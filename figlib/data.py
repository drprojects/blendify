"""Read point clouds from whatever format they arrive in.

Everything funnels into one `PointCloud` object: XYZ plus a named set of RGB
colorizations. The figure script only ever sees that object, so adding support
for a new file format is a matter of writing one reader here.

Currently supported: `.pt` (the legacy dict format), `.ply` / `.ply.gz`,
`.npz`, and `.las` / `.laz` (needs `laspy`).
"""
import gzip
import hashlib
import os
import os.path as osp
from dataclasses import dataclass, field

import numpy as np

from .palettes import colorize, load_palettes

SUPPORTED_SUFFIXES = (".pt", ".ply", ".ply.gz", ".npz", ".las", ".laz")

# Fields that are geometry or photo colour rather than a colorizable variable
_RESERVED = {"x", "y", "z", "red", "green", "blue", "nx", "ny", "nz",
             "intensity_normalized"}


@dataclass
class PointCloud:
    """XYZ plus any number of named (N, 3) uint8 colorizations.

    `void[name]` is a boolean mask marking points that carry no real label in
    that colorization — an explicit void/N-A class, or a non-finite value in a
    continuous field. Only colorizations built from a palette have one.
    """
    pos: np.ndarray
    colors: dict = field(default_factory=dict)
    void: dict = field(default_factory=dict)
    # Raw source columns (semantic labels, elevation metres, ...). Keeping them
    # means a palette change is a re-colorize rather than a re-parse, and the
    # raw values can be carried into the .blend for live editing.
    fields: dict = field(default_factory=dict)
    source: str = ""
    # Translation applied by center(), so other geometry (network graphs,
    # bounding boxes) can be brought into the same frame
    offset: np.ndarray = None

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float32)
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"pos must be (N, 3), got {self.pos.shape}")
        for name, value in list(self.colors.items()):
            value = np.asarray(value)
            if value.shape != self.pos.shape:
                raise ValueError(
                    f"colour {name!r} has shape {value.shape}, expected {self.pos.shape}")
            self.colors[name] = value

    def __len__(self):
        return len(self.pos)

    @property
    def names(self):
        return sorted(self.colors)

    def center(self):
        """Recenter on the XY bounding-box midpoint and drop min Z to zero.

        Records the translation in `self.offset` so anything else placed in the
        scene can be moved by the same amount: `scene_xyz = source_xyz + offset`.
        """
        xy = self.pos[:, :2]
        shift_xy = (xy.max(0) + xy.min(0)) / 2
        shift_z = self.pos[:, 2].min()
        self.pos[:, :2] = xy - shift_xy
        self.pos[:, 2] -= shift_z
        self.offset = np.array(
            [-shift_xy[0], -shift_xy[1], -shift_z], dtype=np.float64)
        return self

    def add_xyz_colorization(self, name="xyz"):
        """Add a colorization encoding normalized position as RGB.

        Each axis is min-max scaled over the cloud's own bounding box, so the
        result reads as a smooth R=x, G=y, B=z gradient. Handy for showing scene
        structure without any semantic content.
        """
        lo, hi = self.pos.min(0), self.pos.max(0)
        span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        normalized = (self.pos - lo) / span
        self.colors[name] = np.clip(
            np.rint(normalized * 255), 0, 255).astype(np.uint8)
        return self

    def _select(self, keep):
        self.pos = self.pos[keep]
        self.colors = {k: v[keep] for k, v in self.colors.items()}
        self.void = {k: v[keep] for k, v in self.void.items()}
        self.fields = {k: v[keep] for k, v in self.fields.items()}
        return self

    def subsample(self, n, seed=0):
        """Keep a random subset, applying the same indices to every colour."""
        if n is None or n >= len(self):
            return self
        keep = np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False))
        return self._select(keep)

    def drop_void(self, names, log=print):
        """Remove points that are void in any of the named colorizations.

        The same points are removed from *every* colorization, so all figures
        of this scene keep an identical point set and stay comparable.
        """
        if not names:
            return self
        if isinstance(names, str):
            names = [names]

        drop = np.zeros(len(self), dtype=bool)
        for name in names:
            if name in self.void:
                drop |= self.void[name]
                continue
            if name in self.colors:
                raise KeyError(
                    f"data.drop_void names {name!r}, but that colorization has no "
                    f"void mask — it was read as ready-made RGB rather than built "
                    f"from a palette, so there is no way to tell which points are "
                    f"unlabelled.")
            raise KeyError(
                f"data.drop_void names {name!r}, which this cloud does not have. "
                f"Layers with a void mask: {sorted(self.void)}")

        if drop.any():
            log(f"Dropping {int(drop.sum())} points void in "
                f"{', '.join(names)} ({100 * drop.mean():.2f}% of the cloud)")
        return self._select(~drop)

    def summary(self):
        extent = (self.pos.max(0) - self.pos.min(0)).round(2)
        lines = [f"{len(self)} points, extent {tuple(extent)}, from {self.source}"]
        for name in self.names:
            value = self.colors[name]
            lines.append(f"  {name:24s} {value.dtype} "
                         f"[{value.min()}, {value.max()}]")
        return "\n".join(lines)


def _as_uint8_rgb(value):
    """Normalize a colour array to (N, 3) uint8 in [0, 255]."""
    value = np.asarray(value)
    if value.dtype.kind in "US":  # "rgb(12, 34, 56)" strings
        value = np.array([
            [int(c) for c in s.replace("rgb(", "").replace(")", "").split(",")]
            for s in value])
    if value.dtype.kind == "f":
        # floats are either already 0-1 or 0-255
        value = value * 255 if value.max() <= 1.0 else value
    return np.clip(value, 0, 255).astype(np.uint8)[:, :3]


# --------------------------------------------------------------------------
# Readers. Each returns a PointCloud.
# --------------------------------------------------------------------------

def _read_pt(path, **kwargs):
    """The legacy format: a dict with `pos` and any number of `*_colors`."""
    import torch
    data = torch.load(path)
    pos = data["pos"]
    pos = pos.numpy() if hasattr(pos, "numpy") else np.asarray(pos)

    colors = {}
    for key, value in data.items():
        if key.endswith("_colors"):
            colors[key[: -len("_colors")]] = _as_uint8_rgb(value)
    return PointCloud(pos, colors, source=path)


def _read_npz(path, **kwargs):
    data = np.load(path, allow_pickle=False)
    colors = {
        key[: -len("_colors")]: _as_uint8_rgb(data[key])
        for key in data.files if key.endswith("_colors")
    }
    void = {
        key[: -len("_void")]: data[key].astype(bool)
        for key in data.files if key.endswith("_void")
    }
    raw = {
        key[: -len("_field")]: data[key]
        for key in data.files if key.endswith("_field")
    }
    return PointCloud(data["pos"], colors, void, raw, source=path)


def _read_ply(path, palettes=None, log=print, **kwargs):
    """Generic PLY reader.

    Picks up, in order of preference per variable:
      1. precomputed `<name>_red/green/blue` channels
      2. a raw scalar/label field, colorized through `palettes`
    plus plain `red/green/blue` as the `rgb` colorization.
    """
    from plyfile import PlyData

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        vertex = PlyData.read(f)["vertex"].data

    fields = set(vertex.dtype.names)
    pos = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    colors, void = {}, {}

    if {"red", "green", "blue"} <= fields:
        colors["rgb"] = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)

    # precomputed <name>_red/green/blue triplets
    triplets = {
        name[: -len("_red")] for name in fields if name.endswith("_red")
    } - {""}
    for name in sorted(triplets):
        channels = [f"{name}_{c}" for c in ("red", "green", "blue")]
        if set(channels) <= fields:
            colors[name] = np.stack(
                [vertex[c] for c in channels], axis=1).astype(np.uint8)
            log(f"  {name}: precomputed RGB channels")

    # Keep every raw column that a palette could consume. Colorization happens
    # later (see `colorize_cloud`) so the cache holds parsed data, not one
    # particular palette's output.
    raw = {}
    for name in sorted(fields - _RESERVED):
        if name.endswith(("_red", "_green", "_blue")):
            continue
        raw[name] = np.asarray(vertex[name])

    return PointCloud(pos, colors, void, raw, source=path)


def _read_las(path, palettes=None, log=print, **kwargs):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "Reading .las/.laz needs laspy. Install it into the blendify env:\n"
            "  ~/miniconda3/envs/blendify/bin/pip install 'laspy[lazrs]'") from exc

    las = laspy.read(path)
    pos = np.stack([las.x, las.y, las.z], axis=1)
    colors, void = {}, {}

    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        rgb = np.stack([las.red, las.green, las.blue], axis=1)
        # LAS stores colour as uint16; scale down when it clearly uses the range
        colors["rgb"] = (rgb / 257 if rgb.max() > 255 else rgb).astype(np.uint8)

    palettes = palettes or {}
    present = {d.name for d in las.point_format.dimensions}
    for name in sorted(present - _RESERVED):
        if name not in palettes:
            continue
        log(f"  {name}: colorized from palette")
        colors[name], void[name] = colorize(
            np.asarray(las[name]), palettes[name],
            on_warning=lambda m: log(f"    {m}"))

    return PointCloud(pos, colors, void, raw, source=path)


_READERS = {
    ".pt": _read_pt,
    ".npz": _read_npz,
    ".ply": _read_ply,
    ".ply.gz": _read_ply,
    ".las": _read_las,
    ".laz": _read_las,
}


def colorize_cloud(cloud, palettes, log=print):
    """Build every colorization the palettes describe from the raw columns.

    Runs on every load, including cache hits, so changing a palette is a
    re-colorize (about a second) rather than a re-parse (minutes). Colours
    already present — photo RGB, precomputed triplets — are left alone.
    """
    palettes = palettes or {}
    for name, palette in sorted(palettes.items()):
        source = palette.get("field", name)
        if source not in cloud.fields:
            continue
        suffix = "" if source == name else f" (from field {source!r})"
        log(f"  {name}: colorized from palette{suffix}")
        cloud.colors[name], cloud.void[name] = colorize(
            cloud.fields[source], palette, on_warning=lambda m: log(f"    {m}"))

    for name in sorted(cloud.fields):
        if name not in cloud.colors and not any(
                p.get("field", k) == name for k, p in palettes.items()):
            log(f"  {name}: no palette entry, skipped")
    return cloud


def _cache_path(path, palettes_path, cache_dir, overrides_path=None):
    """Where the parsed form of `path` gets cached.

    The key covers the source path plus the size and mtime of both the source
    and the palette file, so editing either invalidates the cache.
    """
    # Only the SOURCE file identifies the cache. Palettes are applied after
    # loading, so a colour change must not invalidate a parse.
    parts = [osp.abspath(path)]
    for dependency in (path,):
        if dependency and osp.exists(dependency):
            stat = os.stat(dependency)
            parts.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    stem = osp.basename(path).split(".")[0]
    return osp.join(cache_dir, f"{stem}-{digest}.npz")


def load_point_cloud(path, palettes=None, palette_overrides=None, colors=None,
                     cache=True, cache_dir="data/.figcache", log=print):
    """Read `path` into a PointCloud.

    Parsing a large gzipped PLY costs far more than rendering it, so the parsed
    form is cached to `.npz` and reused until the source or the palette file
    changes. `.npz` inputs are read directly and never re-cached.

    Args:
        path: input file; format is picked from the suffix.
        palettes: path to a palette JSON, or an already-loaded dict. Only used
            by formats that carry raw label/scalar fields.
        colors: optional list of colorization names to keep (default: all).
        cache: set False to always re-parse the source.
        cache_dir: where cached `.npz` files live.
        log: where progress messages go.
    """
    lowered = path.lower()
    suffix = next(
        (s for s in sorted(_READERS, key=len, reverse=True) if lowered.endswith(s)),
        None)
    if suffix is None:
        raise ValueError(
            f"Don't know how to read {osp.basename(path)!r}. "
            f"Supported: {', '.join(SUPPORTED_SUFFIXES)}")

    if not osp.exists(path):
        raise FileNotFoundError(f"No such point cloud: {path}")

    palettes_path = palettes if isinstance(palettes, str) else None
    if isinstance(palettes, str) or palettes is None:
        palettes = load_palettes(palettes, palette_overrides)

    # Serve from cache when we can. The cache holds parsed data, so the palette
    # is applied afterwards either way — a colour change costs a re-colorize,
    # not a re-parse.
    cached = None
    cloud = None
    if cache and suffix != ".npz":
        cached = _cache_path(path, palettes_path, cache_dir)
        if osp.exists(cached):
            log(f"Reading {path}\n  (from cache {cached})")
            cloud = _read_npz(cached)
            cloud.source = path

    if cloud is None:
        log(f"Reading {path}")
        cloud = _READERS[suffix](path, palettes=palettes, log=log)
        if cached is not None:
            os.makedirs(cache_dir, exist_ok=True)
            save_npz(cloud, cached)
            log(f"  cached to {cached} ({osp.getsize(cached) / 1e6:.1f} MB)")

    # Legacy .pt/.npz inputs carry colours but no raw columns; nothing to do.
    if cloud.fields:
        colorize_cloud(cloud, palettes, log=log)

    return _finish(cloud, path, colors)


def _finish(cloud, path, colors):
    if colors:
        missing = [c for c in colors if c not in cloud.colors]
        if missing:
            raise KeyError(
                f"Requested colours {missing} not in {path}. "
                f"Available: {cloud.names}")
        cloud.colors = {c: cloud.colors[c] for c in colors}
        # keep void masks for dropped colorizations: `drop_void` may reference
        # a layer that is not itself being rendered

    if not cloud.colors:
        raise ValueError(
            f"{path} yielded no colorizations. If it carries raw label fields, "
            f"pass a palette file via `data.palettes` in the config.")
    return cloud


def save_npz(cloud, path):
    """Write a PointCloud to a fast-loading .npz cache.

    Stores the raw source columns as well as any colours, so the cache is a
    parsed cloud rather than one palette's rendering of it.
    """
    np.savez_compressed(
        path, pos=cloud.pos,
        **{f"{name}_colors": value for name, value in cloud.colors.items()},
        **{f"{name}_void": value for name, value in cloud.void.items()},
        **{f"{name}_field": value for name, value in cloud.fields.items()})
