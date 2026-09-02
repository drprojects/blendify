"""Add a field that arrived late to a scene, without re-parsing the rest.

When a colleague ships a richer version of a cloud — same points, one extra
column — a full re-import costs a minute of parsing and, worse, a re-export
throws away whatever camera and grading were tuned in the GUI since. This does
the surgical version instead:

  1. verify the new file is the same cloud, point for point, on every field the
     old one already had (this is the whole safety argument — bail on mismatch)
  2. install it as the config's `data.path`, keeping the old file alongside
  3. re-seed the parse cache from the existing one plus the new column, so
     nothing is re-parsed and a future cache rebuild yields the same thing
  4. colorize the new layer through the normal pipeline and inject it into an
     exported .blend, leaving every other layer and all GUI state untouched

    python scripts/inject_field.py \
        --config configs/figures/malibu3d_D075_UU-S1-3.yaml \
        --source data/malibu3d/send_29_07_v2/blender_export/D075_UU-S1-3.ply.gz \
        --field strength

`--dry-run` stops after the verification and prints what it would do.
"""
import argparse
import gzip
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from figlib.blender_material import DEFAULT_GRADE
from figlib.config import load_config
from figlib.data import _cache_path, load_point_cloud
from figlib.grading import srgb_to_linear
from figlib.palettes import is_continuous, load_palettes

PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1",
             "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def read_ply(path):
    """Parse a (gzipped) binary-little-endian PLY into a structured array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    end = raw.index(b"\n", raw.index(b"end_header")) + 1
    count, fields = None, []
    for line in raw[:end].decode("ascii").splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts and parts[0] == "property":
            if parts[1] not in PLY_TYPES:
                raise SystemExit(f"unsupported PLY property type {parts[1]!r}")
            fields.append((parts[2], PLY_TYPES[parts[1]]))
    dtype = np.dtype(fields)
    return np.frombuffer(raw, dtype=dtype, count=count, offset=end)


def same_values(a, b):
    """Bit-identical, treating NaN in both as a match."""
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        both_nan = np.isnan(a) & np.isnan(b)
        return bool(np.array_equal(a[~both_nan], b[~both_nan]))
    return bool(np.array_equal(a, b))


def verify(new, cache_path, field):
    """The new file must be the old cloud plus exactly the new column."""
    print(f"Verifying against {osp.basename(cache_path)}")
    cache = np.load(cache_path)
    ok = True

    n_cache, n_new = len(cache["pos"]), len(new)
    match = n_cache == n_new
    print(f"  {'PASS' if match else 'FAIL'} point count "
          f"{n_new:,} (new) vs {n_cache:,} (cached)")
    if not match:
        return False           # nothing below is meaningful without alignment

    if field not in new.dtype.names:
        print(f"  FAIL new file has no {field!r} field "
              f"(has: {', '.join(new.dtype.names)})")
        return False
    print(f"  PASS new file carries {field!r}")

    pos = np.stack([new["x"], new["y"], new["z"]], axis=1).astype(np.float32)
    match = same_values(cache["pos"], pos)
    print(f"  {'PASS' if match else 'FAIL'} pos bit-identical")
    ok &= match

    if "rgb_colors" in cache.files and {"red", "green", "blue"} <= set(new.dtype.names):
        rgb = np.stack([new["red"], new["green"], new["blue"]], axis=1)
        match = same_values(cache["rgb_colors"], rgb)
        print(f"  {'PASS' if match else 'FAIL'} rgb bit-identical")
        ok &= match

    for key in cache.files:
        if not key.endswith("_field"):
            continue
        name = key[: -len("_field")]
        if name not in new.dtype.names:
            print(f"  FAIL cached field {name!r} is missing from the new file")
            ok = False
            continue
        match = same_values(cache[key], new[name])
        print(f"  {'PASS' if match else 'FAIL'} {name} bit-identical")
        ok &= match
    return bool(ok)


def build_payload(cfg, layer, field, out_path):
    """Colorize through the normal pipeline, then hand the .blend side arrays."""
    c_data = cfg["data"]
    cloud = load_point_cloud(
        c_data["path"], palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"], colors=None,
        cache=c_data["cache"], cache_dir=c_data["cache_dir"], log=print)

    if layer not in cloud.colors:
        raise SystemExit(
            f"{layer!r} is not among the colorizations the pipeline produced "
            f"({', '.join(cloud.names)}). Check the palette entry for it.")

    # The .blend was exported from the cloud *after* these, so the payload has
    # to go through them too or the point counts will not line up.
    cloud = cloud.drop_void(c_data["drop_void"], log=print)
    cloud = cloud.subsample(c_data["subsample"], c_data["seed"])
    print(f"  {len(cloud):,} points after drop_void/subsample")

    rgb = srgb_to_linear(cloud.colors[layer].astype(np.float32) / 255.0).astype(np.float32)
    alpha = np.ones((len(rgb), 1), dtype=np.float32)
    void = cloud.void.get(layer)
    c_void = cfg["void"]
    if void is not None and void.any():
        if c_void["color"] is not None:
            rgb[void] = srgb_to_linear(
                np.asarray(c_void["color"], dtype=np.float32)).astype(np.float32)
        alpha[void] = float(c_void["alpha"])
        print(f"  {int(void.sum()):,} void points muted")
    colors = np.concatenate([rgb, alpha], axis=1)

    values = np.asarray(cloud.fields[field], dtype=np.float32)
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0).astype(np.float32)

    grade = dict(DEFAULT_GRADE)
    grade["attribute"] = f"color_{layer}"
    grade["alpha"] = float((cfg["point_cloud"]["layer_alpha"] or {}).get(
        layer, cfg["point_cloud"]["alpha"]))

    meta = {"name": layer, "grade": grade}
    palettes = load_palettes(c_data["palettes"], c_data["palette_overrides"])
    palette = palettes.get(layer, {})
    if is_continuous(palette):
        # Resolve the range the way `continuous_colors` does. The export's
        # 0..30 fallback is an elevation default and would leave a 0..1 field
        # squashed into the bottom of the GUI ramp.
        if palette.get("vmin") is not None and palette.get("vmax") is not None:
            lo, hi = float(palette["vmin"]), float(palette["vmax"])
            source = "absolute"
        else:
            lo, hi = np.percentile(
                values[finite],
                [palette.get("percentile_low", 2.0),
                 palette.get("percentile_high", 98.0)])
            source = "per-tile percentiles"
        meta["continuous"] = {
            "vmin": float(lo), "vmax": float(hi),
            "gamma": float(palette.get("gamma", 1.0)),
            "unit": "m" if layer == "elevation" else "",
        }
        print(f"  continuous ramp {source} [{lo:.5g}, {hi:.5g}], "
              f"gamma {meta['continuous']['gamma']}")

    np.savez(out_path, colors=colors, values=values, meta=json.dumps(meta))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True,
                        help="the richer file: same cloud, plus the new column")
    parser.add_argument("--field", required=True, help="raw column to bring in")
    parser.add_argument("--layer", default=None,
                        help="colorization name, if it differs from the field")
    parser.add_argument("--blend", default=None,
                        help="the .blend to inject into (default: skip)")
    parser.add_argument("--blender", default="blender", help="Blender binary")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    layer = args.layer or args.field

    cfg = load_config(args.config)
    target = cfg["data"]["path"]
    cache_dir = cfg["data"]["cache_dir"]
    old_cache = _cache_path(target, cfg["data"]["palettes"], cache_dir)
    if not osp.exists(old_cache):
        raise SystemExit(
            f"No cache for {target}. Without one there is nothing to verify "
            f"against and nothing to re-seed — just re-render normally.")

    print(f"Reading {args.source}")
    new = read_ply(args.source)
    print(f"  {len(new):,} points, fields: {', '.join(new.dtype.names)}\n")

    if not verify(new, old_cache, args.field):
        raise SystemExit(
            "\n*** VERIFICATION FAILED — nothing was modified. ***\n"
            "The new file is not the same cloud in the same order, so the "
            "field cannot be attached point-wise.")
    print("\nAll checks passed: same cloud, same order.\n")

    if args.dry_run:
        print(f"--dry-run: would install {args.source} as {target},\n"
              f"           re-seed the cache with {args.field}_field,")
        print(f"           and inject {layer!r} into {args.blend}"
              if args.blend else "           and skip .blend injection")
        return

    # 1. install the new source, keeping the old one — the cache key is derived
    #    from the source's size and mtime, so this also invalidates it cleanly.
    if osp.abspath(args.source) != osp.abspath(target):
        backup = target.replace(".ply.gz", f"_before_{args.field}.ply.gz")
        if osp.exists(target) and not osp.exists(backup):
            shutil.move(target, backup)
            print(f"kept the old source as {backup}")
        shutil.move(args.source, target)
        print(f"installed {osp.basename(args.source)} as {target}")

    # 2. re-seed the cache: everything the old one had, plus the new column
    new_cache = _cache_path(target, cfg["data"]["palettes"], cache_dir)
    cached = dict(np.load(old_cache))
    cached[f"{args.field}_field"] = np.ascontiguousarray(new[args.field])
    np.savez_compressed(new_cache, **cached)
    print(f"re-seeded cache {new_cache} "
          f"({osp.getsize(new_cache) / 1e6:.1f} MB, no re-parse)")
    if osp.abspath(new_cache) != osp.abspath(old_cache):
        print(f"  (the previous cache {osp.basename(old_cache)} is now orphaned "
              f"and can be deleted)")

    # 3. colorize and inject
    with tempfile.TemporaryDirectory() as tmp:
        payload = osp.join(tmp, "layer.npz")
        print("\nColorizing through the pipeline")
        meta = build_payload(cfg, layer, args.field, payload)

        if not args.blend:
            print("\nNo --blend given; cache is updated, "
                  "so renders will pick the layer up. Skipping injection.")
            return

        print(f"\nInjecting {layer!r} into {args.blend}")
        script = osp.join(osp.dirname(osp.abspath(__file__)), "blend_inject_layer.py")
        result = subprocess.run(
            [args.blender, "--background", args.blend, "--python", script,
             "--", "--payload", payload, "--out", args.blend],
            capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("  ") or "Error" in line or "error" in line:
                print(line)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise SystemExit(f"{args.blender} exited {result.returncode}")
    print(f"\nDone. {layer!r} is now a layer in the .blend and in the cache.")


if __name__ == "__main__":
    main()
