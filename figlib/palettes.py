"""Turn raw per-point fields (class labels, scalars) into RGB.

A palette file is JSON mapping a field name to either a categorical entry
(`names` + `colors` + `unknown_color`) or a continuous one (`type: continuous`
with `color_stops_rgb` and clipping percentiles). This is the format MALIBU3D
ships as `palettes.json`, but nothing here is MALIBU3D-specific.
"""
import copy
import json

import numpy as np


def hex_to_rgb(value):
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _read_overrides(source):
    """A palette-override mapping, from a path or an already-loaded dict."""
    if source is None:
        return {}
    if not isinstance(source, str):
        return dict(source)
    with open(source) as handle:
        if source.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def load_palettes(path, overrides=None):
    """Load a palette JSON, optionally patched by an override file.

    The delivered `palettes.json` is a colleague's artefact and should stay
    untouched. Overrides let a figure restyle classes without editing it: a JSON
    or YAML file of the same shape, merged per layer. Colours may be given as a
    full list, or per class by name:

        {"semantic": {"colors_by_name": {"Agricultural soil": "#b8860b"}}}
    """
    if path is None:
        return {}
    with open(path) as f:
        palettes = json.load(f)

    if not overrides:
        return palettes

    # A list applies several override files in order, so a shared restyle and a
    # figure-specific addition can be combined without one having to copy the
    # other (a copy is what goes stale).
    if isinstance(overrides, (list, tuple)):
        merged = {}
        for item in overrides:
            merged.update(_read_overrides(item))
        overrides = merged
    else:
        overrides = _read_overrides(overrides)

    def apply(layer, patch):
        # A new layer is legitimate: a drop may add a field the shared
        # palettes.json never described (e.g. lidar `strength`).
        patch = dict(patch)                      # never mutate the caller's dict
        base = patch.pop("like", None)
        if base is not None:
            if base not in palettes:
                raise KeyError(
                    f"Palette override for {layer!r} says like={base!r}, "
                    f"which is not a known layer: {sorted(palettes)}")
            palettes[layer] = copy.deepcopy(palettes[base])
        entry = palettes.setdefault(layer, {})
        by_name = patch.pop("colors_by_name", None)
        for key, value in patch.items():
            # An explicit null removes an inherited key. A prediction layer
            # needs this: it inherits a derived task's palette but its labels
            # are already in that task's index space, so the source field's
            # `remap` must not be applied a second time.
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        if by_name:
            names = entry.get("names", [])
            index = {n: i for i, n in enumerate(names)}
            for class_name, color in by_name.items():
                if class_name not in index:
                    raise KeyError(
                        f"Palette override for {layer!r} names class "
                        f"{class_name!r}; known classes: {names}")
                entry["colors"][index[class_name]] = color

    # Two passes so `like` can reference a layer this same file defines, and
    # inherits it *after* its own patch has been applied rather than before.
    # Keys starting with "_" are documentation, not layers.
    items = [(k, v) for k, v in (overrides or {}).items()
             if not k.startswith("_")]
    for layer, patch in items:
        if "like" not in patch:
            apply(layer, patch)
    for layer, patch in items:
        if "like" in patch:
            apply(layer, patch)
    return palettes


def is_continuous(palette):
    return palette.get("type") == "continuous" or "color_stops_rgb" in palette


# Class names that mean "no label here" rather than a real category
VOID_NAMES = {"void", "n/a", "na", "none", "unknown", "no data", "nodata"}


def void_class_indices(palette):
    """Indices of classes that mean 'no label', by name."""
    return {i for i, name in enumerate(palette.get("names", []))
            if name.strip().lower() in VOID_NAMES}


def categorical_colors(labels, palette, on_warning=None):
    """Map integer labels through a palette LUT, with a fallback colour.

    Returns (colors, void) where `void` marks points that carry no real label:
    either an explicit void/N-A class, or an index outside the palette.
    """
    labels = np.asarray(labels).astype(np.int64)

    # A derived layer regroups another field's classes: `remap[i]` gives the
    # class of this palette that source class `i` belongs to. Used to split the
    # 44 natural-habitat classes into the four prediction tasks.
    remap = palette.get("remap")
    if remap is not None:
        table = np.asarray(remap, dtype=np.int64)
        labels = table[np.clip(labels, 0, len(table) - 1)]

    colors = np.stack([hex_to_rgb(c) for c in palette["colors"]])
    unknown = hex_to_rgb(palette.get("unknown_color", "#808080"))

    size = max(len(colors), int(labels.max()) + 1 if len(labels) else 0)
    lut = np.repeat(unknown[None], size, axis=0)
    lut[:len(colors)] = colors

    out_of_range = labels >= len(colors)
    n_unknown = int(out_of_range.sum())
    if n_unknown and on_warning:
        on_warning(f"{n_unknown} points fall outside the {len(colors)}-class "
                   f"palette and render as unknown grey")

    void = out_of_range.copy()
    for index in void_class_indices(palette):
        void |= labels == index
    return lut[labels], void


def continuous_colors(values, palette, on_warning=None):
    """Map a float field through a colour ramp, clipped to percentiles.

    Returns (colors, void), where `void` marks the non-finite entries that get
    the fallback colour.
    """
    values = np.asarray(values).astype(np.float64)
    stops = np.asarray(palette["color_stops_rgb"], dtype=np.float64)
    nan_color = hex_to_rgb(palette.get("nan_color", "#808080"))

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Field is entirely NaN; cannot build a colour ramp")

    # An absolute range makes a colour mean the same value on every tile, which
    # per-tile percentiles cannot. Percentiles remain the fallback.
    if palette.get("vmin") is not None and palette.get("vmax") is not None:
        lo, hi = float(palette["vmin"]), float(palette["vmax"])
        source = "absolute"
    else:
        lo, hi = np.percentile(
            values[finite],
            [palette.get("percentile_low", 2.0), palette.get("percentile_high", 98.0)])
        source = "per-tile percentiles"
    if on_warning:
        on_warning(f"ramp {source} [{lo:.2f}, {hi:.2f}], "
                   f"{int((~finite).sum())} non-finite -> fallback colour")

    # Non-finite entries are parked at 0 so they survive the int cast, then
    # overwritten with the fallback colour at the end
    safe = np.where(finite, values, lo)
    t = np.clip((safe - lo) / max(hi - lo, 1e-9), 0, 1)
    # `gamma` reshapes the clipped range before the ramp is sampled. Lidar
    # intensity is heavily right-skewed (99% of `strength` below 0.063), so
    # gamma < 1 expands the crowded low end and gamma > 1 compresses it.
    gamma = float(palette.get("gamma", 1.0))
    if gamma != 1.0:
        t = np.power(t, gamma)
    pos = t * (len(stops) - 1)
    idx = np.clip(pos.astype(int), 0, len(stops) - 2)
    frac = (pos - idx)[:, None]

    out = np.rint(stops[idx] * (1 - frac) + stops[idx + 1] * frac).astype(np.uint8)
    out[~finite] = nan_color
    return out, ~finite


def colorize(values, palette, on_warning=None):
    """Dispatch to the continuous or categorical mapping.

    Returns (colors, void).
    """
    if is_continuous(palette):
        return continuous_colors(values, palette, on_warning)
    return categorical_colors(values, palette, on_warning)
