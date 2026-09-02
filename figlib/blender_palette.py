"""Editable class palettes inside the .blend.

Class colours are baked into the per-point colour arrays, so they cannot be
changed after export — unless the class *index* travels with them. This stores
one int attribute per categorical layer (`class_<layer>`) alongside the colour
attribute, plus the palette itself as a custom property. Recolouring is then a
lookup: `color_<layer> = LUT[class_<layer>]`, which measures ~113 ms for 5.5 M
points, fast enough to drive a colour picker.

Nothing about rendering changes: the shader still reads `color_<layer>` exactly
as before. This only rewrites that attribute's contents.

The class index is recovered by matching each point's colour back to the palette
LUT, which is exact because every class in the MALIBU3D palettes has a distinct
colour. That avoids having to re-parse the source clouds.
"""
import json

import numpy as np

PALETTES_PROP = "figure_palettes"          # currently applied, editable
ORIGINAL_PROP = "figure_palettes_original"  # as exported, for Reset
UNKNOWN_NAME = "Unknown"


def _codes(rgb):
    """Pack uint8 RGB into one int per row, for exact matching."""
    rgb = np.asarray(rgb, dtype=np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def build_lut(palette, void_color=None):
    """Two LUTs plus names: one for matching, one for display.

    `raw` is the palette exactly as delivered — this is what the cached point
    colours were built from, so it is the only thing class indices can be
    matched against.

    `display` additionally replaces void/N-A classes with `void_color`, which is
    what those points actually render as after the figure script mutes them. It
    is the table the editor shows and recolours from, so recolouring reproduces
    the exported appearance rather than reverting to the palette's raw black.
    """
    from .palettes import hex_to_rgb, void_class_indices

    raw = np.stack([hex_to_rgb(c) for c in palette["colors"]]).astype(np.uint8)
    names = list(palette.get("names", [f"class {i}" for i in range(len(raw))]))

    unknown = hex_to_rgb(palette.get("unknown_color", "#808080"))
    raw = np.vstack([raw, unknown[None]])
    names.append(UNKNOWN_NAME)

    # Display LUT is LINEAR float 0-1. Blender colour attributes and colour
    # properties are linear; the palette hex codes are sRGB, so they are decoded
    # here. This also makes the GUI colour picker show the original hex, since
    # Blender's picker encodes linear -> sRGB for display.
    from .grading import srgb_to_linear
    display = srgb_to_linear(raw.astype(np.float64) / 255.0)
    # The trailing "Unknown" slot is unlabelled too, so it belongs to the single
    # global void colour rather than the palette's own unknown_color. Without
    # this it kept #808080 while every other unlabelled class became #CCCCCC.
    void = sorted(set(void_class_indices(palette)) | {len(raw) - 1})
    if void_color is not None:
        linear_void = srgb_to_linear(np.asarray(void_color, dtype=np.float64))
        for index in void:
            display[index] = linear_void
    return raw, display, names, void


def class_indices(rendered_rgb, lut_rgb):
    """Recover the class index of every point from its rendered colour.

    Points matching no class (should not happen) get the trailing unknown slot.
    """
    point_codes = _codes(rendered_rgb)
    lut_codes = _codes(lut_rgb)

    order = np.argsort(lut_codes, kind="stable")
    sorted_codes = lut_codes[order]
    position = np.searchsorted(sorted_codes, point_codes)
    position = np.clip(position, 0, len(sorted_codes) - 1)
    index = order[position]

    unmatched = lut_codes[index] != point_codes
    index = index.astype(np.int32)
    index[unmatched] = len(lut_rgb) - 1        # the unknown slot
    return index, int(unmatched.sum())


def store(obj, tables):
    """Persist the editable palettes on the object (current + pristine copy)."""
    payload = json.dumps(tables)
    obj[PALETTES_PROP] = payload
    obj[ORIGINAL_PROP] = payload


def load(obj, original=False):
    raw = obj.get(ORIGINAL_PROP if original else PALETTES_PROP)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def save_current(obj, tables):
    obj[PALETTES_PROP] = json.dumps(tables)


def recolour(mesh, layer, lut_rgb):
    """Rewrite `color_<layer>` from `class_<layer>` and a LUT of 0-1 RGB.

    The existing alpha channel is preserved, so per-point opacity (the muted
    void points) survives a palette edit.
    """
    index_attr = mesh.attributes.get(f"class_{layer}")
    colour_attr = mesh.color_attributes.get(f"color_{layer}")
    if index_attr is None or colour_attr is None:
        return False

    count = len(mesh.vertices)
    labels = np.empty(count, dtype=np.int32)
    index_attr.data.foreach_get("value", labels)

    existing = np.empty(count * 4, dtype=np.float32)
    colour_attr.data.foreach_get("color", existing)
    existing = existing.reshape(count, 4)

    lut = np.asarray(lut_rgb, dtype=np.float32)
    labels = np.clip(labels, 0, len(lut) - 1)

    out = np.empty((count, 4), dtype=np.float32)
    out[:, :3] = lut[labels]
    out[:, 3] = existing[:, 3]                 # keep void fading
    colour_attr.data.foreach_set("color", out.ravel())
    mesh.update()
    return True
