"""Add one colorization to an already-exported .blend, in place.

Runs *inside* Blender, so it can open files saved by the GUI version — the `bpy`
pinned in the conda env is older and cannot. `scripts/inject_field.py` calls
this for you; you rarely run it by hand.

    blender --background scene.blend --python scripts/blend_inject_layer.py \
        -- --payload layer.npz --out scene.blend

The payload carries the baked RGBA, the raw values, and the layer metadata, all
computed by the normal pipeline on the conda side. Nothing here re-derives
colour — this only writes attributes and custom properties, so a re-export and
an injection produce the same file.
"""
import json
import sys

import bpy
import numpy as np


def find_cloud():
    candidates = [o for o in bpy.context.scene.objects
                  if o.type == "MESH" and o.name.startswith("point_cloud")]
    if not candidates:
        candidates = sorted(
            (o for o in bpy.context.scene.objects if o.type == "MESH"),
            key=lambda o: len(o.data.vertices), reverse=True)[:1]
    if not candidates:
        raise SystemExit("no mesh in this .blend")
    return candidates[0]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    payload_path = argv[argv.index("--payload") + 1]
    out_path = argv[argv.index("--out") + 1]

    payload = np.load(payload_path, allow_pickle=False)
    meta = json.loads(str(payload["meta"]))
    name = meta["name"]
    colors = payload["colors"]          # (M, 4) float32, linear
    values = payload["values"]          # (M,) float32, raw

    obj = find_cloud()
    mesh = obj.data
    count = len(mesh.vertices)
    print(f"  object {obj.name!r}: {count:,} points")
    if count != len(colors):
        raise SystemExit(
            f"point count mismatch: .blend has {count:,}, payload has "
            f"{len(colors):,}. The payload must be built from the same config "
            f"(same data.drop_void / data.subsample) as this export.")

    # Replacing rather than skipping makes the script idempotent, so a bad
    # palette choice can be re-injected without re-exporting the whole scene.
    for attributes, attr_name, kind in (
            (mesh.color_attributes, f"color_{name}", "FLOAT_COLOR"),
            (mesh.attributes, f"value_{name}", "FLOAT")):
        existing = attributes.get(attr_name)
        if existing is not None:
            print(f"  replacing existing {attr_name}")
            attributes.remove(existing)
        attributes.new(name=attr_name, type=kind, domain="POINT")

    mesh.color_attributes[f"color_{name}"].data.foreach_set(
        "color", np.ascontiguousarray(colors, dtype=np.float32).ravel())
    mesh.attributes[f"value_{name}"].data.foreach_set(
        "value", np.ascontiguousarray(values, dtype=np.float32))
    mesh.update()
    print(f"  wrote color_{name} and value_{name}")

    layers = json.loads(obj.get("figure_layers") or "{}")
    layers[name] = meta["grade"]
    obj["figure_layers"] = json.dumps(layers)
    print(f"  figure_layers now: {', '.join(sorted(layers))}")

    continuous = json.loads(obj.get("figure_continuous") or "{}")
    if meta.get("continuous"):
        continuous[name] = meta["continuous"]
        obj["figure_continuous"] = json.dumps(continuous)
        print(f"  figure_continuous now: {', '.join(sorted(continuous))} "
              f"({name} range {meta['continuous']['vmin']:.4g}"
              f"..{meta['continuous']['vmax']:.4g})")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
