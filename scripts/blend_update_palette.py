"""Re-apply a class palette inside an exported .blend, in place.

Class colours are baked into `color_<layer>`, so a palette change in the configs
does not reach a file that was already exported. It does not have to: the class
*index* travels with the points as `class_<layer>`, so recolouring is a lookup
rather than a re-export.

Runs *inside* Blender, so it opens files saved by the GUI version:

    blender --background scene.blend --python scripts/blend_update_palette.py \
        -- --payload palette.json --out scene.blend

The payload is built on the conda side by `scripts/refresh_class_palette.py`,
which uses the same `figlib.blender_palette.build_lut` an export would, so an
updated file and a freshly exported one agree.

The stored `figure_palettes_original` is updated too: the new palette becomes
the baseline that the panel's Reset returns to, not the superseded one.
"""
import json
import sys

import bpy
import numpy as np

PALETTES_PROP = "figure_palettes"
ORIGINAL_PROP = "figure_palettes_original"


def find_cloud():
    objects = [o for o in bpy.context.scene.objects
               if o.type == "MESH" and o.name.startswith("point_cloud")]
    if not objects:
        objects = sorted((o for o in bpy.context.scene.objects if o.type == "MESH"),
                         key=lambda o: len(o.data.vertices), reverse=True)[:1]
    return objects[0] if objects else None


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]

    def arg(flag):
        return argv[argv.index(flag) + 1]

    with open(arg("--payload")) as handle:
        payload = json.load(handle)
    out_path = arg("--out")

    layer = payload["layer"]
    lut = np.asarray(payload["colors"], dtype=np.float32)      # linear RGB

    obj = find_cloud()
    if obj is None:
        raise SystemExit("no mesh in this .blend")
    mesh = obj.data

    index_attr = mesh.attributes.get(f"class_{layer}")
    colour_attr = mesh.color_attributes.get(f"color_{layer}")
    if index_attr is None or colour_attr is None:
        print(f"  SKIP: no class_{layer}/color_{layer} on {obj.name!r} "
              f"(this .blend does not carry that layer)")
        return

    count = len(mesh.vertices)
    labels = np.empty(count, dtype=np.int32)
    index_attr.data.foreach_get("value", labels)
    labels = np.clip(labels, 0, len(lut) - 1)

    existing = np.empty(count * 4, dtype=np.float32)
    colour_attr.data.foreach_get("color", existing)
    existing = existing.reshape(count, 4)

    out = np.empty((count, 4), dtype=np.float32)
    out[:, :3] = lut[labels]
    out[:, 3] = existing[:, 3]        # keep per-point alpha (muted void points)
    changed = int((np.abs(out[:, :3] - existing[:, :3]) > 1e-6).any(axis=1).sum())
    colour_attr.data.foreach_set("color", out.ravel())
    mesh.update()

    counts = np.bincount(labels, minlength=len(lut)).tolist()
    tables = {}
    raw = obj.get(PALETTES_PROP)
    if raw:
        try:
            tables = json.loads(raw)
        except (TypeError, ValueError):
            tables = {}
    tables[layer] = {"names": payload["names"], "colors": lut.tolist(),
                     "counts": counts, "void": payload["void"]}
    serialized = json.dumps(tables)
    obj[PALETTES_PROP] = serialized
    obj[ORIGINAL_PROP] = serialized

    present = sum(1 for c in counts if c)
    print(f"  {obj.name}: recoloured {changed:,}/{count:,} points "
          f"({present} of {len(counts)} classes present)")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
