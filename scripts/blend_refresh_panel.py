"""Refresh the embedded GUI panel and colormap presets inside a .blend.

Both are just text in the file — `figure_panel.py` as a text datablock and the
colormap table as a JSON custom property — so improving either is a swap, not a
re-export. Nothing here touches point data, materials, camera or lighting.

Runs *inside* Blender, so it can open files the GUI saved:

    blender --background scene.blend --python scripts/blend_refresh_panel.py \
        -- --panel scripts/blender_layer_switcher.py \
           --colormaps configs/palettes/colormaps.json --out scene.blend
"""
import json
import sys

import bpy

PANEL_TEXT = "figure_panel.py"
COLORMAPS_PROP = "figure_colormaps"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]

    def arg(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    out_path = arg("--out")
    panel_path = arg("--panel")
    colormaps_path = arg("--colormaps")

    if panel_path:
        with open(panel_path) as handle:
            source = handle.read()
        text = bpy.data.texts.get(PANEL_TEXT)
        if text is None:
            text = bpy.data.texts.new(PANEL_TEXT)
            print(f"  created {PANEL_TEXT}")
        text.clear()
        text.write(source)
        print(f"  {PANEL_TEXT}: {len(source.splitlines())} lines")

    if colormaps_path:
        with open(colormaps_path) as handle:
            payload = handle.read()
        parsed = json.loads(payload)
        count = len(parsed.get("maps", parsed))
        targets = [o for o in bpy.data.objects if o.get(COLORMAPS_PROP) is not None]
        if not targets:
            # A .blend exported before continuous layers existed has no holder;
            # attach to the point cloud so the panel finds it anyway.
            targets = [o for o in bpy.context.scene.objects
                       if o.type == "MESH" and o.name.startswith("point_cloud")]
        for obj in targets:
            obj[COLORMAPS_PROP] = payload
        print(f"  {COLORMAPS_PROP}: {count} colormaps on "
              f"{', '.join(o.name for o in targets) or '(no object found!)'}")
        if not targets:
            raise SystemExit("no object to attach colormaps to")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
