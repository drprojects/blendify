"""Set the render resolution stored inside a .blend, in place.

An exported .blend carries its own `scene.render.resolution_x/y`, so a change to
`render.resolution` in the configs does not reach files that were already
written. This retrofits them without a re-export.

Runs *inside* Blender, so it opens files saved by the GUI version:

    blender --background scene.blend --python scripts/blend_set_resolution.py \
        -- --width 2100 --height 1190 --out scene.blend

The camera is deliberately left alone. `fov_x` is the horizontal field of view
and Blender derives the vertical one from the frame's aspect, so changing only
the resolution crops top and bottom and leaves the horizontal framing identical.
Touching the lens here would silently re-frame every tuned view.
"""
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]

    def arg(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    width = int(arg("--width"))
    height = int(arg("--height"))
    out_path = arg("--out")

    render = bpy.context.scene.render
    before = (render.resolution_x, render.resolution_y, render.resolution_percentage)
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100

    camera = bpy.context.scene.camera
    lens = round(camera.data.lens, 6) if camera else None
    print(f"  resolution {before[0]}x{before[1]} @{before[2]}% "
          f"-> {width}x{height} @100%  (aspect {width / height:.4f})")
    print(f"  camera lens left at {lens} mm, sensor "
          f"{round(camera.data.sensor_width, 4) if camera else '-'} mm")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
