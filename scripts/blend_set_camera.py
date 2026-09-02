"""Write a camera pose into a .blend, in place.

A pose recovered into a YAML config does not reach an already-exported .blend —
the two are independent copies of the scene, and only a re-export normally
reconciles them. This retrofits the camera so the file opens on the intended
view, without touching anything else.

Runs *inside* Blender, so it opens files saved by the GUI version:

    blender --background scene.blend --python scripts/blend_set_camera.py -- \
        --translation 1,2,3 --quaternion 1,0,0,0 --fov 73 \
        --near 0.1 --far 4095 --out scene.blend

FOV is horizontal and is converted to a focal length against the camera's own
sensor width, which is how the rest of the pipeline defines it. `--resolution
WxH` is optional and left alone by default: frame aspect changes what the
vertical field of view covers, so it is a deliberate decision, not a side effect
of moving the camera.
"""
import math
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]

    def arg(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    translation = [float(v) for v in arg("--translation").split(",")]
    quaternion = [float(v) for v in arg("--quaternion").split(",")]
    fov = float(arg("--fov"))
    near, far = arg("--near"), arg("--far")
    resolution = arg("--resolution")
    out_path = arg("--out")

    scene = bpy.context.scene
    camera = scene.camera or next(
        (o for o in scene.objects if o.type == "CAMERA"), None)
    if camera is None:
        raise SystemExit("no camera in this .blend")

    before_loc = [round(v, 4) for v in camera.location]
    before_fov = math.degrees(
        2 * math.atan(0.5 * camera.data.sensor_width / camera.data.lens))

    camera.rotation_mode = "QUATERNION"
    camera.location = translation
    camera.rotation_quaternion = quaternion
    camera.data.type = "PERSP"
    camera.data.lens = (camera.data.sensor_width / 2.0) / math.tan(
        math.radians(fov) / 2.0)
    if near is not None:
        camera.data.clip_start = float(near)
    if far is not None:
        camera.data.clip_end = float(far)

    after_fov = math.degrees(
        2 * math.atan(0.5 * camera.data.sensor_width / camera.data.lens))
    print(f"  location {before_loc} -> {[round(v, 4) for v in camera.location]}")
    print(f"  fov_x    {before_fov:.4f} -> {after_fov:.4f} deg "
          f"(lens {camera.data.lens:.3f} mm)")

    if resolution:
        width, height = (int(v) for v in resolution.lower().split("x"))
        print(f"  resolution {scene.render.resolution_x}x{scene.render.resolution_y}"
              f" -> {width}x{height}")
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
    else:
        print(f"  resolution left at {scene.render.resolution_x}"
              f"x{scene.render.resolution_y}")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
