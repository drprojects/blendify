"""Recover a camera pose by fitting it to an image that was rendered from it.

When a pose is lost — a GUI session closed without saving, and no autosave left
— it can often still be recovered from the render itself, because the scene is
still here. This projects the cached point cloud through a candidate camera with
a plain numpy z-buffer and optimises the pose to match the target image. A
software projection is ~0.3 s where a Cycles render is ~60 s, which is what
makes a search over 7 parameters practical at all.

    python scripts/fit_camera.py \
        --config configs/figures/malibu3d_D068_UN-S1-28_pred.yaml \
        --layer semantic \
        --target .../D068_UN-S1-28_semantic_gt.png \
        --seed-from .../autosave.blend.json

The projection ignores sphere radius, shading and depth of field, so it will
never match a render pixel-for-pixel. It does not need to: class boundaries and
skyline carry more than enough structure to localise a camera, and the answer is
verified afterwards with a real Cycles render.

Reports the fitted pose in the config's own convention (translation, [w,x,y,z]
quaternion, horizontal FOV in degrees).
"""
import argparse
import json
import math
import os.path as osp
import sys

import numpy as np
from PIL import Image
from scipy.optimize import minimize

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from figlib.config import load_config
from figlib.data import load_point_cloud


def quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s,
             (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                 (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                 (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def rotvec_to_quat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.concatenate([[math.cos(theta / 2)], axis * math.sin(theta / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def project(pos, gray, translation, quat, fov_x, size):
    """Z-buffered point projection. Blender cameras look down local -Z."""
    width, height = size
    rotation = quat_to_matrix(quat)
    local = (pos - translation) @ rotation          # world -> camera
    depth = -local[:, 2]
    front = depth > 1e-6
    if not front.any():
        return np.full((height, width), 1.0), np.zeros((height, width), bool)

    focal = (width / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    u = focal * local[front, 0] / depth[front]
    v = focal * local[front, 1] / depth[front]
    px = np.rint(width / 2.0 + u).astype(np.int64)
    py = np.rint(height / 2.0 - v).astype(np.int64)

    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not inside.any():
        return np.full((height, width), 1.0), np.zeros((height, width), bool)
    flat = py[inside] * width + px[inside]
    d = depth[front][inside]
    g = gray[front][inside]

    # Nearest point wins each pixel: sort far-to-near and let later writes
    # overwrite, which is a z-buffer without the loop.
    order = np.argsort(-d, kind="stable")
    image = np.full(width * height, np.nan)
    image[flat[order]] = g[order]
    hit = np.isfinite(image)
    image[~hit] = 1.0                                # background reads as paper
    return image.reshape(height, width), hit.reshape(height, width)


def ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-12 else 0.0


def load_target(path, size):
    image = Image.open(path).convert("RGBA").resize(size, Image.LANCZOS)
    a = np.asarray(image, dtype=np.float64) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", required=True, help="colorization the target shows")
    parser.add_argument("--target", required=True)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--aspect", type=float, default=1.5,
                        help="target aspect; read it off the original PNG")
    parser.add_argument("--seed-translation", default=None, help="x,y,z")
    parser.add_argument("--seed-quaternion", default=None, help="w,x,y,z")
    parser.add_argument("--seed-fov", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=1_500_000,
                        help="thin the cloud for speed; structure survives")
    parser.add_argument("--out", default=None, help="write the fitted pose as JSON")
    args = parser.parse_args()

    size = (args.width, int(round(args.width / args.aspect)))
    cfg = load_config(args.config)
    c_data = cfg["data"]
    cloud = load_point_cloud(
        c_data["path"], palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"], colors=None,
        cache=True, cache_dir=c_data["cache_dir"], log=lambda *a: None)
    cloud = cloud.drop_void(c_data["drop_void"], log=lambda *a: None)
    cloud = cloud.subsample(c_data["subsample"], c_data["seed"])
    if c_data["center"]:
        cloud.center()

    colors = cloud.colors[args.layer].astype(np.float64) / 255.0
    gray = (0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2])
    pos = np.asarray(cloud.pos, dtype=np.float64)
    if len(pos) > args.max_points:
        keep = np.sort(np.random.default_rng(0).choice(
            len(pos), args.max_points, replace=False))
        pos, gray = pos[keep], gray[keep]
    print(f"cloud {len(pos):,} points, target {size[0]}x{size[1]}")

    target = load_target(args.target, size)

    seed_t = (np.array([float(v) for v in args.seed_translation.split(",")])
              if args.seed_translation else np.array(cfg["camera"]["translation"], float))
    seed_q = (np.array([float(v) for v in args.seed_quaternion.split(",")])
              if args.seed_quaternion else np.array(cfg["camera"]["quaternion"], float))
    seed_f = args.seed_fov if args.seed_fov is not None else cfg["camera"]["fov_x_deg"]

    def unpack(p):
        translation = seed_t + p[:3] * scale_t
        quat = qmul(seed_q, rotvec_to_quat(p[3:6] * scale_r))
        fov = seed_f * math.exp(p[6] * 0.25)
        return translation, quat, fov

    def cost(p):
        translation, quat, fov = unpack(p)
        if not (5.0 < fov < 140.0):
            return 1.0
        image, hit = project(pos, gray, translation, quat, fov, size)
        if hit.mean() < 0.2:
            return 1.0
        return -ncc(image, target)

    scale_t = max(pos[:, :2].max(0) - pos[:, :2].min(0)) * 0.25
    scale_r = 0.35

    # A free 6-DOF search from a wrong viewpoint never finds the right basin --
    # the objective is flat once the frames stop overlapping. Scan a look-at
    # parameterisation first (where on a hemisphere the camera sits, aimed at
    # the cloud), which covers the plausible space with far fewer evaluations
    # and is what a person moving a viewport actually does.
    centre = np.array([pos[:, 0].mean(), pos[:, 1].mean(),
                       np.percentile(pos[:, 2], 20)])
    span = float(max(pos[:, :2].max(0) - pos[:, :2].min(0)))

    def look_at(eye, target, roll=0.0):
        forward = target - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        # Blender cameras look down -Z with +Y up
        R = np.stack([right, true_up, -forward], axis=1)
        q = matrix_to_quat(R)
        return qmul(q, rotvec_to_quat(np.array([0.0, 0.0, roll])))

    scan = []
    for azimuth in np.arange(0, 360, 10.0):
        for elevation in (15.0, 25.0, 35.0, 50.0, 65.0):
            for radius in (0.6, 0.9, 1.3, 1.8):
                for fov in (45.0, 60.0, 73.0, 90.0):
                    a, e = math.radians(azimuth), math.radians(elevation)
                    d = span * radius
                    eye = centre + np.array([d * math.cos(e) * math.cos(a),
                                             d * math.cos(e) * math.sin(a),
                                             d * math.sin(e)])
                    quat = look_at(eye, centre)
                    image, hit = project(pos, gray, eye, quat, fov, size)
                    if hit.mean() < 0.25:
                        continue
                    scan.append((ncc(image, target), eye, quat, fov))
    scan.sort(key=lambda r: -r[0])
    print(f"coarse scan: {len(scan)} viable poses, best NCC {scan[0][0]:.4f}, "
          f"top-5 {[round(s[0], 3) for s in scan[:5]]}")

    best = None
    for rank, (score, eye, quat, fov) in enumerate(scan[:args.restarts]):
        seed_t, seed_q, seed_f = eye, quat, fov
        result = minimize(cost, np.zeros(7), method="Powell",
                          options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
        tag = ""
        if best is None or result.fun < best[0].fun:
            best = (result, seed_t, seed_q, seed_f)
            tag = "  <-- best"
        print(f"  seed {rank:2d} (coarse {score:.3f}) -> NCC {-result.fun:.4f}{tag}")

    result, seed_t, seed_q, seed_f = best
    translation, quat, fov = unpack(result.x)
    best = result
    if quat[0] < 0:
        quat = -quat
    print(f"\nfitted NCC {-best.fun:.4f}")
    print(f"  translation {[round(float(v), 6) for v in translation]}")
    print(f"  quaternion  {[round(float(v), 7) for v in quat]}")
    print(f"  fov_x_deg   {fov:.4f}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"translation": [float(v) for v in translation],
                       "quaternion": [float(v) for v in quat],
                       "fov_x_deg": float(fov), "ncc": float(-best.fun)}, handle, indent=1)
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
