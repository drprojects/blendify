"""Generate camera paths for video shots, guaranteed to clear the terrain.

A shot is a named move over one tile. The path is derived from the cloud's own
geometry rather than hand-placed: the camera is framed so the tile fills a
chosen fraction of the frame at the config's field of view, which makes shots of
tiles with very different extents look consistent when cut together.

Two things this exists to get right:

**Never fly into the scene.** The camera height is checked against the DTM that
`figlib.ground` already builds and lifted wherever it would come within
`clearance` of the surface. On a 400 m-relief alpine tile a path that looks fine
over the valley floor will otherwise pass straight through a ridge.

**Never start or stop abruptly.** Every move is eased (smoothstep), so the
camera accelerates out of rest and settles rather than snapping. Linear motion
is the single clearest tell of an amateur render.

Paths are written as JSON — a list of {t, position, quaternion} — which
`examples/00_custom.py --frames` consumes. Keeping them as data means a path can
be inspected, tweaked and re-rendered without touching the figure script.
"""
import argparse
import json
import math
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from figlib import ground as ground_module
from figlib.config import load_config
from figlib.data import load_point_cloud


def smoothstep(u):
    """Ease in and out. u in [0,1]."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Blender camera quaternion [w,x,y,z]: looks down -Z with +Y up."""
    eye = np.asarray(eye, float)
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = np.stack([right, true_up, -forward], axis=1)

    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s,
             (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                 (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                 (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def frame_distance(extent, fov_x_deg, fill):
    """Distance at which `extent` metres spans `fill` of the frame width.

    `fill` above 1.0 means the tile is wider than the frame, i.e. its edges fall
    outside it. Bigger fill = closer camera.
    """
    return extent / (2.0 * fill * math.tan(math.radians(fov_x_deg) / 2.0))


def frame_footprint(eye, target, fov_x_deg, aspect, plane_z):
    """Where the four frame corners land on the horizontal plane `plane_z`.

    Returns None if any corner ray points away from the plane (it looks at the
    sky), which is itself a reason to reject a pose.
    """
    eye = np.asarray(eye, float)
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        return None
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    half_x = math.tan(math.radians(fov_x_deg) / 2.0)
    half_y = half_x / aspect
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            ray = forward + right * (sx * half_x) + up * (sy * half_y)
            if ray[2] >= -1e-6:                 # parallel to, or above, the plane
                return None
            t = (plane_z - eye[2]) / ray[2]
            corners.append(eye + ray * t)
    return np.asarray(corners)


def is_interior(eye, target, fov_x_deg, aspect, plane_z, bounds, margin):
    """True when the whole frame lands inside the tile, with a margin to spare.

    This is what keeps a cut edge out of shot. Checking the frame footprint
    directly is far more reliable than picking a distance by eye, because the
    far edge of an oblique view runs away much faster than the near one.
    """
    corners = frame_footprint(eye, target, fov_x_deg, aspect, plane_z)
    if corners is None:
        return False
    (x0, y0), (x1, y1) = bounds
    return bool((corners[:, 0] > x0 + margin).all()
                and (corners[:, 0] < x1 - margin).all()
                and (corners[:, 1] > y0 + margin).all()
                and (corners[:, 1] < y1 - margin).all())


def build(cfg, move="orbit", duration=4.5, fps=30, azimuth=35.0, elevation=32.0,
          fill=(1.05, 1.45), sweep=18.0, clearance=45.0, target_z_percentile=25.0,
          aspect=16 / 9, interior=True, edge_margin=15.0, fov_x=None,
          fly_travel=0.85, fly_altitude=140.0, fly_look_ahead=0.55,
          look_mode="inward", ease="both", reverse=False, target=None,
          spiral_override=None, log=print):
    """Poses for one shot over the tile described by `cfg`.

    `fill` is (start, end) as a fraction of frame width, and bigger means
    closer, so the default moves IN. Values above 1 put the tile's cut edges
    outside the frame.

    With `interior`, the elevation is raised until the frame footprint sits
    inside the tile for every frame of the shot — the camera looks down enough
    that no edge is ever in view. It is solved once for the whole shot and
    applied uniformly, because varying it per frame would show up as a drift in
    the move.
    """
    c_data = cfg["data"]
    cloud = load_point_cloud(
        c_data["path"], palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"], colors=None,
        cache=True, cache_dir=c_data["cache_dir"], log=lambda *a: None)
    cloud = cloud.drop_void(c_data["drop_void"], log=lambda *a: None)
    cloud = cloud.subsample(c_data["subsample"], c_data["seed"])
    if c_data["center"]:
        cloud.center()
    pos = np.asarray(cloud.pos, float)

    ground = ground_module.build(pos, cell=10.0, percentile=5.0)
    # Frame on the SHORT side. Several ROIs are 2:1 strips, and framing on the
    # long side puts the camera far enough back that the short side always
    # overruns the frame -- no elevation can then hide the cut edge.
    extent = float(min(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])))
    centre = np.array([pos[:, 0].mean(), pos[:, 1].mean(),
                       float(np.percentile(pos[:, 2], target_z_percentile))])
    target_end = None
    if target is not None and len(target) > 3:
        target, target_end = target[:3], target[3:6]
    if target is not None:
        # Orbit a subject, not the tile's centroid. A landmark is rarely at the
        # middle of its tile -- the Eiffel Tower sits 134 m off centre -- and
        # aiming at the centroid puts it at the frame edge, seen from above.
        centre = np.asarray(target, dtype=float)
    fov = fov_x if fov_x is not None else cfg["camera"]["fov_x_deg"]
    log(f"  extent {extent:.0f} m, target {centre.round(1).tolist()}, fov {fov}")

    bounds = ((float(pos[:, 0].min()), float(pos[:, 1].min())),
              (float(pos[:, 0].max()), float(pos[:, 1].max())))
    n = max(int(round(duration * fps)), 2)

    long_extent = float(max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])))
    half_short = extent / 2.0

    # Terrain floor as a SMOOTH function of the parameter, not a per-frame max
    # over a sliding window. The window version steps every time a ridge enters
    # or leaves it, which is exactly the bumpy bounce that made the flyover
    # unwatchable. Here the whole terrain is reduced to one high percentile per
    # radius band, then smoothed, so the altitude curve has no corners.
    def terrain_ceiling(radii):
        out = []
        for r in radii:
            mask = (np.abs(pos[:, 0] - centre[0]) < max(r, 1.0)) & \
                   (np.abs(pos[:, 1] - centre[1]) < max(r, 1.0))
            band = pos[mask, 2] if mask.any() else pos[:, 2]
            out.append(float(np.percentile(band, 99.0)))
        out = np.asarray(out)
        if len(out) >= 5:                      # light smoothing, edge-preserving
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
            kernel /= kernel.sum()
            out = np.convolve(np.pad(out, 2, mode="edge"), kernel, mode="valid")
        return out

    SPIRALS = {
        # (azimuth sweep deg, radius start, radius end, elev start, elev end)
        # radii are multiples of half the tile's SHORT side, so the camera ends
        # over the tile rather than outside it looking at a cut edge.
        "spiral_in":   (150.0, 2.30, 0.80, 58.0, 26.0),
        "descend_arc": (55.0,  2.00, 0.70, 62.0, 20.0),
        "orbit_hold":  (130.0, 1.55, 1.15, 46.0, 30.0),
        "crane_down":  (18.0,  2.40, 0.95, 68.0, 18.0),
    }

    def ease_u(index):
        """Timing curve.

        `both` (smoothstep) starts and stops at rest, which is right for the
        first and last shot of a sequence. It is WRONG in the middle: every shot
        decelerating to a halt before the cut is what makes a montage feel
        stop-start. Interior shots use `none` so the motion carries across the
        cut and the eye follows it into the next tile.
        """
        u = index / (n - 1)
        if ease == "none":
            return u
        if ease == "in":
            return u * u
        if ease == "out":
            return 1.0 - (1.0 - u) ** 2
        return smoothstep(u)

    def spiral_pose(index, spec):
        sweep_deg, r0, r1, e0, e1 = spec
        if spiral_override is not None:
            sweep_deg, r0, r1, e0, e1 = [
                spec[i] if spiral_override[i] is None else spiral_override[i]
                for i in range(5)]
        u = ease_u(index)
        r = half_short * (r0 + (r1 - r0) * u)
        e = math.radians(e0 + (e1 - e0) * u)
        a = math.radians(azimuth + sweep_deg * (u - 0.5))
        eye = centre + np.array([r * math.cos(e) * math.cos(a),
                                 r * math.cos(e) * math.sin(a),
                                 r * math.sin(e)])
        return eye

    def spiral_path(spec):
        """All poses for a spiral, with a smooth clearance floor applied."""
        eyes = np.array([spiral_pose(i, spec) for i in range(n)])
        ground_r = np.hypot(eyes[:, 0] - centre[0], eyes[:, 1] - centre[1])
        floor = terrain_ceiling(ground_r) + clearance
        # Enforce clearance smoothly: lift the whole curve by one constant if it
        # would dip, rather than bending it per frame and reintroducing bumps.
        lift = float(np.max(floor - eyes[:, 2]))
        if lift > 0:
            eyes[:, 2] += lift
        return eyes

    def spiral_targets(spec, eyes):
        """Where each spiral frame aims.

        `inward` holds the scene centre for the whole move, so the frame stays
        full of data and the shot closes in. `outward` swings the aim to the
        horizon over the back half, so the shot opens out as it drops -- more
        dramatic, and it puts the tile edge in the distance rather than the
        subject. Which one reads better depends on the cut that follows, which
        is why both exist.
        """
        if look_mode != "outward":
            if target_end is None:
                return [centre] * n
            # Let the aim travel: starting on the tower's crown and settling
            # onto the scene is what turns an orbit into a reveal -- close on
            # the subject, then opening out to show what it stands in.
            end = np.asarray(target_end, dtype=float)
            return [centre + (end - centre) * smoothstep(ease_u(i))
                    for i in range(n)]
        sweep_deg = spec[0]
        out = []
        for index in range(n):
            u = ease_u(index)
            a = math.radians(azimuth + sweep_deg * (u - 0.5))
            # tangent to the spiral, i.e. the direction of travel
            tangent = np.array([-math.sin(a), math.cos(a), 0.0])
            far = np.array([eyes[index, 0] + tangent[0] * extent * 1.8,
                            eyes[index, 1] + tangent[1] * extent * 1.8,
                            centre[2]])
            blend = smoothstep((u - 0.40) / 0.60)
            out.append(centre * (1 - blend) + far * blend)
        return out

    def fly_pose(index):
        """A low travelling shot across the tile.

        An orbit at 50 deg shows the plan of a tile but almost none of its
        relief: everything is seen from above, so hills read as texture. Flying
        low along a line, looking ahead and slightly down, puts the terrain in
        profile against the horizon -- which is the only way topography reads --
        and covers far more ground than an orbit of the same duration.
        """
        u = smoothstep(index / (n - 1))
        a = math.radians(azimuth)
        direction = np.array([math.cos(a), math.sin(a), 0.0])
        travel = long_extent * fly_travel
        here = centre + direction * (travel * (u - 0.5))

        # Hug the terrain: clear the highest ground within a look-ahead window,
        # so the camera rises over a ridge before reaching it rather than
        # clipping through it.
        window = centre + direction * (travel * (u - 0.5) + np.linspace(
            -0.05, 0.55, 24)[:, None] * travel)
        lateral = np.stack([window + np.array([-direction[1], direction[0], 0.0]) * o
                            for o in (-0.15 * travel, 0.0, 0.15 * travel)])
        floor = float(ground.sample(lateral.reshape(-1, 3)[:, :2]).max())
        eye = np.array([here[0], here[1], floor + fly_altitude])

        look = eye + direction * (travel * fly_look_ahead)
        look_z = float(ground.sample(look[None, :2])[0])
        target = np.array([look[0], look[1],
                           look_z - (fly_altitude * 0.0)])
        # Pitch is set by how far ahead we look relative to how high we are.
        return eye, target

    def pose_at(index, elev):
        u = smoothstep(index / (n - 1))
        d = frame_distance(extent, fov, fill[0] + (fill[1] - fill[0]) * u)
        if move == "orbit":
            a, e = math.radians(azimuth + sweep * (u - 0.5)), math.radians(elev)
        elif move == "push":
            a, e = math.radians(azimuth), math.radians(elev)
        elif move == "descend":
            a, e = math.radians(azimuth), math.radians(elev + 14.0 * (0.5 - u))
        else:
            raise SystemExit(f"unknown move {move!r}")
        eye = centre + np.array([d * math.cos(e) * math.cos(a),
                                 d * math.cos(e) * math.sin(a),
                                 d * math.sin(e)])
        samples = centre + (eye - centre) * np.linspace(0.0, 1.0, 24)[:, None]
        floor = ground.sample(samples[:, :2]).max() + clearance
        if eye[2] < floor:
            eye[2] = floor
        return eye

    if interior and move not in ("fly",) and move not in SPIRALS:
        chosen = None
        for elev in np.arange(elevation, 82.0, 1.5):
            if all(is_interior(pose_at(i, elev), centre, fov, aspect,
                               centre[2], bounds, edge_margin)
                   for i in range(n)):
                chosen = float(elev)
                break
        if chosen is None:
            log(f"  WARNING: no elevation up to 82 deg keeps the tile edges out "
                f"of frame; a cut edge will be visible. Raise fill or lower fov.")
            chosen = elevation
        elif chosen > elevation:
            log(f"  elevation {elevation:.0f} -> {chosen:.0f} deg to keep tile "
                f"edges out of frame")
        elevation = chosen

    spiral_eyes = spiral_path(SPIRALS[move]) if move in SPIRALS else None
    spiral_aim = (spiral_targets(SPIRALS[move], spiral_eyes)
                  if move in SPIRALS else None)
    poses = []
    for index in range(n):
        if move in SPIRALS:
            eye, target = spiral_eyes[index], spiral_aim[index]
        elif move == "fly":
            eye, target = fly_pose(index)
        else:
            eye, target = pose_at(index, elevation), centre
        poses.append({"t": index / fps,
                      "position": [float(v) for v in eye],
                      "quaternion": [float(v) for v in look_at(eye, target)]})
    if reverse:
        # Played backwards a descent becomes a rise. That is what lets two
        # shots meet at the SAME altitude across a cut instead of jumping from
        # a close low frame to a distant high one.
        poses = [{"t": i / fps, "position": p["position"],
                  "quaternion": p["quaternion"]}
                 for i, p in enumerate(reversed(poses))]
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--move", default="orbit",
                        choices=("orbit", "push", "descend", "fly",
                                 "spiral_in", "descend_arc", "orbit_hold",
                                 "crane_down"))
    parser.add_argument("--fly-travel", type=float, default=0.85,
                        help="fraction of the tile's long side to cross")
    parser.add_argument("--fly-altitude", type=float, default=140.0,
                        help="metres above the highest ground ahead")
    parser.add_argument("--fly-look-ahead", type=float, default=0.55,
                        help="how far ahead to aim, as a fraction of travel; "
                             "smaller looks down more")
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=32.0)
    parser.add_argument("--fill-start", type=float, default=1.05)
    parser.add_argument("--fill-end", type=float, default=1.45)
    parser.add_argument("--aspect", type=float, default=16 / 9)
    parser.add_argument("--fov", type=float, default=None,
                        help="override the config fov. A narrower lens (45 deg) "
                             "both looks more cinematic and shrinks the ground "
                             "footprint, which is what keeps tile edges out of frame")
    parser.add_argument("--show-edges", action="store_true",
                        help="allow tile edges in frame (the exploded view wants this)")
    parser.add_argument("--sweep", type=float, default=18.0)
    parser.add_argument("--clearance", type=float, default=45.0)
    parser.add_argument("--ease", default="both",
                        choices=("both", "in", "out", "none"),
                        help="use 'in' for the first shot, 'none' in the middle, "
                             "'out' for the last: a shot that stops before every "
                             "cut makes the montage feel stop-start")
    parser.add_argument("--sweep-deg", type=float, default=None)
    parser.add_argument("--radius0", type=float, default=None,
                        help="start radius, in half-short-sides of the tile")
    parser.add_argument("--radius1", type=float, default=None)
    parser.add_argument("--elev0", type=float, default=None)
    parser.add_argument("--elev1", type=float, default=None)
    parser.add_argument("--target", default=None,
                        help="x,y,z to orbit and aim at, instead of the tile "
                             "centre. Six values x,y,z,x2,y2,z2 move the aim "
                             "from the first point to the second over the shot")
    parser.add_argument("--reverse", action="store_true",
                        help="play the move backwards, turning a descent into a rise")
    parser.add_argument("--look", default="inward", choices=("inward", "outward"),
                        help="spiral aim: hold the scene, or swing to the horizon")
    args = parser.parse_args()

    cfg = load_config(args.config)
    poses = build(cfg, move=args.move, duration=args.duration, fps=args.fps,
                  azimuth=args.azimuth, elevation=args.elevation,
                  fill=(args.fill_start, args.fill_end), sweep=args.sweep,
                  clearance=args.clearance, aspect=args.aspect,
                  interior=not args.show_edges, fov_x=args.fov,
                  fly_travel=args.fly_travel, fly_altitude=args.fly_altitude,
                  fly_look_ahead=args.fly_look_ahead, look_mode=args.look,
                  ease=args.ease, reverse=args.reverse,
                  target=([float(v) for v in args.target.split(",")]
                          if args.target else None),
                  spiral_override=(args.sweep_deg, args.radius0, args.radius1,
                                   args.elev0, args.elev1))
    with open(args.out, "w") as handle:
        json.dump({"fps": args.fps, "poses": poses}, handle, indent=1)
    print(f"{len(poses)} poses -> {args.out}")


if __name__ == "__main__":
    main()
