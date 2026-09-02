"""Choreograph the exploded layer stack: slab positions and camera, per frame.

The sequence has three acts:

  1. **Reveal.** Layers appear one at a time, bottom to top, each one *at its own
     slot* -- see below -- and the camera climbs to stay above the highest one
     that has appeared. Without that climb the later slabs are seen edge-on from
     below, which is the main way this kind of shot goes wrong.
  2. **Survey.** The camera pulls back and orbits the finished stack.
  3. **Fuse.** Every slab sinks back into the bottom one and switches off, so
     the RGB layer is left alone -- all those annotations describing one place.

Slab spacing is derived from the cloud's own z-range rather than guessed: two
slabs a fixed distance apart will interpenetrate wherever the terrain is tall,
so the gap is the full height of the cloud plus breathing space.

Three things this file exists to get right:

**A new layer must not fly up from the layer below.** An earlier version started
slab *k* at slot *k-1* and translated it a full gap into place. It reads as the
annotation being *extruded out of* the layer beneath it, which is the opposite
of the claim -- these are independent descriptions of one place, not derivations
of one another. Each slab now appears directly at its slot and only settles
through `--rise` of a slot (a nudge, ~10%) along the fan axis, which gives the
eye something to latch onto without implying provenance. The collapse in act 3
still travels the full distance, because there the merging *is* the point.

**The camera must be continuous.** Every quantity the pose is built from is a
smooth function of the global frame index, and the near->wide framing is a
single blend weight `w` that is exactly 0 at the last reveal frame and exactly 1
at the first fuse frame. In particular the camera does *not* track the real
`max(z)` of the visible slabs: with slabs popping in at their final slot that
number steps by a whole gap and the camera would jerk with it. It tracks
`cam_level`, a ramp that rises through the same slots continuously.

**The reveal must not be stop-and-go.** An earlier version eased the camera into
each new level and then held there for the rest of that layer's beat. On paper
that was "a beat per layer"; on screen it is one acceleration and one stop every
1.5 s, which the eye reads as juddering rather than as rhythm. The camera now
climbs the stacking axis at **constant speed** for the whole reveal -- an
elevator with a window, the slabs appearing above and being passed one by one --
with exactly two accelerations in the act: out of rest at the very start, and
back into rest as the survey takes over. `--ease-in` / `--ease-out` set how much
of the climb those two cost; everything between them is dead linear. The slabs
themselves still get their per-layer settle, because that motion is *local* to a
slab and does not move the frame.

    python scripts/explode_path.py --config <cfg> --layers rgb,semantic,... \
        --out path.json
"""
import argparse
import json
import math
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from camera_path import look_at, smoothstep
from figlib.config import load_config
from figlib.data import load_point_cloud


def cruise(u, ease_in, ease_out):
    """Position along a move whose *speed* is flat except at the two ends.

    `smoothstep` eases position, which means it accelerates and decelerates
    continuously -- fine for a short move, wrong for a long climb past nine
    slabs, where the eye wants to be carried rather than pushed and pulled.
    Here the smoothstep is applied to the *speed* instead: it ramps up over the
    first `ease_in` of the move, stays flat, and ramps down over the last
    `ease_out`. Position is its integral, so `cruise` is C1 and dead linear in
    between -- the elevator-window read.

    Returns 0 at u=0 and 1 at u=1 for any split, so changing the eases changes
    the feel of the ends and nothing else.
    """
    u = min(max(float(u), 0.0), 1.0)
    a, b = max(float(ease_in), 1e-9), max(float(ease_out), 1e-9)
    if a + b > 1.0:                       # degenerate split: no cruise segment
        a, b = a / (a + b), b / (a + b)
    # Area under smoothstep over [0, x]; the full ramp is worth half a flat one.
    area = lambda x: x ** 3 - 0.5 * x ** 4
    total = 1.0 - 0.5 * a - 0.5 * b
    if u < a:
        travelled = a * area(u / a)
    elif u < 1.0 - b:
        travelled = 0.5 * a + (u - a)
    else:
        travelled = total - b * area((1.0 - u) / b)
    return travelled / total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--reveal", type=float, default=2.90,
                        help="seconds per layer during the reveal. This has to "
                             "cover reading the layer's legend AND looking at "
                             "the scene under it, which is why it is not the "
                             "shortest interval that reads as a beat.")
    parser.add_argument("--hold-first", type=float, default=1.90,
                        help="seconds the base layer holds alone before the "
                             "stack starts building. Without it the first layer "
                             "is the only one that never gets a beat -- it is "
                             "already on screen at frame 0, so the second slab "
                             "arriving on schedule cuts its turn short -- and "
                             "the shot opens by piling on rather than by "
                             "establishing what is being piled on.")
    parser.add_argument("--survey", type=float, default=4.0)
    parser.add_argument("--fuse", type=float, default=2.8)
    parser.add_argument("--outro", type=float, default=5.0,
                        help="seconds of closing push-in on the fused RGB tile")
    parser.add_argument("--outro-fill", type=float, default=0.50,
                        help="fraction of the tile framed at the very end; "
                             "smaller is closer, cf --reveal-fill")
    parser.add_argument("--breathing", type=float, default=1.30,
                        help="extra gap between slabs, as a fraction of cloud height")
    parser.add_argument("--fan", type=float, default=0.10,
                        help="sideways slide per slab, fraction of tile span")
    parser.add_argument("--rise", type=float, default=0.10,
                        help="how far above its own slot a slab appears, as a "
                             "fraction of one slot. Small on purpose: it is a "
                             "settle, not a journey from the layer below")
    parser.add_argument("--appear", type=float, default=0.28,
                        help="fraction of a layer's beat to wait before it "
                             "appears, so the camera has already started "
                             "climbing towards the empty slot")
    parser.add_argument("--settle", type=float, default=0.90,
                        help="fraction of a layer's beat by which it has fully "
                             "settled into its slot")
    parser.add_argument("--ease-in", type=float, default=0.08,
                        help="fraction of the reveal spent accelerating out of "
                             "rest; the climb is constant-speed after it")
    parser.add_argument("--ease-out", type=float, default=0.18,
                        help="fraction of the reveal spent decelerating into "
                             "the survey, so the two acts hand over at rest")
    parser.add_argument("--reveal-fill", type=float, default=0.66,
                        help="fraction of the tile framed during the reveal; "
                             "smaller = closer to the points")
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--aspect", type=float, default=16 / 9)
    parser.add_argument("--azimuth", type=float, default=40.0)
    parser.add_argument("--elevation", type=float, default=33.0)
    parser.add_argument("--sweep", type=float, default=85.0)
    args = parser.parse_args()

    names = [n.strip() for n in args.layers.split(",") if n.strip()]
    n_layers = len(names)
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
    pos = np.asarray(cloud.pos, float)

    span = float(max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])))
    height = float(np.ptp(pos[:, 2]))
    gap = height * (1.0 + args.breathing)
    fan = span * args.fan
    base_z = float(pos[:, 2].min())
    mid_x, mid_y = float(pos[:, 0].mean()), float(pos[:, 1].mean())
    top_slot = gap * (n_layers - 1)
    print(f"cloud {len(pos):,} pts, span {span:.0f} m, height {height:.0f} m")
    print(f"  gap {gap:.0f} m (height + {args.breathing:.0%} breathing) -> "
          f"stack top {top_slot:.0f} m over {n_layers} layers")
    print(f"  slabs appear {gap * args.rise:.0f} m above their slot and settle in")

    reveal_n = int(round(args.reveal * args.fps))
    hold_n = int(round(args.hold_first * args.fps))
    survey_start = hold_n + reveal_n * (n_layers - 1)
    fuse_start = survey_start + int(round(args.survey * args.fps))
    fuse_end = fuse_start + int(round(args.fuse * args.fps))
    total = fuse_end + int(round(args.outro * args.fps))
    fov_y = 2 * math.atan(math.tan(math.radians(args.fov) / 2) / args.aspect)
    half_v = math.tan(fov_y / 2)

    # The two framings the shot blends between, as distances. Both are constants:
    # blending *them* (rather than recomputing a distance from a changing
    # subject) is what keeps the pull-back free of any step at the act boundary.
    d_near = 1.20 * span * args.reveal_fill / (2 * half_v)
    d_wide = 1.25 * max(span, top_slot + height * 1.6) / (2 * half_v)
    # The closing framing. Once everything has collapsed into the RGB tile, the
    # wide distance that framed a nine-slab column leaves one thin slab in the
    # middle of an empty frame, so the shot has to come back in to end on the
    # data rather than on the space the data used to occupy.
    d_close = 1.20 * span * args.outro_fill / (2 * half_v)

    poses, centres = [], []
    for index in range(total):
        # --- continuous phase scalars -------------------------------------
        # `step` counts layer beats and drives the slabs, so they keep arriving
        # on an even pulse. `cam_level` is the height the camera is centred on,
        # in slots, and is deliberately NOT tied to `step`: it is one
        # constant-speed climb across the whole act. Both are continuous in
        # `index` and both reach n_layers-1 by the last reveal frame, so the
        # survey starts from precisely where the reveal stopped.
        if index < survey_start:
            phase = "reveal"
            # The opening hold freezes the beat counter, not the clock, so the
            # slabs still arrive on an even pulse once it ends.
            beat = max(index - hold_n, 0)
            step = beat / reveal_n
            cam_level = (n_layers - 1) * cruise(
                beat / max(survey_start - hold_n, 1), args.ease_in, args.ease_out)
        elif index < fuse_start:
            phase = "survey"
            step = cam_level = float(n_layers - 1)
        elif index < fuse_end:
            phase = "fuse"
            step = cam_level = float(n_layers - 1)
        else:
            phase = "outro"
            step = cam_level = float(n_layers - 1)

        # 0 while revealing, eased 0->1 across the survey, 1 through the fuse.
        w = float(smoothstep(np.clip(
            (index - survey_start) / max(fuse_start - survey_start, 1), 0.0, 1.0)))
        # 0 until the fuse, then eased 0->1 as the stack collapses.
        collapse = float(smoothstep(np.clip(
            (index - fuse_start) / max(fuse_end - fuse_start, 1), 0.0, 1.0))) \
            if index >= fuse_start else 0.0

        # --- slabs ---------------------------------------------------------
        slabs = []
        for k in range(n_layers):
            if phase in ("fuse", "outro"):
                # The one journey along the fan axis that is meant to read as a
                # journey: every slab retraces the stack back into the RGB one.
                s = k * (1.0 - collapse)
                x, z = fan * s, gap * s
                visible, alpha = (k == 0 or collapse < 0.995), 1.0
            elif k == 0:
                x, z, visible, alpha = 0.0, 0.0, True, 1.0
            else:
                # Slab k owns beat k-1. It shows up `appear` into that beat,
                # directly at slot k (plus a `rise` nudge along the fan axis)
                # and settles by `settle`.
                local_k = step - (k - 1)
                p = float(np.clip((local_k - args.appear)
                                  / max(args.settle - args.appear, 1e-6), 0.0, 1.0))
                t = float(smoothstep(p))
                s = k + args.rise * (1.0 - t)
                x, z = fan * s, gap * s
                visible = local_k >= args.appear - 1e-9
                alpha = t
            slabs.append({"x": float(x), "z": float(z),
                          "visible": bool(visible), "alpha": float(alpha)})

        # --- camera ---------------------------------------------------------
        stack_top = gap * cam_level * (1.0 - collapse)
        stack_x = fan * cam_level * (1.0 - collapse)
        # Near: sit on the layer being revealed, because framing the whole stack
        # from the start makes each new slab a distant sliver. Wide: take in the
        # column. One blend, so there is no seam between them.
        centre_near = np.array([mid_x + stack_x, mid_y,
                                base_z + stack_top + height / 2])
        centre_wide = np.array([mid_x + stack_x / 2, mid_y,
                                base_z + stack_top / 2 + height / 2])
        centre = centre_near + (centre_wide - centre_near) * w
        centres.append(centre)
        distance = d_near + (d_wide - d_near) * w
        if phase == "outro":
            push = float(smoothstep((index - fuse_end) / max(total - fuse_end, 1)))
            distance = d_wide + (d_close - d_wide) * push

        # Constant angular rate for the same reason the climb is: a smoothstep
        # here would swell to 1.5x speed mid-shot, and combined with a flat
        # climb that is a camera that mysteriously speeds up and slows down.
        # Eased at the two ends of the *whole shot* only, so it leaves and
        # returns to rest.
        # The sweep completes by the end of the fuse and holds through the
        # outro. `cruise` decelerates into rest at its end, so holding after it
        # adds no velocity step: the orbit simply stops and the push-in takes
        # over as the only motion, which is what makes the last beat read as an
        # ending rather than as more of the same.
        angle = math.radians(args.azimuth + args.sweep * cruise(
            min(index / max(fuse_end - 1, 1), 1.0), args.ease_in, args.ease_out))
        elev = math.radians(args.elevation)
        eye = centre + np.array([distance * math.cos(elev) * math.cos(angle),
                                 distance * math.cos(elev) * math.sin(angle),
                                 distance * math.sin(elev)])
        # Stay above the highest slab that has appeared: looking up at a new
        # layer from underneath is exactly what makes these shots read badly.
        # `stack_top` is continuous, so this max cannot introduce a jump.
        eye[2] = max(eye[2], base_z + stack_top + height * 0.85)
        poses.append({"t": index / args.fps,
                      "position": [float(v) for v in eye],
                      "quaternion": [float(v) for v in look_at(eye, centre)],
                      "slabs": slabs})

    with open(args.out, "w") as handle:
        json.dump({"fps": args.fps, "layers": names, "poses": poses}, handle)
    deltas = np.linalg.norm(np.diff(np.array(
        [p["position"] for p in poses]), axis=0), axis=1)
    print(f"  {total} frames = {total / args.fps:.1f} s "
          f"(hold {hold_n}, reveal {survey_start - hold_n}, survey {fuse_start - survey_start}, "
          f"fuse {fuse_end - fuse_start}, outro {total - fuse_end}) "
          f"-> {args.out}")
    print(f"  outro pushes in {d_wide:.0f} -> {d_close:.0f} m")
    print(f"  camera step: max {deltas.max():.1f} m at frame "
          f"{int(deltas.argmax())}, median {np.median(deltas):.1f} m")
    # The elevator check: the frame's travel along the stacking axis must be
    # flat through the cruise, otherwise the reveal reads as stop-and-go.
    axis = np.array([fan, 0.0, gap])
    axis = axis / np.linalg.norm(axis)
    climb = np.diff(np.asarray(centres)[:survey_start], axis=0) @ axis
    lo = hold_n + int(args.ease_in * (survey_start - hold_n)) + 1
    hi = hold_n + int((1.0 - args.ease_out) * (survey_start - hold_n)) - 1
    cruise_v = climb[lo:hi]
    print(f"  reveal climb along the fan axis: {cruise_v.mean():.2f} m/frame "
          f"+- {cruise_v.std():.4f} over the cruise (frames {lo}-{hi}), "
          f"max deviation {np.abs(cruise_v - cruise_v.mean()).max():.4f} m")


if __name__ == "__main__":
    main()
