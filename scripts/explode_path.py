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
`cam_level`, an eased ramp that rises through the same slots continuously.

**Nothing should feel rushed.** Every act is eased, the orbit is eased over the
whole shot so it starts and ends at rest, and the camera reaches each new level
at ~70% of that layer's beat and then holds -- the pause is what makes the
stack legible.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--reveal", type=float, default=1.45,
                        help="seconds per layer during the reveal")
    parser.add_argument("--survey", type=float, default=4.0)
    parser.add_argument("--fuse", type=float, default=2.8)
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
    parser.add_argument("--lead", type=float, default=0.70,
                        help="fraction of a layer's beat over which the camera "
                             "climbs one slot; it then holds, which is the beat")
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
    survey_start = reveal_n * (n_layers - 1)
    fuse_start = survey_start + int(round(args.survey * args.fps))
    total = fuse_start + int(round(args.fuse * args.fps))
    fov_y = 2 * math.atan(math.tan(math.radians(args.fov) / 2) / args.aspect)
    half_v = math.tan(fov_y / 2)

    # The two framings the shot blends between, as distances. Both are constants:
    # blending *them* (rather than recomputing a distance from a changing
    # subject) is what keeps the pull-back free of any step at the act boundary.
    d_near = 1.20 * span * args.reveal_fill / (2 * half_v)
    d_wide = 1.25 * max(span, top_slot + height * 1.6) / (2 * half_v)

    poses = []
    for index in range(total):
        # --- continuous phase scalars -------------------------------------
        # `step` counts layer beats; `cam_level` is the height the camera is
        # centred on, in slots. Both are continuous in `index`, and both reach
        # exactly n_layers-1 on the last reveal frame, so survey starts from
        # precisely where reveal stopped.
        if index < survey_start:
            phase = "reveal"
            step = index / reveal_n
            beat = int(min(step, n_layers - 2))
            local = step - beat
            cam_level = beat + smoothstep(min(local / args.lead, 1.0))
        else:
            phase = "survey" if index < fuse_start else "fuse"
            step = float(n_layers - 1)
            cam_level = float(n_layers - 1)

        # 0 while revealing, eased 0->1 across the survey, 1 through the fuse.
        w = float(smoothstep(np.clip(
            (index - survey_start) / max(fuse_start - survey_start, 1), 0.0, 1.0)))
        # 0 until the fuse, then eased 0->1 as the stack collapses.
        collapse = float(smoothstep(np.clip(
            (index - fuse_start) / max(total - fuse_start, 1), 0.0, 1.0))) \
            if phase == "fuse" else 0.0

        # --- slabs ---------------------------------------------------------
        slabs = []
        for k in range(n_layers):
            if phase == "fuse":
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
        distance = d_near + (d_wide - d_near) * w

        # Eased over the whole shot, so the orbit starts and ends at rest
        # instead of snapping into motion on frame 0.
        angle = math.radians(args.azimuth + args.sweep
                             * float(smoothstep(index / max(total - 1, 1))))
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
          f"(reveal {survey_start}, survey {fuse_start - survey_start}, "
          f"fuse {total - fuse_start}) -> {args.out}")
    print(f"  camera step: max {deltas.max():.1f} m at frame "
          f"{int(deltas.argmax())}, median {np.median(deltas):.1f} m")


if __name__ == "__main__":
    main()
