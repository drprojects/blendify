"""Per-layer legend inserts for the exploded-stack shot.

Which legend is on screen is derived from the choreography file, not from a
hand-written schedule: the newest slab that has appeared IS the layer being
introduced, and its own `alpha` ramp drives the legend, so caption and slab
arrive together by construction and cannot drift apart when the pacing changes.

This is a library first. `assemble_video.py` calls `schedule()` and
`draw_legend()` *after* the background plate and the saturation boost, which is
the only correct place for them: baking a legend into the rendered frames would
send the class swatches through the same 2x chroma scale as the point cloud, and
a legend whose colours no longer match the paper's is worse than no legend.

The CLI is a preview path — it flattens onto the plate and writes opaque PNGs,
which is fine for eyeballing placement but is not what gets encoded.

    python scripts/compose_legends.py --frames <dir> --path stack.json \
        --legends <dir> --layers rgb,strength,... --out <dir>
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from overlay import frosted_panel, fit_box

CAP_RATIO = 0.85
EXIT = 0.35             # seconds the outgoing legend takes to clear the anchor
HOLD = 0.85             # seconds the last layer's legend holds, in seconds
REFERENCE_WIDTH = 1920.0    # legends are authored for a 1080p frame


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def load_legends(directory, names, frame_width):
    """Legend art keyed by layer name, scaled to `frame_width`.

    Legends are authored at 2x of 1080p, so a preview render gets the same
    apparent size as the final and placement decisions transfer.
    """
    scale = frame_width / REFERENCE_WIDTH / 2.0
    art = {}
    for name in names:
        path = osp.join(directory, f"legend_{name}.png")
        if not osp.exists(path):
            continue
        image = Image.open(path).convert("RGBA")
        size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
        art[name] = np.asarray(image.resize(size, Image.LANCZOS), np.float64) / 255.0
    return art


def schedule(path_json, names, fps=12, outro=0.7):
    """Per-frame `[(layer_name, weight)]`, at most one entry non-zero.

    The fade-in *is* the slab's own alpha ramp, so the caption cannot lag the
    tile it labels however the pacing is retimed. That leaves no room for a
    cross-fade, and none is wanted: the legends differ in width by a factor of
    ten, so two capsules alive at once are two different-sized shapes sharing
    one anchor, which reads as a glitch. The outgoing legend is therefore
    scheduled *backwards* from the frame the next slab appears, reaching zero
    exactly as its successor starts.
    """
    poses = json.load(open(path_json))["poses"]
    alphas = [[s.get("alpha", 1.0 if s.get("visible", True) else 0.0)
               for s in p["slabs"]] for p in poses]
    n_frames = len(alphas)
    n_slabs = len(alphas[0])

    appear = [next((i for i, row in enumerate(alphas) if row[k] > 0.02), None)
              for k in range(n_slabs)]

    # The legend leaves once the last slab has had the same beat as every other
    # layer, then the survey of the finished stack is unobstructed. Anchoring
    # this on the frame the last slab *appears* would give it a quarter of a
    # second and never full opacity — the one layer whose label is least
    # guessable would be the one you cannot read.
    last = n_slabs - 1
    settled = next((i for i, row in enumerate(alphas) if row[last] > 0.99),
                   n_frames - 1)
    hold = max(int(round(HOLD * fps)), 1)
    reveal_end = min(settled + hold, n_frames - 1)
    tail = max(int(outro * fps), 1)
    exit_frames = max(int(round(EXIT * fps)), 1)

    # (frame the legend is gone, frames it takes to get there)
    departure = []
    for k in range(n_slabs):
        nxt = next((appear[j] for j in range(k + 1, n_slabs)
                    if appear[j] is not None), None)
        departure.append((nxt, exit_frames) if nxt is not None
                         else (reveal_end + tail, tail))

    out = []
    for index, row in enumerate(alphas):
        frame = []
        for k, a in enumerate(row):
            if appear[k] is None or index < appear[k] or k >= len(names):
                continue
            stop, span = departure[k]
            w = min(a, 1.0 - smoothstep((index - (stop - span)) / span))
            if w > 0.01:
                frame.append((names[k], w))
        out.append(frame)
    return out, reveal_end


def draw_legend(frame, art, weight, position="top-right", pad_x=42, pad_y=20,
                whiten=0.88):
    """Draw one legend on a frosted capsule. `frame` is opaque RGB float."""
    if art is None or weight <= 0.01:
        return frame
    height, width = frame.shape[:2]
    scale = width / REFERENCE_WIDTH
    lh, lw = art.shape[:2]
    box = fit_box((width, height), (lw, lh), position,
                  pad=(int(pad_x * scale), int(pad_y * scale)),
                  shape="capsule", cap_ratio=CAP_RATIO)
    panelled = frosted_panel(frame, box, whiten=whiten, blur=13.0, shadow=0.20,
                             shape="capsule", cap_ratio=CAP_RATIO)
    frame = frame * (1 - weight) + panelled * weight
    x = int((box[0] + box[2]) / 2 - lw / 2)
    y = int((box[1] + box[3]) / 2 - lh / 2)
    patch = frame[y:y + lh, x:x + lw]
    a = art[..., 3:4] * weight
    frame[y:y + lh, x:x + lw] = patch * (1 - a) + art[..., :3] * a
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--legends", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--position", default="top-right",
                        choices=("top-right", "top-left",
                                 "bottom-left", "bottom-right"))
    parser.add_argument("--pad-x", type=int, default=42,
                        help="capsule end-cap clearance, in 1080p pixels")
    parser.add_argument("--pad-y", type=int, default=20)
    parser.add_argument("--whiten", type=float, default=0.88)
    parser.add_argument("--outro", type=float, default=0.7)
    parser.add_argument("--background", default="gradient")
    args = parser.parse_args()

    from assemble_video import background

    names = [n.strip() for n in args.layers.split(",") if n.strip()]
    files = sorted(glob.glob(osp.join(args.frames, "*.png")))
    plan, reveal_end = schedule(args.path, names, args.fps, args.outro)
    n = min(len(files), len(plan))
    os.makedirs(args.out, exist_ok=True)

    size = Image.open(files[0]).size
    art = load_legends(args.legends, names, size[0])
    missing = [n_ for n_ in names if n_ not in art]
    print(f"  {len(art)} legends" + (f", missing {missing}" if missing else "")
          + f", reveal ends at frame {reveal_end}")
    plate = background(size, args.background)

    for index in range(n):
        rgba = np.asarray(Image.open(files[index]).convert("RGBA"), np.float64) / 255.0
        a = rgba[..., 3:4]
        frame = rgba[..., :3] * a + plate * (1 - a)
        for name, weight in plan[index]:
            frame = draw_legend(frame, art.get(name), weight, args.position,
                                args.pad_x, args.pad_y, args.whiten)
        Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(
            osp.join(args.out, f"frame_{index:05d}.png"))
        if index % 40 == 0:
            print(f"  {index}/{n}")
    print(f"  {n} preview frames -> {args.out}")


if __name__ == "__main__":
    main()
