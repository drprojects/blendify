"""Lay the paper's title over a rendered shot, for a cold open.

The title is compiled by LaTeX from the paper's own macros, so the wordmark,
the icon and the MALIBU highlight colour are the paper's rather than an
imitation. This only places it.

The scene is washed toward white behind the title and the wash lifts as the
title fades, so the shot opens on a title card and resolves into the landscape.
That wash is why the paper's black body text stays readable over an aerial
photograph without inventing a different colour scheme for the video.

    python scripts/compose_title.py --frames <dir> --title title.png --out <dir>
"""
import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from overlay import frosted_panel, fit_box

# Caps are laid OUTSIDE the padded text box, so the straight section IS
# that box and no glyph can land in a curved end. Both fit_box and
# frosted_panel must get the same ratio or the caps and the reserved
# clearance stop agreeing.
CAP_RATIO = 0.85


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--fade-in", type=float, default=0.8)
    parser.add_argument("--hold", type=float, default=2.4)
    parser.add_argument("--fade-out", type=float, default=1.2)
    parser.add_argument("--whiten", type=float, default=0.86,
                        help="opacity of the insert; higher = more readable type")
    parser.add_argument("--title-y", type=float, default=0.42,
                        help="vertical placement, 0=top 1=bottom. Move it off "
                             "the subject rather than over it")
    parser.add_argument("--shape", default="capsule",
                        choices=("rect", "ellipse", "capsule"))
    parser.add_argument("--title-width", type=float, default=0.52,
                        help="title width as a fraction of the frame")
    args = parser.parse_args()

    files = sorted(glob.glob(osp.join(args.frames, "*.png")))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")
    os.makedirs(args.out, exist_ok=True)

    title = Image.open(args.title).convert("RGBA")
    # The LaTeX minipage is far wider than the type, so ~53% of the PNG is
    # transparent margin. Sizing a panel to the uncropped image bakes that in
    # and makes the insert 1.7x wider than it needs to be -- crop to the ink.
    ink = title.getbbox()
    if ink:
        title = title.crop(ink)
    probe = Image.open(files[0])
    width, height = probe.size
    tw = int(width * args.title_width)
    th = int(round(tw * title.height / title.width))
    title = title.resize((tw, th), Image.LANCZOS)
    title_rgba = np.asarray(title, np.float64) / 255.0

    n = len(files)
    for index, path in enumerate(files):
        t = index / args.fps
        if t < args.fade_in:
            alpha = smoothstep(t / args.fade_in)
        elif t < args.fade_in + args.hold:
            alpha = 1.0
        else:
            alpha = 1.0 - smoothstep((t - args.fade_in - args.hold) / max(args.fade_out, 1e-6))

        frame = np.asarray(Image.open(path).convert("RGBA"), np.float64) / 255.0
        rgb, a = frame[..., :3], frame[..., 3:4]
        plate = np.array([0.93, 0.928, 0.922])
        composed = rgb * a + plate * (1 - a)

        box = fit_box((width, height), (tw, th), "centre",
                      pad=(int(width * 0.030), 18),
                      centre_y=args.title_y, shape=args.shape,
                      cap_ratio=CAP_RATIO)
        if alpha > 0.004:
            # The insert carries the type; the rest of the frame is left alone,
            # so the shot stays a picture rather than a tinted plate.
            panelled = frosted_panel(composed, box, whiten=args.whiten,
                                     blur=16.0, shadow=0.22,
                                     shape=args.shape, cap_ratio=CAP_RATIO)
            composed = composed * (1 - alpha) + panelled * alpha

        x = int((box[0] + box[2]) / 2 - tw / 2)
        y = int((box[1] + box[3]) / 2 - th / 2)
        patch = composed[y:y + th, x:x + tw]
        ta = title_rgba[..., 3:4] * alpha
        composed[y:y + th, x:x + tw] = patch * (1 - ta) + title_rgba[..., :3] * ta

        Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8)).save(
            osp.join(args.out, f"frame_{index:05d}.png"))
    print(f"{n} frames -> {args.out}")


if __name__ == "__main__":
    main()
