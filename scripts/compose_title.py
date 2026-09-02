"""The paper's title on a frosted insert, for the cold open and the sign-off.

The title is compiled by LaTeX from the paper's own macros, so the wordmark,
the icon and the MALIBU highlight colour are the paper's rather than an
imitation. This only places it.

This is a library first. `assemble_video.py` calls `load_title()` and
`draw_title()` *after* the background plate and the saturation boost, for the
same reason the legends are drawn there: a title baked into the rendered frames
would be pushed through the same 2x chroma scale as the point cloud, and the
highlight colour would stop being the paper's.

The CLI is the cold-open path — it flattens onto the plate and drives the
weight from a fade-in/hold/fade-out ramp, so the shot opens on a title card and
resolves into the landscape.

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
REFERENCE_WIDTH = 1920.0    # pixel-quoted values below are for a 1080p frame

# The approved cold-open look, recovered from its own output. Fractions of the
# frame width are already resolution-independent; the pixel quantities are
# scaled by REFERENCE_WIDTH so a 960-wide preview and a 1920-wide final differ
# only in sampling.
TITLE_WIDTH = 0.347     # title width as a fraction of the frame
TITLE_Y = 0.5           # dead centre; the title card has no subject to avoid
PAD_X = 0.020           # end-cap clearance, as a fraction of frame width
PAD_Y = 24
WHITEN = 0.90
BLUR = 32.0
SHADOW = 0.22
SHADOW_BLUR = 36.0
SHADOW_OFFSET = 12


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def load_title(path, frame_width, title_width=TITLE_WIDTH):
    """Title art as RGBA float in [0, 1], sized for `frame_width`.

    The LaTeX minipage is far wider than the type, so ~53% of the PNG is
    transparent margin. Sizing a panel to the uncropped image bakes that in and
    makes the insert 1.7x wider than it needs to be -- crop to the ink.
    """
    title = Image.open(path).convert("RGBA")
    ink = title.getbbox()
    if ink:
        title = title.crop(ink)
    tw = int(frame_width * title_width)
    th = int(round(tw * title.height / title.width))
    return np.asarray(title.resize((tw, th), Image.LANCZOS), np.float64) / 255.0


def draw_title(frame, art, weight, title_y=TITLE_Y, pad_x=PAD_X, pad_y=PAD_Y,
               whiten=WHITEN, shape="capsule"):
    """Draw the title on a frosted capsule. `frame` is opaque RGB float.

    `weight` fades the insert and the type together, so the shot resolves into
    the landscape rather than the type lifting off a panel that lingers.
    """
    if art is None or weight <= 0.004:
        return frame
    height, width = frame.shape[:2]
    scale = width / REFERENCE_WIDTH
    th, tw = art.shape[:2]
    box = fit_box((width, height), (tw, th), "centre",
                  pad=(int(width * pad_x), int(pad_y * scale)),
                  centre_y=title_y, shape=shape, cap_ratio=CAP_RATIO)
    # The insert carries the type; the rest of the frame is left alone, so the
    # shot stays a picture rather than a tinted plate.
    panelled = frosted_panel(frame, box, whiten=whiten, blur=BLUR * scale,
                             shadow=SHADOW, shadow_blur=SHADOW_BLUR * scale,
                             shadow_offset=int(round(SHADOW_OFFSET * scale)),
                             shape=shape, cap_ratio=CAP_RATIO)
    frame = frame * (1 - weight) + panelled * weight
    x = int((box[0] + box[2]) / 2 - tw / 2)
    y = int((box[1] + box[3]) / 2 - th / 2)
    patch = frame[y:y + th, x:x + tw]
    a = art[..., 3:4] * weight
    frame[y:y + th, x:x + tw] = patch * (1 - a) + art[..., :3] * a
    return frame


def ramp(index, fps, fade_in, hold, fade_out):
    """Title opacity for frame `index`: fade in, hold, fade out."""
    t = index / fps
    if t < fade_in:
        return smoothstep(t / fade_in)
    if t < fade_in + hold:
        return 1.0
    return 1.0 - smoothstep((t - fade_in - hold) / max(fade_out, 1e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--fade-in", type=float, default=0.9)
    parser.add_argument("--hold", type=float, default=4.5)
    parser.add_argument("--fade-out", type=float, default=1.4)
    parser.add_argument("--whiten", type=float, default=WHITEN,
                        help="opacity of the insert; higher = more readable type")
    parser.add_argument("--title-y", type=float, default=TITLE_Y,
                        help="vertical placement, 0=top 1=bottom")
    parser.add_argument("--pad-x", type=float, default=PAD_X,
                        help="end-cap clearance as a fraction of frame width")
    parser.add_argument("--pad-y", type=int, default=PAD_Y,
                        help="vertical padding, in 1080p pixels")
    parser.add_argument("--shape", default="capsule",
                        choices=("rect", "ellipse", "capsule"))
    parser.add_argument("--title-width", type=float, default=TITLE_WIDTH,
                        help="title width as a fraction of the frame")
    args = parser.parse_args()

    files = sorted(glob.glob(osp.join(args.frames, "*.png")))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")
    os.makedirs(args.out, exist_ok=True)

    width, height = Image.open(files[0]).size
    art = load_title(args.title, width, args.title_width)

    for index, path in enumerate(files):
        rgba = np.asarray(Image.open(path).convert("RGBA"), np.float64) / 255.0
        a = rgba[..., 3:4]
        plate = np.array([0.93, 0.928, 0.922])
        frame = rgba[..., :3] * a + plate * (1 - a)
        frame = draw_title(frame, art,
                           ramp(index, args.fps, args.fade_in, args.hold,
                                args.fade_out),
                           args.title_y, args.pad_x, args.pad_y, args.whiten,
                           args.shape)
        Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(
            osp.join(args.out, f"frame_{index:05d}.png"))
    print(f"{len(files)} frames -> {args.out}")


if __name__ == "__main__":
    main()
