"""Composite two renders into one image, split by an angled margin.

Both images come from the same camera at the same resolution, so they are
pixel-registered: the split is a 2D masking job, not a rendering one.

The seam is fully defined by an angle and a margin width, and passes through the
image centre (unless `--offset` moves it). The first image is kept on the side
the normal points away from, the second on the other side, and the band between
them is filled with the gap colour (transparent by default).

    angle 0   -> vertical seam,   image A left,  image B right
    angle 90  -> horizontal seam, image A top,   image B bottom
    angle 20  -> seam leaning 20 deg clockwise from vertical

Example:
    python scripts/split_composite.py \
        data/malibu3d/renders/palette_v2/semantic.png \
        data/malibu3d/renders/palette_v2/forest.png \
        --angle 20 --margin 1.5% --gap white -o /tmp/split.png
"""
import argparse
import os.path as osp

import numpy as np
from PIL import Image

NAMED_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "none": None,
    "transparent": None,
}


def parse_gap(value):
    """Gap fill: a hex colour, or None for transparent."""
    if value is None:
        return None
    key = value.strip().lower()
    if key in NAMED_COLORS:
        value = NAMED_COLORS[key]
        if value is None:
            return None
    value = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB or a name, got {value!r}")
    rgb = [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return np.array(rgb + [1.0], dtype=np.float64)


def parse_length(value, reference):
    """A length in pixels, or a percentage of the image width."""
    value = str(value).strip()
    if value.endswith("%"):
        return float(value[:-1]) / 100.0 * reference
    return float(value)


def load(path):
    image = Image.open(path).convert("RGBA")
    return np.asarray(image, dtype=np.float64) / 255.0


def composite(image_a, image_b, angle, margin, offset=0.0, gap=None, feather=1.5):
    """Blend two RGBA arrays across a straight seam.

    Args:
        angle: seam orientation in degrees, 0 = vertical, growing clockwise.
        margin: total width of the gap band, in pixels.
        offset: shift of the seam off centre, along its normal, in pixels.
        gap: RGBA fill for the band, or None for transparent.
        feather: width of the antialiasing ramp on each edge, in pixels.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"size mismatch: {image_a.shape[:2]} vs {image_b.shape[:2]}")

    height, width = image_a.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(angle)

    # Signed distance to the seam. The normal is (cos, sin) in display
    # coordinates (x right, y down), so angle 0 gives a vertical seam.
    signed = ((x - (width - 1) / 2.0) * np.cos(theta)
              + (y - (height - 1) / 2.0) * np.sin(theta)) - offset

    half = margin / 2.0
    aa = max(feather, 1e-6)
    weight_a = np.clip(0.5 - (signed + half) / aa, 0.0, 1.0)
    weight_b = np.clip(0.5 + (signed - half) / aa, 0.0, 1.0)
    weight_gap = np.clip(1.0 - weight_a - weight_b, 0.0, 1.0)

    # Mix in premultiplied alpha, otherwise the colour of a transparent
    # background bleeds into the seam.
    def premultiply(image):
        out = image.copy()
        out[..., :3] *= out[..., 3:4]
        return out

    out = premultiply(image_a) * weight_a[..., None]
    out += premultiply(image_b) * weight_b[..., None]
    if gap is not None:
        out += premultiply(gap[None, None, :] * np.ones_like(image_a)) * weight_gap[..., None]

    alpha = out[..., 3:4]
    rgb = np.divide(out[..., :3], alpha, out=np.zeros_like(out[..., :3]), where=alpha > 0)
    return np.concatenate([rgb, alpha], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_a", help="render kept on the first side of the seam")
    parser.add_argument("image_b", help="render kept on the second side")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--angle", type=float, default=20.0,
                        help="seam angle in degrees; 0 = vertical, 90 = horizontal")
    parser.add_argument("--margin", default="1%",
                        help="gap width, in pixels or as a %% of image width")
    parser.add_argument("--offset", default="0",
                        help="shift the seam off centre along its normal")
    parser.add_argument("--gap", default="none",
                        help="gap colour: #RRGGBB, white, black, or none")
    parser.add_argument("--feather", type=float, default=1.5,
                        help="antialiasing ramp width in pixels")
    parser.add_argument("--swap", action="store_true",
                        help="exchange the two sides without reordering the arguments")
    args = parser.parse_args()

    image_a, image_b = load(args.image_a), load(args.image_b)
    if args.swap:
        image_a, image_b = image_b, image_a

    width = image_a.shape[1]
    margin = parse_length(args.margin, width)
    offset = parse_length(args.offset, width)

    out = composite(image_a, image_b, args.angle, margin, offset,
                    parse_gap(args.gap), args.feather)

    Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8), "RGBA").save(args.output)
    print(f"{osp.basename(args.output)}  {out.shape[1]}x{out.shape[0]}  "
          f"angle={args.angle}  margin={margin:.1f}px  gap={args.gap}")


if __name__ == "__main__":
    main()
