"""Turn a render into a pale backdrop to sit behind a plot.

Washing an image toward white and brightening it are different operations and
only one of them does what a backdrop needs.

  * Brightening (the `brightness` input of the grading chain, or any gain)
    scales the image. Shadows are scaled by the same factor as highlights, so
    the *ratio* between them is untouched: the picture gets brighter but stays
    exactly as contrasty, and the bright end clips off against white before the
    dark end has lifted at all.

  * Washing blends every pixel toward white by a fixed fraction. Absolute
    contrast falls by (1 - strength) everywhere at once, blacks lift, nothing
    clips. That is what makes foreground text readable over it.

So do not reach for `brightness` to make a backdrop. Render normally, wash
afterwards — it also iterates in milliseconds instead of minutes.

    python scripts/backdrop.py render.png -o backdrop.png --strength 0.7 --gray

On a white page, blending toward white at strength s is *identical* to leaving
the colours alone and scaling the image's alpha by (1 - s), since the page then
supplies the white itself. `--keep-alpha` does it that way, which keeps the file
honest on any background; the default bakes the wash in.
"""
import argparse

import numpy as np
from PIL import Image

NAMED = {"white": "#FFFFFF", "black": "#000000", "paper": "#FFFFFF"}


def parse_color(value):
    value = NAMED.get(value.strip().lower(), value).lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {value!r}")
    return np.array([int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def wash(image, strength, color, gray=False, keep_alpha=False):
    """Blend toward `color`. Works in display space, matching a PDF overlay."""
    out = image.copy()
    rgb, alpha = out[..., :3], out[..., 3:4]

    if gray:
        rgb = np.repeat(luminance(rgb)[..., None], 3, axis=-1)

    if keep_alpha:
        return np.concatenate([rgb, alpha * (1.0 - strength)], axis=-1)
    return np.concatenate([rgb * (1 - strength) + color * strength, alpha], axis=-1)


def report(label, image):
    """Contrast as seen on a white page, which is where the figure lives."""
    rgb, alpha = image[..., :3], image[..., 3:4]
    over_white = rgb * alpha + (1.0 - alpha)
    lum = luminance(over_white)
    ink = lum[alpha[..., 0] > 0.01]
    print(f"  {label:22s} mean {lum.mean():.3f}  "
          f"range {ink.min():.3f}-{ink.max():.3f}  "
          f"contrast(sd) {ink.std():.4f}  clipped {100 * (lum >= 0.999).mean():5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--strength", type=float, default=0.7,
                        help="0 = untouched, 1 = pure backdrop colour")
    parser.add_argument("--color", default="white")
    parser.add_argument("--gray", action="store_true", help="desaturate first")
    parser.add_argument("--keep-alpha", action="store_true",
                        help="fade via alpha instead of baking the wash")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGBA"), np.float64) / 255
    out = wash(image, args.strength, parse_color(args.color), args.gray,
               args.keep_alpha)

    report("original", image)
    report(f"washed {args.strength}", out)

    Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8),
                    "RGBA").save(args.output)
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
