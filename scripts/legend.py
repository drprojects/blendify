"""Legend inserts for the exploded-stack video: one transparent PNG per layer.

The video walks up a stack of colorizations of the same cloud, and a viewer
cannot tell `soil_chemistry` from `moisture_regime` by looking at it -- both are
three flat colours over the same geometry. So each layer needs a caption, and
for the categorical layers a key.

Three things drive the design:

**The classes come from the project's own palette JSON, never from a literal
list in this file.** The colours in the legend must be the *same objects* the
renderer used; a hand-copied swatch table is a second source of truth that goes
stale the first time someone restyles a class in
`configs/palettes/malibu3d_extra.json`, and it goes stale silently -- the figure
still renders, it just lies. `figlib.palettes.load_palettes` is the same call
`00_custom.py` makes, overrides and all, and nothing here is cached, so a
palette edit (a new colourmap for `elevation`, say) reaches the legend and the
point cloud in one step.

**The title sits at a fixed offset from the top-left corner, always.** These
inserts are shown one after another over a moving camera, so a title that
shifted by a few pixels between layers would read as a jitter in the video --
the eye is far more sensitive to a caption moving than to a panel resizing. The
title is therefore drawn into a fixed-height zone at a fixed baseline, and the
image is cropped only on its right and bottom edges. The insert still sizes
itself to its content; only the title's position inside it is pinned.

**Output is RGBA with nothing behind it.** The caller owns the background:
`scripts/overlay.py` builds a frosted panel out of the frame itself, which it
can only do if it knows the exact content extent. A legend that baked in its own
margin or its own white box would force the panel to guess. For the capsule
panel shape, whose end caps are semicircles that cannot hold text, see
`capsule_padding` below.

Everything is drawn at `--scale` (default 2x) and is expected to be downscaled
by the compositor. PIL's glyph rasteriser is decent but not a typesetter;
rendering large and resampling down is what makes 25px Times look printed rather
than pixellated.
"""
import argparse
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib.palettes import VOID_NAMES, hex_to_rgb, is_continuous, load_palettes  # noqa: E402


# --- what the video says out loud -------------------------------------------
# The renderer's layer keys are field names (`strength`, `grayscale`); those are
# implementation vocabulary and mean nothing to a reader, so every layer carries
# an explicit display title. Order is the order the stack explodes in.
#
# The four habitat tasks are *not* prefixed "Natural habitat --". The prefix is
# the same on all four, so it distinguishes nothing, and it tripled the width of
# the widest inserts in the sequence -- on a capsule panel that width is paid
# for twice, once in the box and once in the end caps.
LAYERS = [
    ("rgb", "RGB"),
    ("strength", "LiDAR intensity"),
    ("semantic", "Land cover"),
    ("forest", "Forest cover"),
    ("habitat_type", "Habitat type"),
    ("moisture_regime", "Moisture regime"),
    ("soil_chemistry", "Soil chemistry"),
    ("bioclimatic_zone", "Bioclimatic zone"),
    ("elevation", "Elevation"),
    ("grayscale", "Road network"),
]

# Column counts that beat the automatic choice. `semantic` has 15 classes; two
# columns of 8 make a tall, narrow block that fights the wide 16:9 frame,
# whereas 4x4 (the last column short) is a squat block that sits happily under a
# one-line title and inside a capsule.
COLUMNS = {"semantic": 4}

# Units and end labels for the continuous ramps. The palette JSON carries the
# numeric range but not what the numbers *are*; `strength` has no absolute range
# at all (it is clipped per tile), so it gets a qualitative pair instead of
# invented numbers.
RAMP_LABELS = {
    "elevation": {"unit": " m", "decimals": 0},
    "strength": {"ends": ("low", "high")},
}

DEFAULT_PALETTES = "data/malibu3d/send_29_07_v2/blender_export/palettes.json"
DEFAULT_OVERRIDES = ["configs/palettes/malibu3d_extra.json"]

FONT_DIR = "/usr/share/fonts/truetype/msttcorefonts"
FONT_REGULAR = os.path.join(FONT_DIR, "Times_New_Roman.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "Times_New_Roman_Bold.ttf")

# Design sizes, in pixels of the *final* 1080p frame; `scale` multiplies them.
TITLE_SIZE = 33
CLASS_SIZE = 25
TITLE_INK = (26, 26, 30, 255)
# The class list is subordinate to the title: a mid grey keeps the hierarchy
# without the list fading out over the frosted panel underneath.
CLASS_INK = (105, 105, 112, 255)
SWATCH_BORDER = (0, 0, 0, 55)

TITLE_GAP = 16          # bottom of the title zone to the first class row
ROW_LEADING = 1.42      # class row pitch, as a multiple of the class size
SWATCH_TEXT_GAP = 11
COLUMN_GAP = 34
RAMP_WIDTH = 250
RAMP_HEIGHT = 15
RAMP_LABEL_GAP = 7

# A legend is an annotation, not the subject: past ~45% of frame height it stops
# reading as an insert and starts competing with the point cloud.
MAX_HEIGHT_FRACTION = 0.45
FRAME_HEIGHT = 1080

# Acronyms and names whose internal capitals must survive sentence casing.
# Everything else is lowercased word by word, see `sentence_case`.
CASE_EXCEPTIONS = ("RGB", "LiDAR", "N/A")


def _px(value, scale):
    return int(round(value * scale))


def sentence_case(text):
    """One capitalisation convention for every title and every class label.

    The palettes disagree with each other -- `semantic` ships Title Case ("Not
    Forest"), the habitat tasks ship lowercase ("open", "mesic") -- and a
    sequence of inserts that mixes conventions looks like a typo rather than a
    style. Sentence case is the least shouty of the three and matches the paper.

    A word keeps its own casing if it carries a capital anywhere but the front
    (LiDAR, N/A, RGB); those are names and acronyms, not sentence starts. Words
    that are merely front-capitalised get folded down, so "Not Forest" becomes
    "Not forest". Accented French names are unaffected: `str.lower` and
    `str.upper` are Unicode-aware, and the habitat names are already lowercase.
    """
    words = text.split(" ")
    out = []
    for word in words:
        core = word.strip(".,;:()[]")
        if core in CASE_EXCEPTIONS or any(c.isupper() for c in core[1:]):
            out.append(word)                     # acronym or CamelCase name
        else:
            out.append(word.lower())
    result = " ".join(out)
    # Capitalise the opening letter, unless the first word is itself an acronym
    # that must be left alone.
    first = out[0].strip(".,;:()[]") if out else ""
    if first in CASE_EXCEPTIONS or any(c.isupper() for c in first[1:]):
        return result
    for i, char in enumerate(result):
        if char.isalpha():
            return result[:i] + char.upper() + result[i + 1:]
    return result


def classes(palette, max_classes=16):
    """(label, "#rrggbb") pairs worth putting in a key.

    Void / N-A classes are dropped: they mark points that were never annotated,
    which the render already mutes to neutral grey (see `void:` in the configs).
    Listing them would advertise an absence as a category.

    Returns `(entries, n_hidden)` so the caller can admit to a truncation rather
    than quietly showing a partial key.
    """
    names = palette.get("names", [])
    colors = palette.get("colors", [])
    kept = [(sentence_case(n), c) for n, c in zip(names, colors)
            if n.strip().lower() not in VOID_NAMES]
    if max_classes and len(kept) > max_classes:
        return kept[:max_classes], len(kept) - max_classes
    return kept, 0


def _swatch(size, color, scale, radius_frac=0.22):
    """A small rounded square of `color` with a hairline border.

    Supersampled 4x for the corners, exactly as `overlay.panel_mask` does: at
    legend sizes the radius is a handful of pixels and aliasing on it is the
    first thing that makes an insert look homemade. The border exists so a pale
    class (e.g. `#F5F0E6`) still reads as a swatch on a near-white panel.
    """
    ss = 4
    big = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    radius = max(1, int(round(size * ss * radius_frac)))
    draw.rounded_rectangle([0, 0, size * ss - 1, size * ss - 1], radius=radius,
                           fill=tuple(color) + (255,), outline=SWATCH_BORDER,
                           width=max(1, int(round(1.2 * scale)) * ss))
    return big.resize((size, size), Image.LANCZOS)


def ramp_colors(width, palette):
    """The RGB the bar shows at each of `width` horizontal positions.

    Split out from the drawing so it can be sampled in a test: a bar whose ends
    do not match the palette's first and last stop means the legend and the
    render disagree about what a colour means.

    `gamma` is applied here for the same reason it is applied in
    `figlib.palettes.continuous_colors`: with gamma 0.65 on `strength` a linear
    bar would claim a mapping the render does not use, and the reader could not
    match a colour in the cloud to a position on the bar.
    """
    stops = [tuple(int(v) for v in s) for s in palette["color_stops_rgb"]]
    gamma = float(palette.get("gamma", 1.0))
    out = []
    for x in range(width):
        t = x / max(width - 1, 1)
        if gamma != 1.0:
            t = t ** gamma
        pos = t * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        f = pos - i
        out.append(tuple(int(round(stops[i][c] * (1 - f) + stops[i + 1][c] * f))
                         for c in range(3)))
    return out


def _ramp(width, height, palette, scale):
    """Horizontal colour-ramp bar sampled from the palette's own stops."""
    bar = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = bar.load()
    for x, rgb in enumerate(ramp_colors(width, palette)):
        for y in range(height):
            pixels[x, y] = rgb + (255,)

    # Round the ends and outline, via a mask, so the bar matches the swatches.
    ss = 4
    mask = Image.new("L", (width * ss, height * ss), 0)
    radius = int(round(height * ss * 0.5))
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, width * ss - 1, height * ss - 1], radius=radius, fill=255)
    bar.putalpha(mask.resize((width, height), Image.LANCZOS))
    outline = Image.new("RGBA", (width * ss, height * ss), (0, 0, 0, 0))
    ImageDraw.Draw(outline).rounded_rectangle(
        [0, 0, width * ss - 1, height * ss - 1], radius=radius,
        outline=SWATCH_BORDER, width=max(1, int(round(1.2 * scale)) * ss))
    return Image.alpha_composite(bar, outline.resize((width, height), Image.LANCZOS))


def _ramp_ends(name, palette):
    """Text for the two ends of a ramp: real values when the palette has them."""
    spec = RAMP_LABELS.get(name, {})
    if "ends" in spec:
        # Sentence-cased like the class labels: these sit in the same slot in
        # the layout and reading "low" under one layer and "Open" under the
        # next is exactly the inconsistency the convention exists to kill.
        return tuple(sentence_case(t) for t in spec["ends"])
    lo, hi = palette.get("vmin"), palette.get("vmax")
    if lo is None or hi is None:
        # Percentile clipping is per tile, so no honest number can be printed.
        return ("low", "high")
    decimals = spec.get("decimals", 0)
    unit = spec.get("unit", "")
    return (f"{float(lo):.{decimals}f}{unit}", f"{float(hi):.{decimals}f}{unit}")


def _text_width(draw, text, font):
    return draw.textlength(text, font=font)


def _auto_columns(count, row_pitch, title_zone, scale):
    """1 or 2 columns: whichever keeps the insert under the height budget.

    Two columns are only worth the extra width when one column would push the
    insert past ~45% of frame height; below that a single column reads faster.
    Layers that want a specific shape override this through `COLUMNS`.
    """
    if count <= 1:
        return 1
    budget = MAX_HEIGHT_FRACTION * FRAME_HEIGHT * scale
    if title_zone + count * row_pitch <= budget or count <= 8:
        return 1
    return 2


def title_zone_height(scale):
    """Height of the fixed block the title is drawn into, in rendered pixels.

    Depends only on the font size, hence only on `scale` -- which is precisely
    what makes the title land at the same offset in every legend of the run.
    """
    font = ImageFont.truetype(FONT_BOLD, _px(TITLE_SIZE, scale))
    ascent, descent = font.getmetrics()
    return ascent + descent + _px(TITLE_GAP, scale)


def title_baseline(scale):
    """Distance from the insert's top edge to the title's baseline."""
    font = ImageFont.truetype(FONT_BOLD, _px(TITLE_SIZE, scale))
    return font.getmetrics()[0]


def capsule_padding(content_height, pad_y, margin=0.25):
    """Horizontal padding a capsule panel needs so no glyph enters an end cap.

    A capsule's caps are semicircles of radius R = panel_height / 2, so the
    panel's left edge at the *top* of the content is not at x = 0 but bitten
    inward. For content of height `h` centred in a panel of height h + 2*pad_y:

        R = h / 2 + pad_y,  bite = R - sqrt(R^2 - (h / 2)^2)

    and the padding must cover that bite, not merely look generous. `margin`
    adds a fraction of the bite back as breathing room. Note the bite grows with
    content height: a tall key like `semantic` needs far more side padding than
    a one-line title, which is the real cost of the capsule shape.
    """
    half = content_height / 2.0
    r = half + pad_y
    bite = r - math.sqrt(max(r * r - half * half, 0.0))
    return int(math.ceil(bite * (1.0 + margin)))


def render_legend(name, title, palettes, scale=2.0, max_classes=16, columns=None):
    """Render one layer's legend to an RGBA image, title pinned to the top-left.

    A layer with no palette (`rgb`, `grayscale`) legitimately gets a title and
    nothing else -- there is no key to give for a photograph or a road network
    drawn in a single colour -- so a missing palette is not an error here.
    """
    title = sentence_case(title)
    title_font = ImageFont.truetype(FONT_BOLD, _px(TITLE_SIZE, scale))
    class_font = ImageFont.truetype(FONT_REGULAR, _px(CLASS_SIZE, scale))

    palette = palettes.get(name)
    continuous = palette is not None and is_continuous(palette)
    entries, hidden = ([], 0)
    if palette is not None and not continuous:
        entries, hidden = classes(palette, max_classes)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    title_w = _text_width(probe, title, title_font)
    zone = title_zone_height(scale)
    baseline = title_baseline(scale)

    swatch = _px(CLASS_SIZE * 0.92, scale)
    row_pitch = _px(CLASS_SIZE * ROW_LEADING, scale)
    gap = _px(SWATCH_TEXT_GAP, scale)
    col_gap = _px(COLUMN_GAP, scale)

    rows = list(entries)
    if hidden:
        rows.append((f"+{hidden} more", None))

    if continuous:
        bar_w, bar_h = _px(RAMP_WIDTH, scale), _px(RAMP_HEIGHT, scale)
        lo_text, hi_text = _ramp_ends(name, palette)
        label_h = sum(class_font.getmetrics())
        body_w = max(bar_w, _text_width(probe, lo_text, class_font)
                     + _text_width(probe, hi_text, class_font) + gap)
        body_h = bar_h + _px(RAMP_LABEL_GAP, scale) + label_h
    elif rows:
        n_cols = columns or COLUMNS.get(name) or _auto_columns(
            len(rows), row_pitch, zone, scale)
        n_cols = max(1, min(n_cols, len(rows)))
        per_col = int(math.ceil(len(rows) / n_cols))
        chunks = [rows[i * per_col:(i + 1) * per_col] for i in range(n_cols)]
        chunks = [c for c in chunks if c]
        widths = [swatch + gap + max(_text_width(probe, t, class_font)
                                     for t, _ in chunk) for chunk in chunks]
        body_w = sum(widths) + col_gap * (len(chunks) - 1)
        body_h = max(len(chunk) for chunk in chunks) * row_pitch
    else:
        body_w = body_h = 0

    width = int(math.ceil(max(title_w, body_w))) + 2
    height = int(math.ceil(zone + body_h)) + 2
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Baseline anchor, not top anchor: "RGB" has no ascenders and "Habitat type"
    # does, so anchoring by ink would move the caption between layers.
    draw.text((0, baseline), title, font=title_font, fill=TITLE_INK, anchor="ls")
    y0 = zone

    if continuous:
        image.alpha_composite(_ramp(bar_w, bar_h, palette, scale), (0, int(y0)))
        y_text = y0 + bar_h + _px(RAMP_LABEL_GAP, scale)
        draw.text((0, y_text), lo_text, font=class_font, fill=CLASS_INK)
        draw.text((bar_w, y_text), hi_text, font=class_font, fill=CLASS_INK,
                  anchor="ra")
    elif rows:
        x = 0
        for chunk, col_w in zip(chunks, widths):
            for i, (text, color) in enumerate(chunk):
                cy = y0 + i * row_pitch + row_pitch / 2.0
                if color is not None:
                    tile = _swatch(swatch, hex_to_rgb(color), scale)
                    image.alpha_composite(
                        tile, (int(x), int(round(cy - swatch / 2.0))))
                draw.text((x + swatch + gap, cy), text, font=class_font,
                          fill=CLASS_INK, anchor="lm")
            x += col_w + col_gap

    # Crop right and bottom only. Trimming the top or the left would undo the
    # fixed title placement, because the topmost and leftmost ink depends on
    # which glyphs the title happens to contain.
    bbox = image.getbbox()
    if bbox:
        image = image.crop((0, 0, bbox[2], bbox[3]))
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer", help="layer key, e.g. semantic")
    parser.add_argument("--out", help="output PNG (with --layer)")
    parser.add_argument("--all", action="store_true",
                        help="render every layer of the stack")
    parser.add_argument("--out-dir", help="output directory (with --all)")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="render at this multiple of final frame pixels")
    parser.add_argument("--max-classes", type=int, default=16,
                        help="classes to list before truncating the key")
    parser.add_argument("--columns", type=int, default=None,
                        help="force a column count for the class key")
    parser.add_argument("--pad-y", type=float, default=34.0,
                        help="caller's vertical panel padding, in final frame "
                             "pixels; used to report the capsule end-cap padding")
    parser.add_argument("--palettes", default=DEFAULT_PALETTES)
    parser.add_argument("--palette-overrides", default=",".join(DEFAULT_OVERRIDES),
                        help="comma-separated override JSON/YAML files")
    args = parser.parse_args(argv)

    overrides = [p for p in (args.palette_overrides or "").split(",") if p]
    palettes = load_palettes(args.palettes, overrides)
    titles = dict(LAYERS)

    def emit(name, title, path):
        image = render_legend(name, title, palettes, args.scale,
                              args.max_classes, args.columns)
        image.save(path)
        # Padding is quoted in final frame pixels, which is what the compositor
        # works in once it has downscaled the insert by `scale`.
        pad_x = capsule_padding(image.height / args.scale, args.pad_y)
        print(f"{path}  {image.width}x{image.height}  "
              f"(final {image.width / args.scale:.0f}x{image.height / args.scale:.0f}"
              f", capsule pad_x >= {pad_x})")

    if args.all:
        if not args.out_dir:
            parser.error("--all needs --out-dir")
        os.makedirs(args.out_dir, exist_ok=True)
        for name, title in LAYERS:
            emit(name, title, os.path.join(args.out_dir, f"legend_{name}.png"))
        return

    if not args.layer or not args.out:
        parser.error("give --layer and --out, or --all and --out-dir")
    title = titles.get(args.layer)
    if title is None:
        parser.error(f"unknown layer {args.layer!r}; known: "
                     f"{[n for n, _ in LAYERS]}")
    emit(args.layer, title, args.out)


if __name__ == "__main__":
    main()
