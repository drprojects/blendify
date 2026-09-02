"""Frosted inserts for text over video frames.

Text laid straight onto an aerial render is unreadable wherever the scene
happens to be light, and washing the whole frame to fix that kills the image.
An insert solves both: it is local, so the picture survives, and it is opaque
enough that the type sits on a controlled background instead of on whatever
pixels are underneath.

The panel is the scene behind it, blurred and lifted toward white, inside a
generously rounded mask with a feathered edge and a soft drop shadow. Blurring
rather than flat-filling keeps a hint of the image showing through, so the
insert reads as part of the frame rather than a sticker on top of it.

Panel shape
-----------
`shape` picks the outline. The three are not interchangeable, and which one
wastes least space depends entirely on the aspect ratio of the text:

    rect     rounded rectangle. Area = w*h minus the four corner bites.
    ellipse  Area = (pi/2) * w*h ~= 1.571 * w*h.
    capsule  stadium; a rectangle with semicircular caps. Area = w*h + pi*h^2/4.

The ellipse figure is the one that surprises people. An ellipse that fully
contains a w x h box has semi-axes w/sqrt(2) and h/sqrt(2), so it must be
1.414x wider *and* 1.414x taller than the text it wraps -- exactly the side
space the shape was supposed to save. Its 57% overhead is also independent of
aspect ratio, so it never improves for wide text.

The capsule's overhead is pi*h^2/4, which depends only on the *height*. For a
wide title block that is a small constant, so the capsule converges on the
rectangle as text gets wider while still losing the hard corners. That makes it
the shape to reach for when a rectangle looks boxy but an ellipse looks
bloated. See `measure_shapes` for the numbers on a concrete title.

Sizing a capsule, and `cap_ratio`
---------------------------------
The caps are laid *outside* the padded text box rather than being carved out of
it: the straight central section is exactly the padded box, and each cap adds a
further r. So the text can never stray into the curved ends no matter how the
panel is sized -- the horizontal padding is the guaranteed clearance, and
`straight_section` will report it back for checking.

That makes the cap cost exactly pi*r^2, and gives two independent levers on it:

    cap_ratio  r = cap_ratio * h/2. At 1.0 the ends are true semicircles. Below
               that they are still fully-rounded arcs, just drawn on a smaller
               circle, and the saving goes as (1 - cap_ratio^2) -- 0.85 sheds
               28% of the cap area while staying visually round.
    vertical padding  h = content height + 2*py, and the cap term is quadratic
               in h, so trimming py pays off twice: a shorter panel and much
               smaller caps.

Vertical padding is the stronger lever of the two and should be spent first;
`cap_ratio` is there for the last bit of tightening once the padding is as
tight as the type can bear. Both are measured in `measure_shapes`.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SHAPES = ("rect", "ellipse", "capsule")


def panel_mask(size, box, shape="rect", radius=None, feather=0.6, cap_ratio=1.0):
    """A soft-edged panel outline, as a float mask in [0, 1].

    Supersampling is what keeps the edge sharp. Drawing the outline at 4x and
    downsampling with LANCZOS resolves the curve to a quarter of a pixel, so
    the 0.6 px blur afterwards only has to kill the residual stair-stepping --
    it is not being asked to hide aliasing, which is why the edge stays crisp
    instead of turning into a haze. Ellipses and capsules need this more than
    rectangles do: their curvature runs along the whole outline rather than
    being confined to four corners.
    """
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    scale = 4
    big = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    draw = ImageDraw.Draw(big)
    x0, y0, x1, y1 = [v * scale for v in box]
    if shape == "ellipse":
        draw.ellipse([x0, y0, x1, y1], fill=255)
    else:
        if shape == "capsule":
            # A stadium is the fully-rounded case: corner radius pinned to half
            # the short side, so the caps become true semicircles. `cap_ratio`
            # below 1 shrinks that circle -- the ends still read as round arcs,
            # but they cost pi*r^2 instead of pi*(h/2)^2.
            r = cap_ratio * min(x1 - x0, y1 - y0) / 2.0
        elif radius is None:
            r = min(x1 - x0, y1 - y0) * 0.34
        else:
            # `radius` is quoted in output pixels, so it scales with the box.
            r = radius * scale
        draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=255)
    mask = big.resize(size, Image.LANCZOS)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(mask, np.float64)[..., None] / 255.0


def rounded_mask(size, box, radius, feather):
    """A soft-edged rounded rectangle, as a float mask in [0, 1].

    Kept as the original name because other modules import it; it is now a thin
    call through to `panel_mask`.
    """
    return panel_mask(size, box, "rect", radius=radius, feather=feather)


def frosted_panel(frame, box, radius=None, whiten=0.80, blur=14.0,
                  feather=0.6, shadow=0.20, shadow_blur=18.0, shadow_offset=6,
                  shape="rect", cap_ratio=1.0):
    """Composite a frosted insert into `frame` (float RGB in [0, 1]).

    `whiten` is what makes text legible: at 0.8 the panel is mostly white with
    the blurred scene showing faintly through, which is enough to carry black
    body text and a light accent colour without either fighting the picture.

    `shape` defaults to "rect" so existing callers are unaffected. `radius` is
    ignored for "ellipse" and "capsule", whose curvature is fixed by the box;
    for "capsule", `cap_ratio` scales the end-cap radius (see the module
    docstring). Pass the same `cap_ratio` you gave `fit_box`, or the caps will
    not line up with the clearance it reserved.
    """
    height, width = frame.shape[:2]
    size = (width, height)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if radius is None:
        radius = int(round(min(x1 - x0, y1 - y0) * 0.34))

    out = frame.copy()

    # Drop shadow first, so the panel sits on top of its own shadow. It uses
    # the same outline as the panel: a rectangular shadow under an elliptical
    # panel would show at the sides and give the shape away.
    if shadow > 0:
        offset = (x0, y0 + shadow_offset, x1, y1 + shadow_offset)
        smask = panel_mask(size, offset, shape, radius=radius,
                           feather=shadow_blur, cap_ratio=cap_ratio)
        out *= (1.0 - shadow * smask)

    blurred = np.asarray(
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(blur)), np.float64) / 255.0
    panel = blurred * (1.0 - whiten) + 1.0 * whiten

    mask = panel_mask(size, (x0, y0, x1, y1), shape, radius=radius,
                      feather=feather, cap_ratio=cap_ratio)
    return out * (1.0 - mask) + panel * mask


def shape_box(content, shape="rect", pad=(46, 34), cap_ratio=1.0):
    """Outer panel size needed to wrap `content` (width, height) in `shape`.

    Padding is applied to the text box first, then the shape is grown around
    that padded box. The growth is where the shapes diverge:

        rect     no growth; the padded box *is* the panel.
        ellipse  sqrt(2) in both axes -- the smallest ellipse containing a
                 rectangle touches it at the corners, and that costs 41% on
                 each axis. This is the cost that makes a naive ellipse waste
                 more side space than the rectangle it replaced.
        capsule  a cap of radius r is added at each end of the long axis, so
                 the straight section is exactly the padded box.

    The capsule rule is what keeps type out of the curved ends. The tempting
    alternative -- letting the caps eat into the padded box, which is cheaper --
    puts the corners of the text inside the arc, and on a centred block whose
    widest line sits at the very bottom that clips glyphs. Reserving the caps
    outside costs pi*r^2 and is worth it. Caps go on the long axis, so tall
    narrow content (a legend column) gets them top and bottom instead.
    """
    cw, ch = content
    px, py = pad
    bw, bh = cw + 2 * px, ch + 2 * py
    if shape == "rect":
        return bw, bh
    if shape == "ellipse":
        return bw * np.sqrt(2.0), bh * np.sqrt(2.0)
    if shape == "capsule":
        if bw >= bh:
            return bw + cap_ratio * bh, bh
        return bw, bh + cap_ratio * bw
    raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")


def straight_section(box, shape="rect", cap_ratio=1.0):
    """The part of `box` that is not curved end cap, as (x0, y0, x1, y1).

    Text must stay inside this. For a capsule it is the panel minus one cap
    radius at each end of the long axis; for the other shapes the question does
    not arise the same way and the box is returned unchanged. Exists so callers
    can *assert* the clearance rather than trust the arithmetic in `shape_box`.
    """
    x0, y0, x1, y1 = box
    if shape != "capsule":
        return (x0, y0, x1, y1)
    w, h = x1 - x0, y1 - y0
    r = cap_ratio * min(w, h) / 2.0
    if w >= h:
        return (x0 + r, y0, x1 - r, y1)
    return (x0, y0 + r, x1, y1 - r)


def fit_box(size, content, position="centre", pad=(46, 34), offset=(0, 0),
            centre_y=0.42, shape="rect", cap_ratio=1.0):
    """Box for a panel wrapping `content` (width, height) pixels.

    `shape` only changes how much the box is grown around the content; the
    placement rules are unchanged, and "rect" reproduces the original box
    exactly. Pass `cap_ratio` on to `frosted_panel` unchanged.
    """
    width, height = size
    bw, bh = shape_box(content, shape, pad, cap_ratio)
    if position == "centre":
        x0, y0 = (width - bw) / 2, (height - bh) * centre_y
    elif position == "bottom-left":
        x0, y0 = width * 0.045, height - bh - height * 0.075
    else:
        raise ValueError(position)
    return (x0 + offset[0], y0 + offset[1], x0 + bw + offset[0], y0 + bh + offset[1])


def measure_shapes(content, pad=(46, 34), ink_area=None, cap_ratio=1.0):
    """Panel area per shape, as a multiple of the text box (and of real ink).

    Reported against two baselines because they answer different questions.
    Against the *text box* it is pure geometry. Against the *ink* -- the count
    of actually-marked pixels, which for a centred title block with a short top
    line and a stats row is far less than its bounding box -- it says how much
    of the insert is genuinely empty, which is what the eye complains about.
    """
    cw, ch = content
    rows = []
    for shape in SHAPES:
        bw, bh = shape_box(content, shape, pad, cap_ratio)
        if shape == "ellipse":
            area = (np.pi / 4.0) * bw * bh
        elif shape == "capsule":
            # Straight section plus two half-caps: bw0*bh0 + pi*r^2 exactly.
            r = cap_ratio * min(bw, bh) / 2.0
            area = (bw - 2 * r) * bh + np.pi * r * r if bw >= bh else \
                   bw * (bh - 2 * r) + np.pi * r * r
        else:
            # Subtract the four corner bites of the default 0.34 radius.
            r = min(bw, bh) * 0.34
            area = bw * bh - (4 - np.pi) * r * r
        rows.append((shape, bw, bh, area, area / (cw * ch),
                     area / ink_area if ink_area else float("nan")))
    return rows
