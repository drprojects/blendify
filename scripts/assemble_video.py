"""Assemble rendered shot frames into a finished video.

Everything after the render happens here: cross-dissolves, corner labels,
background, loop closure, encoding. Keeping it out of Blender means retiming a
cut, restyling a label or reordering shots costs seconds instead of a re-render,
which is the whole reason the shots are rendered as PNG sequences.

Frames keep their alpha, so the background is chosen at assembly time.

    python scripts/assemble_video.py --shots shots.json --out demo.mp4

The shot file is a list of:
    {"name": ..., "frames": <dir>, "label": "dense town", "sublabel": "D068_..."}
"""
import argparse
import json
import glob
import math
import os
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from overlay import frosted_panel, fit_box

FONT = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"


def prepare_legends(shot, size, fps, saturation=1.0):
    """Load a shot's legend art and its per-frame schedule, or None.

    A shot declares legends in the shots JSON:

        "legends": {"path": "stack.json", "dir": "legends",
                    "layers": ["rgb", "strength", ...]}

    so the pacing comes from the same choreography file that drove the render
    and cannot fall out of sync with it.

    The swatches are put through the same chroma scale as the frame. A legend is
    only useful if its colours are the ones on screen, and the boost is applied
    to the whole frame at assembly, so an unboosted swatch would label a class
    with a colour that class no longer has. The operation preserves luma and
    leaves neutrals alone, so the grey type and the panel are untouched.
    """
    spec = shot.get("legends")
    if not spec:
        return None
    from compose_legends import load_legends, schedule
    names = spec["layers"]
    plan, _ = schedule(spec["path"], names, fps, spec.get("outro", 0.7))
    art = load_legends(spec["dir"], names, size[0])
    if saturation != 1.0:
        for name, image in art.items():
            rgb = image[..., :3]
            lum = (0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2]
                   + 0.0722 * rgb[..., 2:3])
            image[..., :3] = np.clip(lum + (rgb - lum) * saturation, 0, 1)
    return {"art": art,
            "plan": plan,
            "position": spec.get("position", "top-right"),
            "whiten": spec.get("whiten", 0.88)}


def prepare_title(shot, size, fps):
    """Load a shot's closing title card, or None.

    Declared per shot:

        "title": {"path": "title.png", "start": 397, "fade_in": 1.0}

    `start` is a frame index within the shot; the card then holds to the end of
    the shot, because this is the last thing on screen and a title that fades
    out leaves the video ending on nothing.
    """
    spec = shot.get("title")
    if not spec:
        return None
    from compose_title import load_title
    return {"art": load_title(spec["path"], size[0]),
            "start": int(spec["start"]),
            "fade": max(int(round(spec.get("fade_in", 1.0) * fps)), 1)}


def background(size, style="gradient"):
    """The plate every frame is composited onto."""
    width, height = size
    if style == "white":
        return np.ones((height, width, 3), np.float64)
    if style == "dark":
        return np.full((height, width, 3), 0.10)
    # A light warm grey, slightly darker at the top: it keeps a pale alpine
    # tile from blowing out into the background while leaving a dark forest
    # tile readable, which a flat white or a flat dark plate cannot both do.
    top = np.array([0.878, 0.874, 0.866])
    bottom = np.array([0.964, 0.960, 0.952])
    ramp = np.linspace(0.0, 1.0, height)[:, None, None]
    return top * (1 - ramp) + bottom * ramp


def load_frame(path, size, flip=False):
    """Load a render. `flip` mirrors it horizontally.

    Mirroring is a composition tool here, not a fix: it moves a shot's empty
    quadrant to the side the inserts live on. It is safe only because these
    frames contain no text and no orientation cue a viewer could check -- there
    is no north arrow and no scale bar in shot. Overlays are drawn afterwards
    and are never mirrored.
    """
    image = Image.open(path).convert("RGBA")
    if image.size != size:
        image = image.resize(size, Image.LANCZOS)
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(image, np.float64) / 255.0


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def draw_label(image, text, sub, alpha, size, position="top-left"):
    """Caption on a frosted insert, faded by `alpha`.

    Same corner as the legend inserts, because a caption in one corner and a
    legend in another makes two design languages out of one video. `fit_box`
    anchors the content, so the type starts at the same x whatever the panel
    around it measures.
    """
    if alpha <= 0.01 or not text:
        return image
    width, height = size
    scale = width / 1920.0
    big = ImageFont.truetype(FONT_BOLD, int(round(38 * scale)))
    small = ImageFont.truetype(FONT, int(round(22 * scale)))

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    tw = probe.textlength(text, font=big)
    sw = probe.textlength(sub, font=small) if sub else 0
    content = (int(max(tw, sw)), int(round((46 if sub else 30) * scale)))
    box = fit_box(size, content, position,
                  pad=(int(30 * scale), int(24 * scale)))

    # Same insert as the title card, so the video reads as one design rather
    # than a title in one language and captions in another.
    panelled = frosted_panel(image, box, whiten=0.84, blur=12.0,
                             shadow=0.18, shadow_blur=14.0)
    out = image * (1 - alpha) + panelled * alpha

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ink = int(round(255 * alpha))
    x = int(box[0] + 30 * scale)
    y = int(box[1] + 20 * scale)
    draw.text((x, y), text, font=big, fill=(26, 26, 30, ink))
    if sub:
        draw.text((x, y + int(40 * scale)), sub, font=small, fill=(105, 105, 112, ink))
    merged = Image.alpha_composite(
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).convert("RGBA"),
        layer)
    return np.asarray(merged, np.float64)[..., :3] / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--crossfade", type=float, default=0.5,
                        help="seconds of dissolve between shots")
    parser.add_argument("--label-position", default="top-left",
                        choices=("top-left", "top-right",
                                 "bottom-left", "bottom-right"))
    parser.add_argument("--background", default="gradient",
                        choices=("gradient", "white", "dark"))
    parser.add_argument("--loop", action="store_true",
                        help="dissolve the tail back into the first frame")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="chroma scale applied in post; values above 1 "
                             "over-saturate, which the shader cannot do")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()

    with open(args.shots) as handle:
        shots = json.load(handle)
    for shot in shots:
        shot["files"] = sorted(glob.glob(osp.join(shot["frames"], "*.png")))
        if not shot["files"]:
            raise SystemExit(f"no frames in {shot['frames']}")
    probe = Image.open(shots[0]["files"][0])
    size = (args.width, int(round(args.width * probe.height / probe.width))) \
        if args.width else probe.size
    plate = background(size, args.background)
    fade = max(int(round(args.crossfade * args.fps)), 0)
    print(f"{len(shots)} shots, {size[0]}x{size[1]}, {args.fps} fps, "
          f"{fade}-frame dissolves")

    # Timeline: each shot overlaps the next by `fade` frames.
    timeline = []
    cursor = 0
    for index, shot in enumerate(shots):
        shot["start"] = cursor
        cursor += len(shot["files"]) - (fade if index < len(shots) - 1 else 0)
    total = cursor
    print(f"  {total} frames = {total / args.fps:.1f} s")

    for shot in shots:
        shot["_legends"] = prepare_legends(shot, size, args.fps,
                                          args.saturation)
        shot["_title"] = prepare_title(shot, size, args.fps)
        if shot["_title"]:
            print(f"  title card from frame {shot['_title']['start']}")
        if shot["_legends"]:
            print(f"  legends: {len(shot['_legends']['art'])} for "
                  f"{osp.basename(shot['files'][0])[:12]}...")

    tmp = tempfile.mkdtemp(prefix="assemble_")
    for frame_index in range(total):
        acc = np.zeros((size[1], size[0], 4))
        caption = ("", "", 0.0)
        lead = None          # (shot, local index) of the most visible shot
        for shot in shots:
            local = frame_index - shot["start"]
            if not (0 <= local < len(shot["files"])):
                continue
            weight = 1.0
            if fade:
                if local < fade and shot["start"] > 0:
                    weight = smoothstep(local / fade)
                remaining = len(shot["files"]) - 1 - local
                if remaining < fade and shot is not shots[-1]:
                    weight = min(weight, smoothstep(remaining / fade))
            acc += load_frame(shot["files"][local], size,
                              shot.get("flip", False)) * weight
            # the label follows whichever shot is most visible
            if weight > caption[2]:
                caption = (shot.get("label", ""), shot.get("sublabel", ""), weight)
                lead = (shot, local)
        alpha = np.clip(acc[..., 3:4], 0, 1)
        rgb = np.divide(acc[..., :3], np.where(alpha > 1e-6, alpha, 1.0))
        if args.saturation != 1.0:
            # Saturation lives here rather than in the shader because Cycles
            # CLAMPS a Mix node's factor at 1.0, so `color.saturation` above 1
            # silently does nothing in a render. Under white light this is
            # exactly equivalent: scaling by a scalar commutes with a chroma
            # scale about luma. Being in post also makes it free to retune.
            lum = (0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2]
                   + 0.0722 * rgb[..., 2:3])
            rgb = np.clip(lum + (rgb - lum) * args.saturation, 0, 1)
        frame = rgb * alpha + plate * (1 - alpha)
        if lead is not None and lead[0].get("_legends"):
            from compose_legends import draw_legend
            spec = lead[0]["_legends"]
            for name, weight in spec["plan"][min(lead[1], len(spec["plan"]) - 1)]:
                frame = draw_legend(frame, spec["art"].get(name),
                                    weight * caption[2], spec["position"],
                                    whiten=spec["whiten"])
        if lead is not None and lead[0].get("_title"):
            from compose_title import draw_title
            spec = lead[0]["_title"]
            weight = smoothstep((lead[1] - spec["start"]) / spec["fade"])
            frame = draw_title(frame, spec["art"], weight * caption[2])
        frame = draw_label(frame, caption[0], caption[1], caption[2], size,
                           args.label_position)
        Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(
            osp.join(tmp, f"f_{frame_index:05d}.png"))
        if frame_index % 40 == 0:
            print(f"  composited {frame_index}/{total}")

    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", osp.join(tmp, "f_%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"-> {args.out} ({osp.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
