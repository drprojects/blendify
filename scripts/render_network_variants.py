"""Render a labelled matrix of network-figure options for visual inspection.

Varies one thing at a time around a baseline, so each image answers a single
question. Outputs land in one directory with self-describing names, plus a
contact sheet per axis.

    python scripts/render_network_variants.py \\
        --config configs/figures/malibu3d_D075_FU-S1-10_networks.yaml \\
        --out data/malibu3d/renders/networks_D075_FU-S1-10
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_config                                   # noqa: E402

PY = osp.expanduser("~/miniconda3/envs/blendify/bin/python")

# (axis, label, extra --set overrides). The baseline appears in several axes so
# each contact sheet is self-contained.
VARIANTS = [
    ("height", "h10", ["graphs.height=10", "graphs.cast_shadow=False"]),
    ("height", "h15", ["graphs.height=15", "graphs.cast_shadow=False"]),
    ("height", "h20", ["graphs.height=20", "graphs.cast_shadow=False"]),
    ("height", "h25", ["graphs.height=25", "graphs.cast_shadow=False"]),
    ("height", "h30", ["graphs.height=30", "graphs.cast_shadow=False"]),

    ("shadow", "h20-shadow-off", ["graphs.height=20", "graphs.cast_shadow=False"]),
    ("shadow", "h20-shadow-on", ["graphs.height=20", "graphs.cast_shadow=True"]),
    ("shadow", "h10-shadow-on", ["graphs.height=10", "graphs.cast_shadow=True"]),

    ("cloud", "subtle", ["graphs.height=20", "graphs.cast_shadow=False",
                         "color.saturation=0.40", "color.exposure=0.40"]),
    ("cloud", "default", ["graphs.height=20", "graphs.cast_shadow=False"]),
    ("cloud", "very-pale", ["graphs.height=20", "graphs.cast_shadow=False",
                            "color.saturation=0.08", "color.exposure=1.35"]),
    ("cloud", "greyscale", ["graphs.height=20", "graphs.cast_shadow=False",
                            "color.saturation=0.0", "color.exposure=0.85"]),

    ("thickness", "thin", ["graphs.height=20", "graphs.cast_shadow=False",
                           "graphs.radius=1.0", "graphs.node_radius=1.5"]),
    ("thickness", "medium", ["graphs.height=20", "graphs.cast_shadow=False"]),
    ("thickness", "thick", ["graphs.height=20", "graphs.cast_shadow=False",
                            "graphs.radius=2.5", "graphs.node_radius=3.5"]),
]


def main(args):
    os.chdir(osp.dirname(osp.dirname(osp.abspath(__file__))))
    cfg = load_config(args.config)
    stem = osp.basename(cfg["data"]["path"]).split(".")[0]
    rendered = osp.join(osp.dirname(cfg["data"]["path"]), f"{stem}_rgb.png")
    os.makedirs(args.out, exist_ok=True)

    common = [f"render.n_samples={args.samples}",
              f"render.resolution=[{args.width},{round(args.width*1.7/3)}]"]

    done, failed = [], []
    for index, (axis, label, extra) in enumerate(VARIANTS, 1):
        target = osp.join(args.out, f"{axis}__{label}.png")
        started = time.time()
        result = subprocess.run(
            [PY, "examples/00_custom.py", "--config", args.config, "--image",
             "--set", *common, *extra],
            capture_output=True, text=True)
        if result.returncode != 0 or not osp.exists(rendered):
            print(f"  [{index:>2}/{len(VARIANTS)}] FAIL {axis}/{label}")
            print("       " + result.stderr.strip().splitlines()[-1][:160])
            failed.append(label)
            continue
        shutil.copy(rendered, target)
        print(f"  [{index:>2}/{len(VARIANTS)}] {axis}/{label:16s} "
              f"{time.time()-started:5.1f}s -> {osp.basename(target)}")
        done.append((axis, label, target))

    # one contact sheet per axis
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    for axis in dict.fromkeys(a for a, _, _ in done):
        items = [(l, p) for a, l, p in done if a == axis]
        cols = min(len(items), 3)
        rows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 5.0 * rows),
                                 squeeze=False)
        for ax, (label, path) in zip(axes.ravel(), items):
            img = mpimg.imread(path)
            alpha = img[..., 3:4]
            ax.imshow(np.clip(img[..., :3] * alpha + (1 - alpha), 0, 1))
            ax.set_title(label, fontsize=13)
            ax.axis("off")
        for ax in axes.ravel()[len(items):]:
            ax.axis("off")
        plt.tight_layout()
        sheet = osp.join(args.out, f"_sheet_{axis}.png")
        plt.savefig(sheet, dpi=72, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  sheet -> {sheet}")

    print(f"\n{len(done)}/{len(VARIANTS)} rendered into {args.out}")
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--width", type=int, default=1100)
    sys.exit(main(parser.parse_args()))
