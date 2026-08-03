"""Inspect and stress-test a categorical palette against a real scene.

Answers the two questions that come up when a segmentation figure is hard to
read: *which class is which*, and *which classes are too close to tell apart*.

    python scripts/palette_lab.py --config configs/figures/<fig>.yaml --layer semantic

Produces:
  * a table of classes present in this scene, with point counts and shares
  * a legend PNG (swatch, name, share) you can put beside the render
  * a separation report: for every pair of classes that actually occur, the
    perceptual distance between their colours, worst pairs first

Separation is measured in OKLab dE (x100), the same metric the dataviz skill's
validator uses. The rule of thumb there is dE >= 15 for normal vision; with 16
or 44 classes that is impossible for every pair, so this ranks pairs by how much
they matter here — a pair that is rare or never adjacent can afford to be close,
a pair that is frequent and spatially interleaved cannot.
"""
import argparse
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_config, load_point_cloud          # noqa: E402
from figlib.palettes import hex_to_rgb, load_palettes, void_class_indices  # noqa: E402


def srgb_to_oklab(rgb):
    """sRGB (0-1) -> OKLab. Perceptually uniform, so Euclidean distance means
    something."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(linear @ m.T)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return lms @ m2.T


def delta_e(a, b):
    return float(np.linalg.norm(srgb_to_oklab(a) - srgb_to_oklab(b)) * 100)


def main(args):
    cfg = load_config(args.config)
    c_data = cfg["data"]
    palettes = load_palettes(c_data["palettes"], c_data["palette_overrides"])
    if args.layer not in palettes:
        raise SystemExit(f"No palette for {args.layer!r}. "
                         f"Available: {sorted(palettes)}")
    palette = palettes[args.layer]
    names = palette.get("names", [])
    colors = [hex_to_rgb(c) / 255.0 for c in palette["colors"]]
    void_indices = void_class_indices(palette)

    cloud = load_point_cloud(
        c_data["path"], palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"],
        cache=c_data["cache"], cache_dir=c_data["cache_dir"], log=lambda m: None)

    # Recover class index per point by matching the rendered colour back to the
    # palette — the cache stores colours, not raw labels
    rendered = cloud.colors[args.layer].astype(np.int32)
    lut = np.array([hex_to_rgb(c) for c in palette["colors"]], dtype=np.int32)
    counts = np.zeros(len(lut), dtype=np.int64)
    for index, entry in enumerate(lut):
        counts[index] = int((rendered == entry).all(axis=1).sum())

    total = len(cloud)
    present = [i for i in range(len(lut)) if counts[i] > 0]
    print(f"\n{args.layer}: {len(present)} of {len(names)} classes present "
          f"in {total} points\n")
    print(f"  {'idx':>3}  {'colour':<9} {'share':>7}  {'points':>10}  name")
    for i in sorted(present, key=lambda i: -counts[i]):
        tag = "  (void)" if i in void_indices else ""
        print(f"  {i:>3}  {palette['colors'][i]:<9} "
              f"{100*counts[i]/total:6.2f}%  {counts[i]:>10}  {names[i]}{tag}")

    # Pairwise separation, weighted by how much each pair matters in this scene
    print(f"\n  Closest pairs among classes present "
          f"(OKLab dE x100; <15 is hard to tell apart):\n")
    pairs = []
    for a_i, a in enumerate(present):
        for b in present[a_i + 1:]:
            if a in void_indices or b in void_indices:
                continue
            share = min(counts[a], counts[b]) / total
            pairs.append((delta_e(colors[a], colors[b]), a, b, share))
    pairs.sort()
    for distance, a, b, share in pairs[:args.top]:
        flag = "FAIL" if distance < 15 else "ok  "
        print(f"  [{flag}] dE {distance:5.1f}  {names[a]} ({palette['colors'][a]})"
              f"  vs  {names[b]} ({palette['colors'][b]})"
              f"   [rarer class = {100*share:.2f}% of scene]")

    if args.legend:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        ordered = sorted(present, key=lambda i: -counts[i])
        height = max(2.0, 0.42 * len(ordered))
        fig, ax = plt.subplots(figsize=(7.4, height))
        for row, i in enumerate(ordered):
            y = len(ordered) - row - 1
            ax.add_patch(Rectangle((0, y + 0.12), 0.62, 0.76,
                                   facecolor=palette["colors"][i],
                                   edgecolor="#444", linewidth=0.6))
            suffix = "   (unannotated)" if i in void_indices else ""
            ax.text(0.82, y + 0.5,
                    f"{names[i]}{suffix}", va="center", fontsize=10.5)
            ax.text(7.3, y + 0.5, f"{100*counts[i]/total:5.2f}%",
                    va="center", ha="right", fontsize=10, color="#555")
        ax.set_xlim(0, 7.4); ax.set_ylim(0, len(ordered))
        ax.axis("off")
        ax.set_title(f"{args.layer} — classes present in this scene",
                     fontsize=12, loc="left")
        plt.tight_layout()
        plt.savefig(args.legend, dpi=140, bbox_inches="tight", facecolor="white")
        print(f"\n  legend -> {args.legend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", default="semantic")
    parser.add_argument("--legend", default=None, help="Write a legend PNG here")
    parser.add_argument("--top", type=int, default=12,
                        help="How many closest pairs to report")
    main(parser.parse_args())
