"""Generate a figure config for every MALIBU3D ROI in a drop.

The ROIs differ in extent (25 to 150 subtiles) and in point density (0.12 to
0.26 m spacing), so a camera and sphere radius copied from one tile do not
transfer. Both are derived here from each cloud's own geometry, anchored on the
hand-tuned reference tile:

  * camera   — the reference pose, with its distance scaled by the ROI's XY
               extent, so every tile is framed the same way
  * voxel    — the reference radius-to-spacing ratio, times this ROI's spacing,
               times sqrt(n_full / n_kept) when subsampling

    python scripts/prepare_rois.py --drop data/malibu3d/send_29_07_v2/blender_export
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_config, load_point_cloud     # noqa: E402

# The hand-tuned reference: pose, and the ROI extent it was framed for
REFERENCE = "malibu3d_D075_UU-S1-3"
REFERENCE_EXTENT = 512.0

# One hue per network type, so the three read apart when several are shown.
# Predictions get a contrasting hue from their ground truth rather than a
# lighter shade of it: on a busy scene a shade reads as "the same network drawn
# twice", which is exactly the comparison the figure needs to avoid. Blue/orange
# for roads is also the safest pair under the common colour-vision deficiencies.
GRAPH_COLORS = {
    ("roads", "gt"): [0.30, 0.55, 1.00],
    ("roads", "pred"): [1.00, 0.42, 0.20],
    ("railroads", "gt"): [1.00, 0.45, 0.25],
    ("railroads", "pred"): [0.95, 0.30, 0.75],
    ("transmission", "gt"): [1.00, 0.85, 0.25],
    ("transmission", "pred"): [0.45, 0.95, 0.55],
}


def classify_graph(filename):
    """(kind, role) from a graph filename, e.g. `..._ROADS_pred_graph.gpkg`."""
    upper = osp.basename(filename).upper()
    if "RAILROAD" in upper:
        kind = "railroads"
    elif "TRANSMISSION" in upper:
        kind = "transmission"
    else:
        kind = "roads"
    return kind, ("pred" if "_PRED" in upper else "gt")


def main(args):
    reference = load_config(f"configs/figures/{REFERENCE}.yaml")
    ref_voxel = float(reference["data"]["voxel"])

    # The camera anchor is separable from the voxel anchor on purpose. Once a
    # reference tile's pose has been tuned for one specific figure — a tight
    # teaser framing, say — propagating it to a fresh ROI produces something
    # unlike the rest of the family. `--camera-from` lets a sibling that still
    # holds the plain overview pose seed the new tiles instead, while the voxel
    # ratio stays anchored on REFERENCE, whose geometry the constant below
    # encodes.
    camera_source = args.camera_from or REFERENCE
    ref_camera = load_config(f"configs/figures/{camera_source}.yaml")["camera"]
    ref_translation = np.asarray(ref_camera["translation"], dtype=float)
    if camera_source != REFERENCE:
        print(f"Camera seeded from {camera_source} "
              f"(fov {ref_camera['fov_x_deg']}), voxel from {REFERENCE}\n")

    # A sub-drop (e.g. predictions/) shares the parent's palettes.json rather
    # than carrying its own copy.
    palettes_path = args.palettes or osp.join(args.drop, "palettes.json")
    if not osp.exists(palettes_path):
        parent = osp.join(osp.dirname(osp.abspath(args.drop)), "palettes.json")
        if not osp.exists(parent):
            raise SystemExit(f"No palettes.json in {args.drop} or its parent")
        palettes_path = parent
        print(f"Using the parent drop's palettes: {palettes_path}\n")

    rois = sorted(glob.glob(osp.join(args.drop, "*", "*_meta.json")))
    print(f"Found {len(rois)} ROI sidecars in {args.drop}\n")

    written, skipped = [], []
    for meta_path in rois:
        directory = osp.dirname(meta_path)
        roi = osp.basename(meta_path).replace("_meta.json", "")
        ply = osp.join(directory, f"{roi}.ply.gz")
        if not osp.exists(ply):
            print(f"  {roi}: SKIP — no point cloud ({osp.basename(ply)} missing)")
            skipped.append(roi)
            continue

        name = f"malibu3d_{roi}{args.suffix}"
        path = osp.join("configs", "figures", f"{name}.yaml")
        # Never clobber a config whose camera has been tuned by hand. Checked
        # before loading, so re-running over a processed drop is instant.
        if name == REFERENCE or (osp.exists(path) and not args.overwrite):
            print(f"  {roi}: keeping existing {osp.basename(path)}")
            written.append(name)
            continue

        cloud = load_point_cloud(
            ply,
            palettes=palettes_path,
            cache_dir=args.cache_dir,
            log=lambda m: None)
        cloud.drop_void(["elevation"], log=lambda m: None)

        extent = cloud.pos.max(0) - cloud.pos.min(0)
        extent_xy = float(max(extent[0], extent[1]))
        n_points = len(cloud)
        spacing = math.sqrt((extent[0] * extent[1]) / n_points)

        keep = min(n_points, args.max_points)
        thinning = math.sqrt(n_points / keep)
        voxel = ref_voxel * (spacing / (REFERENCE_EXTENT / math.sqrt(3972917))) * thinning
        # the reference ratio expressed directly, to avoid drift if it changes
        voxel = round(spacing * (ref_voxel / 0.2570) * thinning, 3)

        scale = extent_xy / REFERENCE_EXTENT
        translation = (ref_translation * scale).round(4).tolist()

        graphs = [osp.relpath(p) for p in sorted(
            glob.glob(osp.join(directory, f"{roi}_*_graph.gpkg")))]
        if not graphs:
            # A single-subtile crop keeps the parent tile's graph filenames, so
            # the ROI-prefixed glob misses them. One directory holds one ROI, so
            # falling back to everything in it is safe.
            graphs = [osp.relpath(p) for p in sorted(
                glob.glob(osp.join(directory, "*_graph.gpkg")))]
            if graphs:
                print(f"    (graphs found under a different prefix: "
                      f"{', '.join(osp.basename(g) for g in graphs)})")

        with open(path, "w") as f:
            f.write(f"""# MALIBU3D ROI {roi} — auto-generated by scripts/prepare_rois.py
# {extent_xy:.0f} x {float(extent[1]):.0f} m, {n_points} points, {spacing:.3f} m spacing.
# Camera is the {camera_source} pose scaled x{scale:.2f} for this ROI's extent —
# a starting point, not a tuned view. Refine it in the GUI and pull it back with
# scripts/scene_to_config.py.
extends: ../malibu3d.yaml

data:
  path: {osp.relpath(ply)}
  voxel: {voxel}
""")
            if args.palette_overrides:
                f.write(f"  palette_overrides:\n")
                for entry in args.palette_overrides.split(","):
                    f.write(f"    - {entry.strip()}\n")
            if keep < n_points:
                f.write(f"  subsample: {keep}   "
                        f"# of {n_points}; voxel already scaled by sqrt(n_full/n_kept)\n")
            f.write(f"""
camera:
  translation: {translation}
  quaternion: {[round(v, 7) for v in ref_camera['quaternion']]}
  fov_x_deg: {ref_camera['fov_x_deg']}
  near: {ref_camera['near']}
  far: {max(1000, int(extent_xy * 4))}
""")
            # Networks actually get drawn, not just listed. A ROI with no
            # graph file simply gets no `graphs:` block, and the figure script
            # skips the whole stage — no crash.
            if graphs:
                f.write("\ngraphs:\n  items:\n")
                for path_g in graphs:
                    kind, role = classify_graph(path_g)
                    colour = GRAPH_COLORS.get((kind, role),
                                              GRAPH_COLORS[("roads", role)])
                    f.write(f"    - path: {path_g}\n")
                    f.write(f"      name: {kind}_{role}\n")
                    f.write(f"      color: {colour}\n")
            else:
                f.write("\n# No network graph delivered for this ROI.\n")

        print(f"  {roi}: {n_points:>9} pts, {extent_xy:6.0f} m, spacing {spacing:.3f} m "
              f"-> voxel {voxel}, keep {keep}, camera x{scale:.2f}")
        written.append(name)
        del cloud

    print(f"\nWrote {len(written)} configs; skipped {len(skipped)}: {skipped}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drop", required=True, help="blender_export directory")
    parser.add_argument("--max-points", type=int, default=8_000_000,
                        help="Cap for preview renders (default 8M)")
    parser.add_argument("--palettes", default=None,
                        help="palettes.json (default: in the drop, else its parent)")
    parser.add_argument("--suffix", default="",
                        help="appended to the config name, e.g. _pred, so a "
                             "second drop of the same ROIs does not collide")
    parser.add_argument("--palette-overrides", default=None,
                        help="comma-separated override files written into "
                             "data.palette_overrides of each config")
    parser.add_argument("--camera-from", default=None,
                        help="config name whose camera seeds new ROIs "
                             "(default: the reference tile). Use a sibling when "
                             "the reference pose has been tuned for one figure.")
    parser.add_argument("--cache-dir", default="data/.figcache")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace configs that already exist (never the reference)")
    main(parser.parse_args())
