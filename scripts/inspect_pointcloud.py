"""Inspect any supported point cloud, and optionally cache it as .npz.

Useful when a colleague sends a new drop and you want to know what
colorizations it yields before writing a figure config. Caching to .npz makes
repeated renders of a big cloud much faster than re-parsing a gzipped PLY.

    python scripts/inspect_pointcloud.py <file> [--palettes palettes.json]
    python scripts/inspect_pointcloud.py <file> --palettes p.json --cache out.npz
"""
import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_point_cloud
from figlib.data import save_npz


def main(args):
    cloud = load_point_cloud(args.path, palettes=args.palettes, colors=args.colors)
    cloud.subsample(args.subsample)
    print()
    print(cloud.summary())
    print()
    print("Colorizations available for `data.default_color` / `--image`:")
    for name in cloud.names:
        print(f"  {name}")

    if args.cache:
        save_npz(cloud, args.cache)
        size_mb = osp.getsize(args.cache) / 1e6
        print(f"\nCached to {args.cache} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Point cloud file (.pt/.ply/.ply.gz/.npz/.las/.laz)")
    parser.add_argument("--palettes", default=None, help="Palette JSON")
    parser.add_argument("--colors", nargs="*", default=None,
                        help="Keep only these colorizations")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Keep at most N points (applies to the cache too)")
    parser.add_argument("--cache", default=None, help="Write a .npz cache here")
    main(parser.parse_args())
