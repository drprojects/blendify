"""Push a changed class palette into every exported .blend, without re-exporting.

A palette edit in `configs/palettes/` reaches new renders immediately — colours
are applied at load, and the caches hold raw label fields, not colours. Already
exported .blend files are the exception: their colours are baked. This rebuilds
the lookup table with the same code an export uses and hands it to
`scripts/blend_update_palette.py`, which runs inside Blender.

    python scripts/refresh_class_palette.py --layer semantic \
        --config configs/figures/malibu3d_D075_UU-S1-3.yaml \
        --blends "data/malibu3d/.../blender_export/*/*.blend"

`--dry-run` prints the table it would apply and touches nothing.
"""
import argparse
import glob
import json
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from figlib import blender_palette
from figlib.config import load_config
from figlib.palettes import load_palettes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--config", required=True,
                        help="any figure config; supplies the palette files and void colour")
    parser.add_argument("--blends", required=True, help="glob of .blend files")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--exclude", default="_before_",
                        help="skip paths containing this (default: backups)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    palettes = load_palettes(cfg["data"]["palettes"], cfg["data"]["palette_overrides"])
    if args.layer not in palettes:
        raise SystemExit(f"{args.layer!r} not in the palettes")

    raw_lut, display, names, void = blender_palette.build_lut(
        palettes[args.layer], void_color=cfg["void"]["color"])

    print(f"{args.layer}: {len(names)} classes, void slots {void}")
    for index, (name, srgb) in enumerate(zip(names, raw_lut)):
        mark = " (void -> muted)" if index in void else ""
        print(f"  {index:2d} {name:24s} #{srgb[0]:02X}{srgb[1]:02X}{srgb[2]:02X}{mark}")

    targets = [p for p in sorted(glob.glob(args.blends)) if args.exclude not in p]
    print(f"\n{len(targets)} .blend files")
    if args.dry_run:
        print("--dry-run: nothing modified")
        return

    payload = {"layer": args.layer, "names": names,
               "colors": np.asarray(display, dtype=float).tolist(), "void": void}
    script = osp.join(osp.dirname(osp.abspath(__file__)), "blend_update_palette.py")

    ok = skipped = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = osp.join(tmp, "palette.json")
        with open(path, "w") as handle:
            json.dump(payload, handle)

        for index, blend in enumerate(targets, 1):
            result = subprocess.run(
                [args.blender, "--background", blend, "--python", script,
                 "--", "--payload", path, "--out", blend],
                capture_output=True, text=True)
            line = next((l for l in result.stdout.splitlines()
                         if "recoloured" in l or "SKIP" in l), "")
            if result.returncode != 0:
                failed += 1
                print(f"[{index:2d}/{len(targets)}] {osp.basename(blend):42s} FAILED")
                print(result.stderr[-1500:])
            elif "SKIP" in line:
                skipped += 1
                print(f"[{index:2d}/{len(targets)}] {osp.basename(blend):42s} skipped "
                      f"(no class_{args.layer})")
            else:
                ok += 1
                print(f"[{index:2d}/{len(targets)}] {osp.basename(blend):42s} "
                      f"{line.strip().split(': ', 1)[-1]}")

    print(f"\nupdated {ok}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
