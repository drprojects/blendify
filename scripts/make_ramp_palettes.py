"""Build palette entries that render one continuous field through several colormaps.

Trying colormaps in the GUI means one render, one manual save, repeat. This
emits a palette-override JSON with one entry per colormap — `elevation_berlin`,
`elevation_oslo`, ... — so a single CLI run renders the whole set, named
`<tile>_elevation_<map>.png`, with everything but the ramp held fixed.

    python scripts/make_ramp_palettes.py --field elevation \
        --maps berlin,coolwarm,managua,cmc.oslo,cmc.tofino,cmc.vanimo \
        --out configs/palettes/malibu3d_elevation_ramps.json

The base overrides are copied in first, so the derived habitat layers and the
absolute elevation range survive; only new entries are added.

`figlib/palettes.py` interpolates `color_stops_rgb` as **evenly spaced** stops,
while `colormaps.json` stores them at greedy non-uniform positions (they drive a
Blender ColorRamp, which carries its own positions). So each map is resampled
onto a uniform grid here — handing the non-uniform stops over directly would
distort every ramp.
"""
import argparse
import json
import os.path as osp

import numpy as np

BASE_OVERRIDES = "configs/palettes/malibu3d_extra.json"
COLORMAPS = "configs/palettes/colormaps.json"


def resample(stops, count):
    """Uniformly spaced 0-255 RGB triples reproducing a non-uniform ramp."""
    stops = np.asarray(stops, dtype=float)
    grid = np.linspace(0.0, 1.0, count)
    rgb = np.stack([np.interp(grid, stops[:, 0], stops[:, c + 1]) for c in range(3)],
                   axis=1)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", required=True, help="raw field to colorize")
    parser.add_argument("--maps", required=True, help="comma-separated colormap names")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stops", type=int, default=64)
    parser.add_argument("--base", default=BASE_OVERRIDES)
    parser.add_argument("--colormaps", default=COLORMAPS)
    args = parser.parse_args()

    with open(args.base) as handle:
        palettes = json.load(handle)
    with open(args.colormaps) as handle:
        colormaps = json.load(handle)["maps"]

    template = dict(palettes.get(args.field, {}))
    if not template:
        raise SystemExit(f"{args.field!r} has no entry in {args.base}")
    print(f"template from {args.field}: vmin={template.get('vmin')} "
          f"vmax={template.get('vmax')} gamma={template.get('gamma')} "
          f"nan={template.get('nan_color')}")

    names = []
    for name in [m.strip() for m in args.maps.split(",") if m.strip()]:
        if name not in colormaps:
            raise SystemExit(f"unknown colormap {name!r}; not in {args.colormaps}")
        short = name.split(".")[-1]
        key = f"{args.field}_{short}"
        entry = dict(template)
        entry["field"] = args.field
        entry["color_stops_rgb"] = resample(colormaps[name]["stops"], args.stops)
        # A percentile range would move with each tile's own distribution and
        # make the comparison meaningless; keep whatever absolute range the
        # field already uses.
        entry.pop("percentile_low", None)
        entry.pop("percentile_high", None)
        palettes[key] = entry
        names.append(key)
        print(f"  {key:24s} <- {name} ({colormaps[name]['family']}, "
              f"{len(colormaps[name]['stops'])} stops -> {args.stops} uniform)")

    with open(args.out, "w") as handle:
        json.dump(palettes, handle, indent=1)
    print(f"\n{len(names)} entries -> {args.out}")
    print("data.colors=[" + ",".join(f"'{n}'" for n in names) + "]")


if __name__ == "__main__":
    main()
