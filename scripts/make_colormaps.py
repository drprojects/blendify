"""Build `configs/palettes/colormaps.json`, the ramp presets embedded in a .blend.

The GUI ramp is a Blender ColorRamp, which caps out at 32 elements, so each
colormap is resampled to the fewest stops that reproduce it to within a
perceptual tolerance, placed where the fit is worst rather than on a uniform
grid. Smooth maps end up with a handful of handles and stay comfortable to drag
by hand; kinked ones get more.

matplotlib is required; cmocean, cmcrameri and colorcet are optional and simply
skipped when absent. They are worth having — cmcrameri is Crameri's scientific
colour maps, the reference set in geoscience, and colorcet is Kovesi's
perceptually uniform set. Neither needs to be installed in the render
environment: this script only emits JSON, which is self-contained.

    python scripts/make_colormaps.py --out configs/palettes/colormaps.json

Qualitative maps (tab10, Set1, Paired, flag, prism, Crameri's `*S`) are
deliberately excluded. This ramp drives a *continuous* field; a categorical
palette stretched along one would read as banding that means nothing. Class
colours are edited through the palette editor instead.
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

# Blender's ColorRamp holds at most 32 elements.
MAX_STOPS = 32
TOLERANCE = 4.0 / 255.0      # max per-channel error, below visual threshold

SEQUENTIAL, DIVERGING, CYCLIC, MISC = "Sequential", "Diverging", "Cyclic", "Misc"

# matplotlib's own documented groupings, flattened to the four that matter here.
MPL_FAMILIES = {
    SEQUENTIAL: """viridis plasma inferno magma cividis
        Greys Purples Blues Greens Oranges Reds YlOrBr YlOrRd OrRd PuRd RdPu
        BuPu GnBu PuBu YlGnBu PuBuGn BuGn YlGn
        binary gist_yarg gist_gray gray bone pink spring summer autumn winter
        cool Wistia hot afmhot gist_heat copper""".split(),
    DIVERGING: """PiYG PRGn BrBG PuOr RdGy RdBu RdYlBu RdYlGn Spectral coolwarm
        bwr seismic berlin managua vanimo""".split(),
    CYCLIC: "twilight twilight_shifted hsv".split(),
    MISC: """ocean gist_earth terrain gist_stern gnuplot gnuplot2 CMRmap
        cubehelix brg gist_rainbow rainbow jet turbo nipy_spectral
        gist_ncar""".split(),
}

QUALITATIVE = {"Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2",
               "Set3", "tab10", "tab20", "tab20b", "tab20c", "flag", "prism"}

# Spelling aliases matplotlib carries for the same ramp; shipping both would
# just pad the dropdown with duplicates.
ALIASES = {"Grays", "grey", "gist_grey", "gist_yerg"}

CMOCEAN_DIVERGING = {"balance", "delta", "curl", "diff", "tarn", "topo"}
CMOCEAN_CYCLIC = {"phase"}

CRAMERI_DIVERGING = {"broc", "cork", "vik", "lisbon", "tofino", "berlin", "roma",
                     "bam", "vanimo", "managua", "oleron", "bukavu", "fes"}

# colorcet's human-readable aliases. The CET_* identifiers cover the same
# ground but are opaque in a dropdown, so only the named ones are shipped.
COLORCET = {
    "fire": SEQUENTIAL, "bgy": SEQUENTIAL, "bgyw": SEQUENTIAL, "bmw": SEQUENTIAL,
    "bmy": SEQUENTIAL, "kbc": SEQUENTIAL, "kgy": SEQUENTIAL, "blues": SEQUENTIAL,
    "kb": SEQUENTIAL, "kg": SEQUENTIAL, "kr": SEQUENTIAL, "gouldian": SEQUENTIAL,
    "dimgray": SEQUENTIAL, "gray": SEQUENTIAL, "isolum": SEQUENTIAL,
    "bkr": DIVERGING, "bky": DIVERGING, "cwr": DIVERGING, "gwv": DIVERGING,
    "coolwarm": DIVERGING, "bjy": DIVERGING,
    "colorwheel": CYCLIC, "cyclic_mygbm_30_95_c78": CYCLIC,
    "rainbow4": MISC, "rainbow": MISC, "glasbey": None,     # None -> skip
}


def fit_stops(sample, tolerance=TOLERANCE):
    """Fewest stops reproducing `sample(t)` within `tolerance`.

    Stops are placed greedily at wherever the current piecewise-linear fit is
    worst, not on a uniform grid. Uniform spacing wastes handles on the flat
    parts of a ramp and, worse, cannot represent the hard step at sea level in
    a multi-sequential map like `oleron` or `topo` at any resolution — the
    greedy pass lands two stops on either side of the jump and reproduces it.
    """
    fine = np.linspace(0, 1, 512)
    truth = sample(fine)
    chosen = [0, len(fine) - 1]

    while True:
        pos = fine[chosen]
        picked = truth[chosen]
        approx = np.stack([np.interp(fine, pos, picked[:, c]) for c in range(3)],
                          axis=1)
        deviation = np.abs(truth - approx).max(axis=1)
        error = float(deviation.max())
        if error <= tolerance or len(chosen) >= MAX_STOPS:
            return pos, picked, error
        chosen = sorted(chosen + [int(deviation.argmax())])


def entry(name, cmap, family, source):
    def sample(t):
        return np.asarray(cmap(np.asarray(t, dtype=float)))[:, :3]
    pos, colors, error = fit_stops(sample)
    stops = [[round(float(p), 5)] + [round(float(c), 5) for c in rgb]
             for p, rgb in zip(pos, colors)]
    return name, {"family": family, "source": source, "stops": stops,
                  "error": round(error * 255, 1)}


def from_matplotlib():
    from matplotlib import colormaps
    lookup = {n: fam for fam, names in MPL_FAMILIES.items() for n in names}
    out, unclassified = {}, []
    for name in colormaps:
        if name.endswith("_r") or name in QUALITATIVE or name in ALIASES:
            continue
        if name.startswith(("cet_", "cmo.", "cmc.", "cmr.")):
            continue                      # registered by an optional library
        family = lookup.get(name)
        if family is None:
            family = MISC
            unclassified.append(name)
        key, value = entry(name, colormaps[name], family, "matplotlib")
        out[key] = value
    if unclassified:
        print(f"  note: filed under {MISC}: {', '.join(sorted(unclassified))}")
    return out


def from_cmocean():
    import cmocean
    out = {}
    for name in cmocean.cm.cmapnames:
        family = (DIVERGING if name in CMOCEAN_DIVERGING else
                  CYCLIC if name in CMOCEAN_CYCLIC else SEQUENTIAL)
        key, value = entry(f"cmo.{name}", getattr(cmocean.cm, name), family, "cmocean")
        out[key] = value
    return out


def from_cmcrameri():
    from cmcrameri import cm
    out = {}
    for name, cmap in cm.cmaps.items():
        if name.endswith("_r") or name.endswith("S"):    # S = categorical
            continue
        family = (CYCLIC if name.endswith("O") else
                  DIVERGING if name in CRAMERI_DIVERGING else SEQUENTIAL)
        key, value = entry(f"cmc.{name}", cmap, family, "cmcrameri")
        out[key] = value
    return out


def from_colorcet():
    import colorcet
    out = {}
    for name, family in COLORCET.items():
        if family is None:
            continue
        cmap = getattr(colorcet, f"m_{name}", None)
        if cmap is None:
            print(f"  note: colorcet has no {name!r}, skipped")
            continue
        key, value = entry(f"cet.{name}", cmap, family, "colorcet")
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="configs/palettes/colormaps.json")
    args = parser.parse_args()

    maps = {}
    for label, builder in (("matplotlib", from_matplotlib), ("cmocean", from_cmocean),
                           ("cmcrameri", from_cmcrameri), ("colorcet", from_colorcet)):
        try:
            found = builder()
        except ImportError:
            print(f"{label:11s} not installed, skipped")
            continue
        overlap = set(found) & set(maps)
        if overlap:
            raise SystemExit(f"name collision from {label}: {sorted(overlap)}")
        maps.update(found)
        print(f"{label:11s} {len(found):3d} colormaps")

    if not maps:
        raise SystemExit("no colormaps built; is matplotlib installed?")

    payload = {"version": 2, "maps": dict(sorted(maps.items()))}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    counts = {}
    stops = []
    for value in maps.values():
        counts[value["family"]] = counts.get(value["family"], 0) + 1
        stops.append(len(value["stops"]))
    rough = sorted((v["error"], k) for k, v in maps.items())[-4:]
    print(f"\n{len(maps)} colormaps -> {args.out} "
          f"({osp.getsize(args.out) / 1024:.0f} KB)")
    print("  by family: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"  stops: min {min(stops)}, median {int(np.median(stops))}, max {max(stops)}")
    print("  least faithful (max channel error /255): "
          + ", ".join(f"{n}={e:.0f}" for e, n in reversed(rough)))


if __name__ == "__main__":
    sys.exit(main())
