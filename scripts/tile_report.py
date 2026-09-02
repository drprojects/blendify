"""Summarise what is actually inside each MALIBU3D tile, into one Markdown file.

Filenames encode a department and a two-letter landscape code and nothing else,
so choosing a tile for a figure otherwise means opening several 900 MB .blend
files and looking. This measures every tile from the parse cache — no
re-parsing, no rendering — and writes a document you can scan instead.

    python scripts/tile_report.py

It writes next to the data by default, not into the repo: the content is
specific to one delivered drop and `data/` is gitignored, so it is a data
artifact rather than a documented part of the pipeline.

Everything here is computed from the data except the `LANDMARKS` table below,
which is hand-written: nothing in a point cloud says "Eiffel Tower". The
latitude/longitude of each tile centre is derived, so an unfamiliar tile can be
identified in one click rather than guessed at.
"""
import argparse
import glob
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from figlib.config import load_config
from figlib.data import _cache_path, load_point_cloud
from figlib.palettes import is_continuous, load_palettes

# French department codes appearing in the tile names.
DEPARTMENTS = {
    "D067": "Bas-Rhin (Strasbourg)",
    "D068": "Haut-Rhin (Mulhouse / Colmar)",
    "D073": "Savoie (northern Alps)",
    "D075": "Paris",
}

LANDSCAPE = {"U": "urban", "A": "agricultural", "F": "forest", "N": "natural"}

# Hand-written; add to it as tiles are recognised. Not derivable from the data.
LANDMARKS = {
    "D075_UU-S1-3": "Eiffel Tower (confirmed by Damien, 2026-08-11)",
}

# Semantic classes grouped into the things you actually want to ask about.
GROUPS = {
    "built": ["Building"],
    "sealed": ["Impervious surface", "Other infrastructures", "Bridge"],
    "tree": ["Deciduous", "Coniferous"],
    "farm": ["Agricultural soil", "Vineyard", "Greenhouse"],
    "open": ["Herbaceous", "Brushwood", "Other soil", "Soil under vegetation"],
    "water": ["Water", "Swimming pool"],
}


def lambert93_to_wgs84(x, y):
    """EPSG:2154 -> lon/lat degrees. Closed form, so no geospatial dependency.

    Lambert Conformal Conic 2SP on GRS80, the standard French projection.
    """
    a, f = 6378137.0, 1 / 298.257222101
    e = np.sqrt(2 * f - f * f)
    lon0, lat0 = np.deg2rad(3.0), np.deg2rad(46.5)
    lat1, lat2 = np.deg2rad(44.0), np.deg2rad(49.0)
    x0, y0 = 700000.0, 6600000.0

    def m(lat):
        return np.cos(lat) / np.sqrt(1 - e**2 * np.sin(lat)**2)

    def t(lat):
        return (np.tan(np.pi / 4 - lat / 2)
                / ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat)))**(e / 2))

    n = np.log(m(lat1) / m(lat2)) / np.log(t(lat1) / t(lat2))
    F = m(lat1) / (n * t(lat1)**n)
    rho0 = a * F * t(lat0)**n

    dx, dy = x - x0, rho0 - (y - y0)
    rho = np.hypot(dx, dy) * np.sign(n)
    theta = np.arctan2(dx, dy)
    t_val = (rho / (a * F))**(1 / n)

    lat = np.pi / 2 - 2 * np.arctan(t_val)
    for _ in range(12):                      # standard iterative inverse
        lat = np.pi / 2 - 2 * np.arctan(
            t_val * ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat)))**(e / 2))
    return np.rad2deg(theta / n + lon0), np.rad2deg(lat)


def breakdown(labels, palette, total):
    """Percent of points per class name, largest first, zero classes dropped."""
    remap = palette.get("remap")
    values = np.asarray(labels, dtype=np.int64)
    if remap is not None:
        table = np.asarray(remap, dtype=np.int64)
        values = table[np.clip(values, 0, len(table) - 1)]
    names = palette.get("names", [])
    counts = np.bincount(values, minlength=len(names))
    rows = [(names[i] if i < len(names) else f"<{i}>", 100.0 * c / total)
            for i, c in enumerate(counts) if c]
    return sorted(rows, key=lambda r: -r[1])


def character(groups, relief, built_pct, habitat, semantic_void):
    """A one-line description, from thresholds rather than impression."""
    # Some tiles are barely annotated semantically (a high glacier is 62% Void),
    # so the semantic groups alone would describe them as empty. The habitat
    # layer still covers them, and is the honest source there.
    if semantic_void >= 50:
        parts = []
        if habitat.get("aquatic", 0) >= 30:
            parts.append("glacier / water surface")
        if habitat.get("mineral", 0) >= 30:
            parts.append("bare rock")
        if parts:
            return (", ".join(parts) + ", mountainous" if relief >= 300
                    else ", ".join(parts))

    if built_pct >= 20:
        head = "dense city"
    elif built_pct >= 8:
        head = "urban"
    elif built_pct >= 3:
        head = "suburban"
    elif built_pct >= 0.5:
        head = "villages / scattered built-up"
    else:
        head = "essentially unbuilt"

    extras = []
    if groups["tree"] >= 40:
        extras.append("heavily wooded")
    elif groups["tree"] >= 15:
        extras.append("wooded")
    if groups["farm"] >= 20:
        extras.append("farmland")
    elif groups["farm"] >= 5:
        extras.append("some farmland")
    if groups["water"] >= 3:
        extras.append("substantial water")
    elif groups["water"] >= 0.5:
        extras.append("water present")
    if relief >= 300:
        extras.append("mountainous")
    elif relief >= 100:
        extras.append("hilly")
    elif relief <= 15:
        extras.append("flat")
    return head + (", " + ", ".join(extras) if extras else "")


def measure(cfg_path, palettes, log=print):
    cfg = load_config(cfg_path)
    c_data = cfg["data"]
    tile = osp.basename(cfg_path)[len("malibu3d_"):-len(".yaml")]
    cache = _cache_path(c_data["path"], c_data["palettes"], c_data["cache_dir"])
    if not osp.exists(cache):
        log(f"  {tile}: no cache, skipped (render it once to build one)")
        return None

    log(f"  {tile}")
    cloud = load_point_cloud(
        c_data["path"], palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"], colors=None,
        cache=True, cache_dir=c_data["cache_dir"], log=lambda *a: None)

    meta_path = osp.join(osp.dirname(c_data["path"]), f"{tile}_meta.json")
    meta = {}
    if osp.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
    # Cached z is relative to the exporter's origin; without adding it back the
    # altitudes are tile-local and a 3000 m glacier reads as "-468 m".
    z_origin = (meta.get("coord_translation") or [0, 0, 0])[2]

    pos = np.asarray(cloud.pos, dtype=np.float64)
    total = len(pos)
    span = pos.max(0) - pos.min(0)
    area = float(span[0] * span[1])

    info = {
        "tile": tile,
        "points": total,
        "span": span,
        "area_km2": area / 1e6,
        "density": total / max(area, 1),
        "config": cfg_path,
    }

    # Ground surface is exact here: `elevation` is height above the delivered
    # MNT, so z - elevation is the terrain, and its spread is the relief.
    height = np.asarray(cloud.fields.get("elevation"), dtype=np.float64)
    finite = np.isfinite(height)
    ground = pos[finite, 2] - height[finite]
    info["relief"] = float(np.percentile(ground, 98) - np.percentile(ground, 2))
    info["ground_range"] = (float(ground.min()) + z_origin,
                            float(ground.max()) + z_origin)
    info["tallest"] = float(np.nanmax(height[finite]))
    info["above_50m"] = int((height[finite] > 50).sum())
    info["elevation_void"] = 100.0 * (~finite).sum() / total

    tasks = {}
    for name, palette in palettes.items():
        if is_continuous(palette):
            continue
        field = palette.get("field", name)
        if field not in cloud.fields:
            continue
        tasks[name] = breakdown(cloud.fields[field], palette, total)
    info["tasks"] = tasks
    info["void"] = {
        name: sum(pct for cls, pct in rows_ if cls in ("Void", "N/A"))
        for name, rows_ in tasks.items()}

    semantic = dict(tasks.get("semantic", []))
    info["groups"] = {g: sum(semantic.get(c, 0.0) for c in members)
                      for g, members in GROUPS.items()}
    info["character"] = character(
        info["groups"], info["relief"], info["groups"]["built"],
        dict(tasks.get("habitat_type", [])), info["void"].get("semantic", 0.0))

    strength = cloud.fields.get("strength")
    info["strength"] = None if strength is None else {
        "median": float(np.median(strength)),
        "p98": float(np.percentile(strength, 98)),
        "max": float(np.max(strength)),
    }

    info["split"] = meta.get("pointcept_split", "?")
    info["graphs"] = meta.get("network_graphs", [])
    translation = meta.get("coord_translation")
    if translation:
        # Cached positions are source-relative, so the absolute centre is the
        # translation plus the cloud's own centre — not the translation alone,
        # which is only the origin the exporter happened to subtract.
        centre = pos[:, :2].mean(axis=0)
        lon, lat = lambert93_to_wgs84(translation[0] + centre[0],
                                      translation[1] + centre[1])
        info["lonlat"] = (float(lon), float(lat))
    else:
        info["lonlat"] = None
    return info


def render(rows, palettes):
    out = []
    w = out.append
    w("# MALIBU3D tiles — what is in each one\n")
    w("Generated by `scripts/tile_report.py` from the parse caches. "
      "Percentages are shares of **points**, not of area — tall vegetation and "
      "building facades carry more points per m² than bare ground, so a canopy "
      "reads higher here than it would on a map.\n")
    w("Tile names are `D<department>_<landscape><landscape>-S<sensor>-<index>`, "
      "where the two letters are the dominant landscapes: "
      + ", ".join(f"`{k}` {v}" for k, v in LANDSCAPE.items()) + ".\n")

    w("## At a glance\n")
    w("| Tile | Split | Character | Built | Tree | Farm | Water | Relief | "
      "Unlabelled sem/hab | Points | Graphs |")
    w("|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        g = r["groups"]
        # GitHub keeps underscores and hyphens in heading anchors and only
        # lowercases; stripping them here produced links that went nowhere.
        w(f"| [`{r['tile']}`](#{r['tile'].lower()}) "
          f"| {r['split']} | {r['character']} "
          f"| {g['built']:.1f}% | {g['tree']:.1f}% | {g['farm']:.1f}% "
          f"| {g['water']:.1f}% | {r['relief']:.0f} m "
          f"| {r['void'].get('semantic', 0):.0f}% / "
          f"{r['void'].get('natural_habitat', 0):.0f}% "
          f"| {r['points'] / 1e6:.1f} M | {len(r['graphs'])} |")
    w("")

    # Which tiles can exercise which task at all
    w("## Task coverage\n")
    w("How many classes actually appear, counting only classes above 0.1% of "
      "points — a class present at 0.001% will not be visible in a figure.\n")
    names = [n for n in ("semantic", "forest", "habitat_type", "moisture_regime",
                         "soil_chemistry", "bioclimatic_zone") if n in palettes]
    w("| Tile | " + " | ".join(names) + " |")
    w("|---" * (len(names) + 1) + "|")
    for r in rows:
        cells = []
        for name in names:
            rows_ = r["tasks"].get(name, [])
            usable = [c for c, pct in rows_ if pct >= 0.1 and c not in ("Void", "N/A")]
            cells.append(str(len(usable)) if usable else "—")
        w(f"| `{r['tile']}` | " + " | ".join(cells) + " |")
    w("")

    for r in rows:
        w(f"## {r['tile']}\n")
        department = DEPARTMENTS.get(r["tile"][:4], "?")
        code = r["tile"].split("_")[1][:2]
        landscapes = " + ".join(dict.fromkeys(LANDSCAPE.get(c, c) for c in code))
        w(f"**{r['character']}** · {department} · declared landscape: {landscapes} "
          f"· `{r['split']}` split\n")
        if r["tile"] in LANDMARKS:
            w(f"> **Landmark:** {LANDMARKS[r['tile']]}\n")

        w(f"- **Size** {r['points'] / 1e6:.2f} M points over "
          f"{r['span'][0]:.0f} x {r['span'][1]:.0f} m "
          f"({r['area_km2']:.3f} km2), {r['density']:.1f} pts/m2")
        w(f"- **Terrain** ground altitude {r['ground_range'][0]:.0f} to "
          f"{r['ground_range'][1]:.0f} m above sea level, "
          f"relief {r['relief']:.0f} m")
        w(f"- **Tallest structure** {r['tallest']:.0f} m above ground; "
          f"{r['above_50m']:,} points above 50 m")
        if r["lonlat"]:
            lon, lat = r["lonlat"]
            w(f"- **Location** {lat:.5f}, {lon:.5f} — "
              f"[map](https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=15/{lat:.5f}/{lon:.5f})")
        if r["graphs"]:
            # Splitting on "_" and taking [-2] turned TRANSMISSION_LINES into
            # LINES; strip the known prefix and suffix instead.
            kinds = [osp.basename(g)[len(r["tile"]) + 1:].replace("_graph.gpkg", "")
                     for g in r["graphs"]]
            w("- **Networks** " + ", ".join(f"`{k}`" for k in kinds))
        else:
            w("- **Networks** none")
        if r["strength"]:
            s = r["strength"]
            w(f"- **Intensity (`strength`)** median {s['median']:.4f}, "
              f"98th pct {s['p98']:.4f}, max {s['max']:.3f}")
        else:
            w("- **Intensity (`strength`)** not delivered for this tile")
        if r["elevation_void"] > 0:
            w(f"- **Void** {r['elevation_void']:.2f}% of points have no "
              f"`elevation` (dropped by `data.drop_void`)")
        w(f"- **Config** `{r['config']}`\n")

        for name, entries in r["tasks"].items():
            shown = [(c, p) for c, p in entries if p >= 0.05]
            if not shown:
                continue
            labelled = len([c for c, p in shown if c not in ("Void", "N/A")])
            unlabelled = r["void"].get(name, 0.0)
            w(f"<details><summary><b>{name}</b> — {labelled} labelled "
              f"class{'es' if labelled != 1 else ''} &ge;0.05%"
              + (f", {unlabelled:.1f}% unlabelled" if unlabelled >= 0.05 else "")
              + "</summary>\n")
            w("| class | % of points |")
            w("|---|--:|")
            for cls, pct in shown:
                w(f"| {cls} | {pct:.2f} |")
            w("\n</details>\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default="data/malibu3d/send_29_07_v2/blender_export/TILES.md")
    parser.add_argument("--configs", default="configs/figures/malibu3d_D*.yaml")
    args = parser.parse_args()

    paths = [p for p in sorted(glob.glob(args.configs))
             if not p.endswith(("_networks.yaml", "_graphs.yaml", "_strength.yaml"))]
    print(f"measuring {len(paths)} tiles")

    reference = load_config(paths[0])["data"]
    palettes = load_palettes(reference["palettes"], reference["palette_overrides"])

    rows = [r for r in (measure(p, palettes) for p in paths) if r]
    with open(args.out, "w") as handle:
        handle.write(render(rows, palettes))
    print(f"\n{len(rows)} tiles -> {args.out} "
          f"({osp.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
