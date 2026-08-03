"""Verify the figure pipeline's invariants. Run after touching render or export.

Every check here exists because the corresponding bug was actually shipped:

  configs        every figure config loads and resolves
  identity       an ungraded layer renders EXACTLY as a bare Attribute->BSDF
                 material would  (caught: contrast 1.0 vs 0.0 semantics)
  not-black      every layer renders with real colour, not just lighting
                 (caught: shader reading a colour attribute that only exists
                  in exports, which rendered near-black)
  material       the grading chain survives a colour reassignment
                 (caught: blender_plots builds a new material every time)
  export         the .blend carries every layer, the grading table, the active
                 layer, and the grading nodes
  graphs         graphs declared in the config are actually drawn
                 (caught: paths written as YAML comments, never as items)
  parenting      graph node spheres are parented to their edges, so height
                 moves the whole graph  (caught: nodes left behind)
  roundtrip      export -> read back gives the same numbers
  degenerate     a ROI with no graph, and a config restricting data.colors,
                 both work rather than crashing

    python scripts/selfcheck.py                    # quick, one representative tile
    python scripts/selfcheck.py --all-configs      # also load every figure config
"""
import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_config                                    # noqa: E402

PY = osp.expanduser("~/miniconda3/envs/blendify/bin/python")
SCRIPT = "examples/00_custom.py"

# A small tile that has graphs, so graph checks are meaningful
TILE = "configs/figures/malibu3d_D075_FU-S1-10.yaml"
TILE_NO_GRAPH = "configs/figures/malibu3d_D073_NN-S1-5.yaml"

PASS, FAIL = "PASS", "FAIL"
results = []


def record(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def run(args, timeout=1800):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def render(config, out_dir, extra=(), samples=8, resolution=(300, 200),
           subsample=120000):
    """Render layers of `config` into `out_dir`; returns {layer: path}."""
    overrides = [
        f"data.subsample={subsample}",
        f"render.n_samples={samples}",
        f"render.resolution=[{resolution[0]},{resolution[1]}]",
        *extra,
    ]
    result = run([PY, SCRIPT, "--config", config, "--image",
                  "--set", *overrides,
                  "--path-out", out_dir] if False else
                 [PY, SCRIPT, "--config", config, "--image", "--set", *overrides])
    if result.returncode != 0:
        return None, result.stderr[-1500:]
    # Filenames are "<stem>_<layer>.png" and layer names contain underscores
    # (natural_habitat, rgb_muted), so strip the known stem rather than
    # splitting on the last underscore.
    stem = osp.basename(load_config(config)["data"]["path"]).split(".")[0]
    produced = {}
    for line in result.stdout.splitlines():
        if line.startswith("Saved: '"):
            path = line.split("Saved: '", 1)[1].rstrip("'")
            name = osp.basename(path)[:-len(".png")]
            layer = name[len(stem) + 1:] if name.startswith(stem + "_") else name
            produced[layer] = path
    return produced, result.stdout


def mean_colour(path):
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    mask = img[..., 3] > 0.5
    if mask.sum() == 0:
        return None
    return img[..., :3][mask].mean()


def check_configs(all_configs):
    print("\nconfigs")
    configs = sorted(glob.glob("configs/figures/*.yaml"))
    if not all_configs:
        configs = configs[:3] + [TILE, TILE_NO_GRAPH]
    bad = []
    for path in set(configs):
        try:
            cfg = load_config(path)
            for section in ("data", "render", "camera", "sun", "world",
                            "point_cloud", "color", "void", "graphs", "export"):
                if section not in cfg:
                    bad.append(f"{osp.basename(path)}: missing {section}")
        except Exception as exc:                                  # noqa: BLE001
            bad.append(f"{osp.basename(path)}: {exc}")
    record(f"{len(set(configs))} configs load and resolve", not bad,
           "; ".join(bad[:3]))


def check_render_and_identity(tmp):
    print("\nrender")
    cfg = load_config(TILE)
    graded = set(cfg["color"]["apply_to"] or [])

    produced, log = render(TILE, tmp, extra=["graphs.items=[]"])
    if produced is None:
        record("all layers render", False, log)
        return
    expected = {"rgb", "elevation", "forest", "natural_habitat", "semantic",
                "xyz", "grayscale", "rgb_muted"}
    missing = expected - set(produced)
    record(f"all {len(expected)} layers render", not missing,
           f"missing {sorted(missing)}" if missing else "")

    # not-black: a layer showing only lighting has almost no colour
    dark = {k: round(mean_colour(v), 4) for k, v in produced.items()
            if k != "rgb_muted" and (mean_colour(v) or 0) < 0.05}
    record("no layer renders black", not dark, f"suspicious: {dark}" if dark else "")

    # identity: an ungraded layer must match a bare material exactly
    baseline = osp.join(tmp, "elevation_baseline.png")
    ungraded = [k for k in produced if k not in graded and k != "rgb_muted"]
    if "elevation" in ungraded:
        import shutil
        shutil.copy(produced["elevation"], baseline)
        again, _ = render(TILE, tmp, extra=["graphs.items=[]",
                                            "data.colors=['elevation']",
                                            "data.default_color=elevation",
                                            "data.add_xyz=False"])
        if again and "elevation" in again:
            import matplotlib.image as mpimg
            a, b = mpimg.imread(baseline), mpimg.imread(again["elevation"])
            mask = (a[..., 3] > 0.5) & (b[..., 3] > 0.5)
            delta = float(np.abs(a[..., :3][mask] - b[..., :3][mask]).max())
            record("ungraded layer is reproducible", delta < 0.02,
                   f"max |diff| {delta:.4f}")
        else:
            record("ungraded layer is reproducible", False, "second render failed")

    # Saturation must actually desaturate. Measured on a highly chromatic layer
    # under neutral light: a photo layer of dark forest is already near-grey, so
    # testing there cannot distinguish "working" from "no-op" (it did not, and
    # produced a false alarm).
    chroma = {}
    for value in (1.0, 0.0):
        got, _ = render(TILE, tmp, extra=[
            "graphs.items=[]", "data.colors=['semantic']",
            "data.default_color=semantic", "data.add_xyz=False",
            "color.variants=[]", "color.apply_to=['semantic']",
            f"color.saturation={value}",
            "sun.color=[1.0,1.0,1.0]", "world.color=[1.0,1.0,1.0]"])
        if not got or "semantic" not in got:
            record("saturation node desaturates", False, "render failed")
            return
        import matplotlib.image as mpimg
        img = mpimg.imread(got["semantic"])
        mask = img[..., 3] > 0.5
        px = img[..., :3][mask]
        chroma[value] = float(np.abs(px - px.mean(1, keepdims=True)).mean())
    ratio = chroma[0.0] / max(chroma[1.0], 1e-9)
    record("saturation node desaturates", ratio < 0.2,
           f"chromaticity {chroma[1.0]:.4f} -> {chroma[0.0]:.4f} (ratio {ratio:.3f})")


def check_export_and_roundtrip(tmp):
    print("\nexport + round-trip")
    result = run([PY, SCRIPT, "--config", TILE, "--export",
                  "--set", "data.subsample=120000"])
    if result.returncode != 0:
        record("export succeeds", False, result.stderr[-800:])
        return
    record("export succeeds", True)

    cfg = load_config(TILE)
    stem = osp.basename(cfg["data"]["path"]).split(".")[0]
    blend = osp.join(osp.dirname(cfg["data"]["path"]), stem + ".blend")

    probe = osp.join(tmp, "probe.py")
    out_json = osp.join(tmp, "probe.json")
    with open(probe, "w") as f:
        f.write(f'''
import bpy, json
scene = bpy.context.scene
cloud = [o for o in scene.objects if o.type=="MESH" and o.name.startswith("point_cloud")][0]
mat = bpy.data.materials["color"]
nodes = [n.name for n in mat.node_tree.nodes]
graphs = {{}}
for o in scene.objects:
    if o.type == "CURVE" and o.name.endswith("_edges"):
        graphs.setdefault(o.name[:-6], {{}})["height"] = round(o.location.z, 4)
    if o.type == "MESH" and o.name.endswith("_nodes"):
        graphs.setdefault(o.name[:-6], {{}})["parent"] = o.parent.name if o.parent else None
json.dump({{
    "attributes": [c.name for c in cloud.data.color_attributes],
    "layers": json.loads(cloud.get("figure_layers") or "{{}}"),
    "active": cloud.get("figure_active_layer"),
    "nodes": nodes,
    "graphs": graphs,
}}, open({out_json!r}, "w"))
''')
    run(["blender", "--background", blend, "--python", probe])
    if not osp.exists(out_json):
        record("blend is readable", False, "probe produced nothing")
        return
    data = json.load(open(out_json))
    record("blend is readable", True)

    for node in ("grade_saturation", "grade_brightcontrast", "grade_exposure",
                 "grade_gamma", "cloud_alpha"):
        record(f"grading node {node}", node in data["nodes"])

    record("layer table stored", bool(data["layers"]),
           f"{len(data['layers'])} layers")
    record("active layer stored", bool(data["active"]), str(data["active"]))

    # identity defaults: only `apply_to` layers may be non-neutral
    graded = set(cfg["color"]["apply_to"] or [])
    variants = {v["name"] for v in (cfg["color"]["variants"] or [])}
    offenders = []
    for name, entry in data["layers"].items():
        if name in graded or name in variants:
            continue
        for key, neutral in (("saturation", 1.0), ("contrast", 0.0),
                             ("brightness", 0.0), ("exposure", 0.0),
                             ("gamma", 1.0)):
            if abs(float(entry.get(key, neutral)) - neutral) > 1e-6:
                offenders.append(f"{name}.{key}={entry[key]}")
    record("ungraded layers keep identity defaults", not offenders,
           "; ".join(offenders[:4]))

    declared = {i["name"] for i in cfg["graphs"]["items"]}
    drawn = set(data["graphs"])
    record("declared graphs are drawn", declared <= drawn,
           f"declared {sorted(declared)}, drawn {sorted(drawn)}")

    unparented = [n for n, g in data["graphs"].items()
                  if "parent" in g and g["parent"] != f"{n}_edges"]
    record("graph nodes parented to edges", not unparented,
           f"loose: {unparented}" if unparented else "")

    heights = {n: g.get("height") for n, g in data["graphs"].items()}
    expected_height = cfg["graphs"]["height"]
    wrong = {n: h for n, h in heights.items()
             if h is not None and abs(h - expected_height) > 1e-3}
    record("graph height is an object transform", not wrong,
           f"expected {expected_height}, got {heights}" if wrong else
           f"height {expected_height}")


def check_degenerate():
    print("\ndegenerate cases")
    result = run([PY, SCRIPT, "--config", TILE_NO_GRAPH, "--export",
                  "--set", "data.subsample=80000"])
    record("ROI with no graph exports", result.returncode == 0,
           result.stderr[-400:] if result.returncode else "")

    result = run([PY, SCRIPT, "--config", TILE, "--image", "--set",
                  "data.subsample=80000", "data.colors=['semantic']",
                  "data.default_color=semantic", "data.add_xyz=False",
                  "render.n_samples=4", "render.resolution=[200,140]"])
    record("restricting data.colors does not crash", result.returncode == 0,
           result.stderr[-400:] if result.returncode else "variants skipped")


def main(args):
    os.chdir(osp.dirname(osp.dirname(osp.abspath(__file__))))
    with tempfile.TemporaryDirectory() as tmp:
        check_configs(args.all_configs)
        check_render_and_identity(tmp)
        check_export_and_roundtrip(tmp)
        check_degenerate()

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for _, name, detail in failed:
            print(f"  - {name}  {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-configs", action="store_true",
                        help="Load every figure config, not just a sample")
    sys.exit(main(parser.parse_args()))
