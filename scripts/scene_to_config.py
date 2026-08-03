"""Read camera / sun / world settings out of a .blend and write them into a figure YAML.

This closes the loop that used to be manual. Instead of reading values off the
Blender scripting console and pasting them into code:

    1. render or export the scene          00_custom.py --config X.yaml --export
    2. open X.blend in the Blender GUI, move the camera, tweak the sun,
       adjust the world background, then save the file
    3. pull the result back into the config
                                           scripts/scene_to_config.py \\
                                               --blend X.blend --config X.yaml

Step 3 replaces the whole `camera:` / `sun:` / `world:` blocks of the YAML.
Comments *inside* those blocks are lost; everything else in the file, including
comments above them, is preserved.
"""
import argparse
import json
import math
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

SECTIONS = ("camera", "sun", "world")
PULLABLE = SECTIONS + ("data", "point_cloud")   # data.voxel, point_cloud.alpha


def _round(values, ndigits=7):
    return [round(float(v), ndigits) for v in values]


def read_scene_external(blend_path, blender_bin):
    """Read the .blend using a standalone Blender binary.

    Preferred over the in-process `bpy`, because the GUI Blender is usually
    newer than the pinned `bpy` and old Blender cannot open new .blend files.
    """
    dumper = osp.join(osp.dirname(osp.abspath(__file__)), "blend_dump.py")
    with tempfile.TemporaryDirectory() as tmp:
        out_path = osp.join(tmp, "params.json")
        result = subprocess.run(
            [blender_bin, "--background", osp.abspath(blend_path),
             "--python", dumper, "--", "--out", out_path],
            capture_output=True, text=True)
        if not osp.exists(out_path):
            raise SystemExit(
                f"{blender_bin} could not read {blend_path}:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        with open(out_path) as f:
            payload = json.load(f)

    version = payload.pop("_blender_version", "?")
    resolution = payload.pop("_resolution", [None, None])
    print(f"  read with {blender_bin} (Blender {version})")
    return payload, resolution


def read_scene(blend_path):
    """Open a .blend with the in-process bpy and pull out the settings."""
    import bpy

    bpy.ops.wm.open_mainfile(filepath=osp.abspath(blend_path))
    scene = bpy.context.scene
    out = {}

    camera = scene.camera
    if camera is None:
        cameras = [o for o in scene.objects if o.type == "CAMERA"]
        camera = cameras[0] if cameras else None
    if camera is not None:
        if camera.rotation_mode != "QUATERNION":
            quaternion = camera.matrix_world.to_quaternion()
        else:
            quaternion = camera.rotation_quaternion
        entry = {
            "translation": _round(camera.location),
            "quaternion": _round([quaternion.w, quaternion.x,
                                  quaternion.y, quaternion.z]),
        }
        data = camera.data
        if data.type == "PERSP" and data.sensor_width:
            fov_x = 2 * math.atan(0.5 * data.sensor_width / data.lens)
            entry["fov_x_deg"] = round(math.degrees(fov_x), 4)
            entry["near"] = round(data.clip_start, 7)
            entry["far"] = round(data.clip_end, 7)
        out["camera"] = entry

    suns = [o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    if suns:
        sun = suns[0]
        out["sun"] = {
            "energy": round(sun.data.energy, 7),
            "color": _round(sun.data.color),
            "location": _round(sun.location),
            "rotation_euler": _round(sun.rotation_euler),
        }
        if len(suns) > 1:
            print(f"  note: {len(suns)} suns in the scene, using {sun.name!r}")

    world = scene.world
    if world is not None and world.use_nodes:
        background = world.node_tree.nodes.get("Background")
        if background is not None:
            out["world"] = {
                "color": _round(background.inputs[0].default_value[:3]),
                "strength": round(background.inputs[1].default_value, 7),
            }

    # Sphere radius of the POINT CLOUD specifically. Graph node objects also
    # carry a Mesh-to-Points node, so match the scatter object by name and fall
    # back to the densest mesh rather than taking whichever comes last.
    candidates = [o for o in scene.objects
                  if o.type == "MESH" and o.name.startswith("point_cloud")]
    if not candidates:
        candidates = sorted(
            (o for o in scene.objects if o.type == "MESH"),
            key=lambda o: len(o.data.vertices), reverse=True)[:1]
    for obj in candidates:
        for modifier in getattr(obj, "modifiers", []):
            if modifier.type != "NODES" or not modifier.node_group:
                continue
            for node in modifier.node_group.nodes:
                if node.type == "MESH_TO_POINTS" and node.inputs.get("Radius"):
                    out["_voxel"] = round(node.inputs["Radius"].default_value, 7)

    resolution = [scene.render.resolution_x, scene.render.resolution_y]
    return out, resolution


def set_scalar(text, section, key, value):
    """Update (or insert) a single `key:` inside a top-level section.

    Unlike replace_section this leaves the rest of the section — including its
    comments — completely alone.
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines)
                  if line.rstrip("\n") == f"{section}:"), None)
    if start is None:
        return text + f"\n{section}:\n  {key}: {value!r}\n"

    end = start + 1
    while end < len(lines):
        if lines[end].strip() and not lines[end][0].isspace():
            break
        end += 1

    for i in range(start + 1, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key}:") and not stripped.startswith("#"):
            indent = lines[i][:len(lines[i]) - len(stripped)]
            lines[i] = f"{indent}{key}: {value!r}\n"
            return "".join(lines)

    lines.insert(end, f"  {key}: {value!r}\n")
    return "".join(lines)


def render_block(name, values):
    lines = [f"{name}:"]
    for key, value in values.items():
        if isinstance(value, list):
            lines.append(f"  {key}: [{', '.join(repr(v) for v in value)}]")
        else:
            lines.append(f"  {key}: {value!r}")
    return "\n".join(lines) + "\n"


def replace_section(text, name, block):
    """Swap a whole top-level YAML section, or append it if absent."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines)
                  if line.rstrip("\n") == f"{name}:"), None)
    if start is None:
        separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        return text + separator + block

    end = start + 1
    while end < len(lines):
        stripped = lines[end]
        if stripped.strip() and not stripped[0].isspace():
            break
        end += 1
    # keep trailing blank lines that belonged after the section
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[:start]) + block + "".join(lines[end:])


def write_layers(text, layers, config):
    """Fold the GUI's per-layer grading back into `color:`.

    The layer whose attribute matches its own name is the base grading; the
    others become `color.variants` entries, which is exactly how they were
    defined going in.
    """
    base_name = config["data"]["default_color"]
    base = layers.get(base_name, {})
    for key, config_key in (("saturation", "saturation"), ("contrast", "contrast"),
                            ("brightness", "brightness"), ("exposure", "exposure"),
                            ("gamma", "gamma")):
        if key in base:
            text = set_scalar(text, "color", config_key, base[key])
    if "alpha" in base:
        text = set_scalar(text, "point_cloud", "alpha", base["alpha"])

    variants = []
    for name, entry in layers.items():
        if name == base_name:
            continue
        source = entry.get("attribute", "").replace("color_", "", 1)
        if source == name:
            continue          # a plain layer, not a variant
        variants.append((name, source, entry))
    if variants:
        block = ["color:", "  variants:"]
        for name, source, entry in variants:
            block.append(f"    - name: {name}")
            block.append(f"      from: {source}")
            for key in ("saturation", "contrast", "brightness", "exposure",
                        "gamma", "alpha"):
                if key in entry:
                    block.append(f"      {key}: {entry[key]!r}")
        print("\n".join(block))
        print("  ^ paste into the config if you want these variants pinned "
              "(automatic merge of list entries is not attempted)")
    return text


def write_graphs(text, graphs, config):
    """Report GUI-tuned graph appearance.

    `graphs.items` is a list keyed by name, so it is printed for you to paste
    rather than merged blindly — an automatic rewrite would risk reordering or
    dropping entries the GUI never saw.
    """
    print("graphs:")
    print("  items:")
    for name, entry in graphs.items():
        print(f"    - name: {name}")
        for key in ("color", "alpha", "emission", "radius", "node_radius", "height"):
            if key in entry:
                print(f"      {key}: {entry[key]!r}")
        if entry.get("hidden"):
            print("      # hidden in render")
    print("  ^ paste into the config to pin these")
    return text


def main(args):
    print(f"Reading {args.blend}")
    blender_bin = args.blender or shutil.which("blender")
    if blender_bin:
        scene, resolution = read_scene_external(args.blend, blender_bin)
    else:
        print("  no `blender` binary found, falling back to the in-process bpy "
              "(this fails if the file was saved by a newer Blender)")
        scene, resolution = read_scene(args.blend)
    if not scene:
        raise SystemExit("Found no camera, sun or world in that .blend")

    voxel = scene.pop("_voxel", None)
    alpha = scene.pop("_alpha", None)
    layers = scene.pop("_layers", None)
    graph_params = scene.pop("_graphs", None)
    wanted = args.only or SECTIONS
    scene = {k: v for k, v in scene.items() if k in wanted}
    pull_voxel = voxel is not None and (args.only is None or "data" in args.only)
    # `cloud_alpha` now carries the ACTIVE layer's opacity, so it is read via
    # the per-layer table rather than pulled into the global default
    pull_alpha = False

    # Don't pin an inherited value at figure level just because it round-tripped
    from figlib import load_config
    config = load_config(args.config)
    if pull_voxel:
        current = config["data"]["voxel"]
        if current is not None and abs(float(current) - voxel) < 1e-7:
            pull_voxel = False
            print(f"(data.voxel unchanged at {voxel!r}, leaving inheritance alone)")
    if pull_alpha:
        current = config["point_cloud"]["alpha"]
        if current is not None and abs(float(current) - alpha) < 1e-7:
            pull_alpha = False
            print(f"(point_cloud.alpha unchanged at {alpha!r}, "
                  f"leaving inheritance alone)")

    print()
    for name in SECTIONS:
        if name in scene:
            print(render_block(name, scene[name]), end="")
    if pull_voxel:
        print(f"data:\n  voxel: {voxel!r}")
    if pull_alpha:
        print(f"point_cloud:\n  alpha: {alpha!r}")
    print(f"\n(scene render resolution is {resolution[0]}x{resolution[1]}; "
          f"render.resolution is left untouched)")

    text = open(args.config).read()
    if layers:
        text = write_layers(text, layers, config)
    if graph_params:
        text = write_graphs(text, graph_params, config)

    if args.dry_run:
        print("\n--dry-run: config not modified")
        return

    for name in SECTIONS:
        if name in scene:
            text = replace_section(text, name, render_block(name, scene[name]))
    if pull_voxel:
        text = set_scalar(text, "data", "voxel", voxel)
    if pull_alpha:
        text = set_scalar(text, "point_cloud", "alpha", alpha)
    open(args.config, "w").write(text)
    print(f"\nUpdated {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blend", required=True, help="The .blend saved from the GUI")
    parser.add_argument("--config", required=True, help="Figure YAML to update")
    parser.add_argument("--only", nargs="*", choices=PULLABLE, default=None,
                        help="Only pull these (default: all). \"data\" means data.voxel.")
    parser.add_argument("--blender", default=None,
                        help="Blender binary to read the .blend with "
                             "(default: whatever `blender` is on PATH)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written, change nothing")
    main(parser.parse_args())
