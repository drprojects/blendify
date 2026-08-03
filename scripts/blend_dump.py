"""Dump camera / sun / world / sphere-radius from a .blend as JSON.

Runs *inside* Blender, so it reads files saved by whatever Blender version you
use in the GUI — the `bpy` module pinned in the conda env is usually older and
cannot open them.

    blender --background scene.blend --python scripts/blend_dump.py -- --out p.json

`scripts/scene_to_config.py` calls this for you; you rarely run it by hand.
"""
import json
import math
import sys

import bpy


def _round(values, ndigits=7):
    return [round(float(v), ndigits) for v in values]


def dump():
    scene = bpy.context.scene
    out = {}

    camera = scene.camera
    if camera is None:
        cameras = [o for o in scene.objects if o.type == "CAMERA"]
        camera = cameras[0] if cameras else None
    if camera is not None:
        quaternion = (camera.rotation_quaternion
                      if camera.rotation_mode == "QUATERNION"
                      else camera.matrix_world.to_quaternion())
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

    # Overall cloud opacity: the named multiply node spliced into the scatter
    # material, so GUI scrubbing round-trips into point_cloud.alpha
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        node = material.node_tree.nodes.get("cloud_alpha")
        if node is not None:
            out["_alpha"] = round(node.inputs[1].default_value, 7)

    # Per-layer grading table, as tuned in the GUI
    for obj in scene.objects:
        raw = obj.get("figure_layers")
        if raw:
            try:
                out["_layers"] = json.loads(raw)
            except (TypeError, ValueError):
                pass

    # Network graph appearance: colour, radii, opacity, glow, height
    graphs = {}
    for obj in scene.objects:
        if obj.type == "CURVE" and obj.name.endswith("_edges"):
            name = obj.name[: -len("_edges")]
            entry = graphs.setdefault(name, {})
            entry["radius"] = round(obj.data.bevel_depth, 7)
            entry["height"] = round(obj.location.z, 7)
            entry["hidden"] = bool(obj.hide_render)
            if obj.data.materials and obj.data.materials[0].use_nodes:
                bsdf = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    entry["color"] = [round(v, 7)
                                      for v in bsdf.inputs["Base Color"].default_value[:3]]
                    entry["alpha"] = round(bsdf.inputs["Alpha"].default_value, 7)
                    if "Emission Strength" in bsdf.inputs:
                        entry["emission"] = round(
                            bsdf.inputs["Emission Strength"].default_value, 7)
        elif obj.type == "MESH" and obj.name.endswith("_nodes"):
            name = obj.name[: -len("_nodes")]
            entry = graphs.setdefault(name, {})
            for modifier in getattr(obj, "modifiers", []):
                if modifier.type == "NODES" and modifier.node_group:
                    for node in modifier.node_group.nodes:
                        if node.type == "MESH_TO_POINTS" and node.inputs.get("Radius"):
                            entry["node_radius"] = round(
                                node.inputs["Radius"].default_value, 7)
    if graphs:
        out["_graphs"] = graphs

    out["_resolution"] = [scene.render.resolution_x, scene.render.resolution_y]
    out["_blender_version"] = bpy.app.version_string
    return out


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else None
    payload = dump()
    if out_path:
        with open(out_path, "w") as f:
            json.dump(payload, f)
    else:
        print(json.dumps(payload, indent=2))
