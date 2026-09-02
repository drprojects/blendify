"""Give each graph's nodes their own material inside an exported .blend.

Before this, edges and nodes shared one material datablock, so the GUI colour
picker moved both together and they could not be told apart without scripting.
This hands the nodes a copy — identical colour, so nothing renders differently
until you deliberately change it — and re-points the geometry-nodes Set Material
at it, which is what actually colours the instanced spheres.

Runs *inside* Blender:

    blender --background scene.blend --python scripts/blend_split_graph_materials.py \
        -- --out scene.blend

Idempotent: a graph whose nodes already have their own material is left alone,
so re-running never stacks up copies. Nothing touches the point cloud, so this
costs a file rewrite and nothing else.
"""
import sys

import bpy


def set_material_nodes(obj):
    """The geometry-nodes Set Material nodes driving this object's instances."""
    found = []
    for modifier in getattr(obj, "modifiers", []):
        if modifier.type == "NODES" and modifier.node_group:
            for node in modifier.node_group.nodes:
                if node.bl_idname == "GeometryNodeSetMaterial":
                    found.append(node)
    return found


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out_path = argv[argv.index("--out") + 1]

    scene = bpy.context.scene
    edge_materials = set()
    for obj in scene.objects:
        if obj.name.endswith("_edges") and obj.data.materials:
            edge_materials.add(obj.data.materials[0].name)

    split = already = 0
    for obj in list(scene.objects):
        if not (obj.type == "MESH" and obj.name.endswith("_nodes")):
            continue
        graph = obj.name[: -len("_nodes")]
        setters = set_material_nodes(obj)
        current = (setters[0].inputs["Material"].default_value if setters
                   else (obj.data.materials[0] if obj.data.materials else None))
        if current is None:
            print(f"  {graph}: no material, skipped")
            continue
        if current.name not in edge_materials:
            already += 1
            print(f"  {graph}: already separate ({current.name!r})")
            continue

        copy = current.copy()
        copy.name = f"{graph}_nodes"
        if obj.data.materials:
            obj.data.materials[0] = copy
        else:
            obj.data.materials.append(copy)
        for node in setters:
            node.inputs["Material"].default_value = copy
        split += 1
        print(f"  {graph}: {current.name!r} -> {copy.name!r} "
              f"({len(setters)} Set Material node(s) repointed)")

    if not split and not already:
        print("  no graph nodes in this .blend")

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False)
    print(f"  saved {out_path} (split {split}, already separate {already})")


if __name__ == "__main__":
    main()
