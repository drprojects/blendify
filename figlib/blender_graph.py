"""Draw network graphs in a Blender scene as floating tubes.

The graphs are bird's-eye centrelines with no Z, so they are drawn on a single
horizontal plane above the point cloud. Because they will often pass straight
through buildings and trees, the default material is translucent and slightly
emissive: it stays readable where it crosses dense geometry instead of either
disappearing behind it or hiding it.

Edges become a bevelled curve (cheap, smooth, and thickness is one number);
nodes are optional spheres.
"""
import bpy
import numpy as np


def build_material(name, color, alpha=0.35, emission=0.2, roughness=0.35,
                   cast_shadow=False):
    """Material for a network layer.

    `cast_shadow` is off by default: a graph floating high above the cloud casts
    a shadow far from itself, which reads as a second spurious network. It is
    worth enabling only when the graph sits low enough for the shadow to land
    near it and act as a height cue.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Config graph colours are sRGB 0-1; Blender shader inputs are linear
    from .grading import srgb_to_linear
    rgba = (*srgb_to_linear(np.asarray(color, dtype=float)), 1.0)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = alpha
    if "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = rgba
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = emission

    # Needed for EEVEE; Cycles honours Alpha directly
    material.blend_method = "BLEND"
    material.shadow_method = "CLIP" if cast_shadow else "NONE"
    return material


def draw_graph(graph, material, radius=1.0, node_radius=0.0, name=None,
               height=0.0, cast_shadow=False, node_material=None):
    """Add one aligned graph to the scene. Returns the created objects.

    `graph` must already be in scene coordinates — see `graphs.align_to_cloud`,
    which leaves it on the z=0 plane. The floating height is applied as an
    object transform rather than baked into the vertices, so it can be dragged
    in the GUI and read back out again.

    `node_material` colours the node spheres separately from the edges. Passing
    None shares one material, which is what the two looked like before this
    existed. The caller normally hands in a *copy* even when the two colours
    agree: a shared datablock cannot be pulled apart in the GUI afterwards
    without scripting, so the copy is what makes the node colour adjustable.
    """
    name = name or graph.get("name", "graph")
    created = []

    edges = [e for e in graph["edges"] if len(e) >= 2]
    if edges:
        curve = bpy.data.curves.new(f"{name}_edges", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 2
        curve.use_fill_caps = True

        for polyline in edges:
            spline = curve.splines.new("POLY")
            spline.points.add(len(polyline) - 1)
            flat = np.column_stack(
                [polyline, np.ones(len(polyline))]).astype(np.float32).ravel()
            spline.points.foreach_set("co", flat)

        obj = bpy.data.objects.new(f"{name}_edges", curve)
        obj.data.materials.append(material)
        obj.location.z = height
        obj.visible_shadow = cast_shadow      # Cycles; shadow_method is EEVEE
        bpy.context.scene.collection.objects.link(obj)
        created.append(obj)

    nodes = graph.get("nodes")
    if node_radius > 0 and nodes is not None and len(nodes):
        mesh = bpy.data.meshes.new(f"{name}_nodes")
        mesh.from_pydata([tuple(p) for p in nodes], [], [])
        mesh.update()

        # The sphere instances are coloured by the geometry-nodes Set Material
        # below, not by this slot; the slot is set too so the outliner and the
        # GUI colour picker agree with what actually renders.
        node_material = node_material or material
        obj = bpy.data.objects.new(f"{name}_nodes", mesh)
        obj.data.materials.append(node_material)
        obj.visible_shadow = cast_shadow
        bpy.context.scene.collection.objects.link(obj)
        # Parent the nodes to the edges so the graph moves as one piece:
        # dragging the height in the GUI must not leave the nodes behind.
        if created:
            obj.parent = created[0]
            obj.matrix_parent_inverse.identity()

        modifier = obj.modifiers.new(f"{name}_nodes", type="NODES")
        group = bpy.data.node_groups.new(f"{name}_nodes", "GeometryNodeTree")
        modifier.node_group = group

        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")
        group_in = group.nodes.new("NodeGroupInput")
        group_out = group.nodes.new("NodeGroupOutput")
        to_points = group.nodes.new("GeometryNodeMeshToPoints")
        to_points.inputs["Radius"].default_value = node_radius
        set_material = group.nodes.new("GeometryNodeSetMaterial")
        set_material.inputs["Material"].default_value = node_material

        group.links.new(group_in.outputs[0], to_points.inputs["Mesh"])
        group.links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
        group.links.new(set_material.outputs["Geometry"], group_out.inputs[0])
        created.append(obj)

    return created
