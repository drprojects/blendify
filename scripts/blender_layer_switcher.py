"""Sidebar panel for tuning a MALIBU3D figure scene inside Blender.

Run this INSIDE Blender, once per session:

    1. open the .blend
    2. click the `Scripting` tab at the top of the window
    3. the text `figure_panel.py` is already loaded — press `Run Script` (the
       play button). Otherwise: Open (folder icon) -> pick this file.
    4. move the mouse into the 3D viewport and press `N`
    5. use the `Figure` tab that appears on the right

Everything the panel changes is written back into the .blend, and
`scripts/scene_to_config.py` reads it out into the YAML config afterwards — so
nothing tuned here has to be retyped.
"""
import json

import bpy

MATERIAL_NAME = "color"
LAYERS_PROP = "figure_layers"
ACTIVE_PROP = "figure_active_layer"

SATURATION = "grade_saturation"
BRIGHT_CONTRAST = "grade_brightcontrast"
EXPOSURE = "grade_exposure"
GAMMA = "grade_gamma"
ALPHA = "cloud_alpha"


# --------------------------------------------------------------------------
# scene lookups — all defensive, a scene may have no cloud and/or no graphs
# --------------------------------------------------------------------------

def scatter_material():
    material = bpy.data.materials.get(MATERIAL_NAME)
    return material if material and material.use_nodes else None


def point_cloud():
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.name.startswith("point_cloud"):
            return obj
    return None


def attribute_node():
    material = scatter_material()
    if material is None:
        return None
    for node in material.node_tree.nodes:
        if node.type == "ATTRIBUTE":
            return node
    return None


def grade_node(name):
    material = scatter_material()
    return material.node_tree.nodes.get(name) if material else None


def layer_table():
    obj = point_cloud()
    if obj is None:
        return {}
    try:
        return json.loads(obj.get(LAYERS_PROP) or "{}")
    except (TypeError, ValueError):
        return {}


def save_layer_table(table):
    obj = point_cloud()
    if obj is not None:
        obj[LAYERS_PROP] = json.dumps(table)


def graph_objects():
    """Curve objects are graph edges; their sibling `_nodes` mesh holds nodes."""
    found = {}
    for obj in bpy.context.scene.objects:
        if obj.type == "CURVE" and obj.name.endswith("_edges"):
            found.setdefault(obj.name[: -len("_edges")], {})["edges"] = obj
        elif obj.type == "MESH" and obj.name.endswith("_nodes"):
            found.setdefault(obj.name[: -len("_nodes")], {})["nodes"] = obj
    return found


def node_radius_input(obj):
    for modifier in getattr(obj, "modifiers", []):
        if modifier.type == "NODES" and modifier.node_group:
            for node in modifier.node_group.nodes:
                if node.type == "MESH_TO_POINTS":
                    return node.inputs.get("Radius")
    return None


# --------------------------------------------------------------------------
# live grading — panel sliders write straight into the shader nodes
# --------------------------------------------------------------------------

def current_layer():
    """The layer being shown, tracked explicitly.

    It cannot be inferred from the shader: a variant shares its source layer's
    colour attribute, and matching on grading values breaks as soon as a slider
    moves. So the active name is stored on the object.
    """
    obj = point_cloud()
    if obj is None:
        return None
    name = obj.get(ACTIVE_PROP)
    table = layer_table()
    if name in table:
        return name
    # first load, or a .blend written before this property existed
    node = attribute_node()
    if node is not None:
        for candidate, entry in table.items():
            if entry.get("attribute") == node.attribute_name:
                return candidate
    return next(iter(table), None)


def set_current_layer(name):
    obj = point_cloud()
    if obj is not None:
        obj[ACTIVE_PROP] = name


def push_grade(entry):
    node = attribute_node()
    if node is None:
        return
    if entry.get("attribute"):
        node.attribute_name = entry["attribute"]
    for node_name, socket, key, default in (
            (SATURATION, "Fac", "saturation", 1.0),
            (BRIGHT_CONTRAST, "Bright", "brightness", 0.0),
            (BRIGHT_CONTRAST, "Contrast", "contrast", 0.0),
            (GAMMA, "Gamma", "gamma", 1.0)):
        target = grade_node(node_name)
        if target is not None:
            target.inputs[socket].default_value = float(entry.get(key, default))
    exposure = grade_node(EXPOSURE)
    if exposure is not None:
        gain = 2.0 ** float(entry.get("exposure", 0.0))
        exposure.inputs["Color2"].default_value = (gain, gain, gain, 1.0)
    alpha = grade_node(ALPHA)
    if alpha is not None:
        alpha.inputs[1].default_value = float(entry.get("alpha", 1.0))


def sync_table_from_nodes():
    """Persist whatever the sliders currently show onto the active layer."""
    name = current_layer()
    table = layer_table()
    if name is None or name not in table:
        return
    import math
    node = attribute_node()
    exposure = grade_node(EXPOSURE)
    gain = exposure.inputs["Color2"].default_value[0] if exposure else 1.0
    table[name].update({
        "attribute": node.attribute_name,
        "saturation": round(grade_node(SATURATION).inputs["Fac"].default_value, 6),
        "brightness": round(grade_node(BRIGHT_CONTRAST).inputs["Bright"].default_value, 6),
        "contrast": round(grade_node(BRIGHT_CONTRAST).inputs["Contrast"].default_value, 6),
        "exposure": round(math.log2(gain) if gain > 0 else 0.0, 6),
        "gamma": round(grade_node(GAMMA).inputs["Gamma"].default_value, 6),
        "alpha": round(grade_node(ALPHA).inputs[1].default_value, 6),
    })
    save_layer_table(table)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

class FIGURE_OT_set_layer(bpy.types.Operator):
    """Show this layer, with its own grading"""
    bl_idname = "figure.set_layer"
    bl_label = "Set layer"
    bl_options = {"REGISTER", "UNDO"}

    layer: bpy.props.StringProperty()

    def execute(self, context):
        sync_table_from_nodes()          # keep edits to the layer we are leaving
        table = layer_table()
        if self.layer not in table:
            self.report({"ERROR"}, f"Unknown layer {self.layer!r}")
            return {"CANCELLED"}
        push_grade(table[self.layer])
        set_current_layer(self.layer)
        self.report({"INFO"}, f"Layer: {self.layer}")
        return {"FINISHED"}


class FIGURE_OT_save_grade(bpy.types.Operator):
    """Store the current sliders onto this layer"""
    bl_idname = "figure.save_grade"
    bl_label = "Remember these settings"

    def execute(self, context):
        sync_table_from_nodes()
        self.report({"INFO"}, f"Saved grading for {current_layer()}")
        return {"FINISHED"}


class FIGURE_OT_greyscale(bpy.types.Operator):
    """Set saturation to 0 (greyscale) or back to 1"""
    bl_idname = "figure.greyscale"
    bl_label = "Toggle greyscale"

    def execute(self, context):
        node = grade_node(SATURATION)
        if node is None:
            return {"CANCELLED"}
        node.inputs["Fac"].default_value = (
            1.0 if node.inputs["Fac"].default_value < 0.01 else 0.0)
        sync_table_from_nodes()
        return {"FINISHED"}


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------

class FIGURE_PT_layers(bpy.types.Panel):
    bl_label = "Layer"
    bl_idname = "FIGURE_PT_layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        table = layer_table()
        if not table:
            layout.label(text="No layer table in this .blend", icon="ERROR")
            layout.label(text="Re-export with a current script version")
            return
        active = current_layer()
        layout.label(text=f"showing: {active}")
        column = layout.column(align=True)
        for name in table:
            row = column.row()
            row.operator("figure.set_layer", text=name,
                         depress=(name == active)).layer = name


class FIGURE_PT_grading(bpy.types.Panel):
    bl_label = "Colour & opacity"
    bl_idname = "FIGURE_PT_grading"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        if scatter_material() is None:
            layout.label(text="No point cloud material", icon="ERROR")
            return

        saturation = grade_node(SATURATION)
        bright_contrast = grade_node(BRIGHT_CONTRAST)
        exposure = grade_node(EXPOSURE)
        gamma = grade_node(GAMMA)
        alpha = grade_node(ALPHA)

        if saturation is None:
            layout.label(text="This .blend predates the grading nodes",
                         icon="ERROR")
            layout.label(text="Re-export to get live sliders")
            return

        column = layout.column(align=True)
        column.prop(saturation.inputs["Fac"], "default_value",
                    text="Saturation", slider=True)
        column.prop(bright_contrast.inputs["Bright"], "default_value",
                    text="Brightness")
        column.prop(bright_contrast.inputs["Contrast"], "default_value",
                    text="Contrast")
        column.prop(exposure.inputs["Color2"], "default_value",
                    text="Exposure gain")
        column.prop(gamma.inputs["Gamma"], "default_value", text="Gamma")
        layout.prop(alpha.inputs[1], "default_value", text="Opacity", slider=True)

        row = layout.row(align=True)
        row.operator("figure.greyscale", icon="IMAGE_ZDEPTH")
        row.operator("figure.save_grade", icon="CHECKMARK")


class FIGURE_PT_graphs(bpy.types.Panel):
    bl_label = "Network graphs"
    bl_idname = "FIGURE_PT_graphs"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        graphs = graph_objects()
        if not graphs:
            layout.label(text="No graphs in this scene", icon="INFO")
            layout.label(text="(this ROI may simply have none)")
            return

        for name, parts in graphs.items():
            box = layout.box()
            box.label(text=name, icon="OUTLINER_OB_CURVE")

            edges = parts.get("edges")
            nodes = parts.get("nodes")

            row = box.row(align=True)
            row.label(text="show")
            for obj in (edges, nodes):
                if obj is not None:
                    row.prop(obj, "hide_viewport", text="", emboss=False)
                    row.prop(obj, "hide_render", text="", emboss=False)

            if edges is not None:
                box.prop(edges.data, "bevel_depth", text="Edge radius")
                if edges.data.materials and edges.data.materials[0].use_nodes:
                    bsdf = edges.data.materials[0].node_tree.nodes.get(
                        "Principled BSDF")
                    if bsdf is not None:
                        box.prop(bsdf.inputs["Base Color"], "default_value",
                                 text="Colour")
                        box.prop(bsdf.inputs["Alpha"], "default_value",
                                 text="Opacity", slider=True)
                        if "Emission Strength" in bsdf.inputs:
                            box.prop(bsdf.inputs["Emission Strength"],
                                     "default_value", text="Glow")
            if nodes is not None:
                radius = node_radius_input(nodes)
                if radius is not None:
                    box.prop(radius, "default_value", text="Node radius")
            if edges is not None:
                box.prop(edges, "location", index=2, text="Height")


class FIGURE_PT_preview(bpy.types.Panel):
    bl_label = "Preview"
    bl_idname = "FIGURE_PT_preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        # Renders use a transparent film so figures drop onto a white page. In
        # the viewport that shows against Blender's dark theme, making anything
        # semi-transparent look far darker than it will in print.
        layout.prop(context.scene.render, "film_transparent",
                    text="Transparent (untick to judge on white)")
        layout.label(text="Viewport -> Rendered (Z) to judge", icon="INFO")


CLASSES = (
    FIGURE_OT_set_layer, FIGURE_OT_save_grade, FIGURE_OT_greyscale,
    FIGURE_PT_layers, FIGURE_PT_grading, FIGURE_PT_graphs, FIGURE_PT_preview,
)


def register():
    for cls in CLASSES:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
    print(f"Figure panel registered. Layers: {list(layer_table())}")
    print(f"Graphs: {list(graph_objects())}")
    print("Press N in the 3D viewport and open the 'Figure' tab.")
