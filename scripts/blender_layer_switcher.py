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
PALETTES_PROP = "figure_palettes"
CONTINUOUS_PROP = "figure_continuous"
COLORMAPS_PROP = "figure_colormaps"
VALUE_ATTR="value_attribute"; RANGE="ramp_range"; RAMP_GAMMA="ramp_gamma"; RAMP="value_ramp"
ORIGINAL_PROP = "figure_palettes_original"

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


def graph_material(obj, part):
    """The material that actually colours this part of a graph.

    Node spheres are instanced by geometry nodes, so what renders is the
    material on the Set Material node, not necessarily the object's slot. Older
    files share one datablock between edges and nodes; editing the colour there
    moves both, which is why the export now hands nodes their own copy.
    """
    if obj is None:
        return None
    if part == "nodes":
        for modifier in getattr(obj, "modifiers", []):
            if modifier.type == "NODES" and modifier.node_group:
                for node in modifier.node_group.nodes:
                    if node.bl_idname == "GeometryNodeSetMaterial":
                        material = node.inputs["Material"].default_value
                        if material is not None:
                            return material
    materials = getattr(obj.data, "materials", None)
    return materials[0] if materials and len(materials) else None


def material_bsdf(material):
    if material is None or not material.use_nodes:
        return None
    return material.node_tree.nodes.get("Principled BSDF")


def draw_material_row(box, material, label, with_extras=True):
    bsdf = material_bsdf(material)
    if bsdf is None:
        return
    box.prop(bsdf.inputs["Base Color"], "default_value", text=label)
    if with_extras:
        box.prop(bsdf.inputs["Alpha"], "default_value", text="Opacity", slider=True)
        if "Emission Strength" in bsdf.inputs:
            box.prop(bsdf.inputs["Emission Strength"], "default_value", text="Glow")


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
    # Only trust the sliders if the shader is actually showing the layer we
    # think is active. If the two disagree, something set the active layer
    # without switching the shader, and persisting now would write another
    # layer's attribute and grading onto this one — silent, and only visible
    # later as a layer that displays the wrong colours.
    recorded = table[name].get("attribute")
    if node is not None and recorded and node.attribute_name != recorded:
        return
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
# class palettes — editable colours for the categorical layers
# --------------------------------------------------------------------------

def palette_tables(original=False):
    obj = point_cloud()
    if obj is None:
        return {}
    try:
        return json.loads(obj.get(ORIGINAL_PROP if original else PALETTES_PROP) or "{}")
    except (TypeError, ValueError):
        return {}


def save_palette_tables(tables):
    obj = point_cloud()
    if obj is not None:
        obj[PALETTES_PROP] = json.dumps(tables)


def recolour_layer(layer, lut_rgb):
    """color_<layer> = LUT[class_<layer>], preserving the alpha channel.

    Alpha carries the per-point void fading, so it must survive a palette edit.
    """
    import numpy as np
    obj = point_cloud()
    if obj is None:
        return False
    mesh = obj.data
    index_attr = mesh.attributes.get(f"class_{layer}")
    colour_attr = mesh.color_attributes.get(f"color_{layer}")
    if index_attr is None or colour_attr is None:
        return False

    count = len(mesh.vertices)
    labels = np.empty(count, dtype=np.int32)
    index_attr.data.foreach_get("value", labels)

    existing = np.empty(count * 4, dtype=np.float32)
    colour_attr.data.foreach_get("color", existing)

    lut = np.asarray(lut_rgb, dtype=np.float32)     # stored as 0-1 floats
    labels = np.clip(labels, 0, len(lut) - 1)

    out = np.empty((count, 4), dtype=np.float32)
    out[:, :3] = lut[labels]
    out[:, 3] = existing.reshape(count, 4)[:, 3]
    colour_attr.data.foreach_set("color", out.ravel())
    mesh.update()
    return True


def _void_update(self, context):
    """One control for the unlabelled points of every categorical layer.

    Only the layer on screen is recoloured here. Recolouring all seven cost
    2.3 s per event on an 8 M-point tile, and a colour picker fires ~30 events
    per second — which made the UI unusable. The others are brought up to date
    when they are next selected (see FIGURE_OT_set_layer) or on Apply, so the
    result is identical, just not paid for during the drag.
    """
    tables = palette_tables()
    if not tables:
        return
    rgb = [float(c) for c in context.scene.figure_void_color]
    for entry in tables.values():
        for index in entry.get("void", []):
            if index < len(entry["colors"]):
                entry["colors"][index] = rgb
    save_palette_tables(tables)

    if not context.scene.figure_live_palette:
        return
    active = current_layer()
    if active in tables:
        recolour_layer(active, tables[active]["colors"])


def _swatch_update(self, context):
    """Live-apply a swatch edit to the point cloud."""
    if not context.scene.figure_live_palette:
        return
    tables = palette_tables()
    entry = tables.get(self.layer)
    if entry is None:
        return
    entry["colors"][self.index] = [int(round(c * 255)) for c in self.color]
    save_palette_tables(tables)
    recolour_layer(self.layer, entry["colors"])


class FigureSwatch(bpy.types.PropertyGroup):
    layer: bpy.props.StringProperty()
    index: bpy.props.IntProperty()
    label: bpy.props.StringProperty()
    count: bpy.props.IntProperty()
    color: bpy.props.FloatVectorProperty(
        subtype="COLOR", min=0.0, max=1.0, size=3, update=_swatch_update)


def rebuild_swatches(context):
    """Refill the swatch list from the stored palettes."""
    swatches = context.scene.figure_swatches
    swatches.clear()
    for layer, entry in palette_tables().items():
        show_all = context.scene.figure_all_classes
        void_indices = set(entry.get("void", []))
        for index, (name, rgb) in enumerate(zip(entry["names"], entry["colors"])):
            count = entry["counts"][index] if index < len(entry["counts"]) else 0
            if not show_all and not count:
                continue
            if index in void_indices:
                continue        # driven by the single global void colour
            item = swatches.add()
            item.layer = layer
            item.index = index
            item.label = name
            item.count = count
            item.color = list(rgb)


# --------------------------------------------------------------------------
# continuous layers — a live ColorRamp in the shader
# --------------------------------------------------------------------------

def continuous_meta():
    obj = point_cloud()
    if obj is None:
        return {}
    try:
        return json.loads(obj.get(CONTINUOUS_PROP) or "{}")
    except (TypeError, ValueError):
        return {}


def colormaps():
    """{name: {"family", "source", "stops"}}, normalising the older flat form.

    v1 files stored `{name: stops}` with no metadata; they still load, they just
    land in one unnamed family.
    """
    obj = point_cloud()
    if obj is None:
        return {}
    try:
        raw = json.loads(obj.get(COLORMAPS_PROP) or "{}")
    except (TypeError, ValueError):
        return {}
    if isinstance(raw, dict) and raw.get("version") == 2:
        return raw.get("maps", {})
    return {name: {"family": "", "source": "", "stops": stops}
            for name, stops in raw.items()}


def srgb_to_linear(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def apply_colormap(name, reverse=False):
    node = grade_node(RAMP)
    entry = colormaps().get(name)
    if node is None or not entry:
        return False
    stops = entry["stops"]
    if reverse:
        stops = [[1.0 - stop[0]] + list(stop[1:4]) for stop in reversed(stops)]
    elements = node.color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])
    first = True
    for stop in stops:
        pos = float(stop[0])
        lin = srgb_to_linear(stop[1:4])
        el = elements[0] if first else elements.new(pos)
        el.position = pos
        el.color = (*lin, 1.0)
        first = False
    return True


FAMILIES = ("Sequential", "Diverging", "Cyclic", "Misc")

# Blender does not keep a reference to the strings an EnumProperty callback
# returns, so a list built fresh each call can be garbage-collected mid-draw and
# render as mojibake. Holding the last result here is the standard workaround.
_ENUM_CACHE = {}


def _family_items(self, context):
    items = [("ALL", "All", "Every colormap")]
    items += [(f, f, "") for f in FAMILIES]
    _ENUM_CACHE["family"] = items
    return items


def _colormap_items(self, context):
    family = getattr(context.scene, "figure_colormap_family", "ALL")
    entries = colormaps()
    names = sorted(entries)
    if family != "ALL":
        names = [n for n in names if entries[n].get("family") == family]
    items = [(n, n, entries[n].get("source", "")) for n in names]
    _ENUM_CACHE["colormap"] = items or [("none", "none", "")]
    return _ENUM_CACHE["colormap"]


def _colormap_update(self, context):
    apply_colormap(context.scene.figure_colormap,
                   context.scene.figure_colormap_reverse)


def _family_update(self, context):
    # The visible list just changed; the previous pick may not be in it.
    items = _colormap_items(self, context)
    if context.scene.figure_colormap not in {i[0] for i in items}:
        context.scene.figure_colormap = items[0][0]


def show_continuous(layer):
    """Point the shader at this layer's raw values and switch to the ramp."""
    from_meta = continuous_meta().get(layer)
    value_node = grade_node(VALUE_ATTR)
    if from_meta is None or value_node is None:
        return False
    value_node.attribute_name = f"value_{layer}"
    rng = grade_node(RANGE)
    if rng is not None:
        rng.inputs["From Min"].default_value = float(from_meta.get("vmin", 0.0))
        rng.inputs["From Max"].default_value = float(from_meta.get("vmax", 30.0))
    gam = grade_node(RAMP_GAMMA)
    if gam is not None:
        gam.inputs[1].default_value = float(from_meta.get("gamma", 1.0))
    return True


def wire_source(continuous):
    """Grading chain reads either the ramp or the baked colour attribute."""
    mat = scatter_material()
    if mat is None or RAMP not in mat.node_tree.nodes:
        return
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    src = (nodes[RAMP].outputs["Color"] if continuous else
           next(n for n in nodes if n.type == "ATTRIBUTE"
                and n.name != VALUE_ATTR).outputs["Color"])
    for target, socket in ((nodes["grade_luminance"], "Color"),
                           (nodes["grade_saturation"], "Color2")):
        for l in list(links):
            if l.to_node is target and l.to_socket.name == socket:
                links.remove(l)
        links.new(src, target.inputs[socket])


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
        is_cont = self.layer in continuous_meta()
        if is_cont:
            show_continuous(self.layer)
        wire_source(is_cont)
        # Palette edits are applied lazily to off-screen layers, so refresh this
        # one now. ~0.3 s once on selection, versus per colour-picker event.
        palettes = palette_tables()
        if self.layer in palettes:
            recolour_layer(self.layer, palettes[self.layer]["colors"])
        self.report({"INFO"}, f"Layer: {self.layer}")
        return {"FINISHED"}


class FIGURE_OT_toggle_pred(bpy.types.Operator):
    """Switch between the current layer and its ground-truth / prediction twin"""
    bl_idname = "figure.toggle_pred"
    bl_label = "Flip ground truth / prediction"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        table = layer_table()
        active = current_layer()
        if active is None:
            return {"CANCELLED"}
        twin = (active[len("pred_"):] if active.startswith("pred_")
                else f"pred_{active}")
        if twin not in table:
            self.report({"INFO"}, f"{active!r} has no counterpart")
            return {"CANCELLED"}
        # Reuse the layer operator so grading is saved and the ramp is rewired
        # exactly as a manual switch would do.
        return bpy.ops.figure.set_layer(layer=twin)


class FIGURE_OT_split_graph_material(bpy.types.Operator):
    """Give this graph's nodes their own material so their colour can differ"""
    bl_idname = "figure.split_graph_material"
    bl_label = "Give nodes their own colour"
    bl_options = {"REGISTER", "UNDO"}

    graph: bpy.props.StringProperty()

    def execute(self, context):
        parts = graph_objects().get(self.graph, {})
        nodes = parts.get("nodes")
        shared = graph_material(nodes, "nodes")
        if nodes is None or shared is None:
            self.report({"ERROR"}, f"No node material on {self.graph!r}")
            return {"CANCELLED"}

        copy = shared.copy()
        copy.name = f"{self.graph}_nodes"
        if nodes.data.materials:
            nodes.data.materials[0] = copy
        else:
            nodes.data.materials.append(copy)
        # The instanced spheres take their colour from Set Material, so this is
        # the assignment that actually changes the render.
        for modifier in getattr(nodes, "modifiers", []):
            if modifier.type == "NODES" and modifier.node_group:
                for node in modifier.node_group.nodes:
                    if node.bl_idname == "GeometryNodeSetMaterial":
                        node.inputs["Material"].default_value = copy
        self.report({"INFO"}, f"{self.graph}: nodes now use {copy.name!r}")
        return {"FINISHED"}


class FIGURE_OT_match_node_colour(bpy.types.Operator):
    """Copy the edge colour onto the nodes of this graph"""
    bl_idname = "figure.match_node_colour"
    bl_label = "Match edge colour"
    bl_options = {"REGISTER", "UNDO"}

    graph: bpy.props.StringProperty()

    def execute(self, context):
        parts = graph_objects().get(self.graph, {})
        edge_bsdf = material_bsdf(graph_material(parts.get("edges"), "edges"))
        node_bsdf = material_bsdf(graph_material(parts.get("nodes"), "nodes"))
        if edge_bsdf is None or node_bsdf is None:
            self.report({"ERROR"}, f"{self.graph!r} has no separate materials")
            return {"CANCELLED"}
        colour = tuple(edge_bsdf.inputs["Base Color"].default_value)
        node_bsdf.inputs["Base Color"].default_value = colour
        if "Emission" in node_bsdf.inputs:
            node_bsdf.inputs["Emission"].default_value = colour
        return {"FINISHED"}


class FIGURE_OT_save_grade(bpy.types.Operator):
    """Store the current sliders onto this layer"""
    bl_idname = "figure.save_grade"
    bl_label = "Remember these settings"

    def execute(self, context):
        sync_table_from_nodes()
        self.report({"INFO"}, f"Saved grading for {current_layer()}")
        return {"FINISHED"}


class FIGURE_OT_load_palettes(bpy.types.Operator):
    """Read the stored class palettes into the editor"""
    bl_idname = "figure.load_palettes"
    bl_label = "Reload palette list"

    def execute(self, context):
        rebuild_swatches(context)
        self.report({"INFO"}, f"{len(context.scene.figure_swatches)} classes")
        return {"FINISHED"}


class FIGURE_OT_apply_palettes(bpy.types.Operator):
    """Apply the current swatches to every categorical layer"""
    bl_idname = "figure.apply_palettes"
    bl_label = "Apply palettes"

    def execute(self, context):
        tables = palette_tables()
        for item in context.scene.figure_swatches:
            entry = tables.get(item.layer)
            if entry is not None:
                entry["colors"][item.index] = [float(c) for c in item.color]
        save_palette_tables(tables)
        for layer, entry in tables.items():
            recolour_layer(layer, entry["colors"])
        self.report({"INFO"}, f"applied {len(tables)} palettes")
        return {"FINISHED"}


class FIGURE_OT_reset_palettes(bpy.types.Operator):
    """Restore the palettes this .blend was exported with"""
    bl_idname = "figure.reset_palettes"
    bl_label = "Reset to exported"

    def execute(self, context):
        original = palette_tables(original=True)
        if not original:
            self.report({"ERROR"}, "no pristine palette stored")
            return {"CANCELLED"}
        save_palette_tables(original)
        for layer, entry in original.items():
            recolour_layer(layer, entry["colors"])
        rebuild_swatches(context)
        self.report({"INFO"}, "palettes reset")
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

        # A prediction tile carries a ground-truth layer and a `pred_` twin for
        # each task. Listing ~20 layers flat makes comparing a pair a hunt, so
        # twins share a row: ground truth left, prediction right. One click each
        # way is the whole point of this figure.
        paired, singles, seen = [], [], set()
        for name in table:
            if name.startswith("pred_"):
                continue
            twin = f"pred_{name}"
            if twin in table:
                paired.append((name, twin))
                seen.update((name, twin))
        for name in table:
            if name not in seen:
                singles.append(name)

        if paired:
            box = layout.box()
            header = box.row(align=True)
            header.label(text="ground truth")
            header.label(text="prediction")
            column = box.column(align=True)
            for gt, pred in paired:
                row = column.row(align=True)
                row.operator("figure.set_layer", text=gt,
                             depress=(gt == active)).layer = gt
                row.operator("figure.set_layer", text="pred",
                             depress=(pred == active)).layer = pred
            box.operator("figure.toggle_pred", icon="ARROW_LEFTRIGHT",
                         text="Flip ground truth / prediction")

        column = layout.column(align=True)
        for name in singles:
            column.operator("figure.set_layer", text=name,
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

            edge_material = graph_material(edges, "edges")
            node_material = graph_material(nodes, "nodes")

            if edges is not None:
                box.prop(edges.data, "bevel_depth", text="Edge radius")
                draw_material_row(box, edge_material, "Edge colour")
            if nodes is not None:
                radius = node_radius_input(nodes)
                if radius is not None:
                    box.prop(radius, "default_value", text="Node radius")
                if node_material is edge_material and node_material is not None:
                    # Pre-split file: one datablock, so a node colour here would
                    # silently drag the edges with it.
                    row = box.row()
                    row.enabled = False
                    row.label(text="Node colour: shared with edges", icon="LINKED")
                    box.operator("figure.split_graph_material",
                                 text="Give nodes their own colour",
                                 icon="UNLINKED").graph = name
                else:
                    draw_material_row(box, node_material, "Node colour",
                                      with_extras=False)
                    op = box.operator("figure.match_node_colour",
                                      text="Match edge colour", icon="COLOR")
                    op.graph = name
            if edges is not None:
                box.prop(edges, "location", index=2, text="Height")


class FIGURE_PT_palettes(bpy.types.Panel):
    bl_label = "Class palettes"
    bl_idname = "FIGURE_PT_palettes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        tables = palette_tables()
        if not tables:
            layout.label(text="No editable palettes here", icon="INFO")
            layout.label(text="(re-export to add them)")
            return

        box = layout.box()
        box.label(text="Unlabelled points (all layers)", icon="SHADING_SOLID")
        box.prop(context.scene, "figure_void_color", text="")
        if not context.scene.figure_live_palette:
            box.label(text="press Apply to see it", icon="INFO")

        row = layout.row(align=True)
        row.prop(context.scene, "figure_live_palette", text="Live")
        row.prop(context.scene, "figure_all_classes", text="All classes")
        row = layout.row(align=True)
        row.operator("figure.load_palettes", text="Reload", icon="FILE_REFRESH")
        row.operator("figure.apply_palettes", text="Apply", icon="CHECKMARK")
        row.operator("figure.reset_palettes", text="Reset", icon="LOOP_BACK")

        swatches = context.scene.figure_swatches
        if not swatches:
            layout.label(text="press Reload to list classes", icon="INFO")
            return

        current = None
        for item in swatches:
            if item.layer != current:
                current = item.layer
                box = layout.box()
                box.label(text=current, icon="COLOR")
            row = box.row(align=True)
            row.prop(item, "color", text="")
            suffix = "" if item.count else "  (absent)"
            row.label(text=f"{item.label}{suffix}")


class FIGURE_PT_ramp(bpy.types.Panel):
    bl_label = "Continuous ramp"
    bl_idname = "FIGURE_PT_ramp"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        layout = self.layout
        meta = continuous_meta()
        if not meta:
            layout.label(text="No continuous layer here", icon="INFO")
            return

        active = current_layer()
        if active not in meta:
            layout.label(text=f"select {', '.join(sorted(meta))} to edit",
                         icon="INFO")
            return

        unit = meta[active].get("unit", "")
        ramp = grade_node(RAMP)
        rng = grade_node(RANGE)
        gam = grade_node(RAMP_GAMMA)
        if ramp is None:
            layout.label(text="this .blend predates the ramp", icon="ERROR")
            return

        box = layout.box()
        entries = colormaps()
        box.label(text=f"Preset colormap ({len(entries)})", icon="COLOR")
        row = box.row(align=True)
        row.prop(context.scene, "figure_colormap_family", text="")
        row.prop(context.scene, "figure_colormap_reverse", text="", icon="ARROW_LEFTRIGHT")
        box.prop(context.scene, "figure_colormap", text="")
        picked = entries.get(context.scene.figure_colormap)
        if picked and picked.get("source"):
            box.label(text=f"{picked['source']} · {picked.get('family', '')}",
                      icon="INFO")

        box = layout.box()
        box.label(text=f"Range ({unit or 'value'})", icon="ARROW_LEFTRIGHT")
        if rng is not None:
            box.prop(rng.inputs["From Min"], "default_value", text="min")
            box.prop(rng.inputs["From Max"], "default_value", text="max")
            box.label(text="same range on every tile = comparable colours",
                      icon="INFO")
        if gam is not None:
            box.prop(gam.inputs[1], "default_value", text="curve (gamma)")
            box.label(text="<1 expands the low end", icon="INFO")

        box = layout.box()
        box.label(text="Ramp — drag stops, right-click to add", icon="IPO_EASE_IN_OUT")
        box.template_color_ramp(ramp, "color_ramp", expand=True)


class FIGURE_PT_lighting(bpy.types.Panel):
    bl_label = "Lighting & shadows"
    bl_idname = "FIGURE_PT_lighting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Figure"

    def draw(self, context):
        import math
        layout = self.layout
        scene = context.scene

        suns = [o for o in scene.objects
                if o.type == "LIGHT" and o.data.type == "SUN"]
        if suns:
            sun = suns[0]
            box = layout.box()
            box.label(text="Sun", icon="LIGHT_SUN")
            box.prop(sun.data, "energy", text="Strength")
            # Angular diameter drives shadow softness. Blender stores radians;
            # degrees is what anyone actually reasons about.
            row = box.row()
            row.prop(sun.data, "angle", text="Softness (angular size)")
            elevation = 90.0 - math.degrees(sun.rotation_euler[0]) % 360.0
            box.label(text=f"elevation above horizon: {elevation:.0f} deg")
            box.prop(sun, "rotation_euler", index=0, text="Tilt")
            box.prop(sun, "rotation_euler", index=2, text="Compass")
            box.prop(sun.data, "color", text="Colour")

        world = scene.world
        if world is not None and world.use_nodes:
            background = world.node_tree.nodes.get("Background")
            if background is not None:
                box = layout.box()
                box.label(text="Ambient (world)", icon="WORLD")
                box.prop(background.inputs[1], "default_value", text="Strength")
                box.prop(background.inputs[0], "default_value", text="Colour")
                box.label(text="lower = deeper shadows", icon="INFO")

        box = layout.box()
        box.label(text="Contact shadows", icon="SHADING_RENDERED")
        box.prop(scene.cycles, "use_fast_gi", text="Fast GI")
        if scene.cycles.use_fast_gi:
            box.prop(scene.cycles, "fast_gi_method", text="Method")
            box.prop(scene.cycles, "ao_bounces_render", text="Bounces")
            if scene.world is not None:
                box.prop(scene.world.light_settings, "distance", text="AO distance")

        camera = scene.camera
        if camera is not None and camera.data.type == "PERSP":
            box = layout.box()
            box.label(text="Camera", icon="CAMERA_DATA")
            box.prop(camera.data, "lens", text="Focal length")
            fov = math.degrees(2 * math.atan(
                0.5 * camera.data.sensor_width / camera.data.lens))
            box.label(text=f"horizontal FOV: {fov:.0f} deg")


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

        # The viewport clips independently of the camera; at 1 km+ tile widths
        # the default 1000 m hides the far side of the scene in the GUI only.
        space = context.space_data
        if getattr(space, "clip_end", None) is not None:
            box = layout.box()
            box.label(text="Viewport clipping", icon="VIEW_CAMERA")
            box.prop(space, "clip_start", text="Near")
            box.prop(space, "clip_end", text="Far")
            camera = context.scene.camera
            if camera and space.clip_end < camera.data.clip_end:
                box.label(text=f"below camera far ({camera.data.clip_end:.0f})",
                          icon="ERROR")

        layout.label(text="Viewport -> Rendered (Z) to judge", icon="INFO")


CLASSES = (
    FIGURE_OT_toggle_pred,
    FIGURE_OT_split_graph_material,
    FIGURE_OT_match_node_colour,
    FigureSwatch,
    FIGURE_OT_set_layer, FIGURE_OT_save_grade, FIGURE_OT_greyscale,
    FIGURE_OT_load_palettes, FIGURE_OT_apply_palettes, FIGURE_OT_reset_palettes,
    FIGURE_PT_layers, FIGURE_PT_grading, FIGURE_PT_graphs, FIGURE_PT_palettes,
    FIGURE_PT_ramp, FIGURE_PT_lighting, FIGURE_PT_preview,
)


def register():
    for cls in CLASSES:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.figure_swatches = bpy.props.CollectionProperty(type=FigureSwatch)
    obj = point_cloud()
    big = obj is not None and len(obj.data.vertices) > 2_000_000
    bpy.types.Scene.figure_live_palette = bpy.props.BoolProperty(
        name="Live", default=not big,
        description="Recolour as each swatch changes. Costs ~0.3 s per event at "
                    "8 M points, so it is off by default on large tiles — press "
                    "Apply instead")
    bpy.types.Scene.figure_all_classes = bpy.props.BoolProperty(
        name="All classes", default=False,
        description="Also list classes absent from this tile")
    # Seed from whatever the .blend was exported with, so the swatch shows the
    # colour actually in use rather than a hardcoded default
    seeded = (0.8, 0.8, 0.8)          # #CCCCCC, matches configs/malibu3d.yaml
    for entry in palette_tables().values():
        for index in entry.get("void", []):
            if index < len(entry["colors"]):
                seeded = tuple(entry["colors"][index])
                break
        break
    bpy.types.Scene.figure_colormap_family = bpy.props.EnumProperty(
        name="Family", items=_family_items, update=_family_update,
        description="Narrow the list. Sequential for a magnitude, diverging "
                    "only when the value has a meaningful zero")
    bpy.types.Scene.figure_colormap_reverse = bpy.props.BoolProperty(
        name="Reverse", default=False, update=_colormap_update,
        description="Flip the ramp end to end")
    bpy.types.Scene.figure_colormap = bpy.props.EnumProperty(
        name="Colormap", items=_colormap_items, update=_colormap_update,
        description="Load a colormap preset into the ramp")
    bpy.types.Scene.figure_void_color = bpy.props.FloatVectorProperty(
        name="Void colour", subtype="COLOR", min=0.0, max=1.0, size=3,
        default=seeded, update=_void_update,
        description="Colour of unlabelled points across every categorical layer")


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
    print(f"Figure panel registered. Layers: {list(layer_table())}")
    print(f"Graphs: {list(graph_objects())}")
    print("Press N in the 3D viewport and open the 'Figure' tab.")
