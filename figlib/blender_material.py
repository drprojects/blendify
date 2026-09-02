"""The point cloud's shader chain, and per-layer grading stored in the .blend.

Grading used to be baked into the colour arrays, which made it invisible and
un-tweakable once a scene was exported. It now lives in named shader nodes, so
the same numbers drive a CLI render and a live GUI slider, and can be read back
out of a saved .blend.

The chain mirrors `figlib.grading`:

    Attribute ─┬──────────────┐
               └─ RGB to BW ──┴─ Mix(fac=saturation) ─ Bright/Contrast
                  ─ Multiply(2^exposure) ─ Gamma ─▶ Base Color
    Attribute ─ Alpha ─ Multiply(cloud_alpha) ─────▶ Alpha

Saturation is a mix between the luminance and the colour, which is exactly the
luma-preserving formula in `grading.grade` — factor 0 is greyscale, 1 is
untouched, >1 boosts.
"""
import json

import bpy

# Node names, shared by the exporter, the GUI panel and the reader
SATURATION = "grade_saturation"
BRIGHT_CONTRAST = "grade_brightcontrast"
EXPOSURE = "grade_exposure"
GAMMA = "grade_gamma"
ALPHA = "cloud_alpha"
LUMINANCE = "grade_luminance"

# Continuous layers are coloured by a live ColorRamp in the shader rather than
# by a baked colour array, so the ramp can be edited with zero recolour cost.
VALUE_ATTR = "value_attribute"
RANGE = "ramp_range"
RAMP_GAMMA = "ramp_gamma"
RAMP = "value_ramp"

# Custom property on the scatter object holding every layer's grading
LAYERS_PROP = "figure_layers"

DEFAULT_GRADE = {
    "attribute": "color_rgb",
    "saturation": 1.0,
    "contrast": 0.0,
    "brightness": 0.0,
    "exposure": 0.0,
    "gamma": 1.0,
    "alpha": 1.0,
}


def build_grading_chain(material):
    """Splice the grading nodes into blender_plots' scatter material.

    Idempotent: re-running on an already-built material just returns the nodes.
    """
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    bsdf = nodes["Principled BSDF"]
    attribute = next(n for n in nodes if n.type == "ATTRIBUTE")

    if SATURATION in nodes:
        return {n.name: n for n in nodes if n.name in
                (SATURATION, BRIGHT_CONTRAST, EXPOSURE, GAMMA, ALPHA)}

    x, y = bsdf.location.x, bsdf.location.y

    luminance = nodes.new("ShaderNodeRGBToBW")
    luminance.name = luminance.label = LUMINANCE
    luminance.location = (x - 1000, y + 200)

    saturation = nodes.new("ShaderNodeMixRGB")
    saturation.name = saturation.label = SATURATION
    saturation.blend_type = "MIX"
    saturation.use_clamp = False          # allow saturation > 1
    saturation.location = (x - 820, y + 100)

    bright_contrast = nodes.new("ShaderNodeBrightContrast")
    bright_contrast.name = bright_contrast.label = BRIGHT_CONTRAST
    bright_contrast.location = (x - 640, y + 100)

    exposure = nodes.new("ShaderNodeMixRGB")
    exposure.name = exposure.label = EXPOSURE
    exposure.blend_type = "MULTIPLY"
    exposure.use_clamp = False
    exposure.inputs["Fac"].default_value = 1.0
    exposure.location = (x - 460, y + 100)

    gamma = nodes.new("ShaderNodeGamma")
    gamma.name = gamma.label = GAMMA
    gamma.location = (x - 280, y + 100)

    alpha = nodes.get(ALPHA)
    if alpha is None:
        alpha = nodes.new("ShaderNodeMath")
        alpha.name = alpha.label = ALPHA
        alpha.operation = "MULTIPLY"
        alpha.inputs[1].default_value = 1.0
        alpha.location = (x - 280, y - 320)

    links.new(attribute.outputs["Color"], luminance.inputs["Color"])
    links.new(luminance.outputs["Val"], saturation.inputs["Color1"])
    links.new(attribute.outputs["Color"], saturation.inputs["Color2"])
    links.new(saturation.outputs["Color"], bright_contrast.inputs["Color"])
    links.new(bright_contrast.outputs["Color"], exposure.inputs["Color1"])
    links.new(exposure.outputs["Color"], gamma.inputs["Color"])
    links.new(gamma.outputs["Color"], bsdf.inputs["Base Color"])

    for link in list(links):
        if link.to_socket is bsdf.inputs["Alpha"]:
            links.remove(link)
    links.new(attribute.outputs["Alpha"], alpha.inputs[0])
    links.new(alpha.outputs["Value"], bsdf.inputs["Alpha"])

    return {SATURATION: saturation, BRIGHT_CONTRAST: bright_contrast,
            EXPOSURE: exposure, GAMMA: gamma, ALPHA: alpha}


def build_ramp_chain(material, stops=None, vmin=0.0, vmax=30.0, gamma=1.0):
    """Add the live ramp path used by continuous layers.

        Attribute(value_<layer>) -> Map Range -> power(gamma) -> ColorRamp

    Its output is NOT linked here; `select_source` decides whether the grading
    chain reads this or the baked colour attribute. Evaluated in the shader, so
    editing the ramp costs nothing per change.
    """
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    if RAMP in nodes:
        return nodes[RAMP]

    bsdf = nodes["Principled BSDF"]
    x, y = bsdf.location.x, bsdf.location.y

    value = nodes.new("ShaderNodeAttribute")
    value.name = value.label = VALUE_ATTR
    value.attribute_name = ""
    value.location = (x - 1400, y + 520)

    mapped = nodes.new("ShaderNodeMapRange")
    mapped.name = mapped.label = RANGE
    mapped.clamp = True
    mapped.inputs["From Min"].default_value = float(vmin)
    mapped.inputs["From Max"].default_value = float(vmax)
    mapped.inputs["To Min"].default_value = 0.0
    mapped.inputs["To Max"].default_value = 1.0
    mapped.location = (x - 1200, y + 520)

    shaped = nodes.new("ShaderNodeMath")
    shaped.name = shaped.label = RAMP_GAMMA
    shaped.operation = "POWER"
    shaped.inputs[1].default_value = float(gamma)
    shaped.location = (x - 1010, y + 520)

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = ramp.label = RAMP
    ramp.location = (x - 830, y + 560)

    links.new(value.outputs["Fac"], mapped.inputs["Value"])
    links.new(mapped.outputs["Result"], shaped.inputs[0])
    links.new(shaped.outputs["Value"], ramp.inputs["Fac"])

    if stops:
        set_ramp_stops(ramp, stops)
    return ramp


MAX_RAMP_ELEMENTS = 32          # Blender's hard limit on a ColorRamp


def set_ramp_stops(ramp, stops):
    """Replace a ColorRamp's elements. `stops` is [[pos, r, g, b], ...] in sRGB.

    A palette may carry more stops than a ColorRamp can hold — the baked colours
    are computed in numpy and have no such limit, and a denser table renders a
    smoother ramp. Overflowing stops are resampled onto the limit rather than
    raising: this widget is a GUI preview of colours that are already baked, so
    losing a little ramp resolution is right and failing the whole render is not.
    """
    from .grading import srgb_to_linear
    import numpy as _np

    stops = _np.asarray(stops, dtype=float)
    if len(stops) > MAX_RAMP_ELEMENTS:
        grid = _np.linspace(float(stops[0, 0]), float(stops[-1, 0]),
                            MAX_RAMP_ELEMENTS)
        stops = _np.column_stack([grid] + [
            _np.interp(grid, stops[:, 0], stops[:, c + 1]) for c in range(3)])

    elements = ramp.color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])
    first = True
    for stop in stops:
        position = float(stop[0])
        linear = srgb_to_linear(_np.asarray(stop[1:4], dtype=float))
        element = elements[0] if first else elements.new(position)
        element.position = position
        element.color = (*linear, 1.0)
        first = False


def read_ramp_stops(ramp):
    """Read a ColorRamp back as [[pos, r, g, b], ...] in sRGB."""
    import numpy as _np
    out = []
    for element in ramp.color_ramp.elements:
        srgb = _np.clip(_np.asarray(element.color[:3], dtype=float), 0, None)
        srgb = _np.where(srgb <= 0.0031308, srgb * 12.92,
                         1.055 * srgb ** (1 / 2.4) - 0.055)
        out.append([round(float(element.position), 5)]
                   + [round(float(v), 5) for v in srgb])
    return out


def select_source(material, continuous):
    """Point the grading chain at the ramp (continuous) or the colour attribute."""
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    if RAMP not in nodes:
        return
    source = nodes[RAMP].outputs["Color"] if continuous else next(
        n for n in nodes if n.type == "ATTRIBUTE" and n.name != VALUE_ATTR
    ).outputs["Color"]

    for target, socket in ((nodes[LUMINANCE], "Color"),
                           (nodes[SATURATION], "Color2")):
        for link in list(links):
            if link.to_node is target and link.to_socket.name == socket:
                links.remove(link)
        links.new(source, target.inputs[socket])


def apply_grade(material, grade, attribute_name=None):
    """Push one layer's grading values into the shader nodes.

    `attribute_name` overrides which colour attribute the shader reads. The
    render path needs this: `scatter.color = ...` writes into `marker_color`,
    whereas the exported .blend carries one `color_<layer>` attribute per layer.
    Pointing the shader at an attribute that does not exist renders black.
    """
    nodes = material.node_tree.nodes
    attribute = next(n for n in nodes if n.type == "ATTRIBUTE")
    name = attribute_name or grade.get("attribute")
    if name:
        attribute.attribute_name = name

    nodes[SATURATION].inputs["Fac"].default_value = float(
        grade.get("saturation", 1.0))
    nodes[BRIGHT_CONTRAST].inputs["Bright"].default_value = float(
        grade.get("brightness", 0.0))
    nodes[BRIGHT_CONTRAST].inputs["Contrast"].default_value = float(
        grade.get("contrast", 0.0))
    gain = 2.0 ** float(grade.get("exposure", 0.0))
    nodes[EXPOSURE].inputs["Color2"].default_value = (gain, gain, gain, 1.0)
    nodes[GAMMA].inputs["Gamma"].default_value = float(grade.get("gamma", 1.0))
    nodes[ALPHA].inputs[1].default_value = float(grade.get("alpha", 1.0))


def read_grade(material):
    """Read the current shader values back out as a grading dict."""
    import math
    nodes = material.node_tree.nodes
    attribute = next(n for n in nodes if n.type == "ATTRIBUTE")
    gain = nodes[EXPOSURE].inputs["Color2"].default_value[0]
    return {
        "attribute": attribute.attribute_name,
        "saturation": round(nodes[SATURATION].inputs["Fac"].default_value, 6),
        "brightness": round(nodes[BRIGHT_CONTRAST].inputs["Bright"].default_value, 6),
        "contrast": round(nodes[BRIGHT_CONTRAST].inputs["Contrast"].default_value, 6),
        "exposure": round(math.log2(gain) if gain > 0 else 0.0, 6),
        "gamma": round(nodes[GAMMA].inputs["Gamma"].default_value, 6),
        "alpha": round(nodes[ALPHA].inputs[1].default_value, 6),
    }


def store_layers(obj, layers):
    """Persist the per-layer grading table on the object as JSON."""
    obj[LAYERS_PROP] = json.dumps(layers)


def load_layers(obj):
    raw = obj.get(LAYERS_PROP)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}
