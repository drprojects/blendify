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
