"""Small support library for the Blender figure scripts.

`config`   — YAML figure configs with `extends` inheritance
`data`     — read point clouds from various formats into one internal object
`palettes` — turn raw label / scalar fields into RGB using a palette file
`grading`  — colour grading (saturation, contrast, exposure) for photo layers
"""
from .config import load_config, apply_overrides, require
from .data import PointCloud, load_point_cloud, SUPPORTED_SUFFIXES
from .grading import grade, is_neutral

# Name of the material node holding overall cloud opacity, so the GUI can
# scrub it and scripts/scene_to_config.py can read it back
ALPHA_NODE_NAME = "cloud_alpha"

__all__ = [
    "load_config", "apply_overrides", "require",
    "PointCloud", "load_point_cloud", "SUPPORTED_SUFFIXES",
    "grade", "is_neutral", "ALPHA_NODE_NAME",
]
