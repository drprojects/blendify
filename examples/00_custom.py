import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import logging

import bpy
import blender_plots as bplt
import numpy as np
import os.path as osp
from videoio import VideoWriter
import subprocess
from scipy.interpolate import CubicSpline

from blendify import scene
from blendify.utils.camera_trajectory import Trajectory
from blendify.utils.bounding_box import draw_bboxes

# Allow running this script from anywhere without installing anything
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import (load_config, apply_overrides, require, load_point_cloud,
                    grade, is_neutral, ALPHA_NODE_NAME)
from figlib.blender_material import (build_grading_chain, apply_grade, read_grade,
                                     store_layers, DEFAULT_GRADE)
from figlib.graphs import read_gpkg_graph, align_to_cloud, find_meta_json
from figlib.blender_graph import build_material, draw_graph


class ColorInterpolator:
    def __init__(self, colors, times, fade_duration=1):
        """
        Args:
            colors: list of 2D numpy arrays (H, W, 3) or (H, W, 4).
            times: list of times (must be strictly increasing).
            fade_duration: duration of fade into each new color.
        """
        assert len(colors) == len(times), "colors and times must have same length"
        self.colors = colors
        self.times = np.array(times, dtype=float)
        self.fade_duration = float(fade_duration)

    def get_color(self, t):
        """
        Compute the color array at time t.
        Fades occur in intervals [t_i - fade_duration, t_i].
        """
        # Before first fade → return first color
        if t <= self.times[0] - self.fade_duration:
            return self.colors[0]

        # After last color time → return last color
        if t >= self.times[-1]:
            return self.colors[-1]

        # Find the active target index
        idx = np.searchsorted(self.times, t)
        if idx == 0:
            return self.colors[0]

        t_target = self.times[idx]        # time when color idx is fully active
        t_fade_start = t_target - self.fade_duration

        if t < t_fade_start:
            # Still in hold period of previous color
            return self.colors[idx - 1]

        # Blend from previous color → current target color
        alpha = (t - t_fade_start) / self.fade_duration
        alpha = np.clip(alpha, 0.0, 1.0)
        return (1 - alpha) * self.colors[idx - 1] + alpha * self.colors[idx]


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    Right-handed, unit quaternion.
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def look_at_quaternion(camera_location, target, up=(0, 0, 1)):
    """Compute quaternion (np.array [w, x, y, z]) for camera looking at
    target.
    """
    cam = np.array(camera_location, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = tgt - cam
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    R = np.stack([right, up, -forward], axis=1)  # 3x3
    return rotmat_to_quat(R)


def time_steps(fps, duration):
    """
    Generate an array of time steps starting at 0.

    Args:
        fps (float): frames per second
        duration (float): total duration in seconds

    Returns:
        np.ndarray of shape (n_frames,), where each element is the timestamp in seconds
    """
    n_frames = int(np.floor(fps * duration))
    return np.arange(n_frames) / fps


def spiral_camera_trajectory(
        start_translation,
        start_quaternion,
        target,
        fps=20,
        duration=10,
        angular_speed=0.5,
        z_speed=1.0,
        radius_growth=1.0,
        z_curve=None,
        radius_curve=None):
    """
    Spiral camera trajectory with optional variable Z and radius profiles (smooth).

    start_translation: (3,)
    start_quaternion: (4,) [w,x,y,z]
    target: (3,)
    fps: int
    duration: float
    angular_speed: float, radians/sec (ignored if radius_curve used)
    z_speed: float, units/sec (ignored if z_curve used)
    radius_growth: float, units/sec (ignored if radius_curve used)
    z_curve: np.ndarray (n_points,2) [[t0, z0], [t1, z1], ...] optional
    radius_curve: np.ndarray (n_points,2) [[t0, r0], [t1, r1], ...] optional

    Returns:
        dict: {frame_index: (translation, quaternion)}
    """
    start_loc = np.array(start_translation, dtype=np.float64)
    start_quat = np.array(start_quaternion, dtype=np.float64)
    start_quat /= np.linalg.norm(start_quat)
    target = np.array(target, dtype=np.float64)

    default_radius = np.linalg.norm(start_loc[:2] - target[:2])
    default_z = start_loc[2]

    # Setup smooth splines if provided
    if z_curve is not None:
        z_curve = np.array(z_curve)
        z_spline = CubicSpline(z_curve[:, 0], z_curve[:, 1])
    else:
        z_spline = None

    if radius_curve is not None:
        radius_curve = np.array(radius_curve)
        r_spline = CubicSpline(radius_curve[:, 0], radius_curve[:, 1])
    else:
        r_spline = None

    trajectory = {}

    for t in time_steps(fps, duration):
        if t == 0:
            loc = start_loc
            quat = start_quat
        else:
            # Compute radius
            if r_spline is not None:
                r = r_spline(t)
            else:
                r = default_radius + radius_growth * t

            # Compute Z
            if z_spline is not None:
                z = z_spline(t)
            else:
                z = default_z + z_speed * t

            # Angle
            theta = angular_speed * t
            x = target[0] + r * np.cos(theta)
            y = target[1] + r * np.sin(theta)
            loc = np.array([x, y, z], dtype=np.float64)

            quat = look_at_quaternion(loc, target)

        trajectory[t] = (loc, quat)

    return trajectory


def spin_around_trajectory(
        location,
        start_quaternion,
        fps=20,
        duration=10,
        angular_speed=0.5):
    """
    Camera spins around the GLOBAL Z axis at a fixed location.

    Args:
        location: (3,) fixed camera position
        start_quaternion: (4,) initial orientation [w, x, y, z]
        fps: frames per second
        duration: total duration in seconds
        angular_speed: radians/second (positive = CCW spin seen from above)

    Returns:
        dict {time: (translation, quaternion)}
    """
    location = np.array(location, dtype=np.float64)
    start_quat = np.array(start_quaternion, dtype=np.float64)
    start_quat /= np.linalg.norm(start_quat)

    # Convert starting quaternion to rotation matrix
    w, x, y, z = start_quat
    R0 = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])

    # Rotation around GLOBAL Z
    n_frames = int(np.floor(fps * duration))
    times = np.arange(n_frames) / fps
    trajectory = {}

    for t in times:
        angle = angular_speed * t
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        R = Rz @ R0  # apply spin in world coordinates
        quat = rotmat_to_quat(R)

        trajectory[t] = (location, quat)

    return trajectory


def look_rotation(forward: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
    """
    Build quaternion [w, x, y, z] from forward and up vectors.
    Similar to Blender's 'track to' constraint.
    """
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=1)

    # Rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def path_trajectory(times, positions, fps=20, up=np.array([0, 0, 1])):
    """
    Generate camera poses along a smooth path.

    Args:
        times: list of keyframe times (ascending)
        positions: list of 3D positions (same length as times)
        fps: frames per second
        up: world up vector for orientation

    Returns:
        dict {time: (position, quaternion)}
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    assert positions.shape[0] == len(times)

    # Build cubic splines per coordinate
    splines = [CubicSpline(times, positions[:, i]) for i in range(3)]

    total_duration = times[-1]
    n_frames = int(np.floor(total_duration * fps))
    frame_times = np.linspace(times[0], total_duration, n_frames)

    trajectory = {}
    for t in frame_times:
        pos = np.array([spline(t) for spline in splines])
        vel = np.array([spline(t, 1) for spline in splines])  # derivative
        if np.linalg.norm(vel) < 1e-6:  # avoid zero velocity
            vel = np.array([1, 0, 0])
        quat = look_rotation(vel, up)
        trajectory[t] = (pos, quat)

    return trajectory


def concat_pose_dicts(d1, d2):
    """
    Concatenate two pose dicts {time: (loc, quat)}, offsetting times of d2.
    """
    if not d1:
        return dict(d2)
    if not d2:
        return dict(d1)

    t_offset = max(d1.keys())
    return {**d1, **{t + t_offset: pose for t, pose in d2.items()}}


def compress_mp4(
        input_path,
        output_path,
        crf=28,
        preset="slow",
        codec="libx264"):
    """
    Compress an MP4 file using ffmpeg.
    Args:
        input_path (str): Path to input mp4
        output_path (str): Path to compressed mp4
        crf (int): Constant Rate Factor (lower=better quality, bigger file). Typical 18–28.
        preset (str): ffmpeg preset ("ultrafast","superfast","veryfast","faster",
                      "fast","medium","slow","slower","veryslow")
        codec (str): "libx264" (H.264) or "libx265" (H.265, better compression but less supported)
    """
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vcodec", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-acodec", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Custom Blendify script for point clouds visualization")

    # Everything about the figure lives in the YAML config; argparse only
    # carries run-time concerns (which config, image vs video, quick overrides)
    cfg = apply_overrides(load_config(args.config), args.set)
    c_data, c_render = cfg["data"], cfg["render"]
    c_cam, c_sun, c_world = cfg["camera"], cfg["sun"], cfg["world"]
    c_pc, c_bbox, c_video = cfg["point_cloud"], cfg["bbox"], cfg["video"]

    data_path = args.path or c_data["path"]
    if data_path is None:
        raise ValueError(
            "No input point cloud. Set `data.path` in the config or pass --path.")
    resolution = tuple(c_render["resolution"])

    # # Attach blender file with scene (walls and floor)
    # logger.info("Attaching blend to the scene")
    # scene.attach_blend("./assets/light_box.blend")

    # Set the renderer
    bpy.context.scene.render.engine = c_render["engine"]
    if c_render["engine"] == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = c_render["n_samples"]  # Temporal Anti-Aliasing
        bpy.context.scene.eevee.taa_samples = 8  # default 8, higher = smoother motion
        bpy.context.scene.eevee.use_ssr = True  # enable screen space reflections
        bpy.context.scene.eevee.use_ssr_refraction = True  # if you have transparent/refractive materials
        bpy.context.scene.eevee.use_soft_shadows = True  # softer shadows
        bpy.context.scene.eevee.shadow_cube_size = '1024'  # shadow resolution
        bpy.context.scene.eevee.shadow_cascade_size = '1024'
        bpy.context.scene.eevee.use_volumetric_lights = False  # optional, faster
        bpy.context.scene.eevee.use_motion_blur = False  # optional, prevents extra flicker
    elif c_render["engine"] == 'CYCLES':
        bpy.context.scene.cycles.max_bounces = 30
        bpy.context.scene.cycles.transmission_bounces = 20
        bpy.context.scene.cycles.transparent_max_bounces = 15
        bpy.context.scene.cycles.diffuse_bounces = 10
        bpy.context.scene.cycles.device = 'CPU' if c_render["cpu"] else 'GPU'
        bpy.context.scene.cycles.samples = c_render["n_samples"]

    # Resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # Color management
    bpy.context.scene.view_settings.view_transform = c_render["view_transform"]

    # Output format
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # Transparent background
    bpy.context.scene.render.film_transparent = c_render["transparent_film"]

    # Add camera to the scene (position will be set in the rendering loop)
    # Tip to position the cameras in the blender GUI:
    # - navigate the general viewport as you'd like
    # - select the camera object in the menu on the right in the GUI
    # - press Ctrl + Alt + Numpad 0 to align the camera with the viewport
    # - (optional) adjust the viewport or the camera's position
    # - in the scripting console, recover the camera position by typing:
    #     ```
    #     bpy.context.object.location
    #     bpy.context.object.rotation_quaternion
    #     ```
    #     or the one-liner:
    #     ```
    #     print([bpy.context.object.location.x, bpy.context.object.location.y, bpy.context.object.location.z]); print([bpy.context.object.rotation_quaternion.w, bpy.context.object.rotation_quaternion.x, # bpy.context.object.rotation_quaternion.y, bpy.context.object.rotation_quaternion.z])
    #     ```
    camera = scene.set_perspective_camera(
        resolution=resolution,
        fov_x=np.deg2rad(c_cam["fov_x_deg"]),
        near=c_cam["near"],
        far=c_cam["far"])
    camera.set_position(
        quaternion=np.asarray(c_cam["quaternion"], dtype=np.float32),
        translation=np.asarray(c_cam["translation"], dtype=np.float32))

    # Set it as the active camera
    bpy.context.scene.camera = camera.blender_camera

    # Add lights to the scene
    logger.info("Setting up the Blender scene")
    # scene.lights.add_point(quaternion=(0.571, 0.169, 0.272, 0.756), translation=(21.0, 0.0, 7.0), strength=10000)
    # scene.lights.add_point(quaternion=(0.571, 0.169, 0.272, 0.756), translation=(0.0, -21, 7.0), strength=10000)

    # Configure the sun
    bpy.ops.object.light_add(
        type="SUN",
        radius=1,
        align="WORLD",
        location=c_sun["location"],
        rotation=c_sun["rotation_euler"],
        scale=[1, 1, 1])
    bpy.context.object.data.energy = c_sun["energy"]
    bpy.context.object.data.color = tuple(c_sun["color"])
    bpy.context.object.rotation_euler = np.asarray(
        c_sun["rotation_euler"], dtype=np.float32)

    # Configure world lighting
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = tuple(c_world["color"]) + (1,)  # color
    bg.inputs[1].default_value = c_world["strength"]  # strength

    # Read input data. Any supported format lands in the same PointCloud object
    root = osp.dirname(data_path)
    filename = osp.basename(data_path).split(".")[0]
    path_blender = osp.join(root, filename + '.blend')
    path_image = osp.join(root, filename)

    cloud = load_point_cloud(
        data_path,
        palettes=c_data["palettes"],
        palette_overrides=c_data["palette_overrides"],
        colors=c_data["colors"],
        cache=c_data["cache"],
        cache_dir=c_data["cache_dir"],
        log=logger.info)
    cloud.drop_void(c_data["drop_void"], log=logger.info)
    if c_data["center"]:
        cloud.center()
    cloud.subsample(c_data["subsample"], seed=c_data["seed"])
    if c_data["add_xyz"]:
        cloud.add_xyz_colorization()
    logger.info(cloud.summary())
    point_size = c_data["voxel"]

    # Blender wants float RGB in [0, 1]
    colorizations = {name: value.astype(np.float32) / 255.
                     for name, value in cloud.colors.items()}

    # Grading no longer touches the colour arrays. Each layer carries a grading
    # entry which is pushed into the shader nodes when that layer is rendered,
    # so the same numbers drive a CLI render and a live GUI slider, and can be
    # read back out of a saved .blend. White balance is the exception: it stays
    # baked, being a rare per-channel fix rather than something to scrub.
    c_color = cfg["color"]
    base_grade = {
        "saturation": c_color["saturation"],
        "contrast": c_color["contrast"],
        "brightness": c_color.get("brightness", 0.0),
        "exposure": c_color["exposure"],
        "gamma": c_color["gamma"],
    }
    graded_layers = set(c_color["apply_to"] or [])

    layer_grades = {}
    for name in colorizations:
        entry = dict(DEFAULT_GRADE)
        entry["attribute"] = f"color_{name}"
        if name in graded_layers:
            entry.update(base_grade)
        layer_grades[name] = entry

    if c_color["white_balance"]:
        for name in graded_layers:
            if name in colorizations:
                colorizations[name] = grade(
                    colorizations[name], white_balance=c_color["white_balance"])
        logger.info(f"Baked white balance {c_color['white_balance']}")

    # A variant is the same source colours under a different grading preset, so
    # it needs no colour array of its own — just its own entry pointing at the
    # source layer's attribute.
    for variant in c_color["variants"] or []:
        source = variant.get("from", "rgb")
        if source not in colorizations:
            # Variants are usually declared dataset-wide, so a figure that
            # restricts `data.colors` may not have loaded the source layer.
            # That is not an error — just skip the variant.
            logger.info(
                f"Skipping colour variant {variant.get('name')!r}: source "
                f"{source!r} not loaded (data.colors restricts to "
                f"{sorted(colorizations)})")
            continue
        entry = dict(layer_grades[source])
        entry["attribute"] = f"color_{source}"
        for key in ("saturation", "contrast", "brightness", "exposure", "gamma", "alpha"):
            if variant.get(key) is not None:
                entry[key] = variant[key]
        layer_grades[variant["name"]] = entry
        logger.info(f"Colour variant {variant['name']!r} from {source!r}: "
                    f"saturation={entry['saturation']}, alpha={entry['alpha']}")

    # Per-layer opacity lives in the grading entry too, so switching layer in
    # the GUI switches opacity with it.
    for name, value in (c_pc["layer_alpha"] or {}).items():
        if name in layer_grades:
            layer_grades[name]["alpha"] = value
    for name in colorizations:
        if name not in (c_pc["layer_alpha"] or {}):
            layer_grades[name].setdefault("alpha", float(c_pc["alpha"]))

    # Unannotated points (Void / N-A) are shown so the geometry stays complete,
    # but they carry no label and no metrics, so they must not compete for
    # attention: recolour them neutral and fade them. This IS baked, because it
    # is per-point rather than per-layer. The scatter material wires the colour
    # attribute's alpha into the BSDF, so handing it RGBA is enough.
    c_void = cfg["void"]
    for name, rgb in colorizations.items():
        alpha = np.ones((len(rgb), 1), dtype=np.float32)
        mask = cloud.void.get(name)
        if mask is not None and mask.any():
            rgb = rgb.copy()
            if c_void["color"] is not None:
                rgb[mask] = np.asarray(c_void["color"], dtype=np.float32)
            alpha[mask] = float(c_void["alpha"])
            logger.info(
                f"{name}: {int(mask.sum())} void points muted "
                f"(colour={c_void['color']}, alpha={c_void['alpha']})")
        colorizations[name] = np.concatenate([rgb, alpha], axis=1)

    # Create the Scatter object holding the point cloud
    default_colorname = c_data["default_color"]
    if default_colorname not in colorizations:
        raise KeyError(
            f"data.default_color={default_colorname!r} is not in {data_path}. "
            f"Available: {sorted(colorizations)}")
    scatter = bplt.Scatter(
        cloud.pos,
        color=colorizations[default_colorname],
        marker_type=c_pc["marker_type"],
        name=f"point_cloud_{default_colorname}",
        radius=point_size)
    bsdf = scatter.color_material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[7].default_value = c_pc["specularity"]
    bsdf.inputs[9].default_value = c_pc["roughness"]

    # Grading lives in named shader nodes rather than in the colour arrays, so
    # it is live-tweakable in the GUI and readable back out of a saved .blend.
    # blender_plots builds a NEW material on every `scatter.color = ...`, which
    # would orphan the grading chain. Hold on to the first one and keep it as
    # the mesh's only material.
    cloud_material = scatter.color_material
    build_grading_chain(cloud_material)
    apply_grade(cloud_material, layer_grades[default_colorname])

    def show_layer(colors, grade_entry):
        """Assign a colour array and its grading, keeping our material."""
        scatter.color = colors
        mesh = scatter.base_object.data
        if list(mesh.materials) != [cloud_material]:
            mesh.materials.clear()
            mesh.materials.append(cloud_material)
        # `scatter.color` writes into `marker_color`, so the shader must read
        # that here rather than the per-layer attribute used in exports
        apply_grade(cloud_material, grade_entry, attribute_name="marker_color")

    scatter.base_object.rotation_euler = np.asarray(
        c_pc["rotation_euler"], dtype=np.float32)

    # Draw network graphs, aligned to the cloud and floating on one plane
    c_graphs = cfg["graphs"]
    if c_graphs["items"]:
        translation = c_graphs["coord_translation"]
        if translation == "auto":
            meta_path = find_meta_json(data_path)
            if meta_path is None:
                raise ValueError(
                    f"graphs.coord_translation is 'auto' but no <roi>_meta.json "
                    f"sits next to {data_path}. Set it explicitly in the config.")
            with open(meta_path) as f:
                translation = json.load(f)["coord_translation"]
            logger.info(f"Graph alignment: coord_translation from "
                        f"{osp.basename(meta_path)} = {translation}")

        for item in c_graphs["items"]:
            item = {"path": item} if isinstance(item, str) else dict(item)
            graph = read_gpkg_graph(item["path"])
            aligned = align_to_cloud(
                graph,
                coord_translation=translation,
                offset=cloud.offset,
                height=0.0)   # height is an object transform, applied below
            material = build_material(
                name=item.get("name", graph["name"]),
                color=item.get("color", c_graphs["color"]),
                alpha=item.get("alpha", c_graphs["alpha"]),
                emission=item.get("emission", c_graphs["emission"]),
                roughness=item.get("roughness", c_graphs["roughness"]))
            draw_graph(
                aligned,
                material,
                radius=item.get("radius", c_graphs["radius"]),
                node_radius=item.get("node_radius", c_graphs["node_radius"]),
                name=item.get("name", graph["name"]),
                height=item.get("height", c_graphs["height"]))
            logger.info(
                f"Drew graph {graph['name']} "
                f"({len(aligned['edges'])} edges, {len(aligned['nodes'])} nodes) "
                f"at z={item.get('height', c_graphs['height'])}")

    # Read and draw bounding boxes
    bbox_path = args.path_bbox or c_bbox["path"]
    if bbox_path is not None:
        draw_bboxes(
            bbox_path,
            face_alpha=c_bbox["face_alpha"],
            sphere_r=c_bbox["sphere_r"],
            edge_r=c_bbox["edge_r"],
            iou_threshold=c_bbox["iou_threshold"],
        )

    # # Make adjustments in case we use the Eevee engine, mostly to
    # # avoid light saturation
    # if args.engine == "BLENDER_EEVEE":
    #     bpy.context.scene.view_settings.view_transform = 'Filmic'
    #     bpy.context.scene.view_settings.look = 'Medium High Contrast'  # optional
    #     bpy.context.scene.render.film_transparent = False
    #     for light in bpy.data.lights:
    #         light.energy *= 0.1
    #     bg.inputs[1].default_value *= 0.1
    #     for mat in bpy.data.materials:
    #         if not mat.node_tree:
    #             continue
    #         for node in mat.node_tree.nodes:
    #             if node.type != 'BSDF_PRINCIPLED':
    #                 continue
    #             node.inputs['Roughness'].default_value = max(node.inputs['Roughness'].default_value, 0.6)
    #             node.inputs['Specular'].default_value = min(node.inputs['Specular'].default_value, 0.5)

    # Render image and save to disk
    if args.image:
        if bbox_path is not None:
            bbox_suffix = f"_bbox-{osp.splitext(bbox_path)[0].split('_')[-1]}"
        else:
            bbox_suffix = ''
        for layername, entry in layer_grades.items():
            source = entry["attribute"].replace("color_", "", 1)
            print(f"Rendering {layername}...")
            bpy.context.scene.render.filepath = f"{path_image}_{layername}{bbox_suffix}.png"
            show_layer(colorizations[source], entry)
            bpy.ops.render.render(write_still=True)
            logger.info(f"Rendering of {layername} complete")

    if args.video:
        # Build the camera trajectory
        logger.info("Creating camera and interpolating its trajectory")
        require(
            cfg,
            "video",
            ["start_position", "start_target", "spiral_target",
             "spin_spiral_ratio", "spin_angle", "spiral_angle", "z_max", "r_max"],
            "--video")
        start_position = np.asarray(c_video["start_position"], dtype=np.float32)
        start_target = np.asarray(c_video["start_target"], dtype=np.float32)
        spiral_target = np.asarray(c_video["spiral_target"], dtype=np.float32)
        spin_spiral_ratio = c_video["spin_spiral_ratio"]
        spin_angle = c_video["spin_angle"]
        spiral_angle = c_video["spiral_angle"]
        z_max = c_video["z_max"]
        r_max = c_video["r_max"]
        start_quaternion = look_at_quaternion(start_position, start_target)
        duration, fps = c_video["duration"], c_video["fps"]
        spin_duration = duration * spin_spiral_ratio
        spiral_duration = duration - spin_duration
        spin_poses = spin_around_trajectory(
            start_position,
            start_quaternion,
            fps=fps,
            duration=spin_duration,
            angular_speed=spin_angle / spin_duration)

        # A figure may replace the "spin" phase with a scripted flythrough;
        # keypoints are spread evenly over the spin duration
        if c_video["path_keypoints"] is not None:
            keypoints = c_video["path_keypoints"]
            spin_poses = path_trajectory(
                np.linspace(0, spin_duration, len(keypoints)),
                keypoints,
                fps=fps)

        spiral_poses = spiral_camera_trajectory(
            spin_poses[list(spin_poses.keys())[-1]][0],
            spin_poses[list(spin_poses.keys())[-1]][1],
            spiral_target,
            fps=fps,
            duration=spiral_duration,
            angular_speed=spiral_angle / spiral_duration,
            z_speed=z_max / spiral_duration,
            radius_growth=r_max / spiral_duration,
            z_curve=None,
            radius_curve=None)
        poses = concat_pose_dicts(spin_poses, spiral_poses)
        camera_trajectory = Trajectory()
        for time, (translation, quaternion) in poses.items():
            camera_trajectory.add_keypoint(
                quaternion=quaternion,
                position=translation,
                time=time)
        camera_trajectory = camera_trajectory.refine_trajectory(
            time_step=1 / fps,
            smoothness=c_video["smoothness"])

        # Create a color interpolator. `color_times` entries are
        # [color_key, factor], where the fade-in time is factor * spin_duration
        require(cfg, "video", ["color_times"], "--video")
        color_times = [
            [key, factor * spin_duration] for key, factor in c_video["color_times"]]
        color_interpolator = ColorInterpolator(
            [colorizations[ct[0]] for ct in color_times],
            [ct[1] for ct in color_times],
            fade_duration=1)

        logger.info("Entering the main drawing loop")
        total_frames = len(camera_trajectory)
        video_path = (
            f"{path_image}"
            f"_engine-{c_render['engine']}"
            f"_fps-{fps}"
            f"_resolution-{resolution[0]}-{resolution[1]}"
            f"_duration-{duration}"
            f"_specularity-{c_pc['specularity']}"
            f"_roughness-{c_pc['roughness']}"
            f"_n_samples-{c_render['n_samples']}"
            f".mp4")
        with VideoWriter(
                video_path,
                resolution=resolution,
                fps=fps) as vw:
            for index, position in enumerate(camera_trajectory):
                logger.info(f"Rendering frame {index:03d} / {total_frames:03d}")

                # Set new camera position
                camera.set_position(
                    quaternion=position["quaternion"],
                    translation=position["position"])

                # Update the point colors at the current time step
                t = index / total_frames * duration
                color = color_interpolator.get_color(t)
                show_layer(color, layer_grades[default_colorname])

                # Render the scene to temporary image
                img = scene.render(
                    use_gpu=not c_render["cpu"],
                    samples=c_render["n_samples"])

                # Read the resulting frame back
                # Frames have transparent background; perform an
                # alpha blending with white background instead
                alpha = img[:, :, 3:4].astype(np.int32)
                img_white_bkg = ((img[:, :, :3] * alpha + 255 * (255 - alpha)) // 255).astype(np.uint8)

                # Add the frame to the video
                vw.write(img_white_bkg)
        logger.info("Rendering complete")
        logger.info("Compressing video")
        compress_mp4(
            video_path,
            f"{osp.splitext(video_path)[0]}_compressed.mp4",
            crf=c_video["crf"],
            preset=c_video["preset"],
            codec=c_video["codec"])
        logger.info("Compressing complete")

    # Optionally save blend file with the scene at frame 0
    if args.export:
        # Store every colorization as an extra colour attribute on the SAME
        # mesh. This costs one float4 per point per layer and nothing else —
        # the positions and, crucially, the geometry-nodes sphere instancing
        # are shared. Separate objects per layer would duplicate both and make
        # the viewport crawl.
        if cfg["export"]["all_layers"]:
            mesh = scatter.base_object.data
            existing = {a.name for a in mesh.color_attributes}
            for name, color in colorizations.items():
                attr_name = f"color_{name}"
                if attr_name in existing:
                    continue
                attribute = mesh.color_attributes.new(
                    name=attr_name, type='FLOAT_COLOR', domain='POINT')
                rgba = color if color.shape[1] == 4 else np.concatenate(
                    [color, np.ones((len(color), 1), dtype=np.float32)], axis=1)
                attribute.data.foreach_set("color", np.ascontiguousarray(rgba).ravel())
            store_layers(scatter.base_object, layer_grades)
            scatter.base_object["figure_active_layer"] = default_colorname
            apply_grade(scatter.color_material, layer_grades[default_colorname])
            logger.info(
                f"Stored {len(layer_grades)} layers "
                f"({', '.join(layer_grades)}) with their grading")
            logger.info(
                f"Stored {len(colorizations)} colour attributes on the mesh: "
                f"{', '.join('color_' + n for n in colorizations)}")
            logger.info(
                "To switch layer in the GUI: select the point cloud object, "
                "Material Properties -> `color` -> Attribute node -> set Name "
                "to one of the above (currently 'marker_color').")

        # Embed the layer-switcher panel so it is already loaded in the .blend's
        # Scripting tab — the user presses Run Script rather than hunting for a
        # file. Not auto-run: that would trip Blender's script security prompt.
        switcher = osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))),
            "scripts", "blender_layer_switcher.py")
        if osp.exists(switcher):
            text_name = "figure_panel.py"
            if text_name in bpy.data.texts:
                bpy.data.texts.remove(bpy.data.texts[text_name])
            text = bpy.data.texts.new(text_name)
            with open(switcher) as f:
                text.write(f.read())
            logger.info(
                "Embedded figure_panel.py — in the GUI: Scripting tab, "
                "Run Script, then press N in the viewport for the 'Figure' tab.")

        scene.export(path_blender)
        logger.info(f"Exported {path_blender}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Render a point cloud figure from a YAML figure config.")

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to the YAML figure config (see configs/figures/)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries, e.g. --set render.n_samples=8 "
             "render.resolution=[800,600]")

    # Run-time concerns, not properties of the figure itself
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        type=str,
        help="Input point cloud .pt file (overrides data.path in the config)")
    parser.add_argument(
        "--path_bbox",
        default=None,
        type=str,
        help="Input bbox .ply file (overrides bbox.path in the config)")
    parser.add_argument(
        "--image",
        action='store_true',
        help="Render one still image per *_colors key in the data")
    parser.add_argument(
        "--video",
        action='store_true',
        help="Render a video along the config's camera trajectory")
    parser.add_argument(
        "--export",
        action='store_true',
        help="Export the scene to a .blend file for inspection in the GUI")

    arguments = parser.parse_args()
    main(arguments)
