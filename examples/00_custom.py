import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import logging

import bpy
import blender_plots as bplt
import numpy as np
import os.path as osp
import torch
import trimesh
from videoio import VideoWriter
import matplotlib.colors as colors
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline


from blendify import scene
from blendify.colors import UniformColors, VertexColors
from blendify.materials import PrincipledBSDFMaterial
from blendify.utils.camera_trajectory import Trajectory
from blendify.utils.pointcloud import estimate_normals_from_pointcloud, approximate_colors_from_camera


def adjust_color(color, scale_red=1, scale_saturation=1):
    if scale_red != 1:
        color[..., 0] *= scale_red
    if scale_saturation != 1:
        color = colors.rgb_to_hsv(color)
        color[..., 1] *= scale_saturation
        color = colors.hsv_to_rgb(color)
    color = np.clip(color, 0., 1.)
    return color


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


# def spiral_camera_trajectory(
#         start_translation,
#         start_quaternion,
#         target,
#         fps=20,
#         duration=10,
#         angular_speed=0.1,
#         z_speed=0.05,
#         radius_growth=0.01):
#     """Generate a spiral trajectory around a target.
#
#     Args:
#         start_translation: (3,) array-like
#         start_quaternion: (4,) array-like [w, x, y, z]
#         target: (3,) array-like
#         fps: int
#         duration: float
#         angular_speed: float, radians per frame
#         z_speed: float
#         radius_growth: float
#
#     Returns:
#         dict: {frame_index: (translation (3,), quaternion (4,))}
#     """
#     start_loc = np.array(start_translation, dtype=np.float64)
#     start_quat = np.array(start_quaternion, dtype=np.float64)
#     start_quat /= np.linalg.norm(start_quat)
#     target = np.array(target, dtype=np.float64)
#
#     r0 = np.linalg.norm(start_loc[:2] - target[:2])
#     z0 = start_loc[2]
#
#     trajectory = {}
#
#     for t in time_steps(fps, duration):
#         if t == 0:
#             loc = start_loc
#             quat = start_quat
#         else:
#             theta = angular_speed * t
#             r = r0 + radius_growth * t
#             x = target[0] + r * np.cos(theta)
#             y = target[1] + r * np.sin(theta)
#             z = z0 + z_speed * t
#             loc = np.array([x, y, z], dtype=np.float64)
#             quat = look_at_quaternion(loc, target)
#
#         trajectory[t] = (loc, quat)
#
#     return trajectory


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


def spin_around_global_z(
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

    # # Attach blender file with scene (walls and floor)
    # logger.info("Attaching blend to the scene")
    # scene.attach_blend("./assets/light_box.blend")

    # Set the renderer
    bpy.context.scene.render.engine = args.engine
    if args.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = args.n_samples  # Temporal Anti-Aliasing
        bpy.context.scene.eevee.taa_samples = 8  # default 8, higher = smoother motion
        bpy.context.scene.eevee.use_ssr = True  # enable screen space reflections
        bpy.context.scene.eevee.use_ssr_refraction = True  # if you have transparent/refractive materials
        bpy.context.scene.eevee.use_soft_shadows = True  # softer shadows
        bpy.context.scene.eevee.shadow_cube_size = '1024'  # shadow resolution
        bpy.context.scene.eevee.shadow_cascade_size = '1024'
        bpy.context.scene.eevee.use_volumetric_lights = False  # optional, faster
        bpy.context.scene.eevee.use_motion_blur = False  # optional, prevents extra flicker
    elif args.engine == 'CYCLES':
        bpy.context.scene.cycles.max_bounces = 30
        bpy.context.scene.cycles.transmission_bounces = 20
        bpy.context.scene.cycles.transparent_max_bounces = 15
        bpy.context.scene.cycles.diffuse_bounces = 10
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = args.n_samples

    # Resolution
    bpy.context.scene.render.resolution_x = args.resolution[0]
    bpy.context.scene.render.resolution_y = args.resolution[1]

    # Color management
    bpy.context.scene.view_settings.view_transform = 'Standard'  # or 'Filmic'

    # Output format
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # Transparent background
    bpy.context.scene.render.film_transparent = True

    # # Interpolation of the camera trajectory
    # # Start, middle and end points of the camera trajectory
    # left_translation, left_rotation = \
    #     np.array([-0.5, -6.5, 2.4], dtype=np.float32), np.array([0.866, 0.5, 0.0, 0.0], dtype=np.float32)
    # middle_translation, middle_rotation = \
    #     np.array([4, -6.5, 2.4], dtype=np.float32), np.array([0.793, 0.505, 0.184, 0.288], dtype=np.float32)
    # right_translation, right_rotation = \
    #     np.array([6.5, -0.7, 2.4], dtype=np.float32), np.array([0.612, 0.354, 0.354, 0.612], dtype=np.float32)
    #
    # # Interpolate camera trajectory
    # logger.info("Creating camera and interpolating its trajectory")
    # camera_trajectory = Trajectory()
    # camera_trajectory.add_keypoint(quaternion=left_rotation, position=left_translation, time=0)
    # camera_trajectory.add_keypoint(quaternion=middle_rotation, position=middle_translation, time=2.5)
    # camera_trajectory.add_keypoint(quaternion=right_rotation, position=right_translation, time=5)
    # camera_trajectory = camera_trajectory.refine_trajectory(time_step=1/30, smoothness=5.0)

    # Add camera to the scene (position will be set in the rendering loop)
    camera = scene.set_perspective_camera(resolution=args.resolution, fov_x=np.deg2rad(73), near=0.1, far=1000)
    if args.mode == 'paper_ezsp_dales':
        translation =  np.array([43.99616622924805, 17.057422637939453, 29.741680145263672], dtype=np.float32)
        quaternion = np.array([0.49730759859085083, 0.24891497194766998, 0.3719913959503174, 0.7432017922401428], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_ezsp_kitti360':
        translation = np.array([-14.493873596191406, -15.842079162597656, 8.457014083862305], dtype=np.float32)
        quaternion = np.array([0.7844890356063843, 0.5345035195350647, -0.17706048488616943, -0.25987112522125244], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_ezsp_s3dis':
        translation = np.array([-0.1313309222459793, -7.50628137588501, 3.815814971923828], dtype=np.float32)
        quaternion = np.array([0.8719961047172546, 0.4894148111343384, -0.00020381153444759548, -0.009800762869417667], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_ezsp_s3dis_2':
        translation = np.array([-0.12195667624473572, -6.851565361022949, 5.126280784606934], dtype=np.float32)
        quaternion = np.array([0.9126590490341187, 0.40860432386398315, 0.0006827776087448001, -0.009779075160622597], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)

    # Set it as the active camera
    bpy.context.scene.camera = camera.blender_camera

    # Add lights to the scene
    logger.info("Setting up the Blender scene")
    # scene.lights.add_point(quaternion=(0.571, 0.169, 0.272, 0.756), translation=(21.0, 0.0, 7.0), strength=10000)
    # scene.lights.add_point(quaternion=(0.571, 0.169, 0.272, 0.756), translation=(0.0, -21, 7.0), strength=10000)

    # Configure the sun
    bpy.ops.object.light_add(type="SUN", radius=1, align="WORLD", location=[0, 0, 5], rotation=[1, 0, 2], scale=[1, 1, 1])
    bpy.context.object.data.energy = 1
    if args.mode == 'paper_ezsp_dales':
        bpy.context.object.data.energy = 5
        bpy.context.object.data.color = (1, 0.581614, 0.316125)  # sunset-ish color
        bpy.context.object.rotation_euler = np.array([-1.0471975803375244, 0.0, 0.1745329201221466],dtype=np.float32)
    elif args.mode == 'paper_ezsp_kitti360':
        bpy.context.object.data.energy = 3.5
        bpy.context.object.data.color = (1.0, 0.8358416557312012, 0.8358416557312012)
        bpy.context.object.rotation_euler = np.array([0.6981316804885864, 0.0, 0.7853981852531433], dtype=np.float32)
    elif args.mode == 'paper_ezsp_s3dis':
        bpy.context.object.data.energy = 2.2
        bpy.context.object.rotation_euler = np.array([0.1745329201221466, 0.0, -0.7853981852531433], dtype=np.float32)
    elif args.mode == 'paper_ezsp_s3dis_2':
        bpy.context.object.data.energy = 2.2
        bpy.context.object.rotation_euler = np.array([0.1745329201221466, 0.0, -0.7853981852531433], dtype=np.float32)

    # Configure world lighting
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (1, 0.934681, 0.78918, 1)  # color
    bg.inputs[1].default_value = 1.3  # strength
    if args.mode == 'paper_ezsp_dales':
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode == 'paper_ezsp_kitti360':
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode == 'paper_ezsp_s3dis':
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode == 'paper_ezsp_s3dis_2':
        bg.inputs[1].default_value = 0.7  # strength

    # # Camera colored PointCloud
    # # source of the mesh https://graphics.stanford.edu/data/3Dscanrep/
    # # load only vertices of the example mesh
    # mesh = trimesh.load("./assets/bunny.obj", process=False, validate=False)
    # vertices = mesh.vertices
    # # estimate normals
    # if args.backend == "orig":
    #     normals = np.array(mesh.vertex_normals)
    # else:
    #     normals = estimate_normals_from_pointcloud(vertices, backend=args.backend, device="cpu" if args.cpu else "cuda")

    # Read input data
    root = osp.dirname(args.path)
    filename = osp.splitext(osp.basename(args.path))[0]
    path_blender = osp.join(root, filename + '.blender')
    path_image = osp.join(root, filename)
    data = torch.load(args.path)
    print(f"Read data. Available attributes:")
    for k in sorted(list(data.keys())):
        print(f"{k} : sample: {data[k][0]}")
    pos = data['pos']
    # pos[:, :2] -= pos[:, :2].mean(dim=0).view(1, -1)
    pos[:, :2] -= (pos[:, :2].max(dim=0).values + pos[:, :2].min(dim=0).values).view(1, -1) / 2
    pos[:, 2] -= pos[:, 2].min()
    point_size = args.voxel

    # Prepare the RGB colors to numpy arrays of float32 in [0, 1]
    for colorname in [k for k in data.keys() if 'color' in k]:
        if data[colorname].dtype == np.dtype('<U18'):
            data[colorname] = np.array([
                [int(c) for c in rgb.replace('rgb(', '').replace(')', '').split(', ')]
                for rgb in data[colorname]])
        data[colorname] = np.asarray(data[colorname]).astype('float32') / 255.
        data[colorname] = adjust_color(data[colorname])

    # Create the Scatter object holding the point cloud
    default_colorname = f"{args.default_color}_colors"
    scatter = bplt.Scatter(
        pos.numpy(),
        color=data[default_colorname],
        marker_type="spheres",
        name=f"point_cloud_{default_colorname}",
        radius=point_size)
    scatter.color_material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = args.specularity
    scatter.color_material.node_tree.nodes["Principled BSDF"].inputs[9].default_value = args.roughness
    if args.mode == 'paper_ezsp_dales':
        scatter.base_object.rotation_euler = np.array([0.0, -0.0, -2.099583387374878], dtype=np.float32)
    elif args.mode == 'paper_ezsp_s3dis_2':
        scatter.base_object.rotation_euler = np.array([0.0, -0.0, -0.4942256808280945], dtype=np.float32)

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
        for colorname in [k for k in data.keys() if 'color' in k]:
            print(f"Rendering {colorname}...")
            bpy.context.scene.render.filepath = f"{path_image}_{colorname}.png"
            scatter.color = data[colorname]
            bpy.ops.render.render(write_still=True)
            # scatter.base_object.hide_render = True
            # bpy.data.objects.remove(scatter.base_object, do_unlink=True)
            logger.info(f"Renedering of {colorname} complete")

    if args.video:
        # # Interpolation of the camera trajectory
        # # Start, middle and end points of the camera trajectory
        # left_translation, left_rotation = \
        #     np.array([-0.5, -6.5, 2.4], dtype=np.float32), np.array([0.866, 0.5, 0.0, 0.0], dtype=np.float32)
        # middle_translation, middle_rotation = \
        #     np.array([4, -6.5, 2.4], dtype=np.float32), np.array([0.793, 0.505, 0.184, 0.288], dtype=np.float32)
        # right_translation, right_rotation = \
        #     np.array([6.5, -0.7, 2.4], dtype=np.float32), np.array([0.612, 0.354, 0.354, 0.612], dtype=np.float32)
        # # Interpolate camera trajectory
        # logger.info("Creating camera and interpolating its trajectory")
        # camera_trajectory = Trajectory()
        # camera_trajectory.add_keypoint(quaternion=left_rotation, position=left_translation, time=0)
        # camera_trajectory.add_keypoint(quaternion=middle_rotation, position=middle_translation, time=2.5)
        # camera_trajectory.add_keypoint(quaternion=right_rotation, position=right_translation, time=5)
        # camera_trajectory = camera_trajectory.refine_trajectory(time_step=1 / 30, smoothness=5.0)

        # Build the camera trajectory
        logger.info("Creating camera and interpolating its trajectory")
        if args.mode == 'paper_ezsp_dales':
            start_position = np.array([0, 0, 10], dtype=np.float32)
            start_target = np.array([1, 0, 10], dtype=np.float32)
            spiral_target = np.array([0, 0, 0], dtype=np.float32)
            spin_spiral_ratio = 0.3
            z_max = 200
            r_max = 200
            color_keys = ['intensity', '0_level', 'pred']
        elif args.mode == 'paper_ezsp_kitti360':
            start_position = None
            start_target = None
            spiral_target = None
            spin_spiral_ratio = None
            z_max = None
            r_max = None
            color_keys = ['rgb', '0_level', 'pred']
        elif args.mode in ['paper_ezsp_s3dis', 'paper_ezsp_s3dis_2']:
            start_position = np.array([0, 3, 1.7], dtype=np.float32)
            start_target = np.array([5, 0, 1.4], dtype=np.float32)
            spiral_target = start_position
            spin_spiral_ratio = 0.5
            z_max = 50
            r_max = 50
            color_keys = ['rgb', '0_level', 'pred']
        start_quaternion = look_at_quaternion(start_position, start_target)
        spin_duration = args.duration * spin_spiral_ratio
        spiral_duration = args.duration - spin_duration
        spin_poses = spin_around_global_z(
            start_position,
            start_quaternion,
            fps=args.fps,
            duration=spin_duration,
            angular_speed=np.pi / spin_duration)
        spiral_poses = spiral_camera_trajectory(
            spin_poses[list(spin_poses.keys())[-1]][0],
            spin_poses[list(spin_poses.keys())[-1]][1],
            spiral_target,
            fps=args.fps,
            duration=spiral_duration,
            angular_speed=3 * np.pi / spiral_duration,
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
            time_step=1 / args.fps,
            smoothness=5.0)

        # Create a color interpolator
        color_interpolator = ColorInterpolator(
            [data[f'{k}_colors'] for k in color_keys],
            [0, spin_duration / 2, 3 * spin_duration / 2],
            fade_duration=1)

        logger.info("Entering the main drawing loop")
        total_frames = len(camera_trajectory)
        video_path = (
            f"{path_image}_{colorname}"
            f"_engine-{args.engine}"
            f"_fps-{args.fps}"
            f"_resolution-{args.resolution[0]}-{args.resolution[1]}"
            f"_duration-{args.duration}"
            f"_specularity-{args.specularity}"
            f"_roughness-{args.roughness}"
            f"_n_samples-{args.n_samples}"
            f".mp4")
        with VideoWriter(
                video_path,
                resolution=args.resolution,
                fps=args.fps) as vw:
            for index, position in enumerate(camera_trajectory):
                logger.info(f"Rendering frame {index:03d} / {total_frames:03d}")

                # Set new camera position
                camera.set_position(
                    quaternion=position["quaternion"],
                    translation=position["position"])

                # # Approximate colors from normals and camera_view_direction
                # camera_viewdir = camera.get_camera_viewdir()
                # per_vertex_recolor = approximate_colors_from_camera(
                #     camera_viewdir, normals, per_vertex_color=pointcloud_colors_init.color,
                #     back_color=(0.0, 0.0, 0.0, 0.0)
                # )
                # # Create VertexColor instance and set it to the PointCloud
                # pointcloud_colors_new = VertexColors(per_vertex_recolor)
                # pointcloud.update_colors(pointcloud_colors_new)

                # Update the point colors at the current time step
                t = index / total_frames * args.duration
                color = color_interpolator.get_color(t)
                scatter.color = color

                # Render the scene to temporary image
                img = scene.render(
                    use_gpu=not args.cpu,
                    samples=args.n_samples)

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
            crf=28,
            preset="slow",
            codec="libx264")
        logger.info("Compressing complete")

    # # create material
    # poincloud_material = PrincipledBSDFMaterial(specular=0.25, roughness=0.2)
    # # create default color (will be changed in the rendering loop)
    # pointcloud_colors_init = UniformColors((51/255, 204/255, 204/255))
    # add pointcloud to the scene
    # pointcloud = scene.renderables.add_pointcloud(
    #     vertices=vertices, material=poincloud_material, colors=pointcloud_colors_init, point_size=0.03,
    #     particle_emission_strength=0.1, quaternion=(1, 0, 0, 0), translation=(0, 0, 0), base_primitive="SPHERE"
    # )

    # scatter = bplt.Scatter(
    #     np.random.rand(n, 3) * 50,
    #     color=np.random.rand(n, 3),
    #     marker_type="spheres",
    #     radius=1.5
    # )

    # Optionally save blend file with the scene at frame 0
    if args.export:
        scene.export(path_blender)

    # # Render the video frame by frame
    # logger.info("Entering the main drawing loop")
    # total_frames = len(camera_trajectory)
    # with VideoWriter(args.path, resolution=args.resolution, fps=args.fps) as vw:
    #     for index, position in enumerate(camera_trajectory):
    #         logger.info(f"Rendering frame {index:03d} / {total_frames:03d}")
    #         # Set new camera position
    #         camera.set_position(quaternion=position["quaternion"], translation=position["position"])
    #         # Approximate colors from normals and camera_view_direction
    #         camera_viewdir = camera.get_camera_viewdir()
    #         per_vertex_recolor = approximate_colors_from_camera(
    #             camera_viewdir, normals, per_vertex_color=pointcloud_colors_init.color, back_color=(0.0, 0.0, 0.0, 0.0)
    #         )
    #         # Create VertexColor instance and set it to the PointCloud
    #         pointcloud_colors_new = VertexColors(per_vertex_recolor)
    #         pointcloud.update_colors(pointcloud_colors_new)
    #         # Render the scene to temporary image
    #         img = scene.render(use_gpu=not args.cpu, samples=args.n_samples)
    #         # Read the resulting frame back
    #         # Frames have transparent background; perform an alpha blending with white background instead
    #         alpha = img[:, :, 3:4].astype(np.int32)
    #         img_white_bkg = ((img[:, :, :3] * alpha + 255 * (255 - alpha)) // 255).astype(np.uint8)
    #         # Add the frame to the video
    #         vw.write(img_white_bkg)
    # logger.info("Rendering complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Blendify example 04: Camera colored PointCloud.")

    # Paths to output files
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the input data")

    # Point cloud parameters
    parser.add_argument(
        "-v",
        "--voxel",
        type=float,
        help="Voxel size")
    parser.add_argument(
        "--specularity",
        default=0.1,
        type=float)
    parser.add_argument(
        "--roughness",
        default=0.2,
        type=float)

    # Rendering parameters
    parser.add_argument(
        "--image",
        action='store_true',
        help="Whether to render images")
    parser.add_argument(
        "--video",
        action='store_true',
        help="Whether to render video")
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"],
        help="Blender rendering engines. Supports 'CYCLES', 'BLENDER_EEVEE', "
             "'BLENDER_WORKBENCH'? (default: CYCLES)")
    parser.add_argument(
        "-n",
        "--n-samples",
        default=64,
        type=int,
        help="Number of paths to trace for each pixel in the render (default: 64)")
    parser.add_argument(
        "--duration",
        default=20,
        type=float,
        help="Duration of the video")
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="Number of frames per second in video renderings (default: 20)")
    parser.add_argument(
        "-res",
        "--resolution",
        default=(2100, 1400),
        nargs=2,
        type=int,
        help="Rendering resolution, (default: (2100, 1400))")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for rendering (by default GPU is used)")

    # Other parameters
    parser.add_argument(
        "--export",
        action='store_true',
        help="Whether to export the scene to a .blender file")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="orig",
        choices=["orig", "open3d", "pytorch3d"],
        help="Backend to use for normal estimation. Orig corresponds to "
             "original mesh normals, i.e. no estimation is performed (default: orig)")

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=None,
        help="Mode for printing a scene. Some predefined recipes are there.")
    parser.add_argument(
        "--default_color",
        type=str,
        default='0_level',
        help="Color that will be used for displaying in blender.")

    arguments = parser.parse_args()
    main(arguments)
