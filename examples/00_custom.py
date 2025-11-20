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
from videoio import VideoWriter
import matplotlib.colors as colors
import subprocess
from scipy.interpolate import CubicSpline

from blendify import scene
from blendify.utils.camera_trajectory import Trajectory
from blendify.utils.bounding_box import draw_bboxes


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
    elif args.mode == 'paper_litept_multiscale_resolutions':
        translation = np.array([-0.730396568775177, -0.17812800407409668, 2.0655770301818848], dtype=np.float32)
        quaternion = np.array([0.830621063709259, 0.4501716196537018, -0.10737811028957367, -0.3096523582935333], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0030_00':
        translation = np.array([0.5710775256156921, 3.436332941055298, 6.617712497711182], dtype=np.float32)
        quaternion = np.array([0.09742767363786697, 0.026836074888706207, 0.2641966640949249, 0.9591599702835083], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0169_00':
        translation = np.array([-1.2411928176879883, -2.975771188735962, 6.083170413970947], dtype=np.float32)
        quaternion = np.array([0.9348678588867188, 0.274120569229126, -0.05881626531481743, -0.21776331961154938], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0378_02':
        translation = np.array([0.6095614433288574, 0.890741229057312, 5.776294231414795], dtype=np.float32)
        quaternion = np.array([0.214557945728302, 0.022599322721362114, 0.11913300305604935, 0.9691550135612488], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0406_02':
        translation = np.array([-0.9438729286193848, 0.38966935873031616, 3.5319409370422363], dtype=np.float32)
        quaternion = np.array([0.10252528637647629, 0.02483103796839714, 0.19308218359947205, 0.975495457649231], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0645_01':
        translation = np.array([0.8784829378128052, 0.3396054804325104, 8.742258071899414], dtype=np.float32)
        quaternion = np.array([0.21838167309761047, 0.0794510766863823, 0.057358015328645706, 0.9709311127662659], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_semseg_scene0651_00':
        translation = np.array([-0.6995250582695007, -0.2804234027862549, 4.767136096954346], dtype=np.float32)
        quaternion = np.array([0.9844140410423279, 0.09081237763166428, -0.027085987851023674, -0.1481495052576065], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03022_room_8765':
        translation = np.array([-0.3661176860332489, -0.011110145598649979, 5.475278854370117], dtype=np.float32)
        quaternion = np.array([0.702695906162262, 0.034513816237449646, -0.028601357713341713, -0.7100768685340881], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03034_room_401':
        translation = np.array([1.1356086730957031, -1.1455254554748535, 8.321700096130371], dtype=np.float32)
        quaternion = np.array([0.9960197806358337, 0.0889928862452507, 0.004876633174717426, -0.0010398455196991563], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03113_room_560':
        translation = np.array([-0.536615252494812, 1.0438051223754883, 5.901821613311768], dtype=np.float32)
        quaternion = np.array([0.001395018887706101, -0.0036766876000910997, 0.11427392065525055, 0.9934415221214294], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03195_room_1764':
        translation = np.array([0.7700258493423462, 0.011294008232653141, 4.823118209838867], dtype=np.float32)
        quaternion = np.array([0.7043673992156982, 0.07746528834104538, 0.08040645718574524, 0.7009996771812439], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03223_room_4894':
        translation = np.array([-0.10034395009279251, -0.057335224002599716, 4.6524505615234375], dtype=np.float32)
        quaternion = np.array([0.9999446272850037, 0.009804977104067802, 0.0036866941954940557, -0.0010302967857569456], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_stru3d_semseg_scene_03237_room_2846':
        translation = np.array([0.3447108864784241, 0.03353601321578026, 4.648366451263428], dtype=np.float32)
        quaternion = np.array([0.7067402601242065, 0.038907717913389206, 0.04402199387550354, 0.7050294876098633], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_049d115cb992491b8de81f45e9ecc803':
        translation = np.array([-1.6669714450836182, 26.397863388061523, 17.029747009277344], dtype=np.float32)
        quaternion = np.array([0.041050706058740616, 0.026643119752407074, -0.5437620282173157, -0.8378113508224487], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_1ccdbec944bd4994b91aa3d0af8d285c':
        translation = np.array([13.539493560791016, -31.683805465698242, 18.330381393432617], dtype=np.float32)
        quaternion = np.array([0.8217780590057373, 0.555506706237793, 0.07675204426050186, 0.10100644081830978], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_2f678cb1e67d42ae9a04401f9cc1e6be':
        translation = np.array([29.993968963623047, -11.73816204071045, 13.352730751037598], dtype=np.float32)
        quaternion = np.array([0.6480386257171631, 0.4415041506290436, 0.3566734492778778, 0.5078426003456116], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_5f8393250fae4960b501cb6055614547':
        translation = np.array([9.246944427490234, 48.104496002197266, 20.54415512084961], dtype=np.float32)
        quaternion = np.array([0.12811481952667236, 0.06082574278116226, 0.465190052986145, 0.873776376247406], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_6bfd64d0778842288608be82d7e36371':
        translation = np.array([-27.46445655822754, 18.978410720825195, 37.58072280883789], dtype=np.float32)
        quaternion = np.array([0.5536229014396667, 0.19715970754623413, -0.27144017815589905, -0.7622007727622986], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_nuscenes_semseg_8f78c446a68d4854bfb7cdfa1c7097d2':
        translation = np.array([-9.895901679992676, 15.226813316345215, 19.999162673950195], dtype=np.float32)
        quaternion = np.array([0.19350597262382507, 0.11946077644824982, 0.5357224345207214, 0.813194990158081], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-11037651371539287009_77_670_97_670_with_camera_labels_1507944303393935':
        translation = np.array([-24.74045181274414, -1.8987843990325928, 17.814373016357422], dtype=np.float32)
        quaternion = np.array([0.6321805715560913, 0.3547615110874176, -0.3300589323043823, -0.6046100854873657], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-18252111882875503115_378_471_398_471_with_camera_labels_1509125955575722':
        translation = np.array([-32.836299896240234, 30.25058937072754, 21.363615036010742], dtype=np.float32)
        quaternion = np.array([0.3692755401134491, 0.20747537910938263, -0.4307706356048584, -0.7968853116035461], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-18333922070582247333_320_280_340_280_with_camera_labels_1507326323829964':
        translation = np.array([-53.11225509643555, 2.3480372428894043, 18.724817276000977], dtype=np.float32)
        quaternion = np.array([0.599420428276062, 0.36403462290763855, -0.3615887463092804, -0.6143513321876526], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-3077229433993844199_1080_000_1100_000_with_camera_labels_1553271550525306':
        translation = np.array([8.384597778320312, -15.586426734924316, 44.270565032958984], dtype=np.float32)
        quaternion = np.array([0.9985811114311218, 0.027893105521798134, 0.0012666245456784964, 0.04534531012177467], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-8956556778987472864_3404_790_3424_790_with_camera_labels_1513450837409246':
        translation = np.array([6.465387344360352, 10.621289253234863, 26.19098663330078], dtype=np.float32)
        quaternion = np.array([0.051066912710666656, 0.013404211029410362, 0.34771621227264404, 0.9361121654510498], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_semseg_segment-9041488218266405018_6454_030_6474_030_with_camera_labels_1508979405218294':
        translation = np.array([-28.028162002563477, 12.82093334197998, 18.361310958862305], dtype=np.float32)
        quaternion = np.array([0.4304896891117096, 0.252314031124115, -0.4262278079986572, -0.7545503377914429], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0011_01':
        translation = np.array([-2.607921600341797, -3.9115700721740723, 4.767398834228516], dtype=np.float32)
        quaternion = np.array([0.862251341342926, 0.3944227695465088, -0.10950090736150742, -0.29826658964157104], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0164_00':
        translation = np.array([0.44784823060035706, 1.346594214439392, 4.8421549797058105], dtype=np.float32)
        quaternion = np.array([0.6361541152000427, -0.14624781906604767, -0.13321393728256226, -0.7457706332206726], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0591_02':
        translation = np.array([0.20522555708885193, 0.7992222905158997, 6.30797004699707], dtype=np.float32)
        quaternion = np.array([0.096808522939682, -0.03859753534197807, -0.07422225922346115, -0.9917808771133423], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0621_00':
        translation = np.array([-0.5188435912132263, 1.7458114624023438, 7.386157989501953], dtype=np.float32)
        quaternion = np.array([0.18237119913101196, 0.021946880966424942, -0.11573700606822968, -0.9761475920677185], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0645_01':
        translation = np.array([-0.4646049439907074, 0.1932399570941925, 9.053607940673828], dtype=np.float32)
        quaternion = np.array([0.225994274020195, 0.004258096218109131, 0.03415100648999214, 0.9735205173492432], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_scannet_insseg_scene0651_02':
        translation = np.array([-3.1515419483184814, -1.9629572629928589, 4.700793266296387], dtype=np.float32)
        quaternion = np.array([0.8152374625205994, 0.3115808665752411, -0.1690923422574997, -0.4579443335533142], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-13336883034283882790_7100_000_7120_000_with_camera_labels_157':
        translation = np.array([42.82041931152344, 4.8843674659729, 6.764627456665039], dtype=np.float32)
        quaternion = np.array([0.6482436060905457, 0.41732335090637207, 0.35241055488586426, 0.5304980874061584], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-13356997604177841771_3360_000_3380_000_with_camera_labels_002':
        translation = np.array([-23.578227996826172, 45.050628662109375, 64.74434661865234], dtype=np.float32)
        quaternion = np.array([0.004924893379211426, 0.0061909714713692665, -0.262095183134079, -0.9650095701217651], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-14300007604205869133_1160_000_1180_000_with_camera_labels_149':
        translation = np.array([34.866790771484375, -17.64608383178711, 16.711875915527344], dtype=np.float32)
        quaternion = np.array([0.8245905041694641, 0.41970524191856384, 0.17838475108146667, 0.33477896451950073], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-30779396576054160_1880_000_1900_000_with_camera_labels_185':
        translation = np.array([-8.743428230285645, -31.42224884033203, 30.81954574584961], dtype=np.float32)
        quaternion = np.array([0.8754737377166748, 0.4165576100349426, -0.09985091537237167, -0.22373054921627045], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-662188686397364823_3248_800_3268_800_with_camera_labels_019':
        translation = np.array([31.259653091430664, -18.483051300048828, 53.053916931152344], dtype=np.float32)
        quaternion = np.array([0.9408805966377258, 0.32648780941963196, 0.034090492874383926, 0.08359020948410034], dtype=np.float32)
        camera.set_position(quaternion=quaternion, translation=translation)
    elif args.mode == 'paper_litept_waymo_det_segment-8956556778987472864_3404_790_3424_790_with_camera_labels_085':
        translation = np.array([-24.89736557006836, -6.524712562561035, 82.25992584228516], dtype=np.float32)
        quaternion = np.array([0.9744954705238342, 0.18442603945732117, -0.03375321254134178, -0.12331458181142807], dtype=np.float32)
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
    elif args.mode.startswith('paper_litept_scannet_semseg'):
        bpy.context.object.data.energy = 1.5
    elif args.mode.startswith('paper_litept_stru3d_semseg'):
        bpy.context.object.data.energy = 1.5
    elif args.mode.startswith('paper_litept_nuscenes_semseg'):
        bpy.context.object.data.energy = 3.5
        bpy.context.object.data.color = (1.0, 0.8358416557312012, 0.8358416557312012)
        bpy.context.object.rotation_euler = np.array([0.6981316804885864, 0.0, 0.7853981852531433], dtype=np.float32)
    elif args.mode.startswith('paper_litept_waymo_semseg'):
        bpy.context.object.data.energy = 3.5
        bpy.context.object.data.color = (1.0, 0.8358416557312012, 0.8358416557312012)
        bpy.context.object.rotation_euler = np.array([0.6981316804885864, 0.0, 0.7853981852531433], dtype=np.float32)
    elif args.mode.startswith('paper_litept_scannet_insseg'):
        bpy.context.object.data.energy = 1.2
    elif args.mode.startswith('paper_litept_waymo_det'):
        bpy.context.object.data.energy = 3.5
        bpy.context.object.data.color = (1.0, 0.8358416557312012, 0.8358416557312012)
        bpy.context.object.rotation_euler = np.array([0.6981316804885864, 0.0, 0.7853981852531433], dtype=np.float32)

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
    elif args.mode.startswith('paper_litept_scannet_semseg'):
        bg.inputs[1].default_value = 0.9  # strength
    elif args.mode.startswith('paper_litept_stru3d_semseg'):
        bg.inputs[1].default_value = 1.2  # strength
    elif args.mode.startswith('paper_litept_nuscenes_semseg'):
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode.startswith('paper_litept_waymo_semseg'):
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode.startswith('paper_litept_scannet_insseg'):
        bg.inputs[1].default_value = 0.7  # strength
    elif args.mode.startswith('paper_litept_waymo_det'):
        bg.inputs[1].default_value = 0.7  # strength

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
    if not args.no_centering:
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

    # Read and draw bounding boxes
    if args.path_bbox is not None:
        draw_bboxes(
            args.path_bbox,
            face_alpha=0.1,
            sphere_r=0.25,
            edge_r=0.15,
            iou_threshold=0.1,
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
        if args.path_bbox is not None:
            bbox_suffix = f"_bbox-{osp.splitext(args.path_bbox)[0].split('_')[-1]}"
        else:
            bbox_suffix = ''
        for colorname in [k for k in data.keys() if 'color' in k]:
            print(f"Rendering {colorname}...")
            bpy.context.scene.render.filepath = f"{path_image}_{colorname}{bbox_suffix}.png"
            scatter.color = data[colorname]
            bpy.ops.render.render(write_still=True)
            # scatter.base_object.hide_render = True
            # bpy.data.objects.remove(scatter.base_object, do_unlink=True)
            logger.info(f"Rendering of {colorname} complete")

    if args.video:
        # Build the camera trajectory
        logger.info("Creating camera and interpolating its trajectory")
        if args.mode == 'paper_ezsp_dales':
            start_position = np.array([0, 0, 30], dtype=np.float32)
            start_target = np.array([0, 40, 0], dtype=np.float32)
            spiral_target = start_position
            spin_spiral_ratio = 0.4
            spin_angle = np.pi / 2
            spiral_angle = 2 * np.pi
            z_max = 150
            r_max = 150
        elif args.mode == 'paper_ezsp_kitti360':
            start_position = np.array([-36, 38, 5], dtype=np.float32)
            start_target = np.array([-65, 20, 2], dtype=np.float32)
            spiral_target = np.array([3, 4, 4.5], dtype=np.float32)
            spin_spiral_ratio = 0.8
            spin_angle = 2 * np.pi / 3
            spiral_angle = np.pi / 2
            z_max = 75
            r_max = 150
        elif args.mode in ['paper_ezsp_s3dis', 'paper_ezsp_s3dis_2']:
            start_position = np.array([0, 3, 1.7], dtype=np.float32)
            start_target = np.array([5, 0, 1.4], dtype=np.float32)
            spiral_target = start_position
            spin_spiral_ratio = 0.5
            spin_angle = np.pi
            spiral_angle = 3 * np.pi
            z_max = 50
            r_max = 50
        start_quaternion = look_at_quaternion(start_position, start_target)
        spin_duration = args.duration * spin_spiral_ratio
        spiral_duration = args.duration - spin_duration
        spin_poses = spin_around_trajectory(
            start_position,
            start_quaternion,
            fps=args.fps,
            duration=spin_duration,
            angular_speed=spin_angle / spin_duration)

        # Dirty overwrite of the "spin" trajectory by a "path"
        # trajectory for KITTI360
        if args.mode == 'paper_ezsp_kitti360':
            spin_poses = path_trajectory(
                [
                    0 * spin_duration / 4,
                    1 * spin_duration / 4,
                    2 * spin_duration / 4,
                    3 * spin_duration / 4,
                    4 * spin_duration / 4,
                ],
                [
                    [-66, 10, 4.5],
                    [-47, 28, 4.5],
                    [-37, 36, 4.5],
                    [-30, 32, 4.5],
                    [3, 4, 4.5],
                ],
                fps=args.fps)

        spiral_poses = spiral_camera_trajectory(
            spin_poses[list(spin_poses.keys())[-1]][0],
            spin_poses[list(spin_poses.keys())[-1]][1],
            spiral_target,
            fps=args.fps,
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
            time_step=1 / args.fps,
            smoothness=5.0)

        # Create a color interpolator
        if args.mode == 'paper_ezsp_dales':
            color_times = [
                ['intensity', 0],
                ['0_level', spin_duration / 2],
                ['pred', 2 * spin_duration]]
        elif args.mode == 'paper_ezsp_kitti360':
            color_times = [
                ['rgb', 0],
                ['0_level', spin_duration / 2],
                ['pred', spin_duration]]
        elif args.mode in ['paper_ezsp_s3dis', 'paper_ezsp_s3dis_2']:
            color_times = [
                ['rgb', 0],
                ['0_level', spin_duration / 2],
                ['pred', 3 * spin_duration / 2]]
        color_interpolator = ColorInterpolator(
            [data[f'{ct[0]}_colors'] for ct in color_times],
            [ct[1] for ct in color_times],
            fade_duration=1)

        logger.info("Entering the main drawing loop")
        total_frames = len(camera_trajectory)
        video_path = (
            f"{path_image}"
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

    # Optionally save blend file with the scene at frame 0
    if args.export:
        scene.export(path_blender)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Blendify example 04: Camera colored PointCloud.")

    # Paths to input files
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the input point cloud data")
    parser.add_argument(
        "--path_bbox",
        default=None,
        type=str,
        help="Path to the input bbox data")

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
    parser.add_argument(
        "--no_centering",
        action='store_true',
        help="Whether to center the point cloud before rendering")

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
