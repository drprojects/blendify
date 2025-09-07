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


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Custom Blendify script for point clouds visualization")

    # # Attach blender file with scene (walls and floor)
    # logger.info("Attaching blend to the scene")
    # scene.attach_blend("./assets/light_box.blend")

    # Set custom parameters to improve quality of rendering
    bpy.context.scene.cycles.max_bounces = 30
    bpy.context.scene.cycles.transmission_bounces = 20
    bpy.context.scene.cycles.transparent_max_bounces = 15
    bpy.context.scene.cycles.diffuse_bounces = 10
    bpy.context.scene.view_settings.view_transform = 'Standard'  # could also be Filmic
    bpy.context.scene.render.resolution_x = args.resolution[0]
    bpy.context.scene.render.resolution_y = args.resolution[1]
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = args.n_samples
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
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
    camera = scene.set_perspective_camera(resolution=args.resolution, fov_x=np.deg2rad(73))
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
        bpy.context.object.data.energy = 2
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
        bg.inputs[1].default_value = 0.5  # strength

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
    specularity = 0.1
    roughness = 0.2
    point_size = args.voxel

    for colorname in [k for k in data.keys() if 'color' in k]:
        # Create Scatter for this layer
        scatter = bplt.Scatter(
            pos.numpy(),
            color=adjust_color(np.asarray(data[colorname]).astype('float32') / 255.),
            marker_type="spheres",
            name=f"point_cloud_{colorname}",
            radius=point_size)
        scatter.color_material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = specularity
        scatter.color_material.node_tree.nodes["Principled BSDF"].inputs[9].default_value = roughness

        if args.mode == 'paper_ezsp_dales':
            scatter.base_object.rotation_euler = np.array([0.0, -0.0, -2.099583387374878], dtype=np.float32)

        # Render image and save to disk
        bpy.context.scene.render.filepath = f"{path_image}_{colorname}.png"
        if args.images:
            print(f"Rendering {colorname}...")
            bpy.ops.render.render(write_still=True)

            # Remove or hide scatter before next iteration
            scatter.base_object.hide_render = True
            bpy.data.objects.remove(scatter.base_object, do_unlink=True)

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
    scene.export(path_blender)

    # # Render the video frame by frame
    # logger.info("Entering the main drawing loop")
    # total_frames = len(camera_trajectory)
    # with VideoWriter(args.path, resolution=args.resolution, fps=30) as vw:
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
    parser = argparse.ArgumentParser(description="Blendify example 04: Camera colored PointCloud.")

    # Paths to output files
    parser.add_argument("-p", "--path", type=str, help="Path to the input data")
    parser.add_argument("-v", "--voxel", type=float, help="Voxel size")

    parser.add_argument("-i", "--images", action='store_true',
                        help="Whether to render images")

    # Rendering parameters
    parser.add_argument("-n", "--n-samples", default=32, type=int,
                        help="Number of paths to trace for each pixel in the render (default: 32)")
    parser.add_argument("-res", "--resolution", default=(2100, 1400), nargs=2, type=int,
                        help="Rendering resolution, (default: (2100, 1400))")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU for rendering (by default GPU is used)")

    # Other parameters
    parser.add_argument("-b", "--backend", type=str, default="orig", choices=["orig", "open3d", "pytorch3d"],
                        help="Backend to use for normal estimation. Orig corresponds to original mesh normals, "
                             "i.e. no estimation is performed (default: orig)")

    parser.add_argument("-m", "--mode", type=str, default=None,
                        help="Mode for printing a scene. Some predefined recipes are there.")

    arguments = parser.parse_args()
    main(arguments)
