import bpy
import bmesh
from mathutils import Vector
import numpy as np
import time
from plyfile import PlyData
from scipy.spatial import ConvexHull


LABEL_MAP = {
    (0, 0, 255): "Vehicle",     # (255,0,0)
    (255, 0, 0): "Pedestrian",  # (0,255,0)
    (0, 255, 0): "Cyclist",     # (0,0,255)
}

COLOR_MAP_256 = {
    "Vehicle": (75, 112, 183),
    "Pedestrian": (230, 25, 75),
    "Cyclist": (245, 158, 37),
}

COLOR_MAP = {k: tuple(c / 256 for c in color) for k, color in COLOR_MAP_256.items()}

BOX_EDGES = [
    (0,1), (1,2), (2,3), (3,0),   # top loop
    (4,5), (5,6), (6,7), (7,4),   # bottom loop
    (0,4), (1,5), (2,6), (3,7)    # vertical edges
]

BOX_FACES = [
    (0,1,2,3),     # top
    (4,5,6,7),     # bottom
    (0,1,5,4),     # side
    (1,2,6,5),
    (2,3,7,6),
    (3,0,4,7)
]

def rgb_to_label(rgb):
    # round to avoid float noise
    key = tuple(np.round(rgb, decimals=3))
    return LABEL_MAP.get(key, "Unknown")


def load_ply_bboxes(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data

    xyz = np.vstack([v['x'], v['y'], v['z']]).T       # (N,3)
    rgb = np.vstack([v['red'], v['green'], v['blue']]).T  # scaled

    assert len(xyz) % 8 == 0, "Expected groups of 8 corners per bbox"
    num = len(xyz) // 8

    bboxes = []
    for i in range(num):
        pts = xyz[i*8:(i+1)*8]
        col = rgb[i*8]   # first point color
        label = rgb_to_label(col)
        col = COLOR_MAP[label]
        bboxes.append((pts, col, label))

    return bboxes


def box_volume(corners):
    hull = ConvexHull(corners)
    return hull.volume


def intersect_boxes(corners_a, corners_b):
    """
    Computes intersection volume of two convex polyhedra given 8 corners each.
    Uses halfspace intersection (scipy).
    """
    from scipy.spatial import HalfspaceIntersection

    def poly_halfspaces(pts):
        hull = ConvexHull(pts)
        eqs = hull.equations  # shape (F, 4): normal + offset
        return eqs

    # get halfspaces
    H = np.vstack([poly_halfspaces(corners_a),
                   poly_halfspaces(corners_b)])

    # interior point needed → take mean of both boxes
    interior = 0.5 * (corners_a.mean(0) + corners_b.mean(0))

    try:
        hs = HalfspaceIntersection(H, interior)
        vol = ConvexHull(hs.intersections).volume
    except:
        vol = 0.0

    return vol


def iou_3d(corners_a, corners_b):
    volA = box_volume(corners_a)
    volB = box_volume(corners_b)
    inter = intersect_boxes(corners_a, corners_b)
    union = volA + volB - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def nms_3d(bboxes, scores, iou_threshold=0.3):
    idxs = np.argsort(scores)[::-1]  # descending
    keep = []

    while len(idxs) > 0:
        best = idxs[0]
        keep.append(best)

        remaining = []
        for i in idxs[1:]:
            iou = iou_3d(bboxes[best], bboxes[i])
            if iou < iou_threshold:
                remaining.append(i)
        idxs = np.array(remaining)

    return keep


def make_solid_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    BSDF = mat.node_tree.nodes["Principled BSDF"]
    BSDF.inputs["Base Color"].default_value = (*color, 1.0)
    BSDF.inputs["Roughness"].default_value = 0.3
    return mat


def make_transparent_material(name, color, alpha):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        if n.name != "Material Output":
            nodes.remove(n)

    output = nodes["Material Output"]
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, alpha)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = 0.3

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'
    return mat


def draw_sphere(location, radius, material, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def draw_cylinder(p0, p1, radius, material, name):
    p0 = Vector(p0)
    p1 = Vector(p1)
    mid = (p0 + p1) / 2
    height = (p1 - p0).length

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=height, location=mid)
    obj = bpy.context.object
    obj.name = name

    direction = (p1 - p0).normalized()
    up = Vector((0,0,1))
    rot = up.rotation_difference(direction)
    obj.rotation_quaternion = rot

    obj.data.materials.append(material)
    return obj


def draw_face(points, indices, material, name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    face_verts = [points[i] for i in indices]
    mesh.from_pydata(face_verts, [], [list(range(len(face_verts)))])
    mesh.update()

    obj.data.materials.append(material)
    return obj


def draw_bbox(
        points,
        color,
        face_alpha=0.2,
        sphere_r=0.06,
        edge_r=0.03,
        name_prefix="bbox"
):

    label = rgb_to_label(color)
    color = COLOR_MAP[label]

    # Parent empty
    empty = bpy.data.objects.new(f"{name_prefix}_{label}", None)
    bpy.context.collection.objects.link(empty)

    # Materials
    edge_mat = make_solid_material(f"{name_prefix}_edge_mat", color)
    sphere_mat = make_solid_material(f"{name_prefix}_sphere_mat", color)
    face_mat = make_transparent_material(f"{name_prefix}_face_mat", color, face_alpha)

    # Corners
    for i, p in enumerate(points):
        obj = draw_sphere(p, sphere_r, sphere_mat, f"{name_prefix}_corner_{i}")
        obj.parent = empty

    # Edges
    for (i, j) in BOX_EDGES:
        obj = draw_cylinder(points[i], points[j], edge_r, edge_mat,
                            f"{name_prefix}_edge_{i}_{j}")
        obj.parent = empty

    # Semi-transparent faces
    for f_idx, f in enumerate(BOX_FACES):
        obj = draw_face(points, f, face_mat, f"{name_prefix}_face_{f_idx}")
        obj.parent = empty

    # Text label above box
    center = np.mean(points, axis=0)
    txt_loc = (center[0], center[1], center[2] + 0.5)
    bpy.ops.object.text_add(location=txt_loc)
    txt = bpy.context.object
    txt.data.body = label
    txt.scale = (0.3, 0.3, 0.3)
    txt.parent = empty

    return empty


def log(msg):
    print(f"[BBox] {msg}")


def build_material(color, alpha=1.0):
    """Build material with given RGB (0-1) and alpha"""
    mat = bpy.data.materials.new(name="bbox_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, alpha)
    bsdf.inputs["Alpha"].default_value = alpha
    mat.blend_method = 'BLEND' if alpha < 1 else 'OPAQUE'
    return mat


def build_glass_material(color, alpha=0.05, transmission=0.9, roughness=0.05):
    mat = bpy.data.materials.new(name="bbox_glass")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*[c/255 for c in color], alpha)
    bsdf.inputs["Transmission"].default_value = transmission
    bsdf.inputs["Roughness"].default_value = roughness
    mat.blend_method = 'BLEND'   # for Eevee
    return mat


def build_alpha_material(color, alpha=0.05):
    """
    color: RGB tuple 0-255
    alpha: 0.0 fully transparent, 1.0 opaque
    """
    mat = bpy.data.materials.new(name="bbox_face_alpha")
    mat.use_nodes = True

    # Principled BSDF
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*[c / 255 for c in color], alpha)
    bsdf.inputs["Alpha"].default_value = alpha

    # Enable transparency in Eevee
    mat.blend_method = 'BLEND'  # allows alpha blending
    mat.show_transparent_back = True  # optional: see through backfaces

    return mat


BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),      # top square
    (4,5),(5,6),(6,7),(7,4),      # bottom square
    (0,4),(1,5),(2,6),(3,7)       # verticals
]


BOX_FACES = [
    (0,1,2,3),
    (4,5,6,7),
    (0,1,5,4),
    (1,2,6,5),
    (2,3,7,6),
    (3,0,4,7)
]

def draw_bbox_fast(
        points,
        color,
        face_alpha=0.05,
        sphere_r=0.05,
        edge_r=0.02,
        name="bbox"
):
    t0 = time.time()

    # ------------------ 1️⃣ Create faces --------------------------
    log("Creating base vertices and faces...")
    bm = bmesh.new()
    base_verts = [bm.verts.new(Vector(p)) for p in points]
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()

    # Add faces (quads only)
    for face_idx in BOX_FACES:
        bm.faces.new([base_verts[k] for k in face_idx])

    # Create mesh object for faces
    mesh_face = bpy.data.meshes.new(f"{name}_faces_mesh")
    obj_face = bpy.data.objects.new(f"{name}_faces", mesh_face)
    bpy.context.scene.collection.objects.link(obj_face)
    bm.to_mesh(mesh_face)
    bm.free()

    # Assign face material
    mat_face = build_alpha_material(
        color,
        alpha=face_alpha,
    )
    obj_face.data.materials.append(mat_face)
    for poly in obj_face.data.polygons:
        poly.material_index = 0  # all quads

    # ------------------ 2️⃣ Create spheres + edges ------------------
    log("Adding corner spheres and edge cylinders...")
    mat_solid = build_material(color, alpha=1.0)

    # Create a single BMesh for all spheres and cylinders
    bm2 = bmesh.new()

    # Spheres
    for p in points:
        sphere = bmesh.ops.create_icosphere(bm2, subdivisions=2, radius=sphere_r)
        bmesh.ops.translate(bm2, verts=sphere["verts"], vec=Vector(p))

    # Cylinders (edges)
    up = Vector((0,0,1))
    for i,j in BOX_EDGES:
        p0 = Vector(points[i])
        p1 = Vector(points[j])
        height = (p1 - p0).length
        mid = (p0 + p1) / 2
        direction = (p1 - p0).normalized()

        cyl = bmesh.ops.create_cone(
            bm2, cap_ends=False, segments=16, radius1=edge_r, radius2=edge_r, depth=height
        )
        q = up.rotation_difference(direction)
        bmesh.ops.rotate(bm2, verts=cyl["verts"], cent=Vector((0,0,0)), matrix=q.to_matrix())
        bmesh.ops.translate(bm2, verts=cyl["verts"], vec=mid)

    # Create mesh object for corners + edges
    mesh_solid = bpy.data.meshes.new(f"{name}_solid_mesh")
    obj_solid = bpy.data.objects.new(f"{name}_solid", mesh_solid)
    bpy.context.scene.collection.objects.link(obj_solid)
    bm2.to_mesh(mesh_solid)
    bm2.free()

    # Assign solid material
    obj_solid.data.materials.append(mat_solid)
    for poly in obj_solid.data.polygons:
        poly.material_index = 0  # all polygons in solid mesh

    log(f"{name} done in {time.time() - t0:.3f}s")
    return obj_face, obj_solid


def draw_bboxes(
        path,
        face_alpha=0.05,
        sphere_r=0.05,
        edge_r=0.02,
        iou_threshold=0.1,
        prefix="bbox",
):
    # Read the bboxes from ply format
    # TODO: this is specific to the waymo-litept files Yuanwen sent...
    bboxes = load_ply_bboxes(path)

    # Remove overlapping boxes
    indices_to_keep = nms_3d(
        [b[0] for b in bboxes],
        scores=np.ones(len(bboxes)),
        iou_threshold=iou_threshold,
    )
    bboxes = [bboxes[i] for i in indices_to_keep]

    # Draw boxes in blender
    for k, (points, color, label) in enumerate(bboxes):
        draw_bbox_fast(
            points,
            color,
            face_alpha=face_alpha,
            sphere_r=sphere_r,
            edge_r=edge_r,
            name=f"{prefix}_{k}",
        )
