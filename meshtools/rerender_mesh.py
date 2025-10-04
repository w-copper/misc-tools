import argparse
import trimesh
import os
import numpy as np
from mesh_tools import (
    render_one,
    load_ccobjs_to_trimesh,
    load_objs_box,
)
import geopandas as gpd
from shapely import geometry as geom
from trimesh.visual import TextureVisuals
from trimesh.visual import material
from PIL import Image
import tqdm

# print(ctypes.CDLL("d:/wt/vcpkg/installed/x64-windows/bin/libEGL.dll"))
# print(ctypes.CDLL("d:/wt/vcpkg/installed/x64-windows/bin/libGLESv2.dll"))

# os.environ["PYOPENGL_PLATFORM"] = "egl"


def _compute_uvw(submesh: trimesh.Trimesh, gap):
    A, B, C = submesh.vertices[submesh.faces[0]]
    N = np.cross(B - A, C - A) * np.sign(gap)
    gap = np.abs(gap)
    N = N / np.linalg.norm(N)  # 归一化
    xy_N = np.array([N[0], N[1], 0])
    xaxis = np.zeros(3)
    if xy_N[0] == 0 and xy_N[1] == 0:  # x与xy平面垂直
        xaxis += np.array([1, 0, 0])
    else:  # 使用向量x和z轴单位向量做叉积计算正交向量
        z_unit_vector = np.array([0, 0, 1])
        orthogonal = np.cross(xy_N, z_unit_vector)
        # 确保正交向量是单位向量，更有通用性
        xaxis += orthogonal / np.linalg.norm(orthogonal)
    xaxis = xaxis / np.linalg.norm(xaxis)  # 归一化
    yaxis = np.cross(N, xaxis)

    rotation_matrix = np.array([xaxis, yaxis, -N]).T
    center = np.array(submesh.centroid)
    center = center + gap * N
    # 步骤4和5: 转换顶点到新坐标系
    vers = np.array(submesh.vertices)

    ss = trimesh.Trimesh(vertices=vers + gap * N, faces=submesh.faces)

    new_vers = np.dot(rotation_matrix.T, (vers - center).T).T

    min_x = np.min(new_vers[:, 0])
    max_x = np.max(new_vers[:, 0])
    min_y = np.min(new_vers[:, 1])
    max_y = np.max(new_vers[:, 1])

    new_center = np.array(
        [min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2, new_vers[0, 2]]
    )
    new_center = np.dot(rotation_matrix, new_center)
    new_center += center
    width = max_x - min_x
    height = max_y - min_y
    new_vers = np.dot(rotation_matrix.T, (vers - new_center).T).T
    min_x = np.min(new_vers[:, 0])
    max_x = np.max(new_vers[:, 0])
    min_y = np.min(new_vers[:, 1])
    max_y = np.max(new_vers[:, 1])
    u = (new_vers[:, 0] - min_x) / (max_x - min_x)
    v = (new_vers[:, 1] - min_y) / (max_y - min_y)

    return ss, xaxis, yaxis, N, new_center, width, height, [u, v]


def compute_uvw(submesh: trimesh.Trimesh, gap):
    posmesh = _compute_uvw(submesh, 0.03)
    negmesh = _compute_uvw(submesh, -0.03)
    return posmesh, negmesh


def set_face_matriels(mesh, uv, image):
    uv = np.array(uv).T
    image = Image.fromarray(image)
    image.format = "JPEG"

    mesh.visual = TextureVisuals(
        uv=uv,
        material=material.SimpleMaterial(
            image=image,
            diffuse=(255, 255, 255),
            ambient=(255, 255, 255),
            specular=(255, 255, 255),
            glossiness=1.0,
        ),
    )

    # trimesh.Scene(mesh).show(flags=Render)
    return mesh


def are_similar_normals(norm1, norm2, threshold=0.1):
    # 比较两个法向量的差值是否小于某个阈值
    return np.linalg.norm(norm1 - norm2) < threshold


def dfs(mesh: trimesh.Trimesh, face_index, visited, cluster, threshold=0.1):
    face_normals = mesh.face_normals
    for edge in mesh.face_adjacency:
        if face_index in edge:
            neighbor_index = edge[1] if edge[0] == face_index else edge[0]
            if neighbor_index not in visited and are_similar_normals(
                face_normals[face_index], face_normals[neighbor_index], threshold
            ):
                visited.add(neighbor_index)
                cluster.append(neighbor_index)
                dfs(mesh, neighbor_index, visited, cluster, threshold)


def cluster_face(mesh: trimesh.Trimesh, threshold=0.05):
    clusters = []
    visited = set()
    for face_index in range(len(mesh.faces)):
        if face_index not in visited:
            cluster = [face_index]
            visited.add(face_index)
            dfs(mesh, face_index, visited, cluster, threshold)
            clusters.append(cluster)

    return clusters


def create_submesh(mesh, faces_indices):
    """
    从原始网格和给定的面的索引中，提取子网格。
    参数：
    - mesh: 原始的完整 trimesh 对象。
    - faces_indices: 这个聚类中所有面的索引列表。

    返回：
    - submesh: 一个新的 trimesh 对象，仅包含指定的面。
    """
    # 提取对应的面（三角形顶点的索引）
    faces = mesh.faces[faces_indices]

    # 提取所有涉及的顶点索引，并去除重复项
    unique_vertices_indices = np.unique(faces)

    # 建立新索引映射
    new_indices_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices_indices)
    }

    # 根据映射，更新面的顶点索引
    new_faces = np.array([[new_indices_map[idx] for idx in face] for face in faces])

    # 提取对应的顶点坐标
    vertices = mesh.vertices[unique_vertices_indices]

    return trimesh.Trimesh(vertices=vertices, faces=new_faces)


def run_one_file(pmesh: gpd.GeoDataFrame, obj, gap=0.3, resolution=0.01):
    mesh: trimesh.Trimesh = trimesh.load(obj, force="mesh")
    center = mesh.centroid
    face_meshes = []
    bounds = mesh.bounds
    bound_geom = geom.Polygon(
        [
            [bounds[0][0], bounds[0][1]],
            [bounds[1][0], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            [bounds[0][0], bounds[1][1]],
        ]
    )
    objs = pmesh.intersects(bound_geom)
    if objs.sum() == 0:
        return mesh
    objs = pmesh[objs]["obj"]
    scene = trimesh.Scene([trimesh.load(o) for o in objs])
    cluster = cluster_face(mesh, threshold=0.05)
    # print("cluster", cluster)
    # return
    for cl in tqdm.tqdm(cluster):
        submesh = create_submesh(mesh, cl)
        for sub in compute_uvw(submesh, gap):
            smesh, xvec, yvec, zvec, center, xmag, ymag, uv = sub

            matrix = np.eye(4)
            matrix[:3, 0] = xvec
            matrix[:3, 1] = yvec
            matrix[:3, 2] = zvec
            matrix[:3, 3] = center
            camera = {
                "type": "orthographic",
                "xmag": xmag / 2,
                "ymag": ymag / 2,
                "width": int(xmag / resolution),
                "zfar": 20,
                "height": int(ymag / resolution),
                "postype": "matrix",
                "matrix": matrix,
            }
            # from pyrender.renderer import RenderFlags
            # import matplotlib.pyplot as plt

            # visgeom = get_orthographic_visual_geom(xmag / 2, ymag / 2, 0.01, 20, matrix)
            # visgeom.visual = ColorVisuals(visgeom, face_colors=(1.0, 0.0, 0.0, 0.5))
            # smesh.visual = ColorVisuals(smesh, face_colors=(0.0, 0.0, 1.0))
            # s = trimesh.Scene([smesh, scene, visgeom])
            # s.show(flags=RenderFlags.FLAT)
            img, _, _ = render_one(camera, scene, bgcolor=(1, 1, 1))
            # plt.imshow(img)
            # plt.show()
            face_meshes.append(set_face_matriels(smesh, uv, img))

    return trimesh.Scene(face_meshes)


def run(objroot, obj, gap=0.3):
    meshes = load_ccobjs_to_trimesh(os.path.join(objroot, "Data"))
    pmesh, _ = load_objs_box(meshes)

    if os.path.isfile(obj):
        mesh = run_one_file(pmesh, obj, gap=gap)
        mesh.export(obj.replace(".obj", "_rerender.obj"))
    else:
        for obj in os.listdir(obj):
            mesh = run_one_file(pmesh, os.path.join(obj), gap=gap)
            mesh.export(os.path.join(obj, obj.replace(".obj", "_rerender.obj")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objroot", type=str, default="data")
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--gap", type=float, default=1)
    args = parser.parse_args()
    run(args.objroot, args.obj, gap=args.gap)

    # python rerender_mesh.py --objroot "P:/Projects/20240419 shanghai/Productions/Production_wd_obj" --obj e:/Data\上海项目\0419万达\万达\rebuilding.obj
