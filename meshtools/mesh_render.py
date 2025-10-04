import trimesh
import pyrender
import numpy as np
import pyvista as pv
import math
import os
import argparse
from shapely.geometry import Polygon
import shapely.geometry as geom
import geopandas as gpd


def reconstraction(planes, textures, out_obj_path):
    recon_scene = pv.Plotter()
    recon_scene.add_axes()
    texcoords1 = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    for i, plane in enumerate(planes):
        heigh = np.array(plane)[:, 2]
        print(heigh)
        if np.all(heigh == heigh[0]):
            polygoncoor = np.array(plane)[:, 0:2]
            # print(polygoncoor)
            polygon = Polygon(polygoncoor)
            v, f = trimesh.creation.triangulate_polygon(polygon)
            # print(f)
            plane_recons = trimesh.Trimesh(vertices=plane, faces=f)
            plane_recons.visual.uv = texcoords1
            recon_scene.add_mesh(plane_recons, texture=textures[i])
        else:
            triangles = [[0, 1, 3, 2]]
            plane_recons = trimesh.Trimesh(vertices=plane, faces=triangles)
            plane_recons.visual.uv = texcoords1

            recon_scene.add_mesh(plane_recons, texture=textures[i])

    recon_scene.export_obj(os.path.join(out_obj_path, "rebuilding.obj"))
    # recon_scene.show()
    # recon_scene.close()

    return


def render_plane_single(
    tri_scene,
    plane,
    bgcolor=(0, 0, 1.0, 0.0),
    reslution=0.01,
):
    y = np.array(plane[0] - plane[1])
    z = np.cross(plane[0] - plane[1], plane[2] - plane[0])
    x = np.cross(y, z)
    vec_norm_y = np.linalg.norm(y)
    unit_vec_y = y / vec_norm_y

    vec_norm_z = np.linalg.norm(z)
    unit_vec_z = z / vec_norm_z

    vec_norm_x = np.linalg.norm(x)
    unit_vec_x = x / vec_norm_x

    points = np.array(plane)

    center = points.mean(axis=0)
    center_forward = center + 5 * unit_vec_z

    matrix = np.eye(4)
    matrix[:3, 0] = unit_vec_x
    matrix[:3, 1] = unit_vec_y
    matrix[:3, 2] = unit_vec_z
    matrix[:3, 3] = center_forward
    # print(matrix)
    proj_points = trimesh.transform_points(points, matrix)

    xmag = math.sqrt(
        (proj_points[0, 0] - proj_points[1, 0]) ** 2
        + (proj_points[0, 1] - proj_points[1, 1]) ** 2
        + (proj_points[0][2] - proj_points[1][2]) ** 2
    )
    ymag = math.sqrt(
        (proj_points[0, 0] - proj_points[2, 0]) ** 2
        + (proj_points[0, 1] - proj_points[2, 1]) ** 2
        + (proj_points[0][2] - proj_points[2][2]) ** 2
    )

    xmag = xmag / 2
    ymag = ymag / 2

    tri_scene = tri_scene
    bgcolor = (0, 0, 1.0, 0.0)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=bgcolor)
    camrea = pyrender.OrthographicCamera(ymag, xmag, znear=0.1, zfar=1000)
    # light = pyrender.DirectionalLight(color=lightcolor, intensity=10.0)
    # scene.add(light,pose=matrix)
    scene.add(camrea, pose=matrix)
    # pyrender.Viewer(scene)

    reslution = 0.01
    width = int(xmag / reslution)
    height = int(ymag / reslution)
    render = pyrender.OffscreenRenderer(height, width)
    color, _ = render.render(scene, pyrender.RenderFlags.FLAT)

    return color


def point_in_quadrilateral(A, B, C, D, P):
    # 计算向量PA、PB、PC和PD与向量AB、BC、CD和DA的叉积
    cross_AB_PA = np.cross(B - A, P - A)
    cross_BC_PB = np.cross(C - B, P - B)
    cross_CD_PC = np.cross(D - C, P - C)
    cross_DA_PD = np.cross(A - D, P - D)

    if (
        cross_AB_PA >= 0 and cross_BC_PB >= 0 and cross_CD_PC >= 0 and cross_DA_PD >= 0
    ) or (
        cross_AB_PA <= 0 and cross_BC_PB <= 0 and cross_CD_PC <= 0 and cross_DA_PD <= 0
    ):
        return True
    else:
        return False


def load_objs_box(objs):
    bounds = []
    # zrange = [-np.inf, np.inf]
    for obj in objs:
        # print(obj)
        box = obj.replace(".obj", "_bbox.obj")
        if os.path.exists(box):
            box = trimesh.load(box)
            bounds.append(box.bounds)
        else:
            o = trimesh.load(obj)
            bounds.append(o.bounds)
            mbox = trimesh.creation.box(bounds=o.bounds)
            mbox.export(box)
    minz = np.min([b[0, 2] for b in bounds])
    maxz = np.max([b[1, 2] for b in bounds])
    zrange = [maxz, minz]
    bounds_geom = [geom.box(b[0, 0], b[0, 1], b[1, 0], b[1, 1]) for b in bounds]
    bounds = gpd.GeoDataFrame(
        geometry=bounds_geom, data={"obj": objs, "bounds": bounds}
    )
    return bounds, zrange


def load_ccobjs_to_trimesh(root_data_path, sub=None):
    paths = os.listdir(root_data_path)
    results = []
    for pth in paths:
        f = os.path.join(root_data_path, pth, f"{pth}.obj")
        f = f.replace("//", "/")
        f = f.replace("\\", "/")
        if os.path.exists(f):
            # print(f)
            f = f.replace("//", "/")
            f = f.replace("\\", "/")
            results.append(f)
    if sub is not None:
        results = results[sub]

    return results


def render_batch_planes(planes, parent_folder):
    # mesh = []
    textures = []
    _planes = []
    for plane in planes:
        _, f = trimesh.creation.triangulate_polygon(geom.Polygon(plane[:, :2]), 0.1)
        m = trimesh.Trimesh(vertices=plane, faces=f)
        p, t = render_trimesh_planes(m, parent_folder)
        _planes.extend(p)
        textures.extend(t)
    return _planes, textures


def render_trimesh_planes(building_mesh, parent_folder):
    # building_mesh: trimesh.Trimesh = trimesh.load(args.inputpath)
    bound = building_mesh.bounds
    # bound = [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    xypolygon = geom.Polygon(
        [
            [bound[0, 0], bound[0, 1]],
            [bound[0, 0], bound[1, 1]],
            [bound[1, 0], bound[1, 1]],
            [bound[1, 0], bound[0, 1]],
        ]
    )
    objs = load_ccobjs_to_trimesh(parent_folder)
    boxes, _ = load_objs_box(objs)
    ins = boxes.intersects(xypolygon)
    if ins.sum() > 0:
        objs = boxes[ins]["obj"]
    else:
        return [], []
    meshes = [trimesh.load(f) for f in objs]
    tri_scene = trimesh.Scene(meshes)

    lenght = building_mesh.facets_boundary

    all_vertices = building_mesh.vertices
    planes = []
    textures = []
    for i in range(len(lenght)):
        plane = []
        indexva = []
        for j in range(len(all_vertices[lenght[i]])):
            if j == 0:
                plane.append(all_vertices[lenght[i][j][0]])
                plane.append(all_vertices[lenght[i][j][1]])
                indexva.append(lenght[i][j][0])
                indexva.append(lenght[i][j][1])
            else:
                if lenght[i][j][0] in indexva:
                    pass
                else:
                    plane.append(all_vertices[lenght[i][j][0]])
                    indexva.append(lenght[i][j][0])
                if lenght[i][j][1] in indexva:
                    pass
                else:
                    plane.append(all_vertices[lenght[i][j][1]])
                    indexva.append(lenght[i][j][1])

        if i == len(lenght) - 3:
            plane[:2], plane[2:] = plane[2:], plane[:2]
        planes.append(plane)
        texture = render_plane_single(tri_scene, plane)
        textures.append(texture)
    return planes, textures


def main():
    parser = argparse.ArgumentParser(description="process some integers")
    parser.add_argument("--inputpath", "-i", type=str, help="Input file path")
    parser.add_argument(
        "--parent_folder", "-p", type=str, help="the parent folder of objs"
    )
    parser.add_argument("--outputpath", "-o", type=str, help="Output file path")
    args = parser.parse_args()
    parent_folder = args.parent_folder

    if args.inputpath.endswith(".txt"):
        planes = []
        with open(args.inputpath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                points = np.array(list(map(float, line.split(",")))).reshape(-1, 3)
                planes.append(points)
        planes, textures = render_batch_planes(planes, parent_folder)

    else:
        mesh = trimesh.load_mesh(args.inputpath)
        planes, textures = render_trimesh_planes(mesh, parent_folder)
    if len(textures) > 0:
        reconstraction(planes, textures, args.outputpath)
    # reconstraction(planes, textures, args.outputpath)


if __name__ == "__main__":
    main()
