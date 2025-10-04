import trimesh
from trimesh import creation
import os
import numpy as np
import json
import tqdm
import pyrender
import PIL.Image as Image
import logging
import argparse
from render_patch import OffRender
from shapely import geometry as geom
import geopandas as gpd


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
        if os.path.exists(f):
            f = f.replace("\\", "/")
            logging.info(f)
            results.append(f)
    if sub is not None:
        results = results[sub]

    return results


def get_matrix(zrot=0, yrot=0, xrot=0, pos=np.zeros(3), xyz="xyz"):
    """
    xyzrot: 顺时针旋转角度,单位为度
    return: np.array with shape 4x4
    """
    zrot = np.deg2rad(zrot)
    yrot = np.deg2rad(yrot)
    xrot = np.deg2rad(xrot)

    # 计算绕U轴旋转的矩阵
    z = np.array(
        [[np.cos(zrot), np.sin(zrot), 0], [-np.sin(zrot), np.cos(zrot), 0], [0, 0, 1]]
    )
    # print(rotation_u.shape)

    # 计算绕N轴旋转的矩阵
    x = np.array(
        [[1, 0, 0], [0, np.cos(xrot), np.sin(xrot)], [0, -np.sin(xrot), np.cos(xrot)]]
    )
    # 计算绕E轴旋转的矩阵
    y = np.array(
        [[np.cos(yrot), 0, -np.sin(yrot)], [0, 1, 0], [np.sin(yrot), 0, np.cos(yrot)]]
    )

    # 计算旋转后的坐标
    # rotated = np.dot(rotation_n, np.dot(rotation_u, np.vstack((e, n, u))))
    xyzs = dict(x=x, y=y, z=z)
    rot_mat = np.eye(3)
    for i in xyz:
        rot_mat = np.dot(rot_mat, xyzs[i])

    # rot_mat = np.dot(np.dot(rotation_e, rotation_n), rotation_u)
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = rot_mat[:, :]
    matrix[:3, 3] = pos[:]
    matrix[-1, -1] = 1
    # matrix[-1,-1] = 1
    return matrix


def compute_zmin_zmax(meshes):
    zmin, zmax = np.inf, -np.inf
    for mesh in meshes:
        if zmin > mesh.bounds[0, 2]:
            zmin = mesh.bounds[0, 2]
        if zmax < mesh.bounds[1, 2]:
            zmax = mesh.bounds[1, 2]

    return zmin, zmax


def get_meshes_bounds_points(meshes):
    points = []
    for mesh in meshes:
        bound = mesh.bounding_box.vertices
        points.append(bound)

    points = np.concatenate(points, axis=0)

    return points


def get_vis_meshs(meshes: list, visual_geom: trimesh.Trimesh, bound_points):
    is_visual = visual_geom.ray.contains_points(bound_points)
    count = len(meshes)
    is_visual = np.reshape(is_visual, (count, -1))
    is_visual = np.any(is_visual, axis=1)
    nmesh = []
    for i in range(count):
        if is_visual[i]:
            nmesh.append(meshes[i])

    return nmesh


def get_corner_rays(width, height, yfov, matrix=None):
    if matrix is None:
        matrix = np.eye(4)
    xx, yy = np.meshgrid([0, width], [0, height])
    z = -height / 2 / np.tan(yfov / 2)
    target = np.ones((4, 3))
    target[:, 0] = xx.flatten() - width / 2
    target[:, 1] = yy.flatten() - height / 2
    target[:, 2] = z
    origin = matrix[:3, 3]
    target = np.dot(matrix[:3, :3], target.T)
    origin = origin.reshape((1, 3)).repeat(4, axis=0)
    target = target.T
    # target = trimesh.transform_points(target, np.linalg.inv(matrix))
    return origin, target


def get_perspective_visual_geom(
    width, height, yfov, zmin, matrix=None, strict_front=True
):
    origin, target = get_corner_rays(width, height, yfov, matrix=matrix)

    tmin = (zmin - origin[:, 2]) / target[:, 2]
    tmin = tmin[:, None].repeat(3, axis=1)
    if strict_front and np.any(tmin < 0):
        return None
    interpoints = origin + tmin * target

    vertexs = [origin[0, :].reshape(-1, 3), interpoints]
    vertexs = np.concatenate(vertexs, axis=0)
    faces = [[0, 1, 2], [0, 2, 4], [0, 3, 1], [0, 4, 3], [1, 3, 2], [2, 3, 4]]
    mesh = trimesh.Trimesh(vertices=vertexs, faces=faces, vertex_colors=(1.0, 0.0, 0.0))
    mesh.fix_normals()
    return mesh


def get_orthographic_visual_geom(xmag, ymag, znear, zfar, matrix=None):
    bbox = creation.box(
        extents=(xmag * 2, ymag * 2, (zfar - znear)), transform=np.eye(4)
    )
    ztrans = np.array([0, 0, -(zfar - znear) / 2 - znear])
    bbox.apply_translation(ztrans)
    bbox.apply_transform(matrix)
    return bbox


def depth_to_dsm(depth, camera_matrix, proj_matrix, xyz_or_depth="xyz"):
    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xx.flatten() - width / 2.0
    yy = height / 2.0 - yy.flatten()
    xyz = np.zeros((width * height, 3))
    xyz[:, 0] = xx / (width / 2)
    xyz[:, 1] = yy / (height / 2)
    xyz[:, 2] = depth.flatten() * 2 - 1
    nodata = (depth == 1).flatten()
    xyz[nodata, :] = np.nan
    stack = np.column_stack([xyz, np.ones(xyz.shape[0])])
    # stack = -stack * z[:, None]
    stack = np.linalg.inv(proj_matrix) @ stack.T
    xyz = stack.T / stack.T[:, 3:]

    xyz = ((camera_matrix) @ xyz.T).T

    if xyz_or_depth == "xyz":
        return xyz
    else:
        return xyz[:, 2].reshape(height, width)


def rcd_to_xyz(xy, depth, camera):
    if camera["type"] == "perspective":
        camera_instance = pyrender.PerspectiveCamera(
            camera["yfov"], aspectRatio=camera["aspec"]
        )
    elif camera["type"] == "orthographic":
        camera_instance = pyrender.OrthographicCamera(
            xmag=camera["xmag"], ymag=camera["ymag"], znear=0.01, zfar=camera["zfar"]
        )
    if camera["postype"] == "matrix":
        camera_matrix = np.array(camera["matrix"])
    elif camera["postype"] == "angles":
        camera_matrix = get_matrix(
            camera["yaw"], camera["pitch"], camera["roll"], camera["pos"]
        )
    proj_matrix = camera_instance.get_projection_matrix()
    height, width = depth.shape
    # xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xy[:, 0] - width / 2.0
    yy = height / 2.0 - xy[:, 1]
    xyz = np.zeros((xy.shape[0], 3))
    xyz[:, 0] = xx / (width / 2)
    xyz[:, 1] = yy / (height / 2)
    d = depth[
        np.clip(xy[:, 1], 0, height - 1).astype(int),
        np.clip(xy[:, 0], 0, width - 1).astype(int),
    ].flatten()
    xyz[:, 2] = d * 2 - 1.0
    valid_idx = d < 1
    if not np.any(valid_idx):
        return np.zeros((0, 3))

    # d = d[valid_idx]
    # xyz = xyz[valid_idx, :]
    stack = np.column_stack([xyz, np.ones(xyz.shape[0])])
    stack = np.linalg.inv(proj_matrix) @ stack.T
    xyz = stack.T / stack.T[:, 3:]

    xyz = ((camera_matrix) @ xyz.T).T
    xyz = xyz[:, :3]
    xyz[~valid_idx, :] = np.nan
    # xyz = trimesh.transform_points(xyz, camera_matrix)

    return xyz


def render_meshes(
    meshes,
    cameras,
    out_dir=None,
    save_depth=False,
    save_dsm=False,
    save_pose=True,
    bgcolor=(0, 0, 1.0, 0.0),
    lightcolor=(1.0, 1.0, 1.0),
    skip_exist=False,
):
    light = pyrender.DirectionalLight(color=lightcolor, intensity=5.0)
    zmin, zmax = compute_zmin_zmax(meshes)
    bound_points = get_meshes_bounds_points(meshes)
    for i, camera in tqdm.tqdm(enumerate(cameras), desc="Start rendering"):
        if out_dir is not None:
            image_file = os.path.join(out_dir, "%d_%s.jpg" % (i, camera["fname"]))
            if skip_exist and os.path.exists(image_file):
                continue

        if camera["type"] == "perspective":
            camera_instance = pyrender.PerspectiveCamera(
                camera["yfov"], aspectRatio=camera["aspec"]
            )
        elif camera["type"] == "orthographic":
            camera_instance = pyrender.OrthographicCamera(
                xmag=camera["xmag"],
                ymag=camera["ymag"],
                znear=0.01,
                zfar=zmax - zmin + 10,
            )
        if camera["postype"] == "matrix":
            camera_pose = camera["matrix"]
        elif camera["postype"] == "angles":
            camera_pose = get_matrix(
                camera["yaw"], camera["pitch"], camera["roll"], camera["pos"]
            )
        else:
            raise NotImplementedError(f"Camera postype {camera['postype']}")
        if camera["type"] == "perspective":
            # camera_pose =
            visual_geom = get_perspective_visual_geom(
                camera["width"], camera["height"], camera["yfov"], zmin, camera_pose
            )
        elif camera["type"] == "orthographic":
            visual_geom = get_orthographic_visual_geom(
                camera["xmag"], camera["ymag"], 0.01, zmax - zmin + 10, camera_pose
            )
        # else:

        dsm = None
        if visual_geom is None:
            color = np.zeros((camera["height"], camera["width"], 3)).astype(np.uint8)
            depth = np.zeros((camera["height"], camera["width"]))
            if out_dir is None:
                yield None
        else:
            visual_meshes = get_vis_meshs(meshes, visual_geom, bound_points)
            scene = pyrender.Scene.from_trimesh_scene(
                trimesh.Scene(visual_meshes), bg_color=bgcolor
            )
            scene.add(light)
            scene.add(camera_instance, pose=camera_pose)
            render = pyrender.OffscreenRenderer(camera["width"], camera["height"])
            color, depth = render.render(scene)
            render.delete()
            scene.clear()
            if save_dsm:
                dsm = depth_to_dsm(
                    depth, camera_pose, camera_instance.get_projection_matrix()
                )
            del scene
            del render
            del visual_geom
        ret = []
        if save_depth:
            if out_dir is not None:
                depth_file = os.path.join(
                    out_dir, "%d_%s_depth.tif" % (i, camera["fname"])
                )
                Image.fromarray(depth).save(depth_file)
            else:
                ret.append(depth)
        if save_dsm:
            if dsm is None:
                dsm = np.zeros_like(depth)
            if out_dir is not None:
                dsm_file = os.path.join(
                    out_dir, "%d_%s_depth.tif" % (i, camera["fname"])
                )
                Image.fromarray(dsm).save(dsm_file)
            else:
                ret.append(dsm)
        if save_pose:
            if out_dir is not None:
                pose_file = os.path.join(
                    out_dir, "%d_%s_pose.json" % (i, camera["fname"])
                )
                with open(pose_file, "w") as f:
                    f.write(json.dumps(camera_pose))
            else:
                ret.append(camera_pose)
        if out_dir is not None:
            image_file = os.path.join(out_dir, "%d_%s.tif" % (i, camera["fname"]))
            Image.fromarray(color).save(image_file)
        else:
            yield (color, *ret)


def orth_depth_to_dsm(
    depth, camera_matrix, proj_matrix, nodata=np.nan, znear=0.01, zfar=50
):
    # z = depth * proj_matrix[2,2] + proj_matrix[2,3]
    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xx.flatten() - width / 2.0
    yy = height / 2.0 - yy.flatten()
    xyz = np.zeros((width * height, 3))
    xyz[:, 0] = xx / (width / 2)
    xyz[:, 1] = yy / (height / 2)
    # xyz[:, 2] = -depth.flatten()
    # n, f = znear, zfar
    # d = depth.flatten()
    xyz[:, 2] = depth.flatten() * 2 - 1.0
    stack = np.column_stack([xyz, np.ones(xyz.shape[0])])
    stack = np.linalg.inv(proj_matrix) @ stack.T
    xyz = stack.T[:, :3]
    xyz = trimesh.transform_points(xyz, camera_matrix)
    dsm = xyz[:, 2]
    dsm = dsm.reshape((height, width))
    dsm[depth == 1] = nodata
    return dsm


def generate_geos(pose, proj, width, height):
    xcenter = pose[0]
    ycenter = pose[1]
    ytop = ycenter + 1.0 / proj[0, 1]
    xtop = xcenter - 1.0 / proj[0, 0]
    xres = 1.0 / proj[0, 0] * 2 / width
    yres = 1.0 / proj[0, 1] * 2 / height

    return [xtop, xres, 0, ytop, 0, -yres]


def load_mesh(pth):
    if os.path.splitext(pth)[1][1:] in trimesh.available_formats():
        mesh = trimesh.load(pth)
    else:
        raise NotImplementedError("Unknown mesh format %s" % pth)
    # print('end %s'%pth)
    return mesh


def set_double_sided_material(scene):
    """
    遍历场景中的所有节点，将具有材质的网格的材质设置为双面的
    """
    for node in scene.nodes:
        # 检查节点是否包含网格
        if node.mesh is not None:
            # 遍历节点网格中的所有原语（Primitives），设置其材质为双面的
            for primitive in node.mesh.primitives:
                # 检查原语是否具有材质
                if primitive.material is not None:
                    # 设置为双面渲染
                    primitive.material.doubleSided = True


def render_one(
    camera,
    meshes,
    bgcolor=(0, 0, 1.0, 0.0),
    lightcolor=(1.0, 1.0, 1.0),
):
    if camera["type"] == "perspective":
        camera_instance = pyrender.PerspectiveCamera(
            camera["yfov"], aspectRatio=camera["aspec"]
        )
    elif camera["type"] == "orthographic":
        camera_instance = pyrender.OrthographicCamera(
            xmag=camera["xmag"], ymag=camera["ymag"], znear=0.01, zfar=camera["zfar"]
        )
    if camera["postype"] == "matrix":
        camera_pose = np.array(camera["matrix"])
    elif camera["postype"] == "angles":
        camera_pose = get_matrix(
            camera["yaw"], camera["pitch"], camera["roll"], camera["pos"]
        )
    if isinstance(meshes, list):
        meshes = [load_mesh(p) for p in meshes]
    elif isinstance(meshes, str):
        meshes = load_mesh(meshes)
    elif isinstance(meshes, (trimesh.Scene, trimesh.Trimesh)):
        meshes = meshes

    scene = pyrender.Scene.from_trimesh_scene(trimesh.Scene(meshes), bg_color=bgcolor)
    # light = pyrender.DirectionalLight(color=lightcolor, intensity=7.0)
    # scene.add(light, pose=camera_pose)
    set_double_sided_material(scene)
    scene.add(camera_instance, pose=camera_pose)

    render = OffRender(camera["width"], camera["height"])
    color, depth = render.render(scene, flags=pyrender.RenderFlags.FLAT)
    dsm = depth_to_dsm(
        depth,
        camera_pose,
        camera_instance.get_projection_matrix(),
        xyz_or_depth="depth",
    )
    render.delete()
    scene.clear()

    return color, depth, dsm


def mesh_info(paths, out):
    bounds = []
    for p in tqdm.tqdm(paths, desc="load mesh bbox"):
        mesh = trimesh.load(p)
        b = mesh.bounds
        bounds.append(dict(bounds=b.tolist(), path=p))
    with open(out, "w", encoding="utf8") as f:
        json.dump(bounds, f)


def find_bounding_box(bounding_boxes, point):
    """
    Find which bounding box the point is in.
    """
    # # 将包围盒列表转换为NumPy数组，以便进行快速计算
    boxes = bounding_boxes

    # 计算点是否在每个包围盒内
    # 这里我们只检查了x和y坐标，假设z坐标在你的场景中不需要
    in_box = (
        (boxes[:, 0, 0] <= point[0])
        & (point[0] <= boxes[:, 1, 0])
        & (boxes[:, 0, 1] <= point[1])
        & (point[1] <= boxes[:, 1, 1])
    )

    # 检查是否有至少一个包围盒包含点
    if np.any(in_box):
        # 返回第一个找到的包围盒的索引
        return np.where(in_box)[0][0]
    else:
        # 如果点不在任何包围盒内，返回-1
        return -1


def is_box_within_area(rect1, rect2):
    # 将包围盒的坐标转化为numpy数组，便于进行计算
    xmin1, ymin1 = rect1[0, 0], rect1[0, 1]
    xmax1, ymax1 = rect1[1, 0], rect1[1, 1]
    xmin2, ymin2 = rect2[0, 0], rect2[0, 1]
    xmax2, ymax2 = rect2[1, 0], rect2[1, 1]

    # [[xmin1, ymin1], [xmax1, ymax1]] = rect1
    # [[xmin2, ymin2], [xmax2, ymax2]] = rect2

    # 检查不相交的情况
    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return False
    else:
        # 如果以上情况都不成立，则相交
        return True


def render_prepare(info, gap=5, zgap=5, buffer=20, out=None):
    with open(info, "r", encoding="utf8") as f:
        infodict = json.load(f)
    paths = [p["path"] for p in infodict]
    bounds = [np.array(p["bounds"]) for p in infodict]
    boxes = [creation.box(bounds=b) for b in bounds]
    for b in boxes:
        b.visual.face_colors[:, 2] = 0
        b.visual.face_colors[:, 3] = 100
    bounds = np.array(bounds)

    x90poses = [get_matrix(0, yr, -90) for yr in [0, 90, 180, 270]]
    x75poses = [m @ get_matrix(0, 0, 15) for m in x90poses]
    # x105poses = [m @ get_matrix(0, 0, -15) for m in x90poses]

    poses = [
        # *x90poses,
        # *x105poses,
        *x75poses,
    ]

    # scene = trimesh.Scene([*boxes])
    # center = creation.box(extents=(0.5, 0.5, 0.5))
    # x, y, z = scene.bounds.mean(axis=0)
    # # center.apply_translation((x, y, z))
    # # center.visual.vertex_colors[:, 0] = 255
    # # center.visual.vertex_colors[:, 1] = 0
    # # center.visual.vertex_colors[:, 2] = 0

    # x90poses = [get_matrix(0, yr, -90) for yr in [0, 90, 180, 270]]
    # x75poses = [m @ get_matrix(0, 0, 15) for m in x90poses]
    # x105poses = [m @ get_matrix(0, 0, -15) for m in x90poses]

    # camera_pose = get_matrix(pos=np.array([x, y, z])) @ x105poses[0]

    # # camera_pose[-1, -1] = 1
    # # pose_2 =
    # visual_geom = get_orthographic_visual_geom(20, 20, 0.01, 100, camera_pose)
    # scene.add_geometry(visual_geom)
    # # scene.add_geometry(center)

    # axis = creation.axis(axis_length=10, transform=camera_pose, axis_radius=1)
    # scene.add_geometry(axis)
    # scene.show()
    # return

    # print(bounds.shape)
    bounds_p = np.reshape(bounds, (-1, 3))
    min_b = bounds_p.min(axis=0)
    max_b = bounds_p.max(axis=0)

    xrange = np.arange(min_b[0] - buffer, max_b[0] + buffer, gap * 2)
    yrange = np.arange(min_b[1] - buffer, max_b[1] + buffer, gap * 2)
    zrange = np.zeros(1)

    xx, yy, zz = np.meshgrid(xrange, yrange, zrange)
    xx, yy, zz = (
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
    )

    zfar = 200
    count = len(boxes)
    result = []
    for x, y, z in tqdm.tqdm(zip(xx, yy, zz), total=len(xx), desc="generate pose"):
        for p in poses:
            idx = find_bounding_box(bounds, (x, y))
            if idx == -1:
                continue
            b = bounds[idx, :, :]
            z = b[1, 2] + 2
            camera_pose = get_matrix(pos=np.array([x, y, z])) @ p
            camera = dict(
                type="orthographic",
                postype="matrix",
                matrix=camera_pose.tolist(),
                zfar=zfar,
                xmag=gap * 1.5,
                ymag=gap * 1.5,
                width=int(gap * 1.5 * 2 / 0.05),
                height=int(gap * 1.5 * 2 / 0.05),
            )
            visual_geom = get_orthographic_visual_geom(
                camera["xmag"], camera["ymag"], 0.01, zfar, camera_pose
            )
            # print(x, y, z, yr, xr)
            vb = visual_geom.bounds

            nmesh = []
            for i in range(count):
                if is_box_within_area(bounds[i, :, :], vb):
                    nmesh.append(paths[i])
                    # print(paths[i])
            if len(nmesh) == 0:
                continue
            nmesh = paths
            # vertex_colors = np.random.random((visual_geom.vertices.shape[0], 4))
            # vertex_colors[:, 3] = 0  # 设置alpha通道为完全不透明
            # center = creation.axis(axis_length=10, axis_radius=2, transform=camera_pose)
            # visual_geom.visual.face_colors[:, 3] = 50
            # # vbox = creation.box(bounds=vb)
            # scene = trimesh.Scene([visual_geom, center, *boxes])
            # scene.show()

            result.append(dict(camera=camera, meshes=nmesh))
    print(f"{len(result)} valid views")
    if out is not None:
        with open(out, "w") as f:
            json.dump(result, f)
    return result


def render_loop(
    cfg, bgcolor=(0, 0, 1.0, 0.0), lightcolor=(1.0, 1.0, 1.0), out=None, skip_exist=True
):
    with open(cfg, "r") as f:
        cfgdict = json.load(f)
    os.makedirs(out, exist_ok=True)
    for i, c in tqdm.tqdm(enumerate(cfgdict), total=len(cfgdict), desc="Start render"):
        if skip_exist and os.path.exists(os.path.join(out, "%08d.jpg" % i)):
            continue
        camera = c["camera"]
        meshes = c["meshes"]
        # print(camera)
        color, depth, dsm = render_one(camera, meshes, bgcolor, lightcolor)
        Image.fromarray(color.astype(np.uint8)).save(os.path.join(out, "%08d.jpg" % i))
        Image.fromarray(depth).save(os.path.join(out, "%08d_depth.tif" % i))
        Image.fromarray(dsm).save(os.path.join(out, "%08d_dsm.tif" % i))


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="demo/Data",
        required=False,
        help="input CC Obj Data path, like 'xx/xx/Data'",
    )
    parser.add_argument("-o", "--output", type=str, required=False)

    args = parser.parse_args()

    mesh_info_json = os.path.join(os.path.dirname(args.input), "mesh_info.json")
    render_info_json = os.path.join(os.path.dirname(args.input), "render_info.json")
    render_output = os.path.join(os.path.dirname(args.input), "render_output")
    os.makedirs(render_output, exist_ok=True)
    logging.info("mesh_info_json:{}".format(mesh_info_json))
    if not os.path.exists(mesh_info_json):
        mesh_list = load_ccobjs_to_trimesh(args.input)
        mesh_info(mesh_list, mesh_info_json)
    logging.info("render_info_json:{}".format(render_info_json))
    render_prepare(mesh_info_json, gap=10, zgap=10, buffer=5, out=render_info_json)
    logging.info("render_output:{}".format(render_output))
    render_loop(render_info_json, out=render_output, skip_exist=False)
