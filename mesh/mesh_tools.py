from osgeo import gdal, osr
import pyproj

import trimesh
from trimesh import creation
import trimesh.exchange.gltf as gltf
import trimesh.exchange.ply as ply
import trimesh.exchange.export as export
from trimesh.visual import TextureVisuals
from trimesh.visual.texture import SimpleMaterial
import os
import numpy as np
import json
from py3dtiles.tileset.utils import TileContentReader
from py3dtiles.tileset.content import B3dm
from pathlib import Path
from io import BytesIO
from collections import deque
import tqdm
import xmltodict
import pyrender
import PIL.Image as Image
from pygltflib import GLTF2, Scene


def load_ccobjs_to_trimesh(root_data_path, sub=None, ret_files=False):
    paths = os.listdir(root_data_path)
    results = []
    for pth in paths:
        f = os.path.join(root_data_path, pth, f"{pth}.obj")
        if os.path.exists(f):
            results.append(f)
    if sub is not None:
        results = results[sub]
    meshes = []
    files = []
    for p in tqdm.tqdm(results, desc="Read OBJ files"):
        mesh: trimesh.Trimesh = trimesh.load(p)
        if isinstance(mesh, trimesh.Scene):
            for i, s in enumerate(mesh.geometry):
                meshes.append(mesh.geometry[s])
                files.append(
                    os.path.join(
                        os.path.dirname(p),
                        f"{os.path.basename(p)}.{i}",
                    )
                )
        elif isinstance(mesh, trimesh.Trimesh):
            # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            meshes.append(mesh)
            files.append(p)
        else:
            raise Exception("Not supported format", mesh)
    if ret_files:
        return meshes, files
    return meshes


def parser_ccxml(ccxml_path):
    """
    TODO: metadata 不在使用,去除相关代码;
    TODO: 视场角仍未能通过参数确定,需要进一步设置
    """
    # SRS = ''
    # ORIGIN = []
    # with open(metadata_path, 'r') as f:
    #     for line in f.readlines():
    #         if line.strip().startswith('<SRS>'):
    #             SRS = CRS.from_string(line.strip()[len('<SRS>'):-len('</SRS>')])
    #         if line.strip().startswith('<SRSOrigin>'):
    #             origin_str = line.strip()[len('<SRSOrigin>'):-len('</SRSOrigin>')]
    #             origin_ls = origin_str.split(',')
    #             ORIGIN.extend((float(origin_ls[0]), float(origin_ls[1]), float(origin_ls[2])))

    # ORIGIN = np.array(ORIGIN).reshape(3)
    # BLOKC_SRSS = dict()

    # ZINV = np.eye(4)
    # ZINV[2,2] = -1
    # R90 = get_matrix(0, 0, -90, np.zeros(3)) # look at y, z up x right

    # return

    def read_to_line(f, endline=None):
        if endline is None:
            return f.readlines()
        else:
            res = []
            while True:
                line = f.readline()
                if line.strip() == endline:
                    return res + [line]
                else:
                    res.append(line)

    with open(ccxml_path, "r") as f:
        line = f.readline().strip()
        # first_line = line
        # current_block_srs = None
        current_image_size = [0, 0]
        current_aspec = 1
        current_distortion = [0.0, 0.0, 0.0, 0.0, 0]
        current_focallength = 0
        while line != "</BlocksExchange>":
            # if line.startswith('<SpatialReferenceSystems>'):
            #     # logging.info('process srs')

            #     res = read_to_line(f, '</SpatialReferenceSystems>')
            #     line = f.readline().strip()
            #     continue
            #     res.insert(0, line)
            #     xmlstr = 	'\n'.join(res)

            #     srss = xmltodict.parse(xmlstr)
            #     for srs in srss['SpatialReferenceSystems']['SRS']:
            #         BLOKC_SRSS[srs['Id']] = CRS.from_string(srs['Definition'])
            #     # logging.info('fine %d srs', len(BLOKC_SRSS))
            # elif line.startswith('<SRSId>'):
            #     line = f.readline().strip()
            #     continue
            # current_block_srs = BLOKC_SRSS[line.split('>')[1].split('<')[0]]
            if line.startswith("<ImageDimensions>"):
                wline = f.readline().strip().split(">")[1].split("<")[0]
                hline = f.readline().strip().split(">")[1].split("<")[0]
                current_image_size[0] = int(wline)
                current_image_size[1] = int(hline)
            # elif line.startswith('<AspectRatio>'):
            #     current_aspec = float(f.readline().strip().split('>')[1].split('<')[0])
            # elif line.startswith('<FocalLength>'):
            #     current_focallength = 	float(line.split('>')[1].split('<')[0])
            elif line.startswith("<Photo>"):
                res = read_to_line(f, "</Photo>")
                res.insert(0, line)
                xmlstr = "\n".join(res)
                image_info = xmltodict.parse(xmlstr)["Photo"]
                image_pose = image_info["Pose"]
                yaw = float(image_pose["Rotation"]["Yaw"])
                pitch = float(image_pose["Rotation"]["Pitch"])
                roll = float(image_pose["Rotation"]["Roll"])

                R90 = get_matrix(0, -90, 0, np.zeros(3))  # look at y, z up x right
                YAW = get_matrix(0, 0, yaw, np.zeros(3))  # cz
                PITCH = get_matrix(0, -pitch, 0, np.zeros(3))  # ccx
                ROLL = get_matrix(-roll, 0, 0, np.zeros(3))  # cy
                # T = get_matrix(0, 0, 0, np.array([camera_pos[-1]])) # T

                pymatrix = ROLL @ PITCH @ YAW @ R90

                # yaw = 180 + yaw
                # if yaw < 0:
                #     yaw = yaw + 360
                # roll, pitch = -pitch, roll
                # rot_matrix = [[ float(image_pose['Rotation']['M_00']), float(image_pose['Rotation']['M_01']), float(image_pose['Rotation']['M_02'])],
                #              [float(image_pose['Rotation']['M_10']), float(image_pose['Rotation']['M_11']), float(image_pose['Rotation']['M_12'])],
                #              [float(image_pose['Rotation']['M_20']), float(image_pose['Rotation']['M_21']), float(image_pose['Rotation']['M_22'])]]
                position = [
                    float(image_pose["Center"]["x"]),
                    float(image_pose["Center"]["y"]),
                    float(image_pose["Center"]["z"]),
                ]
                # position = pyproj.transform(current_block_srs, SRS, *position)
                # position = Transformer.from_crs(current_block_srs, SRS).transform(*position)
                # position = np.array(position).reshape(3)
                # relposition = position[[1,0,2]]   - ORIGIN
                relposition = position
                pymatrix[:3, 3] = relposition[:]
                img_name = os.path.basename(image_info["ImagePath"])
                img_name = os.path.splitext(img_name)[0]
                camrea = dict(
                    yfov=40 / 180 * np.pi,
                    aspec=current_image_size[0] / current_image_size[1],
                    width=current_image_size[0],
                    height=current_image_size[1],
                    postype="matrix",
                    matrix=pymatrix,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    pos=relposition,
                    fname=img_name,
                )
                # print(camrea)
                yield camrea

            line = f.readline().strip()


def load_b3dm_matrix(pth):
    tile_content = TileContentReader.read_file(Path(pth)).to_array()
    tile = B3dm.from_array(tile_content)
    gltfb = tile.body.glTF.to_array()
    gltfio = BytesIO(gltfb.tobytes())
    loaddict = gltf.load_glb(gltfio)
    graph: deque = loaddict["graph"]
    matrix = graph.pop()["matrix"]
    return matrix


def load_b3dm_to_trimesh(pth, transform=None):
    tile_content = TileContentReader.read_file(Path(pth)).to_array()
    tile = B3dm.from_array(tile_content)
    gltfb = tile.body.glTF.to_array()
    gltfio = BytesIO(gltfb.tobytes())
    # glb = GLTF2.load_from_bytes(gltfb.tobytes())
    # t = 0
    loaddict = gltf.load_glb(gltfio, merge_primitives=True)
    meshdict = loaddict["geometry"]
    mesh = trimesh.Trimesh(**meshdict["GLTF"])
    graph: deque = loaddict["graph"]
    matrix = graph.pop()["matrix"]
    yup_to_zup = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(4, 4)
    mesh.apply_transform(matrix)
    mesh.apply_transform(yup_to_zup)
    if transform is not None:
        mesh.apply_transform(transform)

    return mesh


def load_tileset_transform(pth):
    with open(pth, "r") as f:
        data = json.load(f)
    data = data["root"]
    if "transform" in data:
        return np.array(data["transform"]).reshape(4, 4).T
    else:
        return np.eye(4, 4)


def load_tileset_gltf(pth):
    with open(pth, "r") as f:
        data = json.load


def load_sub_tileset(json_path, apply_transform=False):
    with open(json_path, "r") as f:
        data = json.load(f)
    data = data["root"]
    leaf_nodes = []
    nodes = [data]
    if "transform" in data and apply_transform:
        transform = np.array(data["transform"])
        transform = np.reshape(transform, (4, 4))
    else:
        transform = np.eye(4)
    # 深度优先遍历，取所有叶子节点
    while len(nodes) > 0:
        item = nodes.pop(0)
        # matrix = load_b3dm_matrix(os.path.join(os.path.dirname(json_path), item['content']['uri']))
        # print(matrix)
        if "children" in item:
            for ix in range(len(item["children"])):
                nodes.append(item["children"][ix])

        else:
            leaf_nodes.append(item)

    meshes = []

    for leaf in leaf_nodes:
        content = leaf["content"]["uri"]
        path = os.path.join(os.path.dirname(json_path), content)
        if path.endswith("json"):
            t = load_sub_tileset(path, apply_transform=True)
            for m in t:
                m.apply_transform(transform)
            meshes.extend(t)
        elif path.endswith("b3dm"):
            m = load_b3dm_to_trimesh(path)
            box = np.array(leaf["boundingVolume"]["box"]).reshape(4, 3)
            center = box[0, :]
            m.apply_translation(center)
            m.apply_transform(transform)

            meshes.append(m)
        elif path.endswith("i3dm"):
            pass

        elif path.endswith("pnts"):
            pass

        elif path.endswith("cmpt"):
            pass

    return meshes


def load_tileset_json_to_trimesh(json_path, apply_transform=False, show_bar=False):
    with open(json_path, "r") as f:
        data = json.load(f)
    data = data["root"]
    leaf_nodes = []
    nodes = [data]
    if "transform" in data and apply_transform:
        transform = np.array(data["transform"])
        transform = np.reshape(transform, (4, 4))
    else:
        transform = np.eye(4)
    # 深度优先遍历，取所有叶子节点
    while len(nodes) > 0:
        item = nodes.pop(0)
        # matrix = load_b3dm_matrix(os.path.join(os.path.dirname(json_path), item['content']['uri']))
        # print(matrix)
        if "children" in item:
            for ix in range(len(item["children"])):
                nodes.append(item["children"][ix])

        else:
            leaf_nodes.append(item)

    meshes = []
    if show_bar:
        leaf_nodes = tqdm.tqdm(leaf_nodes, desc="load content")
    for leaf in leaf_nodes:
        content = leaf["content"]["uri"]
        path = os.path.join(os.path.dirname(json_path), content)
        if path.endswith("json"):
            t = load_tileset_json_to_trimesh(path, apply_transform=True)
            for m in t:
                m.apply_transform(transform)
            t = trimesh.Scene(t)
            meshes.append(t)
        elif path.endswith("b3dm"):
            m = load_b3dm_to_trimesh(path)
            box = np.array(leaf["boundingVolume"]["box"]).reshape(4, 3)
            center = box[0, :]
            m.apply_translation(center)
            m.apply_transform(transform)

            meshes.append(trimesh.Scene(m))
        elif path.endswith("i3dm"):
            pass

        elif path.endswith("pnts"):
            pass

        elif path.endswith("cmpt"):
            pass

    return meshes


def load_cctileset_json(
    json_path, apply_transform=False, show_bar=False, ret_iter=False
):
    with open(json_path, "r") as f:
        data = json.load(f)
    data = data["root"]
    leaf_nodes = []
    nodes = [data]
    if "transform" in data and apply_transform:
        transform = np.array(data["transform"])
        transform = np.reshape(transform, (4, 4))
    else:
        transform = np.eye(4)
    # 深度优先遍历，取所有叶子节点
    while len(nodes) > 0:
        item = nodes.pop(0)
        # matrix = load_b3dm_matrix(os.path.join(os.path.dirname(json_path), item['content']['uri']))
        # print(matrix)
        if "children" in item:
            for ix in range(len(item["children"])):
                nodes.append(item["children"][ix])

        else:
            leaf_nodes.append(item)

    meshes = []
    names = []
    if show_bar:
        leaf_nodes = tqdm.tqdm(leaf_nodes, desc="load content")
    for leaf in leaf_nodes:
        content = leaf["content"]["uri"]
        path = os.path.join(os.path.dirname(json_path), content)
        if path.endswith("json"):
            t = load_tileset_json_to_trimesh(path, apply_transform=True)
            for m in t:
                m.apply_transform(transform)
            t = trimesh.Scene(t)
            if ret_iter:
                yield t, os.path.splitext(os.path.basename(path))[0]
            else:
                names.append(os.path.splitext(os.path.basename(path))[0])
                meshes.append(t)
        elif path.endswith("b3dm"):
            m = load_b3dm_to_trimesh(path)
            m = trimesh.Scene(m)
            box = np.array(leaf["boundingVolume"]["box"]).reshape(4, 3)
            center = box[0, :]
            m.apply_translation(center)
            m.apply_transform(transform)
            if ret_iter:
                yield t, os.path.splitext(os.path.basename(path))[0]
            else:
                names.append(os.path.splitext(os.path.basename(path))[0])
                meshes.append(t)
            # names.append( os.path.splitext(os.path.basename(path))[0])
            # meshes.append(m)
    if not ret_iter:
        return meshes, names


def get_matrix(zrot=0, yrot=0, xrot=0, pos=np.zeros(3)):
    """
    xyzrot: 顺时针旋转角度,单位为度
    return: np.array with shape 4x4
    """
    zrot = np.deg2rad(zrot)
    yrot = np.deg2rad(yrot)
    xrot = np.deg2rad(xrot)

    # 计算绕U轴旋转的矩阵
    rotation_u = np.array(
        [[np.cos(zrot), np.sin(zrot), 0], [-np.sin(zrot), np.cos(zrot), 0], [0, 0, 1]]
    )
    # print(rotation_u.shape)

    # 计算绕N轴旋转的矩阵
    rotation_n = np.array(
        [[1, 0, 0], [0, np.cos(xrot), np.sin(xrot)], [0, -np.sin(xrot), np.cos(xrot)]]
    )
    # 计算绕E轴旋转的矩阵
    rotation_e = np.array(
        [[np.cos(yrot), 0, -np.sin(yrot)], [0, 1, 0], [np.sin(yrot), 0, np.cos(yrot)]]
    )

    # 计算旋转后的坐标
    # rotated = np.dot(rotation_n, np.dot(rotation_u, np.vstack((e, n, u))))
    rot_mat = np.dot(np.dot(rotation_e, rotation_n), rotation_u)
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = rot_mat[:, :]
    matrix[:3, 3] = pos[:]
    matrix[-1, -1] = 1
    # matrix[-1,-1] = 1
    return matrix


def sample_pose_from_bboxs(meshes: list, gap=5, buffer=20):
    box_box = [b.bounds for b in meshes]
    box_box = np.array(box_box)  # N 2 3
    box_box = box_box.reshape((-1, 3))
    box_min = box_box.min(axis=0)
    box_max = box_box.max(axis=0)
    bounds = np.stack([box_min, box_max], axis=0)

    # large_box = creation.box(bounds=bounds)

    x_range = np.arange(bounds[0, 0] - buffer, bounds[1, 0] + buffer, gap)
    y_range = np.arange(bounds[0, 1] - buffer, bounds[1, 1] + buffer, gap)
    z_range = np.arange(bounds[0, 2] + buffer, bounds[1, 2] + buffer, gap)
    yrot = np.arange(-10, 10, 2)
    xrot = np.arange(-10, 10, 2)

    xx, yy, zz, yrot, xrot = np.meshgrid(x_range, y_range, z_range, yrot, xrot)
    xx, yy, zz, yrot, xrot = (
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
        yrot.flatten(),
        xrot.flatten(),
    )

    return


def batch_export_ply(meshes, outs):
    for mesh, o in zip(meshes, outs):
        # mesh = trimesh.Trimesh()
        with open(o, "rb") as f:
            f.write(ply.export_ply(mesh))


def batch_export(meshes, outs, *args, **kwargs):
    for mesh, o in zip(meshes, outs):
        # mesh = trimesh.Trimesh()
        export.export_mesh(mesh, o, *args, **kwargs)


def batch_export_bbox(meshes, outdir):
    bboxs = [creation.box(bounds=m.bounds) for m in meshes]
    batch_export_ply(
        bboxs, [os.path.join(outdir, "%d_box.ply" % i) for i in range(len(bboxs))]
    )


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
        extents=(xmag * 2, ymag * 2, (zfar - znear) / 2), transform=np.eye(4)
    )
    ztrans = np.array([0, 0, -(zfar - znear) / 2 - znear])
    bbox.apply_translation(ztrans)
    bbox.apply_transform(matrix)
    return bbox


def depth_to_dsm(depth, camera_matrix, proj_matrix):
    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xx.flatten() - width / 2.0
    yy = height / 2.0 - yy.flatten()
    xyz = np.zeros((width * height, 3))
    xyz[:, 0] = xx / (width / 2)
    xyz[:, 1] = yy / (height / 2)
    z = -(depth.flatten() - proj_matrix[2, 3])
    k = -proj_matrix[2, 3] / z + 1
    xyz[:, 2] = k

    stack = np.column_stack([xyz, np.ones(xyz.shape[0])])
    stack = -stack * z[:, None]
    stack = np.linalg.inv(proj_matrix) @ stack.T
    xyz = stack.T[:, :3]
    xyz = trimesh.transform_points(xyz, camera_matrix)

    dsm = xyz[:, 2]
    dsm = dsm.reshape((height, width))

    return dsm


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


def orth_depth_to_dsm(depth, camera_matrix, proj_matrix, nodata=np.nan):
    # z = depth * proj_matrix[2,2] + proj_matrix[2,3]
    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xx.flatten() - width / 2.0
    yy = height / 2.0 - yy.flatten()
    xyz = np.zeros((width * height, 3))
    xyz[:, 0] = xx / (width / 2)
    xyz[:, 1] = yy / (height / 2)
    # xyz[:, 2] = -depth.flatten()

    stack = np.column_stack([xyz, np.ones(xyz.shape[0])])
    stack = np.linalg.inv(proj_matrix) @ stack.T
    xyz = stack.T[:, :3]
    xyz[:, 2] = -depth.flatten()

    xyz = trimesh.transform_points(xyz, camera_matrix)
    dsm = xyz[:, 2]
    dsm = dsm.reshape((height, width))
    dsm[depth == 0] = nodata
    return dsm


def generate_geos(pose, proj, width, height):
    xcenter = pose[0]
    ycenter = pose[1]
    ytop = ycenter + 1.0 / proj[0, 1]
    xtop = xcenter - 1.0 / proj[0, 0]
    xres = 1.0 / proj[0, 0] * 2 / width
    yres = 1.0 / proj[0, 1] * 2 / height

    return [xtop, xres, 0, ytop, 0, -yres]


def dom_dsm(
    meshes,
    out_dir,
    res=0.5,
):
    pass


def generate_orth(bboxs, size=50):
    bounds = get_meshes_bounds_points(bboxs)
    min_b = bounds.min(axis=0)
    max_b = bounds.max(axis=0)

    xrange = np.arange(min_b[0], max_b[0], size)
    yrange = np.arange(min_b[1], max_b[1], size)

    z = max_b[2] + 10


import cachetools


# @cachetools.cached(cachetools.LRUCache(maxsize=1000))
def load_mesh(pth):
    # print('load %s'%pth)
    if pth.endswith(".json"):
        mesh = load_tileset_json_to_trimesh(pth)
        mesh = trimesh.Scene(mesh)
    elif pth.endswith(".b3dm"):
        mesh = load_b3dm_to_trimesh(pth)
    elif os.path.splitext(pth)[1] in trimesh.available_formats():
        mesh = trimesh.load(pth)
    else:
        raise NotImplementedError("Unknown mesh format %s" % pth)
    # print('end %s'%pth)
    return mesh


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

    meshes = [load_mesh(p) for p in meshes]
    scene = pyrender.Scene.from_trimesh_scene(trimesh.Scene(meshes), bg_color=bgcolor)
    light = pyrender.DirectionalLight(color=lightcolor, intensity=5.0)
    scene.add(light)
    scene.add(camera_instance, pose=camera_pose)

    render = pyrender.OffscreenRenderer(camera["width"], camera["height"])
    color, depth = render.render(scene)
    render.delete()
    scene.clear()

    if camera["type"] == "perspective":
        dsm = depth_to_dsm(depth, camera_pose, camera_instance.get_projection_matrix())
    elif camera["type"] == "orthographic":
        dsm = orth_depth_to_dsm(
            depth, camera_pose, camera_instance.get_projection_matrix()
        )

    return color, depth, dsm


def mesh_info(paths, out):
    bounds = []
    for p in tqdm.tqdm(paths, desc="load mesh bbox"):
        mesh = trimesh.load(p)
        b = mesh.bounds
        bounds.append(dict(bounds=b.tolist(), path=p))
    with open(out, "w") as f:
        json.dump(bounds, f)


def render_plane(
    meshes,
    plane: list,
    bgcolor=(0, 0, 1.0, 0.0),
    lightcolor=(1.0, 1.0, 1.0),
    reslution=0.1,
):
    y = np.array([0, 0, 1])
    z1 = np.cross(plane[0] - plane[1], plane[1] - plane[2])
    x1 = -np.cross(y, z1)

    z2 = -z1
    x2 = -np.cross(y, z2)
    points = np.array(plane)
    center = points.mean(axis=0)

    matrix1 = np.eye(4)
    matrix1[:3, 0] = x1
    matrix1[:3, 1] = y
    matrix1[:3, 2] = z1
    matrix1[:3, 3] = center + z1 * 0.01

    matrix2 = np.eye(4)
    matrix2[:3, 0] = x2
    matrix2[:3, 1] = y
    matrix2[:3, 2] = z2
    matrix2[:3, 3] = center + z2 * 0.01

    proj_points = trimesh.transform_points(points, matrix1)
    proj_points = proj_points[:, :2]
    bounds = [np.max(proj_points, axis=0), np.min(proj_points, axis=0)]
    bounds = np.array(bounds)
    xmag = bounds[0, 0] - bounds[1, 0]
    ymag = bounds[0, 1] - bounds[1, 1]
    xmag = xmag / 2
    ymag = ymag / 2

    meshes = [load_mesh(m) for m in meshes]
    tri_scene = trimesh.Scene(meshes)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=bgcolor)

    camrea = pyrender.OrthographicCamera(xmag, ymag, znear=0.1, zfar=100)
    node = scene.add(camrea, pose=matrix1)

    light = pyrender.DirectionalLight(color=lightcolor, intensity=5.0)
    scene.add(light)
    scene.add(camrea, pose=matrix1)

    width = int(xmag / reslution * 2)
    height = int(ymag / reslution * 2)

    render = pyrender.OffscreenRenderer(width, height)
    color1, depth1 = render.render(scene)
    # render.delete()
    # scene.clear()
    scene.remove_node(node)

    scene.add(camrea, pose=np.eye(4))
    color2, depth2 = render.render(scene)
    from shapely.geometry import Polygon

    geom = creation.extrude_polygon(
        Polygon(proj_points), 0.02, transform=np.linalg.inv(matrix1)
    )

    material = SimpleMaterial(image=Image.fromarray(np.vstack((color1, color2))))
    # TODO: check point rings rank
    uvs = []
    proj_points = proj_points - np.min(proj_points, axis=0, keepdims=True)
    for i in range(proj_points.shape[0]):
        p = proj_points[i, :]
        v = p[0] / xmag / 2
        u = (1 - p[1] / ymag / 2) / 2
        uvs.append((u, v))
    for i in range(proj_points.shape[0]):
        p = proj_points[i, :]
        v = p[0] / xmag / 2
        u = 1 - p[1] / ymag / 4

        uvs.append((u, v))

    uvs = np.array(uvs).reshape(-1, 2)

    visual = TextureVisuals(uv=uvs, material=material)
    geom.visual = visual

    return geom


def render_prepare(info, gap=5, zgap=5, buffer=20, out=None):
    with open(info, "r") as f:
        infodict = json.load(f)
    paths = [p["path"] for p in infodict]
    bounds = [np.array(p["bounds"]) for p in infodict]
    boxes = [creation.box(bounds=b) for b in bounds]
    bounds = np.array(bounds)
    bounds_p = np.reshape(bounds, (-1, 3))
    min_b = bounds_p.min(axis=0)
    max_b = bounds_p.max(axis=0)

    xrange = np.arange(min_b[0] - buffer, max_b[0] + buffer, gap)
    yrange = np.arange(min_b[1] - buffer, max_b[1] + buffer, gap)
    zrange = np.arange(max_b[2] - buffer, max_b[2] + buffer, zgap)
    yrot = np.arange(-10, 10, 2)
    xrot = np.arange(-10, 10, 2)

    xx, yy, zz, yrot, xrot = np.meshgrid(xrange, yrange, zrange, yrot, xrot)
    xx, yy, zz, yrot, xrot = (
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
        yrot.flatten(),
        xrot.flatten(),
    )

    zfar = max_b[2] - min_b[2] + 10
    count = len(boxes)
    result = []
    for x, y, z, yr, xr in tqdm.tqdm(zip(xx, yy, zz, yrot, xrot), desc="generate pose"):
        camera_pose = get_matrix(0, yr, xr, pos=np.array([x, y, z]))
        camera = dict(
            type="orthographic",
            postype="matrix",
            matrix=camera_pose.tolist(),
            zfar=zfar,
            xmag=gap * 1.5,
            ymag=gap * 1.5,
            width=512,
            height=512,
        )
        visual_geom = get_orthographic_visual_geom(
            camera["xmag"], camera["ymag"], 0.01, zfar, camera_pose
        )
        is_visual = visual_geom.ray.contains_points(bounds_p)
        is_visual = np.reshape(is_visual, (count, -1))
        is_visual = np.any(is_visual, axis=1)
        nmesh = []
        for i in range(count):
            if is_visual[i]:
                nmesh.append(paths[i])

        if len(nmesh) == 0:
            continue

        result.append(dict(camera=camera, meshes=nmesh))

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
    for i, c in tqdm.tqdm(enumerate(cfgdict), desc="Start render"):
        if skip_exist and os.path.exists(os.path.join(out, "%08d.jpg" % i)):
            continue
        camera = c["camera"]
        meshes = c["meshes"]
        color, depth, dsm = render_one(camera, meshes, bgcolor, lightcolor)
        Image.fromarray(color.astype(np.uint8)).save(os.path.join(out, "%08d.jpg" % i))
        Image.fromarray(depth).save(os.path.join(out, "%08d_depth.tif" % i))
        Image.fromarray(dsm).save(os.path.join(out, "%08d_dsm.tif" % i))


def orth_3dtile_with_coords(
    meshes,
    bgcolor=(0.0, 0.0, 0.0),
    lightcolor=(1.0, 1.0, 1.0),
    intensity=5,
    resolution=0.2,
    out="",
    is_xyz=True,
    transform=np.eye(4),
    epsg="4547",
):
    # print(meshes)

    meshes = [load_mesh(p) for p in meshes]
    tri_scene = trimesh.Scene(meshes)
    # tri_scene.show()
    scene = pyrender.Scene(ambient_light=lightcolor, bg_color=bgcolor)
    # export.export_mesh(tri_scene, out + '.obj', file_type='obj')
    for geom in tri_scene.geometry.values():
        # geom.show()
        mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        for p in mesh.primitives:
            p.material.doubleSided = True
        scene.add(mesh)
    # scene = pyrender.Scene.from_trimesh_scene(
    #             tri_scene, bg_color=bgcolor, ambient_light=bgcolor
    #         )

    center_pose = tri_scene.centroid
    bounds = tri_scene.bounds
    center_pose[2] = bounds[1, 2] + 0.2  # 上方0.2m
    length = bounds[1, :] - bounds[0, :]
    xl = length[0]
    yl = length[1]
    # print(xl, yl, center_pose)
    if xl <= 1e-1 or yl <= 1e-1:
        return None, None
    camera_pose = get_matrix(0, 0, 0, center_pose)
    camera_instance = pyrender.OrthographicCamera(
        xmag=xl / 2.0, ymag=yl / 2.0, znear=0.01, zfar=length[2] + 10
    )
    light = pyrender.DirectionalLight(color=lightcolor, intensity=intensity)
    scene.add(light)
    scene.add(camera_instance, pose=camera_pose)
    bounds[:, 2] -= 20
    scene.add(pyrender.Mesh.from_trimesh(creation.box(bounds=bounds), smooth=False))
    # print(length[2] + 11)
    # v = pyrender.Viewer(scene)

    width = int(xl / resolution + 1)
    height = int(yl / resolution + 1)
    render = pyrender.OffscreenRenderer(width, height)
    color, depth = render.render(scene)
    render.delete()
    scene.clear()
    dsm = orth_depth_to_dsm(depth, camera_pose, camera_instance.get_projection_matrix())
    bounds = trimesh.transform_points(bounds, transform)
    if is_xyz:
        # trans1 = pyproj.Proj('EPSG:4978')
        # trans2 = pyproj.Proj('EPSG:%d'%epsg)
        trans = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:%d" % epsg)
        y, x, z = trans.transform(bounds[:, 0], bounds[:, 1], bounds[:, 2])
        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        xres = dx / width
        yres = dy / height
        geo = [np.min(x), xres, 0, np.max(y), 0, -yres]
    else:
        geo = [bounds[0, 0], resolution, 0, bounds[1, 1], 0, -resolution]

    basename = os.path.splitext(out)

    dataset = gdal.GetDriverByName("GTiff").Create(
        basename[0] + "_dom.tif", width, height, 3, gdal.GDT_Byte
    )
    dataset.SetGeoTransform(geo)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    dataset.SetProjection(proj.ExportToWkt())

    for i in range(3):
        dataset.GetRasterBand(i + 1).WriteArray(color[:, :, i])
    # dataset.GetRasterBand(1).WriteArray( color[:,:,0])
    # dataset.GetRasterBand(2).WriteArray( color[:,:,1])
    # dataset.GetRasterBand(3).WriteArray( color[:,:,2])
    dsmds = gdal.GetDriverByName("GTiff").Create(
        basename[0] + "_dsm.tif", width, height, 1, gdal.GDT_Float32
    )
    dsmds.SetGeoTransform(geo)
    dsmds.SetProjection(proj.ExportToWkt())
    dsmds.GetRasterBand(1).WriteArray(dsm)

    del dataset
    del dsmds

    return basename[0] + "_dom.tif", basename[0] + "_dsm.tif"


def orth_invmesh_with_coords(
    meshes,
    bgcolor=(0, 0, 0.0, 0.0),
    lightcolor=(1.0, 1.0, 1.0),
    intensity=10,
    resolution=0.2,
    out="",
    is_xyz=True,
    transform=np.eye(4),
    epsg="4547",
):
    meshes = [load_mesh(p) for p in meshes]
    tri_scene = trimesh.Scene(meshes)
    # tri_scene.show()
    invz = np.eye(4)
    invz[2, 2] = -1
    # tri_scene = tri_scene.apply_transform(invz)
    # tri_scene.show()
    scene = pyrender.Scene(ambient_light=lightcolor, bg_color=bgcolor)
    for geom in tri_scene.geometry.values():
        mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        for p in mesh.primitives:
            p.material.doubleSided = True
        scene.add(mesh, pose=invz)
    center_pose = scene.centroid
    # center_pose[2] = - center_pose[2]
    bounds = scene.bounds
    # bounds[:,2] = - bounds[:,2]
    # bounds[0,2], bounds[1,2] = bounds[1,2], bounds[0,2]
    center_pose[2] = bounds[1, 2] + 0.2  # 上方0.2m
    length = bounds[1, :] - bounds[0, :]
    xl = length[0]
    yl = length[1]

    camera_pose = get_matrix(0, 0, 0, center_pose)
    # camera_pose[2,2] = -1
    camera_instance = pyrender.OrthographicCamera(
        xmag=xl / 2.0, ymag=yl / 2.0, znear=0.01, zfar=length[2] + 10
    )
    light = pyrender.DirectionalLight(color=lightcolor, intensity=intensity)

    # scene.add(pyrender.Mesh.from_trimesh(creation.box(bounds = scene.bounds)))
    scene.add(light)
    # import trimesh.creation
    # geom  = get_orthographic_visual_geom(xmag = xl / 2.0, ymag= yl/ 2.0, znear = 0.01, zfar = length[2], matrix=camera_pose)
    # scene.add(pyrender.Mesh.from_trimesh(geom))
    scene.add(camera_instance, pose=camera_pose)
    # v = pyrender.Viewer(scene)

    width = int(xl / resolution + 1)
    height = int(yl / resolution + 1)
    render = pyrender.OffscreenRenderer(width, height)
    color, depth = render.render(scene)
    render.delete()
    scene.clear()
    dsm = orth_depth_to_dsm(depth, camera_pose, camera_instance.get_projection_matrix())
    color = np.flip(color, axis=1)
    dsm = np.flip(dsm, axis=1)
    bounds = trimesh.transform_points(tri_scene.bounds, transform)
    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.imshow(color)
    # plt.subplot(122)
    # plt.imshow(depth)
    # plt.show()

    if is_xyz:
        # trans1 = pyproj.Proj('EPSG:4978')
        # trans2 = pyproj.Proj('EPSG:%d'%epsg)
        trans = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:%d" % epsg)
        y, x, z = trans.transform(bounds[:, 0], bounds[:, 1], bounds[:, 2])
        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        xres = dx / width
        yres = dy / height
        geo = [np.min(x), xres, 0, np.max(y), 0, -yres]
    else:
        geo = [bounds[0, 0], resolution, 0, bounds[1, 1], 0, -resolution]

    basename = os.path.splitext(out)

    dataset = gdal.GetDriverByName("GTiff").Create(
        basename[0] + "_dom.tif", width, height, 3, gdal.GDT_Byte
    )
    dataset.SetGeoTransform(geo)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    dataset.SetProjection(proj.ExportToWkt())

    for i in range(3):
        dataset.GetRasterBand(i + 1).WriteArray(color[:, :, i])
    # dataset.GetRasterBand(1).WriteArray( color[:,:,0])
    # dataset.GetRasterBand(2).WriteArray( color[:,:,1])
    # dataset.GetRasterBand(3).WriteArray( color[:,:,2])
    dsmds = gdal.GetDriverByName("GTiff").Create(
        basename[0] + "_dsm.tif", width, height, 1, gdal.GDT_Float32
    )
    dsmds.SetGeoTransform(geo)
    dsmds.SetProjection(proj.ExportToWkt())
    dsmds.GetRasterBand(1).WriteArray(dsm)

    del dataset
    del dsmds

    return basename[0] + "_dom.tif", basename[0] + "_dsm.tif"


if __name__ == "__main__":
    import glob

    root = r"D:\data\3dpointcloud\xuanen"
    ext = ".obj"
    objs = glob.glob(root + "/*" + ext)[4:8]
    # orth_3dtile_with_coords(objs, out='./test/mesh2.tif',  epsg=4326, is_xyz=True, transform=np.array([-0.912198727927758,
    #         -0.409748069875845,
    #         0.0,
    #         0.0,
    #         0.20804322229683,
    #         -0.463154744793949,
    #         0.861513609893194,
    #         0.0,
    #         -0.353003538825508,
    #         0.785871619037023,
    #         0.507734477821623,
    #         0.0,
    #         -2253443.30022558,
    #         5016712.12885994,
    #         3219490.32158955,
    #         1.0]).reshape(4,4).T)

    mesh_info(objs, out="./test/meshinfos.json")
    render_prepare(
        "./test/meshinfos.json", out="./test/render.json", gap=100, buffer=50
    )

    # pth = './test/render.json'
    # render_loop(pth, out='./test/render_out', skip_exist=False)
