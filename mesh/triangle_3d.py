import numpy as np
import triangle
import trimesh
import pyproj
import PIL.Image as Image
import trimesh.visual
import os
import json
import xmltodict
import shutil
import argparse


def lonlat2enu(lonlat, ref_point):
    """
    将经纬度坐标转换为相对于参考点的ENU坐标。
    :param lat_lon_alt: 包含目标点纬度、经度和高度的元组或list。
    :param ref_point: 包含参考点纬度、经度和高度的元组或list。
    :return: ENU坐标系下的坐标。
    """

    # 创建经纬度到ECEF坐标的转换器
    lla2ecef = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=False,
    )

    # 创建从ECEF到ENU的旋转矩阵
    def ecef_to_enu_matrix(lat0, lon0):
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

        R = np.array(
            [
                [-np.sin(lon0), np.cos(lon0), 0],
                [
                    -np.sin(lat0) * np.cos(lon0),
                    -np.sin(lat0) * np.sin(lon0),
                    np.cos(lat0),
                ],
                [
                    np.cos(lat0) * np.cos(lon0),
                    np.cos(lat0) * np.sin(lon0),
                    np.sin(lat0),
                ],
            ]
        )
        # print(x0)
        # S = [
        #     [-np.sin(lon0), np.cos(lon0), 0],
        #     [
        #         -np.sin(lat0) * np.cos(lon0),
        #         -np.sin(lat0) * np.sin(lon0),
        #         np.cos(lat0),
        #     ],
        #     [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        # ]
        return R

    # 将参考点和目标点从经纬度转换到ECEF坐标
    ecef_ref = np.array(lla2ecef.transform(ref_point[0], ref_point[1], 0)).reshape(
        3, -1
    )

    ecef_target = np.array(lla2ecef.transform(lonlat[:, 1], lonlat[:, 0], lonlat[:, 2]))

    # 计算目标点相对于参考点的ECEF向量
    ecef_vector = ecef_target - ecef_ref
    # 生成ECEF到ENU的旋转矩阵
    R = ecef_to_enu_matrix(ref_point[0], ref_point[1])

    # 将ECEF向量转换到ENU坐标系
    enu = R @ ecef_vector

    return enu.T


def lonlat2local(lat_lon_alt, ref_point, epsg):
    """
    将经纬度坐标转换为相对于参考点的ENU坐标。
    :param lat_lon_alt: 包含目标点纬度、经度和高度的元组或list。
    :param ref_point: 包含参考点纬度、经度和高度的元组或list。
    :return: ENU坐标系下的坐标。
    """
    llh2epsg = pyproj.Transformer.from_crs(
        "EPSG:4326",
        epsg,
        always_xy=False,
    )
    local = np.array(
        llh2epsg.transform(lat_lon_alt[:, 1], lat_lon_alt[:, 0], lat_lon_alt[:, 2])
    )
    local = local - ref_point
    return local.T


def coord_trans_inv(srs_dict, points):
    srs = srs_dict["ModelMetadata"]["SRS"]
    if srs.startswith("EPSG:"):
        origin = srs_dict["ModelMetadata"]["SRSOrigin"]
        origin = origin.split(",")
        origin_xy = [float(origin[0]), float(origin[1]), 0]
        origin_xy = np.array(origin_xy).reshape(3, 1)
        points = lonlat2local(points, origin_xy, srs)
    elif srs.startswith("ENU"):
        center = srs.split(":")[-1].split(",")
        center = np.array(center).astype(float)
        points = lonlat2enu(points, center)
    return points


def create_polygon_mesh(points, image=None, res=0.05, upnormal=False):
    # sourcery skip: extract-method
    points = np.array(points)
    points_xy = points[:, :2]
    seg = [[i, i + 1] for i in range(len(points) - 1)]
    seg.append([len(seg) - 1, 0])
    seg = np.array(seg)

    A = {
        "vertices": points_xy,
        "segments": seg,
    }
    B = triangle.triangulate(A, "p")
    mesh = trimesh.Trimesh(
        vertices=points,
        faces=B["triangles"],
    )
    trimesh.repair.fix_normals(mesh)
    if upnormal:
        up_normal = np.array([0, 0, 1])
        for i, face in enumerate(mesh.faces):
            normal = mesh.face_normals[i]
            if np.dot(normal, up_normal) < 0:
                mesh.faces[i] = mesh.faces[i][::-1]
        trimesh.repair.fix_normals(mesh)
    if image is not None:
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("image must be a string or PIL.Image.Image")
        width = img.width
        height = img.height

        points_xy = points[:, :2] - np.min(points[:, :2], axis=0)
        points_xy = points_xy / res
        xyu = points_xy[:, 0] // width + (points_xy[:, 0] % width) / width
        xyv = points_xy[:, 1] // height + (points_xy[:, 1] % height) / height
        uv = np.vstack([xyu, xyv]).T
        mesh.visual.uv = uv
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.SimpleMaterial(image=img)
        )
    else:
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, face_colors=[[255, 0, 0, 255]] * len(mesh.faces)
        )

    return mesh


def anyimage_flatten(img):
    image = Image.open(img)
    image = np.array(image)
    image1 = np.flip(image, axis=0)
    image2 = np.flip(image, axis=1)
    image3 = np.flip(image1, axis=1)

    vstack_image1 = np.vstack([image, image1])
    vstack_image2 = np.vstack([image2, image3])
    return np.hstack([vstack_image1, vstack_image2])


def points_to_mesh(points, image=None, res=0.1, output_path=None, xml_path=None):
    esjson = {"data": []}
    with open(points, "r", encoding="utf-8") as f:
        for line in f.readlines():
            esjson["data"].append(
                {
                    "points": np.array(list(map(line.split(","), float))).reshape(
                        -1, 3
                    ),
                }
            )
    esjson_to_mesh(
        esjson, image=image, res=res, output_path=output_path, xml_path=xml_path
    )


def esjson_to_mesh(esjson_path, image=None, res=0.1, output_path=None, xml_path=None):
    if output_path is None:
        output_path = "mesh"
    os.makedirs(output_path, exist_ok=True)
    mesh_path = os.path.join(output_path, "Data")
    os.makedirs(mesh_path, exist_ok=True)
    if isinstance(esjson_path, str):
        with open(esjson_path, "r", encoding="utf-8") as f:
            esjson = json.load(f)
    else:
        esjson = esjson_path
    if xml_path is None:
        # center = None
        for feature in esjson["data"]:
            points = feature["points"]
            points = np.array(points)
            center = np.mean(points, axis=0)
            break
        xmlstr = f"""<?xml version="1.0" encoding="UTF-8"?>
<ModelMetadata version="1">
    <!--Spatial Reference System-->
    <SRS>ENU:{center[1]},{center[0]}</SRS>
    <!--Origin in Spatial Reference System-->
    <SRSOrigin>0,0,0</SRSOrigin>
    <Texture>
        <ColorSource>Visible</ColorSource>
    </Texture>
</ModelMetadata>"""
        srs_dict = xmltodict.parse(xmlstr)
    else:
        with open(xml_path, "r", encoding="utf-8") as f:
            srs_dict = xmltodict.parse(f.read())

    image = anyimage_flatten(image)
    image = Image.fromarray(image)
    image.format = "jpeg"
    for i, feature in enumerate(esjson["data"]):
        points = feature["points"]
        points = np.array(points)
        enu = coord_trans_inv(srs_dict, points)
        mesh = create_polygon_mesh(enu, image=image, res=res)
        name = f"{i}_{int(np.random.rand() * 100000)}"
        obji_path = os.path.join(mesh_path, f"Object_{name}")
        os.makedirs(obji_path, exist_ok=True)
        obj_path = os.path.join(obji_path, f"Object_{name}.obj")
        mesh.export(obj_path)
        enu[:, 1] = -enu[:, 1]
        mesh.vertices = enu[:, [0, 2, 1]]
        obj_temp_path = os.path.join(obji_path, f"Object_{name}_temp.obj")
        mesh.export(obj_temp_path)
        osgb_path = os.path.join(obji_path, f"Object_{name}.osgb")
        os.system(
            f"Z:/CVEO成果/3D-Software/software/osgconv.exe {obj_temp_path} {osgb_path}"
        )
        os.remove(obj_temp_path)
    if xml_path is None:
        with open(
            os.path.join(output_path, "metadata.xml"), "w", encoding="utf-8"
        ) as f:
            f.write(xmlstr)
    elif not os.path.exists(os.path.join(output_path, "metadata.xml")):
        shutil.copy(xml_path, os.path.join(output_path, "metadata.xml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--image", "-i", type=str, default=None)
    parser.add_argument("--res", "-r", type=float, default=0.1)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--xml", "-x", type=str, default=None)
    parser.add_argument(
        "--type", "-t", type=str, default="points", choices=["points", "esjson"]
    )

    args = parser.parse_args()
    if args.type == "points":
        points_to_mesh(
            args.input,
            image=args.image,
            res=args.res,
            output_path=args.output,
            xml_path=args.xml,
        )
    elif args.type == "esjson":
        esjson_to_mesh(
            args.input,
            image=args.image,
            res=args.res,
            output_path=args.output,
            xml_path=args.xml,
        )
