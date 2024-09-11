import numpy as np
import triangle
import trimesh
import pyproj
import PIL.Image as Image
import trimesh.visual


def lonlat2enu(lat_lon_alt, ref_point):
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
    ecef_ref = np.array(lla2ecef.transform(ref_point[1], ref_point[0], 0)).reshape(
        3, -1
    )

    ecef_target = np.array(
        lla2ecef.transform(lat_lon_alt[:, 1], lat_lon_alt[:, 0], lat_lon_alt[:, 2])
    )

    # 计算目标点相对于参考点的ECEF向量
    ecef_vector = ecef_target - ecef_ref
    # 生成ECEF到ENU的旋转矩阵
    R = ecef_to_enu_matrix(ref_point[1], ref_point[0])

    # 将ECEF向量转换到ENU坐标系
    enu = R @ ecef_vector

    return enu.T

    # 创建从ENU到ECEF的旋转矩阵


def coord_trans(srs_dict, x, y, z=0):
    srs = srs_dict["ModelMetadata"]["SRS"]
    if srs.startswith("EPSG:"):
        proj1 = pyproj.CRS(srs)
        proj2 = pyproj.CRS("EPSG:4326")
        trans = pyproj.Transformer.from_crs(proj1, proj2, always_xy=True)
        origin = srs_dict["ModelMetadata"]["SRSOrigin"]
        origin = origin.split(",")
        origin_xy = [float(origin[0]), float(origin[1])]
        x = x + origin_xy[0]
        y = y + origin_xy[1]
        lon, lat = trans.transform(x, y)
        return lon, lat
    elif srs.startswith("ENU"):
        lla2ecef = pyproj.Transformer.from_crs(
            "EPSG:4326", {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"}
        )
        ecef2lla = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"}, "EPSG:4326"
        )
        center = srs.split(":")[-1].split(",")
        lat0 = np.deg2rad(float(center[0]))
        lon0 = np.deg2rad(float(center[1]))
        x0, y0, z0 = lla2ecef.transform(lat0, lon0, 0, radians=True)
        # print(x0)
        S = [
            [-np.sin(lon0), np.cos(lon0), 0],
            [
                -np.sin(lat0) * np.cos(lon0),
                -np.sin(lat0) * np.sin(lon0),
                np.cos(lat0),
            ],
            [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ]
        S = np.array(S)
        # print(np.array([x, y, z]).reshape(3, -1))
        dxyz = S.T @ np.array([x, y, z]).reshape(3, -1)
        x = x0 + dxyz[0, :]
        y = y0 + dxyz[1, :]
        z = z0 + dxyz[2, :]
        # print(x, y, z)
        lat, lon, alt = ecef2lla.transform(x, y, z, radians=True)
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)
        if len(lon) == 1:
            lon = lon[0]
            lat = lat[0]
        return lon, lat


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


def lonlat23857(points):
    p = pyproj.Proj("epsg:4326")
    ll = pyproj.Proj("epsg:3857")
    trans = pyproj.Transformer.from_proj(p, ll)
    xyz = trans.transform(points[:, 1], points[:, 0], points[:, 2])
    return np.vstack(xyz).T


def lonlat2car3(points):
    trans = pyproj.Transformer.from_crs(
        "epsg:4326",
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )
    xyz = trans.transform(points[:, 1], points[:, 0], points[:, 2])
    return np.vstack(xyz).T


def anyimage_flatten(img):
    image = Image.open(img)
    image = np.array(image)
    image1 = np.flip(image, axis=0)
    image2 = np.flip(image, axis=1)
    image3 = np.flip(image1, axis=1)

    vstack_image1 = np.vstack([image, image1])
    vstack_image2 = np.vstack([image2, image3])
    hstack_image = np.hstack([vstack_image1, vstack_image2])

    return hstack_image


def shp_to_mesh(shp_path, image=None, res=0.1, output_path=None):
    pass


import os
import json


def esjson_to_mesh(esjson_path, image=None, res=0.1, output_path=None):
    if output_path is None:
        output_path = os.path.dirname(esjson_path)
        output_path = os.path.join(output_path, "mesh")
    os.makedirs(output_path, exist_ok=True)
    mesh_path = os.path.join(output_path, "Data")
    os.makedirs(mesh_path, exist_ok=True)
    with open(esjson_path, "r", encoding="utf-8") as f:
        esjson = json.load(f)
    center = None
    image = anyimage_flatten(image)
    image = Image.fromarray(image)
    image.format = "jpeg"
    for i, feature in enumerate(esjson["data"]):
        points = feature["points"]
        points = np.array(points)
        if center is None:
            center = np.mean(points, axis=0)
        enu = lonlat2enu(points, center)
        mesh = create_polygon_mesh(enu, image=image, res=res)
        obji_path = os.path.join(mesh_path, f"Object_{i}")
        os.makedirs(obji_path, exist_ok=True)
        enu[:, 1] = -enu[:, 1]
        mesh.vertices = enu[:, [0, 2, 1]]
        obj_path = os.path.join(obji_path, f"Object_{i}.obj")
        osgb_path = os.path.join(obji_path, f"Object_{i}.osgb")

        mesh.export(obj_path)
        os.system(
            f"Z:/CVEO成果/3D-Software/software/osgconv.exe {obj_path} {osgb_path}"
        )
    with open(os.path.join(output_path, "metadata.xml"), "w", encoding="utf-8") as f:
        f.write(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<ModelMetadata version="1">
    <!--Spatial Reference System-->
    <SRS>ENU:{center[1]},{center[0]}</SRS>
    <!--Origin in Spatial Reference System-->
    <SRSOrigin>0,0,0</SRSOrigin>
    <Texture>
        <ColorSource>Visible</ColorSource>
    </Texture>
</ModelMetadata>"""
        )


if __name__ == "__main__":
    esjson_to_mesh("./mesh/polygons.json", "./water-5.jpg", 0.1, "output")
