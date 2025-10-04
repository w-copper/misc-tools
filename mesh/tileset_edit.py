import pyproj
import numpy as np
import os
import json
import trimesh.visual
import xmltodict
import trimesh
import trimesh.creation as creation
import shapely.geometry as sg
import geopandas as gpd
import shutil
import tempfile
import platform

_search_path = os.environ.get("PATH", "")
if platform.system() == "Windows":
    # try to find Blender install on Windows
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(";") if len(i) > 0]
    for pf in [r"C:\Program Files", r"C:\Program Files (x86)"]:
        pf = os.path.join(pf, "Blender Foundation")
        if os.path.exists(pf):
            for p in os.listdir(pf):
                if "Blender" in p:
                    _search_path.append(os.path.join(pf, p))
    _search_path = ";".join(_search_path)
    print("searching for blender in: %s", _search_path)

if platform.system() == "Darwin":
    # try to find Blender on Mac OSX
    _search_path = [i for i in _search_path.split(":") if len(i) > 0]
    _search_path.append("/Applications/blender.app/Contents/MacOS")
    _search_path.append("/Applications/Blender.app/Contents/MacOS")
    _search_path.append("/Applications/Blender/blender.app/Contents/MacOS")
    _search_path = ":".join(_search_path)
    print("searching for blender in: %s", _search_path)

_blender_executable = shutil.which("blender", path=_search_path)
_clip_script_path = os.path.join(os.path.dirname(__file__), "blender_diff.py")
_merge_script_path = os.path.join(os.path.dirname(__file__), "blender_merge.py")
exists = _blender_executable is not None


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


def coord_trans_inv(points, srs_dict):
    srs = srs_dict["ModelMetadata"]["SRS"]
    if srs.startswith("EPSG:"):
        origin = srs_dict["ModelMetadata"]["SRSOrigin"]
        origin = origin.split(",")
        origin_xy = [float(origin[0]), float(origin[1]), 0]
        origin_xy = np.array(origin_xy).reshape(3, 1)
        points = lonlat2local(points, origin_xy, srs)
    elif srs.startswith("ENU"):
        center = srs.split(":")[-1].split(",")
        points = lonlat2enu(points, center)
    return points


def load_ccobjs_to_trimesh(root_data_path, sub=None):
    paths = os.listdir(root_data_path)
    results = []
    for pth in paths:
        f = os.path.join(root_data_path, pth, f"{pth}.obj")
        f = f.replace("//", "/")
        f = f.replace("\\", "/")
        if os.path.exists(f):
            print(f)
            f = f.replace("//", "/")
            f = f.replace("\\", "/")
            results.append(f)
    if sub is not None:
        results = results[sub]

    return results


def load_cc_tileset(tileset_path):
    """
    Load a tileset from a path and return the tileset as a numpy array.
    """
    with open(tileset_path, "r", encoding="utf-8") as f:
        tileset = json.load(f)

    return tileset


def tile_coord_convert(tileset):
    root_dict = tileset["root"]
    childrens = root_dict["children"]
    root_transform = root_dict["transform"]
    root_matrix = np.array(root_transform).reshape(4, 4)
    centers = []
    for child in childrens:
        box = box = np.array(child["boundingVolume"]["box"]).reshape(4, 3)
        center = box[0, :]
        extend = box[1, :]
        uri = child["content"]["uri"]

        # print(center, url)


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


def cc_coord_convert(metadata_path, output_path):
    srs_dict = xmltodict.parse(open(metadata_path, "r").read())
    data_dir = os.path.join(os.path.dirname(metadata_path), "Data")

    objs = load_ccobjs_to_trimesh(data_dir)

    geoms = []
    fobjs = []
    for p in objs:
        # print(p)
        pbox = str(p).replace(".obj", "_bbox.obj")
        if not os.path.exists(pbox):
            mesh = trimesh.load(p)
            box = creation.box(bounds=mesh.bounds)
            box.export(pbox)
        else:
            mesh = trimesh.load(pbox)

        b = mesh.bounds
        lon, lat = coord_trans(srs_dict, b[:, 0], b[:, 1], b[:, 2])
        g = sg.Polygon(
            [
                [lon[0], lat[0]],  # minx miny
                [lon[0], lat[1]],  # maxx miny
                [lon[1], lat[1]],  # maxx maxy
                [lon[1], lat[0]],  # minx maxy
            ]
        )
        fobjs.append(p)
        geoms.append(g)
        del mesh
    gp = gpd.GeoDataFrame({"geometry": geoms, "obj": fobjs}, crs="EPSG:4326")
    gp.to_file(output_path)
    return gp


def delete_tileset_block(tileset_path, blocks):
    with open(tileset_path, "r") as f:
        data = json.load(f)
    childrens = data["root"]["children"]
    saved_children = []
    del_childeren = []
    for c in childrens:
        name = c["content"]["uri"]
        name = os.path.basename(name)
        name = os.path.splitext(name)[0]
        if name in blocks:
            shutil.rmtree(os.path.join(os.path.dirname(tileset_path), name))
            del_childeren.append(c)
        else:
            saved_children.append(c)

    data["root"]["children"] = saved_children
    with open(tileset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def set_point_bellow_ground(metadata_path, shp_path, polygon: sg.Polygon):
    shp_df: gpd.GeoDataFrame = gpd.read_file(shp_path)
    rel_objs = shp_df[shp_df.intersects(polygon)]
    objs = rel_objs["obj"].tolist()
    srs_dict = xmltodict.parse(open(metadata_path, "r").read())

    for obj in objs:
        mesh: trimesh.Trimesh = trimesh.load(obj)
        points = mesh.vertices
        lon, lat = coord_trans(srs_dict, points[:, 0], points[:, 1], points[:, 2])
        lonlath = np.c_[lon, lat, points[:, 2]]
        ppg = [sg.Point(lonlath[i, 0], lonlath[i, 1]) for i in range(len(lonlath))]
        idx = polygon.contains(ppg)
        points[idx, 2] = -999


def clip_and_merge(
    obj_path: str,
    obj2_path: str,
    clip_obj2: bool,
    polygon: sg.Polygon,
    zmin=None,
    zmax=None,
):
    obj = trimesh.load(obj_path)
    zmin = obj.bounds[2, 0] if zmin is None else zmin
    zmax = obj.bounds[2, 1] if zmax is None else zmax

    area = trimesh.creation.extrude_polygon(polygon, zmax - zmin)
    area.vertices[:, 2] = area.vertices[:, 2] + zmin
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        temp_area = tempfile.NamedTemporaryFile(suffix=".obj", dir=tempdir)
        temp_out1 = tempfile.NamedTemporaryFile(suffix=".obj", dir=tempdir)
        temp_out2 = tempfile.NamedTemporaryFile(suffix=".obj", dir=tempdir)
        temp_box = tempfile.NamedTemporaryFile(suffix=".obj", dir=tempdir)
        temp_obj2 = tempfile.NamedTemporaryFile(suffix=".obj", dir=tempdir)
        if clip_obj2:
            box = trimesh.creation.box(bounds=obj.bounds)
            box.export(temp_box.name)
            os.system(
                _blender_executable,
                " --background --python "
                + _clip_script_path
                + " "
                + obj2_path
                + " "
                + temp_box.name
                + " "
                + temp_obj2.name,
            )
            obj2_path = temp_obj2.name
        area.export(temp_area.name)
        os.system(
            _blender_executable
            + " --background --python "
            + _clip_script_path
            + " "
            + obj_path
            + " "
            + temp_area.name
            + " "
            + temp_out1.name
        )
        os.system(
            _blender_executable
            + " --background --python "
            + _merge_script_path
            + " "
            + temp_out1.name
            + " "
            + obj2_path
            + " "
            + temp_out2.name
        )

        mesh = trimesh.load(temp_out2.name)

    return mesh


def warp_clip_and_merge(
    obj_root: str,
    shp_path: str,
    image_path: str,
):
    patch_df_path = os.path.join(obj_root, "patchs.geojson")
    metadata_path = os.path.join(obj_root, "metadata.xml")
    if not os.path.exists(patch_df_path):
        cc_coord_convert(metadata_path, patch_df_path)
    patch_df: gpd.GeoDataFrame = gpd.read_file(patch_df_path)
    shp_df: gpd.GeoDataFrame = gpd.read_file(shp_path)
    for i, row in patch_df.iterrows():
        box = row["geometry"]
        obj_path = row["obj"]
        shps = shp_df[shp_df.intersects(box)]
    pass


if __name__ == "__main__":
    # shp_path = "e:/Data/上海项目/0419万达/clip_patch.shp"
    # shp_df: gpd.GeoDataFrame = gpd.read_file(shp_path)
    # tileset_path = "e:/Data/上海项目/0419万达/tileset4/tileset.json"
    # obj1s = shp_df[shp_df["obj"].str.contains("Production_obj_4")]
    # tiles = obj1s["obj"].map(lambda x: os.path.basename(x).split(".")[0]).tolist()
    # delete_tileset_block(tileset_path, tiles)
    # # delete_tileset_block(tileset_path, shp_df["name"].tolist())
    obj_path = "e:/Data/上海项目/0419万达/Production_obj_3/Data/Tile_+001_+005/Tile_+001_+005.obj"
    # http://e/Data/%E4%B8%8A%E6%B5%B7%E9%A1%B9%E7%9B%AE/0419%E4%B8%87%E8%BE%BE/Production_obj_3/Data/Tile_+002_+005/Tile_+002_+005.obj
    mesh: trimesh.Trimesh = trimesh.load(obj_path)
    waters = []
    for i in range(7):
        w: trimesh.Trimesh = trimesh.load(
            f"local/output/Data/Object_{i}/Object_{i}.obj"
        )
        waters.append(w)
    scene = trimesh.Scene([mesh, *waters])
    scene.show()
