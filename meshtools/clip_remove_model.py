import trimesh.visual
from trimesh.visual.material import SimpleMaterial
from bpy_model_act import intersect_model, clip_model
import trimesh
import numpy as np
import os
import shapely.geometry as geom
import geopandas as gpd
import argparse
import pyproj
from tqdm import tqdm
import xmltodict
import glob


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
        center = np.array(center).astype(float)
        points = lonlat2enu(points, center)
    return points


def gpd_coord_trans_inv(gdf: gpd.GeoDataFrame, srs_dict):
    crs = gdf.crs
    if crs is None:
        return
    if crs.to_epsg() == 4326:
        pass
    else:
        gdf.to_crs(epsg=4326, inplace=True)
    geoms = []
    for g in gdf["geometry"]:
        lon, lat = g.exterior.xy
        points = np.array([lon, lat]).T
        points = np.column_stack([points, np.ones(points.shape[0])])
        points = coord_trans_inv(points, srs_dict)
        geoms.append(geom.Polygon(points[:, :2]))

    gdf["geometry"] = geoms
    gdf.crs = None
    return gdf


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


def make_light(m):
    if isinstance(m, trimesh.Trimesh):
        v = m.visual
        if isinstance(v, trimesh.visual.TextureVisuals):
            if isinstance(v.material, SimpleMaterial):
                v.material.diffuse = np.array([255, 255, 255, 255])
                v.material.ambient = np.array([255, 255, 255, 255])
                v.material.specular = np.array([255, 255, 255, 255])
                v.material.glossiness = 1
            elif hasattr(v.material, "to_simple"):
                v.material = v.material.to_simple()
                v.material.diffuse = np.array([255, 255, 255, 255])
                v.material.ambient = np.array([255, 255, 255, 255])
                v.material.specular = np.array([255, 255, 255, 255])
                v.material.glossiness = 1
    elif isinstance(m, trimesh.Scene):
        for mi in m.geometry:
            make_light(m.geometry[mi])
    else:
        return


def remove_mesh_by_polygons(mesh, polygons):
    m = trimesh.load(mesh)
    polys = []
    for poly in polygons:
        if isinstance(poly, geom.Polygon):
            pm = trimesh.creation.extrude_polygon(
                poly, height=m.bounds[1, 2] - m.bounds[0, 2]
            )
            pm.apply_translation(np.array([0, 0, m.bounds[0, 2]]))
            polys.append(pm)
        elif isinstance(poly, trimesh.Trimesh):
            polys.append(poly)
    # print(polys)
    polys = trimesh.Scene(polys)
    poly_export = os.path.abspath("poly.obj")
    polys.export(poly_export)
    reuslts = clip_model(mesh, poly_export)
    results: trimesh.Trimesh = trimesh.load(reuslts)
    make_light(results)
    os.remove(poly_export)
    return results


def clip_meshes_by_polygon(meshes, polygon: geom.Polygon, zrange=(999, -999)):
    results = []
    maxz = zrange[0]
    minz = zrange[1]

    polymesh = trimesh.creation.extrude_polygon(polygon, height=maxz - minz)
    polymesh.apply_translation(np.array([0, 0, minz]))
    poly_export = os.path.abspath("poly.obj")
    polymesh.export(poly_export)

    for mesh in meshes:
        result = intersect_model(mesh, poly_export)
        results.append(result)
    results = trimesh.Scene(results)
    os.remove(poly_export)
    return results


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


def load_objs(objs):
    meshes = []
    for obj in objs:
        mesh = trimesh.load(obj)
        meshes.append(mesh)
    return meshes


def batch_clip_run(objroot, outroot, gppoly, has_mesh=True):
    ccobjs = load_ccobjs_to_trimesh(os.path.join(objroot, "Data"))
    boxes, zrange = load_objs_box(ccobjs)
    # gppoly = gpd.GeoDataFrame(geometry=polys)
    pbar = tqdm(total=len(ccobjs))

    # boxes iter
    for i, box in boxes.iterrows():
        pbar.set_description(f"Processing {i+1}")
        insect = gppoly.intersects(box["geometry"])
        # print(insect)
        if insect.sum() > 0:
            obj_path = box["obj"]
            polys = gppoly[insect]
            multi_poly: gpd.GeoSeries = polys["geometry"]
            if multi_poly.union_all().covers(box["geometry"]):
                result = trimesh.Trimesh()
            else:
                if has_mesh:
                    result = remove_mesh_by_polygons(obj_path, polys["obj"])
                else:
                    result = remove_mesh_by_polygons(obj_path, polys["geometry"])
            obj_name = os.path.basename(obj_path)
            path_name = os.path.splitext(obj_name)[0]
            os.makedirs(os.path.join(outroot, path_name), exist_ok=True)
            result.export(os.path.join(outroot, path_name, obj_name))


def batch_clip_run_txt(objroot, outroot, points):
    with open(points, "r", encoding="utf-8") as f:
        lines = f.readlines()

    polys = []
    for line in lines:
        p = np.array(list(map(float, line.split(",")))).reshape(-1, 3)
        poly = geom.Polygon(p[:, :2])
        polys.append(poly)
    polys = gpd.GeoDataFrame(geometry=polys)
    batch_clip_run(objroot, outroot, polys)


def batch_clip_run_shp(objroot, outroot, points):
    metadata_path = os.path.join(objroot, "metadata.xml")
    srs_dict = xmltodict.parse(open(metadata_path, "r", encoding="utf-8").read())

    bgdata: gpd.GeoDataFrame = gpd.read_file(points)
    bgdata = gpd_coord_trans_inv(bgdata, srs_dict)

    batch_clip_run(objroot, outroot, bgdata)


def batch_run_mesh(objroot, outroot, dir):
    clip_objs = glob.glob(os.path.join(dir, "*.obj"))
    clip_objs = [trimesh.load(obj) for obj in clip_objs]
    clip_boxes = [
        geom.Polygon(
            [
                [obj.bounds[0], obj.bounds[2]],
                [obj.bounds[1], obj.bounds[2]],
                [obj.bounds[1], obj.bounds[3]],
                [obj.bounds[0], obj.bounds[3]],
            ]
        )
        for obj in clip_objs
    ]
    clip_gpd = gpd.GeoDataFrame(data={"obj": clip_objs}, geometry=clip_boxes)

    batch_clip_run(objroot, outroot, clip_gpd, has_mesh=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objroot", type=str, required=True)
    parser.add_argument("--outroot", type=str, required=True)
    parser.add_argument("--points", type=str, required=True)

    args = parser.parse_args()
    if args.points.endswith(".txt"):
        batch_clip_run_txt(args.objroot, args.outroot, args.points)
    elif (
        args.points.endswith(".shp")
        or args.points.endswith(".geojson")
        or args.points.endswith(".gpkg")
        or args.points.endswith(".kml")
        or args.points.endswith(".gpx")
    ):
        batch_clip_run_shp(args.objroot, args.outroot, args.points)
    else:
        raise ValueError("points file must be .txt or .shp")
