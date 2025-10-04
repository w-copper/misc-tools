import os

if os.path.exists("share"):
    os.environ["PROJ_LIB"] = os.path.join(os.getcwd(), "share", "proj")
    os.environ["GDAL_DATA"] = os.path.join(os.getcwd(), "share", "gdal")

import numpy as np
import shapely.geometry as geom
import geopandas as gpd
import pyproj


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
        if g.type == "Polygon":
            lon, lat = g.exterior.xy
            points = np.array([lon, lat]).T
            points = np.column_stack([points, np.ones(points.shape[0])])
            points = coord_trans_inv(points, srs_dict)
            geoms.append(geom.Polygon(points[:, :2]))
        elif g.type == "MultiPolygon":
            for g1 in g.geoms:
                lon, lat = g1.exterior.xy
                points = np.array([lon, lat]).T
                points = np.column_stack([points, np.ones(points.shape[0])])
                points = coord_trans_inv(points, srs_dict)
                geoms.append(geom.Polygon(points[:, :2]))
        elif g.type == "LineString":
            lon, lat = g.xy
            points = np.array([lon, lat]).T
            points = np.column_stack([points, np.ones(points.shape[0])])
            points = coord_trans_inv(points, srs_dict)
            geoms.append(geom.LineString(points[:, :2]))
        elif g.type == "Point":
            lon, lat = g.xy
            points = np.array([lon, lat]).T
            points = np.column_stack([points, np.ones(points.shape[0])])
            points = coord_trans_inv(points, srs_dict)
            geoms.append(geom.Point(points[:, :2]))
    gdf["geometry"] = geoms
    gdf.crs = None
    return gdf


if __name__ == "__main__":
    local_coord = [
        -899.5689425,
        -127.64999045107797,
        30.635602680990093,
        -897.04941125,
        -125.8873983965351,
        30.608783602644106,
        -923.41659875,
        -100.91890557504504,
        35.747335285604706,
        -925.2409087509092,
        -94.23379524999999,
        30.594044063641217,
        -925.4389747069406,
        -94.17520149999999,
        33.02806171220655,
        -894.4126925,
        -83.92113593592609,
        27.943537895810394,
        -925.3971596350135,
        -94.27285774999999,
        33.16179502617454,
        -897.04941125,
        -125.8873983965351,
        30.608783602644106,
        -899.5689425,
        -127.64709837698737,
        30.595008081760124,
        -923.41659875,
        -100.91890557504504,
        35.747335285604706,
        -923.12363,
        -99.43672518094637,
        32.69105043417223,
        -910.0470144980507,
        -136.636139,
        27.20145920792359,
        -910.1259622716502,
        -136.69473275,
        27.341446313448103,
    ]

    local_coord = np.array(local_coord).reshape((-1, 3))
    x = local_coord[:, 0]
    y = local_coord[:, 1]
    z = local_coord[:, 2]

    import xmltodict

    srs_dict = xmltodict.parse(open("demo/metadata.xml").read())
    # for i in range(len(x)):
    print(coord_trans(srs_dict, x, y, z))
