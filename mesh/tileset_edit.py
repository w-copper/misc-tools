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


def delete_area(mesh: trimesh.Trimesh, area: trimesh.Trimesh):
    mesh.intersection(area)


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


from scipy.spatial import KDTree
from PIL import Image

# import numpy as np
if __name__ == "__main__":
    # shp_path = "e:/Data/上海项目/0419万达/clip_patch.shp"
    # shp_df: gpd.GeoDataFrame = gpd.read_file(shp_path)
    # tileset_path = "e:/Data/上海项目/0419万达/tileset4/tileset.json"
    # obj1s = shp_df[shp_df["obj"].str.contains("Production_obj_4")]
    # tiles = obj1s["obj"].map(lambda x: os.path.basename(x).split(".")[0]).tolist()
    # delete_tileset_block(tileset_path, tiles)
    # # delete_tileset_block(tileset_path, shp_df["name"].tolist())
    obj_path = "e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/Tile_+000_+010.obj"
    mesh: trimesh.Trimesh = trimesh.load(obj_path)
    center = mesh.centroid
    trans = np.eye(4)
    trans[:3, 3] = center
    box = trimesh.creation.box(extents=[50, 10, 30], transform=trans)
    bound_box = trimesh.creation.box(bounds=mesh.bounds)
    box.visual = trimesh.visual.ColorVisuals(
        box, face_colors=np.array([[1, 0, 0, 0.5]] * box.faces.shape[0])
    ).to_texture()

    box.export("e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/box.obj")

    # intersect = mesh.union(box, engine="blender")
    # origin_uv = mesh.visual.uv
    # # mesh.faces

    # face_xyz = mesh.vertices[mesh.faces]
    # face_center = face_xyz.mean(axis=-1)
    # tree = KDTree(face_center, compact_nodes=True)
    # # origin_uv = mesh.visual.uv

    # inter_face_xyz = intersect.vertices[intersect.faces]
    # inter_face_center = inter_face_xyz.mean(axis=-1)
    # d, i = tree.query(inter_face_center, k=1, eps=1e-3)
    # face_map_xyz = face_xyz[i]
    # distance = inter_face_xyz[:, :, None, :] - face_map_xyz[:, None, :, :]
    # distance = np.linalg.norm(distance, axis=-1)
    # distance = np.argmin(distance, axis=-1)
    # face_map_idx = mesh.faces[i]
    # face_map_idx = np.take_along_axis(face_map_idx, distance, axis=-1)
    # new_uv = np.zeros((intersect.vertices.shape[0], 2))
    # new_uv[np.array(intersect.faces).flatten()] = origin_uv[face_map_idx.flatten()]

    # # tree = KDTree(mesh.vertices, compact_nodes=True)
    # # d, i = tree.query(intersect.vertices, k=1, eps=1e-3)

    # # new_uv = origin_uv[i]

    # # d, ni = tree.query(second_p, k=1, eps=1e-6)
    # # new_uv[~correct_index] = 0

    # intersect.visual = trimesh.visual.TextureVisuals(
    #     uv=new_uv, material=mesh.visual.material
    # )
    # # image.save("e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/uv.png")
    # intersect.export(
    #     "e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/intersect.obj"
    # )
    # # trimesh.repair.fix_normals(intersect)

    # scene = trimesh.Scene([box, intersect])
    # scene.show()
    # print(mesh.bounds)
