import geopandas as gpd
import numpy as np
import os
from mesh_tools import load_ccobjs_to_trimesh, load_objs_box
import shapely.geometry as geom
import argparse
from coord_trans import gpd_coord_trans_inv
import xmltodict
import tqdm
import trimesh
import trimesh.voxel.ops as voxel_ops


def run_one_polygon(
    boxes: gpd.GeoDataFrame, polygon: geom.Polygon, sample: float, out_path: str
):
    objs = boxes[boxes.intersects(polygon)]["obj"]
    if len(objs) == 0:
        return
    points = []
    for obj in objs:
        mesh = trimesh.load(obj, force="mesh")
        vertices = mesh.vertices
        point_gp = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(vertices[:, 0], vertices[:, 1]),
        )
        point_in_polygon = point_gp.intersects(polygon)
        point_xyz = vertices[point_in_polygon]
        points.append(point_xyz)
    points = np.concatenate(points)
    if sample > 0:
        voxel_grid = voxel_ops.points_to_marching_cubes(points, pitch=sample)
        points = voxel_grid.vertices
    np.savetxt(out_path, points, delimiter="\t", fmt="%f", encoding="utf8")


def load_polygon_file(pth, metadata_path=None):
    # gdf = None
    if (
        pth.endswith(".shp")
        or pth.endswith(".geojson")
        or pth.endswith(".gml")
        or pth.endswith(".gpkg")
        or pth.endswith(".kml")
        or pth.endswith(".kmz")
    ):
        gdf = gpd.read_file(pth)
        if metadata_path is not None:
            srs_dict = xmltodict.parse(open(metadata_path, "r", encoding="utf8").read())
            gdf = gpd_coord_trans_inv(gdf, srs_dict)
    elif pth.endswith(".txt"):
        with open(pth, "r", encoding="utf-8") as f:
            lines = f.readlines()
        polys = []
        for line in lines:
            p = np.array(list(map(float, line.split(",")))).reshape(-1, 3)
            poly = geom.Polygon(p[:, :2])
            polys.append(poly)
        gdf = gpd.GeoDataFrame(geometry=polys)
    return gdf


def run(objroot, shp, sample=0):
    objs = load_ccobjs_to_trimesh(os.path.join(objroot, "Data"))
    boxes, _ = load_objs_box(objs)
    gdf = load_polygon_file(shp, os.path.join(objroot, "metadata.xml"))
    outdir = os.path.join(
        os.path.dirname(shp), f"{os.path.basename(shp).split('.')[0]}_pnts"
    )
    os.makedirs(outdir, exist_ok=True)
    for i, polygon in tqdm.tqdm(enumerate(gdf.geometry)):
        out_path = os.path.join(outdir, f"pnts_{i}.txt")
        run_one_polygon(boxes, polygon, sample, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objroot", type=str, default="data")
    parser.add_argument("--shp", type=str, required=True)
    parser.add_argument("--sample", type=float, default=0.5)
    args = parser.parse_args()
    run(args.objroot, args.shp, sample=args.sample)
