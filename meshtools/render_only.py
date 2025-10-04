import argparse
import os
import ultralytics
import trimesh
import numpy as np
import geopandas as gpd
import shapely.geometry as geom
import xmltodict
import PIL.Image as Image
import json
import pickle
from collections import defaultdict
from scipy.spatial import KDTree
from tqdm import tqdm
from mesh_tools import (
    render_one,
    rcd_to_xyz,
    get_matrix,
    get_orthographic_visual_geom,
    get_perspective_visual_geom,
    load_ccobjs_to_trimesh,
    load_objs_box,
)
from coord_trans import coord_trans, gpd_coord_trans_inv

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def render_detect(scene, camera, save=False, outpath="./render_out"):
    color, depth, dsm = render_one(camera, scene)
    color = Image.fromarray(color)

    if save:
        os.makedirs(outpath, exist_ok=True)
        fname = os.path.join(outpath, f"render_{int(np.random.rand() * 100000)}")
        color.save(f"{fname}_color.jpg")
        Image.fromarray((depth * 255).astype(np.uint8)).save(f"{fname}_depth.jpg")
        Image.fromarray((dsm)).save(f"{fname}_dsm.tif")
        with open(f"{fname}_camera.pkl", "wb") as f:
            pickle.dump(camera, f)


def run_point(
    pmesh: gpd.GeoDataFrame,
    point: geom.Point,
    resultion=0.01,
    height=40,
    xrot=0,
    yrot=0,
    zrot=0,
    save_render=False,
    render_out=None,
):
    matrix = get_matrix(zrot, yrot, xrot, np.array([point.x, point.y, height]))
    print(matrix)
    scene = trimesh.Scene([trimesh.load(f) for f in pmesh["obj"]])
    print(scene.bounds)
    # scene.show()
    camera = {
        "postype": "matrix",
        "matrix": matrix,
        "type": "perspective",
        "yfov": 60,
        "aspec": 1.0,
        "width": 640,
        "height": 640,
    }

    rgb, depth, dsm = render_one(camera, scene)
    
    import matplotlib.pyplot as plt

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.subplot(1, 3, 2)
    plt.imshow(depth)
    plt.subplot(1, 3, 3)
    plt.imshow(dsm)
    plt.show()


def batch_run_points(objroot, polyinput, save_render=False, render_out=None):
    metadata_path = os.path.join(objroot, "metadata.xml")
    srs_dict = xmltodict.parse(open(metadata_path, "r", encoding="utf-8").read())
    if polyinput.endswith(".txt"):
        polys = []
        with open(args.inputpath, "r", encoding="utf-8") as f:
            for line in f:
                points = np.array(list(map(float, line.split(",")))).reshape(-1, 3)
                polygon = geom.LineString(points[:, :2])
                polys.append(polygon)
        gpp = gpd.GeoDataFrame(geometry=polys)
    elif (
        polyinput.endswith(".shp")
        or polyinput.endswith(".geojson")
        or polyinput.endswith(".gpkg")
        or polyinput.endswith(".kml")
    ):
        gpp = gpd.read_file(polyinput)
        gpp = gpd_coord_trans_inv(gpp, srs_dict)
    else:
        raise ValueError("polyinput must be a .txt or .shp file")
    ccobjs = load_ccobjs_to_trimesh(os.path.join(objroot, "Data"))
    boxes, zrange = load_objs_box(ccobjs)
    pbar = tqdm(total=len(gpp))

    for i, poly in gpp.iterrows():
        pbar.set_description(f"Processing {i+1}")

        if isinstance(poly["geometry"], geom.Point):
            run_point(
                boxes,
                poly["geometry"],
                height=50,
                xrot=0,
                yrot=0,
                zrot=0,
            )
        else:
            print("WARNING: geometry type not supported")
            print(poly["geometry"])
        pbar.update(1)
    pbar.close()


def main(args):
    render_out = (
        args.render_out
        if args.render_out is not None
        else os.path.join(args.outroot, "render")
    )
    batch_run_points(args.objroot, args.polyinput, args.save_render, render_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objroot", type=str, default="F:/0422模型-滨江/objsub")
    parser.add_argument("--outroot", type=str, default="data")
    parser.add_argument(
        "--polyinput", type=str, default="F:/0422模型-滨江/objsub/points.shp"
    )
    parser.add_argument("--save_render", action="store_true", default=False)
    parser.add_argument("--render_out", type=str, default=None)
    args = parser.parse_args()
    main(args)
