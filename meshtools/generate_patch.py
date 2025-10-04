import xmltodict
import geopandas as gpd
import os
from coord_trans import coord_trans
from mesh_tools import load_ccobjs_to_trimesh, load_objs_box
import shapely.geometry as geom
import numpy as np


def generate_tile_grid(root):
    metadata = os.path.join(root, "metadata.xml")
    srs_dict = xmltodict.parse(open(metadata).read())
    # srs = srs_dict["ModelMetadata"]["SRS"]

    objs = load_ccobjs_to_trimesh(os.path.join(root, "Data"))
    boxes, _ = load_objs_box(objs)
    new_geoms = []
    for box in boxes["geometry"]:
        # box = geom.Polygon()
        coords = box.exterior.coords.xy
        # print(coords)
        x = coords[0]
        y = coords[1]
        xx, yy = coord_trans(srs_dict, x, y, np.zeros_like(x))
        box = geom.Polygon(list(zip(xx, yy)))
        print(box)
        new_geoms.append(box)
    boxes["geometry"] = gpd.GeoSeries(new_geoms)
    boxes["bounds"] = None
    boxes.crs = "EPSG:4326"
    # gdf = gpd.GeoDataFrame(geometry=new_geoms,  crs="EPSG:4326")
    boxes.to_file(os.path.join(root, "tile_grid.geojson"))


generate_tile_grid("F:/0422模型-滨江/objsub")
