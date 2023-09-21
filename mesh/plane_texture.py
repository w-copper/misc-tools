import numpy as np
from osgeo import gdal, ogr, osr
import torchvision.ops as torch_ops
import torch
from shapely.wkt import loads
import json
import trimesh 
from typing import List

def load_polygon(file:str):
    ds = ogr.Open(file)
    layer = ds.GetLayer(0)
    polygons = []
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        geom =  feature.GetGeometryRef()
        polygon = loads(geom.ExportToWkt())
        polygons.append(polygon)
    return polygons

def create_planes(polygons:list, dsmfile:str) -> List[trimesh.Trimesh]:
    ds = gdal.Open(dsmfile)
    for polygon in polygons:
        pass
    return []


def get_meshes(plane:trimesh.Trimesh, meshinfo:dict) -> List[str]:
    bounds = plane.bounds
    target = [ bounds[0,0], bounds[0,1], bounds[1,1], bounds[1,0] ]
    target = np.array(target)
    target = torch.from_numpy(target).reshape(-1, 4)
    boxs = []
    for mesh in meshinfo['meshes']:
        b = mesh['bounds']
        box = [ b[0], b[1], b[3], b[4] ]
        box = np.array(box)
        box = torch.from_numpy(box).reshape(-1, 4)
        boxs.append(box)
    boxs = torch.cat(boxs, dim=0)
    ious = torch_ops.box_iou( target, boxs ) # 1xM
    ious = ious.numpy().reshape(-1)
    inter_id = np.argwhere(ious > 1e-10)
    vismesh = []
    for i, mesh in enumerate(meshinfo['meshes']):
        if i in inter_id:
            m = trimesh.load_mesh(mesh['file'])
            vismesh.append(m)
    
    scene = trimesh.Scene(vismesh)

    center = plane.centroid
    y_axis = np.array([0, 0, 1])
    # 法线向外还是向内?
    z_axis = plane.face_normals[0, :]
    x_axis = np.cross(z_axis, y_axis)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = np.vstack((x_axis, y_axis, z_axis))
    pose_matrix[:3, 3] = center

    rot_bounds = plane.bounding_box_oriented
    


    
    return []

def convert_coords(plane, meshinfo:dict):
    
    return plane

def capture_texture(meshinfo:dict, plane:trimesh.Trimesh):
    meshes = get_meshes(plane, meshinfo)

    return plane, None, None

def main(shpfile:str, dsmfile:str, meshinfo:str):
    '''
    SRS must be the same
    '''
    polygons = load_polygon(shpfile)
    with open(meshinfo, 'w') as f:
        meshinfos = json.load(f)
    planes = create_planes(polygons, dsmfile)
    
    for plane in planes:
        plane, rgb, depth = capture_texture(meshinfo, plane)


    



if __name__ == '__main__':
    load_polygon(r"D:\data\3dpointcloud\shanghai\building-test.shp")