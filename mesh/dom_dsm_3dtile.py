import os
import glob

import mesh_tools as mt
import tqdm
import tempfile

import os
import sys
from osgeo import gdal

def batch_merge(inputs, output):
    
    exe = os.path.dirname(sys.executable)
    gdalwarp = os.path.join(exe, 'Library', 'bin', 'gdalwarp.exe')
    args = [ "--config", "GDAL_CACHEMAX", "3000", "-wm", "3000", *inputs, output ]
    os.system(gdalwarp + ' ' + ' '.join(args))

def fund(listTemp, n):
    resules = []
    for i in range(0, len(listTemp), n):
        temp = listTemp[i:i + n]
        resules.append(temp)
    return resules


    
def for_xuanen(root, output):
    folders = os.listdir(root)
    subs = []
    for folder in folders:
        if folder.startswith('Tile'):
            f = os.path.join(root, folder, folder + '.json')
            subs.append(f)
    
    root_json = os.path.join(root, 'tileset.json')

    transform = mt.load_tileset_transform(root_json) 
    os.makedirs(output, exist_ok=True)
    doms = []
    dsms = []
    for i, s in tqdm.tqdm(list(enumerate(subs)), desc='rendering'):
        f = os.path.join(output, '%d.tif'%i)
        domf, dsmf = mt.orth_3dtile_with_coords([s], transform=transform, epsg=4326, out=f, intensity=2, resolution=0.1)
        doms.append(domf)
        dsms.append(dsmf)


def build_vrt(root):
    doms = glob.glob(os.path.join(root, '*_dom.tif'))
    dsms = glob.glob(os.path.join(root, '*_dsm.tif'))
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False, srcNodata=0, VRTNodata=0, xRes=0.1, yRes=0.1, outputSRS='EPSG:4547', allowProjectionDifference=True)
    dom_vrt = gdal.BuildVRT(os.path.join(root, 'dommerge.vrt'), doms, options=vrt_options)
    dom_vrt = None

    dsm_vrt = gdal.BuildVRT(os.path.join(root, 'dsmmerge.vrt'), dsms, options=vrt_options)
    dsm_vrt = None

def for_xuanen_inv(root, output):
    folders = os.listdir(root)
    subs = []
    for folder in folders:
        if folder.startswith('Tile'):
            f = os.path.join(root, folder, folder + '.json')
            subs.append(f)
    
    root_json = os.path.join(root, 'tileset.json')

    transform = mt.load_tileset_transform(root_json) 
    os.makedirs(output, exist_ok=True)
    doms = []
    dsms = []
    for i, s in tqdm.tqdm(list(enumerate(subs)), desc='rendering'):
        f = os.path.join(output, '%d.tif'%i)
        domf, dsmf = mt.orth_invmesh_with_coords([s], transform=transform, epsg=4326, out=f, intensity=10, resolution=0.1)
        doms.append(domf)
        dsms.append(dsmf)



def for_jiangshan(root, output):
    folders = os.listdir(root)
    subs = []
    for folder in folders:
        if folder.startswith('Tile'):
            f = os.path.join(root, folder, folder + '.json')
            subs.append(f)
    
    root_json = os.path.join(root, 'tileset.json')

    transform = mt.load_tileset_transform(root_json) 
    os.makedirs(output, exist_ok=True)
    doms = []
    dsms = []
    for i, s in tqdm.tqdm(list(enumerate(subs)), desc='rendering'):
        f = os.path.join(output, '%d.tif'%i)
        domf, dsmf = mt.orth_3dtile_with_coords([s], transform=transform, epsg=4326, out=f, intensity=10, resolution=0.1)
        doms.append(domf)
        dsms.append(dsmf)


if __name__ == '__main__':

    # for_xuanen('E:/xuanen/data', 'E:/xuanen/domdsm')
    # for_xuanen_inv('E:/xuanen/data', 'E:/xuanen/domdsminv')
    for_jiangshan('G:/江山图画/output', 'E:/jiangshan/domdsm')
    # build_vrt('E:/xuanen/domdsm')
    # files = glob.glob("D:/tmpy7dj1fk3/*.tif")
