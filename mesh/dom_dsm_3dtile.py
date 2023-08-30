import os
import glob

import mesh_tools as mt
import tqdm
import tempfile

import os
import sys

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


def batch_merge_files(inputs, output):

    if len(inputs) <= 10:
        batch_merge(inputs, output)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        i = 0
        while True:
            subs = fund(inputs, 10)
            inputs = []
            for s in subs:
                f = os.path.join(temp_dir, '%d.tif'%i)
                i+= 1
                batch_merge(s, f)
                inputs.append(f)
            if len(inputs) <= 10:
                batch_merge(inputs, output)
                return

    
def for_xuanen(root, output):
    folders = os.listdir(root)
    subs = []
    for folder in folders:
        if folder.startswith('Tile'):
            f = os.path.join(root, folder, folder + '.json')
            subs.append(f)
    
    root_json = os.path.join(root, 'tileset.json')

    transform = mt.load_tileset_transform(root_json) 
    with tempfile.TemporaryDirectory(dir='D:/') as tmpdirname:
        doms = []
        dsms = []
        for i, s in tqdm.tqdm(list(enumerate(subs)), desc='rendering'):
            f = os.path.join(tmpdirname, '%d.tif'%i)
            domf, dsmf = mt.orth_3dtile_with_coords([s], transform=transform, epsg=4326, out=f, intensity=1, resolution=0.1)
            doms.append(domf)
            dsms.append(dsmf)
        print('start merge')
        batch_merge_files(doms, os.path.splitext(output)[0] + '_dom.tif')
        batch_merge_files(dsms, os.path.splitext(output)[0] + '_dsm.tif')
    


if __name__ == '__main__':

    for_xuanen('E:/xuanen/3dtiles', 'E:/xuanen/3dtiles/xuanen.tif')
    # files = glob.glob("D:/tmpy7dj1fk3/*.tif")
