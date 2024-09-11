import os
import glob
import geopandas as gpd
# import _winapi
import shapely
from shapely.geometry import Polygon
from osgeo import gdal, osr

def get_file_list(path):
    path = os.path.join(path, 'Productions')
    if not os.path.exists(path):
        return []
    pths = os.listdir(path)
    fs = []
    for pth in pths:
        r = os.path.join(path, pth, '*ortho_merge.tif')
        fs.extend(glob.glob(r))
    
    return fs



def for_productions():
    root = "Z:/CCProject/Projects"
    # dirs =  ['2023 Q1', '2023 Q4-10', '2023 Q11', '2023 Q12', '2023 Q13', 'PP1', 'PP2', 'Q14-Q17', '2023 TooMany2', '2023 TooMany3', '2023 TooMany4', '2023 TooMany5', '2023 TooMany626', '2023 TooMany627', '2023 TooMany628']
    dirs = os.listdir(root)
    doms = []
    outdir = 'Z:/CCProject/Projects'
    for d in dirs:
        doms.extend(get_file_list( os.path.join(root,d)))
    geoms = []

    for dom in doms:
        ds = gdal.Open(dom)
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize

        geom = ds.GetGeoTransform()

        extent = Polygon([(geom[0], geom[3]), (geom[0] + xsize*geom[1], geom[3]), (geom[0] + xsize*geom[1], geom[3] + ysize*geom[5]), (geom[0], geom[3] + ysize*geom[5])])

        srs = ds.GetProjection()
        srs = osr.SpatialReference(wkt=srs)
        srs.AutoIdentifyEPSG()
        epsg = srs.GetAuthorityCode(None)
        print(epsg)

        # print(extent.area)
        geoms.append(extent)
    
    tables = gpd.GeoDataFrame({'geometry': geoms, 'dom': doms}, crs='EPSG:4547')
    os.makedirs(outdir, exist_ok=True)
    
    tables.to_file(os.path.join(outdir, 'domains.geojson'), driver='GeoJSON')




    # print('\n'.join(doms))
    # print('Find ', len(doms))
    # os.makedirs(outdir, exist_ok=True)
    # for dom in doms:
    #     _winapi.CreateJunction(dom, outdir)
        # os.symlink(dom, os.path.join(outdir, os.path.basename(dom)))

    # vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False, srcNodata=0, VRTNodata=0, outputSRS='EPSG:4547', allowProjectionDifference=True)
    # dom_vrt = gdal.BuildVRT(os.path.join(root, 'dommerge.vrt'), doms, options=vrt_options)
    # dom_vrt = None

if __name__ == '__main__':

    root = 'E:/jiangshan/domdsm'
    doms = glob.glob(os.path.join(root, '*_dom.tif'))

    outdir = 'E:/jiangshan'
    geoms = []

    for dom in doms:
        ds = gdal.Open(dom)
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize

        geom = ds.GetGeoTransform()

        extent = Polygon([(geom[0], geom[3]), (geom[0] + xsize*geom[1], geom[3]), (geom[0] + xsize*geom[1], geom[3] + ysize*geom[5]), (geom[0], geom[3] + ysize*geom[5])])

        srs = ds.GetProjection()
        srs = osr.SpatialReference(wkt=srs)
        srs.AutoIdentifyEPSG()
        epsg = srs.GetAuthorityCode(None)
        # print(epsg)

        # print(extent.area)
        geoms.append(extent)
    
    tables = gpd.GeoDataFrame({'geometry': geoms, 'dom': doms}, crs='EPSG:4326')
    os.makedirs(outdir, exist_ok=True)
    
    tables.to_file(os.path.join(outdir, 'jiangshan.geojson'), driver='GeoJSON')

