from osgeo import gdal
import json
import tqdm
import argparse
def dsm2json(dsm, out, res = 10, rect = None):
    ds =  gdal.Open(dsm)
    geo = 	ds.GetGeoTransform()

    # width = ds.RasterXSize
    # height = ds.RasterYSize

    xoff, xres, _, yoff, _, yres =  geo
    # breakpoint()
    if rect is not None:
        xstart, ystart, xend, yend = rect 
    else:
        xstart,  ystart, xend, yend = xoff, yoff, xoff + ds.RasterXSize * xres, yoff + ds.RasterYSize * 	yres

    # xstart = xstart - xoff)
    band = ds.GetRasterBand(1)
    
    xcount = int((xend - xstart) / 	xres / res + 1)
    ycount = int((yend - ystart) / 	yres / res + 1)
    bar = tqdm.tqdm(total=xcount * ycount)
    with open(out, 'w', encoding='utf-8') as f:
        full_data = dict(
            data = []
        )
        for jy in range(ycount):
            for ix in range(xcount):
                x = xstart + ix * res * xres
                x = (x - xoff) / xres
                y = ystart + jy * res * yres
                y = (y - yoff) / yres
                xe = xend if ix == xcount - 1 else xstart + (ix + 1) * res* xres 
                xe = (xe - xoff) / xres
                ye = yend if jy == ycount - 1 else ystart + (jy + 1) *  res* yres
                ye = (ye - yoff) / yres

                block = band.ReadAsArray(int(x), int(y), res, res).astype(float)
                block = block.max()

                data = dict(
                    x0 = x * xres + xoff,
                    y0 = y * yres + yoff,
                    x1 = xe * xres + xoff,
                    y1 = ye * yres + yoff,
                    max_value = block
                )
                full_data['data'].append(data)
                bar.update(1)

        f.write(json.dumps(full_data))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Convert DSM.tif to json, with grid max value'
    )

    parser.add_argument('-i', '--input', 	required=True, 	help='Input raster file')
    parser.add_argument('-o', '--output',   required=True,   help='Output json file')
    parser.add_argument('-r', '--res',	type=int,	required=True,	help= 	'Pixel size')

    parser.add_argument('--rect', default=None, type=float, nargs='+', help='Rect sub region' )


    args = parser.parse_args()
   
    dsm2json(args.input, args.output, args.res, args.rect)