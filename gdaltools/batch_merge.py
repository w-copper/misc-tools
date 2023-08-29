import os
import sys

def run(inputs, output):

    exe = os.path.dirname(sys.executable)
    gdalwarp = os.path.join(exe, 'Library', 'bin', 'gdalwarp.exe')
    args = [ "--config", "GDAL_CACHEMAX", "3000", "-wm", "3000", *inputs, output ]
    os.system(gdalwarp + ' ' + ' '.join(args))


    

