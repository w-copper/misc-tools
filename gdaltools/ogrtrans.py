import json
import os
import copy
def batch_json(path, crs, outdir, outjson):
    files = os.listdir(path)
    # print(files)
    results = []
    result_dict = dict(
        PARAMETERS= dict(CRS='QgsCoordinateReferenceSystem(\'%s\')'%crs,
        INPUT='',),
        OUTPUTS=dict(OUTPUT='',),
    )
    for f in files:
        if f.endswith(".geojson"):
            d = copy.deepcopy(result_dict)
            # f = os.path.join(path, f).replace('\\', '/')
            d['PARAMETERS']['INPUT'] = "'" + os.path.join(path, f).replace('\\', '/') + "'"
            d['OUTPUTS']['OUTPUT'] = os.path.join(outdir, f).replace('\\', '/')

            results.append(d)
    
    with open(outjson, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    batch_json('D:/data/3dpointcloud/jingshan/buildingclip',
               'EPSG:4547',
               'D:/data/3dpointcloud/jingshan/building_crs',
               "D:/data/3dpointcloud/jingshan/batch.json")