import os
import glob
import shutil
import re



def change_0():
    root = r'D:\data\deeplearning\ISPRS\Potsdam\1_DSM_normalisation'

    files = glob.glob(os.path.join(root, '*.jpg'))

    search_p = re.compile(r'(0\d)')

    for file in files:
        results = re.search(search_p, file)
        if not results:
            continue
        for g in results.groups():
            gr = int(g)
            if gr < 10:
                gr = str(gr)
                # file.replace(g, gr)
                shutil.move(file, file.replace(g, gr))
        # print(results)

# change_0()


def chagne_vai():
    p = r'D:\data\deeplearning\ISPRS\Vaihingen\dsm'
    files = glob.glob(os.path.join(p, '*.tif'))
    patten = re.compile(r'.*_area\d*\.tif')
    for f in files:
        if re.search(patten, f):
            # nf = os.path.basename(f)
            nf = f.replace('dsm_09cm_matching_', 'top_mosaic_09cm_')
            shutil.move(f, nf)

chagne_vai()
# files = glob.glob(os.path.join(root, '*.jpg'))

# for file in files:

#     shutil.move(file, file.replace('_dsm', ''))