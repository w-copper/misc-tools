import pycocotools.coco as COCO
import pycocotools.mask as M
import os
import skimage.io as io
import tqdm
import numpy as np
import argparse

def convert_tif(input_ann_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    coco = COCO.COCO(input_ann_json)
    for i in tqdm.tqdm(coco.getImgIds(catIds=coco.getCatIds())):
        img = coco.loadImgs(i)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
        m = 0
        for ai, ann in enumerate(anns):
            rle = M.frPyObjects(ann['segmentation'], img['height'], img['width'])
            m += M.decode(rle) * (ai + 1)
        m = m.reshape((img['height'], img['width']))
        m = m.astype(np.uint8)
        outname = os.path.splitext(img['file_name'])[0] + '.tif'
        outpath = os.path.join(output_dir, outname)
        if os.path.exists(outpath):
            os.remove(outpath)
        io.imsave(outpath, m, check_contrast=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Converts the MSCOCO dataset to the format needed for the task of object detection.'
    )
    parser.add_argument('--input-ann-json', '-i', required=True, help='path to the input annotation json file')
    parser.add_argument('--output-dir', '-o', required=True, help='path to the output directory')
    args = parser.parse_args()
    convert_tif(args.input_ann_json, args.output_dir)

    # convert_tif(
    #     r"D:\data\deeplearning\crawdai\train\annotation.json",
    #     r"D:\data\deeplearning\crawdai\train\masks"
    # )