import mesh.mesh_tools as mt
import argparse
import os
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', 	required=True, 	help='input tileset.json')
    parser.add_argument('-o', '--output', 	required=True, 	help='output path')

    args = parser.parse_args()

    meshes = mt.load_cctileset_json(args.input, ret_iter=True)
    os.makedirs(args.output, exist_ok=True)
    for i, (mesh, name) in tqdm.tqdm(enumerate(meshes)):
        # if hasattr(mesh.visual.material, 'to_simple'):
        #     mesh.visual.material =mesh.visual.material.to_simple()
        # mesh.visual.material.name = '%s'%(name)
        # mesh.show(smooth=False)
        mt.export.export_scene(mesh, os.path.join(args.output, '%s.glb'%(name)))
