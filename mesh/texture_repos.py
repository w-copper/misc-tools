import trimesh
from trimesh.exchange import export
from trimesh.visual.material import SimpleMaterial
from trimesh.visual import TextureVisuals
import numpy as np
import logging
import fire

def find_no_texture(pth:str, out:str, no_texture=(127,127,127), strict=1):
    # TODO: 使用strict参数，忽略那些只有一个点在无纹理区域内的面
    scene = trimesh.load_mesh(pth)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    logging.info('Load mesh success ')
    notexutre_geom = []
    for geom in scene.geometry:
        mesh:trimesh.Trimesh = scene.geometry[geom]
        if not isinstance( mesh.visual, TextureVisuals):
            logging.warning('Not a texture mesh, skip')
            continue
        material = mesh.visual.material
        
        
        if not isinstance(material, SimpleMaterial):
            if hasattr(material, 'to_simple'):
                material = material.to_simple()
            else:
                logging.warning('Not a simple material, skip')
                continue
        image = material.image
        image = np.array(image)
        mask = np.sum((image == np.array(no_texture).reshape((1,1,3))), axis=-1) == 3
        
        uv = mesh.visual.uv
        vertex = mesh.vertices
        faces = mesh.faces
        uv = np.copy(uv)
        uv[:,0] = uv[:,0] * mask.shape[1]
        uv[:,1] = (1 - uv[:,1]) * mask.shape[0]
        uv = np.round(uv).astype(int)
        uvmask = mask[uv[:,1], uv[:,0]]
        vertex_faces = mesh.vertex_faces
        notexture = np.argwhere(uvmask)
        
        notexture = notexture.reshape(-1)
        if notexture.shape[0] == 0:
            logging.info('not found no texture vertex, skip')
            continue
      
        subpoints = list(notexture.tolist())
        facei = vertex_faces[notexture,:]
        facei = facei[facei != -1]

        facei = np.array(list(set(facei.tolist())))
        subfaces = faces[facei,:]
        face_normals = mesh.face_normals[facei,:]
        addp = subfaces.reshape((-1))
        addp = list(set(addp.tolist()))
        subpoints.extend(addp)
        # print(subpoints)
        subpoints = list(set(subpoints))
        newsubfaces = np.copy(subfaces)
        points = []
        for i in subpoints:
            points.append(vertex[i,:])
            subfaces[newsubfaces == i] = len(points) -1
        points = np.array(points)
        faces = np.copy(subfaces)
        
        ngeom = trimesh.Trimesh(vertices=points, vertex_colors=(255,0,0), face_normals=face_normals, faces=faces, face_colors=(255.0, 0.0, 0.0))
        notexutre_geom.append(ngeom)

    if len(notexutre_geom) == 0:
        logging.warning('not found no texture vertex, we will not export any file')
        return
    notexutre_geom = trimesh.util.concatenate(notexutre_geom)
    logging.info('export notexutre file')
    export.export_mesh(notexutre_geom, out)

if __name__ == '__main__':
    fire.Fire(find_no_texture)