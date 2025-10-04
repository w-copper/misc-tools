import trimesh
import pyrender
import numpy as np
from trimesh import creation
# from trimesh.visual import TextureVisuals
import pyvista as pv
import math
import os 
import argparse
import json
from shapely.geometry import Polygon

def reconstraction(planes,textures,out_obj_path):
    
    recon_scene = pv.Plotter()
    recon_scene.add_axes()
    texcoords1 = np.array([[1, 1],
                [1, 0],
                [0, 1],
                [0, 0]
                ])
    for i,plane in enumerate(planes):
        # if i==0:
            
            heigh = np.array(plane)[:,2]
            print(heigh)
            if np.all(heigh==heigh[0]):
                polygoncoor = np.array(plane)[:,0:2]
                # print(polygoncoor)
                polygon = Polygon(polygoncoor)
                v, f = trimesh.creation.triangulate_polygon(polygon)
                # print(f)
                plane_recons = trimesh.Trimesh(vertices=plane, faces=f)
                recon_scene.add_mesh(plane_recons,color = [0.537,0.537,0.537])
            else:
                triangles = [[0,1,3,2]]
                plane_recons = trimesh.Trimesh(vertices=plane, faces=triangles)
                plane_recons.visual.uv = texcoords1
                
                recon_scene.add_mesh(plane_recons,texture=textures[i])

        
    recon_scene.export_obj(os.path.join(out_obj_path,'rebuilding.obj')) 
    recon_scene.show()
    recon_scene.close()

    return

def render_plane_single(tri_scene,i,plane, bgcolor=(0, 0, 1.0, 0.0),lightcolor=(1.0, 1.0, 1.0), reslution=0.01):
    

    y = np.array(plane[0]-plane[1])
    z = np.cross(plane[0] - plane[1], plane[2] - plane[0])
    x = np.cross(y,z)
    vec_norm_y = np.linalg.norm(y)
    unit_vec_y = y / vec_norm_y

    vec_norm_z = np.linalg.norm(z)
    unit_vec_z = z / vec_norm_z

    vec_norm_x = np.linalg.norm(x)
    unit_vec_x = x / vec_norm_x

    points = np.array(plane)
    
    center = points.mean(axis=0)
    center_forward = center + 5*unit_vec_z
 
    matrix = np.eye(4)
    matrix[:3,0] = unit_vec_x
    matrix[:3,1] = unit_vec_y
    matrix[:3,2] = unit_vec_z
    matrix[:3,3] = center_forward
    # print(matrix)
    proj_points = trimesh.transform_points(points, matrix)


    import math
    xmag = math.sqrt((proj_points[0,0]-proj_points[1,0])**2+(proj_points[0,1]-proj_points[1,1])**2+(proj_points[0][2]-proj_points[1][2])**2)
    ymag = math.sqrt((proj_points[0, 0] - proj_points[2, 0]) ** 2 + (proj_points[0, 1] - proj_points[2, 1]) ** 2+(proj_points[0][2]-proj_points[2][2])**2)

    xmag = xmag / 2
    ymag = ymag / 2

    tri_scene = tri_scene
    bgcolor=(0, 0, 1.0, 0.0)
    lightcolor=(1.0, 1.0, 1.0)
    scene = pyrender.Scene.from_trimesh_scene(tri_scene, bg_color=bgcolor)

    camrea = pyrender.OrthographicCamera(ymag,xmag, znear=0.1, zfar=1000)
    light = pyrender.DirectionalLight(color=lightcolor, intensity=10.0)
    scene.add(light,pose=matrix)
    scene.add(camrea, pose=matrix)
    # pyrender.Viewer(scene)

    reslution = 0.01
    width = int(xmag / reslution)
    height = int(ymag / reslution)
    render = pyrender.OffscreenRenderer(height, width)
    color, _ = render.render(scene)


    return color

def point_in_quadrilateral(A, B, C, D, P):
    # 计算向量PA、PB、PC和PD与向量AB、BC、CD和DA的叉积
    cross_AB_PA = np.cross(B - A, P - A)
    cross_BC_PB = np.cross(C - B, P - B)
    cross_CD_PC = np.cross(D - C, P - C)
    cross_DA_PD = np.cross(A - D, P - D)

    if (cross_AB_PA >= 0 and cross_BC_PB >= 0 and
        cross_CD_PC >= 0 and cross_DA_PD >= 0) or \
       (cross_AB_PA <= 0 and cross_BC_PB <= 0 and
        cross_CD_PC <= 0 and cross_DA_PD <= 0):
        return True
    else:
        return False

def main():

    parser = argparse.ArgumentParser(description="process some integers")
    parser.add_argument("--inputpath","-i",type=str,help="Input file path")
    parser.add_argument("--parent_folder","-p",type=str,help="the parent folder of objs")
    parser.add_argument("--outputpath",'-o',type=str,help="Output file path")
    args = parser.parse_args()
    parent_folder = args.parent_folder
 
    building_mesh = trimesh.load(args.inputpath)
    building_mesh_box = building_mesh.bounding_box_oriented 
    building_plane = building_mesh_box.vertices
    point = building_mesh_box.centroid
  
    building_x_min = np.min(building_plane[:,0])
    building_y_min = np.min(building_plane[:,1])
    building_x_max = np.max(building_plane[:,0])
    building_y_max = np.max(building_plane[:,1])

    building_tr_p = np.array([building_x_min,building_y_max])
    building_dr_p = np.array([building_x_min,building_y_min])
    building_tl_p = np.array([building_x_max,building_y_max])
    building_dl_p = np.array([building_x_max,building_y_min])
    right_block = []

    index_path = os.path.join(parent_folder,"index.txt")
    with open(index_path, 'r') as file:
        for line in file:
           
            parsed_line = line.strip().split(':') 
            list_data = json.loads(parsed_line[1])
            A = np.array(list_data[0])
            B = np.array(list_data[3])
            C = np.array(list_data[1])
            D = np.array(list_data[2])

            result_tr = point_in_quadrilateral(A,B,C,D,building_tr_p)
            result_dr = point_in_quadrilateral(A,B,C,D,building_dr_p)
            result_tl = point_in_quadrilateral(A,B,C,D,building_tl_p)
            result_dl = point_in_quadrilateral(A,B,C,D,building_dl_p)

            if result_tr or result_dr or result_tl or result_dl:
                
                file_path = os.path.join(parent_folder,parsed_line[0],parsed_line[0]+".obj")

                right_block.append(file_path)

    mesh1 = trimesh.load(right_block[0])
    tri_scene = trimesh.Scene(mesh1)
    if len(right_block)!=1:
        for i in range(len(right_block)-1):
            
            tri_scene.add_geometry(trimesh.load(right_block[i+1]))


    lenght = building_mesh.facets_boundary

    all_vertices = building_mesh.vertices
    planes = []
    textures = []
    for i in range(len(lenght)):
        
        plane = []
        indexva = []
        for j in range(len(all_vertices[lenght[i]])):

            if j == 0:
                plane.append(all_vertices[lenght[i][j][0]])
                plane.append(all_vertices[lenght[i][j][1]])
                indexva.append(lenght[i][j][0])
                indexva.append(lenght[i][j][1])
            else:
                if lenght[i][j][0] in indexva:
                    pass
                else:
                    plane.append(all_vertices[lenght[i][j][0]])
                    indexva.append(lenght[i][j][0])
                if lenght[i][j][1] in indexva:
                    pass
                else:
                    plane.append(all_vertices[lenght[i][j][1]])
                    indexva.append(lenght[i][j][1])

        if i ==len(lenght)-3:
            plane[:2],plane[2:] = plane[2:],plane[:2]
        planes.append(plane)
        texture = render_plane_single(tri_scene, i,plane)
        textures.append(texture)
    
    reconstraction(planes,textures,args.outputpath)

if __name__ == "__main__":
    main()