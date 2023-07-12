import trimesh
from trimesh import creation
from trimesh import visual
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import tqdm
from pyproj import CRS, Transformer
import xmltodict
from collections import defaultdict
from collections import Counter

def read_objs(scene, pths):

    for p in tqdm.tqdm(pths, desc='Read OBJ files'):
        mesh:trimesh.Trimesh = trimesh.load(p)
        if isinstance(mesh, trimesh.Scene):
            for s in mesh.geometry:
                m = pyrender.Mesh.from_trimesh(mesh.geometry[s], smooth=False)
                scene.add(m)
        elif isinstance(mesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            scene.add(mesh)
        else:
            raise  Exception('Not supported format', mesh)


    return scene

def read_objs_to_scene(pths):
    scene = trimesh.Scene()
    for p in tqdm.tqdm(pths, desc='Read OBJ files'):
        mesh:trimesh.Trimesh = trimesh.load(p)
        if isinstance(mesh, trimesh.Scene):
            for s in mesh.geometry:
                scene.add_geometry(s)
        elif isinstance(mesh, trimesh.Trimesh):
            # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            scene.add_geometry(mesh)
        else:
            raise  Exception('Not supported format', mesh)

    return scene

def get_matrix(zrot = 0, yrot = 0, xrot = 0, pos = np.zeros(3)):
   '''
   xyzrot: 顺时针旋转角度,单位为度
   return: np.array with shape 4x4
   '''
   zrot = np.deg2rad(zrot)
   yrot = np.deg2rad(yrot)
   xrot = np.deg2rad(xrot)

   # 计算绕U轴旋转的矩阵
   rotation_u = np.array([[np.cos(zrot), np.sin(zrot), 0],
                        [-np.sin(zrot), np.cos(zrot), 0],
                        [0, 0, 1]])
   # print(rotation_u.shape)

   # 计算绕N轴旋转的矩阵
   rotation_n = np.array([[1, 0, 0],
                        [0, np.cos(xrot), np.sin(xrot)],
                        [0, -np.sin(xrot), np.cos(xrot)]])
   # 计算绕E轴旋转的矩阵
   rotation_e = np.array([[np.cos(yrot), 0, -np.sin(yrot)],
                        [0, 1, 0],
                        [np.sin(yrot), 0, np.cos(yrot)]])

   # 计算旋转后的坐标
   # rotated = np.dot(rotation_n, np.dot(rotation_u, np.vstack((e, n, u))))
   rot_mat = np.dot(np.dot(rotation_e, rotation_n), rotation_u)
   matrix = np.zeros((4,4))
   matrix[:3,:3] = rot_mat[:,:]
   matrix[:3,3] = pos[:]
   matrix[-1,-1] = 1
   # matrix[-1,-1] = 1
   return matrix
   
def render_one(scene:pyrender.Scene, camera):
    camera_instance = pyrender.PerspectiveCamera(camera['yfov'], aspectRatio=camera['aspec'])
    if camera['postype'] == 'matrix':
        camera_pose = camera['matrix']
    elif camera['postype'] == 'angles':
        camera_pose = get_matrix(camera['yaw'], camera['pitch'], camera['roll'], camera['pos'])
    else:
        raise NotImplementedError(f"Camera postype {camera['postype']}")
    render = pyrender.OffscreenRenderer(camera['width'], camera['height'])

    camera_node = scene.add(camera_instance,  pose=camera_pose)
    color, depth = render.render(scene)
    render.delete()
    scene.remove_node(camera_node)
    return color, depth

def main_loop(objpaths, camreas, out_dir,  
              save_depth = False, bgcolor = (0,0,1.0,0.0), lightcolor = (1.0,1.0,1.0)):
    
    
    scene = pyrender.Scene(bg_color=bgcolor)
    light = pyrender.DirectionalLight(color=lightcolor, intensity=5.0)
    scene.add(light)
    scene = read_objs(scene, objpaths)

    os.makedirs(out_dir, exist_ok=True)

    for i,camera in tqdm.tqdm(enumerate(camreas), desc='Start rendering'):
        color, depth = render_one(scene, camera)
        if save_depth:
            depth_file = os.path.join(out_dir, '%d_%s_depth.tif'%(i, camera['fname']))
            Image.fromarray(depth).save(depth_file)
        image_file = os.path.join(out_dir, '%d_%s.jpg'%(i, camera['fname']))
        Image.fromarray(color).save(image_file)

def load_productions(root_data_path, sub=None):
    paths = os.listdir(root_data_path)
    results = []
    for pth in paths:
        f = os.path.join(root_data_path, pth, f'{pth}.obj')
        if os.path.exists(f):
            results.append(f)
    if sub is not None:
        results = results[sub]
    return results

def parser_ccxml(ccxml_path, metadata_path):
    '''
    TODO: metadata 不在使用,去除相关代码;
    TODO: 视场角仍未能通过参数确定,需要进一步设置
    '''
    # SRS = ''
    # ORIGIN = []
    # with open(metadata_path, 'r') as f:
    #     for line in f.readlines():
    #         if line.strip().startswith('<SRS>'):
    #             SRS = CRS.from_string(line.strip()[len('<SRS>'):-len('</SRS>')])
    #         if line.strip().startswith('<SRSOrigin>'):
    #             origin_str = line.strip()[len('<SRSOrigin>'):-len('</SRSOrigin>')]
    #             origin_ls = origin_str.split(',')
    #             ORIGIN.extend((float(origin_ls[0]), float(origin_ls[1]), float(origin_ls[2])))
    
    # ORIGIN = np.array(ORIGIN).reshape(3)
    # BLOKC_SRSS = dict()

    # ZINV = np.eye(4)
    # ZINV[2,2] = -1
    # R90 = get_matrix(0, 0, -90, np.zeros(3)) # look at y, z up x right

    # return

    def read_to_line(f, endline = None):
        if endline is None:
            return f.readlines()
        else:
            res = []
            while True:
                line = f.readline()
                if line.strip() == endline:
                    return res + [line]
                else:
                    res.append(line)
    
    with open(ccxml_path, 'r') as f:
        line = f.readline().strip()
        # first_line = line
        # current_block_srs = None
        current_image_size = [0, 0]
        current_aspec = 1
        current_distortion = [ 0.0, 0.0, 0.0, 0.0, 0 ]
        current_focallength = 0
        while line != '</BlocksExchange>':
            if line.startswith('<SpatialReferenceSystems>'):
                # logging.info('process srs')
                
                res = read_to_line(f, '</SpatialReferenceSystems>')       
                line = f.readline().strip()         
                continue
                res.insert(0, line)
                xmlstr = 	'\n'.join(res)

                srss = xmltodict.parse(xmlstr)
                for srs in srss['SpatialReferenceSystems']['SRS']:
                    BLOKC_SRSS[srs['Id']] = CRS.from_string(srs['Definition'])
                # logging.info('fine %d srs', len(BLOKC_SRSS))
            elif line.startswith('<SRSId>'):
                line = f.readline().strip()
                continue
                current_block_srs = BLOKC_SRSS[line.split('>')[1].split('<')[0]]
            elif line.startswith('<ImageDimensions>'):
                wline = f.readline().strip().split('>')[1].split('<')[0]
                hline = f.readline().strip().split('>')[1].split('<')[0]
                current_image_size[0] = int(wline)
                current_image_size[1] = int(hline)
            elif line.startswith('<AspectRatio>'):
                current_aspec = float(f.readline().strip().split('>')[1].split('<')[0])
            elif line.startswith('<FocalLength>'):
                current_focallength = 	float(line.split('>')[1].split('<')[0])
            elif line.startswith('<Photo>'):
                res = read_to_line(f, '</Photo>')
                res.insert(0, line)
                xmlstr = 	'\n'.join(res)
                image_info = xmltodict.parse(xmlstr)['Photo']
                image_pose = image_info['Pose']
                yaw = float(image_pose['Rotation']['Yaw'])
                pitch = float(image_pose['Rotation']['Pitch'])
                roll =  float(image_pose['Rotation']['Roll'])

                R90 = get_matrix(0, -90, 0, np.zeros(3)) # look at y, z up x right
                YAW = get_matrix(0, 0, yaw, np.zeros(3)) # cz
                PITCH = get_matrix(0, -pitch, 0, np.zeros(3)) # ccx
                ROLL = get_matrix(-roll, 0, 0, np.zeros(3)) # cy
                # T = get_matrix(0, 0, 0, np.array([camera_pos[-1]])) # T

                pymatrix = ROLL @ PITCH @ YAW @ R90
                
                # yaw = 180 + yaw
                # if yaw < 0:
                #     yaw = yaw + 360
                # roll, pitch = -pitch, roll
                # rot_matrix = [[ float(image_pose['Rotation']['M_00']), float(image_pose['Rotation']['M_01']), float(image_pose['Rotation']['M_02'])],
                #              [float(image_pose['Rotation']['M_10']), float(image_pose['Rotation']['M_11']), float(image_pose['Rotation']['M_12'])],           
                #              [float(image_pose['Rotation']['M_20']), float(image_pose['Rotation']['M_21']), float(image_pose['Rotation']['M_22'])]]
                position = [float(image_pose['Center']['x']), 	float(image_pose['Center']['y']), 	float(image_pose['Center']['z'])]
                # position = pyproj.transform(current_block_srs, SRS, *position)
                # position = Transformer.from_crs(current_block_srs, SRS).transform(*position)
                # position = np.array(position).reshape(3)
                # relposition = position[[1,0,2]]   - ORIGIN
                relposition = position
                pymatrix[:3,3] = relposition[:]
                img_name = os.path.basename(image_info['ImagePath'])
                img_name = os.path.splitext(img_name)[0]
                camrea = dict(
                    yfov= 40 / 180 * np.pi,
                    aspec = current_image_size[0] / current_image_size[1],
                    width = 	current_image_size[0],
                    height = 	current_image_size[1],
                    postype = 'matrix',
                    matrix = pymatrix,
                    yaw = yaw,
                    pitch = pitch,
                    roll = roll,
                    pos = relposition,
                    fname = img_name
                )
                # print(camrea)
                yield camrea
            
            line = f.readline().strip()
    
import functools

def random_render( scene:pyrender.Scene, count = 0, 
                    yfov = [50, 150],
                    zrot = [0, 360],
                    xrot = [-60, 60],
                    yrot = [-60, 60],
                    width = 500,
                    height = 500
                    ):
    
    if count <= 0:
        count = np.inf
    
    def get_rand(r):
        if isinstance(r, (tuple, list)):
            return np.random.rand() * (r[1] - r[0]) + r[0]
        elif isinstance(r, (int,float)):
            return r
        else:
            raise NotImplementedError(f"Cannot handle type {type(r)}")

    rand_yfov = functools.partial(get_rand, yfov)
    rand_zrot = functools.partial(get_rand, zrot)
    rand_xrot = functools.partial(get_rand, xrot)
    rand_yrot = functools.partial(get_rand, yrot)
    rand_width = 	functools.partial(get_rand, width)
    rand_height = functools.partial(get_rand, height)
    # rand_aspect = functools.partial(get_rand, aspect)

    def rand_pos():
        '''
        TODO: fix it
        '''
        return scene.centroid

    c = 0
    while c < count:
        width, height = rand_width(), rand_height()
        yfov = rand_yfov() / 180.0 * np.pi
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / height)
        matrix = get_matrix(rand_zrot(), 	rand_yrot(), 	rand_xrot(), rand_pos())
        camera_node = scene.add(camera, pose=matrix)
        render = pyrender.OffscreenRenderer()
        color, depth = render.render(scene)
        render.delete()
        scene.remove_node(camera_node)

        yield color, depth, matrix, yfov
        
        c += 1

import copy

def remap_one(obj:trimesh.Trimesh, img:np.array, matrix:np.array, yfov:float, to_vers = True, to_face = False):
    if not (to_vers or to_face):
        return obj
    if img is None:
        return obj
    
    y, x = np.arange(img.shape[0]), np.arange(img.shape[1])
    y = y - 	img.shape[0]/2.0
    x = x - 	img.shape[1]/2.0
    xx, yy = np.meshgrid(x, y)
    hh, ww = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    # xy = np.stack([xx, yy], axis=-1) # N x 2
    hfov = np.deg2rad(yfov) / 2
    z = img.shape[0] / 2 / np.tan(hfov)
    z = np.ones((xx.shape[0])) * z
    xyz = np.stack([xx, yy, z], axis=-1) # N x 3
    direction = matrix[:3,:3] @ xyz.T # 3 x N
    direction = direction.T # N x 3
    origin = matrix[:3,3]
    origin = origin.reshape(1,3)
    origin = np.repeat(origin, xyz.shape[0], axis=0) # N x 3

    tri_indexs, ray_indexs, locations = obj.ray.intersects_id(origin, direction, return_locations=True, multiple_hist = False) # N
    hh = hh[ray_indexs]
    ww = ww[ray_indexs]
    colors = img[hh, ww,:] # K x 3
    if to_vers:
        neardist, nearvers = obj.nearest.vertex(locations)
        if 'neardist' in obj.vertex_attributes:
            predist = obj.vertex_attributes['neardist'][nearvers]
            valid = neardist < predist
            nearvers = nearvers[valid]
            obj.vertex_attributes['neardist'][nearvers][valid] = neardist[valid]
            colors = colors[valid]
            obj.vertex_attributes['cate_colors'][nearvers,:][valid,:] = colors[valid,:]
        else:
            obj.vertex_attributes['neardist'] = np.ones(len(obj.vertices)) * np.inf
            obj.vertex_attributes['cate_colors'] = np.zeros((len(obj.vertices), img.shape[2]))
            obj.vertex_attributes['neardist'][nearvers] = neardist
            obj.vertex_attributes['cate_colors'][nearvers,:] = colors
    if to_face:
        if 'rayinter' not in obj.face_attributes:
            obj.face_attributes['rayinter'] = defaultdict(list)
        face_attr = obj.face_attributes['rayinter']
        for i in range(len(ray_indexs)):
            face_attr[tri_indexs[i]].append(colors[ray_indexs[i],:])
    return obj
    
def merge_remap(obj:trimesh.Trimesh):
    vers_colors = None
    faces_colors = None
    if 'cate_colors' in obj.vertex_attributes:
        cates = obj.vertex_attributes['cate_colors']
        vers_colors = cates
    if 'rayinter' in  obj.face_attributes:
        face_attr = obj.face_attributes['rayinter']
        faces_colors = np.zeros((len(obj.faces), 3))
        for i in range(len(obj.faces)):
            if len(face_attr[i]) == 0:
                continue
            counter = Counter(face_attr[i])
            color = counter.most_common(1)[0][0]
            faces_colors[i,:] = color[:]
    
    obj.visual = visual.ColorVisuals(obj, face_colors=faces_colors, vertex_colors=vers_colors)

    return obj


def view_ccobjs(cc_data, sub = None, camera = None):
    
    scene = pyrender.Scene(bg_color=(0.0, 0.0, 1.0, 0.5))
    objpaths = load_productions(cc_data, sub)
    
    light = pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=5.0)
    scene.add(light)
    scene = read_objs(scene, objpaths)
    o = np.eye(4)
    o[:3,3] = np.array(scene.centroid)[:]
    axis = creation.axis(1, transform = o )
    axis = pyrender.Mesh.from_trimesh(axis, smooth= False)
    scene.add(axis)
    # axis_trans = np.eye(4)
    if camera is not None:
        camera_instance = pyrender.PerspectiveCamera(camera['yfov'], aspectRatio=camera['aspec'])
        if camera['postype'] == 'matrix':
            camera_pose = camera['matrix']
        elif camera['postype'] == 'angles':
            camera_pose = get_matrix(camera['yaw'], camera['pitch'], camera['roll'], camera['pos'])
        else:
            raise NotImplementedError(f"Camera postype {camera['postype']}")
        scene.add(camera_instance, pose=camera_pose)
    # axis_trans[:3,3] = scene.centroid[:]
    # axis = creation.axis(10, transform=axis_trans)
    # axis = pyrender.Mesh.from_trimesh(axis)
    # scene.add(axis)


    return pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)


def  mesh_to_bbox(sence:trimesh.Scene):
    bbox_sence = trimesh.Scene()

    for name in sence.geometry:
        mesh:trimesh.Trimesh = sence.geometry[name]
        bbox_sence.add(mesh.bound)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='render ccobjs')

    parser.add_argument('-d', '--data-root', type=str, help='Data 目录位置')
    parser.add_argument('-x',  '--export-xml', type=str, help='导出的xml文件位置')
    parser.add_argument('-o', '--outdir', 	type=str, 	help='保存的目录')
    parser.add_argument('--depth', default=False, action='store_true', help='是否保存深度信息')
    parser.add_argument('--bgcolor', 	type=float, nargs=4, default=(0.0, 0.0, 0.0, 0), help='背景颜色')
    parser.add_argument('--sub', type=int, default=None)

    args = parser.parse_args()

    cameras = parser_ccxml(args.export_xml, 
                 os.path.join(args.data_root, '..', 'metadata.xml'))
    sub = None if args.sub is None else slice(args.sub)
    objs = load_productions(args.data_root, sub=sub)

    main_loop(objs, cameras, args.outdir, save_depth=args.depth, bgcolor=args.bgcolor)

    # for camera in cameres:
    #     print(camera)
    #     break

    # view_ccobjs(r"Z:\CCProject\Projects\2023 Shanghai\Productions\Production_6\Data", 
    #             sub=slice(100))

    