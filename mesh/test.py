import open3d as o3d
import open3d.visualization.rendering as rendering
import mesh_tools as mt
import trimesh
import matplotlib.pyplot as plt
import numpy as np

mesh = mt.load_mesh(r"D:\data\3dpointcloud\xuanen\Tile_+006_+005.obj")
o3dmesh = mesh.as_open3d()

bounds = mesh.bounds
center = mesh.centroid
dx, dy, dz = bounds[1,:] - bounds[0,:] 
center[2] = bounds[1,2] + 10 # Up 10m
extrinsic_matrix = mt.get_matrix(0, 0,0, center) # 外参矩阵 World -> camera
intrinsic_matrix = np.array( [[ 2/ dx, 0, 0,],
                            [0, 2/ dy, 0,] ,
                            [0, 0, -2/ dz,]
                            ] )

render = rendering.OffscreenRenderer(640, 480)
render.scene.add_geometry("cyl", o3dmesh)

render.setup_camera()
render.scene.show_axes(True)
img = render.render_to_image()

plt.imshow(img)
plt.show()