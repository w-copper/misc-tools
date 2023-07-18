import numpy as np

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