import trimesh
import trimesh.voxel.ops as voxel_ops

# 假设 points 是一个形状为 (n_points, 3) 的 NumPy 数组，包含了点云中每个点的 x, y, z 坐标
points = trimesh.load(
    "e:/Data/上海项目/0419万达/Production_obj_1/building/building_0.obj"
)
points = points.vertices

# 抽稀点云
# 这里我们通过将点云转换为一个体素模型，在将其转换回点来间接完成抽稀
# 首先，定义体素的大小（影响抽稀的程度）
voxel_size = 1.0
# 计算体素网格
voxel_grid = voxel_ops.points_to_marching_cubes(points, pitch=voxel_size)

# 从体素网格中提取顶点作为抽稀后的点云
downsampled_points = voxel_grid.vertices

# save to obj
trimesh.Trimesh(vertices=downsampled_points, faces=[]).export(
    "e:/Data/上海项目/0419万达/Production_obj_1/building/building_0_downsampled.obj"
)

# 可以打印查看结果
print(f"Original number of points: {len(points)}")
print(f"Downsampled number of points: {len(downsampled_points)}")
