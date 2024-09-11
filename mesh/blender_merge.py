import bpy
import os

# 请替换这些路径为你的OBJ文件的路径
obj_path_1 = (
    "e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/Tile_+000_+010.obj"
)
obj_path_2 = "e:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/box.obj"
output_path = "D:/wt/misc-tools/mesh/merge.obj"
bpy.ops.wm.read_homefile()
try:
    bpy.ops.object.mode_set(mode="OBJECT")
except BaseException:
    pass
# 删除场景中现有的所有对象
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# 导入第一个OBJ文件
bpy.ops.import_scene.obj(filepath=obj_path_1)

# 导入第二个OBJ文件
bpy.ops.import_scene.obj(filepath=obj_path_2)

# 合并逻辑
# 获取所有meshes
meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
# 确保至少有两个mesh对象可以合并
if len(meshes) < 2:
    print("需要至少两个mesh对象来合并。")
else:
    # 将每个对象的所有顶点转移到第一个mesh对象中
    ctx = bpy.context.copy()
    ctx["active_object"] = meshes[0]
    meshes[0].select_set(True)
    for obj in meshes[1:]:
        obj.select_set(True)
    bpy.ops.object.join(ctx)  # 合并选定的对象

# 导出合并后的OBJ文件，包括纹理
bpy.ops.export_scene.obj(filepath=output_path, use_materials=True, path_mode="RELATIVE")

print("合并并导出完成。导出文件路径：", output_path)
