import bpy
import sys
import os

operate = sys.argv[-4]
result_obj_path = sys.argv[-1]
obj_file_2 = sys.argv[-2]
obj_file_1 = sys.argv[-3]
# print(sys.argv)
try:
    bpy.ops.object.mode_set(mode="OBJECT")
except BaseException:
    pass
# 删除场景中现有的所有对象
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


def import_model(pth):
    pth = os.path.abspath(pth)
    if pth.endswith(".obj"):
        # 加载第一个OBJ文件到场景中
        bpy.ops.import_scene.obj(filepath=pth)
    elif pth.endswith(".fbx"):
        # 加载第一个FBX文件到场景中
        bpy.ops.import_scene.fbx(filepath=pth)
    elif pth.endswith(".gltf") or pth.endswith(".glb"):
        # 加载第一个GLTF文件到场景中
        bpy.ops.import_scene.gltf(filepath=pth)
    elif pth.endswith(".ply"):
        # 加载第一个PLY文件到场景中
        bpy.ops.import_mesh.ply(filepath=pth)
    elif pth.endswith(".stl"):
        # 加载第一个STL文件到场景中
        bpy.ops.import_mesh.stl(filepath=pth)


import_model(obj_file_1)
# 为了简化操作，我们假设导入的对象是场景中的第一个对象
obj1 = bpy.context.selected_objects[:]
import_model(obj_file_2)
# 同样假设导入的第二个对象是选中的对象
obj2 = bpy.context.selected_objects[0]


def bool_intersect(obj1, obj2):
    # 为第一个对象添加布尔修改器，并进行差集操作
    bool_modifier = obj1.modifiers.new(
        name="Bool Modifier " + str(obj1.name), type="BOOLEAN"
    )
    if operate.upper() == "INTERSECT":
        bool_modifier.operation = operate.upper()
        bool_modifier.use_hole_tolerant = True
        bool_modifier.solver = "EXACT"
    elif operate.upper() == "DIFFERENCE":
        bool_modifier.operation = operate.upper()
        bool_modifier.use_hole_tolerant = True
        bool_modifier.solver = "EXACT"

    bool_modifier.object = obj2  # 设置布尔操作的目标对象为第二个导入的对象
    bpy.context.view_layer.objects.active = obj1  # 将第一个对象设置为活动对象
    bpy.ops.object.modifier_apply(modifier=bool_modifier.name)


for obj in obj1:
    bool_intersect(obj, obj2)

# 删除第二个对象，因为我们已经完成了布尔差集操作
bpy.data.objects.remove(obj2, do_unlink=True)

if result_obj_path.endswith(".obj"):
    bpy.ops.export_scene.obj(filepath=result_obj_path, use_materials=True)
elif result_obj_path.endswith(".fbx"):
    bpy.ops.export_scene.fbx(filepath=result_obj_path, use_selection=True)
elif result_obj_path.endswith(".gltf") or result_obj_path.endswith(".glb"):
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(result_obj_path))
