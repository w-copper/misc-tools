import bpy
import os

# 设定OBJ文件的路径
obj_file_1 = (
    r"e:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\Tile_+000_+010.obj"
)
obj_file_2 = r"e:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\box.obj"

# 导出结果的路径
result_obj_path = (
    r"e:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\result.obj"
)
bpy.ops.wm.read_homefile()
try:
    bpy.ops.object.mode_set(mode="OBJECT")
except BaseException:
    pass
# 删除场景中现有的所有对象
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# 加载第一个OBJ文件到场景中
bpy.ops.import_scene.obj(filepath=obj_file_1)

# 为了简化操作，我们假设导入的对象是场景中的第一个对象
obj1 = bpy.context.selected_objects[0]

# 加载第二个OBJ文件到场景中
bpy.ops.import_scene.obj(filepath=obj_file_2)

# 同样假设导入的第二个对象是选中的对象
obj2 = bpy.context.selected_objects[0]

# 为第一个对象添加布尔修改器，并进行差集操作
bool_modifier = obj1.modifiers.new(type="BOOLEAN", name="boolean_diff")
bool_modifier.object = obj2  # 设置布尔操作的目标对象为第二个导入的对象
bool_modifier.operation = "DIFFERENCE"

# 应用修改器
bpy.context.view_layer.objects.active = obj1  # 将第一个对象设置为活动对象
bpy.ops.object.modifier_apply(modifier=bool_modifier.name)

# 删除第二个对象，因为我们已经完成了布尔差集操作
bpy.data.objects.remove(obj2, do_unlink=True)

# 导出结果，带有纹理信息
bpy.ops.export_scene.obj(filepath=result_obj_path, use_materials=True)
