import bpy

# 调整这些路径为你自己的文件路径
obj_path = (
    r"e:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\Tile_+000_+010.obj"
)
output_path = r"e:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\ssss.obj"
texture_path = r"E:\Data\上海项目\0419万达\Production_obj_4\Data\Tile_+000_+010\Tile_+000_+010_0.jpg"

# 清空当前场景
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# 导入OBJ文件
bpy.ops.import_scene.obj(filepath=obj_path)
model_a = bpy.context.selected_objects[0]

# 简化模型（修改这里的ratio值以控制简化的程度）
bpy.ops.object.select_all(action="DESELECT")
model_a.select_set(True)
bpy.context.view_layer.objects.active = model_a
bpy.ops.object.modifier_add(type="DECIMATE")
bpy.context.object.modifiers["Decimate"].ratio = 0.5  # 简化50%
bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Decimate")

# 重命名简化后的模型为模型B
model_b = model_a
model_b.name = "Model_B"

# 准备烘焙（假定模型已经包含一个材质和对应的纹理）
# 您可能需要根据具体情况调整这些设置
bpy.ops.object.select_all(action="DESELECT")
model_b.select_set(True)
bpy.context.view_layer.objects.active = model_b
bpy.ops.object.bake(type="DIFFUSE", use_clear=True, margin=2)

# 将简化的模型保存为新的OBJ文件
bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)

# 保存纹理
# 注意：这里的纹理保存过程可能需要手动进行或使用特定的脚本代码来完成，
# 因为直接从烘焙操作保存纹理的自动化流程可能涉及对材质节点的访问和操作。
