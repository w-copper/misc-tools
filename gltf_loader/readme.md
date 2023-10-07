# 修复trimesh gltf问题

修复了trimesh无法识别带有 KHR_draco_mesh_compression extension的问题，使用DracoPy对buffer进行解码

注意本文件夹下的 gltf.py 来自于trimesh/exchange/gltf.py ，其中修复了上述提到的问题，可将其与原trimesh库中的文件进行覆盖
