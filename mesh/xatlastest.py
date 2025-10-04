import trimesh
import xatlas

# We use trimesh (https://github.com/mikedh/trimesh) to load a mesh but you can use any library.
mesh: trimesh.Trimesh = trimesh.load_mesh(
    "E:/Data/上海项目/0419万达/Production_obj_4/Data/Tile_+000_+010/Tile_+000_+010.obj"
)

# The parametrization potentially duplicates vertices.
# `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
# `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
# `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

# Trimesh needs a material to export uv coordinates and always creates a *.mtl file.
# Alternatively, we can use the `export` helper function to export the mesh as obj.
xatlas.export("output.obj", mesh.vertices[vmapping], indices, uvs)
