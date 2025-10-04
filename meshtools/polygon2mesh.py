import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import trimesh.creation as trimesh_creation
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import SimpleMaterial
from trimesh.bounds import oriented_bounds_2D
import trimesh
from coord_trans import coord_trans_inv
import os
import PIL.Image as Image
import random
import matplotlib.pyplot as plt

# import xmltodict
import tqdm

PICLIBS = {
    "door": [
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/door1.png",
            "width": 3,
            "height": 3,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/door2.png",
            "width": 3,
            "height": 3,
        },
    ],
    "window": [
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win1.png",
            "width": 2,
            "height": 2,
            "hgap": 1.4,
            "wgap": 2,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win2.png",
            "width": 2,
            "height": 2,
            "hgap": 1.4,
            "wgap": 2,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win3.png",
            "width": 2,
            "height": 2,
            "hgap": 1.4,
            "wgap": 2,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win4.png",
            "width": 2,
            "height": 2,
            "hgap": 1.4,
            "wgap": 2,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win5.png",
            "width": 2,
            "height": 2,
            "hgap": 1.4,
            "wgap": 2,
        },
        {
            "path": "D:/wt/misc-tools/meshtools/piclibs/win6.png",
            "width": 5,
            "height": 3,
            "hgap": 0.3,
            "wgap": 0.3,
        },
    ],
    "wall": [
        {"path": "D:/wt/misc-tools/meshtools/piclibs/w1.png", "type": "n1", "res": 1},
        {"path": "D:/wt/misc-tools/meshtools/piclibs/w2.png", "type": "n2", "res": 1},
        {"path": "D:/wt/misc-tools/meshtools/piclibs/w3.png", "type": "n3", "res": 1},
        {"path": "D:/wt/misc-tools/meshtools/piclibs/w4.png", "type": "n4", "res": 1},
    ],
}

ROOF_LIBS = [
    {
        "path": "D:/wt/misc-tools/meshtools/piclibs/roof1.png",
    },
    {
        "path": "D:/wt/misc-tools/meshtools/piclibs/roof2.png",
    },
    {
        "path": "D:/wt/misc-tools/meshtools/piclibs/roof3.png",
    },
    {
        "path": "D:/wt/misc-tools/meshtools/piclibs/roof4.png",
    },
    {
        "path": "D:/wt/misc-tools/meshtools/piclibs/roof5.png",
    },
]


def draw_nodoor(width, height, res=0.1):
    return draw_normal(width, height, res, False)


def get_wall(width, height, wtype="random"):
    walls = PICLIBS["wall"]
    if wtype == "random":
        wall = random.choice(walls)
    else:
        for w in walls:
            if w["type"] == wtype:
                wall = w
                break
    wall_img: Image.Image = Image.open(wall["path"]).convert("RGB")

    wall_img.resize(
        (int(wall_img.width * wall["res"]), int(wall_img.height * wall["res"]))
    )

    if width < wall_img.width and height < wall_img.height:
        wall_img = np.array(wall_img)
        wall_img = wall_img[0:height, 0:width]
        return wall_img
    wall_img = np.array(wall_img)
    wall_img_1 = wall_img
    wall_img_2 = np.flip(wall_img, 0)
    wall_img_3 = np.flip(wall_img, 1)
    wall_img_4 = np.flip(wall_img_2, 1)

    big_wall_img = np.concatenate((wall_img_1, wall_img_2), axis=0)
    big_wall_img1 = np.concatenate((wall_img_3, wall_img_4), axis=0)
    big_wall_img = np.concatenate((big_wall_img, big_wall_img1), axis=1)

    if width < big_wall_img.shape[1] and height < big_wall_img.shape[0]:
        wall_img = big_wall_img[0:height, 0:width]
        return wall_img

    if width < big_wall_img.shape[1]:
        big_wall_img = big_wall_img[:, 0:width]

    if height < big_wall_img.shape[0]:
        big_wall_img = big_wall_img[0:height, :]

    wall_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, width, big_wall_img.shape[1]):
        for j in range(0, height, big_wall_img.shape[0]):
            w = big_wall_img.shape[1]
            if i + big_wall_img.shape[1] > width:
                w = width - i
            h = big_wall_img.shape[0]
            if j + big_wall_img.shape[0] > height:
                h = height - j
            wall_img[j : j + h, i : i + w] = big_wall_img[0:h, 0:w]
    return wall_img


def draw_normal(width, height, door_cfg, window_cfg, wall_type, res=0.1, door=True):
    # door_cfg = random.choice(PICLIBS["door"])
    # window_cfg = random.choice(PICLIBS["window"])
    window_height = int((door_cfg["height"] + np.random.rand()) / res)
    window_width = int((door_cfg["width"] + np.random.rand()) / res)
    window_h_gap = int(window_cfg["hgap"] / res)
    window_w_gap = int(window_cfg["wgap"] / res)

    if door:
        door_height = int((3 + np.random.rand()) / res)
        door_width = int((3 + np.random.rand()) / res)
        if width < (door_width + 6):
            return get_wall(width, height, wall_type)
        door_start_row = np.random.randint(0, 3)
        door_start_col = np.random.randint(0, width - door_width - 3)
    else:
        door_height = int(0.3 / res)
        door_width = 0
        door_start_row = 0
        door_start_col = 0
    if height < door_start_row + door_height + window_height + 10:
        height = door_height + window_height + window_h_gap + 10
    if width < max(door_width, window_width) + 10:
        width = max(door_width, window_width) + 10
    normal = get_wall(width, height, wall_type)
    # normal[:, :] = [127, 127, 127]
    if door:
        door_img: Image.Image = Image.open(door_cfg["path"])
        door_img = door_img.convert("RGB").resize((door_width, door_height))

        normal[
            door_start_row : door_start_row + door_height,
            door_start_col : door_start_col + door_width,
            :,
        ] = np.flip((np.array(door_img)), axis=0)

    window_img = Image.open(window_cfg["path"])
    window_img = window_img.convert("RGB").resize((window_width, window_height))

    window_array = np.flip(np.array(window_img), 0)
    start_row = door_start_row + door_height + 10
    end_row = height - window_height - 10
    start_col = 10
    end_col = width - window_width - 10
    for i in range(start_row, end_row, window_height + window_h_gap):
        for j in range(start_col, end_col, window_width + window_w_gap):
            normal[i : i + window_height, j : j + window_width, :] = window_array

    return normal


def generate_wall_matrial(
    wall: trimesh.Trimesh, door_cfg, window_cfg, wall_type, wtype="normal"
):
    verteices = wall.vertices
    if len(verteices) < 4:
        return wall
    p1 = verteices[0, :2]
    p2 = verteices[1, :2]
    width = np.linalg.norm(p1 - p2)
    if width < 3:
        return wall
    height = np.max(verteices[:, 2]) - np.min(verteices[:, 2])

    img_width = int(width / 0.1)
    img_height = int(height / 0.1)
    try:
        if wtype == "normal":
            img = draw_normal(
                img_width, img_height, door_cfg, window_cfg, wall_type, res=0.1
            )
        elif wtype == "nodoor":
            img = draw_normal(
                img_width,
                img_height,
                door_cfg,
                window_cfg,
                wall_type,
                res=0.1,
                door=False,
            )
        else:
            img = draw_normal(
                img_width, img_height, door_cfg, window_cfg, wall_type, res=0.1
            )
    except Exception as e:
        print(e)
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8) + 127

    img = np.flip(img, axis=0)
    uv = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ]
    uv = np.array(uv)
    image = Image.fromarray(img)
    image.format = "jpg"

    wall.visual = TextureVisuals(
        uv=uv,
        material=SimpleMaterial(
            image=image,
            diffuse=(255, 255, 255, 255),
            ambient=(255, 255, 255, 255),
            specular=(255, 255, 255, 255),
            glossiness=1.0,
        ),
    )
    return wall


def generate_roof(roof: trimesh.Trimesh):
    verteices = roof.vertices

    min_x = np.min(verteices[:, 0])
    min_y = np.min(verteices[:, 1])
    max_x = np.max(verteices[:, 0])
    max_y = np.max(verteices[:, 1])

    uvs = verteices[:, :2] - np.array([min_x, min_y])
    uvs[:, 0] /= max_x - min_x
    uvs[:, 1] /= max_y - min_y

    trans, extents = oriented_bounds_2D(uvs)
    uvs = trimesh.transform_points(uvs, trans)

    min_uv = np.min(uvs, axis=0)
    max_uv = np.max(uvs, axis=0)
    uvs[:, 0] -= min_uv[0]
    uvs[:, 1] -= min_uv[1]
    uvs[:, 0] /= max_uv[0] - min_uv[0]
    uvs[:, 1] /= max_uv[1] - min_uv[1]

    # roof.visual.uv = uvs
    image = random.choice(ROOF_LIBS)
    image = Image.open(image["path"]).convert("RGB")

    roof.visual = TextureVisuals(
        uv=uvs,
        material=SimpleMaterial(
            image=image,
        ),
    )
    return roof


def make_light(m):
    if isinstance(m, trimesh.Trimesh):
        v = m.visual
        if isinstance(v, trimesh.visual.TextureVisuals):
            if isinstance(v.material, SimpleMaterial):
                v.material.diffuse = np.array([1.0, 1.0, 1.0, 1.0])
                v.material.ambient = np.array([1.0, 1.0, 1.0, 1.0])
                v.material.specular = np.array([1.0, 1.0, 1.0, 1.0])
                v.material.glossiness = 1

                max_length = 3000
                if v.material.image.width > max_length:
                    v.material.image = v.material.image.resize(
                        (
                            max_length,
                            int(
                                max_length
                                * v.material.image.height
                                / v.material.image.width
                            ),
                        )
                    )
                if v.material.image.height > max_length:
                    v.material.image = v.material.image.resize(
                        (
                            int(
                                max_length
                                * v.material.image.width
                                / v.material.image.height
                            ),
                            max_length,
                        )
                    )

            elif hasattr(v.material, "to_simple"):
                v.material = v.material.to_simple()
                v.material.diffuse = np.array([1.0, 1.0, 1.0, 1.0])
                v.material.ambient = np.array([1.0, 1.0, 1.0, 1.0])
                v.material.specular = np.array([1.0, 1.0, 1.0, 1.0])

                v.material.glossiness = 1
                max_length = 3000
                if v.material.image.width > max_length:
                    v.material.image = v.material.image.resize(
                        (
                            max_length,
                            int(
                                max_length
                                * v.material.image.height
                                / v.material.image.width
                            ),
                        )
                    )
                if v.material.image.height > max_length:
                    v.material.image = v.material.image.resize(
                        (
                            int(
                                max_length
                                * v.material.image.width
                                / v.material.image.height
                            ),
                            max_length,
                        )
                    )

    elif isinstance(m, trimesh.Scene):
        for mi in m.geometry:
            make_light(m.geometry[mi])
    else:
        return


def polygon2mesh(polygon: Polygon, height=20, bottom=12, srs_dict=None):
    walls = []
    if polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            mesh = polygon2mesh(poly, height=height, bottom=bottom)
            walls.append(mesh)
        mesh = trimesh.util.concatenate(walls)
        # make_light(mesh)
        return mesh

    if polygon.exterior.coords.xy[0].__len__() < 3:
        return trimesh.Trimesh()
    door_number = [0]
    max_doors = 3
    door_cfg = random.choice(PICLIBS["door"])
    window_cfg = random.choice(PICLIBS["window"])
    if height < 15:
        wall_type = random.choice(["n1", "n2", "n3", "n4"])
    else:
        wall_type = random.choice(["n1", "n2", "n3"])

    def gen_walls(points, inv=False, has_door=False):
        for i in range(len(points)):
            p1 = points[i, :]
            p2 = points[(i + 1) % len(points), :]  # 循环到第一个点
            # 构建墙的四个角点

            if inv:
                pp = [
                    [p2[0], p2[1], bottom],
                    [p1[0], p1[1], bottom],
                    [p1[0], p1[1], height],
                    [p2[0], p2[1], height],
                ]
            else:
                pp = [
                    [p1[0], p1[1], bottom],
                    [p2[0], p2[1], bottom],
                    [p2[0], p2[1], height],
                    [p1[0], p1[1], height],
                ]
            # print(pp)
            pp = np.array(pp)
            # print(pp)
            if srs_dict is not None:
                pp = coord_trans_inv(pp, srs_dict)
            # 生成墙面
            wall = trimesh.Trimesh(vertices=pp, faces=[[0, 1, 2], [0, 2, 3]])

            has_door = door_number[0] < max_doors and np.random.rand() > 0.5
            if has_door:
                wall = generate_wall_matrial(
                    wall,
                    door_cfg=door_cfg,
                    window_cfg=window_cfg,
                    wall_type=wall_type,
                    wtype="normal",
                )
                door_number[0] = door_number[0] + 1
            else:
                wall = generate_wall_matrial(
                    wall,
                    door_cfg=door_cfg,
                    window_cfg=window_cfg,
                    wall_type=wall_type,
                    wtype="nodoor",
                )

            walls.append(wall)

    points = polygon.exterior.coords.xy
    points = np.array(points).T
    gen_walls(points, True)
    for i in polygon.interiors:
        points = np.array(i.coords.xy).T
        gen_walls(points, True, has_door=False)
    # print(points.shape)
    vertices, faces = trimesh_creation.triangulate_polygon(polygon)
    vertices = np.insert(vertices, 2, height, axis=1)
    if srs_dict is not None:
        vertices = coord_trans_inv(vertices, srs_dict)
    roof = trimesh.Trimesh(vertices=vertices, faces=faces)
    roof = generate_roof(roof)
    vertices[:, 2] = np.min(walls[0].vertices[:, 2])
    inv_faces = np.flip(faces, axis=1)
    bottom = trimesh.Trimesh(vertices=vertices, faces=inv_faces)
    mesh = trimesh.util.concatenate(walls + [roof, bottom])
    # make_light(mesh)
    return mesh


def run_polygon_build(path, out_dir):
    # polygons: gpd.GeoDataFrame = gpd.read_file("F:/data/subbuild1_dsm.shp")
    polygons: gpd.GeoDataFrame = gpd.read_file(path)
    # centerx = polygons.geometry.centroid.x
    # centery = polygons.geometry.centroid.y
    centerx_mean = polygons.geometry.centroid.x.mean()
    centery_mean = polygons.geometry.centroid.y.mean()

    srs_dict = {"ModelMetadata": {"SRS": f"ENU:{centery_mean},{centerx_mean}"}}
    # polygons = gpd_coord_trans_inv(polygons, srs_dict)
    # out_dir = "F:/data/meshes"
    out_data = os.path.join(out_dir, "Data")
    os.makedirs(out_data, exist_ok=True)
    # write metadata.xml
    with open(os.path.join(out_dir, "metadata.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<ModelMetadata version="1.0">\n')
        f.write(f"\t<SRS>ENU:{centery_mean},{centerx_mean}</SRS>\n")
        f.write("\t<SRSOrigin>0,0,0</SRSOrigin>\n")
        f.write("\t<Texture>\n\t\t<ColorSource>Visible</ColorSource>\n\t</Texture>\n")
        f.write("</ModelMetadata>\n")

    for row in tqdm.tqdm(polygons.itertuples(), total=len(polygons)):
        polygon = row.geometry
        # print(row["dsmmax"])
        mesh = polygon2mesh(polygon, height=row.dsmmax, bottom=12, srs_dict=srs_dict)
        objdir = os.path.join(out_data, f"building_{row.Index}")
        os.makedirs(objdir, exist_ok=True)
        make_light(mesh)
        mesh.export(os.path.join(objdir, f"building_{row.Index}.obj"))
        # break


if __name__ == "__main__":
    run_polygon_build("F:/data/subbuild1_dsm.shp", "F:/data/meshes")
