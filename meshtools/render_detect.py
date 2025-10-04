import argparse
import os
import ultralytics
import trimesh
import numpy as np
import geopandas as gpd
import shapely.geometry as geom
import xmltodict
import PIL.Image as Image
import json
import pickle
from collections import defaultdict
from scipy.spatial import KDTree
from tqdm import tqdm
from mesh_tools import (
    render_one,
    rcd_to_xyz,
    get_orthographic_visual_geom,
    load_ccobjs_to_trimesh,
    load_objs_box,
)
from coord_trans import coord_trans, gpd_coord_trans_inv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = None


def detect_one_model(color, depth, camera, yolo_model, fname=None):
    global device
    if device is None:
        try:
            result = yolo_model(color, imgsz=640, device="cuda")[0]
            device = "cuda"
        except:
            result = yolo_model(color, imgsz=640, device="cpu")[0]
            device = "cpu"
    else:
        result = yolo_model(color, imgsz=640, device=device)[0]
    detection_results = defaultdict(list)
    if fname is not None:
        rimage = result.plot(conf=True, boxes=True)
        rimage = Image.fromarray(rimage)
        rimage.save(f"{fname}_pred.jpg")

    boxes = result.boxes
    if boxes.shape[0] == 0:
        return detection_results

    xywh = boxes.xywh.cpu().numpy()
    nan_idx = np.isnan(xywh).any(axis=1)
    xy = xywh[:, :2]
    xy = xy[~nan_idx, :]
    if xy.shape[0] == 0:
        return detection_results

    xy = xy.astype(int)
    cate = boxes.cls.cpu().numpy().astype(int)
    cate = cate[~nan_idx]
    cate_name = [result.names[c] for c in cate]

    xyz = rcd_to_xyz(xy, depth, camera)
    if xyz is None:
        return detection_results
    if xyz.shape[0] == 0:
        return detection_results
    for j in range(len(cate_name)):
        if np.isnan(xyz[j, 0]):
            continue
        detection_results[cate_name[j]].append(xyz[j, None, :])
    return detection_results


def render_detect(scene, camera, yolo_model=None, save=False, outpath="./render_out"):
    if yolo_model is None:
        yolo_model = []
    color, depth, dsm = render_one(camera, scene)
    color = Image.fromarray(color)
    results = []
    if save:
        os.makedirs(outpath, exist_ok=True)
        fname = os.path.join(outpath, f"render_{int(np.random.rand() * 100000)}")
        color.save(f"{fname}_color.jpg")
        Image.fromarray((depth * 255).astype(np.uint8)).save(f"{fname}_depth.jpg")
        with open(f"{fname}_camera.pkl", "wb") as f:
            pickle.dump(camera, f)
    else:
        fname = None
    for i, model in enumerate(yolo_model):
        result = detect_one_model(
            color, depth, camera, model, fname=f"{fname}_{str(i)}" if save else None
        )
        results.append(result)
    return results


def segment_line(line, max_length=50):
    # 分割LineString，使每段长度不超过max_length
    distances = []
    points = list(line.coords)
    for i in range(1, len(points)):
        p1, p2 = geom.Point(points[i - 1]), geom.Point(points[i])
        distance = p1.distance(p2)
        count = int(np.ceil(distance / max_length))
        segment_length = distance / count
        distances.extend([segment_length] * count)
    print(distances)
    new_points = [points[0]]
    for i in range(1, len(distances) + 1):
        # prev_point = geom.Point(new_points[-1])
        next_point = line.interpolate(sum(distances[:i]))
        new_points.append(next_point.coords[0])

    return geom.LineString(new_points)


def generate_surrounding_points(polygon: geom.Polygon, max_distance=50):
    if isinstance(polygon, geom.Polygon):
        rotated_rect = polygon.minimum_rotated_rectangle
        exterior_coords = list(rotated_rect.exterior.coords)

        # 旋转矩形的四个顶点，去掉重复的最后一个点
        corners = exterior_coords[:-1]

        # 计算四边中点
        midpoints = [
            [
                (corners[i][0] + corners[(i + 1) % 4][0]) / 2.0,
                (corners[i][1] + corners[(i + 1) % 4][1]) / 2.0,
            ]
            for i in range(4)
        ]

        edges = [
            (corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))
        ]
        additional_points = []

        for edge in edges:
            line = geom.LineString([edge[0], edge[1]])
            edge_length = line.length
            if edge_length > max_distance:
                num_points_to_add = int(np.floor(edge_length / max_distance) - 1)
                for i in range(1, num_points_to_add + 1):
                    point = line.interpolate(
                        (edge_length / (num_points_to_add + 1)) * i
                    )
                    additional_points.append((point.x, point.y))

        # 合并顶点和中点作为结果
        return corners + midpoints + additional_points

    elif isinstance(polygon, geom.LineString):
        line = segment_line(polygon, max_length=max_distance)
        points = list(line.coords)
        directions = []
        for i in range(len(points)):
            if i == 0:  # 起点，只有后方向
                p1 = geom.Point(points[i])
                p2 = geom.Point(points[i + 1])
                vector = (p2.x - p1.x, p2.y - p1.y)
                left_direction = (-vector[1], vector[0])
                right_direction = (vector[1], -vector[0])
            elif i == len(points) - 1:  # 终点，只有前方向
                p0 = geom.Point(points[i - 1])
                p1 = geom.Point(points[i])
                vector = (p1.x - p0.x, p1.y - p0.y)
                left_direction = (-vector[1], vector[0])
                right_direction = (vector[1], -vector[0])
            else:
                p0 = geom.Point(points[i - 1])
                p1 = geom.Point(points[i])
                p2 = geom.Point(points[i + 1])
                vector1 = (p1.x - p0.x, p1.y - p0.y)
                left_direction = (-vector1[1], vector1[0])
                right_direction = (vector1[1], -vector1[0])

            directions.append(
                ({"point": points[i], "left": left_direction, "right": right_direction})
            )

        return directions


def run_polylines(
    pmesh: gpd.GeoDataFrame,
    ppoly: geom.LineString,
    distance_break=50,
    resultion=0.01,
    yolo_models=None,
    save_render=False,
    render_out=None,
):
    if yolo_models is None:
        yolo_models = []
    directions = generate_surrounding_points(ppoly, max_distance=distance_break)
    detect_results = []
    for direction in directions:
        point = direction["point"]
        left_direction = direction["left"]
        right_direction = direction["right"]
        yvec = np.array([0, 0, 1])
        for zv in [left_direction, right_direction]:
            zvec = np.array([zv[0], zv[1], 0])
            zvec = zvec / np.linalg.norm(zvec)
            xvec = np.cross(yvec, zvec)
            xvec = xvec / np.linalg.norm(xvec)
            center = np.array([point[0], point[1], 0])
            mt = np.eye(4)
            mt[:3, 0] = xvec
            mt[:3, 1] = yvec
            mt[:3, 2] = zvec
            mt[:3, 3] = center
            vg = get_orthographic_visual_geom(
                distance_break,
                20,
                0.1,
                100,
                matrix=mt,
            )
            bounds_polygon = geom.Polygon(
                [
                    [vg.bounds[0][0], vg.bounds[0][1]],
                    [vg.bounds[1][0], vg.bounds[0][1]],
                    [vg.bounds[1][0], vg.bounds[1][1]],
                    [vg.bounds[0][0], vg.bounds[1][1]],
                ]
            )
            vis_objs = pmesh.intersects(bounds_polygon)
            if vis_objs.sum() > 0:
                vis_objs = pmesh[vis_objs]
                scene = trimesh.Scene([trimesh.load(f) for f in vis_objs["obj"]])
                bounds = scene.bounds
                matrix = np.eye(4)
                zmean = np.mean(bounds[:, 2])
                matrix[:3, 0] = xvec
                matrix[:3, 1] = yvec
                matrix[:3, 2] = zvec
                matrix[:3, 3] = np.array([point[0], point[1], zmean])
                camera = {
                    "type": "orthographic",
                    "xmag": distance_break,
                    "ymag": (bounds[1, 2] - bounds[0, 2]) / 2,
                    "width": int(distance_break / resultion),
                    "zfar": np.sqrt(np.sum((bounds[1, :2] - bounds[0, :2]) ** 2)),
                    "height": int((bounds[1, 2] - bounds[0, 2]) / 2 / resultion),
                    "postype": "matrix",
                    "matrix": matrix,
                }
                dr = render_detect(
                    scene, camera, yolo_models, save=save_render, outpath=render_out
                )

                detect_results.extend(dr)
    return detect_results


def run_single_poygon(
    meshes,
    polygon: geom.Polygon,
    distance_break=50,
    resultion=0.01,
    yolo_models=None,
    save_render=False,
    render_out=None,
):
    if yolo_models is None:
        yolo_models = []
    meshes = [trimesh.load(f) for f in meshes]
    sence = trimesh.Scene(meshes)

    bounds = sence.bounds
    zmean = np.mean(bounds[:, 2])

    buffer_poly = polygon.buffer(1, cap_style="round", join_style="round")
    detect_results = []
    for point in generate_surrounding_points(buffer_poly, distance_break):
        point = np.array(point)
        yvec = np.array([0, 0, 1])
        zve = point - np.array(buffer_poly.centroid.xy).reshape(2)
        # print(zve)
        zvec = np.array([zve[0], zve[1], 0])
        zvec = zvec / np.linalg.norm(zvec)
        xvec = np.cross(yvec, zvec)
        xvec = xvec / np.linalg.norm(xvec)

        matrix = np.eye(4)
        matrix[:3, 0] = xvec
        matrix[:3, 1] = yvec
        matrix[:3, 2] = zvec
        matrix[:3, 3] = np.array([point[0], point[1], zmean])
        matrix[3, 3] = 1
        camera = {
            "type": "orthographic",
            "xmag": distance_break,
            "ymag": (bounds[1, 2] - bounds[0, 2]) / 2,
            "width": int(distance_break / resultion),
            "zfar": np.sqrt(np.sum((bounds[1, :2] - bounds[0, :2]) ** 2)),
            "height": int((bounds[1, 2] - bounds[0, 2]) / 2 / resultion),
            "postype": "matrix",
            "matrix": matrix,
        }
        dr = render_detect(
            sence, camera, yolo_models, save=save_render, outpath=render_out
        )

        detect_results.extend(dr)

    return detect_results


def merge_close_points(points, distance_threshold):
    """
    使用KD树优化合并距离很近的点。

    :param points: numpy数组，形状为(n, 3)。
    :param distance_threshold: 合并点的距离阈值。
    :return: 合并后的点和对应的索引映射。
    """
    # 创建KD树
    tree = KDTree(points)

    # 初始化一个集合，用来存放合并后的点
    merged_points = []
    # 初始化一个字典，用来存放点的新旧索引映射关系
    index_mapping = {}

    # 遍历所有点
    for i, point in enumerate(points):
        # 如果点还没有被合并
        if i not in index_mapping:
            # 查找所有与当前点距离小于阈值的点
            nearby_indices = tree.query_ball_point(point, distance_threshold)
            # 将当前点添加到合并后的点集
            merged_points.append(point)
            # 更新映射关系
            for j in nearby_indices:
                index_mapping[j] = len(merged_points) - 1

    return np.array(merged_points)


def merge_detections(detection_results):
    origin_results = defaultdict(list)
    # print(detection_results)
    for result in detection_results:
        if result is None:
            continue
        for k, v in result.items():
            origin_results[k].extend(v)

    results = []

    for k, v in origin_results.items():
        if len(v) == 0:
            continue
        # xyz = np.array(v
        xyz = np.concatenate(v, axis=0)
        if xyz.shape[0] > 1:
            xyz = merge_close_points(xyz, 1)
        results.extend(
            {"name": k, "x": xyz[i, 0], "y": xyz[i, 1], "z": xyz[i, 2]}
            for i in range(xyz.shape[0])
        )
    return results


def batch_run_polygons(objroot, polyinput, model, save_render=False, render_out=None):
    metadata_path = os.path.join(objroot, "metadata.xml")
    srs_dict = xmltodict.parse(open(metadata_path, "r", encoding="utf-8").read())
    if polyinput.endswith(".txt"):
        polys = []
        with open(args.inputpath, "r", encoding="utf-8") as f:
            for line in f:
                points = np.array(list(map(float, line.split(",")))).reshape(-1, 3)
                polygon = geom.LineString(points[:, :2])
                polys.append(polygon)
        gpp = gpd.GeoDataFrame(geometry=polys)
    elif (
        polyinput.endswith(".shp")
        or polyinput.endswith(".geojson")
        or polyinput.endswith(".gpkg")
        or polyinput.endswith(".kml")
    ):
        gpp = gpd.read_file(polyinput)
        gpp = gpd_coord_trans_inv(gpp, srs_dict)
    else:
        raise ValueError("polyinput must be a .txt or .shp file")
    ccobjs = load_ccobjs_to_trimesh(os.path.join(objroot, "Data"))
    boxes, zrange = load_objs_box(ccobjs)
    pbar = tqdm(total=len(gpp))
    results = []
    with open(model, "r", encoding="utf8") as f:
        model_dict = json.load(f)
    yolo_models = []
    for mp in model_dict["models"]:
        m = ultralytics.YOLO(os.path.join(os.path.dirname(model), mp))
        try:
            me = m.cuda()
        except:
            me = m
        m = me
        print(mp, m.names)
        yolo_models.append(m)
    for i, poly in gpp.iterrows():
        pbar.set_description(f"Processing {i+1}")
        if isinstance(poly["geometry"], geom.Polygon):
            insect = boxes.intersects(poly["geometry"])
            if insect.sum() > 0:
                objs = boxes[insect]["obj"].values
                result = run_single_poygon(
                    objs,
                    poly["geometry"],
                    yolo_models=yolo_models,
                    save_render=save_render,
                    render_out=render_out,
                )
                results.extend(result)
        elif isinstance(poly["geometry"], geom.LineString):
            result = run_polylines(
                boxes,
                poly["geometry"],
                yolo_models=yolo_models,
                save_render=save_render,
                render_out=render_out,
            )
            results.extend(result)
        else:
            print("WARNING: geometry type not supported")
            print(poly["geometry"])
        pbar.update(1)
    pbar.close()
    return merge_detections(results)


def format_save(results, objroot, outroot, polyinput, model):
    metadata_path = os.path.join(objroot, "metadata.xml")
    srs_dict = xmltodict.parse(open(metadata_path, "r", encoding="utf-8").read())
    os.makedirs(outroot, exist_ok=True)
    with open(model, "r", encoding="utf8") as f:
        model_dict = json.load(f)
        trans_dict = model_dict["objs"]
        # translation_names = dict((i, v) for i, v in enumerate(trans_dict.values()))
        colors = model_dict["colors"]
    with open(
        os.path.join(
            outroot,
            f"{os.path.splitext(os.path.basename(polyinput))[0]}_local_result.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            color = colors[r["name"]] if r["name"] in colors else [0, 0, 0]
            cate = trans_dict[r["name"]] if r["name"] in trans_dict else r["name"]
            f.write(
                f"{r['x']},{r['y']},{r['z']},{color[0]},{color[1]},{color[2]},{cate} \n"
            )
    with open(
        os.path.join(
            outroot,
            f"{os.path.splitext(os.path.basename(polyinput))[0]}_blh_result.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            color = colors[r["name"]] if r["name"] in colors else [0, 0, 0]
            cate = trans_dict[r["name"]] if r["name"] in trans_dict else r["name"]
            lon, lat = coord_trans(srs_dict, r["x"], r["y"], r["z"])
            f.write(f"{lon},{lat},{r['z']},{color[0]},{color[1]},{color[2]},{cate} \n")


def main(args):
    render_out = (
        args.render_out
        if args.render_out is not None
        else os.path.join(args.outroot, "render")
    )
    results = batch_run_polygons(
        args.objroot, args.polyinput, args.model, args.save_render, render_out
    )
    format_save(results, args.objroot, args.outroot, args.polyinput, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objroot", type=str, default="data")
    parser.add_argument("--outroot", type=str, default="data")
    parser.add_argument("--polyinput", type=str, default="data/polygons.shp")
    parser.add_argument("--model", type=str, default="data/models.json")
    parser.add_argument("--save_render", action="store_true", default=False)
    parser.add_argument("--render_out", type=str, default=None)
    args = parser.parse_args()
    main(args)
