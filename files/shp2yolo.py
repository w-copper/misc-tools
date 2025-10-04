import rasterio
from rasterio import windows
from rasterio.enums import Resampling
import geopandas as gpd
from itertools import product
from shapely.geometry import box
import numpy as np
import os

# 读取栅格文件
raster_path = "F:/DOM/Production_dsm_ortho_merge_32651.tif"
# raster_path = "F:/DOM/Production_domdsm_1_ortho_merge.tif"
dataset: rasterio.rasterio.DatasetReader = rasterio.open(raster_path)
out_dir = "F:/停车场-小区口标注/tiles_" + os.path.basename(raster_path).split(".")[0]
os.makedirs(out_dir, exist_ok=True)
# 读取矢量文件
vector_path = "F:/停车场-小区口标注/merged.geojson"
gdf: gpd.GeoDataFrame = gpd.read_file(vector_path)
# gdf = gdf.where((gdf["类型"] == "出入口") | (gdf["类型1"] == "出入口"))
gdf = gdf[(gdf["类型"] == "出入口") | (gdf["类型1"] == "出入口")]
print(((gdf["类型"] == "出入口") | (gdf["类型1"] == "出入口")).sum())
print(gdf)
tile_size = 1024

class_index_map = {"停车位": 0, "出入口": 1}


def get_class_index(class_name, class_name1):
    if class_name1 is None and class_name is None:
        return 0
    if class_name is None:
        return class_index_map[class_name1]
    return class_index_map[class_name]


def get_tiles(ds, width=512, height=512):
    nols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, nols, width // 2), range(0, nrows, height // 2))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        if col_off + width > nols:
            col_off = nols - width
        if row_off + height > nrows:
            row_off = nrows - height

        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def polygon_to_rotated_box(polygon):
    # 使用最小外接矩形获得旋转矩形，返回四个角的坐标
    return np.array(polygon.minimum_rotated_rectangle.exterior.coords[:-1])


for window, transform in get_tiles(dataset, width=tile_size, height=tile_size):
    # print(window, transform)
    # 裁剪栅格

    # 裁剪对应的矢量数据
    # 转换window为栅格坐标系的范围
    bounds = rasterio.windows.bounds(window, dataset.transform)
    bbox = box(*bounds)
    # print(bbox)
    # 选择落在当前窗口内的矢量对象
    clipped_gdf = gdf.clip(bbox)
    if len(clipped_gdf) == 0:
        continue

    data = dataset.read(window=window, resampling=Resampling.nearest)
    nodata = np.sum((data == 0), axis=0) == 3
    if nodata.sum() / nodata.size > 0.1:
        continue

    clipped_gdf.to_file(
        os.path.join(
            out_dir, f"tile_{int(window.col_off)}_{int(window.row_off)}.geojson"
        )
    )

    with rasterio.open(
        os.path.join(out_dir, f"tile_{int(window.col_off)}_{int(window.row_off)}.tif"),
        "w",
        crs=dataset.crs,
        transform=transform,
        driver="GTiff",
        count=3,
        dtype=dataset.dtypes[0],
        height=window.height,
        width=window.width,
    ) as outds:
        outds.write(data)
    # 对每个矢量对象转换坐标并保存旋转矩形
    label_lines = []
    for index, row in clipped_gdf.iterrows():
        if row["geometry"].area < 10:
            continue
        rotated_box = polygon_to_rotated_box(row["geometry"])
        # 坐标转换为像素
        pixel_coords = np.array([~transform * (x, y) for x, y in rotated_box])
        # 转换为相对坐标
        relative_coords = pixel_coords / tile_size
        relative_coords = np.clip(relative_coords, 0, 1)
        class_index = 0
        label_line = [class_index] + relative_coords.flatten().tolist()
        label_lines.append(" ".join(map(str, label_line)))

    # 保存标签文件
    with open(
        os.path.join(out_dir, f"tile_{int(window.col_off)}_{int(window.row_off)}.txt"),
        "w",
    ) as file:
        file.write("\n".join(label_lines))
