from pyper import task
import geopandas as gpd
import shapely.geometry as sg
import numpy as np
import pyproj
import os
from osgeo import gdal
import typing
from dataclasses import dataclass
import tqdm
from rasterio import features


@dataclass
class ZoneStatsInput:
    geom: sg.Point | sg.Polygon
    raster_path: str
    attribute: dict
    crs: typing.Optional[pyproj.CRS] = None


@dataclass
class ZoneStatsOutput:
    geom: sg.Point | sg.Polygon
    attribute: dict
    stats: list[dict]
    raster_path: str
    crs: typing.Optional[pyproj.CRS] = None

    def to_dict(self):
        result = self.attribute.copy()
        result["raster_path"] = self.raster_path
        for band_index, band_stats in enumerate(self.stats, start=1):
            for stat_name, stat_value in band_stats.items():
                result[f"band_{band_index}_{stat_name}"] = stat_value
        result["geometry"] = self.geom
        return result


def zone_sts_one_geom(
    inputs: ZoneStatsInput,
    bands: typing.Union[list, str] = "all",
    stats: typing.Optional[list] = None,
    nodata: typing.Optional[float] = None,
) -> ZoneStatsOutput:
    """Calculate zonal statistics for a single geometry.

    Parameters
    ----------
    geom : shapely.geometry, Point or Polygon
        The geometry to calculate zonal statistics for.
    raster_path : str
        Path to the raster file.
    bands : list or 'all', optional
        List of band indices (1-based) to calculate statistics for, or 'all' to use all bands. Default is 'all'.
    stats : list, optional
        List of statistics to calculate. Supported statistics are:
        'min', 'max', 'mean', 'median', 'std', 'sum', 'count', 'nodata_count'.
        Default is None, which calculates all supported statistics.

    Returns
    -------
    dict
        Dictionary containing the calculated statistics for each band and statistic.
    """
    raster_path = inputs.raster_path
    # Open the raster dataset
    ds = gdal.Open(raster_path)

    # Get the number of bands in the raster
    num_bands = ds.RasterCount
    if bands == "all":
        bands = list(range(1, num_bands + 1))
    if stats is None:
        stats = ["min", "max", "mean", "median", "std", "sum", "count", "nodata_count"]

    geo = ds.GetGeoTransform()
    bound = sg.Polygon.from_bounds(
        geo[0],
        geo[3] + geo[5] * ds.RasterYSize,
        geo[0] + geo[1] * ds.RasterXSize,
        geo[3],
    )
    geom = inputs.geom
    if not bound.contains(geom):
        results = [
            {stat: None for stat in stats} if stats is not None else {} for _ in bands
        ]
        return ZoneStatsOutput(
            geom=geom,
            attribute=inputs.attribute,
            stats=results,
            raster_path=raster_path,
        )

    # If stats is None, use all supported statistics

    # Create a dictionary to store the results
    results = []

    # Loop through each band and calculate the statistics

    for band_index in bands:
        # Get the band object
        band = ds.GetRasterBand(band_index)
        # Get the band data
        if isinstance(geom, sg.Point):
            # For point geometries, sample the raster value at the point location
            gt = ds.GetGeoTransform()
            px = int((geom.x - gt[0]) / gt[1])
            py = int((geom.y - gt[3]) / gt[5])
            data = band.ReadAsArray(px, py, 1, 1).flatten()
        elif isinstance(geom, sg.Polygon):
            geom_bounds = geom.bounds
            gt = ds.GetGeoTransform()

            # Convert the geometry bounds to pixel coordinates
            min_x = int((geom_bounds[0] - gt[0]) / gt[1])
            max_x = int((geom_bounds[2] - gt[0]) / gt[1])
            max_y = int((geom_bounds[1] - gt[3]) / gt[5])
            min_y = int((geom_bounds[3] - gt[3]) / gt[5])
            data = band.ReadAsArray(
                min_x, min_y, max_x - min_x, max_y - min_y
            ).flatten()
            # TODO: mask the data array with the geometry
            # mask = features.rasterize(
            #     [(geom, 1)],
            #     out_shape=(max_y - min_y, max_x - min_x),
            #     transform=(
            #         gt[1],
            #         0,
            #         gt[0] + min_x * gt[1],
            #         0,
            #         gt[5],
            #         gt[3] + max_y * gt[5],
            #     ),
            # )
            # print(mask)
            # data = data[mask.flatten() == 1]

        else:
            raise ValueError("Geometry must be a Point or Polygon")
        # Handle nodata values
        if nodata is not None:
            data = data[data != nodata]
        # Calculate the statistics
        band_stats = {}
        for stat in stats:
            if stat == "min":
                band_stats["min"] = np.nanmin(data)
            elif stat == "max":
                band_stats["max"] = np.nanmax(data)
            elif stat == "mean":
                band_stats["mean"] = np.nanmean(data)
            elif stat == "median":
                band_stats["median"] = np.nanmedian(data)
            elif stat == "std":
                band_stats["std"] = np.nanstd(data)
            elif stat == "sum":
                band_stats["sum"] = np.nansum(data)
            elif stat == "count":
                band_stats["count"] = len(data)
            elif stat == "nodata_count":
                band_stats["nodata_count"] = np.sum(data == nodata)
        results.append(band_stats)

    return ZoneStatsOutput(
        geom=geom,
        attribute=inputs.attribute,
        stats=results,
        raster_path=raster_path,
        crs=inputs.crs,
    )


def generate_geoms_from_file(
    file_path: str, raster_path: str, buffer: float = 0
) -> typing.Generator[ZoneStatsInput, None, None]:
    gdf = gpd.read_file(file_path)
    ds = gdal.Open(raster_path)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    proj_obj = pyproj.CRS.from_wkt(proj)
    gdf = gdf.to_crs(proj_obj.to_epsg())

    raster_bound = sg.Polygon.from_bounds(
        geo[0],
        geo[3] + geo[5] * ds.RasterYSize,
        geo[0] + geo[1] * ds.RasterXSize,
        geo[3],
    )
    ds = None

    # print(raster_bound)
    gdf = gdf[gdf.intersects(raster_bound)]

    if buffer > 0:
        gdf.geometry = (
            gdf.to_crs(epsg=3857).buffer(buffer).to_crs(epsg=proj_obj.to_epsg())
        )

    for _, row in gdf.iterrows():
        # print(idx)
        geom = row.geometry
        attribute = row.drop(labels="geometry").to_dict()
        yield ZoneStatsInput(
            geom=geom,
            attribute=attribute,
            raster_path=raster_path,
            crs=proj_obj,
        )


def save_zone_stats(
    zone_stats: typing.Generator[ZoneStatsOutput, None, None], output_path: str
):
    results = []
    for zone_stat in zone_stats:
        results.append(zone_stat.to_dict())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf = gpd.GeoDataFrame.from_records(results)
    gdf.set_geometry("geometry", inplace=True)
    gdf.set_crs(epsg=zone_stat.crs.to_epsg(), inplace=True)
    gdf.to_file(output_path, driver="GPKG")

    return output_path


def main(
    vector_path,
    raster_path,
    bands="all",
    stats=None,
    nodata=None,
    buffer=0,
    output_path=None,
):
    pipeline = task(
        generate_geoms_from_file,
        branch=True,
    ) | task(
        zone_sts_one_geom,
        bind=task.bind(bands=bands, stats=stats, nodata=nodata),
        multiprocess=True,
        workers=os.cpu_count(),
    )

    zones = [
        zone
        for zone in tqdm.tqdm(
            pipeline(file_path=vector_path, raster_path=raster_path, buffer=buffer),
            total=10000,
        )
    ]
    if output_path is not None:
        save_zone_stats(zones, output_path)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main(
        vector_path="D:/points.gpkg",
        raster_path=r"D:\project\gedi-agb\data\shp1_2018\shp1_2018\embading_para_shp1_2018-0000000000-0000000000.tif",
        bands="all",
        stats=["min", "max", "mean", "median", "std", "sum", "count", "nodata_count"],
        nodata=None,
        buffer=20,
        output_path="D:/output.gpkg",
    )
