from concurrent import futures
from datetime import datetime
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import List, Tuple, Union

import geofileops as gfo
import geopandas as gpd
import pandas as pd

from cropclassification.helpers import pandas_helper as pdh
from . import _general_helper as general_helper
from . import _raster_helper as raster_helper
from . import _vector_helper as vector_helper
from . import Statistic

# Set path for qgis
qgis_path = Path(os.environ["CONDA_PREFIX"]) / "Library/python"
sys.path.insert(0, str(qgis_path))
import qgis.core  # noqa: E402
import qgis.analysis  # noqa: E402

logger = logging.getLogger(__name__)


def zonal_stats(
    vector_path: Path,
    columns: List[str],
    rasters_bands: List[Tuple[Path, List[str]]],
    output_dir: Path,
    stats: List[Statistic],
    nb_parallel: int = -1,
    force: bool = False,
):
    """
    Calculate zonal statistics.

    Args:
        features_path (Path): _description_
        id_column (str): _description_
        images_bands (List[Tuple[Path, List[str]]]): _description_
        stats (List[Statistic]): statistics to calculate.
        output_dir (Path): _description_
        nb_parallel (int, optional): the number of parallel processes to use.
            Defaults to -1: use all available processors.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    # Some checks on the input parameters
    if len(rasters_bands) == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.gettempdir()) / "zonal_stats_bulk_pyqgis"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=tmp_dir))

    start_time = datetime.now()
    nb_todo = 0
    nb_done_total = 0
    if nb_parallel < 1:
        nb_parallel += multiprocessing.cpu_count()

    # Loop over all images and bands to calculate zonal stats in parallel...
    calc_queue = {}
    pool = futures.ProcessPoolExecutor(nb_parallel)
    try:
        for raster_path, bands in rasters_bands:
            raster_info = raster_helper.get_image_info(raster_path)
            # Create base output filename
            # TODO: hoort hier niet echt thuis
            orbit = None
            if raster_info.imagetype.lower() == "s1-grd-sigma0-asc":
                orbit = "ASCENDING"
            elif raster_info.imagetype.lower() == "s1-grd-sigma0-desc":
                orbit = "DESCENDING"
            output_base_path = general_helper._format_output_path(
                vector_path,
                raster_path,
                output_dir,
                orbit,
                band=None,
            )
            output_base_busy_path = tmp_dir / f"BUSY_{output_base_path.name}"

            # Check for which bands there is a valid output file already
            if bands is None:
                bands = raster_info.bands
                assert bands is not None
            bands_todo = {}

            for band in bands:
                # Prepare the output paths...
                output_band_path = general_helper._format_output_path(
                    vector_path, raster_path, output_dir, orbit, band
                )
                output_band_busy_path = tmp_dir / f"BUSY_{output_band_path.name}"

                # If a busy output file exists, remove it, otherwise we can get
                # double data in it...
                if output_band_busy_path.exists():
                    output_band_busy_path.unlink()

                # Check if the output file exists already
                if output_band_path.exists():
                    if not force:
                        logger.debug(f"Output file for band exists {output_band_path}")
                        continue
                    else:
                        output_band_path.unlink()
                bands_todo[band] = output_band_path

                # If all bands already processed, skip image...
                if len(bands_todo) == 0:
                    logger.info(
                        "SKIP image: output files for all bands exist already for "
                        f"{output_base_path}"
                    )
                    continue

                nb_todo += 1
                future = pool.submit(
                    zonal_stats_band_tofile,
                    vector_path=vector_path,
                    raster_path=raster_path,
                    band=band,
                    stats=stats,
                    tmp_dir=tmp_dir,
                    columns=columns,
                    output_band_path=output_band_path,
                )
                calc_queue[future] = {
                    "vector_path": vector_path,
                    "raster_path": raster_path,
                    "bands": bands,
                    "bands_todo": bands_todo,
                    "prepare_calc_future": future,
                    "image_info": raster_info,
                    "prepare_calc_starttime": datetime.now(),
                    "output_base_path": output_base_path,
                    "output_base_busy_path": output_base_busy_path,
                    "status": "IMAGE_PREPARE_CALC_BUSY",
                }

        # Write progress for each completed calculation
        for future in futures.as_completed(calc_queue):
            try:
                _ = future.result()
            except Exception as ex:
                raise Exception(f"Error calculating {calc_queue[future]}: {ex}") from ex
            nb_done_total += 1
            progress_msg = general_helper._format_progress_message(
                nb_todo, nb_done_total, start_time
            )
            logger.info(progress_msg)
    finally:
        pool.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def zonal_stats_band(
    vector_path,
    raster_path: Path,
    band: str,
    tmp_dir: Path,
    stats: List[Statistic],
    columns: List[str],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    # Init
    stats_mask = None
    for stat in stats:
        if stats_mask is None:
            stats_mask = stat_to_qgisstat(stat)
        else:
            stats_mask |= stat_to_qgisstat(stat)

    # Get the image info
    image_info = raster_helper.get_image_info(raster_path)

    # Reproject the vector data
    tmp_dir.mkdir(exist_ok=True, parents=True)
    vector_proj_path = vector_helper.reproject_synced(
        path=vector_path,
        columns=columns + ["geometry"],
        target_epsg=image_info.image_epsg,
        dst_dir=tmp_dir,
    )
    layer = gfo.get_only_layer(vector_proj_path)

    # Init qgis
    # qgis.core.QgsApplication.setPrefixPath(str(qgis_path), True) ???
    qgs = qgis.core.QgsApplication([], False)
    qgs.initQgis()
    # Read the vector file + copy to memory layer for:
    #   - improved performance
    #   - QgsZonalStatistics actually adds the data to the input file, so copy needed
    vector = qgis.core.QgsVectorLayer(str(vector_proj_path), layer, "ogr")
    vector_mem = vector.materialize(
        qgis.core.QgsFeatureRequest().setFilterFids(vector.allFeatureIds())
    )
    del vector

    # Calculates zonal stats with raster
    raster = qgis.core.QgsRasterLayer(image_info.bands[band].path)

    # rasterband is the 1 based index in the raster file. Verify, as QgsZonalStatistics
    # doesn't give an error if this is wrong.
    band_index = image_info.bands[band].band_index
    if band_index < 1 or band_index > raster.bandCount():
        raise ValueError(f"invalid {band_index=} in {image_info.bands[band]}")

    try:
        zoneStats = qgis.analysis.QgsZonalStatistics(
            polygonLayer=vector_mem,
            rasterLayer=raster,
            attributePrefix="",
            rasterBand=band_index,
            stats=stats_mask,
        )
    except Exception:
        raise

    status = zoneStats.calculateStatistics(None)
    if zoneStats.calculateStatistics(None) != qgis.analysis.QgsZonalStatistics.Success:
        raise RuntimeError(
            f"Error: calculateStatistics returned {status} for zonal stats between "
            f"{vector_proj_path.name} and {raster_path.name}"
        )

    # Convert result to (geo)dataframe
    if "geometry" in columns:
        columns = [f.name() for f in vector_mem.fields()] + ["geometry"]
        data = [
            dict(zip(columns, f.attributes() + [f.geometry().asWkt()]))
            for f in vector_mem.getFeatures()
        ]
    else:
        columns = [f.name() for f in vector_mem.fields()]
        data = [dict(zip(columns, f.attributes())) for f in vector_mem.getFeatures()]
    stats_df = pd.DataFrame(data, columns=columns)

    if "geometry" in columns:
        stats_df["geometry"] = gpd.GeoSeries.from_wkt(stats_df["geometry"])
        stats_df = gpd.GeoDataFrame(
            stats_df, geometry="geometry", crs=vector_mem.crs().toWkt()
        )

    return stats_df


def zonal_stats_band_tofile(
    vector_path,
    raster_path: Path,
    output_band_path: Path,
    band: str,
    tmp_dir: Path,
    stats: List[Statistic],
    columns: List[str],
    force: bool = False,
) -> Path:
    if output_band_path.exists():
        if force:
            output_band_path.unlink()
        else:
            return output_band_path

    stats_df = zonal_stats_band(
        vector_path=vector_path,
        raster_path=raster_path,
        band=band,
        stats=stats,
        tmp_dir=tmp_dir,
        columns=columns,
    )
    # Remove rows with empty data
    # stats_df.dropna(inplace=True)

    logger.info(
        f"Write data for {len(stats_df.index)} parcels found to {output_band_path}"
    )
    pdh.to_file(stats_df, output_band_path, index=False)
    return output_band_path


def stat_to_qgisstat(stat: str):
    if stat == "count":
        return qgis.analysis.QgsZonalStatistics.Count
    elif stat == "sum":
        return qgis.analysis.QgsZonalStatistics.Sum
    elif stat == "mean":
        return qgis.analysis.QgsZonalStatistics.Mean
    elif stat == "median":
        return qgis.analysis.QgsZonalStatistics.Median
    elif stat == "std":
        return qgis.analysis.QgsZonalStatistics.StDev
    elif stat == "min":
        return qgis.analysis.QgsZonalStatistics.Min
    elif stat == "max":
        return qgis.analysis.QgsZonalStatistics.Max
    elif stat == "range":
        return qgis.analysis.QgsZonalStatistics.Range
    elif stat == "minority":
        return qgis.analysis.QgsZonalStatistics.Minority
    elif stat == "majority":
        return qgis.analysis.QgsZonalStatistics.Majority
    elif stat == "variance":
        return qgis.analysis.QgsZonalStatistics.Variance
    else:
        raise ValueError(f"unsupported value in stats: {stat}")
