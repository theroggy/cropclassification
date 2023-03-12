from collections import defaultdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pprint
import time
from typing import List, Optional, Tuple

import geofileops as gfo
import openeo
import openeo.rest.job
import pyproj

from osgeo import gdal
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')
import rasterio

# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# The real work
# -------------------------------------------------------------

"""
def calc_periodic_mosaic(
    roi_path: Path,
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    sensordata_to_get: List[str],
    output_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Path]:
    result_paths = []

    # Prepare period_name:
    if period_name is None:
        if days_per_period == 7:
            period_name = "weekly"
        elif days_per_period == 14:
            period_name = "biweekly"
        else:
            raise Exception("Unknown period name, please specify")

    # Determine bbox of roi
    roi_info = gfo.get_layerinfo(roi_path)
    roi_bounds = list(roi_info.total_bounds)
    if roi_info.crs is None:
        logger.info("The crs of the roi is None, so it is assumed to be WGS84")
    elif roi_info.crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(
            roi_info.crs, "epsg:4326", always_xy=True
        )
        roi_bounds[0], roi_bounds[1] = transformer.transform(
            roi_bounds[0], roi_bounds[1]
        )
        roi_bounds[2], roi_bounds[3] = transformer.transform(
            roi_bounds[2], roi_bounds[3]
        )
    roi_extent = {
        "west": roi_bounds[0],
        "east": roi_bounds[2],
        "south": roi_bounds[1],
        "north": roi_bounds[3],
    }

    # Connect with VITO openeo backend
    conn = openeo.connect("https://openeo.vito.be").authenticate_oidc(provider_id="egi")

    # Save periodic aggregate of asked images
    logger.info(
        f"Download masked images for roi {roi_extent} per {days_per_period} days "
        f"to {output_dir}"
    )

    if delete_existing_openeo_jobs:
        # Delete all jobs that are running already
        jobs = conn.list_jobs()
        if len(jobs) > 0:
            for job in jobs:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete job '{job['title']}', id: {job['id']}")
                batch_job.delete_job()
    else:
        # Process jobs that are still on the server
        result_paths, job_errors = get_job_results(conn, output_dir)
        for result_path in result_paths:
            add_overviews(result_path)
        if raise_errors and len(job_errors) > 0:
            raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    period_start_date = start_date
    job_options = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "2",
    }
    while period_start_date <= (end_date - timedelta(days=days_per_period)):

        for sensordata_type in sensordata_to_get:
            # Period in openeo is inclusive for startdate and excludes enddate
            period_end_date = period_start_date + timedelta(days=days_per_period)
            period = [
                period_start_date.strftime("%Y-%m-%d"),
                period_end_date.strftime("%Y-%m-%d"),
            ]
            satellite, band = sensordata_type.split("-")

            output_basestem = (
                f"{satellite}_mosaic_{period_name}_{period[0]}_{period[1]}"
            )

            logger.info(f"Get images for period {period}")

            if sensordata_type == "S2-landcover":
                save_rgb = False
                save_scl = False
                save_landcover = True

                # Load cube of relevant S2 images
                # You can look which layers are available here:
                # https://openeo.cloud/data-collections/
                s2_cube = conn.load_collection(
                    "TERRASCOPE_S2_TOC_V2",
                    spatial_extent=roi_extent,
                    temporal_extent=period,  # type: ignore
                    bands=["B04", "B03", "B02", "SCL"],
                )

                # Save RGB image
                if save_rgb:
                    job_title = f"{output_basestem}_rgb.tif"
                    result_path = output_dir / job_title
                    if file_needs_processing(result_path, force):
                        # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
                        # Use the "min" reducer filters out "lightly clouded areas"
                        _ = (
                            s2_cube.process(
                                "mask_scl_dilation",
                                data=s2_cube,
                                scl_band_name="SCL",
                            )
                            .filter_bands(bands=["B04", "B03", "B02"])
                            .reduce_dimension(dimension="t", reducer="min")
                            .create_job(
                                out_format="GTiff",
                                title=job_title,
                                job_options=job_options,
                            )
                            .start_job()
                        )

                # Save scl layer if available
                # SCL mask values:
                #   * 0: NO_DATA
                #   * 1: SATURATED_OR_DEFECTIVE
                #   * 2: DARK_AREA_PIXELS
                #   * 3: CLOUD_SHADOWS
                #   * 4: VEGETATION
                #   * 5: NOT_VEGETATED
                #   * 6: WATER
                #   * 7: UNCLASSIFIED
                #   * 8: CLOUD_MEDIUM_PROBABILITY
                #   * 9: CLOUD_HIGH_PROBABILITY
                #   * 10: THIN_CIRRUS
                #   * 11: SNOW
                if save_scl:
                    job_title = f"{output_basestem}_scl.tif"
                    result_path = output_dir / job_title
                    if file_needs_processing(result_path, force):
                        _ = (
                            s2_cube.band("SCL")
                            .reduce_dimension(dimension="t", reducer="max")
                            .create_job(
                                out_format="GTiff",
                                title=job_title,
                                job_options=job_options,
                            )
                            .start_job()
                        )

                if save_landcover:
                    job_title = f"{output_basestem}_{band}.tif"
                    if (
                        job_title
                        == "S2_mosaic_weekly_2021-11-29_2021-12-06_landcover.tif"
                    ):
                        period_start_date = period_end_date
                        continue

                    result_path = output_dir / job_title
                    result_paths.append(result_path)
                    if file_needs_processing(result_path, force):

                        # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
                        s2_cube_scl = s2_cube.filter_bands(bands=["SCL"])
                        s2_cube_scl_data = s2_cube_scl.rename_labels(
                            dimension="bands", target=["SCL_DATA"], source=["SCL"]
                        )
                        s2_cube_scl_merged = s2_cube_scl_data.merge_cubes(s2_cube_scl)
                        _ = (
                            s2_cube_scl_merged.process(
                                "mask_scl_dilation",
                                data=s2_cube_scl_merged,
                                scl_band_name="SCL",
                            )
                            .filter_bands(bands=["SCL_DATA"])
                            .reduce_dimension(dimension="t", reducer="max")
                            .create_job(
                                out_format="GTiff",
                                title=job_title,
                                job_options=job_options,
                            )
                            .start_job()
                        )

            elif sensordata_type == "S2-ndvi":
                job_title = f"{output_basestem}_{band}.tif"
                result_path = output_dir / job_title
                result_paths.append(result_path)
                if file_needs_processing(result_path, force):

                    # Load cube of relevant S2 images
                    # You can look which layers are available here:
                    #    https://openeo.cloud/data-collections/
                    s2_ndvi_cube = conn.load_collection(
                        "TERRASCOPE_S2_NDVI_V2",
                        spatial_extent=roi_extent,
                        temporal_extent=period,  # type: ignore
                        bands=["NDVI_10M", "SCENECLASSIFICATION_20M"],
                    )

                    # Create + start job
                    # Use "aggressive" algorythm to mask more clouds based on SCL
                    _ = (
                        s2_ndvi_cube.process(
                            "mask_scl_dilation",
                            data=s2_ndvi_cube,
                            scl_band_name="SCENECLASSIFICATION_20M",
                        )
                        .filter_bands(bands=["NDVI_10M"])
                        .reduce_dimension(dimension="t", reducer="max")
                        .apply(lambda x: 0.004 * x - 0.08)
                        .create_job(
                            out_format="GTiff", title=job_title, job_options=job_options
                        )
                        .start_job()
                    )
            else:
                raise ValueError(
                    f"invalid value for sensordata_to_get: {sensordata_type}"
                )

            period_start_date = period_end_date

    # Get the results
    output_paths, job_errors = get_job_results(conn, output_dir)
    for output_path in output_paths:
        add_overviews(output_path)
    if raise_errors and len(job_errors) > 0:
        raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    result_paths.extend(output_paths)
    return result_paths
"""


def calc_periodic_mosaic(
    roi_path: Path,
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    sensordata_to_get: List[str],
    output_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Path]:

    # Prepare period_name:
    if period_name is None:
        if days_per_period == 7:
            period_name = "weekly"
        elif days_per_period == 14:
            period_name = "biweekly"
        else:
            raise Exception("Unknown period name, please specify")

    # Determine bbox of roi
    roi_info = gfo.get_layerinfo(roi_path)
    roi_bounds = list(roi_info.total_bounds)
    if roi_info.crs is None:
        logger.info("The crs of the roi is None, so it is assumed to be WGS84")
    elif roi_info.crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(
            roi_info.crs, "epsg:4326", always_xy=True
        )
        roi_bounds[0], roi_bounds[1] = transformer.transform(
            roi_bounds[0], roi_bounds[1]
        )
        roi_bounds[2], roi_bounds[3] = transformer.transform(
            roi_bounds[2], roi_bounds[3]
        )
    roi_extent = {
        "west": roi_bounds[0],
        "east": roi_bounds[2],
        "south": roi_bounds[1],
        "north": roi_bounds[3],
    }

    # Connect with openeo backend as configured OPENEO_CLIENT_CONFIG 
    conn = openeo.connect()

    # Save periodic aggregate of asked images
    logger.info(
        f"Download masked images for roi {roi_extent} per {days_per_period} days "
        f"to {output_dir}"
    )

    if delete_existing_openeo_jobs:
        # Delete all jobs that are running already
        jobs = conn.list_jobs()
        if len(jobs) > 0:
            for job in jobs:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete job '{job['title']}', id: {job['id']}")
                batch_job.delete_job()
    else:
        # Process jobs that are still on the server
        processed_paths, job_errors = get_job_results(conn, output_dir)
        for processed_path in processed_paths:
            add_overviews(processed_path)
        if raise_errors and len(job_errors) > 0:
            raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    period_start_date = start_date
    job_options = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "2",
    }
    mosaic_paths = []
    while period_start_date <= (end_date - timedelta(days=days_per_period)):
        period_end_date = period_start_date + timedelta(days=days_per_period)
        for sensordata_type in sensordata_to_get:
            # Period in openeo is inclusive for startdate and excludes enddate
            sensordata_type = sensordata_type.lower()
            satellite, bands_code = sensordata_type.split("-")
            collection = None
            bands = None
            process_options = {}
            reducer = None
            if sensordata_type == "s2-ndvi":
                collection = "TERRASCOPE_S2_NDVI_V2"
                bands = ["NDVI_10M"]
                process_options["filter_clouds"] = "SCENECLASSIFICATION_20M"
                reducer = "max"
            elif satellite == "s2":
                collection = "TERRASCOPE_S2_TOC_V2"
                process_options["filter_clouds"] = "SCL"
                # Use the "min" reducer filters out "lightly clouded areas"
                reducer = "min"
                if bands_code == "rgb":
                    bands = ["B04", "B03", "B02"]
                elif bands_code == "agri":
                    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
            elif satellite == "s1":
                reducer = "max"
                if bands_code == "asc":
                    collection = "S1_GRD_SIGMA0_ASCENDING"
                    bands = ["VV", "VH", "angle"]
                elif bands_code == "desc":
                    collection = "S1_GRD_SIGMA0_DESCENDING"
                    bands = ["VV", "VH", "angle"]
                elif bands_code == "coh":
                    collection = "TERRASCOPE_S1_SLC_COHERENCE_V1"
                    bands = ["VV", "VH"]
            if collection is None or bands is None:
                raise ValueError(f"sensordata_type not known: {sensordata_type}")

            processed_path = start_job_mosaic(
                conn=conn,
                collection=collection,
                spatial_extent=roi_extent,
                start_date=period_start_date,
                end_date=period_end_date,
                bands=bands,
                period_name=period_name,
                reducer=reducer,
                output_dir=output_dir,
                job_options=job_options,
                process_options=process_options,
                force=force,
            )
            mosaic_paths.append(processed_path)

        period_start_date = period_end_date

    # Get the results
    processed_paths, job_errors = get_job_results(conn, output_dir)

    # TODO: ideally, all mosaic_paths would be checked if they have overviews.
    for processed_path in processed_paths:
        add_overviews(processed_path)
    if raise_errors and len(job_errors) > 0:
        raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    if True:
        for path in mosaic_paths:
            logger.info(f"Info of {path.name}:")
            with rasterio.open(path, "r") as dst:
                logger.info(dst.profile)

    return mosaic_paths


def start_job_mosaic(
    conn: openeo.Connection,
    collection: str,
    spatial_extent,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    output_dir: Path,
    period_name: str,
    reducer: str,
    job_options: dict,
    process_options: dict,
    force: bool = False,
) -> Path:
    # Check and process input params
    period = [
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]
    bands_str = "-".join(bands)
    collection_str = collection.replace("_", "-")
    name = (
        f"{collection_str}_mosaic_{period_name}_{period[0]}_{period[1]}_{bands_str}.tif"
    )
    output_path = output_dir / name
    if file_exists(output_path, force):
        return output_path

    period = [
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]

    bands_to_load = list(bands)
    filter_clouds_band = None
    for option in process_options:
        if option == "filter_clouds":
            filter_clouds_band = process_options[option]
            bands_to_load.append(filter_clouds_band)
        else:
            raise ValueError(f"unknown processing option: {option}")

    logger.info(f"{output_path.stem} needs to be scheduled")
    # Load cube of relevant images
    # You can look which layers are available here:
    # https://openeo.cloud/data-collections/
    cube = conn.load_collection(
        collection,
        spatial_extent=spatial_extent,
        temporal_extent=period,  # type: ignore
        bands=bands_to_load,
    )

    job_title = output_path.name

    # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
    if filter_clouds_band is not None:
        cube = cube.process(
            "mask_scl_dilation",
            data=cube,
            scl_band_name=filter_clouds_band,
        )

    cube = cube.filter_bands(bands=bands)
    cube = cube.reduce_dimension(dimension="t", reducer=reducer)

    # The NDVI collection needs to be recalculated
    if collection == "TERRASCOPE_S2_NDVI_V2":
        cube = cube.apply(lambda x: 0.004 * x - 0.08)

    cube = cube.create_job(
        out_format="GTiff",
        title=job_title,
        job_options=job_options,
    )
    cube.start_job()
    return output_path


def file_exists(path: Path, force: bool) -> bool:
    """
    Check if the output file exists already. If force is True, the file is removed.

    Args:
        path (Path): the file to check for.
        force (bool): If True, remove the file if it exists.

    Returns:
        bool: True if the file exists.
    """
    if path.exists():
        if force is True:
            path.unlink()
        else:
            logger.info(f"{path.name} exists, so skip")
            return True

    return False


def get_job_results(
    conn: openeo.Connection, output_dir: Path
) -> Tuple[List[Path], List[str]]:
    """Get results of the completed jobs."""

    output_paths = []
    errors = []
    while True:
        jobs = conn.list_jobs()
        if len(jobs) == 0:
            break

        jobs_per_status = defaultdict(list)
        for job in jobs:
            jobs_per_status[job["status"]].append(job)

        message = "jobs: "
        message += f"{','.join(f'{k}: {len(v)}' for k, v in jobs_per_status.items())}"
        logger.info(message)

        for job in jobs_per_status["finished"]:
            batch_job = openeo.rest.job.BatchJob(job["id"], conn)

            # If there is no title specified, delete job
            if "title" not in job:
                logger.info(f"job {job} doesn't have a title, so just delete it")
                batch_job.delete_job()
            else:
                # Download results + delete job
                output_path = output_dir / job["title"]
                logger.info(f"job {job} finished, so download results")
                batch_job.get_results().download_file(target=output_path)
                batch_job.delete_job()
                output_paths.append(output_path)

        # As long as processing is needed, keep polling
        if "queued" in jobs_per_status or "running" in jobs_per_status:
            time.sleep(30)
        else:
            # We are ready, so deal with error jobs and break
            for job in jobs_per_status["error"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.error(f"Error processing job '{job['title']}', id: {job['id']}")
                errorlog = pprint.pformat(batch_job.logs())
                logger.error(errorlog)
                batch_job.delete_job()
                errors.append(f"Error for job '{job['title']}', id: {job['id']}")
            for job in jobs_per_status["created"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete created job '{job['title']}', id: {job['id']}")
                batch_job.delete_job()

            break

    return output_paths, errors


def add_overviews(path: Path):

    # Add overviews
    with rasterio.open(path, "r+") as dst:
        factors = []
        for power in range(1, 999):
            factor = pow(2, power)
            if dst.width / factor < 256 or dst.height / factor < 256:
                break
            factors.append(factor)
        if len(factors) > 0:
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
