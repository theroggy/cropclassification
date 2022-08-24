
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import List

import geofileops as gfo
import openeo
import openeo.rest.job
import pyproj
import rasterio

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def calc_periodic_images(
    roi_path: Path,
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    filename_suffix: str,
    output_dir: Path,
    sensordata_to_get: List[str],
    force: bool = False,
):

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

    # Save periodic aggregate of S2 images
    logger.info(
        f"Download masked images for roi {roi_extent} per {days_per_period} days to {output_dir}"
    )

    # Delete all jobs that are running already
    jobs = conn.list_jobs()
    if len(jobs) > 0:
        for job in jobs:
            batch_job = openeo.rest.job.BatchJob(job["id"], conn)
            logger.info(f"delete job '{job['title']}', id: {job['id']}")
            batch_job.delete_job()

    period_start_date = start_date
    while period_start_date <= (end_date - timedelta(days=days_per_period)):

        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=days_per_period)
        period = [
            period_start_date.strftime("%Y-%m-%d"),
            period_end_date.strftime("%Y-%m-%d"),
        ]
        output_basestem = f"{period[0]}_{period[1]}{filename_suffix}"

        logger.info(f"Get images for period {period}")

        if save_rgb or save_scl or save_landcover:

            # Load cube of relevant S2 images
            # You can look which layers are available here:
            #    https://openeo.cloud/data-collections/
            s2_cube = conn.load_collection(
                "TERRASCOPE_S2_TOC_V2",
                spatial_extent=roi_extent,
                temporal_extent=period,  # type: ignore
                bands=["B04", "B03", "B02", "SCL"],
            )

            # Save RGB image
            if save_rgb:
                job_title = f"{output_basestem}_rgb.tif"
                output_path = output_dir / job_title
                if check_output_file(output_path, force):
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
                        .create_job(out_format="GTiff", title=job_title)
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
                output_path = output_dir / job_title
                if check_output_file(output_path, force):
                    _ = (
                        s2_cube.band("SCL")
                        .reduce_dimension(dimension="t", reducer="max")
                        .create_job(out_format="GTiff", title=job_title)
                        .start_job()
                    )

            if save_landcover:
                job_title = f"{output_basestem}_landcover.tif"
                output_path = output_dir / job_title
                if check_output_file(output_path, force):

                    # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
                    s2_cube_scl = s2_cube.filter_bands(bands=["SCL"])
                    s2_cube_scl_data = s2_cube_scl.rename_labels(
                        dimension="bands", target=["SCL_DATA"], source=["SCL"]
                    )
                    s2_cube_scl_merged = s2_cube_scl_data.merge_cubes(s2_cube_scl)
                    _ = (
                        s2_cube_scl_merged
                        .process(
                            "mask_scl_dilation",
                            data=s2_cube_scl_merged,
                            scl_band_name="SCL"
                        )
                        .filter_bands(bands=["SCL_DATA"])
                        .reduce_dimension(dimension="t", reducer="max")
                        .create_job(out_format="GTiff", title=job_title)
                        .start_job()
                    )

        if save_ndvi:
            job_title = f"{output_basestem}_ndvi.tif"
            output_path = output_dir / job_title
            if check_output_file(output_path, force):

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
                    .reduce_dimension(dimension="t", reducer="max")
                    .create_job(out_format="GTiff", title=job_title)
                    .start_job()
                )

        period_start_date = period_end_date

    # Get the results
    output_paths = get_job_results(conn, output_dir)
    for output_path in output_paths:
        add_overviews(output_path)


def check_output_file(path: Path, force: bool) -> bool:
    """
    Check if the output file exists already. If force is True, the file is removed.

    Args:
        path (Path): the file to test.
        force (bool): If True, remove the file if it exists.

    Returns:
        bool: True if the file exists.
    """
    if path.exists():
        if force is True:
            path.unlink()
        else:
            logger.info(f"{path.name} exists, so skip")
            return False

    logger.info(f"{path.name} needs to be processed")
    return True


def get_job_results(conn: openeo.Connection, output_dir: Path) -> List[Path]:
    """Get results of the completed jobs."""

    output_paths = []
    while True:
        jobs = conn.list_jobs()
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
                logger.info(f"job {job} didn't have a title, so just delete it")
                batch_job.delete_job()
            else:
                # Download results + delete job
                output_path = output_dir / job["title"]
                batch_job.get_results().download_file(target=output_path)
                batch_job.delete_job()
                output_paths += output_paths

        # As long as processing is needed, keep polling
        if "queued" in jobs_per_status or "running" in jobs_per_status:
            time.sleep(30)
        else:
            # We are ready, so deal with error jobs and break
            for job in jobs_per_status["error"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"Error processing job '{job['title']}', id: {job['id']}")
                logger.info(batch_job.logs())
                batch_job.delete_job()
            for job in jobs_per_status["created"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete created job '{job['title']}', id: {job['id']}")
                batch_job.delete_job()

            break

    return output_paths


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
