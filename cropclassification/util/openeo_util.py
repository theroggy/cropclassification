from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import json
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
gdal.PushErrorHandler("CPLQuietErrorHandler")
import rasterio


# Get a logger...
logger = logging.getLogger(__name__)


class ImageProfile:
    def __init__(
        self,
        name: str,
        satellite: str,
        collection: str,
        bands: List[str],
        process_options: dict,
    ):
        self.name = name
        self.satellite = satellite
        self.collection = collection
        self.bands = bands
        self.process_options = process_options


def calc_periodic_mosaic(
    roi_path: Path,
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    images_to_get: List[ImageProfile],
    output_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = True,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Tuple[Path, ImageProfile]]:
    # Validate time_dimension_reducer
    for imageprofile in images_to_get:
        reducer = imageprofile.process_options.get("time_dimension_reducer", None)
        if reducer is None:
            raise ValueError(
                "process_options must contain a time_dimension_reducer for "
                "calc_periodic_mosaic"
            )

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

    # Connect with openeo backend as configured in file specified in the
    # OPENEO_CLIENT_CONFIG environment variable.
    #
    # More info about the openeo authentication options can be found here:
    # https://open-eo.github.io/openeo-python-client/auth.html#default-url-and-auto-auth
    #
    # More info on the configuration file format can be found here:
    # https://open-eo.github.io/openeo-python-client/configuration.html#configuration-files  # noqa: E501
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
        image_paths, job_errors = get_job_results(conn, output_dir)
        for image_path in image_paths:
            add_overviews(image_path)
        if raise_errors and len(job_errors) > 0:
            raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    period_start_date = start_date
    job_options = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "2",
    }
    result = []
    while period_start_date <= (end_date - timedelta(days=days_per_period)):
        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=days_per_period)
        for imageprofile in images_to_get:
            # Prepare path. Remark: in the image path the end_date should be inclusive.
            end_date_incl = period_end_date
            if days_per_period > 1:
                end_date_incl = period_end_date - timedelta(days=1)
            image_path = prepare_image_path(
                imageprofile.name,
                start_date=period_start_date,
                end_date=end_date_incl,
                bands=imageprofile.bands,
                dir=output_dir,
            )
            if not file_exists(image_path, force):
                process_options = deepcopy(imageprofile.process_options)
                reducer = process_options["time_dimension_reducer"]
                del process_options["time_dimension_reducer"]

                create_mosaic_job(
                    conn=conn,
                    collection=imageprofile.collection,
                    spatial_extent=roi_extent,
                    start_date=period_start_date,
                    end_date=period_end_date,
                    bands=imageprofile.bands,
                    time_dimension_reducer=reducer,
                    output_name=image_path.name,
                    job_options=job_options,
                    process_options=process_options,
                ).start_job()
            result.append((image_path, imageprofile))

        period_start_date = period_end_date

    # Get the results
    image_paths, job_errors = get_job_results(conn, output_dir)

    # TODO: ideally, all mosaic_paths would be checked if they have overviews.
    for image_path in image_paths:
        add_overviews(image_path)
    if raise_errors and len(job_errors) > 0:
        raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    return result


def prepare_image_path(
    imageprofile: str,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    dir: Path,
) -> Path:
    """
    Returns an image_path + saves a metadata file for the image as f"{image_path}.json".

    Args:
        imageprofile (str): _description_
        start_date (datetime): _description_
        end_date (datetime):
        bands (List[str]): _description_
        dir (Path): _description_

    Raises:
        ValueError: _description_

    Returns:
        Path: _description_
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    weeks = list(
        range(int(start_date.strftime("%W")), int(start_date.strftime("%W")) + 1)
    )
    bands_str = "-".join(bands)
    name = f"{imageprofile}_{start_date_str}_{end_date_str}_{bands_str}.tif"

    # If the image metadata file doesn't exist, create it
    image_path = dir / name
    imagemeta_path = dir / f"{name}.json"
    if not imagemeta_path.exists():
        imageprofile_parts = imageprofile.split("-")
        satellite = imageprofile_parts[0].lower()
        metadata = {
            "imageprofile": imageprofile,
            "satellite": satellite,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "weeks": weeks,
            "bands": bands,
            "path": image_path.as_posix(),
        }
        if imageprofile.lower() == "s1-grd-sigma0-asc":
            metadata["orbit"] = "asc"
        elif imageprofile.lower() == "s1-grd-sigma0-desc":
            metadata["orbit"] = "desc"

        # Write to file
        with open(imagemeta_path, "w") as outfile:
            outfile.write(json.dumps(metadata, indent=4))

    return image_path


def create_mosaic_job(
    conn: openeo.Connection,
    collection: str,
    spatial_extent,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    output_name: str,
    time_dimension_reducer: str,
    job_options: dict,
    process_options: dict,
) -> openeo.BatchJob:
    logger.info(f"schedule {output_name}")

    # Check and process input params
    period = [
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]

    bands_to_load = list(bands)
    cloud_filter_band = None
    for option in process_options:
        if option == "cloud_filter_band":
            cloud_filter_band = process_options[option]
            bands_to_load.append(cloud_filter_band)
        else:
            raise ValueError(f"unknown processing option: {option}")

    # Load cube of relevant images
    # You can look which layers are available here:
    # https://openeo.cloud/data-collections/
    cube = conn.load_collection(
        collection,
        spatial_extent=spatial_extent,
        temporal_extent=period,  # type: ignore
        bands=bands_to_load,
    )

    # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
    if cloud_filter_band is not None:
        cube = cube.process(
            "mask_scl_dilation",
            data=cube,
            scl_band_name=cloud_filter_band,
        )

    cube = cube.filter_bands(bands=bands)
    cube = cube.reduce_dimension(dimension="t", reducer=time_dimension_reducer)

    # The NDVI collection needs to be recalculated
    if collection == "TERRASCOPE_S2_NDVI_V2":
        cube = cube.apply(lambda x: 0.004 * x - 0.08)

    return cube.create_job(
        out_format="GTiff",
        title=output_name,
        job_options=job_options,
    )


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

    logger.info(f"{path.name} doesn't exist")
    return False


def get_job_results(
    conn: openeo.Connection, output_dir: Path, ignore_errors: bool = False
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

    if not ignore_errors and len(errors) > 0:
        raise RuntimeError(f"openeo processing errors: {pprint.pformat(errors)}")

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
