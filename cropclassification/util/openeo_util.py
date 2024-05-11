from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import pprint
import time
from typing import Any, Dict, List, Optional, Tuple

import openeo
import openeo.rest.job
import pyproj

from . import raster_util
from .io_util import output_exists

# Get a logger...
logger = logging.getLogger(__name__)


def get_images(
    images_to_get: List[Dict[str, Any]],
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
):
    """
    Get a list of images from openeo.

    ``images_to_get`` is a list with a dict for each image to get with the following
    properties:

      - path (Path): the path to save the image to.
      - roi_bounds (Tuple): bounds of the image to get as Tuple[minx, miny, maxx, maxy].
      - roi_crs (int): crs the roi bounds.
      - collection (str): openeo collection to get the image from.
      - start_date (datetime): start date for the images in the collection to use for
        this image.
      - end_date (datetime): exclusive end date for the images in the collection to use
        for this image.
      - bands (List[str]): the bands to get.
      - time_reducer (str): the reducer to be used to aggregate the images in
        the time dimension: one of "mean", "min", "max",...
      - max_cloud_cover (float): the maximum cloud cover images should have to be used.
        If None, no cloud cover filtering is applied.
      - process_options (dict): process options to use for the image.
      - job_options (dict): job options to pass to openeo.


    Args:
        images_to_get (List[Dict[str, Any]]): list of dicts with information about the
            images to get.
        output_base_dir (Path): base directory to save the images to. The images will be
            saved in a subdirectory based on the image profile name.
        delete_existing_openeo_jobs (bool, optional): True to delete existing openeo
            jobs. If False, they are just left running and the results are downloaded if
            they are ready like other jobs. Defaults to False.
        raise_errors (bool, optional): True to raise if an error occurs. If False,
            errors are only logged. Defaults to True.
        force (bool, optional): True to force recreation of existing output files.
            Defaults to False.

    Raises:
        ValueError: _description_
        Exception: _description_
        RuntimeError: _description_
        RuntimeError: _description_
    """
    # Do some checks on already existing images
    images_to_get_todo = {}
    for image_to_get in images_to_get:
        if not output_exists(image_to_get["path"], remove_if_exists=force):
            images_to_get_todo[image_to_get["path"]] = image_to_get
        else:
            # Make sure the band descriptions are in the image
            raster_util.set_band_descriptions(
                image_to_get["path"], image_to_get["bands"], overwrite=False
            )

    if len(images_to_get_todo) == 0:
        return

    # Connect with openeo backend as configured in file specified in the
    # OPENEO_CLIENT_CONFIG environment variable.
    #
    # More info about the openeo authentication options can be found here:
    # https://open-eo.github.io/openeo-python-client/auth.html#default-url-and-auto-auth
    #
    # More info on the configuration file format can be found here:
    # https://open-eo.github.io/openeo-python-client/configuration.html#configuration-files  # noqa: E501
    conn = openeo.connect()
    logger.info(conn.describe_account())

    images_to_get_dict = {image["path"].as_posix(): image for image in images_to_get}

    if delete_existing_openeo_jobs:
        # Delete all jobs that are running already
        jobs = conn.list_jobs()
        if len(jobs) > 0:
            for job in jobs:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete job {job.get('title')}, id: {job['id']}")
                try:
                    batch_job.delete()
                except Exception as ex:
                    logger.warning(ex)
    else:
        # Process jobs that are still on the server
        image_paths, job_errors = get_job_results(conn)
        for image_path in image_paths:
            band_descriptions = None
            if image_path in images_to_get_dict:
                band_descriptions = images_to_get_dict[image_path]["bands"]
            postprocess_image(image_path, band_descriptions)
        if raise_errors and len(job_errors) > 0:
            raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    for image_to_get in images_to_get_todo.values():
        roi_bounds = list(image_to_get["roi_bounds"])

        # Determine bbox of roi
        roi_crs = image_to_get["roi_crs"]
        if roi_crs is None:
            logger.info("The crs of the roi is None, so it is assumed to be WGS84")
        else:
            roi_crs = pyproj.CRS(roi_crs)

        if roi_crs.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(
                roi_crs, "epsg:4326", always_xy=True
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

        create_mosaic_job(
            conn=conn,
            collection=image_to_get["collection"],
            spatial_extent=roi_extent,
            start_date=image_to_get["start_date"],
            end_date=image_to_get["end_date"],
            bands=image_to_get["bands"],
            time_reducer=image_to_get["time_reducer"],
            output_path=image_to_get["path"],
            max_cloud_cover=image_to_get.get("max_cloud_cover", None),
            job_options=image_to_get.get("job_options", {}),
            process_options=image_to_get.get("process_options", {}),
        ).start_job()

    # Get the results
    image_paths, job_errors = get_job_results(conn)

    # Postprocess the images created
    for image_path in image_paths:
        band_descriptions = None
        if image_path.as_posix() in images_to_get_dict:
            band_descriptions = images_to_get_dict[image_path.as_posix()]["bands"]
        postprocess_image(image_path, band_descriptions)
    if raise_errors and len(job_errors) > 0:
        raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    return


def create_mosaic_job(
    conn: openeo.Connection,
    collection: str,
    spatial_extent,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    output_path: Path,
    time_reducer: str,
    max_cloud_cover: Optional[float],
    job_options: dict,
    process_options: dict,
) -> openeo.BatchJob:
    logger.info(f"schedule {output_path}")

    # Check and process input params
    period = [
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]
    if process_options is None:
        process_options = {}

    bands_to_load = list(bands)
    cloud_filter_band = None
    cloud_filter_band_dilated = None
    for option in process_options:
        if option == "cloud_filter_band":
            band = process_options[option]
            cloud_filter_band = band
            if band not in bands_to_load:
                bands_to_load.append(band)
        elif option == "cloud_filter_band_dilated":
            band = process_options[option]
            cloud_filter_band_dilated = band
            if band not in bands_to_load:
                bands_to_load.append(band)
        else:
            raise ValueError(f"unknown processing option: {option}")

    if cloud_filter_band is not None and cloud_filter_band_dilated is not None:
        raise ValueError(
            "process_option cloud_filter_band and cloud_filter_band_dilated cannot be "
            "used together"
        )

    # Load cube of relevant images
    # You can look which layers are available here:
    # https://openeo.cloud/data-collections/
    cube = conn.load_collection(
        collection,
        spatial_extent=spatial_extent,
        temporal_extent=period,
        bands=bands_to_load,
        max_cloud_cover=max_cloud_cover,
    )

    # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
    if cloud_filter_band_dilated is not None:
        cube = cube.process(
            "mask_scl_dilation",
            data=cube,
            scl_band_name=cloud_filter_band,
        )
    elif cloud_filter_band is not None:
        # Select the scl band from the data cube
        scl_band = cube.band(cloud_filter_band)
        # Build mask to mask out everything but classes
        #   - 4: vegetation
        #   - 5: non vegetated
        #   - 6: water
        mask = ~((scl_band == 4) | (scl_band == 5) | (scl_band == 6))

        # The mask needs to have the same "ground sample distance" as the cube it is
        # applied to.
        mask = mask.resample_cube_spatial(cube)

        # Apply the mask
        cube = cube.mask(mask)

    cube = cube.filter_bands(bands=bands)
    cube = cube.reduce_dimension(dimension="t", reducer=time_reducer)

    # The NDVI collection needs to be recalculated
    if collection == "TERRASCOPE_S2_NDVI_V2":
        cube = cube.apply(lambda x: 0.004 * x - 0.08)

    return cube.create_job(
        out_format="GTiff",
        title=output_path.as_posix(),
        job_options=job_options,
    )


def get_job_results(
    conn: openeo.Connection, ignore_errors: bool = False
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
            if "title" in job:
                jobs_per_status[job["status"]].append(job)

        message = "jobs: "
        message += f"{','.join(f'{k}: {len(v)}' for k, v in jobs_per_status.items())}"
        logger.info(message)

        for job in jobs_per_status["finished"]:
            batch_job = openeo.rest.job.BatchJob(job["id"], conn)

            # If there is no title specified, delete job
            if "title" not in job:
                logger.info(f"job {job} doesn't have a title, so just delete it")
                try:
                    batch_job.delete()
                except Exception as ex:
                    logger.warning(ex)
            else:
                # Download results + delete job
                try:
                    logger.info(f"job {job} finished, so download results")

                    # Download to tmp file first so we are sure download was complete
                    output_path = Path(job["title"])
                    output_tmp_path = Path(f"{output_path}.download")
                    if output_path.exists():
                        output_path.unlink()
                    if output_tmp_path.exists():
                        output_tmp_path.unlink()

                    asset = batch_job.get_results().get_asset()
                    import requests
                    import shutil

                    def download_file(url, target):
                        with requests.get(url, stream=True) as r:
                            r.raw.decode_content = True
                            with open(target, "wb") as f:
                                shutil.copyfileobj(r.raw, f, length=1014 * 1024)

                    # Download result. If it fails, delete the job anyway
                    try:
                        download_file(asset.href, target=output_tmp_path)

                        # Ready, now we can rename tmp file
                        output_tmp_path.rename(output_path)
                        output_paths.append(output_path)
                    finally:
                        batch_job.delete()

                except Exception as ex:
                    raise RuntimeError(f"Error downloading {output_path}: {ex}")

        # As long as processing is needed, keep polling
        if "queued" in jobs_per_status or "running" in jobs_per_status:
            print(
                f"Waiting for {jobs_per_status['queued']} and "
                f"{jobs_per_status['running']}"
            )
            time.sleep(30)
        else:
            # We are ready, so deal with error jobs and break
            for job in jobs_per_status["error"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.error(f"Error processing job '{job['title']}', id: {job['id']}")
                errorlog = pprint.pformat(batch_job.logs())
                logger.error(errorlog)
                batch_job.delete()
                errors.append(f"Error for job '{job['title']}', id: {job['id']}")
            for job in jobs_per_status["created"]:
                batch_job = openeo.rest.job.BatchJob(job["id"], conn)
                logger.info(f"delete created job '{job['title']}', id: {job['id']}")
                batch_job.delete()

            break

    if not ignore_errors and len(errors) > 0:
        raise RuntimeError(f"openeo processing errors: {pprint.pformat(errors)}")

    return output_paths, errors


def postprocess_image(path: Path, band_descriptions: Optional[List[str]]):
    raster_util.add_overviews(path)

    # if band_descriptions is None, try to read them from the json metadata file.
    if band_descriptions is None:
        json_path = Path(f"{path}.json")
        if json_path.exists():
            with open(json_path) as file:
                json_str = file.read()
                data = json.loads(json_str)
                band_descriptions = data["bands"]

    if band_descriptions is not None:
        raster_util.set_band_descriptions(path, band_descriptions)
    else:
        logger.warning(f"no band_descriptions specified for {path}")
