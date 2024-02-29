from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pprint
import time
from typing import List, Optional, Tuple

import openeo
import openeo.rest.job
import pyproj
import rioxarray

from osgeo import gdal

# ... and suppress errors
gdal.PushErrorHandler("CPLQuietErrorHandler")
import rasterio  # noqa: E402


# Get a logger...
logger = logging.getLogger(__name__)


class ImageProfile:
    """
    Profile of the image to be processed via openeo.
    """

    def __init__(
        self,
        name: str,
        satellite: str,
        image_source: str,
        collection: str,
        bands: List[str],
        process_options: dict,
        job_options: dict,
    ):
        """
        Constructor of ImageProfile.

        Args:
            name (str): name of the image profile.
            satellite (str): name of the satellite.
            image_source (str): source of the image.
            collection (str): collection on openeo to load to create this image.
            bands (List[str]): bands of the collection to include in the image.
            process_options (dict): process options.
            job_options (dict): job options. Job options available on the VITO OpenEO
                backend are documented here: https://docs.openeo.cloud/federation/#customizing-batch-job-resources-on-terrascope
        """
        self.name = name
        self.satellite = satellite
        self.image_source = image_source
        self.collection = collection
        self.bands = bands
        self.process_options = process_options
        self.job_options = job_options


def calc_periodic_mosaic(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    images_to_get: List[ImageProfile],
    output_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Tuple[Path, ImageProfile]]:
    """
    Download a periodic mosaic.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        days_per_period (int): number of days per period.
        images_to_get (List[ImageProfile]): list of imageprofiles to create the mosaic
            with.
        output_dir (Path): directory to save the images to.
        period_name (Optional[str], optional): name of the period. If None, default
            names are used: if ``days_per_period=7``: "weekly", if
            ``days_per_period=14``: "biweekly", for other values of ``days_per_period``
            a ValueError is thrown. Defaults to None.
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

    Returns:
        List[Tuple[Path, ImageProfile]]: _description_
    """
    if start_date == end_date:
        raise ValueError(f"start date and end date are the same: {start_date}")
    if end_date > datetime.now():
        logger.warning(f"end_date is in the future: {end_date}")
    roi_bounds = list(roi_bounds)

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
    if roi_crs is None:
        logger.info("The crs of the roi is None, so it is assumed to be WGS84")
    else:
        roi_crs = pyproj.CRS(roi_crs)

    if roi_crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(roi_crs, "epsg:4326", always_xy=True)
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
                logger.info(f"delete job {job.get('title')}, id: {job['id']}")
                try:
                    batch_job.delete()
                except Exception as ex:
                    logger.warning(ex)
    else:
        # Process jobs that are still on the server
        image_paths, job_errors = get_job_results(conn, output_dir)
        for image_path in image_paths:
            add_overviews(image_path)
        if raise_errors and len(job_errors) > 0:
            raise RuntimeError(f"Errors occured: {pprint.pformat(job_errors)}")

    period_start_date = start_date

    result = []
    while period_start_date <= (end_date - timedelta(days=days_per_period)):
        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=days_per_period)
        if period_end_date > datetime.now():
            logger.info(
                f"skip period ({period_start_date}, {period_end_date}): it is in the "
                "future!"
            )
            break

        for imageprofile in images_to_get:
            # Prepare path. Remark: in the image path the end_date should be inclusive.
            end_date_incl = period_end_date
            if days_per_period > 1:
                end_date_incl = period_end_date - timedelta(days=1)
            image_path, image_relative_path = prepare_image_path(
                imageprofile.name,
                start_date=period_start_date,
                end_date=end_date_incl,
                bands=imageprofile.bands,
                time_dimension_reducer=imageprofile.process_options[
                    "time_dimension_reducer"
                ],
                dir=output_dir,
            )
            if not file_exists(image_path, force):
                # time_dimension_reducer is expected as a parameter rather than in the
                # process_options, so extract it...
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
                    output_name=image_relative_path,
                    job_options=imageprofile.job_options,
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
    time_dimension_reducer: str,
    dir: Path,
) -> Tuple[Path, str]:
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
        Tuple[Path, str]: the full path + the relative path
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    weeks = list(
        range(int(start_date.strftime("%W")), int(start_date.strftime("%W")) + 1)
    )
    # Concat bands, but remove all chars used as separators
    bands_str = "-".join([band.replace("_", "").replace("-", "") for band in bands])
    name = (
        f"{imageprofile}_{start_date_str}_{end_date_str}_{bands_str}_"
        f"{time_dimension_reducer}.tif"
    )

    # If the image metadata file doesn't exist, create it
    image_dir = dir / imageprofile
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / name
    imagemeta_path = image_dir / f"{name}.json"
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
            "time_dimension_reducer": time_dimension_reducer,
            "path": image_path.as_posix(),
        }
        if imageprofile.lower() == "s1-grd-sigma0-asc":
            metadata["orbit"] = "asc"
        elif imageprofile.lower() == "s1-grd-sigma0-desc":
            metadata["orbit"] = "desc"

        # Write to file
        with open(imagemeta_path, "w") as outfile:
            outfile.write(json.dumps(metadata, indent=4))

    return (image_path, f"{imageprofile}/{name}")


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
        temporal_extent=period,  # type: ignore
        bands=bands_to_load,
        max_cloud_cover=80,
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
                    output_path = output_dir / job["title"]
                    output_tmp_path = output_dir / f"{job['title']}.download"
                    if output_path.exists():
                        output_path.unlink()
                    if output_tmp_path.exists():
                        output_tmp_path.unlink()

                    # Use chunk size 50 MB to see some progress
                    batch_job.get_results().get_asset().download(
                        target=output_tmp_path, chunk_size=50 * 1024 * 1024
                    )

                    # Ready, now we can rename tmp file
                    output_tmp_path.rename(output_path)

                    batch_job.delete()
                    output_paths.append(output_path)

                except Exception as ex:
                    raise RuntimeError(f"Error downloading {output_path}: {ex}")

        # As long as processing is needed, keep polling
        if "queued" in jobs_per_status or "running" in jobs_per_status:
            print(
                f"Waiting for {jobs_per_status['queued']} and {jobs_per_status['running']}"
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
