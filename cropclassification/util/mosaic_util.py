from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pyproj
import openeo_util
from openeo_util import ImageProfile


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
    Generate a periodic mosaic.

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
    images_to_get_openeo = []
    images_to_get_local = []
    for image_to_get in images_to_get:
        if image_to_get.image_source == "openeo":
            images_to_get_openeo.append(image_to_get)
        elif image_to_get.image_source == "local":
            images_to_get_local.append(image_to_get)
        else:
            raise ValueError(
                f"unsupported image_source in image_to_get: {image_to_get.image_source}"
            )

    info = openeo_util.calc_periodic_mosaic(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        images_to_get=images_to_get_openeo,
        output_dir=output_dir,
        period_name=period_name,
        delete_existing_openeo_jobs=delete_existing_openeo_jobs,
        raise_errors=raise_errors,
        force=force,
    )
    
    calc_periodic_mosaic_local(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        images_to_get=images_to_get_local,
        output_dir=output_dir,
        period_name=period_name,
        raise_errors=raise_errors,
        force=force,
    )

def calc_periodic_mosaic_local:
