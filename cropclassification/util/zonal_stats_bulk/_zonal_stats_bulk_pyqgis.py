from concurrent import futures
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Tuple, Union

import qgis.core  # type: ignore
import qgis.analysis  # type: ignore

from cropclassification.util import io_util

logger = logging.getLogger(__name__)


def zonal_stats(
    features_path: Path,
    id_column: str,
    images_bands: List[Tuple[Path, List[str]]],
    output_dir: Path,
    temp_dir: Path,
    log_dir: Path,
    log_level: Union[str, int],
    cloud_filter_band: Optional[str] = None,
    calc_bands_parallel: bool = True,
    force: bool = False,
):
    """
    Calculate zonal statistics.

    Args:
        features_path (Path): _description_
        id_column (str): _description_
        images_bands (List[Tuple[Path, List[str]]]): _description_
        output_dir (Path): _description_
        temp_dir (Path): _description_
        log_dir (Path): _description_
        log_level (Union[str, int]): _description_
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    raise Exception("Not implemented")
