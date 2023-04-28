from pathlib import Path
from typing import List, Optional, Tuple, Union

from ._raster_helper import *  # noqa: F401, F403


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
    engine: str = "rasterstats",
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
    if engine == "pyqgis":
        from _zonal_stats_bulk_pyqgis import zonal_stats

        return zonal_stats(
            features_path=features_path,
            id_column=id_column,
            images_bands=images_bands,
            output_dir=output_dir,
            temp_dir=temp_dir,
            log_dir=log_dir,
            log_level=log_level,
            cloud_filter_band=cloud_filter_band,
            calc_bands_parallel=calc_bands_parallel,
            force=force,
        )
    elif engine == "rasterstats":
        from _zonal_stats_bulk_rs import zonal_stats

        return zonal_stats(
            features_path=features_path,
            id_column=id_column,
            images_bands=images_bands,
            output_dir=output_dir,
            temp_dir=temp_dir,
            log_dir=log_dir,
            log_level=log_level,
            cloud_filter_band=cloud_filter_band,
            calc_bands_parallel=calc_bands_parallel,
            force=force,
        )
    else:
        raise ValueError(f"invalid engine: {engine}")
