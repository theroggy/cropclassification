from pathlib import Path
from typing import List, Literal, Optional, Tuple

from ._raster_helper import *  # noqa: F401, F403


def zonal_stats(
    vector_path: Path,
    id_column: str,
    rasters_bands: List[Tuple[Path, List[str]]],
    output_dir: Path,
    stats: Literal["count", "mean", "median", "std", "min", "max"],
    cloud_filter_band: Optional[str] = None,
    calc_bands_parallel: bool = True,
    engine: str = "rasterstats",
    force: bool = False,
):
    """
    Calculate zonal statistics.

    Args:
        vector_path (Path): _description_
        id_column (str): _description_
        rasters_bands (List[Tuple[Path, List[str]]]): _description_
        output_dir (Path): _description_
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    if engine == "pyqgis":
        if cloud_filter_band is not None:
            raise ValueError(
                'cloud_filter_band parameter not supported for engine "pyqgis"'
            )
        from . import _zonal_stats_bulk_pyqgis

        return _zonal_stats_bulk_pyqgis.zonal_stats(
            vector_path=vector_path,
            rasters_bands=rasters_bands,
            output_dir=output_dir,
            stats=stats,
            columns=[id_column],
            force=force,
        )
    elif engine == "rasterstats":
        from . import _zonal_stats_bulk_rs

        return _zonal_stats_bulk_rs.zonal_stats(
            vector_path=vector_path,
            id_column=id_column,
            rasters_bands=rasters_bands,
            output_dir=output_dir,
            stats=stats,
            cloud_filter_band=cloud_filter_band,
            calc_bands_parallel=calc_bands_parallel,
            force=force,
        )
    else:
        raise ValueError(f"invalid engine: {engine}")
