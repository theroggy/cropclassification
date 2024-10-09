"""Calculate zonal statistics for a vector file with many raster files."""

from pathlib import Path
from typing import Optional, Union

from ._raster_helper import *  # noqa: F403


def zonal_stats(
    vector_path: Path,
    id_column: str,
    rasters_bands: list[tuple[Path, list[str]]],
    output_dir: Path,
    stats: Union[list[str], str] = ["count", "median"],
    cloud_filter_band: Optional[str] = None,
    calc_bands_parallel: bool = True,
    engine: str = "rasterstats",
    nb_parallel: int = -1,
    force: bool = False,
):
    """Calculate zonal statistics.

    Args:
        vector_path (Path): input file with vector data.
        id_column (str): column in vector_path with the id that will be retained in the
            output files.
        rasters_bands (List[Tuple[Path, List[str]]]): List of tuples with the path to
            the raster files and the bands to calculate the zonal statistics on.
        output_dir (Path): directory to write the results to.
        stats (List[str]): statistics to calculate. Available statistics and
            special options are dependent on the `engine` specified:

                - "rasterstats": `rasterstats documentation <https://pythonhosted.org/rasterstats/manual.html#statistics>`_
                - "pyqgis": "count", "sum", "mean", "median", "std", "min", "max",
                        "range", "minority", "majority" and "variance".
                - "exactextract": `exactextract documentation <https://isciences.github.io/exactextract/operations.html>`_

        cloud_filter_band (str, optional): the band to use as a cloud filter. Only
            supported for engine "rasterstats". Defaults to None.
        calc_bands_parallel (bool, optional): True to calculate the bands in parallel.
            Only supported for engine "rasterstats". Defaults to True.
        engine (str, optional): the engine to use for the calculation. Options are
            "exactextract", "rasterstats" and "pyqgis". Defaults to "rasterstats".
        nb_parallel (int, optional): the number of parallel processes to use.
            Defaults to -1: use all available processors.
        force (bool, optional): False to skip calculating existing output files. True to
            recalculate and overwrite existing output files. Defaults to False.
    """
    if isinstance(stats, str):
        stats = [stats]

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
            nb_parallel=nb_parallel,
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
            nb_parallel=nb_parallel,
            force=force,
        )
    elif engine == "exactextract":
        from . import _zonal_stats_bulk_exactextract

        return _zonal_stats_bulk_exactextract.zonal_stats(
            vector_path=vector_path,
            rasters_bands=rasters_bands,
            output_dir=output_dir,
            stats=stats,
            columns=[id_column],
            nb_parallel=nb_parallel,
            force=force,
        )
    else:
        raise ValueError(f"invalid engine: {engine}")
