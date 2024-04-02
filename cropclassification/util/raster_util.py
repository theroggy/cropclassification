from pathlib import Path
from typing import Dict, Iterable, Union

from osgeo import gdal
import rasterio

# Suppress errors
gdal.PushErrorHandler("CPLQuietErrorHandler")


def add_overviews(path: Path):
    """
    Add overviews to the file.

    Args:
        path (Path): path to the file.
    """
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


def add_band_descriptions(
    path: Path, band_descriptions: Union[Iterable[str], Dict[int, str]]
):
    """
    Add band decriptions to a raster file.

    Args:
        path (Path): the file to add band descriptions to
        band_descriptions (Iterable[str]): an Iterable with the band descriptions or a
            Dict with the band index as key (starting with 1) and the description as
            value.
    """
    # Add band descriptions
    with rasterio.open(path, "r+") as file:
        # If band_descriptions is no dict, there should be a description for each band.
        if not isinstance(band_descriptions, dict):
            if file.count != len(band_descriptions):
                raise ValueError(
                    f"number of bands ({file.count}) != number of band_descriptions "
                    f"({len(band_descriptions)}): {path}"
                )
            band_descriptions = dict(enumerate(band_descriptions, start=1))

        for index, band_description in band_descriptions.items():
            file.set_band_description(index, band_description)
