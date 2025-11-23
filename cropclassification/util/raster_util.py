"""Raster utility functions."""

from collections.abc import Iterable
from pathlib import Path

import rasterio
from osgeo import gdal

# Suppress errors
gdal.PushErrorHandler("CPLQuietErrorHandler")


def add_overviews(
    path: Path, min_pixels: int = 512, resampling: str = "average"
) -> None:
    """Add overviews to the file.

    Args:
        path (Path): path to the file.
        min_pixels (int, optional): minimum number of pixels in a zoom level to
            calculate overviews for. Defaults to 512.
        resampling (str, optional): resampling method. Defaults to 'average'.
    """
    with rasterio.open(path, "r+") as dst:
        factors = []
        for power in range(1, 999):
            factor = pow(2, power)
            if dst.width / factor < min_pixels or dst.height / factor < min_pixels:
                break
            factors.append(factor)
        if len(factors) > 0:
            dst.build_overviews(factors, rasterio.enums.Resampling[resampling])
            dst.update_tags(ns="rio_overview", resampling=resampling)


def get_band_descriptions(path: Path) -> dict[str, int]:
    """Get the band descriptions of a raster file.

    Args:
        path (Path): the file to get the band descriptions from

    Returns:
        dict: the band descriptions
    """
    with rasterio.open(path, "r") as file:
        return {name: index for index, name in enumerate(file.descriptions, start=1)}


def set_band_descriptions(
    path: Path,
    band_descriptions: Iterable[str] | dict[int, str] | str,
    overwrite: bool = True,
) -> None:
    """Add band descriptions to a raster file.

    Args:
        path (Path): the file to add band descriptions to
        band_descriptions (Iterable, dict, str): an Iterable with the band descriptions,
            a Dict with the band index as key (starting with 1) and the description
            as value or a string if the file has a single band.
        overwrite (bool): True to overwrite existing band descriptions. If False, if any
            band does not have a description, all band descriptions are overwritten.
            Defaults to True.
    """
    # If band_descriptions is a string, make it a list to avoid each char being treated
    # as a band name.
    if isinstance(band_descriptions, str):
        band_descriptions = [band_descriptions]

    # Add band descriptions
    with rasterio.open(path, "r+") as file:
        # If overwrite is False and all bands already have a description, return.
        if not overwrite and all(file.descriptions):
            return

        # If band_descriptions is no dict, there should be a description for each band.
        if not isinstance(band_descriptions, dict):
            band_descriptions = list(band_descriptions)
            if file.count != len(band_descriptions):
                raise ValueError(
                    f"number of bands ({file.count}) != number of band_descriptions "
                    f"({len(band_descriptions)}): {path}"
                )
            band_descriptions = dict(enumerate(band_descriptions, start=1))

        for index, band_description in band_descriptions.items():
            index = int(index)

            # Compare the current band description if it exists.
            if len(file.descriptions) >= index:
                curr_description = file.descriptions[index - 1]
                if curr_description == band_description:
                    continue

            file.set_band_description(index, band_description)
