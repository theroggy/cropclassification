# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of geopandas.
"""

import os
from pathlib import Path
from typing import List, Optional

import fiona
import geopandas as gpd

from cropclassification.helpers import pandas_helper as pdh


def read_file(
    path: Path, layer: str = "", columns: Optional[List[str]] = None, bbox=None
) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the path
    extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()

    if layer == "":
        layer = path.stem

    if ext_lower == ".shp":
        return gpd.read_file(str(path), bbox=bbox)
    elif ext_lower == ".gpkg":
        return gpd.read_file(str(path), layer=layer, bbox=bbox)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")


def to_file(gdf: gpd.GeoDataFrame, path: Path, layer: str = "", index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the path
    extension.

    # TODO: think about if possible/how to support adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()

    if layer == "":
        layer = path.stem

    if ext_lower == ".shp":
        if index is True:
            gdf = gdf.reset_index(inplace=False)
        gdf.to_file(str(path))

    elif ext_lower == ".gpkg":
        gdf.to_file(str(path), layer=layer, driver="GPKG")
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")


def get_crs(path: Path):
    with fiona.open(str(path), "r") as geofile:
        return geofile.crs


def get_totalbounds(path: Path):
    """
    Gets the total bounds of a geofile. Return a tuple with the bounds and the crs.

    Remark: implementation is not at all efficient!!!

    Args:
        path (str): The path to the geofile
    """
    gdf = gpd.read_file(str(path))
    return (gdf.total_bounds, gdf.crs)


def is_geofile(path: Path) -> bool:
    """
    Determines based on the path if this is a geofile.
    """
    _, file_ext = os.path.splitext(path)
    return is_geofile_ext(file_ext)


def is_geofile_ext(file_ext) -> bool:
    """
    Determines based on the file extension if this is a geofile.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower in (".shp", ".gpkg", ".geojson"):
        return True
    else:
        return False
