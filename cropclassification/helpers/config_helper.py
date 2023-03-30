# -*- coding: utf-8 -*-
"""
Module that manages configuration data.
"""

import configparser
import json
from pathlib import Path
import pprint
import tempfile
from typing import Dict, List, Optional

from cropclassification.util.openeo_util import ImageProfile

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


class SensorData:
    def __init__(
        self,
        imageprofile_name: str,
        imageprofile: Optional[ImageProfile] = None,
        bands: Optional[List[str]] = None,
    ):
        self.imageprofile_name = imageprofile_name
        if imageprofile is not None:
            self.imageprofile = imageprofile
        else:
            self.imageprofile = image_profiles[imageprofile_name]
        if bands is not None:
            self.bands = bands
        else:
            self.bands = self.imageprofile.bands


def read_config(config_paths: List[Path], default_basedir: Optional[Path] = None):
    # Read the configuration
    global config
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={
            "list": lambda x: [i.strip() for i in x.split(",")],
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [float(i.strip()) for i in x.split(",")],
            "dict": lambda x: json.loads(x),
            "path": lambda x: None if x is None else Path(x),
        },
        allow_no_value=True,
    )

    # Check if all config paths are ok
    for config_path in config_paths:
        if not config_path.exists():
            raise Exception(f"Config file doesn't exist: {config_path}")

    config.read(config_paths)

    # If the data_dir parameter is a relative path, try to resolve it towards
    # the default basedir of, if specfied.
    data_dir = config["dirs"].getpath("data_dir")
    if not data_dir.is_absolute():
        if default_basedir is None:
            raise Exception(
                "Config parameter dirs.data_dir is relative, but no default_basedir "
                "supplied!"
            )
        data_dir_absolute = (default_basedir / data_dir).resolve()
        print(
            "Config parameter dirs.data_dir was relative, so is now resolved to "
            f"{data_dir_absolute}"
        )
        config["dirs"]["data_dir"] = data_dir_absolute.as_posix()

    # If the marker_basedir parameter is a relative path, try to resolve it towards
    # the default basedir of, if specfied.
    marker_basedir = config["dirs"].getpath("marker_basedir")
    if not marker_basedir.is_absolute():
        if default_basedir is None:
            raise Exception(
                "Config parameter dirs.marker_basedir is relative, but no "
                "default_basedir supplied!"
            )
        marker_basedir_absolute = (default_basedir / marker_basedir).resolve()
        print(
            "Config parameter dirs.marker_basedir was relative, so is now resolved to "
            f"{marker_basedir_absolute}"
        )
        config["dirs"]["marker_basedir"] = marker_basedir_absolute.as_posix()

    # Fill out placeholder in the temp_dir (if it is there)
    tmp_dir_str = tempfile.gettempdir()
    config["dirs"]["temp_dir"] = config["dirs"]["temp_dir"].format(tmp_dir=tmp_dir_str)

    global config_paths_used
    config_paths_used = config_paths

    # Now set global variables to each section as shortcuts
    global general
    general = config["general"]
    global calc_timeseries_params
    calc_timeseries_params = config["calc_timeseries_params"]
    global calc_marker_params
    calc_marker_params = config["calc_marker_params"]
    global marker
    marker = config["marker"]
    global timeseries
    timeseries = config["timeseries"]
    global preprocess
    preprocess = config["preprocess"]
    global classifier
    classifier = config["classifier"]
    global postprocess
    postprocess = config["postprocess"]
    global columns
    columns = config["columns"]
    global dirs
    dirs = config["dirs"]

    global image_profiles
    image_profiles = _get_raster_profiles()


def parse_sensordata_to_use(input) -> Dict[str, SensorData]:
    result = None
    sensordata_parsed = None
    try:
        sensordata_parsed = json.loads(input)
    except Exception:
        pass

    if sensordata_parsed is not None:
        # It was a json object, so parse as such
        result = {}
        for imageprofile in sensordata_parsed:
            if isinstance(imageprofile, str):
                result[imageprofile] = SensorData(imageprofile)
            elif isinstance(imageprofile, dict):
                if len(imageprofile) != 1:
                    raise ValueError(
                        "invalid sensordata_to_use: this should be a single key dict: "
                        f"{imageprofile}"
                    )
                imageprofile_name = list(imageprofile.keys())[0]
                bands = list(imageprofile.values())[0]
                result[imageprofile_name] = SensorData(imageprofile_name, bands=bands)
            else:
                raise ValueError(
                    "invalid sensordata_to_use: only str or dict elements allowed, "
                    f"not: {imageprofile}"
                )
    else:
        # It was no json object, so it must be a list
        result = {i.strip(): SensorData(i.strip()) for i in input.split(",")}

    return result


def _get_raster_profiles() -> Dict[str, ImageProfile]:
    # TODO: this should move to a config file
    profiles = {}
    profiles["s2-agri"] = ImageProfile(
        name="s2-agri",
        satellite="s2",
        collection="TERRASCOPE_S2_TOC_V2",
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
        # Use the "min" reducer filters out "lightly clouded areas"
        process_options={"time_dimension_reducer": "min", "cloud_filter_band": "SCL"},
    )
    profiles["s2-ndvi"] = ImageProfile(
        name="s2-ndvi",
        satellite="s2",
        collection="TERRASCOPE_S2_NDVI_V2",
        bands=["NDVI"],
        process_options={
            "time_dimension_reducer": "max",
            "cloud_filter_band": "SCENECLASSIFICATION_20M",
        },
    )
    profiles["s1-grd-sigma0-asc"] = ImageProfile(
        name="s1-grd-sigma0-asc",
        satellite="s1",
        collection="S1_GRD_SIGMA0_ASCENDING",
        bands=["VV", "VH", "angle"],
        process_options={
            "time_dimension_reducer": "min",
        },
    )
    profiles["s1-grd-sigma0-desc"] = ImageProfile(
        name="s1-grd-sigma0-desc",
        satellite="s1",
        collection="S1_GRD_SIGMA0_DESCENDING",
        bands=["VV", "VH", "angle"],
        process_options={
            "time_dimension_reducer": "min",
        },
    )
    profiles["s1-coh"] = ImageProfile(
        name="s1-coh",
        satellite="s1",
        collection="TERRASCOPE_S1_SLC_COHERENCE_V1",
        bands=["VV", "VH"],
        process_options={
            "time_dimension_reducer": "min",
        },
    )

    return profiles


def pformat_config():
    message = f"Config files used: {pprint.pformat(config_paths_used)} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(
        {section: dict(config[section]) for section in config.sections()}
    )
    return message


def as_dict():
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict
