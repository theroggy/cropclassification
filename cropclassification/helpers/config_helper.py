"""
Module that manages configuration data.
"""

import configparser
import json
from pathlib import Path
import pprint
import tempfile
from typing import Any, Dict, List, Optional

from cropclassification.util.mosaic_util import ImageProfile

config: configparser.ConfigParser
config_paths_used: List[Path]
general: Any
calc_timeseries_params: Any
calc_marker_params: Any
calc_periodic_mosaic_params: Any
marker: Any
timeseries: Any
preprocess: Any
classifier: Any
postprocess: Any
columns: Any
dirs: Any
image_profiles: Any


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
    global calc_periodic_mosaic_params
    calc_periodic_mosaic_params = config["calc_periodic_mosaic_params"]
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
    image_profiles = _get_image_profiles(
        marker.getpath("image_profiles_config_filepath")
    )


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
                imageprofile_name = next(iter(imageprofile.keys()))
                bands = next(iter(imageprofile.values()))
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


def _get_image_profiles(image_profiles_path: Path) -> Dict[str, ImageProfile]:
    # Cropclassification gives best results with time_dimension_reducer "mean" for both
    # sentinel 2 and sentinel 1 images.
    # Init
    if not image_profiles_path.exists():
        raise ValueError(f"Config file specified does not exist: {image_profiles_path}")

    # Read config file...
    profiles_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={
            "list": lambda x: [i.strip() for i in x.split(",")],
            "dict": lambda x: None if x == "None" else json.loads(x),
        },
        allow_no_value=True,
    )
    profiles_config.read(image_profiles_path)

    # Prepare data
    profiles = {}
    for profile in profiles_config.sections():
        profiles[profile] = ImageProfile(
            name=profile,
            satellite=profiles_config[profile].get("satellite"),
            index_type=profiles_config[profile].get("index_type"),
            image_source=profiles_config[profile].get("image_source"),
            base_image_profile=profiles_config[profile].get("base_image_profile"),
            collection=profiles_config[profile].get("collection"),
            bands=profiles_config[profile].getlist("bands"),
            max_cloud_cover=profiles_config[profile].getfloat("max_cloud_cover"),
            process_options=profiles_config[profile].getdict("process_options"),
            job_options=profiles_config[profile].getdict("job_options"),
        )

    # Do some extra validations on the profiles read.
    _validate_image_profiles(profiles)

    return profiles


def _validate_image_profiles(profiles: Dict[str, ImageProfile]):
    # Check that all base_image_profile s are actually existing image profiles.
    for profile in profiles:
        base_image_profile = profiles[profile].base_image_profile
        if base_image_profile is not None and base_image_profile not in profiles:
            raise ValueError(
                f"{base_image_profile=} not found for profile {profiles[profile]}"
            )


def pformat_config():
    message = f"Config files used: {pprint.pformat(config_paths_used)} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(as_dict())

    return message


def as_dict():
    """
    Converts the config objects into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    the_dict["image_profiles"] = {}
    for image_profile in image_profiles:
        the_dict["image_profiles"][image_profile] = image_profiles[
            image_profile
        ].__dict__

    return the_dict
