from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from cropclassification.helpers import log_helper
import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess._timeseries_helper as ts_helper
from cropclassification.util import openeo_util


def calc_periodic_mosaic_task(config_paths: List[Path], default_basedir: Path):
    """
    Runs a periodic mosaic using the setting in the config_paths.

    Args:
        config_paths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config
            file to.

    Raises:
        Exception: [description]
        Exception: [description]
    """
    # Read the configuration files
    conf.read_config(config_paths=config_paths, default_basedir=default_basedir)

    # Initialisation of the logging
    log_level = conf.general.get("log_level")
    log_dir = conf.dirs.getpath("log_dir")
    logger = log_helper.main_log_init(log_dir, __name__, log_level)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Init some variables
    start_date = datetime.fromisoformat(
        conf.calc_periodic_mosaic_params["start_date_str"]
    )
    end_date_str = conf.calc_periodic_mosaic_params["end_date_str"]
    if end_date_str == "{now}":
        end_date = datetime.now()
    else:
        end_date = datetime.fromisoformat(
            conf.calc_periodic_mosaic_params["end_date_str"]
        )
    end_date_subtract_days = conf.calc_periodic_mosaic_params["end_date_subtract_days"]
    if end_date_subtract_days is not None:
        end_date = end_date - timedelta(int(end_date_subtract_days))

    sensors = conf.calc_periodic_mosaic_params.getlist("sensors")
    imageprofiles = conf._get_image_profiles(
        Path(conf.marker["image_profiles_config_filepath"])
    )
    sensordata_to_get = [imageprofiles[i] for i in sensors if i in imageprofiles]

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_helper.get_monday(start_date)
    end_date = ts_helper.get_monday(end_date)

    if not conf.calc_periodic_mosaic_params.getboolean("simulate"):
        _ = openeo_util.calc_periodic_mosaic(
            roi_bounds=[161_000, 188_000, 162_000, 189_000],
            roi_crs=conf.calc_periodic_mosaic_params.getint("roi_crs"),
            start_date=start_date,
            end_date=end_date,
            days_per_period=conf.calc_periodic_mosaic_params.getint("days_per_period"),
            output_dir=Path(conf.calc_periodic_mosaic_params["dest_image_data_dir"]),
            images_to_get=sensordata_to_get,
            force=False,
        )
