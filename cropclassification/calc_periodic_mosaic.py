"""Generate periodic mosaics."""

from datetime import datetime
from pathlib import Path

import cropclassification.helpers.config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.util import mosaic_util


def calc_periodic_mosaic_task(config_paths: list[Path], default_basedir: Path):
    """Runs a periodic mosaic using the setting in the config_paths.

    Args:
        config_paths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config file to.
    """
    # Read the configuration files
    conf.read_config(config_paths=config_paths, default_basedir=default_basedir)

    # Initialisation of the logging
    log_level = conf.general.get("log_level")
    log_dir = conf.paths.getpath("log_dir")
    logger = log_helper.main_log_init(log_dir, __name__, log_level)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Init some variables
    start_date = datetime.fromisoformat(conf.period["start_date"])
    end_date_str = conf.period["end_date"]
    if end_date_str == "{now}":
        now = datetime.now()
        end_date = datetime(now.year, now.month, now.day)
    else:
        end_date = datetime.fromisoformat(conf.period["end_date"])

    imageprofiles_to_get = list(conf.parse_image_config(conf.images["images"]))
    imageprofiles = conf._get_image_profiles(
        Path(conf.paths["image_profiles_config_filepath"])
    )

    if not conf.calc_periodic_mosaic_params.getboolean("simulate"):
        _ = mosaic_util.calc_periodic_mosaic(
            roi_bounds=tuple(conf.roi.getlistfloat("roi_bounds")),
            roi_crs=conf.roi.getint("roi_crs"),
            start_date=start_date,
            end_date=end_date,
            output_base_dir=conf.paths.getpath("images_periodic_dir"),
            imageprofiles_to_get=imageprofiles_to_get,
            imageprofiles=imageprofiles,
            images_available_delay=conf.period["images_available_delay"],
            force=False,
        )
