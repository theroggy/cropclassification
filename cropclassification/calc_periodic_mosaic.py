from datetime import datetime, timedelta
from pathlib import Path

import cropclassification.helpers.config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.util import mosaic_util


def calc_periodic_mosaic_task(config_paths: list[Path], default_basedir: Path):
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

    imageprofiles_to_get = conf.calc_periodic_mosaic_params.getlist(
        "imageprofiles_to_get"
    )
    imageprofiles = conf._get_image_profiles(
        Path(conf.marker["image_profiles_config_filepath"])
    )

    if not conf.calc_periodic_mosaic_params.getboolean("simulate"):
        _ = mosaic_util.calc_periodic_mosaic(
            roi_bounds=conf.calc_periodic_mosaic_params.getint("roi_bounds"),
            roi_crs=conf.calc_periodic_mosaic_params.getint("roi_crs"),
            start_date=start_date,
            end_date=end_date,
            output_base_dir=Path(
                conf.calc_periodic_mosaic_params["dest_image_data_dir"]
            ),
            imageprofiles_to_get=imageprofiles_to_get,
            imageprofiles=imageprofiles,
            force=False,
        )
