"""Run a cover classification."""

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# Import geofilops here already, if tensorflow is loaded first leads to dll load errors
import geofileops as gfo  # noqa: F401
import pyproj

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.preprocess import _timeseries_helper as ts_helper
from cropclassification.preprocess import timeseries as ts
from cropclassification.util import mosaic_util


def run_cover(
    config_paths: list[Path],
    default_basedir: Path,
    config_overrules: list[str] = [],
):
    """Runs a the cover marker using the setting in the config_paths.

    Args:
        config_paths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config
            file to.
        config_overrules (List[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified as a
            list of "<section>.<parameter>=<value>". Defaults to [].
    """
    # Read the configuration files
    conf.read_config(
        config_paths, default_basedir=default_basedir, overrules=config_overrules
    )

    # Main initialisation of the logging
    log_level = conf.general.get("log_level")
    log_dir = conf.paths.getpath("log_dir")
    logger = log_helper.main_log_init(log_dir, __name__, log_level)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Read the info about the run
    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    input_parcel_filetype = conf.calc_marker_params["input_parcel_filetype"]
    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")

    input_dir = conf.paths.getpath("input_dir")
    input_parcel_path = input_dir / input_parcel_filename

    refe_dir = conf.paths.getpath("refe_dir")
    classes_refe_path = refe_dir / classes_refe_filename

    # Check if the necessary input files exist...
    for path in [classes_refe_path, input_parcel_path]:
        if path is not None and not path.exists():
            message = f"Input file doesn't exist, so STOP: {path}"
            logger.critical(message)
            raise ValueError(message)

    # Get some general config
    data_ext = conf.general["data_ext"]
    geofile_ext = conf.general["geofile_ext"]

    # -------------------------------------------------------------
    # The real work
    # -------------------------------------------------------------
    # STEP 1: prepare parcel data for classification and image data extraction
    # -------------------------------------------------------------

    # Prepare the input data for optimal image data extraction:
    #    1) apply a negative buffer on the parcel to evade mixels
    #    2) remove features that became null because of buffer
    input_preprocessed_dir = conf.paths.getpath("input_preprocessed_dir")
    buffer = conf.timeseries.getfloat("buffer")
    input_parcel_nogeo_path = (
        input_preprocessed_dir / f"{input_parcel_filename.stem}{data_ext}"
    )
    imagedata_input_parcel_filename = (
        f"{input_parcel_filename.stem}_bufm{buffer:g}{geofile_ext}"
    )
    imagedata_input_parcel_path = (
        input_preprocessed_dir / imagedata_input_parcel_filename
    )
    ts_helper.prepare_input(
        input_parcel_path=input_parcel_path,
        output_imagedata_parcel_input_path=imagedata_input_parcel_path,
        output_parcel_nogeo_path=input_parcel_nogeo_path,
    )

    # STEP 2: Calculate the timeseries data needed
    # -------------------------------------------------------------
    # Get the time series data (eg. S1, S2,...) to be used for the classification
    # Result: data is put in files in timeseries_periodic_dir, in one file per
    #         date/period
    timeseries_periodic_dir = conf.paths.getpath("timeseries_periodic_dir")
    timeseries_periodic_dir /= f"{imagedata_input_parcel_path.stem}"
    start_date = datetime.fromisoformat(conf.period["start_date"])
    end_date = datetime.fromisoformat(conf.period["end_date"])
    images_to_use = conf.parse_image_config(conf.images["images"])

    ts.calc_timeseries_data(
        input_parcel_path=imagedata_input_parcel_path,
        roi_bounds=tuple(conf.roi.getlistfloat("roi_bounds")),
        roi_crs=pyproj.CRS.from_user_input(conf.roi.get("roi_crs")),
        start_date=start_date,
        end_date=end_date,
        images_to_use=images_to_use,
        timeseries_periodic_dir=timeseries_periodic_dir,
    )

    # STEP 3: Determine the cover for the parcels for all periods
    # -------------------------------------------------------------
    # Loop over all periods
    periods = mosaic_util._prepare_periods(
        start_date, end_date, period_name="weekly", period_days=None
    )
    cover_periodic_dir = conf.paths.getpath("cover_periodic_dir")
    cover_dir = cover_periodic_dir / input_parcel_nogeo_path.stem
    cover_dir.mkdir(parents=True, exist_ok=True)
    export_geo = True
    force = True

    for period in periods:
        period_start_date = period["start_date"].strftime("%Y-%m-%d")
        period_end_date = period["end_date"].strftime("%Y-%m-%d")
        output_name = f"cover_{period_start_date}_{period_end_date}{data_ext}"
        output_path = cover_dir / output_name

        if output_path.exists():
            if force:
                gfo.remove(output_path)
            else:
                continue

        # Collect all data needed to do the classification in one input file
        tmp_dir = Path(tempfile.mkdtemp(prefix="calc_index_"))
        tmp_path = tmp_dir / f"{input_parcel_nogeo_path.stem}.sqlite"
        ts.collect_and_prepare_timeseries_data(
            input_parcel_path=input_parcel_nogeo_path,
            timeseries_dir=timeseries_periodic_dir,
            output_path=tmp_path,
            start_date=period["start_date"],
            end_date=period["end_date"],
            images_to_use=images_to_use,
            parceldata_aggregations_to_use=["mean", "median", "stdev"],
        )

        columns = [
            column.lower()
            for column in gfo.get_layerinfo(
                tmp_path, layer="info", raise_on_nogeom=False
            ).columns
        ]
        vv_mean = next(column for column in columns if column.endswith("vv_mean"))
        vv_median = next(column for column in columns if column.endswith("vv_median"))
        vh_median = next(column for column in columns if column.endswith("vh_median"))
        vh_stdev = next(column for column in columns if column.endswith("vh_stdev"))

        sql_stmt = f"""
            SELECT "UID"
                  ,CASE
                     WHEN "{vv_mean}" IS NULL THEN 'NODATA'
                     WHEN ( ("{vv_median}" < 0.027 OR "{vh_stdev}" < 0.0027)
                            AND "{vh_median}" < 0.07
                          ) THEN 'water'
                     WHEN "{vv_median}"/"{vh_median}" >= 5.9 AND "{vh_median}" < 0.014
                          THEN 'bare-soil'
                     WHEN "{vv_mean}" > 0.25 THEN 'urban'
                     ELSE 'other'
                   END AS cover
              FROM "info"
        """
        result_df = gfo.read_file(tmp_path, sql_stmt=sql_stmt)
        gfo.to_file(result_df, output_path)

        if export_geo:
            parcels_gdf = gfo.read_file(input_parcel_path, columns=["UID"])
            result_gdf = parcels_gdf.merge(result_df, on="UID")
            geo_name = f"cover_{period_start_date}_{period_end_date}{geofile_ext}"
            geo_path = cover_dir / geo_name
            gfo.to_file(result_gdf, geo_path)

        shutil.rmtree(tmp_dir)

    logging.shutdown()
