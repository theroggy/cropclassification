"""Run a cover classification."""

import logging
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

# Import geofilops here already, if tensorflow is loaded first leads to dll load errors
import geofileops as gfo
import pandas as pd
import pyproj

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import dir_helper, log_helper
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.preprocess import _timeseries_helper as ts_helper
from cropclassification.preprocess import timeseries as ts
from cropclassification.util import mosaic_util


def run_cover(
    config_paths: list[Path],
    default_basedir: Path,
    config_overrules: list[str] | None = None,
) -> None:
    """Run a the cover marker using the setting in the config_paths.

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
    global logger  # noqa: PLW0603
    logger = log_helper.main_log_init(log_dir, __name__, log_level)

    logger.warning("This is a POC for a cover marker, so not for operational use!")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    force = True
    force = False
    markertype = conf.marker["markertype"]

    if not markertype.startswith(("COVER", "ONBEDEKT")):
        raise ValueError(f"Invalid markertype {markertype}, expected COVER_XXX")

    # Create run dir to be used for the results
    reuse_last_run_dir = conf.calc_marker_params.getboolean("reuse_last_run_dir")
    run_dir = dir_helper.create_run_dir(
        conf.paths.getpath("marker_dir"), reuse_last_run_dir
    )

    parcels_marker_path = (
        run_dir / f"{markertype}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    )
    parcel_selected_sqlite_path = parcels_marker_path.with_suffix(".sqlite")

    start_date = datetime.fromisoformat(conf.period["start_date"])
    end_date = datetime.fromisoformat(conf.period["end_date"])

    # If the config needs to be reused as well, load it, else write it
    config_used_path = run_dir / "config_used.ini"
    if reuse_last_run_dir and run_dir.exists() and config_used_path.exists():
        config_paths.append(config_used_path)
        logger.info(f"Run dir config needs to be reused, so {config_paths}")
        conf.read_config(
            config_paths=config_paths,
            default_basedir=default_basedir,
            overrules=config_overrules,
        )
        logger.info(
            "Write new config_used.ini, because some parameters might have been added"
        )
        with config_used_path.open("w") as config_used_file:
            conf.config.write(config_used_file)
    else:
        # Copy the config files to a config dir for later notice
        configfiles_used_dir = run_dir / "configfiles_used"
        if configfiles_used_dir.exists():
            configfiles_used = sorted(configfiles_used_dir.glob("*.ini"))
            conf.read_config(
                config_paths=configfiles_used,
                default_basedir=default_basedir,
                overrules=config_overrules,
            )
        else:
            configfiles_used_dir.mkdir(parents=True)
            for idx, config_path in enumerate(config_paths):
                # Prepend with idx so the order of config files is retained...
                dst = configfiles_used_dir / f"{idx}_{config_path.name}"
                shutil.copy(config_path, dst)

            # Write the resolved complete config, so it can be reused
            logger.info("Write config_used.ini, so it can be reused later on")
            with config_used_path.open("w") as config_used_file:
                conf.config.write(config_used_file)

    if not parcels_marker_path.exists() or force:
        # Read the info about the run
        input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
        input_dir = conf.paths.getpath("input_dir")
        input_parcel_path = input_dir / input_parcel_filename

        # Check if the necessary input files exist...
        for path in [input_parcel_path]:
            if path is not None and not path.exists():
                message = f"Input file doesn't exist, so STOP: {path}"
                logger.critical(message)
                raise ValueError(message)

        # Depending on the specific markertype, export only the relevant parcels
        input_preprocessed_dir = conf.paths.getpath("input_preprocessed_dir")
        input_preprocessed_dir.mkdir(parents=True, exist_ok=True)
        if markertype in ("COVER", "COVER_EEF_VOORJAAR", "ONBEDEKT_NA_WINTER"):
            input_parcel_filename = f"{input_parcel_path.stem}_{markertype}.gpkg"
            input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

            where = """
                "ALL_BEST" like '%EEF%'
            """
            gfo.copy_layer(
                input_parcel_path, input_parcel_filtered_path, where=where, force=force
            )
            input_parcel_path = input_parcel_filtered_path
        elif markertype in ("ONBEDEKT_NA_ZOMER", "COVER_BMG_MEG_MEV_EEF_NAJAAR"):
            input_parcel_filename = f"{input_parcel_path.stem}_{markertype}.gpkg"
            input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

            where = """
                "ALL_BEST" like '%BMG%'
                OR "ALL_BEST" like '%MEV%'
                OR "ALL_BEST" like '%MEG%'
                OR "ALL_BEST" like '%EEF%'
                OR "ALL_BEST" like '%TBG%'
                OR "ALL_BEST" like '%EBG%'
            """
            gfo.copy_layer(
                input_parcel_path, input_parcel_filtered_path, where=where, force=force
            )
            input_parcel_path = input_parcel_filtered_path

        elif markertype == "COVER_EEB_VOORJAAR":
            # Fallow parcels with a premium having as requirement that there is no activity
            # on them from 15/01 till 10/04 + that they have maize as main crop.
            input_parcel_filename = f"{input_parcel_path.stem}_EEB.gpkg"
            input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

            where = "ALL_BEST like '%EEB%' AND GWSCOD_V = '83' AND GWSCOD_H IN ('201', '202')"
            gfo.copy_layer(
                input_parcel_path, input_parcel_filtered_path, where=where, force=force
            )
            input_parcel_path = input_parcel_filtered_path

        elif markertype in ("COVER_TBG_BMG_VOORJAAR", "COVER_TBG_BMG_NAJAAR"):
            # Grassland parcels with a premium having a requirement that they cannot be
            # resown -> should never be ploughed.
            input_parcel_filename = f"{input_parcel_path.stem}_TBG_BMG.gpkg"
            input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

            where = "ALL_BEST like '%TBG%' OR ALL_BEST like '%BMG%'"
            gfo.copy_layer(
                input_parcel_path, input_parcel_filtered_path, where=where, force=force
            )
            input_parcel_path = input_parcel_filtered_path

        else:
            raise ValueError(f"Invalid {markertype=}")

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
        buffer = conf.timeseries.getfloat("buffer")
        input_parcel_nogeo_path = (
            input_preprocessed_dir / f"{input_parcel_path.stem}{data_ext}"
        )

        imagedata_input_parcel_filename = (
            f"{input_parcel_path.stem}_bufm{buffer:g}{geofile_ext}"
        )
        imagedata_input_parcel_path = (
            input_preprocessed_dir / imagedata_input_parcel_filename
        )
        ts_helper.prepare_input(
            input_parcel_path=input_parcel_path,
            output_imagedata_parcel_input_path=imagedata_input_parcel_path,
            output_parcel_nogeo_path=input_parcel_nogeo_path,
            force=force,
        )

        # STEP 2: Calculate the timeseries data needed
        # -------------------------------------------------------------
        # Get the time series data (eg. S1, S2,...) to be used for the classification
        # Result: data is put in files in timeseries_periodic_dir, in one file per
        #         date/period
        timeseries_periodic_dir = conf.paths.getpath("timeseries_periodic_dir")
        timeseries_periodic_dir /= f"{imagedata_input_parcel_path.stem}"
        images_to_use = conf.parse_image_config(conf.images["images"])

        ts.calc_timeseries_data(
            input_parcel_path=imagedata_input_parcel_path,
            roi_bounds=tuple(conf.roi.getlistfloat("roi_bounds")),
            roi_crs=pyproj.CRS.from_user_input(conf.roi.get("roi_crs")),
            start_date=start_date,
            end_date=end_date,
            images_to_use=images_to_use,
            timeseries_periodic_dir=timeseries_periodic_dir,
            force=force,
        )

        # STEP 3: Determine the cover for the parcels for all periods
        # -------------------------------------------------------------
        # Loop over all periods
        periods = mosaic_util._prepare_periods(
            start_date, end_date, period_name="weekly", period_days=None
        )
        cover_dir = run_dir / input_parcel_nogeo_path.stem
        cover_dir.mkdir(parents=True, exist_ok=True)

        on_error = "warn"
        parcels_cover_paths = []

        for period in periods:
            period_start_date = period["start_date"].strftime("%Y-%m-%d")
            period_end_date = period["end_date"].strftime("%Y-%m-%d")
            output_name = f"cover_{period_start_date}_{period_end_date}{data_ext}"
            output_path = cover_dir / output_name

            try:
                geo_path = output_path.with_suffix(geofile_ext)
                _calc_cover(
                    input_parcel_path=input_parcel_path,
                    timeseries_periodic_dir=timeseries_periodic_dir,
                    images_to_use=images_to_use,
                    start_date=period["start_date"],
                    end_date=period["end_date"],
                    parcel_columns=None,
                    output_path=output_path,
                    output_geo_path=geo_path,
                    force=force,
                )
                parcels_cover_paths.append(geo_path)

            except Exception as ex:
                message = f"Error calculating {output_path.stem}"
                if on_error == "warn":
                    logger.exception(f"Error calculating {output_path.stem}")
                    warnings.warn(message, category=RuntimeWarning, stacklevel=1)
                elif on_error == "raise":
                    raise RuntimeError(message) from ex
                else:
                    logger.error(
                        f"invalid value for on_error: {on_error}, 'raise' assumed"
                    )
                    raise RuntimeError(message) from ex

        # STEP 4: Create the final list of parcels
        # ----------------------------------------
        # Consolidate the list of parcels that need to be controlled
        parcels_selected = None
        for path in parcels_cover_paths:
            parcels_selected_path = path
            parcels = gfo.read_file(parcels_selected_path, ignore_geometry=True)

            if parcels_selected is None:
                parcels_selected = parcels
            else:
                parcels_selected = pd.concat([parcels_selected, parcels])

        # Determine max probability for every parcel
        assert parcels_selected is not None
        input_info = gfo.get_layerinfo(input_parcel_path)
        cols_to_keep = [*list(input_info.columns), "pred1"]
        if "provincie" in parcels_selected.columns:
            cols_to_keep.append("provincie")

        parcels_selected = (
            parcels_selected[[*cols_to_keep, "pred1_prob"]]
            .sort_values("pred1_prob", ascending=False)
            .groupby(conf.columns["id"], dropna=False, as_index=False)
            .first()  # Take the highest pred1_prob per id
        )

        # Add pred_consolidated based on max pred1_proba
        parcels_selected["pred_consolidated"] = parcels_selected["pred1_prob"].apply(
            _categorize_pred
        )

        # Add pred_cons_status based on pred_consolidated
        parcels_selected["pred_cons_status"] = parcels_selected[
            "pred_consolidated"
        ].apply(lambda x: "NOK" if x in ("NODATA", "DOUBT") else "OK")

        # Export to excel
        pdh.to_excel(parcels_selected, parcels_marker_path, index=False)
        pdh.to_file(parcels_selected, parcel_selected_sqlite_path, index=False)

    # STEP 5: reporting
    # ----------------------------------------

    input_dir = conf.paths.getpath("input_dir")
    input_groundtruth_filename = conf.calc_marker_params.getpath(
        "input_groundtruth_filename"
    )

    if input_groundtruth_filename is not None:
        input_groundtruth_path = input_dir / input_groundtruth_filename
    else:
        input_groundtruth_path = None

    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")
    refe_dir = conf.paths.getpath("refe_dir")
    classes_refe_path = refe_dir / classes_refe_filename

    original_input_parcel_filename = conf.calc_marker_params.getpath(
        "input_parcel_filename"
    )
    original_input_parcel_path = input_dir / original_input_parcel_filename

    report(
        input_parcel_path=original_input_parcel_path,
        parcels_selected_path=parcel_selected_sqlite_path,
        ground_truth_path=input_groundtruth_path,
        classes_refe_path=classes_refe_path,
        id_column=conf.columns["id"],
        start_date=start_date,
        end_date=end_date,
    )

    logging.shutdown()


def _categorize_pred(x: float | str) -> str:
    if pd.isna(x):
        return "NODATA"
    try:
        x_num = float(x)
        if x_num > 0.5:
            return "ONBEDEKT"
        elif x_num > 0.4:
            return "DOUBT"
        else:
            return "BEDEKT"
    except (ValueError, TypeError):
        return "NODATA"


def _calc_cover(
    input_parcel_path: Path,
    timeseries_periodic_dir: Path,
    images_to_use: dict[str, conf.ImageConfig],
    start_date: datetime,
    end_date: datetime,
    parcel_columns: list[str] | None,
    output_path: Path,
    output_geo_path: Path | None = None,
    force: bool = False,
) -> None:
    logger.info(f"start processing {output_path}")

    if output_path.exists():
        if force:
            gfo.remove(output_path)
        elif output_geo_path is not None and not output_geo_path.exists():
            # Geo file is asked but missing: we need to recalculate the output file as
            # well because we need temporary files to create the geo file.
            gfo.remove(output_path)
        else:
            return

    id_column = conf.columns["id"]
    result_df = None
    # Collect all data needed to do the classification in one input file
    tmp_dir = Path(tempfile.mkdtemp(prefix="calc_index_"))
    tmp_path = tmp_dir / f"{input_parcel_path.stem}.sqlite"
    ts.collect_and_prepare_timeseries_data(
        input_parcel_path=input_parcel_path,
        timeseries_dir=timeseries_periodic_dir,
        output_path=tmp_path,
        start_date=start_date,
        end_date=end_date,
        images_to_use=images_to_use,
        parceldata_aggregations_to_use=["count", "mean", "median", "stdev"],
        max_fraction_null=1,
        force=force,
    )

    info = gfo.get_layerinfo(tmp_path, layer="info", raise_on_nogeom=False)
    columns = {}
    for column in info.columns:
        column_lower = column.lower()
        if "-asc-" in column_lower:
            orbit = "asc"
        elif "-desc-" in column_lower:
            orbit = "desc"
        elif column_lower.startswith("s2-ndvi-weekly"):
            if column_lower.endswith("_ndvi_median"):
                columns["ndvi_median"] = column
        else:
            continue

        key = f"s1{orbit}_{'_'.join(column_lower.split('-')[-1].split('_')[-2:])}"
        columns[key] = column

    # Minimum NDVI where we judge some vegetation is present
    ndvi_vegetation_min = 0.35

    # Remarks:
    #   - for asc, asc_vh_median seems typically lower -> different thresshold
    sql = f"""
        SELECT "{id_column}"
            ,CASE
                WHEN cover_s1_asc IN ('NODATA', 'multi') THEN cover_s1_desc
                WHEN cover_s1_desc IN ('NODATA', 'multi') THEN cover_s1_asc
                WHEN cover_s1_asc = 'water' AND cover_s1_desc = 'water' THEN 'water'
                --WHEN (cover_s1_asc = 'bare-soil' OR cover_s1_desc = 'bare-soil') THEN 'bare-soil'
                WHEN (cover_s1_asc IN ('bare-soil', 'water')
                    AND cover_s1_desc IN ('bare-soil', 'water')
                    ) THEN 'bare-soil'
                WHEN cover_s1_asc = 'urban' or cover_s1_desc = 'urban' THEN 'urban'
                ELSE 'other'
            END cover_s1
            ,cover_s1_asc
            ,cover_s1_desc
            ,cover_s2_ndvi
            ,s2_ndvi_median
        FROM (
              SELECT "{id_column}"
                ,CASE
                    WHEN "{columns["s1asc_vv_mean"]}" IS NULL OR "{columns["s1asc_vv_mean"]}" = 0
                        THEN 'NODATA'
                    WHEN ( "{columns["s1asc_vh_stdev"]}" > 0.5
                            OR "{columns["s1asc_vv_stdev"]}" > 0.5
                        ) THEN 'multi'
                    WHEN ( "{columns["s1asc_vh_median"]}" < 0.07
                            AND ("{columns["s1asc_vv_median"]}" < 0.027
                                OR "{columns["s1asc_vh_stdev"]}" < 0.0027)
                        ) THEN 'water'
                    WHEN "{columns["s1asc_vv_median"]}"/"{columns["s1asc_vh_median"]}" >= 5.9
                        AND "{columns["s1asc_vh_median"]}" < 0.013
                        THEN 'bare-soil'
                    WHEN "{columns["s1asc_vv_mean"]}" > 0.25 THEN 'urban'
                    ELSE 'other'
                END AS cover_s1_asc
                ,CASE
                    WHEN "{columns["s1desc_vv_mean"]}" IS NULL OR "{columns["s1desc_vv_mean"]}" = 0
                        THEN 'NODATA'
                    WHEN ( "{columns["s1desc_vh_stdev"]}" > 0.5
                            OR "{columns["s1desc_vv_stdev"]}" > 0.5
                        ) THEN 'multi'
                    WHEN ( "{columns["s1desc_vh_median"]}" < 0.07
                            AND ("{columns["s1desc_vv_median"]}" < 0.027
                                OR "{columns["s1desc_vh_stdev"]}" < 0.0027)
                        ) THEN 'water'
                    WHEN "{columns["s1desc_vv_median"]}"/"{columns["s1desc_vh_median"]}" >= 5.9
                        AND "{columns["s1desc_vh_median"]}" < 0.015
                        THEN 'bare-soil'
                    WHEN "{columns["s1desc_vv_mean"]}" > 0.25 THEN 'urban'
                    ELSE 'other'
                END AS cover_s1_desc
                ,CASE WHEN "{columns["ndvi_median"]}" IS NULL
                                OR "{columns["ndvi_median"]}" = 0 THEN 'NODATA'
                      WHEN "{columns["ndvi_median"]}" < {ndvi_vegetation_min} THEN 'bare-soil'
                      ELSE 'other'
                 END AS cover_s2_ndvi
                 ,"{columns["ndvi_median"]}" AS s2_ndvi_median
              FROM "info" info
            ) covers
    """  # noqa: E501

    # Add pred1_prob column based on all previous columns
    # TODO: "AND s2_ndvi_median <> 0" should not be in below expression, but apparently
    # s2_ndvi_median is never NULL -> problem with zonal stats?
    sql = f"""
        SELECT sub.*
                ,CASE
                    WHEN s2_ndvi_median IS NOT NULL AND s2_ndvi_median >= {ndvi_vegetation_min} THEN 0.0
                    WHEN s2_ndvi_median IS NOT NULL AND s2_ndvi_median <> 0 AND s2_ndvi_median < {ndvi_vegetation_min} THEN 1 - s2_ndvi_median
                    WHEN cover_s1 = 'NODATA' THEN NULL
                    WHEN cover_s1_asc = 'bare-soil' THEN 0.5
                    ELSE 0.0
                END AS pred1_prob
          FROM ({sql}) sub
    """  # noqa: E501

    # Add pred1 column based on pred1_prob
    sql = f"""
        SELECT sub2.*
                ,CASE
                    WHEN pred1_prob IS NULL THEN 'NODATA'
                    WHEN pred1_prob > 0.5 THEN 'bare-soil'
                    WHEN pred1_prob > 0.4 THEN 'DOUBT'
                    ELSE 'other'
                END AS pred1
          FROM ({sql}) sub2
    """
    result_df = pdh.read_file(tmp_path, sql=sql)
    pdh.to_file(result_df, output_path)

    if output_geo_path is not None:
        # If a geo file is asked, always recreate it so it is in sync with the output
        # file
        gfo.remove(output_geo_path, missing_ok=True)
        parcels_gdf = pdh.read_file(
            input_parcel_path,
            ignore_geometry=False,
            columns=parcel_columns,
        )

        # Determine province
        parcels_gdf["PRC_NIS"] = parcels_gdf["PRC_NIS"].fillna(990000)
        prov = {
            10000: "ANTW",
            20000: "VLBR",
            30000: "WVLA",
            40000: "OVLA",
            70000: "LIMB",
            990000: "ONBEKEND",
        }
        parcels_gdf["provincie"] = (parcels_gdf["PRC_NIS"].astype(int) / 10000).astype(
            int
        ) * 10000
        parcels_gdf["provincie"] = parcels_gdf["provincie"].map(prov)

        # Merge satellite data
        if result_df is None:
            result_df = pdh.read_file(output_path)
        result_gdf = parcels_gdf.merge(result_df, on=id_column)

        satdata_df = pdh.read_file(tmp_path)

        """
        # Add the ratio of the median of the VV and VH bands
        satdata_df["vvdvh_median_asc"] = (
            satdata_df[columns["asc_vv_median"]] / satdata_df[columns["asc_vh_median"]]
        )
        satdata_df["vvdvh_median_desc"] = (
            satdata_df[columns["desc_vv_median"]]
            / satdata_df[columns["desc_vh_median"]]
        )
        """

        result_gdf = result_gdf.merge(satdata_df, on=id_column)
        gfo.to_file(result_gdf, output_geo_path)

    shutil.rmtree(tmp_dir)


def _select_parcels_BMG_MEG_MEV_EEF(
    input_geo_path: Path, output_geo_path: Path
) -> None:
    """Select parcels based on the cover marker."""
    # Select the relevant parcels based on the cover marker
    info = gfo.get_layerinfo(input_geo_path)
    columns = {}
    for column in info.columns:
        column_lower = column.lower()
        if not column_lower.startswith("s2-ndvi-weekly"):
            continue
        if column_lower.endswith("_ndvi_median"):
            columns["ndvi_median"] = column

    # Zone used in the selection of 10/2024 to reduce number parcels
    zone_filter = """
        AND ( PRC_NIS IN (
                12014, 23025, 23096, 24028, 24054, 24107, 24133, 24135, 32010, 32011
              )
              OR (PRC_NIS > 70000 AND PRC_NIS <> 73109)
            )
    """
    zone_filter = ""

    # Filter used in the selection of 10/2024
    where = f"""
            ( ("ALL_BEST" like '%MEV%'
                OR "ALL_BEST" like '%MEG%'
                OR "ALL_BEST" like '%EEF%'
              )
              AND ( ( "{columns["ndvi_median"]}" <> 0
                      AND "{columns["ndvi_median"]}" < 0.3
                    )
                    OR "cover_s1" = 'bare-soil'
                  )
            )
            OR ("ALL_BEST" like '%BMG%'
                AND (  ( "{columns["ndvi_median"]}"<> 0
                        and "{columns["ndvi_median"]}"< 0.3
                        )
                        OR ("cover_s1" = 'bare-soil'
                            --AND "s1-grd-sigma0-desc-weekly_20240930_VH_count"<> 0
                            --AND "s1-grd-sigma0-asc-weekly_20240930_VH_count"<> 0
                            {zone_filter}
                        )
                    )
            )
    """
    gfo.copy_layer(input_geo_path, output_geo_path, where=where)


def _filter_parcels_with_enough_sentinel_data(
    input_geo_path: Path, output_geo_path: Path
) -> None:
    """Select parcels based on the ndvi median/s1 cover."""
    # Select the relevant parcels based on the cover marker
    info = gfo.get_layerinfo(input_geo_path)
    columns = {}
    for column in info.columns:
        column_lower = column.lower()
        if not column_lower.startswith("s2-ndvi-weekly"):
            continue
        if column_lower.endswith("_ndvi_median"):
            columns["ndvi_median"] = column

    where = f"""
             ( ( "{columns["ndvi_median"]}" <> 0
                      AND "{columns["ndvi_median"]}" < 0.35
                    )
                    OR ( cover_s1 = 'bare-soil'
                         AND "{columns["ndvi_median"]}" = 0  -- no NDVI available
               )
             )
    """
    gfo.copy_layer(input_geo_path, output_geo_path, where=where)


def report(
    input_parcel_path: Path,
    parcels_selected_path: Path,
    ground_truth_path: Path | None = None,
    classes_refe_path: Path | None = None,
    id_column: str = "UID",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force: bool = False,
) -> None:
    """Create a report for the cover marker.

    Args:
        input_parcel_path: Path to the original input parcel file.
        parcels_selected_path: Path to the selected parcels file with predictions.
        ground_truth_path: Path to the ground truth file. If None, only general
            statistics are generated.
        classes_refe_path: Path to the classes reference file to determine crop
        id_column: Name of the ID column to use for matching parcels.
        start_date: Start date of the detection period. Ground truth observations
            before this date are filtered out.
        end_date: End date of the detection period. Ground truth observations
            after end_date + 2 months are filtered out.
        force: Whether to force re-creation of the report.
    """
    report_dir = parcels_selected_path.parent
    report_path = report_dir / f"{parcels_selected_path.stem}_accuracy_report.html"

    if report_path.exists() and not force:
        logger.info(f"Report already exists, skipping: {report_path}")
        return

    logger.info("Generating report...")
    stats_list = []

    parcels_selected_df = pdh.read_file(parcels_selected_path)
    logger.info(f"Loaded {len(parcels_selected_df)} parcels from results file")
    total_parcels = len(parcels_selected_df)

    pred_column = None
    if "pred_consolidated" in parcels_selected_df.columns:
        pred_column = "pred_consolidated"
    elif "pred1" in parcels_selected_df.columns:
        pred_column = "pred1"
    else:
        logger.warning("No prediction column found in results file")
        return

    pred_counts = parcels_selected_df[pred_column].value_counts()
    logger.info(f"Prediction distribution: {pred_counts.to_dict()}")

    # General statistics by prediction
    for pred_cat, count in pred_counts.items():
        percentage = (count / total_parcels) * 100
        stats_list.append(
            {
                "category": f"Prediction: {pred_cat}",
                "count": count,
                "percentage": percentage,
                "total_parcels": total_parcels,
            }
        )

    # Add overall summary
    stats_list.insert(
        0,
        {
            "category": "Total parcels",
            "count": total_parcels,
            "percentage": 100.0,
            "total_parcels": total_parcels,
        },
    )

    if (
        ground_truth_path is not None
        and ground_truth_path.exists()
        and classes_refe_path is not None
        and classes_refe_path.exists()
    ):
        logger.info("Ground truth available, calculating accuracy metrics...")
        classes_refe_df = pdh.read_file(classes_refe_path)
        logger.info(f"Loaded {len(classes_refe_df)} crop codes from reference file")

        groundtruth_df = pdh.read_file(ground_truth_path)
        logger.info(f"Loaded {len(groundtruth_df)} parcels from ground truth file")

        # Filter out records that have neither NATEELT_CTRL_COD nor VASTSTELLINGEN
        # These records cannot be used for validation
        if "NATEELT_CTRL_COD" in groundtruth_df.columns:
            before_filter = len(groundtruth_df)

            has_nateelt = groundtruth_df["NATEELT_CTRL_COD"].notna() & (
                groundtruth_df["NATEELT_CTRL_COD"].astype(str).str.strip() != ""
            )

            has_vaststellingen = False
            if "VASTSTELLINGEN" in groundtruth_df.columns:
                has_vaststellingen = groundtruth_df["VASTSTELLINGEN"].notna() & (
                    groundtruth_df["VASTSTELLINGEN"].astype(str).str.strip() != ""
                )

            groundtruth_df = groundtruth_df[has_nateelt | has_vaststellingen].copy()
            logger.info(
                f"Filtered out records without NATEELT_CTRL_COD and VASTSTELLINGEN: "
                f"{before_filter} -> {len(groundtruth_df)} parcels"
            )

        if start_date is not None and "CONTROLEDATUM" in groundtruth_df.columns:
            from dateutil.relativedelta import relativedelta

            groundtruth_df["CONTROLEDATUM"] = pd.to_datetime(
                groundtruth_df["CONTROLEDATUM"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce",
            )
            before_filter = len(groundtruth_df)

            # Filter CONTROLEDATUM >= start_date
            date_filter = (groundtruth_df["CONTROLEDATUM"] >= start_date) & (
                groundtruth_df["CONTROLEDATUM"].notna()
            )

            # And CONTROLEDATUM <= end_date + 2 months (if end_date provided)
            if end_date is not None:
                max_date = end_date + relativedelta(months=2)
                date_filter = date_filter & (
                    groundtruth_df["CONTROLEDATUM"] <= max_date
                )
                date_range_msg = f">= {start_date.date()} and <= {max_date.date()}"
            else:
                date_range_msg = f">= {start_date.date()}"

            groundtruth_df = groundtruth_df[date_filter].copy()
            logger.info(
                f"Filtered ground truth by date ({date_range_msg}): "
                f"{before_filter} -> {len(groundtruth_df)} parcels"
            )

        # Sanity check: compare ground truth with original input file
        # input_parcel_df = gfo.read_file(input_parcel_path, ignore_geometry=True)
        # if id_column in input_parcel_df.columns:
        #     input_parcel_df[id_column] = (
        #         input_parcel_df[id_column].astype(str).str.strip()
        #     )
        #     input_ids = set(input_parcel_df[id_column])
        #     logger.info(
        #         f"SANITY CHECK - Original input file has {len(input_ids)} unique IDs"
        #     )

        #     if id_column in groundtruth_df.columns:
        #         groundtruth_df[id_column] = (
        #             groundtruth_df[id_column].astype(str).str.strip()
        #         )
        #         gt_ids_temp = set(groundtruth_df[id_column])
        #         common_with_input = input_ids & gt_ids_temp
        #         logger.info(
        #             f"SANITY CHECK - Ground truth overlaps with original input: "
        #             f"{len(common_with_input)} IDs"
        #         )

        if id_column in parcels_selected_df.columns:
            parcels_selected_df[id_column] = (
                parcels_selected_df[id_column].astype(str).str.strip()
            )
        if id_column in groundtruth_df.columns:
            groundtruth_df[id_column] = (
                groundtruth_df[id_column].astype(str).str.strip()
            )

        groundtruth_df = groundtruth_df.merge(
            classes_refe_df[["CROPCODE", "MON_VRU", "CROP_DESC"]],
            left_on="NATEELT_CTRL_COD",
            right_on="CROPCODE",
            how="left",
        )

        merged_df = parcels_selected_df.merge(groundtruth_df, on=id_column, how="inner")
        logger.info(
            f"Matched {len(merged_df)} parcels between results and ground truth "
            f"(after date filtering)"
        )

        if len(merged_df) > 0:

            def determine_gt_cover(row: pd.Series) -> str:
                vaststellingen = row.get("VASTSTELLINGEN")
                onbedekt_vaststellingen = [
                    "PCSCHEUR",
                    "PCPLOZM",
                    "PC35ET",
                    "PC35ME",
                    "PC35EG",
                    "PCVEGWEG",
                ]
                if not pd.isna(vaststellingen):
                    vaststellingen_str = str(vaststellingen).upper()
                    if any(
                        vaststelling in vaststellingen_str
                        for vaststelling in onbedekt_vaststellingen
                    ):
                        return "ONBEDEKT"

                # fallback to the MON_VRU mapping
                mon_vru = row.get("MON_VRU")
                if pd.isna(mon_vru):
                    return "NODATA"
                mon_vru_str = str(mon_vru)
                if mon_vru_str == "MON_VRU_BRAAK":
                    return "ONBEDEKT"
                else:
                    return "BEDEKT"

            merged_df["gt_cover"] = merged_df.apply(determine_gt_cover, axis=1)
            merged_df["correct"] = merged_df["gt_cover"] == merged_df[pred_column]

            valid_df = merged_df[
                (merged_df["gt_cover"] != "NODATA")
                & (merged_df[pred_column] != "NODATA")
                # & (merged_df[pred_column] != "DOUBT")
            ].copy()

            logger.info(f"Valid parcels for accuracy calculation: {len(valid_df)}")

            if len(valid_df) > 0:
                total_valid = len(valid_df)
                total_correct = valid_df["correct"].sum()
                accuracy = total_correct / total_valid if total_valid > 0 else 0

                stats_list.append(
                    {
                        "category": "--- ACCURACY METRICS ---",
                        "count": "",
                        "percentage": "",
                        "total_parcels": "",
                    }
                )
                stats_list.append(
                    {
                        "category": "Overall Accuracy",
                        "count": f"{total_correct}/{total_valid}",
                        "percentage": accuracy * 100,
                        "total_parcels": "",
                    }
                )

                for gt_cat in valid_df["gt_cover"].unique():
                    gt_subset = valid_df[valid_df["gt_cover"] == gt_cat]
                    if len(gt_subset) > 0:
                        correct = gt_subset["correct"].sum()
                        total = len(gt_subset)
                        stats_list.append(
                            {
                                "category": f"Accuracy for GT: {gt_cat}",
                                "count": f"{correct}/{total}",
                                "percentage": (correct / total) * 100,
                                "total_parcels": "",
                            }
                        )

                logger.info(
                    f"Overall accuracy: {accuracy:.2%} ({total_correct}/{total_valid})"
                )
        else:
            logger.warning("No parcels matched between results and ground truth.")
            merged_df = parcels_selected_df
    else:
        logger.info("No ground truth provided, generating general statistics only.")
        merged_df = parcels_selected_df

    stats_df = pd.DataFrame(stats_list)

    report_name = f"{parcels_selected_path.stem}_accuracy_report.html"
    report_html_path = report_dir / report_name

    html_parts = []
    html_parts.append("<html><head><style>")
    html_parts.append("table { border-collapse: collapse; margin: 20px 0; }")
    html_parts.append(
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
    )
    html_parts.append("th { background-color: #4CAF50; color: white; }")
    html_parts.append("h2 { color: #333; }")
    html_parts.append(".incorrect { background-color: #ffcccc; }")
    html_parts.append(".correct { background-color: #ccffcc; }")
    html_parts.append(".nodata { background-color: #f0f0f0; }")
    html_parts.append("</style></head><body>")
    html_parts.append("<h1>Cover Marker Report</h1>")

    html_parts.append("<h2>Statistics</h2>")
    html_parts.append(stats_df.to_html(index=False, border=0))

    html_parts.append("<h2>Ground truth</h2>")
    columns_to_save = [
        id_column,
        "pred1",
        "pred1_prob",
        "pred_consolidated",
        "pred_cons_status",
    ]

    if "gt_cover" in merged_df.columns:
        columns_to_save.extend(
            [
                "CONTROLEDATUM",
                "VASTSTELLINGEN",
                "NATEELT_CTRL_COD",
                "gt_cover",
                "correct",
            ]
        )

    available_columns = [col for col in columns_to_save if col in merged_df.columns]
    details_df = merged_df[available_columns]

    # Generate HTML with conditional row coloring based on correctness
    if "correct" in details_df.columns:
        html_table = '<table border="0">\n<thead>\n<tr style="text-align: right;">\n'
        for col in details_df.columns:
            html_table += f"<th>{col}</th>\n"
        html_table += "</tr>\n</thead>\n<tbody>\n"

        for _, row in details_df.iterrows():
            gt_cover = row.get("gt_cover")
            if gt_cover == "NODATA" or pd.isna(gt_cover):
                row_class = "nodata"
            elif pd.isna(row.get("correct")):
                row_class = "nodata"
            elif row.get("correct"):
                row_class = "correct"
            else:
                row_class = "incorrect"

            html_table += f'<tr class="{row_class}">\n'
            for col in details_df.columns:
                html_table += f"<td>{row[col]}</td>\n"
            html_table += "</tr>\n"

        html_table += "</tbody>\n</table>"
        html_parts.append(html_table)
    else:
        html_parts.append(details_df.to_html(index=False, border=0))

    html_parts.append("</body></html>")

    with report_html_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info(f"Accuracy report saved to: {report_html_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    report(
        input_parcel_path=Path(r".."),
        parcels_selected_path=Path(r".."),
        ground_truth_path=Path(r".."),
        classes_refe_path=Path(r".."),
        id_column="UID",
    )
