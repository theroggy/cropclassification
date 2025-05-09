"""Run a cover classification."""

import logging
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    config_overrules: list[str] = [],
):
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
    global logger
    logger = log_helper.main_log_init(log_dir, __name__, log_level)

    logger.warning("This is a POC for a cover marker, so not for operational use!")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    markertype = conf.marker["markertype"]
    if not markertype.startswith("COVER"):
        raise ValueError(f"Invalid markertype {markertype}, expected COVER_XXX")

    # Create run dir to be used for the results
    reuse_last_run_dir = conf.calc_marker_params.getboolean("reuse_last_run_dir")
    run_dir = dir_helper.create_run_dir(
        conf.paths.getpath("marker_dir"), reuse_last_run_dir
    )

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
    if markertype in ("COVER_TBG_BMG_VOORJAAR", "COVER_TBG_BMG_NAJAAR"):
        # Grassland parcels with a premium having a requirement that they cannot be
        # resown -> should never be ploughed.
        input_parcel_filename = f"{input_parcel_path.stem}_TBG_BMG.gpkg"
        input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

        where = "ALL_BEST like '%TBG%' OR ALL_BEST like '%BMG%'"
        gfo.copy_layer(input_parcel_path, input_parcel_filtered_path, where=where)
        input_parcel_path = input_parcel_filtered_path

    elif markertype == "COVER_EEB_VOORJAAR":
        # Fallow parcels with a premium having as requirement that there is no activity
        # on them from 15/01 till 10/04 + that they have maize as main crop.
        input_parcel_filename = f"{input_parcel_path.stem}_EEB.gpkg"
        input_parcel_filtered_path = input_preprocessed_dir / input_parcel_filename

        where = (
            "ALL_BEST like '%EEB%' AND GWSCOD_V = '83' AND GWSCOD_H IN ('201', '202')"
        )
        gfo.copy_layer(input_parcel_path, input_parcel_filtered_path, where=where)
        input_parcel_path = input_parcel_filtered_path

    elif markertype == "COVER":
        pass
    else:
        raise ValueError(f"Invalid {markertype=}")

    # Get some general config
    data_ext = conf.general["data_ext"]
    geofile_ext = conf.general["geofile_ext"]
    id_column = conf.columns["id"]
    parcel_columns = [
        id_column,
        "ALL_BEST",
        "GWSCOD_H",
        "GWSNAM_H",
        "GWSCOD_N",
        "GWSNAM_N",
        "GWSCOD_N2",
        "GWSNAM_N2",
        "PRC_NIS",
        "ALV_NUMMER",
        "PRC_NMR",
    ]

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
    force = False
    on_error = "warn"
    parcels_selected_paths = []

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
                parcel_columns=parcel_columns,
                output_path=output_path,
                output_geo_path=geo_path,
                force=force,
            )

            if markertype == "COVER_EEF_VOORJAAR":
                parcels_selected_path = run_dir / f"{geo_path.stem}_EEF{geofile_ext}"
                _select_parcels_EEF(geo_path, parcels_selected_path)
                parcels_selected_paths.append(parcels_selected_path)
            elif markertype == "COVER_BMG_MEG_MEV_NAJAAR":
                parcels_selected_path = run_dir / f"{geo_path.stem}_BMG{geofile_ext}"
                _select_parcels_BMG_MEG_MEV_EEF(geo_path, parcels_selected_path)
                parcels_selected_paths.append(parcels_selected_path)
            else:
                parcels_selected_path = run_dir / f"{geo_path.stem}_filter{geofile_ext}"
                _select_parcels(geo_path, parcels_selected_path)
                parcels_selected_paths.append(parcels_selected_path)

        except Exception as ex:
            message = f"Error calculating {output_path.stem}"
            if on_error == "warn":
                logger.exception(f"Error calculating {output_path.stem}")
                warnings.warn(message, category=RuntimeWarning, stacklevel=1)
            elif on_error == "raise":
                raise RuntimeError(message) from ex
            else:
                logger.error(f"invalid value for on_error: {on_error}, 'raise' assumed")
                raise RuntimeError(message) from ex

    # STEP 4: Create the final list of parcels
    # ----------------------------------------
    # Consolidate the list of parcels that need to be controlled
    parcels_selected_path = (
        run_dir / f"selectie_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    )
    parcels_selected = None
    for path in parcels_selected_paths:
        columns = [*parcel_columns, "provincie"]
        parcels = gfo.read_file(path, columns=columns, ignore_geometry=True)

        if parcels_selected is None:
            parcels_selected = parcels
        else:
            parcels_selected = pd.concat([parcels_selected, parcels])

    # Create list with unique parcels
    assert parcels_selected is not None
    parcels_selected = parcels_selected.drop_duplicates()
    pdh.to_excel(parcels_selected, parcels_selected_path, index=False)

    logging.shutdown()


def _calc_cover(
    input_parcel_path,
    timeseries_periodic_dir,
    images_to_use,
    start_date: datetime,
    end_date: datetime,
    parcel_columns: list[str],
    output_path: Path,
    output_geo_path: Optional[Path] = None,
    force: bool = False,
):
    logger.info(f"start processing {output_path}")

    if output_path.exists():
        if force:
            gfo.remove(output_path)
        elif output_geo_path is not None and not output_geo_path.exists():
            # Geo file is asked bu missing: we need to recalculate the output file as
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
        else:
            continue

        key = f"s1{orbit}_{'_'.join(column_lower.split('-')[-1].split('_')[-2:])}"
        columns[key] = column

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
            FROM "info" info
        ) covers
    """  # noqa: E501
    result_df = pdh.read_file(tmp_path, sql=sql)
    pdh.to_file(result_df, output_path)

    if output_geo_path is not None:
        # If a geo file is asked, always recreated it so it is in sync with the output
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


def _select_parcels_BMG_MEG_MEV_EEF(input_geo_path, output_geo_path):
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


def _select_parcels_EEF(input_geo_path, output_geo_path):
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

    # Filter used in the selection of 03/2025
    where = f"""
            ( ALL_BEST like '%EEF%'
              AND ( ( "{columns["ndvi_median"]}" <> 0
                      AND "{columns["ndvi_median"]}" < 0.35
                    )
                    OR ( cover_s1 = 'bare-soil'
                         AND "{columns["ndvi_median"]}" = 0  -- no NDVI available
                    )
                  )
            )
    """
    gfo.copy_layer(input_geo_path, output_geo_path, where=where)


def _select_parcels(input_geo_path, output_geo_path):
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

    # Filter used in the selection of 05/2025
    where = f"""
            ( 1=1
              AND ( ( "{columns["ndvi_median"]}" <> 0
                      AND "{columns["ndvi_median"]}" < 0.35
                    )
                    OR ( cover_s1 = 'bare-soil'
                         AND "{columns["ndvi_median"]}" = 0  -- no NDVI available
                    )
                  )
            )
    """
    gfo.copy_layer(input_geo_path, output_geo_path, where=where)


def report():
    """Create a report for the cover marker."""
    # Read parcels selected to be controlled for the cover marker
    prc_selectie_path = Path(
        r"X:\__IT_TEAM_ANG_GIS\Taken\2024\2024-10-16_baresoil_selectie\baresoil_selectie_2024-10-16.xlsx"
    )
    force = False
    force_update = True

    report_dir = prc_selectie_path.parent / "evaluatie"

    # Create output file with the result of the cover marker + the groundtruth
    result_geo_path = report_dir / f"{prc_selectie_path.stem}_gt.gpkg"
    if force or not result_geo_path.exists():
        prc_selectie_df = pd.read_excel(prc_selectie_path)
        prc_selectie_df["selectie_reden"] = "?"
        prc_selectie_df.loc[
            prc_selectie_df["ALL_BEST"].str.contains("MEV", regex=False),
            "selectie_reden",
        ] = "MEV"
        prc_selectie_df.loc[
            prc_selectie_df["ALL_BEST"].str.contains("MEG", regex=False),
            "selectie_reden",
        ] = "MEG"
        prc_selectie_df.loc[
            prc_selectie_df["ALL_BEST"].str.contains("EEF", regex=False),
            "selectie_reden",
        ] = "EEF"
        prc_selectie_df.loc[
            prc_selectie_df["ALL_BEST"].str.contains("BMG", regex=False),
            "selectie_reden",
        ] = "BMG"

        logger.info(f"{len(prc_selectie_df)=}")

        # Read groundtruth and join with parcels
        input_groundtruth_filename = "Prc_BEFL_2024_2025_02_21_groundtruth_baresoil.tsv"
        # input_dir = conf.paths.getpath("input_dir")
        input_dir = Path(r"X:\Monitoring\Markers\dev\_inputdata")
        input_groundtruth_path = input_dir / input_groundtruth_filename
        groundtruth_df = pdh.read_file(input_groundtruth_path)
        prc_gt_df = prc_selectie_df.merge(
            groundtruth_df, on=["ALV_NUMMER", "PRC_NMR"], how="left"
        )
        logger.info(f"{len(prc_gt_df)=}")

        # Read geometries of parcels and join with controled parcels
        prc_path = input_dir / "Prc_BEFL_2024_2024-10-13.gpkg"
        prc_gdf = gfo.read_file(prc_path)
        prc_gdf["ALV_NUMMER"] = prc_gdf["ALV_NUMMER"].astype("int64")
        prc_gt_gdf = prc_gdf[["geometry", "ALV_NUMMER", "PRC_NMR"]].merge(
            prc_gt_df, on=["ALV_NUMMER", "PRC_NMR"], how="inner"
        )
        logger.info(f"{len(prc_gt_gdf)=}")

        # Read the result of the cover marker and join with controled parcels
        prc_cover_path = Path(
            r"X:\Monitoring\Markers\dev\_cover_periodic\Prc_BEFL_2024_2024-10-13\cover_2024-09-30_2024-10-07.gpkg"
        )
        prc_cover_df = gfo.read_file(
            prc_cover_path,
            table_name=prc_cover_path.stem.replace("-", "_"),
            ignore_geometry=True,
        )
        prc_cover_df.rename(columns={"uid": "UID"}, inplace=True)
        prc_gt_gdf = prc_gt_gdf.merge(prc_cover_df, on="UID", how="inner")

        # Write the result to a gpkg
        if "fid" in prc_gt_gdf.columns:
            prc_gt_gdf.drop(columns=["fid"], inplace=True)
        gfo.to_file(prc_gt_gdf, result_geo_path)

    s2_ndvi = "s2-ndvi-weekly_20240930_ndvi_median"
    expression = f"""
        CASE
            WHEN "{s2_ndvi}" = 0 THEN 'NODATA'
            WHEN ( (selectie_reden = 'BMG' AND "{s2_ndvi}" < 0.25) OR
                   (selectie_reden <> 'BMG' AND "{s2_ndvi}" < 0.3)
                 ) THEN 'bare-soil'
            ELSE 'other'
        END
    """
    gfo.add_column(
        result_geo_path,
        "cover_ndvi",
        "TEXT",
        expression=expression,
        force_update=force_update,
    )

    expression_template = """
        CASE
            WHEN ctl_vastst IS NULL AND hoofdteelt_ctl IS NULL AND nateelt_ctl IS NULL THEN 'no_groundtruth'
            WHEN {cover_col} = 'NODATA' THEN 'NODATA'
            WHEN selectie_reden = 'BMG' AND {cover_col} = 'bare-soil' AND ctl_vastst IS NOT NULL THEN 'yes'
            WHEN selectie_reden = 'BMG' AND {cover_col} = 'bare-soil'
                AND hoofdteelt_ctl IS NOT NULL AND hoofdteelt_ctl NOT IN ('60', '63', '638', '660', '700', '745', '9827', '9828') THEN 'yes'
            WHEN selectie_reden = 'BMG' AND {cover_col} <> 'bare-soil' AND ctl_vastst IS NULL THEN 'yes'
            WHEN selectie_reden = 'EEF' AND {cover_col} = 'bare-soil'
                AND ((hoofdteelt_ctl IS NOT NULL AND hoofdteelt_ctl <> '98')
                        OR (nateelt_ctl IS NOT NULL AND nateelt_ctl <> '98')
                    ) THEN 'yes'
            WHEN selectie_reden = 'EEF' AND {cover_col} = 'bare-soil' AND ctl_vastst IS NOT NULL THEN 'yes'
            WHEN selectie_reden = 'EEF' AND {cover_col} <> 'bare-soil' AND (hoofdteelt_ctl IS NULL OR hoofdteelt_ctl = '98') THEN 'yes'
            WHEN selectie_reden = 'MEV' AND {cover_col} = 'bare-soil'
                AND ((hoofdteelt_ctl IS NOT NULL AND hoofdteelt_ctl NOT IN ('660', '700', '723', '732'))
                        OR ctl_vastst IS NOT NULL
                    ) THEN 'yes'
            WHEN selectie_reden = 'MEV' AND {cover_col} <> 'bare-soil'
                AND (hoofdteelt_ctl IS NULL OR hoofdteelt_ctl IN ('660', '700', '723', '732')) THEN 'yes'
            WHEN selectie_reden = 'MEG' AND {cover_col} = 'bare-soil'
                AND ((hoofdteelt_ctl IS NOT NULL AND hoofdteelt_ctl NOT IN ('63'))
                        OR ctl_vastst IS NOT NULL
                    ) THEN 'yes'
            WHEN selectie_reden = 'MEG' AND {cover_col} <> 'bare-soil'
                AND (hoofdteelt_ctl IS NULL OR hoofdteelt_ctl IN ('63')) THEN 'yes'
            ELSE 'no'
        END
    """  # noqa: E501
    gfo.add_column(
        result_geo_path,
        "ndvi_correct",
        "TEXT",
        expression=expression_template.format(cover_col="cover_ndvi"),
        force_update=force_update,
    )

    gfo.add_column(
        result_geo_path,
        "s1_correct",
        "TEXT",
        expression=expression_template.format(cover_col="cover_s1"),
        force_update=force_update,
    )

    # Create statistics output
    sql = """
        SELECT selectie_reden, count(*) aantal
              ,SUM(CASE WHEN ndvi_correct = 'no_groundtruth' THEN 1 ELSE 0 END) AS nb_no_groundtruth
              ,SUM(CASE WHEN ndvi_correct = 'yes' THEN 1 ELSE 0 END) AS nb_ndvi_correct
              ,SUM(CASE WHEN ndvi_correct = 'no' THEN 1 ELSE 0 END) AS nb_ndvi_wrong
              ,SUM(CASE WHEN ndvi_correct = 'NODATA' THEN 1 ELSE 0 END) AS nb_ndvi_nodata
              ,SUM(CASE WHEN s1_correct = 'yes' THEN 1 ELSE 0 END) AS nb_s1_correct
              ,SUM(CASE WHEN s1_correct = 'no' THEN 1 ELSE 0 END) AS nb_s1_wrong
              ,SUM(CASE WHEN s1_correct = 'NODATA' THEN 1 ELSE 0 END) AS nb_s1_nodata
         FROM "{input_layer}"
         GROUP BY selectie_reden
    """  # noqa: E501
    stats_df = gfo.read_file(result_geo_path, sql_stmt=sql)
    stats_path = report_dir / f"{prc_selectie_path.stem}_stats.xlsx"
    pdh.to_excel(stats_df, stats_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    report()
