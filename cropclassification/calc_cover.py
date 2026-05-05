"""Run a cover classification."""

import logging
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

# Import before tensorflow to avoid DLL load errors.
import geofileops as gfo
import numpy as np
import pandas as pd
import pyproj
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import dir_helper, log_helper
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.preprocess import _timeseries_helper as ts_helper
from cropclassification.preprocess import timeseries as ts
from cropclassification.util import mosaic_util

_S2_NDVI_COLUMN_REGEX = re.compile(r"^s2-ndvi-.*_ndvi_median$", re.IGNORECASE)
_S1_COLUMN_REGEX = re.compile(r"^s1-.*", re.IGNORECASE)
_PERIOD_REGEX = re.compile(r"(?:daily|weekly|monthly)_\d{8}_", re.IGNORECASE)
_ONBEDEKT_VASTSTELLINGEN_REGEX = re.compile(
    "|".join(
        re.escape(code)
        for code in (
            "PCSCHEUR",
            "PCPLOZM",
            "PC35ET",
            "PC35ME",
            "PC35EG",
            "PCVEGWEG",
        )
    )
)

_DEFAULT_ML_DOUBT = 0.65
_DEFAULT_THRESHOLD_NDVI = 0.35  # =1-x
_THRESHOLD_ONBEDEKT = 0.5
_THRESHOLD_DOUBT = 0.4
_PRED_SOURCE_THRESHOLD = "THRESHOLD"
_PRED_SOURCE_ML = "ML"


def _get_cover_output_name(period: dict[str, datetime], data_ext: str) -> str:
    period_start_str = period["start_date"].strftime("%Y-%m-%d")
    period_end_str = period["end_date"].strftime("%Y-%m-%d")
    return f"cover_{period_start_str}_{period_end_str}{data_ext}"


def _get_required_ts_periods(
    cover_periods: list[dict[str, datetime]],
    training_periods: list[dict[str, datetime]],
    include_training_periods: bool,
) -> list[dict[str, datetime]]:
    required_periods = list(cover_periods)
    if include_training_periods:
        seen_period_keys = {
            (period["start_date"], period["end_date"]) for period in cover_periods
        }
        for period in training_periods:
            period_key = (period["start_date"], period["end_date"])
            if period_key not in seen_period_keys:
                required_periods.append(period)
                seen_period_keys.add(period_key)
    return required_periods


def _get_markertype_filter(markertype: str) -> str:
    # TODO: zijn de oude codes nog nodig?
    if markertype in ("COVER", "COVER_EEF_VOORJAAR", "ONBEDEKT_NA_WINTER"):
        return """
                "ALL_BEST" like '%EEF%'
            """

    if markertype in ("ONBEDEKT_NA_ZOMER", "COVER_BMG_MEG_MEV_EEF_NAJAAR"):
        return """
                   "ALL_BEST" like '%BMG%'
                OR "ALL_BEST" like '%MEV%'
                OR "ALL_BEST" like '%MEG%'
                OR "ALL_BEST" like '%EEF%'
                OR "ALL_BEST" like '%TBG%'
                OR "ALL_BEST" like '%EBG%'
            """

    if markertype in ("COVER_EEB_VOORJAAR", "ONBEDEKT_LENTE"):
        return """
                (
                    ("ALL_BEST" like '%EEB%'
                 AND "GWSCOD_V" = '83' AND "GWSCOD_H" IN ('201', '202'))
                 OR ("ALL_BEST" like '%TBG%')
                 OR ("ALL_BEST" like '%BMG%')
                )
            """

    if markertype in ("COVER_TBG_BMG_VOORJAAR", "COVER_TBG_BMG_NAJAAR"):
        return "ALL_BEST like '%TBG%' OR ALL_BEST like '%BMG%'"

    raise ValueError(f"Invalid {markertype=}")


def _filter_input_parcels_for_markertype(
    input_parcel_path: Path,
    input_preprocessed_dir: Path,
    markertype: str,
    force: bool,
) -> Path:
    where = _get_markertype_filter(markertype)
    output_path = input_preprocessed_dir / f"{input_parcel_path.stem}_{markertype}.gpkg"
    if force or not output_path.exists():
        gfo.copy_layer(input_parcel_path, output_path, where=where, force=force)
    else:
        logger.info(f"Filtered parcel file already exists, skipping: {output_path}")
    return output_path


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
    conf.read_config(
        config_paths, default_basedir=default_basedir, overrules=config_overrules
    )

    log_level = conf.general.get("log_level")
    log_dir = conf.paths.getpath("log_dir")
    global logger  # noqa: PLW0603
    logger = log_helper.main_log_init(log_dir, __name__, log_level)

    logger.info(f"Config used: \n{conf.pformat_config()}")

    # force = True
    force = False
    markertype = conf.marker["markertype"]

    if not markertype.startswith(("COVER", "ONBEDEKT")):
        raise ValueError(f"Invalid markertype {markertype}")

    reuse_last_run_dir = conf.calc_marker_params.getboolean("reuse_last_run_dir")
    run_dir = dir_helper.create_run_dir(
        conf.paths.getpath("marker_dir"), reuse_last_run_dir
    )
    input_dir = conf.paths.getpath("input_dir")
    _input_groundtruth_filename = conf.calc_marker_params.getpath(
        "input_groundtruth_filename"
    )
    input_groundtruth_path = (
        input_dir / _input_groundtruth_filename
        if _input_groundtruth_filename is not None
        else None
    )
    if (input_groundtruth_path is not None) and not input_groundtruth_path.exists():
        message = (
            f"Input groundtruth file doesn't exist, so STOP: {input_groundtruth_path}"
        )
        logger.critical(message)
        raise ValueError(message)

    _store_config_files(
        config_paths, default_basedir, config_overrules, reuse_last_run_dir, run_dir
    )

    parcels_marker_path = (
        run_dir / f"{markertype}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    )
    parcel_selected_sqlite_path = parcels_marker_path.with_suffix(".sqlite")
    ndvi_threshold = conf.calc_marker_params.getfloat(
        "ndvi_threshold", fallback=_DEFAULT_THRESHOLD_NDVI
    )
    ml_onbedekt_threshold = conf.calc_marker_params.getfloat(
        "ml_onbedekt_threshold", fallback=0.65
    )

    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    all_input_parcel_path = input_dir / input_parcel_filename
    if not all_input_parcel_path.exists():
        message = f"Input file doesn't exist, so STOP: {all_input_parcel_path}"
        logger.critical(message)
        raise ValueError(message)

    data_ext = conf.general["data_ext"]
    geofile_ext = conf.general["geofile_ext"]
    buffer = conf.timeseries.getfloat("buffer")
    input_preprocessed_dir = conf.paths.getpath("input_preprocessed_dir")
    input_preprocessed_dir.mkdir(parents=True, exist_ok=True)
    images_to_use = conf.parse_image_config(conf.images["images"])

    all_bufm_filename = f"{all_input_parcel_path.stem}_bufm{buffer:g}{geofile_ext}"
    all_bufm_path = input_preprocessed_dir / all_bufm_filename
    all_nogeo_path = input_preprocessed_dir / f"{all_input_parcel_path.stem}{data_ext}"
    timeseries_periodic_dir = (
        conf.paths.getpath("timeseries_periodic_dir") / all_bufm_path.stem
    )

    cover_dir = run_dir / all_input_parcel_path.stem
    cover_dir.mkdir(parents=True, exist_ok=True)

    ts_periods_cache_dir = run_dir / "_ts_periods"
    ts_periods_cache_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.fromisoformat(conf.period["start_date"])
    end_date = datetime.fromisoformat(conf.period["end_date"])
    cover_periods = list(
        mosaic_util._prepare_periods(
            start_date=start_date,
            end_date=end_date,
            period_name="weekly",
            period_days=None,
        )
    )

    # Optional extra training window for the S1 ML model.
    # Useful when S2 cloud coverage is poor in the detection period.
    conf_training_start = conf.period.get("training_start_date")
    conf_training_end = conf.period.get("training_end_date")
    training_start_date = (
        datetime.fromisoformat(conf_training_start)
        if conf_training_start
        else start_date
    )
    training_end_date = (
        datetime.fromisoformat(conf_training_end) if conf_training_end else end_date
    )
    training_periods = list(
        mosaic_util._prepare_periods(
            start_date=training_start_date,
            end_date=training_end_date,
            period_name="weekly",
            period_days=None,
        )
    )

    filtered_input_parcel_path = (
        input_preprocessed_dir / f"{all_input_parcel_path.stem}_{markertype}.gpkg"
    )

    cover_output_paths = {
        (period["start_date"], period["end_date"]): cover_dir
        / _get_cover_output_name(period, data_ext)
        for period in cover_periods
    }

    needs_model_training = force or any(
        not output_path.exists() for output_path in cover_output_paths.values()
    )
    needs_cover_outputs = force or any(
        not (output_path.exists() and output_path.with_suffix(geofile_ext).exists())
        for output_path in cover_output_paths.values()
    )
    needs_consolidation = (
        force
        or needs_cover_outputs
        or not (parcels_marker_path.exists() and parcel_selected_sqlite_path.exists())
    )
    needs_filtered_parcels = force or (
        (needs_cover_outputs or needs_consolidation)
        and not filtered_input_parcel_path.exists()
    )

    ts_period_paths: dict[tuple[datetime, datetime], Path] = {}
    classifier: RandomForestClassifier | None = None
    feature_names: list[str] = []

    if needs_cover_outputs:
        # -------------------------------------------------------------------
        # STEP 1: Buffer all parcels and extract their timeseries.
        # -------------------------------------------------------------------
        ts_helper.prepare_input(
            input_parcel_path=all_input_parcel_path,
            output_imagedata_parcel_input_path=all_bufm_path,
            output_parcel_nogeo_path=all_nogeo_path,
            force=force,
        )

        ts.calc_timeseries_data(
            input_parcel_path=all_bufm_path,
            roi_bounds=tuple(conf.roi.getlistfloat("roi_bounds")),
            roi_crs=pyproj.CRS.from_user_input(conf.roi.get("roi_crs")),
            start_date=min(start_date, training_start_date),
            end_date=max(end_date, training_end_date),
            images_to_use=images_to_use,
            timeseries_periodic_dir=timeseries_periodic_dir,
            force=force,
        )

        # -------------------------------------------------------------------
        # STEP 2: Collect weekly timeseries for the periods we still need.
        # -------------------------------------------------------------------
        required_periods = _get_required_ts_periods(
            cover_periods=cover_periods,
            training_periods=training_periods,
            include_training_periods=needs_model_training,
        )

        parceldata_aggregations_to_use = conf.marker.getlist(
            "parceldata_aggregations_to_use"
        )
        for period in required_periods:
            period_start = period["start_date"]
            period_end = period["end_date"]
            pstart_str = period_start.strftime("%Y-%m-%d")
            pend_str = period_end.strftime("%Y-%m-%d")
            ts_path = (
                ts_periods_cache_dir
                / f"{all_nogeo_path.stem}_{pstart_str}_{pend_str}.sqlite"
            )
            try:
                ts.collect_and_prepare_timeseries_data(
                    input_parcel_path=all_nogeo_path,
                    timeseries_dir=timeseries_periodic_dir,
                    output_path=ts_path,
                    start_date=period_start,
                    end_date=period_end,
                    images_to_use=images_to_use,
                    parceldata_aggregations_to_use=parceldata_aggregations_to_use,
                    max_fraction_null=1,
                    force=force,
                )
                ts_period_paths[(period_start, period_end)] = ts_path

            except Exception:
                logger.warning(
                    f"Skipping period {period_start.date()} for ts collection",
                    exc_info=True,
                )

    else:
        logger.info("All cover period outputs already exist")

    if needs_model_training:
        # -------------------------------------------------------------------
        # STEP 3: Train the ML model.
        # -------------------------------------------------------------------

        logger.info("Training ML model")
        training_period_dfs: list[tuple[datetime, pd.DataFrame]] = []
        for period in training_periods:
            period_key = (period["start_date"], period["end_date"])
            if period_key not in ts_period_paths:
                continue

            training_period_dfs.append(
                (
                    period["start_date"],
                    pdh.read_file(ts_period_paths[period_key]),
                )
            )

        classifier, feature_names = _train_ml_cover_model(
            period_timeseries=training_period_dfs,
            id_column=conf.columns["id"],
            ndvi_threshold=ndvi_threshold,
            ground_truth_path=input_groundtruth_path,
            output_test_eval_path=run_dir / "s1_ml_test_eval.txt",
            output_gt_eval_path=run_dir / "s1_ml_groundtruth_eval.txt",
        )
        logger.info(f"S1 ML model trained on {len(feature_names)} features.")
    else:
        logger.info("All cover output files exist, skipping ML training")

    # -------------------------------------------------------------------
    # STEP 4: Filter parcels for the specific markertype.
    # -------------------------------------------------------------------
    if needs_filtered_parcels:
        input_parcel_path = _filter_input_parcels_for_markertype(
            input_parcel_path=all_input_parcel_path,
            input_preprocessed_dir=input_preprocessed_dir,
            markertype=markertype,
            force=force,
        )
    else:
        input_parcel_path = filtered_input_parcel_path
        logger.info("Filtered parcel output already exists, skipping")

    # ---------------------------------------------------------------------
    # STEP 5: Calculate cover and prediction per week for filtered parcels.
    # ---------------------------------------------------------------------
    on_error = "warn"
    parcels_cover_paths: list[Path] = []

    if not needs_cover_outputs:
        logger.info("All cover outputs already exist, skipping")

    for period in cover_periods:
        period_key = (period["start_date"], period["end_date"])
        output_path = cover_output_paths[period_key]
        geo_path = output_path.with_suffix(geofile_ext)

        if not force and output_path.exists() and geo_path.exists():
            parcels_cover_paths.append(geo_path)
            continue

        if period_key not in ts_period_paths:
            logger.warning(
                f"No timeseries for period {period['start_date'].date()}, skipping"
            )
            continue

        try:
            _calc_cover_and_predict(
                input_parcel_path=input_parcel_path,
                ts_path=ts_period_paths[period_key],
                parcel_columns=None,
                ndvi_threshold=ndvi_threshold,
                output_path=output_path,
                classifier=classifier,
                feature_names=feature_names,
                output_geo_path=geo_path,
                force=force,
            )
            if geo_path.exists():
                parcels_cover_paths.append(geo_path)

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

    # ---------------------------------------------------------------
    # STEP 6: Consolidate period results into one row per parcel
    # ---------------------------------------------------------------
    if needs_consolidation:
        parcels_selected = _consolidate_cover_predictions(
            input_parcel_path=input_parcel_path,
            parcels_cover_paths=parcels_cover_paths,
            id_column=conf.columns["id"],
            ml_threshold_onbedekt=ml_onbedekt_threshold,
        )

        pred_source_counts = parcels_selected["pred_source"].value_counts().to_dict()
        logger.info(f"Consolidation source distribution: {pred_source_counts}")

        pdh.to_excel(parcels_selected, parcels_marker_path, index=False)
        pdh.to_file(parcels_selected, parcel_selected_sqlite_path, index=False)
    else:
        logger.info("Final cover outputs already exist, skipping step 6")

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------
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


def _store_config_files(
    config_paths: list[Path],
    default_basedir: Path,
    config_overrules: list[str] | None,
    reuse_last_run_dir: bool,
    run_dir: Path,
) -> None:
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
        # Copy the config files to a config dir for later reference
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


def _categorize_threshold_detection(
    probabilities: pd.Series,
    onbedekt_probability_threshold: float = _THRESHOLD_ONBEDEKT,
    doubt_probability_threshold: float = _THRESHOLD_DOUBT,
) -> pd.Series:
    """Categorize THRESHOLD detection scores into BEDEKT/DOUBT/ONBEDEKT."""
    numeric_probabilities = pd.to_numeric(probabilities, errors="coerce")
    categories = pd.Series("NODATA", index=probabilities.index, dtype="object")

    valid_mask = numeric_probabilities.notna()
    categories.loc[valid_mask] = "BEDEKT"

    doubt_mask = numeric_probabilities.gt(doubt_probability_threshold)
    onbedekt_mask = numeric_probabilities.gt(onbedekt_probability_threshold)

    categories.loc[valid_mask & doubt_mask] = "DOUBT"
    categories.loc[valid_mask & onbedekt_mask] = "ONBEDEKT"

    return categories


def _categorize_ml_detection(
    probabilities: pd.Series,
    ml_onbedekt_threshold: float,
) -> pd.Series:
    """Categorize ML probabilities into BEDEKT/ONBEDEKT."""
    numeric_probabilities = pd.to_numeric(probabilities, errors="coerce")
    categories = pd.Series("NODATA", index=probabilities.index, dtype="object")

    valid_mask = numeric_probabilities.notna()
    categories.loc[valid_mask] = "BEDEKT"
    categories.loc[valid_mask & numeric_probabilities.gt(ml_onbedekt_threshold)] = (
        "ONBEDEKT"
    )
    return categories


def _categorize_consolidated_predictions(
    probabilities: pd.Series,
    sources: pd.Series,
    threshold_onbedekt_threshold: float = _THRESHOLD_ONBEDEKT,
    threshold_doubt_threshold: float = _THRESHOLD_DOUBT,
    ml_onbedekt_doubt: float = _DEFAULT_ML_DOUBT,
) -> pd.Series:
    """Categorize final parcel probabilities using explicit THRESHOLD/ML logic."""
    categories = pd.Series("NODATA", index=sources.index, dtype="object")

    threshold_mask = sources.eq(_PRED_SOURCE_THRESHOLD)
    if threshold_mask.any():
        categories.loc[threshold_mask] = _categorize_threshold_detection(
            probabilities=probabilities.loc[threshold_mask],
            onbedekt_probability_threshold=threshold_onbedekt_threshold,
            doubt_probability_threshold=threshold_doubt_threshold,
        )

    ml_mask = sources.eq(_PRED_SOURCE_ML)
    if ml_mask.any():
        categories.loc[ml_mask] = _categorize_ml_detection(
            probabilities=probabilities.loc[ml_mask],
            ml_onbedekt_threshold=ml_onbedekt_doubt,
        )

    return categories


def _consolidate_cover_predictions(
    input_parcel_path: Path,
    parcels_cover_paths: list[Path],
    id_column: str,
    ml_threshold_onbedekt: float,
    threshold_onbedekt: float = _THRESHOLD_ONBEDEKT,
    threshold_doubt: float = _THRESHOLD_DOUBT,
) -> pd.DataFrame:
    parcels_stacked = pd.concat(
        [gfo.read_file(path, ignore_geometry=True) for path in parcels_cover_paths],
        ignore_index=True,
    )

    input_info = gfo.get_layerinfo(input_parcel_path)
    attr_cols = list(input_info.columns)
    if "provincie" in parcels_stacked.columns and "provincie" not in attr_cols:
        attr_cols.append("provincie")

    best_s2_df = (
        parcels_stacked[[*attr_cols, "pred1", "pred1_prob"]]
        .sort_values("pred1_prob", ascending=False)
        .groupby(id_column, dropna=False, as_index=False)
        .first()
        .rename(columns={"pred1_prob": "thresh_prob", "pred1": "thresh_pred"})
    )

    if "cover_s2_ndvi" in parcels_stacked.columns:
        s2_available_df = (
            parcels_stacked.assign(
                s2_available=parcels_stacked["cover_s2_ndvi"].ne("NODATA")
            )
            .groupby(id_column, dropna=False, as_index=False)["s2_available"]
            .any()
        )
    else:
        s2_available_df = best_s2_df[[id_column]].copy()
        s2_available_df["s2_available"] = True

    if "ml_prob" in parcels_stacked.columns:
        s1_df = (
            parcels_stacked.groupby(id_column, dropna=False, as_index=False)["ml_prob"]
            .max()
            .rename(columns={"ml_prob": "ml_prob_max"})
        )
    else:
        s1_df = best_s2_df[[id_column]].copy()
        s1_df["ml_prob_max"] = float("nan")

    parcels_selected = best_s2_df.merge(s2_available_df, on=id_column, how="left")
    parcels_selected = parcels_selected.merge(s1_df, on=id_column, how="left")

    has_ml = parcels_selected["ml_prob_max"].notna()
    parcels_selected["pred_source"] = _PRED_SOURCE_THRESHOLD
    parcels_selected.loc[~parcels_selected["s2_available"] & has_ml, "pred_source"] = (
        _PRED_SOURCE_ML
    )

    parcels_selected["pred1_prob"] = np.nan
    threshold_mask = parcels_selected["pred_source"] == _PRED_SOURCE_THRESHOLD
    ml_mask = parcels_selected["pred_source"] == _PRED_SOURCE_ML
    parcels_selected.loc[threshold_mask, "pred1_prob"] = parcels_selected.loc[
        threshold_mask, "thresh_prob"
    ]
    parcels_selected.loc[ml_mask, "pred1_prob"] = parcels_selected.loc[
        ml_mask, "ml_prob_max"
    ]

    parcels_selected["pred_consolidated"] = _categorize_consolidated_predictions(
        probabilities=parcels_selected["pred1_prob"],
        sources=parcels_selected["pred_source"],
        threshold_onbedekt_threshold=threshold_onbedekt,
        threshold_doubt_threshold=threshold_doubt,
        ml_onbedekt_doubt=ml_threshold_onbedekt,
    )
    parcels_selected["pred_cons_status"] = np.where(
        parcels_selected["pred_consolidated"].isin(("NODATA", "DOUBT")),
        "NOK",
        "OK",
    )
    return parcels_selected


def _determine_groundtruth_cover_series(merged_df: pd.DataFrame) -> pd.Series:
    gt_cover = pd.Series("NODATA", index=merged_df.index, dtype="object")

    if "VASTSTELLINGEN" in merged_df.columns:
        onbedekt_from_vaststellingen = (
            merged_df["VASTSTELLINGEN"]
            .astype("string")
            .str.upper()
            .str.contains(_ONBEDEKT_VASTSTELLINGEN_REGEX, na=False)
        )
        gt_cover.loc[onbedekt_from_vaststellingen] = "ONBEDEKT"

    if "MON_VRU" not in merged_df.columns:
        return gt_cover

    mon_vru = merged_df["MON_VRU"].astype("string")
    has_mon_vru = mon_vru.notna()
    remaining_mask = gt_cover.eq("NODATA") & has_mon_vru
    gt_cover.loc[remaining_mask] = "BEDEKT"
    gt_cover.loc[remaining_mask & mon_vru.eq("MON_VRU_BRAAK")] = "ONBEDEKT"
    return gt_cover


def _calc_cover(
    input_parcel_path: Path,
    ts_path: Path,
    ndvi_threshold: float,
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
    parcel_ids = pdh.read_file(
        input_parcel_path, ignore_geometry=True, columns=[id_column]
    )[id_column]

    info = gfo.get_layerinfo(ts_path, layer="info", raise_on_nogeom=False)
    columns = {}
    for column in info.columns:
        column_lower = column.lower()
        if "-asc-" in column_lower:
            orbit = "asc"
        elif "-desc-" in column_lower:
            orbit = "desc"
        elif column_lower.startswith("s2-ndvi"):
            if column_lower.endswith("_ndvi_median"):
                columns["s2_ndvi_median"] = column
        else:
            continue

        key = f"s1{orbit}_{'_'.join(column_lower.split('-')[-1].split('_')[-2:])}"
        columns[key] = column

    if "s2_ndvi_median" not in columns:
        # No s2 data found for this period
        return

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
                ,CASE WHEN "{columns["s2_ndvi_median"]}" IS NULL
                                OR "{columns["s2_ndvi_median"]}" = 0 THEN 'NODATA'
                      WHEN "{columns["s2_ndvi_median"]}" < {ndvi_threshold} THEN 'bare-soil'
                      ELSE 'other'
                 END AS cover_s2_ndvi
                 ,"{columns["s2_ndvi_median"]}" AS s2_ndvi_median
              FROM "info" info
            ) covers
    """  # noqa: E501

    # Add pred1_prob column based on all previous columns.
    #
    # Important: pred1_prob is a legacy rule-based score, not raw NDVI.
    # - When S2 NDVI is available, low NDVI is mapped to a higher bare-soil score.
    # - When S2 is unavailable, the historical S1 heuristic fallback can still emit
    #   a score of 0.5 for bare-soil.
    #
    # The configured ndvi_threshold is only used to build this
    # score. The later 0.5 ONBEDEKT cutoff applies to the score itself.
    # TODO: "AND s2_ndvi_median <> 0" should not be in below expression, but apparently
    # s2_ndvi_median is never NULL -> problem with zonal stats?
    sql = f"""
        SELECT sub.*
                ,CASE
                    WHEN s2_ndvi_median IS NOT NULL AND s2_ndvi_median >= {ndvi_threshold} THEN 0.0
                    WHEN s2_ndvi_median IS NOT NULL AND s2_ndvi_median <> 0 AND s2_ndvi_median < {ndvi_threshold} THEN 1 - s2_ndvi_median
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
                    WHEN pred1_prob > {_THRESHOLD_ONBEDEKT} THEN 'bare-soil'
                    WHEN pred1_prob > {_THRESHOLD_DOUBT} THEN 'DOUBT'
                    ELSE 'other'
                END AS pred1
          FROM ({sql}) sub2
    """
    result_df = pdh.read_file(ts_path, sql=sql)
    result_df = result_df[result_df[id_column].isin(parcel_ids)]
    pdh.to_file(result_df, output_path)


def _get_generic_sentinel_column_mapping(
    columns_list: list[str],
) -> tuple[dict[str, str], str | None]:
    """Map raw period-specific S1 column names to generic names.

    eg: ``s1-grd-sigma0-asc-weekly_20250901_VV_median``
        is mapped to ``s1-grd-sigma0-asc_vv_median``.
    This allows the trained model to be used for multiple time periods.

    Args:
        columns_list: list of raw column names (as returned by gfo.get_layerinfo).

    Returns:
        Tuple of:
            - s1_col_map: {generic_name: raw_column_name} for all S1 columns found.
            - ndvi_col: raw column name of the NDVI median, or None if not present.
    """
    s1_col_map: dict[str, str] = {}
    ndvi_col: str | None = None
    for column in columns_list:
        if _S2_NDVI_COLUMN_REGEX.match(column):
            ndvi_col = column
        elif _S1_COLUMN_REGEX.match(column):
            s1_col_map[_PERIOD_REGEX.sub("", column.lower())] = column
    return s1_col_map, ndvi_col


def _train_ml_cover_model(
    period_timeseries: list[tuple[datetime, pd.DataFrame]],
    id_column: str,
    ndvi_threshold: float,
    ground_truth_path: Path | None = None,
    output_test_eval_path: Path | None = None,
    output_gt_eval_path: Path | None = None,
) -> tuple:
    """Train a Random Forest on S1 features labelled by S2 NDVI.

    Accepts per-period DataFrames that were already collected by the caller
    (one DataFrame per weekly period, containing both S1 bands and S2 NDVI).
    A single model is trained on samples pooled across all periods, which gives
    more training data than any single-period subset.

    At prediction time the model is applied per-period to S1 features only, so
    it can produce cover predictions even when S2 is cloudy.

    Args:
        period_timeseries: List of ``(period_start_date, DataFrame)`` tuples.
            Each DataFrame contains the aggregated S1 and S2 NDVI columns for
            all parcels in that period.
        id_column: Name of the parcel ID column.
        ndvi_threshold: NDVI threshold to binarize S2 labels into ONBEDEKT vs BEDEKT.
        ground_truth_path: Optional path to a GeoPackage with ground truth parcel IDs
            to exclude from training and testing the model
        output_test_eval_path: Optional path to write a text file with metrics.
        output_gt_eval_path: Optional path to write a text file with metrics.

    Returns:
        Tuple ``(classifier, s1_column_names)`` where
        *classifier* is a fitted ``RandomForestClassifier`` and
        *s1_column_names* is the ordered list of generic S1 column names
          the model was trained on.
    """
    # one record per parcel per period that had enough S2 data to use as labels
    s1_features_per_period: list = []
    bare_soil_labels_per_period: list = []
    parcel_ids_per_period: list = []
    s1_column_names: list[str] | None = None

    for period_start_date, period_df in period_timeseries:
        s1_col_map, ndvi_column = _get_generic_sentinel_column_mapping(
            list(period_df.columns)
        )

        if not s1_col_map or ndvi_column is None:
            logger.debug(
                f"No S1/NDVI columns for period {period_start_date.date()}, skipping"
            )
            continue

        ndvi_values = period_df[ndvi_column]
        rows_with_s2_data = ndvi_values.notna() & (ndvi_values != 0)
        if rows_with_s2_data.count() == 0:
            logger.debug(f"No S2 observations for period {period_start_date.date()}")
            continue

        # Rename date-specific S1 columns to stable generic names
        # so one model can be applied across different time periods
        if s1_column_names is None:
            s1_column_names = sorted(s1_col_map.keys())

        period_s1_features = (
            period_df.loc[rows_with_s2_data, list(s1_col_map.values())]
            .rename(columns={v: k for k, v in s1_col_map.items()})
            .reindex(columns=s1_column_names, fill_value=0)
            .fillna(0)
            .to_numpy()
        )
        period_bare_soil_labels = (
            (ndvi_values.loc[rows_with_s2_data] < ndvi_threshold).astype(int).to_numpy()
        )
        period_parcel_ids = period_df.loc[rows_with_s2_data, id_column].to_numpy()

        s1_features_per_period.append(period_s1_features)
        bare_soil_labels_per_period.append(period_bare_soil_labels)
        parcel_ids_per_period.append(period_parcel_ids)

    if not s1_features_per_period or s1_column_names is None:
        raise RuntimeError(
            "Could not collect any S2-labelled training samples for the S1 ML model. "
            "Make sure both S1 and S2 NDVI timeseries data are present."
        )

    # Pool all periods together
    all_s1_features = np.vstack(s1_features_per_period)
    all_bare_soil_labels = np.concatenate(bare_soil_labels_per_period)
    all_parcel_ids = np.concatenate(parcel_ids_per_period)

    # Exclude ground truth parcels so the model is never trained or tested on them
    gt_parcel_ids_set: set = set()
    if ground_truth_path is not None and ground_truth_path.exists():
        try:
            gt_df = pdh.read_file(ground_truth_path)
            gt_parcel_ids_set = set(gt_df[id_column].astype(str))
            logger.info(
                f"Excluding {len(gt_parcel_ids_set)} ground truth parcels from "
                "S1 ML training/test pool"
            )
        except Exception as ex:
            logger.warning(f"Could not load ground truth IDs for exclusion: {ex}")

    all_parcel_ids_str = all_parcel_ids.astype(str)
    is_gt_parcel = np.array([pid in gt_parcel_ids_set for pid in all_parcel_ids_str])
    gt_features = all_s1_features[is_gt_parcel]
    gt_labels = all_bare_soil_labels[is_gt_parcel]

    non_gt_mask = ~is_gt_parcel
    pool_features = all_s1_features[non_gt_mask]
    pool_labels = all_bare_soil_labels[non_gt_mask]
    pool_parcel_ids = all_parcel_ids[non_gt_mask]

    test_size = conf.classifier.getfloat("test_size")

    # Parcel-based train/test split to prevent data leakage.
    # Make the split on unique ids, because an id can appear in multiple periods.
    # This ensures the test set only contains parcels the model has never seen.
    unique_parcel_ids = np.unique(pool_parcel_ids)
    train_parcel_ids, test_parcel_ids = train_test_split(
        unique_parcel_ids, test_size=test_size, random_state=42
    )
    train_parcel_set = set(train_parcel_ids)
    is_train_parcel = np.array([pid in train_parcel_set for pid in pool_parcel_ids])
    is_test_parcel = ~is_train_parcel

    train_features = pool_features[is_train_parcel]
    test_features = pool_features[is_test_parcel]
    train_labels = pool_labels[is_train_parcel]
    test_labels = pool_labels[is_test_parcel]

    n_baresoil_in_train = int(train_labels.sum())
    n_covered_in_train = len(train_labels) - n_baresoil_in_train
    logger.info(
        f"Training S1 Random Forest on {len(train_labels)} samples "
        f"({len(test_labels)} held out for test) from "
        f"{len(s1_features_per_period)} periods "
        f"(train: {n_baresoil_in_train} ONBEDEKT / {n_covered_in_train} BEDEKT)"
    )
    classifier_kwargs = conf.classifier.getdict("classifier_sklearn_kwargs")
    if classifier_kwargs is None:
        classifier_kwargs = {}

    configured_classifier_type = conf.classifier.get(
        "classifier_type", fallback="randomforest"
    )
    if configured_classifier_type.lower() != "randomforest":
        # TODO: andere types toevoegen indien nodig
        logger.warning(
            "Cover ML only supports RandomForest, ignoring classifier_type=%s",
            configured_classifier_type,
        )
    classifier = RandomForestClassifier(**classifier_kwargs)
    classifier.fit(train_features, train_labels)
    labels = [0, 1]
    target_labels = ["BEDEKT", "ONBEDEKT"]

    # TODO: eens testen met huidige week + x-1 week + x+1 week in pred zetten?
    if len(test_labels) > 0:
        test_report = classification_report(
            test_labels,
            classifier.predict(test_features),
            labels=labels,
            target_names=target_labels,
            zero_division=0,
        )
        logger.info(f"S1 ML model test-set evaluation:\n{test_report}")
        if output_test_eval_path is not None:
            output_test_eval_path.write_text(test_report)

    # Evaluate separately on ground truth parcels (S2-labelled, if any)
    if len(gt_labels) > 0:
        gt_test_report = classification_report(
            gt_labels,
            classifier.predict(gt_features),
            labels=labels,
            target_names=target_labels,
            zero_division=0,
        )
        logger.info(
            f"S1 ML model evaluation on ground truth parcels "
            f"({len(gt_labels)} samples):\n{gt_test_report}"
        )
        if output_gt_eval_path is not None:
            output_gt_eval_path.write_text(gt_test_report)
    else:
        logger.info("No ground truth parcels found in the S2-labelled training pool.")

    return classifier, s1_column_names


def _calc_cover_and_predict(
    input_parcel_path: Path,
    ts_path: Path,
    parcel_columns: list[str] | None,
    ndvi_threshold: float,
    output_path: Path,
    classifier: RandomForestClassifier,
    feature_names: list[str],
    output_geo_path: Path | None = None,
    force: bool = False,
) -> None:
    """Run rule-based cover detection and add S1 ML prediction for comparison.

    First runs the standard :func:`_calc_cover` to produce ``pred1_prob``
    (S2-dominated threshold-based result), then applies the pre-trained *classifier*
    to the S1 features of this period to produce ``s1_ml_prob``.

    Output columns added on top of :func:`_calc_cover`:
        s1_ml_prob -- probability of ONBEDEKT according to the S1 ML model
    """
    needs_result_refresh = force or not output_path.exists()
    needs_geo_refresh = output_geo_path is not None and (
        force or not output_geo_path.exists()
    )
    if not needs_result_refresh and not needs_geo_refresh:
        logger.info(f"Output already calculated {output_path}, skipping")
        return

    logger.info(f"start ML+rules cover processing {output_path}")

    result_df = None

    if needs_result_refresh:
        # Step 1: Run standard rule-based detection.
        _calc_cover(
            input_parcel_path=input_parcel_path,
            ts_path=ts_path,
            ndvi_threshold=ndvi_threshold,
            output_path=output_path,
            output_geo_path=None,
            force=force,
        )

        # Step 2: Predict using classifier
        try:
            id_column = conf.columns["id"]
            parcel_ids = pdh.read_file(
                input_parcel_path, ignore_geometry=True, columns=[id_column]
            )[id_column]
            info = gfo.get_layerinfo(ts_path, layer="info", raise_on_nogeom=False)
            s1_col_map, _ = _get_generic_sentinel_column_mapping(list(info.columns))

            raw_df = pdh.read_file(ts_path)
            raw_df = raw_df[raw_df[id_column].isin(parcel_ids)].reset_index(drop=True)
            generic_df = raw_df.rename(columns={v: k for k, v in s1_col_map.items()})
            model_input_df = generic_df.reindex(columns=feature_names, fill_value=0)

            # Parcels with at least some valid S1 signal
            s1_raw_cols = [
                column for column in s1_col_map.values() if column in raw_df.columns
            ]
            s1_has_data_series = raw_df[s1_raw_cols].fillna(0).ne(0).any(axis=1)
            s1_ml_prob_series = pd.Series(float("nan"), index=raw_df.index, dtype=float)
            if feature_names and s1_has_data_series.any():
                proba = classifier.predict_proba(
                    model_input_df.loc[s1_has_data_series].fillna(0).to_numpy()
                )
                classes = classifier.classes_.tolist()
                onbedekt_idx = classes.index(1) if 1 in classes else None
                if onbedekt_idx is not None:
                    s1_ml_prob_series.loc[s1_has_data_series] = proba[:, onbedekt_idx]

            # Add ML column to the rule-based output and re-save
            result_df = pdh.read_file(output_path)
            ml_df = pd.DataFrame(
                {
                    id_column: raw_df[id_column],
                    "s1_ml_prob": s1_ml_prob_series.to_numpy(),
                }
            )
            result_df = result_df.merge(ml_df, on=id_column, how="left")
            gfo.remove(output_path, missing_ok=True)
            pdh.to_file(result_df, output_path)
        except Exception as e:
            logger.error(f"Error in ML cover calculation for {output_path}: {e}")
            raise

    if output_geo_path is not None:
        # If a geo file is asked, always recreate it so it is in sync with the output
        gfo.remove(output_geo_path, missing_ok=True)
        id_column = conf.columns["id"]
        parcel_ids = pdh.read_file(
            input_parcel_path, ignore_geometry=True, columns=[id_column]
        )[id_column]
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

        satdata_df = pdh.read_file(ts_path)
        satdata_df = satdata_df[satdata_df[id_column].isin(parcel_ids)]

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
                      AND "{columns["ndvi_median"]}" < {_DEFAULT_THRESHOLD_NDVI}
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
    _ = input_parcel_path
    report_dir = parcels_selected_path.parent
    report_path = report_dir / f"{parcels_selected_path.stem}_accuracy_report.html"

    if report_path.exists() and not force:
        logger.info(f"Report already exists, skipping: {report_path}")
        return

    logger.info("Generating report...")
    stats_list = []
    classif_report_str: str | None = None

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

    # S1 source breakdown
    if (
        "pred_source" in parcels_selected_df.columns
        and "s2_available" in parcels_selected_df.columns
    ):
        is_ml = parcels_selected_df["pred_source"] == _PRED_SOURCE_ML
        n_natural_ml = int((is_ml & ~parcels_selected_df["s2_available"]).sum())
        stats_list.append(
            {
                "category": "ML natural fallback (geen S2)",
                "count": n_natural_ml,
                "percentage": (n_natural_ml / total_parcels) * 100,
                "total_parcels": total_parcels,
            }
        )
        logger.info(f"ML source breakdown: natural fallback={n_natural_ml}")

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

            has_vaststellingen: pd.Series | bool = False
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
            merged_df["gt_cover"] = _determine_groundtruth_cover_series(merged_df)
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

                try:
                    classif_report_str = classification_report(
                        valid_df["gt_cover"],
                        valid_df[pred_column],
                        zero_division=0,
                    )
                except Exception as ex:
                    logger.warning(f"Could not compute classification_report: {ex}")
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

    # Include sklearn classification_report if available
    if classif_report_str is not None:
        html_parts.append("<h2>Ground truth classification report</h2>")
        html_parts.append("<p>Ground truth accuracies.</p>")
        html_parts.append(f"<pre>{classif_report_str}</pre>")

    # Include S1 ML model test-set evaluation if available
    s1_ml_test_eval_path = report_dir / "s1_ml_test_eval.txt"
    if s1_ml_test_eval_path.exists():
        html_parts.append("<h2>S1 ML Model - Test Set Evaluation (S2-labelled)</h2>")
        html_parts.append("<p>Test set ML accuracies.</p>")
        html_parts.append(f"<pre>{s1_ml_test_eval_path.read_text()}</pre>")

    # Include S1 ML model evaluation on ground truth parcels if available
    s1_ml_eval_path = report_dir / "s1_ml_groundtruth_eval.txt"
    if s1_ml_eval_path.exists():
        html_parts.append(
            "<h2>S1 ML Model - Ground truth Set Evaluation (S2-labelled)</h2>"
        )
        html_parts.append("<p>Ground truth ML accuracies.</p>")
        html_parts.append(f"<pre>{s1_ml_eval_path.read_text()}</pre>")

    html_parts.append("<h2>Ground truth</h2>")
    columns_to_save = [
        id_column,
        # Final outputs
        "pred_consolidated",
        "pred_cons_status",
        "pred1_prob",
        "pred_source",
        # Threshold rule-based
        "thresh_pred",
        "thresh_prob",
        "s2_available",
        # ML
        "ml_prob_max",
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
