# -*- coding: utf-8 -*-
"""
Module with postprocessing functions on classification results.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geofileops as gfo
import geopandas as gpd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def calc_top3_and_consolidation(
    input_parcel_path: Path,
    input_parcel_probabilities_path: Path,
    input_parcel_geopath: Path,
    output_predictions_path: Path,
    output_predictions_geopath: Path,
    output_predictions_output_path: Optional[Path] = None,
    force: bool = False,
):
    """
    Calculate the top3 prediction and a consolidation prediction.

    Remark: in this logic the declared crop/class (class_declared) is used, as we want
    to compare with the declaration of the farmer, rather than taking into account
    corrections already.

    Args:
        input_parcel_path (Path): [description]
        input_parcel_probabilities_path (Path): [description]
        output_predictions_path (Path): [description]
        output_predictions_geopath (Path): [description]
        output_predictions_output_path (Path, optional): [description].
            Defaults to None.
        force (bool, optional): [description]. Defaults to False.
    """

    # If force is false and output exists, already, return
    if force is False and output_predictions_path.exists():
        logger.warning(
            "calc_top3_and_consolidation: output file exist and force is False, so "
            f"stop: {output_predictions_path}"
        )
        return

    # Read input files
    logger.info("Read input file")
    proba_df = pdh.read_file(input_parcel_probabilities_path)

    top3_df = calc_top3(proba_df)

    # Read input files
    logger.info("Read input file")
    input_parcel_df = pdh.read_file(input_parcel_path)

    # All input parcels must stay in the output, so left join input with pred
    top3_df.set_index(conf.columns["id"], inplace=True)
    if input_parcel_df.index.name != conf.columns["id"]:
        input_parcel_df.set_index(conf.columns["id"], inplace=True)
    cols_to_join = top3_df.columns.difference(input_parcel_df.columns)
    pred_df = input_parcel_df.join(top3_df[cols_to_join], how="left")

    # The parcels added by the join don't have a prediction yet, so apply it
    # For the ignore classes, set the prediction to the ignore type
    classes_to_ignore = conf.marker.getlist("classes_to_ignore")
    pred_df.loc[
        pred_df[conf.columns["class_declared"]].isin(classes_to_ignore), "pred1"
    ] = pred_df[conf.columns["class_declared"]]
    # For all other parcels without prediction there must have been no data
    # available for a classification, so set prediction to NODATA
    pred_df["pred1"].fillna("NODATA", inplace=True)

    # Add doubt columns
    add_doubt_column(
        pred_df=pred_df,
        new_pred_column=conf.columns["prediction_cons"],
        apply_doubt_pct_proba=True,
        apply_doubt_min_nb_pixels=True,
        apply_doubt_marker_specific=False,
    )
    add_doubt_column(
        pred_df=pred_df,
        new_pred_column=conf.columns["prediction_full_alpha"],
        apply_doubt_pct_proba=True,
        apply_doubt_min_nb_pixels=True,
        apply_doubt_marker_specific=True,
    )

    # Calculate the status of the consolidated prediction (OK=usable, NOK=not)
    pred_df.loc[
        pred_df[conf.columns["prediction_full_alpha"]].isin(proba_df.columns),
        conf.columns["prediction_cons_status"],
    ] = "OK"
    pred_df[conf.columns["prediction_cons_status"]].fillna("NOK", inplace=True)

    logger.info("Write full prediction data to file")
    pdh.to_file(pred_df, output_predictions_path)

    # Output to geo file
    input_parcel_gdf = gfo.read_file(input_parcel_geopath)
    pred_gdf = gpd.GeoDataFrame(input_parcel_gdf.merge(pred_df, how="inner"))
    pred_gdf.to_file(output_predictions_geopath, engine="pyogrio")

    # Create final output file with the most important info
    if output_predictions_output_path is not None:
        # First add some aditional columns specific for the export
        pred_df["markercode"] = conf.marker["markertype"]
        pred_df["run_id"] = conf.general["run_id"]
        today = datetime.date.today()
        pred_df["cons_date"] = today
        pred_df["modify_date"] = today
        logger.info("Write final output prediction data to file")
        pred_df.reset_index(inplace=True)
        pred_df = pred_df[conf.columns.getlist("output_columns")]
        pdh.to_file(
            pred_df, output_predictions_output_path, index=False  # type: ignore
        )

        # Write oracle sqlldr file
        table_name = None
        table_columns = None
        if conf.marker["markertype"] in ["LANDCOVER", "LANDCOVER-EARLY"]:
            table_name = "mon_marker_landcover"
            table_columns = (
                "layer_id, prc_id, versienummer, markercode, run_id, cons_input, "
                "cons_landcover, cons_status, cons_date date 'yyyy-mm-dd', landcover1, "
                "probability1, landcover2, probability2, landcover3, probability3, "
                "modify_date date 'yyyy-mm-dd'"
            )
        elif conf.marker["markertype"] in ["CROPGROUP", "CROPGROUP-EARLY"]:
            table_name = "mon_marker_cropgroup"
            table_columns = (
                "layer_id, prc_id, versienummer, markercode, run_id, cons_input, "
                "cons_cropgroup, cons_status, cons_date date 'yyyy-mm-dd', cropgroup1, "
                "probability1, cropgroup2, probability2, cropgroup3, probability3, "
                "modify_date date 'yyyy-mm-dd'"
            )
        else:
            table_name = None
            logger.warning(
                f"Table unknown for marker type {conf.marker['markertype']}, so cannot "
                "write .ctl file"
            )

        if table_name is not None and table_columns is not None:
            with open(str(output_predictions_output_path) + ".ctl", "w") as ctlfile:
                # SKIP=1 to skip the columns names line, the other ones to evade
                # more commits than needed
                ctlfile.write(
                    "OPTIONS (SKIP=1, ROWS=10000, BINDSIZE=40000000, "
                    "READSIZE=40000000)\n"
                )
                ctlfile.write("LOAD DATA\n")
                ctlfile.write(
                    f"INFILE '{output_predictions_output_path.name}'  \"str '\\n'\"\n"
                )
                ctlfile.write(f"INSERT INTO TABLE {table_name} APPEND\n")
                # A tab as seperator is apparently X'9'
                ctlfile.write("FIELDS TERMINATED BY X'9'\n")
                ctlfile.write(f"({table_columns})\n")


def calc_top3(proba_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the top 3 predictions
    logger.info("Calculate top3")
    proba_tmp_df = proba_df.copy()
    for column in proba_tmp_df.columns:
        if column in conf.preprocess.getlist("dedicated_data_columns"):
            proba_tmp_df.drop(column, axis=1, inplace=True)

    # Get the top 3 predictions for each row
    # First get the indices of the top 3 predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-4": the last 3 elements of the values
    #     - ":-1]": and than reverse the order with a negative step
    top3_pred_classes_idx = np.argsort(proba_tmp_df.values, axis=1)[:, :-4:-1]
    # Convert the indices to classes
    top3_pred_classes = np.take(proba_tmp_df.columns, top3_pred_classes_idx)
    # Get the values of the top 3 predictions
    top3_pred_values = np.sort(proba_tmp_df.values, axis=1)[:, :-4:-1]
    # Concatenate both
    top3_pred = np.concatenate([top3_pred_classes, top3_pred_values], axis=1)
    # Concatenate the ids, the classes and the top3 predictions
    id_class_top3 = np.concatenate(
        [
            proba_df[[conf.columns["id"], conf.columns["class_declared"]]].values,
            top3_pred,
        ],
        axis=1,
    )

    # Convert to dataframe
    # Also apply infer_objects: otherwise the float columns are of object dtype.
    top3_df = pd.DataFrame(
        id_class_top3,
        columns=[
            conf.columns["id"],
            conf.columns["class_declared"],
            "pred1",
            "pred2",
            "pred3",
            "pred1_prob",
            "pred2_prob",
            "pred3_prob",
        ],
    ).infer_objects()

    return top3_df


def add_doubt_column(
    pred_df: pd.DataFrame,
    new_pred_column: str,
    apply_doubt_pct_proba: bool,
    apply_doubt_min_nb_pixels: bool,
    apply_doubt_marker_specific: bool,
):
    # Calculate predictions with doubt column
    classes_to_ignore = conf.marker.getlist("classes_to_ignore")

    # Init with the standard prediction
    pred_df[new_pred_column] = "UNDEFINED"

    # If NODATA OR ignore class, retained those from pred1
    pred_df.loc[
        (pred_df[new_pred_column] == "UNDEFINED")
        & (
            (pred_df["pred1"] == "NODATA")
            | (pred_df[conf.columns["class_declared"]].isin(classes_to_ignore))
        ),
        new_pred_column,
    ] = pred_df["pred1"]

    # Doubt based on percentage probability
    if apply_doubt_pct_proba:
        # Apply doubt for parcels with a low percentage of probability -> = doubt!
        doubt_proba1_st_2_x_proba2 = conf.postprocess.getboolean(
            "doubt_proba1_st_2_x_proba2"
        )
        if doubt_proba1_st_2_x_proba2 is True:
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (
                    pred_df["pred1_prob"].map(float)
                    < 2.0 * pred_df["pred2_prob"].map(float)
                ),
                new_pred_column,
            ] = "DOUBT:PROBA1<2*PROBA2"

        # Apply doubt for parcels with prediction != unverified input
        doubt_pred_ne_input_proba1_st_pct = conf.postprocess.getfloat(
            "doubt_pred_ne_input_proba1_st_pct"
        )
        if doubt_pred_ne_input_proba1_st_pct > 0:
            if doubt_pred_ne_input_proba1_st_pct > 100:
                raise Exception(
                    "doubt_pred_ne_input_proba1_st_pct should be float from 0 till 100,"
                    f" not {doubt_pred_ne_input_proba1_st_pct}"
                )
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df["pred1"] != pred_df[conf.columns["class_declared"]])
                & (
                    pred_df["pred1_prob"].map(float)
                    < (doubt_pred_ne_input_proba1_st_pct / 100)
                ),
                new_pred_column,
            ] = "DOUBT:PRED<>INPUT-PROBA1<X"

        # Apply doubt for parcels with prediction == unverified input
        doubt_pred_eq_input_proba1_st_pct = conf.postprocess.getfloat(
            "doubt_pred_eq_input_proba1_st_pct"
        )
        if doubt_pred_eq_input_proba1_st_pct > 0:
            if doubt_pred_eq_input_proba1_st_pct > 100:
                raise Exception(
                    "doubt_pred_ne_input_proba1_st_pct should be float from 0 till 100,"
                    f" not {doubt_pred_eq_input_proba1_st_pct}"
                )
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df["pred1"] == pred_df[conf.columns["class_declared"]])
                & (
                    pred_df["pred1_prob"].map(float)
                    < (doubt_pred_eq_input_proba1_st_pct / 100)
                ),
                new_pred_column,
            ] = "DOUBT:PRED=INPUT-PROBA1<X"

    # Marker specific doubt
    if apply_doubt_marker_specific is True:
        # Apply some extra, marker-specific doubt algorythms
        if conf.marker["markertype"] in ("LANDCOVER", "LANDCOVER-EARLY"):
            logger.info("Apply some marker-specific doubt algorythms")

            # Remarks:
            #   - To be sure apply RISKY_DOUBT first, so if there would be
            #     overlapping criteria, RISKY_DOUBT "wins".
            #   - Don't overwrite existing DOUBT on parcels, because the
            #     general doubt reasons should win compared to the marker
            #     specific doubt/risky_doubt reasons.
            #   - RISKY DOUBT is used when ground truth resulted in partly alpha
            #     errors, partly correct, classifications.

            # If parcel was declared as grassland, and is classified as arable, set to
            # risky doubt.
            # Remark: those gave 50% false positives for LANDCOVER marker
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] == "MON_LC_GRASSES")
                & (pred_df["pred1"] == "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:GRASS-SEEN-AS-ARABLE"

            # If parcel was declared as fallow, and is classified as something else,
            # set to doubt.
            # Remark: those gave 50% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] == "MON_LC_FALLOW")
                & (pred_df["pred1"] != "MON_LC_FALLOW"),
                new_pred_column,
            ] = "RISKY_DOUBT:FALLOW-UNCONFIRMED"

            # If parcel was declared as "9603: Boomkweek-sierplanten", but is not
            # classified as such: doubt
            # Remark: they gave 50% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9603"]))
                & (pred_df["pred1"] == "MON_LC_GRASSES"),
                new_pred_column,
            ] = "RISKY_DOUBT:BOOMSIER-SEEN-AS-GRASSES"

            # If parcel was declared as grain, but is not classified as MON_LC_ARABLE:
            # doubt
            # Remark: - those gave > 50% false positives for marker LANDCOVER-EARLY
            #         - gave > 33 % false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (
                    pred_df[conf.columns["crop_declared"]].isin(
                        ["39", "311", "321", "322", "331", "342", "639", "646"]
                    )
                )
                & (pred_df["pred1"] != "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:GRAIN-UNCONFIRMED"

            # If parcel was declared as one of the following fabaceae, but is not
            # classified as such: doubt
            # Remark: - those gave > 50% false positives for marker LANDCOVER-EARLY
            #         - gave 33-100% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (
                    pred_df[conf.columns["crop_declared"]].isin(
                        [
                            "43",
                            "51",
                            "52",
                            "721",
                            "722",
                            "731",
                            "732",
                            "831",
                            "931",
                            "8410",
                        ]
                    )
                )
                & (pred_df["pred1"] != "MON_LC_FABACEAE"),
                new_pred_column,
            ] = "RISKY_DOUBT:FABACEAE-UNCONFIRMED"

            # Declared class was not correct, but groundtruth class is permanent
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] != pred_df["pred1"])
                & (pred_df["pred1"].isin(["MON_LC_BOS", "MON_LC_HEIDE"])),
                new_pred_column,
            ] = "RISKY_DOUBT:DECL<>PRED-PRED=" + pred_df["pred1"].map(str)

            # If parcel was declared as some arable crops (9534 = knolvenkel), and is
            # classified as fabaceae, set to doubt
            # Remark: those gave 100% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9534"]))
                & (pred_df["pred1"] == "MON_LC_FABACEAE"),
                new_pred_column,
            ] = "DOUBT:ARABLE-SEEN-AS-FABACEAE"

            # If parcel was declared as 'other herbs' or 'flowers', but is not
            # confirmed as MON_LC_ARABLE classified as such: doubt
            # Remark: - those gave > 50% false positives for marker LANDCOVER-EARLY
            #         - gave 33-50% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["956", "957", "9831"]))
                & (pred_df["pred1"] != "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:HERBS-UNCONFIRMED"

            # If parcel was declared as 'aardbeien', but is not confirmed as
            # MON_LC_ARABLE classified as such: doubt
            # Remark: - those gave > 50% false positives for marker LANDCOVER-EARLY
            #         - gave 33-50% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9516"]))
                & (pred_df["pred1"] != "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:AARDBEIEN-UNCONFIRMED"

            # Declared class was not correct, but groundtruth class is permanent
            # TODO: probably dirty this is hardcoded here!
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] != pred_df["pred1"])
                & (
                    pred_df["pred1"].isin(
                        ["MON_LC_BOS", "MON_LC_HEIDE", "MON_LC_OVERK_LOO"]
                    )
                ),
                new_pred_column,
            ] = "RISKY_DOUBT:DECL<>PRED-PRED=" + pred_df["pred1"].map(str)

            # If parcel was declared as 9410 and classified as ARABLE, set to doubt
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9410"]))
                & (pred_df["pred1"] == "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:STAMSLABONEN-SEEN-AS-ARABLE"

            # If parcel was declared as 9602 and classified as FRUIT, set to doubt
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9602"]))
                & (pred_df["pred1"] == "MON_LC_FRUIT"),
                new_pred_column,
            ] = "RISKY_DOUBT:BOOM/FRUITKWEEK-UNCONFIRMED"

            # Parcel was declared as one of the following + classified as GRASSES
            #
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (
                    pred_df[conf.columns["crop_declared"]].isin(
                        [
                            "36",
                            "895",
                            "9582",
                            #                            "744",
                            #                            "9202",
                            #                            "9714",
                            #                            "832",
                            #                            "9602",
                            #                            "9730",
                        ]
                    )
                )
                & (pred_df["pred1"] == "MON_LC_GRASSES"),
                new_pred_column,
            ] = "RISKY_DOUBT:SEEN-AS-GRASSES"

            # Parcel was declared as one of the following and classified as ARABLE
            # Ground truth in some cases confirmed the classification, in others
            # resulted in alpha errors -> RISKY_DOUBT
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (
                    pred_df[conf.columns["crop_declared"]].isin(
                        ["601", "644", "932", "9412", "9584", "8411"]
                    )
                )
                & (pred_df["pred1"] == "MON_LC_ARABLE"),
                new_pred_column,
            ] = "RISKY_DOUBT:SEEN-AS-ARABLE"

            # If parcel was declared as one of the following and classified as
            # FABACEAE, set to RISKY_DOUBT
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9546"]))
                & (pred_df["pred1"] == "MON_LC_FABACEAE"),
                new_pred_column,
            ] = "RISKY_DOUBT:SAVOOIKOOL-SEEN-AS-FABACEAE"

        elif conf.marker["markertype"] in ("CROPGROUP", "CROPGROUP-EARLY"):
            logger.info("Apply some marker-specific doubt algorythms")

            # Red parcels declared as MON_TRITICALE always seem to be predicted wrong,
            # so place them in doubt
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] != pred_df["pred1"])
                & (pred_df[conf.columns["class_declared"]] == "MON_TRITICALE"),
                new_pred_column,
            ] = "RISKY_DOUBT:TRITICALE-UNCONFIRMED"

            # Red parcels with MON_HEIDE seem to have a difficult time seeing the
            # difference between MON_HEIDE and MON_GRASSEN
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["class_declared"]] != pred_df["pred1"])
                & (pred_df["pred1"] == "MON_HEIDE"),
                new_pred_column,
            ] = "RISKY_DOUBT:HEIDE-UNCONFIRMED"

            """
            # Note: this will INCREASE the alpha errors, because we remove a large
            # portion of red parcels to doubt (~886 parcels => 16 alpha errors)
            pred_df.loc[
                ((pred_df[new_pred_column] == 'UNDEFINED') |
                        (~pred_df[new_pred_column].str.startswith('DOUBT')))
                    & (pred_df[conf.columns['class_declared']] != pred_df['pred1'])
                    & (pred_df[conf.columns['class_declared']] == 'MON_GRASSEN'),
                    new_pred_column
            ] = 'RISKY_DOUBT:GRASSEN-UNCONFIRMED'
            """

    # Accuracy with few pixels might be lower, so set those to doubt
    if apply_doubt_min_nb_pixels is True:
        pred_df.loc[
            (
                pred_df[conf.columns["pixcount_s1s2"]]
                < conf.marker.getfloat("min_nb_pixels")
            )
            & (~pred_df[new_pred_column].str.startswith("DOUBT")),
            new_pred_column,
        ] = "DOUBT:NOT_ENOUGH_PIXELS"

    # Finally, predictions that have no value yet, get the original prediction
    pred_df.loc[pred_df[new_pred_column] == "UNDEFINED", new_pred_column] = pred_df[
        "pred1"
    ]
