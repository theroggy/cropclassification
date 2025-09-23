"""Module with postprocessing functions on classification results."""

import datetime
import logging
from pathlib import Path

import geofileops as gfo
import numpy as np
import pandas as pd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)


def calc_top_classes_and_consolidation(
    input_parcel_path: Path,
    input_parcel_probabilities_path: Path,
    input_parcel_geopath: Path,
    output_predictions_path: Path,
    output_predictions_geopath: Path,
    top_classes: int,
    output_predictions_output_path: Path | None = None,
    force: bool = False,
):
    """Calculate the top3 prediction and a consolidation prediction.

    Remark: in this logic the declared crop/class (class_declared) is used, as we want
    to compare with the declaration of the farmer, rather than taking into account
    corrections already.

    Args:
        input_parcel_path (Path): path to the input parcel file
        input_parcel_probabilities_path (Path): path to the input probabilities file
        input_parcel_geopath (Path): path to the input parcel geofile
        output_predictions_path (Path): path to the output predictions file
        output_predictions_geopath (Path): path to the output predictions geofile
        top_classes (int, optional): Number of top predictions to retain. Defaults to 3.
        output_predictions_output_path (Path, optional): path to the output predictions
            file prepared to load in oracle. Defaults to None.
        force (bool, optional): True to overwrite existing output files.
            Defaults to False.
    """
    # If force is false and output exists, already, return
    if force is False and output_predictions_path.exists():
        logger.warning(
            "calc_top3_and_consolidation: output file exist and force is False, so "
            f"stop: {output_predictions_path}"
        )
        return
    if not input_parcel_probabilities_path.exists():
        raise ValueError(f"file does not exist: {input_parcel_probabilities_path=}")
    if not input_parcel_path.exists():
        raise ValueError(f"file does not exist: {input_parcel_path=}")

    # Read input files
    logger.info("Read input file")
    proba_df = pdh.read_file(input_parcel_probabilities_path)

    top_classes_df = calc_top_classes(proba_df, top_classes)

    # Read input files
    logger.info("Read input file")
    input_parcel_df = pdh.read_file(input_parcel_path)

    # All input parcels must stay in the output, so left join input with pred
    top_classes_df.set_index(conf.columns["id"], inplace=True)
    if input_parcel_df.index.name != conf.columns["id"]:
        input_parcel_df.set_index(conf.columns["id"], inplace=True)
    cols_to_join = top_classes_df.columns.difference(input_parcel_df.columns)
    pred_df = input_parcel_df.join(top_classes_df[cols_to_join], how="left")

    # The parcels added by the join don't have a prediction yet, so apply it
    # For the ignore classes, set the prediction to the ignore type
    classes_to_ignore = conf.marker.getlist("classes_to_ignore")
    pred_df.loc[
        pred_df[conf.columns["class_declared"]].isin(classes_to_ignore), "pred1"
    ] = pred_df[conf.columns["class_declared"]]
    # For all other parcels without prediction there must have been no data
    # available for a classification, so set prediction to NODATA
    pred_df.fillna({"pred1": "NODATA"}, inplace=True)

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
    pred_df.fillna({conf.columns["prediction_cons_status"]: "NOK"}, inplace=True)

    # Output to geo file
    input_parcel_gdf = gfo.read_file(input_parcel_geopath).set_index(conf.columns["id"])
    pred_gdf = input_parcel_gdf[["geometry"]].join(pred_df, how="inner")
    pred_gdf.to_file(output_predictions_geopath, engine="pyogrio")

    # Create final output file with the most important info
    if output_predictions_output_path is not None:
        # First add some aditional columns specific for the export
        pred_output_df = pred_df.copy()
        pred_output_df["markercode"] = conf.marker["markertype"]
        today = datetime.date.today()
        pred_output_df["cons_date"] = today
        pred_output_df["modify_date"] = today
        logger.info("Write final output prediction data to file")
        pred_output_df.reset_index(inplace=True)
        pred_output_df = pred_output_df[conf.columns.getlist("output_columns")]
        pdh.to_file(pred_output_df, output_predictions_output_path, index=False)

        # Write oracle sqlldr file
        table_name = None
        table_columns = None
        if conf.marker["markertype"] in ["LANDCOVER", "LANDCOVER-EARLY"]:
            table_name = "mon_marker_landcover"
            table_columns = (
                "layer_id, prc_id, versienummer, markercode, cons_input, "
                "cons_landcover, cons_status, cons_date date 'yyyy-mm-dd', landcover1, "
                "probability1, landcover2, probability2, landcover3, probability3, "
                "modify_date date 'yyyy-mm-dd'"
            )
        elif conf.marker["markertype"] in ["CROPGROUP", "CROPGROUP-EARLY"]:
            table_name = "mon_marker_cropgroup"
            table_columns = (
                "layer_id, prc_id, versienummer, markercode, cons_input, "
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

    logger.info("Write full prediction data to file")
    pdh.to_file(pred_df, output_predictions_path)


def calc_top_classes(proba_df: pd.DataFrame, top_classes: int = 3) -> pd.DataFrame:
    """Calculate the top x predictions for each row.

    Args:
        proba_df (pd.DataFrame): dataframe with the probabilities
        top_classes (int, optional): the number or top predicted classes to retain.
            Defaults to 3.

    Returns:
        pd.DataFrame: dataframe with the top x predictions for each row.
    """
    # Calculate the top x predictions
    logger.info("Calculate top classes")
    proba_tmp_df = proba_df.copy()
    for column in proba_tmp_df.columns:
        if column in conf.preprocess.getlist("dedicated_data_columns"):
            proba_tmp_df.drop(column, axis=1, inplace=True)

    # Get the top x predictions for each row
    # First get the indices of the top x predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-top_filter": the last x elements of the values (= the top x predictions)
    #     - ":-1]": and than reverse the order with a negative step
    top_filter = top_classes + 1
    top_pred_classes_idx = np.argsort(proba_tmp_df.values, axis=1)[:, :-top_filter:-1]
    # Convert the indices to classes
    top_pred_classes = np.take(proba_tmp_df.columns, top_pred_classes_idx)
    # Get the values of the top predictions
    top_pred_values = np.sort(proba_tmp_df.values, axis=1)[:, :-top_filter:-1]
    # Concatenate both
    top_pred = np.concatenate([top_pred_classes, top_pred_values], axis=1)
    # Concatenate the ids, the classes and the top predictions
    id_class_top = np.concatenate(
        [
            proba_df[[conf.columns["id"], conf.columns["class_declared"]]].values,
            top_pred,
        ],
        axis=1,
    )

    # Convert to dataframe
    # Also apply infer_objects: otherwise the float columns are of object dtype.
    columns = [conf.columns["id"], conf.columns["class_declared"]]
    for i in range(top_classes):
        columns.append(f"pred{i + 1}")
    for i in range(top_classes):
        columns.append(f"pred{i + 1}_prob")
    top_df = pd.DataFrame(id_class_top, columns=columns).infer_objects()

    return top_df


def add_doubt_column(
    pred_df: pd.DataFrame,
    new_pred_column: str,
    apply_doubt_pct_proba: bool,
    apply_doubt_min_nb_pixels: bool,
    apply_doubt_marker_specific: bool,
):
    """Add a doubt column to the prediction dataframe.

    Args:
        pred_df (pd.DataFrame): the dataframe with the predictions
        new_pred_column (str): the column name for the new column
        apply_doubt_pct_proba (bool): True to apply doubt based on percentage
            probability
        apply_doubt_min_nb_pixels (bool): True to apply doubt based on minimum number of
            pixels
        apply_doubt_marker_specific (bool): True to apply doubt based on marker specific
            rules
    """
    # Calculate predictions with doubt column
    classes_to_ignore = conf.marker.getlist("classes_to_ignore")

    # Init with the standard prediction
    pred_df[new_pred_column] = "UNDEFINED"

    # If declared is UNKNOWN, retain it
    pred_df.loc[
        (pred_df[new_pred_column] == "UNDEFINED")
        & (pred_df[conf.columns["class_declared"]] == "UNKNOWN"),
        new_pred_column,
    ] = "IGNORE:NOT_DECLARED"

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
                & (pred_df["pred1_prob"] < 2.0 * pred_df["pred2_prob"]),
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
                & (pred_df["pred1_prob"] < (doubt_pred_ne_input_proba1_st_pct / 100)),
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
                & (pred_df["pred1_prob"] < (doubt_pred_eq_input_proba1_st_pct / 100)),
                new_pred_column,
            ] = "DOUBT:PRED=INPUT-PROBA1<X"

    # Marker specific doubt
    if apply_doubt_marker_specific is True:
        # Apply some extra, marker-specific doubt algorythms
        if conf.marker["markertype"] in ("LANDCOVER", "LANDCOVER-EARLY"):
            logger.info("Apply some marker-specific doubt algorythms")

            # If parcel was declared as some arable crops (9534 = knolvenkel), and is
            # classified as fabaceae, set to doubt
            # Remark: those gave 100% false positives for marker LANDCOVER
            pred_df.loc[
                (pred_df[new_pred_column] == "UNDEFINED")
                & (pred_df[conf.columns["crop_declared"]].isin(["9534"]))
                & (pred_df["pred1"] == "MON_LC_FABACEAE"),
                new_pred_column,
            ] = "DOUBT:ARABLE-SEEN-AS-FABACEAE"

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
