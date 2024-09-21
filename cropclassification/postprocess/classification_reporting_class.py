"""
Module for a dtailed report per class.
"""

import logging
from pathlib import Path
from string import Template
from typing import Optional

import geofileops as gfo
import pandas as pd

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)


def write_class_reports(
    parcel_predictions_geopath: Path,
    output_dir: Path,
    parcel_ground_truth_path: Optional[Path] = None,
    parcel_train_path: Optional[Path] = None,
    parcel_classification_data_path: Optional[Path] = None,
    force: bool = False,
):
    """Writes a report about the accuracy of the predictions to a file.

    Args:
        parcel_predictions_geopath: File name of geofile with the parcels with their
            predictions.
        output_dir: Directory to write the reports per class to.
        parcel_ground_truth_path: List of parcels with ground truth to calculate
        parcel_train_path:
        parcel_classification_data_path:
            eg. alfa and beta errors. If None, the part of the report that is based on
            this data is skipped

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    classname = "MON_HAVER"

    write_class_report(
        parcel_predictions_geopath=parcel_predictions_geopath,
        classname=classname,
        output_dir=output_dir,
        parcel_ground_truth_path=parcel_ground_truth_path,
        parcel_train_path=parcel_train_path,
        parcel_classification_data_path=parcel_classification_data_path,
        force=force,
    )


def write_class_report(
    parcel_predictions_geopath: Path,
    output_dir: Path,
    classname: str,
    parcel_ground_truth_path: Optional[Path] = None,
    parcel_train_path: Optional[Path] = None,
    parcel_classification_data_path: Optional[Path] = None,
    force: bool = False,
):
    # If force == False Check and the output file exists already, stop.
    output_report_html = output_dir / f"{classname}.html"
    if not force and output_report_html.exists():
        return

    # Init placeholders for the report
    # --------------------------------
    empty_string = "''"
    html_data = {
        "CLASS_REPORT_TITLE": empty_string,
        "CLASS_REPORT_TEXT": empty_string,
        "CLASS_REPORT_TABLE": empty_string,
        "GROUND_TRUTH_TEXT": empty_string,
        "GROUND_TRUTH_TABLE": empty_string,
    }

    # Process global data for the class
    # ---------------------------------
    html_data["CLASS_REPORT_TITLE"] = f"REPORT FOR CLASS {classname}"
    html_data["CLASS_REPORT_TEXT"] = f"Report for class {classname}"

    # Process prediction data
    predict_df = gfo.read_file(
        parcel_predictions_geopath,
        where=f"{conf.columns['class']} = '{classname}'",
    )
    predict_df.set_index(conf.columns["id"], inplace=True)

    info_lines = []
    info_lines.append(("Global statistics", "Number declared", len(predict_df)))

    gt_info_df = pd.DataFrame(info_lines, columns=["type", "key", "value"])
    html_data["CLASS_REPORT_TABLE"] = gt_info_df.to_html(index=False)

    # If a ground truth file is provided, report on the ground truth
    # --------------------------------------------------------------
    if parcel_ground_truth_path is not None:
        # Read ground truth
        logger.info(
            "Read csv with ground truth (with their classes): "
            f"{parcel_ground_truth_path}"
        )
        parcel_gt_df = pdh.read_file(parcel_ground_truth_path)
        parcel_gt_df.set_index(conf.columns["id"], inplace=True)
        logger.info(f"Read csv with ground truth ready, shape: {parcel_gt_df.shape}")

        # Join the prediction data
        cols_to_join = predict_df.columns.difference(parcel_gt_df.columns)
        parcel_gt_df = predict_df[cols_to_join].join(parcel_gt_df, how="inner")
        logger.info(
            "After join of ground truth with predictions, shape: "
            f"{parcel_gt_df.shape}"
        )

        if len(parcel_gt_df.index) == 0:
            message = (
                "After join of ground truth with predictions the result was empty, "
                "so probably a wrong ground truth file was used!"
            )
            logger.critical(message)
            raise Exception(message)

        # Fil in general text section
        html_data["GROUND_TRUTH_TEXT"] = (
            f"Reporting on ground truth for class {classname}"
        )

        # Calculate statistics
        # --------------------
        gt_info_lines: list[tuple[str, float, float, str]] = []
        nb_groundtruth = len(parcel_gt_df)
        gt_info_lines.append(("Number parcels with groundtruth", nb_groundtruth, 1, ""))

        # Parcels NOK based on ground truth
        nb_gt_nok = len(parcel_gt_df.query("classname_gt != @classname"))
        gt_info_lines.append(
            (
                f"Parcels NOK ground truth (gt != {classname})",
                nb_gt_nok,
                nb_gt_nok / nb_groundtruth,
                "",
            )
        )

        # Parcels with pred1 != classname
        nb_pred1_nok = len(parcel_gt_df.query("pred1 != @classname"))
        nb_pred1_nok_gt_nok = len(
            parcel_gt_df.query("pred1 != @classname and classname_gt != @classname")
        )

        info = (
            f"Parcels NOK prediction (pred1 != {classname})",
            nb_pred1_nok,
            nb_pred1_nok / nb_groundtruth,
            f"{nb_pred1_nok_gt_nok} ({nb_pred1_nok_gt_nok/nb_gt_nok:.2%} of "
            f"total gt errors) is also gt NOK: {nb_pred1_nok_gt_nok/nb_pred1_nok:.2%}",
        )
        gt_info_lines.append(info)

        # Determine the probability of the class being reported on
        tot_proba_df = parcel_gt_df.copy()
        top_classes = 4
        pred_cols = [
            c
            for c in tot_proba_df.columns
            if len(c) == 5 and c.startswith("pred") and c <= f"pred{top_classes}"
        ]
        for column in pred_cols:
            query = f"{column} != @classname"
            tot_proba_df.loc[tot_proba_df.query(query).index, f"{column}_prob"] = 0
        pred_prob_cols = [f"{c}_prob" for c in pred_cols]
        parcel_gt_df["pred_prob_class"] = tot_proba_df[pred_prob_cols].sum(axis=1)
        del tot_proba_df

        # Parcels with pred_prob_class <= x%
        for prob in [0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.3]:
            nb_pred_prob_class_nok = len(parcel_gt_df.query("pred_prob_class <= @prob"))
            nb_pred_prob_class_nok_gt_nok = len(
                parcel_gt_df.query(
                    "pred_prob_class <= @prob and classname_gt != @classname"
                )
            )

            info = (
                f"Parcels NOK prediction (pred {classname} <= {prob:.0%})",
                nb_pred_prob_class_nok,
                nb_pred_prob_class_nok / nb_groundtruth,
                f"{nb_pred_prob_class_nok_gt_nok} "
                f"({nb_pred_prob_class_nok_gt_nok/nb_gt_nok:.2%} of "
                "total gt errors) are also gt NOK: "
                f"{nb_pred_prob_class_nok_gt_nok/nb_pred_prob_class_nok:.2%}",
            )
            gt_info_lines.append(info)

        # Number/% parcels with pred1 == classname
        nb_pred1_ok = len(parcel_gt_df.query("pred1 == @classname"))
        gt_info_lines.append(
            (
                f"Parcels with pred1 == {classname}",
                nb_pred1_ok,
                nb_pred1_ok / nb_groundtruth,
                "",
            )
        )

        # Create dataframe with the statistics
        gt_info_df = pd.DataFrame(
            gt_info_lines, columns=["statistic", "nb parcels", "% parcels", "remark"]
        )
        gt_info_df["% parcels"] = gt_info_df["% parcels"].map("{:.2%}".format)
        html_data["GROUND_TRUTH_TABLE"] = gt_info_df.to_html(index=False)

    # Write the report to a file
    # --------------------------
    with open(output_report_html, "w") as outputfile:
        script_dir = Path(__file__).resolve().parent
        html_template_path = script_dir / "html_report_class_template.html"
        html_template_file = open(html_template_path).read()
        src = Template(html_template_file)
        # replace strings and write to file
        output = src.substitute(html_data)
        outputfile.write(output)
