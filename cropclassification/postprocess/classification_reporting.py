"""
Module with some helper functions to report on the classification results.
"""

import logging
from pathlib import Path
from string import Template
from typing import Optional

import geofileops as gfo
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.postprocess import classification_postprocess as class_postpr

# Get a logger...
logger = logging.getLogger(__name__)

# TODO: improve reporting to divide between eligible versus ineligible classes?
# TODO?: report based on area instead of number parcel
#     -> seems like being a bit "detached from reality", as for RFV the most important
#        parameter is the number of parcels


def write_full_report(
    parcel_predictions_geopath: Path,
    parcel_train_path: Optional[Path],
    output_report_txt: Path,
    parcel_ground_truth_path: Optional[Path] = None,
    force: bool = False,
):
    """Writes a report about the accuracy of the predictions to a file.

    Args:
        parcel_predictions_geopath: File name of geofile with the parcels with their
            predictions.
        prediction_columnname: Column name of the column that contains the predictions.
        output_report_txt: File name of txt file the report will be written to.
        parcel_ground_truth_path: List of parcels with ground truth to calculate
            eg. alfa and beta errors. If None, the part of the report that is based on
            this data is skipped

    TODO: refactor function to split logic more...
    """

    # If force == False Check and the output file exists already, stop.
    output_report_html = Path(str(output_report_txt).replace(".txt", ".html"))
    if force is False and output_report_txt.exists() and output_report_html.exists():
        logger.warning(
            f"collect_and_prepare_timeseries_data: output files already exist and "
            f"force == False, so stop: {output_report_txt}"
        )
        return

    logger.info("Start write_full_report")

    pandas_option_context_list = [
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        300,
        "display.width",
        2000,
        "display.colheader_justify",
        "left",
    ]
    logger.info(f"Read file with predictions: {parcel_predictions_geopath}")
    df_predict = gfo.read_file(parcel_predictions_geopath)
    df_predict.set_index(conf.columns["id"], inplace=True)

    # Convert all columns to numeric, for the actual numeric ones this will stick.
    # For new files this isn't necessary anymore, but to be able to update reports for
    # old runs this needs to stay here.
    for column in df_predict.columns:
        try:
            df_predict[column] = pd.to_numeric(df_predict[column])
        except Exception:
            _ = None

    # Add column that shows if the parcel was used for training
    if parcel_train_path is not None:
        parcel_train_df = gfo.read_file(parcel_train_path, ignore_geometry=True)
        parcel_train_df.set_index(conf.columns["id"], inplace=True)
        df_predict["used_for_train"] = df_predict.index.isin(parcel_train_df.index)

    # Python template engine expects all values to be present, so initialize to empty
    empty_string = "''"
    html_data = {
        "GENERAL_ACCURACIES_TABLE": empty_string,
        "GENERAL_ACCURACIES_TEXT": empty_string,
        "GENERAL_ACCURACIES_DATA": empty_string,
        "CONFUSION_MATRICES_TABLE": empty_string,
        "CONFUSION_MATRICES_DATA": empty_string,
        "CONFUSION_MATRICES_CONSOLIDATED_TABLE": empty_string,
        "CONFUSION_MATRICES_CONSOLIDATED_DATA": empty_string,
        "PREDICTION_QUALITY_CONS_OVERVIEW_TEXT": empty_string,
        "PREDICTION_QUALITY_CONS_OVERVIEW_TABLE": empty_string,
        "PREDICTION_QUALITY_FULL_ALPHA_OVERVIEW_TEXT": empty_string,
        "PREDICTION_QUALITY_FULL_ALPHA_OVERVIEW_TABLE": empty_string,
        "PREDICTION_QUALITY_ALPHA_TEXT": empty_string,
        "PREDICTION_QUALITY_BETA_TEXT": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TEXT": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TABLE": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_CLASS_TEXT": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_CLASS_TABLE": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_CROP_TEXT": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_CROP_TABLE": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_PROBABILITY_TEXT": empty_string,
        "PREDICTION_QUALITY_ALPHA_PER_PROBABILITY_TABLE": empty_string,
        "PREDICTION_QUALITY_BETA_PER_PIXCOUNT_TEXT": empty_string,
        "PREDICTION_QUALITY_BETA_PER_PIXCOUNT_TABLE": empty_string,
        "PREDICTION_QUALITY_BETA_PER_CLASS_TEXT": empty_string,
        "PREDICTION_QUALITY_BETA_PER_CLASS_TABLE": empty_string,
        "PREDICTION_QUALITY_BETA_PER_CROP_TEXT": empty_string,
        "PREDICTION_QUALITY_BETA_PER_CROP_TABLE": empty_string,
        "PREDICTION_QUALITY_BETA_PER_PROBABILITY_TEXT": empty_string,
        "PREDICTION_QUALITY_BETA_PER_PROBABILITY_TABLE": empty_string,
    }

    # Build and write report...
    with open(output_report_txt, "w") as outputfile:
        outputfile.write("**************************************************\n")
        outputfile.write("**************** PARAMETERS USED *****************\n")
        outputfile.write("**************************************************\n\n")
        message = "Main parameters used for the marker"
        outputfile.write(f"\n{message}\n")
        html_data["PARAMETERS_USED_TEXT"] = message

        logger.info(f"{dict(conf.marker)}")
        parameter_list = [["marker", key, value] for key, value in conf.marker.items()]
        parameter_list += [
            ["calc_marker_params", key, value] for key, value in conf.timeseries.items()
        ]
        parameter_list += [
            ["timeseries", key, value] for key, value in conf.timeseries.items()
        ]
        parameter_list += [
            ["preprocess", key, value] for key, value in conf.preprocess.items()
        ]
        parameter_list += [
            ["classifier", key, value] for key, value in conf.classifier.items()
        ]
        parameter_list += [
            ["postprocess", key, value] for key, value in conf.postprocess.items()
        ]

        parameters_used_df = pd.DataFrame(
            parameter_list, columns=["parameter_type", "parameter", "value"]
        )
        with pd.option_context(*pandas_option_context_list):  # type: ignore[arg-type]
            outputfile.write(f"\n{parameters_used_df}\n")
            logger.info(f"{parameters_used_df}\n")
            html_data["PARAMETERS_USED_TABLE"] = parameters_used_df.to_html(index=False)

        outputfile.write("\n**************************************************\n")
        outputfile.write("*********** RECAP OF GENERAL RESULTS *************\n")
        outputfile.write("**************************************************\n\n")
        outputfile.write("**************************************************\n")
        outputfile.write("*        GENERAL CONSOLIDATED CONCLUSIONS        *\n")
        outputfile.write("**************************************************\n")
        # Calculate + write general conclusions for consolidated prediction
        _add_prediction_conclusion(
            in_df=df_predict,
            new_columnname=conf.columns["prediction_conclusion_cons"],
            prediction_column_to_use=conf.columns["prediction_full_alpha"],
            detailed=False,
        )

        # Get the number of 'unimportant' ignore parcels and report them here
        df_predict_unimportant = df_predict[
            df_predict[conf.columns["prediction_conclusion_cons"]]
            == "IGNORE_UNIMPORTANT"
        ]
        # Now they can be removed for the rest of the reportings...
        df_predict = df_predict[
            df_predict[conf.columns["prediction_conclusion_cons"]]
            != "IGNORE_UNIMPORTANT"
        ]

        message = (
            f"Prediction conclusions cons general overview, for {len(df_predict.index)}"
            f" predicted cases. The {len(df_predict_unimportant.index)} "
            "IGNORE_UNIMPORTANT parcels are excluded from the reporting!"
        )
        outputfile.write(f"\n{message}\n")
        html_data["GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_TEXT"] = message

        count_per_class = (
            df_predict.groupby(conf.columns["prediction_conclusion_cons"])
            .size()
            .to_frame("count")
        )
        values = 100 * count_per_class["count"] / count_per_class["count"].sum()
        count_per_class.insert(loc=1, column="pct", value=values)

        with pd.option_context(*pandas_option_context_list):  # type: ignore[arg-type]
            outputfile.write(f"\n{count_per_class}\n")
            logger.info(f"{count_per_class}\n")
            html_data["GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_TABLE"] = (
                count_per_class.to_html()
            )
            html_data["GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_DATA"] = (
                count_per_class.to_dict()
            )

        # Output general accuracies
        outputfile.write("\n*************************************************\n")
        outputfile.write("*              OVERALL ACCURACIES                *\n")
        outputfile.write("**************************************************\n")
        overall_accuracies_list = []

        # Calculate overall accuracies for all parcels
        try:
            oa = (
                skmetrics.accuracy_score(
                    df_predict[conf.columns["class"]],
                    df_predict["pred1"],
                    normalize=True,
                    sample_weight=None,
                )
                * 100
            )
            overall_accuracies_list.append(
                {"parcels": "All", "prediction_type": "standard", "accuracy": oa}
            )

            oa = (
                skmetrics.accuracy_score(
                    df_predict[conf.columns["class"]],
                    df_predict[conf.columns["prediction_full_alpha"]],
                    normalize=True,
                    sample_weight=None,
                )
                * 100
            )
            overall_accuracies_list.append(
                {"parcels": "All", "prediction_type": "consolidated", "accuracy": oa}
            )
        except Exception:
            logger.exception("Error calculating overall accuracies!")

        # Calculate while ignoring the classes to be ignored...
        df_predict_accuracy_no_ignore = df_predict[
            ~df_predict[conf.columns["class"]].isin(
                conf.marker.getlist("classes_to_ignore_for_train")
            )
        ]
        df_predict_accuracy_no_ignore = df_predict_accuracy_no_ignore[
            ~df_predict_accuracy_no_ignore[conf.columns["class"]].isin(
                conf.marker.getlist("classes_to_ignore")
            )
        ]

        oa = (
            skmetrics.accuracy_score(
                df_predict_accuracy_no_ignore[conf.columns["class"]],
                df_predict_accuracy_no_ignore["pred1"],
                normalize=True,
                sample_weight=None,
            )
            * 100
        )
        overall_accuracies_list.append(
            {
                "parcels": "Exclude classes_to_ignore(_for_train) classes",
                "prediction_type": "standard",
                "accuracy": oa,
            }
        )

        oa = (
            skmetrics.accuracy_score(
                df_predict_accuracy_no_ignore[conf.columns["class"]],
                df_predict_accuracy_no_ignore[conf.columns["prediction_full_alpha"]],
                normalize=True,
                sample_weight=None,
            )
            * 100
        )
        overall_accuracies_list.append(
            {
                "parcels": "Exclude classes_to_ignore(_for_train) classes",
                "prediction_type": "consolidated",
                "accuracy": oa,
            }
        )

        # Calculate ignoring both classes to ignored + parcels not having a valid
        # prediction
        df_predict_no_ignore_has_prediction = df_predict_accuracy_no_ignore.loc[
            (
                df_predict_accuracy_no_ignore[conf.columns["prediction_full_alpha"]]
                != "NODATA"
            )
            & (
                df_predict_accuracy_no_ignore[conf.columns["prediction_full_alpha"]]
                != "DOUBT:NOT_ENOUGH_PIXELS"
            )
        ]
        oa = (
            skmetrics.accuracy_score(
                df_predict_no_ignore_has_prediction[conf.columns["class"]],
                df_predict_no_ignore_has_prediction["pred1"],
                normalize=True,
                sample_weight=None,
            )
            * 100
        )
        overall_accuracies_list.append(
            {
                "parcels": (
                    "Exclude ignored ones + with prediction (= excl. NODATA, "
                    "NOT_ENOUGH_PIXELS)"
                ),
                "prediction_type": "standard",
                "accuracy": oa,
            }
        )

        oa = (
            skmetrics.accuracy_score(
                df_predict_no_ignore_has_prediction[conf.columns["class"]],
                df_predict_no_ignore_has_prediction[
                    conf.columns["prediction_full_alpha"]
                ],
                normalize=True,
                sample_weight=None,
            )
            * 100
        )
        overall_accuracies_list.append(
            {
                "parcels": (
                    "Exclude ignored ones + with prediction (= excl. NODATA, "
                    "NOT_ENOUGH_PIXELS)"
                ),
                "prediction_type": "consolidated",
                "accuracy": oa,
            }
        )

        # Output the resulting overall accuracies
        message = "Overall accuracies for different sub-groups of the data"
        outputfile.write(f"\n{message}\n")
        html_data["OVERALL_ACCURACIES_TEXT"] = message

        overall_accuracies_df = pd.DataFrame(
            overall_accuracies_list, columns=["parcels", "prediction_type", "accuracy"]
        )
        overall_accuracies_df.set_index(
            keys=["parcels", "prediction_type"], inplace=True
        )
        with pd.option_context(*pandas_option_context_list):  # type: ignore[arg-type]
            outputfile.write(f"\n{overall_accuracies_df}\n")
            logger.info(f"{overall_accuracies_df}\n")
            html_data["OVERALL_ACCURACIES_TABLE"] = overall_accuracies_df.to_html()

        # Write the recall, F1 score,... per class
        # message = skmetrics.classification_report(
        #     df_predict[gs.class_column],
        #     df_predict[gs.prediction_column],
        #     labels=classes
        # )
        # outputfile.write(message)

        outputfile.write("**************************************************\n")
        outputfile.write("**************** DETAILED RESULTS ****************\n")
        outputfile.write("**************************************************\n")
        outputfile.write("\n**************************************************\n")
        outputfile.write("*        DETAILED PREDICTION CONCLUSIONS         *\n")
        outputfile.write("**************************************************\n")

        # Calculate detailed conclusions for the predictions
        logger.info("Calculate the detailed conclusions for the predictions")

        # Write the conclusions for the consolidated predictions
        _add_prediction_conclusion(
            in_df=df_predict,
            new_columnname=conf.columns["prediction_conclusion_detail_cons"],
            prediction_column_to_use=conf.columns["prediction_cons"],
            detailed=True,
        )
        message = (
            "Prediction conclusions cons (doubt + not_enough_pixels) overview, for "
            f"{len(df_predict.index)} predicted cases:"
        )
        outputfile.write(f"\n{message}\n")
        html_data["PREDICTION_CONCLUSION_DETAIL_CONS_OVERVIEW_TEXT"] = message

        count_per_class = (
            df_predict.groupby(conf.columns["prediction_conclusion_detail_cons"])
            .size()
            .to_frame("count")
        )
        values = 100 * count_per_class["count"] / count_per_class["count"].sum()
        count_per_class.insert(loc=1, column="pct", value=values)

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            outputfile.write(f"\n{count_per_class}\n")
            logger.info(f"{count_per_class}\n")
            html_data["PREDICTION_CONCLUSION_DETAIL_CONS_OVERVIEW_TABLE"] = (
                count_per_class.to_html()
            )

        # Calculate detailed conclusions for the predictions
        logger.info("Calculate the detailed conclusions for the predictions")

        # Write the conclusions for the consolidated predictions
        _add_prediction_conclusion(
            in_df=df_predict,
            new_columnname=conf.columns["prediction_conclusion_detail_full_alpha"],
            prediction_column_to_use=conf.columns["prediction_full_alpha"],
            detailed=True,
        )
        message = (
            "Prediction conclusions full alpha (doubt + not_enough_pixels) overview, "
            f"for {len(df_predict.index)} predicted cases:"
        )
        outputfile.write(f"\n{message}\n")
        html_data["PREDICTION_CONCLUSION_DETAIL_FULL_ALPHA_OVERVIEW_TEXT"] = message

        count_per_class = (
            df_predict.groupby(conf.columns["prediction_conclusion_detail_full_alpha"])
            .size()
            .to_frame("count")
        )
        values = 100 * count_per_class["count"] / count_per_class["count"].sum()
        count_per_class.insert(loc=1, column="pct", value=values)

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            outputfile.write(f"\n{count_per_class}\n")
            logger.info(f"{count_per_class}\n")
            html_data["PREDICTION_CONCLUSION_DETAIL_FULL_ALPHA_OVERVIEW_TABLE"] = (
                count_per_class.to_html()
            )

        outputfile.write("\n**************************************************\n")
        outputfile.write("* CONFUSION MATRICES FOR PARCELS WITH PREDICTIONS*\n")
        outputfile.write("**************************************************\n")
        # Calculate an extended confusion matrix with the standard prediction column
        # and write it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, "pred1")
        outputfile.write(
            "\nExtended confusion matrix of the predictions: Rows: true/input classes, "
            "columns: predicted classes\n"
        )
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", 2000
        ):
            outputfile.write(f"{df_confmatrix_ext}\n")
            html_data["CONFUSION_MATRICES_TABLE"] = df_confmatrix_ext.to_html()
            html_data["CONFUSION_MATRICES_DATA"] = df_confmatrix_ext.to_json()

        # Calculate an extended confusion matrix with the full alpha prediction column
        # and write it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(
            df_predict, conf.columns["prediction_full_alpha"]
        )
        outputfile.write(
            "\nExtended confusion matrix of the consolidated predictions: "
            "Rows: true/input classes, columns: predicted classes\n"
        )
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", 2000
        ):
            outputfile.write(f"{df_confmatrix_ext}\n\n")
            html_data["CONFUSION_MATRICES_CONSOLIDATED_TABLE"] = (
                df_confmatrix_ext.to_html()
            )
            html_data["CONFUSION_MATRICES_CONSOLIDATED_DATA"] = (
                df_confmatrix_ext.to_json()
            )

        # If the pixcount is available, write the OA per pixcount
        if conf.columns["pixcount_s1s2"] in df_predict.columns:
            pixcount_output_report_txt = Path(
                str(output_report_txt) + "_OA_per_pixcount.txt"
            )
            _write_OA_per_pixcount(
                df_parcel_predictions=df_predict,
                output_report_txt=pixcount_output_report_txt,
                force=force,
            )

        # If a ground truth file is provided, report on the ground truth
        if parcel_ground_truth_path is not None:
            outputfile.write("**************************************************\n")
            outputfile.write("*  REPORT PREDICT QUALITY BASED ON GROUND TRUTH  *\n")
            outputfile.write("**************************************************\n")

            # Read ground truth
            logger.info(
                "Read csv with ground truth (with their classes): "
                f"{parcel_ground_truth_path}"
            )
            df_parcel_gt = pdh.read_file(parcel_ground_truth_path)
            df_parcel_gt.set_index(conf.columns["id"], inplace=True)
            logger.info(
                f"Read csv with ground truth ready, shape: {df_parcel_gt.shape}"
            )

            # Join the prediction data
            cols_to_join = df_predict.columns.difference(df_parcel_gt.columns)
            df_parcel_gt = df_predict[cols_to_join].join(df_parcel_gt, how="inner")
            logger.info(
                "After join of ground truth with predictions, shape: "
                f"{df_parcel_gt.shape}"
            )

            if len(df_parcel_gt.index) == 0:
                message = (
                    "After join of ground truth with predictions the result was empty, "
                    "so probably a wrong ground truth file was used!"
                )
                logger.critical(message)
                raise Exception(message)

            # General ground truth statistics
            # -------------------------------
            # Calculate the conclusions based on ground truth

            # Calculate and write the result for the consolidated predictions
            _add_gt_conclusions(df_parcel_gt, conf.columns["prediction_cons"])
            message = (
                "Prediction quality cons (doubt + not_enough_pixels) overview, for "
                f"{len(df_parcel_gt.index)} predicted cases in ground truth:"
            )
            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_CONS_OVERVIEW_TEXT"] = message

            count_per_class = (
                df_parcel_gt.groupby(f"gt_conclusion_{conf.columns['prediction_cons']}")
                .size()
                .to_frame("count")  # type: ignore[operator]
            )
            values = 100 * count_per_class["count"] / count_per_class["count"].sum()
            count_per_class.insert(loc=1, column="pct", value=values)

            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
            ):
                outputfile.write(f"\n{count_per_class}\n")
                logger.info(f"{count_per_class}\n")
                html_data["PREDICTION_QUALITY_CONS_OVERVIEW_TABLE"] = (
                    count_per_class.to_html()
                )

            # Calculate and write the result for the full alpha predictions
            _add_gt_conclusions(df_parcel_gt, conf.columns["prediction_full_alpha"])
            message = (
                "Prediction quality cons (doubt + not_enough_pixels) overview, for "
                f"{len(df_parcel_gt.index)} predicted cases in ground truth:"
            )
            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_FULL_ALPHA_OVERVIEW_TEXT"] = message

            count_per_class = (
                df_parcel_gt.groupby(
                    f"gt_conclusion_{conf.columns['prediction_full_alpha']}"
                )
                .size()
                .to_frame("count")  # type: ignore[operator]
            )
            values = 100 * count_per_class["count"] / count_per_class["count"].sum()
            count_per_class.insert(loc=1, column="pct", value=values)

            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
            ):
                outputfile.write(f"\n{count_per_class}\n")
                logger.info(f"{count_per_class}\n")
                html_data["PREDICTION_QUALITY_FULL_ALPHA_OVERVIEW_TABLE"] = (
                    count_per_class.to_html()
                )

            # Write the ground truth conclusions to files
            pdh.to_file(
                df_parcel_gt,
                Path(str(output_report_txt) + "_groundtruth_pred_quality_details.tsv"),
            )
            output_path = Path(
                str(output_report_txt) + "_groundtruth_pred_quality_details.gpkg"
            )
            gfo.to_file(gdf=df_parcel_gt, path=output_path)

            # Alpha and beta error statistics based on CONS prediction
            # ******************************************************************
            # Pct Alpha errors=nb incorrect errors/total nb errors
            columnname = f"gt_conclusion_{conf.columns['prediction_cons']}"
            alpha_numerator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname] == "FARMER-CORRECT_PRED-WRONG:ERROR_ALPHA"
                ].index
            )
            alpha_denominator = alpha_numerator + len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].isin(
                        ["FARMER-WRONG_PRED-CORRECT", "FARMER-WRONG_PRED-WRONG"]
                    )
                ].index
            )
            if alpha_denominator > 0:
                message = (
                    f"Alpha error for cons: {alpha_numerator}/{alpha_denominator} = "
                    f"{(alpha_numerator/alpha_denominator):.04f}"
                )
            else:
                message = (
                    f"Alpha error for cons: {alpha_numerator}/{alpha_denominator} = ?"
                )

            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_ALPHA_TEXT"] = message

            # Pct BETA errors=nb incorrect judged OK/total nb judged OK
            beta_numerator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname]
                    == "FARMER-WRONG_PRED-DOESNT_OPPOSE:ERROR_BETA"
                ].index
            )
            beta_denominator = beta_numerator + len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].str.startswith("FARMER-CORRECT")
                ].index
            )
            message = "Beta error for cons (=Beta/(Beta + all FARMER-CORRECT)): "
            if beta_denominator > 0:
                message += (
                    f"{beta_numerator}/{beta_denominator} = "
                    f"{(beta_numerator/beta_denominator):.04f}"
                )
            else:
                message += f"{beta_numerator}/{beta_denominator} = ?"

            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_BETA_TEXT"] = message

            # Alpha and beta error statistics based on FULL ALPHA prediction
            # ******************************************************************
            # Pct ALPHA errors=alpha errors/(alpha errors + real errors)
            alpha_numerator_conclusions = ["FARMER-CORRECT_PRED-WRONG:ERROR_ALPHA"]
            alpha_denominator_conclusions = [
                "FARMER-WRONG_PRED-CORRECT",
                "FARMER-WRONG_PRED-WRONG",
            ] + alpha_numerator_conclusions
            columnname = f"gt_conclusion_{conf.columns['prediction_full_alpha']}"
            alpha_numerator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].isin(alpha_numerator_conclusions)
                ].index
            )
            alpha_denominator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].isin(alpha_denominator_conclusions)
                ].index
            )
            if alpha_denominator > 0:
                message = (
                    f"Alpha error full: {alpha_numerator}/{alpha_denominator} = "
                    f"{(alpha_numerator/alpha_denominator):.04f}"
                )
            else:
                message = f"Alpha error full: {alpha_numerator}/{alpha_denominator} = ?"

            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_ALPHA_TEXT"] += "<br/>" + message

            # Pct BETA errors=beta errors/(beta errors + correct farmer declarations)
            beta_numerator_conclusions = ["FARMER-WRONG_PRED-DOESNT_OPPOSE:ERROR_BETA"]
            beta_denominator_conclusions = (
                df_parcel_gt[columnname]
                .loc[df_parcel_gt[columnname].str.startswith("FARMER-CORRECT")]
                .unique()
                .tolist()
                + beta_numerator_conclusions
            )
            beta_numerator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].isin(beta_numerator_conclusions)
                ].index
            )
            beta_denominator = len(
                df_parcel_gt.loc[
                    df_parcel_gt[columnname].isin(beta_denominator_conclusions)
                ].index
            )
            message = "Beta error full (=Beta/(Beta + all FARMER-CORRECT)): "
            if beta_denominator > 0:
                message += (
                    f"{beta_numerator}/{beta_denominator} = "
                    f"{(beta_numerator/beta_denominator):.04f}"
                )
            else:
                message += f"{beta_numerator}/{beta_denominator} = ?"

            outputfile.write(f"\n{message}\n")
            html_data["PREDICTION_QUALITY_BETA_TEXT"] += "<br/>" + message

            # Some more detailed reports for alpha and beta errors
            # ******************************************************************

            # If the pixcount is available, write the number of ALFA errors per
            # pixcount (for the prediction with doubt)
            pred_quality_full_doubt_column = (
                f"gt_conclusion_{conf.columns['prediction_full_alpha']}"
            )
            if conf.columns["pixcount_s1s2"] in df_parcel_gt.columns:
                # Convert pixcounts to bins
                pixcount_bins_column = f"{conf.columns['pixcount_s1s2']}_bins"
                bins = np.array(
                    [0, 1, 5, 10, 20, 30, 40, 50, 100, 200, 1000, 5000, 10000]
                )
                df_parcel_gt[pixcount_bins_column] = pd.cut(
                    x=df_parcel_gt[conf.columns["pixcount_s1s2"]],
                    bins=bins,
                    include_lowest=True,
                    right=True,
                )  # type: ignore[call-overload]

                # ALPHA errors
                message = (
                    "Number of ERROR_ALFA parcels per pixcount for the ground truth "
                    "parcels without applying doubt based on pixcount:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TEXT"] = message

                # For pixcount report, use error conclusions without min_nb_pixels
                class_postpr.add_doubt_column(
                    pred_df=df_parcel_gt,
                    new_pred_column="pred_cons_no_min_pix",
                    apply_doubt_pct_proba=True,
                    apply_doubt_min_nb_pixels=False,
                    apply_doubt_marker_specific=True,
                )
                _add_gt_conclusions(df_parcel_gt, "pred_cons_no_min_pix")

                # Calc data and write
                pred_quality_column = "gt_conclusion_" + "pred_cons_no_min_pix"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=pixcount_bins_column,
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=alpha_numerator_conclusions,
                    error_codes_denominator=alpha_denominator_conclusions,
                    error_type="alpha",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TABLE"] = (
                        df_per_column.to_html()
                    )

                # BETA errors
                message = (
                    "Number of ERROR_BETA parcels per pixcount for the ground truth "
                    "parcels without applying doubt based on pixcount:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_BETA_PER_PIXCOUNT_TEXT"] = message

                # Calc data and write
                pred_quality_column = "gt_conclusion_" + "pred_cons_no_min_pix"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=pixcount_bins_column,
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=beta_numerator_conclusions,
                    error_codes_denominator=beta_denominator_conclusions,
                    error_type="beta",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_BETA_PER_PIXCOUNT_TABLE"] = (
                        df_per_column.to_html()
                    )

            # If cropclass is available, write the number of ALFA errors per cropclass
            # (for the prediction with doubt)
            if conf.columns["class_declared"] in df_parcel_gt.columns:
                # ALPHA errors
                message = (
                    "Number of ERROR_ALFA parcels per declared cropclass for the "
                    "ground truth parcels without applying crop/class based doubt:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_ALPHA_PER_CLASS_TEXT"] = message

                # For class report, use error conclusions without marker specific stuff
                class_postpr.add_doubt_column(
                    pred_df=df_parcel_gt,
                    new_pred_column="pred_cons_no_marker_specific",
                    apply_doubt_pct_proba=True,
                    apply_doubt_min_nb_pixels=True,
                    apply_doubt_marker_specific=False,
                )
                _add_gt_conclusions(df_parcel_gt, "pred_cons_no_marker_specific")

                # Calc data and write
                pred_quality_column = "gt_conclusion_pred_cons_no_marker_specific"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=conf.columns["class_declared"],
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=alpha_numerator_conclusions,
                    error_codes_denominator=alpha_denominator_conclusions,
                    error_type="alpha",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_ALPHA_PER_CLASS_TABLE"] = (
                        df_per_column.to_html()
                    )

                # BETA errors
                message = (
                    "Number of ERROR_BETA parcels per declared cropclass for the "
                    "ground truth parcels without applying crop/class based doubt:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_BETA_PER_CLASS_TEXT"] = message

                # Calc data and write
                pred_quality_column = "gt_conclusion_pred_cons_no_marker_specific"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=conf.columns["class_declared"],
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=beta_numerator_conclusions,
                    error_codes_denominator=beta_denominator_conclusions,
                    error_type="beta",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_BETA_PER_CLASS_TABLE"] = (
                        df_per_column.to_html()
                    )

            # If crop is available, write the number of ALFA errors per cropclass
            # (for the prediction with doubt)
            if conf.columns["crop_declared"] in df_parcel_gt.columns:
                # ALPHA errors
                message = (
                    "Number of ERROR_ALPHA parcels per declared crop for the ground "
                    "truth parcels, without applying marker specific doubt:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_ALPHA_PER_CROP_TEXT"] = message

                # For crop report, use error conclusions without marker specific stuff
                class_postpr.add_doubt_column(
                    pred_df=df_parcel_gt,
                    new_pred_column="pred_cons_no_marker_specific",
                    apply_doubt_pct_proba=True,
                    apply_doubt_min_nb_pixels=True,
                    apply_doubt_marker_specific=False,
                )
                _add_gt_conclusions(df_parcel_gt, "pred_cons_no_marker_specific")

                # Calc data and write
                pred_quality_column = "gt_conclusion_pred_cons_no_marker_specific"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=conf.columns["crop_declared"],
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=alpha_numerator_conclusions,
                    error_codes_denominator=alpha_denominator_conclusions,
                    error_type="alpha",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_ALPHA_PER_CROP_TABLE"] = (
                        df_per_column.to_html()
                    )

                # BETA errors
                message = (
                    "Number of ERROR_BETA parcels per declared crop for the ground "
                    "truth parcels, without applying marker specific doubt:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_BETA_PER_CROP_TEXT"] = message

                # Calc data and write
                pred_quality_column = "gt_conclusion_pred_cons_no_marker_specific"
                df_per_column = _get_errors_per_column(
                    groupbycolumn=conf.columns["crop_declared"],
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=beta_numerator_conclusions,
                    error_codes_denominator=beta_denominator_conclusions,
                    error_type="beta",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_BETA_PER_CROP_TABLE"] = (
                        df_per_column.to_html()
                    )

            # If probability is available, write the number of ALFA errors per
            # probability (for the prediction with doubt)
            if "pred1_prob" in df_parcel_gt.columns:
                # ALPHA errors
                message = (
                    "Number of ERROR_ALFA parcels per % probability for the ground "
                    "truth parcels, without doubt based on probability:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_ALPHA_PER_PROBABILITY_TEXT"] = message

                # For pixcount report, use error conclusions without doubt reasons
                # that use the pct probability + round the probabilities
                class_postpr.add_doubt_column(
                    pred_df=df_parcel_gt,
                    new_pred_column="pred_cons_no_pct_prob",
                    apply_doubt_pct_proba=False,
                    apply_doubt_min_nb_pixels=True,
                    apply_doubt_marker_specific=True,
                )
                _add_gt_conclusions(df_parcel_gt, "pred_cons_no_pct_prob")
                df_parcel_gt["pred1_prob_rounded"] = df_parcel_gt["pred1_prob"].round(2)

                # Calc data and write
                pred_quality_column = "gt_conclusion_" + "pred_cons_no_pct_prob"
                df_per_column = _get_errors_per_column(
                    groupbycolumn="pred1_prob_rounded",
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=alpha_numerator_conclusions,
                    error_codes_denominator=alpha_denominator_conclusions,
                    ascending=False,
                    error_type="alpha",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_ALPHA_PER_PROBABILITY_TABLE"] = (
                        df_per_column.to_html()
                    )

                # BETA errors
                message = (
                    "Number of ERROR_BETA parcels per % probability for the ground "
                    "truth parcels, without doubt based on probability:"
                )
                outputfile.write(f"\n{message}\n")
                html_data["PREDICTION_QUALITY_BETA_PER_PROBABILITY_TEXT"] = message

                # Calc data and write
                pred_quality_column = "gt_conclusion_" + "pred_cons_no_pct_prob"
                df_per_column = _get_errors_per_column(
                    groupbycolumn="pred1_prob_rounded",
                    df_predquality=df_parcel_gt,
                    pred_quality_column=pred_quality_column,
                    pred_quality_full_doubt_column=pred_quality_full_doubt_column,
                    error_codes_numerator=beta_numerator_conclusions,
                    error_codes_denominator=beta_denominator_conclusions,
                    ascending=False,
                    error_type="beta",
                )
                # df_per_column.dropna(inplace=True)
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    2000,
                ):
                    outputfile.write(f"\n{df_per_column}\n")
                    logger.info(f"{df_per_column}\n")
                    html_data["PREDICTION_QUALITY_BETA_PER_PROBABILITY_TABLE"] = (
                        df_per_column.to_html()
                    )

    with open(output_report_html, "w") as outputfile:
        script_dir = Path(__file__).resolve().parent
        html_template_path = script_dir / "html_rapport_template.html"
        html_template_file = open(html_template_path).read()
        src = Template(html_template_file)
        # replace strings and write to file
        output = src.substitute(html_data)
        outputfile.write(output)


def _get_confusion_matrix_ext(df_predict, prediction_column_to_use: str):
    """Returns a dataset with an extended confusion matrix."""

    classes = sorted(
        np.unique(
            np.append(
                df_predict[prediction_column_to_use], df_predict[conf.columns["class"]]
            )
        )
    )
    logger.debug(f"Input shape: {df_predict.shape}, Unique classes found: {classes}")

    # Calculate standard confusion matrix
    np_confmatrix = skmetrics.confusion_matrix(
        df_predict[conf.columns["class"]],
        df_predict[prediction_column_to_use],
        labels=classes,
    )
    df_confmatrix_ext = pd.DataFrame(np_confmatrix, classes, classes)

    # Add some more columns to the confusion matrix
    # Insert column with the total nb of parcel for each class that were the input
    # values (=sum of the row of each class)
    values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=1)
    df_confmatrix_ext.insert(loc=0, column="nb_input", value=values)
    # Insert column with the total nb of parcel for each class that were predicted to
    # have this value (=sum of the column of each class)
    values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=0)
    df_confmatrix_ext.insert(loc=1, column="nb_predicted", value=values)
    # Insert column with the total nb of correctly predicted classes
    df_confmatrix_ext.insert(loc=2, column="nb_predicted_correct", value=0)
    for column in df_confmatrix_ext.columns:
        if column not in ["nb_input", "nb_predicted", "nb_predicted_correct"]:
            df_confmatrix_ext.at[column, "nb_predicted_correct"] = df_confmatrix_ext.at[
                column, column
            ]
    # Insert column with the total nb of parcel for each class that were predicted to
    # have this value (=sum of the column of each class)
    values = (
        df_confmatrix_ext["nb_predicted"] - df_confmatrix_ext["nb_predicted_correct"]
    )
    df_confmatrix_ext.insert(loc=3, column="nb_predicted_wrong", value=values)
    # Insert columns with percentages
    values = (
        df_confmatrix_ext["nb_predicted_correct"] * 100 / df_confmatrix_ext["nb_input"]
    )
    df_confmatrix_ext.insert(loc=4, column="pct_predicted_correct", value=values)
    values = (
        (df_confmatrix_ext["nb_predicted"] - df_confmatrix_ext["nb_predicted_correct"])
        * 100
        / df_confmatrix_ext["nb_input"]
    )
    df_confmatrix_ext.insert(loc=5, column="pct_predicted_wrong", value=values)

    return df_confmatrix_ext


def _add_prediction_conclusion(
    in_df, new_columnname, prediction_column_to_use, detailed: bool
):
    """
    Calculate the "conclusions" for the predictions

    REMARK: calculating it like this, using native pandas operations, is 300 times
            faster than using DataFrame.apply() with a function!!!
    """
    # Get a lists of the classes to ignore
    all_classes_to_ignore = conf.marker.getlist(
        "classes_to_ignore_for_train"
    ) + conf.marker.getlist("classes_to_ignore")
    classes_to_ignore_unimportant = conf.marker.getlist("classes_to_ignore_unimportant")

    # Add the new column with a fixed value first
    in_df[new_columnname] = "UNDEFINED"

    # Some conclusions are different if detailed info is asked...
    if detailed:
        # The classes that are defined as unimportant
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (
                in_df[conf.columns["class_declared"]].isin(
                    classes_to_ignore_unimportant
                )
            ),
            new_columnname,
        ] = "IGNORE_UNIMPORTANT:INPUTCLASSNAME=" + in_df[conf.columns["class"]]

        # Parcels that were ignored for trainig and/or prediction, get an ignore
        # conclusion
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[conf.columns["class_declared"]].isin(all_classes_to_ignore)),
            new_columnname,
        ] = "IGNORE:INPUTCLASSNAME=" + in_df[conf.columns["class"]]
        # If conclusion still UNDEFINED, check if doubt
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[prediction_column_to_use].str.startswith("DOUBT")),
            new_columnname,
        ] = "DOUBT:REASON=" + in_df[prediction_column_to_use]
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[prediction_column_to_use] == "NODATA"),
            new_columnname,
        ] = "DOUBT:REASON=" + in_df[prediction_column_to_use]
    else:
        # The classes that are defined as unimportant
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (
                in_df[conf.columns["class_declared"]].isin(
                    classes_to_ignore_unimportant
                )
            ),
            new_columnname,
        ] = "IGNORE_UNIMPORTANT"
        # Parcels that were ignored for trainig and/or prediction, get an ignore
        # conclusion
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[conf.columns["class_declared"]].isin(all_classes_to_ignore)),
            new_columnname,
        ] = "IGNORE"
        # If conclusion still UNDEFINED, check if doubt
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[prediction_column_to_use].str.startswith("DOUBT")),
            new_columnname,
        ] = "DOUBT"
        # If conclusion still UNDEFINED, check if doubt
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[prediction_column_to_use].str.startswith("RISKY_DOUBT")),
            new_columnname,
        ] = "RISKY_DOUBT"
        in_df.loc[
            (in_df[new_columnname] == "UNDEFINED")
            & (in_df[prediction_column_to_use] == "NODATA"),
            new_columnname,
        ] = "DOUBT"

    # If conclusion still UNDEFINED, check if prediction equals the input class
    in_df.loc[
        (in_df[new_columnname] == "UNDEFINED")
        & (in_df[conf.columns["class_declared"]] == in_df[prediction_column_to_use]),
        new_columnname,
    ] = "OK:PREDICTION=INPUT_CLASS"
    # If conclusion still UNDEFINED, prediction is different from input
    in_df.loc[in_df[new_columnname] == "UNDEFINED", new_columnname] = (
        "NOK:PREDICTION<>INPUT_CLASS"
    )


def _add_gt_conclusions(in_df, prediction_column_to_use):
    """Add some columns with groundtruth conclusions."""

    # Add the new column with a fixed value first
    gt_vs_declared_column = f"gt_vs_input_{prediction_column_to_use}"
    gt_vs_prediction_column = f"gt_vs_prediction_{prediction_column_to_use}"
    gt_conclusion_column = f"gt_conclusion_{prediction_column_to_use}"
    all_classes_to_ignore = conf.marker.getlist(
        "classes_to_ignore_for_train"
    ) + conf.marker.getlist("classes_to_ignore")

    # Calculate gt_vs_input_column
    # ----------------------------
    # If ground truth same as input class, farmer OK, unless it is an ignore class
    in_df[gt_vs_declared_column] = "UNDEFINED"
    in_df.loc[
        (in_df[gt_vs_declared_column] == "UNDEFINED")
        & (
            in_df[conf.columns["class_declared"]]
            == in_df[conf.columns["class_groundtruth"]]
        )
        & (in_df[conf.columns["class_groundtruth"]].isin(all_classes_to_ignore)),
        gt_vs_declared_column,
    ] = (
        "FARMER-CORRECT:IGNORE:DECLARED=GROUNDTRUTH="
        + in_df[conf.columns["class_groundtruth"]]
    )
    in_df.loc[
        (in_df[gt_vs_declared_column] == "UNDEFINED")
        & (
            in_df[conf.columns["class_declared"]]
            == in_df[conf.columns["class_groundtruth"]]
        ),
        gt_vs_declared_column,
    ] = "FARMER-CORRECT"
    if conf.columns["class_declared2"] in in_df.columns:
        in_df.loc[
            (in_df[gt_vs_declared_column] == "UNDEFINED")
            & (
                in_df[conf.columns["class_declared2"]]
                == in_df[conf.columns["class_groundtruth"]]
            ),
            gt_vs_declared_column,
        ] = "FARMER-CORRECT"

    """
    in_df.loc[
        (in_df[gt_vs_declared_column] == "UNDEFINED")
        & (in_df[conf.columns["class_declared"]].isin(all_classes_to_ignore)),
        gt_vs_declared_column,
    ] = "FARMER-WRONG:IGNORE:DECLARED=" + in_df[conf.columns["class_declared"]]
    in_df.loc[
        (in_df[gt_vs_declared_column] == "UNDEFINED")
        & (in_df[conf.columns["class_groundtruth"]].isin(all_classes_to_ignore)),
        gt_vs_declared_column,
    ] = "FARMER-WRONG:IGNORE:GROUNDTRUTH=" + in_df[
        conf.columns["class_groundtruth"]
    ]
    """

    # If conclusion still UNDEFINED, farmer was simply wrong
    in_df.loc[in_df[gt_vs_declared_column] == "UNDEFINED", gt_vs_declared_column] = (
        "FARMER-WRONG"
    )

    # Calculate gt_vs_prediction_column
    # ---------------------------------
    # If ground truth same as prediction but it is IGNORED, correct but ignore
    in_df[gt_vs_prediction_column] = "UNDEFINED"
    in_df.loc[
        (in_df[prediction_column_to_use] == in_df[conf.columns["class_groundtruth"]])
        & (in_df[prediction_column_to_use].isin(all_classes_to_ignore)),
        gt_vs_prediction_column,
    ] = "PRED-CORRECT:IGNORE:PREDICTION=GROUNDTRUTH=" + in_df[prediction_column_to_use]
    # If not set yet and ground truth same as prediction, prediction OK
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[prediction_column_to_use] == in_df[conf.columns["class_groundtruth"]]),
        gt_vs_prediction_column,
    ] = "PRED-CORRECT"
    # If there is a second declared crop, also compare the prediction with the declared
    # crops in addition to the groundtruth
    if conf.columns["class_declared2"] in in_df.columns:
        in_df.loc[
            (in_df[gt_vs_prediction_column] == "UNDEFINED")
            & (
                in_df[prediction_column_to_use]
                == in_df[conf.columns["class_declared2"]]
            ),
            gt_vs_prediction_column,
        ] = "PRED-CORRECT"
        in_df.loc[
            (in_df[gt_vs_prediction_column] == "UNDEFINED")
            & (
                in_df[prediction_column_to_use] == in_df[conf.columns["class_declared"]]
            ),
            gt_vs_prediction_column,
        ] = "PRED-CORRECT"

    # If declared class in ignored + DOUBT: doubt conclusion
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[conf.columns["class_declared"]].isin(all_classes_to_ignore))
        & (in_df[prediction_column_to_use].str.startswith("DOUBT")),
        gt_vs_prediction_column,
    ] = (
        "PRED-DOUBT:REASON="
        + in_df[prediction_column_to_use]
        + ":DECLARED="
        + in_df[conf.columns["class_declared"]]
    )

    # If conclusion still UNDEFINED, check if DOUBT
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[prediction_column_to_use].str.startswith("DOUBT")),
        gt_vs_prediction_column,
    ] = "PRED-DOUBT:REASON=" + in_df[prediction_column_to_use]
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[prediction_column_to_use] == "NODATA"),
        gt_vs_prediction_column,
    ] = "PRED-DOUBT:REASON=" + in_df[prediction_column_to_use]

    # If conclusion still UNDEFINED, check if RISKY_DOUBT
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[prediction_column_to_use].str.startswith("RISKY_DOUBT")),
        gt_vs_prediction_column,
    ] = "PRED-RISKY_DOUBT:REASON=" + in_df[prediction_column_to_use]
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[prediction_column_to_use] == "NODATA"),
        gt_vs_prediction_column,
    ] = "PRED-RISKY_DOUBT:REASON=" + in_df[prediction_column_to_use]

    # If groundtruth class in ignored for trainig and/or prediction: an ignore
    # conclusion
    in_df.loc[
        (in_df[gt_vs_prediction_column] == "UNDEFINED")
        & (in_df[conf.columns["class_groundtruth"]].isin(all_classes_to_ignore)),
        gt_vs_prediction_column,
    ] = "PRED-WRONG:IGNORE:GROUNDTRUTH=" + in_df[conf.columns["class_groundtruth"]]

    # If conclusion still UNDEFINED, it was wrong
    in_df.loc[
        in_df[gt_vs_prediction_column] == "UNDEFINED", gt_vs_prediction_column
    ] = "PRED-WRONG"

    # Calculate gt_conclusion_column
    # ------------------------------
    # Declared class was correct
    in_df[gt_conclusion_column] = "UNDEFINED"
    in_df.loc[
        (in_df[gt_vs_declared_column] == "FARMER-CORRECT")
        & (in_df[gt_vs_prediction_column] == "PRED-WRONG"),
        gt_conclusion_column,
    ] = "FARMER-CORRECT_PRED-WRONG:ERROR_ALPHA"
    in_df.loc[
        (in_df[gt_conclusion_column] == "UNDEFINED")
        & (in_df[gt_vs_declared_column] == "FARMER-CORRECT"),
        gt_conclusion_column,
    ] = "FARMER-CORRECT_" + in_df[gt_vs_prediction_column]

    # Declared class was not correct
    in_df.loc[
        (in_df[gt_conclusion_column] == "UNDEFINED")
        & (in_df[gt_vs_declared_column] == "FARMER-WRONG")
        & (in_df[conf.columns["class_declared"]] == in_df[prediction_column_to_use]),
        gt_conclusion_column,
    ] = "FARMER-WRONG_PRED-DOESNT_OPPOSE:ERROR_BETA"
    in_df.loc[
        (in_df[gt_conclusion_column] == "UNDEFINED")
        & (in_df[gt_vs_declared_column] == "FARMER-WRONG"),
        gt_conclusion_column,
    ] = "FARMER-WRONG_" + in_df[gt_vs_prediction_column]

    # Declared or groundtruth class was ignore
    in_df.loc[(in_df[gt_conclusion_column] == "UNDEFINED"), gt_conclusion_column] = (
        in_df[gt_vs_declared_column]
    )


def _get_errors_per_column(
    groupbycolumn: str,
    df_predquality,
    pred_quality_column: str,
    pred_quality_full_doubt_column: str,
    error_codes_numerator: list[str],
    error_codes_denominator: list[str],
    error_type: str,
    ascending: bool = True,
):
    """
    Returns a dataset with detailed information about the number of alfa errors per
    column that was passed on
    """
    # First filter on the parcels we need to calculate the pct alpha errors
    df_predquality_filtered = df_predquality[
        df_predquality[pred_quality_column].isin(error_codes_denominator)
    ]

    # Calculate the number of parcels per groupbycolumn, the cumulative sum +
    # the pct of all
    df_predquality_count = (
        df_predquality_filtered.groupby(groupbycolumn).size().to_frame("count_all")
    )
    df_predquality_count.sort_index(ascending=ascending, inplace=True)
    values = df_predquality_count["count_all"].cumsum(axis=0)
    df_predquality_count.insert(
        loc=len(df_predquality_count.columns),
        column="count_all_cumulative",
        value=values,
    )
    values = (
        100
        * df_predquality_count["count_all_cumulative"]
        / df_predquality_count["count_all"].sum()
    )
    df_predquality_count.insert(
        loc=len(df_predquality_count.columns), column="pct_all_cumulative", value=values
    )

    # Now calculate the number of alfa errors per groupbycolumn
    df_alfa_error = df_predquality_filtered[
        df_predquality_filtered[pred_quality_column].isin(error_codes_numerator)
    ]
    df_alfa_per_column = (
        df_alfa_error.groupby(groupbycolumn)
        .size()
        .to_frame(f"count_error_{error_type}")
    )
    df_alfa_per_column.sort_index(ascending=ascending, inplace=True)

    # Now calculate the number of alfa errors with full doubt per groupbycolumn
    df_alfa_error_full_doubt = df_predquality_filtered[
        df_predquality_filtered[pred_quality_full_doubt_column].isin(
            error_codes_numerator
        )
    ]
    df_alfa_full_doubt_per_column = (
        df_alfa_error_full_doubt.groupby(groupbycolumn)
        .size()
        .to_frame(f"count_error_{error_type}_full_doubt")
    )

    # Join everything together
    df_alfa_per_column = df_predquality_count.join(df_alfa_per_column, how="left")
    df_alfa_per_column = df_alfa_per_column.join(
        df_alfa_full_doubt_per_column, how="left"
    )

    # Calculate the total number of parcels with full doubt applied per groupbycolumn
    df_predquality_full_doubt_filtered = df_predquality[
        df_predquality[pred_quality_full_doubt_column].isin(error_codes_denominator)
    ]

    values = (
        df_predquality_full_doubt_filtered.groupby(groupbycolumn)
        .size()
        .to_frame("count_all_full_doubt")
    )
    df_alfa_per_column = pd.concat([df_alfa_per_column, values], axis=1)

    # Finally calculate all alfa error percentages
    values = (
        df_alfa_per_column[f"count_error_{error_type}"]
        .cumsum(axis=0)
        .to_frame(f"count_error_{error_type}_cumulative")
    )
    df_alfa_per_column = pd.concat([df_alfa_per_column, values], axis=1)

    values = (
        100
        * df_alfa_per_column[f"count_error_{error_type}"]
        / df_alfa_per_column["count_all"]
    ).to_frame(f"pct_error_{error_type}_of_all")
    # df_alfa_per_column.insert(
    #     loc=len(df_alfa_per_column.columns),
    #     column=f"pct_error_{error_type}_of_all",
    #     value=values,
    # )
    df_alfa_per_column = pd.concat([df_alfa_per_column, values], axis=1)

    values = (
        100
        * df_alfa_per_column[f"count_error_{error_type}_cumulative"]
        / df_alfa_per_column[f"count_error_{error_type}"].sum()
    ).to_frame(f"pct_error_{error_type}_of_{error_type}_cumulative")
    # df_alfa_per_column.insert(
    #     loc=len(df_alfa_per_column.columns),
    #     column=f"pct_error_{error_type}_of_{error_type}_cumulative",
    #     value=values
    # )
    df_alfa_per_column = pd.concat([df_alfa_per_column, values], axis=1)

    values = (
        100
        * df_alfa_per_column[f"count_error_{error_type}_cumulative"]
        / df_alfa_per_column["count_all"].sum()
    ).to_frame(f"pct_error_{error_type}_of_all_cumulative")
    # df_alfa_per_column.insert(
    #     loc=len(df_alfa_per_column.columns),
    #     column=f"pct_error_{error_type}_of_all_cumulative",
    #     value=values
    # )
    df_alfa_per_column = pd.concat([df_alfa_per_column, values], axis=1)

    # MARINA
    values = (
        df_alfa_per_column[f"count_error_{error_type}_cumulative"]
        / df_alfa_per_column["count_all_cumulative"]
    )
    df_alfa_per_column.insert(
        loc=len(df_alfa_per_column.columns), column="new_column", value=values
    )

    return df_alfa_per_column


def _write_OA_per_pixcount(
    df_parcel_predictions: pd.DataFrame, output_report_txt: Path, force: bool = False
):
    """Write a report of the overall accuracy that parcels per pixcount get."""
    # If force == False Check and the output file exists already, stop.
    if force is False and output_report_txt.exists():
        logger.warning(
            "collect_and_prepare_timeseries_data: output file already exists and "
            f"force is False, so stop: {output_report_txt}"
        )
        return

    # Write output...
    nb_predictions_total = len(df_parcel_predictions.index)
    with open(output_report_txt, "w") as outputfile:
        for i in range(40):
            df_result_cur_pixcount = df_parcel_predictions[
                df_parcel_predictions[conf.columns["pixcount_s1s2"]] == i
            ]
            nb_predictions_pixcount = len(df_result_cur_pixcount.index)
            if nb_predictions_pixcount == 0:
                continue

            overall_accuracy = 100.0 * skmetrics.accuracy_score(
                df_result_cur_pixcount[conf.columns["class"]],
                df_result_cur_pixcount["pred1"],
                normalize=True,
                sample_weight=None,
            )
            message = (
                f"OA for pixcount {i:2}: {overall_accuracy:3.2f} %, with "
                f"{nb_predictions_pixcount} elements "
                f"({100*(nb_predictions_pixcount/nb_predictions_total):.4f} % "
                f"of {nb_predictions_total})"
            )
            logger.info(message)
            outputfile.write(f"{message}\n")
