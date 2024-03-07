# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import ast
import glob
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

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


def train_test_predict(
    input_parcel_train_path: Path,
    input_parcel_test_path: Path,
    input_parcel_all_path: Path,
    input_parcel_classification_data_path: Path,
    output_classifier_basepath: Path,
    output_predictions_test_path: Path,
    output_predictions_all_path: Path,
    force: bool = False,
):
    """Train a classifier, test it and do full predictions.

    Args
        input_parcel_classes_train_path: the list of parcels with classes to train the
            classifier, without data!
        input_parcel_classes_test_path: the list of parcels with classes to test the
            classifier, without data!
        input_parcel_classes_all_path: the list of parcels with classes that need to be
            classified, without data!
        input_parcel_classification_data_path: the data to be used for the
            classification for all parcels.
        output_classifier_basepath: the file path where to save the classifier.
        output_predictions_test_path: the file path where to save the test predictions.
        output_predictions_all_path: the file path where to save the predictions for
            all parcels.
        force: if True, overwrite all existing output files, if False, don't overwrite
            them.
    """

    logger.info("train_test_predict: Start")

    if (
        force is False
        # and os.path.exists(output_classifier_basepath)
        and output_predictions_test_path.exists()
        and output_predictions_all_path.exists()
    ):
        logger.warning(
            "predict: output files exist and force is False, so stop: "
            f"{output_classifier_basepath}, {output_predictions_test_path}, "
            f"{output_predictions_all_path}"
        )
        return

    # Read the classification data from the csv so we can pass it on to the other
    # functions to improve performance...
    logger.info(
        f"Read classification data file: {input_parcel_classification_data_path}"
    )
    input_parcel_classification_data_df = pdh.read_file(
        input_parcel_classification_data_path
    )
    if input_parcel_classification_data_df.index.name != conf.columns["id"]:
        input_parcel_classification_data_df.set_index(conf.columns["id"], inplace=True)
    logger.debug("Read classification data file ready")
    if len(input_parcel_classification_data_df.columns) == 0:
        raise ValueError(
            f"no classification data found in {input_parcel_classification_data_path}"
        )

    # Train the classification
    output_classifier_path = train(
        input_parcel_train_path=input_parcel_train_path,
        input_parcel_test_path=input_parcel_test_path,
        input_parcel_classification_data_path=input_parcel_classification_data_path,
        output_classifier_basepath=output_classifier_basepath,
        force=force,
        input_parcel_classification_data_df=input_parcel_classification_data_df,
    )

    # Predict the test parcels
    predict(
        input_parcel_path=input_parcel_test_path,
        input_parcel_classification_data_path=input_parcel_classification_data_path,
        input_classifier_basepath=output_classifier_basepath,
        input_classifier_path=output_classifier_path,
        output_predictions_path=output_predictions_test_path,
        force=force,
        input_parcel_classification_data_df=input_parcel_classification_data_df,
    )

    # Predict all parcels
    predict(
        input_parcel_path=input_parcel_all_path,
        input_parcel_classification_data_path=input_parcel_classification_data_path,
        input_classifier_basepath=output_classifier_basepath,
        input_classifier_path=output_classifier_path,
        output_predictions_path=output_predictions_all_path,
        force=force,
        input_parcel_classification_data_df=input_parcel_classification_data_df,
    )


def train(
    input_parcel_train_path: Path,
    input_parcel_test_path: Path,
    input_parcel_classification_data_path: Path,
    output_classifier_basepath: Path,
    force: bool = False,
    input_parcel_classification_data_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Train a classifier and test it by predicting the test cases."""

    logger.info("train_and_test: Start")
    if force is False and output_classifier_basepath.exists():
        logger.warning(
            "predict: classifier already exist and force == False, so don't retrain: "
            f"{output_classifier_basepath}"
        )
        return output_classifier_basepath

    # If the classification data isn't passed as dataframe, read it from file
    if input_parcel_classification_data_df is None:
        logger.info(
            f"Read classification data file: {input_parcel_classification_data_path}"
        )
        input_parcel_classification_data_df = pdh.read_file(
            input_parcel_classification_data_path
        )
        if input_parcel_classification_data_df.index.name != conf.columns["id"]:
            input_parcel_classification_data_df.set_index(
                conf.columns["id"], inplace=True
            )
        logger.debug("Read classification data file ready")

    # Read the train parcels
    logger.info(f"Read train file: {input_parcel_train_path}")
    train_df = pdh.read_file(
        input_parcel_train_path, columns=[conf.columns["id"], conf.columns["class"]]
    )
    if train_df.index.name != conf.columns["id"]:
        train_df.set_index(conf.columns["id"], inplace=True)
    logger.debug("Read train file ready")

    # Join the columns of input_parcel_classification_data_df that aren't yet in
    # train_df
    logger.info("Join train sample with the classification data")
    train_df = train_df.join(input_parcel_classification_data_df, how="inner")

    # Read the test/validation data
    logger.info(f"Read test file: {input_parcel_test_path}")
    test_df = pdh.read_file(
        input_parcel_test_path, columns=[conf.columns["id"], conf.columns["class"]]
    )
    if test_df.index.name != conf.columns["id"]:
        test_df.set_index(conf.columns["id"], inplace=True)
    logger.debug("Read test file ready")

    # Join the columns of input_parcel_classification_data_df that aren't yet in test_df
    logger.info("Join test sample with the classification data")
    test_df = test_df.join(input_parcel_classification_data_df, how="inner")

    # Train
    if conf.classifier["classifier_type"].lower() == "keras_multilayer_perceptron":
        import cropclassification.predict.classification_keras as class_core_keras

        return class_core_keras.train(
            train_df=train_df,
            test_df=test_df,
            output_classifier_basepath=output_classifier_basepath,
        )
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn

        return class_core_sklearn.train(
            train_df=train_df,
            output_classifier_basepath=output_classifier_basepath,
        )


def predict(
    input_parcel_path: Path,
    input_parcel_classification_data_path: Path,
    input_classifier_basepath: Path,
    input_classifier_path: Path,
    output_predictions_path: Path,
    force: bool = False,
    input_parcel_classification_data_df: Optional[pd.DataFrame] = None,
):
    """Predict the classes for the input data."""

    # If force is False, and the output file exist already, return
    if force is False and output_predictions_path.exists():
        logger.warning(
            "predict: predictions output file already exists and force is false, so "
            f"stop: {output_predictions_path}"
        )
        return

    # Read the input parcels
    logger.info(f"Read input file: {input_parcel_path}")
    input_parcel_df = pdh.read_file(
        input_parcel_path,
        columns=[
            conf.columns["id"],
            conf.columns["class"],
            conf.columns["class_declared"],
        ],
    )
    if input_parcel_df.index.name != conf.columns["id"]:
        input_parcel_df.set_index(conf.columns["id"], inplace=True)
    logger.debug("Read train file ready")

    # For parcels of a class that should be ignored, don't predict
    input_parcel_df = input_parcel_df.loc[
        ~input_parcel_df[conf.columns["class_declared"]].isin(
            conf.marker.getlist("classes_to_ignore")
        )
    ]

    # get the expected columns from the classifier
    datacolumns_path = glob.glob(
        os.path.join(os.path.dirname(input_classifier_path), "*datacolumns.txt")
    )[0]
    with open(datacolumns_path, "r") as f:
        input_classifier_datacolumns = ast.literal_eval(f.readline())

    # If the classification data isn't passed as dataframe, read it from the csv
    if input_parcel_classification_data_df is None:
        logger.info(
            f"Read classification data file: {input_parcel_classification_data_path}"
        )
        input_parcel_classification_data_df = pdh.read_file(
            input_parcel_classification_data_path
        )
        if input_parcel_classification_data_df.index.name != conf.columns["id"]:
            input_parcel_classification_data_df.set_index(
                conf.columns["id"], inplace=True
            )
        logger.debug("Read classification data file ready")

    # only take the required columns as expected by the classifier
    input_parcel_classification_data_df = input_parcel_classification_data_df[
        input_classifier_datacolumns
    ]

    # Join the data to send to prediction logic
    logger.info("Join input parcels with the classification data")
    input_parcel_for_predict_df = input_parcel_df.join(
        input_parcel_classification_data_df,
        how="inner",
    )

    # Predict!
    logger.info(f"Predict using this model: {input_classifier_path}")
    if conf.classifier["classifier_type"].lower() == "keras_multilayer_perceptron":
        import cropclassification.predict.classification_keras as class_core_keras

        class_core_keras.predict_proba(
            parcel_df=input_parcel_for_predict_df,
            classifier_basepath=input_classifier_basepath,
            classifier_path=input_classifier_path,
            output_parcel_predictions_path=output_predictions_path,
        )
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn

        class_core_sklearn.predict_proba(
            parcel_df=input_parcel_for_predict_df,
            classifier_basepath=input_classifier_basepath,
            classifier_path=input_classifier_path,
            output_parcel_predictions_path=output_predictions_path,
        )


# If the script is run directly...
if __name__ == "__main__":
    logger.critical("Not implemented exception!")
    raise Exception("Not implemented")
