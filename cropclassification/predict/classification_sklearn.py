# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import ast
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.svm import SVC

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def train(train_df: pd.DataFrame, output_classifier_basepath: Path) -> Path:
    """
    Train a classifier and output the trained classifier to the output file.

    Args
        train_df: pandas DataFrame containing the train data. Columns:
            * global_settings.id_column: the id of the parcel
            * global_settings.class_column: the class of the parcel
            * ... all columns that will be used as classification data
        output_classifier_basepath: the path where the classifier can be written
    """
    output_classifier_path_noext, _ = os.path.splitext(output_classifier_basepath)
    output_classifier_path = output_classifier_basepath
    output_classifier_datacolumns_path = Path(
        f"{output_classifier_path_noext}_datacolumns.txt"
    )

    # Split the input dataframe in one with the train classes and one with the train
    # data
    train_classes_df = train_df[conf.columns["class"]]
    cols_to_keep = train_df.columns.difference(
        [conf.columns["id"], conf.columns["class"]]  # type: ignore
    )
    train_data_df = train_df[cols_to_keep]

    logger.info(
        "Train file processed and rows with missing data removed, data shape: "
        f"{train_data_df.shape}, labels shape: {train_classes_df.shape}"
    )
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # type: ignore
        logger.info(f"Resulting Columns for training data: {train_data_df.columns}")
    with open(output_classifier_datacolumns_path, "w") as file:
        file.write(str(list(train_data_df.columns)))

    # Using almost all defaults for the classifier seems to work best...
    logger.info("Start training")
    classifier_type_lower = conf.classifier["classifier_type"].lower()
    if classifier_type_lower == "randomforest":
        n_estimators = conf.classifier.getint("randomforest_n_estimators")
        max_depth = conf.classifier.getint("randomforest_max_depth")
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
    elif classifier_type_lower == "nearestneighbor":
        classifier = KNeighborsClassifier(weights="distance", n_jobs=-1)
    elif classifier_type_lower == "multilayer_perceptron":
        hidden_layer_sizes = tuple(
            conf.classifier.getlistint("multilayer_perceptron_hidden_layer_sizes")
        )
        max_iter = conf.classifier.getint("multilayer_perceptron_max_iter")
        learning_rate_init = conf.classifier.getfloat(
            "multilayer_perceptron_learning_rate_init"
        )
        classifier = MLPClassifier(
            max_iter=max_iter,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
        )
    elif classifier_type_lower == "svm":
        # cache_size=1000 (MB) should speed up training
        # probability=True is necessary to be able to use predict_proba
        classifier = SVC(
            C=64.0, gamma=0.125, probability=True, cache_size=1000  # type: ignore
        )
    else:
        message = (
            "Unsupported classifier in conf.classifier['classifier_type']: "
            f"{conf.classifier['classifier_type']}"
        )
        logger.critical(message)
        raise Exception(message)

    logger.info(f"Start fitting classifier:\n{classifier}")
    classifier.fit(train_data_df, train_classes_df)

    # Write the learned model to a file...
    logger.info(f"Write the learned model file to {output_classifier_path}")
    joblib.dump(classifier, output_classifier_path)

    return output_classifier_path


def predict_proba(
    parcel_df: pd.DataFrame,
    classifier_basepath: Path,
    classifier_path: Path,
    output_parcel_predictions_path: Path,
) -> pd.DataFrame:
    """
    Predict the probabilities for all input data using the classifier provided and
    write it to the output file.

    Args
        parcel_df: pandas DataFrame containing the data to classify. Columns:
            * global_settings.id_column: the id of the parcel.
            * global_settings.class_column: the class of the parcel. Isn't really used.
            * ... all columns that will be used as classification data.
        classifier_path: the path where the classifier can be written.
        output_parcel_predictions_path: file to write the predictions to.
    """

    # Some basic checks that input is ok
    column_class = conf.columns["class"]
    column_class_declared = conf.columns["class_declared"]
    parcel_df.reset_index(inplace=True)
    if (
        conf.columns["id"] not in parcel_df.columns
        or column_class not in parcel_df.columns
    ):
        message = (
            f"Columns {conf.columns['id']} and {column_class} are mandatory for input "
            "parameter parcel_df!"
        )
        logger.critical(message)
        raise Exception(message)

    # Now do final preparation for the classification
    parcel_classes_df = parcel_df[conf.columns["class"]]
    cols_to_keep = parcel_df.columns.difference(
        [conf.columns["id"], column_class, column_class_declared]  # type: ignore
    )
    parcel_data_df = parcel_df[cols_to_keep]

    logger.info(
        "Train file processed and rows with missing data removed, data shape: "
        f"{parcel_data_df.shape}, labels shape: {parcel_classes_df.shape}"
    )
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # type: ignore
        logger.info(f"Resulting Columns for training data: {parcel_data_df.columns}")

    # Check of the input data columns match the columns needed for the classifier
    classifier_datacolumns_path = glob.glob(
        os.path.join(os.path.dirname(classifier_path), "*_datacolumns.txt")
    )[0]
    with open(classifier_datacolumns_path, "r") as file:
        classifier_datacolumns = ast.literal_eval(file.readline())
    if classifier_datacolumns != list(parcel_data_df.columns):
        raise Exception(
            "Input datacolumns for predict don't match needed columns for classifier: "
            f"\ninput: {parcel_data_df.columns}, \nneeded: {classifier_datacolumns}"
        )

    # Load the classifier
    classifier = joblib.load(classifier_path)
    logger.info(f"Classifier has the following columns: {classifier.classes_}")

    logger.info(f"Predict classes with probabilities: {len(parcel_df.index)} rows")
    class_proba = classifier.predict_proba(parcel_data_df)
    logger.info("Predict classes with probabilities ready")

    # Convert probabilities to dataframe, combine with input data and write to file
    id_class_proba = np.concatenate(
        [
            parcel_df[[conf.columns["id"], column_class, column_class_declared]].values,
            class_proba,
        ],
        axis=1,
    )
    cols = [conf.columns["id"], column_class, column_class_declared]
    cols.extend(classifier.classes_)
    proba_df = pd.DataFrame(id_class_proba, columns=cols)
    proba_df = proba_df.set_index(keys=conf.columns["id"])

    # If output path provided, write results
    if output_parcel_predictions_path:
        pdh.to_file(proba_df, output_parcel_predictions_path)

    return proba_df
