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
import tensorflow as tf

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.model_helper as mh
import cropclassification.helpers.pandas_helper as pdh

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# Set number of parallel threads for keras
"""
num_cores = os.cpu_count()
logger.info(f"Cores found: {num_cores}")
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count = {'CPU': num_cores})
session = tf.Session(config=config)
K.set_session(session)
"""

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def train(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_classifier_basepath: Path
) -> Path:
    """
    Train a classifier and output the trained classifier to the output file.

    Args
        train_df: pandas DataFrame containing the train data. Columns:
            * global_settings.id_column: the id of the parcel
            * global_settings.class_column: the class of the parcel
            * ... all columns that will be used as classification data
        test_df: pandas DataFrame containing the test/validation data.
        output_classifier_path: the path where the classifier can be written
    """

    # Prepare and check some input + init some variables
    output_classifier_path_noext, output_ext = os.path.splitext(
        output_classifier_basepath
    )
    output_classifier_classes_path = Path(f"{output_classifier_path_noext}_classes.txt")
    output_classifier_datacolumns_path = Path(
        f"{output_classifier_path_noext}_datacolumns.txt"
    )

    if output_ext.lower() != ".hdf5":
        message = f"Keras only supports saving in extension .hdf5, not in {output_ext}"
        logger.error(message)
        raise Exception(message)

    # Keras wants numeric classes, so prepare *_classes_df for that
    # First create dict with the conversion, and save it
    classes_dict = {
        key: value for value, key in enumerate(train_df[conf.columns["class"]].unique())
    }
    with open(output_classifier_classes_path, "w") as file:
        file.write(str(classes_dict))
    # Replace the string values with the ints
    column_class = conf.columns["class"]

    # TODO: doesn't seem to be the safest way to implement this: what if the
    # classes are integers instead of strings???
    train_df[column_class].replace(classes_dict, inplace=True)
    test_df[column_class].replace(classes_dict, inplace=True)

    # The test dataset also should only include classes we are training on...
    # I don't exactly why (don't know why the notnull/isnull must be there), but this
    # seems the only way it works?
    if test_df.dtypes[column_class] == "object":
        test_removed_df = test_df[test_df[column_class].str.isnumeric().notnull()]
        logger.info(
            "Removed following classes from test_classes_df: "
            f"{test_removed_df[column_class].unique()}"
        )
        test_df = test_df[test_df[column_class].str.isnumeric().isnull()]

    # Split the input dataframe in one with the train classes and one with the train
    # data
    train_classes_df = train_df[column_class]
    cols_to_keep = train_df.columns.difference(
        [conf.columns["id"], column_class]  # type: ignore
    )
    train_data_df = train_df[cols_to_keep]
    train_data_df.sort_index(axis=1, inplace=True)

    test_classes_df = test_df[column_class]
    cols_to_keep = test_df.columns.difference(
        [conf.columns["id"], column_class]  # type: ignore
    )
    test_data_df = test_df[cols_to_keep]
    test_data_df.sort_index(axis=1, inplace=True)

    logger.info(
        "Train file processed and rows with missing data removed, data shape: "
        f"{train_data_df.shape}, labels shape: {train_classes_df.shape}"
    )
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None  # type: ignore
    ):
        logger.info(
            f"Resulting Columns for training data: {list(train_data_df.columns)}"
        )
    with open(output_classifier_datacolumns_path, "w") as file:
        file.write(str(list(train_data_df.columns)))

    classifier_type_lower = conf.classifier["classifier_type"].lower()
    if classifier_type_lower != "keras_multilayer_perceptron":
        message = (
            "Unsupported classifier in conf.classifier['classifier_type']: "
            f"{conf.classifier['classifier_type']}"
        )
        logger.critical(message)
        raise Exception(message)

    # Keras not only want numeric classes, it wants a column per class
    train_classes_df = tf.keras.utils.to_categorical(
        train_classes_df, len(classes_dict)
    )
    test_classes_df = tf.keras.utils.to_categorical(test_classes_df, len(classes_dict))

    # Get some config from the config file
    hidden_layer_sizes = conf.classifier.getlistint(
        "multilayer_perceptron_hidden_layer_sizes"
    )
    if len(hidden_layer_sizes) == 0:
        raise Exception("Having no hidden layers is currently not supported")
    max_iter = conf.classifier.getint("multilayer_perceptron_max_iter")
    learning_rate_init = conf.classifier.getfloat(
        "multilayer_perceptron_learning_rate_init"
    )

    # Create neural network
    model = tf.keras.models.Sequential()
    # Create the hidden layers as specified in config
    dropout_pct = conf.classifier.getfloat("multilayer_perceptron_dropout_pct")
    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        # For the first layer, the input size needs to be specified
        if i == 0:
            model.add(
                tf.keras.layers.Dense(
                    hidden_layer_size,
                    activation="relu",
                    input_shape=(len(train_data_df.columns),),
                )
            )
            if dropout_pct > 0:
                model.add(tf.keras.layers.Dropout(dropout_pct / 100))
        else:
            model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))
            if dropout_pct > 0:
                model.add(tf.keras.layers.Dropout(dropout_pct / 100))

    # Add the final layer that will produce the outputs
    model.add(tf.keras.layers.Dense(len(classes_dict), activation="softmax"))

    # Prepare model for training + train!
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_init)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    logger.info(f"Start fitting classifier:\n{model.summary()}")
    acc_metric_mode = "min"
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode=acc_metric_mode,
        factor=0.2,
        patience=50,
        verbose=1,
        epsilon=1e-4,
    )
    callbacks = [reduce_lr_loss]

    # Several best_model strategies are possible, but it seems the standard/typical
    # val_loss gives the best results for this case.
    best_model_strategy = "VAL_LOSS"
    if best_model_strategy == "VAL_LOSS":
        to_be_formatted_by_callback = (
            "{val_loss:.5f}_{loss:.5f}_{val_loss:.5f}_{epoch:02d}"
        )
        best_model_path = (
            f"{output_classifier_path_noext}_{to_be_formatted_by_callback}{output_ext}"
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                save_best_only=True,
                monitor="val_loss",
                mode=acc_metric_mode,
            )
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode=acc_metric_mode, patience=200, verbose=0
            )
        )
    elif best_model_strategy == "LOSS":
        to_be_formatted_by_callback = "{loss:.5f}_{loss:.5f}_{val_loss:.5f}_{epoch:02d}"
        best_model_path = (
            f"{output_classifier_path_noext}_{to_be_formatted_by_callback}{output_ext}"
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                save_best_only=True,
                monitor="loss",
                mode=acc_metric_mode,
            )
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", mode=acc_metric_mode, patience=100, verbose=0
            )
        )
    elif best_model_strategy == "AVG(VAL_LOSS,LOSS)":
        # Custom callback that saves the best models using both train and validation
        # metric
        callbacks.append(
            mh.ModelCheckpointExt(
                output_classifier_basepath.parent,
                output_classifier_basepath.name,
                acc_metric_train="loss",
                acc_metric_validation="val_loss",
                acc_metric_mode=acc_metric_mode,
            )
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", mode=acc_metric_mode, patience=100, verbose=0
            )
        )

    csv_log_path = Path(f"{output_classifier_path_noext}_train_log.csv")
    callbacks.append(
        tf.keras.callbacks.CSVLogger(csv_log_path, append=True, separator=";")
    )
    model.fit(
        train_data_df,
        train_classes_df,
        batch_size=128,
        epochs=max_iter,
        callbacks=callbacks,
        validation_data=(test_data_df, test_classes_df),
    )

    # Get the best model after fitting to return it...
    best_model_info = mh.get_best_model(
        output_classifier_basepath.parent, acc_metric_mode=acc_metric_mode
    )
    best_model_path = best_model_info["path"]  # type: ignore

    return Path(best_model_path)


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
    parcel_classes_df = parcel_df[column_class]
    cols_to_keep = parcel_df.columns.difference(
        [conf.columns["id"], column_class, column_class_declared]  # type: ignore
    )
    parcel_data_df = parcel_df[cols_to_keep]
    parcel_data_df.sort_index(axis=1, inplace=True)

    logger.info(
        "Input predict file processed and rows with missing data removed, data shape: "
        f"{parcel_data_df.shape}, labels shape: {parcel_classes_df.shape}"
    )

    # Check of the input data columns match the columns needed for the neural net
    classifier_datacolumns_path = glob.glob(
        os.path.join(os.path.dirname(classifier_path), "*_datacolumns.txt")
    )[0]
    with open(classifier_datacolumns_path, "r") as file:
        classifier_datacolumns = ast.literal_eval(file.readline())
    if classifier_datacolumns != list(parcel_data_df.columns):
        raise Exception(
            "Input datacolumns for predict don't match needed columns for neural net: "
            f"\ninput: {parcel_data_df.columns}, \nneeded: {classifier_datacolumns}"
        )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None  # type: ignore
    ):
        logger.info(f"Resulting Columns for predicting data: {parcel_data_df.columns}")

    # Load the classifier and predict
    model = tf.keras.models.load_model(classifier_path)
    logger.info(f"Predict classes with probabilities: {len(parcel_df.index)} rows")
    class_proba = model.predict(parcel_data_df)
    logger.info("Predict classes with probabilities ready")

    # Convert probabilities to dataframe, combine with input data and write to file
    # Load the classes from the classes file
    classifier_classes_path = glob.glob(
        os.path.join(os.path.dirname(classifier_path), "*_classes.txt")
    )[0]
    with open(classifier_classes_path, "r") as file:
        classes_dict = ast.literal_eval(file.readline())

    id_class_proba = np.concatenate(
        [
            parcel_df[[conf.columns["id"], column_class, column_class_declared]].values,
            class_proba,
        ],
        axis=1,
    )
    cols = [conf.columns["id"], column_class, column_class_declared]
    cols.extend(classes_dict)
    proba_df = pd.DataFrame(id_class_proba, columns=cols)
    proba_df.set_index(keys=conf.columns["id"], inplace=True)

    # If output path provided, write results
    if output_parcel_predictions_path:
        pdh.to_file(proba_df, output_parcel_predictions_path)

    return proba_df


# If the script is run directly...
if __name__ == "__main__":
    logger.critical("Not implemented exception!")
    raise Exception("Not implemented")
