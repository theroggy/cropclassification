# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import logging
import os

import keras
from keras import backend as K
import numpy as np
import pandas as pd
import tensorflow as tf

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.model_helper as mh
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# Set number of parallel threads for keras
num_cores = os.cpu_count()
logger.info(f"Cores found: {num_cores}")
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, 
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True, 
                        device_count = {'CPU': num_cores})
session = tf.Session(config=config)
K.set_session(session)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(train_df: pd.DataFrame,
          test_df: pd.DataFrame,
          output_classifier_basefilepath: str) -> str:
    """
    Train a classifier and output the trained classifier to the output file.

    Args
        train_df: pandas DataFrame containing the train data. Columns:
            * global_settings.id_column: the id of the parcel
            * global_settings.class_column: the class of the parcel
            * ... all columns that will be used as classification data
        test_df: pandas DataFrame containing the test/validation data.
        output_classifier_filepath: the filepath where the classifier can be written
    """

    # Prepare and check some input + init some variables
    output_classifier_filepath_noext, output_ext = os.path.splitext(output_classifier_basefilepath)
    output_classifier_classes_filepath = f"{output_classifier_filepath_noext}_classes.txt"
    output_classifier_datacolumns_filepath = f"{output_classifier_filepath_noext}_datacolumns.txt"
    output_classifier_dir, output_classifier_filename = os.path.split(output_classifier_basefilepath)
    csv_log_filepath = f"{output_classifier_filepath_noext}_train_log.csv"

    if output_ext.lower() != '.hdf5':
        message = f"Keras only supports saving in extension .hdf5, not in {output_ext}"
        logger.error(message)
        raise Exception(message)

    # Keras wants numeric classes, so prepare *_classes_df for that
    # First create dict with the conversion, and save it 
    classes_dict = {key:value for value, key in enumerate(train_df[conf.columns['class']].unique())}
    with open(output_classifier_classes_filepath, "w") as file:
        file.write(str(classes_dict))
    # Replace the string values with the ints
    column_class = conf.columns['class']
    train_df[column_class].replace(classes_dict, inplace=True)
    test_df[column_class].replace(classes_dict, inplace=True)

    # The test dataset also should only include classes we are training on...
    # I don't exactly why (don't know why the notnull/isnull must be there), but this seems the 
    # only way it works?
    test_removed_df = test_df[test_df[column_class].str.isnumeric().notnull()]
    logger.info(f"Removed following classes from test_classes_df: {test_removed_df[column_class].unique()}")
    test_df = test_df[test_df[column_class].str.isnumeric().isnull()]

    # Split the input dataframe in one with the train classes and one with the train data
    train_classes_df = train_df[column_class]
    cols_to_keep = train_df.columns.difference([conf.columns['id'], column_class])
    train_data_df = train_df[cols_to_keep]
    train_data_df.sort_index(axis=1, inplace=True)

    test_classes_df = test_df[column_class]
    cols_to_keep = test_df.columns.difference([conf.columns['id'], column_class])
    test_data_df = test_df[cols_to_keep]
    test_data_df.sort_index(axis=1, inplace=True)

    logger.info(f"Train file processed and rows with missing data removed, data shape: {train_data_df.shape}, labels shape: {train_classes_df.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {list(train_data_df.columns)}")
    with open(output_classifier_datacolumns_filepath, "w") as file:
        file.write(str(list(train_data_df.columns)))

    classifier_type_lower = conf.classifier['classifier_type'].lower()
    if classifier_type_lower != 'keras_multilayer_perceptron':
        message = f"Unsupported classifier in conf.classifier['classifier_type']: {conf.classifier['classifier_type']}"
        logger.critical(message)
        raise Exception(message)

    # Keras not only want numeric classes, it wants a column per class
    train_classes_df = keras.utils.to_categorical(train_classes_df, len(classes_dict))
    test_classes_df = keras.utils.to_categorical(test_classes_df, len(classes_dict))

    # Get some config from the config file
    hidden_layer_sizes = conf.classifier.getlistint('multilayer_perceptron_hidden_layer_sizes')
    if len(hidden_layer_sizes) == 0:
        raise Exception("Having no hidden layers is currently not supported")
    max_iter = conf.classifier.getint('multilayer_perceptron_max_iter')
    learning_rate_init = conf.classifier.getfloat('multilayer_perceptron_learning_rate_init')
    
    # Create neural network
    model = keras.models.Sequential()
    # Create the hidden layers as specified in config
    dropout_pct = conf.classifier.getfloat('multilayer_perceptron_dropout_pct')
    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        # For the first layer, the input size needs to be specified
        if i == 0:
            model.add(keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(len(train_data_df.columns),)))
            if dropout_pct > 0:
                model.add(keras.layers.Dropout(dropout_pct/100))
        else:
            model.add(keras.layers.Dense(hidden_layer_size, activation='relu'))
            if dropout_pct > 0:
                model.add(keras.layers.Dropout(dropout_pct/100))

    # Add the final layer that will produce the outputs
    model.add(keras.layers.Dense(len(classes_dict), activation='softmax'))

    # Prepare model for training + train!
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logger.info(f"Start fitting classifier:\n{model.summary()}")
    acc_metric_mode = 'min'
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode=acc_metric_mode,
            factor=0.2, patience=30, verbose=1, epsilon=1e-4)
    
    # Several best_model strategies are possible, but it seems the standard/typical val_loss 
    # gives the best results for this case.
    best_model_strategy = 'VAL_LOSS'
    if(best_model_strategy == 'VAL_LOSS'):
        best_model_filepath = output_classifier_basefilepath        
        mcp_saver = keras.callbacks.ModelCheckpoint(
                best_model_filepath, save_best_only=True, 
                monitor='val_loss', mode=acc_metric_mode)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode=acc_metric_mode,
                patience=100, verbose=0)    
    elif(best_model_strategy == 'LOSS'):
        best_model_filepath = f"{output_classifier_filepath_noext}_train_loss{output_ext}"
        mcp_saver = keras.callbacks.ModelCheckpoint(
                best_model_filepath, save_best_only=True, 
                monitor='loss', mode=acc_metric_mode)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', mode=acc_metric_mode,
                patience=100, verbose=0)
    elif(best_model_strategy == 'AVG(VAL_LOSS,LOSS)'):
        # Custom callback that saves the best models using both train and validation metric
        mcp_saver = mh.ModelCheckpointExt(
                output_classifier_dir, output_classifier_filename, 
                acc_metric_train='loss', acc_metric_validation='val_loss', 
                acc_metric_mode=acc_metric_mode)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', mode=acc_metric_mode,
                patience=100, verbose=0)
                
    csv_logger = keras.callbacks.CSVLogger(csv_log_filepath, append=True, separator=';')
    model.fit(train_data_df, train_classes_df, batch_size=128, epochs=1000, 
              callbacks=[earlyStopping, reduce_lr_loss, mcp_saver, csv_logger],
              validation_data=(test_data_df, test_classes_df))

    # For this strategy, the best model need to be looked for after fitting...
    if(best_model_strategy == 'AVG(VAL_LOSS,LOSS)'):
        best_model_info = mh.get_best_model(output_classifier_dir, acc_metric_mode=acc_metric_mode)
        best_model_filepath = best_model_info['filepath']

    return best_model_filepath

def predict_proba(parcel_df: pd.DataFrame,
                  classifier_basefilepath: str,
                  classifier_filepath: str,
                  output_parcel_predictions_filepath: str) -> pd.DataFrame:
    """
    Predict the probabilities for all input data using the classifier provided and write it
    to the output file.

    Args
        parcel_df: pandas DataFrame containing the data to classify. Columns:
            * global_settings.id_column: the id of the parcel.
            * global_settings.class_column: the class of the parcel. Isn't really used.
            * ... all columns that will be used as classification data.
        classifier_filepath: the filepath where the classifier can be written.
    """

    # Some basic checks that input is ok
    column_class = conf.columns['class']
    column_class_declared = conf.columns['class_declared']
    parcel_df.reset_index(inplace=True)
    if(conf.columns['id'] not in parcel_df.columns
       or column_class not in parcel_df.columns):
        message = f"Columns {conf.columns['id']} and {column_class} are mandatory for input parameter parcel_df!"
        logger.critical(message)
        raise Exception(message)

    # Now do final preparation for the classification
    parcel_classes_df = parcel_df[column_class]
    cols_to_keep = parcel_df.columns.difference(
            [conf.columns['id'], column_class, column_class_declared])
    parcel_data_df = parcel_df[cols_to_keep]
    parcel_data_df.sort_index(axis=1, inplace=True)

    logger.info(f"Input predict file processed and rows with missing data removed, data shape: {parcel_data_df.shape}, labels shape: {parcel_classes_df.shape}")

    # Check of the input data columns match the columns needed for the neural net
    classifier_basefilepath_noext, _ = os.path.splitext(classifier_basefilepath)
    classifier_datacolumns_filepath = classifier_basefilepath_noext + '_datacolumns.txt'
    with open(classifier_datacolumns_filepath, "r") as file:
        classifier_datacolumns = eval(file.readline())
    if classifier_datacolumns != list(parcel_data_df.columns):
        raise Exception(f"Input datacolumns for predict don't match needed columns for neural net: \ninput: {parcel_data_df.columns}, \nneeded: {classifier_datacolumns}" )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for predicting data: {parcel_data_df.columns}")

    # Load the classifier and predict
    model = keras.models.load_model(classifier_filepath)
    logger.info(f"Predict classes with probabilities: {len(parcel_df.index)} rows")
    class_proba = model.predict_proba(parcel_data_df)
    logger.info(f"Predict classes with probabilities ready")

    # Convert probabilities to dataframe, combine with input data and write to file
    # Load the classes from the classes file
    classifier_classes_filepath = classifier_basefilepath_noext + '_classes.txt'
    with open(classifier_classes_filepath, "r") as file:
        classes_dict = eval(file.readline())
    id_class_proba = np.concatenate(
            [parcel_df[[conf.columns['id'], column_class, column_class_declared]].values, 
             class_proba], axis=1)
    cols = [conf.columns['id'], column_class, column_class_declared]
    cols.extend(classes_dict)
    proba_df = pd.DataFrame(id_class_proba, columns=cols)
    proba_df.set_index(keys=conf.columns['id'], inplace=True)
    
    # If output path provided, write results
    if output_parcel_predictions_filepath:
        pdh.to_file(proba_df, output_parcel_predictions_filepath)

    return proba_df

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
