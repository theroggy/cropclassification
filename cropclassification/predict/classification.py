"""Module that implements the classification logic."""

import ast
import glob
import logging
import os
from pathlib import Path

import geofileops as gfo
import pandas as pd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
from cropclassification.helpers import model_helper as mh
from cropclassification.predict import data_balancing

# Get a logger...
logger = logging.getLogger(__name__)


def classify(
    classifier_type: str,
    parcel_path: Path,
    parcel_classification_data_path: Path,
    output_dir: Path,
    output_base_filename: str,
    test_size: float,
    cross_pred_models: int,
    input_model_to_use_path: Path | None,
    force: bool = False,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Classify the parcels.

    Args:
        classifier_type (str): the classifier to use.
        parcel_path (Path): path to the parcels to classify.
        parcel_classification_data_path (Path): path to the classification data.
        output_dir (Path): output directory.
        output_base_filename (str): base filename for the output files.
        test_size (float): size of the test set to use. Float between 0 and 1.
        cross_pred_models (int): the number of models to use for cross prediction.
            If <= 1, no cross prediction models are used.
        input_model_to_use_path (Optional[Path]): path to an existing model to use.
        force (bool, optional): True to force training a new model. Defaults to False.

    Returns:
        Tuple with the following paths:
            - output_proba_all_path: file with the predictions for all parcels
            - output_proba_test_path: file with the predictions for the test parcels
            - parcel_train_path: file with the training parcels
            - parcel_test_path: file with the test parcels

    """
    # Prepare output filenames
    data_ext = conf.general["data_ext"]
    classifier_ext = conf.classifier["classifier_ext"]
    output_proba_all_path = (
        output_dir / f"{output_base_filename}_predict_proba_all{data_ext}"
    )
    output_proba_test_path = (
        output_dir / f"{output_base_filename}_predict_proba_test{data_ext}"
    )

    # If the predictions file doesn't exist, do the classification
    if not force and output_proba_all_path.exists():
        return (output_proba_all_path, output_proba_test_path, None, None)

    cross_pred_model_id_column = conf.columns["cross_pred_model_id"]

    if cross_pred_models <= 1:
        # Simplifies code further on to set to 1 when not using cross prediction models
        cross_pred_models = 1
    else:
        # Use "cross models": train multiple models, so prediction of a parcel is
        # always done with a model where the parcel wasn't used to train it.
        if input_model_to_use_path is not None:
            raise ValueError(
                "cross_pred_models not supported with input_model_to_use_path"
            )

        # Assign each parcel to a model where it will be predicted with.
        # To avoid impacting the data balancing, the are divided evenly over the models
        # based on the class_balancing_column.
        class_balancing_column = conf.columns["class_balancing"]
        add_cross_pred_model_id(
            parcel_path=parcel_path,
            cross_pred_models=cross_pred_models,
            columnname=cross_pred_model_id_column,
            class_balancing_column=class_balancing_column,
        )

        # No need to split the data in test and train when using cross prediction models
        # with an sklearn model.
        if not classifier_type.lower().startswith("keras_"):
            test_size = 0.0

    pred_all_files = []
    for cross_pred_model_idx in range(cross_pred_models):
        if cross_pred_models <= 1:
            # Only one model, so no need to create subdirectory
            model_dir = output_dir

            # Output can be the final file immediately
            parcel_preds_proba_test_model_path = output_proba_test_path
            parcel_preds_proba_all_model_path = output_proba_all_path

            # No need to filter percels based on cross model indexes
            training_query = None
            predict_query = None
        else:
            # Create a subfolder for each model
            model_dir = output_dir / f"cross_pred_model_{cross_pred_model_idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Temporary output partial files in the subdirectory
            parcel_preds_proba_test_model_path = model_dir / output_proba_test_path.name
            parcel_preds_proba_all_model_path = model_dir / output_proba_all_path.name
            pred_all_files.append(parcel_preds_proba_all_model_path)

            # The training will be done on all parcels with another index
            training_cross_pred_model_ids = [
                idx for idx in range(cross_pred_models) if idx != cross_pred_model_idx
            ]
            training_query = (
                f"{cross_pred_model_id_column} in {training_cross_pred_model_ids}"
            )
            predict_query = f"{cross_pred_model_id_column} == {cross_pred_model_idx}"
            # TODO: is there a use of keeping test data seperate if
            # cross-prediction-models are used?

        # Check if a model exists already
        if input_model_to_use_path is None:
            best_model = mh.get_best_model(model_dir, acc_metric_mode="min")
            if best_model is not None:
                input_model_to_use_path = best_model["path"]

        # If we cannot reuse an existing model, train one!
        parcel_train_path = model_dir / f"parcel_train{data_ext}"
        parcel_test_path = model_dir / f"parcel_test{data_ext}"
        classifier_basepath = (
            model_dir / f"marker_01_{classifier_type.replace('_', '-')}{classifier_ext}"
        )
        if input_model_to_use_path is None:
            # Create the training sample.
            # Remark: this creates a list of representative test parcel + a list of
            # (candidate) training parcels
            balancing_strategy = conf.marker["balancing_strategy"]
            data_balancing.create_train_test_sample(
                input_parcel_path=parcel_path,
                output_parcel_train_path=parcel_train_path,
                output_parcel_test_path=parcel_test_path,
                balancing_strategy=balancing_strategy,
                test_size=test_size,
                training_query=training_query,
                force=force,
            )

            # Train the classifier and output predictions
            train_test_predict(
                classifier_type=classifier_type,
                input_parcel_train_path=parcel_train_path,
                input_parcel_test_path=parcel_test_path,
                input_parcel_all_path=parcel_path,
                input_parcel_classification_data_path=parcel_classification_data_path,
                output_classifier_basepath=classifier_basepath,
                output_predictions_test_path=parcel_preds_proba_test_model_path,
                output_predictions_all_path=parcel_preds_proba_all_model_path,
                predict_query=predict_query,
                force=force,
            )
        else:
            # There exists already a classifier, so just use it!
            predict(
                input_parcel_path=parcel_path,
                input_parcel_classification_data_path=parcel_classification_data_path,
                input_classifier_basepath=classifier_basepath,
                input_classifier_path=input_model_to_use_path,
                output_predictions_path=parcel_preds_proba_all_model_path,
                predict_query=predict_query,
                force=force,
            )

    # Merge all predictions to a single output file
    if cross_pred_models > 1:
        # Merge all "all" predictions to single output file.
        # The "test" predictions are not very useful when cross prediction models
        # are used, as the "all" predictions are independent of the training.
        pred_all_df = None
        for path in pred_all_files:
            df = pdh.read_file(path)
            pred_all_df = pd.concat([pred_all_df, df], ignore_index=True)
        pdh.to_file(pred_all_df, output_proba_all_path, index=False)

        return (output_proba_all_path, output_proba_test_path, None, None)

    return (
        output_proba_all_path,
        output_proba_test_path,
        parcel_train_path,
        parcel_test_path,
    )


def add_cross_pred_model_id(
    parcel_path: Path,
    cross_pred_models: int,
    columnname: str | None = None,
    class_balancing_column: str | None = None,
):
    """Add a column to the parcel file that assigns each parcel to a model.

    Args:
        parcel_path (Path): path with parcels to add the column to.
        cross_pred_models (int): the number of models to divide the parcels over.
        columnname (Optional[str], optional): the column name for the column to add to
            `parcel_path`. If None, conf.columns["cross_pred_model_id"] is used.
            Defaults to None.
        class_balancing_column (Optional[str], optional): the column in the
            `parcel_path` file to use for balancing the model ids. If None,
            conf.columns["class_balancing"] is used. Defaults to None.
    """
    if columnname is None:
        columnname = conf.columns["cross_pred_model_id"]
    if class_balancing_column is None:
        class_balancing_column = conf.columns["class_balancing"]
    assert columnname is not None
    assert class_balancing_column is not None

    layer = gfo.get_only_layer(parcel_path)
    gfo.add_column(path=parcel_path, layer=layer, name=columnname, type="int")
    # Remarks:
    #   - ORDER BY rowid is used to get a deterministic order
    #   - row_number() starts at 1, so we subtract 1 to get a 0-based number
    sql = f"""
            WITH tmp AS
              (SELECT rowid
                     ,row_number() OVER
                        (PARTITION BY "{class_balancing_column}" ORDER BY rowid)
                      AS rownum
                 FROM "{layer}"
              )
            UPDATE "{layer}"
               SET "{columnname}" = (
                     SELECT (rownum -1) % {cross_pred_models} AS model_id FROM tmp
                      WHERE tmp.rowid = "{layer}".rowid
                   )
        """
    gfo.execute_sql(path=parcel_path, sql_stmt=sql)


def train_test_predict(
    classifier_type: str,
    input_parcel_train_path: Path,
    input_parcel_test_path: Path,
    input_parcel_all_path: Path,
    input_parcel_classification_data_path: Path,
    output_classifier_basepath: Path,
    output_predictions_test_path: Path,
    output_predictions_all_path: Path,
    predict_query: str | None = None,
    force: bool = False,
):
    """Train a classifier, test it and do full predictions.

    Args:
        classifier_type: the type of classifier to use.
        input_parcel_train_path: the list of parcels with classes to train the
            classifier, without data!
        input_parcel_test_path: the list of parcels with classes to test the
            classifier, without data!
        input_parcel_all_path: the list of parcels with classes that need to be
            classified, without data!
        input_parcel_classification_data_path: the data to be used for the
            classification for all parcels.
        output_classifier_basepath: the file path where to save the classifier.
        output_predictions_test_path: the file path where to save the test predictions.
        output_predictions_all_path: the file path where to save the predictions for
            all parcels.
        predict_query: only predict the parcels that comply with the query. Defaults to
            None, which means all parcels are predicted.
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
        classifier_type=classifier_type,
        input_parcel_train_path=input_parcel_train_path,
        input_parcel_test_path=input_parcel_test_path,
        input_parcel_classification_data_path=input_parcel_classification_data_path,
        output_classifier_basepath=output_classifier_basepath,
        force=force,
        input_parcel_classification_data_df=input_parcel_classification_data_df,
    )

    # Predict the test parcels
    if input_parcel_test_path.exists():
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
        predict_query=predict_query,
    )


def train(
    classifier_type: str,
    input_parcel_train_path: Path,
    input_parcel_test_path: Path,
    input_parcel_classification_data_path: Path,
    output_classifier_basepath: Path,
    force: bool = False,
    input_parcel_classification_data_df: pd.DataFrame | None = None,
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
    test_df = None
    if input_parcel_test_path.exists():
        logger.info(f"Read test file: {input_parcel_test_path}")
        test_df = pdh.read_file(
            input_parcel_test_path, columns=[conf.columns["id"], conf.columns["class"]]
        )
        if test_df.index.name != conf.columns["id"]:
            test_df.set_index(conf.columns["id"], inplace=True)
        logger.debug("Read test file ready")

        # Join the columns of input_parcel_classification_data_df that aren't in test_df
        logger.info("Join test sample with the classification data")
        test_df = test_df.join(input_parcel_classification_data_df, how="inner")

    # Train
    if classifier_type.lower() == "keras_multilayer_perceptron":
        if test_df is None:
            raise ValueError("test_df is mandatory when using a keras classifier")

        import cropclassification.predict.classification_keras as class_core_keras  # noqa: PLC0415

        return class_core_keras.train(
            train_df=train_df,
            test_df=test_df,
            output_classifier_basepath=output_classifier_basepath,
        )
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn  # noqa: PLC0415

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
    input_parcel_classification_data_df: pd.DataFrame | None = None,
    predict_query: str | None = None,
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
    columns = [
        conf.columns["id"],
        conf.columns["class"],
        conf.columns["class_declared"],
    ]
    if predict_query is not None:
        columns.append(conf.columns["cross_pred_model_id"])
    input_parcel_df = pdh.read_file(input_parcel_path, columns=columns)
    if input_parcel_df.index.name != conf.columns["id"]:
        input_parcel_df.set_index(conf.columns["id"], inplace=True)

    if predict_query is not None:
        logger.info(f"Filter predict parcels with query: {predict_query}")
        input_parcel_df = input_parcel_df.query(predict_query)
        input_parcel_df = input_parcel_df.drop(
            columns=[conf.columns["cross_pred_model_id"]]
        )
    logger.debug("Read predict input file ready")

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
    with open(datacolumns_path) as f:
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
    assert input_parcel_classification_data_df is not None
    input_parcel_for_predict_df = input_parcel_df.join(
        input_parcel_classification_data_df, how="inner"
    )

    # Predict!
    logger.info(f"Predict using this model: {input_classifier_path}")
    if conf.classifier["classifier_type"].lower() == "keras_multilayer_perceptron":
        import cropclassification.predict.classification_keras as class_core_keras  # noqa: PLC0415

        class_core_keras.predict_proba(
            parcel_df=input_parcel_for_predict_df,
            classifier_path=input_classifier_path,
            output_parcel_predictions_path=output_predictions_path,
        )
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn  # noqa: PLC0415

        class_core_sklearn.predict_proba(
            parcel_df=input_parcel_for_predict_df,
            classifier_path=input_classifier_path,
            output_parcel_predictions_path=output_predictions_path,
        )
