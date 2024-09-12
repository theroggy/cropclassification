"""
Main script to do a classification.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

# Import geofilops here already, if tensorflow is loaded first leads to dll load errors
import geofileops as gfo
import pandas as pd
import pyproj

import cropclassification.helpers.pandas_helper as pdh
from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import dir_helper, log_helper
from cropclassification.helpers import model_helper as mh
from cropclassification.postprocess import classification_postprocess as class_post
from cropclassification.postprocess import classification_reporting as class_report
from cropclassification.predict import classification, data_balancing
from cropclassification.preprocess import _timeseries_helper as ts_helper
from cropclassification.preprocess import prepare_input
from cropclassification.preprocess import timeseries as ts

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------


def calc_marker_task(
    config_paths: list[Path],
    default_basedir: Path,
    config_overrules: list[str] = [],
):
    """
    Runs a marker using the setting in the config_paths.

    Args:
        config_paths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config
            file to.
        config_overrules (List[str], optional): list of config options that will
            overrule other ways to supply configuration.
            They should be specified as a list of
            "<section>.<parameter>=<value>" strings. Defaults to [].

    Raises:
        Exception: [description]
        Exception: [description]
    """
    # Read the configuration files
    conf.read_config(
        config_paths, default_basedir=default_basedir, overrules=config_overrules
    )

    # Create run dir to be used for the results
    reuse_last_run_dir = conf.calc_marker_params.getboolean("reuse_last_run_dir")
    reuse_last_run_dir_config = conf.calc_marker_params.getboolean(
        "reuse_last_run_dir_config"
    )
    run_dir = dir_helper.create_run_dir(
        conf.paths.getpath("marker_dir"), reuse_last_run_dir
    )
    print(run_dir)
    if not run_dir.exists():
        os.makedirs(run_dir)

    # Main initialisation of the logging
    log_level = conf.general.get("log_level")
    logger = log_helper.main_log_init(run_dir, __name__, log_level)
    logger.info(f"Run dir with reuse_last_run_dir: {reuse_last_run_dir}, {run_dir}")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # If running in conda, export the environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env is not None:
        environment_yml_path = run_dir / f"{conda_env}.yml"
        logger.info(f"Export conda environment used to {environment_yml_path}")
        os.system(f"conda env export > {environment_yml_path}")

    # If the config needs to be reused as well, load it, else write it
    config_used_path = run_dir / "config_used.ini"
    if (
        reuse_last_run_dir
        and reuse_last_run_dir_config
        and run_dir.exists()
        and config_used_path.exists()
    ):
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
        with open(config_used_path, "w") as config_used_file:
            conf.config.write(config_used_file)
    else:
        # Copy the config files to a config dir for later notice
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
            with open(config_used_path, "w") as config_used_file:
                conf.config.write(config_used_file)

    # Read the info about the run
    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    input_parcel_filetype = conf.calc_marker_params["input_parcel_filetype"]
    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")
    input_groundtruth_filename = conf.calc_marker_params.getpath(
        "input_groundtruth_filename"
    )
    input_model_to_use_relativepath = conf.calc_marker_params.getpath(
        "input_model_to_use_relativepath"
    )

    # Prepare input paths
    if input_model_to_use_relativepath is not None:
        input_model_to_use_path = (
            conf.paths.getpath("model_dir") / input_model_to_use_relativepath
        )
        if not input_model_to_use_path.exists():
            raise Exception(
                "Input file input_model_to_use_path doesn't exist: "
                f"{input_model_to_use_path}"
            )
    else:
        input_model_to_use_path = None

    input_dir = conf.paths.getpath("input_dir")
    input_parcel_path = input_dir / input_parcel_filename
    if input_groundtruth_filename is not None:
        input_groundtruth_path = input_dir / input_groundtruth_filename
    else:
        input_groundtruth_path = None

    refe_dir = conf.paths.getpath("refe_dir")
    classes_refe_path = refe_dir / classes_refe_filename

    # Check if the necessary input files exist...
    for path in [classes_refe_path, input_parcel_path]:
        if path is not None and not path.exists():
            message = f"Input file doesn't exist, so STOP: {path}"
            logger.critical(message)
            raise ValueError(message)

    # Get some general config
    data_ext = conf.general["data_ext"]
    output_ext = conf.general["output_ext"]
    geofile_ext = conf.general["geofile_ext"]

    # -------------------------------------------------------------
    # The real work
    # -------------------------------------------------------------
    # STEP 1: prepare parcel data for classification and image data extraction
    # -------------------------------------------------------------

    # Prepare the input data for optimal image data extraction:
    #    1) apply a negative buffer on the parcel to evade mixels
    #    2) remove features that became null because of buffer
    input_preprocessed_dir = conf.paths.getpath("input_preprocessed_dir")
    buffer = conf.timeseries.getfloat("buffer")
    input_parcel_nogeo_path = (
        input_preprocessed_dir / f"{input_parcel_filename.stem}{data_ext}"
    )
    imagedata_input_parcel_filename = (
        f"{input_parcel_filename.stem}_bufm{buffer:g}{geofile_ext}"
    )
    imagedata_input_parcel_path = (
        input_preprocessed_dir / imagedata_input_parcel_filename
    )
    ts_helper.prepare_input(
        input_parcel_path=input_parcel_path,
        output_imagedata_parcel_input_path=imagedata_input_parcel_path,
        output_parcel_nogeo_path=input_parcel_nogeo_path,
    )

    # STEP 2: Get the timeseries data needed for the classification
    # -------------------------------------------------------------
    # Get the time series data (eg. S1, S2,...) to be used for the classification
    # Result: data is put in files in timeseries_periodic_dir, in one file per
    #         date/period
    timeseries_periodic_dir = conf.paths.getpath("timeseries_periodic_dir")
    timeseries_periodic_dir /= f"{imagedata_input_parcel_path.stem}"
    start_date = datetime.fromisoformat(conf.period["start_date"])
    end_date = datetime.fromisoformat(conf.period["end_date"])
    images_to_use = conf.parse_image_config(conf.images["images"])
    parceldata_aggregations_to_use = conf.marker.getlist(
        "parceldata_aggregations_to_use"
    )
    ts.calc_timeseries_data(
        input_parcel_path=imagedata_input_parcel_path,
        roi_bounds=tuple(conf.roi.getlistfloat("roi_bounds")),
        roi_crs=pyproj.CRS.from_user_input(conf.roi.get("roi_crs")),
        start_date=start_date,
        end_date=end_date,
        images_to_use=images_to_use,
        timeseries_periodic_dir=timeseries_periodic_dir,
    )

    # STEP 3: Preprocess all data needed for the classification
    # -------------------------------------------------------------
    # Prepare the basic input file with the classes that will be classified to.
    # Remarks:
    #    - this is typically specific for the input dataset and result wanted!!!
    #    - the result is/should be a file with the following columns
    #           - id (=id_column): unique ID for each parcel
    #           - classname (=class_column): the class that must
    #             be classified to.
    #             Remarks: - if in classes_to_ignore_for_train, class not used for train
    #                      - if in classes_to_ignore, the class will be ignored
    #           - pixcount:
    #             the number of S1/S2 pixels in the parcel.
    #             Is -1 if the parcel doesn't have any S1/S2 data.
    classtype_to_prepare = conf.preprocess["classtype_to_prepare"]
    min_parcels_in_class = conf.preprocess.getint("min_parcels_in_class")
    parcel_path = run_dir / f"{input_parcel_filename.stem}_parcel{data_ext}"
    base_filename = f"{input_parcel_filename.stem}_bufm{buffer:g}_weekly"
    prepare_input.prepare(
        input_parcel_path=input_parcel_nogeo_path,
        input_parcel_filetype=input_parcel_filetype,
        timeseries_periodic_dir=timeseries_periodic_dir,
        base_filename=base_filename,
        data_ext=data_ext,
        classtype_to_prepare=classtype_to_prepare,
        classes_refe_path=classes_refe_path,
        min_parcels_in_class=min_parcels_in_class,
        output_parcel_path=parcel_path,
    )

    # Collect all data needed to do the classification in one input file
    parcel_classification_data_path = (
        run_dir / f"{base_filename}_parcel_classdata{data_ext}"
    )
    ts.collect_and_prepare_timeseries_data(
        input_parcel_path=input_parcel_nogeo_path,
        timeseries_dir=timeseries_periodic_dir,
        base_filename=base_filename,
        output_path=parcel_classification_data_path,
        start_date=start_date,
        end_date=end_date,
        images_to_use=images_to_use,
        parceldata_aggregations_to_use=parceldata_aggregations_to_use,
    )

    # STEP 4: Train and test if necessary... and predict
    # -------------------------------------------------------------
    markertype = conf.marker.get("markertype")
    parcel_predictions_proba_all_path = (
        run_dir / f"{base_filename}_predict_proba_all{data_ext}"
    )
    parcel_predictions_proba_test_path = (
        run_dir / f"{base_filename}_predict_proba_test{data_ext}"
    )
    cross_pred_models = conf.classifier.get("cross_pred_models", 1)
    cross_pred_models = (
        1
        if cross_pred_models is None or int(cross_pred_models) < 1
        else int(cross_pred_models)
    )
    classifier_ext = conf.classifier["classifier_ext"]
    cross_pred_model_id_column = conf.columns["cross_pred_model_id"]
    parcel_test_path = None
    parcel_train_path = None

    # If the predictions file doesn't exist, do the classification
    if not parcel_predictions_proba_all_path.exists():
        if cross_pred_models > 1:
            # Use "cross models": train multiple models, so prediction of a parcel is
            # always done with a model where the parcel wasn't used to train it.

            # Assign each parcel to a model where it will be predicted with.
            gfo.add_column(
                path=parcel_path,
                name=cross_pred_model_id_column,
                type="int",
                expression=f"ABS(RANDOM() % {cross_pred_models})",
            )
            if input_model_to_use_path is not None:
                raise ValueError(
                    "cross_pred_models not supported with input_model_to_use_path"
                )

        pred_test_files = []
        pred_all_files = []
        for cross_pred_model_idx in range(cross_pred_models):
            if cross_pred_models == 1:
                # Only one model, so no need to create subdirectory
                model_dir = run_dir

                # Output can be the final file immediately
                parcel_preds_proba_test_model_path = parcel_predictions_proba_test_path
                parcel_preds_proba_all_model_path = parcel_predictions_proba_all_path

                # No need to filter percels based on cross model indexes
                training_query = None
                predict_query = None
            else:
                # Create a subfolder for each model
                model_dir = run_dir / f"cross_pred_model_{cross_pred_model_idx}"
                model_dir.mkdir(parents=True, exist_ok=True)

                # Temporary output files in the subdirectory
                parcel_preds_proba_test_model_path = (
                    model_dir / parcel_predictions_proba_test_path.name
                )
                pred_test_files.append(parcel_preds_proba_test_model_path)
                parcel_preds_proba_all_model_path = (
                    model_dir / parcel_predictions_proba_all_path.name
                )
                pred_all_files.append(parcel_preds_proba_all_model_path)

                # The training will be done on all parcels with another index
                training_cross_pred_model_ids = [
                    idx
                    for idx in range(cross_pred_models)
                    if idx != cross_pred_model_idx
                ]
                training_query = (
                    f"{cross_pred_model_id_column} in {training_cross_pred_model_ids}"
                )
                predict_query = (
                    f"{cross_pred_model_id_column} == {cross_pred_model_idx}"
                )
                # TODO: is there a use of keeping test data seperate if
                # cross-prediction-models are used?

            # Check if a model exists already
            if input_model_to_use_path is None:
                best_model = mh.get_best_model(model_dir, acc_metric_mode="min")
                if best_model is not None:
                    input_model_to_use_path = best_model["path"]

            # If we cannot reuse an existing model, train one!
            parcel_train_path = model_dir / f"{base_filename}_parcel_train{data_ext}"
            parcel_test_path = model_dir / f"{base_filename}_parcel_test{data_ext}"
            classifier_basepath = model_dir / f"{markertype}_01_mlp{classifier_ext}"
            if input_model_to_use_path is None:
                # Create the training sample.
                # Remark: this creates a list of representative test parcel + a list of
                # (candidate) training parcel
                balancing_strategy = conf.marker["balancing_strategy"]
                data_balancing.create_train_test_sample(
                    input_parcel_path=parcel_path,
                    output_parcel_train_path=parcel_train_path,
                    output_parcel_test_path=parcel_test_path,
                    balancing_strategy=balancing_strategy,
                    training_query=training_query,
                )

                # Train the classifier and output predictions
                classification.train_test_predict(
                    input_parcel_train_path=parcel_train_path,
                    input_parcel_test_path=parcel_test_path,
                    input_parcel_all_path=parcel_path,
                    input_parcel_classification_data_path=parcel_classification_data_path,
                    output_classifier_basepath=classifier_basepath,
                    output_predictions_test_path=parcel_preds_proba_test_model_path,
                    output_predictions_all_path=parcel_preds_proba_all_model_path,
                    predict_query=predict_query,
                )
            else:
                # There exists already a classifier, so just use it!
                classification.predict(
                    input_parcel_path=parcel_path,
                    input_parcel_classification_data_path=parcel_classification_data_path,
                    input_classifier_basepath=classifier_basepath,
                    input_classifier_path=input_model_to_use_path,
                    output_predictions_path=parcel_preds_proba_all_model_path,
                    predict_query=predict_query,
                )

        # Merge all prediction to single output file
        if cross_pred_models > 1:
            # Merge all "all" predictions to single output file.
            # The "test" predictions are not very useful when cross prediction models
            # are used, as the "all" predictions are independent of the training.
            pred_all_df = None
            for path in pred_all_files:
                df = pdh.read_file(path)
                pred_all_df = pd.concat([pred_all_df, df], ignore_index=True)
            pdh.to_file(pred_all_df, parcel_predictions_proba_all_path, index=False)

    # STEP 5: if necessary, do extra postprocessing
    # -------------------------------------------------------------
    """if postprocess_to_groups is not None:
        # TODO
    """

    # STEP 6: do the default, mandatory postprocessing
    # -------------------------------------------------------------
    # If it was necessary to train, there will be a test prediction... so postprocess it
    parcel_predictions_test_path = None
    parcel_predictions_test_geopath = None
    if (
        input_model_to_use_path is None
        and parcel_test_path is not None
        and parcel_predictions_proba_test_path is not None
    ):
        parcel_predictions_test_path = (
            run_dir / f"{base_filename}_predict_test{data_ext}"
        )
        parcel_predictions_test_geopath = (
            run_dir / f"{base_filename}_predict_test{geofile_ext}"
        )
        class_post.calc_top3_and_consolidation(
            input_parcel_path=parcel_test_path,
            input_parcel_probabilities_path=parcel_predictions_proba_test_path,
            input_parcel_geopath=input_parcel_path,
            output_predictions_path=parcel_predictions_test_path,
            output_predictions_geopath=parcel_predictions_test_geopath,
        )

    # Postprocess predictions
    parcel_predictions_all_path = run_dir / f"{base_filename}_predict_all{data_ext}"
    parcel_predictions_all_geopath = (
        run_dir / f"{base_filename}_predict_all{geofile_ext}"
    )
    parcel_predictions_all_output_path = (
        run_dir / f"{base_filename}_predict_all_output{output_ext}"
    )
    class_post.calc_top3_and_consolidation(
        input_parcel_path=parcel_path,
        input_parcel_probabilities_path=parcel_predictions_proba_all_path,
        input_parcel_geopath=input_parcel_path,
        output_predictions_path=parcel_predictions_all_path,
        output_predictions_geopath=parcel_predictions_all_geopath,
        output_predictions_output_path=parcel_predictions_all_output_path,
    )

    # STEP 7: Report on the accuracy, incl. ground truth
    # -------------------------------------------------------------
    # Preprocess the ground truth data if it is provided
    groundtruth_path = None
    if input_groundtruth_path is not None:
        groundtruth_path = (
            run_dir
            / f"{input_groundtruth_path.stem}_classes{input_groundtruth_path.suffix}"
        )
        prepare_input.prepare(
            input_parcel_path=input_groundtruth_path,
            input_parcel_filetype=input_parcel_filetype,
            timeseries_periodic_dir=timeseries_periodic_dir,
            base_filename=base_filename,
            data_ext=data_ext,
            classtype_to_prepare=conf.preprocess["classtype_to_prepare_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            output_parcel_path=groundtruth_path,
        )

    # If we trained a model, there is a test prediction we want to report on
    if input_model_to_use_path is None and parcel_predictions_test_geopath is not None:
        # Print full reporting on the accuracy of the test dataset
        report_txt = Path(f"{parcel_predictions_test_path!s}_accuracy_report.txt")
        class_report.write_full_report(
            parcel_predictions_geopath=parcel_predictions_test_geopath,
            output_report_txt=report_txt,
            parcel_ground_truth_path=groundtruth_path,
            parcel_train_path=parcel_train_path,
            parcel_classification_data_path=parcel_classification_data_path,
        )

    # Print full reporting on the accuracy of the full dataset
    report_txt = Path(f"{parcel_predictions_all_path!s}_accuracy_report.txt")
    class_report.write_full_report(
        parcel_predictions_geopath=parcel_predictions_all_geopath,
        output_report_txt=report_txt,
        parcel_ground_truth_path=groundtruth_path,
        parcel_train_path=parcel_train_path,
        parcel_classification_data_path=parcel_classification_data_path,
    )

    logging.shutdown()
