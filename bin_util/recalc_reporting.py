import logging
from pathlib import Path

import cropclassification.helpers.config_helper as conf
from cropclassification.postprocess import classification_postprocess as class_post
from cropclassification.postprocess import classification_reporting as class_report
from cropclassification.preprocess import classification_preprocess as class_pre


def main():
    logging.basicConfig(level=logging.INFO)

    # Some variables that need to be choosen
    year = 2023
    if year == 2022:
        input_groundtruth_filename = "Prc_BEFL_2022_2024_05_13_groundtruth.tsv"
    elif year == 2023:
        input_groundtruth_filename = "Prc_BEFL_2023_2024_05_13_groundtruth.tsv"
    else:
        raise ValueError(f"invalid year: {year}")

    run_dir = Path(
        "X:/Monitoring/Markers/dev/2023_CROPGROUP/Run_2023-08-08_001_prd_groundtruth"
    )
    basedir = run_dir.parent.parent
    overrules = ["columns.class_declared2=NOT_AVAILABLE", "roi.roi_name=BEFL"]
    """
    input_groundtruth_filename = "Prc_BEFL_2022_2024_05_13_groundtruth.tsv"
    run_dir = Path(
        "X:/Monitoring/Markers/dev/2022_CROPGROUP/Run_2024-05-17_008_96.13_rf_min80pct"
    )
    basedir = run_dir.parent.parent
    overrules = []  # ["roi.roi_name=BEFL"]
    """
    # Init
    if not run_dir.exists():
        raise ValueError(f"run_dir does not exist: {run_dir}")
    configfiles_used = sorted((run_dir / "configfiles_used").glob("*.ini"))
    if len(configfiles_used) == 0:
        raise ValueError(f"No config files found in {run_dir / 'configfiles_used'}")

    # Check if the config files are prepended with an index. If not, raise, as the order
    # the config files are loaded is important.
    for configfile in configfiles_used:
        first_part = configfile.stem.split("_")[0]
        if not str.isnumeric(first_part):
            raise ValueError(
                "configfile_used file names must start with an index to know the "
                f"order, not: {configfile}"
            )

    conf.read_config(configfiles_used, default_basedir=basedir, overrules=overrules)

    input_dir = conf.paths.getpath("input_dir")
    input_groundtruth_path = input_dir / input_groundtruth_filename
    input_parcel_filetype = conf.calc_marker_params["input_parcel_filetype"]
    input_preprocessed_dir = conf.paths.getpath("input_preprocessed_dir")

    data_ext = conf.general["data_ext"]
    geofile_ext = conf.general["geofile_ext"]
    output_ext = conf.general["output_ext"]

    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    parcel_path = run_dir / f"{input_parcel_filename.stem}_parcel{data_ext}"
    input_parcel_path = input_dir / input_parcel_filename
    input_parcel_nogeo_path = (
        input_preprocessed_dir / f"{input_parcel_filename.stem}{data_ext}"
    )

    buffer = conf.timeseries.getfloat("buffer")
    imagedata_input_parcel_filename = (
        f"{input_parcel_filename.stem}_bufm{buffer:g}{geofile_ext}"
    )
    imagedata_input_parcel_path = (
        input_preprocessed_dir / imagedata_input_parcel_filename
    )

    # TODO: periode_name shouldn't be here!
    base_filename = f"{input_parcel_filename.stem}_bufm{buffer:g}_weekly"
    period_name = conf.marker.get("period_name", "weekly")
    timeseries_periodic_dir = conf.paths.getpath("timeseries_periodic_dir")
    timeseries_periodic_dir = (
        timeseries_periodic_dir / f"{imagedata_input_parcel_path.stem}_{period_name}"
    )

    refe_dir = conf.paths.getpath("refe_dir")
    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")
    classes_refe_path = refe_dir / classes_refe_filename

    parcel_train_path = run_dir / f"{base_filename}_parcel_train{data_ext}"
    parcel_test_path = run_dir / f"{base_filename}_parcel_test{data_ext}"
    parcel_predictions_proba_test_path = (
        run_dir / f"{base_filename}_predict_proba_test{data_ext}"
    )
    parcel_predictions_test_path = run_dir / f"{base_filename}_predict_test{data_ext}"
    parcel_predictions_test_geopath = (
        run_dir / f"{base_filename}_predict_test{geofile_ext}"
    )

    parcel_predictions_proba_all_path = (
        run_dir / f"{base_filename}_predict_proba_all{data_ext}"
    )
    parcel_predictions_all_path = run_dir / f"{base_filename}_predict_all{data_ext}"
    parcel_predictions_all_geopath = (
        run_dir / f"{base_filename}_predict_all{geofile_ext}"
    )

    min_parcels_in_class = conf.preprocess.getint("min_parcels_in_class")
    classtype_to_prepare = conf.preprocess["classtype_to_prepare"]

    # Prepare parcel input file
    class_pre.prepare_input(
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

    # Recalculate prediction consolidations
    # -------------------------------------
    # If it was necessary to train, there will be a test prediction... so postprocess it
    parcel_predictions_test_path = None
    parcel_predictions_test_geopath = None
    if parcel_test_path is not None and parcel_predictions_proba_test_path is not None:
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

    # Recalculate reports
    # -------------------
    if input_groundtruth_path is not None:
        groundtruth_path = (
            run_dir
            / f"{input_groundtruth_path.stem}_classes{input_groundtruth_path.suffix}"
        )
        class_pre.prepare_input(
            input_parcel_path=input_groundtruth_path,
            input_parcel_filetype=input_parcel_filetype,
            timeseries_periodic_dir=timeseries_periodic_dir,
            base_filename=base_filename,
            data_ext=data_ext,
            classtype_to_prepare=conf.preprocess["classtype_to_prepare_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            output_parcel_path=groundtruth_path,
            force=True,
        )

    # Print full reporting on the accuracy of the test dataset
    report_txt = Path(f"{parcel_predictions_test_path!s}_accuracy_report.txt")
    class_report.write_full_report(
        parcel_predictions_geopath=parcel_predictions_test_geopath,
        parcel_train_path=parcel_train_path,
        output_report_txt=report_txt,
        parcel_ground_truth_path=groundtruth_path,
        force=True,
    )

    # Print full reporting on the accuracy of the full dataset
    report_txt = Path(f"{parcel_predictions_all_path!s}_accuracy_report.txt")
    class_report.write_full_report(
        parcel_predictions_geopath=parcel_predictions_all_geopath,
        parcel_train_path=parcel_train_path,
        output_report_txt=report_txt,
        parcel_ground_truth_path=groundtruth_path,
        force=True,
    )


if __name__ == "__main__":
    main()
