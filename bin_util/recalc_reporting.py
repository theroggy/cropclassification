import logging
from pathlib import Path

import cropclassification.helpers.config_helper as conf
from cropclassification.preprocess import classification_preprocess as class_pre
from cropclassification.postprocess import classification_reporting as class_report


def main():
    logging.basicConfig(level=logging.INFO)

    # Some variables that need to be choosen

    input_groundtruth_filename = "Prc_BEFL_2022_2024_05_13_groundtruth.tsv"
    run_dir = Path(
        "X:/Monitoring/Markers/dev/2022_CROPGROUP/Run_2022-08-02_001_groundtruth"
    )
    basedir = run_dir.parent.parent
    overrules = [
        "columns.class_declared2=NOT_AVAILABLE",
        "marker.roi_name=BEFL",
    ]
    """
    input_groundtruth_filename = "Prc_BEFL_2022_2024_05_13_groundtruth.tsv"
    run_dir = Path(
        "X:/Monitoring/Markers/dev/2022_CROPGROUP/Run_2024-05-17_001_96.07_min80pct"
    )
    basedir = run_dir.parent.parent
    overrules = []  # ["marker.roi_name=BEFL"]
    """
    # Init
    configfiles_used = sorted((run_dir / "configfiles_used").glob("*.ini"))

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

    input_dir = conf.dirs.getpath("input_dir")
    input_groundtruth_path = input_dir / input_groundtruth_filename
    input_parcel_filetype = conf.calc_marker_params["input_parcel_filetype"]
    input_preprocessed_dir = conf.dirs.getpath("input_preprocessed_dir")

    data_ext = conf.general["data_ext"]
    geofile_ext = conf.general["geofile_ext"]

    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    buffer = conf.marker.getint("buffer")
    imagedata_input_parcel_filename = (
        f"{input_parcel_filename.stem}_bufm{buffer}{geofile_ext}"
    )
    imagedata_input_parcel_path = (
        input_preprocessed_dir / imagedata_input_parcel_filename
    )

    # TODO: periode_name shouldn't be here!
    base_filename = f"{input_parcel_filename.stem}_bufm{buffer}_weekly"
    period_name = conf.marker.get("period_name", "weekly")
    timeseries_periodic_dir = conf.dirs.getpath("timeseries_periodic_dir")
    timeseries_periodic_dir = (
        timeseries_periodic_dir / f"{imagedata_input_parcel_path.stem}_{period_name}"
    )

    refe_dir = conf.dirs.getpath("refe_dir")
    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")
    classes_refe_path = refe_dir / classes_refe_filename

    parcel_predictions_test_path = run_dir / f"{base_filename}_predict_test{data_ext}"
    parcel_predictions_test_geopath = (
        run_dir / f"{base_filename}_predict_test{geofile_ext}"
    )
    parcel_predictions_all_path = run_dir / f"{base_filename}_predict_all{data_ext}"
    parcel_predictions_all_geopath = (
        run_dir / f"{base_filename}_predict_all{geofile_ext}"
    )

    # Run
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
            output_parcel_path=groundtruth_path,
        )

    # Print full reporting on the accuracy of the test dataset
    report_txt = Path(f"{str(parcel_predictions_test_path)}_accuracy_report.txt")
    class_report.write_full_report(
        parcel_predictions_geopath=parcel_predictions_test_geopath,
        output_report_txt=report_txt,
        parcel_ground_truth_path=groundtruth_path,
        force=True,
    )

    # Print full reporting on the accuracy of the full dataset
    report_txt = Path(f"{str(parcel_predictions_all_path)}_accuracy_report.txt")
    class_report.write_full_report(
        parcel_predictions_geopath=parcel_predictions_all_geopath,
        output_report_txt=report_txt,
        parcel_ground_truth_path=groundtruth_path,
        force=True,
    )


if __name__ == "__main__":
    main()
