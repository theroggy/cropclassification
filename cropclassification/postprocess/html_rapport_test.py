"""Test script to generate a classification report for a classification."""

from pathlib import Path

import cropclassification.postprocess.classification_reporting as class_report
from cropclassification.helpers import config_helper as conf

base_dir = Path(r"x:/Monitoring/Markers/DEV")
marker_dir = base_dir / r"2025_LANDCOVER-EARLY\Run_2025-04-29_001"
input_dir = base_dir / "_inputdata"
ts_dir = base_dir / "_ts_periodic"
refe_dir = base_dir / "_refe"

parcel_pixcount_path = (
    input_dir / "Prc_BEFL_2025_2025-04-18_bufm5_weekly_pixcount.sqlite"
)
# input_groundtruth_path = input_dir / "Prc_BEFL_2018_groundTruth.csv"
# groundtruth_path = refe_dir / "Prc_BEFL_2018_groundTruth_classes.csv"

conf.read_config(Path(marker_dir / "config_used.ini"))

parcel_predictions_all_path = (
    marker_dir / "Prc_BEFL_2025_2025-04-18_bufm5_weekly_predict_all.sqlite"
)
report_txt = marker_dir / "test_accuracy_report.txt"
report_html = Path(str(report_txt).replace(".txt", ".html"))

if report_txt.exists():
    report_txt.unlink()
if report_html.exists():
    report_html.unlink()

class_report.write_full_report(
    parcel_predictions_geopath=parcel_predictions_all_path,
    parcel_train_path=None,
    output_report_txt=report_txt,
    # parcel_ground_truth_path=groundtruth_path,
    force=True,
)
