import os
from pathlib import Path

import cropclassification.postprocess.classification_reporting as class_report

base_dir = Path("x:/Monitoring/Markers/PlayGround/JoeBro")
input_dir = base_dir / "InputData"
class_base_dir = base_dir / "HTML"
input_groundtruth_path = input_dir / "Prc_BEFL_2018_groundTruth.csv"
parcel_pixcount_path = input_dir / "BEFL2018_bufm10_weekly_pixcount.csv"

parcel_predictions_all_path = class_base_dir / "BEFL2018_bufm10_weekly_predict_all.csv"
groundtruth_path = class_base_dir / "Prc_BEFL_2018_groundTruth_classes.csv"
report_txt = class_base_dir / "testje_accuracy_report.txt"
report_html = Path(str(report_txt).replace(".txt", ".html"))

if report_txt.exists():
    os.remove(report_txt)
if report_html.exists():
    os.remove(report_html)

class_report.write_full_report(
    parcel_predictions_geopath=parcel_predictions_all_path,
    parcel_train_path=None,
    output_report_txt=report_txt,
    parcel_ground_truth_path=groundtruth_path,
    force=True,
)
