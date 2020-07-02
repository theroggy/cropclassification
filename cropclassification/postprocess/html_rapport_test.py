import os
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.postprocess.classification_reporting as class_report
import cropclassification.preprocess.classification_preprocess as class_pre

base_dir = Path('x:/Monitoring/Markers/PlayGround/JoeBro')
input_dir = base_dir / 'InputData'
class_base_dir = base_dir / "HTML" 
input_groundtruth_filepath = input_dir / "Prc_BEFL_2018_groundTruth.csv"
parcel_pixcount_filepath = input_dir / "BEFL2018_bufm10_weekly_pixcount.csv"

parcel_predictions_all_filepath = class_base_dir / f"BEFL2018_bufm10_weekly_predict_all.csv"
groundtruth_filepath = class_base_dir / "Prc_BEFL_2018_groundTruth_classes.csv"
report_txt = class_base_dir / f"testje_accuracy_report.txt"
report_html = Path(str(report_txt).replace(".txt", ".html"))

if report_txt.exists():
    os.remove(report_txt)
if report_html.exists():
    os.remove(report_html)

'''class_pre.prepare_input(input_parcel_filepath=input_groundtruth_filepath,
                            input_filetype='BEFL',
                            input_parcel_pixcount_filepath=parcel_pixcount_filepath,
                            input_classtype_to_prepare="MONITORING_LANDCOVER_GROUNDTRUTH",
                            output_parcel_filepath=groundtruth_filepath,                            
                            force=True)'''

class_report.write_full_report(parcel_predictions_filepath=parcel_predictions_all_filepath,
                               output_report_txt=report_txt,
                               parcel_ground_truth_filepath=groundtruth_filepath,
                               force=True)