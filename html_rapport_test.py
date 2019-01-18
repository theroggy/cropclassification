# -*- coding: utf-8 -*-
import os
import classification_reporting as class_report


base_dir = 'X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-05-04_Monitoring_Classificatie'        
input_dir = os.path.join(base_dir, 'InputData')  
class_base_dir = os.path.join(base_dir, f"HTML") 
parcel_predictions_all_csv = os.path.join(class_base_dir, f"BEFL2018_bufm10_weekly_predict_all (2).csv")
groundtruth_csv = os.path.join(input_dir, "Prc_BEFL_2018_groundTruth.csv")
report_txt = os.path.join(class_base_dir, f"testje_accuracy_report.txt")
report_html = report_txt.replace(".txt", ".html")

if os.path.exists(report_txt):
    os.remove(report_txt)
if os.path.exists(report_html):
    os.remove(report_html)

class_report.write_full_report(parcel_predictions_csv=parcel_predictions_all_csv,
                               output_report_txt=report_txt,
                               parcel_ground_truth_csv=None,#groundtruth_csv,
                               force=True)
