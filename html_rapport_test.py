import os
import classification_reporting as class_report
import classification_preprocess as class_pre

base_dir = 'x:\\Monitoring\\Markers\\PlayGround\\JoeBro'        
input_dir = os.path.join(base_dir, 'InputData')  
class_base_dir = os.path.join(base_dir, f"HTML") 
input_groundtruth_csv = os.path.join(input_dir, "Prc_BEFL_2018_groundTruth.csv")
parcel_pixcount_csv = os.path.join(input_dir, "BEFL2018_bufm10_weekly_pixcount.csv")

parcel_predictions_all_csv = os.path.join(class_base_dir, f"BEFL2018_bufm10_weekly_predict_all.csv")
groundtruth_csv = os.path.join(class_base_dir, "Prc_BEFL_2018_groundTruth_classes.csv")
report_txt = os.path.join(class_base_dir, f"testje_accuracy_report.txt")
report_html = report_txt.replace(".txt", ".html")

if os.path.exists(report_txt):
    os.remove(report_txt)
if os.path.exists(report_html):
    os.remove(report_html)

'''class_pre.prepare_input(input_parcel_filepath=input_groundtruth_csv,
                            input_filetype='BEFL',
                            input_parcel_pixcount_csv=parcel_pixcount_csv,
                            input_classtype_to_prepare="MONITORING_LANDCOVER_GROUNDTRUTH",
                            output_parcel_filepath=groundtruth_csv,                            
                            force=True)'''

class_report.write_full_report(parcel_predictions_csv=parcel_predictions_all_csv,
                               output_report_txt=report_txt,
                               parcel_ground_truth_csv=groundtruth_csv,
                               force=True)