import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from cropclassification.preprocess import timeseries_calc_dias_onda_per_image as calc_ts

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_calc_stats_per_image_s1_bs(tmpdir):
    
    # Test raw version
    input_features_filepath = Path('/srv/data/playground/_inputdata_preprocessed/Prc_BEFL_2020_2020-07-01_bufm5.shp')
    input_image_filepaths = [Path('/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2020/07/11/S1A_IW_GRDH_1SDV_20200711T172447_20200711T172512_033410_03DF06_66B7_Orb_RBN_RTN_Cal_TC_20200714T100938.L1TC.CARD')]
    tmp_dir = Path(tmpdir) / 'raw'
    try:
        start_time = datetime.datetime.now()
        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column='UID',
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=tmp_dir,
                temp_dir=tmp_dir / 'tmp',
                log_dir=tmp_dir / 'log')
        print(f"calc_stats_per_image ready in {(datetime.datetime.now()-start_time).total_seconds():.2f}")
    except Exception as ex:
        raise Exception(f"Exception calculating for {input_features_filepath} on {input_image_filepaths}") from ex

    # Test .tif version
    input_image_filepaths = [Path('/mnt/NAS5/SAMPLES_CARD/S1A_IW_GRDH_1SDV_20200711T172447_20200711T172512_033410_03DF06_66B7_Orb_RBN_RTN_Cal_TC_20200722T143946.L1TC.CARD')]
    tmp_dir = Path(tmpdir) / 'tif'
    try:
        start_time = datetime.datetime.now()
        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column='UID',
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=tmp_dir,
                temp_dir=tmp_dir / 'tmp',
                log_dir=tmp_dir / 'log')
        print(f"calc_stats_per_image ready in {(datetime.datetime.now()-start_time).total_seconds():.2f}")
    except Exception as ex:
        raise Exception(f"Exception calculating for {input_features_filepath} on {input_image_filepaths}") from ex

if __name__ == '__main__':
    import tempfile
    tmpdir = '/srv/data/playground/tmp'#tempfile.gettempdir()
    print(f"tmpdir used for test: {tmpdir}")
    test_calc_stats_per_image_s1_bs(tmpdir)
