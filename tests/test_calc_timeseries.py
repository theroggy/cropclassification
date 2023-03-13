from datetime import datetime
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cropclassification.preprocess import _timeseries_calc_per_image as calc_ts


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_calc_stats_per_image_s1_bs(tmpdir):

    # Test raw version
    input_features_path = Path(
        "/srv/data/playground/_inputdata_preprocessed/"
        "Prc_BEFL_2020_2020-07-01_bufm5.shp"
    )
    input_image_paths = [
        Path(
            "/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2020/07/11/"
            "S1A_IW_GRDH_1SDV_20200711T172447_20200711T172512_033410_03DF06_66B7_"
            "Orb_RBN_RTN_Cal_TC_20200714T100938.L1TC.CARD"
        )
    ]
    tmp_dir = Path(tmpdir) / "raw"
    try:
        start_time = datetime.now()
        calc_ts.calc_stats_per_image(
            features_path=input_features_path,
            id_column="UID",
            image_paths=input_image_paths,
            bands=["VV", "VH"],
            output_dir=tmp_dir,
            temp_dir=tmp_dir / "tmp",
            log_dir=tmp_dir / "log",
            log_level=logging.INFO,
        )
        print(
            "calc_stats_per_image ready in "
            f"{(datetime.now()-start_time).total_seconds():.2f}"
        )
    except Exception as ex:
        raise Exception(
            f"Exception calculating for {input_features_path} on {input_image_paths}"
        ) from ex

    # Test .tif version
    input_image_paths = [
        Path(
            "/mnt/NAS5/SAMPLES_CARD/S1A_IW_GRDH_1SDV_20200711T172447_20200711T172512_"
            "033410_03DF06_66B7_Orb_RBN_RTN_Cal_TC_20200722T143946.L1TC.CARD"
        )
    ]
    tmp_dir = Path(tmpdir) / "tif"
    try:
        start_time = datetime.now()
        calc_ts.calc_stats_per_image(
            features_path=input_features_path,
            id_column="UID",
            image_paths=input_image_paths,
            bands=["VV", "VH"],
            output_dir=tmp_dir,
            temp_dir=tmp_dir / "tmp",
            log_dir=tmp_dir / "log",
            log_level=logging.INFO,
        )
        print(
            "calc_stats_per_image ready in "
            f"{(datetime.now()-start_time).total_seconds():.2f}"
        )
    except Exception as ex:
        raise Exception(
            f"Exception calculating for {input_features_path} on {input_image_paths}"
        ) from ex


if __name__ == "__main__":
    tmpdir = "/srv/data/playground/tmp"  # tempfile.gettempdir()
    print(f"tmpdir used for test: {tmpdir}")
    test_calc_stats_per_image_s1_bs(tmpdir)
