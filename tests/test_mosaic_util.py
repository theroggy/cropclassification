from datetime import datetime
from pathlib import Path
import pytest
import shutil

from cropclassification.util import mosaic_util
from tests import test_helper


def test_calc_periodic_mosaic(tmp_path):
    sample_dir = test_helper.SampleDirs.marker_basedir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)
    output_base_dir = test_dir / test_helper.SampleDirs.image_dir.name / "roi_test"
    days_per_period = 7
    start_date = datetime(2024, 3, 4)
    end_date = datetime(2024, 3, 11)

    result_infos = mosaic_util.calc_periodic_mosaic(
        roi_bounds=[160_000, 188_000, 160_500, 188_500],
        roi_crs=31370,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        time_dimension_reducer="mean",
        output_base_dir=output_base_dir,
        imageprofiles_to_get=["s2-agri", "s2-ndvi"],
        imageprofiles=test_helper.IMAGEPROFILES,
        force=False,
    )

    assert result_infos is not None
    assert len(result_infos) == 2
    for result_info in result_infos:
        assert result_info["path"].exists()


@pytest.mark.parametrize(
    "start_date, end_date, days_per_period, expected_nb_periods",
    [(datetime(2024, 1, 1), datetime(2024, 1, 17), 7, 2)],
)
def test_prepare_periods(start_date, end_date, days_per_period, expected_nb_periods):
    result = mosaic_util._prepare_periods(
        start_date, end_date, days_per_period=days_per_period
    )
    assert len(result) == expected_nb_periods


def test_prepare_mosaic_image_path():
    # Very basic test, as otherwise the tests just reimplements all code
    result_path = mosaic_util._prepare_mosaic_image_path(
        imageprofile="s2-agri",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        bands=["B01", "B02"],
        time_dimension_reducer="mean",
        output_base_dir=Path("/tmp"),
    )

    expected_path = Path("/tmp/s2-agri/s2-agri_2024-01-01_2024-01-02_B01-B02_mean.tif")
    assert result_path == expected_path
