import shutil
from datetime import datetime
from pathlib import Path

import pytest

from cropclassification.helpers import config_helper as conf
from cropclassification.util import mosaic_util
from tests.test_helper import SampleData


def test_calc_periodic_mosaic_s2(tmp_path):
    # Prepare test data
    sample_dir = SampleData.marker_basedir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    # Init parameters
    image_profiles_path = test_dir / "_config" / "image_profiles.ini"
    imageprofiles = conf._get_image_profiles(image_profiles_path)
    output_image_roi_dir = test_dir / SampleData.image_dir.name / SampleData.roi_name

    # Make sure the s2-agri input file was copied
    s2_agri_output_path = (
        output_image_roi_dir
        / SampleData.image_s2_mean_path.parent.name
        / SampleData.image_s2_mean_path.name
    )
    assert s2_agri_output_path.exists()

    # Test
    result_infos = mosaic_util.calc_periodic_mosaic(
        roi_bounds=SampleData.roi_bounds,
        roi_crs=SampleData.roi_crs,
        start_date=SampleData.start_date,
        end_date=SampleData.end_date,
        output_base_dir=output_image_roi_dir,
        imageprofiles_to_get=["s2-agri-weekly", "s2-ndvi-weekly"],
        imageprofiles=imageprofiles,
        force=False,
    )

    assert result_infos is not None
    assert len(result_infos) == 2
    for result_info in result_infos:
        assert result_info["path"].exists()


def test_calc_periodic_mosaic_s1_local_index_dprvi(tmp_path):
    # Prepare test data
    sample_dir = SampleData.marker_basedir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    # Init parameters
    image_profiles_path = test_dir / "_config" / "image_profiles.ini"
    imageprofiles = conf._get_image_profiles(image_profiles_path)
    output_image_roi_dir = test_dir / SampleData.image_dir.name / SampleData.roi_name

    # Make sure the s1 input files were copied
    s1_asc_output_path = (
        output_image_roi_dir
        / SampleData.image_s1_asc_path.parent.name
        / SampleData.image_s1_asc_path.name
    )
    assert s1_asc_output_path.exists()
    s1_desc_output_path = (
        output_image_roi_dir
        / SampleData.image_s1_desc_path.parent.name
        / SampleData.image_s1_desc_path.name
    )
    assert s1_desc_output_path.exists()

    result_infos = mosaic_util.calc_periodic_mosaic(
        roi_bounds=SampleData.roi_bounds,
        roi_crs=SampleData.roi_crs,
        start_date=SampleData.start_date,
        end_date=SampleData.end_date,
        output_base_dir=output_image_roi_dir,
        imageprofiles_to_get=["s1-dprvi-asc-weekly", "s1-dprvi-desc-weekly"],
        imageprofiles=imageprofiles,
        force=False,
    )

    assert result_infos is not None
    assert len(result_infos) == 4
    for result_info in result_infos:
        assert result_info["path"].exists()


def test_ImageProfile_local():
    image_profile = mosaic_util.ImageProfile(
        name="s2-ndvi",
        satellite="s2",
        image_source="local",
        index_type="ndvi",
        bands=["ndvi"],
        base_image_profile="s2-agri",
    )
    assert image_profile.name == "s2-ndvi"
    assert image_profile.index_type == "ndvi"
    assert image_profile.base_imageprofile == "s2-agri"


@pytest.mark.parametrize(
    "exp_error, image_source, collection, bands, base_image_profile, index_type",
    [
        ("image_source='wrong' is not supported", "wrong", None, None, None, None),
        ("collection must be None", "local", "collection", None, "s2-agri", "ndvi"),
        ("index_type can't be None", "local", None, None, "s2-agri", None),
        ("base_image_profile can't be None", "local", None, None, None, "ndvi"),
        ("collection can't be None", "openeo", None, ["B"], None, None),
        ("collection can't be None", "openeo", None, ["B"], None, None),
        ("base_image_profile must be None", "openeo", "col", ["B"], "profile", None),
    ],
)
def test_ImageProfile_invalid(
    exp_error, image_source, collection, bands, base_image_profile, index_type
):
    with pytest.raises(ValueError, match=exp_error):
        mosaic_util.ImageProfile(
            name="test-name",
            satellite="s2",
            image_source=image_source,
            collection=collection,
            bands=bands,
            base_image_profile=base_image_profile,
            index_type=index_type,
        )


@pytest.mark.parametrize(
    "exp_error, image_source, period_name, period_days",
    [
        ("period_name must be None", "local", "weekly", 7),
        ("both period_name and period_days are None", "openeo", None, None),
    ],
)
def test_ImageProfile_period_info_invalid(
    exp_error, image_source, period_name, period_days
):
    with pytest.raises(ValueError, match=exp_error):
        mosaic_util.ImageProfile(
            name="test-name",
            satellite="s2",
            image_source=image_source,
            collection="test",
            bands=["A"],
            period_name=period_name,
            period_days=period_days,
        )


def test_ImageProfile_openeo():
    image_profile = mosaic_util.ImageProfile(
        name="s2-agri",
        satellite="s2",
        image_source="openeo",
        collection="TERRASCOPE_S2_TOC_V2",
        bands=["B01", "B02"],
        period_name="weekly",
    )
    assert image_profile.name == "s2-agri"
    assert image_profile.collection == "TERRASCOPE_S2_TOC_V2"
    assert image_profile.bands == ["B01", "B02"]


@pytest.mark.parametrize(
    "start_date, end_date, period_name, period_days, expected_nb_periods",
    [
        (datetime(2024, 1, 1), datetime(2024, 1, 17), None, 7, 2),
        (datetime(2024, 1, 1), datetime(2024, 1, 17), "weekly", None, 2),
        (datetime(2024, 1, 2), datetime(2024, 1, 17), "weekly", None, 1),
        (datetime(2024, 3, 3), datetime(2024, 3, 20), "weekly", None, 2),
        (datetime(2024, 1, 1), datetime(2024, 1, 30), "biweekly", None, 2),
        (datetime(2024, 1, 2), datetime(2024, 1, 30), "biweekly", None, 1),
    ],
)
def test_prepare_periods(
    start_date, end_date, period_name, period_days, expected_nb_periods
):
    periods = mosaic_util._prepare_periods(
        start_date, end_date, period_name=period_name, period_days=period_days
    )
    assert len(periods) == expected_nb_periods
    for period in periods:
        assert start_date <= period["start_date"]
        assert end_date >= period["end_date"]


@pytest.mark.parametrize(
    "exp_error, start_date, end_date, period_name, period_days",
    [
        (
            "period_name is None and period_days is not 7 or 14",
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            None,
            3,
        ),
        ("start_date == end_date", datetime(2024, 1, 1), datetime(2024, 1, 1), None, 7),
        (
            "start_date == end_date",
            datetime(2024, 1, 2),
            datetime(2024, 1, 26),
            "biweekly",
            None,
        ),
    ],
)
def test_prepare_periods_invalid(
    exp_error, start_date, end_date, period_name, period_days
):
    with pytest.raises(ValueError, match=exp_error):
        _ = mosaic_util._prepare_periods(
            start_date, end_date, period_name, period_days=period_days
        )


def test_prepare_mosaic_image_path():
    # Very basic test, as otherwise the tests just reimplements all code
    result_path = mosaic_util._prepare_mosaic_image_path(
        imageprofile="s2-agri-weekly",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        bands=["B01", "B02"],
        time_reducer="mean",
        output_base_dir=Path("/tmp"),
    )

    expected_path = Path(
        "/tmp/s2-agri-weekly/s2-agri-weekly_2024-01-01_2024-01-02_B01-B02_mean.tif"
    )
    assert result_path == expected_path
