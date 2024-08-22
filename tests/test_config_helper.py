from copy import deepcopy
from pathlib import Path
from typing import Optional

import pytest

from cropclassification.helpers import config_helper as conf
from tests.test_helper import IMAGEPROFILES, SampleData


@pytest.mark.parametrize(
    "sensor, exp_max_cloud_cover",
    [
        ("s2-agri-weekly", 80),
        ("s2-scl-weekly", 80),
        ("s2-ndvi-weekly", None),
        ("s1-grd-sigma0-asc-weekly", None),
        ("s1-grd-sigma0-desc-weekly", None),
        ("s1-coh-weekly", None),
    ],
)
def test_get_image_profiles(sensor: str, exp_max_cloud_cover: Optional[float]):
    config_path = SampleData.config_dir / "image_profiles.ini"
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
    assert profile.name == sensor
    if exp_max_cloud_cover is None:
        assert profile.max_cloud_cover is None
    else:
        assert profile.max_cloud_cover == exp_max_cloud_cover


def test_read_config():
    config_paths = [
        SampleData.config_dir / "cropgroup.ini",
        SampleData.tasks_dir / "local_overrule.ini",
    ]
    conf.read_config(
        config_paths=config_paths,
        default_basedir=SampleData.marker_basedir,
    )

    assert conf.marker["markertype"] == "CROPGROUP"
    assert conf.calc_marker_params["country_code"] == "BEFL"
    assert conf.marker["roi_name"] == "roi_test"


@pytest.mark.parametrize(
    "error, config_paths, default_basedir, preload_defaults, overrules",
    [
        ("config_paths is None and preload_defaults is False", None, None, False, []),
        ("Config file doesn't exist", Path("not_existing_config.ini"), None, False, []),
        ("default_basedir does not exist", None, Path("not_existing_dir"), True, []),
        (
            "dirs.data_dir is relative, but no default_basedir supplied",
            None,
            None,
            True,
            [],
        ),
        (
            "calc_marker_params.country_code must be overridden",
            None,
            SampleData.marker_basedir,
            True,
            [],
        ),
    ],
)
def test_read_config_invalid(
    error, config_paths, default_basedir, preload_defaults, overrules
):
    with pytest.raises(ValueError, match=error):
        conf.read_config(
            config_paths=config_paths,
            default_basedir=default_basedir,
            preload_defaults=preload_defaults,
            overrules=overrules,
        )


def test_read_config_overrule():
    config_paths = SampleData.config_dir / "cropgroup.ini"
    conf.read_config(
        config_paths=config_paths,
        default_basedir=SampleData.marker_basedir,
        overrules=[
            "calc_marker_params.country_code=COUNTRY_TEST",
            "marker.roi_name=ROI_NAME_TEST",
        ],
    )

    assert conf.marker["markertype"] == "CROPGROUP"
    assert conf.calc_marker_params["country_code"] == "COUNTRY_TEST"
    assert conf.marker["roi_name"] == "ROI_NAME_TEST"


def test_validate_image_profiles():
    conf._validate_image_profiles(IMAGEPROFILES)


def test_validate_image_profiles_invalid():
    test_profiles = deepcopy(IMAGEPROFILES)
    test_profiles["s2-ndvi"].base_imageprofile = "invalid"
    with pytest.raises(
        ValueError, match="base_image_profile='invalid' not found for profile"
    ):
        conf._validate_image_profiles(test_profiles)
