from copy import deepcopy
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
def test_get_image_profiles(sensor: str, exp_max_cloud_cover: float):
    config_path = SampleData.config_dir / "image_profiles.ini"
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
    assert profile.name == sensor
    if exp_max_cloud_cover is None:
        profile.max_cloud_cover is None
    else:
        assert profile.max_cloud_cover == exp_max_cloud_cover


def test_validate_image_profiles():
    conf._validate_image_profiles(IMAGEPROFILES)


def test_validate_image_profiles_invalid():
    test_profiles = deepcopy(IMAGEPROFILES)
    test_profiles["s2-ndvi"].base_imageprofile = "invalid"
    with pytest.raises(
        ValueError, match="base_image_profile='invalid' not found for profile"
    ):
        conf._validate_image_profiles(test_profiles)
