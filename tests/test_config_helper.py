import pytest

from cropclassification.helpers import config_helper as conf
from tests import test_helper


@pytest.mark.parametrize(
    "sensor",
    [
        ("s2-agri"),
        ("s2-scl"),
        ("s2-ndvi"),
        ("s1-grd-sigma0-asc"),
        ("s1-grd-sigma0-desc"),
        ("s1-coh"),
    ],
)
def test_get_image_profiles(sensor: str):
    config_path = test_helper.SampleDirs.config_dir / "image_profiles.ini"
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
