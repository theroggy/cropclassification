from pathlib import Path
import pytest

from cropclassification.helpers import config_helper as conf


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
    data_dir = Path(__file__).resolve().parent / "data"
    config_path = data_dir / "image_profiles.ini"
    config_path = conf.marker.getpath("image_profiles_config_filepath")
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
