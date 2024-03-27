from pathlib import Path
import pytest

from cropclassification.helpers import config_helper as conf

sampleprojects_dir = Path(__file__).resolve().parent.parent / "sample_projects"
project_dir = sampleprojects_dir / "markers"
config_dir = project_dir / "_config"


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
    config_path = config_dir / "image_profiles.ini"
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
