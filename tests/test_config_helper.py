import pytest

from cropclassification.helpers import config_helper as conf


@pytest.mark.parametrize(
    "raster_profile",
    [
        ("s2-agri"),
        ("s2-scl"),
        ("s2-ndvi"),
        ("s1-grd-sigma0-asc"),
        ("s1-grd-sigma0-desc"),
        ("s1-coh")
    ]
)
def test_get_raster_profiles(raster_profile: str):
    raster_profiles = conf._get_raster_profiles()
    
    profile = raster_profiles.get(raster_profile)
    assert profile is not None
