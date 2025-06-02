"""Some tests on the preparation of the input data for BEFL."""

import pytest

import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess._prepare_input_BEFL as befl
from tests.test_helper import SampleData


@pytest.mark.parametrize(
    "classtype_to_prepare",
    [
        "CROPGROUP",
        "CROPGROUP-EARLY",
        "CROPROTATION",
        "CROPROTATION-EARLY",
        "CARBONSUPPLY",
        "CARBONSUPPLY-EARLY",
        "LANDCOVER",
        "LANDCOVER-EARLY",
        "LATECROP-EARLY",
        "LATECROP-LATE",
        "RUGGENTEELT",
        "RUGGENTEELT-EARLY",
    ],
)
@pytest.mark.parametrize("groundtruth", [False])
def test_befl_prepare_input(tmp_path, classtype_to_prepare, groundtruth):
    """Basic test on the preparation of BEFL input data for a classtype/markertype."""
    if groundtruth:
        classtype_to_prepare = f"{classtype_to_prepare}-GROUNDTRUTH"
    conf.read_config(
        config_paths=[
            SampleData.tasks_dir / "local_overrule.ini",
            SampleData.config_dir / "cropgroup.ini",
        ],
        default_basedir=SampleData.markers_dir,
    )

    output_dir = tmp_path
    df_parceldata = befl.prepare_input(
        input_parcel_path=SampleData.input_parcel_path,
        classtype_to_prepare=classtype_to_prepare,
        classes_refe_path=SampleData.classes_refe_path,
        min_parcels_in_class=50,
        output_dir=output_dir,
    )

    assert df_parceldata is not None
