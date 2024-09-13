import shutil

import geofileops as gfo

from cropclassification.predict import classification
from tests import test_helper


def test_add_cross_pred_model_id(tmp_path):
    parcel_path = test_helper.SampleData.inputdata_dir / "Prc_BEFL_2023_2023-07-24.gpkg"
    test_path = tmp_path / parcel_path.name
    shutil.copy(parcel_path, test_path)
    columnname = "test_column"
    cross_pred_models = 4

    classification.add_cross_pred_model_id(
        test_path,
        cross_pred_models=cross_pred_models,
        columnname=columnname,
        class_balancing_column="GWSCOD_H",
    )

    result_gdf = gfo.read_file(test_path)
    assert columnname in result_gdf.columns
    assert result_gdf[columnname].max() == cross_pred_models - 1
    assert result_gdf[columnname].min() == 0
