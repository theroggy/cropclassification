import shutil
from contextlib import nullcontext

import pytest

from cropclassification.helpers import config_helper as conf
from cropclassification.preprocess._timeseries_helper import exclude_erase_layer
from tests.test_helper import SampleData


@pytest.mark.parametrize(
    "ignore_erase_layer_column_exists, exp_error", [(True, False), (False, True)]
)
def test_exclude_erase_layer(tmp_path, ignore_erase_layer_column_exists, exp_error):
    marker_basedir = tmp_path / SampleData.marker_basedir.name
    shutil.copytree(SampleData.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_marker.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    # Read the configuration files
    config_paths = [tasks_dir / "local_overrule.ini", tasks_dir / task_ini]
    overrules = [
        "calc_marker_params.erase_layer_filename=trees_19_162_BEFL-2022-2023-ofw.gpkg",
    ]
    if ignore_erase_layer_column_exists:
        overrules.append(
            "calc_marker_params.classes_refe_filename=BEFL_2023_mon_refe_2023-07-24_eraselayer.tsv"
        )
    conf.read_config(
        config_paths=config_paths,
        default_basedir=marker_basedir,
        overrules=overrules,
    )

    input_parcel_filename = conf.calc_marker_params.getpath("input_parcel_filename")
    classes_refe_filename = conf.calc_marker_params.getpath("classes_refe_filename")
    erase_layer_filename = conf.calc_marker_params.getpath("erase_layer_filename")

    # Call the exclude_erase_layer function
    output_imagedata_parcel_input_path = tmp_path / "output_imagedata_parcel_input.gpkg"
    if exp_error:
        matchstr = "IGNORE_ERASE_LAYER column not found in classes reference file"
        handler = pytest.raises(Exception, match=matchstr)
    else:
        handler = nullcontext()
    with handler:
        result = exclude_erase_layer(
            input_parcel_path=SampleData.inputdata_dir / input_parcel_filename,
            output_imagedata_parcel_input_path=output_imagedata_parcel_input_path,
            erase_layer_path=SampleData.inputdata_dir / erase_layer_filename,
            classes_refe_path=SampleData.refe_dir / classes_refe_filename,
        )

        # Assert that the output file exists
        assert result.exists()

        # Assert that the output file is not empty
        assert result.stat().st_size > 0
