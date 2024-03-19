import configparser
from pathlib import Path
import shutil
import tempfile
from cropclassification import calc_periodic_mosaic
from tests import test_helper


testprojects_dir = (
    Path(tempfile.gettempdir())
    / "cropclassification_test_calc_periodic_mosaic/sample_projects"
)
markers_dir = testprojects_dir / "markers"
tasks_dir = markers_dir / "_tasks"


def test_1_init_testproject():
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)


def test_2_calc_periodic_mosaic_task():
    # Create configparser and read task file!
    task_path = tasks_dir / "task_2023_LANDCOVER_calc_periodic_mosaic.ini"
    task_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={"list": lambda x: [i.strip() for i in x.split(",")]},
        allow_no_value=True,
    )
    task_config.read(task_path)

    default_basedir = task_path.parent.parent
    # Determine the config files to load
    script_dir = Path(__file__).resolve().parent.parent
    config_paths = [script_dir / "cropclassification/general.ini"]
    extra_config_files_to_load = task_config["task"].getlist(
        "extra_config_files_to_load"
    )
    if extra_config_files_to_load is not None:
        for config_file in extra_config_files_to_load:
            config_file_formatted = Path(
                config_file.format(
                    task_filepath=task_path,
                    task_path=task_path,
                    tasks_dir=task_path.parent,
                )
            )
            if not config_file_formatted.is_absolute():
                config_file_formatted = (
                    default_basedir / config_file_formatted
                ).resolve()
            config_paths.append(Path(config_file_formatted))
    calc_periodic_mosaic.calc_periodic_mosaic_task(
        config_paths=config_paths, default_basedir=default_basedir
    )

    # cropclassification.cropclassification(tasksdir=tasks_dir)


if __name__ == "__main__":
    test_1_init_testproject()
    test_2_calc_periodic_mosaic_task()
