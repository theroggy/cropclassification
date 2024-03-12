import configparser
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
    tasksdir = Path("x:/monitoring/markers/dev/_tasks")
    task_paths = sorted(tasksdir.glob("task_*.ini"))
    for task_path in task_paths:
        # Create configparser and read task file!
        task_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={"list": lambda x: [i.strip() for i in x.split(",")]},
            allow_no_value=True,
        )
        task_config.read(task_path)

        # Now get the info we want from the task config
        default_basedir = task_path.parent.parent

        # Determine the config files to load
        script_dir = Path(__file__).resolve().parent.parent
        config_paths = [script_dir / "cropclassification" / "general.ini"]
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

    # Read the configuration files
    conf.read_config(config_paths, default_basedir=default_basedir)
    image_profiles = conf.image_profiles

    profile = image_profiles.get(sensor)
    assert profile is not None


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
def test_get_image_profiles_testdata(sensor: str):
    data_dir = Path(__file__).resolve().parent / "data"
    config_path = data_dir / "image_profiles.ini"
    config_path = conf.marker.getpath("image_profiles_config_filepath")
    image_profiles = conf._get_image_profiles(config_path)

    profile = image_profiles.get(sensor)
    assert profile is not None
