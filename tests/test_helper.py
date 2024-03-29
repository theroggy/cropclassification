from pathlib import Path


class SampleDirs:
    marker_basedir = Path(__file__).resolve().parent.parent / "sample_marker_basedir"
    tasks_dir = marker_basedir / "_tasks"
    config_dir = marker_basedir / "_config"
    task_path = tasks_dir / "task_test_calc_periodic_mosaic.ini"
