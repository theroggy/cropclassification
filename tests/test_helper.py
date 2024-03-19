from pathlib import Path


sampleprojects_dir = Path(__file__).resolve().parent.parent / "sample_projects"


class SampleProjectTasks:
    project_dir = sampleprojects_dir / "markers"
    tasks_dir = project_dir / "_tasks"
    config_dir = project_dir / "_config"
    task_path = tasks_dir / "task_2023_LANDCOVER_calc_periodic_mosaic.ini"


class TestData:
    dir = Path(__file__).resolve().parent / "data"
