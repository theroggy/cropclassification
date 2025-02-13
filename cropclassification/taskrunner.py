"""Run the tasks in the tasks directory."""

import argparse
import configparser
import sys
from pathlib import Path

# Import geofilops here already, if tensorflow is loaded first leads to dll load errors
import geofileops as gfo  # noqa: F401


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with open(version_path) as file:
        return file.readline()


__version__ = _get_version()


def main():
    """Parse the arguments and run the tasks."""
    # Interprete arguments
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-t",
        "--tasksdir",
        help="The path to the dir where tasks (*.ini) to be run can be found.",
    )
    # Add back help
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    args = parser.parse_args()

    # If tasks dir is specified, use it
    if args.tasksdir is not None:
        return run_tasks(tasksdir=Path(args.tasksdir))
    else:
        usertasksdir = Path.home() / "cropclassification" / "tasks"
        if usertasksdir.exists():
            return run_tasks(tasksdir=usertasksdir)
        else:
            print(
                f"Error: no tasksdir specified, and default tasks dir ({usertasksdir}) "
                "does not exist, so stop\n"
            )
            parser.print_help()
            sys.exit(1)


def run_tasks(tasksdir: Path, config_overrules: list[str] = []):
    """Run the tasks in the tasks directory.

    Args:
        tasksdir (Path): path to the directory where the tasks are located.
        config_overrules (list[str], optional): list of overrules to apply on the
            configuration. They should be specified as a list of
            "<section>.<parameter>=<value>" strings. Defaults to [].
    """
    # Get the tasks and treat them
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
        action = task_config["task"].get("action")
        default_basedir = None
        if "paths" in task_config:
            default_basedir = Path(task_config["paths"].get("marker_basedir"))
        if default_basedir is None:
            default_basedir = task_path.parent.parent

        # Determine the config files to load
        script_dir = Path(__file__).resolve().parent
        config_paths = [script_dir / "general.ini"]
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

        if action in ("calc_marker", "calc_cropclass"):
            from cropclassification import calc_cropclass

            calc_cropclass.run_cropclass(
                config_paths=config_paths,
                default_basedir=default_basedir,
                config_overrules=config_overrules,
            )
        elif action == "calc_periodic_mosaic":
            from cropclassification import calc_periodic_mosaic

            calc_periodic_mosaic.calc_periodic_mosaic_task(
                config_paths=config_paths, default_basedir=default_basedir
            )
        else:
            raise Exception(f"Action not supported: {action}")


if __name__ == "__main__":
    main()
