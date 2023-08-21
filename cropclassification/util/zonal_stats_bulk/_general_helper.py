from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _format_output_path(
    features_path: Path,
    image_path: Path,
    output_dir: Path,
    orbit_properties_pass: Optional[str],
    band: Optional[Union[str, int]],
) -> Path:
    """
    Prepare the output path.
    """
    # Interprete the orbit...
    if orbit_properties_pass is not None:
        if orbit_properties_pass == "ASCENDING":
            orbit = "_ASC"
        elif orbit_properties_pass == "DESCENDING":
            orbit = "_DESC"
        else:
            message = f"Unknown orbit_properties_pass: {orbit_properties_pass}"
            logger.error(message)
            raise Exception(message)
    else:
        orbit = ""

    # Format + return output path
    output_stem = f"{features_path.stem}__{image_path.stem}{orbit}"
    if band is not None:
        output_stem = f"{output_stem}_{band}"
    output_path = output_dir / f"{output_stem}.sqlite"
    return output_path


def _format_progress_message(
    nb_todo: int,
    nb_done_total: int,
    start_time: datetime,
    nb_done_latestbatch: Optional[int] = None,
    start_time_latestbatch: Optional[datetime] = None,
) -> str:
    """
    Returns a progress message based on the input.

    Args
        nb_todo: total number of items that need(ed) to be processed
        nb_done_total: total number of items that have been processed already
        nb_done_latestbatch: number of items that were processed in the latest batch
        start_time: datetime the processing started
        start_time_latestbatch: datetime the latest batch started
    """
    time_passed_s = (datetime.now() - start_time).total_seconds()
    time_passed_latestbatch_s = None
    if start_time_latestbatch is not None:
        time_passed_latestbatch_s = (
            datetime.now() - start_time_latestbatch
        ).total_seconds()

    # Calculate the overall progress
    large_number = 9999999999
    if time_passed_s > 0:
        nb_per_hour = (nb_done_total / time_passed_s) * 3600
    else:
        nb_per_hour = large_number
    hours_to_go = (int)((nb_todo - nb_done_total) / nb_per_hour)
    min_to_go = (int)((((nb_todo - nb_done_total) / nb_per_hour) % 1) * 60)

    # Format message
    message = (
        f"{hours_to_go}:{min_to_go} left for {nb_todo-nb_done_total} todo at "
        f"{nb_per_hour:0.0f}/h"
    )
    # Add speed of the latest batch to message if appropriate
    if (
        time_passed_latestbatch_s is not None
        and nb_done_latestbatch is not None
        and time_passed_latestbatch_s > 0
    ):
        nb_per_hour_latestbatch = (
            nb_done_latestbatch / time_passed_latestbatch_s
        ) * 3600
        message = f"{message} ({nb_per_hour_latestbatch:0.0f}/h last batch)"

    return message
