from datetime import datetime
from typing import Union


def get_monday(date: Union[str, datetime], before: bool = True) -> datetime:
    """
    If date is no monday yet, return the monday before or after it.

    Args:
       date (datetime, str): if a string, expected format: %Y-%m-%d
       before (bool): if True, return the monday before the date, otherwise the monday
        after. Defaults to True.

    Return
        datetime: a monday.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    # It is already a monday, so return it.
    if date.strftime("%w") == "1":
        return date

    # Determine the monday before or after the date
    year = date.strftime("%Y")
    week = int(date.strftime("%W"))
    if not before:
        week += 1
    year_week_monday = datetime.strptime(f"{year}_{week:02d}_1", "%Y_%W_%w")

    return year_week_monday


def get_monday_biweekly(date: Union[str, datetime], before: bool = True) -> datetime:
    """
    Determines the "biweekly" monday for the date provided.

    In practice the first monday before or after the date will be returned, aligned to
    every second week of the year.

    Args:
       date (datetime, str): if a string, expected format: %Y-%m-%d
       before (bool): if True, return the monday before the date, otherwise the monday
            after. Defaults to True.

    Return
        datetime: a monday.
    """
    # First determine the monday before or after the date
    monday = get_monday(date, before=before)

    # If the week number is even, it is not a "biweekly monday", so we need to move
    # another monday forward/backward...
    monday_week = int(monday.strftime("%W"))
    if monday_week % 2 == 0:
        if before:
            monday_week -= 1
        else:
            monday_week += 1
        monday = datetime.strptime(f"{monday.year}_{monday_week}_1", "%Y_%W_%w")

    return monday
