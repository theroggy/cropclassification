from datetime import datetime
from typing import Union


def get_monday(date: Union[str, datetime]) -> datetime:
    """
    If date is no monday yet, return the first monday before it.

    Args:
       date (datetime, str): if a string, expected format: %Y-%m-%d

    Return
        datetime: a monday.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    year_week = date.strftime("%Y_%W")
    year_week_monday = datetime.strptime(year_week + "_1", "%Y_%W_%w")
    return year_week_monday


def get_monday_biweekly(date: Union[str, datetime]) -> datetime:
    """
    This function gets the first monday before the date provided, aligned to the every
    second week of the year.

    Args:
       date (datetime, str): if a string, expected format: %Y-%m-%d

    Return
        datetime: a monday.
    """
    # First determine the previous monday
    monday = get_monday(date)

    # If the week number is even, it is not a "biweekly monday", so we need to move
    # another monday forward...
    monday_week = int(monday.strftime("%W"))
    if monday_week % 2 == 0:
        monday = datetime.strptime(f"{monday.year}_{monday_week-1}_1", "%Y_%W_%w")

    return monday
