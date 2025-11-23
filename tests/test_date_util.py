from datetime import datetime

import pytest

from cropclassification.util import date_util


@pytest.mark.parametrize(
    "input_date, before, expected",
    [
        ("2024-01-01", True, datetime(2024, 1, 1)),
        ("2024-01-05", True, datetime(2024, 1, 1)),
        ("2023-01-01", True, datetime(2022, 12, 26)),
        (datetime(2024, 1, 5), True, datetime(2024, 1, 1)),
        ("2024-01-01", False, datetime(2024, 1, 1)),
        ("2024-01-05", False, datetime(2024, 1, 8)),
        ("2022-12-27", False, datetime(2023, 1, 2)),
        (datetime(2024, 1, 5), False, datetime(2024, 1, 8)),
    ],
)
def test_get_monday(input_date, before, expected):
    result = date_util.get_monday(input_date, before=before)
    assert result == expected
    assert result.strftime("%w") == "1"


@pytest.mark.parametrize(
    "input_date, before, expected",
    [
        ("2024-01-01", True, datetime(2024, 1, 1)),
        ("2024-01-05", True, datetime(2024, 1, 1)),
        ("2024-01-10", True, datetime(2024, 1, 1)),
        ("2023-01-01", True, datetime(2022, 12, 19)),
        (datetime(2024, 1, 5), True, datetime(2024, 1, 1)),
        (datetime(2024, 2, 1), True, datetime(2024, 1, 29)),
        (datetime(2024, 2, 5), True, datetime(2024, 1, 29)),
        ("2024-01-01", False, datetime(2024, 1, 1)),
        ("2024-01-05", False, datetime(2024, 1, 15)),
        ("2024-01-10", False, datetime(2024, 1, 15)),
        ("2022-12-20", False, datetime(2023, 1, 2)),
    ],
)
def test_get_monday_biweekly(input_date, before, expected):
    result = date_util.get_monday_biweekly(input_date, before=before)
    assert result == expected
    assert result.strftime("%w") == "1"
