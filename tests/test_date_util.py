from datetime import datetime

import pytest
from cropclassification.util import date_util


@pytest.mark.parametrize(
    "input, expected",
    [
        ("2024-01-01", datetime(2024, 1, 1)),
        ("2024-01-05", datetime(2024, 1, 1)),
        ("2023-01-01", datetime(2022, 12, 26)),
        (datetime(2024, 1, 5), datetime(2024, 1, 1)),
    ],
)
def test_get_monday(input, expected):
    assert date_util.get_monday(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("2024-01-01", datetime(2024, 1, 1)),
        ("2024-01-05", datetime(2024, 1, 1)),
        ("2024-01-10", datetime(2024, 1, 1)),
        ("2023-01-01", datetime(2022, 12, 19)),
        (datetime(2024, 1, 5), datetime(2024, 1, 1)),
        (datetime(2024, 2, 1), datetime(2024, 1, 29)),
        (datetime(2024, 2, 5), datetime(2024, 1, 29)),
    ],
)
def test_get_monday_biweekly(input, expected):
    assert date_util.get_monday_biweekly(input) == expected
