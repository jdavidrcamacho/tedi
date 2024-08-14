import datetime as dt

from src.tedi.utils import time


def test_mjd_to_jd() -> None:
    modified_julian_day = 58000
    expected_julian_day = 2458000.5
    assert time.mjd_to_jd(modified_julian_day) == expected_julian_day


def test_jd_to_mjd() -> None:
    julian_day = 2458000.5
    expected_modified_julian_day = 58000
    assert time.jd_to_mjd(julian_day) == expected_modified_julian_day


def test_date_to_jd() -> None:
    year, month, day = 1985, 2, 17.25
    expected_julian_day = 2446113.75
    assert time.date_to_jd(year, month, day) == expected_julian_day


def test_jd_to_date() -> None:
    julian_day = 2446113.75
    expected_date = (1985, 2, 17.25)
    assert time.jd_to_date(julian_day) == expected_date


def test_hmsm_to_days() -> None:
    hour, minute, second, microsecond = 6, 0, 0, 0
    expected_fractional_days = 0.25
    assert (
        time.hmsm_to_days(hour, minute, second, microsecond)
        == expected_fractional_days  # NOQA
    )


def test_days_to_hmsm() -> None:
    fractional_days = 0.25
    expected_time = (6, 0, 0, 0)
    assert time.days_to_hmsm(fractional_days) == expected_time


def test_datetime_to_jd() -> None:
    date = dt.datetime(1985, 2, 17, 6)
    expected_julian_day = 2446113.75
    assert time.datetime_to_jd(date) == expected_julian_day


def test_jd_to_datetime() -> None:
    julian_day = 2446113.75
    expected_date = dt.datetime(1985, 2, 17, 6)
    assert time.jd_to_datetime(julian_day) == expected_date


def test_timedelta_to_days() -> None:
    time_delta = dt.timedelta(days=4, seconds=43200)  # 4.5 days
    expected_days = 4.5
    assert time.timedelta_to_days(time_delta) == expected_days
