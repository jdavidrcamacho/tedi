"""Functions for converting dates to/from JD and MJD.

Note: Python datetime module assumes an infinitely valid Gregorian calendar.
      The Gregorian calendar took effect after 10-15-1582 and the dates 10-05
      through 10-14-1582 never occurred. Python datetime objects will produce
      incorrect time deltas if one date is from before 10-15-1582.
"""

import datetime as dt
from math import modf, trunc


def mjd_to_jd(modified_julian_day: float) -> float:
    """Convert Modified Julian Day (MJD) to Julian Day (JD).

    Args:
        modified_julian_day (float): Modified Julian Day.

    Returns:
        float: Julian Day.
    """
    return modified_julian_day + 2400000.5


def jd_to_mjd(julian_day: float) -> float:
    """Convert Julian Day (JD) to Modified Julian Day (MJD).

    Args:
        julian_day (float): Julian Day.

    Returns:
        float: Modified Julian Day.
    """
    return julian_day - 2400000.5


def date_to_jd(year: int, month: int, day: float) -> float:
    """Convert a date to Julian Day (JD).

    The algorithm is derived from 'Practical Astronomy with your Calculator
    or Spreadsheet', 4th ed., Duffet-Smith and Zwart, 2011.

    Args:
        year (int): Year as integer. Years preceding 1 A.D. should be 0 or
            negative. For example, the year before 1 A.D. is 0, and 10 B.C.
            is year -9.
        month (int): Month as integer, where January = 1, February = 2, etc.
        day (float): Day as float, may include a fractional part.

    Returns:
        float: Julian Day.

    Examples:
        Convert 6 a.m., February 17, 1985 to Julian Day:

        $ date_to_jd(1985, 2, 17.25)
        2446113.75
    """
    if month == 1 or month == 2:
        adjusted_year = year - 1
        adjusted_month = month + 12
    else:
        adjusted_year = year
        adjusted_month = month

    # Determine if date is before or after the start of the Gregorian calendar
    if (
        (year < 1582)
        or (year == 1582 and month < 10)
        or (year == 1582 and month == 10 and day < 15)
    ):
        # Before the start of the Gregorian calendar
        gregorian_offset = 0
    else:
        # After the start of the Gregorian calendar
        century = trunc(adjusted_year / 100.0)
        gregorian_offset = 2 - century + trunc(century / 4.0)

    if adjusted_year < 0:
        leap_year_adjustment = trunc((365.25 * adjusted_year) - 0.75)
    else:
        leap_year_adjustment = trunc(365.25 * adjusted_year)

    month_adjustment = trunc(30.6001 * (adjusted_month + 1))
    julian_day = gregorian_offset + leap_year_adjustment + month_adjustment
    julian_day = julian_day + day + 1720994.5  # type: ignore

    return julian_day


def jd_to_date(julian_day: float) -> tuple[int, int, float]:
    """Convert Julian Day (JD) to a date.

    The algorithm is derived from 'Practical Astronomy with your Calculator
    or Spreadsheet', 4th ed., Duffett-Smith and Zwart, 2011.

    Args:
        julian_day (float): Julian Day.

    Returns:
        tuple: A tuple containing:
            year (int): Year as integer. Years preceding 1 A.D. should be 0 or
                negative. For example, the year before 1 A.D. is 0, and
                10 B.C. is year -9.
            month (int): Month as integer, with January = 1, February = 2, etc.
            day (float): Day as float, may include a fractional part.

    Examples:
        Convert Julian Day 2446113.75 to year, month, and day:

        $ jd_to_date(2446113.75)
        (1985, 2, 17.25)
    """
    julian_day += 0.5
    fractional_day, integer_day = modf(julian_day)
    integer_day = int(integer_day)
    gregorian_offset = trunc((integer_day - 1867216.25) / 36524.25)

    if integer_day > 2299160:
        corrected_day = (
            integer_day + 1 + gregorian_offset - trunc(gregorian_offset / 4.0)
        )
    else:
        corrected_day = integer_day

    intermediate_day = corrected_day + 1524
    year_day_fraction = trunc((intermediate_day - 122.1) / 365.25)
    integer_days_in_year = trunc(365.25 * year_day_fraction)
    adjusted_month = trunc((intermediate_day - integer_days_in_year) / 30.6001)
    day = intermediate_day - integer_days_in_year + fractional_day
    day = day - trunc(30.6001 * adjusted_month)

    if adjusted_month < 13.5:
        month = adjusted_month - 1
    else:
        month = adjusted_month - 13

    if month > 2.5:
        year = year_day_fraction - 4716
    else:
        year = year_day_fraction - 4715

    return year, month, day


def hmsm_to_days(
    hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0
) -> float:
    """Convert hours, minutes, seconds, and microseconds to fractional days.

    Args:
        hour (int, optional): Hour. Defaults to 0.
        minute (int, optional): Minute. Defaults to 0.
        second (int, optional): Second. Defaults to 0.
        microsecond (int, optional): Microsecond. Defaults to 0.

    Returns:
        float: Fractional days.

    Examples:
        $ hmsm_to_days(hour=6)
        0.25
    """
    total_seconds = second + (microsecond / 1.0e6)
    total_minutes = minute + (total_seconds / 60.0)
    total_hours = hour + (total_minutes / 60.0)
    return total_hours / 24.0


def days_to_hmsm(fractional_days: float) -> tuple[int, int, int, int]:
    """Convert fractional days to hours, minutes, seconds, and microseconds.

    Precision beyond microseconds is rounded to the nearest microsecond.

    Args:
        fractional_days (float): Number of days. Must be less than 1.

    Returns:
        tuple: A tuple containing:
            hour (int): Hour.
            minute (int): Minute.
            second (int): Second.
            microsecond (int): Microsecond.

    Raises:
        ValueError: If `fractional_days` is >= 1.

    Examples:
        $ days_to_hmsm(0.1)
        (2, 24, 0, 0)
    """
    if fractional_days >= 1:
        raise ValueError("Input 'fractional_days' must be less than 1.")

    total_hours = fractional_days * 24.0
    fractional_hours, hour = modf(total_hours)
    total_minutes = fractional_hours * 60.0
    fractional_minutes, minute = modf(total_minutes)
    total_seconds = fractional_minutes * 60.0
    fractional_seconds, second = modf(total_seconds)
    microsecond = round(fractional_seconds * 1.0e6)

    return int(hour), int(minute), int(second), int(microsecond)


def datetime_to_jd(date: dt.datetime) -> float:
    """Convert a `datetime.datetime` object to Julian Day (JD).

    Args:
        date (datetime.datetime): `datetime.datetime` instance.

    Returns:
        float: Julian Day.

    Examples:
        $ d = datetime.datetime(1985, 2, 17, 6)
        $ d
        datetime.datetime(1985, 2, 17, 6, 0)
        $ datetime_to_jd(d)
        2446113.75
    """
    fractional_day = date.day + hmsm_to_days(
        date.hour, date.minute, date.second, date.microsecond
    )

    return date_to_jd(date.year, date.month, fractional_day)


def jd_to_datetime(julian_day: float) -> dt.datetime:
    """Convert Julian Day (JD) to `datetime.datetime` object.

    Args:
        julian_day (float): Julian Day.

    Returns:
        datetime.datetime: `datetime.datetime` object equivalent to Julian Day.

    Examples:
        $ jd_to_datetime(2446113.75)
        datetime.datetime(1985, 2, 17, 6, 0)
    """
    year, month, day = jd_to_date(julian_day)
    fractional_days, day = modf(day)
    day = int(day)
    hour, minute, second, microsecond = days_to_hmsm(fractional_days)

    return dt.datetime(year, month, day, hour, minute, second, microsecond)


def timedelta_to_days(time_delta: dt.timedelta) -> float:
    """Convert `datetime.timedelta` object to a total number of days.

    Args:
        time_delta (datetime.timedelta): `datetime.timedelta` instance.

    Returns:
        float: Total number of days in the `datetime.timedelta` object.

    Examples:
        $ td = datetime.timedelta(4.5)
        $ td
        datetime.timedelta(4, 43200)
        $ timedelta_to_days(td)
        4.5
    """
    seconds = 24.0 * 3600.0
    total_days = (
        time_delta.days
        + (time_delta.seconds + (time_delta.microseconds / 1.0e6)) / seconds
    )

    return total_days
