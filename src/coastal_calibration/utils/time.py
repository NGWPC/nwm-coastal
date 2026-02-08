"""Time utilities for coastal calibration workflows.

This module consolidates all date / time helpers used across the package:

* Functions ported from the legacy bash/perl scripts ``advance_time.sh`` and
  ``advance_cymdh.pl`` (compact ``YYYYMMDDHH`` format).
* Flexible ``datetime`` parsing previously duplicated in ``config.schema``
  and ``downloader``.
* Hour-range iteration previously in ``downloader``.

.. note::
    This is an internal module.  Callers are responsible for validating
    user input before passing it to the compact-format helpers.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_DATE_RE = re.compile(r"^\d{10}$")


def _parse_date(date_string: str) -> datetime:
    """Parse a YYYYMMDDHH string into a datetime, with strict validation."""
    if not isinstance(date_string, str) or not _DATE_RE.match(date_string):
        raise ValueError(
            f"date_string must be exactly 10 digits in YYYYMMDDHH format, got {date_string!r}"
        )
    return datetime.strptime(date_string, "%Y%m%d%H")


def advance_time(date_string: str, hours: int) -> str:
    """Advance a date string by a specified number of hours.

    This function replaces the functionality of advance_time.sh and advance_cymdh.pl.
    It handles date arithmetic including end of day, month, year, and leap years.

    Parameters
    ----------
    date_string : str
        Date string in YYYYMMDDHH format (e.g., "2024010112" for Jan 1, 2024 at 12:00)
    hours : int
        Number of hours to advance (can be negative to go backwards in time)

    Returns
    -------
    str
        Advanced date string in YYYYMMDDHH format

    Examples
    --------
    >>> advance_time("2024010100", 24)
    '2024010200'
    >>> advance_time("2024010100", -48)
    '2023123000'
    >>> advance_time("2024022800", 24)  # Leap year
    '2024022900'
    """
    dt = _parse_date(date_string) + timedelta(hours=hours)
    return dt.strftime("%Y%m%d%H")


def parse_date_components(date_string: str) -> dict[str, str]:
    """Parse a date string into its components.

    Parameters
    ----------
    date_string : str
        Date string in YYYYMMDDHH format

    Returns
    -------
    dict
        Dictionary with keys: year, month, day, hour, pdy (YYYYMMDD), cyc (HH)
    """
    dt = _parse_date(date_string)
    return {
        "year": dt.strftime("%Y"),
        "month": dt.strftime("%m"),
        "day": dt.strftime("%d"),
        "hour": dt.strftime("%H"),
        "pdy": dt.strftime("%Y%m%d"),
        "cyc": dt.strftime("%H"),
    }


def format_forcing_date(date_string: str) -> str:
    """Format a date string for forcing file naming (YYYYMMDDHH00).

    Parameters
    ----------
    date_string : str
        Date string in YYYYMMDDHH format

    Returns
    -------
    str
        Date string in YYYYMMDDHH00 format
    """
    _parse_date(date_string)
    return f"{date_string}00"


# ---------------------------------------------------------------------------
# Flexible datetime parsing (consolidates schema._parse_datetime and
# downloader._parse_datetime)
# ---------------------------------------------------------------------------


def parse_datetime(value: str | datetime | date) -> datetime:
    """Parse a datetime from various formats.

    Supports:
    - ``datetime`` objects (returned as-is)
    - ``date`` objects (converted to datetime at midnight)
    - ISO format strings: ``"2021-06-11T00:00:00"``, ``"2021-06-11"``
    - Date strings: ``"2021-06-11"``, ``"20210611"``
    - Date with time: ``"2021-06-11 00:00:00"``, ``"2021-06-11 00:00"``

    Parameters
    ----------
    value : str, datetime, or date
        The value to parse.

    Returns
    -------
    datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If the value cannot be parsed as a datetime.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    value_str = str(value).strip()

    # Try ISO format first (handles both "2021-06-11T00:00:00" and "2021-06-11")
    try:
        return datetime.fromisoformat(value_str)
    except ValueError:
        pass

    # Try date with space separator: "2021-06-11 00:00:00"
    try:
        return datetime.strptime(value_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Try date with space and no seconds: "2021-06-11 00:00"
    try:
        return datetime.strptime(value_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    # Try compact date format: "20210611"
    try:
        return datetime.strptime(value_str, "%Y%m%d")
    except ValueError:
        pass

    raise ValueError(
        f"Cannot parse datetime from '{value}'. "
        "Supported formats: '2021-06-11', '2021-06-11T00:00:00', "
        "'2021-06-11 00:00:00', '20210611'"
    )


# ---------------------------------------------------------------------------
# Hour-range iteration (moved from downloader)
# ---------------------------------------------------------------------------


def iter_hours(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield each whole hour from *start* up to (but not including) *end*.

    Parameters
    ----------
    start : datetime
        The first hour to yield.
    end : datetime
        The exclusive upper bound.

    Yields
    ------
    datetime
        Each hour in the range ``[start, end)``.
    """
    current = start
    while current < end:
        yield current
        current += timedelta(hours=1)
