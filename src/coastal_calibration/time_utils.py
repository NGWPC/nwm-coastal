"""Time utilities for coastal calibration workflows.

This module replaces the bash/perl scripts advance_time.sh and advance_cymdh.pl
with native Python datetime operations.
"""

from __future__ import annotations

from datetime import datetime, timedelta


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
    dt = datetime.strptime(date_string[:10], "%Y%m%d%H")
    dt += timedelta(hours=hours)
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
    dt = datetime.strptime(date_string[:10], "%Y%m%d%H")
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
    return f"{date_string[:10]}00"
