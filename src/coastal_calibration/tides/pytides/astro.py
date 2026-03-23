"""Astronomical parameter calculations for tidal analysis.

Computes lunar/solar ephemeris parameters needed by the harmonic
constituent model.  Based on Meeus's *Astronomical Algorithms* and
the modernized `pytides-py3 <https://github.com/pytides/pytides-py3>`__ library.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np

_D2R: float = np.pi / 180.0
_R2D: float = 180.0 / np.pi

_HOURS_PER_DAY: float = 24.0
_MINUTES_PER_DAY: float = 24.0 * 60.0
_SECONDS_PER_DAY: float = 24.0 * 60.0 * 60.0
_DAYS_PER_YEAR: float = 365.25
_JULIAN_EPOCH: float = 2451545.0  # J2000.0
_JULIAN_CENTURY: float = 36525.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AstronomicalParameter:
    """Scalar value + speed pair for one astronomical quantity."""

    __slots__ = ("value", "speed")

    def __init__(self, value: float, speed: float) -> None:
        self.value = value
        self.speed = speed


def _s2d(degrees: float, arcmins: float = 0.0, arcsecs: float = 0.0) -> float:
    """Convert sexagesimal angle to decimal degrees."""
    return degrees + arcmins / 60.0 + arcsecs / 3600.0


def _polynomial(coefficients: np.ndarray, argument: float) -> float:
    return float(np.polyval(coefficients[::-1], argument))


def _d_polynomial(coefficients: np.ndarray, argument: float) -> float:
    if len(coefficients) == 0:
        return 0.0
    deriv = np.polyder(coefficients[::-1])
    return float(np.polyval(deriv, argument))


def _JD(t: datetime) -> float:  # noqa: N802
    """Julian Day Number (Meeus formula 7.1)."""
    Y, M = t.year, t.month  # noqa: N806
    D = (  # noqa: N806
        t.day
        + t.hour / _HOURS_PER_DAY
        + t.minute / _MINUTES_PER_DAY
        + t.second / _SECONDS_PER_DAY
        + t.microsecond / (_SECONDS_PER_DAY * 1e6)
    )
    if M <= 2:
        Y -= 1  # noqa: N806
        M += 12  # noqa: N806
    A = np.floor(Y / 100.0)  # noqa: N806
    B = 2 - A + np.floor(A / 4.0)  # noqa: N806
    return float(np.floor(_DAYS_PER_YEAR * (Y + 4716)) + np.floor(30.6001 * (M + 1)) + D + B - 1524.5)


def _T(t: datetime) -> float:
    """Julian centuries since J2000.0 (Meeus formula 11.1)."""
    return (_JD(t) - _JULIAN_EPOCH) / _JULIAN_CENTURY


# ---------------------------------------------------------------------------
# Polynomial coefficients
# ---------------------------------------------------------------------------

_terrestrial_obliquity = np.array(
    [
        _s2d(23, 26, 21.448),
        -_s2d(0, 0, 4680.93),
        -_s2d(0, 0, 1.55),
        _s2d(0, 0, 1999.25),
        -_s2d(0, 0, 51.38),
        -_s2d(0, 0, 249.67),
        -_s2d(0, 0, 39.05),
        _s2d(0, 0, 7.12),
        _s2d(0, 0, 27.87),
        _s2d(0, 0, 5.79),
        _s2d(0, 0, 2.45),
    ],
    dtype=np.float64,
)
# Adjust for parameter T (Julian centuries) instead of U
_terrestrial_obliquity_adj = np.array(
    [c * (1e-2) ** i for i, c in enumerate(_terrestrial_obliquity)],
    dtype=np.float64,
)

_solar_perigee = np.array(
    (280.46645 - 357.52910, 36000.76932 - 35999.05030, 0.0003032 + 0.0001559, 0.00000048),
    dtype=np.float64,
)
_solar_longitude = np.array((280.46645, 36000.76983, 0.0003032), dtype=np.float64)
_lunar_inclination = np.array((5.145,), dtype=np.float64)
_lunar_longitude = np.array(
    (218.3164591, 481267.88134236, -0.0013268, 1 / 538841.0 - 1 / 65194000.0),
    dtype=np.float64,
)
_lunar_node = np.array(
    (125.0445550, -1934.1361849, 0.0020762, 1 / 467410.0, -1 / 60616000.0),
    dtype=np.float64,
)
_lunar_perigee = np.array(
    (83.3532430, 4069.0137111, -0.0103238, -1 / 80053.0, 1 / 18999000.0),
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Auxiliary parameters (Schureman Table 6)
# ---------------------------------------------------------------------------


def _I(N: float, i: float, omega: float) -> float:  # noqa: E741, N802
    N, i, omega = _D2R * N, _D2R * i, _D2R * omega  # noqa: N806
    cosI = np.cos(i) * np.cos(omega) - np.sin(i) * np.sin(omega) * np.cos(N)
    return float(_R2D * np.arccos(np.clip(cosI, -1.0, 1.0)))


def _xi(N: float, i: float, omega: float) -> float:
    N, i, omega = _D2R * N, _D2R * i, _D2R * omega  # noqa: N806
    e1 = np.cos(0.5 * (omega - i)) / np.cos(0.5 * (omega + i)) * np.tan(0.5 * N)
    e2 = np.sin(0.5 * (omega - i)) / np.sin(0.5 * (omega + i)) * np.tan(0.5 * N)
    return float(_R2D * 0.5 * (np.arctan(e1) + np.arctan(e2)))


def _nu(N: float, i: float, omega: float) -> float:
    N, i, omega = _D2R * N, _D2R * i, _D2R * omega  # noqa: N806
    e1 = np.cos(0.5 * (omega - i)) / np.cos(0.5 * (omega + i)) * np.tan(0.5 * N)
    e2 = np.sin(0.5 * (omega - i)) / np.sin(0.5 * (omega + i)) * np.tan(0.5 * N)
    return float(_R2D * 0.5 * (np.arctan(e1) - np.arctan(e2)))


def _nup(N: float, i: float, omega: float) -> float:
    return float(_R2D * np.arctan(np.sin(_D2R * omega) * np.tan(_D2R * N)))


def _nupp(N: float, i: float, omega: float) -> float:
    return float(_R2D * np.arctan(np.sin(_D2R * omega) * np.tan(2 * _D2R * N)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def astro(t: datetime | Iterable[datetime]) -> dict[str, AstronomicalParameter]:
    """Return astronomical parameters at time *t*.

    Parameters
    ----------
    t : datetime or iterable of datetimes
        When a single datetime is given the returned values are scalars.

    Returns
    -------
    dict[str, AstronomicalParameter]
        Mapping of parameter names to ``AstronomicalParameter(value, speed)`` pairs.
    """
    if isinstance(t, datetime):
        times = [t]
    elif isinstance(t, Iterable):
        times = list(t)
    else:
        times = [t]

    # Only single-time path is needed by the tidal prediction code
    t_val = _T(times[0])
    jd_val = _JD(times[0])

    s = _polynomial(_lunar_longitude, t_val) % 360.0
    h = _polynomial(_solar_longitude, t_val) % 360.0
    p = _polynomial(_lunar_perigee, t_val) % 360.0
    N = _polynomial(_lunar_node, t_val) % 360.0  # noqa: N806
    pp = _polynomial(_solar_perigee, t_val) % 360.0
    omega = _polynomial(_terrestrial_obliquity_adj, t_val)
    i = _polynomial(_lunar_inclination, t_val)

    I_val = _I(N, i, omega)  # noqa: N806
    xi_val = _xi(N, i, omega)
    nu_val = _nu(N, i, omega)
    nup_val = _nup(N, i, omega)
    nupp_val = _nupp(N, i, omega)

    # Speeds (degrees per hour)
    s_speed = _d_polynomial(_lunar_longitude, t_val) / (_JULIAN_CENTURY * 24.0)
    h_speed = _d_polynomial(_solar_longitude, t_val) / (_JULIAN_CENTURY * 24.0)
    p_speed = _d_polynomial(_lunar_perigee, t_val) / (_JULIAN_CENTURY * 24.0)
    N_speed = _d_polynomial(_lunar_node, t_val) / (_JULIAN_CENTURY * 24.0)  # noqa: N806
    pp_speed = _d_polynomial(_solar_perigee, t_val) / (_JULIAN_CENTURY * 24.0)

    t_plus_h_minus_s = ((jd_val - np.floor(jd_val)) * 360.0 + h - s) % 360.0

    return {
        "s": AstronomicalParameter(s, s_speed),
        "h": AstronomicalParameter(h, h_speed),
        "p": AstronomicalParameter(p, p_speed),
        "N": AstronomicalParameter(N, N_speed),
        "pp": AstronomicalParameter(pp, pp_speed),
        "90": AstronomicalParameter(90.0, 0.0),
        "omega": AstronomicalParameter(omega, 0.0),
        "i": AstronomicalParameter(i, 0.0),
        "I": AstronomicalParameter(I_val, 0.0),
        "xi": AstronomicalParameter(xi_val, 0.0),
        "nu": AstronomicalParameter(nu_val, 0.0),
        "nup": AstronomicalParameter(nup_val, 0.0),
        "nupp": AstronomicalParameter(nupp_val, 0.0),
        "T+h-s": AstronomicalParameter(t_plus_h_minus_s, 15.0 + h_speed - s_speed),
        "P": AstronomicalParameter(p, p_speed),
    }
