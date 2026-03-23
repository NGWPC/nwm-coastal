"""Nodal correction factors for tidal constituents.

Implements Schureman equations for amplitude (f) and phase (u) nodal
corrections.  All functions accept a dictionary of astronomical
parameters (in degrees) and return dimensionless scale factors.

Based on the modernized `pytides-py3 <https://github.com/pytides/pytides-py3>`__ library.
"""

from __future__ import annotations

import numpy as np

_D2R: float = np.pi / 180.0
_R2D: float = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Amplitude factors (f)
# ---------------------------------------------------------------------------


def f_unity(a: dict) -> float:
    """No correction (unity)."""
    return 1.0


def f_Mm(a: dict) -> float:
    """Schureman equations 73, 65."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = (2 / 3.0 - np.sin(omega) ** 2) * (1 - 3 / 2.0 * np.sin(i) ** 2)
    return (2 / 3.0 - np.sin(I) ** 2) / mean


def f_Mf(a: dict) -> float:
    """Schureman equations 74, 66."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = np.sin(omega) ** 2 * np.cos(0.5 * i) ** 4
    return np.sin(I) ** 2 / mean


def f_O1(a: dict) -> float:
    """Schureman equations 75, 67."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = np.sin(omega) * np.cos(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return (np.sin(I) * np.cos(0.5 * I) ** 2) / mean


def f_J1(a: dict) -> float:
    """Schureman equations 76, 68."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = np.sin(2 * omega) * (1 - 3 / 2.0 * np.sin(i) ** 2)
    return np.sin(2 * I) / mean


def f_OO1(a: dict) -> float:
    """Schureman equations 77, 69."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = np.sin(omega) * np.sin(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return np.sin(I) * np.sin(0.5 * I) ** 2 / mean


def f_M2(a: dict) -> float:
    """Schureman equations 78, 70."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    mean = np.cos(0.5 * omega) ** 4 * np.cos(0.5 * i) ** 4
    return np.cos(0.5 * I) ** 4 / mean


def f_K1(a: dict) -> float:
    """Schureman equations 227, 226, 68."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    nu = _D2R * a["nu"].value
    sin_2I = np.sin(2 * I)
    mean = 0.5023 * np.sin(2 * omega) * (1 - 3 / 2.0 * np.sin(i) ** 2) + 0.1681
    return float(
        np.clip(
            (0.2523 * sin_2I**2 + 0.1689 * sin_2I * np.cos(nu) + 0.0283) ** 0.5 / mean,
            0.0,
            np.inf,
        )
    )


def f_L2(a: dict) -> float:
    """Schureman equations 215, 213, 204."""
    P = _D2R * a["P"].value  # noqa: N806
    I = _D2R * a["I"].value  # noqa: E741
    tan_half_I = np.tan(0.5 * I)
    R_a_inv = (1 - 12 * tan_half_I**2 * np.cos(2 * P) + 36 * tan_half_I**4) ** 0.5
    return f_M2(a) * R_a_inv


def f_K2(a: dict) -> float:
    """Schureman equations 235, 234, 71."""
    omega = _D2R * a["omega"].value
    i = _D2R * a["i"].value
    I = _D2R * a["I"].value  # noqa: E741
    nu = _D2R * a["nu"].value
    mean = 0.5023 * np.sin(omega) ** 2 * (1 - 3 / 2.0 * np.sin(i) ** 2) + 0.0365
    return float(
        np.clip(
            (0.2533 * np.sin(I) ** 4 + 0.0367 * np.sin(I) ** 2 * np.cos(2 * nu) + 0.0013) ** 0.5
            / mean,
            0.0,
            np.inf,
        )
    )


def f_M1(a: dict) -> float:
    """Schureman equations 206, 207, 195."""
    P = _D2R * a["P"].value  # noqa: N806
    I = _D2R * a["I"].value  # noqa: E741
    cos_I = np.cos(I)
    cos_half_I = np.cos(0.5 * I)
    Q_a_inv = (
        0.25
        + 1.5 * cos_I * np.cos(2 * P) * cos_half_I ** (-0.5)
        + 2.25 * cos_I**2 * cos_half_I ** (-4)
    ) ** 0.5
    return f_O1(a) * Q_a_inv


def f_Modd(a: dict, n: int) -> float:
    """Schureman equation 149."""
    return f_M2(a) ** (n / 2.0)


# ---------------------------------------------------------------------------
# Phase factors (u)
# ---------------------------------------------------------------------------


def u_zero(a: dict) -> float:
    """No correction (zero)."""
    return 0.0


def u_Mf(a: dict) -> float:
    return -2.0 * a["xi"].value


def u_O1(a: dict) -> float:
    return 2.0 * a["xi"].value - a["nu"].value


def u_J1(a: dict) -> float:
    return -a["nu"].value


def u_OO1(a: dict) -> float:
    return -2.0 * a["xi"].value - a["nu"].value


def u_M2(a: dict) -> float:
    return 2.0 * a["xi"].value - 2.0 * a["nu"].value


def u_K1(a: dict) -> float:
    return -a["nup"].value


def u_L2(a: dict) -> float:
    """Schureman 214."""
    I = _D2R * a["I"].value  # noqa: E741
    P = _D2R * a["P"].value  # noqa: N806
    tan_half_I = np.tan(0.5 * I)
    if tan_half_I == 0:
        return 2.0 * a["xi"].value - 2.0 * a["nu"].value
    R = _R2D * np.arctan(np.sin(2 * P) / (1 / 6.0 * tan_half_I ** (-2) - np.cos(2 * P)))
    return 2.0 * a["xi"].value - 2.0 * a["nu"].value - R


def u_K2(a: dict) -> float:
    return -2.0 * a["nupp"].value


def u_M1(a: dict) -> float:
    """Schureman 202."""
    I = _D2R * a["I"].value  # noqa: E741
    P = _D2R * a["P"].value  # noqa: N806
    cos_I = np.cos(I)
    if cos_I == 0:
        return a["xi"].value - a["nu"].value
    Q = _R2D * np.arctan((5 * cos_I - 1) / (7 * cos_I + 1) * np.tan(P))
    return a["xi"].value - a["nu"].value + Q


def u_Modd(a: dict, n: int) -> float:
    return n / 2.0 * u_M2(a)
