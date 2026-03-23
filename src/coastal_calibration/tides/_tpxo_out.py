"""TPXOOut — parser for OTPSnc predict_tide output files.

Ported verbatim from ``tpxo_to_open_bnds_hgrid/TPXOOut.py``.
"""

from __future__ import annotations

import math

import pandas as pd


def _isclose(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    """Approximate equality (Python 2 style)."""
    if rel_tol < 0 or abs_tol < 0:
        msg = "tolerances must be non-negative"
        raise ValueError(msg)
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    diff = math.fabs(b - a)
    return (diff <= math.fabs(rel_tol * b)) or (diff <= math.fabs(rel_tol * a)) or (
        diff <= abs_tol
    )


class TPXOOut:
    """Store one timeseries from OTPSnc predict_tide output.

    Parameters
    ----------
    otpsncoutfile : str or Path
        Path to the whitespace-delimited predict_tide output file.
    """

    def __init__(self, otpsncoutfile: str) -> None:
        self.df = pd.read_csv(otpsncoutfile, sep=r"\s+", header=3, on_bad_lines="skip")

    def get_number_of_locations(self) -> int:
        """Return the number of tide gauge locations (always 1 per file)."""
        return 1
