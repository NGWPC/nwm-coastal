"""Tidal constituent definitions.

Defines the 37 NOAA standard harmonic constituents used for tidal
analysis and prediction.  Each constituent carries its Doodson (XDO)
coefficients and references to the appropriate nodal correction
functions from :mod:`~coastal_calibration.tides.pytides.nodal_corrections`.

Based on the modernized `pytides-py3 <https://github.com/pytides/pytides-py3>`__ library.
"""

from __future__ import annotations

import operator as op
import string
from functools import reduce

import numpy as np

import coastal_calibration.tides.pytides.nodal_corrections as nc


class BaseConstituent:
    """A single harmonic tidal constituent."""

    __slots__ = ("coefficients", "name", "u", "f")

    xdo_int: dict[str, int] = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9,
        "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17,
        "R": -8, "S": -7, "T": -6, "U": -5, "V": -4, "W": -3, "X": -2, "Y": -1,
        "Z": 0,
    }
    int_xdo: dict[int, str] = {v: k for k, v in xdo_int.items()}

    def __init__(
        self,
        name: str,
        xdo: str = "",
        coefficients: list[float] | None = None,
        u=nc.u_zero,
        f=nc.f_unity,
    ) -> None:
        if xdo:
            self.coefficients = np.asarray(
                [self.xdo_int[c.upper()] for c in xdo if c in string.ascii_letters],
                dtype=np.float64,
            )
        else:
            self.coefficients = np.asarray(coefficients or [], dtype=np.float64)
        self.name = name
        self.u = u
        self.f = f

    # ------------------------------------------------------------------
    # Astronomical helpers
    # ------------------------------------------------------------------

    def _astro_xdo(self, a: dict) -> list:
        return [a["T+h-s"], a["s"], a["h"], a["p"], a["N"], a["pp"], a["90"]]

    def V(self, a: dict) -> float:  # noqa: N802
        """Equilibrium argument (degrees)."""
        vals = np.asarray([x.value for x in self._astro_xdo(a)], dtype=np.float64)
        return float(np.mod(self.coefficients @ vals, 360.0))

    def speed(self, a: dict) -> float:
        """Constituent speed (degrees per hour)."""
        speeds = np.asarray([x.speed for x in self._astro_xdo(a)], dtype=np.float64)
        return float(self.coefficients @ speeds)

    # Two constituents with the same speed are considered equal
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseConstituent):
            return NotImplemented
        return bool(np.all(self.coefficients[:-1] == other.coefficients[:-1]))

    def __hash__(self) -> int:
        return hash(tuple(self.coefficients[:-1]))

    def __repr__(self) -> str:
        return self.name


class CompoundConstituent(BaseConstituent):
    """A constituent built from weighted combinations of base constituents."""

    def __init__(self, members: list[tuple[BaseConstituent, int]], **kwargs) -> None:
        self.members = members
        kwargs.setdefault("u", self.u)
        kwargs.setdefault("f", self.f)
        super().__init__(**kwargs)
        self.coefficients = reduce(op.add, [c.coefficients * n for c, n in members])

    def speed(self, a: dict) -> float:
        return float(reduce(op.add, [n * c.speed(a) for c, n in self.members]))

    def V(self, a: dict) -> float:  # noqa: N802
        return float(reduce(op.add, [n * c.V(a) for c, n in self.members]))

    def u(self, a: dict) -> float:
        return float(reduce(op.add, [n * c.u(a) for c, n in self.members]))

    def f(self, a: dict) -> float:
        return float(reduce(op.mul, [c.f(a) ** abs(n) for c, n in self.members]))


# =====================================================================
# Standard constituents
# =====================================================================

# Long-term
_Z0 = BaseConstituent(name="Z0", xdo="Z ZZZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_Sa = BaseConstituent(name="Sa", xdo="Z ZAZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_Ssa = BaseConstituent(name="Ssa", xdo="Z ZBZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_Mm = BaseConstituent(name="Mm", xdo="Z AZY ZZZ", u=nc.u_zero, f=nc.f_Mm)
_Mf = BaseConstituent(name="Mf", xdo="Z BZZ ZZZ", u=nc.u_Mf, f=nc.f_Mf)

# Diurnal
_Q1 = BaseConstituent(name="Q1", xdo="A XZA ZZA", u=nc.u_O1, f=nc.f_O1)
_O1 = BaseConstituent(name="O1", xdo="A YZZ ZZA", u=nc.u_O1, f=nc.f_O1)
_K1 = BaseConstituent(name="K1", xdo="A AZZ ZZY", u=nc.u_K1, f=nc.f_K1)
_J1 = BaseConstituent(name="J1", xdo="A BZY ZZY", u=nc.u_J1, f=nc.f_J1)
_M1 = BaseConstituent(name="M1", xdo="A ZZZ ZZA", u=nc.u_M1, f=nc.f_M1)
_P1 = BaseConstituent(name="P1", xdo="A AXZ ZZA", u=nc.u_zero, f=nc.f_unity)
_S1 = BaseConstituent(name="S1", xdo="A AYZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_OO1 = BaseConstituent(name="OO1", xdo="A CZZ ZZY", u=nc.u_OO1, f=nc.f_OO1)

# Semi-diurnal
_2N2 = BaseConstituent(name="2N2", xdo="B XZB ZZZ", u=nc.u_M2, f=nc.f_M2)
_N2 = BaseConstituent(name="N2", xdo="B YZA ZZZ", u=nc.u_M2, f=nc.f_M2)
_nu2 = BaseConstituent(name="nu2", xdo="B YBY ZZZ", u=nc.u_M2, f=nc.f_M2)
_M2 = BaseConstituent(name="M2", xdo="B ZZZ ZZZ", u=nc.u_M2, f=nc.f_M2)
_lambda2 = BaseConstituent(name="lambda2", xdo="B AXA ZZB", u=nc.u_M2, f=nc.f_M2)
_L2 = BaseConstituent(name="L2", xdo="B AZY ZZB", u=nc.u_L2, f=nc.f_L2)
_T2 = BaseConstituent(name="T2", xdo="B BWZ ZAZ", u=nc.u_zero, f=nc.f_unity)
_S2 = BaseConstituent(name="S2", xdo="B BXZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_R2 = BaseConstituent(name="R2", xdo="B BYZ ZYB", u=nc.u_zero, f=nc.f_unity)
_K2 = BaseConstituent(name="K2", xdo="B BZZ ZZZ", u=nc.u_K2, f=nc.f_K2)

# Third-diurnal
_M3 = BaseConstituent(
    name="M3", xdo="C ZZZ ZZZ",
    u=lambda a: nc.u_Modd(a, 3), f=lambda a: nc.f_Modd(a, 3),
)

# Compound constituents
_MSF = CompoundConstituent(name="MSF", members=[(_S2, 1), (_M2, -1)])
_2Q1 = CompoundConstituent(name="2Q1", members=[(_N2, 1), (_J1, -1)])
_rho1 = CompoundConstituent(name="rho1", members=[(_nu2, 1), (_K1, -1)])
_mu2 = CompoundConstituent(name="mu2", members=[(_M2, 2), (_S2, -1)])
_2SM2 = CompoundConstituent(name="2SM2", members=[(_S2, 2), (_M2, -1)])
_2MK3 = CompoundConstituent(name="2MK3", members=[(_M2, 1), (_O1, 1)])
_MK3 = CompoundConstituent(name="MK3", members=[(_M2, 1), (_K1, 1)])
_MN4 = CompoundConstituent(name="MN4", members=[(_M2, 1), (_N2, 1)])
_M4 = CompoundConstituent(name="M4", members=[(_M2, 2)])
_MS4 = CompoundConstituent(name="MS4", members=[(_M2, 1), (_S2, 1)])
_S4 = CompoundConstituent(name="S4", members=[(_S2, 2)])
_M6 = CompoundConstituent(name="M6", members=[(_M2, 3)])
_S6 = CompoundConstituent(name="S6", members=[(_S2, 3)])
_M8 = CompoundConstituent(name="M8", members=[(_M2, 4)])

# The 37 NOAA standard constituents
noaa: list[BaseConstituent] = [
    _M2, _S2, _N2, _K1, _M4, _O1, _M6, _MK3, _S4, _MN4, _nu2, _S6, _mu2,
    _2N2, _OO1, _lambda2, _S1, _M1, _J1, _Mm, _Ssa, _Sa, _MSF, _Mf,
    _rho1, _Q1, _T2, _R2, _2Q1, _P1, _2SM2, _M3, _L2, _2MK3, _K2,
    _M8, _MS4,
]
