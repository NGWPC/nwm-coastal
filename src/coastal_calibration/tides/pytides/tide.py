"""Tidal prediction using harmonic constituents.

Provides the :class:`Tide` class for predicting water levels from
harmonic constituent amplitudes and phases.  Only the prediction
(forward) path is included — harmonic analysis (fitting observed data)
is not needed by this package.

Based on the modernized `pytides-py3 <https://github.com/pytides/pytides-py3>`__ library.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import numpy as np

from coastal_calibration.tides.pytides.astro import astro

_D2R: float = np.pi / 180.0
_R2D: float = 180.0 / np.pi


class Tide:
    """Harmonic tidal model for prediction.

    A ``Tide`` is initialised with a structured NumPy array whose rows
    are ``(constituent, amplitude, phase)`` triples (see :attr:`dtype`).
    Call :meth:`at` to evaluate the model at a sequence of times.

    Parameters
    ----------
    model : np.ndarray
        Structured array with dtype :attr:`Tide.dtype`.
    radians : bool
        If ``True`` the phases in *model* are in radians; they are
        converted to degrees internally.
    """

    dtype = np.dtype([("constituent", object), ("amplitude", float), ("phase", float)])

    def _normalize(self) -> None:
        """Ensure positive amplitudes and phases in [0, 360)."""
        for i, (_, amp, pha) in enumerate(self.model):
            if amp < 0:
                self.model["amplitude"][i] = -amp
                self.model["phase"][i] = pha + 180.0
            self.model["phase"][i] = np.mod(self.model["phase"][i], 360.0)

    def __init__(self, model: np.ndarray, radians: bool = False) -> None:
        if model.dtype != Tide.dtype:
            raise ValueError("model must be a numpy array with dtype == Tide.dtype")
        if radians:
            model = model.copy()
            model["phase"] = _R2D * model["phase"]
        self.model = model[:]
        self._normalize()

    @staticmethod
    def _hours(t0: datetime, t) -> np.ndarray:
        """Return hourly offsets of *t* from *t0*."""
        if not isinstance(t, Iterable):
            return Tide._hours(t0, [t])[0]
        if isinstance(t[0], datetime):
            return np.array([(ti - t0).total_seconds() / 3600.0 for ti in t])
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(constituents, t0, t=None, radians=True):
        """Compute constituent speeds, node factors, and equilibrium arguments."""
        if isinstance(t0, Iterable):
            t0 = t0[0]
        if t is None:
            t = [t0]
        if not isinstance(t, Iterable):
            t = [t]

        a0 = astro(t0)
        a = [astro(ti) for ti in t]

        V0 = np.asarray([c.V(a0) for c in constituents], dtype=np.float64)[:, np.newaxis]
        speed = np.asarray([c.speed(a0) for c in constituents], dtype=np.float64)[:, np.newaxis]
        u = [
            np.mod(
                np.asarray([c.u(ai) for c in constituents], dtype=np.float64)[:, np.newaxis],
                360.0,
            )
            for ai in a
        ]
        f = [
            np.mod(
                np.asarray([c.f(ai) for c in constituents], dtype=np.float64)[:, np.newaxis],
                360.0,
            )
            for ai in a
        ]

        if radians:
            speed = _D2R * speed
            V0 = _D2R * V0
            u = [_D2R * each for each in u]
        return speed, u, f, V0

    @staticmethod
    def _tidal_series(t, amplitude, phase, speed, u, f, V0):  # noqa: N803
        """Vectorised harmonic summation."""
        return np.sum(amplitude * f * np.cos(speed * t + V0 + u - phase), axis=0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def at(self, t: list[datetime]) -> np.ndarray:
        """Return modelled tidal heights at given times.

        Parameters
        ----------
        t : list[datetime]
            Sequence of datetime objects.

        Returns
        -------
        np.ndarray
            Water level at each time.
        """
        if not isinstance(t, Iterable) or len(t) == 0:
            raise ValueError("t must be a non-empty sequence of datetimes.")

        t0 = t[0]
        hours = self._hours(t0, t)

        speed, u, f, V0 = self._prepare(self.model["constituent"], t0, radians=True)
        H = self.model["amplitude"][:, np.newaxis]
        p = _D2R * self.model["phase"][:, np.newaxis]

        u_single = u[0] if isinstance(u, list) else u
        f_single = f[0] if isinstance(f, list) else f
        hours_arr = np.asarray(hours).reshape(1, -1)

        return self._tidal_series(hours_arr, H, p, speed, u_single, f_single, V0)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _times(
        t0: datetime,
        hours: Iterable[float] | Iterable[datetime] | float | datetime,
    ) -> list[datetime] | datetime:
        """Convert hourly offsets from *t0* to datetimes (or vice-versa)."""
        if not isinstance(hours, Iterable):
            result = Tide._times(t0, [hours])  # ty: ignore[invalid-argument-type]
            if not isinstance(result, list):
                msg = "Expected list from _times"
                raise TypeError(msg)
            return result[0]
        hours_list = list(hours)
        if not isinstance(hours_list[0], datetime):
            return [t0 + timedelta(hours=float(h)) for h in hours_list]
        return [h for h in hours_list if isinstance(h, datetime)]
