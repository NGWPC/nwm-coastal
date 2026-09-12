"""Tests for coastal_calibration.utils.idw_interpolate."""

from __future__ import annotations

import numpy as np

from coastal_calibration.utils import idw_interpolate

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
VALUES = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])


def test_exact_source_point_returns_its_value():
    out = idw_interpolate(SQUARE, np.array([[1.0, 0.0]]), VALUES)
    np.testing.assert_allclose(out, [[2.0], [20.0]])


def test_equidistant_neighbours_average():
    out = idw_interpolate(SQUARE, np.array([[0.5, 0.5]]), VALUES)
    np.testing.assert_allclose(out, [[2.5], [25.0]])


def test_k_is_capped_at_source_count():
    out = idw_interpolate(SQUARE[:2], np.array([[0.25, 0.0]]), VALUES[:, :2], k=10)
    assert out.shape == (2, 1)
    assert 1.0 < out[0, 0] < 1.5
