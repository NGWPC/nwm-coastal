"""Tests for :func:`coastal_calibration.plotting.plot_station_comparison`.

Covers the input-validation contract: the function rejects runs whose
elevation arrays don't line up with ``station_ids`` before attempting
to draw, so callers see a clear ValueError instead of an IndexError
inside the per-panel helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coastal_calibration.plotting.stations import plot_station_comparison


class TestShapeValidation:
    def test_empty_runs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="`runs` is empty"):
            plot_station_comparison(
                runs={},
                station_ids=["A", "B"],
                figs_dir=tmp_path,
            )

    def test_wrong_column_count_raises(self, tmp_path):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        # station_ids has 2 entries, but run provides 3 columns
        elev = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="2 columns"):
            plot_station_comparison(
                runs={"run_a": (times, elev)},
                station_ids=["A", "B"],
                figs_dir=tmp_path,
            )

    def test_one_of_many_runs_flagged(self, tmp_path):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        good = np.zeros((3, 2), dtype=np.float64)
        bad = np.zeros((3, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="'run_b'"):
            plot_station_comparison(
                runs={"run_a": (times, good), "run_b": (times, bad)},
                station_ids=["A", "B"],
                figs_dir=tmp_path,
            )

    def test_1d_array_rejected(self, tmp_path):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        elev_1d = np.zeros(3, dtype=np.float64)
        with pytest.raises(ValueError, match="2-D"):
            plot_station_comparison(
                runs={"run_a": (times, elev_1d)},
                station_ids=["A", "B"],
                figs_dir=tmp_path,
            )
