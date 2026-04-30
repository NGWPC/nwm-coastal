"""Tests for :mod:`coastal_calibration.plotting.animate`.

Exercises all three dispatch paths (regular, unstructured triangular/quad,
UGRID quadtree) with tiny synthetic datasets so the full suite stays fast
and display-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_calibration.plotting import (
    animate_water_level,
    animate_water_level_comparison,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic datasets (same layouts used by test_plotting_spatial.py).
# ---------------------------------------------------------------------------


def _schism_dataset(n_time: int = 4) -> xr.Dataset:
    node_x = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    face_nodes = np.array(
        [[0, 1, 3, -1], [1, 4, 3, -1], [1, 2, 4, -1]],
        dtype=np.int64,
    )
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    elevation = np.outer(np.arange(1.0, n_time + 1.0), np.arange(1.0, 6.0))
    return xr.Dataset(
        data_vars={
            "elevation": (("time", "node"), elevation.astype(np.float32)),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(5), "face": np.arange(3)},
        attrs={"mesh_type": "ugrid-triangle-or-quad"},
    )


def _sfincs_regular_dataset(n_time: int = 4, n_y: int = 4, n_x: int = 5) -> xr.Dataset:
    x = np.linspace(0.0, 100.0, n_x)
    y = np.linspace(0.0, 80.0, n_y)
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    t_idx = np.arange(n_time).reshape(-1, 1, 1)
    y_idx = np.arange(n_y).reshape(1, -1, 1)
    x_idx = np.arange(n_x).reshape(1, 1, -1)
    zs = (t_idx + 1.0) * (y_idx + x_idx + 1.0)
    return xr.Dataset(
        data_vars={"zs": (("time", "y", "x"), zs.astype(np.float32))},
        coords={"time": times, "y": y, "x": x},
        attrs={"mesh_type": "regular"},
    )


def _sfincs_quadtree_dataset(n_time: int = 4) -> xr.Dataset:
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    face_nodes = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int64)
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    zs = np.outer(np.arange(1.0, n_time + 1.0), np.arange(1.0, 5.0)).astype(np.float32)
    return xr.Dataset(
        data_vars={
            "zs": (("time", "face"), zs),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(9), "face": np.arange(4)},
        attrs={"mesh_type": "ugrid-quadtree"},
    )


@pytest.fixture(autouse=True)
def _close_all_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# GIF path — uses PillowWriter so ffmpeg is not required in the test env.
# ---------------------------------------------------------------------------


class TestAnimateGif:
    def test_regular_grid_writes_gif(self, tmp_path: Path):
        ds = _sfincs_regular_dataset(n_time=3)
        out = animate_water_level(ds, tmp_path / "regular.gif", fps=5, dpi=60)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_unstructured_writes_gif(self, tmp_path: Path):
        ds = _schism_dataset(n_time=3)
        out = animate_water_level(ds, tmp_path / "schism.gif", fps=5, dpi=60)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_quadtree_writes_gif(self, tmp_path: Path):
        ds = _sfincs_quadtree_dataset(n_time=3)
        out = animate_water_level(ds, tmp_path / "sfincs_quadtree.gif", fps=5, dpi=60)
        assert out.is_file()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# MP4 path — runs only when ffmpeg is on PATH.
# ---------------------------------------------------------------------------


def _have_ffmpeg() -> bool:
    from matplotlib.animation import writers

    return writers.is_available("ffmpeg")


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
class TestAnimateMp4:
    def test_regular_grid_writes_mp4(self, tmp_path: Path):
        ds = _sfincs_regular_dataset(n_time=3)
        out = animate_water_level(ds, tmp_path / "regular.mp4", fps=5, dpi=60)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_time_stride_reduces_frames(self, tmp_path: Path):
        ds = _sfincs_regular_dataset(n_time=6)
        out = animate_water_level(ds, tmp_path / "strided.mp4", fps=5, dpi=60, time_stride=3)
        assert out.is_file()
        # Two output frames (0, 3); mp4 can't be inspected without ffprobe, but
        # file should at least exist and be non-empty.
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_suffix_raises(self, tmp_path: Path):
        ds = _sfincs_regular_dataset(n_time=3)
        with pytest.raises(ValueError, match="infer animation writer"):
            animate_water_level(ds, tmp_path / "out.bogus", fps=5, dpi=60)

    def test_empty_time_axis_after_stride(self, tmp_path: Path):
        # Only one timestep; stride of 10 still yields one frame (index 0).
        ds = _sfincs_regular_dataset(n_time=1)
        out = animate_water_level(ds, tmp_path / "single.gif", fps=5, dpi=60, time_stride=10)
        assert out.is_file()

    def test_explicit_writer_override(self, tmp_path: Path):
        """An explicit pillow writer should happily render to a .gif path."""
        ds = _sfincs_regular_dataset(n_time=2)
        out = animate_water_level(ds, tmp_path / "explicit.gif", fps=5, dpi=60, writer="pillow")
        assert out.is_file()

    def test_colour_limits_propagate(self, tmp_path: Path):
        ds = _sfincs_regular_dataset(n_time=2)
        out = animate_water_level(
            ds,
            tmp_path / "limited.gif",
            fps=5,
            dpi=60,
            vmin=-1.0,
            vmax=10.0,
        )
        assert out.is_file()

    def test_title_prefix_applied(self, tmp_path: Path):
        """Smoke-test the title-prefix kwarg by rendering one frame."""
        ds = _sfincs_regular_dataset(n_time=1)
        out = animate_water_level(
            ds,
            tmp_path / "titled.gif",
            fps=5,
            dpi=60,
            title_prefix="Prefix",
        )
        assert out.is_file()

    def test_mask_dry_per_frame(self, tmp_path: Path):
        """When the dataset has dryFlagNode, the mask is applied to every frame."""
        ds = _schism_dataset(n_time=3)
        # Add a dry flag where node 4 flips wet→dry between frames.
        dry = np.zeros((ds.sizes["time"], ds.sizes["node"]), dtype=np.int8)
        dry[2, 4] = 1
        ds = ds.assign(dryFlagNode=(("time", "node"), dry))

        out = animate_water_level(
            ds,
            tmp_path / "masked.gif",
            fps=5,
            dpi=60,
            mask_dry=True,
        )
        assert out.is_file()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Side-by-side comparison animation.
# ---------------------------------------------------------------------------


class TestAnimateComparison:
    def test_schism_vs_sfincs_quadtree_writes_gif(self, tmp_path: Path):
        ds_l = _schism_dataset(n_time=4)
        ds_r = _sfincs_quadtree_dataset(n_time=4)
        out = animate_water_level_comparison(
            ds_l,
            ds_r,
            tmp_path / "compare.gif",
            labels=("SCHISM", "SFINCS"),
            fps=5,
            dpi=60,
        )
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_schism_vs_regular_writes_gif(self, tmp_path: Path):
        ds_l = _schism_dataset(n_time=3)
        ds_r = _sfincs_regular_dataset(n_time=3)
        out = animate_water_level_comparison(
            ds_l,
            ds_r,
            tmp_path / "compare_reg.gif",
            labels=("SCHISM", "SFINCS-reg"),
            fps=5,
            dpi=60,
        )
        assert out.is_file()

    def test_different_cadence_picks_smaller_clock(self, tmp_path: Path):
        """Different time axes — animation runs on the fewer-frame clock."""
        ds_l = _schism_dataset(n_time=2)  # hourly @ 2024-01-01..02
        ds_r = _sfincs_quadtree_dataset(n_time=4)  # hourly @ 2024-01-01..04
        out = animate_water_level_comparison(
            ds_l,
            ds_r,
            tmp_path / "different_cadence.gif",
            fps=5,
            dpi=60,
        )
        assert out.is_file()

    def test_no_overlap_raises(self):
        ds_l = _schism_dataset(n_time=2)
        ds_r = _sfincs_quadtree_dataset(n_time=2)
        # Shift right dataset out of left's window.
        ds_r = ds_r.assign_coords(
            time=pd.date_range("2030-01-01", periods=ds_r.sizes["time"], freq="1h")
        )
        with pytest.raises(ValueError, match="overlapping time window"):
            animate_water_level_comparison(ds_l, ds_r, "/tmp/never.gif")

    def test_explicit_limits(self, tmp_path: Path):
        ds_l = _schism_dataset(n_time=2)
        ds_r = _sfincs_quadtree_dataset(n_time=2)
        out = animate_water_level_comparison(
            ds_l,
            ds_r,
            tmp_path / "limits.gif",
            fps=5,
            dpi=60,
            vmin=-1.0,
            vmax=20.0,
        )
        assert out.is_file()
