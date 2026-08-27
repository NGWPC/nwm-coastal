"""Tests for coastal_calibration.data.streamflow module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np
import pandas as pd
import pytest

from coastal_calibration.data.streamflow import (
    _read_from_chrtout,
    _read_from_troute,
    read_streamflow,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _create_chrtout_file(
    path: Path,
    timestamp: datetime,
    feature_ids: NDArray[np.integer[Any]],
    streamflow: NDArray[np.floating[Any]],
) -> None:
    """Write a minimal NWM-like CHRTOUT netCDF file."""
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("feature_id", len(feature_ids))
        ds.createDimension("time", 1)

        fid_var = ds.createVariable("feature_id", "i8", ("feature_id",))
        fid_var[:] = feature_ids

        sf_var = ds.createVariable("streamflow", "f4", ("feature_id",))
        sf_var[:] = streamflow

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var.setncatts({"units": "minutes since 1970-01-01 00:00:00", "calendar": "standard"})
        t_var[:] = netCDF4.date2num(timestamp, units=t_var.units, calendar=t_var.calendar)


@pytest.fixture
def chrtout_dir(tmp_path: Path) -> tuple[list[Path], NDArray[np.int64]]:
    """Create a directory with 3 hourly CHRTOUT files."""
    feature_ids = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    files: list[Path] = []

    for hour in range(3):
        dt = datetime(2020, 6, 1, hour, tzinfo=UTC)
        fname = f"2020060100{hour:02d}00.CHRTOUT_DOMAIN1"
        path = tmp_path / fname
        sf = np.full(len(feature_ids), 10.0 + hour, dtype=np.float32)
        _create_chrtout_file(path, dt, feature_ids, sf)
        files.append(path)

    return files, feature_ids


class TestReadFromChrtout:
    """Tests for the netCDF4 direct-read path."""

    def test_basic_read(self, chrtout_dir: tuple[list[Path], NDArray[np.int64]]) -> None:
        files, _feature_ids = chrtout_dir
        df = _read_from_chrtout(files, [100, 300, 500])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {100, 300, 500}
        assert df.iloc[0, 0] == pytest.approx(10.0)
        assert df.iloc[2, 0] == pytest.approx(12.0)

    def test_subset_feature_ids(self, chrtout_dir: tuple[list[Path], NDArray[np.int64]]) -> None:
        files, _ = chrtout_dir
        df = _read_from_chrtout(files, [200])

        assert list(df.columns) == [200]
        assert len(df) == 3

    def test_missing_feature_ids(self, chrtout_dir: tuple[list[Path], NDArray[np.int64]]) -> None:
        files, _ = chrtout_dir
        df = _read_from_chrtout(files, [999999])

        assert df.empty

    def test_empty_files_list(self) -> None:
        df = _read_from_chrtout([], [100])
        assert df.empty

    def test_mixed_feature_id_layouts(self, tmp_path: Path) -> None:
        """Files with different feature_id arrays must not crash (GH-19)."""
        # File 1: large feature_id array (like CONUS NWM, 2.7M reaches)
        big_fids = np.arange(1, 10_001, dtype=np.int64)
        big_sf = np.full(len(big_fids), 5.0, dtype=np.float32)
        f1 = tmp_path / "202401090000.CHRTOUT_DOMAIN1"
        _create_chrtout_file(f1, datetime(2024, 1, 9, tzinfo=UTC), big_fids, big_sf)

        # File 2: small feature_id array (like Hawaii NWM, 13K reaches)
        small_fids = np.arange(1, 101, dtype=np.int64)
        small_sf = np.full(len(small_fids), 3.0, dtype=np.float32)
        f2 = tmp_path / "202401090100.CHRTOUT_DOMAIN1"
        _create_chrtout_file(f2, datetime(2024, 1, 9, 1, tzinfo=UTC), small_fids, small_sf)

        # Request a feature_id that exists only in the big file.
        # The small file is skipped (no matching features) so we get 1 row.
        df = _read_from_chrtout([f1, f2], [5000])
        assert len(df) == 1
        assert 5000 in df.columns
        assert df.iloc[0][5000] == pytest.approx(5.0)


class TestReadStreamflow:
    """Tests for the public read_streamflow interface."""

    def test_nwm_ana_with_files(self, chrtout_dir: tuple[list[Path], NDArray[np.int64]]) -> None:
        files, _ = chrtout_dir
        df = read_streamflow(
            [100, 300],
            datetime(2020, 6, 1, 0, tzinfo=UTC),
            datetime(2020, 6, 1, 2, tzinfo=UTC),
            meteo_source="nwm_ana",
            chrtout_files=files,
        )

        assert len(df) == 3
        assert set(df.columns) == {100, 300}

    def test_nwm_ana_requires_files(self) -> None:
        with pytest.raises(ValueError, match="chrtout_files is required"):
            read_streamflow(
                [100],
                datetime(2020, 6, 1),
                datetime(2020, 6, 2),
                meteo_source="nwm_ana",
            )

    def test_empty_feature_ids(self) -> None:
        df = read_streamflow(
            [],
            datetime(2020, 6, 1),
            datetime(2020, 6, 2),
            meteo_source="nwm_retro",
        )
        assert df.empty

    def test_ngen_forecast_requires_troute_file(self) -> None:
        with pytest.raises(ValueError, match="troute_file is required"):
            read_streamflow(
                [100],
                datetime(2026, 3, 30, 7),
                datetime(2026, 3, 31, 1),
                meteo_source="ngen_forecast",
            )


def _create_troute_file(
    path: Path,
    feature_ids: NDArray[np.integer[Any]],
    ref: datetime,
    n_times: int = 18,
) -> NDArray[np.floating[Any]]:
    """Write a minimal t-route output netCDF and return the flow array.

    Mirrors the real ``troute_output_*.nc`` layout: ``feature_id`` int64,
    ``time`` in ``seconds since <ref>`` starting at +3600 s, and
    ``flow(feature_id, time)`` in m³/s (feature-major, i.e. transposed
    versus CHRTOUT).
    """
    n_fid = len(feature_ids)
    # Distinct value per (reach, step) so placement/transpose is checkable.
    flow = (
        np.arange(n_fid, dtype=np.float32)[:, None] * 100.0
        + np.arange(n_times, dtype=np.float32)[None, :]
    )
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("feature_id", n_fid)
        ds.createDimension("time", n_times)

        fid_var = ds.createVariable("feature_id", "i8", ("feature_id",))
        fid_var[:] = feature_ids

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var.units = f"seconds since {ref.strftime('%Y-%m-%d %H:%M:%S')}"
        t_var.calendar = "standard"
        t_var[:] = (np.arange(n_times, dtype=np.float64) + 1) * 3600.0  # +1h..+n h

        f_var = ds.createVariable("flow", "f4", ("feature_id", "time"))
        f_var.units = "m3 s-1"
        f_var[:] = flow
    return flow


class TestReadFromTroute:
    """Tests for the t-route (ngen_forecast) read path."""

    @pytest.fixture
    def troute_file(self, tmp_path: Path) -> tuple[Path, NDArray[np.int64], NDArray[np.floating[Any]]]:
        # 16-digit NextGen hydrofabric-style ids.
        fids = np.array(
            [1072639236903480, 1073115932546594, 1075116176753856, 1272226984494809],
            dtype=np.int64,
        )
        path = tmp_path / "troute_output_202603300700.nc"
        flow = _create_troute_file(path, fids, datetime(2026, 3, 30, 7))
        return path, fids, flow

    def test_basic_read_shape_and_time(self, troute_file) -> None:
        path, fids, _ = troute_file
        df = _read_from_troute(
            path, [int(fids[0]), int(fids[2])], datetime(2026, 3, 30, 7), datetime(2026, 3, 31, 3)
        )
        # 18 hourly steps, 2 requested reaches.
        assert df.shape == (18, 2)
        assert list(df.columns) == [int(fids[0]), int(fids[2])]
        # time axis starts at ref + 1h, hourly, time-major after transpose.
        assert df.index[0] == pd.Timestamp("2026-03-30 08:00:00")
        assert df.index[-1] == pd.Timestamp("2026-03-31 01:00:00")

    def test_transpose_places_values_correctly(self, troute_file) -> None:
        path, fids, flow = troute_file
        # reach at file-row 2 -> its timeseries is flow[2, :]
        df = _read_from_troute(
            path, [int(fids[2])], datetime(2026, 3, 30, 7), datetime(2026, 3, 31, 3)
        )
        np.testing.assert_allclose(df[int(fids[2])].to_numpy(), flow[2, :])

    def test_missing_ids_dropped(self, troute_file) -> None:
        path, fids, _ = troute_file
        df = _read_from_troute(
            path, [int(fids[1]), 999999999999999], datetime(2026, 3, 30, 7), datetime(2026, 3, 31, 3)
        )
        assert list(df.columns) == [int(fids[1])]

    def test_no_ids_found_returns_empty(self, troute_file) -> None:
        path, _, _ = troute_file
        df = _read_from_troute(
            path, [111111111111111], datetime(2026, 3, 30, 7), datetime(2026, 3, 31, 3)
        )
        assert df.empty

    def test_window_filters_time(self, troute_file) -> None:
        path, fids, _ = troute_file
        # Only the first 3 output hours (08,09,10).
        df = _read_from_troute(
            path, [int(fids[0])], datetime(2026, 3, 30, 7), datetime(2026, 3, 30, 10)
        )
        assert df.shape == (3, 1)
        assert df.index[-1] == pd.Timestamp("2026-03-30 10:00:00")

    def test_public_interface_ngen_forecast(self, troute_file) -> None:
        path, fids, _ = troute_file
        df = read_streamflow(
            [int(fids[0])],
            datetime(2026, 3, 30, 7),
            datetime(2026, 3, 31, 3),
            meteo_source="ngen_forecast",
            troute_file=path,
        )
        assert df.shape == (18, 1)
