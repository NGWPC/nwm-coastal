"""Tests for coastal_calibration.utils.streamflow module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pandas as pd
import pytest

from coastal_calibration.utils.streamflow import (
    _read_from_chrtout,
    read_streamflow,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_chrtout_file(
    path: Path,
    timestamp: datetime,
    feature_ids: np.ndarray,
    streamflow: np.ndarray,
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
        t_var.units = "minutes since 1970-01-01 00:00:00"
        t_var.calendar = "standard"
        t_var[:] = netCDF4.date2num(timestamp, units=t_var.units, calendar=t_var.calendar)


@pytest.fixture
def chrtout_dir(tmp_path: Path) -> tuple[list[Path], np.ndarray]:
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

    def test_basic_read(self, chrtout_dir: tuple[list[Path], np.ndarray]) -> None:
        files, _feature_ids = chrtout_dir
        df = _read_from_chrtout(files, [100, 300, 500])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {100, 300, 500}
        assert df.iloc[0, 0] == pytest.approx(10.0)
        assert df.iloc[2, 0] == pytest.approx(12.0)

    def test_subset_feature_ids(self, chrtout_dir: tuple[list[Path], np.ndarray]) -> None:
        files, _ = chrtout_dir
        df = _read_from_chrtout(files, [200])

        assert list(df.columns) == [200]
        assert len(df) == 3

    def test_missing_feature_ids(self, chrtout_dir: tuple[list[Path], np.ndarray]) -> None:
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

    def test_nwm_ana_with_files(self, chrtout_dir: tuple[list[Path], np.ndarray]) -> None:
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
