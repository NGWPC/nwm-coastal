"""Tests for coastal_calibration.data.glofs (offline; remote reads are faked)."""

from __future__ import annotations

from datetime import datetime

import netCDF4
import numpy as np
import pytest

from coastal_calibration.data import glofs
from coastal_calibration.data.downloader import (
    get_date_range,
    get_default_sources,
    validate_date_ranges,
)
from coastal_calibration.data.glofs import (
    ensure_glofs_waterlevel,
    fetch_glofs,
    glofs_sources,
    glofs_waterlevel_path,
)

BASE = glofs.GLOFS_BASE_URL


def _one(model: str, hour: datetime) -> glofs.GlofsSource:
    (src,) = glofs_sources(model, hour, hour.replace(minute=0) + glofs._HOUR)
    return src


class TestGlofsSources:
    """Hour → file mapping, checked against files listed on NCEI (2026-09)."""

    @pytest.mark.parametrize(
        ("model", "lake_dir"),
        [
            ("leofs", "lake-erie-operational-forecast-system-leofs"),
            ("lmhofs", "lake-michigan-and-huron-operational-forecast-system-lmhofs"),
            ("loofs", "lake-ontario-operational-forecast-system-loofs"),
            ("lsofs", "lake-superior-operational-forecast-system-lsofs"),
        ],
    )
    def test_lake_directories(self, model, lake_dir):
        src = _one(model, datetime(2025, 6, 10, 0))
        assert src.url == f"{BASE}/{lake_dir}/2025/06/{model}.t06z.20250610.fields.n000.nc"
        assert src.fmt == "hdf5"

    def test_nowcast_cycle_looks_back(self):
        """00Z lives in t06z n000; 05Z in t06z n005; 06Z in t12z n000."""
        names = [
            s.url.rsplit("/", 1)[1]
            for s in glofs_sources("leofs", datetime(2024, 8, 10, 0), datetime(2024, 8, 10, 7))
        ]
        assert names[0] == "nos.leofs.fields.n000.20240810.t06z.nc"
        assert names[5] == "nos.leofs.fields.n005.20240810.t06z.nc"
        assert names[6] == "nos.leofs.fields.n000.20240810.t12z.nc"

    def test_late_hours_roll_into_next_days_t00z(self):
        src = _one("leofs", datetime(2024, 8, 10, 23))
        assert src.url.endswith("/2024/08/nos.leofs.fields.n005.20240811.t00z.nc")

    def test_month_folder_follows_cycle_date(self):
        src = _one("lsofs", datetime(2025, 1, 31, 23))
        assert src.url.endswith("/2025/02/lsofs.t00z.20250201.fields.n005.nc")

    def test_hdf5_layout_starts_with_cycle_2024_09_09(self):
        assert _one("leofs", datetime(2024, 9, 8, 17)).fmt == "netcdf3"
        after = _one("leofs", datetime(2024, 9, 8, 18))
        assert after.fmt == "hdf5"
        assert after.url.endswith("/leofs.t00z.20240909.fields.n000.nc")

    def test_pom_era_holds_six_hours_ending_at_cycle(self):
        at_cycle = _one("loofs", datetime(2022, 6, 10, 6))
        first = _one("loofs", datetime(2022, 6, 10, 1))
        assert at_cycle.url == first.url
        assert at_cycle.url.endswith("/glofs.loofs.fields.nowcast.20220610.t06z.nc")
        assert (first.step, at_cycle.step, at_cycle.fmt) == (0, 5, "pom")
        midnight = _one("loofs", datetime(2022, 6, 10, 0))
        assert midnight.url.endswith(".20220610.t00z.nc")
        assert midnight.step == 5

    def test_fvcom_replaces_pom_from_cycle_2022_10_20(self):
        assert _one("loofs", datetime(2022, 10, 19, 17)).fmt == "pom"
        assert _one("loofs", datetime(2022, 10, 19, 18)).url.endswith(
            "/nos.loofs.fields.n000.20221020.t00z.nc"
        )

    def test_erie_and_michigan_huron_have_no_pom_era(self):
        assert _one("leofs", datetime(2017, 1, 1, 0)).fmt == "netcdf3"

    def test_window_is_half_open(self):
        assert len(glofs_sources("leofs", datetime(2025, 1, 1), datetime(2025, 1, 2))) == 24

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown GLOFS model"):
            glofs_sources("lcofs", datetime(2025, 1, 1), datetime(2025, 1, 1, 1))


def _fake_reader(calls, n_nodes=5, grid_shift_at=None, time_offset_hours=0):
    """Stand-in for glofs._read_remote that reports each step's valid time."""

    def read(url, fmt, steps):
        calls.append((url, tuple(steps)))
        srcs = [s for s in _fake_reader.index[url] if s.step in steps]
        shift = time_offset_hours
        times = [s.valid_time.replace(hour=(s.valid_time.hour + shift) % 24) for s in srcs]
        lon = np.linspace(-83.0, -79.0, n_nodes)
        if grid_shift_at is not None and srcs[0].valid_time >= grid_shift_at:
            lon = lon + 0.01
        lat = np.full(n_nodes, 42.0)
        zeta = np.array([[s.valid_time.hour / 10.0] * n_nodes for s in srcs], dtype=np.float32)
        return lon, lat, zeta, times

    return read


@pytest.fixture
def fake_remote(monkeypatch):
    """Patch remote reads; returns the list of (url, steps) calls made."""
    calls: list[tuple[str, tuple[int, ...]]] = []
    real_sources = glofs.glofs_sources

    def indexed_sources(model, start, end):
        srcs = real_sources(model, start, end)
        _fake_reader.index = {}
        for s in srcs:
            _fake_reader.index.setdefault(s.url, []).append(s)
        return srcs

    monkeypatch.setattr(glofs, "glofs_sources", indexed_sources)
    monkeypatch.setattr(glofs, "_read_remote", _fake_reader(calls))
    return calls


class TestFetchGlofs:
    def test_caches_each_hour_and_resumes(self, tmp_path, fake_remote):
        start, end = datetime(2025, 6, 10, 0), datetime(2025, 6, 10, 3)
        first = fetch_glofs("leofs", start, end, tmp_path)
        assert (first.successful, first.failed, len(fake_remote)) == (3, 0, 3)

        again = fetch_glofs("leofs", start, end, tmp_path)
        assert again.successful == 3
        assert len(fake_remote) == 3  # nothing re-read

    def test_reads_each_pom_file_once(self, tmp_path, fake_remote):
        fetch_glofs("loofs", datetime(2022, 6, 10, 1), datetime(2022, 6, 10, 7), tmp_path)
        assert len(fake_remote) == 1
        assert fake_remote[0][1] == (0, 1, 2, 3, 4, 5)

    def test_rejects_file_valid_at_a_different_hour(self, tmp_path, monkeypatch, fake_remote):
        monkeypatch.setattr(glofs, "_read_remote", _fake_reader([], time_offset_hours=6))
        with pytest.raises(ValueError, match="expected 2025-06-10 00:00"):
            fetch_glofs("leofs", datetime(2025, 6, 10), datetime(2025, 6, 10, 1), tmp_path)

    def test_read_failure_is_reported_or_raised(self, tmp_path, monkeypatch, fake_remote):
        def boom(url, fmt, steps):
            raise FileNotFoundError(url)

        monkeypatch.setattr(glofs, "_read_remote", boom)
        start, end = datetime(2025, 6, 10), datetime(2025, 6, 10, 2)
        result = fetch_glofs("leofs", start, end, tmp_path, raise_on_error=False)
        assert (result.successful, result.failed, len(result.errors)) == (0, 2, 2)
        with pytest.raises(RuntimeError, match="could not read"):
            fetch_glofs("leofs", start, end, tmp_path, raise_on_error=True)


class TestEnsureGlofsWaterlevel:
    def test_merges_window_inclusive_of_end_hour(self, tmp_path, fake_remote):
        start = datetime(2025, 6, 10, 0)
        fetch_glofs("leofs", start, datetime(2025, 6, 10, 4), tmp_path)
        out = ensure_glofs_waterlevel(tmp_path, "leofs", start, 3)

        assert out == glofs_waterlevel_path(tmp_path, "leofs", start, 3)
        assert out.name == "glofs_leofs_2025061000_2025061003.nc"
        with netCDF4.Dataset(out) as ds:
            times = netCDF4.num2date(ds["time"][:], ds["time"].units)
            assert [t.hour for t in times] == [0, 1, 2, 3]
            assert ds["zeta"].dimensions == ("time", "node")
            assert set(ds.variables) >= {"x", "y", "zeta", "time"}
            np.testing.assert_allclose(ds["zeta"][:, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-6)

    def test_reuses_existing_merge(self, tmp_path, fake_remote):
        start = datetime(2025, 6, 10, 0)
        fetch_glofs("leofs", start, datetime(2025, 6, 10, 2), tmp_path)
        out = ensure_glofs_waterlevel(tmp_path, "leofs", start, 1)
        mtime = out.stat().st_mtime_ns
        assert ensure_glofs_waterlevel(tmp_path, "leofs", start, 1).stat().st_mtime_ns == mtime

    def test_missing_hours_name_the_gap(self, tmp_path, fake_remote):
        start = datetime(2025, 6, 10, 0)
        fetch_glofs("leofs", start, datetime(2025, 6, 10, 2), tmp_path)
        gap = r"2 of 4 .*2025-06-10 02:00 … 2025-06-10 03:00"
        with pytest.raises(FileNotFoundError, match=gap):
            ensure_glofs_waterlevel(tmp_path, "leofs", start, 3)

    def test_grid_change_inside_window(self, tmp_path, monkeypatch, fake_remote):
        monkeypatch.setattr(
            glofs, "_read_remote", _fake_reader([], grid_shift_at=datetime(2025, 6, 10, 2))
        )
        start = datetime(2025, 6, 10, 0)
        fetch_glofs("leofs", start, datetime(2025, 6, 10, 4), tmp_path)
        with pytest.raises(ValueError, match="changed grids at 2025-06-10 02:00"):
            ensure_glofs_waterlevel(tmp_path, "leofs", start, 3)


class TestGlofsDateRanges:
    def test_per_lake_first_day(self):
        assert get_date_range("glofs", "leofs").start == datetime(2016, 3, 10)
        assert get_date_range("glofs", "lmhofs").start == datetime(2019, 9, 17)
        assert get_date_range("glofs", "loofs").start == datetime(2016, 3, 1)

    def test_validation_uses_the_lake(self):
        start, end = datetime(2018, 1, 1), datetime(2018, 1, 2)
        assert validate_date_ranges(start, end, "nwm_retro", "glofs", "greatlakes", "leofs") == []
        errors = validate_date_ranges(start, end, "nwm_retro", "glofs", "greatlakes", "lmhofs")
        assert len(errors) == 1
        assert "2019-09-17" in errors[0]

    def test_greatlakes_defaults_to_glofs(self):
        _, boundary, start = get_default_sources("greatlakes")
        assert boundary == "glofs"
        assert start >= datetime(2016, 3, 10)
