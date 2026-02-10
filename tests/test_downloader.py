"""Tests for coastal_calibration.downloader module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from coastal_calibration.downloader import (
    DateRange,
    DownloadResult,
    DownloadResults,
    _build_glofs_urls,
    _build_nwm_ana_forcing_urls,
    _build_nwm_ana_streamflow_urls,
    _build_nwm_retro_forcing_urls,
    _build_nwm_retro_streamflow_urls,
    _build_stofs_urls,
    _filter_existing,
    get_date_range,
    get_default_sources,
    get_overlapping_range,
    validate_date_ranges,
)
from coastal_calibration.utils.time import iter_hours as _iter_hours
from coastal_calibration.utils.time import parse_datetime as _parse_datetime


class TestDateRange:
    def test_validate_within_range(self):
        dr = DateRange(
            start=datetime(2020, 1, 1),
            end=datetime(2023, 12, 31),
            description="Test",
        )
        assert dr.validate(datetime(2021, 6, 1), datetime(2021, 7, 1)) is None

    def test_validate_start_too_early(self):
        dr = DateRange(
            start=datetime(2020, 1, 1),
            end=datetime(2023, 12, 31),
            description="Test",
        )
        error = dr.validate(datetime(2019, 1, 1), datetime(2021, 7, 1))
        assert error is not None
        assert "before" in error

    def test_validate_end_too_late(self):
        dr = DateRange(
            start=datetime(2020, 1, 1),
            end=datetime(2023, 12, 31),
            description="Test",
        )
        error = dr.validate(datetime(2021, 1, 1), datetime(2025, 1, 1))
        assert error is not None
        assert "after" in error

    def test_validate_open_ended(self):
        dr = DateRange(
            start=datetime(2020, 1, 1),
            end=None,
            description="Test",
        )
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        # Past date should be fine
        assert dr.validate(datetime(2021, 1, 1), datetime(2021, 2, 1)) is None

    def test_validate_future_start_open_ended(self):
        dr = DateRange(
            start=datetime(2020, 1, 1),
            end=None,
            description="Test",
        )
        error = dr.validate(datetime(2099, 1, 1), datetime(2099, 2, 1))
        assert error is not None
        assert "future" in error


class TestGetDateRange:
    def test_retro_conus(self):
        dr = get_date_range("nwm_retro", "conus")
        assert dr is not None
        assert dr.start == datetime(1979, 2, 1)

    def test_retro_hawaii(self):
        dr = get_date_range("nwm_retro", "hawaii")
        assert dr is not None
        assert dr.start == datetime(1994, 1, 1)

    def test_retro_atlgulf_maps_to_conus(self):
        dr = get_date_range("nwm_retro", "atlgulf")
        assert dr is not None
        assert dr.start == datetime(1979, 2, 1)  # same as CONUS

    def test_retro_pacific_maps_to_conus(self):
        dr = get_date_range("nwm_retro", "pacific")
        assert dr is not None
        assert dr.start == datetime(1979, 2, 1)  # same as CONUS

    def test_ana_conus(self):
        dr = get_date_range("nwm_ana", "conus")
        assert dr is not None
        assert dr.start == datetime(2018, 10, 1)
        assert dr.end is None

    def test_ana_hawaii(self):
        dr = get_date_range("nwm_ana", "hawaii")
        assert dr is not None
        assert dr.start == datetime(2021, 4, 21)
        assert dr.end is None

    def test_ana_prvi(self):
        dr = get_date_range("nwm_ana", "prvi")
        assert dr is not None
        assert dr.start == datetime(2023, 10, 1)
        assert dr.end is None

    def test_ana_alaska(self):
        dr = get_date_range("nwm_ana", "alaska")
        assert dr is not None
        assert dr.start == datetime(2023, 10, 1)
        assert dr.end is None

    def test_ana_atlgulf_maps_to_conus(self):
        dr = get_date_range("nwm_ana", "atlgulf")
        assert dr is not None
        assert dr.start == datetime(2018, 10, 1)  # same as CONUS

    def test_ana_pacific_maps_to_conus(self):
        dr = get_date_range("nwm_ana", "pacific")
        assert dr is not None
        assert dr.start == datetime(2018, 10, 1)  # same as CONUS

    def test_unknown_source(self):
        assert get_date_range("unknown_source") is None

    def test_stofs_default(self):
        dr = get_date_range("stofs")
        assert dr is not None
        assert dr.end is None


class TestGetOverlappingRange:
    def test_retro_tpxo_returns_meteo_range(self):
        overlap = get_overlapping_range("nwm_retro", "tpxo", "conus")
        assert overlap is not None
        # TPXO doesn't constrain range
        meteo = get_date_range("nwm_retro", "conus")
        assert overlap.start == meteo.start

    def test_retro_stofs_overlap(self):
        overlap = get_overlapping_range("nwm_retro", "stofs", "conus")
        assert overlap is not None
        # Overlap start should be the later of the two starts
        meteo = get_date_range("nwm_retro", "conus")
        stofs = get_date_range("stofs")
        assert overlap.start == max(meteo.start, stofs.start)

    def test_unknown_meteo_source(self):
        assert get_overlapping_range("unknown", "stofs", "conus") is None

    def test_unknown_coastal_source(self):
        assert get_overlapping_range("nwm_retro", "unknown_coastal", "conus") is None

    def test_ana_stofs_both_open_ended(self):
        overlap = get_overlapping_range("nwm_ana", "stofs", "conus")
        assert overlap is not None
        assert overlap.end is None


class TestGetDefaultSources:
    def test_pacific(self):
        meteo, boundary, start = get_default_sources("pacific")
        assert meteo in ("nwm_retro", "nwm_ana")
        assert boundary in ("stofs", "tpxo")
        assert isinstance(start, datetime)

    def test_hawaii(self):
        meteo, _boundary, _start = get_default_sources("hawaii")
        assert meteo in ("nwm_retro", "nwm_ana")

    def test_prvi(self):
        _meteo, _boundary, start = get_default_sources("prvi")
        assert isinstance(start, datetime)


class TestDownloadResult:
    def test_default_values(self):
        r = DownloadResult(source="test")
        assert r.total_files == 0
        assert r.successful == 0
        assert r.failed == 0
        assert r.file_paths == []
        assert r.errors == []


class TestDownloadResults:
    def test_has_errors(self):
        results = DownloadResults(
            meteo=DownloadResult(source="meteo"),
            hydro=DownloadResult(source="hydro"),
            coastal=DownloadResult(source="coastal", errors=["fail"]),
        )
        assert results.has_errors is True

    def test_no_errors(self):
        results = DownloadResults(
            meteo=DownloadResult(source="meteo"),
            hydro=DownloadResult(source="hydro"),
            coastal=DownloadResult(source="coastal"),
        )
        assert results.has_errors is False

    def test_iter(self):
        results = DownloadResults(
            meteo=DownloadResult(source="meteo"),
            hydro=DownloadResult(source="hydro"),
            coastal=DownloadResult(source="coastal"),
        )
        items = list(results)
        assert len(items) == 3


class TestIterHours:
    def test_basic(self):
        start = datetime(2024, 1, 1, 0)
        end = datetime(2024, 1, 1, 3)
        hours = list(_iter_hours(start, end))
        assert len(hours) == 3
        assert hours[0] == start
        assert hours[-1] == datetime(2024, 1, 1, 2)

    def test_empty_range(self):
        dt = datetime(2024, 1, 1)
        assert list(_iter_hours(dt, dt)) == []


class TestParseDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2021, 6, 11)
        assert _parse_datetime(dt) == dt

    def test_iso_string(self):
        assert _parse_datetime("2021-06-11") == datetime(2021, 6, 11)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            _parse_datetime("bad")


class TestBuildUrls:
    def test_retro_forcing_urls_conus(self, tmp_path):
        start = datetime(2021, 6, 11, 0)
        end = datetime(2021, 6, 11, 2)
        urls, paths = _build_nwm_retro_forcing_urls(start, end, tmp_path, "conus")
        assert len(urls) == 2
        assert len(paths) == 2
        assert "noaa-nwm-retrospective-3-0-pds" in urls[0]
        assert "CONUS" in urls[0]
        assert "LDASIN_DOMAIN1" in urls[0]

    def test_retro_forcing_urls_hawaii(self, tmp_path):
        start = datetime(2010, 1, 1, 0)
        end = datetime(2010, 1, 1, 1)
        urls, _paths = _build_nwm_retro_forcing_urls(start, end, tmp_path, "hawaii")
        assert "Hawaii" in urls[0]

    def test_retro_streamflow_urls(self, tmp_path):
        start = datetime(2021, 6, 11, 0)
        end = datetime(2021, 6, 11, 1)
        urls, _paths = _build_nwm_retro_streamflow_urls(start, end, tmp_path, "conus")
        assert len(urls) == 1
        assert "CHRTOUT" in urls[0]

    def test_retro_streamflow_urls_hawaii_subhourly(self, tmp_path):
        start = datetime(2010, 1, 1, 0)
        end = datetime(2010, 1, 1, 1)
        urls, _paths = _build_nwm_retro_streamflow_urls(start, end, tmp_path, "hawaii")
        # Hawaii has 4 files per hour (hourly + 15/30/45 min)
        assert len(urls) == 4

    def test_ana_forcing_urls(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 2)
        urls, paths = _build_nwm_ana_forcing_urls(start, end, tmp_path, "conus")
        assert len(urls) == 2
        assert "storage.googleapis.com" in urls[0]
        assert "analysis_assim" in urls[0]
        # Local paths use native .nc names (no LDASIN renaming)
        assert paths[0].suffix == ".nc"
        assert "analysis_assim.forcing" in paths[0].name

    def test_ana_forcing_urls_hawaii(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, paths = _build_nwm_ana_forcing_urls(start, end, tmp_path, "hawaii")
        assert "hawaii" in urls[0]
        assert paths[0].suffix == ".nc"

    def test_ana_streamflow_urls_conus(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, _paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "conus")
        assert len(urls) == 1
        assert "channel_rt" in urls[0]

    def test_ana_streamflow_urls_hawaii_old_naming(self, tmp_path):
        """Before 2021-04-21: 1 hourly file with tm02 pattern."""
        start = datetime(2021, 4, 1, 0)
        end = datetime(2021, 4, 1, 1)
        urls, _paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "hawaii")
        assert len(urls) == 1
        assert "channel_rt.tm02.hawaii.nc" in urls[0]

    def test_ana_streamflow_urls_hawaii_new_naming(self, tmp_path):
        """From 2021-04-21: 4 sub-hourly files with tm0200/tm0145/tm0130/tm0115."""
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, _paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "hawaii")
        assert len(urls) == 4
        assert "channel_rt.tm0200.hawaii.nc" in urls[0]
        assert "channel_rt.tm0145.hawaii.nc" in urls[1]
        assert "channel_rt.tm0130.hawaii.nc" in urls[2]
        assert "channel_rt.tm0115.hawaii.nc" in urls[3]

    def test_ana_streamflow_urls_alaska(self, tmp_path):
        start = datetime(2024, 1, 15, 1)
        end = datetime(2024, 1, 15, 2)
        urls, _paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "alaska")
        assert len(urls) == 1
        assert "analysis_assim_alaska" in urls[0]
        assert "channel_rt.tm02.alaska.nc" in urls[0]

    def test_ana_forcing_urls_alaska(self, tmp_path):
        start = datetime(2024, 1, 15, 1)
        end = datetime(2024, 1, 15, 2)
        urls, _paths = _build_nwm_ana_forcing_urls(start, end, tmp_path, "alaska")
        assert "forcing_analysis_assim_alaska" in urls[0]
        assert "forcing.tm02.alaska.nc" in urls[0]

    def test_stofs_urls_old_naming(self, tmp_path):
        start = datetime(2022, 6, 1, 12)
        urls, paths = _build_stofs_urls(start, tmp_path)
        assert len(urls) == 1
        assert "estofs" in urls[0]
        assert "estofs.20220601" in str(paths[0])

    def test_stofs_urls_new_naming(self, tmp_path):
        start = datetime(2023, 6, 1, 12)
        urls, paths = _build_stofs_urls(start, tmp_path)
        assert len(urls) == 1
        assert "stofs_2d_glo" in urls[0]
        assert "stofs_2d_glo.20230601" in str(paths[0])

    def test_stofs_cycle_rounding(self, tmp_path):
        # 14:00 should round to t12z cycle
        start = datetime(2023, 6, 1, 14)
        urls, _ = _build_stofs_urls(start, tmp_path)
        assert "t12z" in urls[0]

    def test_stofs_path_includes_date(self, tmp_path):
        """Different dates produce different local paths to avoid skip_existing collisions."""
        _, paths_a = _build_stofs_urls(datetime(2022, 6, 1, 0), tmp_path)
        _, paths_b = _build_stofs_urls(datetime(2022, 7, 1, 0), tmp_path)
        assert paths_a[0] != paths_b[0]

    def test_glofs_urls(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 3)
        urls, _paths = _build_glofs_urls(start, end, tmp_path, "leofs")
        assert len(urls) == 3
        assert "leofs" in urls[0]
        assert "lake-erie" in urls[0]

    def test_glofs_urls_lmhofs(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, _paths = _build_glofs_urls(start, end, tmp_path, "lmhofs")
        assert "lake-michigan-huron" in urls[0]


class TestFilterExisting:
    # Valid HDF5 magic header (8 bytes) + padding.
    _HDF5_HEADER = b"\x89HDF\r\n\x1a\n" + b"\x01" * 1024

    def test_no_existing(self, tmp_path):
        urls = ["http://a", "http://b"]
        paths = [tmp_path / "a.nc", tmp_path / "b.nc"]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 2
        assert existing == 0

    @patch("coastal_calibration.downloader.check_downloads", return_value={})
    def test_some_existing(self, mock_check, tmp_path):
        """Valid .nc file passes magic-byte + remote-size check."""
        urls = ["http://a", "http://b"]
        f = tmp_path / "a.nc"
        f.write_bytes(self._HDF5_HEADER)
        paths = [f, tmp_path / "b.nc"]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 1
        assert existing == 1
        assert pending_urls[0] == "http://b"

    def test_empty_file_not_counted(self, tmp_path):
        urls = ["http://a"]
        f = tmp_path / "a.nc"
        f.write_text("")
        paths = [f]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 1
        assert existing == 0

    def test_corrupt_nc_requeued(self, tmp_path):
        """A .nc file with invalid magic bytes is deleted and re-queued."""
        urls = ["http://a"]
        f = tmp_path / "a.nc"
        f.write_text("not-a-netcdf-file")
        paths = [f]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 1
        assert existing == 0
        assert not f.exists(), "corrupt file should be deleted"

    @patch("coastal_calibration.downloader.check_downloads")
    def test_truncated_nc_requeued(self, mock_check, tmp_path):
        """A .nc file that passes magic-byte check but fails size check is re-queued."""
        f = tmp_path / "a.nc"
        f.write_bytes(self._HDF5_HEADER)
        mock_check.return_value = {f: 99999}
        urls = ["http://a"]
        paths = [f]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 1
        assert existing == 0
        assert not f.exists(), "truncated file should be deleted"

    def test_non_nc_file_skipped_by_size(self, tmp_path):
        """Non-.nc files are accepted with a simple size > 0 check."""
        urls = ["http://a"]
        f = tmp_path / "a.txt"
        f.write_text("data")
        paths = [f]
        pending_urls, _pending_paths, existing = _filter_existing(urls, paths)
        assert len(pending_urls) == 0
        assert existing == 1


class TestValidateDateRanges:
    def test_valid_range(self):
        errors = validate_date_ranges(
            datetime(2021, 6, 11),
            datetime(2021, 6, 12),
            "nwm_retro",
            "stofs",
            "conus",
        )
        assert len(errors) == 0

    def test_invalid_meteo_range(self):
        errors = validate_date_ranges(
            datetime(1970, 1, 1),
            datetime(1970, 2, 1),
            "nwm_retro",
            "tpxo",
            "conus",
        )
        assert len(errors) > 0

    def test_tpxo_skips_coastal_validation(self):
        errors = validate_date_ranges(
            datetime(2021, 6, 11),
            datetime(2021, 6, 12),
            "nwm_retro",
            "tpxo",
            "conus",
        )
        assert len(errors) == 0
