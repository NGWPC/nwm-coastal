"""Tests for coastal_calibration.data.downloader module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coastal_calibration.data.downloader import (
    DateRange,
    DownloadResult,
    DownloadResults,
    _build_glofs_urls,
    _build_nwm_ana_forcing_urls,
    _build_nwm_ana_streamflow_urls,
    _build_nwm_retro_forcing_urls,
    _build_stofs_mesh_urls,
    _build_stofs_urls,
    _execute_download,
    _hour_range,
    _stofs_cycle_url,
    get_date_range,
    get_default_sources,
    get_overlapping_range,
    get_stofs_path,
    resolve_stofs_cycle,
    validate_date_ranges,
    write_nwm_grid_sidecar,
)


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
        assert dr.start == datetime(1994, 1, 2)

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
    def test_retro_harmonic_returns_meteo_range(self):
        overlap = get_overlapping_range("nwm_retro", "harmonic", "conus")
        assert overlap is not None
        # Harmonic tide prediction doesn't constrain the date range
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
        assert boundary in ("stofs", "harmonic")
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


class TestHourRange:
    def test_basic(self):
        start = datetime(2024, 1, 1, 0)
        end = datetime(2024, 1, 1, 3)
        assert list(_hour_range(start, end)) == [0, 1, 2]

    def test_empty_range(self):
        dt = datetime(2024, 1, 1)
        assert list(_hour_range(dt, dt)) == []


class TestParseDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2021, 6, 11)
        assert pd.to_datetime(dt).to_pydatetime() == dt

    def test_iso_string(self):
        assert pd.to_datetime("2021-06-11").to_pydatetime() == datetime(2021, 6, 11)

    def test_invalid(self):
        with pytest.raises(ValueError, match="bad"):
            pd.to_datetime("bad", format="mixed")


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
        assert paths[0].parent == tmp_path / "meteo" / "nwm_retro" / "conus"

    def test_retro_forcing_urls_hawaii(self, tmp_path):
        start = datetime(2010, 1, 1, 0)
        end = datetime(2010, 1, 1, 1)
        urls, paths = _build_nwm_retro_forcing_urls(start, end, tmp_path, "hawaii")
        assert "Hawaii" in urls[0]
        assert paths[0].parent == tmp_path / "meteo" / "nwm_retro" / "hawaii"

    def test_ana_forcing_urls(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 2)
        urls, paths = _build_nwm_ana_forcing_urls(start, end, tmp_path, "conus")
        assert len(urls) == 2
        assert "storage.googleapis.com" in urls[0]
        assert "analysis_assim" in urls[0]
        # Local paths use YYYYMMDDHH.LDASIN_DOMAIN1 naming (same as nwm_retro)
        assert paths[0].name == "2023010100.LDASIN_DOMAIN1"
        assert paths[1].name == "2023010101.LDASIN_DOMAIN1"

    def test_ana_forcing_urls_hawaii(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, paths = _build_nwm_ana_forcing_urls(start, end, tmp_path, "hawaii")
        assert "hawaii" in urls[0]
        assert paths[0].name == "2023010100.LDASIN_DOMAIN1"
        assert paths[0].parent == tmp_path / "meteo" / "nwm_ana" / "hawaii"

    def test_ana_streamflow_urls_conus(self, tmp_path):
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "conus")
        assert len(urls) == 1
        assert "channel_rt" in urls[0]
        assert paths[0].parent.name == "conus"

    def test_ana_streamflow_urls_hawaii_old_naming(self, tmp_path):
        """Before 2021-04-21: 1 hourly file with tm02 pattern."""
        start = datetime(2021, 4, 1, 0)
        end = datetime(2021, 4, 1, 1)
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "hawaii")
        assert len(urls) == 1
        assert "channel_rt.tm02.hawaii.nc" in urls[0]
        assert paths[0].parent.name == "hawaii"

    def test_ana_streamflow_urls_hawaii_new_naming(self, tmp_path):
        """From 2021-04-21: 4 sub-hourly files with tm0200/tm0145/tm0130/tm0115."""
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "hawaii")
        assert len(urls) == 4
        assert "channel_rt.tm0200.hawaii.nc" in urls[0]
        assert "channel_rt.tm0145.hawaii.nc" in urls[1]
        assert "channel_rt.tm0130.hawaii.nc" in urls[2]
        assert "channel_rt.tm0115.hawaii.nc" in urls[3]
        assert all(p.parent.name == "hawaii" for p in paths)

    def test_ana_streamflow_urls_alaska(self, tmp_path):
        start = datetime(2024, 1, 15, 1)
        end = datetime(2024, 1, 15, 2)
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "alaska")
        assert len(urls) == 1
        assert "analysis_assim_alaska" in urls[0]
        assert "channel_rt.tm02.alaska.nc" in urls[0]
        assert paths[0].parent.name == "alaska"

    def test_ana_streamflow_domain_isolation(self, tmp_path):
        """CONUS and Hawaii files go to separate subdirectories (GH-19)."""
        start = datetime(2023, 1, 1, 0)
        end = datetime(2023, 1, 1, 1)
        _, conus_paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "conus")
        _, hawaii_paths = _build_nwm_ana_streamflow_urls(start, end, tmp_path, "hawaii")
        # They must not share a directory
        assert conus_paths[0].parent != hawaii_paths[0].parent

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
        """Different dates produce different local paths."""
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


class TestExecuteDownload:
    """Re-run behavior: existing final files must skip the network call."""

    def test_skips_when_all_finals_exist(self, tmp_path, monkeypatch):
        finals = [tmp_path / "a.nc", tmp_path / "b.nc"]
        for f in finals:
            f.write_bytes(b"cached")

        calls: list[list[str]] = []

        def fake_download(urls, paths, **_kwargs):
            calls.append(list(urls))

        monkeypatch.setattr("coastal_calibration.data.downloader.download", fake_download)

        result = _execute_download(
            ["http://example/a", "http://example/b"],
            finals,
            "test",
            timeout=1,
            raise_on_error=False,
        )

        assert calls == []  # downloader never invoked
        assert result.successful == 2
        assert result.failed == 0
        assert result.errors == []

    def test_downloads_only_missing(self, tmp_path, monkeypatch):
        cached = tmp_path / "cached.nc"
        missing = tmp_path / "missing.nc"
        cached.write_bytes(b"cached")

        seen_urls: list[str] = []

        def fake_download(urls, paths, **_kwargs):
            seen_urls.extend(urls)
            for p in paths:
                Path(p).write_bytes(b"new")

        monkeypatch.setattr("coastal_calibration.data.downloader.download", fake_download)

        result = _execute_download(
            ["http://example/cached", "http://example/missing"],
            [cached, missing],
            "test",
            timeout=1,
            raise_on_error=False,
        )

        assert seen_urls == ["http://example/missing"]
        assert result.successful == 2
        assert result.failed == 0
        assert missing.exists()
        assert missing.read_bytes() == b"new"

    def test_zero_byte_file_not_treated_as_cached(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty.nc"
        empty.write_bytes(b"")  # zero bytes — not a valid cached file

        seen_urls: list[str] = []

        def fake_download(urls, paths, **_kwargs):
            seen_urls.extend(urls)
            for p in paths:
                Path(p).write_bytes(b"x")

        monkeypatch.setattr("coastal_calibration.data.downloader.download", fake_download)

        result = _execute_download(
            ["http://example/empty"], [empty], "test", timeout=1, raise_on_error=False
        )

        assert seen_urls == ["http://example/empty"]
        assert result.successful == 1
        assert empty.read_bytes() == b"x"


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
            "harmonic",
            "conus",
        )
        assert len(errors) > 0

    def test_harmonic_skips_coastal_validation(self):
        errors = validate_date_ranges(
            datetime(2021, 6, 11),
            datetime(2021, 6, 12),
            "nwm_retro",
            "harmonic",
            "conus",
        )
        assert len(errors) == 0


class TestBuildStofsMeshUrls:
    """The pre-2023 estofs product needs its connectivity fetched separately."""

    def test_old_product_fetches_maxele_companion(self):
        urls, paths = _build_stofs_mesh_urls(datetime(2022, 9, 28), Path("/tmp/dl"))
        assert len(urls) == 1
        assert urls[0].endswith("estofs.20220928/estofs.t00z.fields.cwl.maxele.nc")
        # Lands beside the main fields file so regrid_estofs finds it by name.
        assert paths[0].name == "estofs.t00z.fields.cwl.maxele.nc"
        assert paths[0].parent == get_stofs_path(datetime(2022, 9, 28), Path("/tmp/dl")).parent

    def test_new_product_needs_nothing_extra(self):
        # stofs_2d_glo carries ``element`` inline.
        urls, paths = _build_stofs_mesh_urls(datetime(2024, 1, 9), Path("/tmp/dl"))
        assert urls == []
        assert paths == []

    def test_boundary_date_is_new_product(self):
        assert _build_stofs_mesh_urls(datetime(2023, 1, 8), Path("/tmp/dl")) == ([], [])
        assert _build_stofs_mesh_urls(datetime(2023, 1, 7), Path("/tmp/dl"))[0]


class TestWriteNwmGridSidecar:
    """PRVI and Alaska Retrospective forcing ships with no georeferencing."""

    def test_existing_sidecar_is_not_refetched(self, tmp_path: Path):
        sidecar = tmp_path / "grid_prvi.json"
        sidecar.write_text('{"domain": "prvi"}')

        # A network read here would blow up the offline test run.
        assert write_nwm_grid_sidecar(tmp_path, "prvi") == sidecar
        assert sidecar.read_text() == '{"domain": "prvi"}'

    def test_records_grid_from_the_store(self, tmp_path: Path, monkeypatch):
        """The sidecar carries the domain's grid verbatim from ldasout.zarr."""
        import xarray as xr

        store = xr.Dataset(
            data_vars={"crs": ((), 0, {"GeoTransform": " -149999.716023 1000.0 0 5 0 -1000.0 "})},
            coords={"x": np.arange(300.0), "y": np.arange(110.0)},
        )
        monkeypatch.setattr("fsspec.get_mapper", lambda *_a, **_k: {})
        monkeypatch.setattr(xr, "open_zarr", lambda *_a, **_k: store)

        sidecar = write_nwm_grid_sidecar(tmp_path / "meteo", "prvi")

        assert sidecar == tmp_path / "meteo" / "grid_prvi.json"
        record = json.loads(sidecar.read_text())
        assert record["domain"] == "prvi"
        assert record["geotransform"] == "-149999.716023 1000.0 0 5 0 -1000.0"
        assert record["shape"] == [300, 110]
        assert record["source"].endswith("PR/zarr/ldasout.zarr")

    def test_unreadable_store_is_non_fatal(self, tmp_path: Path, monkeypatch):
        """The domains that need a sidecar also have a built-in fallback."""

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("no network")

        monkeypatch.setattr("fsspec.get_mapper", _boom)

        assert write_nwm_grid_sidecar(tmp_path, "prvi") is None
        # No sidecar and no .tmp debris left for the next run to trip over.
        assert not list(tmp_path.glob("grid_*"))

    def test_write_failure_leaves_no_partial_sidecar(self, tmp_path: Path, monkeypatch):
        """The canonical path is only ever created by an atomic rename."""
        import xarray as xr

        store = xr.Dataset(
            data_vars={"crs": ((), 0, {"GeoTransform": "0 1 0 0 0 -1"})},
            coords={"x": np.arange(3.0), "y": np.arange(2.0)},
        )
        monkeypatch.setattr("fsspec.get_mapper", lambda *_a, **_k: {})
        monkeypatch.setattr(xr, "open_zarr", lambda *_a, **_k: store)
        monkeypatch.setattr(
            Path, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
        )

        assert write_nwm_grid_sidecar(tmp_path, "prvi") is None
        assert not (tmp_path / "grid_prvi.json").exists()


class TestForcingCacheIsolation:
    """Every NWM domain names its hourly forcing identically."""

    @pytest.mark.parametrize(
        "builder", [_build_nwm_retro_forcing_urls, _build_nwm_ana_forcing_urls]
    )
    def test_domains_do_not_share_a_cached_file(self, tmp_path: Path, builder):
        """Regression: a cached Hawaii file must not be served for a PRVI run.

        ``_execute_download`` treats any existing non-empty file at the target
        path as a cache hit, so two domains landing on one path silently feed
        the wrong forcing into a run.
        """
        start, end = datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2)
        hawaii_urls, hawaii_paths = builder(start, end, tmp_path, "hawaii")
        prvi_urls, prvi_paths = builder(start, end, tmp_path, "prvi")

        assert hawaii_urls != prvi_urls
        assert {p.name for p in hawaii_paths} == {p.name for p in prvi_paths}
        assert not set(hawaii_paths) & set(prvi_paths)

    def test_conus_domains_share_one_copy(self, tmp_path: Path):
        """Atlgulf and pacific pull byte-identical CONUS forcing."""
        start, end = datetime(2021, 6, 11, 0), datetime(2021, 6, 11, 1)
        _, atlgulf = _build_nwm_retro_forcing_urls(start, end, tmp_path, "atlgulf")
        _, pacific = _build_nwm_retro_forcing_urls(start, end, tmp_path, "pacific")

        assert atlgulf == pacific


class TestStofsCycleUrl:
    def test_exact_cycle_no_rounding(self):
        url = _stofs_cycle_url(datetime(2024, 1, 9, 12))
        assert url.endswith("stofs_2d_glo.20240109/stofs_2d_glo.t12z.fields.cwl.nc")

    def test_old_product_naming(self):
        url = _stofs_cycle_url(datetime(2022, 9, 28, 0))
        assert url.endswith("estofs.20220928/estofs.t00z.fields.cwl.nc")


class TestResolveStofsCycle:
    """`exists` is injected so this is fully offline/deterministic."""

    def test_naive_candidate_exists_returns_immediately(self):
        checked = []

        def exists(cycle: datetime) -> bool:
            checked.append(cycle)
            return True

        result = resolve_stofs_cycle(datetime(2025, 9, 15, 12, 30), exists=exists)
        assert result == datetime(2025, 9, 15, 12)
        assert checked == [datetime(2025, 9, 15, 12)]

    def test_walks_backward_until_found(self):
        # Naive candidate (18z) and the next one back (12z) don't exist yet
        # (publish lag) -- 06z does.
        available = {datetime(2026, 8, 5, 6)}

        result = resolve_stofs_cycle(
            datetime(2026, 8, 5, 19), exists=lambda c: c in available
        )
        assert result == datetime(2026, 8, 5, 6)

    def test_crosses_day_boundary(self):
        available = {datetime(2026, 8, 4, 18)}

        result = resolve_stofs_cycle(
            datetime(2026, 8, 5, 2), exists=lambda c: c in available
        )
        assert result == datetime(2026, 8, 4, 18)

    def test_raises_when_nothing_found_within_lookback(self):
        with pytest.raises(ValueError, match="No STOFS cycle found"):
            resolve_stofs_cycle(
                datetime(2026, 8, 5, 12),
                exists=lambda _c: False,
                max_lookback_hours=12,
            )

    def test_only_checks_within_lookback_window(self):
        checked = []

        def exists(cycle: datetime) -> bool:
            checked.append(cycle)
            return False

        with pytest.raises(ValueError, match="No STOFS cycle found"):
            resolve_stofs_cycle(
                datetime(2026, 8, 5, 12), exists=exists, max_lookback_hours=12
            )
        assert checked == [
            datetime(2026, 8, 5, 12),
            datetime(2026, 8, 5, 6),
            datetime(2026, 8, 5, 0),
        ]
        assert min(checked) >= datetime(2026, 8, 5, 12) - timedelta(hours=12)
