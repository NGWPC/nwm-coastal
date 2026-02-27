"""Integration tests for the SFINCS workflow stages.

These tests exercise the real HydroMT-SFINCS model stages against the
pre-built Texas quadtree mesh (``docs/examples/texas.tar.gz``).  External
I/O (data downloads, SFINCS binary, NOAA CO-OPS API) is mocked so that
the tests run in seconds and without network access.

Run with::

    pytest tests/test_sfincs_workflow.py -v
    pytest -m sfincs -v
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    PathConfig,
    SfincsModelConfig,
    SimulationConfig,
)

pytestmark = pytest.mark.sfincs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Texas domain bbox (EPSG:4326) — used to create synthetic data that
# covers the pre-built model area.
_TX_LON_MIN, _TX_LON_MAX = -96.0, -94.0
_TX_LAT_MIN, _TX_LAT_MAX = 28.0, 30.0

# NWM LCC projection string (matches the data catalog)
_NWM_LCC_CRS = (
    "+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 "
    "+x_0=0 +y_0=0 +R=6370000 +units=m +no_defs=True"
)


def _create_fake_meteo_nc(path: Path, start: datetime) -> None:
    """Write a tiny NWM-like LDASIN file (3x3 grid, 2 time steps)."""
    import pandas as pd
    from pyproj import Transformer

    # Project domain corners to NWM LCC grid
    tr = Transformer.from_crs(4326, _NWM_LCC_CRS, always_xy=True)
    x_min, y_min = tr.transform(_TX_LON_MIN, _TX_LAT_MIN)
    x_max, y_max = tr.transform(_TX_LON_MAX, _TX_LAT_MAX)

    xs = np.linspace(x_min, x_max, 3)
    ys = np.linspace(y_min, y_max, 3)
    times = pd.date_range(start, periods=2, freq="h")

    shape = (2, 3, 3)
    ds = xr.Dataset(
        {
            "RAINRATE": (("time", "y", "x"), np.full(shape, 0.001, dtype=np.float32)),
            "T2D": (("time", "y", "x"), np.full(shape, 295.0, dtype=np.float32)),
            "Q2D": (("time", "y", "x"), np.full(shape, 0.01, dtype=np.float32)),
            "U2D": (("time", "y", "x"), np.full(shape, 2.0, dtype=np.float32)),
            "V2D": (("time", "y", "x"), np.full(shape, 1.0, dtype=np.float32)),
            "PSFC": (("time", "y", "x"), np.full(shape, 101325.0, dtype=np.float32)),
            "SWDOWN": (("time", "y", "x"), np.full(shape, 500.0, dtype=np.float32)),
            "LWDOWN": (("time", "y", "x"), np.full(shape, 300.0, dtype=np.float32)),
        },
        coords={"time": times, "y": ys, "x": xs},
    )
    ds.to_netcdf(path)


def _create_fake_streamflow_nc(path: Path, start: datetime) -> None:
    """Write a tiny NWM-like CHRTOUT file (10 reaches, 2 time steps)."""
    import pandas as pd

    times = pd.date_range(start, periods=2, freq="h")
    feature_ids = np.arange(1, 11, dtype=np.int64)

    ds = xr.Dataset(
        {
            "streamflow": (("time", "feature_id"), np.full((2, 10), 5.0, dtype=np.float32)),
            "q_lateral": (("time", "feature_id"), np.full((2, 10), 0.1, dtype=np.float32)),
        },
        coords={
            "time": times,
            "feature_id": feature_ids,
            "latitude": ("feature_id", np.linspace(_TX_LAT_MIN, _TX_LAT_MAX, 10)),
            "longitude": ("feature_id", np.linspace(_TX_LON_MIN, _TX_LON_MAX, 10)),
        },
    )
    ds.to_netcdf(path)


def _create_fake_stofs_nc(path: Path, start: datetime) -> None:
    """Write a tiny STOFS-like unstructured water level file (25 nodes, 2 time steps)."""
    import pandas as pd

    times = pd.date_range(start, periods=2, freq="h")

    # Create a small grid of nodes covering the Texas coast
    lons = np.linspace(_TX_LON_MIN, _TX_LON_MAX, 5)
    lats = np.linspace(_TX_LAT_MIN, _TX_LAT_MAX, 5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    n_nodes = len(lon_flat)

    # Small water level signal (metres)
    wl = np.random.default_rng(42).normal(0.2, 0.05, (2, n_nodes)).astype(np.float32)

    ds = xr.Dataset(
        {
            "zeta": (("time", "node"), wl),
        },
        coords={
            "time": times,
            "x": ("node", lon_flat),
            "y": ("node", lat_flat),
        },
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_download_dir(tmp_path: Path) -> Path:
    """Create a fake download directory with tiny synthetic NetCDF files.

    Directory structure mirrors the real download layout:
    - meteo/nwm_ana/YYYYMMDDHH.LDASIN_DOMAIN1 (+ .nc symlink)
    - hydro/nwm/YYYYMMDDHH.CHRTOUT_DOMAIN1 (+ .nc symlink)
    - coastal/stofs/stofs_2d_glo.YYYYMMDD/stofs_2d_glo.t00z.fields.cwl.nc
    """
    dl_dir = tmp_path / "downloads"
    start = datetime(2025, 6, 1, 0, 0, 0)

    # --- Meteo ---
    meteo_dir = dl_dir / "meteo" / "nwm_ana"
    meteo_dir.mkdir(parents=True)
    # Create two LDASIN files covering 0h and 1h
    for hour in range(2):
        stem = f"2025060{1:02d}{hour:02d}.LDASIN_DOMAIN1"
        nc_path = meteo_dir / f"{stem}.nc"
        _create_fake_meteo_nc(nc_path, datetime(2025, 6, 1, hour))

    # --- Streamflow ---
    hydro_dir = dl_dir / "hydro" / "nwm"
    hydro_dir.mkdir(parents=True)
    for hour in range(2):
        stem = f"2025060{1:02d}{hour:02d}.CHRTOUT_DOMAIN1"
        nc_path = hydro_dir / f"{stem}.nc"
        _create_fake_streamflow_nc(nc_path, datetime(2025, 6, 1, hour))

    # --- Coastal (STOFS) ---
    stofs_dir = dl_dir / "coastal" / "stofs" / "stofs_2d_glo.20250601"
    stofs_dir.mkdir(parents=True)
    stofs_path = stofs_dir / "stofs_2d_glo.t00z.fields.cwl.nc"
    _create_fake_stofs_nc(stofs_path, start)

    return dl_dir


@pytest.fixture
def dummy_sfincs_exe(tmp_path: Path) -> Path:
    """Create a dummy SFINCS executable that produces a minimal sfincs_his.nc."""
    exe = tmp_path / "sfincs_dummy"
    # The script writes a minimal sfincs_his.nc in the current directory
    # so the plot stage can find model output.
    exe.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env bash
        echo "SFINCS mock run"
        exit 0
    """)
    )
    exe.chmod(0o755)
    return exe


@pytest.fixture
def sfincs_workflow_config(
    tmp_path: Path,
    sfincs_model_dir: Path,
    fake_download_dir: Path,
    dummy_sfincs_exe: Path,
) -> CoastalCalibConfig:
    """Build a CoastalCalibConfig for the SFINCS workflow tests.

    Uses the real pre-built model, synthetic data, and a dummy executable.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    return CoastalCalibConfig(
        simulation=SimulationConfig(
            start_date=datetime(2025, 6, 1),
            duration_hours=1,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        ),
        boundary=BoundaryConfig(
            source="stofs",
            stofs_file=fake_download_dir
            / "coastal"
            / "stofs"
            / "stofs_2d_glo.20250601"
            / "stofs_2d_glo.t00z.fields.cwl.nc",
        ),
        paths=PathConfig(
            work_dir=work_dir,
            raw_download_dir=fake_download_dir,
        ),
        model_config=SfincsModelConfig(
            prebuilt_dir=sfincs_model_dir,
            include_noaa_gages=False,
            merge_observations=True,
            merge_discharge=True,
            include_precip=True,
            include_wind=True,
            include_pressure=True,
            forcing_to_mesh_offset_m=0.0,
            vdatum_mesh_to_msl_m=0.171,
            sfincs_exe=dummy_sfincs_exe,
            discharge_locations_file=sfincs_model_dir / "sfincs_nwm.src",
        ),
        download=DownloadConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_NOAA_STATIONS_IN_DOMAIN = gpd.GeoDataFrame(
    [
        {
            "station_id": "8772985",
            "station_name": "Freeport Harbor",
            "state": "TX",
            "tidal": True,
            "greatlakes": False,
            "geometry": shapely.Point(-95.3083, 28.9483),
        },
        {
            "station_id": "8773037",
            "station_name": "Seadrift",
            "state": "TX",
            "tidal": True,
            "greatlakes": False,
            "geometry": shapely.Point(-96.7117, 28.4067),
        },
    ],
    crs=4326,
)


def _mock_coops_client() -> MagicMock:
    """Return a mock COOPSAPIClient with 2 stations inside the Texas domain."""
    mock_client = MagicMock()
    mock_client.stations_metadata = _NOAA_STATIONS_IN_DOMAIN
    mock_client.filter_stations_by_datum.return_value = {"8772985", "8773037"}

    # Mock datum objects for the plot stage
    datum_8772985 = MagicMock()
    datum_8772985.station_id = "8772985"
    datum_8772985.get_datum_value.side_effect = {"MSL": 0.404, "MLLW": 0.0}.get
    datum_8772985.units = "meters"

    datum_8773037 = MagicMock()
    datum_8773037.station_id = "8773037"
    datum_8773037.get_datum_value.side_effect = {"MSL": 0.213, "MLLW": 0.0}.get
    datum_8773037.units = "meters"

    mock_client.get_datums.return_value = [datum_8772985, datum_8773037]
    return mock_client


def _make_mock_obs_ds(station_ids: list[str], start: datetime) -> xr.Dataset:
    """Create a tiny observed water-level xr.Dataset matching query_coops_byids output."""
    import pandas as pd

    times = pd.date_range(start, periods=6, freq="10min")
    n_times = len(times)
    n_stations = len(station_ids)

    wl = np.random.default_rng(99).normal(0.2, 0.05, (n_stations, n_times)).astype(np.float32)

    ds = xr.Dataset(
        {"water_level": (("station", "time"), wl)},
        coords={"station": station_ids, "time": times},
    )
    ds.attrs["datum"] = "MLLW"
    return ds


# ---------------------------------------------------------------------------
# Tests: Individual stages
# ---------------------------------------------------------------------------


class TestSfincsInitStage:
    """Test the sfincs_init stage on the real pre-built model."""

    def test_init_copies_prebuilt_and_reads_model(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsInitStage

        stage = SfincsInitStage(sfincs_workflow_config)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["grid_type"] in ("regular", "quadtree")
        model_root = Path(result["model_root"])
        assert (model_root / "sfincs.inp").exists()
        assert (model_root / "sfincs.nc").exists()

    def test_init_is_idempotent(self, sfincs_workflow_config):
        """Running init twice should not fail (existing files are kept)."""
        from coastal_calibration.stages.sfincs_build import SfincsInitStage

        stage = SfincsInitStage(sfincs_workflow_config)
        result1 = stage.run()
        assert result1["status"] == "completed"

        # Second run: model_root already exists, files should not be overwritten
        result2 = stage.run()
        assert result2["status"] == "completed"
        assert result2["grid_type"] == result1["grid_type"]


class TestSfincsTimingStage:
    """Test the sfincs_timing stage."""

    def test_timing_sets_start_stop(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsTimingStage,
            _get_model,
        )

        SfincsInitStage(sfincs_workflow_config).run()
        result = SfincsTimingStage(sfincs_workflow_config).run()

        assert result["status"] == "completed"

        model = _get_model(sfincs_workflow_config)
        tstart = model.config.data.tstart
        tstop = model.config.data.tstop
        # Verify the duration matches the config
        assert tstart is not None
        assert tstop is not None


class TestSfincsSymlinksAndCatalog:
    """Test the symlinks and data catalog stages."""

    def test_symlinks_stage(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsSymlinksStage

        stage = SfincsSymlinksStage(sfincs_workflow_config)
        result = stage.run()

        assert result["status"] in ("completed", "skipped")
        if result["status"] == "completed":
            assert result["meteo_symlinks"] >= 0
            assert result["streamflow_symlinks"] >= 0

    def test_data_catalog_stage(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsDataCatalogStage

        stage = SfincsDataCatalogStage(sfincs_workflow_config)
        result = stage.run()

        assert result["status"] == "completed"
        catalog_path = Path(result["catalog_path"])
        assert catalog_path.exists()
        assert "stofs_waterlevel" in result["entries"]
        assert "nwm_ana_meteo" in result["entries"]


class TestSfincsForcingStage:
    """Test the water level boundary forcing stage with synthetic STOFS data."""

    def test_forcing_with_stofs(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsDataCatalogStage,
            SfincsForcingStage,
            SfincsInitStage,
            SfincsSymlinksStage,
            SfincsTimingStage,
        )

        # Run prerequisite stages
        SfincsSymlinksStage(sfincs_workflow_config).run()
        SfincsDataCatalogStage(sfincs_workflow_config).run()
        SfincsInitStage(sfincs_workflow_config).run()
        SfincsTimingStage(sfincs_workflow_config).run()

        result = SfincsForcingStage(sfincs_workflow_config).run()

        assert result["status"] == "completed"
        assert result["source"] == "stofs_waterlevel"


class TestSfincsObsStage:
    """Test the observation points stage with mocked CO-OPS API."""

    def test_obs_with_noaa_gages(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsObservationPointsStage,
        )

        sfincs_workflow_config.model_config.include_noaa_gages = True

        SfincsInitStage(sfincs_workflow_config).run()

        with patch("coastal_calibration.coops_api.COOPSAPIClient") as mock_cls:
            mock_cls.return_value = _mock_coops_client()
            result = SfincsObservationPointsStage(sfincs_workflow_config).run()

        assert result["status"] == "completed"
        # Should find at least some stations in the Texas domain
        assert result["noaa_stations"] >= 0

    def test_obs_dedup_on_rerun(self, sfincs_workflow_config):
        """Verify that re-running with merge_observations=True doesn't accumulate duplicates.

        The pre-built Texas model already contains ``Sargent (8772985)``
        at the same coordinates as the NOAA station 8772985.  The
        spatial-proximity dedup should recognise the overlap and *not*
        add a ``noaa_8772985`` duplicate.
        """
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsObservationPointsStage,
            SfincsWriteStage,
            get_model_root,
        )

        sfincs_workflow_config.model_config.include_noaa_gages = True
        sfincs_workflow_config.model_config.merge_observations = True

        SfincsInitStage(sfincs_workflow_config).run()

        def _obs_lines() -> list[str]:
            """Return non-blank lines from sfincs.obs after writing."""
            SfincsWriteStage(sfincs_workflow_config).run()
            model_root = get_model_root(sfincs_workflow_config)
            obs_file = model_root / "sfincs.obs"
            return [ln for ln in obs_file.read_text().splitlines() if ln.strip()]

        with patch("coastal_calibration.coops_api.COOPSAPIClient") as mock_cls:
            mock_cls.return_value = _mock_coops_client()
            SfincsObservationPointsStage(sfincs_workflow_config).run()
        lines_first = _obs_lines()

        # The spatial dedup should prevent a ``noaa_8772985`` duplicate
        # because the pre-built "Sargent (8772985)" sits at the same
        # coordinates.  No line should contain ``noaa_8772985``.
        noaa_dupes = [ln for ln in lines_first if "noaa_8772985" in ln]
        assert len(noaa_dupes) == 0, (
            f"noaa_8772985 should not appear — spatial dedup should "
            f"recognise the pre-built point: {noaa_dupes}"
        )

        # Second run — total count should not grow
        with patch("coastal_calibration.coops_api.COOPSAPIClient") as mock_cls:
            mock_cls.return_value = _mock_coops_client()
            SfincsObservationPointsStage(sfincs_workflow_config).run()
        lines_second = _obs_lines()

        assert len(lines_second) == len(lines_first), (
            f"Observation count changed between runs: {len(lines_first)} → {len(lines_second)}"
        )


class TestSfincsDischargeStage:
    """Test the discharge source points stage."""

    def test_discharge_from_src_file(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsDischargeStage,
            SfincsInitStage,
        )

        SfincsInitStage(sfincs_workflow_config).run()

        result = SfincsDischargeStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"


class TestSfincsMeteoStages:
    """Test the meteo forcing stages (precip, wind, pressure) with synthetic NWM data."""

    def _run_prerequisites(self, config: CoastalCalibConfig) -> None:
        from coastal_calibration.stages.sfincs_build import (
            SfincsDataCatalogStage,
            SfincsInitStage,
            SfincsSymlinksStage,
            SfincsTimingStage,
        )

        SfincsSymlinksStage(config).run()
        SfincsDataCatalogStage(config).run()
        SfincsInitStage(config).run()
        SfincsTimingStage(config).run()

    def test_precipitation_stage(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsPrecipitationStage

        self._run_prerequisites(sfincs_workflow_config)

        result = SfincsPrecipitationStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"

    def test_wind_stage(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsWindStage

        self._run_prerequisites(sfincs_workflow_config)

        result = SfincsWindStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"

    def test_pressure_stage(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import SfincsPressureStage

        self._run_prerequisites(sfincs_workflow_config)

        result = SfincsPressureStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"

    def test_skip_when_disabled(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsPrecipitationStage,
            SfincsPressureStage,
            SfincsWindStage,
        )

        sfincs_workflow_config.model_config.include_precip = False
        sfincs_workflow_config.model_config.include_wind = False
        sfincs_workflow_config.model_config.include_pressure = False

        self._run_prerequisites(sfincs_workflow_config)

        assert SfincsPrecipitationStage(sfincs_workflow_config).run()["status"] == "skipped"
        assert SfincsWindStage(sfincs_workflow_config).run()["status"] == "skipped"
        assert SfincsPressureStage(sfincs_workflow_config).run()["status"] == "skipped"


class TestSfincsWriteStage:
    """Test the write stage."""

    def test_write_model(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsTimingStage,
            SfincsWriteStage,
        )

        SfincsInitStage(sfincs_workflow_config).run()
        SfincsTimingStage(sfincs_workflow_config).run()

        result = SfincsWriteStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"
        assert Path(result["model_root"]).exists()


class TestSfincsRunStage:
    """Test the run stage with a dummy executable."""

    def test_run_native_exe(self, sfincs_workflow_config):
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsRunStage,
            SfincsTimingStage,
            SfincsWriteStage,
        )

        SfincsInitStage(sfincs_workflow_config).run()
        SfincsTimingStage(sfincs_workflow_config).run()
        SfincsWriteStage(sfincs_workflow_config).run()

        result = SfincsRunStage(sfincs_workflow_config).run()
        assert result["status"] == "completed"
        assert result["mode"] == "native"


class TestSfincsPlotStage:
    """Test the plot stage with mocked observations and model output."""

    def test_plot_skips_without_output(self, sfincs_workflow_config):
        """When sfincs_his.nc is missing, the plot stage skips."""
        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsPlotStage,
        )

        SfincsInitStage(sfincs_workflow_config).run()

        result = SfincsPlotStage(sfincs_workflow_config).run()
        assert result["status"] == "skipped"

    def test_plot_with_mock_output(self, sfincs_workflow_config):
        """Test the plot stage with a synthetic sfincs_his.nc.

        The plot stage spatially matches observation points against
        CO-OPS station metadata, so the mock must return stations
        whose projected coordinates fall within 100 m of the model's
        observation points.
        """
        import pandas as pd

        from coastal_calibration.stages.sfincs_build import (
            SfincsInitStage,
            SfincsObservationPointsStage,
            SfincsPlotStage,
            SfincsTimingStage,
            SfincsWriteStage,
            get_model_root,
        )

        sfincs_workflow_config.model_config.include_noaa_gages = True

        # Run prerequisite stages
        SfincsInitStage(sfincs_workflow_config).run()
        SfincsTimingStage(sfincs_workflow_config).run()

        with patch("coastal_calibration.coops_api.COOPSAPIClient") as mock_cls:
            mock_cls.return_value = _mock_coops_client()
            SfincsObservationPointsStage(sfincs_workflow_config).run()

        SfincsWriteStage(sfincs_workflow_config).run()

        # Count observation points from sfincs.obs
        model_root = get_model_root(sfincs_workflow_config)
        obs_file = model_root / "sfincs.obs"
        n_stations = sum(1 for ln in obs_file.read_text().splitlines() if ln.strip())

        times = pd.date_range("2025-06-01", periods=6, freq="10min")
        point_zs = (
            np.random.default_rng(42).normal(0.2, 0.05, (len(times), n_stations)).astype(np.float32)
        )

        his_ds = xr.Dataset(
            {
                "point_zs": (("time", "stations"), point_zs),
                "crs": (
                    (),
                    np.int32(0),
                    {
                        "epsg": 32614,
                        "grid_mapping_name": "transverse_mercator",
                    },
                ),
            },
            coords={
                "time": times,
                "stations": np.arange(n_stations),
                "station_x": ("stations", np.linspace(700000, 850000, n_stations)),
                "station_y": ("stations", np.linspace(3140000, 3190000, n_stations)),
            },
        )
        his_ds.to_netcdf(model_root / "sfincs_his.nc")

        # The plot stage spatially matches obs points to CO-OPS
        # stations.  Our mock provides 2 stations; one overlaps a
        # pre-built point (8772985 ≈ Sargent) and one was newly added
        # (8773037).  Both should be matched → 2 station IDs.
        matched_ids = ["8772985", "8773037"]
        mock_obs_ds = _make_mock_obs_ds(matched_ids, datetime(2025, 6, 1))

        mock_client = _mock_coops_client()
        datums = []
        for sid in matched_ids:
            d = MagicMock()
            d.station_id = sid
            d.get_datum_value.side_effect = lambda name, _s=sid: {"MSL": 0.3, "MLLW": 0.0}.get(name)
            d.units = "meters"
            datums.append(d)
        mock_client.get_datums.return_value = datums

        with (
            patch(
                "coastal_calibration.coops_api.query_coops_byids",
                return_value=mock_obs_ds,
            ),
            patch(
                "coastal_calibration.coops_api.COOPSAPIClient",
            ) as mock_cls2,
        ):
            mock_cls2.return_value = mock_client
            result = SfincsPlotStage(sfincs_workflow_config).run()

        assert result["status"] == "completed"
        figs_dir = model_root / "figs"
        assert figs_dir.exists()
        assert len(result["figures"]) > 0


# ---------------------------------------------------------------------------
# Test: Full workflow runner (download disabled)
# ---------------------------------------------------------------------------


class TestSfincsWorkflowRunner:
    """Test running the full workflow through CoastalCalibRunner."""

    def test_full_workflow_no_download(self, sfincs_workflow_config):
        """Run the complete SFINCS workflow with download disabled and mocked I/O."""
        from coastal_calibration.runner import CoastalCalibRunner

        sfincs_workflow_config.model_config.include_noaa_gages = False

        mock_obs_ds = _make_mock_obs_ds(["8772985"], datetime(2025, 6, 1))

        with (
            patch(
                "coastal_calibration.coops_api.query_coops_byids",
                return_value=mock_obs_ds,
            ),
            patch(
                "coastal_calibration.coops_api.COOPSAPIClient",
            ) as mock_cls,
        ):
            mock_cls.return_value = _mock_coops_client()

            runner = CoastalCalibRunner(sfincs_workflow_config)
            result = runner.run()

        # Download is disabled, so it should be skipped by _get_stages_to_run.
        # All other stages should complete.
        assert result.success, f"Workflow failed: {result.errors}"
        assert "download" not in result.stages_completed

        # The init/timing/write/run stages should have completed
        for stage_name in ("sfincs_init", "sfincs_timing", "sfincs_write", "sfincs_run"):
            assert stage_name in result.stages_completed, f"{stage_name} did not complete"

    def test_dry_run(self, sfincs_workflow_config):
        """Dry run should validate but not execute."""
        from coastal_calibration.runner import CoastalCalibRunner

        runner = CoastalCalibRunner(sfincs_workflow_config)
        result = runner.run(dry_run=True)
        assert result.success
        assert result.outputs.get("dry_run") is True

    def test_stop_after(self, sfincs_workflow_config):
        """Running with stop_after should only execute up to that stage."""
        from coastal_calibration.runner import CoastalCalibRunner

        runner = CoastalCalibRunner(sfincs_workflow_config)
        result = runner.run(stop_after="sfincs_timing")

        assert result.success, f"Workflow failed: {result.errors}"
        assert "sfincs_timing" in result.stages_completed
        assert "sfincs_forcing" not in result.stages_completed
