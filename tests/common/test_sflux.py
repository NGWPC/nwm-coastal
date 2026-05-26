"""Tests for coastal_calibration.schism.sflux."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pytest

from coastal_calibration.schism.sflux import (
    _compute_subset_indices,
    _pressure_to_msl,
    _round_down,
    make_atmo_sflux,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestRoundDown:
    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [
            (3.7, 0, 3.0),
            (-3.7, 0, -4.0),
            (1.234567, 3, 1.234),
            (1.0, 5, 1.0),
            (0.9999999, 6, 0.999999),
        ],
    )
    def test_basic(self, value, decimals, expected):
        assert _round_down(value, decimals) == pytest.approx(expected)


class TestSlp:
    def test_zero_height_returns_input_pressure(self):
        # At height = 0 the exponent is exp(0) = 1, so SLP equals surface pressure.
        temp = np.full((2, 2), 288.15, dtype=np.float64)
        mixing = np.full((2, 2), 0.005, dtype=np.float64)
        height = np.zeros((2, 2), dtype=np.float64)
        press = np.full((2, 2), 101325.0, dtype=np.float64)
        out = _pressure_to_msl(temp, mixing, height, press)
        np.testing.assert_allclose(out, press)

    def test_positive_height_increases_pressure(self):
        # Reducing pressure to sea level from a positive elevation should
        # produce a value larger than the surface pressure.
        temp = np.array([[288.15]])
        mixing = np.array([[0.005]])
        height = np.array([[1000.0]])  # 1 km elevation
        press = np.array([[90000.0]])
        out = _pressure_to_msl(temp, mixing, height, press)
        assert out.shape == press.shape
        assert (out > press).all()

    def test_dry_air_matches_hypsometric(self):
        # With zero specific humidity Tv == T and the formula reduces to
        # the dry hypsometric equation: SLP = p0 / exp(-z / (Rd*T/g0)).
        g0, Rd = 9.80665, 287.058  # noqa: N806
        temp = np.array([[300.0]])
        height = np.array([[500.0]])
        press = np.array([[95000.0]])
        out = _pressure_to_msl(temp, np.zeros_like(temp), height, press)
        expected = press / np.exp(-height / (Rd * temp / g0))
        np.testing.assert_allclose(out, expected)


@pytest.fixture
def geogrid_file(tmp_path: Path) -> Path:
    """Synthetic WRF geogrid: 3 lat by 4 lon, single time slice."""
    path = tmp_path / "geo_em_TEST.nc"
    ny, nx = 3, 4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createVariable("HGT_M", "f4", ("Time", "south_north", "west_east"))[:] = np.linspace(
            0, 200, ny * nx, dtype=np.float32
        ).reshape(1, ny, nx)
        ds.createVariable("XLAT_M", "f4", ("Time", "south_north", "west_east"))[:] = (
            np.broadcast_to(np.linspace(20.0, 22.0, ny, dtype=np.float32)[:, None], (1, ny, nx))
        )
        ds.createVariable("XLONG_M", "f4", ("Time", "south_north", "west_east"))[:] = (
            np.broadcast_to(np.linspace(-156.0, -153.0, nx, dtype=np.float32)[None, :], (1, ny, nx))
        )
    return path


def _write_ldasin(path: Path, t2d: float, q2d: float, u2d: float, v2d: float, psfc: float) -> None:
    """Write a single-timestep LDASIN file with constant fields."""
    ny, nx = 3, 4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        for name, val in [
            ("T2D", t2d),
            ("Q2D", q2d),
            ("U2D", u2d),
            ("V2D", v2d),
            ("PSFC", psfc),
        ]:
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = np.full(
                (1, ny, nx), val, dtype=np.float32
            )


class TestMakeAtmoSflux:
    def test_raises_when_no_ldasin_files(self, tmp_path: Path, geogrid_file: Path):
        forcing_dir = tmp_path / "forcing"
        forcing_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No LDASIN_DOMAIN1 files"):
            make_atmo_sflux(
                forcing_input_dir=forcing_dir,
                work_dir=tmp_path / "work",
                start_dt=datetime(2024, 1, 1, 0),
                geogrid_file=geogrid_file,
            )

    def test_writes_expected_output(self, tmp_path: Path, geogrid_file: Path):
        forcing_dir = tmp_path / "forcing"
        forcing_dir.mkdir()
        # Two hourly LDASIN files with distinct values per timestep so we
        # can verify they land on the right slots and the trailing
        # duplicate copy is wired to the second one.
        _write_ldasin(
            forcing_dir / "2024010100.LDASIN_DOMAIN1",
            t2d=295.0,
            q2d=0.005,
            u2d=2.0,
            v2d=-1.0,
            psfc=100000.0,
        )
        _write_ldasin(
            forcing_dir / "2024010101.LDASIN_DOMAIN1",
            t2d=296.0,
            q2d=0.006,
            u2d=3.0,
            v2d=-2.0,
            psfc=99500.0,
        )
        work_dir = tmp_path / "work"

        make_atmo_sflux(
            forcing_input_dir=forcing_dir,
            work_dir=work_dir,
            start_dt=datetime(2024, 1, 1, 0),
            geogrid_file=geogrid_file,
        )

        out_path = work_dir / "sflux" / "sflux_air_1.0001.nc"
        assert out_path.is_file()

        with netCDF4.Dataset(out_path) as ds:
            # Dimensions: time = N + 1 (trailing duplicate), grid matches geogrid.
            assert ds.dimensions["time"].size == 3
            assert ds.dimensions["ny_grid"].size == 3
            assert ds.dimensions["nx_grid"].size == 4

            # Coordinate metadata is set.
            assert ds["time"].units == "days since 2024-01-01"
            assert list(ds["time"].base_date) == [2024, 1, 1, 0]
            assert ds["lon"].units == "degrees_east"
            assert ds["lat"].units == "degrees_north"
            assert ds["uwind"].units == "m/s"
            assert ds["prmsl"].units == "Pa"
            assert ds["spfh"].units == "kg/kg"

            # Time axis: hourly cadence in fractions of a day, starting at 0.
            np.testing.assert_allclose(
                ds["time"][:], np.array([0.0, 1 / 24, 2 / 24], dtype=np.float64), atol=1e-6
            )

            # Each variable's first slice matches the first file's value,
            # second slice the second file's value, and the trailing
            # duplicate equals the second slice.
            np.testing.assert_allclose(ds["uwind"][0], np.full((3, 4), 2.0, dtype=np.float32))
            np.testing.assert_allclose(ds["uwind"][1], np.full((3, 4), 3.0, dtype=np.float32))
            np.testing.assert_allclose(ds["uwind"][-1], ds["uwind"][-2])
            np.testing.assert_allclose(ds["vwind"][1], np.full((3, 4), -2.0, dtype=np.float32))
            np.testing.assert_allclose(ds["stmp"][0], np.full((3, 4), 295.0, dtype=np.float32))
            np.testing.assert_allclose(ds["spfh"][1], np.full((3, 4), 0.006, dtype=np.float32))

            # PSFC is reduced to MSL via the _pressure_to_msl formula. The geogrid's
            # first cell has height = 0 so its prmsl equals psfc; cells
            # at positive elevations have prmsl > psfc.
            prmsl0 = ds["prmsl"][0]
            assert (prmsl0 >= 100000.0).all()
            assert (prmsl0 > 100000.0).any()

    def test_start_hour_offsets_time_axis(self, tmp_path: Path, geogrid_file: Path):
        # When start_dt has a non-zero hour the time axis shifts by
        # start_hour/24 days.
        forcing_dir = tmp_path / "forcing"
        forcing_dir.mkdir()
        _write_ldasin(
            forcing_dir / "2024010112.LDASIN_DOMAIN1",
            t2d=290.0,
            q2d=0.001,
            u2d=1.0,
            v2d=0.0,
            psfc=101000.0,
        )

        make_atmo_sflux(
            forcing_input_dir=forcing_dir,
            work_dir=tmp_path / "work",
            start_dt=datetime(2024, 1, 1, 12),
            geogrid_file=geogrid_file,
        )
        with netCDF4.Dataset(tmp_path / "work" / "sflux" / "sflux_air_1.0001.nc") as ds:
            np.testing.assert_allclose(
                ds["time"][:], np.array([12 / 24, 13 / 24], dtype=np.float64), atol=1e-6
            )


class TestComputeSubsetIndices:
    @staticmethod
    def _grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
        # Geogrid spanning lon ∈ [-130, -60], lat ∈ [20, 50] (CONUS-ish).
        lat1d = np.linspace(20.0, 50.0, ny, dtype=np.float64)
        lon1d = np.linspace(-130.0, -60.0, nx, dtype=np.float64)
        lons, lats = np.meshgrid(lon1d, lat1d)
        return lats, lons

    def test_returns_tight_window_around_bbox(self):
        lats, lons = self._grid(31, 71)  # 1-deg resolution
        # Mesh covers lon ∈ [-125, -123], lat ∈ [38, 40] (Mendocino-ish).
        j0, j1, i0, i1 = _compute_subset_indices(
            lats, lons, (-125.0, 38.0, -123.0, 40.0), buffer_deg=0.5
        )
        # Buffered bbox: lon ∈ [-125.5, -122.5], lat ∈ [37.5, 40.5].
        # 1-deg grid puts ~3 lat rows and ~3 lon cols in scope.
        assert 0 <= j0 < j1 <= 31
        assert 0 <= i0 < i1 <= 71
        assert (j1 - j0) < 31
        assert (i1 - i0) < 71
        sub_lats, sub_lons = lats[j0:j1, i0:i1], lons[j0:j1, i0:i1]
        assert sub_lats.min() <= 38.0
        assert sub_lats.max() >= 40.0
        assert sub_lons.min() <= -125.0
        assert sub_lons.max() >= -123.0

    def test_buffer_is_applied(self):
        lats, lons = self._grid(31, 71)
        j0_b0, j1_b0, _, _ = _compute_subset_indices(
            lats, lons, (-125.0, 38.0, -123.0, 40.0), buffer_deg=0.0
        )
        j0_b2, j1_b2, _, _ = _compute_subset_indices(
            lats, lons, (-125.0, 38.0, -123.0, 40.0), buffer_deg=2.0
        )
        # A larger buffer widens the window in lat space.
        assert (j1_b2 - j0_b2) > (j1_b0 - j0_b0)

    def test_raises_when_bbox_outside_geogrid(self):
        lats, lons = self._grid(31, 71)
        # Far west of the geogrid even after buffering.
        with pytest.raises(ValueError, match="no overlap with the geogrid"):
            _compute_subset_indices(lats, lons, (-180.0, 38.0, -170.0, 40.0), buffer_deg=0.5)


class TestMakeAtmoSfluxBbox:
    def test_mesh_bbox_shrinks_output_dimensions(self, tmp_path: Path):
        # 21x41 geogrid spanning lon ∈ [-130, -110], lat ∈ [30, 40].
        geogrid_path = tmp_path / "geo_em_LARGE.nc"
        ny, nx = 21, 41
        lat1d = np.linspace(30.0, 40.0, ny, dtype=np.float32)
        lon1d = np.linspace(-130.0, -110.0, nx, dtype=np.float32)
        with netCDF4.Dataset(geogrid_path, "w") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("south_north", ny)
            ds.createDimension("west_east", nx)
            ds.createVariable("HGT_M", "f4", ("Time", "south_north", "west_east"))[:] = np.zeros(
                (1, ny, nx), dtype=np.float32
            )
            ds.createVariable("XLAT_M", "f4", ("Time", "south_north", "west_east"))[:] = (
                np.broadcast_to(lat1d[:, None], (1, ny, nx))
            )
            ds.createVariable("XLONG_M", "f4", ("Time", "south_north", "west_east"))[:] = (
                np.broadcast_to(lon1d[None, :], (1, ny, nx))
            )

        forcing_dir = tmp_path / "forcing"
        forcing_dir.mkdir()
        with netCDF4.Dataset(forcing_dir / "2024010100.LDASIN_DOMAIN1", "w") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("south_north", ny)
            ds.createDimension("west_east", nx)
            for name, val in [
                ("T2D", 295.0),
                ("Q2D", 0.005),
                ("U2D", 1.0),
                ("V2D", 0.5),
                ("PSFC", 100000.0),
            ]:
                ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = np.full(
                    (1, ny, nx), val, dtype=np.float32
                )

        # Mesh covers a 2x2-deg subdomain near (lat=35, lon=-120).
        make_atmo_sflux(
            forcing_input_dir=forcing_dir,
            work_dir=tmp_path / "work",
            start_dt=datetime(2024, 1, 1, 0),
            geogrid_file=geogrid_path,
            mesh_bbox=(-121.0, 34.0, -119.0, 36.0),
            bbox_buffer_deg=0.5,
        )

        out_path = tmp_path / "work" / "sflux" / "sflux_air_1.0001.nc"
        with netCDF4.Dataset(out_path) as ds:
            assert ds.dimensions["ny_grid"].size < ny
            assert ds.dimensions["nx_grid"].size < nx
            # The retained window must still cover the (buffered) mesh extent.
            assert ds["lat"][:].min() <= 34.0
            assert ds["lat"][:].max() >= 36.0
            assert ds["lon"][:].min() <= -121.0
            assert ds["lon"][:].max() >= -119.0
            # Variables are written on the subset grid.
            assert ds["stmp"].shape == (
                2,
                ds.dimensions["ny_grid"].size,
                ds.dimensions["nx_grid"].size,
            )

    def test_raises_when_bbox_outside_geogrid(self, tmp_path: Path, geogrid_file: Path):
        forcing_dir = tmp_path / "forcing"
        forcing_dir.mkdir()
        _write_ldasin(
            forcing_dir / "2024010100.LDASIN_DOMAIN1",
            t2d=295.0,
            q2d=0.005,
            u2d=1.0,
            v2d=0.0,
            psfc=100000.0,
        )
        # Geogrid fixture spans lon ∈ [-156, -153], lat ∈ [20, 22].
        with pytest.raises(ValueError, match="no overlap with the geogrid"):
            make_atmo_sflux(
                forcing_input_dir=forcing_dir,
                work_dir=tmp_path / "work",
                start_dt=datetime(2024, 1, 1, 0),
                geogrid_file=geogrid_file,
                mesh_bbox=(-100.0, 40.0, -98.0, 42.0),
            )
