"""Tests for coastal_calibration.data.nwm_forcing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
import xarray as xr

from coastal_calibration.data.nwm_forcing import normalize_wrf_forcing

if TYPE_CHECKING:
    from pathlib import Path

# Mirrors the NWM Hawaii grid: 1 km cells, origin at the north-west corner.
# The GeoTransform declares a north-up raster (negative y step) even though
# the rows are stored south to north.
GEOTRANSFORM = "-295000 1000 0 194999 0 -1000 "
N_X, N_Y = 6, 4


def _wrf_dataset(
    *, geotransform: str | None = GEOTRANSFORM, n_x: int = N_X, n_y: int = N_Y
) -> xr.Dataset:
    """Build a raw WRF-layout LDASIN dataset like non-CONUS Retrospective.

    ``geotransform=None`` drops the grid-mapping variable altogether, which
    is how the PRVI and Alaska files actually arrive.
    """
    # Row 0 is the southernmost row; make each row identifiable.
    field = np.arange(n_y, dtype="float32")[None, :, None] * np.ones((1, n_y, n_x), "float32")
    data: dict[str, Any] = {
        "RAINRATE": (("Time", "south_north", "west_east"), field),
        "valid_time": (
            ("Time",),
            np.array([1285632000.0]),
            {"units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"},
        ),
        "Times": (("Time", "DateStrLen"), np.array([list("2010-09-28_00:00:00")])),
    }
    if geotransform is not None:
        data["lambert_conformal_conic"] = xr.DataArray(0, attrs={"GeoTransform": geotransform})
    return xr.Dataset(data)


def _write_sidecar(directory: Path, domain: str, geotransform: str, shape: list[int]) -> None:
    """Write the grid sidecar the downloader records next to the forcing."""
    (directory / f"grid_{domain}.json").write_text(
        json.dumps({"domain": domain, "geotransform": geotransform, "shape": shape})
    )


def _round_trip(ds: xr.Dataset, path: Path) -> xr.Dataset:
    """Write and reopen so ``encoding["source"]`` is populated as in production."""
    ds.to_netcdf(path)
    return xr.open_dataset(path)


class TestGridSidecar:
    """The downloader records each domain's grid from NOAA's ldasout.zarr."""

    # Deliberately not a shape in the built-in fallback table, so a passing
    # test can only mean the sidecar was read.
    SHAPE: ClassVar[list[int]] = [7, 5]
    GEOTRANSFORM = "1000 100 0 6000 0 -100"

    def test_sidecar_supplies_missing_geotransform(self, tmp_path: Path):
        _write_sidecar(tmp_path, "prvi", self.GEOTRANSFORM, self.SHAPE)
        raw = _wrf_dataset(geotransform=None, n_x=self.SHAPE[0], n_y=self.SHAPE[1])

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc"))

        np.testing.assert_allclose(ds.x.values[0], 1050.0)
        np.testing.assert_allclose(ds.y.values[0], 5550.0)
        assert ds.y.values[0] < ds.y.values[-1]

    def test_in_file_geotransform_wins(self, tmp_path: Path):
        """A file that carries its own grid mapping ignores the sidecar."""
        _write_sidecar(tmp_path, "prvi", self.GEOTRANSFORM, [N_X, N_Y])
        raw = _wrf_dataset()

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2010092800.LDASIN_DOMAIN1.nc"))

        np.testing.assert_allclose(ds.x.values[0], -294500.0)

    def test_sidecar_for_another_grid_is_ignored(self, tmp_path: Path):
        """Domains sharing a download directory each get their own sidecar."""
        _write_sidecar(tmp_path, "hawaii", self.GEOTRANSFORM, [590, 390])
        raw = _wrf_dataset(geotransform=None, n_x=300, n_y=110)

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc"))

        # Falls through to the built-in PRVI entry, not Hawaii's sidecar.
        np.testing.assert_allclose(ds.x.values[0], -149499.716023)

    def test_numerically_equal_sidecars_do_not_conflict(self, tmp_path: Path):
        """NOAA writes ``1000`` for one domain and ``1000.0`` for another."""
        _write_sidecar(tmp_path, "prvi", "1000 100 0 6000 0 -100", self.SHAPE)
        _write_sidecar(tmp_path, "twin", "1000.0 100.0 0.0 6000.0 0.0 -100.0", self.SHAPE)
        raw = _wrf_dataset(geotransform=None, n_x=self.SHAPE[0], n_y=self.SHAPE[1])

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc"))

        np.testing.assert_allclose(ds.x.values[0], 1050.0)

    @pytest.mark.parametrize(
        "geotransform",
        ["1000 100 0 6000 0", "1000 100 0 6000 0 -100 7", "nan 100 0 6000 0 -100", "junk"],
        ids=["too-short", "too-long", "nonfinite", "non-numeric"],
    )
    def test_malformed_sidecar_is_skipped(self, tmp_path: Path, geotransform: str):
        """A sidecar that cannot yield six finite numbers is not usable."""
        _write_sidecar(tmp_path, "prvi", geotransform, [300, 110])
        raw = _wrf_dataset(geotransform=None, n_x=300, n_y=110)

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc"))

        # Falls through to the built-in entry rather than making NaN coords.
        np.testing.assert_allclose(ds.x.values[0], -149499.716023)

    def test_malformed_in_file_geotransform_raises(self, tmp_path: Path):
        """A file's own broken grid mapping is an error, not a fallback."""
        raw = _wrf_dataset(geotransform="-295000 1000 0")
        opened = _round_trip(raw, tmp_path / "2010092800.LDASIN_DOMAIN1.nc")

        with pytest.raises(ValueError, match="six finite numbers"):
            normalize_wrf_forcing(opened)

    def test_conflicting_sidecars_raise(self, tmp_path: Path):
        """Two domains claiming one shape cannot be resolved by shape alone."""
        _write_sidecar(tmp_path, "prvi", self.GEOTRANSFORM, self.SHAPE)
        _write_sidecar(tmp_path, "elsewhere", "0 100 0 0 0 -100", self.SHAPE)
        raw = _wrf_dataset(geotransform=None, n_x=self.SHAPE[0], n_y=self.SHAPE[1])
        opened = _round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc")

        with pytest.raises(ValueError, match="disagree on its GeoTransform"):
            normalize_wrf_forcing(opened)

    def test_unreadable_sidecar_is_skipped(self, tmp_path: Path):
        (tmp_path / "grid_prvi.json").write_text("{ not json")
        raw = _wrf_dataset(geotransform=None, n_x=300, n_y=110)

        ds = normalize_wrf_forcing(_round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc"))

        np.testing.assert_allclose(ds.x.values[0], -149499.716023)

    def test_builtin_fallback_warns(self, tmp_path: Path, caplog):
        """Running off the built-in table must be visible, not silent."""
        raw = _wrf_dataset(geotransform=None, n_x=300, n_y=110)
        opened = _round_trip(raw, tmp_path / "2017092000.LDASIN_DOMAIN1.nc")

        with caplog.at_level("WARNING"):
            normalize_wrf_forcing(opened)

        assert "No grid sidecar" in caplog.text


class TestNormalizeWrfForcing:
    def test_renames_dims_and_builds_coordinates(self):
        ds = normalize_wrf_forcing(_wrf_dataset())

        assert set(ds.sizes) == {"time", "y", "x"}
        assert ds.sizes["x"] == N_X
        assert ds.sizes["y"] == N_Y
        # Cell centers, i.e. half a cell in from the grid edges.
        np.testing.assert_allclose(ds.x.values[0], -294500.0)
        np.testing.assert_allclose(ds.x.values[-1], -294500.0 + (N_X - 1) * 1000)

    def test_y_ascends_despite_north_up_geotransform(self):
        """Regression: deriving y from the y_res sign flips the field.

        The rows are stored south to north, so y must ascend. Trusting the
        GeoTransform's negative step instead mirrors every field about the
        middle of the domain, which silently misplaces the forcing.
        """
        ds = normalize_wrf_forcing(_wrf_dataset())

        assert ds.y.values[0] < ds.y.values[-1]
        # Row 0 of the source array must stay at the southern edge.
        southern_row = ds["RAINRATE"].isel(time=0).sel(y=ds.y.min()).values
        np.testing.assert_array_equal(southern_row, np.zeros(N_X, "float32"))

    def test_filename_stamp_wins_over_valid_time(self):
        """Regression: PRVI pins ``valid_time`` to 00Z in every hourly file.

        Every file then claims the same timestamp and ``open_mfdataset``
        cannot order them, so the ``YYYYMMDDHH`` filename stamp is the only
        reliable source.
        """
        ds = _wrf_dataset()
        ds.encoding["source"] = "/data/nwm_retro/2010092803.LDASIN_DOMAIN1.nc"

        assert normalize_wrf_forcing(ds).time.values[0] == np.datetime64("2010-09-28T03:00:00")

    def test_promotes_valid_time_and_drops_helper_variables(self):
        ds = normalize_wrf_forcing(_wrf_dataset())

        # No source path, so ``valid_time`` is the fallback.
        assert ds.time.values[0] == np.datetime64("2010-09-28T00:00:00")
        # ``Times`` carries a DateStrLen dimension hydromt cannot handle.
        assert "Times" not in ds.variables
        assert "DateStrLen" not in ds.sizes
        assert "lambert_conformal_conic" not in ds.variables
        assert "RAINRATE" in ds.data_vars

    def test_cf_layout_passes_through_unchanged(self):
        """CONUS Retrospective and every Analysis file already use x/y/time."""
        cf = xr.Dataset(
            {"RAINRATE": (("time", "y", "x"), np.zeros((1, N_Y, N_X), "float32"))},
            coords={"x": np.arange(N_X, dtype=float), "y": np.arange(N_Y, dtype=float)},
        )
        assert normalize_wrf_forcing(cf) is cf

    @pytest.mark.parametrize(
        ("n_x", "n_y", "x0", "y0"),
        [
            (300, 110, -149499.716023, -54498.7304575),
            (879, 459, -1130389.442087718, -3163014.413754986),
        ],
        ids=["prvi", "alaska"],
    )
    def test_known_domain_shape_falls_back_to_builtin_geotransform(self, n_x, n_y, x0, y0):
        """PRVI and Alaska Retrospective files carry no grid-mapping variable."""
        ds = normalize_wrf_forcing(_wrf_dataset(geotransform=None, n_x=n_x, n_y=n_y))

        # Cell centers, half a cell in from the grid edges.
        np.testing.assert_allclose(ds.x.values[0], x0)
        np.testing.assert_allclose(ds.y.values[0], y0)
        np.testing.assert_allclose(ds.x.values[-1], x0 + (n_x - 1) * 1000)
        np.testing.assert_allclose(ds.y.values[-1], y0 + (n_y - 1) * 1000)

    def test_missing_geotransform_on_unknown_grid_raises(self):
        """Without a GeoTransform or a known grid shape, coordinates are lost."""
        with pytest.raises(ValueError, match="GeoTransform"):
            normalize_wrf_forcing(_wrf_dataset(geotransform=None))

    def test_open_mfdataset_orders_files_by_filename_stamp(self, tmp_path):
        """End-to-end guard: hourly PRVI files all share the same ``valid_time``.

        This is the path that actually matters, since it also pins down that
        ``open_mfdataset`` populates ``encoding["source"]`` before calling the
        preprocessor.
        """
        paths = []
        for hour in (2, 0, 1):  # out of order on purpose
            ds = _wrf_dataset(geotransform=None, n_x=300, n_y=110)
            ds["RAINRATE"][:] = hour
            path = tmp_path / f"201709200{hour}.LDASIN_DOMAIN1.nc"
            ds.to_netcdf(path)
            paths.append(path)

        merged = xr.open_mfdataset(paths, preprocess=normalize_wrf_forcing, combine="by_coords")

        np.testing.assert_array_equal(
            merged.time.values,
            np.array(
                ["2017-09-20T00", "2017-09-20T01", "2017-09-20T02"],
                "datetime64[ns]",
            ),
        )
        # Each file's payload must travel with its own timestamp.
        np.testing.assert_array_equal(merged["RAINRATE"].max(("y", "x")).values, [0, 1, 2])
