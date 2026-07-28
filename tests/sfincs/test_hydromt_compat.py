"""Tests for coastal_calibration.sfincs._hydromt_compat."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from coastal_calibration.sfincs._hydromt_compat import normalize_wrf_forcing

# Mirrors the NWM Hawaii grid: 1 km cells, origin at the north-west corner.
# The GeoTransform declares a north-up raster (negative y step) even though
# the rows are stored south to north.
GEOTRANSFORM = "-295000 1000 0 194999 0 -1000 "
N_X, N_Y = 6, 4


def _wrf_dataset(*, geotransform: str | None = GEOTRANSFORM) -> xr.Dataset:
    """Build a raw WRF-layout LDASIN dataset like non-CONUS Retrospective."""
    # Row 0 is the southernmost row; make each row identifiable.
    field = np.arange(N_Y, dtype="float32")[None, :, None] * np.ones((1, N_Y, N_X), "float32")
    grid_mapping = xr.DataArray(
        0, attrs={} if geotransform is None else {"GeoTransform": geotransform}
    )
    return xr.Dataset(
        {
            "RAINRATE": (("Time", "south_north", "west_east"), field),
            "valid_time": (
                ("Time",),
                np.array([1285632000.0]),
                {"units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"},
            ),
            "Times": (("Time", "DateStrLen"), np.array([list("2010-09-28_00:00:00")])),
            "lambert_conformal_conic": grid_mapping,
        }
    )


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

    def test_promotes_valid_time_and_drops_helper_variables(self):
        ds = normalize_wrf_forcing(_wrf_dataset())

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

    def test_missing_geotransform_raises(self):
        """Without a GeoTransform the coordinates cannot be rebuilt."""
        with pytest.raises(ValueError, match="GeoTransform"):
            normalize_wrf_forcing(_wrf_dataset(geotransform=None))
