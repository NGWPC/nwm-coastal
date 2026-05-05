"""Tests for coastal_calibration._nc_io helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pytest

from coastal_calibration._nc_io import (
    ELEV2D_MISSING,
    create_var,
    write_elev2d_th,
    write_var,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestWriteVar:
    def test_writes_matching_shape(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 4)
            v = create_var(ds, "lon", "f4", ("x",))
            write_var(v, np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(
                ds["lon"][:], np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
            )

    def test_casts_data_to_var_dtype(self, tmp_path: Path):
        # Float64 input written into a float32 variable should arrive
        # on disk as float32 (downcast made explicit by the helper).
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 3)
            v = create_var(ds, "lon", "f4", ("x",))
            write_var(v, np.array([1.5, 2.5, 3.5], dtype=np.float64))

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            assert ds["lon"][:].dtype == np.float32
            np.testing.assert_array_equal(ds["lon"][:], np.array([1.5, 2.5, 3.5], dtype=np.float32))

    def test_raises_on_shape_mismatch(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 4)
            v = create_var(ds, "lon", "f4", ("x",))
            with pytest.raises(ValueError, match=r"Shape mismatch.*'lon'"):
                write_var(v, np.zeros(5, dtype=np.float32))

    def test_writes_per_slab_with_index(self, tmp_path: Path):
        # Loop pattern: write one slab per timestep along the leading dim.
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("y", 2)
            ds.createDimension("x", 4)
            v = create_var(ds, "u", "f4", ("time", "y", "x"))
            for i in range(3):
                write_var(v, np.full((2, 4), float(i), dtype=np.float32), index=i)

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(ds["u"][0], np.zeros((2, 4), dtype=np.float32))
            np.testing.assert_array_equal(ds["u"][1], np.ones((2, 4), dtype=np.float32))
            np.testing.assert_array_equal(ds["u"][2], np.full((2, 4), 2.0, dtype=np.float32))

    def test_index_supports_negative_for_trailing_writes(self, tmp_path: Path):
        # ``var[-1] = chunk`` style — used by sflux to duplicate the
        # last real timestep into the trailing pad slot.
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("n", 4)
            v = create_var(ds, "u", "f4", ("time", "n"))
            write_var(v, np.full((4,), 7.0, dtype=np.float32), index=-1)

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(ds["u"][-1], np.full((4,), 7.0, dtype=np.float32))

    def test_raises_on_per_slab_shape_mismatch(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("y", 2)
            ds.createDimension("x", 4)
            v = create_var(ds, "u", "f4", ("time", "y", "x"))
            with pytest.raises(ValueError, match=r"Shape mismatch writing slab 0.*'u'"):
                write_var(v, np.zeros((2, 5), dtype=np.float32), index=0)

    def test_error_message_names_both_shapes(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 3)
            ds.createDimension("y", 4)
            v = create_var(ds, "u", "f4", ("x", "y"))
            with pytest.raises(ValueError, match=r"Shape mismatch") as exc_info:
                write_var(v, np.zeros((3, 5), dtype=np.float32))
            assert "(3, 5)" in str(exc_info.value)
            assert "(3, 4)" in str(exc_info.value)


class TestCreateVar:
    def test_creates_variable_without_attrs(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 5)
            v = create_var(ds, "foo", "f4", ("x",))
            v[:] = np.arange(5, dtype=np.float32)

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(ds["foo"][:], np.arange(5, dtype=np.float32))
            assert ds["foo"].ncattrs() == []

    def test_applies_attrs_dict(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("y", 3)
            create_var(
                ds,
                "bar",
                "i4",
                ("y",),
                attrs={"long_name": "Bar variable", "units": "m", "axis": "Y"},
            )

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            assert ds["bar"].long_name == "Bar variable"
            assert ds["bar"].units == "m"
            assert ds["bar"].axis == "Y"

    def test_passes_kwargs_to_create_variable(self, tmp_path: Path):
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("t", 4)
            ds.createDimension("n", 6)
            v = create_var(
                ds,
                "data",
                "f8",
                ("t", "n"),
                attrs={"units": "Pa"},
                fill_value=-9999.0,
                zlib=True,
            )
            assert v.filters()["zlib"] is True

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            assert ds["data"]._FillValue == -9999.0
            assert ds["data"].units == "Pa"

    def test_empty_attrs_dict_is_a_no_op(self, tmp_path: Path):
        # Passing attrs={} (or None) should not call setncatts; the
        # variable should end up with no user attributes.
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 2)
            create_var(ds, "a", "i4", ("x",), attrs={})
            create_var(ds, "b", "i4", ("x",), attrs=None)

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            assert ds["a"].ncattrs() == []
            assert ds["b"].ncattrs() == []

    def test_returns_variable_for_caller_to_populate(self, tmp_path: Path):
        # The function never writes data itself; the caller always
        # captures the returned Variable and assigns to it explicitly.
        # That keeps reads/writes visible at the call site.
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("x", 4)
            v = create_var(ds, "lon", "f4", ("x",), attrs={"units": "degrees_east"})
            v[:] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(
                ds["lon"][:], np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
            )
            assert ds["lon"].units == "degrees_east"

    def test_per_timestep_population(self, tmp_path: Path):
        # The same returned-Variable contract supports timestep loops.
        with netCDF4.Dataset(tmp_path / "out.nc", "w", format="NETCDF4") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("n", 2)
            v = create_var(ds, "u", "f4", ("time", "n"))
            for i in range(3):
                v[i, :] = i

        with netCDF4.Dataset(tmp_path / "out.nc") as ds:
            np.testing.assert_array_equal(
                ds["u"][:], np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
            )


class TestWriteElev2dTh:
    def test_writes_canonical_schema(self, tmp_path: Path):
        nt, nb = 3, 4
        time_seconds = np.arange(0, nt * 3600.0, 3600.0)
        series = np.arange(nt * nb, dtype=np.float64).reshape(nt, nb)

        out = tmp_path / "elev2D.th.nc"
        write_elev2d_th(
            out,
            n_open_bnd_nodes=nb,
            time_seconds=time_seconds,
            time_step_seconds=3600.0,
            time_series=series,
        )

        with netCDF4.Dataset(out) as ds:
            # Required dimensions present with correct sizes.
            assert ds.dimensions["time"].size == nt
            assert ds.dimensions["nOpenBndNodes"].size == nb
            assert ds.dimensions["nLevels"].size == 1
            assert ds.dimensions["nComponents"].size == 1
            assert ds.dimensions["one"].size == 1

            # Required variables present with correct shapes.
            assert ds["time_step"].shape == (1,)
            assert ds["time"].shape == (nt,)
            assert ds["time_series"].shape == (nt, nb, 1, 1)
            assert ds["time_series"]._FillValue == ELEV2D_MISSING
            np.testing.assert_array_equal(ds["time"][:], time_seconds)
            np.testing.assert_array_equal(ds["time_step"][:], [3600.0])
            # 2-D input got reshaped to 4-D with the singleton trailing dims.
            np.testing.assert_array_equal(
                ds["time_series"][:, :, 0, 0].data, series.astype(np.float64)
            )

    def test_accepts_4d_series_unchanged(self, tmp_path: Path):
        nt, nb = 2, 3
        series = np.arange(nt * nb, dtype=np.float64).reshape(nt, nb, 1, 1)
        out = tmp_path / "out.nc"
        write_elev2d_th(
            out,
            n_open_bnd_nodes=nb,
            time_seconds=np.arange(0.0, nt * 3600.0, 3600.0),
            time_step_seconds=3600.0,
            time_series=series,
        )
        with netCDF4.Dataset(out) as ds:
            np.testing.assert_array_equal(ds["time_series"][:].data, series)

    def test_zeroes_fill_value_entries(self, tmp_path: Path):
        # Values <= missing should be zeroed before writing — this
        # matches the legacy writer behavior.
        series = np.array([[1.0, -9999.0], [2.0, -100000.0]])
        out = tmp_path / "out.nc"
        write_elev2d_th(
            out,
            n_open_bnd_nodes=2,
            time_seconds=np.array([0.0, 3600.0]),
            time_step_seconds=3600.0,
            time_series=series,
        )
        with netCDF4.Dataset(out) as ds:
            data = ds["time_series"][:, :, 0, 0]
            # Reading back through netCDF4 returns a masked array where
            # the _FillValue cells are masked, so compare against a
            # masked-aware reference.
            expected = np.array([[1.0, 0.0], [2.0, 0.0]])
            np.testing.assert_array_equal(np.asarray(data), expected)

    def test_applies_time_attrs(self, tmp_path: Path):
        out = tmp_path / "out.nc"
        write_elev2d_th(
            out,
            n_open_bnd_nodes=1,
            time_seconds=np.array([0.0]),
            time_step_seconds=60.0,
            time_series=np.zeros((1, 1)),
            time_attrs={
                "units": "seconds since 2024-01-01",
                "long_name": "model time",
                "start_time": 0.0,
            },
        )
        with netCDF4.Dataset(out) as ds:
            assert ds["time"].units == "seconds since 2024-01-01"
            assert ds["time"].long_name == "model time"
            assert ds["time"].start_time == 0.0

    def test_rejects_mismatched_2d_shape(self, tmp_path: Path):
        with pytest.raises(ValueError, match="2-D shape"):
            write_elev2d_th(
                tmp_path / "x.nc",
                n_open_bnd_nodes=4,
                time_seconds=np.arange(3),
                time_step_seconds=1.0,
                time_series=np.zeros((3, 5)),  # nb mismatch (5 vs 4)
            )

    def test_rejects_mismatched_4d_shape(self, tmp_path: Path):
        with pytest.raises(ValueError, match="4-D shape"):
            write_elev2d_th(
                tmp_path / "x.nc",
                n_open_bnd_nodes=4,
                time_seconds=np.arange(3),
                time_step_seconds=1.0,
                time_series=np.zeros((3, 4, 1, 2)),  # nComponents mismatch
            )

    def test_rejects_3d_shape(self, tmp_path: Path):
        with pytest.raises(ValueError, match="must be 2-D or 4-D"):
            write_elev2d_th(
                tmp_path / "x.nc",
                n_open_bnd_nodes=4,
                time_seconds=np.arange(3),
                time_step_seconds=1.0,
                time_series=np.zeros((3, 4, 1)),
            )

    def test_path_argument_can_be_pathlib(self, tmp_path: Path):
        # Helper accepts both str and PathLike.
        out = tmp_path / "via_path.nc"
        write_elev2d_th(
            out,  # PathLike
            n_open_bnd_nodes=1,
            time_seconds=np.array([0.0]),
            time_step_seconds=1.0,
            time_series=np.zeros((1, 1)),
        )
        assert out.is_file()
