"""Tests for :func:`coastal_calibration.utils.raster.clip_and_reproject`.

Run with::

    pixi run --environment sfincs pytest tests/test_raster_utils.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from hydromt.gis import raster as _  # noqa: F401  — registers accessor

from coastal_calibration.utils.raster import clip_and_reproject


def _make_raster(
    *,
    bounds: tuple[float, float, float, float] = (-97.0, 28.0, -94.0, 31.0),
    res: float = 0.01,
    crs: str = "EPSG:4326",
    fill: float = 1.0,
) -> xr.DataArray:
    """Create a synthetic raster DataArray with the hydromt raster accessor."""
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin + res / 2, xmax, res)
    ys = np.arange(ymax - res / 2, ymin, -res)
    data = np.full((len(ys), len(xs)), fill, dtype=np.float32)
    da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
    da.raster.set_crs(crs)
    return da


class TestClipAndReproject:
    """Tests for clip_and_reproject."""

    def test_output_bounded_to_dst_bounds(self) -> None:
        """Output grid stays within dst_bounds ± buffer."""
        src = _make_raster(crs="EPSG:4326")
        dst_crs = "EPSG:32615"  # UTM 15N
        dst_bounds = (200_000.0, 3_200_000.0, 300_000.0, 3_300_000.0)
        buffer = 5_000.0
        dst_res = 1_000.0

        result = clip_and_reproject(
            src,
            dst_bounds,
            dst_crs,
            dst_res,
            buffer=buffer,
        )
        out_bounds = result.raster.bounds
        xmin, ymin, xmax, ymax = out_bounds

        assert xmin >= dst_bounds[0] - buffer - dst_res
        assert ymin >= dst_bounds[1] - buffer - dst_res
        assert xmax <= dst_bounds[2] + buffer + dst_res
        assert ymax <= dst_bounds[3] + buffer + dst_res

    def test_output_shape_sensible(self) -> None:
        """Width and height are consistent with dst_bounds and dst_res."""
        src = _make_raster(crs="EPSG:4326")
        dst_bounds = (200_000.0, 3_200_000.0, 300_000.0, 3_300_000.0)
        dst_res = 1_000.0
        buffer = 5_000.0

        result = clip_and_reproject(
            src,
            dst_bounds,
            "EPSG:32615",
            dst_res,
            buffer=buffer,
        )
        expected_w = int(np.ceil((100_000 + 2 * buffer) / dst_res))
        expected_h = int(np.ceil((100_000 + 2 * buffer) / dst_res))
        __, y_dim, x_dim = result.dims if len(result.dims) == 3 else (None, *result.dims)
        assert result.sizes[x_dim] == expected_w
        assert result.sizes[y_dim] == expected_h

    def test_geographic_source_crs(self) -> None:
        """Source in EPSG:4326 (degrees) works without buffer/unit issues."""
        src = _make_raster(
            bounds=(-97.0, 28.0, -94.0, 31.0),
            res=0.01,
            crs="EPSG:4326",
        )
        dst_bounds = (200_000.0, 3_200_000.0, 250_000.0, 3_250_000.0)
        result = clip_and_reproject(
            src,
            dst_bounds,
            "EPSG:32615",
            1_000.0,
            buffer=5_000.0,
        )
        assert result.sizes[result.raster.dims[0]] > 0
        assert result.sizes[result.raster.dims[1]] > 0

    def test_projected_source_crs(self) -> None:
        """Source in a projected CRS (metres) also works correctly."""
        src = _make_raster(
            bounds=(150_000.0, 3_100_000.0, 350_000.0, 3_400_000.0),
            res=1_000.0,
            crs="EPSG:32615",
        )
        dst_bounds = (200_000.0, 3_200_000.0, 300_000.0, 3_300_000.0)
        result = clip_and_reproject(
            src,
            dst_bounds,
            "EPSG:32615",
            1_000.0,
            buffer=5_000.0,
        )
        assert result.sizes[result.raster.dims[0]] > 0
        assert result.sizes[result.raster.dims[1]] > 0

    def test_invalid_dst_res_raises(self) -> None:
        """dst_res <= 0 raises ValueError."""
        src = _make_raster()
        with pytest.raises(ValueError, match="dst_res must be positive"):
            clip_and_reproject(
                src,
                (0, 0, 1, 1),
                "EPSG:32615",
                dst_res=0,
            )
        with pytest.raises(ValueError, match="dst_res must be positive"):
            clip_and_reproject(
                src,
                (0, 0, 1, 1),
                "EPSG:32615",
                dst_res=-100,
            )

    def test_inverted_bounds_raises(self) -> None:
        """Inverted dst_bounds (xmin > xmax or ymin > ymax) raises ValueError."""
        src = _make_raster()
        with pytest.raises(ValueError, match="Invalid dst_bounds"):
            clip_and_reproject(
                src,
                (300_000, 3_200_000, 200_000, 3_300_000),
                "EPSG:32615",
                1_000.0,
            )
        with pytest.raises(ValueError, match="Invalid dst_bounds"):
            clip_and_reproject(
                src,
                (200_000, 3_300_000, 300_000, 3_200_000),
                "EPSG:32615",
                1_000.0,
            )

    def test_dataset_input(self) -> None:
        """Works with xr.Dataset, not just DataArray."""
        da = _make_raster(crs="EPSG:4326")
        ds = da.to_dataset(name="var")
        dst_bounds = (200_000.0, 3_200_000.0, 250_000.0, 3_250_000.0)
        result = clip_and_reproject(
            ds,
            dst_bounds,
            "EPSG:32615",
            1_000.0,
            buffer=5_000.0,
        )
        assert isinstance(result, xr.Dataset)
        assert "var" in result.data_vars
