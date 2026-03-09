"""Raster clipping and reprojection utilities.

Provides a memory-safe clip-before-reproject workflow for xarray
raster data.  The core problem this solves: when reprojecting between
dissimilar projections (e.g. Lambert Conformal Conic → UTM),
``rasterio.warp.calculate_default_transform`` can inflate the output
grid to cover the entire source extent, even if only a small region
is needed.  By clipping in the **source** CRS first and constraining
the destination grid, memory stays proportional to the area of interest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def clip_and_reproject(
    data: xr.Dataset | xr.DataArray,
    dst_bounds: tuple[float, float, float, float],
    dst_crs: object,
    dst_res: float,
    *,
    fill_value: float = 0.0,
    buffer: float = 10_000.0,
    reproject_method: str = "nearest_index",
) -> xr.Dataset | xr.DataArray:
    """Clip a raster in its source CRS, then reproject to a constrained grid.

    Steps:

    1. Transform *dst_bounds* from *dst_crs* into the data's native CRS.
    2. Clip the data in its native CRS with a generous buffer.
    3. Reproject to *dst_crs* with an output grid exactly covering
       *dst_bounds* ± *buffer* at *dst_res* resolution.

    This avoids the grid-inflation problem where
    ``rasterio.warp.calculate_default_transform`` produces an output
    grid spanning the entire source extent (e.g. CONUS-scale for NWM
    LCC → UTM reprojections).

    Parameters
    ----------
    data
        Input :class:`xarray.Dataset` or :class:`xarray.DataArray`
        with a CRS set (via the ``raster`` accessor from hydromt or
        rioxarray).
    dst_bounds
        ``(xmin, ymin, xmax, ymax)`` target bounding box in *dst_crs*
        units (e.g. metres for a UTM zone).
    dst_crs
        Target CRS (anything accepted by ``rasterio``/``pyproj``).
    dst_res
        Target resolution in *dst_crs* units.
    fill_value
        Value for pixels with no source data after reprojection.
    buffer
        Extra margin in *dst_crs* units added around *dst_bounds*
        for the output grid.  Default 10 km.
    reproject_method
        Resampling method passed to ``raster.reproject()``.
        Default ``"nearest_index"`` (KDTree-based, good for meteo).

    Returns
    -------
    Same type as *data*
        Clipped and reprojected raster.  Spatial dimensions are
        **not** renamed (caller handles SFINCS ``x``/``y`` convention).
    """
    import rasterio.warp
    from affine import Affine

    # ------------------------------------------------------------------
    # 1. Clip in source CRS
    # ------------------------------------------------------------------
    src_crs = data.raster.crs
    src_bounds = rasterio.warp.transform_bounds(dst_crs, src_crs, *dst_bounds)

    # Buffer in source-CRS units: at least ``buffer`` (which is in
    # dst_crs units, used as a rough lower bound), or 30 source cells.
    src_res = abs(float(data.raster.res[0]))
    src_buf = max(buffer, src_res * 30)
    src_clip = (
        src_bounds[0] - src_buf,
        src_bounds[1] - src_buf,
        src_bounds[2] + src_buf,
        src_bounds[3] + src_buf,
    )
    data = data.raster.clip_bbox(src_clip)

    # ------------------------------------------------------------------
    # 2. Reproject with a constrained output grid
    # ------------------------------------------------------------------
    xmin = dst_bounds[0] - buffer
    ymin = dst_bounds[1] - buffer
    xmax = dst_bounds[2] + buffer
    ymax = dst_bounds[3] + buffer

    dst_width = int(np.ceil((xmax - xmin) / dst_res))
    dst_height = int(np.ceil((ymax - ymin) / dst_res))
    dst_transform = Affine(dst_res, 0.0, xmin, 0.0, -dst_res, ymax)

    return (
        data.raster.reproject(
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
            method=reproject_method,
        )
        .fillna(fill_value)
    )
