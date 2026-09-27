"""Generate a SCHISM ``manning.gr3`` from ESA WorldCover land cover.

Needed when a mesh directory is assembled without one: ``param.nml``'s
``nchi = -1`` makes SCHISM read ``manning.gr3`` at startup and abort if it is
absent or if its counts line disagrees with ``hgrid.gr3``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from coastal_calibration.schism.project_reader import NWMSCHISMProject

__all__ = [
    "ESA_WORLDCOVER_MANNING",
    "generate_manning_from_landcover",
    "load_manning_mapping",
]

# From hydromt_sfincs/data/lulc/esa_worldcover_mapping.csv. Vendored because
# hydromt-sfincs is a pixi-only dependency and its data path is not an API.
ESA_WORLDCOVER_MANNING: dict[int, float] = {
    10: 0.12,
    20: 0.05,
    30: 0.034,
    40: 0.037,
    50: 0.10,
    60: 0.023,
    70: 0.01,
    80: 0.02,
    90: 0.035,
    95: 0.07,
    100: 0.025,
}

DEFAULT_FALLBACK_MANNING = 0.02


def load_manning_mapping(path: Path | None = None) -> dict[int, float]:
    """Return the class-to-Manning mapping, from *path* or the built-in table.

    A CSV must have ``esa_worldcover`` and ``N`` columns; rows with a negative
    ``N`` are no-data markers and are dropped.
    """
    if path is None:
        return dict(ESA_WORLDCOVER_MANNING)

    import pandas as pd

    df = pd.read_csv(path)
    missing = {"esa_worldcover", "N"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    df = df[df["N"] >= 0]
    return {int(c): float(n) for c, n in zip(df["esa_worldcover"], df["N"], strict=True)}


def _sample_classes(
    vrt_path: Path,
    coords: NDArray[np.float64],
    stripe_rows: int,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Sample land-cover classes at *coords*, reading one row stripe at a time."""
    import rasterio
    from rasterio.windows import Window

    n = coords.shape[0]
    classes = np.zeros(n, dtype=np.uint8)

    with rasterio.open(vrt_path) as src:
        rows, cols = rasterio.transform.rowcol(
            src.transform, coords[:, 0], coords[:, 1], op=np.floor
        )
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if not inside.any():
            return classes, inside

        col_lo = int(cols[inside].min())
        col_hi = int(cols[inside].max())
        width = col_hi - col_lo + 1

        idx = np.flatnonzero(inside)
        stripe = rows[idx] // stripe_rows
        order = np.argsort(stripe, kind="stable")
        idx = idx[order]
        stripe = stripe[order]

        starts = np.flatnonzero(np.r_[True, stripe[1:] != stripe[:-1]])
        for begin, end in zip(starts, np.r_[starts[1:], stripe.size], strict=True):
            sel = idx[begin:end]
            row_off = int(stripe[begin]) * stripe_rows
            height = min(stripe_rows, src.height - row_off)
            band = src.read(1, window=Window(col_lo, row_off, width, height))
            classes[sel] = band[rows[sel] - row_off, cols[sel] - col_lo]

    return classes, inside


def _write_manning_gr3(
    project: NWMSCHISMProject,
    output_file: Path,
    values: NDArray[np.float64],
    chunk_size: int,
) -> None:
    """Write ``manning.gr3``, copying node coords and elements from ``hgrid.gr3``."""
    tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    n_nodes = project.n_nodes
    n_elements = project.n_elements

    with (
        project.hgrid_file.open("r", buffering=project.buffer_size) as fin,
        tmp_path.open("w", buffering=project.buffer_size) as fout,
    ):
        fin.readline()
        fin.readline()
        fout.write("manning.gr3 from ESA WorldCover 2020 v100\n")
        fout.write(f"{n_elements} {n_nodes}\n")

        buf: list[str] = []
        for i in range(n_nodes):
            parts = fin.readline().split("!")[0].split()
            buf.append(f"{parts[0]} {float(parts[1]):.6f} {float(parts[2]):.6f} {values[i]:.3f}\n")
            if len(buf) >= chunk_size:
                fout.writelines(buf)
                buf = []
        if buf:
            fout.writelines(buf)

        buf = []
        for _ in range(n_elements):
            buf.append(fin.readline())
            if len(buf) >= chunk_size:
                fout.writelines(buf)
                buf = []
        if buf:
            fout.writelines(buf)

    tmp_path.replace(output_file)


def generate_manning_from_landcover(
    project: NWMSCHISMProject,
    output_file: Path,
    *,
    cache_dir: Path | None = None,
    fallback_manning: float = DEFAULT_FALLBACK_MANNING,
    mapping: dict[int, float] | None = None,
    stripe_rows: int = 512,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Write a node-wise ``manning.gr3`` for *project* from ESA WorldCover.

    Nodes with no land-cover class -- outside the tiles, over an unpublished
    ocean tile, or over in-tile water -- get *fallback_manning*.
    """
    from coastal_calibration.data.esa_worldcover import fetch_esa_worldcover_vrt

    mapping = mapping if mapping is not None else dict(ESA_WORLDCOVER_MANNING)
    coords = project.geographic_coordinates

    lon_span = float(coords[:, 0].max() - coords[:, 0].min())
    if lon_span > 180.0:
        raise ValueError(
            f"Mesh spans {lon_span:.1f} degrees of longitude and appears to cross the "
            "antimeridian, which is not supported."
        )

    bbox = (
        float(coords[:, 0].min()),
        float(coords[:, 1].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].max()),
    )
    cache = cache_dir if cache_dir is not None else output_file.parent / ".esa_worldcover_cache"
    vrt_path, found, missing = fetch_esa_worldcover_vrt(bbox, cache)

    classes, inside = _sample_classes(vrt_path, coords, stripe_rows)

    lut = np.full(256, np.nan, dtype=np.float64)
    for cls, n in mapping.items():
        lut[cls] = n
    values = lut[classes]

    bad = ~inside | np.isnan(values)
    n_fallback = int(bad.sum())
    values[bad] = fallback_manning
    if n_fallback:
        logger.warning(
            "%d of %d nodes (%.2f%%) had no ESA WorldCover class; using n=%.3f",
            n_fallback,
            values.size,
            100.0 * n_fallback / values.size,
            fallback_manning,
        )

    _write_manning_gr3(project, output_file, values, chunk_size)

    uniq, counts = np.unique(values, return_counts=True)
    return {
        "n_nodes": project.n_nodes,
        "n_elements": project.n_elements,
        "n_tiles": len(found),
        "n_tiles_missing": len(missing),
        "n_fallback": n_fallback,
        "value_counts": {float(v): int(c) for v, c in zip(uniq, counts, strict=True)},
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }
