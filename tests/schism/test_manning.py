"""Tests for generating manning.gr3 from ESA WorldCover."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from coastal_calibration.schism.manning import (
    ESA_WORLDCOVER_MANNING,
    generate_manning_from_landcover,
    load_manning_mapping,
)
from coastal_calibration.schism.project_reader import NWMSCHISMProject
from tests.schism.schism_testkit import generate_test_case

GRID = (9, 9)
RES = (0.01, 0.01)


@pytest.fixture
def mesh_dir(tmp_path: Path) -> Path:
    """Build a small SCHISM mesh and remove its manning.gr3."""
    base = tmp_path / "mesh"
    generate_test_case(
        grid_size=GRID,
        resolution=RES,
        boundary_type="ocean",
        base_dir=base,
        station_output=False,
    )
    (base / "manning.gr3").unlink()
    return base


def _write_raster(path: Path, classes: np.ndarray, *, west=-0.01, north=0.09, pixel=0.005) -> Path:
    """Write a single-band uint8 GeoTIFF of land-cover classes in EPSG:4326."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(west, north, pixel, pixel),
    ) as dst:
        dst.write(classes, 1)
    return path


@pytest.fixture
def patch_fetch(monkeypatch):
    """Replace the tile download with a locally written raster."""

    def _apply(raster: Path, missing=()):
        def _fake(bbox, cache_dir, *, buffer_deg=0.05, log=None):
            return raster, [(0, 0)], list(missing)

        monkeypatch.setattr(
            "coastal_calibration.data.esa_worldcover.fetch_esa_worldcover_vrt", _fake
        )

    return _apply


def _split_gr3(path: Path, n_nodes: int, n_elements: int):
    lines = path.read_text().splitlines()
    return lines[0], lines[1], lines[2 : 2 + n_nodes], lines[2 + n_nodes : 2 + n_nodes + n_elements]


class TestManningStructure:
    def test_matches_hgrid_and_has_no_boundary_block(self, mesh_dir, patch_fetch, tmp_path):
        classes = np.full((20, 20), 10, dtype=np.uint8)
        classes[:10, :] = 80
        patch_fetch(_write_raster(tmp_path / "lc.tif", classes))

        project = NWMSCHISMProject(mesh_dir, validate=False)
        out = mesh_dir / "manning.gr3"
        stats = generate_manning_from_landcover(project, out)

        n_nodes, n_elements = project.n_nodes, project.n_elements
        lines = out.read_text().splitlines()
        assert len(lines) == 2 + n_nodes + n_elements
        assert lines[1].split() == [str(n_elements), str(n_nodes)]

        _, _, _, man_elems = _split_gr3(out, n_nodes, n_elements)
        _, _, _, hgrid_elems = _split_gr3(mesh_dir / "hgrid.gr3", n_nodes, n_elements)
        assert man_elems == hgrid_elems

        assert stats["n_nodes"] == n_nodes
        assert set(stats["value_counts"]) == {0.02, 0.12}

    def test_values_follow_land_cover(self, mesh_dir, patch_fetch, tmp_path):
        classes = np.full((20, 20), 30, dtype=np.uint8)
        patch_fetch(_write_raster(tmp_path / "lc.tif", classes))

        project = NWMSCHISMProject(mesh_dir, validate=False)
        out = mesh_dir / "manning.gr3"
        generate_manning_from_landcover(project, out)

        _, _, nodes, _ = _split_gr3(out, project.n_nodes, project.n_elements)
        values = {float(line.split()[3]) for line in nodes}
        assert values == {ESA_WORLDCOVER_MANNING[30]}

    def test_project_validates_afterwards(self, mesh_dir, patch_fetch, tmp_path):
        patch_fetch(_write_raster(tmp_path / "lc.tif", np.full((20, 20), 10, dtype=np.uint8)))
        generate_manning_from_landcover(
            NWMSCHISMProject(mesh_dir, validate=False), mesh_dir / "manning.gr3"
        )
        NWMSCHISMProject(mesh_dir, validate=True)


class TestSampling:
    def test_stripe_size_does_not_change_output(self, mesh_dir, patch_fetch, tmp_path):
        rng = np.random.default_rng(0)
        classes = rng.choice([10, 30, 50, 80], size=(20, 20)).astype(np.uint8)
        patch_fetch(_write_raster(tmp_path / "lc.tif", classes))

        project = NWMSCHISMProject(mesh_dir, validate=False)
        one = mesh_dir / "one.gr3"
        many = mesh_dir / "many.gr3"
        generate_manning_from_landcover(project, one, stripe_rows=1)
        generate_manning_from_landcover(project, many, stripe_rows=512)
        assert one.read_text() == many.read_text()

    def test_nodata_uses_fallback(self, mesh_dir, patch_fetch, tmp_path, caplog):
        patch_fetch(_write_raster(tmp_path / "lc.tif", np.zeros((20, 20), dtype=np.uint8)))

        project = NWMSCHISMProject(mesh_dir, validate=False)
        out = mesh_dir / "manning.gr3"
        with caplog.at_level(logging.WARNING):
            stats = generate_manning_from_landcover(project, out, fallback_manning=0.031)

        assert stats["n_fallback"] == project.n_nodes
        assert stats["value_counts"] == {0.031: project.n_nodes}
        assert "no ESA WorldCover class" in caplog.text

    def test_nodes_outside_raster_use_fallback(self, mesh_dir, patch_fetch, tmp_path):
        raster = _write_raster(
            tmp_path / "lc.tif",
            np.full((20, 20), 10, dtype=np.uint8),
            west=50.0,
            north=50.0,
        )
        patch_fetch(raster)

        project = NWMSCHISMProject(mesh_dir, validate=False)
        stats = generate_manning_from_landcover(project, mesh_dir / "manning.gr3")
        assert stats["n_fallback"] == project.n_nodes

    def test_antimeridian_rejected(self, mesh_dir, patch_fetch, tmp_path, monkeypatch):
        patch_fetch(_write_raster(tmp_path / "lc.tif", np.full((20, 20), 10, dtype=np.uint8)))
        project = NWMSCHISMProject(mesh_dir, validate=False)

        coords = project.geographic_coordinates.copy()
        coords[: len(coords) // 2, 0] -= 179.0
        coords[len(coords) // 2 :, 0] += 179.0
        monkeypatch.setattr(type(project), "geographic_coordinates", property(lambda self: coords))

        with pytest.raises(ValueError, match="antimeridian"):
            generate_manning_from_landcover(project, mesh_dir / "manning.gr3")


class TestMapping:
    def test_custom_mapping_csv(self, tmp_path):
        csv = tmp_path / "map.csv"
        csv.write_text("esa_worldcover,description,landuse,N\n10,Tree,10,0.5\n0,No data,0,-999\n")
        assert load_manning_mapping(csv) == {10: 0.5}

    def test_missing_column_is_reported(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("esa_worldcover,description\n10,Tree\n")
        with pytest.raises(ValueError, match="missing column"):
            load_manning_mapping(csv)

    def test_matches_hydromt_sfincs_table(self):
        pytest.importorskip("hydromt_sfincs")
        import hydromt_sfincs

        csv = Path(hydromt_sfincs.__file__).parent / "data/lulc/esa_worldcover_mapping.csv"
        if not csv.exists():
            pytest.skip(f"{csv} not shipped with this hydromt-sfincs build")
        assert load_manning_mapping(csv) == ESA_WORLDCOVER_MANNING
