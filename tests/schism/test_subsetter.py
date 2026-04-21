"""Tests for SCHISM mesh subsetter boundary splitting.

Covers three boundary topologies:
- shore: cut crosses open boundary (1 point) + land boundary (1 point)
- ocean: cut crosses a single open boundary at 2 points
- island: cut crosses open boundary (2 points) + island land boundary (2 points)

No SCHISM binaries are needed; tests exercise the splitting algorithm only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import shapely

from coastal_calibration.schism.project_reader import NWMSCHISMProject
from coastal_calibration.schism.subsetter import (
    MeshClassifier,
    MeshSubsetter,
    _build_cut_boundaries,
    _extract_side_segments,
)
from tests.schism.schism_testkit import generate_test_case

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID_SIZE = (9, 7)
RESOLUTION = (1.0, 1.0)


def _make_project(tmp_path: Path, boundary_type: Literal["shore", "ocean", "island"]) -> Path:
    d = tmp_path / boundary_type
    generate_test_case(
        grid_size=GRID_SIZE,
        resolution=RESOLUTION,
        boundary_type=boundary_type,
        base_dir=d,
        station_output=False,
    )
    # Create stub NWM files required by MeshSubsetter._write_nodes
    import shutil

    shutil.copy2(d / "hgrid.gr3", d / "hgrid.cpp")
    shutil.copy2(d / "manning.gr3", d / "windrot_geo2proj.gr3")
    shutil.copy2(d / "manning.gr3", d / "elev.ic")
    # element_areas.txt: one area value per element
    n_elements = int((d / "hgrid.gr3").read_text().splitlines()[1].split()[0])
    (d / "element_areas.txt").write_text("\n".join(["1.0"] * n_elements) + "\n")
    # nwmReaches.csv: empty but present
    (d / "nwmReaches.csv").write_text("")
    return d


@pytest.fixture
def shore_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, "shore")


@pytest.fixture
def ocean_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, "ocean")


@pytest.fixture
def island_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, "island")


@pytest.fixture
def cut_line() -> shapely.LineString:
    """Vertical cut at x = 4.5, spanning the full y range."""
    return shapely.LineString([(4.5, -1.0), (4.5, 7.0)])


# ---------------------------------------------------------------------------
# Unit tests: _extract_side_segments
# ---------------------------------------------------------------------------


class TestExtractSideSegments:
    def test_all_on_side(self):
        result = _extract_side_segments([1, 2, 3, 4, 5], {1, 2, 3, 4, 5})
        assert result == [[1, 2, 3, 4, 5]]

    def test_none_on_side(self):
        result = _extract_side_segments([1, 2, 3], {10, 20})
        assert result == []

    def test_split_middle(self):
        result = _extract_side_segments([1, 2, 3, 4, 5], {1, 2, 4, 5})
        assert result == [[1, 2], [4, 5]]

    def test_single_node_segment(self):
        result = _extract_side_segments([1, 2, 3, 4, 5], {3})
        assert result == [[3]]

    def test_island_wraparound(self):
        result = _extract_side_segments([1, 2, 3, 4, 5, 6], {1, 2, 5, 6}, is_closed=True)
        assert result == [[5, 6, 1, 2]]

    def test_island_no_wrap(self):
        result = _extract_side_segments([1, 2, 3, 4, 5, 6], {3, 4}, is_closed=True)
        assert result == [[3, 4]]

    def test_island_all_on_side(self):
        result = _extract_side_segments([1, 2, 3, 4], {1, 2, 3, 4}, is_closed=True)
        assert result == [[1, 2, 3, 4]]

    def test_empty_boundary(self):
        result = _extract_side_segments([], {1, 2, 3})
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests: _build_cut_boundaries
# ---------------------------------------------------------------------------


class TestBuildCutBoundaries:
    def test_single_path(self):
        adjacency = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        mapping = {i: i for i in range(1, 5)}
        result = _build_cut_boundaries([1, 4], adjacency, mapping)
        assert len(result) == 1
        assert result[0] == [1, 2, 3, 4]

    def test_no_terminals(self):
        adjacency = {1: {2}, 2: {1}}
        mapping = {1: 1, 2: 2}
        assert _build_cut_boundaries([], adjacency, mapping) == []

    def test_single_terminal(self):
        adjacency = {1: {2}, 2: {1}}
        mapping = {1: 1, 2: 2}
        assert _build_cut_boundaries([1], adjacency, mapping) == []

    def test_two_disjoint_paths(self):
        adjacency = {
            1: {2},
            2: {1, 3},
            3: {2},
            10: {11},
            11: {10, 12},
            12: {11},
        }
        mapping = {i: i for i in [1, 2, 3, 10, 11, 12]}
        result = _build_cut_boundaries([1, 3, 10, 12], adjacency, mapping)
        assert len(result) == 2

    def test_remapping(self):
        adjacency = {10: {20}, 20: {10, 30}, 30: {20}}
        mapping = {10: 1, 20: 2, 30: 3}
        result = _build_cut_boundaries([10, 30], adjacency, mapping)
        assert len(result) == 1
        assert result[0] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _divide_and_check(
    project_dir: Path,
    cut_line: shapely.LineString,
    tmp_path: Path,
) -> dict:
    """Run the full divide pipeline and return stats for assertions."""
    project = NWMSCHISMProject(project_dir, validate=False)

    classifier = MeshClassifier(project, cut_line)
    classification, elements_a, elements_b = classifier.classify_nodes()

    output_a = tmp_path / "side_a"
    output_b = tmp_path / "side_b"

    subsetter = MeshSubsetter(
        project,
        classification.side_a,
        classification.side_b,
    )
    subset_result = subsetter.subset_mesh(
        output_a,
        output_b,
        classification.shared,
        elements_a,
        elements_b,
        write_boundaries=True,
        write_bctides=True,
        write_hgrid_ll=False,
        write_netcdf=False,
    )

    return {
        "classification": classification,
        "subset": subset_result,
        "output_a": output_a,
        "output_b": output_b,
        "n_nodes_original": project.n_nodes,
        "n_elements_original": project.n_elements,
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDivideShore:
    """Shore case: cut crosses open boundary + land boundary (1 point each)."""

    def test_node_conservation(self, shore_project, cut_line, tmp_path):
        r = _divide_and_check(shore_project, cut_line, tmp_path)
        c = r["classification"]
        assert c.n_side_a + c.n_side_b - c.n_shared == r["n_nodes_original"]

    def test_element_conservation(self, shore_project, cut_line, tmp_path):
        r = _divide_and_check(shore_project, cut_line, tmp_path)
        s = r["subset"]
        assert s.side_a.n_elements + s.side_b.n_elements == r["n_elements_original"]

    def test_boundaries_exist(self, shore_project, cut_line, tmp_path):
        r = _divide_and_check(shore_project, cut_line, tmp_path)
        for side_dir in (r["output_a"], r["output_b"]):
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            assert bs.n_open >= 1, f"No open boundaries in {side_dir}"

    def test_boundary_node_ids_valid(self, shore_project, cut_line, tmp_path):
        r = _divide_and_check(shore_project, cut_line, tmp_path)
        for side_dir, side_data in [
            (r["output_a"], r["subset"].side_a),
            (r["output_b"], r["subset"].side_b),
        ]:
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            n = side_data.n_nodes
            for bnd in bs.open_boundaries:
                assert all(1 <= nid <= n for nid in bnd), f"Invalid open bnd IDs in {side_dir}"
            for lb in bs.land_boundaries:
                assert all(1 <= nid <= n for nid in lb.nodes), f"Invalid land bnd IDs in {side_dir}"

    def test_has_land_boundaries(self, shore_project, cut_line, tmp_path):
        r = _divide_and_check(shore_project, cut_line, tmp_path)
        proj_a = NWMSCHISMProject(r["output_a"], validate=False)
        proj_b = NWMSCHISMProject(r["output_b"], validate=False)
        total_land = proj_a.read_boundaries().n_land + proj_b.read_boundaries().n_land
        assert total_land >= 1


class TestDivideOcean:
    """Ocean case: cut crosses a single open boundary at 2 points."""

    def test_node_conservation(self, ocean_project, cut_line, tmp_path):
        r = _divide_and_check(ocean_project, cut_line, tmp_path)
        c = r["classification"]
        assert c.n_side_a + c.n_side_b - c.n_shared == r["n_nodes_original"]

    def test_element_conservation(self, ocean_project, cut_line, tmp_path):
        r = _divide_and_check(ocean_project, cut_line, tmp_path)
        s = r["subset"]
        assert s.side_a.n_elements + s.side_b.n_elements == r["n_elements_original"]

    def test_boundaries_exist(self, ocean_project, cut_line, tmp_path):
        r = _divide_and_check(ocean_project, cut_line, tmp_path)
        for side_dir in (r["output_a"], r["output_b"]):
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            assert bs.n_open >= 1

    def test_island_preserved(self, ocean_project, cut_line, tmp_path):
        r = _divide_and_check(ocean_project, cut_line, tmp_path)
        proj_a = NWMSCHISMProject(r["output_a"], validate=False)
        proj_b = NWMSCHISMProject(r["output_b"], validate=False)
        total_land = proj_a.read_boundaries().n_land + proj_b.read_boundaries().n_land
        assert total_land >= 1, "Island land boundary lost during split"

    def test_boundary_node_ids_valid(self, ocean_project, cut_line, tmp_path):
        r = _divide_and_check(ocean_project, cut_line, tmp_path)
        for side_dir, side_data in [
            (r["output_a"], r["subset"].side_a),
            (r["output_b"], r["subset"].side_b),
        ]:
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            n = side_data.n_nodes
            for bnd in bs.open_boundaries:
                assert all(1 <= nid <= n for nid in bnd)
            for lb in bs.land_boundaries:
                assert all(1 <= nid <= n for nid in lb.nodes)


class TestDivideIsland:
    """Island case: cut crosses open bnd (2 pts) + island land bnd (2 pts)."""

    def test_node_conservation(self, island_project, cut_line, tmp_path):
        r = _divide_and_check(island_project, cut_line, tmp_path)
        c = r["classification"]
        assert c.n_side_a + c.n_side_b - c.n_shared == r["n_nodes_original"]

    def test_element_conservation(self, island_project, cut_line, tmp_path):
        r = _divide_and_check(island_project, cut_line, tmp_path)
        s = r["subset"]
        assert s.side_a.n_elements + s.side_b.n_elements == r["n_elements_original"]

    def test_boundaries_exist(self, island_project, cut_line, tmp_path):
        r = _divide_and_check(island_project, cut_line, tmp_path)
        for side_dir in (r["output_a"], r["output_b"]):
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            assert bs.n_open >= 1

    def test_island_split(self, island_project, cut_line, tmp_path):
        r = _divide_and_check(island_project, cut_line, tmp_path)
        proj_a = NWMSCHISMProject(r["output_a"], validate=False)
        proj_b = NWMSCHISMProject(r["output_b"], validate=False)
        assert proj_a.read_boundaries().n_land >= 1
        assert proj_b.read_boundaries().n_land >= 1

    def test_boundary_node_ids_valid(self, island_project, cut_line, tmp_path):
        r = _divide_and_check(island_project, cut_line, tmp_path)
        for side_dir, side_data in [
            (r["output_a"], r["subset"].side_a),
            (r["output_b"], r["subset"].side_b),
        ]:
            proj = NWMSCHISMProject(side_dir, validate=False)
            bs = proj.read_boundaries()
            n = side_data.n_nodes
            for bnd in bs.open_boundaries:
                assert all(1 <= nid <= n for nid in bnd)
            for lb in bs.land_boundaries:
                assert all(1 <= nid <= n for nid in lb.nodes)
