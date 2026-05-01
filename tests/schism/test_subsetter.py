"""Tests for SCHISM mesh subsetter boundary splitting.

Covers three boundary topologies:
- shore: cut crosses open boundary (1 point) + land boundary (1 point)
- ocean: cut crosses a single open boundary at 2 points
- island: cut crosses open boundary (2 points) + island land boundary (2 points)

No SCHISM binaries are needed; tests exercise the splitting algorithm only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import shapely

from coastal_calibration.schism.project_reader import NWMSCHISMProject
from coastal_calibration.schism.stages import _chain_ring
from coastal_calibration.schism.subsetter import (
    MeshClassifier,
    MeshSubsetter,
    _build_cut_boundaries,
    _build_shared_nodes_graph,
    _extract_side_segments,
    extract_mesh,
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


# ---------------------------------------------------------------------------
# Unit tests: _build_shared_nodes_graph
# ---------------------------------------------------------------------------


class TestBuildSharedNodesGraph:
    """Adjacency must follow real element edges, never quad diagonals."""

    def test_empty_elements(self):
        elements = np.empty((0, 5), dtype=np.int64)
        assert _build_shared_nodes_graph([], elements) == {}

    def test_no_shared_nodes_in_elements(self):
        elements = np.array([[1, 1, 2, 3, 0]], dtype=np.int64)
        assert _build_shared_nodes_graph([10, 20], elements) == {}

    def test_triangle_two_shared(self):
        elements = np.array([[1, 1, 2, 3, 0]], dtype=np.int64)
        adj = _build_shared_nodes_graph([1, 2], elements)
        assert adj == {1: {2}, 2: {1}}

    def test_triangle_all_three_shared(self):
        # All triangle vertices form actual edges, so all pairs are adjacent
        elements = np.array([[1, 1, 2, 3, 0]], dtype=np.int64)
        adj = _build_shared_nodes_graph([1, 2, 3], elements)
        assert adj == {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}

    def test_quad_three_shared_excludes_diagonal(self):
        # Quad 1-2-3-4 (CCW). Edges: 1-2, 2-3, 3-4, 4-1. Diagonals: 1-3, 2-4.
        # If 1, 2, 3 are shared, the diagonal 1-3 must NOT be added — that
        # was the old bug.
        elements = np.array([[1, 1, 2, 3, 4]], dtype=np.int64)
        adj = _build_shared_nodes_graph([1, 2, 3], elements)
        assert adj == {1: {2}, 2: {1, 3}, 3: {2}}
        assert 3 not in adj.get(1, set()), "diagonal 1-3 must not be in adjacency"
        assert 1 not in adj.get(3, set()), "diagonal 1-3 must not be in adjacency"

    def test_quad_diagonal_pair_only_no_edge(self):
        # Only the diagonal pair 1, 3 shared: not edge-adjacent on the quad
        elements = np.array([[1, 1, 2, 3, 4]], dtype=np.int64)
        assert _build_shared_nodes_graph([1, 3], elements) == {}
        # Other diagonal: 2, 4
        assert _build_shared_nodes_graph([2, 4], elements) == {}

    def test_quad_all_four_shared(self):
        # All 4 edges, but NOT the 2 diagonals
        elements = np.array([[1, 1, 2, 3, 4]], dtype=np.int64)
        adj = _build_shared_nodes_graph([1, 2, 3, 4], elements)
        assert adj == {1: {2, 4}, 2: {1, 3}, 3: {2, 4}, 4: {1, 3}}

    def test_mixed_tri_and_quad(self):
        elements = np.array(
            [
                [1, 1, 2, 3, 0],  # triangle
                [2, 3, 4, 5, 6],  # quad (edges 3-4, 4-5, 5-6, 6-3)
            ],
            dtype=np.int64,
        )
        adj = _build_shared_nodes_graph([1, 2, 3, 4, 5, 6], elements)
        assert adj[1] == {2, 3}
        assert adj[2] == {1, 3}
        # Node 3 is shared between tri (with 1, 2) and quad (with 4, 6)
        assert adj[3] == {1, 2, 4, 6}
        assert adj[4] == {3, 5}
        assert adj[5] == {4, 6}
        assert adj[6] == {3, 5}


# ---------------------------------------------------------------------------
# Unit tests: _chain_ring (used by both _build_domain_polygon and the QGIS plugin)
# ---------------------------------------------------------------------------


class TestChainRing:
    """Must handle arbitrary segment order and orientation."""

    def test_empty(self):
        assert _chain_ring([]) == []

    def test_single_segment(self):
        assert _chain_ring([[1, 2, 3, 4]]) == [1, 2, 3, 4]

    def test_two_segments_in_order(self):
        # Segment A's last node == segment B's first node
        result = _chain_ring([[1, 2, 3], [3, 4, 5]])
        assert result == [1, 2, 3, 4, 5]

    def test_two_segments_out_of_order(self):
        # B[-1] matches A's first node — B prepends (reversed-form match
        # against `first`). With (B=[3,4,5], A=[1,2,3]) the algorithm pops
        # segment[0] first and then matches the rest.
        result = _chain_ring([[3, 4, 5], [1, 2, 3]])
        assert result == [1, 2, 3, 4, 5]

    def test_segment_reversed(self):
        # Second segment given in reverse direction: seg[-1] == ring[-1]
        result = _chain_ring([[1, 2, 3], [5, 4, 3]])
        assert result == [1, 2, 3, 4, 5]

    def test_extract_mesh_like_layout(self):
        # Mimics extract_mesh output for a single-cut subdomain:
        #   open_kept   = [sa, x, sb]
        #   cut_segment = [sa, z, sc]   (direction not aligned with CCW)
        #   land_kept   = [sc, y, sb]   (direction not aligned with CCW)
        # Naive concatenation would emit jump-lines sb -> sa and sc -> sc.
        sa, sb, sc = 100, 200, 300
        x, y, z = 11, 12, 13
        segments = [
            [sa, x, sb],  # original open kept
            [sa, z, sc],  # cut boundary
            [sc, y, sb],  # original land kept
        ]
        ring = _chain_ring(segments)
        assert ring[0] == ring[-1], f"ring not closed: starts {ring[0]} ends {ring[-1]}"
        all_nodes = {n for seg in segments for n in seg}
        assert set(ring) == all_nodes
        # 3 segments * 3 nodes - 3 shared endpoints + 1 closing repeat = 7
        assert len(ring) == 7

    def test_disconnected_segments_concatenate(self):
        # When no greedy match is possible, the chain falls back to
        # concatenating the next segment in file order — preserving
        # the pre-greedy behavior so every segment ends up on the
        # perimeter, even with a one-node visual jump where the gap is.
        # See test_off_by_one_endpoints_concatenate for the real-world
        # case this protects (SCHISM hgrid with adjacent-but-not-equal
        # node IDs at the open/land transition).
        result = _chain_ring([[1, 2, 3], [10, 11, 12]])
        assert result == [1, 2, 3, 10, 11, 12]

    def test_off_by_one_endpoints_concatenate(self):
        # Real SCHISM hgrid files occasionally have spatially-adjacent
        # but not-identical node IDs at the open/land transition (the
        # Pacific mesh hits this: open ends at 214834, land starts at
        # 214833). The chain must include every segment — falling back
        # to concatenation when greedy matching can't connect them —
        # rather than ending early and yielding only the first piece.
        open_b = [1, 2, 3]  # ends at 3
        land_b = [4, 5, 6, 7, 0]  # starts at 4 (one off from open's last)
        result = _chain_ring([open_b, land_b])
        assert result == [1, 2, 3, 4, 5, 6, 7, 0]


# ---------------------------------------------------------------------------
# Integration test: extract_mesh produces a chainable, simple polygon
# ---------------------------------------------------------------------------


def _extracted_polygon(mesh_dir, polygon: shapely.Polygon, tmp_path) -> shapely.Polygon:
    """Run extract_mesh and return the polygon built from its boundaries.

    Chains ``open + exterior land`` segments — exactly what the QGIS
    plugin and ``_build_domain_polygon`` produce for visualization or
    CO-OPS gauge filtering.
    """
    out_dir = tmp_path / "extracted"
    extract_mesh(
        input_dir=mesh_dir,
        polygon=polygon,
        output_dir=out_dir.parent,
        output_name=out_dir.name,
        write_netcdf=False,
        crs=4326,
    )

    project = NWMSCHISMProject(out_dir, validate=False)
    bs = project.read_boundaries()
    coords = project.nodes_coordinates

    segments = list(bs.open_boundaries)
    segments.extend(list(lb.nodes) for lb in bs.land_boundaries if lb.is_exterior)
    islands = [list(lb.nodes) for lb in bs.land_boundaries if lb.is_island]

    ring_ids = _chain_ring(segments)
    assert ring_ids, "chained ring empty — extract_mesh produced no exterior segments"

    outer_pts = coords[np.array(ring_ids) - 1].tolist()
    holes = [coords[np.array(isl) - 1].tolist() for isl in islands]
    return shapely.Polygon(outer_pts, holes=holes)


class TestExtractMeshBoundaryChain:
    """End-to-end regression for the QGIS-plugin "diagonal jump" bug.

    A correct extraction must produce boundary segments that chain into a
    simple (non-self-intersecting) polygon — i.e. no jump-lines between
    non-adjacent segments.
    """

    def _inner_rectangle(self) -> shapely.Polygon:
        # Inner rectangle of the 9x7 mesh that cuts both open and land sides.
        return shapely.Polygon([(1.5, 0.5), (6.5, 0.5), (6.5, 4.5), (1.5, 4.5)])

    def test_shore_subset_polygon_is_simple(self, shore_project, tmp_path):
        result = _extracted_polygon(shore_project, self._inner_rectangle(), tmp_path)
        assert result.is_valid, f"extracted polygon invalid: {shapely.is_valid_reason(result)}"
        assert result.is_simple, "extracted polygon must be simple (no self-intersections)"
        assert result.area > 0, "extracted polygon has zero area"

    def test_ocean_subset_polygon_is_simple(self, ocean_project, tmp_path):
        # A polygon containing the ocean-island region but cutting the outer
        # boundary on at least two sides.
        poly = shapely.Polygon([(0.5, 0.5), (7.5, 0.5), (7.5, 5.5), (0.5, 5.5)])
        result = _extracted_polygon(ocean_project, poly, tmp_path)
        assert result.is_valid, f"extracted polygon invalid: {shapely.is_valid_reason(result)}"
        assert result.is_simple, "extracted polygon must be simple (no self-intersections)"

    def test_chained_ring_visits_each_node_once(self, shore_project, tmp_path):
        # In a correctly-chained ring, only the closing node repeats — any
        # other repeat means a jump-line was inserted.
        out_dir = tmp_path / "extracted"
        extract_mesh(
            input_dir=shore_project,
            polygon=self._inner_rectangle(),
            output_dir=out_dir.parent,
            output_name=out_dir.name,
            write_netcdf=False,
            crs=4326,
        )
        project = NWMSCHISMProject(out_dir, validate=False)
        bs = project.read_boundaries()

        segments = list(bs.open_boundaries)
        segments.extend(list(lb.nodes) for lb in bs.land_boundaries if lb.is_exterior)
        ring_ids = _chain_ring(segments)

        interior = ring_ids[:-1] if ring_ids and ring_ids[0] == ring_ids[-1] else ring_ids
        assert len(interior) == len(set(interior)), (
            "chained ring revisits a node — extract_mesh boundary order is "
            "not chainable, or _chain_ring is choosing wrong matches"
        )
