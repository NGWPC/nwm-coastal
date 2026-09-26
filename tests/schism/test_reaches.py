"""Tests for deriving SCHISM source reaches from a hydrofabric."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from coastal_calibration.schism.project_reader import BoundarySet, BoundaryType, LandBoundary
from coastal_calibration.schism.reaches import (
    _classify_crossings,
    domain_polygon,
    generate_reaches,
    infer_region,
    land_rings,
    resolve_nwm_layer,
)

# Annulus geometry shared by the fixtures: a 10x10 domain with a 4..6 hole.
OUTER = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
HOLE = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]


class StubProject:
    """Minimal NWMSCHISMProject stand-in backed by an explicit annulus mesh.

    ``open_ring`` picks which ring is the open boundary.  ``"inner"`` is a lake
    mesh: the nearshore ring is land and the lake is forced through the hole.
    ``"outer"`` is an ocean mesh: open water surrounds an island coastline.
    """

    def __init__(self, open_ring: str = "inner") -> None:
        nodes: list[tuple[float, float]] = []
        lookup: dict[tuple[int, int], int] = {}
        for j in range(11):
            for i in range(11):
                lookup[i, j] = len(nodes)
                nodes.append((float(i), float(j)))
        self.nodes_coordinates = np.array(nodes, dtype=np.float64)

        elements: list[list[int]] = []
        for j in range(10):
            for i in range(10):
                if 4 <= i < 6 and 4 <= j < 6:
                    continue  # the hole
                elements.append(
                    [lookup[i, j], lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]]
                )
        self.element_connections = np.array(elements, dtype=np.int64)
        self.n_elements = len(elements)
        self._open_ring = open_ring

    def _ring(self, corners: list[tuple[float, float]]) -> list[int]:
        index = {tuple(c): n + 1 for n, c in enumerate(self.nodes_coordinates.tolist())}
        ring: list[int] = []
        for start, end in zip(corners, [*corners[1:], corners[0]], strict=True):
            steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1])))
            for step in range(steps):
                t = step / steps
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                ring.append(index[(x, y)])
        ring.append(ring[0])
        return ring

    def read_boundaries(self) -> BoundarySet:
        outer, hole = self._ring(OUTER), self._ring(HOLE)
        open_nodes, island = (hole, outer) if self._open_ring == "inner" else (outer, hole)
        return BoundarySet(
            open_boundaries=[open_nodes],
            land_boundaries=[LandBoundary(nodes=island, boundary_type=BoundaryType.ISLAND)],
        )


def _write_layer(path, lines, ids, crs="EPSG:4326", layer="flowpaths"):
    gpd.GeoDataFrame({"fp_id": ids}, geometry=lines, crs=crs).to_file(
        path, layer=layer, driver="GPKG"
    )
    return path


class TestRegionResolution:
    """Region inference and layer lookup."""

    @pytest.mark.parametrize(
        ("bounds", "expected"),
        [
            ((-83.6, 41.2, -78.7, 43.2), "conus"),
            ((-84.1, 24.5, -80.0, 31.7), "conus"),
            ((-154.3, 58.8, -141.2, 61.7), "alaska"),
            ((-158.2, 20.5, -155.9, 22.3), "hawaii"),
            ((-67.3, 17.9, -65.2, 18.6), "puertorico"),
        ],
    )
    def test_infers_region_from_bounds(self, bounds, expected):
        assert infer_region(bounds) == expected

    def test_rejects_bounds_outside_all_regions(self):
        with pytest.raises(ValueError, match="outside all known NWM regions"):
            infer_region((2.0, 48.0, 3.0, 49.0))

    def test_accepts_a_region_or_a_coastal_domain(self):
        assert resolve_nwm_layer("conus") == "nwm_reaches_conus"
        assert resolve_nwm_layer("greatlakes") == "nwm_reaches_conus"
        assert resolve_nwm_layer("atlgulf") == "nwm_reaches_conus"
        assert resolve_nwm_layer("prvi") == "nwm_reaches_puertorico"
        assert resolve_nwm_layer("alaska") == "nwm_reaches_alaska"

    def test_rejects_unknown_domain(self):
        with pytest.raises(ValueError, match="Unknown region or coastal domain"):
            resolve_nwm_layer("atlantis")


class TestDomainGeometry:
    """The domain polygon keeps its holes; land rings exclude open water."""

    def test_hole_is_kept(self):
        polygon = domain_polygon(StubProject())
        assert len(polygon.interiors) == 1
        assert polygon.area == pytest.approx(96.0)

    def test_inverted_rings_are_repaired(self):
        """A shoreline stored as an island must not leave the lake as the shell."""
        polygon = domain_polygon(StubProject())
        assert polygon.exterior.length == pytest.approx(Polygon(OUTER).exterior.length)

    def test_open_boundary_ring_is_not_land(self):
        project = StubProject()
        rings = land_rings(project, domain_polygon(project))
        assert len(rings.geoms) == 1
        assert rings.geoms[0].length == pytest.approx(Polygon(OUTER).exterior.length)

    def test_island_hole_counts_as_land(self):
        """When the open boundary is the outer ring, the hole is a coastline."""
        project = StubProject(open_ring="outer")
        rings = land_rings(project, domain_polygon(project))
        assert len(rings.geoms) == 1
        assert rings.geoms[0].length == pytest.approx(Polygon(HOLE).exterior.length)


class TestClassifyCrossings:
    """Crossings are found in vertex order and classified in/out."""

    def test_single_inbound_crossing(self):
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        crossings = _classify_crossings(LineString([(-2.0, 5.0), (2.0, 5.0)]), rings, polygon)
        assert len(crossings) == 1
        point, outbound = crossings[0]
        assert (point.x, point.y) == pytest.approx((0.0, 5.0))
        assert outbound is False

    def test_single_outbound_crossing(self):
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        crossings = _classify_crossings(LineString([(2.0, 5.0), (-2.0, 5.0)]), rings, polygon)
        assert len(crossings) == 1
        assert crossings[0][1] is True

    def test_transit_is_in_then_out(self):
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        line = LineString([(-2.0, 2.5), (5.0, 2.5), (5.0, 12.0)])
        crossings = _classify_crossings(line, rings, polygon)
        assert [outbound for _, outbound in crossings] == [False, True]

    def test_ordered_from_the_first_vertex(self):
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        line = LineString([(-2.0, 2.5), (12.0, 2.5)])
        crossings = _classify_crossings(line, rings, polygon)
        assert [round(p.x, 6) for p, _ in crossings] == [0.0, 10.0]

    def test_line_crossing_into_the_hole_enters_at_the_outer_ring(self):
        """Regression: on an annulus the source must never land on the hole edge."""
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        crossings = _classify_crossings(LineString([(-2.0, 5.0), (5.0, 5.0)]), rings, polygon)
        assert len(crossings) == 1
        point = crossings[0][0]
        assert (point.x, point.y) == pytest.approx((0.0, 5.0))
        assert not Polygon(HOLE).exterior.intersects(point.buffer(1e-9))

    def test_no_crossings_when_the_line_stays_inside(self):
        project = StubProject()
        polygon = domain_polygon(project)
        rings = land_rings(project, polygon)
        assert _classify_crossings(LineString([(1.0, 1.0), (2.0, 2.0)]), rings, polygon) == []


class TestGenerateReaches:
    """End-to-end generation against a GeoPackage."""

    def test_writes_one_source_per_crossing(self, tmp_path):
        src = _write_layer(
            tmp_path / "fp.gpkg",
            [LineString([(-2.0, 2.5), (2.0, 2.5)]), LineString([(-2.0, 7.5), (2.0, 7.5)])],
            [111, 222],
        )
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["crossing"] == 2
        assert stats["sources"] == 2
        assert stats["resolved_by_nearest"] == 0
        lines = out.read_text().splitlines()
        assert lines[0] == "2"
        assert sorted(int(ln.split()[1]) for ln in lines[1:3]) == [111, 222]

    def test_transit_is_a_source_and_a_sink(self, tmp_path):
        """A river crossing in, out and in again yields two sources and one sink."""
        line = LineString(
            [(-2.0, 2.5), (5.0, 2.5), (5.0, 12.0), (11.0, 12.0), (11.0, 7.5), (5.0, 7.5)]
        )
        src = _write_layer(tmp_path / "fp.gpkg", [line], [111])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["inbound"] == 2
        assert stats["outbound"] == 1
        assert stats["sources"] == 2
        assert stats["sinks"] == 1
        assert stats["distinct_source_reaches"] == 1

        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert lines[0] == "2"
        source_rows = [ln.split() for ln in lines[1:3]]
        assert {int(r[1]) for r in source_rows} == {111}
        assert len({int(r[0]) for r in source_rows}) == 2  # distinct elements
        assert lines[3] == "1"
        assert int(lines[4].split()[1]) == 111

    def test_two_chained_reaches_each_entering_are_both_sources(self, tmp_path):
        upstream = LineString([(-2.0, 2.5), (5.0, 2.5), (5.0, 12.0), (11.0, 12.0)])
        downstream = LineString([(11.0, 12.0), (11.0, 2.5), (5.0, 2.5)])
        src = _write_layer(tmp_path / "fp.gpkg", [upstream, downstream], [111, 222])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["sources"] == 2
        assert stats["distinct_source_reaches"] == 2

    def test_outlet_becomes_a_sink(self, tmp_path):
        """A lake outlet carries water out of the domain, so it is a sink."""
        inlet = LineString([(-2.0, 5.0), (4.0, 5.0)])
        outlet = LineString([(6.0, 5.0), (12.0, 5.0)])
        src = _write_layer(tmp_path / "fp.gpkg", [inlet, outlet], [111, 333])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["sources"] == 1
        assert stats["sinks"] == 1
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert int(lines[1].split()[1]) == 111
        assert int(lines[3].split()[1]) == 333

    def test_a_stream_flowing_out_of_the_mesh_is_a_sink(self, tmp_path):
        local = LineString([(2.0, 2.0), (2.0, -2.0)])
        src = _write_layer(tmp_path / "fp.gpkg", [local], [444])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["sources"] == 0
        assert stats["sinks"] == 1

    def test_one_row_per_element(self, tmp_path):
        """Two rivers entering the same element collapse to the longer one."""
        longer = LineString([(-4.0, 2.4), (0.5, 2.4)])
        shorter = LineString([(-1.0, 2.6), (0.5, 2.6)])
        src = _write_layer(tmp_path / "fp.gpkg", [longer, shorter], [111, 222])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        assert stats["crossing"] == 2
        assert stats["merged"] == 1
        assert stats["sources"] == 1
        assert int(out.read_text().splitlines()[1].split()[1]) == 111

    def test_projected_source_matches_geographic_source(self, tmp_path):
        """A non-4326 layer must select the same reaches."""
        lines = [LineString([(-2.0, 2.5), (2.0, 2.5)])]
        results = []
        for name, crs in (("ll.gpkg", "EPSG:4326"), ("proj.gpkg", "EPSG:3857")):
            src = tmp_path / name
            geo = gpd.GeoDataFrame({"fp_id": [111]}, geometry=lines, crs="EPSG:4326")
            geo.to_crs(crs).to_file(src, layer="flowpaths", driver="GPKG")
            out = tmp_path / f"{name}.csv"
            generate_reaches(
                StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
            )
            results.append(out.read_text())
        assert results[0] == results[1]

    def test_raises_when_nothing_crosses(self, tmp_path):
        src = _write_layer(tmp_path / "fp.gpkg", [LineString([(20.0, 20.0), (21.0, 21.0)])], [1])
        with pytest.raises(ValueError, match="cross a land boundary"):
            generate_reaches(
                StubProject(),
                src,
                tmp_path / "out.csv",
                layer="flowpaths",
                id_column="fp_id",
                log=lambda _: None,
            )

    def test_output_round_trips_through_the_schism_parser(self, tmp_path):
        from coastal_calibration.schism.prep import _parse_reach_rows

        project = StubProject()
        src = _write_layer(
            tmp_path / "fp.gpkg",
            [LineString([(-2.0, 2.5), (2.0, 2.5)]), LineString([(-2.0, 7.5), (2.0, 7.5)])],
            [111, 222],
        )
        out = tmp_path / "ngenReaches.csv"
        generate_reaches(
            project, src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )

        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        nso = int(lines[0])
        elements, ids = _parse_reach_rows(lines[1 : 1 + nso], nso, "source")
        rest = lines[1 + nso :]
        assert int(rest[0]) == 0  # sinks are always declared, and always empty
        assert all(1 <= e <= project.n_elements for e in elements)
        assert len(set(elements)) == len(elements)
        assert sorted(ids) == [111, 222]

    def test_check_mode_writes_nothing_and_compares(self, tmp_path):
        src = _write_layer(
            tmp_path / "fp.gpkg",
            [LineString([(-2.0, 2.5), (2.0, 2.5)]), LineString([(-2.0, 7.5), (2.0, 7.5)])],
            [111, 222],
        )
        out = tmp_path / "ngenReaches.csv"
        generate_reaches(
            StubProject(), src, out, layer="flowpaths", id_column="fp_id", log=lambda _: None
        )
        before = out.read_text()

        stats = generate_reaches(
            StubProject(),
            src,
            out,
            layer="flowpaths",
            id_column="fp_id",
            dry_run=True,
            log=lambda _: None,
        )
        assert out.read_text() == before
        assert stats["dry_run"] is True
        assert stats["existing_sources"] == 2
        assert stats["shared_source_rows"] == 2
        assert stats["shared_source_reaches"] == 2

    def test_check_mode_without_an_existing_file(self, tmp_path):
        src = _write_layer(tmp_path / "fp.gpkg", [LineString([(-2.0, 2.5), (2.0, 2.5)])], [111])
        out = tmp_path / "ngenReaches.csv"
        stats = generate_reaches(
            StubProject(),
            src,
            out,
            layer="flowpaths",
            id_column="fp_id",
            dry_run=True,
            log=lambda _: None,
        )
        assert not out.exists()
        assert stats["sources"] == 1
        assert "existing_sources" not in stats
