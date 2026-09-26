"""Derive SCHISM source reaches by intersecting the mesh boundary with a hydrofabric.

A SCHISM model needs a reaches crosswalk (``nwmReaches.csv`` or
``ngenReaches.csv``) mapping mesh element IDs to the routed reaches that
discharge into them.  Meshes that are hand-assembled or copied without
going through the ``create`` workflow ship without one, which silently
disables river inflow (``if_source = 0``).

This module rebuilds the file from the mesh boundary, emitting one row per
boundary crossing: a flowline entering the mesh becomes a source and one
leaving it becomes a sink, each at the element holding the crossing.  A river
transiting the domain is therefore both, and its discharge nets out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

# NWM v3 hydrofabric gdb reach layer per region.
NWM_REACH_LAYERS: dict[str, str] = {
    "conus": "nwm_reaches_conus",
    "alaska": "nwm_reaches_alaska",
    "hawaii": "nwm_reaches_hawaii",
    "puertorico": "nwm_reaches_puertorico",
}

DOMAIN_REGIONS: dict[str, str] = {
    "greatlakes": "conus",
    "atlgulf": "conus",
    "pacific": "conus",
    "alaska": "alaska",
    "hawaii": "hawaii",
    "prvi": "puertorico",
}

# Disjoint (west, south, east, north) extents, tested in order.
_REGION_EXTENTS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("alaska", (-180.0, 50.0, -128.0, 72.0)),
    ("hawaii", (-161.0, 17.0, -153.0, 24.0)),
    ("puertorico", (-68.5, 16.5, -64.0, 19.5)),
    ("conus", (-127.0, 22.0, -64.0, 50.0)),
)


def infer_region(bounds: tuple[float, float, float, float]) -> str:
    """Infer the NWM hydrofabric region from mesh *bounds* in EPSG:4326."""
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    for name, (w, s, e, n) in _REGION_EXTENTS:
        if w <= cx <= e and s <= cy <= n:
            return name
    raise ValueError(f"Mesh centroid ({cx:.3f}, {cy:.3f}) is outside all known NWM regions")


def resolve_nwm_layer(region_or_domain: str) -> str:
    """Return the NWM gdb reach layer for a region or a coastal domain."""
    region = DOMAIN_REGIONS.get(region_or_domain, region_or_domain)
    try:
        return NWM_REACH_LAYERS[region]
    except KeyError:
        raise ValueError(
            f"Unknown region or coastal domain {region_or_domain!r}; expected one of "
            f"{sorted(NWM_REACH_LAYERS)} or {sorted(DOMAIN_REGIONS)}"
        ) from None


def domain_polygon(project: Any) -> Any:
    """Return the mesh domain as a polygon, interior holes intact."""
    from coastal_calibration.schism.stages import _build_domain_polygon

    polygon = _build_domain_polygon(project)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    return polygon


def land_rings(project: Any, polygon: Any, *, open_share: float = 0.9) -> Any:
    """Return the boundary rings of *polygon* that are land, as a MultiLineString.

    Flowlines enter across land, so only these rings can hold a crossing.  Which
    ring is land varies by mesh: an ocean domain carries its islands as holes and
    is bounded by open water, while a lake domain is the reverse.  A ring built
    almost entirely from open-boundary nodes is water; every other ring is land.
    """
    from shapely.geometry import MultiLineString

    open_nodes = {
        tuple(np.round(project.nodes_coordinates[node - 1], 9))
        for segment in project.read_boundaries().open_boundaries
        for node in segment
    }

    def is_open(ring: Any) -> bool:
        coords = np.round(np.asarray(ring.coords)[:, :2], 9)
        hits = sum(1 for c in coords if tuple(c) in open_nodes)
        return hits >= open_share * len(coords)

    rings = [polygon.exterior, *polygon.interiors]
    land = [r for r in rings if not is_open(r)]
    if not land:
        raise ValueError("Mesh boundary has no land ring for flowlines to cross")
    return MultiLineString([list(r.coords) for r in land])


def _as_single_line(geom: Any) -> Any:
    """Collapse a (Multi)LineString to one LineString, preserving vertex order."""
    import shapely

    if geom.geom_type == "LineString":
        return geom
    merged = shapely.line_merge(geom)
    if merged.geom_type == "LineString":
        return merged
    return max(merged.geoms, key=lambda g: g.length)


def _crossing_points(line: Any, ring: Any) -> list[Any]:
    """Flatten the intersection of *line* with *ring* into a list of points."""
    from shapely.geometry import LineString, Point

    crossings = line.intersection(ring)
    if crossings.is_empty:
        return []
    if isinstance(crossings, Point):
        return [crossings]
    points: list[Any] = []
    for sub in getattr(crossings, "geoms", []):
        if isinstance(sub, Point):
            points.append(sub)
        elif isinstance(sub, LineString):
            points.extend(Point(c[:2]) for c in sub.coords)
    return points


def _classify_crossings(line: Any, ring: Any, polygon: Any) -> list[tuple[Any, bool]]:
    """Return every boundary crossing of *line* as ``(point, is_outbound)``.

    Flowlines are digitized upstream to downstream, so vertex order gives the
    flow direction.  Endpoint membership cannot: on an annulus both ends of a
    river lie outside the filled polygon.  Each crossing is classified from the
    midpoint of the stretch before it, which survives a grazing touch.
    """
    points = _crossing_points(line, ring)
    if not points:
        return []
    ordered = sorted(((line.project(p), p) for p in points), key=lambda item: item[0])
    crossings: list[tuple[Any, bool]] = []
    previous = 0.0
    for distance, point in ordered:
        inside = polygon.covers(line.interpolate(0.5 * (previous + distance)))
        crossings.append((point, bool(inside)))
        previous = distance
    return crossings


def _element_centroids(project: Any, chunk_size: int = 250_000) -> NDArray[np.float64]:
    """Arithmetic mean of each element's node coordinates."""
    conn = project.element_connections
    coords = project.nodes_coordinates
    out = np.empty((conn.shape[0], 2), dtype=np.float64)
    for start in range(0, conn.shape[0], chunk_size):
        block = conn[start : start + chunk_size]
        mask = block >= 0
        idx = np.where(mask, block, 0)
        pts = coords[idx]
        counts = mask.sum(axis=1, keepdims=True)
        out[start : start + chunk_size] = (pts * mask[..., None]).sum(axis=1) / counts
    return out


def _element_polygon(project: Any, element: int) -> Any:
    """Build the polygon for a 0-based element index."""
    from shapely.geometry import Polygon

    nodes = project.element_connections[element]
    return Polygon(project.nodes_coordinates[nodes[nodes >= 0]])


def _resolve_elements(
    project: Any, points: list[Any], log: Callable[[str], None]
) -> tuple[list[int], int, int]:
    """Map points to 1-based element IDs, preferring the element hosting each."""
    import shapely
    from shapely.strtree import STRtree

    centroids = _element_centroids(project)
    centroid_tree = STRtree(shapely.points(centroids))
    # Characteristic element width, sizing the candidate window.
    xmin, ymin = centroids.min(axis=0)
    xmax, ymax = centroids.max(axis=0)
    span = ((xmax - xmin) * (ymax - ymin) / len(centroids)) ** 0.5

    # Crossings sit on the boundary, so only nearby elements can host them.
    candidates = {int(i) for pt in points for i in centroid_tree.query(pt.buffer(span * 3.0))}
    candidates.update(int(centroid_tree.query_nearest(pt)[0]) for pt in points)
    index = np.fromiter(sorted(candidates), dtype=np.int64)
    log(f"testing {len(index)} candidate element(s) near the boundary")

    polygons = [_element_polygon(project, int(i)) for i in index]
    polygon_tree = STRtree(polygons)

    # Crossings land on an element edge; the slack absorbs reprojection round-off.
    tolerance = span * 0.01
    hit, distance = polygon_tree.query_nearest(points, return_distance=True, all_matches=False)
    # Array input returns (input_index, tree_index) pairs.
    order = np.argsort(hit[0])
    elements = [int(index[int(i)]) + 1 for i in np.asarray(hit[1])[order]]
    hosted = int((np.asarray(distance)[order] <= tolerance).sum())
    snapped = len(points) - hosted
    log(f"resolved {hosted} point(s) inside an element, {snapped} snapped to the nearest")
    return elements, hosted, snapped


def _one_row_per_element(
    elements: list[int], crossings: list[tuple[int, float]]
) -> tuple[list[tuple[int, int]], int]:
    """Collapse crossings sharing an element, keeping the one on the longest flowline.

    Elements are distinct within a block, so an element carries one row even
    when a river crosses into it more than once.
    """
    best: dict[int, tuple[float, int]] = {}
    for elem, (reach_id, length) in zip(elements, crossings, strict=True):
        if elem not in best or length > best[elem][0]:
            best[elem] = (length, reach_id)
    pairs = sorted((elem, reach_id) for elem, (_, reach_id) in best.items())
    return pairs, len(elements) - len(best)


def generate_reaches(
    project: Any,
    source: Path,
    output_file: Path,
    *,
    layer: str,
    id_column: str,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write a reaches crosswalk for the flowlines crossing the mesh boundary.

    One row is written per crossing: a flowline entering the mesh becomes a
    source and one leaving it becomes a sink, so a river transiting the domain
    contributes both and its discharge nets out correctly.

    Parameters
    ----------
    project
        Loaded :class:`NWMSCHISMProject`.
    source
        Hydrofabric dataset (NWM v3 gdb or NextGen GeoPackage).
    output_file
        Destination ``nwmReaches.csv`` / ``ngenReaches.csv``.
    layer
        Layer within *source* holding the flowlines.
    id_column
        Reach identifier column in *layer*.
    dry_run
        Report what would be written, and how it compares to *output_file* if
        that already exists, without writing anything.
    log
        Optional logging callback.

    Returns
    -------
    dict
        Counts for each stage of the selection.
    """
    import geopandas as gpd
    import pyogrio

    from coastal_calibration.schism.ngen_reaches import read_reaches_blocks, write_reaches_blocks

    _log = log if log is not None else logger.info

    polygon = domain_polygon(project)
    rings = land_rings(project, polygon)

    crs = pyogrio.read_info(source, layer=layer)["crs"]
    # pyogrio has no bbox_crs, so move the geometry into the layer CRS.
    moved = gpd.GeoSeries([polygon, rings], crs="EPSG:4326").to_crs(crs)
    polygon_src, ring_src = moved.iloc[0], moved.iloc[1]
    gdf = pyogrio.read_dataframe(source, layer=layer, bbox=polygon_src.bounds).reset_index(
        drop=True
    )
    _log(f"{layer}: {len(gdf)} flowline(s) in the mesh bounding box")

    selected = np.flatnonzero(gdf.intersects(ring_src).to_numpy())
    _log(f"{layer}: {len(selected)} flowline(s) cross a land boundary")
    if not len(selected):
        raise ValueError(f"No flowlines in {layer} cross a land boundary of the mesh")

    inbound: list[tuple[int, float, Any]] = []
    outbound: list[tuple[int, float, Any]] = []
    for row in selected:
        line = _as_single_line(gdf.geometry.iloc[row])
        reach_id = int(gdf[id_column].iloc[row])
        for point, is_outbound in _classify_crossings(line, ring_src, polygon_src):
            bucket = outbound if is_outbound else inbound
            bucket.append((reach_id, line.length, point))
    _log(f"{layer}: {len(inbound)} inbound and {len(outbound)} outbound crossing(s)")

    combined = inbound + outbound
    to_ll = gpd.GeoSeries([p for _, _, p in combined], crs=crs).to_crs("EPSG:4326")
    elements, contained, nearest = _resolve_elements(project, list(to_ll), _log)

    split = len(inbound)
    keys = [(reach_id, length) for reach_id, length, _ in combined]
    sources, source_merged = _one_row_per_element(elements[:split], keys[:split])
    sinks, sink_merged = _one_row_per_element(elements[split:], keys[split:])
    merged = source_merged + sink_merged
    if merged:
        _log(f"{layer}: merged {merged} crossing(s) sharing an element")

    stats: dict[str, Any] = {
        "output_file": output_file,
        "layer": layer,
        "in_bbox": len(gdf),
        "crossing": len(selected),
        "inbound": len(inbound),
        "outbound": len(outbound),
        "merged": merged,
        "resolved_by_containment": contained,
        "resolved_by_nearest": nearest,
        "sources": len(sources),
        "sinks": len(sinks),
        "distinct_source_reaches": len({rid for _, rid in sources}),
        "distinct_sink_reaches": len({rid for _, rid in sinks}),
        "dry_run": dry_run,
    }

    if not dry_run:
        write_reaches_blocks(output_file, sources, sinks)
        _log(f"wrote {len(sources)} source(s) and {len(sinks)} sink(s) to {output_file}")
        return stats

    if output_file.exists():
        existing_sources, existing_sinks = read_reaches_blocks(output_file)
        stats.update(
            existing_sources=len(existing_sources),
            existing_sinks=len(existing_sinks),
            shared_source_reaches=len(
                {rid for _, rid in sources} & {rid for _, rid in existing_sources}
            ),
            shared_sink_reaches=len({rid for _, rid in sinks} & {rid for _, rid in existing_sinks}),
            shared_source_rows=len(set(sources) & set(existing_sources)),
            shared_sink_rows=len(set(sinks) & set(existing_sinks)),
        )
    _log(f"would write {len(sources)} source(s) and {len(sinks)} sink(s); nothing written")
    return stats
