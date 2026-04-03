"""SCHISM hgrid mesh subsetter.

Utilities to divide SCHISM hgrid meshes using a dividing line.
Memory-efficient processing with geodesic classification and chunked I/O.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.io import netcdf_file
from shapely import STRtree

from coastal_calibration.logging import logger
from coastal_calibration.schism.constants import GeoSpatial, Performance, SCHISMFiles, Size
from coastal_calibration.schism.project_reader import (
    BoundarySet,
    LandBoundary,
    NWMSCHISMProject,
    geod,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from shapely import LineString

__all__ = [
    "MeshClassifier",
    "MeshDivisionResult",
    "MeshSubsetter",
    "NodeClassification",
    "SubsetResult",
    "divide_mesh",
]


@contextmanager
def temporary_file(suffix: str, directory: Path, buffer_size: int):
    """Create a temporary file with automatic cleanup.

    Parameters
    ----------
    suffix : str
        File suffix for the temporary file.
    directory : pathlib.Path
        Directory in which to create the temporary file.
    buffer_size : int
        Buffering argument passed to NamedTemporaryFile.

    Yields
    ------
    tuple
        A tuple of (file handle, Path) for the temporary file.
    """
    fd, name = tempfile.mkstemp(suffix=suffix, dir=directory)
    os.close(fd)
    temp_path = Path(name)
    temp_file = temp_path.open("w", buffering=buffer_size)
    try:
        yield temp_file, temp_path
    finally:
        temp_file.close()
        for attempt in range(Performance.MAX_TEMP_FILE_CLEANUP_RETRIES):
            try:
                if temp_path.exists():
                    temp_path.unlink()
                break
            except PermissionError:
                if attempt < Performance.MAX_TEMP_FILE_CLEANUP_RETRIES - 1:
                    time.sleep(Performance.TEMP_FILE_CLEANUP_RETRY_DELAY)


@dataclass(frozen=True)
class NodeClassification:
    """Results from node classification along dividing line.

    Attributes
    ----------
    side_a : numpy.ndarray
        Node IDs assigned to side A.
    side_b : numpy.ndarray
        Node IDs assigned to side B.
    shared : numpy.ndarray
        Node IDs shared between both sides.
    n_cut_elements : int
        Number of elements cut by the dividing line.
    """

    side_a: NDArray[np.int64]
    side_b: NDArray[np.int64]
    shared: NDArray[np.int64]
    n_cut_elements: np.int64

    @property
    def side_a_exclusive(self) -> NDArray[np.int64]:
        """Nodes only on side A, not shared."""
        return np.setdiff1d(self.side_a, self.shared, assume_unique=True)

    @property
    def side_b_exclusive(self) -> NDArray[np.int64]:
        """Nodes only on side B, not shared."""
        return np.setdiff1d(self.side_b, self.shared, assume_unique=True)

    @property
    def n_side_a(self) -> int:
        """Number of nodes on side A."""
        return len(self.side_a)

    @property
    def n_side_b(self) -> int:
        """Number of nodes on side B."""
        return len(self.side_b)

    @property
    def n_shared(self) -> int:
        """Number of shared nodes."""
        return len(self.shared)

    @property
    def n_side_a_exclusive(self) -> int:
        """Number of nodes only on side A, not shared."""
        return self.n_side_a - self.n_shared

    @property
    def n_side_b_exclusive(self) -> int:
        """Number of nodes only on side B, not shared."""
        return self.n_side_b - self.n_shared

    def __str__(self) -> str:
        return (
            f"Classification info:\n"
            f"  Side A: {self.n_side_a:,} nodes ({self.n_side_a_exclusive:,} exclusive)\n"
            f"  Side B: {self.n_side_b:,} nodes ({self.n_side_b_exclusive:,} exclusive)\n"
            f"  Shared: {self.n_shared:,} nodes\n"
            f"  Cut elements: {self.n_cut_elements:,}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SideData:
    """Container for side-specific mesh data.

    Parameters
    ----------
    nodes : numpy.ndarray
        Node IDs included in this side.
    elements : numpy.ndarray
        Element rows included in this side.
    """

    nodes: NDArray[np.int64]
    elements: NDArray[np.int64]
    mapping: dict[int, int] = field(init=False)

    def __post_init__(self):
        self.mapping = {
            original_id: new_id for new_id, original_id in enumerate(self.nodes, start=1)
        }

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def write_mapping_to_file(self, file_path: Path | str) -> None:
        """Write full to subset node ID mapping to a text file.

        Parameters
        ----------
        file_path : pathlib.Path or str
            Destination file path for the mapping output.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as f:
            f.write("# Original_Node_ID Subset_Node_ID\n")
            for original_id in sorted(self.mapping.keys()):
                subset_id = self.mapping[original_id]
                f.write(f"{original_id} {subset_id}\n")


@dataclass(frozen=True)
class SubsetResult:
    """Results from creating mesh subsets.

    Attributes
    ----------
    side_a : SideData
        Side A data container.
    side_b : SideData
        Side B data container.
    shared_nodes : numpy.ndarray
        Shared node IDs between subsets.
    """

    side_a: SideData
    side_b: SideData
    shared_nodes: NDArray[np.int64]

    @property
    def n_shared(self) -> int:
        """Number of shared nodes."""
        return len(self.shared_nodes)

    @property
    def total_nodes(self) -> int:
        """Total number of unique nodes in the subset."""
        return self.side_a.n_nodes + self.side_b.n_nodes - self.n_shared

    @property
    def total_elements(self) -> int:
        """Total number of elements in both subsets."""
        return self.side_a.n_elements + self.side_b.n_elements

    @property
    def overlap_ratio_a(self) -> float:
        """Overlap ratio for side A."""
        return self.n_shared / self.side_a.n_nodes if self.side_a.n_nodes > 0 else 0.0

    @property
    def overlap_ratio_b(self) -> float:
        """Overlap ratio for side B."""
        return self.n_shared / self.side_b.n_nodes if self.side_b.n_nodes > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Subset info:\n"
            f"  Side A: {self.side_a.n_nodes:,} nodes, {self.side_a.n_elements:,} elements "
            f"({self.overlap_ratio_a:.1%} shared)\n"
            f"  Side B: {self.side_b.n_nodes:,} nodes, {self.side_b.n_elements:,} elements "
            f"({self.overlap_ratio_b:.1%} shared)\n"
            f"  Overlap: {self.n_shared:,} shared nodes\n"
            f"  Total: {self.total_nodes:,} unique nodes, {self.total_elements:,} elements\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class MeshDivisionResult:
    """Complete results from mesh division.

    Attributes
    ----------
    input_dir : pathlib.Path
        Input project directory.
    output_dir_a : pathlib.Path
        Output directory for side A.
    output_dir_b : pathlib.Path
        Output directory for side B.
    classification : NodeClassification
        Node classification results.
    subset : SubsetResult
        Subset creation results.
    crs : str or int
        Coordinate reference system used.
    """

    input_dir: Path
    output_dir_a: Path
    output_dir_b: Path
    classification: NodeClassification
    subset: SubsetResult
    crs: str | int

    def __str__(self) -> str:
        crs = pyproj.CRS.from_user_input(self.crs)
        return (
            f"Mesh division info:\n"
            f"  Input: {self.input_dir}\n"
            f"  Output A: {self.output_dir_a}\n"
            f"  Output B: {self.output_dir_b}\n"
            f"  CRS: EPSG:{crs.to_epsg()}\n"
            f"  {self.classification}\n"
            f"  {self.subset}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


class MeshClassifier:
    """Classify mesh nodes using geodesic calculations.

    Parameters
    ----------
    project : NWMSCHISMProject
        Project object providing mesh data.
    cut_line : numpy.ndarray or shapely.LineString
        Dividing line as Nx2 array or shapely LineString.
    crs : str, optional
        Coordinate reference system (default 'EPSG:4326').
    chunk_size : int, optional
        Chunk size for element iteration.
    """

    def _load_mesh(self) -> None:
        logger.info(f"Loading mesh with {self.project.n_nodes:,} nodes...")

        self.mesh_vertices = np.zeros((self.project.n_nodes, 3), dtype=np.float64)
        self.node_ids = np.zeros(self.project.n_nodes, dtype=np.int64)

        idx = 0
        for node_ids_chunk, coords_chunk, depths_chunk in self.project.iter_nodes(self.chunk_size):
            chunk_size = len(node_ids_chunk)
            self.node_ids[idx : idx + chunk_size] = node_ids_chunk
            self.mesh_vertices[idx : idx + chunk_size, :2] = coords_chunk
            self.mesh_vertices[idx : idx + chunk_size, 2] = depths_chunk
            idx += chunk_size

        elements_list = list(self.project.iter_elements(self.chunk_size))
        self.mesh_elements = np.vstack(elements_list) if elements_list else np.array([])

        logger.info(
            f"Loaded {len(self.mesh_vertices):,} nodes, {len(self.mesh_elements):,} elements"
        )

    def __init__(
        self,
        project: NWMSCHISMProject,
        cut_line: NDArray[np.float64] | LineString,
        chunk_size: int = 50000,
    ):
        self.project = project

        if isinstance(cut_line, shapely.LineString):
            self.cut_line = cut_line
        elif isinstance(cut_line, np.ndarray):
            if cut_line.ndim != 2 or cut_line.shape[1] != 2 or cut_line.shape[0] < 2:
                raise ValueError("cut_line must be Nx2 array with at least 2 points")
            self.cut_line = shapely.LineString(cut_line)
        else:
            raise TypeError("cut_line must be numpy array or shapely LineString")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size

        self._mesh_vertices: NDArray[np.float64] | None = None
        self._mesh_elements: NDArray[np.int64] | None = None
        self._node_ids: NDArray[np.int64] | None = None

        self._load_mesh()

    @property
    def mesh_vertices(self) -> NDArray[np.float64]:
        """Get mesh vertices."""
        if self._mesh_vertices is None:
            msg = "Mesh vertices not loaded."
            raise ValueError(msg)
        return self._mesh_vertices

    @mesh_vertices.setter
    def mesh_vertices(self, vertices: NDArray[np.float64]) -> None:
        """Set mesh vertices."""
        if not isinstance(vertices, np.ndarray):
            raise TypeError("vertices must be a numpy array")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must be a Nx3 array")
        self._mesh_vertices = vertices

    @property
    def mesh_elements(self) -> NDArray[np.int64]:
        """Get mesh elements."""
        if self._mesh_elements is None:
            msg = "Mesh elements not loaded."
            raise ValueError(msg)
        return self._mesh_elements

    @mesh_elements.setter
    def mesh_elements(self, elements: NDArray[np.int64]) -> None:
        """Set mesh elements."""
        if not isinstance(elements, np.ndarray):
            raise TypeError("elements must be a numpy array")
        if elements.ndim != 2 or elements.shape[1] not in (4, 5):
            raise ValueError("elements must be a Nx4 or Nx5 array")
        self._mesh_elements = elements

    @property
    def node_ids(self) -> NDArray[np.int64]:
        """Get node IDs."""
        if self._node_ids is None:
            msg = "Node IDs not loaded."
            raise ValueError(msg)
        return self._node_ids

    @node_ids.setter
    def node_ids(self, ids: NDArray[np.int64]) -> None:
        """Set node IDs."""
        if not isinstance(ids, np.ndarray):
            raise TypeError("ids must be a numpy array")
        if ids.ndim != 1:
            raise ValueError("ids must be a 1D array")
        self._node_ids = ids

    def _compute_tangent_angles_geodesic(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute forward azimuths along a polyline using geodesic inv.

        Parameters
        ----------
        coords : numpy.ndarray
            Polyline coordinates shape (N,2).

        Returns
        -------
        numpy.ndarray
            Azimuths in degrees for each input point.
        """
        n_pts = len(coords)
        tangent_azimuths = np.zeros(n_pts)

        az_fwd, _, _ = geod.inv(coords[0, 0], coords[0, 1], coords[1, 0], coords[1, 1])
        tangent_azimuths[0] = az_fwd

        if n_pts > 2:
            az_fwd, _, _ = geod.inv(coords[:-2, 0], coords[:-2, 1], coords[2:, 0], coords[2:, 1])
            tangent_azimuths[1:-1] = az_fwd

        az_fwd, _, _ = geod.inv(coords[-2, 0], coords[-2, 1], coords[-1, 0], coords[-1, 1])
        tangent_azimuths[-1] = az_fwd

        return tangent_azimuths

    def _classify_nodes_geodesic(
        self, line_4326: LineString, mesh_4326: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Classify points relative to a line using geodesic methods.

        Parameters
        ----------
        line_4326 : LineString
            The dividing line in lon/lat coordinates.
        mesh_4326 : numpy.ndarray
            Mesh vertex coordinates shape (n_vertices, 2).

        Returns
        -------
        numpy.ndarray
            Indices of points classified on the 'right' side.
        """
        line_length_m = geod.geometry_length(line_4326)
        target_spacing_m = min(GeoSpatial.LINE_DENSIFICATION_TARGET_M, line_length_m / 100)

        approx_deg_spacing = target_spacing_m / GeoSpatial.APPROX_METERS_PER_DEGREE
        line = shapely.segmentize(line_4326, approx_deg_spacing)

        coords = shapely.get_coordinates(line)

        logger.debug(f"Densified line: {len(coords)} points (original: {len(line_4326.coords)})")

        tree = STRtree(shapely.points(coords))  # pyright:ignore[reportArgumentType]

        query_points = shapely.points(mesh_4326)
        _, nearest_indices = tree.query_nearest(query_points, all_matches=False)

        tangent_azimuths = self._compute_tangent_angles_geodesic(coords)

        nearest_coords = coords[nearest_indices]
        nearest_azimuths = tangent_azimuths[nearest_indices]

        az_to_points, _, _ = geod.inv(
            nearest_coords[:, 0],
            nearest_coords[:, 1],
            mesh_4326[:, 0],
            mesh_4326[:, 1],
        )

        angle_diff = (az_to_points - nearest_azimuths + 360) % 360
        is_right = (angle_diff > 0) & (angle_diff < 180)

        right_count = np.sum(is_right)
        left_count = len(is_right) - right_count

        logger.debug(f"Geodesic classification: {right_count} on right, {left_count} on left")

        return np.where(is_right)[0]

    def _geodesic_classification(self) -> NDArray[np.int8]:
        """Classify nodes by side using geodesic side-of-line test.

        Returns
        -------
        numpy.ndarray
            Integer array marking side for each mesh vertex.
        """
        logger.info(f"Classifying {len(self.mesh_vertices):,} nodes using geodesic calculations...")

        vertices_xy = self.mesh_vertices[:, :2]

        right_indices = self._classify_nodes_geodesic(self.cut_line, vertices_xy)

        node_side = np.ones(len(vertices_xy), dtype=np.int8)
        node_side[right_indices] = 0

        num_side_a = np.sum(node_side == 0)
        num_side_b = np.sum(node_side == 1)

        logger.info(
            f"Initial classification: {num_side_a} nodes on side A, {num_side_b} nodes on side B"
        )
        return node_side

    def classify_nodes(
        self,
    ) -> tuple[
        NodeClassification,
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        """Classify nodes and extract cut boundaries for subsetting.

        Returns
        -------
        tuple
            (NodeClassification, elements_side_a, elements_side_b)
        """
        node_side = self._geodesic_classification()
        msk = node_side == 0
        side_a = self.node_ids[msk]
        side_b = self.node_ids[~msk]

        elements_by_side = {"a": {"tri": [], "quad": []}, "b": {"tri": [], "quad": []}}
        n_cut_elements = np.int64(0)

        for elem in self.project.iter_elements(self.chunk_size):
            is_tri = elem[:, -1] == 0

            if np.any(is_tri):
                tri = elem[is_tri, :-1]
                in_a = np.isin(tri[:, 1:], side_a, assume_unique=True).all(axis=1)
                in_b = np.isin(tri[:, 1:], side_b, assume_unique=True).all(axis=1)
                bnd = ~(in_a | in_b)
                n_cut_elements += np.sum(bnd)

                tri_with_padding = np.insert(tri, 4, 0, axis=1)
                elements_by_side["a"]["tri"].append(tri_with_padding[in_a | bnd])
                elements_by_side["b"]["tri"].append(tri_with_padding[in_b])

            if np.any(~is_tri):
                quad = elem[~is_tri]
                in_a = np.isin(quad[:, 1:], side_a, assume_unique=True).all(axis=1)
                in_b = np.isin(quad[:, 1:], side_b, assume_unique=True).all(axis=1)
                bnd = ~(in_a | in_b)
                n_cut_elements += np.sum(bnd)

                elements_by_side["a"]["quad"].append(quad[in_a | bnd])
                elements_by_side["b"]["quad"].append(quad[in_b])

        side_a_elements = (
            np.vstack(elements_by_side["a"]["tri"] + elements_by_side["a"]["quad"])
            if (elements_by_side["a"]["tri"] or elements_by_side["a"]["quad"])
            else np.array([])
        )

        side_b_elements = (
            np.vstack(elements_by_side["b"]["tri"] + elements_by_side["b"]["quad"])
            if (elements_by_side["b"]["tri"] or elements_by_side["b"]["quad"])
            else np.array([])
        )

        side_a = np.unique(side_a_elements[:, 1:], return_index=False)
        side_a = side_a[side_a != 0]
        side_b = np.unique(side_b_elements[:, 1:], return_index=False)
        side_b = side_b[side_b != 0]
        shared = np.intersect1d(side_a, side_b, assume_unique=True)

        logger.info(f"Side A: {len(side_a):,} nodes, {len(side_a_elements):,} elements")
        logger.info(f"Side B: {len(side_b):,} nodes, {len(side_b_elements):,} elements")
        logger.info(f"Shared: {len(shared):,} nodes in overlap zone")

        classification = NodeClassification(
            side_a=side_a,
            side_b=side_b,
            shared=shared,
            n_cut_elements=n_cut_elements,
        )
        return classification, side_a_elements, side_b_elements


def _build_shared_nodes_graph(
    shared_nodes: NDArray[np.int64] | Sequence[int] | Sequence[np.int64],
    elements: NDArray[np.int64],
) -> dict[np.int64, set[np.int64]]:
    """Build undirected graph from shared nodes and element connectivity.

    Parameters
    ----------
    shared_nodes : numpy.ndarray
        Shared node IDs.
    elements : numpy.ndarray
        Element connectivity rows used to build adjacency.

    Returns
    -------
    dict
        Mapping of node_id -> set(neighbor_node_ids).
    """
    elem_nodes = elements[:, 1:5].copy()
    elem_nodes[elements[:, 4] == 0, 3] = 0

    is_shared = np.isin(elem_nodes, np.asarray(shared_nodes))
    shared_in_elems = np.where(is_shared, elem_nodes, 0)
    shared_count = (shared_in_elems != 0).sum(axis=1)
    valid_mask = shared_count >= 2

    if not np.any(valid_mask):
        return {}

    valid_shared = shared_in_elems[valid_mask]
    adjacency = defaultdict(set)

    for elem_shared in valid_shared:
        nodes = elem_shared[elem_shared != 0]

        if len(nodes) >= 2:
            i_idx, j_idx = np.triu_indices(len(nodes), k=1)
            pairs_i = nodes[i_idx]
            pairs_j = nodes[j_idx]

            for ni, nj in zip(pairs_i, pairs_j, strict=False):
                adjacency[int(ni)].add(int(nj))
                adjacency[int(nj)].add(int(ni))

    return adjacency


def _find_path(
    adjacency: dict[np.int64, set[np.int64]], start_node: np.int64, end_node: np.int64
) -> NDArray[np.int64]:
    """Find a path between two nodes in a line graph.

    Parameters
    ----------
    adjacency : dict
        Node adjacency mapping.
    start_node : int
        Starting node ID.
    end_node : int
        Ending node ID.

    Returns
    -------
    numpy.ndarray
        Node ID array representing a path from start to end.
    """
    if start_node == end_node:
        return np.array([start_node], dtype=np.int64)

    path = [start_node]
    visited = {start_node}
    current = start_node

    while current != end_node:
        neighbors = [n for n in adjacency[current] if n not in visited]

        if not neighbors:
            break

        next_node = neighbors[0]
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return np.array(path, dtype=np.int64)


def _split_boundary(
    full_boundary: NDArray[np.int64],
    submesh_nodes: NDArray[np.int64],
    shared_nodes: NDArray[np.int64],
    shared_adjacency: dict[np.int64, set[np.int64]],
) -> NDArray[np.int64]:
    """Split a full boundary for a single submesh using graph traversal.

    Parameters
    ----------
    full_boundary : numpy.ndarray
        Full boundary node sequence.
    submesh_nodes : numpy.ndarray
        Node IDs present in the submesh.
    shared_nodes : numpy.ndarray
        Shared node IDs across both submeshes.
    shared_adjacency : dict
        Adjacency graph of shared nodes.

    Returns
    -------
    numpy.ndarray
        New boundary node sequence for the submesh (1-based IDs).
    """
    subgraph = full_boundary[np.isin(full_boundary, submesh_nodes, assume_unique=True)]

    if subgraph.size == 0:
        return np.array([], dtype=np.int64)

    shared_in_subgraph = np.intersect1d(subgraph, shared_nodes, assume_unique=True)

    if shared_in_subgraph.size < 2:
        return subgraph

    start_shared = shared_in_subgraph[0]
    start_idx = np.where(subgraph == start_shared)[0][0]

    start_idx_in_full = np.where(full_boundary == start_shared)[0][0]
    next_after_start = full_boundary[(start_idx_in_full + 1) % len(full_boundary)]
    start_is_terminal = next_after_start not in submesh_nodes

    new_boundary = []

    if start_is_terminal:
        other_shared = shared_in_subgraph[1] if len(shared_in_subgraph) > 1 else start_shared

        shared_path = _find_path(shared_adjacency, start_shared, other_shared)
        new_boundary.extend(shared_path)

        other_idx = np.where(subgraph == other_shared)[0][0]
        idx = (other_idx + 1) % len(subgraph)

        while idx != start_idx:
            node = subgraph[idx]
            if node not in shared_nodes:
                new_boundary.append(node)
            idx = (idx + 1) % len(subgraph)
    else:
        idx = start_idx
        terminal_shared = None

        for _ in range(len(subgraph) + 1):
            node = subgraph[idx]
            new_boundary.append(node)

            if node in shared_nodes and node != start_shared:
                idx_in_full = np.where(full_boundary == node)[0][0]
                next_in_full = full_boundary[(idx_in_full + 1) % len(full_boundary)]

                if next_in_full not in submesh_nodes:
                    terminal_shared = np.int64(node)
                    break

            idx = (idx + 1) % len(subgraph)

        if terminal_shared is not None and terminal_shared != start_shared:
            shared_path = _find_path(shared_adjacency, terminal_shared, start_shared)
            new_boundary.extend(shared_path[1:-1])

    return np.array(new_boundary, dtype=np.int64)


@dataclass
class _WorkingSideData:
    """Container for side-specific data during I/O passes.

    Attributes
    ----------
    side_data : SideData
        SideData object with mapping and element info.
    output_path : pathlib.Path
        Final output file path for the side.
    temp_file : IO[str]
        Temporary file handle used during writing.
    temp_path : pathlib.Path
        Path to the temporary file.
    """

    side_data: SideData
    output_path: Path
    temp_file: IO[str]
    temp_path: Path


def subset_nwm_reaches_file(
    input_file: Path,
    output_file: Path,
    element_mapping: dict[int, int],
) -> dict[str, int]:
    """Subset NWM reaches file based on element mapping.

    Parameters
    ----------
    input_file : pathlib.Path
        Input NWM reaches file path.
    output_file : pathlib.Path
        Output path for subset file.
    element_mapping : dict
        Mapping from original element IDs to new IDs.

    Returns
    -------
    dict
        Statistics about blocks/elements in/out and dropped counts.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Subsetting NWM reaches file: {input_file} -> {output_file}")

    blocks_out = []
    total_blocks_in = 0
    total_elements_in = 0
    total_elements_dropped = 0

    with input_file.open("r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                n_elements = int(line.strip())
            except ValueError:
                continue

            total_blocks_in += 1
            total_elements_in += n_elements

            block_entries = []
            for _ in range(n_elements):
                entry_line = f.readline()
                if not entry_line.strip():
                    logger.warning(f"Unexpected end of file in block with {n_elements} elements")
                    break

                parts = entry_line.strip().split()
                if len(parts) != 2:
                    logger.warning(f"Skipping invalid entry: {entry_line.strip()}")
                    total_elements_dropped += 1
                    continue

                try:
                    original_element_id = int(parts[0])
                    comid = int(parts[1])
                except ValueError:
                    logger.warning(f"Skipping entry with invalid integers: {entry_line.strip()}")
                    total_elements_dropped += 1
                    continue

                if original_element_id in element_mapping:
                    new_element_id = element_mapping[original_element_id]
                    block_entries.append(f"{new_element_id} {comid}")
                else:
                    total_elements_dropped += 1

            if block_entries:
                blocks_out.append(block_entries)

    with output_file.open("w") as f:
        for block in blocks_out:
            f.write(f"{len(block)}\n")
            for entry in block:
                f.write(f"{entry}\n")
            f.write("\n")

    stats = {
        "blocks_in": total_blocks_in,
        "blocks_out": len(blocks_out),
        "blocks_dropped": total_blocks_in - len(blocks_out),
        "elements_in": total_elements_in,
        "elements_out": sum(len(block) for block in blocks_out),
        "elements_dropped": total_elements_dropped,
    }

    logger.info(
        f"Subset complete: {stats['blocks_out']}/{stats['blocks_in']} blocks, "
        f"{stats['elements_out']}/{stats['elements_in']} elements "
        f"({stats['elements_dropped']} dropped)"
    )

    return stats


def subset_nongrid_file(
    project: NWMSCHISMProject,
    input_file: Path,
    output_file: Path,
    node_mapping: dict[int, int],
    element_mapping: dict[int, int],
    chunk_size: int = Performance.DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    """Subset non-grid GR3-style file using node/element mappings.

    Parameters
    ----------
    project : NWMSCHISMProject
        Project object for buffer settings and counts.
    input_file : pathlib.Path
        Input GR3-like file to subset.
    output_file : pathlib.Path
        Destination file path.
    node_mapping : dict
        Mapping of original node IDs to new subset IDs.
    element_mapping : dict
        Mapping of original element IDs to new subset IDs.
    chunk_size : int, optional
        Chunk size for processing.

    Returns
    -------
    dict
        Simple stats: nodes_in, nodes_out, elements_in, elements_out.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input GR3 file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Subsetting GR3 file: {input_file} -> {output_file}")

    # Read metadata from input file
    with input_file.open("r", buffering=project.buffer_size) as f:
        description = f.readline().strip()
        line = f.readline().split()
        n_elements = int(line[0])
        n_nodes = int(line[1])

    nodes_out = len(node_mapping)
    elements_out = len(element_mapping)

    logger.debug(f"Subsetting {n_nodes:,} nodes to {nodes_out:,}")
    logger.debug(f"Subsetting {n_elements:,} elements to {elements_out:,}")

    with output_file.open("w", buffering=project.buffer_size) as f_out:
        f_out.write(f"{description} - Subset\n")
        f_out.write(f"{elements_out} {nodes_out}\n")

        # Read and write nodes in chunks
        with input_file.open("r", buffering=project.buffer_size) as f_in:
            f_in.readline()
            f_in.readline()

            nodes_read = 0
            while nodes_read < n_nodes:
                current_chunk = min(chunk_size, n_nodes - nodes_read)

                for _ in range(current_chunk):
                    line = f_in.readline()
                    parts = line.split()
                    original_node_id = int(parts[0])

                    if original_node_id in node_mapping:
                        new_node_id = node_mapping[original_node_id]
                        x, y = float(parts[1]), float(parts[2])
                        n_value = float(parts[3])
                        f_out.write(f"{new_node_id} {x:.6f} {y:.6f} {n_value:.3f}\n")

                nodes_read += current_chunk

            # Read and write elements in chunks
            elem_id_new = 1
            elements_read = 0
            while elements_read < n_elements:
                current_chunk = min(chunk_size, n_elements - elements_read)

                for _ in range(current_chunk):
                    line = f_in.readline()
                    parts = line.split()
                    original_elem_id = int(parts[0])

                    if original_elem_id in element_mapping:
                        elem_type = int(parts[1])
                        if elem_type == 3:
                            elem_nodes = [int(parts[i]) for i in range(2, 5)]
                        else:
                            elem_nodes = [int(parts[i]) for i in range(2, 6)]

                        if all(n in node_mapping for n in elem_nodes):
                            new_nodes = [node_mapping[n] for n in elem_nodes]
                            node_str = " ".join(map(str, new_nodes))
                            f_out.write(f"{elem_id_new} {elem_type} {node_str}\n")
                            elem_id_new += 1

                elements_read += current_chunk

    stats = {
        "nodes_in": n_nodes,
        "nodes_out": nodes_out,
        "elements_in": n_elements,
        "elements_out": elements_out,
    }

    logger.info(
        f"Subset complete: {stats['nodes_out']}/{stats['nodes_in']} nodes, "
        f"{stats['elements_out']}/{stats['elements_in']} elements"
    )

    return stats


class MeshSubsetter:
    """Create subset meshes with chunked I/O.

    Parameters
    ----------
    project : NWMSCHISMProject
        Project providing mesh files and settings.
    side_a_nodes : numpy.ndarray
        Node IDs for side A.
    side_b_nodes : numpy.ndarray
        Node IDs for side B.
    chunk_size : int, optional
        Chunk size for node/element passes.
    input_file_override : pathlib.Path or None, optional
        Optional override for the input grid file.
    """

    def __init__(
        self,
        project: NWMSCHISMProject,
        side_a_nodes: NDArray[np.int64],
        side_b_nodes: NDArray[np.int64],
        chunk_size: int = Performance.DEFAULT_CHUNK_SIZE,
        input_file_override: Path | None = None,
    ):
        self.project = project
        self.side_a_nodes = side_a_nodes
        self.side_b_nodes = side_b_nodes
        self.chunk_size = chunk_size
        self.input_file_override = input_file_override

    def _get_input_file(self) -> Path:
        """Return the input file used for reading mesh data.

        Returns
        -------
        pathlib.Path
            Path to the input file (override or default hgrid.gr3).
        """
        return self.input_file_override or self.project.hgrid_file

    def _write_nodes(
        self,
        side_a_working: _WorkingSideData,
        side_b_working: _WorkingSideData,
        n_elem_a: int,
        n_elem_b: int,
        description: str,
    ) -> None:
        """Write node lines for both sides in a single pass.

        Parameters
        ----------
        side_a_working : _WorkingSideData
            Working data holder for side A.
        side_b_working : _WorkingSideData
            Working data holder for side B.
        n_elem_a, n_elem_b : int
            Number of elements for sides A and B.
        description : str
            Mesh description/header string to write.
        """
        input_file = self._get_input_file()
        project = NWMSCHISMProject(
            input_file.parent, buffer_size=self.project.buffer_size, validate=False
        )
        nwm_required_files = set(project.optional_files.values()) - {project.elev_corr_file}
        nwm_required_files = set(project.required_files.values()) | nwm_required_files
        missing_files = [f for f in nwm_required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"The following required NWM-SCHISM files are missing: "
                f"{', '.join(str(f) for f in missing_files)}"
            )

        for node_ids, coords, values in project.iter_nodes(self.chunk_size):
            for i in range(len(node_ids)):
                node_id = node_ids[i]
                x, y, v = coords[i, 0], coords[i, 1], values[i]
                line = f"{x:.6f} {y:.6f} {v:.3f}\n"

                if node_id in side_a_working.side_data.mapping:
                    new_id = side_a_working.side_data.mapping[node_id]
                    side_a_working.temp_file.write(f"{new_id} {line}")

                if node_id in side_b_working.side_data.mapping:
                    new_id = side_b_working.side_data.mapping[node_id]
                    side_b_working.temp_file.write(f"{new_id} {line}")

        side_a_working.temp_file.close()
        side_b_working.temp_file.close()

        for working, n_elem, side_label in [
            (side_a_working, n_elem_a, "A"),
            (side_b_working, n_elem_b, "B"),
        ]:
            with working.output_path.open("w", buffering=self.project.buffer_size) as f_out:
                f_out.write(f"{description} - Side {side_label}\n")
                f_out.write(f"{n_elem} {len(working.side_data.nodes)}\n")
                with working.temp_path.open("r", buffering=self.project.buffer_size) as f_in:
                    f_out.write(f_in.read())

    def _write_elements(
        self,
        output_path: Path,
        elements: NDArray[np.int64],
        mapping: dict[int, int],
    ) -> None:
        """Append elements to an hgrid file with remapped node IDs.

        Parameters
        ----------
        output_path : pathlib.Path
            File to append element lines to.
        elements : numpy.ndarray
            Element connectivity rows.
        mapping : dict
            Mapping from original node IDs to new node IDs.
        """
        with output_path.open("a", buffering=self.project.buffer_size) as f:
            for i, elem in enumerate(elements, start=1):
                if elem[4] == 0:
                    nodes = " ".join(str(mapping[int(n)]) for n in elem[1:4])
                    f.write(f"{i} 3 {nodes}\n")
                else:
                    nodes = " ".join(str(mapping[int(n)]) for n in elem[1:])
                    f.write(f"{i} 4 {nodes}\n")

    def _split_boundaries_for_side(
        self,
        boundary_set: BoundarySet,
        side_nodes: NDArray[np.int64],
        shared_nodes: NDArray[np.int64],
        shared_adjacency: dict[np.int64, set[np.int64]],
        mapping: dict[int, int],
    ) -> BoundarySet:
        """Split and remap boundaries for a single side.

        Parameters
        ----------
        boundary_set : BoundarySet
            Full set of boundaries for the mesh.
        side_nodes : numpy.ndarray
            Node IDs for the side.
        shared_nodes : numpy.ndarray
            Shared node IDs.
        shared_adjacency : dict
            Shared-node adjacency graph.
        mapping : dict
            Node ID mapping from original to subset IDs.

        Returns
        -------
        BoundarySet
            New BoundarySet with remapped node IDs for the side.
        """
        open_bnd_cross = {
            i
            for i, bnd in enumerate(boundary_set.open_boundaries)
            if np.isin(bnd, shared_nodes, assume_unique=True).any()
        }
        land_bnd_cross = {
            i
            for i, bnd in enumerate(boundary_set.land_boundaries)
            if np.isin(bnd.nodes, shared_nodes, assume_unique=True).any()
        }

        open_bnds = []
        open_flags = []

        for idx, open_bnd in enumerate(boundary_set.open_boundaries):
            if idx not in open_bnd_cross:
                if np.isin(open_bnd, side_nodes, assume_unique=True).all():
                    open_bnds.append([mapping[n] for n in open_bnd])
                    # Preserve flags if available
                    if idx < len(boundary_set.open_boundary_flags):
                        open_flags.append(boundary_set.open_boundary_flags[idx])
                continue
            bnd = _split_boundary(np.array(open_bnd), side_nodes, shared_nodes, shared_adjacency)
            if bnd.size > 0:
                open_bnds.append([mapping[n] for n in bnd])
                # Preserve flags if available
                if idx < len(boundary_set.open_boundary_flags):
                    open_flags.append(boundary_set.open_boundary_flags[idx])

        land_bnds = []
        for idx, land_bnd in enumerate(boundary_set.land_boundaries):
            if idx not in land_bnd_cross:
                if np.isin(land_bnd.nodes, side_nodes, assume_unique=True).all():
                    remapped_nodes = [mapping[n] for n in land_bnd.nodes]
                    land_bnds.append(
                        LandBoundary(nodes=remapped_nodes, boundary_type=land_bnd.boundary_type)
                    )
                continue
            bnd = _split_boundary(
                np.array(land_bnd.nodes), side_nodes, shared_nodes, shared_adjacency
            )
            if bnd.size > 0:
                remapped_nodes = [mapping[n] for n in bnd]
                land_bnds.append(
                    LandBoundary(nodes=remapped_nodes, boundary_type=land_bnd.boundary_type)
                )

        return BoundarySet(
            open_boundaries=open_bnds,
            land_boundaries=land_bnds,
            bctides_header=boundary_set.bctides_header,
            ntip_line=boundary_set.ntip_line,
            nbfr_line=boundary_set.nbfr_line,
            open_boundary_flags=open_flags,
        )

    def write_esmf_netcdf(
        self,
        output_dir: Path,
        boundary_set: BoundarySet | None = None,
    ) -> dict[str, int]:
        """Write ESMF NetCDF files for a mesh subset.

        The function writes hgrid.nc and optionally open_bnds_hgrid.nc in the
        given output directory using ESMF unstructured conventions.

        Parameters
        ----------
        output_dir : pathlib.Path
            Directory where NetCDF files will be written.
        boundary_set : BoundarySet or None, optional
            Optional boundary information to include in open boundary file.

        Returns
        -------
        dict
            Stats including n_nodes, n_elements, max_nodes_per_element and
            n_open_bnd_nodes.
        """
        hgrid_file = output_dir / SCHISMFiles.HGRID_GR3
        output_nc_file = output_dir / SCHISMFiles.HGRID_NC
        open_bnds_nc_file = output_dir / SCHISMFiles.OPEN_BOUNDS_NC

        logger.info(f"Writing ESMF NetCDF files from {hgrid_file.name}")
        project = NWMSCHISMProject(output_dir, buffer_size=self.project.buffer_size, validate=False)

        is_tri = project.element_connections[:, -1] == 0
        num_element_conn = ~is_tri + 3

        logger.debug(f"Writing {output_nc_file.name}...")
        with netcdf_file(str(output_nc_file), "w", version=2) as nc:
            nc.createDimension("nodeCount", project.n_nodes)
            nc.createDimension("elementCount", project.n_elements)
            nc.createDimension("maxNodePElement", project.max_nodes_per_element)
            nc.createDimension("coordDim", 2)

            node_coords_var = nc.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
            node_coords_var[:] = project.nodes_coordinates
            node_coords_var.units = "degrees"

            element_conn_var = nc.createVariable(
                "elementConn", "i4", ("elementCount", "maxNodePElement")
            )
            element_conn_var[:] = project.element_connections
            element_conn_var.long_name = "Node indices that define the element connectivity"
            element_conn_var.start_index = 0

            num_element_conn_var = nc.createVariable("numElementConn", "i1", ("elementCount",))
            num_element_conn_var[:] = num_element_conn
            num_element_conn_var.long_name = "Number of nodes per element"

            center_coords_var = nc.createVariable(
                "centerCoords", "f8", ("elementCount", "coordDim")
            )
            center_coords_var[:] = project.elements_centroid
            center_coords_var.units = "degrees"

            nc.gridType = "unstructured"
            nc.version = "0.9"

        # Write open boundaries NetCDF file (open_bnds_hgrid.nc)
        n_open_bnd_nodes = 0
        if boundary_set is not None and boundary_set.n_open > 0:
            logger.debug(f"Writing {open_bnds_nc_file.name}...")

            # Collect all open boundary nodes (already in subset numbering from 1-based)
            all_open_nodes = []
            for boundary in boundary_set.open_boundaries:
                all_open_nodes.extend(boundary)

            # Convert to 0-based indexing
            all_open_nodes = np.array(all_open_nodes, dtype=np.int32) - 1
            n_open_bnd_nodes = len(all_open_nodes)

            with netcdf_file(str(open_bnds_nc_file), "w", version=2) as nc:
                nc.createDimension("nodeCount", project.n_nodes)
                nc.createDimension("coordDim", 2)
                nc.createDimension("openBndNodeCount", n_open_bnd_nodes)
                nc.createDimension("elementCount", project.n_elements)
                nc.createDimension("maxNodePElement", project.max_nodes_per_element)

                node_coords_var = nc.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
                node_coords_var[:] = project.nodes_coordinates
                node_coords_var.units = "degrees"

                open_nodes_var = nc.createVariable("openBndNodes", "i4", ("openBndNodeCount",))
                open_nodes_var[:] = all_open_nodes
                open_nodes_var.long_name = "Open boundary node indices (0-based)"
                open_nodes_var.start_index = 0

                element_conn_var = nc.createVariable(
                    "elementConn", "i4", ("elementCount", "maxNodePElement")
                )
                element_conn_var[:] = project.element_connections
                element_conn_var.long_name = "Node indices that define the element connectivity"
                element_conn_var.start_index = 0

                num_element_conn_var = nc.createVariable("numElementConn", "i1", ("elementCount",))
                num_element_conn_var[:] = num_element_conn
                num_element_conn_var.long_name = "Number of nodes per element"

                center_coords_var = nc.createVariable(
                    "centerCoords", "f8", ("elementCount", "coordDim")
                )
                center_coords_var[:] = project.elements_centroid
                center_coords_var.units = "degrees"

                nc.gridType = "unstructured"
                nc.version = "0.9"
                nc.n_open_boundary_segments = boundary_set.n_open

            logger.info(
                f"Open boundaries: {n_open_bnd_nodes:,} nodes in {boundary_set.n_open} segments"
            )
        else:
            logger.warning("No open boundaries to write")

        stats = {
            "n_nodes": project.n_nodes,
            "n_elements": project.n_elements,
            "max_nodes_per_element": project.max_nodes_per_element,
            "n_open_bnd_nodes": n_open_bnd_nodes,
        }

        logger.info(
            f"ESMF mesh written: {stats['n_nodes']:,} nodes, "
            f"{stats['n_elements']:,} elements, "
            f"max {stats['max_nodes_per_element']} nodes/element"
        )

        return stats

    def subset_mesh(
        self,
        output_dir_a: Path | str,
        output_dir_b: Path | str,
        shared_nodes: NDArray[np.int64],
        elements_a: NDArray[np.int64],
        elements_b: NDArray[np.int64],
        output_filename: str = SCHISMFiles.HGRID_GR3,
        write_boundaries: bool = True,
        write_bctides: bool = True,
        write_hgrid_ll: bool = True,
        write_netcdf: bool = False,
    ) -> SubsetResult:
        """Create both subset meshes ensuring no overlapping elements.

        Parameters
        ----------
        output_dir_a, output_dir_b : pathlib.Path or str
            Output directories for side A and side B.
        shared_nodes : numpy.ndarray
            Node IDs shared between both sides.
        elements_a, elements_b : numpy.ndarray
            Element rows for side A and side B.
        output_filename : str, optional
            Output filename to write (default 'hgrid.gr3').
        write_boundaries : bool, optional
            Whether to include boundary sections in outputs.
        write_bctides : bool, optional
            Whether to write bctides.in files for each side.
        write_hgrid_ll : bool, optional
            Whether to create hgrid.ll copies of hgrid.
        write_netcdf : bool, optional
            Whether to write ESMF NetCDF representations.

        Returns
        -------
        SubsetResult
            Object containing mappings and shared node list.
        """
        output_dir_a = Path(output_dir_a)
        output_dir_b = Path(output_dir_b)
        output_dir_a.mkdir(parents=True, exist_ok=True)
        output_dir_b.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating subset meshes: {output_filename}")

        output_a = output_dir_a / output_filename
        output_b = output_dir_b / output_filename

        side_a_data = SideData(nodes=self.side_a_nodes, elements=elements_a)
        side_b_data = SideData(nodes=self.side_b_nodes, elements=elements_b)

        # Get description from input file
        input_file = self._get_input_file()
        with input_file.open("r", buffering=self.project.buffer_size) as f:
            description = f.readline().strip()

        logger.debug("Pass 1: Writing nodes...")
        with (
            temporary_file(".tmp", output_a.parent, self.project.buffer_size) as (
                temp_a_file,
                temp_a,
            ),
            temporary_file(".tmp", output_b.parent, self.project.buffer_size) as (
                temp_b_file,
                temp_b,
            ),
        ):
            side_a_working = _WorkingSideData(
                side_data=side_a_data,
                output_path=output_a,
                temp_file=temp_a_file,
                temp_path=temp_a,
            )
            side_b_working = _WorkingSideData(
                side_data=side_b_data,
                output_path=output_b,
                temp_file=temp_b_file,
                temp_path=temp_b,
            )

            self._write_nodes(
                side_a_working, side_b_working, len(elements_a), len(elements_b), description
            )

        logger.debug("Pass 2: Writing elements...")
        self._write_elements(output_a, elements_a, side_a_data.mapping)
        self._write_elements(output_b, elements_b, side_b_data.mapping)

        boundaries_a = None
        boundaries_b = None

        if write_boundaries:
            logger.debug("Pass 3: Writing boundaries...")
            logger.debug("Building shared node graph...")
            elements = elements_a if len(elements_a) <= len(elements_b) else elements_b
            shared_adjacency = _build_shared_nodes_graph(shared_nodes, elements)

            logger.debug("Reading boundaries...")
            boundary_set = self.project.read_boundaries()

            logger.debug("Splitting boundaries...")
            boundaries_a = self._split_boundaries_for_side(
                boundary_set, self.side_a_nodes, shared_nodes, shared_adjacency, side_a_data.mapping
            )
            boundaries_b = self._split_boundaries_for_side(
                boundary_set, self.side_b_nodes, shared_nodes, shared_adjacency, side_b_data.mapping
            )

            logger.info(
                f"Side A: {boundaries_a.n_open} open boundaries, {boundaries_a.n_land} land boundaries"
            )
            logger.info(
                f"Side B: {boundaries_b.n_open} open boundaries, {boundaries_b.n_land} land boundaries"
            )

            with output_a.open("a", buffering=self.project.buffer_size) as f:
                boundaries_a.write_to_file(f)

            with output_b.open("a", buffering=self.project.buffer_size) as f:
                boundaries_b.write_to_file(f)

            if write_bctides:
                logger.debug("Writing bctides.in files...")
                with (output_dir_a / SCHISMFiles.BCTIDES).open("w") as f:
                    boundaries_a.write_bctides_file(f)

                with (output_dir_b / SCHISMFiles.BCTIDES).open("w") as f:
                    boundaries_b.write_bctides_file(f)

                logger.info(f"Side A: {boundaries_a.n_open_with_bctides} boundaries with tidal BC")
                logger.info(f"Side B: {boundaries_b.n_open_with_bctides} boundaries with tidal BC")

        if write_hgrid_ll and output_filename == SCHISMFiles.HGRID_GR3:
            logger.debug("Creating hgrid.ll files...")
            shutil.copy2(output_a, output_dir_a / SCHISMFiles.HGRID_LL)
            shutil.copy2(output_b, output_dir_b / SCHISMFiles.HGRID_LL)

        logger.info("Writing full to subset node mappings...")
        mapping_a_file = output_dir_a / SCHISMFiles.NODE_MAPPING
        side_a_data.write_mapping_to_file(mapping_a_file)
        mapping_b_file = output_dir_b / SCHISMFiles.NODE_MAPPING
        side_b_data.write_mapping_to_file(mapping_b_file)

        if write_netcdf:
            logger.info("Writing ESMF NetCDF files...")
            self.write_esmf_netcdf(output_dir_a, boundaries_a)
            self.write_esmf_netcdf(output_dir_b, boundaries_b)

        subset_result = SubsetResult(
            side_a=side_a_data,
            side_b=side_b_data,
            shared_nodes=shared_nodes,
        )

        return subset_result


def divide_mesh(
    input_dir: Path | str,
    dividing_line: NDArray[np.float64] | LineString,
    output_dir: Path | str,
    *,
    side_a_name: str = "mesh_a",
    side_b_name: str = "mesh_b",
    write_netcdf: bool = True,
    crs: str | int = 4326,
    re_calc_area: bool = False,
    chunk_size: int = Performance.DEFAULT_CHUNK_SIZE,
    buffer_size: int = Performance.DEFAULT_BUFFER_SIZE,
) -> MeshDivisionResult:
    """Divide a SCHISM mesh into two valid subsets based on a dividing line.

    Parameters
    ----------
    input_dir : pathlib.Path or str
        Directory containing NWM-SCHISM project files.
    dividing_line : numpy.ndarray or shapely.LineString
        Either a shapely LineString or Nx2 array of coordinates.
    output_dir : pathlib.Path or str
        Base output directory to create side subfolders.
    side_a_name, side_b_name : str, optional
        Subdirectory names for side A and side B.
    write_netcdf : bool, optional
        Whether to write ESMF NetCDF files.
    crs : str, optional
        Coordinate reference system used for processing.
    re_calc_area : bool, optional
        Whether to recalculate element areas instead of subsetting from
        existing area file. The area is calculated using the geodesic method
        which is more accurate than the existing area file but is a
        computationally expensive step.
    chunk_size : int, optional
        Number of nodes/elements to process in each chunk.
    buffer_size : int, optional
        File I/O buffer size in bytes.

    Returns
    -------
    MeshDivisionResult
        Paths and statistics describing the division result.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir_a = output_dir / side_a_name
    output_dir_b = output_dir / side_b_name
    output_dir_a.mkdir(parents=True, exist_ok=True)
    output_dir_b.mkdir(parents=True, exist_ok=True)

    if buffer_size >= Size.GB:
        buffer_size_str = f"{buffer_size / Size.GB:.2f} GB"
    elif buffer_size >= Size.MB:
        buffer_size_str = f"{buffer_size / Size.MB:.2f} MB"
    elif buffer_size >= Size.KB:
        buffer_size_str = f"{buffer_size / Size.KB:.2f} KB"
    else:
        buffer_size_str = f"{buffer_size} bytes"

    logger.info(f"Using chunk size: {chunk_size:,}, buffer size: {buffer_size_str}")

    project = NWMSCHISMProject(input_dir, buffer_size)

    classifier = MeshClassifier(project, dividing_line, chunk_size)
    classification, elements_a, elements_b = classifier.classify_nodes()

    logger.info("Subsetting hgrid.gr3...")
    subsetter = MeshSubsetter(
        project,
        classification.side_a,
        classification.side_b,
        chunk_size,
    )

    subset_result = subsetter.subset_mesh(
        output_dir_a,
        output_dir_b,
        classification.shared,
        elements_a,
        elements_b,
        output_filename=SCHISMFiles.HGRID_GR3,
        write_boundaries=True,
        write_bctides=True,
        write_hgrid_ll=True,
        write_netcdf=write_netcdf,
    )

    logger.info("Subsetting hgrid.cpp...")
    subsetter_cpp = MeshSubsetter(
        project,
        classification.side_a,
        classification.side_b,
        chunk_size,
        input_file_override=project.hgrid_cpp_file,
    )

    subsetter_cpp.subset_mesh(
        output_dir_a,
        output_dir_b,
        classification.shared,
        elements_a,
        elements_b,
        output_filename=SCHISMFiles.HGRID_CPP,
        write_bctides=False,  # Don't write bctides for cpp file
        write_hgrid_ll=False,  # Don't write hgrid.ll for cpp file
        write_netcdf=False,  # Don't write NetCDF for cpp file
    )

    element_mapping_a = {int(elem[0]): new_id for new_id, elem in enumerate(elements_a, 1)}
    element_mapping_b = {int(elem[0]): new_id for new_id, elem in enumerate(elements_b, 1)}

    logger.info("Subsetting NWM reaches files...")
    _ = subset_nwm_reaches_file(
        project.nwm_reaches_file,
        output_dir_a / SCHISMFiles.NWM_REACHES,
        element_mapping_a,
    )
    _ = subset_nwm_reaches_file(
        project.nwm_reaches_file,
        output_dir_b / SCHISMFiles.NWM_REACHES,
        element_mapping_b,
    )

    logger.info("Subsetting Manning's coefficient files...")
    _ = subset_nongrid_file(
        project,
        project.manning_file,
        output_dir_a / SCHISMFiles.MANNING,
        subset_result.side_a.mapping,
        element_mapping_a,
        chunk_size,
    )
    _ = subset_nongrid_file(
        project,
        project.manning_file,
        output_dir_b / SCHISMFiles.MANNING,
        subset_result.side_b.mapping,
        element_mapping_b,
        chunk_size,
    )

    logger.info("Subsetting wind rotation files...")
    _ = subset_nongrid_file(
        project,
        project.windrot_file,
        output_dir_a / SCHISMFiles.WINDROT,
        subset_result.side_a.mapping,
        element_mapping_a,
        chunk_size,
    )
    _ = subset_nongrid_file(
        project,
        project.windrot_file,
        output_dir_b / SCHISMFiles.WINDROT,
        subset_result.side_b.mapping,
        element_mapping_b,
        chunk_size,
    )

    if re_calc_area:
        logger.info("Recalculating element areas...")
        areas = project.element_areas
    else:
        logger.info("Reading element area file...")
        areas = np.loadtxt(project.elem_area_file, dtype=np.float64)

    logger.info("Subsetting element area...")
    np.savetxt(output_dir_a / SCHISMFiles.ELEM_AREA, areas[elements_a[:, 0] - 1], fmt="%.3f")
    np.savetxt(output_dir_b / SCHISMFiles.ELEM_AREA, areas[elements_b[:, 0] - 1], fmt="%.3f")

    logger.info("Subsetting elevation initial condition files...")
    _ = subset_nongrid_file(
        project,
        project.elev_ic_file,
        output_dir_a / SCHISMFiles.ELEV_IC,
        subset_result.side_a.mapping,
        element_mapping_a,
        chunk_size,
    )
    _ = subset_nongrid_file(
        project,
        project.elev_ic_file,
        output_dir_b / SCHISMFiles.ELEV_IC,
        subset_result.side_b.mapping,
        element_mapping_b,
        chunk_size,
    )

    if project.elev_corr_file.exists():
        logger.info("Subsetting elevation correction files...")
        corrections = pd.read_csv(project.elev_corr_file, index_col=0)
        elems_a_ids = [int(elem[0]) for elem in elements_a]
        elems_b_ids = [int(elem[0]) for elem in elements_b]
        corrections.loc[elems_a_ids].to_csv(output_dir_a / SCHISMFiles.ELEV_CORRECTION)
        corrections.loc[elems_b_ids].to_csv(output_dir_b / SCHISMFiles.ELEV_CORRECTION)
    else:
        logger.warning("Elevation correction file not found; skipping subsetting.")

    logger.info(f"Complete: Side A={output_dir_a}, Side B={output_dir_b}")
    logger.info(f"Shared boundary: {len(classification.shared):,} nodes")

    return MeshDivisionResult(
        input_dir=input_dir,
        output_dir_a=output_dir_a,
        output_dir_b=output_dir_b,
        classification=classification,
        subset=subset_result,
        crs=crs,
    )
