"""Process and manage NWM-SCHISM project files."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shapely
from pyproj import Geod

if TYPE_CHECKING:
    from collections.abc import Iterator
    from io import TextIOWrapper

    from numpy.typing import NDArray
    from shapely import Polygon


from coastal_calibration.logging import logger
from coastal_calibration.schism.constants import Performance, SCHISMFiles

__all__ = ["BoundarySet", "LandBoundary", "NWMSCHISMProject", "geod"]


geod = Geod(ellps="WGS84")


class BoundaryType(IntEnum):
    """SCHISM boundary types.

    Values
    ------
    EXTERIOR
        Exterior (land) boundary.
    ISLAND
        Island (interior) boundary.
    """

    EXTERIOR = 0
    ISLAND = 1


@dataclass(frozen=True)
class LandBoundary:
    """SCHISM land boundary segment.

    Parameters
    ----------
    nodes : list[int]
        Node IDs for the boundary segment.
    boundary_type : BoundaryType
        Type of boundary (EXTERIOR or ISLAND).
    """

    nodes: list[int]
    boundary_type: BoundaryType

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the boundary."""
        return len(self.nodes)

    @property
    def is_island(self) -> bool:
        """Whether the boundary is an island (interior) boundary."""
        return self.boundary_type == BoundaryType.ISLAND

    @property
    def is_exterior(self) -> bool:
        """Whether the boundary is an exterior (land) boundary."""
        return self.boundary_type == BoundaryType.EXTERIOR


@dataclass(frozen=True)
class BoundarySet:
    """Complete set of boundaries for a mesh.

    Attributes
    ----------
    open_boundaries : list of list of int
        Open boundary node lists.
    land_boundaries : list of Boundary
        Land boundary objects.
    bctides_header : str
        Header text from bctides.in.
    ntip_line : str
        NTIP line content from bctides.in.
    nbfr_line : str
        NBFR line content from bctides.in.
    open_boundary_flags : list of list of int
        Flags for open boundaries read from bctides.in.
    """

    open_boundaries: list[list[int]] = field(default_factory=list)
    land_boundaries: list[LandBoundary] = field(default_factory=list)
    bctides_header: str = ""
    ntip_line: str = ""
    nbfr_line: str = ""
    open_boundary_flags: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def n_open(self) -> int:
        """Number of open boundaries."""
        return len(self.open_boundaries)

    @property
    def n_land(self) -> int:
        """Number of land boundaries."""
        return len(self.land_boundaries)

    @property
    def total_open_nodes(self) -> int:
        """Total number of open boundary nodes."""
        return sum(len(b) for b in self.open_boundaries)

    @property
    def total_land_nodes(self) -> int:
        """Total number of land boundary nodes."""
        return sum(b.n_nodes for b in self.land_boundaries)

    @property
    def n_open_with_bctides(self) -> int:
        """Number of open boundaries with tidal BC applied."""
        return len(self.open_boundary_flags)

    def add_open_boundary(
        self, node_ids: list[int], flags: tuple[int, int, int, int] | None = None
    ) -> None:
        """Add an open boundary segment.

        Parameters
        ----------
        node_ids : list of int
            Node IDs for the open boundary.
        flags : list of int, optional
            Flags for the open boundary (if using bctides).
        """
        self.open_boundaries.append(node_ids)
        if flags is not None:
            if len(flags) != 4 or not all(isinstance(f, int) for f in flags):
                raise ValueError("Open boundary flags must be a tuple of four integers.")
            self.open_boundary_flags.append(flags)

    def add_land_boundary(self, node_ids: list[int], boundary_type: BoundaryType) -> None:
        """Add a land boundary segment.

        Parameters
        ----------
        node_ids : list of int
            Node IDs for the land boundary.
        boundary_type : BoundaryType
            Type of the land boundary (EXTERIOR or ISLAND).
        """
        self.land_boundaries.append(LandBoundary(node_ids, boundary_type))

    def write_to_file(self, file_handle: TextIOWrapper) -> None:
        """Write boundaries to an open file handle."""
        file_handle.write(f"{self.n_open} = Number of open boundaries\n")
        file_handle.write(f"{self.total_open_nodes} = Total number of open boundary nodes\n")

        for i, boundary in enumerate(self.open_boundaries, start=1):
            if len(boundary) == 0:
                raise ValueError(f"Open boundary {i} has no nodes.")
            file_handle.write(f"{len(boundary)} = Number of nodes for open boundary {i}\n")
            file_handle.writelines(f"{node_id}\n" for node_id in boundary)

        file_handle.write(f"{self.n_land} = Number of land boundaries\n")
        file_handle.write(f"{self.total_land_nodes} = Total number of land boundary nodes\n")

        btype_str = {BoundaryType.EXTERIOR: "land", BoundaryType.ISLAND: "island"}
        for i, boundary in enumerate(self.land_boundaries, start=1):
            btype = boundary.boundary_type
            file_handle.write(
                f"{boundary.n_nodes} {btype} = Number of nodes for "
                f"{btype_str[btype]} boundary {i}\n"
            )
            file_handle.writelines(f"{node_id}\n" for node_id in boundary.nodes)

    def write_bctides_file(self, file_handle: TextIOWrapper) -> None:
        """Write bctides.in to an open file handle."""
        if not self.bctides_header or not self.ntip_line or not self.nbfr_line:
            raise ValueError("BoundarySet does not contain bctides.in data")

        # Write header lines as-is
        file_handle.write(f"{self.bctides_header}\n")
        file_handle.write(f"{self.ntip_line}\n")
        file_handle.write(f"{self.nbfr_line}\n")

        nope = self.n_open_with_bctides
        file_handle.write(f"{nope} nope\n")
        for boundary, flags in zip(self.open_boundaries, self.open_boundary_flags, strict=False):
            n_nodes = len(boundary)
            flags_str = " ".join(str(f) for f in flags)
            file_handle.write(f"{n_nodes} {flags_str}\n")

    def __repr__(self) -> str:
        result = (
            "BoundarySet:\n"
            f"  Total open nodes: {self.total_open_nodes} nodes\n"
            f"  Open boundaries: {self.n_open} segments\n"
            f"  Number of nodes in each open boundary:\n"
            f"    " + ", ".join(str(len(b)) for b in self.open_boundaries) + "\n"
            f"  Total land nodes: {self.total_land_nodes} nodes\n"
            f"  Land boundaries: {self.n_land} segments\n"
            f"  Number of nodes in each land boundary:\n"
            f"    " + ", ".join(str(b.n_nodes) for b in self.land_boundaries) + "\n"
        )
        if self.n_open_with_bctides > 0:
            result += f"  Open boundaries with tidal BC: {self.n_open_with_bctides}\n"
        return result


class NWMSCHISMProject:
    """NWM-SCHISM project with required input files.

    Parameters
    ----------
    project_dir : pathlib.Path or str
        Root directory of the project.
    buffer_size : int, optional
        File I/O buffering size in bytes.
    validate : bool, optional
        Whether to validate required files on initialization.
    """

    def _validate_required_files(self) -> None:
        """Validate that all required files exist.

        Raises
        ------
        FileNotFoundError
            If any required file is missing in the project directory.
        """
        missing_files = []
        for name, filepath in self.required_files.items():
            if not filepath.exists():
                missing_files.append(name)

        if missing_files:
            raise FileNotFoundError(
                f"Required files missing from {self.project_dir}:\n"
                + "\n".join(f"  - {name}" for name in missing_files)
            )

        logger.info(f"Validated NWM-SCHISM project: {self.project_dir}")
        for filepath in self.required_files.values():
            logger.debug(f"  - Found required file: {filepath}")

        for filepath in self.optional_files.values():
            if filepath.exists():
                logger.debug(f"  - Found optional file: {filepath}")

    def _read_metadata(self) -> None:
        """Read hgrid.gr3 metadata to populate project attributes.

        This reads counts of nodes/elements and boundary summaries where
        present.
        """
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:

            def _readline(is_desc: bool = False) -> str:
                if is_desc:
                    return f.readline().split("!")[0].split(" = ")[0].strip()
                return f.readline().split("!")[0].strip()

            self.description = _readline()
            line = _readline().split()
            if len(line) < 2:
                raise ValueError(f"Invalid header format in {self.hgrid_file}")

            self.n_elements = int(line[0])
            self.n_nodes = int(line[1])

            if self.n_elements < 0 or self.n_nodes < 0:
                raise ValueError(
                    f"Invalid mesh dimensions: {self.n_elements} elements, {self.n_nodes} nodes"
                )

            for _ in range(self.n_nodes):
                f.readline()

            # Skip element lines to read max nodes per element from element types
            self.max_nodes_per_element = max(
                int(_readline().split()[1]) for _ in range(self.n_elements)
            )

            try:
                self.n_open_bnds = int(_readline(True))
            except ValueError:
                logger.debug(f"No open boundaries found in {self.hgrid_file}.")
                return

            self.total_open_nodes = int(_readline(True))

            for _ in range(self.n_open_bnds):
                n_nodes = int(_readline(True))
                _ = [f.readline() for _ in range(n_nodes)]

            n_land_bnds = _readline(True)
            if n_land_bnds:
                self.n_land_bnds = int(n_land_bnds)
            total_land_nodes = _readline(True)
            if total_land_nodes:
                self.total_land_nodes = int(total_land_nodes)

    def __init__(
        self,
        project_dir: Path | str,
        buffer_size: int = Performance.DEFAULT_BUFFER_SIZE,
        validate: bool = True,
    ):
        self.project_dir = Path(project_dir)
        self.buffer_size = buffer_size

        if not self.project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")

        self.hgrid_file = self.project_dir / SCHISMFiles.HGRID_GR3
        self.hgrid_cpp_file = self.project_dir / SCHISMFiles.HGRID_CPP
        self.manning_file = self.project_dir / SCHISMFiles.MANNING
        self.windrot_file = self.project_dir / SCHISMFiles.WINDROT
        self.elem_area_file = self.project_dir / SCHISMFiles.ELEM_AREA
        self.elev_ic_file = self.project_dir / SCHISMFiles.ELEV_IC
        self.nwm_reaches_file = self.project_dir / SCHISMFiles.NWM_REACHES
        self.bctides_file = self.project_dir / SCHISMFiles.BCTIDES
        self.elev_corr_file = self.project_dir / SCHISMFiles.ELEV_CORRECTION

        self.required_files = {
            "hgrid.gr3": self.hgrid_file,
            "manning.gr3": self.manning_file,
            "bctides.in": self.bctides_file,
        }
        self.optional_files = {
            "elevation_correction.csv": self.elev_corr_file,
            "hgrid.cpp": self.hgrid_cpp_file,
            "windrot_geo2proj.gr3": self.windrot_file,
            "element_areas.txt": self.elem_area_file,
            "elev.ic": self.elev_ic_file,
            "nwmReaches.csv": self.nwm_reaches_file,
        }
        if validate:
            self._validate_required_files()

        self.description = ""
        self.n_elements = 0
        self.n_nodes = 0
        self.n_open_bnds = 0
        self.total_open_nodes = 0
        self.n_land_bnds = 0
        self.total_land_nodes = 0
        self.max_nodes_per_element = 0
        self._read_metadata()
        self._is_geographic: bool | None = None
        self._node_coordinates: NDArray[np.float64] | None = None
        self._element_connections: NDArray[np.int64] | None = None
        self._element_centroids: NDArray[np.float64] | None = None
        self._element_areas: NDArray[np.float64] | None = None

    def _check_geographic(self) -> bool:
        """Detect geographic (deg) coordinates."""
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:
            f.readline()
            f.readline()
            parts = f.readline().split()
            x, y = float(parts[1]), float(parts[2])

        return -180 <= x <= 180 and -90 <= y <= 90

    @property
    def is_geographic(self) -> bool:
        """Whether the mesh uses geographic coordinates (degrees)."""
        if self._is_geographic is None:
            self._is_geographic = self._check_geographic()
        return self._is_geographic

    def _get_node_coordinates(self) -> NDArray[np.float64]:
        """Collect all node coordinates into a Nx2 array."""
        coords = np.zeros((self.n_nodes, 2), dtype=np.float64)
        idx = 0
        for _, coords_chunk, _ in self.iter_nodes(self.buffer_size):
            chunk_size = len(coords_chunk)
            coords[idx : idx + chunk_size, :] = coords_chunk
            idx += chunk_size
        return coords

    @property
    def nodes_coordinates(self) -> NDArray[np.float64]:
        """Get all node coordinates (lon, lat), loading lazily if needed."""
        if self._node_coordinates is None:
            self._node_coordinates = self._get_node_coordinates()
        return self._node_coordinates

    def _read_node_coords_from(self, path: Path) -> NDArray[np.float64]:
        """Read node coordinates from a specific hgrid-format file."""
        coords = np.zeros((self.n_nodes, 2), dtype=np.float64)
        with path.open("r", buffering=self.buffer_size) as f:
            f.readline()  # description
            f.readline()  # header
            for i in range(self.n_nodes):
                parts = f.readline().split("!")[0].split()
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
        return coords

    def _get_geographic_coords(self) -> NDArray[np.float64]:
        """Get node coordinates in geographic (lon/lat) space.

        Uses ``hgrid.ll`` when the primary mesh is projected;
        otherwise returns :attr:`nodes_coordinates` directly.
        """
        if self.is_geographic:
            return self.nodes_coordinates
        hgrid_ll = self.project_dir / SCHISMFiles.HGRID_LL
        if hgrid_ll.exists():
            return self._read_node_coords_from(hgrid_ll)
        raise FileNotFoundError(
            f"Projected mesh requires hgrid.ll for geodesic area computation, "
            f"but {hgrid_ll} was not found. Provide either hgrid.ll or "
            f"a pre-computed element_areas.txt."
        )

    def _compute_element_areas(self) -> NDArray[np.float64]:
        """Compute geodesic areas for all elements.

        Reads lon/lat coordinates from ``hgrid.ll`` when the primary
        mesh (``hgrid.gr3``) uses projected coordinates.
        """
        geo_coords = self._get_geographic_coords()
        areas = np.zeros(self.n_elements, dtype=np.float64)
        idx = 0
        for elem_chunk in self.iter_elements(self.buffer_size):
            node_indices = elem_chunk[:, 1:5] - 1  # 0-indexed
            for i, node_ids in enumerate(node_indices):
                valid_ids = node_ids[node_ids >= 0]
                lons, lats = geo_coords[valid_ids].T
                areas[idx + i] = abs(geod.polygon_area_perimeter(lons, lats)[0])
            idx += len(elem_chunk)
        return areas

    @property
    def element_areas(self) -> NDArray[np.float64]:
        """Get element areas in m².

        Prefers the pre-computed ``element_areas.txt`` file when present
        (faster and correct for both geographic and projected meshes).
        Falls back to geodesic computation using ``hgrid.ll`` for
        projected meshes or ``hgrid.gr3`` for geographic meshes.
        """
        if self._element_areas is None:
            if self.elem_area_file.exists():
                self._element_areas = np.loadtxt(self.elem_area_file, dtype=np.float64)
            else:
                self._element_areas = self._compute_element_areas()
        return self._element_areas

    def _get_element_connections(self) -> NDArray[np.int64]:
        """Collect all element connectivity into an array."""
        element_conn = np.concatenate([e[:, 1:] for e in self.iter_elements(self.buffer_size)])
        if self.max_nodes_per_element == 3:
            element_conn = element_conn[:, :-1]
        # Convert to 0-based indexing
        # The tri elements will have their last node index set to -1
        # which is the convention used by NWM for padding
        element_conn -= 1
        return element_conn

    @property
    def element_connections(self) -> NDArray[np.int64]:
        """Get 0-based indexed element connectivity where tris have -1 for the fourth node."""
        if self._element_connections is None:
            self._element_connections = self._get_element_connections()
        return self._element_connections

    @staticmethod
    def _compute_element_centroids(
        nodes: NDArray[np.float64], elements: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Compute geodesic centroids for mixed tri/quad elements.

        Parameters
        ----------
        nodes : numpy.ndarray
            Node coordinates as (lon, lat) with shape (n_nodes, 2).
        elements : numpy.ndarray
            Element connectivity with shape (n_elements, max_vertices).
            Use -1 or 0 padding for missing vertices in triangular elements.

        Returns
        -------
        numpy.ndarray
            Array of centroid coordinates shape (n_elements, 2).
        """
        if elements.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        n_elem = elements.shape[0]
        n_vert = elements.shape[1]

        mask = np.ones((n_elem, n_vert), dtype=bool) if n_vert == 3 else elements >= 0

        coords = nodes[np.where(mask, elements, 0)]
        lons = coords[:, :, 0]
        lats = coords[:, :, 1]

        lat_rad = np.deg2rad(lats)
        cos_lat = np.cos(lat_rad)

        w_sum = (cos_lat * mask).sum(axis=1, keepdims=True)
        cent_lon = (lons * cos_lat * mask).sum(axis=1) / w_sum.squeeze()
        cent_lat = (lats * cos_lat * mask).sum(axis=1) / w_sum.squeeze()

        # Iteratively refine centroid using inverse distance weighting
        n_iter = 3
        for _ in range(n_iter):
            _, _, dists = geod.inv(
                np.repeat(cent_lon[:, np.newaxis], n_vert, axis=1)[mask],
                np.repeat(cent_lat[:, np.newaxis], n_vert, axis=1)[mask],
                lons[mask],
                lats[mask],
            )

            dist_arr = np.zeros((n_elem, n_vert))
            dist_arr[mask] = dists

            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where((dist_arr > 1.0) & mask, 1.0 / dist_arr, 0.0)

            w_sum = weights.sum(axis=1, keepdims=True)
            w_sum = np.where(w_sum > 0, w_sum, 1.0)

            cent_lon = (lons * weights).sum(axis=1) / w_sum.squeeze()
            cent_lat = (lats * weights).sum(axis=1) / w_sum.squeeze()

        return np.c_[cent_lon, cent_lat]

    @property
    def elements_centroid(self) -> NDArray[np.float64]:
        """Get all element centroids computed in geodesic space (lon, lat)."""
        if self._element_centroids is None:
            self._element_centroids = self._compute_element_centroids(
                self.nodes_coordinates, self.element_connections
            )
        return self._element_centroids

    def iter_nodes(
        self, chunk_size: int = Performance.DEFAULT_CHUNK_SIZE
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]]:
        """Iterate over nodes in chunks.

        Yields
        ------
        tuple
            (node_ids, coords, values) arrays for each chunk.
        """
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:

            def _readline() -> str:
                return f.readline().split("!")[0].strip()

            # Skip header
            f.readline()
            f.readline()

            nodes_read = 0
            while nodes_read < self.n_nodes:
                current_chunk = min(chunk_size, self.n_nodes - nodes_read)

                node_ids = np.zeros(current_chunk, dtype=np.int64)
                coords = np.zeros((current_chunk, 2), dtype=np.float64)
                values = np.zeros(current_chunk, dtype=np.float64)

                for i in range(current_chunk):
                    parts = _readline().split()
                    node_ids[i] = int(parts[0])
                    coords[i, 0] = float(parts[1])
                    coords[i, 1] = float(parts[2])
                    values[i] = float(parts[3])

                nodes_read += current_chunk
                yield node_ids, coords, values

    def iter_elements(
        self, chunk_size: int = Performance.DEFAULT_CHUNK_SIZE
    ) -> Iterator[NDArray[np.int64]]:
        """Iterate over elements in chunks.

        Yields
        ------
        ndarray
            Element connectivity arrays for each chunk. Each row contains:
            ``[elem_id, node1_id, node2_id, node3_id, node4_id]``
            For triangular elements, ``node4_id`` is set to 0.
        """
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:

            def _readline() -> str:
                return f.readline().split("!")[0].strip()

            # Skip header and nodes
            f.readline()
            f.readline()
            for _ in range(self.n_nodes):
                f.readline()

            elements_read = 0
            while elements_read < self.n_elements:
                current_chunk = min(chunk_size, self.n_elements - elements_read)

                elements = np.zeros((current_chunk, 5), dtype=np.int64)

                for i in range(current_chunk):
                    parts = _readline().split()
                    elem_id = int(parts[0])
                    elem_type = int(parts[1])

                    elements[i, 0] = elem_id
                    elements[i, 1] = int(parts[2])
                    elements[i, 2] = int(parts[3])
                    elements[i, 3] = int(parts[4])
                    elements[i, 4] = int(parts[5]) if elem_type == 4 else 0
                elements_read += current_chunk
                yield elements

    def iter_element_polys(
        self, chunk_size: int = Performance.DEFAULT_CHUNK_SIZE
    ) -> Iterator[list[Polygon]]:
        """Iterate over element polygons in chunks.

        Yields
        ------
        list of shapely.Polygon
            Shapely Polygons for each element chunk.
        """
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:

            def _readline() -> str:
                return f.readline().split("!")[0].strip()

            # Skip header and nodes
            f.readline()
            f.readline()
            for _ in range(self.n_nodes):
                f.readline()

            elements_read = 0
            while elements_read < self.n_elements:
                current_chunk = min(chunk_size, self.n_elements - elements_read)

                elements: list[shapely.Polygon] = []

                for _ in range(current_chunk):
                    parts = _readline().split()
                    elem_type = int(parts[1])
                    p_range = 6 if elem_type == 4 else 5
                    node_ids = [int(parts[j]) - 1 for j in range(2, p_range)]
                    elements.append(shapely.Polygon(self.nodes_coordinates[node_ids]))
                elements_read += current_chunk
                yield elements

    def read_bctides(self) -> tuple[str, str, str, list[tuple[int, int, int, int]]]:
        """Read bctides.in and return header and boundary flags.

        Returns
        -------
        tuple
            (header_line, ntip_line, nbfr_line, boundary_flags)
        """
        with self.bctides_file.open("r", buffering=self.buffer_size) as f:

            def _readline() -> str:
                return f.readline().split("!")[0].strip()

            # Read first three lines as-is
            header_line = f.readline().strip()
            ntip_line = _readline()
            nbfr_line = _readline()

            # Read nope line
            nope = int(_readline().split()[0])

            # Read boundary flags
            boundary_flags = []
            for _ in range(nope):
                parts = _readline().split()
                # First value is node count, rest are flags
                flags = [int(parts[i]) for i in range(1, len(parts))]
                boundary_flags.append(flags)

        return header_line, ntip_line, nbfr_line, boundary_flags

    def read_boundaries(self) -> BoundarySet:
        """Read boundary definitions from hgrid.gr3 and bctides.in.

        Returns
        -------
        BoundarySet
            BoundarySet with open and land boundaries and bctides data.
        """
        with self.hgrid_file.open("r", buffering=self.buffer_size) as f:

            def _readline(is_desc: bool = False) -> str:
                if is_desc:
                    return f.readline().split("!")[0].split(" = ")[0].strip()
                return f.readline().split("!")[0].strip()

            f.readline()
            f.readline()

            for _ in range(self.n_nodes):
                f.readline()
            for _ in range(self.n_elements):
                f.readline()

            f.readline()
            f.readline()

            open_boundaries: list[list[int]] = []
            for _ in range(self.n_open_bnds):
                n_nodes = int(_readline(True))
                boundary = [int(_readline().split()[0]) for _ in range(n_nodes)]
                open_boundaries.append(boundary)

            f.readline()
            f.readline()

            land_boundaries: list[LandBoundary] = []
            for _ in range(self.n_land_bnds):
                n_nodes, btype = _readline(True).split()
                boundary_nodes = []
                for _ in range(int(n_nodes)):
                    node_id = int(_readline(True).split()[0])
                    boundary_nodes.append(node_id)
                land_boundaries.append(
                    LandBoundary(nodes=boundary_nodes, boundary_type=BoundaryType(int(btype)))
                )

        bctides_header, ntip_line, nbfr_line, boundary_flags = self.read_bctides()

        return BoundarySet(
            open_boundaries,
            land_boundaries,
            bctides_header,
            ntip_line,
            nbfr_line,
            boundary_flags,
        )

    def coords_byid(self, index: list[int] | NDArray[np.int64]) -> NDArray[np.float64]:
        """Get coordinates for specified node IDs.

        Parameters
        ----------
        index : list of int or numpy.ndarray
            1-based node IDs (SCHISM convention).

        Returns
        -------
        numpy.ndarray
            Coordinates array with shape (N, 2).
        """
        to_0based = np.array(index, dtype=np.int64) - 1
        return self.nodes_coordinates[to_0based]
