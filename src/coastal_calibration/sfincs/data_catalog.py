"""HydroMT data catalog generation and NC symlink helpers for SFINCS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from coastal_calibration.config.schema import MeteoSource, PathConfig
from coastal_calibration.data.nwm_forcing import normalize_wrf_forcing
from coastal_calibration.logging import logger
from coastal_calibration.sfincs._hydromt_compat import apply_all_patches

if TYPE_CHECKING:
    import xarray as xr

    from coastal_calibration.config.schema import CoastalCalibConfig, SimulationConfig
    from coastal_calibration.data.downloader import CoastalSource

#: hydromt's ConventionResolver expands ``{year}``/``{month}`` across the
#: requested time range, so one entry covers a run of any length. Emitting one
#: entry per month instead would suffix the extras ``_1``, ``_2``, ... and every
#: consumer asks for the bare name, so those months were silently never loaded.
_MONTH_GLOB = "{year}{month:02d}"

#: Name under which :func:`_register_ldasin_preprocessor` registers our reader
#: hook with hydromt, and the value written as ``preprocess`` in the catalog.
LDASIN_PREPROCESSOR = "nwm_ldasin"


def _register_ldasin_preprocessor() -> None:
    """Teach hydromt how to read NWM LDASIN forcing.

    This is not a patch: :func:`normalize_wrf_forcing` supplies coordinates
    and timestamps that the PRVI and Alaska files simply do not carry, and no
    hydromt release can make that unnecessary. It lives here, beside the code
    that writes ``preprocess`` into the catalog, rather than in
    :mod:`coastal_calibration.sfincs._hydromt_compat`, so that module stays
    deletable.

    The rounding *is* a hydromt workaround and can go once upstream widens its
    regularity tolerance: NWM stores projected coordinates in meters with
    float error up to ~0.25 m, and hydromt's ``atol=5e-4`` rejects the grid as
    irregular. Coordinates rebuilt by ``normalize_wrf_forcing`` are already
    exact, so this only matters for the CF-layout CONUS files.
    """
    try:
        from hydromt.data_catalog.drivers.preprocessing import PREPROCESSORS
    except ImportError:
        return

    if LDASIN_PREPROCESSOR in PREPROCESSORS:
        return

    import numpy as np

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        ds = normalize_wrf_forcing(ds)
        x_dim, y_dim = ds.raster.x_dim, ds.raster.y_dim
        ds[x_dim] = np.round(ds[x_dim], decimals=0)
        ds[y_dim] = np.round(ds[y_dim], decimals=0)
        return ds

    PREPROCESSORS[LDASIN_PREPROCESSOR] = _preprocess
    logger.debug("Registered '%s' preprocessor in hydromt.", LDASIN_PREPROCESSOR)


apply_all_patches()
_register_ldasin_preprocessor()


DataType = Literal["RasterDataset", "GeoDataset", "GeoDataFrame", "DataFrame"]
Category = Literal[
    "geography",
    "topography",
    "hydrography",
    "meteo",
    "landuse",
    "ocean",
    "socio-economic",
    "observed data",
]


@dataclass
class DataAdapter:
    """HydroMT data adapter for variable harmonization.

    Parameters
    ----------
    rename : dict[str, str], optional
        Mapping from original variable names to HydroMT conventions.
    unit_mult : dict[str, float], optional
        Multiplication factors for unit conversion.
    unit_add : dict[str, float], optional
        Additive adjustments for unit conversion.
    """

    rename: dict[str, str] = field(default_factory=dict)
    unit_mult: dict[str, float] = field(default_factory=dict)
    unit_add: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding empty fields."""
        result = {}
        if self.rename:
            result["rename"] = self.rename
        if self.unit_mult:
            result["unit_mult"] = self.unit_mult
        if self.unit_add:
            result["unit_add"] = self.unit_add
        return result


@dataclass
class CatalogMetadata:
    """Metadata for a HydroMT data catalog entry.

    Parameters
    ----------
    crs : int or str, optional
        Coordinate reference system (e.g., 4326 for EPSG:4326).
    temporal_extent : tuple[str, str], optional
        Start and end dates as ISO format strings.
    spatial_extent : dict[str, float], optional
        Bounding box with keys: west, south, east, north.
    category : Category, optional
        Data category (geography, topography, hydrography, meteo, etc.).
    source_url : str, optional
        URL to the original data source.
    source_license : str, optional
        License of the data source.
    source_version : str, optional
        Version of the data source.
    paper_ref : str, optional
        Reference to a related publication.
    paper_doi : str, optional
        DOI of the related publication.
    notes : str, optional
        Additional notes about the dataset.
    """

    crs: int | str | None = None
    temporal_extent: tuple[str, str] | None = None
    spatial_extent: dict[str, float] | None = None
    category: Category | None = None
    source_url: str | None = None
    source_license: str | None = None
    source_version: str | None = None
    paper_ref: str | None = None
    paper_doi: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None fields."""
        result = {}
        if self.crs is not None:
            result["crs"] = self.crs
        if self.temporal_extent is not None:
            result["temporal_extent"] = list(self.temporal_extent)
        if self.spatial_extent is not None:
            result["spatial_extent"] = self.spatial_extent
        if self.category is not None:
            result["category"] = self.category
        if self.source_url is not None:
            result["source_url"] = self.source_url
        if self.source_license is not None:
            result["source_license"] = self.source_license
        if self.source_version is not None:
            result["source_version"] = self.source_version
        if self.paper_ref is not None:
            result["paper_ref"] = self.paper_ref
        if self.paper_doi is not None:
            result["paper_doi"] = self.paper_doi
        if self.notes is not None:
            result["notes"] = self.notes
        return result


@dataclass
class CatalogEntry:
    """A single entry in a HydroMT data catalog.

    Parameters
    ----------
    name : str
        Unique identifier for this dataset.
    data_type : DataType
        Format category (RasterDataset, GeoDataset, GeoDataFrame, DataFrame).
    driver : str or dict
        Driver for reading data (e.g., "netcdf", "zarr", "raster").
    uri : str
        URI pointing to where the data can be queried. Relative paths are combined
        with the global root option. Supports glob patterns like "path/to/*.nc".
    metadata : CatalogMetadata, optional
        Dataset metadata.
    data_adapter : DataAdapter, optional
        Variable harmonization configuration.
    version : str, optional
        Dataset version.
    """

    name: str
    data_type: DataType
    driver: str | dict[str, Any]
    uri: str
    metadata: CatalogMetadata | None = None
    data_adapter: DataAdapter | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            "data_type": self.data_type,
            "driver": self.driver,
            "uri": self.uri,
        }
        if self.metadata is not None:
            meta_dict = self.metadata.to_dict()
            if meta_dict:
                result["metadata"] = meta_dict
        if self.data_adapter is not None:
            adapter_dict = self.data_adapter.to_dict()
            if adapter_dict:
                result["data_adapter"] = adapter_dict
        if self.version is not None:
            result["version"] = self.version
        return result


@dataclass
class DataCatalog:
    """HydroMT data catalog container.

    Parameters
    ----------
    entries : list[CatalogEntry]
        List of catalog entries.
    name : str, optional
        Catalog identifier.
    version : str, optional
        Catalog version number.
    hydromt_version : str, optional
        Compatible HydroMT versions (PEP 440 format).
    roots : list[str], optional
        Root directories for relative paths.
    """

    entries: list[CatalogEntry] = field(default_factory=list)
    name: str | None = None
    version: str | None = None
    hydromt_version: str | None = None
    roots: list[str] | None = None

    def add_entry(self, entry: CatalogEntry) -> None:
        """Add an entry to the catalog."""
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        """Convert catalog to dictionary for YAML serialization."""
        result: dict[str, Any] = {}

        # Add global metadata if present
        meta: dict[str, Any] = {}
        if self.name is not None:
            meta["name"] = self.name
        if self.version is not None:
            meta["version"] = self.version
        if self.hydromt_version is not None:
            meta["hydromt_version"] = self.hydromt_version
        if self.roots is not None:
            meta["roots"] = self.roots
        if meta:
            result["meta"] = meta

        # Add entries
        for entry in self.entries:
            result[entry.name] = entry.to_dict()

        return result

    def to_yaml(self, path: Path | str) -> None:
        """Write catalog to YAML file.

        Parameters
        ----------
        path : Path or str
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                self.to_dict(),
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        )


# Variable mappings for NWM data to HydroMT conventions
NWM_METEO_RENAME = {
    "RAINRATE": "precip",
    "T2D": "temp",
    "Q2D": "humidity",
    "U2D": "wind10_u",
    "V2D": "wind10_v",
    "PSFC": "press_msl",
    "SWDOWN": "kin",
    "LWDOWN": "kout",
}

NWM_METEO_UNIT_MULT = {
    "precip": 3600.0,  # mm/s to mm/hr
}

NWM_METEO_UNIT_ADD = {
    "temp": -273.15,  # K to C
}


def _get_temporal_extent(
    sim: SimulationConfig,
) -> tuple[str, str]:
    """Get temporal extent from simulation config."""
    from datetime import timedelta

    start = sim.start_date
    end = start + timedelta(hours=sim.duration_hours)
    return (start.isoformat(), end.isoformat())


def _build_meteo_entry(
    sim: SimulationConfig,
    meteo_source: MeteoSource,
) -> list[CatalogEntry]:
    """Build catalog entries for meteorological forcing data (LDASIN).

    Returns one entry per month covered by the simulation window so
    that the glob is tightly scoped and stale files from other months
    are never loaded.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    meteo_source : MeteoSource
        Meteorological data source (nwm_retro or nwm_ana).

    Returns
    -------
    list[CatalogEntry]
        One catalog entry per simulation month.
    """
    temporal_extent = _get_temporal_extent(sim)

    # Both NWM Retrospective and Analysis LDASIN files use the same projected
    # grid (coordinates in meters) for a given domain, but that projection
    # differs between domains, so it must be looked up per coastal domain.
    crs = sim.meteo_crs

    # Determine source URL based on meteo source
    if meteo_source == "nwm_retro":
        source_url = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
        notes = "NWM Retrospective 3.0 LDASIN forcing files"
        source_version = "3.0"
    else:
        source_url = "https://storage.googleapis.com/national-water-model"
        notes = "NWM Analysis and Assimilation forcing files"
        source_version = "operational"

    metadata = CatalogMetadata(
        crs=crs,
        temporal_extent=temporal_extent,
        category="meteo",
        source_url=source_url,
        source_license="Public Domain",
        source_version=source_version,
        notes=notes,
    )

    data_adapter = DataAdapter(
        rename=NWM_METEO_RENAME,
        unit_mult=NWM_METEO_UNIT_MULT,
        unit_add=NWM_METEO_UNIT_ADD,
    )

    # ``_register_ldasin_preprocessor`` rebuilds the coordinates and
    # timestamps the raw WRF-layout files omit, then rounds x/y so hydromt
    # accepts the grid as regular.
    #
    # Each LDASIN file also carries a scalar ``reference_time`` coordinate
    # (model initialization time).  When ``open_mfdataset`` concatenates
    # the files, ``reference_time`` becomes a new dimension and inflates
    # the data to 4-D (reference_time, time, y, x).  hydromt only supports
    # 2-D/3-D arrays, so we drop it via ``drop_variables``.
    driver: dict[str, Any] = {
        "name": "raster_xarray",
        "options": {
            "preprocess": LDASIN_PREPROCESSOR,
            "drop_variables": ["reference_time", "crs"],
        },
    }

    meteo_subdir = PathConfig.meteo_subdir(meteo_source, sim.coastal_domain).as_posix()

    return [
        CatalogEntry(
            name=f"{meteo_source}_meteo",
            data_type="RasterDataset",
            driver=driver,
            uri=f"{meteo_subdir}/{_MONTH_GLOB}*.LDASIN_DOMAIN1.nc",
            metadata=metadata,
            data_adapter=data_adapter,
            version=temporal_extent[0][:10],
        )
    ]


def _read_forecast_crs(forecast_file: Path) -> str:
    """Read the projection of a forecast forcing file as a WKT string.

    The ngen forecast forcing engine writes the grid mapping into a
    ``crs`` variable (``spatial_ref``/``esri_pe_string`` WKT).  hydromt's
    raster accessor does not pick this up automatically, so the catalog
    entry must carry the crs explicitly.  Reading it from the file (rather
    than hardcoding a per-domain projection) keeps the entry correct for
    whatever domain the forecast engine produced.

    Raises
    ------
    ValueError
        If the file has no readable ``crs`` variable.
    """
    import xarray as xr
    from pyproj import CRS

    with xr.open_dataset(forecast_file) as ds:
        attrs = ds["crs"].attrs if "crs" in ds.variables else {}
        wkt = attrs.get("spatial_ref") or attrs.get("esri_pe_string")
    if not wkt:
        raise ValueError(
            f"Forecast meteo file has no readable 'crs' variable: {forecast_file}. "
            "Cannot determine the projection for the SFINCS data catalog."
        )
    return CRS.from_user_input(wkt).to_wkt()


def _build_forecast_meteo_entry(
    sim: SimulationConfig,
    forecast_file: Path,
) -> CatalogEntry:
    """Build the catalog entry for ngen forecast meteorological forcing.

    Unlike NWM LDASIN (one extension-less file per hour), the ngen
    forecast forcing engine emits a single multi-timestep netCDF on a
    projected (domain-specific LCC) grid.  The entry therefore points at
    that one file, carries the crs read from the file, and uses the
    ``forecast_meteo_coords`` preprocessor to promote the ``Time``
    variable to a ``time`` coordinate and regularize ``x``/``y``.  The
    variable names match LDASIN, so the same rename/unit adapters apply.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration (for the temporal extent).
    forecast_file : Path
        Path to the ngen forecast forcing file
        (``paths.forecast_meteo_file``).

    Returns
    -------
    CatalogEntry
        A single entry named ``<meteo_source>_meteo`` (matching the name
        the precip/wind/pressure stages resolve).
    """
    temporal_extent = _get_temporal_extent(sim)

    metadata = CatalogMetadata(
        crs=_read_forecast_crs(forecast_file),
        temporal_extent=temporal_extent,
        category="meteo",
        source_url="local ngen forecast forcing engine",
        source_license="Public Domain",
        source_version="ngen_forecast",
        notes="ngen forecast forcing engine output (single multi-timestep file)",
    )

    data_adapter = DataAdapter(
        rename=NWM_METEO_RENAME,
        unit_mult=NWM_METEO_UNIT_MULT,
        unit_add=NWM_METEO_UNIT_ADD,
    )

    # The datetime axis lives in a ``Time`` variable and the projected
    # coordinates carry sub-meter float noise; ``forecast_meteo_coords``
    # promotes ``Time`` -> ``time`` and rounds ``x``/``y``.  ``Time`` must
    # not be dropped at open time (the preprocessor needs it), so no
    # ``drop_variables`` here.
    driver: dict[str, Any] = {
        "name": "raster_xarray",
        "options": {"preprocess": "forecast_meteo_coords"},
    }

    return CatalogEntry(
        name=f"{sim.meteo_source}_meteo",
        data_type="RasterDataset",
        driver=driver,
        uri=str(forecast_file),
        metadata=metadata,
        data_adapter=data_adapter,
        version=temporal_extent[0][:10],
    )


def _build_streamflow_entry(
    sim: SimulationConfig,
    meteo_source: MeteoSource,
) -> list[CatalogEntry]:
    """Build catalog entries for downloaded streamflow data (CHRTOUT).

    Returns one entry per month covered by the simulation window, and
    nothing at all for ``nwm_retro``, whose streamflow is read straight
    from the S3 Zarr store by
    :func:`coastal_calibration.data.streamflow.read_streamflow` and never
    lands on disk.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    meteo_source : MeteoSource
        Meteorological data source (determines streamflow path).

    Returns
    -------
    list[CatalogEntry]
        One catalog entry per simulation month.
    """
    if meteo_source == "nwm_retro":
        return []

    subdir = PathConfig.streamflow_subdir(sim.coastal_domain).as_posix()
    source_url = "https://storage.googleapis.com/national-water-model"
    notes = "NWM Analysis channel_rt streamflow files"
    source_version = "operational"

    temporal_extent = _get_temporal_extent(sim)

    metadata = CatalogMetadata(
        crs=4326,
        temporal_extent=temporal_extent,
        category="hydrography",
        source_url=source_url,
        source_license="Public Domain",
        source_version=source_version,
        notes=notes,
    )

    data_adapter = DataAdapter(
        rename={
            "streamflow": "discharge",
            "q_lateral": "discharge_lateral",
        },
    )

    return [
        CatalogEntry(
            name=f"{meteo_source}_streamflow",
            data_type="GeoDataset",
            driver="geodataset_xarray",
            uri=f"{subdir}/{_MONTH_GLOB}*.CHRTOUT_DOMAIN1.nc",
            metadata=metadata,
            data_adapter=data_adapter,
            version=temporal_extent[0][:10],
        )
    ]


def _stofs_uri(sim: SimulationConfig) -> str:
    """Build the URI for the STOFS file matching this simulation.

    The path mirrors the layout produced by
    :func:`coastal_calibration.data.downloader.get_stofs_path`.  Using an
    exact path instead of a recursive glob (``stofs/**/*.fields.cwl.nc``)
    avoids picking up STOFS files from other simulations that may sit in
    the same shared download cache.  Different STOFS versions can have
    incompatible mesh dimensions (e.g. ``nbou``, ``node``), and xarray
    cannot concatenate files whose unindexed dimensions differ in size.
    """
    from datetime import datetime as _dt

    name_change_date = _dt(2023, 1, 8)
    start = sim.start_date
    product = "estofs" if start < name_change_date else "stofs_2d_glo"
    date_str = start.strftime("%Y%m%d")
    cycle_hour = (start.hour // 6) * 6
    hour_str = f"{cycle_hour:02d}"
    return (
        f"{PathConfig.COASTAL_SUBDIR}/stofs/"
        f"{product}.{date_str}/{product}.t{hour_str}z.fields.cwl.nc"
    )


def _build_coastal_stofs_entry(
    sim: SimulationConfig,
) -> CatalogEntry:
    """Build catalog entry for STOFS coastal water level data.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.

    Returns
    -------
    CatalogEntry
        Catalog entry for STOFS data.
    """
    # URI points to the specific file for *this* simulation, not a
    # recursive glob.  The shared download cache may contain STOFS
    # files from other runs whose mesh dimensions (``node``, ``nbou``,
    # ``nvel``) differ across STOFS versions, and xarray cannot
    # concatenate files with incompatible unindexed dimensions.
    uri = _stofs_uri(sim)

    temporal_extent = _get_temporal_extent(sim)

    metadata = CatalogMetadata(
        crs=4326,
        temporal_extent=temporal_extent,
        category="ocean",
        source_url="https://noaa-gestofs-pds.s3.amazonaws.com",
        source_license="Public Domain",
        source_version="operational",
        notes="STOFS 2D Global water level fields",
    )

    data_adapter = DataAdapter(
        rename={
            "zeta": "waterlevel",
            "cwl": "waterlevel",
        },
    )

    # Use a dict driver to pass ``drop_variables``.  The STOFS netCDF
    # (ADCIRC output) contains UGRID-like mesh topology variables
    # (``adcirc_mesh``, ``element``) that cause xugrid to fail because
    # ADCIRC uses 1-based face-node indexing while UGRID requires 0-based.
    # We also drop the scalar ``nvel`` which clashes with the ``nvel``
    # dimension.  Only the node coordinates (``x``, ``y``) and the water
    # level variable (``zeta``/``cwl``) are needed.
    #
    # The boundary-topology variables (``nvell``, ``ibtype``, ``nbvv``,
    # ``max_nvell``) and the bathymetry (``depth``) are also dropped
    # because they are unused and would inflate memory for the
    # ~12-million-node STOFS mesh.
    driver: dict[str, Any] = {
        "name": "geodataset_xarray",
        "options": {
            "drop_variables": [
                "nvel",
                "adcirc_mesh",
                "element",
                "mesh",
                "nvell",
                "ibtype",
                "nbvv",
                "max_nvell",
                "depth",
            ],
        },
    }

    return CatalogEntry(
        name="stofs_waterlevel",
        data_type="GeoDataset",
        driver=driver,
        uri=uri,
        metadata=metadata,
        data_adapter=data_adapter,
        version=temporal_extent[0][:10],
    )


def _build_coastal_glofs_entry(
    sim: SimulationConfig,
    glofs_model: str = "leofs",
) -> list[CatalogEntry]:
    """Build catalog entries for GLOFS coastal water level data.

    Returns one entry per month covered by the simulation window, so a
    download directory holding several runs of the same lake is not swept
    up wholesale. This mirrors the meteo and streamflow entries.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    glofs_model : str
        GLOFS model name (leofs, loofs, lsofs, lmhofs).

    Returns
    -------
    list[CatalogEntry]
        One catalog entry per simulation month.
    """
    temporal_extent = _get_temporal_extent(sim)

    metadata = CatalogMetadata(
        crs=4326,
        temporal_extent=temporal_extent,
        category="ocean",
        source_url="https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
        source_license="Public Domain",
        source_version="operational",
        notes=f"GLOFS {glofs_model.upper()} water level fields (Great Lakes)",
    )

    data_adapter = DataAdapter(
        rename={
            "zeta": "waterlevel",
        },
    )

    # GLOFS files: {model}.t{cycle}z.{YYYYMMDD}.fields.n{hour}.nc
    return [
        CatalogEntry(
            name=f"glofs_{glofs_model}_waterlevel",
            data_type="GeoDataset",
            driver="geodataset_xarray",
            uri=f"{PathConfig.COASTAL_SUBDIR}/glofs/{glofs_model}.*.{_MONTH_GLOB}*.fields.*.nc",
            metadata=metadata,
            data_adapter=data_adapter,
            version=temporal_extent[0][:10],
        )
    ]


def generate_data_catalog(
    config: CoastalCalibConfig,
    output_path: Path | str | None = None,
    *,
    catalog_name: str = "coastal_calibration",
    catalog_version: str = "1.0",
    hydromt_version: str = ">=0.9.0",
    include_meteo: bool = True,
    include_streamflow: bool = True,
    include_coastal: bool = True,
    coastal_source: CoastalSource | None = None,
    glofs_model: str = "leofs",
) -> DataCatalog:
    """Generate a HydroMT data catalog for downloaded coastal calibration data.

    Parameters
    ----------
    config : CoastalCalibConfig
        Coastal calibration configuration.
    output_path : Path or str, optional
        Path to write the catalog YAML file. If None, catalog is not written.
    catalog_name : str, optional
        Name identifier for the catalog. Default is "coastal_calibration".
    catalog_version : str, optional
        Version number for the catalog. Default is "1.0".
    hydromt_version : str, optional
        Compatible HydroMT version constraint (PEP 440). Default is ">=0.9.0".
    include_meteo : bool, optional
        Include meteorological forcing data entry. Default is True.
    include_streamflow : bool, optional
        Include streamflow data entry. Default is True.
    include_coastal : bool, optional
        Include coastal water level data entry. Default is True.
    coastal_source : CoastalSource, optional
        Coastal data source (stofs, glofs, harmonic). If None, uses config.boundary.source.
    glofs_model : str, optional
        GLOFS model name if using GLOFS coastal source. Default is "leofs".

    Returns
    -------
    DataCatalog
        The generated data catalog.

    Examples
    --------
    >>> from coastal_calibration import CoastalCalibConfig
    >>> config = CoastalCalibConfig.from_yaml("config.yaml")  # doctest: +SKIP
    >>> catalog = generate_data_catalog(config, "data_catalog.yml")  # doctest: +SKIP
    """
    download_dir = config.paths.download_dir.resolve()
    # The catalog root must exist for hydromt to load it. With ngen_forecast
    # meteo (external absolute-path file) and download disabled, download_dir
    # may not exist yet — create it so the produced catalog stays loadable.
    download_dir.mkdir(parents=True, exist_ok=True)
    sim = config.simulation
    meteo_source = sim.meteo_source
    effective_coastal_source = coastal_source or config.boundary.source

    catalog = DataCatalog(
        name=catalog_name,
        version=catalog_version,
        hydromt_version=hydromt_version,
        roots=[str(download_dir)],
    )

    if include_meteo:
        if meteo_source == "ngen_forecast":
            forecast_file = config.paths.forecast_meteo_file
            if forecast_file is None:
                raise ValueError(
                    "paths.forecast_meteo_file is required when "
                    "simulation.meteo_source is 'ngen_forecast'"
                )
            catalog.add_entry(_build_forecast_meteo_entry(sim, forecast_file))
        else:
            for entry in _build_meteo_entry(sim, meteo_source):
                catalog.add_entry(entry)

    # ngen forecast streamflow comes from t-route output, which is not wired
    # up yet — no streamflow catalog entry for it (mirrors the SCHISM side).
    if include_streamflow and meteo_source != "ngen_forecast":
        for entry in _build_streamflow_entry(sim, meteo_source):
            catalog.add_entry(entry)

    if include_coastal:
        # ``harmonic`` forcing is handled directly by SfincsForcingStage via
        # pyTMD, so it contributes no catalog entry.
        coastal_entries: list[CatalogEntry] = []
        if effective_coastal_source == "stofs":
            coastal_entries = [_build_coastal_stofs_entry(sim)]
        elif effective_coastal_source == "glofs":
            coastal_entries = _build_coastal_glofs_entry(sim, glofs_model)

        for entry in coastal_entries:
            catalog.add_entry(entry)

    if output_path is not None:
        catalog.to_yaml(output_path)

    return catalog


def _symlink_dir(directory: Path, glob_pattern: str, nc_suffix: str) -> tuple[list[Path], int]:
    """Create ``.nc`` symlinks in *directory* for files matching *glob_pattern*."""
    created: list[Path] = []
    n_existing = 0
    if not directory.exists():
        return created, n_existing
    for src in directory.glob(glob_pattern):
        dst = src.with_suffix(nc_suffix)
        if dst.is_symlink():
            n_existing += 1
        elif dst.exists():
            logger.warning("Non-symlink file exists at %s, skipping", dst)
        else:
            dst.symlink_to(src.name)
            created.append(dst)
    return created, n_existing


def create_nc_symlinks(
    download_dir: Path | str,
    *,
    meteo_source: MeteoSource = "nwm_retro",
    coastal_domain: str,
    include_meteo: bool = True,
    include_streamflow: bool = True,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Create .nc symlinks for NWM files to work around HydroMT extension check bug.

    HydroMT's raster_xarray driver has a bug where the `ext_override` option is not
    respected for netCDF files (only for zarr). This function creates symlinks with
    `.nc` extension pointing to the original NWM files.

    Parameters
    ----------
    download_dir : Path or str
        Root download directory containing meteo and streamflow subdirectories.
    meteo_source : MeteoSource, optional
        Meteorological data source (nwm_retro or nwm_ana). Default is "nwm_retro".
    coastal_domain : str
        Coastal domain, which selects the per-domain meteo subdirectory.
    include_meteo : bool, optional
        Create symlinks for LDASIN meteo files. Default is True.
    include_streamflow : bool, optional
        Create symlinks for CHRTOUT streamflow files. Default is True.

    Returns
    -------
    created : dict[str, list[Path]]
        Dictionary with keys "meteo" and "streamflow" containing lists of newly
        created symlink paths.
    existing : dict[str, int]
        Dictionary with keys "meteo" and "streamflow" containing counts of
        symlinks that already existed.

    Examples
    --------
    >>> from coastal_calibration.sfincs.data_catalog import create_nc_symlinks
    >>> symlinks = create_nc_symlinks("./data/downloads")  # doctest: +SKIP
    >>> print(f"Created {len(symlinks['meteo'])} meteo symlinks")  # doctest: +SKIP

    Notes
    -----
    This is a workaround for a HydroMT bug. See:
    https://github.com/Deltares/hydromt/issues/1361

    The symlinks are created in the same directory as the original files with
    the pattern: `{original_name}.nc` -> `{original_name}`
    """
    download_dir = Path(download_dir)
    created: dict[str, list[Path]] = {"meteo": [], "streamflow": []}
    existing: dict[str, int] = {"meteo": 0, "streamflow": 0}

    # ngen forecast forcing is a single already-".nc" file referenced by the
    # catalog directly, so there are no extension-less LDASIN files to link.
    if include_meteo and meteo_source != "ngen_forecast":
        meteo_dir = download_dir / PathConfig.meteo_subdir(meteo_source, coastal_domain)
        # Both nwm_retro and nwm_ana downloads use extension-less
        # YYYYMMDDHH.LDASIN_DOMAIN1 naming.  We create .nc symlinks to
        # work around a HydroMT ext_override bug.
        new, n_existing = _symlink_dir(meteo_dir, "*.LDASIN_DOMAIN1", ".LDASIN_DOMAIN1.nc")
        created["meteo"] = new
        existing["meteo"] = n_existing

    # nwm_retro streamflow is read from the S3 Zarr store, so only nwm_ana
    # ever puts CHRTOUT files on disk.
    if include_streamflow and meteo_source != "nwm_retro":
        streamflow_dir = download_dir / PathConfig.streamflow_subdir(coastal_domain)

        new, n_existing = _symlink_dir(streamflow_dir, "*.CHRTOUT_DOMAIN1", ".CHRTOUT_DOMAIN1.nc")
        created["streamflow"] = new
        existing["streamflow"] = n_existing

    return created, existing


def remove_nc_symlinks(
    download_dir: Path | str,
    *,
    meteo_source: MeteoSource = "nwm_retro",
    coastal_domain: str,
    include_meteo: bool = True,
    include_streamflow: bool = True,
) -> dict[str, int]:
    """Remove .nc symlinks created by create_nc_symlinks.

    Parameters
    ----------
    download_dir : Path or str
        Root download directory containing meteo and streamflow subdirectories.
    meteo_source : MeteoSource, optional
        Meteorological data source (nwm_retro or nwm_ana). Default is "nwm_retro".
    coastal_domain : str
        Coastal domain, which selects the per-domain meteo subdirectory.
    include_meteo : bool, optional
        Remove symlinks for LDASIN meteo files. Default is True.
    include_streamflow : bool, optional
        Remove symlinks for CHRTOUT streamflow files. Default is True.

    Returns
    -------
    dict[str, int]
        Dictionary with keys "meteo" and "streamflow" containing counts of removed
        symlinks.
    """
    download_dir = Path(download_dir)
    removed: dict[str, int] = {"meteo": 0, "streamflow": 0}

    if include_meteo and meteo_source != "ngen_forecast":
        meteo_dir = download_dir / PathConfig.meteo_subdir(meteo_source, coastal_domain)
        # Both sources use extension-less LDASIN_DOMAIN1 naming; remove
        # the .nc symlinks we created as a HydroMT workaround.
        if meteo_dir.exists():
            for link in meteo_dir.glob("*.LDASIN_DOMAIN1.nc"):
                if link.is_symlink():
                    link.unlink()
                    removed["meteo"] += 1

    if include_streamflow and meteo_source != "nwm_retro":
        streamflow_dir = download_dir / PathConfig.streamflow_subdir(coastal_domain)

        if streamflow_dir.exists():
            for link in streamflow_dir.glob("*.CHRTOUT_DOMAIN1.nc"):
                if link.is_symlink():
                    link.unlink()
                    removed["streamflow"] += 1

    return removed
