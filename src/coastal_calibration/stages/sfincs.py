"""HydroMT data catalog generation and NC symlink helpers for SFINCS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from coastal_calibration.config.schema import MeteoSource, PathConfig
from coastal_calibration.stages._hydromt_compat import apply_all_patches

apply_all_patches()

if TYPE_CHECKING:
    from coastal_calibration.config.schema import CoastalCalibConfig, SimulationConfig
    from coastal_calibration.downloader import CoastalSource

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
) -> CatalogEntry:
    """Build catalog entry for meteorological forcing data (LDASIN).

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    meteo_source : MeteoSource
        Meteorological data source (nwm_retro or nwm_ana).

    Returns
    -------
    CatalogEntry
        Catalog entry for meteo data.
    """
    # URI is relative to the root (download_dir).
    # Both nwm_retro and nwm_ana downloads use YYYYMMDDHH.LDASIN_DOMAIN1
    # naming (extension-less).  We create .nc symlinks to work around a
    # HydroMT ext_override bug.
    uri = f"{PathConfig.METEO_SUBDIR}/{meteo_source}/*.LDASIN_DOMAIN1.nc"

    temporal_extent = _get_temporal_extent(sim)

    # Both NWM Retrospective and Analysis LDASIN files use the same
    # Lambert Conformal Conic (LCC) projected grid with coordinates in meters.
    crs = "+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs=True"

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

    # NWM LDASIN files store projected (LCC) coordinates with floating-point
    # rounding errors up to ~0.125 m.  hydromt's raster accessor rejects
    # them as non-regular because its tolerance (atol=5e-4) is far too tight
    # for metre-scale coordinates.  The custom ``round_coords`` preprocessor
    # (registered in ``_hydromt_compat``) rounds x/y to the nearest integer,
    # making the grid perfectly regular.
    #
    # Each LDASIN file also carries a scalar ``reference_time`` coordinate
    # (model initialisation time).  When ``open_mfdataset`` concatenates
    # the files, ``reference_time`` becomes a new dimension and inflates
    # the data to 4-D (reference_time, time, y, x).  hydromt only supports
    # 2-D/3-D arrays, so we drop it via ``drop_variables``.
    driver: dict[str, Any] = {
        "name": "raster_xarray",
        "options": {
            "preprocess": "round_coords",
            "drop_variables": ["reference_time", "crs"],
        },
    }

    return CatalogEntry(
        name=f"{meteo_source}_meteo",
        data_type="RasterDataset",
        driver=driver,
        uri=uri,
        metadata=metadata,
        data_adapter=data_adapter,
        version=temporal_extent[0][:10],  # Use start date as version
    )


def _build_streamflow_entry(
    sim: SimulationConfig,
    meteo_source: MeteoSource,
) -> CatalogEntry:
    """Build catalog entry for streamflow data (CHRTOUT).

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    meteo_source : MeteoSource
        Meteorological data source (determines streamflow path).

    Returns
    -------
    CatalogEntry
        Catalog entry for streamflow data.
    """
    # URI is relative to the root (download_dir)
    # Use .nc symlinks to work around HydroMT ext_override bug
    if meteo_source == "nwm_retro":
        uri = f"{PathConfig.STREAMFLOW_SUBDIR}/nwm_retro/*.CHRTOUT_DOMAIN1.nc"
        source_url = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
        notes = "NWM Retrospective 3.0 CHRTOUT streamflow files"
        source_version = "3.0"
    else:
        uri = f"{PathConfig.HYDRO_SUBDIR}/nwm/*.CHRTOUT_DOMAIN1.nc"
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

    # Streamflow data adapter - rename to HydroMT conventions
    data_adapter = DataAdapter(
        rename={
            "streamflow": "discharge",
            "q_lateral": "discharge_lateral",
        },
    )

    return CatalogEntry(
        name=f"{meteo_source}_streamflow",
        data_type="GeoDataset",
        driver="geodataset_xarray",
        uri=uri,
        metadata=metadata,
        data_adapter=data_adapter,
        version=temporal_extent[0][:10],
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
    # URI is relative to the root (download_dir)
    # STOFS files have specific naming: estofs or stofs_2d_glo
    uri = f"{PathConfig.COASTAL_SUBDIR}/stofs/**/*.fields.cwl.nc"

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
    driver: dict[str, Any] = {
        "name": "geodataset_xarray",
        "options": {
            "drop_variables": [
                "nvel",
                "adcirc_mesh",
                "element",
                "mesh",
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
) -> CatalogEntry:
    """Build catalog entry for GLOFS coastal water level data.

    Parameters
    ----------
    sim : SimulationConfig
        Simulation configuration.
    glofs_model : str
        GLOFS model name (leofs, loofs, lsofs, lmhofs).

    Returns
    -------
    CatalogEntry
        Catalog entry for GLOFS data.
    """
    # URI is relative to the root (download_dir)
    # GLOFS files: {model}.t{cycle}z.{date}.fields.n{hour}.nc
    uri = f"{PathConfig.COASTAL_SUBDIR}/glofs/{glofs_model}.*.fields.*.nc"

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

    return CatalogEntry(
        name=f"glofs_{glofs_model}_waterlevel",
        data_type="GeoDataset",
        driver="geodataset_xarray",
        uri=uri,
        metadata=metadata,
        data_adapter=data_adapter,
        version=temporal_extent[0][:10],
    )


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
        Coastal data source (stofs, glofs, tpxo). If None, uses config.boundary.source.
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
        meteo_entry = _build_meteo_entry(sim, meteo_source)
        catalog.add_entry(meteo_entry)

    if include_streamflow:
        streamflow_entry = _build_streamflow_entry(sim, meteo_source)
        catalog.add_entry(streamflow_entry)

    if include_coastal:
        if effective_coastal_source == "stofs":
            coastal_entry = _build_coastal_stofs_entry(sim)
        elif effective_coastal_source == "glofs":
            coastal_entry = _build_coastal_glofs_entry(sim, glofs_model)
        elif effective_coastal_source == "tpxo":
            # TPXO forcing is handled directly by SfincsForcingStage
            # using predict_tide, not via the HydroMT data catalog.
            coastal_entry = None
        else:
            coastal_entry = None

        if coastal_entry is not None:
            catalog.add_entry(coastal_entry)

    if output_path is not None:
        catalog.to_yaml(output_path)

    return catalog


def create_nc_symlinks(
    download_dir: Path | str,
    *,
    meteo_source: MeteoSource = "nwm_retro",
    include_meteo: bool = True,
    include_streamflow: bool = True,
) -> dict[str, list[Path]]:
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
    include_meteo : bool, optional
        Create symlinks for LDASIN meteo files. Default is True.
    include_streamflow : bool, optional
        Create symlinks for CHRTOUT streamflow files. Default is True.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary with keys "meteo" and "streamflow" containing lists of created
        symlink paths.

    Examples
    --------
    >>> from coastal_calibration.stages.sfincs import create_nc_symlinks
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

    if include_meteo:
        meteo_dir = download_dir / PathConfig.METEO_SUBDIR / meteo_source
        # Both nwm_retro and nwm_ana downloads use extension-less
        # YYYYMMDDHH.LDASIN_DOMAIN1 naming.  We create .nc symlinks to
        # work around a HydroMT ext_override bug.
        if meteo_dir.exists():
            for src in meteo_dir.glob("*.LDASIN_DOMAIN1"):
                dst = src.with_suffix(".LDASIN_DOMAIN1.nc")
                if not dst.exists():
                    dst.symlink_to(src.name)
                    created["meteo"].append(dst)

    if include_streamflow:
        if meteo_source == "nwm_retro":
            streamflow_dir = download_dir / PathConfig.STREAMFLOW_SUBDIR / "nwm_retro"
        else:
            streamflow_dir = download_dir / PathConfig.HYDRO_SUBDIR / "nwm"

        if streamflow_dir.exists():
            for src in streamflow_dir.glob("*.CHRTOUT_DOMAIN1"):
                dst = src.with_suffix(".CHRTOUT_DOMAIN1.nc")
                if not dst.exists():
                    dst.symlink_to(src.name)
                    created["streamflow"].append(dst)

    return created


def remove_nc_symlinks(
    download_dir: Path | str,
    *,
    meteo_source: MeteoSource = "nwm_retro",
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

    if include_meteo:
        meteo_dir = download_dir / PathConfig.METEO_SUBDIR / meteo_source
        # Both sources use extension-less LDASIN_DOMAIN1 naming; remove
        # the .nc symlinks we created as a HydroMT workaround.
        if meteo_dir.exists():
            for link in meteo_dir.glob("*.LDASIN_DOMAIN1.nc"):
                if link.is_symlink():
                    link.unlink()
                    removed["meteo"] += 1

    if include_streamflow:
        if meteo_source == "nwm_retro":
            streamflow_dir = download_dir / PathConfig.STREAMFLOW_SUBDIR / "nwm_retro"
        else:
            streamflow_dir = download_dir / PathConfig.HYDRO_SUBDIR / "nwm"

        if streamflow_dir.exists():
            for link in streamflow_dir.glob("*.CHRTOUT_DOMAIN1.nc"):
                if link.is_symlink():
                    link.unlink()
                    removed["streamflow"] += 1

    return removed
