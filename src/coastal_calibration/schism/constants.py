"""Constants used throughout schism_subsetter."""

from __future__ import annotations


class Performance:
    """Performance and memory-related constants."""

    DEFAULT_CHUNK_SIZE = 10000
    DEFAULT_BUFFER_SIZE = 1024 * 1024  # 1 MB
    MAX_TEMP_FILE_CLEANUP_RETRIES = 3
    TEMP_FILE_CLEANUP_RETRY_DELAY = 0.1  # seconds


class Size:
    """Size conversion constants."""

    GB = 1024 * 1024 * 1024
    MB = 1024 * 1024
    KB = 1024


class GeoSpatial:
    """Geospatial calculation constants."""

    LINE_DENSIFICATION_TARGET_M = 1000  # meters
    APPROX_METERS_PER_DEGREE = 111000


class SCHISMFiles:
    """Standard NWM-SCHISM filenames."""

    HGRID_GR3 = "hgrid.gr3"
    HGRID_LL = "hgrid.ll"
    HGRID_CPP = "hgrid.cpp"
    HGRID_NC = "hgrid.nc"
    OPEN_BOUNDS_NC = "open_bnds_hgrid.nc"
    MANNING = "manning.gr3"
    WINDROT = "windrot_geo2proj.gr3"
    ELEM_AREA = "element_areas.txt"
    ELEV_IC = "elev.ic"
    ELEV_CORRECTION = "elevation_correction.csv"
    NWM_REACHES = "nwmReaches.csv"
    BCTIDES = "bctides.in"
    NODE_MAPPING = "node_mapping.txt"
