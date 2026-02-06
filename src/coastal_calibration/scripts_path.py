"""Utility module for resolving paths to bundled scripts.

This module provides functions to locate the bundled bash and Python scripts
that are distributed with the coastal_calibration package. This allows the
package to work without relying on external script dependencies on the cluster.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def get_scripts_dir() -> Path:
    """Get the path to the bundled scripts directory.

    Returns the absolute path to the scripts directory within the installed
    coastal_calibration package. This directory contains all bash scripts
    and Python modules needed for the workflow.

    Returns
    -------
    Path
        Absolute path to the scripts directory.
    """
    return Path(str(importlib.resources.files("coastal_calibration") / "scripts"))


def get_wrf_hydro_dir() -> Path:
    """Get the path to the bundled WRF-Hydro workflow scripts.

    Returns the absolute path to the wrf_hydro_workflow_dev directory
    containing the coastal and forcings modules.

    Returns
    -------
    Path
        Absolute path to the wrf_hydro_workflow_dev directory.
    """
    return get_scripts_dir() / "wrf_hydro_workflow_dev"


def get_tpxo_scripts_dir() -> Path:
    """Get the path to the bundled TPXO processing scripts.

    Returns the absolute path to the tpxo_to_open_bnds_hgrid directory
    containing Python scripts for TPXO tide processing.

    Returns
    -------
    Path
        Absolute path to the tpxo_to_open_bnds_hgrid directory.
    """
    return get_scripts_dir() / "tpxo_to_open_bnds_hgrid"


def get_coastal_scripts_dir() -> Path:
    """Get the path to the bundled coastal workflow scripts.

    Returns the absolute path to the coastal subdirectory within
    wrf_hydro_workflow_dev, containing scripts like makeAtmo.py,
    regrid_estofs.py, and correct_elevation.py.

    Returns
    -------
    Path
        Absolute path to the coastal scripts directory.
    """
    return get_wrf_hydro_dir() / "coastal"


def get_forcings_dir() -> Path:
    """Get the path to the bundled forcing generation scripts.

    Returns the absolute path to the forcings subdirectory within
    wrf_hydro_workflow_dev, containing the WrfHydroFECPP workflow driver.

    Returns
    -------
    Path
        Absolute path to the forcings directory.
    """
    return get_wrf_hydro_dir() / "forcings"


def get_script_environment_vars() -> dict[str, str]:
    """Get environment variables for bundled script paths.

    Returns a dictionary of environment variables that should be set
    to allow bash scripts to find the bundled Python scripts.

    The variables are:
    - SCRIPTS_DIR: Root scripts directory
    - WRF_HYDRO_DIR: wrf_hydro_workflow_dev directory
    - TPXO_SCRIPTS_DIR: tpxo_to_open_bnds_hgrid directory
    - COASTAL_SCRIPTS_DIR: coastal scripts directory
    - FORCINGS_SCRIPTS_DIR: forcings scripts directory

    Returns
    -------
    dict[str, str]
        Dictionary mapping environment variable names to paths.
    """
    return {
        "SCRIPTS_DIR": str(get_scripts_dir()),
        "WRF_HYDRO_DIR": str(get_wrf_hydro_dir()),
        "TPXO_SCRIPTS_DIR": str(get_tpxo_scripts_dir()),
        "COASTAL_SCRIPTS_DIR": str(get_coastal_scripts_dir()),
        "FORCINGS_SCRIPTS_DIR": str(get_forcings_dir()),
    }
