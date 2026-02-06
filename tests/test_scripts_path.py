"""Tests for coastal_calibration.scripts_path module."""

from __future__ import annotations

from pathlib import Path

from coastal_calibration.scripts_path import (
    get_coastal_scripts_dir,
    get_forcings_dir,
    get_script_environment_vars,
    get_scripts_dir,
    get_tpxo_scripts_dir,
    get_wrf_hydro_dir,
)


class TestGetScriptsDir:
    def test_returns_path(self):
        result = get_scripts_dir()
        assert isinstance(result, Path)
        assert "scripts" in str(result)

    def test_path_exists(self):
        result = get_scripts_dir()
        assert result.exists()


class TestGetWrfHydroDir:
    def test_returns_path(self):
        result = get_wrf_hydro_dir()
        assert isinstance(result, Path)
        assert "wrf_hydro_workflow_dev" in str(result)


class TestGetTpxoScriptsDir:
    def test_returns_path(self):
        result = get_tpxo_scripts_dir()
        assert isinstance(result, Path)
        assert "tpxo_to_open_bnds_hgrid" in str(result)


class TestGetCoastalScriptsDir:
    def test_returns_path(self):
        result = get_coastal_scripts_dir()
        assert isinstance(result, Path)
        assert "coastal" in str(result)


class TestGetForcingsDir:
    def test_returns_path(self):
        result = get_forcings_dir()
        assert isinstance(result, Path)
        assert "forcings" in str(result)


class TestGetScriptEnvironmentVars:
    def test_returns_dict(self):
        result = get_script_environment_vars()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_script_environment_vars()
        expected_keys = [
            "SCRIPTS_DIR",
            "WRF_HYDRO_DIR",
            "TPXO_SCRIPTS_DIR",
            "COASTAL_SCRIPTS_DIR",
            "FORCINGS_SCRIPTS_DIR",
        ]
        for key in expected_keys:
            assert key in result

    def test_all_values_are_strings(self):
        result = get_script_environment_vars()
        for value in result.values():
            assert isinstance(value, str)
