"""Shared fixtures for coastal_calibration tests."""

from __future__ import annotations

from datetime import datetime

import pytest
import yaml

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    MonitoringConfig,
    PathConfig,
    SchismModelConfig,
    SimulationConfig,
    SlurmConfig,
)


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Create a temporary work directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return work_dir


@pytest.fixture
def tmp_download_dir(tmp_path):
    """Create a temporary download directory."""
    dl_dir = tmp_path / "downloads"
    dl_dir.mkdir()
    return dl_dir


@pytest.fixture
def sample_slurm_config():
    """Create a sample SlurmConfig."""
    return SlurmConfig(
        job_name="test_job",
        partition="test-partition",
        user="testuser",
    )


@pytest.fixture
def sample_simulation_config():
    """Create a sample SimulationConfig."""
    return SimulationConfig(
        start_date=datetime(2021, 6, 11, 0, 0, 0),
        duration_hours=3,
        coastal_domain="pacific",
        meteo_source="nwm_retro",
    )


@pytest.fixture
def sample_boundary_config():
    """Create a sample BoundaryConfig."""
    return BoundaryConfig(source="tpxo")


@pytest.fixture
def sample_path_config(tmp_work_dir, tmp_download_dir):
    """Create a sample PathConfig with temp directories."""
    return PathConfig(
        work_dir=tmp_work_dir,
        raw_download_dir=tmp_download_dir,
    )


@pytest.fixture
def sample_config(
    sample_slurm_config,
    sample_simulation_config,
    sample_boundary_config,
    sample_path_config,
):
    """Create a complete sample CoastalCalibConfig."""
    return CoastalCalibConfig(
        slurm=sample_slurm_config,
        simulation=sample_simulation_config,
        boundary=sample_boundary_config,
        paths=sample_path_config,
        model_config=SchismModelConfig(),
        monitoring=MonitoringConfig(),
        download=DownloadConfig(enabled=False),
    )


@pytest.fixture
def sample_config_yaml(tmp_path, sample_config):
    """Write a sample config to YAML and return the path."""
    config_path = tmp_path / "config.yaml"
    sample_config.to_yaml(config_path)
    return config_path


@pytest.fixture
def minimal_config_dict(tmp_work_dir, tmp_download_dir):
    """Return a minimal config dictionary."""
    return {
        "model": "schism",
        "slurm": {"user": "testuser"},
        "simulation": {
            "start_date": "2021-06-11",
            "duration_hours": 3,
            "coastal_domain": "pacific",
            "meteo_source": "nwm_retro",
        },
        "boundary": {"source": "tpxo"},
        "paths": {
            "work_dir": str(tmp_work_dir),
            "raw_download_dir": str(tmp_download_dir),
        },
    }


@pytest.fixture
def minimal_config_yaml(tmp_path, minimal_config_dict):
    """Write a minimal config dict to YAML and return the path."""
    config_path = tmp_path / "minimal_config.yaml"
    config_path.write_text(yaml.dump(minimal_config_dict))
    return config_path
