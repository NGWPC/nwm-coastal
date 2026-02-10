"""Tests for coastal_calibration.config.schema module."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    MonitoringConfig,
    PathConfig,
    SchismModelConfig,
    SfincsModelConfig,
    SimulationConfig,
    SlurmConfig,
    _build_interpolation_context,
    _deep_merge,
    _interpolate_config,
    _interpolate_value,
)
from coastal_calibration.utils.system import get_cpu_count
from coastal_calibration.utils.time import parse_datetime as _parse_datetime


class TestSlurmConfig:
    def test_defaults(self):
        cfg = SlurmConfig()
        assert cfg.job_name == "coastal_calibration"
        assert cfg.user is None

    def test_custom_values(self):
        cfg = SlurmConfig(
            job_name="my_job",
            partition="gpu",
            user="alice",
            time_limit="12:00:00",
            account="my_account",
            qos="normal",
        )
        assert cfg.time_limit == "12:00:00"
        assert cfg.user == "alice"


class TestSimulationConfig:
    def test_start_properties(self, sample_simulation_config):
        sim = sample_simulation_config
        assert sim.start_pdy == "20210611"
        assert sim.start_cyc == "00"

    def test_inland_domain_mapping(self):
        for domain, expected in [
            ("prvi", "domain_puertorico"),
            ("hawaii", "domain_hawaii"),
            ("atlgulf", "domain"),
            ("pacific", "domain"),
        ]:
            sim = SimulationConfig(
                start_date=datetime(2021, 1, 1),
                duration_hours=3,
                coastal_domain=domain,
                meteo_source="nwm_retro",
            )
            assert sim.inland_domain == expected

    def test_nwm_domain_mapping(self):
        for domain, expected in [
            ("prvi", "prvi"),
            ("hawaii", "hawaii"),
            ("atlgulf", "conus"),
            ("pacific", "conus"),
        ]:
            sim = SimulationConfig(
                start_date=datetime(2021, 1, 1),
                duration_hours=3,
                coastal_domain=domain,
                meteo_source="nwm_retro",
            )
            assert sim.nwm_domain == expected

    def test_geo_grid_mapping(self):
        for domain, expected in [
            ("prvi", "geo_em_PRVI.nc"),
            ("hawaii", "geo_em_HI.nc"),
            ("atlgulf", "geo_em_CONUS.nc"),
            ("pacific", "geo_em_CONUS.nc"),
        ]:
            sim = SimulationConfig(
                start_date=datetime(2021, 1, 1),
                duration_hours=3,
                coastal_domain=domain,
                meteo_source="nwm_retro",
            )
            assert sim.geo_grid == expected

    def test_default_timestep(self):
        sim = SimulationConfig(
            start_date=datetime(2021, 1, 1),
            duration_hours=3,
            coastal_domain="pacific",
            meteo_source="nwm_retro",
        )
        assert sim.timestep_seconds == 3600


class TestBoundaryConfig:
    def test_defaults(self):
        cfg = BoundaryConfig()
        assert cfg.source == "tpxo"
        assert cfg.stofs_file is None

    def test_stofs_source(self):
        cfg = BoundaryConfig(source="stofs", stofs_file=Path("/tmp/stofs.nc"))
        assert cfg.source == "stofs"
        assert cfg.stofs_file == Path("/tmp/stofs.nc")


class TestPathConfig:
    def test_post_init_converts_to_path(self):
        cfg = PathConfig(work_dir="/tmp/work")
        assert isinstance(cfg.work_dir, Path)

    def test_download_dir_fallback(self, tmp_work_dir):
        cfg = PathConfig(work_dir=tmp_work_dir)
        assert cfg.download_dir == tmp_work_dir / "downloads"

    def test_download_dir_explicit(self, tmp_work_dir, tmp_download_dir):
        cfg = PathConfig(work_dir=tmp_work_dir, raw_download_dir=tmp_download_dir)
        assert cfg.download_dir == tmp_download_dir

    def test_otps_dir(self, tmp_work_dir):
        cfg = PathConfig(work_dir=tmp_work_dir)
        assert "OTPSnc" in str(cfg.otps_dir)

    def test_meteo_dir(self, tmp_work_dir, tmp_download_dir):
        cfg = PathConfig(work_dir=tmp_work_dir, raw_download_dir=tmp_download_dir)
        assert cfg.meteo_dir("nwm_retro") == tmp_download_dir / "meteo" / "nwm_retro"

    def test_streamflow_dir_retro(self, tmp_work_dir, tmp_download_dir):
        cfg = PathConfig(work_dir=tmp_work_dir, raw_download_dir=tmp_download_dir)
        assert cfg.streamflow_dir("nwm_retro") == tmp_download_dir / "streamflow" / "nwm_retro"

    def test_streamflow_dir_ana(self, tmp_work_dir, tmp_download_dir):
        cfg = PathConfig(work_dir=tmp_work_dir, raw_download_dir=tmp_download_dir)
        assert cfg.streamflow_dir("nwm_ana") == tmp_download_dir / "hydro" / "nwm"

    def test_coastal_dir(self, tmp_work_dir, tmp_download_dir):
        cfg = PathConfig(work_dir=tmp_work_dir, raw_download_dir=tmp_download_dir)
        assert cfg.coastal_dir("stofs") == tmp_download_dir / "coastal" / "stofs"

    def test_derived_paths(self, tmp_work_dir):
        cfg = PathConfig(work_dir=tmp_work_dir)
        assert "nwm.v3.0.6" in str(cfg.nwm_version_dir)
        assert "ush" in str(cfg.ush_nwm)
        assert "exec" in str(cfg.exec_nwm)
        assert "parm" in str(cfg.parm_nwm)

    def test_geogrid_file(self, tmp_work_dir, sample_simulation_config):
        cfg = PathConfig(work_dir=tmp_work_dir)
        geogrid = cfg.geogrid_file(sample_simulation_config)
        assert "geo_em_CONUS.nc" in str(geogrid)


class TestSchismModelConfig:
    def test_defaults(self):
        cfg = SchismModelConfig()
        assert cfg.nodes == 2
        assert cfg.ntasks_per_node == 18
        assert cfg.exclusive is True
        assert cfg.nscribes == 2
        assert cfg.omp_num_threads == 2
        assert cfg.oversubscribe is False
        assert cfg.include_noaa_gages is False

    def test_total_tasks(self):
        cfg = SchismModelConfig(nodes=3, ntasks_per_node=10)
        assert cfg.total_tasks == 30

    def test_model_name(self):
        cfg = SchismModelConfig()
        assert cfg.model_name == "schism"

    def test_stage_order(self):
        cfg = SchismModelConfig()
        expected = [
            "download",
            "pre_forcing",
            "nwm_forcing",
            "post_forcing",
            "update_params",
            "boundary_conditions",
            "pre_schism",
            "schism_run",
            "post_schism",
        ]
        assert cfg.stage_order == expected

    def test_schism_mesh(self, sample_simulation_config, tmp_work_dir):
        cfg = SchismModelConfig()
        paths = PathConfig(work_dir=tmp_work_dir)
        mesh = cfg.schism_mesh(sample_simulation_config, paths)
        assert "hgrid.nc" in str(mesh)
        assert "pacific" in str(mesh)

    def test_to_dict(self):
        cfg = SchismModelConfig()
        d = cfg.to_dict()
        assert d["nodes"] == 2
        assert d["ntasks_per_node"] == 18
        assert d["nscribes"] == 2
        assert "binary" in d
        assert d["include_noaa_gages"] is False

    def test_generate_job_script_lines(self, sample_config):
        cfg = SchismModelConfig()
        lines = cfg.generate_job_script_lines(sample_config)
        assert any("-N 2" in line for line in lines)
        assert any("ntasks-per-node=18" in line for line in lines)
        assert any("exclusive" in line for line in lines)


class TestSfincsModelConfig:
    def test_defaults(self, tmp_path):
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path)
        assert cfg.omp_num_threads == get_cpu_count()
        assert cfg.container_tag == "latest"
        assert cfg.merge_observations is False
        assert cfg.merge_discharge is False
        assert cfg.include_noaa_gages is False

    def test_model_name(self, tmp_path):
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path)
        assert cfg.model_name == "sfincs"

    def test_stage_order(self, tmp_path):
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path)
        expected = [
            "download",
            "sfincs_symlinks",
            "sfincs_data_catalog",
            "sfincs_init",
            "sfincs_timing",
            "sfincs_forcing",
            "sfincs_obs",
            "sfincs_discharge",
            "sfincs_precip",
            "sfincs_write",
            "sfincs_run",
        ]
        assert cfg.stage_order == expected

    def test_to_dict(self, tmp_path):
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path)
        d = cfg.to_dict()
        assert d["prebuilt_dir"] == str(tmp_path)
        assert d["omp_num_threads"] == get_cpu_count()
        assert "container_tag" in d
        assert d["include_noaa_gages"] is False

    def test_explicit_omp_num_threads(self, tmp_path):
        """Explicit omp_num_threads is preserved (e.g., cluster YAML)."""
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path, omp_num_threads=36)
        assert cfg.omp_num_threads == 36

    def test_generate_job_script_lines(self, tmp_path, sample_config):
        cfg = SfincsModelConfig(prebuilt_dir=tmp_path)
        lines = cfg.generate_job_script_lines(sample_config)
        assert any("-N 1" in line for line in lines)
        assert any("ntasks=1" in line for line in lines)
        assert any("cpus-per-task" in line for line in lines)


class TestMonitoringConfig:
    def test_defaults(self):
        cfg = MonitoringConfig()
        assert cfg.log_level == "INFO"
        assert cfg.log_file is None
        assert cfg.enable_progress_tracking is True
        assert cfg.enable_timing is True


class TestDownloadConfig:
    def test_defaults(self):
        cfg = DownloadConfig()
        assert cfg.enabled is True
        assert cfg.skip_existing is True
        assert cfg.timeout == 600
        assert cfg.raise_on_error is True


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_replaces_non_dict(self):
        base = {"a": {"x": 1}}
        override = {"a": "string"}
        result = _deep_merge(base, override)
        assert result == {"a": "string"}


class TestParseDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2021, 6, 11, 12, 0, 0)
        assert _parse_datetime(dt) == dt

    def test_date_to_datetime(self):
        d = date(2021, 6, 11)
        result = _parse_datetime(d)
        assert result == datetime(2021, 6, 11)

    def test_iso_format_date(self):
        assert _parse_datetime("2021-06-11") == datetime(2021, 6, 11)

    def test_iso_format_datetime(self):
        assert _parse_datetime("2021-06-11T12:00:00") == datetime(2021, 6, 11, 12)

    def test_compact_date(self):
        assert _parse_datetime("20210611") == datetime(2021, 6, 11)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            _parse_datetime("not-a-date")


class TestInterpolation:
    def test_interpolate_value(self):
        ctx = {"slurm.user": "john", "simulation.coastal_domain": "hawaii"}
        result = _interpolate_value("/data/${slurm.user}/${simulation.coastal_domain}", ctx)
        assert result == "/data/john/hawaii"

    def test_interpolate_value_unresolved(self):
        ctx = {"slurm.user": "john"}
        result = _interpolate_value("/data/${slurm.user}/${missing.key}", ctx)
        assert result == "/data/john/${missing.key}"

    def test_interpolate_non_string(self):
        assert _interpolate_value(42, {}) == 42
        assert _interpolate_value(None, {}) is None

    def test_build_interpolation_context(self):
        data = {
            "slurm": {"user": "john", "partition": "default"},
            "simulation": {"coastal_domain": "hawaii"},
        }
        ctx = _build_interpolation_context(data)
        assert ctx["slurm.user"] == "john"
        assert ctx["simulation.coastal_domain"] == "hawaii"

    def test_interpolate_config(self):
        data = {
            "slurm": {"user": "john"},
            "simulation": {"coastal_domain": "hawaii"},
            "paths": {"work_dir": "/data/${slurm.user}/${simulation.coastal_domain}"},
        }
        result = _interpolate_config(data)
        assert result["paths"]["work_dir"] == "/data/john/hawaii"

    def test_model_variable_interpolation(self):
        data = {
            "model": "sfincs",
            "slurm": {"user": "john"},
            "simulation": {"coastal_domain": "hawaii"},
            "paths": {"work_dir": "/data/${model}/${slurm.user}"},
        }
        result = _interpolate_config(data)
        assert result["paths"]["work_dir"] == "/data/sfincs/john"


class TestCoastalCalibConfig:
    def test_from_yaml(self, minimal_config_yaml):
        cfg = CoastalCalibConfig.from_yaml(minimal_config_yaml)
        assert cfg.slurm.user == "testuser"
        assert cfg.simulation.coastal_domain == "pacific"
        assert cfg.simulation.duration_hours == 3

    def test_from_yaml_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CoastalCalibConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_empty(self, tmp_path):
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            CoastalCalibConfig.from_yaml(empty_path)

    def test_to_dict(self, sample_config):
        d = sample_config.to_dict()
        assert "slurm" in d
        assert "simulation" in d
        assert "boundary" in d
        assert "paths" in d
        assert "model_config" in d
        assert "model" in d
        assert "monitoring" in d
        assert "download" in d

    def test_model_property(self, sample_config):
        assert sample_config.model == "schism"

    def test_to_yaml_and_back(self, sample_config, tmp_path):
        yaml_path = tmp_path / "roundtrip.yaml"
        sample_config.to_yaml(yaml_path)
        assert yaml_path.exists()
        loaded = CoastalCalibConfig.from_yaml(yaml_path)
        assert loaded.slurm.user == sample_config.slurm.user
        assert loaded.simulation.duration_hours == sample_config.simulation.duration_hours
        assert loaded.simulation.coastal_domain == sample_config.simulation.coastal_domain

    def test_yaml_inheritance(self, tmp_path, minimal_config_dict):
        # Write base config
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.dump(minimal_config_dict))

        # Write child config that overrides duration_hours
        child_dict = {
            "_base": str(base_path),
            "simulation": {"duration_hours": 12},
        }
        child_path = tmp_path / "child.yaml"
        child_path.write_text(yaml.dump(child_dict))

        cfg = CoastalCalibConfig.from_yaml(child_path)
        assert cfg.simulation.duration_hours == 12
        assert cfg.slurm.user == "testuser"  # inherited

    def test_validate_positive_duration(self, sample_config):
        sample_config.simulation.duration_hours = 0
        errors = sample_config.validate()
        assert any("duration_hours" in e for e in errors)

    def test_validate_user_required(self, sample_config):
        sample_config.slurm.user = None
        errors = sample_config.validate()
        assert any("user" in e for e in errors)

    def test_validate_nodes_positive(self, sample_config):
        """SchismModelConfig validates nodes must be at least 1."""
        assert isinstance(sample_config.model_config, SchismModelConfig)
        sample_config.model_config.nodes = 0
        errors = sample_config.validate()
        assert any("nodes" in e for e in errors)

    def test_validate_nscribes_less_than_total(self, sample_config):
        assert isinstance(sample_config.model_config, SchismModelConfig)
        sample_config.model_config.nscribes = sample_config.model_config.total_tasks
        errors = sample_config.validate()
        assert any("nscribes" in e for e in errors)

    def test_validate_raw_download_dir_required(self, sample_config):
        sample_config.paths.raw_download_dir = None
        errors = sample_config.validate()
        assert any("raw_download_dir" in e for e in errors)

    def test_to_dict_boundary_stofs_file(self, sample_config):
        sample_config.boundary.stofs_file = Path("/tmp/stofs.nc")
        d = sample_config.to_dict()
        assert d["boundary"]["stofs_file"] == "/tmp/stofs.nc"

    def test_to_dict_boundary_stofs_file_none(self, sample_config):
        d = sample_config.to_dict()
        assert d["boundary"]["stofs_file"] is None

    def test_model_registry_dispatch_schism(self, tmp_path):
        """YAML with model: schism creates SchismModelConfig."""
        config_dict = {
            "model": "schism",
            "slurm": {"user": "test"},
            "simulation": {
                "start_date": "2021-06-11",
                "duration_hours": 3,
                "coastal_domain": "pacific",
                "meteo_source": "nwm_retro",
            },
            "boundary": {"source": "tpxo"},
            "paths": {
                "work_dir": str(tmp_path / "work"),
                "raw_download_dir": str(tmp_path / "dl"),
            },
        }
        config_path = tmp_path / "schism.yaml"
        config_path.write_text(yaml.dump(config_dict))
        cfg = CoastalCalibConfig.from_yaml(config_path)
        assert isinstance(cfg.model_config, SchismModelConfig)
        assert cfg.model == "schism"

    def test_model_registry_dispatch_sfincs(self, tmp_path):
        """YAML with model: sfincs creates SfincsModelConfig."""
        config_dict = {
            "model": "sfincs",
            "slurm": {"user": "test"},
            "simulation": {
                "start_date": "2021-06-11",
                "duration_hours": 3,
                "coastal_domain": "pacific",
                "meteo_source": "nwm_retro",
            },
            "boundary": {"source": "tpxo"},
            "paths": {
                "work_dir": str(tmp_path / "work"),
                "raw_download_dir": str(tmp_path / "dl"),
            },
            "model_config": {
                "prebuilt_dir": str(tmp_path / "prebuilt"),
            },
        }
        config_path = tmp_path / "sfincs.yaml"
        config_path.write_text(yaml.dump(config_dict))
        cfg = CoastalCalibConfig.from_yaml(config_path)
        assert isinstance(cfg.model_config, SfincsModelConfig)
        assert cfg.model == "sfincs"

    def test_unknown_model_type(self, tmp_path):
        """Unknown model type raises ValueError."""
        config_dict = {
            "model": "unknown_model",
            "slurm": {"user": "test"},
            "simulation": {
                "start_date": "2021-06-11",
                "duration_hours": 3,
                "coastal_domain": "pacific",
                "meteo_source": "nwm_retro",
            },
            "boundary": {"source": "tpxo"},
            "paths": {
                "work_dir": str(tmp_path / "work"),
                "raw_download_dir": str(tmp_path / "dl"),
            },
        }
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump(config_dict))
        with pytest.raises(ValueError, match="Unknown model type"):
            CoastalCalibConfig.from_yaml(config_path)
