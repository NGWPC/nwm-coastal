"""Tests for coastal_calibration.stages module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    MonitoringConfig,
    MPIConfig,
    PathConfig,
    SimulationConfig,
    SlurmConfig,
)
from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.boundary import (
    BoundaryConditionStage,
    STOFSBoundaryStage,
    UpdateParamsStage,
)
from coastal_calibration.stages.download import DownloadStage
from coastal_calibration.stages.forcing import (
    NWMForcingStage,
    PostForcingStage,
    PreForcingStage,
)
from coastal_calibration.stages.schism import (
    PostSCHISMStage,
    PreSCHISMStage,
    SCHISMRunStage,
)
from coastal_calibration.utils.logging import WorkflowMonitor


class TestWorkflowStageBase:
    def test_abstract_cant_instantiate(self):
        with pytest.raises(TypeError):
            WorkflowStage(None, None)

    def test_build_date_env(self, sample_config):
        """Test that _build_date_env sets correct environment variables."""
        # Create a concrete subclass for testing
        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        env = {}
        stage._build_date_env(env)

        assert "FORCING_BEGIN_DATE" in env
        assert "FORCING_END_DATE" in env
        assert "SCHISM_BEGIN_DATE" in env
        assert "SCHISM_END_DATE" in env
        assert "END_DATETIME" in env
        assert "PDY" in env
        assert "cyc" in env
        assert "start_dt" in env
        assert "end_dt" in env
        assert "FORCING_START_YEAR" in env
        assert "FORCING_START_MONTH" in env
        assert "FORCING_START_DAY" in env
        assert "FORCING_START_HOUR" in env

    def test_build_environment(self, sample_config):
        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        env = stage.build_environment()

        assert env["STARTPDY"] == "20210611"
        assert env["STARTCYC"] == "00"
        assert env["COASTAL_DOMAIN"] == "pacific"
        assert env["METEO_SOURCE"] == "NWM_RETRO"
        assert env["USE_TPXO"] == "YES"

    def test_validate_default_returns_empty(self, sample_config):
        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        assert stage.validate() == []

    def test_log_with_monitor(self, sample_config):
        monitor = WorkflowMonitor(MonitoringConfig())

        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, monitor)
        stage._log("test message")  # Should not raise

    def test_log_without_monitor(self, sample_config):
        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        stage._log("test message")  # Should not raise

    def test_update_substep_with_monitor(self, sample_config):
        monitor = WorkflowMonitor(MonitoringConfig())
        monitor.register_stages(["test"])

        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, monitor)
        stage._update_substep("sub1")
        assert "sub1" in monitor.stages["test"].substeps


class TestStageNames:
    def test_download_stage(self, sample_config):
        stage = DownloadStage(sample_config)
        assert stage.name == "download"

    def test_pre_forcing_stage(self, sample_config):
        stage = PreForcingStage(sample_config)
        assert stage.name == "pre_forcing"

    def test_nwm_forcing_stage(self, sample_config):
        stage = NWMForcingStage(sample_config)
        assert stage.name == "nwm_forcing"

    def test_post_forcing_stage(self, sample_config):
        stage = PostForcingStage(sample_config)
        assert stage.name == "post_forcing"

    def test_update_params_stage(self, sample_config):
        stage = UpdateParamsStage(sample_config)
        assert stage.name == "update_params"

    def test_boundary_condition_stage(self, sample_config):
        stage = BoundaryConditionStage(sample_config)
        assert stage.name == "boundary_conditions"

    def test_pre_schism_stage(self, sample_config):
        stage = PreSCHISMStage(sample_config)
        assert stage.name == "pre_schism"

    def test_schism_run_stage(self, sample_config):
        stage = SCHISMRunStage(sample_config)
        assert stage.name == "schism_run"

    def test_post_schism_stage(self, sample_config):
        stage = PostSCHISMStage(sample_config)
        assert stage.name == "post_schism"


class TestSTOFSBoundaryStage:
    def test_validate_download_enabled(self, sample_config):
        sample_config.boundary = BoundaryConfig(source="stofs")
        sample_config.download.enabled = True
        stage = STOFSBoundaryStage(sample_config)
        assert stage.validate() == []

    def test_validate_no_stofs_file(self, sample_config):
        sample_config.boundary = BoundaryConfig(source="stofs")
        sample_config.download.enabled = False
        stage = STOFSBoundaryStage(sample_config)
        errors = stage.validate()
        assert len(errors) > 0

    def test_validate_stofs_file_not_found(self, sample_config, tmp_path):
        sample_config.boundary = BoundaryConfig(
            source="stofs", stofs_file=tmp_path / "nonexistent.nc"
        )
        sample_config.download.enabled = False
        stage = STOFSBoundaryStage(sample_config)
        errors = stage.validate()
        assert len(errors) > 0
        assert "not found" in errors[0]

    def test_validate_stofs_file_exists(self, sample_config, tmp_path):
        stofs_file = tmp_path / "stofs.nc"
        stofs_file.write_text("data")
        sample_config.boundary = BoundaryConfig(source="stofs", stofs_file=stofs_file)
        sample_config.download.enabled = False
        stage = STOFSBoundaryStage(sample_config)
        assert stage.validate() == []


class TestBoundaryConditionStage:
    def test_validate_tpxo(self, sample_config):
        sample_config.boundary = BoundaryConfig(source="tpxo")
        stage = BoundaryConditionStage(sample_config)
        assert stage.validate() == []

    def test_validate_stofs_no_file(self, sample_config):
        sample_config.boundary = BoundaryConfig(source="stofs")
        sample_config.download.enabled = False
        stage = BoundaryConditionStage(sample_config)
        errors = stage.validate()
        assert len(errors) > 0


class TestBuildDateEnvNegativeDuration:
    """Test _build_date_env with negative duration_hours (reanalysis)."""

    def test_negative_duration(self, tmp_path):
        config = CoastalCalibConfig(
            slurm=SlurmConfig(user="test"),
            simulation=SimulationConfig(
                start_date=datetime(2021, 6, 11),
                duration_hours=-24,
                coastal_domain="pacific",
                meteo_source="nwm_retro",
            ),
            boundary=BoundaryConfig(source="tpxo"),
            paths=PathConfig(work_dir=tmp_path, raw_download_dir=tmp_path / "dl"),
            download=DownloadConfig(enabled=False),
        )

        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test"

            def run(self):
                return {}

        stage = ConcreteStage(config, None)
        env = {}
        stage._build_date_env(env)
        # With negative duration, SCHISM_BEGIN_DATE should be before SCHISM_END_DATE
        assert env["SCHISM_END_DATE"] == "202106110000"
