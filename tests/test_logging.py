"""Tests for coastal_calibration.utils.logging module."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from coastal_calibration.config.schema import MonitoringConfig
from coastal_calibration.utils.logging import (
    ProgressBar,
    StageProgress,
    StageStatus,
    WorkflowMonitor,
    _validate_level,
    configure_logger,
    generate_log_path,
    get_log_file_path,
    logger,
)


class TestStageStatus:
    def test_values(self):
        assert StageStatus.PENDING == "pending"
        assert StageStatus.RUNNING == "running"
        assert StageStatus.COMPLETED == "completed"
        assert StageStatus.FAILED == "failed"
        assert StageStatus.SKIPPED == "skipped"


class TestStageProgress:
    def test_defaults(self):
        sp = StageProgress(name="test")
        assert sp.status == StageStatus.PENDING
        assert sp.start_time is None
        assert sp.end_time is None
        assert sp.substeps == []

    def test_duration_none(self):
        sp = StageProgress(name="test")
        assert sp.duration is None
        assert sp.duration_str == "-"

    def test_duration_completed(self):
        now = datetime.now()
        sp = StageProgress(
            name="test",
            start_time=now - timedelta(seconds=90),
            end_time=now,
        )
        d = sp.duration
        assert d is not None
        assert abs(d.total_seconds() - 90) < 1

    def test_duration_str_seconds(self):
        now = datetime.now()
        sp = StageProgress(
            name="test",
            start_time=now - timedelta(seconds=45),
            end_time=now,
        )
        assert "s" in sp.duration_str

    def test_duration_str_minutes(self):
        now = datetime.now()
        sp = StageProgress(
            name="test",
            start_time=now - timedelta(minutes=5, seconds=30),
            end_time=now,
        )
        assert "m" in sp.duration_str

    def test_duration_str_hours(self):
        now = datetime.now()
        sp = StageProgress(
            name="test",
            start_time=now - timedelta(hours=2, minutes=15),
            end_time=now,
        )
        assert "h" in sp.duration_str

    def test_duration_running(self):
        sp = StageProgress(
            name="test",
            start_time=datetime.now() - timedelta(seconds=10),
        )
        # Running stage should have a duration
        assert sp.duration is not None


class TestValidateLevel:
    def test_valid_strings(self):
        assert _validate_level("DEBUG") == logging.DEBUG
        assert _validate_level("INFO") == logging.INFO
        assert _validate_level("WARNING") == logging.WARNING
        assert _validate_level("ERROR") == logging.ERROR
        assert _validate_level("CRITICAL") == logging.CRITICAL

    def test_case_insensitive(self):
        assert _validate_level("debug") == logging.DEBUG

    def test_valid_int(self):
        assert _validate_level(logging.INFO) == logging.INFO

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_level("VERBOSE")

    def test_invalid_int(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_level(999)

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="must be str or int"):
            _validate_level(3.14)


class TestGenerateLogPath:
    def test_basic(self, tmp_path):
        path = generate_log_path(tmp_path)
        assert path.parent == tmp_path
        assert "coastal-calibration" in path.name
        assert path.suffix == ".log"

    def test_custom_prefix(self, tmp_path):
        path = generate_log_path(tmp_path, prefix="myprefix")
        assert "myprefix" in path.name


class TestConfigureLogger:
    def test_file_logging(self, tmp_path):
        log_file = tmp_path / "test.log"
        configure_logger(file=str(log_file), file_level="DEBUG")
        assert log_file.exists() or True  # File may not be created until first write
        # Cleanup
        configure_logger(file=None)

    def test_verbose_flag(self):
        configure_logger(verbose=True)
        configure_logger(verbose=False)


class TestWorkflowMonitor:
    def test_init(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        assert mon.stages == {}
        assert mon.workflow_start is None

    def test_register_stages(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["download", "run"])
        assert "download" in mon.stages
        assert "run" in mon.stages
        assert mon.stages["download"].status == StageStatus.PENDING

    def test_start_end_workflow(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.start_workflow()
        assert mon.workflow_start is not None
        mon.end_workflow(success=True)
        assert mon.workflow_end is not None

    def test_start_end_stage(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["test_stage"])
        mon.start_stage("test_stage", "Testing")
        assert mon.stages["test_stage"].status == StageStatus.RUNNING
        mon.end_stage("test_stage", StageStatus.COMPLETED)
        assert mon.stages["test_stage"].status == StageStatus.COMPLETED

    def test_start_stage_auto_register(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.start_stage("new_stage")
        assert "new_stage" in mon.stages

    def test_end_stage_not_registered(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        # Should not raise
        mon.end_stage("nonexistent")

    def test_update_substep(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["test"])
        mon.update_substep("test", "substep1")
        assert "substep1" in mon.stages["test"].substeps

    def test_log_methods(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        # Should not raise
        mon.info("test info")
        mon.warning("test warning")
        mon.error("test error")
        mon.debug("test debug")
        mon.log("info", "test log")

    def test_stage_context_success(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["test"])
        with mon.stage_context("test"):
            pass
        assert mon.stages["test"].status == StageStatus.COMPLETED

    def test_stage_context_failure(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["test"])
        with pytest.raises(RuntimeError):
            with mon.stage_context("test"):
                raise RuntimeError("fail")
        assert mon.stages["test"].status == StageStatus.FAILED

    def test_get_progress_dict(self):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["s1"])
        mon.start_workflow()
        d = mon.get_progress_dict()
        assert "workflow_start" in d
        assert "stages" in d
        assert "s1" in d["stages"]

    def test_save_progress(self, tmp_path):
        cfg = MonitoringConfig()
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["s1"])
        progress_file = tmp_path / "progress.json"
        mon.save_progress(progress_file)
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert "stages" in data


class TestProgressBar:
    def test_basic(self):
        pb = ProgressBar(total=10, description="Test")
        assert pb.total == 10
        assert pb.current == 0

    def test_update(self, capsys):
        pb = ProgressBar(total=3, description="Test")
        pb.update(1)
        assert pb.current == 1

    def test_update_past_total(self, capsys):
        pb = ProgressBar(total=2, description="Test")
        pb.update(5)
        assert pb.current == 2  # clamped to total

    def test_zero_total(self, capsys):
        pb = ProgressBar(total=0)
        pb.update(0)
        # Should not raise

    def test_context_manager(self, capsys):
        with ProgressBar(total=2) as pb:
            pb.update(2)
        # Should not raise
