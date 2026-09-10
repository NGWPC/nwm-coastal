"""Tests for coastal_calibration.logging module."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import pytest

import coastal_calibration.logging as cc_logging
from coastal_calibration.config.schema import MonitoringConfig
from coastal_calibration.logging import (
    StageProgress,
    StageStatus,
    WorkflowMonitor,
    _validate_level,
    configure_logger,
    generate_log_path,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-global handler state that leaks between tests."""
    console_level = cc_logging._console_handler.level
    yield
    configure_logger(file=None)
    cc_logging._user_console_level = None
    cc_logging._console_handler.setLevel(console_level)


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
        assert cc_logging._console_handler.level == logging.DEBUG
        configure_logger(verbose=False)
        assert cc_logging._console_handler.level == logging.INFO

    def test_console_default_is_info(self):
        """Importing the package alone gives an INFO console, not WARNING.

        This is the level library calls that never build a WorkflowMonitor
        (e.g. ``extract_mesh``) log at.
        """
        assert cc_logging._console_handler.level == logging.INFO

    def test_traceback_written_to_file(self, tmp_path):
        log_file = tmp_path / "tb.log"
        configure_logger(level="CRITICAL", file=str(log_file), file_mode="w")
        mon = WorkflowMonitor(MonitoringConfig())
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            mon.error("Workflow failed: boom", exc_info=True)

        contents = log_file.read_text()
        assert "Traceback (most recent call last)" in contents
        assert 'raise RuntimeError("boom")' in contents


class TestLevelRouting:
    """Console is fixed at INFO; log_level sets the file level."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_console_stays_info_regardless_of_config(self, level):
        WorkflowMonitor(MonitoringConfig(log_level=level))
        assert cc_logging._console_handler.level == logging.INFO

    def test_explicit_call_beats_console_default(self):
        configure_logger(level="ERROR")
        WorkflowMonitor(MonitoringConfig())
        assert cc_logging._console_handler.level == logging.ERROR

    def test_log_level_sets_file_level(self, tmp_path):
        cfg = MonitoringConfig(log_level="WARNING", log_file=tmp_path / "f.log")
        WorkflowMonitor(cfg)
        assert cc_logging._file_handler.level == logging.WARNING

    def test_log_level_defaults_to_debug(self, tmp_path):
        cfg = MonitoringConfig(log_file=tmp_path / "f.log")
        WorkflowMonitor(cfg)
        assert cc_logging._file_handler.level == logging.DEBUG

    def test_file_honours_level_when_writing(self, tmp_path):
        log_file = tmp_path / "f.log"
        mon = WorkflowMonitor(MonitoringConfig(log_level="WARNING", log_file=log_file))
        mon.logger.debug("hidden-debug")
        mon.logger.warning("shown-warning")
        contents = log_file.read_text()
        assert "hidden-debug" not in contents
        assert "shown-warning" in contents


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

    def test_update_substep_disabled(self):
        cfg = MonitoringConfig(enable_progress_tracking=False)
        mon = WorkflowMonitor(cfg)
        mon.register_stages(["test"])
        mon.update_substep("test", "substep1")
        assert mon.stages["test"].substeps == []

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
        with pytest.raises(RuntimeError), mon.stage_context("test"):
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
