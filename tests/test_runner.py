"""Tests for coastal_calibration.runner module."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from coastal_calibration.runner import CoastalCalibRunner, WorkflowResult


class TestWorkflowResult:
    def test_duration_seconds(self):
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)
        result = WorkflowResult(
            success=True,
            job_id=None,
            start_time=start,
            end_time=end,
            stages_completed=["s1"],
            stages_failed=[],
            outputs={},
            errors=[],
        )
        assert result.duration_seconds == 3600.0

    def test_duration_seconds_no_end(self):
        result = WorkflowResult(
            success=True,
            job_id=None,
            start_time=datetime.now(),
            end_time=None,
            stages_completed=[],
            stages_failed=[],
            outputs={},
            errors=[],
        )
        assert result.duration_seconds is None

    def test_to_dict(self):
        result = WorkflowResult(
            success=True,
            job_id="123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1),
            stages_completed=["s1"],
            stages_failed=[],
            outputs={"key": "value"},
            errors=[],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["job_id"] == "123"
        assert d["duration_seconds"] == 3600.0
        assert d["stages_completed"] == ["s1"]

    def test_save(self, tmp_path):
        result = WorkflowResult(
            success=True,
            job_id=None,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1),
            stages_completed=[],
            stages_failed=[],
            outputs={},
            errors=[],
        )
        path = tmp_path / "result.json"
        result.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["success"] is True

    def test_save_creates_parent_dirs(self, tmp_path):
        result = WorkflowResult(
            success=True,
            job_id=None,
            start_time=datetime(2024, 1, 1),
            end_time=None,
            stages_completed=[],
            stages_failed=[],
            outputs={},
            errors=[],
        )
        path = tmp_path / "deep" / "nested" / "result.json"
        result.save(path)
        assert path.exists()


class TestCoastalCalibRunner:
    def test_stage_order(self):
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
        assert CoastalCalibRunner.STAGE_ORDER == expected

    def test_init(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        assert runner.config is sample_config
        assert runner._slurm is None

    def test_get_stages_to_run_all(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        # download is disabled in sample_config
        stages = runner._get_stages_to_run(None, None)
        assert "download" not in stages
        assert "pre_forcing" in stages
        assert "post_schism" in stages

    def test_get_stages_to_run_start_from(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        stages = runner._get_stages_to_run("boundary_conditions", None)
        assert "pre_forcing" not in stages
        assert "boundary_conditions" in stages
        assert stages[0] == "boundary_conditions"

    def test_get_stages_to_run_stop_after(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        stages = runner._get_stages_to_run(None, "nwm_forcing")
        assert "nwm_forcing" in stages
        assert "post_forcing" not in stages

    def test_get_stages_to_run_invalid_stage(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        with pytest.raises(ValueError, match="Unknown stage"):
            runner._get_stages_to_run("nonexistent", None)
        with pytest.raises(ValueError, match="Unknown stage"):
            runner._get_stages_to_run(None, "nonexistent")

    def test_get_stages_with_download_enabled(self, sample_config):
        sample_config.download.enabled = True
        runner = CoastalCalibRunner(sample_config)
        stages = runner._get_stages_to_run(None, None)
        assert "download" in stages
