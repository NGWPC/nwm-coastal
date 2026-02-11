"""Tests for coastal_calibration.runner module."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from coastal_calibration.config.schema import SchismModelConfig, SfincsModelConfig
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
    def test_schism_stage_order(self):
        """SchismModelConfig.stage_order returns the correct SCHISM stages."""
        expected = [
            "download",
            "pre_forcing",
            "nwm_forcing",
            "post_forcing",
            "update_params",
            "schism_obs",
            "boundary_conditions",
            "pre_schism",
            "schism_run",
            "post_schism",
            "schism_plot",
        ]
        assert expected == SchismModelConfig().stage_order

    def test_sfincs_stage_order(self, tmp_path):
        """SfincsModelConfig.stage_order returns the correct SFINCS stages."""
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
            "sfincs_wind",
            "sfincs_pressure",
            "sfincs_write",
            "sfincs_run",
            "sfincs_plot",
        ]
        assert expected == SfincsModelConfig(prebuilt_dir=tmp_path).stage_order

    def test_stage_order_property(self, sample_config):
        runner = CoastalCalibRunner(sample_config)
        # Default model is "schism" -> stage_order comes from SchismModelConfig
        assert SchismModelConfig().stage_order == runner.STAGE_ORDER

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


class TestSplitStagesForSubmit:
    """Tests for _split_stages_for_submit stage partitioning."""

    def test_schism_split_full_pipeline(self, sample_config):
        """SCHISM: download→pre_job, forcing/boundary/run→job, plot→post_job."""
        sample_config.download.enabled = True
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        stages = runner._get_stages_to_run(None, None)
        pre_job, job, post_job = runner._split_stages_for_submit(stages)

        # download is python-only, before first container stage
        assert "download" in pre_job
        # schism_obs is sandwiched between container stages, promoted to pre_job
        assert "schism_obs" in pre_job
        # All container stages should be in job
        assert "pre_forcing" in job
        assert "nwm_forcing" in job
        assert "post_forcing" in job
        assert "update_params" in job
        assert "boundary_conditions" in job
        assert "pre_schism" in job
        assert "schism_run" in job
        assert "post_schism" in job
        # schism_plot is python-only, after last container stage
        assert "schism_plot" in post_job
        # No stage should appear in multiple groups
        all_stages = pre_job + job + post_job
        assert len(all_stages) == len(set(all_stages))
        assert set(all_stages) == set(stages)

    def test_sfincs_split_full_pipeline(self, tmp_path, sample_config):
        """SFINCS: 13 stages→pre_job, sfincs_run→job, sfincs_plot→post_job."""
        from coastal_calibration.config.schema import SfincsModelConfig

        sample_config.model_config = SfincsModelConfig(prebuilt_dir=tmp_path)
        sample_config.download.enabled = True
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        stages = runner._get_stages_to_run(None, None)
        pre_job, job, post_job = runner._split_stages_for_submit(stages)

        # All stages before sfincs_run are python-only → pre_job
        assert len(pre_job) == 12  # download + 11 build stages
        assert job == ["sfincs_run"]
        assert post_job == ["sfincs_plot"]

    def test_schism_split_start_from_container(self, sample_config):
        """--start-from=schism_run: no pre_job, schism_run+post_schism in job."""
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        stages = runner._get_stages_to_run("schism_run", None)
        pre_job, job, post_job = runner._split_stages_for_submit(stages)

        assert pre_job == []
        assert "schism_run" in job
        assert "post_schism" in job
        assert "schism_plot" in post_job

    def test_schism_split_stop_after_forcing(self, sample_config):
        """--stop-after=post_forcing: no post_job."""
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        stages = runner._get_stages_to_run(None, "post_forcing")
        _pre_job, job, post_job = runner._split_stages_for_submit(stages)

        assert post_job == []
        assert "pre_forcing" in job
        assert "nwm_forcing" in job
        assert "post_forcing" in job

    def test_no_container_stages(self, sample_config):
        """When only python-only stages remain, everything goes to pre_job."""
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        # schism_obs and schism_plot are python-only
        # If we start from schism_plot and stop after schism_plot
        stages = runner._get_stages_to_run("schism_plot", "schism_plot")
        pre_job, job, post_job = runner._split_stages_for_submit(stages)

        assert pre_job == ["schism_plot"]
        assert job == []
        assert post_job == []

    def test_schism_obs_promoted_to_pre_job(self, sample_config):
        """schism_obs (sandwiched between container stages) is promoted."""
        runner = CoastalCalibRunner(sample_config)
        runner._init_stages()

        # Full pipeline includes schism_obs between post_forcing and boundary_conditions
        stages = runner._get_stages_to_run(None, None)
        pre_job, job, _post_job = runner._split_stages_for_submit(stages)

        # schism_obs must be in pre_job, NOT in job
        assert "schism_obs" in pre_job
        assert "schism_obs" not in job


class TestPreparePromotedStageDeps:
    """Tests for _prepare_promoted_stage_deps dependency pre-creation."""

    @pytest.fixture
    def config_with_local_parm(self, sample_config, tmp_path):
        """Override parm_dir with a temp directory so tests can create files."""
        sample_config.paths.parm_dir = tmp_path / "parm_root"
        sample_config.paths.parm_dir.mkdir()
        return sample_config

    def test_schism_obs_symlinks_hgrid_from_parm_nwm(self, config_with_local_parm):
        """schism_obs dep creates hgrid.gr3 symlink from parm_nwm, not parm_dir."""
        config = config_with_local_parm
        runner = CoastalCalibRunner(config)
        runner._init_stages()

        work_dir = config.paths.work_dir
        parm_nwm = config.paths.parm_nwm  # parm_dir / "parm"
        domain = config.simulation.coastal_domain

        # Create the source hgrid.gr3 at the parm_nwm path
        src_dir = parm_nwm / "coastal" / domain
        src_dir.mkdir(parents=True, exist_ok=True)
        hgrid_src = src_dir / "hgrid.gr3"
        hgrid_src.write_text("test hgrid content")

        runner._prepare_promoted_stage_deps("schism_obs")

        hgrid_dst = work_dir / "hgrid.gr3"
        assert hgrid_dst.exists()
        assert hgrid_dst.is_symlink()
        assert hgrid_dst.resolve() == hgrid_src.resolve()

    def test_schism_obs_does_not_use_parm_dir_directly(self, config_with_local_parm):
        """Ensure the symlink source is parm_nwm (parm_dir/parm), not parm_dir."""
        config = config_with_local_parm
        runner = CoastalCalibRunner(config)
        runner._init_stages()

        work_dir = config.paths.work_dir
        parm_dir = config.paths.parm_dir
        domain = config.simulation.coastal_domain

        # Create hgrid.gr3 at the WRONG path (parm_dir/coastal/...)
        wrong_dir = parm_dir / "coastal" / domain
        wrong_dir.mkdir(parents=True, exist_ok=True)
        (wrong_dir / "hgrid.gr3").write_text("wrong location")

        # Don't create it at the correct path (parm_nwm/coastal/...)
        runner._prepare_promoted_stage_deps("schism_obs")

        # Should NOT have created the symlink (source doesn't exist at correct path)
        hgrid_dst = work_dir / "hgrid.gr3"
        assert not hgrid_dst.exists()

    def test_schism_obs_skips_if_hgrid_already_exists(self, config_with_local_parm):
        """If hgrid.gr3 already exists in work_dir, don't overwrite."""
        config = config_with_local_parm
        runner = CoastalCalibRunner(config)
        runner._init_stages()

        work_dir = config.paths.work_dir
        hgrid_dst = work_dir / "hgrid.gr3"
        hgrid_dst.write_text("existing hgrid")

        # Should not raise or modify the existing file
        runner._prepare_promoted_stage_deps("schism_obs")
        assert hgrid_dst.read_text() == "existing hgrid"
        assert not hgrid_dst.is_symlink()

    def test_non_schism_obs_stage_is_noop(self, config_with_local_parm):
        """Non-schism_obs stages should not create any files."""
        config = config_with_local_parm
        runner = CoastalCalibRunner(config)
        runner._init_stages()

        work_dir = config.paths.work_dir
        before = set(work_dir.iterdir())

        runner._prepare_promoted_stage_deps("download")
        runner._prepare_promoted_stage_deps("schism_plot")

        after = set(work_dir.iterdir())
        assert before == after
