"""Tests for coastal_calibration.workflow_utils module."""

from __future__ import annotations

from pathlib import Path

import pytest

from coastal_calibration.workflow_utils import (
    post_nwm_coastal,
    pre_nwm_forcing_coastal,
)


class TestPreNwmForcingCoastal:
    def test_creates_directories_and_symlinks(self, tmp_path):
        # Set up mock directories
        date_string = "2021061100"
        forcing_dir = tmp_path / "coastal_forcing"
        nwm_dir = tmp_path / "nwm_forcing"
        nwm_dir.mkdir()
        data_exec = tmp_path / "exec"
        data_exec.mkdir()

        # Create mock LDASIN files
        for i in range(4):
            from coastal_calibration.time_utils import advance_time

            fname = f"{advance_time(date_string, i)}.LDASIN_DOMAIN1"
            (nwm_dir / fname).write_text("data")

        env_vars = pre_nwm_forcing_coastal(
            date_string=date_string,
            coastal_forcing_output_dir=forcing_dir,
            length_hrs=3,
            nwm_forcing_retro_dir=nwm_dir,
            data_exec=data_exec,
        )

        assert forcing_dir.exists()
        assert "FORCING_BEGIN_DATE" in env_vars
        assert env_vars["FORCING_BEGIN_DATE"] == "202106110000"
        assert env_vars["LENGTH_HRS"] == "3"
        assert env_vars["FORCING_START_YEAR"] == "2021"
        assert env_vars["FORCING_START_MONTH"] == "06"

        # Check symlinks were created
        forcing_input = data_exec / "forcing_input" / "2021061100"
        assert forcing_input.exists()
        symlinks = list(forcing_input.iterdir())
        assert len(symlinks) == 4

    def test_env_vars_complete(self, tmp_path):
        nwm_dir = tmp_path / "nwm"
        nwm_dir.mkdir()
        data_exec = tmp_path / "exec"
        data_exec.mkdir()
        forcing_dir = tmp_path / "forcing"

        env_vars = pre_nwm_forcing_coastal(
            date_string="2021061112",
            coastal_forcing_output_dir=forcing_dir,
            length_hrs=1,
            nwm_forcing_retro_dir=nwm_dir,
            data_exec=data_exec,
        )

        required_keys = [
            "FORCING_BEGIN_DATE",
            "FORCING_END_DATE",
            "NWM_FORCING_OUTPUT_DIR",
            "COASTAL_FORCING_INPUT_DIR",
            "COASTAL_WORK_DIR",
            "FORCING_START_YEAR",
            "FORCING_START_MONTH",
            "FORCING_START_DAY",
            "FORCING_START_HOUR",
            "COASTAL_FORCING_OUTPUT_DIR",
            "LENGTH_HRS",
            "FECPP_JOB_INDEX",
            "FECPP_JOB_COUNT",
        ]
        for key in required_keys:
            assert key in env_vars, f"Missing env var: {key}"


class TestPostNwmCoastal:
    def test_no_fatal_error(self, tmp_path):
        """Should run without error when no fatal.error file exists."""
        data_exec = tmp_path / "exec"
        data_exec.mkdir()
        outputs = data_exec / "outputs"
        outputs.mkdir()
        # No fatal.error file -> should not raise
        post_nwm_coastal(data_exec=data_exec, length_hrs=3)

    def test_empty_fatal_error(self, tmp_path):
        """Empty fatal.error file should not raise."""
        data_exec = tmp_path / "exec"
        data_exec.mkdir()
        outputs = data_exec / "outputs"
        outputs.mkdir()
        (outputs / "fatal.error").write_text("")
        post_nwm_coastal(data_exec=data_exec, length_hrs=3)

    def test_fatal_error_raises(self, tmp_path):
        """Non-empty fatal.error file should raise RuntimeError."""
        data_exec = tmp_path / "exec"
        data_exec.mkdir()
        outputs = data_exec / "outputs"
        outputs.mkdir()
        (outputs / "fatal.error").write_text("Something went wrong")
        with pytest.raises(RuntimeError, match="program failed"):
            post_nwm_coastal(data_exec=data_exec, length_hrs=3)

    def test_creates_outputs_dir(self, tmp_path):
        """Should create outputs dir if it doesn't exist."""
        data_exec = tmp_path / "exec"
        data_exec.mkdir()
        post_nwm_coastal(data_exec=data_exec, length_hrs=3)
        assert (data_exec / "outputs").exists()
