"""Tests for coastal_calibration.cli module."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from coastal_calibration.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIStages:
    def test_stages_command(self, runner):
        result = runner.invoke(cli, ["stages"])
        assert result.exit_code == 0
        # Both SCHISM and SFINCS stages shown by default
        assert "download" in result.output
        assert "schism_run" in result.output
        assert "boundary_conditions" in result.output
        assert "sfincs_run" in result.output

    def test_stages_schism_only(self, runner):
        result = runner.invoke(cli, ["stages", "--model", "schism"])
        assert result.exit_code == 0
        assert "schism_run" in result.output
        assert "sfincs_run" not in result.output

    def test_stages_sfincs_only(self, runner):
        result = runner.invoke(cli, ["stages", "--model", "sfincs"])
        assert result.exit_code == 0
        assert "sfincs_run" in result.output
        assert "schism_run" not in result.output


class TestCLIInit:
    def test_init_default(self, runner, tmp_path):
        output_path = tmp_path / "config.yaml"
        result = runner.invoke(cli, ["init", str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "coastal_domain" in content

    def test_init_with_domain(self, runner, tmp_path):
        output_path = tmp_path / "config.yaml"
        result = runner.invoke(cli, ["init", str(output_path), "--domain", "hawaii"])
        assert result.exit_code == 0
        content = output_path.read_text()
        assert "hawaii" in content

    def test_init_force_overwrite(self, runner, tmp_path):
        output_path = tmp_path / "config.yaml"
        output_path.write_text("existing")
        result = runner.invoke(cli, ["init", str(output_path), "--force"])
        assert result.exit_code == 0
        # Should overwrite
        content = output_path.read_text()
        assert "coastal_domain" in content

    def test_init_no_overwrite_abort(self, runner, tmp_path):
        output_path = tmp_path / "config.yaml"
        output_path.write_text("existing")
        result = runner.invoke(cli, ["init", str(output_path)], input="n\n")
        assert result.exit_code != 0  # Abort

    def test_init_sfincs(self, runner, tmp_path):
        output_path = tmp_path / "config.yaml"
        result = runner.invoke(cli, ["init", str(output_path), "--model", "sfincs"])
        assert result.exit_code == 0
        content = output_path.read_text()
        assert "model: sfincs" in content
        assert "prebuilt_dir" in content


class TestCLIValidate:
    def test_validate_valid_config(self, runner, sample_config_yaml):
        """Validate will report errors since Singularity image won't exist."""
        result = runner.invoke(cli, ["validate", str(sample_config_yaml)])
        # Config won't fully validate in test env (no singularity, etc.)
        # but should not crash
        assert result.exit_code in (0, 1)

    def test_validate_nonexistent_config(self, runner, tmp_path):
        result = runner.invoke(cli, ["validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0


class TestCLIRun:
    def test_run_nonexistent_config(self, runner, tmp_path):
        result = runner.invoke(cli, ["run", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0
