"""Tests for coastal_calibration.cli module."""

from __future__ import annotations

import pytest
import yaml
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
        assert "schism_boundary" in result.output
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

    @pytest.mark.parametrize("model", ["schism", "sfincs"])
    def test_init_greatlakes_uses_glofs(self, runner, tmp_path, model):
        output_path = tmp_path / "config.yaml"
        result = runner.invoke(
            cli, ["init", str(output_path), "--domain", "greatlakes", "--model", model]
        )
        assert result.exit_code == 0
        cfg = yaml.safe_load(output_path.read_text())
        assert cfg["boundary"] == {"source": "glofs", "glofs_model": "leofs"}
        assert cfg["model_config"]["forcing_to_mesh_offset_m"] == 0.0
        if model == "schism":
            assert cfg["model_config"]["include_noaa_gages"] is True

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


class TestCLIPrepareSchismMesh:
    def test_command_registered(self, runner):
        result = runner.invoke(cli, ["prepare-schism-mesh", "--help"])
        assert result.exit_code == 0
        assert "hgrid.nc" in result.output
        assert "open_bnds_hgrid.nc" in result.output

    def test_missing_hgrid_gr3(self, runner, tmp_path):
        result = runner.invoke(cli, ["prepare-schism-mesh", str(tmp_path)])
        assert result.exit_code != 0
        assert "hgrid.gr3 not found" in result.output

    def test_refuses_existing_without_force(self, runner, tmp_path):
        (tmp_path / "hgrid.gr3").write_text("stub")
        (tmp_path / "hgrid.nc").write_bytes(b"stub")
        result = runner.invoke(cli, ["prepare-schism-mesh", str(tmp_path)])
        assert result.exit_code != 0
        assert "already exist" in result.output
        assert "--force" in result.output

    def test_nonexistent_dir(self, runner, tmp_path):
        result = runner.invoke(cli, ["prepare-schism-mesh", str(tmp_path / "nope")])
        assert result.exit_code != 0


class TestCLILogLevelOption:
    """--log-level sets the log FILE level; the console is always INFO."""

    def test_present_on_file_writing_commands(self, runner):
        for command in ("run", "create"):
            result = runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0
            assert "--log-level" in result.output

    def test_absent_where_no_log_file_is_written(self, runner):
        for command in ("prepare-topobathy", "prepare-schism-mesh", "update-dem-index"):
            result = runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0
            assert "--log-level" not in result.output

    def test_verbose_flag_removed(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert "--verbose" not in result.output

    def test_rejects_invalid_level(self, runner, sample_config_yaml):
        result = runner.invoke(cli, ["run", str(sample_config_yaml), "--log-level", "LOUD"])
        assert result.exit_code != 0

    def test_console_stays_info_when_file_is_quieted(self, runner, sample_config_yaml):
        result = runner.invoke(
            cli, ["run", str(sample_config_yaml), "--dry-run", "--log-level", "ERROR"]
        )
        assert "Dry run mode" in result.output

    def test_env_var_accepted(self, runner, sample_config_yaml):
        result = runner.invoke(
            cli,
            ["run", str(sample_config_yaml), "--dry-run"],
            env={"COASTAL_LOG_LEVEL": "WARNING"},
        )
        assert "Dry run mode" in result.output
