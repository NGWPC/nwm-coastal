"""Tests for coastal_calibration.schism.prep module.

Unit tests for the pure-Python SCHISM pre/post processing functions
that replaced the former bash scripts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from coastal_calibration.schism.prep import (
    _write_th_file,
    partition_mesh,
    run_combine_sink_source,
    stage_chrtout_files,
    stage_ldasin_files,
    update_params,
)

# ---------------------------------------------------------------------------
# stage_chrtout_files
# ---------------------------------------------------------------------------


class TestStageChrtoutFiles:
    """Tests for CHRTOUT file staging (symlink logic)."""

    def test_creates_staging_dirs(self, tmp_path):
        streamflow = tmp_path / "streamflow"
        streamflow.mkdir()

        # Create a dummy CHRTOUT file
        dt = datetime(2021, 6, 11, tzinfo=UTC)
        fname = "202106110000.CHRTOUT_DOMAIN1"
        (streamflow / fname).write_text("dummy")

        nwm_out, nwm_ana = stage_chrtout_files(
            work_dir=tmp_path,
            start_date=dt,
            duration_hours=1,
            coastal_domain="atlgulf",
            streamflow_dir=streamflow,
        )
        assert nwm_out.exists()
        assert nwm_ana.exists()

    def test_hawaii_creates_subhourly_links(self, tmp_path):
        streamflow = tmp_path / "streamflow"
        streamflow.mkdir()

        dt = datetime(2021, 6, 11, tzinfo=UTC)
        # Create sub-hourly CHRTOUT files
        for suffix in ["00", "15", "30", "45"]:
            fname = f"2021061100{suffix}.CHRTOUT_DOMAIN1"
            (streamflow / fname).write_text("dummy")
        for suffix in ["00", "15", "30", "45"]:
            fname = f"2021061101{suffix}.CHRTOUT_DOMAIN1"
            (streamflow / fname).write_text("dummy")

        nwm_out, nwm_ana = stage_chrtout_files(
            work_dir=tmp_path,
            start_date=dt,
            duration_hours=1,
            coastal_domain="hawaii",
            streamflow_dir=streamflow,
        )
        # Ana dir should have the first 00 file
        assert any(nwm_ana.iterdir())
        # Output dir should have sub-hourly files
        out_files = list(nwm_out.iterdir())
        assert len(out_files) > 0


# ---------------------------------------------------------------------------
# _write_th_file
# ---------------------------------------------------------------------------


class TestWriteThFile:
    def test_writes_correct_format(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "test.th"
        _write_th_file(path, data, tstep=3600.0)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        # First line: time=0, then values
        parts = lines[0].split("\t")
        assert parts[0] == "0.0"
        assert parts[1] == "1.0"
        assert parts[2] == "2.0"
        # Second line: time=3600
        parts = lines[1].split("\t")
        assert parts[0] == "3600.0"


# ---------------------------------------------------------------------------
# run_combine_sink_source
# ---------------------------------------------------------------------------


class TestRunCombineSinkSource:
    def test_raises_on_missing_binary(self, tmp_path):
        """Should raise when binary is not on PATH."""
        with patch("coastal_calibration.schism.prep.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 127
            mock_run.return_value.stderr = "combine_sink_source: not found"
            with pytest.raises(RuntimeError, match="combine_sink_source failed"):
                run_combine_sink_source(tmp_path)

    def test_passes_correct_stdin(self, tmp_path):
        r"""Should pass '1\\n2\\n' as stdin to the binary."""
        with patch("coastal_calibration.schism.prep.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            run_combine_sink_source(tmp_path)
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["input"] == "1\n2\n"
            assert call_kwargs.kwargs["cwd"] == tmp_path


# ---------------------------------------------------------------------------
# partition_mesh
# ---------------------------------------------------------------------------


class TestPartitionMesh:
    def test_calls_metis_prep_and_gpmetis(self, tmp_path):
        """Verify correct subprocess commands and partition.prop generation."""
        # Create a fake graphinfo.part.34 file
        n_compute = 36 - 2  # total_tasks - nscribes
        part_file = tmp_path / f"graphinfo.part.{n_compute}"
        part_file.write_text("\n".join(str(i % 4) for i in range(100)))

        with patch("coastal_calibration.schism.prep.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = partition_mesh(
                work_dir=tmp_path,
                total_tasks=36,
                nscribes=2,
            )

        # Should have called metis_prep then gpmetis
        assert mock_run.call_count == 2
        metis_call = mock_run.call_args_list[0]
        assert metis_call.args[0][0] == "metis_prep"
        gpmetis_call = mock_run.call_args_list[1]
        assert gpmetis_call.args[0][0] == "gpmetis"
        assert str(n_compute) in gpmetis_call.args[0]

        # partition.prop should exist with line numbers
        assert result.exists()
        lines = result.read_text().splitlines()
        assert len(lines) == 100
        assert lines[0].startswith("1 ")
        assert lines[99].startswith("100 ")

    def test_raises_on_metis_prep_failure(self, tmp_path):
        with patch("coastal_calibration.schism.prep.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "error"
            with pytest.raises(RuntimeError, match="metis_prep failed"):
                partition_mesh(work_dir=tmp_path, total_tasks=4, nscribes=1)


# ---------------------------------------------------------------------------
# stage_ldasin_files
# ---------------------------------------------------------------------------


class TestStageLdasinFiles:
    def test_creates_forcing_dirs(self, tmp_path):
        nwm_dir = tmp_path / "meteo"
        nwm_dir.mkdir()
        dt = datetime(2020, 8, 26, tzinfo=UTC)
        # Create dummy LDASIN files
        for h in range(3):
            t = dt + timedelta(hours=h)
            fname = f"{t.strftime('%Y%m%d%H')}.LDASIN_DOMAIN1"
            (nwm_dir / fname).write_text("dummy")

        forcing_input, coastal_output = stage_ldasin_files(
            work_dir=tmp_path,
            start_date=dt,
            duration_hours=2,
            nwm_forcing_dir=nwm_dir,
        )
        assert forcing_input.exists()
        assert coastal_output.exists()
        # Should have symlinks in the subdirectory
        subdir = forcing_input / "2020082600"
        assert subdir.exists()
        assert len(list(subdir.iterdir())) == 3

    def test_handles_missing_files(self, tmp_path):
        """Should log warning but not crash for missing LDASIN files."""
        nwm_dir = tmp_path / "meteo"
        nwm_dir.mkdir()
        dt = datetime(2020, 8, 26, tzinfo=UTC)

        forcing_input, _ = stage_ldasin_files(
            work_dir=tmp_path,
            start_date=dt,
            duration_hours=1,
            nwm_forcing_dir=nwm_dir,
        )
        subdir = forcing_input / "2020082600"
        assert subdir.exists()
        # No symlinks since no source files exist
        assert len(list(subdir.iterdir())) == 0


# ---------------------------------------------------------------------------
# update_params
# ---------------------------------------------------------------------------


class TestUpdateParams:
    def _create_template(self, prebuilt_dir):
        """Create a minimal param.nml template for testing."""
        prebuilt_dir.mkdir(parents=True, exist_ok=True)

        param_text = """\
&CORE
  ipre = 0
  ibc = 1
  rnday = 10
  dt = 200.
  nspool = 18
  ihfskip = 324
/

&OPT
  start_year = 2000
  start_month = 1
  start_day = 1
  start_hour = 0
  ihot = 1
  if_source = 1
  nws = 2
  wtiminc = 600
  impose_net_flux = 0
  isconsv = 1
  isav = 0
  vclose_surf_frac = 0.0
/

&SCHOUT
  nhot = 1
  nhot_write = 18
/
"""
        (prebuilt_dir / "param.nml").write_text(param_text)

        # Create dummy mesh files
        for fname in ["hgrid.gr3", "vgrid.in", "bctides.in", "manning.gr3"]:
            (prebuilt_dir / fname).write_text("dummy")

        # Create sflux directory
        sflux_dir = prebuilt_dir / "sflux"
        sflux_dir.mkdir()
        (sflux_dir / "sflux_inputs.txt").write_text("&sflux_inputs\n/\n")

        return prebuilt_dir

    def test_removes_deprecated_params(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "impose_net_flux" not in text
        assert "isconsv" not in text
        assert "isav" not in text
        assert "vclose_surf_frac" not in text

    def test_adds_mandatory_params(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "nbins_veg_vert = 1" in text
        assert "nmarsh_types = 1" in text

    def test_updates_date_params(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26, 12, 30)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "start_year = 2020" in text
        assert "start_month = 08" in text
        assert "start_day = 26" in text
        assert "start_hour = 12.50" in text

    def test_cold_start_without_hotstart(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "ihot = 0" in text

    def test_symlinks_mesh_files(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        assert (work_dir / "hgrid.gr3").is_symlink()
        assert (work_dir / "vgrid.in").is_symlink()
        assert (work_dir / "bctides.in").is_symlink()
        assert (work_dir / "sflux" / "sflux_inputs.txt").exists()

    def test_sets_if_source_netcdf(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        dt = datetime(2020, 8, 26)
        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=dt,
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "if_source = -1" in text
