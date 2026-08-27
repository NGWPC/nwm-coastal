"""Tests for coastal_calibration.schism.prep module.

Unit tests for the pure-Python SCHISM pre/post processing functions
that replaced the former bash scripts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

import netCDF4
import numpy as np
import pandas as pd
import pytest

from coastal_calibration.schism.prep import (
    _write_th_file,
    make_discharge,
    merge_source_sink,
    partition_mesh,
    run_combine_sink_source,
    stage_chrtout_files,
    stage_forecast_forcing,
    stage_ldasin_files,
    update_params,
    validate_param_nml,
)

if TYPE_CHECKING:
    from pathlib import Path

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

    def test_reruns_do_not_mix_dates(self, tmp_path):
        """Regression: staged links from an earlier run must not survive.

        ``make_discharge`` globs the staging directory and derives every
        timestamp from the files themselves, so a leftover link would
        silently splice another run's dates into the discharge series.
        """
        streamflow = tmp_path / "streamflow"
        streamflow.mkdir()
        for day in ("20211001", "20240520"):
            for hour in range(4):
                (streamflow / f"{day}{hour:02d}00.CHRTOUT_DOMAIN1").write_text("dummy")

        staged_dates = []
        for day in ("20211001", "20240520"):
            nwm_out, _ = stage_chrtout_files(
                work_dir=tmp_path,
                start_date=datetime(int(day[:4]), int(day[4:6]), int(day[6:]), tzinfo=UTC),
                duration_hours=1,
                coastal_domain="atlgulf",
                streamflow_dir=streamflow,
            )
            staged_dates.append({p.name[:8] for p in nwm_out.glob("*CHRTOUT*")})

        assert staged_dates == [{"20211001"}, {"20240520"}]

    def test_rerun_drops_files_beyond_a_shorter_window(self, tmp_path):
        """A shorter rerun at the same start date must not keep extra hours."""
        streamflow = tmp_path / "streamflow"
        streamflow.mkdir()
        for hour in range(6):
            (streamflow / f"202106110{hour}00.CHRTOUT_DOMAIN1").write_text("dummy")

        dt = datetime(2021, 6, 11, tzinfo=UTC)
        kwargs = {
            "work_dir": tmp_path,
            "start_date": dt,
            "coastal_domain": "atlgulf",
            "streamflow_dir": streamflow,
        }
        long_out, _ = stage_chrtout_files(duration_hours=4, **kwargs)
        n_long = len(list(long_out.glob("*CHRTOUT*")))
        short_out, _ = stage_chrtout_files(duration_hours=1, **kwargs)
        n_short = len(list(short_out.glob("*CHRTOUT*")))

        assert n_short < n_long


# ---------------------------------------------------------------------------
# _write_th_file
# ---------------------------------------------------------------------------


class TestWriteThFile:
    def test_writes_correct_format(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        times = np.array([0.0, 3600.0])
        path = tmp_path / "test.th"
        _write_th_file(path, data, times)
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

    def test_writes_times_as_given(self, tmp_path):
        """The caller (make_discharge) owns t=0 padding; this just writes
        whatever elapsed-time array it's handed, verbatim."""
        data = np.array([[1.0], [2.0], [3.0]])
        times = np.array([0.0, 3600.0, 7200.0])
        path = tmp_path / "test.th"
        _write_th_file(path, data, times)
        lines = path.read_text().splitlines()
        assert [line.split("\t")[0] for line in lines] == ["0.0", "3600.0", "7200.0"]


# ---------------------------------------------------------------------------
# make_discharge
# ---------------------------------------------------------------------------


class TestMakeDischarge:
    """Reading the source and sink blocks of ``nwmReaches.csv``."""

    @staticmethod
    def _run(work_dir: Path, reaches_text: str, fids: list[int]):
        """Run ``make_discharge`` against a canned streamflow frame."""
        (work_dir / "nwmReaches.csv").write_text(reaches_text)
        start = datetime(2021, 6, 11, tzinfo=UTC)
        # Distinct series per feature ID so an assertion pins which column
        # a discharge series landed in, not just its shape.
        df = pd.DataFrame(
            {fid: [(i + 1) * 10.0 + step for step in range(3)] for i, fid in enumerate(fids)},
            index=pd.date_range(start, periods=3, freq="h"),
        )
        with patch(
            "coastal_calibration.data.streamflow.read_streamflow", return_value=df
        ) as mock_read:
            make_discharge(
                work_dir=work_dir,
                nwm_output_dir=work_dir,
                meteo_source="nwm_retro",
                start_date=start,
                end_date=start + timedelta(hours=2),
            )
        return mock_read

    def test_reads_sources_and_sinks(self, tmp_path):
        self._run(tmp_path, "1\n101 5001\n\n1\n301 6001\n", [5001, 6001])

        assert (tmp_path / "source_sink.in").read_text() == "1\n101\n\n1\n301\n"
        # Reach 5001 feeds the source column, 6001 the sink column, negated.
        src_rows = [ln.split("\t") for ln in (tmp_path / "vsource.th").read_text().splitlines()]
        assert [r[1] for r in src_rows] == ["10.0", "11.0", "12.0"]
        sink_rows = [ln.split("\t") for ln in (tmp_path / "vsink.th").read_text().splitlines()]
        assert [r[1] for r in sink_rows] == ["-20.0", "-21.0", "-22.0"]

    def test_accepts_a_missing_sink_block(self, tmp_path):
        """A subset with only sources may end right after the source block."""
        mock_read = self._run(tmp_path, "2\n101 5001\n205 5002\n", [5001, 5002])

        assert mock_read.call_args.args[0] == [5001, 5002]
        assert (tmp_path / "source_sink.in").read_text() == "2\n101\n205\n\n0\n"
        # vsink.th keeps its time column and carries no discharge values.
        assert [ln.split("\t") for ln in (tmp_path / "vsink.th").read_text().splitlines()] == [
            ["0.0"],
            ["3600.0"],
            ["7200.0"],
        ]

    def test_accepts_an_explicit_zero_sink_count(self, tmp_path):
        self._run(tmp_path, "2\n101 5001\n205 5002\n\n0\n", [5001, 5002])

        assert (tmp_path / "source_sink.in").read_text() == "2\n101\n205\n\n0\n"

    def test_rejects_a_truncated_block(self, tmp_path):
        """A declared count with missing rows is a corrupt file, not zero sinks."""
        with pytest.raises(ValueError, match="declares 2 sink rows but has 1"):
            self._run(tmp_path, "1\n101 5001\n\n2\n301 6001\n", [5001, 6001])

    def test_rejects_sink_rows_beyond_the_declared_count(self, tmp_path):
        """An under-declared count would silently drop sink forcing."""
        with pytest.raises(ValueError, match="declares 1 sink rows but has 2"):
            self._run(tmp_path, "1\n101 5001\n\n1\n301 6001\n302 6002\n", [5001, 6001])

    def test_rejects_sink_rows_under_a_zero_count(self, tmp_path):
        with pytest.raises(ValueError, match="declares 0 sink rows but has 1"):
            self._run(tmp_path, "1\n101 5001\n\n0\n301 6001\n", [5001, 6001])


# ---------------------------------------------------------------------------
# merge_source_sink
# ---------------------------------------------------------------------------


class TestMergeSourceSink:
    def test_writes_source_nc_without_sinks(self, tmp_path):
        """The sink-free path must still produce a readable ``source.nc``.

        ``combine_sink_source`` echoes a time-only ``vsink.th.1`` back when
        the mesh subset has no sinks, and netCDF turns a zero-length
        ``nsinks`` dimension into an unlimited one.
        """
        ntime, n_elem = 4, 6
        (tmp_path / "source_sink.in").write_text("2\n2\n5\n\n0\n")
        (tmp_path / "vsink.th.1").write_text("".join(f"{i * 3600.0}\n" for i in range(ntime)))
        (tmp_path / "vsource.th.1").write_text(
            "".join(f"{i * 3600.0}\t10.0\t20.0\n" for i in range(ntime))
        )
        with netCDF4.Dataset(tmp_path / "precip_source.nc", "w") as ds:
            ds.createDimension("time", ntime)
            ds.createDimension("nsources", n_elem)
            ds.createVariable("vsource", "f8", ("time", "nsources"))[:] = np.ones((ntime, n_elem))

        merge_source_sink(work_dir=tmp_path, element_areas=np.full(n_elem, 100.0))

        with netCDF4.Dataset(tmp_path / "source.nc") as ds:
            assert ds.dimensions["nsinks"].size == 0
            assert ds.variables["vsink"].shape == (ntime, 0)
            assert ds.variables["sink_elem"].shape == (0,)
            # Elements 2 and 5 (1-based) carry precipitation plus discharge.
            src = ds.variables["vsource"][:]
            assert src[0, 1] == 11.0
            assert src[0, 4] == 21.0
            assert src[0, 0] == 1.0


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


class TestStageForecastForcing:
    def test_symlinks_single_multitimestep_file(self, tmp_path):
        """A single forecast file is symlinked with a LDASIN_DOMAIN1 name."""
        forecast_file = tmp_path / "Hawaii_202509150000.nc"
        forecast_file.write_text("dummy")
        dt = datetime(2025, 9, 15, tzinfo=UTC)

        forcing_input, coastal_output = stage_forecast_forcing(
            work_dir=tmp_path,
            start_date=dt,
            forecast_file=forecast_file,
        )

        assert coastal_output.exists()
        subdir = forcing_input / "2025091500"
        assert subdir.exists()
        links = list(subdir.iterdir())
        # Exactly one symlink, discoverable by the *LDASIN_DOMAIN1 globs.
        assert len(links) == 1
        link = links[0]
        assert link.name == "Hawaii_202509150000.LDASIN_DOMAIN1"
        assert link.is_symlink()
        assert link.resolve() == forecast_file.resolve()

    def test_raises_when_forecast_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Forecast meteo file not found"):
            stage_forecast_forcing(
                work_dir=tmp_path,
                start_date=datetime(2025, 9, 15, tzinfo=UTC),
                forecast_file=tmp_path / "does_not_exist.nc",
            )


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

    def test_if_source_disabled_when_no_discharge(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            discharge_enabled=False,
        )

        text = (work_dir / "param.nml").read_text()
        assert "if_source = 0" in text
        assert "if_source = -1" not in text

    def test_default_output_freq_writes_hourly(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
        )

        text = (work_dir / "param.nml").read_text()
        assert "nspool = 18" in text
        assert "ihfskip = 18" in text
        assert "nhot_write = 18" in text  # hourly hotstart, independent of run length

    def test_single_output_file_extends_ihfskip(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=50,
            single_output_file=True,
        )

        text = (work_dir / "param.nml").read_text()
        # 50h * 3600 / dt(200) = 900 timesteps
        assert "ihfskip = 900" in text
        assert "nhot_write = 900" in text  # rounded up to a multiple of 900

    def test_output_freq_hours_scales_nspool(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            output_freq_hours=2.0,
        )

        text = (work_dir / "param.nml").read_text()
        assert "nspool = 36" in text  # 2h * 3600 / 200 = 36
        assert "ihfskip = 36" in text  # not single-file mode, ihfskip == nspool

    def test_run_param_overrides_replace_existing_keys(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            run_param_overrides={"dt": 100, "wtiminc": 300},
        )

        text = (work_dir / "param.nml").read_text()
        assert "dt = 100" in text
        assert "wtiminc = 300" in text

    def test_run_param_overrides_insert_new_key(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            run_param_overrides={"iwbl": 1},
        )

        text = (work_dir / "param.nml").read_text()
        assert "iwbl = 1" in text

    def test_ihfskip_override_rederives_nhot_write(self, tmp_path):
        # Pacific forecast template: ihfskip=324 (18 hourly outputs per
        # file) and nhot_write=324. Overriding ihfskip alone should
        # produce a matching nhot_write so the SCHISM divisibility
        # constraint stays satisfied without the user setting both.
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            run_param_overrides={"ihfskip": 324},
        )

        text = (work_dir / "param.nml").read_text()
        assert "ihfskip = 324" in text
        assert "nhot_write = 324" in text  # auto-bumped to a multiple of 324

    def test_timestep_seconds_drives_dt_and_nspool(self, tmp_path):
        # dt should be written into param.nml verbatim, and nspool
        # should rescale so the wall-clock output cadence stays the
        # same (default output_freq_hours=1.0 → hourly snapshots).
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            timestep_seconds=100,  # half the default; nspool should double
        )

        text = (work_dir / "param.nml").read_text()
        assert "dt = 100" in text
        assert "nspool = 36" in text  # 1h * 3600 / 100 = 36
        assert "ihfskip = 36" in text  # default single_output_file=False

    def test_ihfskip_and_nhot_write_overrides_both_respected(self, tmp_path):
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        prebuilt = tmp_path / "prebuilt"
        self._create_template(prebuilt)

        update_params(
            work_dir=work_dir,
            prebuilt_dir=prebuilt,
            start_date=datetime(2020, 8, 26),
            duration_hours=6,
            run_param_overrides={"ihfskip": 864, "nhot_write": 8640},
        )

        text = (work_dir / "param.nml").read_text()
        assert "ihfskip = 864" in text
        assert "nhot_write = 8640" in text


class TestValidateParamNml:
    def _write(self, tmp_path, body: str) -> Path:
        p = tmp_path / "param.nml"
        p.write_text(body)
        return p

    def test_consistent_namelist_passes(self, tmp_path):
        p = self._write(
            tmp_path,
            "&CORE\n  nspool = 18\n  ihfskip = 18\n/\n"
            "&SCHOUT\n  iout_sta = 1\n  nspool_sta = 18\n  nhot = 1\n  nhot_write = 324\n/\n",
        )
        assert validate_param_nml(p) == []

    def test_nhot_write_not_multiple_of_ihfskip(self, tmp_path):
        p = self._write(
            tmp_path,
            "&CORE\n  nspool = 18\n  ihfskip = 900\n/\n"
            "&SCHOUT\n  nhot = 1\n  nhot_write = 324\n/\n",
        )
        errors = validate_param_nml(p)
        assert any("nhot_write" in e and "ihfskip" in e for e in errors)

    def test_ihfskip_not_multiple_of_nspool(self, tmp_path):
        p = self._write(
            tmp_path,
            "&CORE\n  nspool = 18\n  ihfskip = 25\n/\n&SCHOUT\n  nhot = 0\n/\n",
        )
        errors = validate_param_nml(p)
        assert any("ihfskip" in e and "nspool" in e for e in errors)

    def test_nhot_write_not_multiple_of_nspool_sta(self, tmp_path):
        p = self._write(
            tmp_path,
            "&CORE\n  nspool = 18\n  ihfskip = 18\n/\n"
            "&SCHOUT\n  iout_sta = 1\n  nspool_sta = 25\n  nhot = 1\n  nhot_write = 324\n/\n",
        )
        errors = validate_param_nml(p)
        # Multiple constraints could fire here; pick out the nspool_sta one.
        assert any("nspool_sta" in e for e in errors)

    def test_nhot_zero_skips_nhot_write_check(self, tmp_path):
        # nhot_write doesn't have to be a multiple of ihfskip when nhot=0
        p = self._write(
            tmp_path,
            "&CORE\n  nspool = 18\n  ihfskip = 900\n/\n"
            "&SCHOUT\n  nhot = 0\n  nhot_write = 324\n/\n",
        )
        # nspool divides ihfskip cleanly here, so no errors expected
        assert validate_param_nml(p) == []
