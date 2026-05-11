"""Tests for coastal_calibration.base and coastal_calibration.data.download_stage modules."""

from __future__ import annotations

import importlib.util
from datetime import datetime

import pytest

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

from coastal_calibration.base import WorkflowStage
from coastal_calibration.config.schema import (
    BoundaryConfig,
    MonitoringConfig,
    SchismModelConfig,
)
from coastal_calibration.data.download_stage import DownloadStage
from coastal_calibration.logging import WorkflowMonitor
from coastal_calibration.plotting.stations import plot_station_comparison
from coastal_calibration.schism.boundary import (
    BoundaryConditionStage,
    STOFSBoundaryStage,
    UpdateParamsStage,
)
from coastal_calibration.schism.forcing import (
    NWMForcingStage,
    PostForcingStage,
    PreForcingStage,
)
from coastal_calibration.schism.stages import (
    PostSCHISMStage,
    PreSCHISMStage,
    SCHISMRunStage,
    _patch_param_nml,
)


class TestWorkflowStageBase:
    def test_abstract_cant_instantiate(self):
        with pytest.raises(TypeError):
            WorkflowStage(None, None)

    def test_build_environment(self, sample_config):
        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        env = stage.build_environment()

        # Core runtime variables set by WorkflowStage
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"
        assert "OMP_NUM_THREADS" in env
        # Thread-pinning policy is intentionally NOT a default — users
        # supply OMP_PROC_BIND / OMP_PLACES via runtime_env when their
        # cluster benefits from NUMA pinning. Confirm we don't set them.
        assert "OMP_PLACES" not in env or env["OMP_PLACES"] != "cores"
        assert "OMP_PROC_BIND" not in env or env["OMP_PROC_BIND"] != "close"
        assert "PATH" in env

    def test_build_environment_schism_mpi_vars(self, sample_config):
        """SchismModelConfig.build_environment() delegates to build_mpi_env."""
        from unittest.mock import patch

        from coastal_calibration.utils import MpiImpl

        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)

        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = stage.build_environment()
            assert env["MPICH_OFI_STARTUP_CONNECT"] == "1"

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
        assert stage.name == "schism_forcing_prep"

    def test_nwm_forcing_stage(self, sample_config):
        stage = NWMForcingStage(sample_config)
        assert stage.name == "schism_forcing"

    def test_post_forcing_stage(self, sample_config):
        stage = PostForcingStage(sample_config)
        assert stage.name == "schism_sflux"

    def test_update_params_stage(self, sample_config):
        stage = UpdateParamsStage(sample_config)
        assert stage.name == "schism_params"

    def test_boundary_condition_stage(self, sample_config):
        stage = BoundaryConditionStage(sample_config)
        assert stage.name == "schism_boundary"

    def test_pre_schism_stage(self, sample_config):
        stage = PreSCHISMStage(sample_config)
        assert stage.name == "schism_prep"

    def test_schism_run_stage(self, sample_config):
        stage = SCHISMRunStage(sample_config)
        assert stage.name == "schism_run"

    def test_post_schism_stage(self, sample_config):
        stage = PostSCHISMStage(sample_config)
        assert stage.name == "schism_postprocess"


class TestRuntimeEnvPlumbing:
    """Regression tests for ``runtime_env`` propagation.

    ``runtime_env`` must be merged into the subprocess environment for
    every stage that spawns one, so users can override MPI / fabric
    env vars on a per-cluster basis without monkey-patching the
    package.
    """

    def test_nwm_forcing_stage_applies_runtime_env(self, sample_config):
        """NWMForcingStage's subprocess env must include model.runtime_env.

        Exercised via the ``_build_run_env`` seam to avoid having to set
        up the full forcing-generation fixture stack.
        """
        sample_config.model_config.runtime_env = {
            "OMPI_MCA_btl": "self,tcp",
            "OMPI_MCA_mtl": "^ofi",
            "OMPI_MCA_pml": "ob1",
        }
        stage = NWMForcingStage(sample_config)
        env = stage._build_run_env()

        assert env["OMPI_MCA_btl"] == "self,tcp"
        assert env["OMPI_MCA_mtl"] == "^ofi"
        assert env["OMPI_MCA_pml"] == "ob1"
        # base build_environment vars still present:
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"

    def test_nwm_forcing_stage_runtime_env_empty_is_safe(self, sample_config):
        """No runtime_env override should leave build_environment intact."""
        # default runtime_env is {} (empty)
        stage = NWMForcingStage(sample_config)
        env = stage._build_run_env()
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"

    def test_sfincs_forcing_stage_applies_runtime_env(self, tmp_path):
        """SFINCS forcing stage must merge sfincs.runtime_env.

        Specifically, ``_run_predict_tide`` must pass an env to
        subprocess.run that includes the user-supplied overrides on
        ``SfincsModelConfig.runtime_env``.
        """
        from coastal_calibration.config.schema import (
            BoundaryConfig,
            CoastalCalibConfig,
            DownloadConfig,
            MonitoringConfig,
            PathConfig,
            SfincsModelConfig,
            SimulationConfig,
        )
        from coastal_calibration.sfincs.stages import SfincsForcingStage

        prebuilt = tmp_path / "model"
        prebuilt.mkdir()
        config = CoastalCalibConfig(
            simulation=SimulationConfig(
                start_date=datetime(2021, 6, 11, 0, 0, 0),
                duration_hours=3,
                coastal_domain="pacific",
                meteo_source="nwm_retro",
            ),
            boundary=BoundaryConfig(source="tpxo"),
            paths=PathConfig(
                work_dir=tmp_path / "work",
                raw_download_dir=tmp_path / "downloads",
            ),
            model_config=SfincsModelConfig(
                prebuilt_dir=prebuilt,
                runtime_env={
                    "OMPI_MCA_btl": "self,tcp",
                    "OMPI_MCA_mtl": "^ofi",
                },
            ),
            monitoring=MonitoringConfig(),
            download=DownloadConfig(enabled=False),
        )

        stage = SfincsForcingStage(config)
        env = stage._build_run_env()

        assert env["OMPI_MCA_btl"] == "self,tcp"
        assert env["OMPI_MCA_mtl"] == "^ofi"
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"

    def test_make_stofs_boundary_applies_runtime_env(self, tmp_path, monkeypatch):
        """make_stofs_boundary launches MPI; runtime_env must reach its env.

        ``STOFSBoundaryStage`` calls this helper for the elev2D regrid
        and previously built env from ``os.environ.copy()`` only,
        bypassing ``model.runtime_env`` entirely. This regression test
        captures the env passed to subprocess.run via a stub.
        """
        import subprocess as sp
        from datetime import datetime as _dt

        from coastal_calibration.schism.prep import make_stofs_boundary

        captured: dict[str, dict[str, str]] = {}

        def fake_run(cmd, *, env, **kwargs):
            captured["env"] = env
            (tmp_path / "elev2D.th.nc").touch()
            return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(sp, "run", fake_run)

        stofs = tmp_path / "stofs.nc"
        stofs.touch()
        prebuilt = tmp_path / "prebuilt"
        prebuilt.mkdir()

        make_stofs_boundary(
            work_dir=tmp_path,
            start_date=_dt(2025, 11, 26),
            duration_hours=50,
            stofs_file=stofs,
            prebuilt_dir=prebuilt,
            mpi_tasks=1,
            runtime_env={
                "OMPI_MCA_btl": "self,tcp",
                "FI_PROVIDER": "tcp",
            },
        )

        assert captured["env"]["OMPI_MCA_btl"] == "self,tcp"
        assert captured["env"]["FI_PROVIDER"] == "tcp"
        # baseline HDF5 var still present:
        assert captured["env"]["HDF5_USE_FILE_LOCKING"] == "FALSE"


class TestSchismRunCommandConstruction:
    """Seam-based tests for SCHISMRunStage._build_mpi_command."""

    def test_default_command(self, sample_config):
        from pathlib import Path

        stage = SCHISMRunStage(sample_config)
        exe = Path("/usr/bin/pschism")
        cmd = stage._build_mpi_command(exe)
        assert cmd[0] == "mpiexec"
        assert "-n" in cmd
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == str(sample_config.model_config.total_tasks)
        assert cmd[-2] == str(exe)
        assert cmd[-1] == str(sample_config.model_config.nscribes)

    def test_oversubscribe_flag_openmpi(self, sample_config):
        from pathlib import Path
        from unittest.mock import patch

        from coastal_calibration.utils import MpiImpl

        sample_config.model_config.oversubscribe = True
        stage = SCHISMRunStage(sample_config)
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.OPENMPI):
            cmd = stage._build_mpi_command(Path("/usr/bin/pschism"))
            assert "--oversubscribe" in cmd

    def test_oversubscribe_flag_mpich_ignored(self, sample_config):
        from pathlib import Path
        from unittest.mock import patch

        from coastal_calibration.utils import MpiImpl

        sample_config.model_config.oversubscribe = True
        stage = SCHISMRunStage(sample_config)
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            cmd = stage._build_mpi_command(Path("/usr/bin/pschism"))
            assert "--oversubscribe" not in cmd

    def test_no_oversubscribe_by_default(self, sample_config):
        from pathlib import Path

        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command(Path("/usr/bin/pschism"))
        assert "--oversubscribe" not in cmd

    def test_custom_exe(self, sample_config):
        from pathlib import Path

        exe = Path("/opt/wcoss2/bin/pschism_custom")
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command(exe)
        assert str(exe) in cmd

    def test_task_count_from_config(self, sample_config):
        from pathlib import Path

        sample_config.model_config.nodes = 4
        sample_config.model_config.ntasks_per_node = 36
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command(Path("/usr/bin/pschism"))
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == "144"

    def test_nscribes_passed_as_argument(self, sample_config):
        from pathlib import Path

        sample_config.model_config.nscribes = 4
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command(Path("/usr/bin/pschism"))
        assert cmd[-1] == "4"


class TestSchismModelConfigDefaults:
    """Verify SchismModelConfig defaults after removing Singularity."""

    def test_default_schism_exe_is_none(self):
        config = SchismModelConfig()
        assert config.schism_exe is None

    def test_validate_no_singularity_check(self, sample_config):
        """validate() should not fail due to missing Singularity image."""
        errors = sample_config.model_config.validate(sample_config)
        # The only errors should be about paths, not about SIF
        for err in errors:
            assert "singularity" not in err.lower()
            assert "sif" not in err.lower()


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


class TestPatchParamNml:
    """Tests for _patch_param_nml station output patching."""

    def test_replaces_existing_iout_sta(self, tmp_path):
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  iout_sta = 0\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        assert "iout_sta = 1" in text
        assert "iout_sta = 0" not in text

    def test_inserts_iout_sta_after_schout(self, tmp_path):
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  some_param = 1\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        assert "iout_sta = 1" in text

    def test_sets_nspool_sta(self, tmp_path):
        """nspool_sta must be set when enabling station output."""
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  iout_sta = 0\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        assert "nspool_sta = 18" in text

    def test_replaces_existing_nspool_sta(self, tmp_path):
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  iout_sta = 0\n  nspool_sta = 99\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        assert "nspool_sta = 18" in text
        assert "nspool_sta = 99" not in text

    def test_custom_nspool_sta(self, tmp_path):
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  iout_sta = 0\n/\n")
        _patch_param_nml(p, nspool_sta=36)
        text = p.read_text()
        assert "nspool_sta = 36" in text

    def test_inserts_nspool_sta_after_iout_sta(self, tmp_path):
        """When nspool_sta doesn't exist, insert it after iout_sta."""
        p = tmp_path / "param.nml"
        p.write_text("&SCHOUT\n  iout_sta = 0\n  other = 5\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        lines = text.splitlines()
        iout_idx = next(i for i, line in enumerate(lines) if "iout_sta" in line)
        nspool_idx = next(i for i, line in enumerate(lines) if "nspool_sta" in line)
        assert nspool_idx == iout_idx + 1

    def test_fallback_appends_schout_block(self, tmp_path):
        """When &SCHOUT is missing, append a new block."""
        p = tmp_path / "param.nml"
        p.write_text("&CORE\n  dt = 200\n/\n")
        _patch_param_nml(p)
        text = p.read_text()
        assert "iout_sta = 1" in text
        assert "nspool_sta = 18" in text

    def test_nhot_write_divisibility(self, tmp_path):
        """Default nspool_sta=18 divides all nhot_write values.

        Covers runtime values from update_param.bash (18, 72, 162, 2160)
        and every domain template (hawaii=162, atlgulf/pacific=324, prvi=8640).
        """
        nhot_values = [18, 72, 162, 324, 2160, 8640, 18 * 5, 18 * 12]
        for nhot in nhot_values:
            assert nhot % 18 == 0, f"nhot_write={nhot} not divisible by nspool_sta=18"

    @pytest.mark.parametrize(
        ("nhot_write", "old_nspool_sta"),
        [
            (162, 10),  # hawaii
            (324, 18),  # atlgulf / pacific
            (8640, 10),  # prvi
        ],
        ids=["hawaii", "atlgulf_pacific", "prvi"],
    )
    def test_domain_template_param_nml(self, tmp_path, nhot_write, old_nspool_sta):
        """Patch real domain templates so mod(nhot_write, nspool_sta)==0."""
        p = tmp_path / "param.nml"
        p.write_text(
            "&SCHOUT\n"
            f"  nhot_write = {nhot_write}\n"
            "  iout_sta = 0\n"
            f"  nspool_sta = {old_nspool_sta}"
            " !needed if iout_sta/=0; mod(nhot_write,nspool_sta) must=0\n"
            "/\n"
        )
        _patch_param_nml(p)
        text = p.read_text()
        assert "iout_sta = 1" in text
        assert "nspool_sta = 18" in text
        assert f"nspool_sta = {old_nspool_sta}" not in text or old_nspool_sta == 18
        # The inline comment should be preserved
        assert "!needed if" in text
        # Verify the constraint: mod(nhot_write, nspool_sta) == 0
        assert nhot_write % 18 == 0


@pytest.mark.skipif(not _has_matplotlib, reason="requires matplotlib (sfincs/test env)")
class TestPlotStationComparison:
    """Tests for :func:`plot_station_comparison`."""

    @staticmethod
    def _make_obs_ds(station_ids, n_times=10, fill_value=0.0):
        import numpy as np
        import xarray as xr

        t0 = np.datetime64("2021-06-11")
        times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        data = np.full((len(station_ids), n_times), fill_value)
        return xr.Dataset(
            {"water_level": (["station", "time"], data)},
            coords={"station": station_ids, "time": times},
        )

    def _make_run(self, n_times: int, n_stations: int, value: float):
        import numpy as np

        t0 = np.datetime64("2021-06-11")
        times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        elev = np.full((n_times, n_stations), value)
        return times, elev

    def test_two_runs_and_obs_overlay(self, tmp_path):
        """Two model permutations overlaid with observed data → one figure."""
        ids = ["A", "B"]
        run_a = self._make_run(10, 2, 1.0)
        run_b = self._make_run(10, 2, 1.5)
        obs = self._make_obs_ds(ids, n_times=10, fill_value=1.2)

        paths = plot_station_comparison(
            {"baseline": run_a, "tuned": run_b},
            ids,
            tmp_path / "figs",
            obs_ds=obs,
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_runs_without_obs(self, tmp_path):
        """Pure model-vs-model comparison — ``obs_ds`` is optional."""
        ids = ["A", "B"]
        run_a = self._make_run(10, 2, 1.0)
        run_b = self._make_run(10, 2, 1.5)

        paths = plot_station_comparison(
            {"baseline": run_a, "tuned": run_b},
            ids,
            tmp_path / "figs",
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_single_run_with_obs(self, tmp_path):
        """A single-entry ``runs`` dict is the common sim-vs-obs case."""
        ids = ["A"]
        run = self._make_run(10, 1, 1.0)
        obs = self._make_obs_ds(ids, fill_value=1.0)

        paths = plot_station_comparison({"Simulated": run}, ids, tmp_path / "figs", obs_ds=obs)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_plotable_keeps_station_when_any_run_has_data(self, tmp_path):
        """Plotability rule: any run (or obs) with finite data keeps the station."""
        import numpy as np

        ids = ["A", "B"]
        run_a_t, run_a_e = self._make_run(5, 2, 1.0)
        run_a_e[:, 1] = np.nan  # run A has no data at station B
        run_b = self._make_run(5, 2, 2.0)  # run B has data everywhere

        paths = plot_station_comparison(
            {"a": (run_a_t, run_a_e), "b": run_b},
            ids,
            tmp_path / "figs",
        )
        assert len(paths) == 1

    def test_obs_only_station_is_plotable(self, tmp_path):
        """A station with obs data but NaN in all runs is kept."""
        import numpy as np

        ids = ["A"]
        t, e = self._make_run(5, 1, np.nan)
        obs = self._make_obs_ds(ids, n_times=5, fill_value=1.0)

        paths = plot_station_comparison({"sim": (t, e)}, ids, tmp_path / "figs", obs_ds=obs)
        assert len(paths) == 1

    def test_empty_runs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            plot_station_comparison({}, ["A"], tmp_path / "figs")

    def test_all_nan_yields_empty(self, tmp_path):
        """Station with NaN everywhere (runs + obs) produces no figures."""
        import numpy as np

        t0 = np.datetime64("2021-06-11")
        times = np.arange(t0, t0 + np.timedelta64(5, "h"), np.timedelta64(1, "h"))
        bad_elev = np.full((5, 2), np.nan)
        bad_obs = self._make_obs_ds(["A", "B"], n_times=5, fill_value=np.nan)

        paths = plot_station_comparison(
            {"x": (times, bad_elev), "y": (times, bad_elev)},
            ["A", "B"],
            tmp_path / "figs",
            obs_ds=bad_obs,
        )
        assert paths == []

    def test_pagination_over_four_stations(self, tmp_path):
        """More than 4 plotable stations → multiple figures (4 per fig)."""
        n = 6
        ids = [f"S{i}" for i in range(n)]
        run = self._make_run(10, n, 1.0)
        obs = self._make_obs_ds(ids, fill_value=1.0)

        paths = plot_station_comparison({"sim": run}, ids, tmp_path / "figs", obs_ds=obs)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)

    def test_single_station_layout(self, tmp_path):
        """A single plotable station → one 1-by-1 figure."""
        ids = ["A"]
        run = self._make_run(10, 1, 1.0)
        obs = self._make_obs_ds(ids, fill_value=1.0)

        paths = plot_station_comparison({"sim": run}, ids, tmp_path / "figs", obs_ds=obs)
        assert len(paths) == 1
        assert paths[0].exists()
