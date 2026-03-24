"""Tests for coastal_calibration.stages module."""

from __future__ import annotations

import importlib.util

import pytest

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

from coastal_calibration.config.schema import (
    BoundaryConfig,
    MonitoringConfig,
    SchismModelConfig,
)
from coastal_calibration.plotting.stations import (
    plot_station_comparison,
)
from coastal_calibration.plotting.stations import (
    plotable_stations as _plotable_stations,
)
from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.boundary import (
    BoundaryConditionStage,
    STOFSBoundaryStage,
    UpdateParamsStage,
)
from coastal_calibration.stages.download import DownloadStage
from coastal_calibration.stages.forcing import (
    NWMForcingStage,
    PostForcingStage,
    PreForcingStage,
)
from coastal_calibration.stages.schism import (
    PostSCHISMStage,
    PreSCHISMStage,
    SCHISMRunStage,
    _patch_param_nml,
)
from coastal_calibration.utils.logging import WorkflowMonitor


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

        # Core runtime variables
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"
        assert "PATH" in env

    def test_build_environment_schism_vars(self, sample_config):
        """SchismModelConfig.build_environment() sets SCHISM-specific vars."""

        class ConcreteStage(WorkflowStage):
            name = "test"
            description = "test stage"

            def run(self):
                return {}

        stage = ConcreteStage(sample_config, None)
        env = stage.build_environment()

        # Model-specific vars delegated to SchismModelConfig.build_environment
        assert "OMP_NUM_THREADS" in env
        assert "OMP_PLACES" in env
        assert "MPICH_OFI_STARTUP_CONNECT" in env

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


class TestSchismRunCommandConstruction:
    """Seam-based tests for SCHISMRunStage._build_mpi_command."""

    def test_default_command(self, sample_config):
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        assert cmd[0] == "mpiexec"
        assert "-n" in cmd
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == str(sample_config.model_config.total_tasks)
        assert cmd[-2] == "pschism"
        assert cmd[-1] == str(sample_config.model_config.nscribes)

    def test_oversubscribe_flag(self, sample_config):
        sample_config.model_config.oversubscribe = True
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        assert "--oversubscribe" in cmd

    def test_no_oversubscribe_by_default(self, sample_config):
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        assert "--oversubscribe" not in cmd

    def test_custom_binary(self, sample_config):
        sample_config.model_config.binary = "pschism_custom"
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        assert "pschism_custom" in cmd

    def test_task_count_from_config(self, sample_config):
        sample_config.model_config.nodes = 4
        sample_config.model_config.ntasks_per_node = 36
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == "144"

    def test_nscribes_passed_as_argument(self, sample_config):
        sample_config.model_config.nscribes = 4
        stage = SCHISMRunStage(sample_config)
        cmd = stage._build_mpi_command()
        assert cmd[-1] == "4"


class TestSchismModelConfigDefaults:
    """Verify SchismModelConfig defaults after removing Singularity."""

    def test_default_binary_is_pschism(self):
        config = SchismModelConfig()
        assert config.binary == "pschism"

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


class TestPlotableStations:
    """Tests for _plotable_stations pre-filter."""

    @staticmethod
    def _make_obs_ds(station_ids, n_times=10, fill_value=0.0):
        """Create a minimal xr.Dataset mimicking CO-OPS observations."""
        import numpy as np
        import xarray as xr

        t0 = np.datetime64("2021-06-11")
        times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        data = np.full((len(station_ids), n_times), fill_value)
        return xr.Dataset(
            {"water_level": (["station", "time"], data)},
            coords={"station": station_ids, "time": times},
        )

    def test_all_valid(self):
        """Stations with both sim and obs should all be returned."""
        import numpy as np

        sim = np.ones((5, 3))
        obs = self._make_obs_ds(["A", "B", "C"])
        result = _plotable_stations(["A", "B", "C"], sim, obs)
        assert len(result) == 3
        assert [sid for sid, _ in result] == ["A", "B", "C"]

    def test_sim_only_excluded(self):
        """Station with sim but no obs is excluded (no comparison possible)."""
        import numpy as np

        sim = np.ones((5, 2))
        obs = self._make_obs_ds(["A"], fill_value=np.nan)
        result = _plotable_stations(["A", "B"], sim, obs)
        # "A" has sim (all 1s) but NaN obs -> excluded
        # "B" has sim (all 1s) but not in obs -> excluded
        assert len(result) == 0

    def test_obs_only_excluded(self):
        """Station with obs but NaN sim is excluded (no comparison possible)."""
        import numpy as np

        sim = np.full((5, 1), np.nan)
        obs = self._make_obs_ds(["A"], fill_value=1.0)
        result = _plotable_stations(["A"], sim, obs)
        assert len(result) == 0

    def test_neither_obs_nor_sim(self):
        """Station with all-NaN sim and all-NaN obs is excluded."""
        import numpy as np

        sim = np.full((5, 2), np.nan)
        obs = self._make_obs_ds(["A", "B"], fill_value=np.nan)
        result = _plotable_stations(["A", "B"], sim, obs)
        assert len(result) == 0

    def test_mixed_keeps_both_only(self):
        """Only stations with both sim and obs are kept for comparison."""
        import numpy as np

        # Station 0 ("A"): sim valid, obs NaN -> excluded (no comparison)
        # Station 1 ("B"): sim NaN, obs NaN -> excluded
        # Station 2 ("C"): sim NaN, obs valid -> excluded (no comparison)
        # Station 3 ("D"): sim valid, obs valid -> KEPT
        sim = np.full((5, 4), np.nan)
        sim[:, 0] = 1.0
        sim[:, 3] = 2.0
        obs = self._make_obs_ds(["A", "B", "C", "D"], fill_value=np.nan)
        obs.water_level.loc[{"station": "C"}] = 1.0
        obs.water_level.loc[{"station": "D"}] = 3.0
        result = _plotable_stations(["A", "B", "C", "D"], sim, obs)
        assert [sid for sid, _ in result] == ["D"]
        assert [idx for _, idx in result] == [3]


@pytest.mark.skipif(not _has_matplotlib, reason="requires matplotlib (sfincs/test env)")
class TestPlotFigures:
    """Tests for plot_station_comparison."""

    @staticmethod
    def _make_obs_ds(station_ids, n_times=10, fill_value=0.0):
        """Create a minimal xr.Dataset mimicking CO-OPS observations."""
        import numpy as np
        import xarray as xr

        t0 = np.datetime64("2021-06-11")
        times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        data = np.full((len(station_ids), n_times), fill_value)
        return xr.Dataset(
            {"water_level": (["station", "time"], data)},
            coords={"station": station_ids, "time": times},
        )

    def test_skips_station_with_no_data(self, tmp_path):
        """Station with all-NaN sim and all-NaN obs produces no panel."""
        import numpy as np

        n_times = 10
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        # 2 stations: first has data, second is all NaN
        sim = np.full((n_times, 2), np.nan)
        sim[:, 0] = np.linspace(0, 1, n_times)
        obs = self._make_obs_ds(["A", "B"], n_times=n_times)
        obs.water_level.loc[{"station": "B"}] = np.nan

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A", "B"], obs, figs_dir)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_returns_empty_when_all_nan(self, tmp_path):
        """When all stations lack data, no figures are created."""
        import numpy as np

        n_times = 5
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.full((n_times, 2), np.nan)
        obs = self._make_obs_ds(["A", "B"], n_times=n_times, fill_value=np.nan)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A", "B"], obs, figs_dir)
        assert paths == []

    def test_produces_multiple_figures(self, tmp_path):
        """More than 4 valid stations yields multiple figures."""
        import numpy as np

        n_times = 10
        n_stations = 6
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.ones((n_times, n_stations))
        ids = [f"S{i}" for i in range(n_stations)]
        obs = self._make_obs_ds(ids, n_times=n_times, fill_value=1.0)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ids, obs, figs_dir)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)

    def test_single_station_layout(self, tmp_path):
        """A single valid station should produce a 1x1 figure."""
        import numpy as np

        n_times = 10
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.ones((n_times, 1))
        obs = self._make_obs_ds(["A"], n_times=n_times, fill_value=1.0)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A"], obs, figs_dir)
        assert len(paths) == 1
        assert paths[0].exists()


# ---------------------------------------------------------------------------
# SFINCS plot stage tests
# ---------------------------------------------------------------------------


class TestSfincsPlotableStations:
    """Tests for _plotable_stations in sfincs_build (identical logic to SCHISM)."""

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

    def test_all_valid(self):
        import numpy as np

        sim = np.ones((5, 3))
        obs = self._make_obs_ds(["A", "B", "C"])
        result = _plotable_stations(["A", "B", "C"], sim, obs)
        assert len(result) == 3

    def test_sim_only_excluded(self):
        import numpy as np

        sim = np.ones((5, 2))
        obs = self._make_obs_ds(["A"], fill_value=np.nan)
        result = _plotable_stations(["A", "B"], sim, obs)
        assert len(result) == 0

    def test_obs_only_excluded(self):
        import numpy as np

        sim = np.full((5, 1), np.nan)
        obs = self._make_obs_ds(["A"], fill_value=1.0)
        result = _plotable_stations(["A"], sim, obs)
        assert len(result) == 0

    def test_mixed_keeps_both_only(self):
        import numpy as np

        sim = np.full((5, 3), np.nan)
        sim[:, 2] = 2.0
        obs = self._make_obs_ds(["A", "B", "C"], fill_value=np.nan)
        obs.water_level.loc[{"station": "C"}] = 3.0
        result = _plotable_stations(["A", "B", "C"], sim, obs)
        assert [sid for sid, _ in result] == ["C"]


@pytest.mark.skipif(not _has_matplotlib, reason="requires matplotlib (sfincs/test env)")
class TestSfincsPlotFigures:
    """Tests for plot_station_comparison (identical to SCHISM)."""

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

    def test_skips_station_with_no_data(self, tmp_path):
        import numpy as np

        n_times = 10
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.full((n_times, 2), np.nan)
        sim[:, 0] = np.linspace(0, 1, n_times)
        obs = self._make_obs_ds(["A", "B"], n_times=n_times)
        obs.water_level.loc[{"station": "B"}] = np.nan

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A", "B"], obs, figs_dir)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_returns_empty_when_all_nan(self, tmp_path):
        import numpy as np

        n_times = 5
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.full((n_times, 2), np.nan)
        obs = self._make_obs_ds(["A", "B"], n_times=n_times, fill_value=np.nan)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A", "B"], obs, figs_dir)
        assert paths == []

    def test_produces_multiple_figures(self, tmp_path):
        import numpy as np

        n_times = 10
        n_stations = 6
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.ones((n_times, n_stations))
        ids = [f"S{i}" for i in range(n_stations)]
        obs = self._make_obs_ds(ids, n_times=n_times, fill_value=1.0)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ids, obs, figs_dir)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)

    def test_single_station_layout(self, tmp_path):
        import numpy as np

        n_times = 10
        t0 = np.datetime64("2021-06-11")
        sim_times = np.arange(t0, t0 + np.timedelta64(n_times, "h"), np.timedelta64(1, "h"))
        sim = np.ones((n_times, 1))
        obs = self._make_obs_ds(["A"], n_times=n_times, fill_value=1.0)

        figs_dir = tmp_path / "figs"
        paths = plot_station_comparison(sim_times, sim, ["A"], obs, figs_dir)
        assert len(paths) == 1
        assert paths[0].exists()
