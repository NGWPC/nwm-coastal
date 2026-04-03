"""Tests for the SFINCS model creation workflow.

Covers config loading/validation, stage execution with mocked HydroMT API,
the ``SfincsCreator`` runner, and CLI integration.

Run with::

    pytest tests/test_sfincs_create.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from coastal_calibration.cli import cli
from coastal_calibration.config.create_schema import (
    ElevationConfig,
    ElevationDataset,
    GridConfig,
    RefinementLevel,
    RiverDischargeConfig,
    SfincsCreateConfig,
    SubgridConfig,
)
from coastal_calibration.sfincs.create import (
    CreateBoundaryStage,
    CreateDischargeStage,
    CreateElevationStage,
    CreateFetchDataStage,
    CreateFetchElevationStage,
    CreateGridStage,
    CreateMaskStage,
    CreateStage,
    CreateSubgridStage,
    CreateWriteStage,
    SfincsCreator,
    _clear_model,
    _get_model,
    _set_model,
    create_stages,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aoi_file(tmp_path: Path) -> Path:
    """Create a minimal GeoJSON AOI polygon in EPSG:4326."""
    import geopandas as gpd
    from shapely import box

    # Bounding box around Lavaca Bay area (lon/lat)
    gdf = gpd.GeoDataFrame(geometry=[box(-95.5, 29.0, -95.0, 29.5)], crs="EPSG:4326")
    path = tmp_path / "aoi.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Return a fresh output directory."""
    d = tmp_path / "model_output"
    d.mkdir()
    return d


@pytest.fixture
def minimal_create_config(aoi_file: Path, output_dir: Path) -> SfincsCreateConfig:
    """Return a ``SfincsCreateConfig`` with all defaults."""
    return SfincsCreateConfig(aoi=aoi_file, output_dir=output_dir)


@pytest.fixture
def minimal_config_dict(aoi_file: Path, output_dir: Path) -> dict[str, Any]:
    """Return a minimal config dictionary for YAML tests."""
    return {
        "aoi": str(aoi_file),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def minimal_config_yaml(tmp_path: Path, minimal_config_dict: dict[str, Any]) -> Path:
    """Write a minimal config YAML and return its path."""
    path = tmp_path / "create_config.yaml"
    path.write_text(yaml.dump(minimal_config_dict))
    return path


@pytest.fixture
def full_config_dict(aoi_file: Path, output_dir: Path) -> dict[str, Any]:
    """Return a fully-specified config dictionary."""
    return {
        "aoi": str(aoi_file),
        "output_dir": str(output_dir),
        "grid": {"resolution": 100, "crs": "EPSG:32615", "rotated": False},
        "elevation": {
            "datasets": [
                {"name": "copdem_30m", "zmin": 0.001},
                {"name": "gebco_15arcs", "zmin": -20000},
            ],
            "buffer_cells": 2,
        },
        "mask": {
            "zmin": -10.0,
            "boundary_zmax": -10.0,
            "reset_bounds": False,
        },
        "subgrid": {
            "nr_subgrid_pixels": 10,
            "lulc_dataset": "esa_worldcover_2021",
            "manning_land": 0.05,
            "manning_sea": 0.025,
        },
        "data_catalog": {"data_libs": ["artifact_data"]},
        "monitoring": {"log_level": "DEBUG"},
    }


@pytest.fixture
def full_config_yaml(tmp_path: Path, full_config_dict: dict[str, Any]) -> Path:
    """Write a full config YAML and return its path."""
    path = tmp_path / "create_full.yaml"
    path.write_text(yaml.dump(full_config_dict))
    return path


@pytest.fixture
def mock_sfincs_model() -> MagicMock:
    """Return a mock ``SfincsModel`` with the expected quadtree API."""
    sf = MagicMock(name="SfincsModel")
    sf.quadtree_grid.create_from_region = MagicMock()
    sf.quadtree_elevation.create = MagicMock()
    sf.quadtree_mask.create_active = MagicMock()
    sf.quadtree_mask.create_boundary = MagicMock()
    sf.quadtree_subgrid.create = MagicMock()
    sf.write = MagicMock()
    return sf


# ===================================================================
# Config unit tests
# ===================================================================


class TestSfincsCreateConfig:
    """Test SfincsCreateConfig loading and validation."""

    def test_from_yaml_minimal(self, minimal_config_yaml: Path) -> None:
        cfg = SfincsCreateConfig.from_yaml(minimal_config_yaml)
        assert cfg.aoi.exists()
        assert cfg.grid.resolution == 50.0
        assert cfg.grid.crs == "utm"
        assert cfg.subgrid.nr_subgrid_pixels == 5

    def test_from_yaml_full(self, full_config_yaml: Path) -> None:
        cfg = SfincsCreateConfig.from_yaml(full_config_yaml)
        assert cfg.grid.resolution == 100
        assert cfg.grid.crs == "EPSG:32615"
        assert cfg.grid.rotated is False
        assert cfg.elevation.buffer_cells == 2
        assert len(cfg.elevation.datasets) == 2
        assert cfg.mask.zmin == -10.0
        assert cfg.subgrid.manning_land == 0.05
        assert cfg.subgrid.nr_subgrid_pixels == 10
        assert cfg.data_catalog.data_libs == ["artifact_data"]
        assert cfg.monitoring.log_level == "DEBUG"

    def test_from_yaml_missing_aoi_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"output_dir": str(tmp_path)}))
        with pytest.raises(ValueError, match="'aoi' is required"):
            SfincsCreateConfig.from_yaml(path)

    def test_from_yaml_missing_output_dir(self, tmp_path: Path, aoi_file: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"aoi": str(aoi_file)}))
        with pytest.raises(ValueError, match="'output_dir' is required"):
            SfincsCreateConfig.from_yaml(path)

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            SfincsCreateConfig.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            SfincsCreateConfig.from_yaml(path)

    def test_validate_ok(self, minimal_create_config: SfincsCreateConfig) -> None:
        errors = minimal_create_config.validate()
        assert errors == []

    def test_validate_missing_aoi(self, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(aoi=Path("/nonexistent/aoi.geojson"), output_dir=output_dir)
        errors = cfg.validate()
        assert any("AOI file not found" in e for e in errors)

    def test_validate_bad_resolution(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            grid=GridConfig(resolution=-10),
        )
        errors = cfg.validate()
        assert any("resolution must be positive" in e for e in errors)

    def test_validate_empty_datasets(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(datasets=[]),
        )
        errors = cfg.validate()
        assert any("at least one entry" in e for e in errors)

    def test_validate_bad_manning(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            subgrid=SubgridConfig(manning_land=0, manning_sea=-1),
        )
        errors = cfg.validate()
        assert any("manning_land" in e for e in errors)
        assert any("manning_sea" in e for e in errors)

    def test_validate_missing_reclass_table(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            subgrid=SubgridConfig(reclass_table=Path("/nonexistent.csv")),
        )
        errors = cfg.validate()
        assert any("reclass_table not found" in e for e in errors)

    def test_stage_order(self, minimal_create_config: SfincsCreateConfig) -> None:
        stages = minimal_create_config.stage_order
        assert "create_subgrid" in stages
        assert "create_roughness" not in stages
        assert stages[0] == "create_grid"
        assert stages[-1] == "create_write"
        # Default config has source-backed datasets, so fetch is included
        assert "create_fetch_data" in stages

    def test_to_dict_roundtrip(self, minimal_create_config: SfincsCreateConfig) -> None:
        d = minimal_create_config.to_dict()
        assert d["aoi"] == str(minimal_create_config.aoi)
        assert d["grid"]["resolution"] == 50.0
        assert d["subgrid"]["nr_subgrid_pixels"] == 5
        assert d["subgrid"]["lulc_source"] == "esa_worldcover"

    def test_to_dict_includes_buffer_m(self, aoi_file: Path, output_dir: Path) -> None:
        """buffer_m should round-trip through to_dict when non-zero."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            grid=GridConfig(
                refinement=[RefinementLevel(polygon=aoi_file, level=3, buffer_m=-3072.0)]
            ),
        )
        d = cfg.to_dict()
        ref = d["grid"]["refinement"][0]
        assert ref["buffer_m"] == -3072.0

    def test_to_dict_omits_zero_buffer_m(self, aoi_file: Path, output_dir: Path) -> None:
        """buffer_m=0 (default) should not appear in serialized output."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            grid=GridConfig(refinement=[RefinementLevel(polygon=aoi_file, level=2)]),
        )
        d = cfg.to_dict()
        ref = d["grid"]["refinement"][0]
        assert "buffer_m" not in ref

    def test_data_catalog_data_libs_is_list(self) -> None:
        """DataCatalogConfig.data_libs should default to an empty list, not a string.

        Regression: a bare triple-quoted string after the field was a
        no-op expression, not a docstring. Verify the default is correct.
        """
        from coastal_calibration.config.create_schema import DataCatalogConfig

        dc = DataCatalogConfig()
        assert isinstance(dc.data_libs, list)
        assert dc.data_libs == []

    def test_to_yaml(self, tmp_path: Path, minimal_create_config: SfincsCreateConfig) -> None:
        path = tmp_path / "out.yaml"
        minimal_create_config.to_yaml(path)
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert loaded["grid"]["resolution"] == 50.0

    def test_relative_paths_resolved(self, tmp_path: Path) -> None:
        """Relative AOI path in YAML should resolve relative to the YAML file."""
        aoi = tmp_path / "my_aoi.geojson"
        aoi.write_text('{"type":"FeatureCollection","features":[]}')
        cfg_data = {"aoi": "my_aoi.geojson", "output_dir": str(tmp_path / "out")}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        cfg = SfincsCreateConfig.from_yaml(cfg_path)
        assert cfg.aoi == aoi.resolve()

    # --- Source config tests ---

    def test_default_includes_fetch_stage(self, minimal_create_config: SfincsCreateConfig) -> None:
        """Default datasets have source set, so fetch stage should be included."""
        stages = minimal_create_config.stage_order
        assert "create_fetch_data" in stages
        assert stages.index("create_fetch_data") < stages.index("create_elevation")

    def test_source_none_excludes_fetch_stage(self, aoi_file: Path, output_dir: Path) -> None:
        """Datasets with source=None and lulc_source=None exclude fetch stage."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(datasets=[ElevationDataset(name="copdem_30m")]),
            subgrid=SubgridConfig(lulc_source=None),
        )
        assert "create_fetch_data" not in cfg.stage_order

    def test_source_noaa_includes_fetch_stage(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="noaa_tb", source="noaa_3m")]
            ),
            subgrid=SubgridConfig(lulc_source=None),
        )
        stages = cfg.stage_order
        assert "create_fetch_data" in stages
        assert stages.index("create_fetch_data") < stages.index("create_elevation")

    def test_source_invalid_fails_validation(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(datasets=[ElevationDataset(name="bad", source="s3")]),
        )
        errors = cfg.validate()
        assert any("source must be one of" in e for e in errors)

    def test_lulc_source_includes_fetch_stage(self, aoi_file: Path, output_dir: Path) -> None:
        """lulc_source alone should trigger the fetch stage."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(datasets=[ElevationDataset(name="copdem_30m")]),
            subgrid=SubgridConfig(lulc_source="esa_worldcover"),
        )
        assert "create_fetch_data" in cfg.stage_order

    def test_lulc_source_invalid_fails_validation(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            subgrid=SubgridConfig(lulc_source="bad_lulc"),
        )
        errors = cfg.validate()
        assert any("lulc_source must be one of" in e for e in errors)

    def test_esa_worldcover_2021_normalized(self) -> None:
        """Backward compat: esa_worldcover_2021 → esa_worldcover."""
        sc = SubgridConfig(lulc_dataset="esa_worldcover_2021")
        assert sc.lulc_dataset == "esa_worldcover"

    def test_noaa_dataset_without_source_fails(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="oops", noaa_dataset="TX_Coastal_DEM")]
            ),
        )
        errors = cfg.validate()
        assert any("noaa_dataset is set but source is not" in e for e in errors)

    def test_download_dir_defaults(self, minimal_create_config: SfincsCreateConfig) -> None:
        assert minimal_create_config.download_dir is None
        assert minimal_create_config.effective_download_dir == (
            minimal_create_config.output_dir / "downloads"
        )

    def test_download_dir_explicit(self, aoi_file: Path, output_dir: Path) -> None:
        dl = output_dir / "my_dl"
        cfg = SfincsCreateConfig(aoi=aoi_file, output_dir=output_dir, download_dir=dl)
        assert cfg.effective_download_dir == dl.resolve()

    def test_to_dict_with_noaa_source(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            download_dir=output_dir / "dl",
            elevation=ElevationConfig(
                datasets=[
                    ElevationDataset(
                        name="noaa_tb",
                        source="noaa_3m",
                        noaa_dataset="TX_Coastal_DEM_2018_8899",
                    )
                ]
            ),
        )
        d = cfg.to_dict()
        ds = d["elevation"]["datasets"][0]
        assert ds["source"] == "noaa_3m"
        assert ds["noaa_dataset"] == "TX_Coastal_DEM_2018_8899"
        assert "download_dir" in d

    def test_yaml_roundtrip_with_noaa(
        self, tmp_path: Path, aoi_file: Path, output_dir: Path
    ) -> None:
        cfg_data = {
            "aoi": str(aoi_file),
            "output_dir": str(output_dir),
            "download_dir": str(tmp_path / "dl"),
            "elevation": {
                "datasets": [{"name": "noaa_tb", "zmin": -20000, "source": "noaa_3m"}],
            },
        }
        cfg_path = tmp_path / "noaa.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        cfg = SfincsCreateConfig.from_yaml(cfg_path)
        assert cfg.elevation.datasets[0].source == "noaa_3m"
        assert cfg.download_dir is not None
        assert "create_fetch_data" in cfg.stage_order


# ===================================================================
# Stage unit tests (mocked HydroMT)
# ===================================================================


class TestCreateStages:
    """Test individual creation stages with a mocked SfincsModel."""

    def test_create_grid(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        with (
            patch(
                "hydromt_sfincs.SfincsModel",
                return_value=mock_sfincs_model,
            ),
            patch(
                "coastal_calibration.sfincs.create.suppress_hydromt_output",
                return_value=MagicMock(
                    __enter__=MagicMock(), __exit__=MagicMock(return_value=False)
                ),
            ),
        ):
            stage = CreateGridStage(minimal_create_config)
            result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.quadtree_grid.create_from_region.assert_called_once()

    def test_create_elevation(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateElevationStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.quadtree_elevation.create.assert_called_once()
        _clear_model(minimal_create_config)

    def test_create_mask(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateMaskStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.quadtree_mask.create_active.assert_called_once_with(
            zmin=-5.0,
        )
        _clear_model(minimal_create_config)

    def test_create_boundary(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateBoundaryStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.quadtree_mask.create_boundary.assert_called_once_with(
            btype="waterlevel", zmax=-5.0, reset_bounds=True
        )
        _clear_model(minimal_create_config)

    def test_create_subgrid(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateSubgridStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.quadtree_subgrid.create.assert_called_once()
        call_kwargs = mock_sfincs_model.quadtree_subgrid.create.call_args[1]
        assert call_kwargs["manning_land"] == 0.04
        assert call_kwargs["manning_water"] == 0.02
        assert call_kwargs["nr_subgrid_pixels"] == 5
        _clear_model(minimal_create_config)

    def test_create_write(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateWriteStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "completed"
        mock_sfincs_model.write.assert_called_once()
        # Registry should be cleared
        with pytest.raises(RuntimeError):
            _get_model(minimal_create_config)

    def test_get_model_missing_raises(self, minimal_create_config: SfincsCreateConfig) -> None:
        _clear_model(minimal_create_config)
        with pytest.raises(RuntimeError, match="No SfincsModel found"):
            _get_model(minimal_create_config)

    def test_create_fetch_data_noaa(
        self,
        aoi_file: Path,
        output_dir: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="noaa_tb", source="noaa_3m", zmin=-20000)]
            ),
            subgrid=SubgridConfig(lulc_source=None),
        )
        _set_model(cfg, mock_sfincs_model)

        dl_dir = cfg.effective_download_dir

        tif = dl_dir / "noaa_tb.tif"
        cat = dl_dir / "noaa_tb_catalog.yml"

        def _fake_fetch(**kwargs):
            dl_dir.mkdir(parents=True, exist_ok=True)
            tif.write_bytes(b"\x00")
            cat.write_text("meta: {}")
            return tif, cat, "noaa_tb"

        with patch("coastal_calibration.data.topobathy_noaa.fetch_noaa_dem") as mock_fetch:
            mock_fetch.side_effect = _fake_fetch

            stage = CreateFetchDataStage(cfg)
            result = stage.run()

        assert result["status"] == "completed"
        assert "noaa_tb" in result["fetched"]
        mock_fetch.assert_called_once()
        _clear_model(cfg)

    def test_create_fetch_data_copdem30(
        self,
        aoi_file: Path,
        output_dir: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="copdem_30m", source="copdem_30m", zmin=0.001)]
            ),
            subgrid=SubgridConfig(lulc_source=None),
        )
        _set_model(cfg, mock_sfincs_model)

        dl_dir = cfg.effective_download_dir

        def _fake_fetch(**kwargs):
            dl_dir.mkdir(parents=True, exist_ok=True)
            (dl_dir / "copdem_30m.tif").write_bytes(b"\x00")
            cat = dl_dir / "copdem_30m_catalog.yml"
            cat.write_text("meta: {}")
            return dl_dir / "copdem_30m.tif", cat, "copdem_30m"

        with patch("coastal_calibration.data.copdem.fetch_copdem30") as mock_fetch:
            mock_fetch.side_effect = _fake_fetch
            stage = CreateFetchDataStage(cfg)
            result = stage.run()

        assert result["status"] == "completed"
        assert "copdem_30m" in result["fetched"]
        mock_fetch.assert_called_once()
        _clear_model(cfg)

    def test_create_fetch_data_gebco(
        self,
        aoi_file: Path,
        output_dir: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="gebco_15arcs", source="gebco_15arcs", zmin=-20000)]
            ),
            subgrid=SubgridConfig(lulc_source=None),
        )
        _set_model(cfg, mock_sfincs_model)

        dl_dir = cfg.effective_download_dir

        def _fake_fetch(**kwargs):
            dl_dir.mkdir(parents=True, exist_ok=True)
            (dl_dir / "gebco_15arcs.tif").write_bytes(b"\x00")
            cat = dl_dir / "gebco_15arcs_catalog.yml"
            cat.write_text("meta: {}")
            return dl_dir / "gebco_15arcs.tif", cat, "gebco_15arcs"

        with patch("coastal_calibration.data.gebco_wms.fetch_gebco") as mock_fetch:
            mock_fetch.side_effect = _fake_fetch
            stage = CreateFetchDataStage(cfg)
            result = stage.run()

        assert result["status"] == "completed"
        assert "gebco_15arcs" in result["fetched"]
        mock_fetch.assert_called_once()
        _clear_model(cfg)

    def test_create_fetch_data_lulc(
        self,
        aoi_file: Path,
        output_dir: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """Test that LULC fetching works via lulc_source."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(datasets=[ElevationDataset(name="copdem_30m")]),
            subgrid=SubgridConfig(lulc_source="esa_worldcover"),
        )
        _set_model(cfg, mock_sfincs_model)

        dl_dir = cfg.effective_download_dir

        def _fake_fetch(**kwargs):
            dl_dir.mkdir(parents=True, exist_ok=True)
            (dl_dir / "esa_worldcover.tif").write_bytes(b"\x00")
            cat = dl_dir / "esa_worldcover_catalog.yml"
            cat.write_text("meta: {}")
            return dl_dir / "esa_worldcover.tif", cat, "esa_worldcover"

        with patch("coastal_calibration.data.esa_worldcover.fetch_esa_worldcover") as mock_fetch:
            mock_fetch.side_effect = _fake_fetch
            stage = CreateFetchDataStage(cfg)
            result = stage.run()

        assert result["status"] == "completed"
        assert "esa_worldcover" in result["fetched"]
        mock_fetch.assert_called_once()
        _clear_model(cfg)

    def test_create_fetch_data_reuses_existing(
        self,
        aoi_file: Path,
        output_dir: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """When GeoTIFF + catalog already exist, skip fetch."""
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            elevation=ElevationConfig(
                datasets=[ElevationDataset(name="noaa_tb", source="noaa_3m", zmin=-20000)]
            ),
            subgrid=SubgridConfig(lulc_source=None),
        )
        _set_model(cfg, mock_sfincs_model)

        dl_dir = cfg.effective_download_dir
        dl_dir.mkdir(parents=True, exist_ok=True)
        (dl_dir / "noaa_tb.tif").write_bytes(b"\x00")
        (dl_dir / "noaa_tb_catalog.yml").write_text("meta: {}")

        with patch("coastal_calibration.data.topobathy_noaa.fetch_noaa_dem") as mock_fetch:
            stage = CreateFetchDataStage(cfg)
            result = stage.run()

        assert result["status"] == "completed"
        mock_fetch.assert_not_called()
        _clear_model(cfg)

    def test_backward_compat_alias(self) -> None:
        """CreateFetchElevationStage should be an alias for CreateFetchDataStage."""
        assert CreateFetchElevationStage is CreateFetchDataStage

    def test_create_stages_helper(self, minimal_create_config: SfincsCreateConfig) -> None:
        stages_dict = create_stages(minimal_create_config)
        assert set(stages_dict.keys()) == set(minimal_create_config.stage_order)
        for stage in stages_dict.values():
            assert isinstance(stage, CreateStage)


# ===================================================================
# Runner tests
# ===================================================================


class TestSfincsCreator:
    """Test the SfincsCreator runner."""

    def test_dry_run(self, minimal_create_config: SfincsCreateConfig) -> None:
        creator = SfincsCreator(minimal_create_config)
        result = creator.run(dry_run=True)
        assert result.success is True
        assert result.outputs.get("dry_run") is True

    def test_validation_failure(self, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=Path("/nonexistent/aoi.geojson"),
            output_dir=output_dir,
        )
        creator = SfincsCreator(cfg)
        result = creator.run()
        assert result.success is False
        assert any("AOI file not found" in e for e in result.errors)

    def test_start_from_stop_after(self, minimal_create_config: SfincsCreateConfig) -> None:
        creator = SfincsCreator(minimal_create_config)
        creator._init_stages()
        stages = creator._get_stages_to_run("create_mask", "create_subgrid")
        assert stages == ["create_mask", "create_boundary", "create_subgrid"]

    def test_unknown_stage_raises(self, minimal_create_config: SfincsCreateConfig) -> None:
        creator = SfincsCreator(minimal_create_config)
        creator._init_stages()
        with pytest.raises(ValueError, match="Unknown stage"):
            creator._get_stages_to_run("nonexistent", None)

    def test_status_tracking(self, minimal_create_config: SfincsCreateConfig) -> None:
        creator = SfincsCreator(minimal_create_config)
        assert creator._load_status() == {}
        creator._save_stage_status("create_grid")
        status = creator._load_status()
        assert "create_grid" in status["completed_stages"]

    def test_check_prerequisites_missing(self, minimal_create_config: SfincsCreateConfig) -> None:
        creator = SfincsCreator(minimal_create_config)
        errors = creator._check_prerequisites("create_elevation")
        assert len(errors) == 1
        assert "create_grid" in errors[0]


# ===================================================================
# CLI tests
# ===================================================================


class TestCLI:
    """Test the ``create`` and ``stages`` CLI commands."""

    def test_create_dry_run(self, minimal_config_yaml: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["create", str(minimal_config_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output

    def test_create_missing_config(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["create", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_stages_create(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stages", "--model", "create"])
        assert result.exit_code == 0
        assert "create_grid" in result.output
        assert "create_write" in result.output
        assert "create_roughness" not in result.output

    def test_stages_all_includes_create(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stages"])
        assert result.exit_code == 0
        assert "creation stages" in result.output.lower()


# ===================================================================
# NWM Discharge config tests
# ===================================================================


@pytest.fixture
def flowlines_geojson(tmp_path: Path) -> Path:
    """Create a minimal GeoJSON with flowpath linestrings."""
    import geopandas as gpd
    from shapely import LineString

    # Create flowpath linestrings that cross the AOI boundary
    # AOI is [-95.5, 29.0] to [-95.0, 29.5]
    gdf = gpd.GeoDataFrame(
        {"id": [1001, 1002, 1003]},
        geometry=[
            # Crosses left boundary of AOI
            LineString([(-96.0, 29.25), (-95.4, 29.25)]),
            # Crosses top boundary of AOI
            LineString([(-95.3, 29.6), (-95.3, 29.3)]),
            # Entirely inside AOI — no intersection with boundary
            LineString([(-95.3, 29.2), (-95.2, 29.2)]),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "flowlines.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def river_discharge_config(flowlines_geojson: Path) -> RiverDischargeConfig:
    """Return a RiverDischargeConfig with valid test values."""
    return RiverDischargeConfig(
        flowlines=flowlines_geojson,
        nwm_id_column="id",
        max_snap_distance_m=100_000.0,
    )


@pytest.fixture
def discharge_create_config(
    aoi_file: Path,
    output_dir: Path,
    river_discharge_config: RiverDischargeConfig,
) -> SfincsCreateConfig:
    """Return a SfincsCreateConfig with river_discharge configured."""
    return SfincsCreateConfig(
        aoi=aoi_file,
        output_dir=output_dir,
        river_discharge=river_discharge_config,
    )


class TestRiverDischargeConfig:
    """Test RiverDischargeConfig loading and validation."""

    def test_config_loads_with_river_discharge(
        self,
        tmp_path: Path,
        aoi_file: Path,
        output_dir: Path,
        flowlines_geojson: Path,
    ) -> None:
        cfg_data = {
            "aoi": str(aoi_file),
            "output_dir": str(output_dir),
            "river_discharge": {
                "flowlines": str(flowlines_geojson),
                "nwm_id_column": "id",
            },
        }
        cfg_path = tmp_path / "with_discharge.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        cfg = SfincsCreateConfig.from_yaml(cfg_path)
        assert cfg.river_discharge is not None
        assert cfg.river_discharge.nwm_id_column == "id"

    def test_relative_flowlines_path_resolved(
        self, tmp_path: Path, aoi_file: Path, flowlines_geojson: Path
    ) -> None:
        """Relative flowlines path resolves against YAML dir."""
        import shutil

        # Put the YAML in a subdirectory so the geojson copy isn't same-file.
        sub = tmp_path / "sub"
        sub.mkdir()
        local_geojson = sub / "flowlines.geojson"
        shutil.copy(flowlines_geojson, local_geojson)
        cfg_data = {
            "aoi": str(aoi_file),
            "output_dir": str(sub / "out"),
            "river_discharge": {
                "flowlines": "flowlines.geojson",
                "nwm_id_column": "id",
            },
        }
        cfg_path = sub / "rel.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        cfg = SfincsCreateConfig.from_yaml(cfg_path)
        assert cfg.river_discharge is not None
        assert cfg.river_discharge.flowlines == local_geojson.resolve()

    def test_validate_missing_flowlines(self, aoi_file: Path, output_dir: Path) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=Path("/nonexistent/flowlines.geojson"),
                nwm_id_column="id",
            ),
        )
        errors = cfg.validate()
        assert any("flowlines not found" in e for e in errors)

    def test_stage_order_includes_discharge(
        self, discharge_create_config: SfincsCreateConfig
    ) -> None:
        stages = discharge_create_config.stage_order
        assert "create_discharge" in stages
        assert stages.index("create_discharge") == stages.index("create_boundary") + 1
        assert stages.index("create_discharge") < stages.index("create_subgrid")

    def test_stage_order_excludes_discharge_when_none(
        self, minimal_create_config: SfincsCreateConfig
    ) -> None:
        assert minimal_create_config.river_discharge is None
        assert "create_discharge" not in minimal_create_config.stage_order

    def test_to_dict_with_discharge(self, discharge_create_config: SfincsCreateConfig) -> None:
        d = discharge_create_config.to_dict()
        assert d["river_discharge"] is not None
        assert d["river_discharge"]["nwm_id_column"] == "id"

    def test_to_dict_without_discharge(self, minimal_create_config: SfincsCreateConfig) -> None:
        d = minimal_create_config.to_dict()
        assert d["river_discharge"] is None


# ===================================================================
# Discharge stage validation tests
# ===================================================================


class TestDischargeStageValidation:
    """Test CreateDischargeStage.validate() with GeoJSON checks."""

    def test_validate_skips_when_no_config(self, minimal_create_config: SfincsCreateConfig) -> None:
        stage = CreateDischargeStage(minimal_create_config)
        errors = stage.validate()
        assert errors == []

    def test_validate_missing_flowlines(
        self,
        aoi_file: Path,
        output_dir: Path,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=Path("/nonexistent/flowlines.geojson"),
                nwm_id_column="id",
            ),
        )
        stage = CreateDischargeStage(cfg)
        errors = stage.validate()
        assert any("flowlines not found" in e for e in errors)

    def test_validate_wrong_extension(
        self,
        aoi_file: Path,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        bad_file = tmp_path / "flowlines.gpkg"
        bad_file.touch()
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=bad_file,
                nwm_id_column="id",
            ),
        )
        stage = CreateDischargeStage(cfg)
        errors = stage.validate()
        assert any(".geojson" in e for e in errors)

    def test_validate_missing_column(
        self,
        aoi_file: Path,
        output_dir: Path,
        flowlines_geojson: Path,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=flowlines_geojson,
                nwm_id_column="nonexistent_col",
            ),
        )
        stage = CreateDischargeStage(cfg)
        errors = stage.validate()
        assert any("nonexistent_col" in e and "not found" in e for e in errors)

    def test_validate_valid_geojson(
        self,
        aoi_file: Path,
        output_dir: Path,
        flowlines_geojson: Path,
    ) -> None:
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=flowlines_geojson,
                nwm_id_column="id",
            ),
        )
        stage = CreateDischargeStage(cfg)
        errors = stage.validate()
        assert errors == []


# ===================================================================
# Discharge stage execution tests
# ===================================================================


class TestDischargeStageRun:
    """Test CreateDischargeStage.run() with mocked model."""

    def test_run_skipped_when_no_config(
        self,
        minimal_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        _set_model(minimal_create_config, mock_sfincs_model)
        stage = CreateDischargeStage(minimal_create_config)
        result = stage.run()
        assert result["status"] == "skipped"
        _clear_model(minimal_create_config)

    def test_run_adds_discharge_points(
        self,
        discharge_create_config: SfincsCreateConfig,
        mock_sfincs_model: MagicMock,
    ) -> None:
        import numpy as np

        # Mock grid in UTM zone 15N — the run() method reprojects
        # EPSG:4326 GeoJSON data to this CRS automatically.
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:32615"
        # Face centers inside the AOI (UTM equivalents of the 4326 AOI).
        face_x = np.array([261677.0, 271629.0, 281563.0, 291478.0])
        face_y = np.array([3227087.0, 3237973.0, 3248867.0, 3259768.0])
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_x = face_x
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_y = face_y
        mask_mock = MagicMock()
        mask_mock.to_numpy.return_value = np.array([1, 1, 1, 1])
        mock_sfincs_model.quadtree_grid.data.__getitem__ = lambda self, k: (
            mask_mock if k == "mask" else MagicMock()
        )
        _set_model(discharge_create_config, mock_sfincs_model)

        stage = CreateDischargeStage(discharge_create_config)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["points_added"] > 0
        # Create stage writes a .src file but does NOT add points to the
        # model — the run stage handles that with real timeseries.
        mock_sfincs_model.discharge_points.add_point.assert_not_called()
        src_file = discharge_create_config.output_dir / "sfincs_nwm.src"
        assert src_file.exists()
        src_lines = src_file.read_text().strip().splitlines()
        assert len(src_lines) == result["points_added"]
        _clear_model(discharge_create_config)

    def test_run_empty_geojson(
        self,
        aoi_file: Path,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """An empty GeoJSON produces skipped status."""
        import geopandas as gpd

        empty_geojson = tmp_path / "empty.geojson"
        gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326").to_file(
            empty_geojson, driver="GeoJSON"
        )
        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=empty_geojson,
                nwm_id_column="id",
            ),
        )
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:32615"
        _set_model(cfg, mock_sfincs_model)
        stage = CreateDischargeStage(cfg)
        result = stage.run()
        assert result["status"] == "skipped"
        _clear_model(cfg)

    def test_run_flowpath_inside_aoi_uses_endpoint(
        self,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """Flowpath inside AOI uses downstream endpoint as discharge point."""
        import geopandas as gpd
        import numpy as np
        from shapely import LineString, box

        # AOI in EPSG:4326 (will be reprojected to model CRS by run())
        aoi = tmp_path / "aoi.geojson"
        gpd.GeoDataFrame(geometry=[box(-95.5, 29.0, -95.0, 29.5)], crs="EPSG:4326").to_file(
            aoi, driver="GeoJSON"
        )

        # Flowpath entirely inside the AOI — endpoint at (-95.2, 29.3)
        geojson = tmp_path / "inside.geojson"
        gdf = gpd.GeoDataFrame(
            {"id": [2001]},
            geometry=[LineString([(-95.3, 29.2), (-95.2, 29.3)])],
            crs="EPSG:4326",
        )
        gdf.to_file(geojson, driver="GeoJSON")

        cfg = SfincsCreateConfig(
            aoi=aoi,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=geojson,
                nwm_id_column="id",
                max_snap_distance_m=100_000.0,
            ),
        )
        # UTM equivalents of (-95.25, 29.25) and (-95.15, 29.35)
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:32615"
        face_x = np.array([281563.0, 291478.0])
        face_y = np.array([3248867.0, 3259768.0])
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_x = face_x
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_y = face_y
        mask_mock = MagicMock()
        mask_mock.to_numpy.return_value = np.array([1, 1])
        mock_sfincs_model.quadtree_grid.data.__getitem__ = lambda self, k: (
            mask_mock if k == "mask" else MagicMock()
        )
        _set_model(cfg, mock_sfincs_model)
        stage = CreateDischargeStage(cfg)
        result = stage.run()
        assert result["points_added"] == 1
        mock_sfincs_model.discharge_points.add_point.assert_not_called()
        src_file = cfg.output_dir / "sfincs_nwm.src"
        assert src_file.exists()
        assert len(src_file.read_text().strip().splitlines()) == 1
        _clear_model(cfg)

    def test_run_multilinestring_merged(
        self,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """MultiLineString geometries are merged into LineStrings."""
        import geopandas as gpd
        import numpy as np
        from shapely import MultiLineString, box

        aoi = tmp_path / "aoi.geojson"
        gpd.GeoDataFrame(geometry=[box(-95.5, 29.0, -95.0, 29.5)], crs="EPSG:4326").to_file(
            aoi, driver="GeoJSON"
        )

        # MultiLineString that should be merged into a single LineString
        geojson = tmp_path / "multi.geojson"
        gdf = gpd.GeoDataFrame(
            {"id": [3001]},
            geometry=[
                MultiLineString([[(-95.4, 29.2), (-95.3, 29.25)], [(-95.3, 29.25), (-95.2, 29.3)]])
            ],
            crs="EPSG:4326",
        )
        gdf.to_file(geojson, driver="GeoJSON")

        cfg = SfincsCreateConfig(
            aoi=aoi,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=geojson,
                nwm_id_column="id",
                max_snap_distance_m=100_000.0,
            ),
        )
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:32615"
        face_x = np.array([281563.0, 291478.0])
        face_y = np.array([3248867.0, 3259768.0])
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_x = face_x
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_y = face_y
        mask_mock = MagicMock()
        mask_mock.to_numpy.return_value = np.array([1, 1])
        mock_sfincs_model.quadtree_grid.data.__getitem__ = lambda self, k: (
            mask_mock if k == "mask" else MagicMock()
        )
        _set_model(cfg, mock_sfincs_model)
        stage = CreateDischargeStage(cfg)
        result = stage.run()
        assert result["points_added"] == 1
        _clear_model(cfg)

    def _make_outside_domain_cfg(
        self,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
        *,
        max_snap_distance_m: float = 2000.0,
    ) -> SfincsCreateConfig:
        """Create config with a flowpath far outside the AOI."""
        import geopandas as gpd
        import numpy as np
        from shapely import LineString, box

        # AOI in EPSG:4326 (will be reprojected to model CRS)
        aoi = tmp_path / "aoi.geojson"
        gpd.GeoDataFrame(geometry=[box(-95.5, 29.0, -95.0, 29.5)], crs="EPSG:4326").to_file(
            aoi, driver="GeoJSON"
        )

        # Flowpath entirely outside the AOI (endpoint at -96.0, 29.25)
        geojson = tmp_path / "outside.geojson"
        gdf = gpd.GeoDataFrame(
            {"id": [9001]},
            geometry=[LineString([(-96.5, 29.25), (-96.0, 29.25)])],
            crs="EPSG:4326",
        )
        gdf.to_file(geojson, driver="GeoJSON")

        cfg = SfincsCreateConfig(
            aoi=aoi,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(
                flowlines=geojson,
                nwm_id_column="id",
                max_snap_distance_m=max_snap_distance_m,
            ),
        )

        # Grid in UTM 15N — face centers inside the AOI.
        # The outside flowpath endpoint (-96.0, 29.25) projects to
        # ~(208436, 3239416) in UTM, which is ~54,650 m from the
        # nearest face center at (261677, 3227087).
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:32615"
        face_x = np.array([261677.0, 271629.0, 281563.0, 291478.0])
        face_y = np.array([3227087.0, 3237973.0, 3248867.0, 3259768.0])
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_x = face_x
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_y = face_y
        mask_mock = MagicMock()
        mask_mock.to_numpy.return_value = np.array([1, 1, 1, 1])
        mock_sfincs_model.quadtree_grid.data.__getitem__ = lambda self, k: (
            mask_mock if k == "mask" else MagicMock()
        )
        _set_model(cfg, mock_sfincs_model)
        return cfg

    def test_run_outside_domain_dropped_exceeds_max_snap_distance(
        self,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """Discharge point beyond max_snap_distance_m is dropped with a warning."""
        # Outside point is ~54,650 m from nearest face center — exceeds 2000 m.
        cfg = self._make_outside_domain_cfg(
            output_dir, tmp_path, mock_sfincs_model, max_snap_distance_m=2000.0
        )
        stage = CreateDischargeStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["points_added"] == 0
        _clear_model(cfg)

    def test_run_outside_domain_snaps_within_max_distance(
        self,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """Discharge point outside domain snaps when max_snap_distance_m is large enough."""
        # Outside point is ~54,650 m from nearest face — 100 km threshold allows it.
        cfg = self._make_outside_domain_cfg(
            output_dir, tmp_path, mock_sfincs_model, max_snap_distance_m=100_000.0
        )
        stage = CreateDischargeStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["points_added"] == 1

        # Verify snapped coordinates are on the grid (UTM), not at
        # the original outside-domain location (~208436 easting).
        src_file = cfg.output_dir / "sfincs_nwm.src"
        assert src_file.exists()
        src_line = src_file.read_text().strip()
        x_written, y_written = (float(v) for v in src_line.split()[:2])
        assert 256000 <= x_written <= 292000, f"Snapped x={x_written} should be inside the grid"
        assert 3_210_000 <= y_written <= 3_266_000, (
            f"Snapped y={y_written} should be inside the grid"
        )
        _clear_model(cfg)

    def test_run_geographic_crs_raises(
        self,
        aoi_file: Path,
        output_dir: Path,
        tmp_path: Path,
        mock_sfincs_model: MagicMock,
    ) -> None:
        """A geographic CRS (degree-based) raises ValueError."""
        import geopandas as gpd
        import numpy as np
        from shapely import LineString

        geojson = tmp_path / "flow.geojson"
        gpd.GeoDataFrame(
            {"id": [1001]},
            geometry=[LineString([(-96.0, 29.25), (-95.4, 29.25)])],
            crs="EPSG:4326",
        ).to_file(geojson, driver="GeoJSON")

        cfg = SfincsCreateConfig(
            aoi=aoi_file,
            output_dir=output_dir,
            river_discharge=RiverDischargeConfig(flowlines=geojson, nwm_id_column="id"),
        )

        mock_sfincs_model.quadtree_grid.data.ugrid.grid.crs = "EPSG:4326"
        face_x = np.array([-95.25])
        face_y = np.array([29.25])
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_x = face_x
        mock_sfincs_model.quadtree_grid.data.ugrid.grid.face_y = face_y
        mask_mock = MagicMock()
        mask_mock.to_numpy.return_value = np.array([1])
        mock_sfincs_model.quadtree_grid.data.__getitem__ = lambda self, k: (
            mask_mock if k == "mask" else MagicMock()
        )
        _set_model(cfg, mock_sfincs_model)

        stage = CreateDischargeStage(cfg)
        with pytest.raises(ValueError, match="geographic"):
            stage.run()
        _clear_model(cfg)

    def test_create_stages_includes_discharge(
        self, discharge_create_config: SfincsCreateConfig
    ) -> None:
        stages_dict = create_stages(discharge_create_config)
        assert "create_discharge" in stages_dict
        assert isinstance(stages_dict["create_discharge"], CreateDischargeStage)
