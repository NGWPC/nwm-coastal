"""Tests for SFINCS data catalog functionality in coastal_calibration.stages.sfincs.

Note: These tests cover the data catalog and symlink functionality,
NOT the SFINCS model build/run stages which require hydromt-sfincs.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import yaml

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    PathConfig,
    SchismModelConfig,
    SimulationConfig,
    SlurmConfig,
)
from coastal_calibration.stages.sfincs import (
    CatalogEntry,
    CatalogMetadata,
    DataAdapter,
    DataCatalog,
    create_nc_symlinks,
    generate_data_catalog,
    remove_nc_symlinks,
)
from coastal_calibration.stages.sfincs_build import SfincsDataCatalogStage


class TestDataAdapter:
    def test_to_dict_empty(self):
        da = DataAdapter()
        assert da.to_dict() == {}

    def test_to_dict_with_rename(self):
        da = DataAdapter(rename={"old": "new"})
        d = da.to_dict()
        assert d == {"rename": {"old": "new"}}

    def test_to_dict_full(self):
        da = DataAdapter(
            rename={"a": "b"},
            unit_mult={"c": 1.5},
            unit_add={"d": -273.15},
        )
        d = da.to_dict()
        assert "rename" in d
        assert "unit_mult" in d
        assert "unit_add" in d


class TestCatalogMetadata:
    def test_to_dict_empty(self):
        cm = CatalogMetadata()
        assert cm.to_dict() == {}

    def test_to_dict_with_fields(self):
        cm = CatalogMetadata(
            crs=4326,
            category="meteo",
            source_url="https://example.com",
            temporal_extent=("2021-01-01", "2021-12-31"),
        )
        d = cm.to_dict()
        assert d["crs"] == 4326
        assert d["category"] == "meteo"
        assert d["source_url"] == "https://example.com"
        assert d["temporal_extent"] == ["2021-01-01", "2021-12-31"]

    def test_to_dict_excludes_none(self):
        cm = CatalogMetadata(crs=4326)
        d = cm.to_dict()
        assert "source_url" not in d
        assert "notes" not in d


class TestCatalogEntry:
    def test_to_dict_minimal(self):
        entry = CatalogEntry(
            name="test",
            data_type="RasterDataset",
            driver="netcdf",
            uri="path/to/data.nc",
        )
        d = entry.to_dict()
        assert d["data_type"] == "RasterDataset"
        assert d["driver"] == "netcdf"
        assert d["uri"] == "path/to/data.nc"
        assert "metadata" not in d
        assert "data_adapter" not in d

    def test_to_dict_with_metadata_and_adapter(self):
        entry = CatalogEntry(
            name="test",
            data_type="GeoDataset",
            driver="zarr",
            uri="path/*.nc",
            metadata=CatalogMetadata(crs=4326),
            data_adapter=DataAdapter(rename={"a": "b"}),
            version="1.0",
        )
        d = entry.to_dict()
        assert "metadata" in d
        assert "data_adapter" in d
        assert d["version"] == "1.0"


class TestDataCatalog:
    def test_empty_catalog(self):
        cat = DataCatalog()
        d = cat.to_dict()
        assert d == {}

    def test_add_entry(self):
        cat = DataCatalog()
        entry = CatalogEntry(
            name="test_entry",
            data_type="RasterDataset",
            driver="netcdf",
            uri="data.nc",
        )
        cat.add_entry(entry)
        assert len(cat.entries) == 1
        d = cat.to_dict()
        assert "test_entry" in d

    def test_with_metadata(self):
        cat = DataCatalog(
            name="my_catalog",
            version="2.0",
            hydromt_version=">=0.9.0",
            roots=["/data"],
        )
        d = cat.to_dict()
        assert d["meta"]["name"] == "my_catalog"
        assert d["meta"]["version"] == "2.0"
        assert d["meta"]["roots"] == ["/data"]

    def test_to_yaml(self, tmp_path):
        cat = DataCatalog(name="test")
        entry = CatalogEntry(
            name="test_entry",
            data_type="RasterDataset",
            driver="netcdf",
            uri="data.nc",
        )
        cat.add_entry(entry)

        yaml_path = tmp_path / "catalog.yml"
        cat.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = yaml.safe_load(yaml_path.read_text())
        assert "test_entry" in loaded
        assert loaded["meta"]["name"] == "test"


class TestGenerateDataCatalog:
    @pytest.fixture
    def catalog_config(self, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        dl_dir = tmp_path / "downloads"
        dl_dir.mkdir()

        return CoastalCalibConfig(
            slurm=SlurmConfig(user="test"),
            simulation=SimulationConfig(
                start_date=datetime(2021, 6, 11),
                duration_hours=3,
                coastal_domain="pacific",
                meteo_source="nwm_retro",
            ),
            boundary=BoundaryConfig(source="stofs"),
            paths=PathConfig(work_dir=work_dir, raw_download_dir=dl_dir),
            model_config=SchismModelConfig(),
            download=DownloadConfig(enabled=False),
        )

    def test_generate_all(self, catalog_config, tmp_path):
        catalog = generate_data_catalog(catalog_config)
        assert len(catalog.entries) == 3  # meteo, streamflow, coastal
        names = [e.name for e in catalog.entries]
        assert "nwm_retro_meteo" in names
        assert "nwm_retro_streamflow" in names
        assert "stofs_waterlevel" in names

    def test_generate_meteo_only(self, catalog_config):
        catalog = generate_data_catalog(
            catalog_config,
            include_meteo=True,
            include_streamflow=False,
            include_coastal=False,
        )
        assert len(catalog.entries) == 1
        assert catalog.entries[0].name == "nwm_retro_meteo"

    def test_generate_with_output_path(self, catalog_config, tmp_path):
        output = tmp_path / "cat.yml"
        generate_data_catalog(catalog_config, output_path=output)
        assert output.exists()

    def test_generate_glofs_source(self, catalog_config):
        catalog = generate_data_catalog(
            catalog_config,
            coastal_source="glofs",
            glofs_model="leofs",
            include_meteo=False,
            include_streamflow=False,
        )
        assert len(catalog.entries) == 1
        assert "glofs" in catalog.entries[0].name

    def test_generate_tpxo_source(self, catalog_config):
        """TPXO forcing is handled directly by SfincsForcingStage, not via the catalog."""
        catalog = generate_data_catalog(
            catalog_config,
            coastal_source="tpxo",
            include_meteo=False,
            include_streamflow=False,
        )
        # No TPXO catalog entry — predict_tide runs outside HydroMT
        assert len(catalog.entries) == 0

    def test_catalog_name_and_version(self, catalog_config):
        catalog = generate_data_catalog(
            catalog_config,
            catalog_name="custom",
            catalog_version="3.0",
        )
        assert catalog.name == "custom"
        assert catalog.version == "3.0"

    def test_nwm_ana_meteo_uri_uses_ldasin_glob(self, tmp_path):
        """nwm_ana meteo entry should use *.LDASIN_DOMAIN1.nc glob (same as nwm_retro)."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        dl_dir = tmp_path / "downloads"
        dl_dir.mkdir()

        cfg = CoastalCalibConfig(
            slurm=SlurmConfig(user="test"),
            simulation=SimulationConfig(
                start_date=datetime(2021, 6, 11),
                duration_hours=3,
                coastal_domain="atlgulf",
                meteo_source="nwm_ana",
            ),
            boundary=BoundaryConfig(source="stofs"),
            paths=PathConfig(work_dir=work_dir, raw_download_dir=dl_dir),
            model_config=SchismModelConfig(),
            download=DownloadConfig(enabled=False),
        )
        catalog = generate_data_catalog(
            cfg,
            include_meteo=True,
            include_streamflow=False,
            include_coastal=False,
        )
        entry = catalog.entries[0]
        assert entry.name == "nwm_ana_meteo"
        assert entry.uri.endswith("*.LDASIN_DOMAIN1.nc")


class TestCreateNcSymlinks:
    def test_creates_meteo_symlinks(self, tmp_path):
        meteo_dir = tmp_path / "meteo" / "nwm_retro"
        meteo_dir.mkdir(parents=True)
        (meteo_dir / "2021061100.LDASIN_DOMAIN1").write_text("data")
        (meteo_dir / "2021061101.LDASIN_DOMAIN1").write_text("data")

        result = create_nc_symlinks(tmp_path, include_streamflow=False)
        assert len(result["meteo"]) == 2
        for link in result["meteo"]:
            assert link.name.endswith(".LDASIN_DOMAIN1.nc")
            assert link.is_symlink()

    def test_creates_streamflow_symlinks_retro(self, tmp_path):
        stream_dir = tmp_path / "streamflow" / "nwm_retro"
        stream_dir.mkdir(parents=True)
        (stream_dir / "202106110000.CHRTOUT_DOMAIN1").write_text("data")

        result = create_nc_symlinks(tmp_path, include_meteo=False)
        assert len(result["streamflow"]) == 1
        assert result["streamflow"][0].name.endswith(".CHRTOUT_DOMAIN1.nc")

    def test_creates_streamflow_symlinks_ana(self, tmp_path):
        stream_dir = tmp_path / "hydro" / "nwm"
        stream_dir.mkdir(parents=True)
        (stream_dir / "202306010000.CHRTOUT_DOMAIN1").write_text("data")

        result = create_nc_symlinks(
            tmp_path,
            meteo_source="nwm_ana",
            include_meteo=False,
        )
        assert len(result["streamflow"]) == 1

    def test_skip_existing_symlinks(self, tmp_path):
        meteo_dir = tmp_path / "meteo" / "nwm_retro"
        meteo_dir.mkdir(parents=True)
        original = meteo_dir / "2021061100.LDASIN_DOMAIN1"
        original.write_text("data")
        link = meteo_dir / "2021061100.LDASIN_DOMAIN1.nc"
        link.symlink_to(original.name)

        result = create_nc_symlinks(tmp_path, include_streamflow=False)
        assert len(result["meteo"]) == 0  # Already existed

    def test_creates_meteo_symlinks_for_nwm_ana(self, tmp_path):
        """nwm_ana downloads use LDASIN naming — .nc symlinks are needed."""
        meteo_dir = tmp_path / "meteo" / "nwm_ana"
        meteo_dir.mkdir(parents=True)
        (meteo_dir / "2021042100.LDASIN_DOMAIN1").write_text("data")
        (meteo_dir / "2021042101.LDASIN_DOMAIN1").write_text("data")

        result = create_nc_symlinks(
            tmp_path,
            meteo_source="nwm_ana",
            include_streamflow=False,
        )
        assert len(result["meteo"]) == 2
        for link in result["meteo"]:
            assert link.name.endswith(".LDASIN_DOMAIN1.nc")
            assert link.is_symlink()

    def test_nonexistent_dir(self, tmp_path):
        result = create_nc_symlinks(tmp_path)
        assert result["meteo"] == []
        assert result["streamflow"] == []


class TestRemoveNcSymlinks:
    def test_removes_meteo_symlinks(self, tmp_path):
        meteo_dir = tmp_path / "meteo" / "nwm_retro"
        meteo_dir.mkdir(parents=True)
        original = meteo_dir / "2021061100.LDASIN_DOMAIN1"
        original.write_text("data")
        link = meteo_dir / "2021061100.LDASIN_DOMAIN1.nc"
        link.symlink_to(original.name)

        result = remove_nc_symlinks(tmp_path, include_streamflow=False)
        assert result["meteo"] == 1
        assert not link.exists()
        assert original.exists()

    def test_removes_streamflow_symlinks(self, tmp_path):
        stream_dir = tmp_path / "streamflow" / "nwm_retro"
        stream_dir.mkdir(parents=True)
        original = stream_dir / "202106110000.CHRTOUT_DOMAIN1"
        original.write_text("data")
        link = stream_dir / "202106110000.CHRTOUT_DOMAIN1.nc"
        link.symlink_to(original.name)

        result = remove_nc_symlinks(tmp_path, include_meteo=False)
        assert result["streamflow"] == 1

    def test_nonexistent_dir(self, tmp_path):
        result = remove_nc_symlinks(tmp_path)
        assert result["meteo"] == 0
        assert result["streamflow"] == 0

    def test_does_not_remove_real_files(self, tmp_path):
        meteo_dir = tmp_path / "meteo" / "nwm_retro"
        meteo_dir.mkdir(parents=True)
        # Create a real file with .nc extension (not a symlink)
        real_file = meteo_dir / "2021061100.LDASIN_DOMAIN1.nc"
        real_file.write_text("real data")

        result = remove_nc_symlinks(tmp_path, include_streamflow=False)
        assert result["meteo"] == 0
        assert real_file.exists()


class TestSfincsDataCatalogStage:
    def test_validate_download_dir_missing(self, sample_config, tmp_path):
        sample_config.paths.raw_download_dir = tmp_path / "nonexistent"
        stage = SfincsDataCatalogStage(sample_config)
        errors = stage.validate()
        assert any("does not exist" in e for e in errors)

    def test_validate_download_dir_exists(self, sample_config):
        stage = SfincsDataCatalogStage(sample_config)
        errors = stage.validate()
        assert len(errors) == 0
