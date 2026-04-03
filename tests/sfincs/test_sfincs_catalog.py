"""Tests for SFINCS data catalog functionality in coastal_calibration.sfincs.data_catalog.

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
)
from coastal_calibration.sfincs.data_catalog import (
    CatalogEntry,
    CatalogMetadata,
    DataAdapter,
    DataCatalog,
    _build_coastal_stofs_entry,
    _stofs_uri,
    create_nc_symlinks,
    generate_data_catalog,
    remove_nc_symlinks,
)
from coastal_calibration.sfincs.stages import SfincsDataCatalogStage


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

        created, existing = create_nc_symlinks(tmp_path, include_streamflow=False)
        assert len(created["meteo"]) == 2
        for link in created["meteo"]:
            assert link.name.endswith(".LDASIN_DOMAIN1.nc")
            assert link.is_symlink()
        assert existing["meteo"] == 0

    def test_creates_streamflow_symlinks_retro(self, tmp_path):
        stream_dir = tmp_path / "streamflow" / "nwm_retro"
        stream_dir.mkdir(parents=True)
        (stream_dir / "202106110000.CHRTOUT_DOMAIN1").write_text("data")

        created, _existing = create_nc_symlinks(tmp_path, include_meteo=False)
        assert len(created["streamflow"]) == 1
        assert created["streamflow"][0].name.endswith(".CHRTOUT_DOMAIN1.nc")

    def test_creates_streamflow_symlinks_ana(self, tmp_path):
        stream_dir = tmp_path / "hydro" / "nwm"
        stream_dir.mkdir(parents=True)
        (stream_dir / "202306010000.CHRTOUT_DOMAIN1").write_text("data")

        created, _existing = create_nc_symlinks(
            tmp_path,
            meteo_source="nwm_ana",
            include_meteo=False,
        )
        assert len(created["streamflow"]) == 1

    def test_skip_existing_symlinks(self, tmp_path):
        meteo_dir = tmp_path / "meteo" / "nwm_retro"
        meteo_dir.mkdir(parents=True)
        original = meteo_dir / "2021061100.LDASIN_DOMAIN1"
        original.write_text("data")
        link = meteo_dir / "2021061100.LDASIN_DOMAIN1.nc"
        link.symlink_to(original.name)

        created, existing = create_nc_symlinks(tmp_path, include_streamflow=False)
        assert len(created["meteo"]) == 0
        assert existing["meteo"] == 1  # Reports the pre-existing symlink

    def test_creates_meteo_symlinks_for_nwm_ana(self, tmp_path):
        """nwm_ana downloads use LDASIN naming — .nc symlinks are needed."""
        meteo_dir = tmp_path / "meteo" / "nwm_ana"
        meteo_dir.mkdir(parents=True)
        (meteo_dir / "2021042100.LDASIN_DOMAIN1").write_text("data")
        (meteo_dir / "2021042101.LDASIN_DOMAIN1").write_text("data")

        created, _existing = create_nc_symlinks(
            tmp_path,
            meteo_source="nwm_ana",
            include_streamflow=False,
        )
        assert len(created["meteo"]) == 2
        for link in created["meteo"]:
            assert link.name.endswith(".LDASIN_DOMAIN1.nc")
            assert link.is_symlink()

    def test_nonexistent_dir(self, tmp_path):
        created, existing = create_nc_symlinks(tmp_path)
        assert created["meteo"] == []
        assert created["streamflow"] == []
        assert existing["meteo"] == 0
        assert existing["streamflow"] == 0


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


class TestStofsUri:
    """Tests for ``_stofs_uri`` — URI builder for STOFS catalog entries.

    Regression tests for a bug where the STOFS data catalog entry used a
    recursive glob (``stofs/**/*.fields.cwl.nc``) that matched all cached
    files.  When the shared download cache contained STOFS files from
    different mesh versions (pre/post v2.1, mid-January 2024), xarray
    could not concatenate them because the ``nbou`` / ``node`` / ``nvel``
    dimensions differed in size.  The fix replaced the glob with an exact
    path computed by ``_stofs_uri``.
    """

    def test_new_naming_convention(self):
        """Dates after 2023-01-08 should use ``stofs_2d_glo`` naming."""
        sim = SimulationConfig(
            start_date=datetime(2024, 1, 9),
            duration_hours=60,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        )
        uri = _stofs_uri(sim)
        assert uri == "coastal/stofs/stofs_2d_glo.20240109/stofs_2d_glo.t00z.fields.cwl.nc"

    def test_old_naming_convention(self):
        """Dates before 2023-01-08 should use ``estofs`` naming."""
        sim = SimulationConfig(
            start_date=datetime(2022, 6, 1),
            duration_hours=3,
            coastal_domain="atlgulf",
            meteo_source="nwm_retro",
        )
        uri = _stofs_uri(sim)
        assert uri == "coastal/stofs/estofs.20220601/estofs.t00z.fields.cwl.nc"

    def test_cycle_hour_rounding(self):
        """Non-cycle hours should round down to the nearest 6-hour cycle."""
        sim = SimulationConfig(
            start_date=datetime(2024, 9, 1, 14),
            duration_hours=3,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        )
        uri = _stofs_uri(sim)
        assert "t12z" in uri

    def test_uri_matches_downloader_path(self):
        """URI must match the layout produced by ``get_stofs_path``."""
        from pathlib import Path

        from coastal_calibration.data.downloader import get_stofs_path

        for start_date in [
            datetime(2022, 3, 15),
            datetime(2024, 1, 9),
            datetime(2025, 6, 1, 6),
        ]:
            sim = SimulationConfig(
                start_date=start_date,
                duration_hours=3,
                coastal_domain="atlgulf",
                meteo_source="nwm_ana",
            )
            uri = _stofs_uri(sim)
            expected = str(get_stofs_path(start_date, Path()))
            assert uri == expected, f"URI mismatch for {start_date}: {uri!r} != {expected!r}"

    def test_uri_is_not_a_glob(self):
        """Regression: URI must not contain wildcard characters."""
        sim = SimulationConfig(
            start_date=datetime(2024, 1, 9),
            duration_hours=60,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        )
        uri = _stofs_uri(sim)
        assert "*" not in uri
        assert "?" not in uri


class TestStofsEntryDropVariables:
    """Verify the STOFS catalog entry drops all mesh-topology variables.

    Regression tests for a bug where ADCIRC mesh topology variables
    (``nvell``, ``ibtype``, ``nbvv``, ``max_nvell``) were not dropped,
    causing xarray to fail when files from different STOFS mesh versions
    were loaded together.  Even with a specific URI (no glob), dropping
    these unused variables reduces memory for the ~12-million-node mesh.
    """

    def _get_stofs_entry(self, start_date=None):
        sim = SimulationConfig(
            start_date=start_date or datetime(2024, 1, 9),
            duration_hours=60,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        )
        return _build_coastal_stofs_entry(sim)

    def test_drops_mesh_topology_variables(self):
        """All ADCIRC mesh topology variables must be dropped."""
        entry = self._get_stofs_entry()
        drop = entry.driver["options"]["drop_variables"]
        for var in ("adcirc_mesh", "element", "mesh"):
            assert var in drop, f"{var!r} must be dropped"

    def test_drops_nbou_dimension_variables(self):
        """Variables on the ``nbou`` dimension must be dropped.

        The ``nbou`` dimension (number of flow boundary segments) changed
        from 2530 to 262 when STOFS-2D-Global updated its mesh (v2.1,
        mid-January 2024).
        """
        entry = self._get_stofs_entry()
        drop = entry.driver["options"]["drop_variables"]
        for var in ("nvell", "ibtype"):
            assert var in drop, f"{var!r} (nbou dim) must be dropped"

    def test_drops_nvel_dimension_variables(self):
        """Variables on the ``nvel`` dimension must be dropped.

        ``nvel`` is used as both a scalar variable and a dimension in
        STOFS files — the scalar is dropped to avoid the clash, and
        ``nbvv`` / ``max_nvell`` are dropped because they ride on the
        ``nvel``/``nbou`` dimensions.
        """
        entry = self._get_stofs_entry()
        drop = entry.driver["options"]["drop_variables"]
        for var in ("nvel", "nbvv", "max_nvell"):
            assert var in drop, f"{var!r} (nvel dim) must be dropped"

    def test_drops_depth_variable(self):
        """Bathymetry ``depth`` is unused and inflates memory."""
        entry = self._get_stofs_entry()
        drop = entry.driver["options"]["drop_variables"]
        assert "depth" in drop

    def test_entry_uses_specific_uri(self):
        """Regression: the entry URI must be a specific path, not a glob."""
        entry = self._get_stofs_entry()
        assert "*" not in entry.uri
        assert "**" not in entry.uri
        assert entry.uri.endswith(".fields.cwl.nc")

    def test_catalog_stofs_entry_uri_varies_by_date(self):
        """Different simulation dates must produce different URIs."""
        entry_a = self._get_stofs_entry(datetime(2024, 1, 9))
        entry_b = self._get_stofs_entry(datetime(2024, 9, 1))
        assert entry_a.uri != entry_b.uri
        assert "20240109" in entry_a.uri
        assert "20240901" in entry_b.uri
