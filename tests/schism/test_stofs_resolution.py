"""Tests for STOFS boundary file resolution in :mod:`coastal_calibration.schism.boundary`."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    PathConfig,
    SchismModelConfig,
    SimulationConfig,
)
from coastal_calibration.data.downloader import get_stofs_path
from coastal_calibration.schism.boundary import STOFSBoundaryStage

if TYPE_CHECKING:
    from pathlib import Path

# 2022 uses the ``estofs`` product, 2024 uses ``stofs_2d_glo``.
OLD_RUN = datetime(2022, 9, 28)
NEW_RUN = datetime(2024, 1, 9, 12)


def _stage(tmp_path: Path, start_date: datetime) -> STOFSBoundaryStage:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    config = CoastalCalibConfig(
        simulation=SimulationConfig(
            start_date=start_date,
            duration_hours=3,
            coastal_domain="atlgulf",
            meteo_source="nwm_ana",
        ),
        boundary=BoundaryConfig(source="stofs"),
        paths=PathConfig(work_dir=work_dir, raw_download_dir=tmp_path / "downloads"),
        model_config=SchismModelConfig(),
        download=DownloadConfig(enabled=False),
    )
    return STOFSBoundaryStage(config)


def _cache(tmp_path: Path, start_date: datetime) -> Path:
    path = get_stofs_path(start_date, (tmp_path / "downloads").resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{start_date:%Y-%m-%d}")
    return path


class TestResolveStofsFile:
    def test_uses_the_file_for_this_start_date(self, tmp_path: Path):
        _cache(tmp_path, OLD_RUN)
        expected = _cache(tmp_path, NEW_RUN)

        assert _stage(tmp_path, NEW_RUN)._resolve_stofs_file() == expected

    def test_missing_file_raises_instead_of_substituting(self, tmp_path: Path):
        """Regression: any cached STOFS file used to satisfy any run.

        ``estofs`` also sorts before ``stofs_2d_glo``, so the old product
        won every time once both were cached.
        """
        _cache(tmp_path, OLD_RUN)

        with pytest.raises(FileNotFoundError, match="2024-01-09"):
            _stage(tmp_path, NEW_RUN)._resolve_stofs_file()

    def test_explicit_config_path_still_wins(self, tmp_path: Path):
        override = tmp_path / "elsewhere.nc"
        override.write_text("explicit")
        stage = _stage(tmp_path, NEW_RUN)
        stage.config.boundary.stofs_file = override

        assert stage._resolve_stofs_file() == override
