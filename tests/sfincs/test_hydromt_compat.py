"""Tests for coastal_calibration.sfincs._hydromt_compat."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path


class TestQuadtreeOutputPatch:
    """``patch_quadtree_output_read`` only rescues structured (n, m) files."""

    def _patched_reader(self, monkeypatch, boom: Exception):
        """Install the patch over a stub reader that always raises *boom*."""
        import sys
        import types

        module = types.ModuleType("hydromt_sfincs.components.output")

        class SfincsOutput:
            def read_map_file(self, fn_map="sfincs_map.nc", drop=None, **kwargs):
                raise boom

        module.SfincsOutput = SfincsOutput  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setitem(sys.modules, "hydromt_sfincs.components.output", module)

        from coastal_calibration.sfincs._hydromt_compat import patch_quadtree_output_read

        patch_quadtree_output_read()
        return SfincsOutput

    def test_non_structured_failure_propagates(self, tmp_path: Path, monkeypatch):
        """A genuine UGRID read failure must not be masked as "no data"."""
        reader = self._patched_reader(monkeypatch, RuntimeError("corrupt UGRID"))
        xr.Dataset({"zs": (("time", "face"), np.zeros((1, 2)))}).to_netcdf(
            tmp_path / "sfincs_map.nc"
        )

        model = type("M", (), {"grid_type": "quadtree"})()
        with pytest.raises(RuntimeError, match="corrupt UGRID"):
            reader.read_map_file(
                type("S", (), {"model": model})(), fn_map=str(tmp_path / "sfincs_map.nc")
            )
