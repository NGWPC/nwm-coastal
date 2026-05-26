"""Root conftest — only environment setup and truly shared fixtures."""

from __future__ import annotations

import os

# Force non-interactive backend for the entire test suite so stages that
# trigger matplotlib indirectly (e.g., via hydromt-sfincs) never attempt
# to open a display and hang the process.  Set via env var to avoid
# importing matplotlib at collection time (which can stall on CI).
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Create a temporary work directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return work_dir


@pytest.fixture
def tmp_download_dir(tmp_path):
    """Create a temporary download directory."""
    dl_dir = tmp_path / "downloads"
    dl_dir.mkdir()
    return dl_dir
