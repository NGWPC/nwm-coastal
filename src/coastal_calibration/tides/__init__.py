"""SCHISM tidal boundary condition utilities.

Public API
----------
make_otps_input
    Write an OTPSnc input file for TPXO tidal predictions.
otps_to_open_bnds
    Convert OTPSnc output to SCHISM ``elev2D.th.nc``.
generate_ocean_tide
    Extend boundary conditions with tidal predictions for long forecasts.
"""

from __future__ import annotations

from pathlib import Path

from coastal_calibration.tides._ocean_tide import generate_ocean_tide
from coastal_calibration.tides._otps import make_otps_input, otps_to_open_bnds

# Directory containing bundled OTPS configuration templates
# (setup_tpxo.txt, Model_tpxo10_atlas)
TIDES_DATA_DIR: Path = Path(__file__).resolve().parent

__all__ = [
    "TIDES_DATA_DIR",
    "generate_ocean_tide",
    "make_otps_input",
    "otps_to_open_bnds",
]
