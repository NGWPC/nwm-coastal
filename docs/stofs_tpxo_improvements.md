# STOFS Boundary Condition Pipeline: Analysis and Improvement Plan

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Current implementation](#current-implementation)
3. [Root cause analysis](#root-cause-analysis)
4. [STOFS data availability](#stofs-data-availability)
5. [Issues with TPXO tidal prediction](#issues-with-tpxo-tidal-prediction)
6. [Proposed solution](#proposed-solution)

---

## Problem statement

The boundary condition pipeline has three issues that affect
simulation quality, portability, and scalability:

1. **Blind TPXO fallback for simulations > 180 hours.** When
   `boundary.source == "stofs"` and the simulation exceeds 180 hours,
   the code unconditionally replaces hours 181+ with tidal-only
   predictions. This degrades quality for retrospective runs where
   STOFS data covers the full period; the tidal-only fill loses
   storm surge, wind setup, and pressure effects that STOFS captures.

2. **Single-cycle download.** The downloader fetches one STOFS
   forecast file (matching the simulation start time) regardless of
   duration.  Each STOFS file covers at most 180 hours.  For longer
   simulations, additional STOFS cycles are needed but never
   downloaded.

3. **Fragile and inaccurate TPXO prediction stack.** The tidal
   prediction path relies on a compiled Fortran binary
   (`predict_tide`) invoked via subprocess, a bundled legacy Python
   library (`pytides`) with only 8 constituents, and ad-hoc spatial
   interpolation via `scipy.griddata` that ignores the native TPXO
   grid structure.  This is both a portability problem (binary
   dependency) and an accuracy problem.

---

## Current implementation

### Download (`downloader.py`)

`_build_stofs_urls()` constructs a single URL based on the simulation
start date and the nearest 6-hourly cycle:

```text
s3://noaa-gestofs-pds/stofs_2d_glo.{YYYYMMDD}/stofs_2d_glo.t{HH}z.fields.cwl.nc
```

Only one file is downloaded per simulation.  The end date / duration
is not considered.

### Regridding (`regrid_estofs.py`)

`regrid_estofs` receives the single STOFS file, a cycle date/time,
and `--length-hrs`.  It extracts time steps starting at a fixed
forecast offset (`FORECAST_START = 5`) and interpolates the
unstructured STOFS grid onto the SCHISM open boundary nodes via ESMF.
Output: `elev2D.th.nc`.

If the requested duration exceeds the file's available time steps,
the regridder produces output only for the hours present in the file
(up to ~180 hours of forecast).

### TPXO fallback (`schism_prep.py`, lines 907–924)

```python
raw_length = abs(int(duration_hours))
if raw_length > 180:
    generate_ocean_tide(
        hgrid_gr3=..., output_file=...,
        start_dt=..., duration_hours=raw_length,
        tidal_constants_dir=...,
    )
```

This unconditionally overwrites `elev2D.th.nc` from hour 181 onward
with tidal-only predictions.  There is no check for whether the
simulation is retrospective (STOFS data exists for the full window)
or prospective (a real-time forecast beyond the STOFS horizon).

### `pytides` tidal fill (`_ocean_tide.py`)

`generate_ocean_tide()`:

1. Reads 8 separate TPXO constituent grid files (`k1.nc` …
   `s2.nc`).
2. Interpolates amplitude/phase to SCHISM boundary nodes via
   `scipy.griddata` (linear, ignoring the native Arakawa C-grid
   staggering).
3. Sums harmonics for 8 constituents using the bundled `pytides`
   library.
4. Opens `elev2D.th.nc` in append mode and overwrites from index 181.

### TPXO boundary path (`make_tpxo_boundary`)

When `boundary.source == "tpxo"`, the code shells out to the Fortran
`predict_tide` binary, then converts its text output to
`elev2D.th.nc` via `otps_to_open_bnds()`.  This path is functionally
correct but carries a compiled binary dependency that complicates
portability and CI.

---

## Root cause analysis

The 180-hour threshold originates from the **NWM operational forecast
pipeline on WCOSS/AWS**.  In that context:

- STOFS-2D-Global produces forecasts to exactly 180 hours (7.5 days),
  4 cycles per day.
- NWM medium-range forecasts extend to 240 hours and extended-range
  to 720 hours.
- For the hours beyond the STOFS forecast horizon (181+), the only
  option was tidal-only extrapolation.

This was correct for real-time operational use.  The code was ported
to this package without accounting for retrospective runs, where
overlapping STOFS cycles can be stitched to cover any historical
period.

---

## STOFS data availability

| Property | Value |
| -------- | ----- |
| Archive start | 2020-12-30 (`estofs` naming) / 2023-01-08 (`stofs_2d_glo` naming) |
| Forecast horizon | 180 hours per cycle |
| Cycle frequency | Every 6 hours (00, 06, 12, 18 UTC) |
| Archive location | `s3://noaa-gestofs-pds` (public, no auth) |
| File format | Unstructured NetCDF (ADCIRC triangular mesh) |
| File size | ~12 GB per cycle (global) |
| Subsetting API | None (no OPeNDAP/THREDDS endpoint for STOFS-2D-Global) |

For any retrospective simulation starting after 2020-12-30, STOFS
data exists for the entire period.  A 240-hour simulation starting
2024-01-09T00Z can be covered by:

- Cycle 2024-01-09T00Z → hours 0–180
- Cycle 2024-01-09T06Z → hours 6–186 (overlap provides hours
  180–186)
- Cycle 2024-01-09T12Z → hours 12–192 (provides hours 186–192)
- … and so on every 6 hours.

---

## Issues with TPXO tidal prediction

Even when the TPXO fallback is genuinely needed (prospective
forecasts beyond the STOFS horizon), the current implementation has
significant quality and portability problems:

### Quality limitations

- **Only 8 tidal constituents** (K1, K2, M2, N2, O1, P1, Q1, S2).
  The TPXO10 atlas provides 32 constituents and 18 additional minor
  constituents can be inferred via admittance methods.  Using only 8
  loses significant tidal energy, especially in regions with strong
  shallow-water harmonics (M4, M6, MS4).
- **No minor constituent inference.**  The OTPS Fortran code and
  modern implementations like `pyTMD` use Richard Ray's PERTH2
  admittance method to infer 18 minor constituents from the 8 major
  ones.  The current `pytides` path does not do this.
- **Incorrect spatial interpolation.**  `scipy.griddata` with
  `method='linear'` treats the TPXO data as scattered points.  TPXO
  uses an Arakawa C-grid where elevation, u-transport, and
  v-transport live on different staggered nodes.  Proper bilinear
  interpolation on the C-grid (with land masking and periodic
  longitude wrapping) is required for correct results.
- **Node-by-node loop.**  The current code loops over each boundary
  node individually (`for i in range(amp.shape[0])`) to create a
  `Tide` object and call `tide.at()`.  A vectorized implementation
  over all nodes would be significantly faster.
- **No solid Earth tide correction.**  The OTPS Fortran code applies
  a `BETA_SE` scaling factor (~0.94–0.954) to remove first-order
  load tides.  `pytides` does not.

### Portability problems

- **Compiled Fortran binary dependency.**  The `predict_tide` binary
  must be compiled from the OTPS source code (or provided via pixi).
  This complicates CI, cross-platform builds, and container-less
  deployment.
- **Text file I/O pipeline.**  The current TPXO path writes a text
  input file (`otps_lat_lon_time.txt`), invokes `predict_tide` via
  subprocess, parses its text output (`otps_out.txt`), then converts
  to NetCDF.  This is fragile and slow.
- **Separate constituent files.**  The `pytides` fallback path requires
  8 separate TPXO constituent files (`k1.nc` … `s2.nc`) in a
  specific directory, separate from the atlas files used by
  `predict_tide`.  This creates a confusing dual data requirement.

### Suspected bugs in the OTPS Fortran code

Preliminary review of the OTPS Fortran source (`subs.f90`,
`arguments` subroutine, originally by Richard Ray / NASA GSFC) has
identified three suspected bugs in the nodal correction factor
computation that require systematic investigation and quantification:

1. **L2 nodal factor: possible missing radian conversion.**  The
   `sin(2p)` term in the L2 amplitude factor may omit the
   degree-to-radian conversion while the adjacent `cos(2p*rad)`
   term applies it.  L2 is absent from the TPXO10 atlas v2 set
   but would affect non-atlas models.

2. **MS4 nodal factor: possible wrong compound formula.**  MS4 is
   a compound of M2 and S2, so its factor should be `f_M2 * f_S2`.
   The code appears to assign `f_M2^2` (the M4/MN4 factor)
   instead.  MS4 IS present in the TPXO10 atlas.

3. **M3 nodal factor: possible hardcoded value.**  The M3 factor
   appears hardcoded to 1.0 rather than `f_M2^1.5`.  Absent from
   the TPXO10 atlas v2 set.

These need to be confirmed by direct comparison of the Fortran
output against an independent implementation (e.g., `pyTMD`) across
the full 18.6-year nodal cycle and at stations spanning different
tidal regimes.  If confirmed, the errors would be systematic and
oscillatory, scaling with tidal range, providing further motivation
for replacing the Fortran binary with a validated pure-Python
implementation where such bugs can be corrected.

### What a proper Python replacement needs

A pure-Python TPXO prediction module that:

1. Reads the standard TPXO10 atlas NetCDF files directly (the same
   files `predict_tide` uses).
2. Implements proper Arakawa C-grid bilinear interpolation with land
   masking and periodic longitude wrapping.
3. Computes astronomical arguments and nodal corrections using the
   Cartwright & Tayler formulas, with any confirmed Fortran bugs
   corrected.
4. Supports all 32 TPXO constituents.
5. Infers 18 minor constituents via the PERTH2 admittance method.
6. Applies solid Earth tide corrections.
7. Vectorizes over all locations and times for performance.
8. Returns NumPy arrays directly; no subprocess, no text parsing.
9. Is validated against the Fortran OTPS binary and cross-validated
   against `pyTMD` for numerical parity.

---

## Proposed solution

### Phase 1: Fix the TPXO fallback logic

Replace the blind `> 180` check with a data-availability test:

```python
sim_end = start_date + timedelta(hours=abs(duration_hours))
now = datetime.now(timezone.utc)

if sim_end > now + timedelta(hours=6):
    # Prospective: simulation extends into the future beyond latest
    # available STOFS cycle.  Fall back to TPXO for uncovered hours.
    last_stofs_hour = ...  # determine from downloaded data
    generate_tidal_fill(start_hour=last_stofs_hour, ...)
else:
    # Retrospective: STOFS covers the full window.  No TPXO needed.
    pass
```

This is a minimal, low-risk change that eliminates the quality
degradation for all retrospective STOFS runs longer than 180 hours.

### Phase 2: Multi-cycle STOFS download and stitching

Extend the download and regridding pipeline to handle multiple STOFS
cycles:

**Download stage changes (`downloader.py`):**

- `_build_stofs_urls()` accepts `start` and `end` date-times.
- Generates one URL per 6-hourly cycle needed to cover the full
  simulation window.
- Downloads only the cycles not already cached locally.

**Regridding stage changes (`regrid_estofs.py`):**

- Accept multiple STOFS files as input.
- For each output time step, select the best available STOFS cycle
  (e.g., prefer the cycle whose forecast hour is closest to the
  analysis time, earlier forecast hours are more accurate).
- Produce a single continuous `elev2D.th.nc` spanning the full
  duration.

### Phase 3: Spatial subsetting of unstructured STOFS grids

Reduce download volume by cropping STOFS files to the model domain
before regridding:

**Approach:**

1. Open each STOFS file lazily from S3 via `xarray` + `fsspec`
   (no full download).
2. Normalize the ADCIRC-format STOFS variable/dimension names to a
   standard schema (node coordinates, triangle connectivity).
3. Crop to the SCHISM model's bounding box: select all nodes within
   the bounding box, keep only triangles where all three vertices are
   inside, and remap the connectivity array to reflect the new
   sequential node numbering.
4. Materialize only the cropped subset to local disk.

This could reduce per-cycle data from ~12 GB (global) to a few
hundred MB (regional), making multi-cycle stitching practical even
for extended-range simulations.

The crop algorithm for unstructured triangular meshes is
well-established; the [Thalassa](https://github.com/ec-jrc/Thalassa)
library (JRC, EUPL-1.2 license) implements exactly this pattern in
its `crop()` function using NumPy boolean indexing and
`numpy_indexed.remap()`. Since Thalassa has not been updated in two
years and carries a copyleft (EUPL-1.2) license that would propagate
to derivative works, we should implement an independent standalone
crop module following the same algorithmic approach rather than
extracting their code directly.  The logic is straightforward:

```text
nodes_in_bbox = where(lon >= xmin & lon <= xmax & lat >= ymin & lat <= ymax)
faces_in_bbox = where(all three vertex indices are in nodes_in_bbox)
remap vertex indices to new sequential numbering
```

This requires only `numpy` (and optionally `numpy_indexed` for the
remap step, which can also be done with `np.searchsorted`).

### Phase 4: Pure-Python TPXO prediction module

Develop a Python module that replaces both the Fortran `predict_tide`
binary and the bundled `pytides` library.  This module would:

- Read the standard TPXO10 atlas NetCDF files directly.
- Implement Arakawa C-grid bilinear interpolation with land masking.
- Compute all 32 constituents + 18 inferred minor constituents.
- Apply solid Earth tide corrections.
- Return NumPy arrays directly (no subprocess, no text parsing).
- Be validated against the Fortran OTPS binary for numerical parity.

This replaces three current components with one:

| Current | Replacement |
| ------- | ----------- |
| `predict_tide` (Fortran binary via subprocess) | Pure-Python prediction function |
| `otps_to_open_bnds()` (text output parser) | Direct NumPy array return |
| `generate_ocean_tide()` + `pytides` (8 constituents, `scipy.griddata`) | Same function with 32 constituents, proper C-grid interpolation |

Once implemented, the following can be removed:

- `src/coastal_calibration/tides/pytides/`: 4 files, ~680 lines.
- `src/coastal_calibration/tides/_ocean_tide.py`: `pytides`-based
  tidal fill.
- `predict_tide` binary dependency.
- `_otps.py` (`make_otps_input`, `otps_to_open_bnds`): text I/O
  helpers.
- The separate TPXO constituent files (`k1.nc` … `s2.nc`): the
  module reads the atlas directly.
