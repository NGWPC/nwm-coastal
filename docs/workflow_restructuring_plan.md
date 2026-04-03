# SCHISM/SFINCS Workflow Restructuring Plan

## Context

The coastal-calibration codebase currently mixes SCHISM-specific and SFINCS-specific
code under a flat `stages/` module and scattered root-level files (`schism_prep.py`,
`sflux.py`, `creator.py`). A new `src/coastal_calibration/schism/` module
(`project_reader.py`, `subsetter.py`, `constants.py`) has been added but is not yet
integrated with the main workflow.

Three files -- `nwmReaches.csv`, `element_areas.txt`, `elevation_correction.csv` -- are
tightly coupled into the core SCHISM workflow even though they only serve discharge
generation and boundary datum correction. The goal is to:

1. Decouple these files from the core workflow into dedicated stages
1. Integrate the new `schism/` module to replace duplicated hgrid parsing logic
1. Reorganize the package so each model has its own self-contained package

This is a three-phase effort. Each phase delivers standalone value and keeps the
codebase runnable.

### File Trace Summary

**`nwmReaches.csv`** -- Maps NWM river reach IDs to SCHISM source/sink element IDs.

- `config/schema.py:384` -- Listed as a **required** file in
    `SchismModelConfig.validate()` for `prebuilt_dir` (main coupling point).
- `stages/schism.py:360-366` (`PreSCHISMStage.run`) -- Copies it from `prebuilt_dir` to
    `work_dir` (both `nwm_retro` and `nwm_ana` paths).
- `schism_prep.py:134-136` (`stage_chrtout_files`) -- Also copies it during CHRTOUT
    staging.
- `schism_prep.py:179-194` (`make_discharge`) -- **Only consumer**: reads it to build
    source/sink element lists and NWM feature IDs.
- `schism_prep.py:50` (`clean_run_directory`) -- Deletes it during cleanup.
- `schism/project_reader.py:327,342` -- Stored as `optional_files` in
    `NWMSCHISMProject`.

**`element_areas.txt`** -- Per-element geodesic area values used as minimum thresholds
when merging river discharge into precipitation sources.

- `schism_prep.py:732` (`update_params`) -- Symlinked from `prebuilt_dir` as a static
    mesh file alongside `hgrid.gr3`, `manning.gr3`, etc.
- `schism_prep.py:369` (`merge_source_sink`) -- **Only consumer**:
    `np.genfromtxt(root_dir / "element_areas.txt")` to filter negligible sources.
- `schism/project_reader.py:396-415` -- `NWMSCHISMProject` can compute areas from mesh
    (geodesic calculation). The `subsetter.py:1679-1686` uses this for subsetting.

**`elevation_correction.csv`** -- Per-node datum corrections (6th CSV column),
subtracted from `elev2D.th.nc` tidal boundary forcing.

- `schism_prep.py:742` (`update_params`) -- Symlinked as an optional file from
    `prebuilt_dir`.
- `schism_prep.py:874-877` (`make_tpxo_boundary`) -- If it exists, calls
    `correct_elevation()`.
- `schism_prep.py:981-986` (`make_stofs_boundary`) -- Same pattern.
- `schism_prep.py:766-785` (`correct_elevation`) -- Reads CSV column 5, subtracts from
    each time step of `time_series` in the netCDF.

______________________________________________________________________

## Phase 1: Discharge & Elevation Correction Decoupling

**Goal:** Extract discharge operations into a dedicated `SchismDischargeStage` and make
elevation correction an explicit optional parameter rather than an implicit file
existence check. Remove `nwmReaches.csv` from the required files list and
`element_areas.txt` from the static symlink list.

### 1.1 Create `SchismDischargeStage`

Extract steps 1-3 from `PreSCHISMStage.run()` into a new stage class.

**File:** `src/coastal_calibration/stages/schism.py` (add new class)

The new stage encapsulates:

- Staging CHRTOUT files + copying `nwmReaches.csv` (current step 1)
- `make_discharge()` (current step 1 continued)
- `run_combine_sink_source()` (current step 2)
- `merge_source_sink()` (current step 3)

```python
class SchismDischargeStage(WorkflowStage):
    name = "schism_discharge"
    description = "Generate river discharge forcing"

    def run(self):
        # 1. Copy nwmReaches.csv from prebuilt_dir (only if it exists)
        # 2. Stage CHRTOUT files (for nwm_ana) or skip (for nwm_retro)
        # 3. make_discharge(...)
        # 4. run_combine_sink_source(...)
        # 5. merge_source_sink(...) -- read element_areas.txt from prebuilt_dir
        # 6. Verify source.nc was produced
```

**Validation:** The stage itself checks for `nwmReaches.csv` in `prebuilt_dir` at the
start of `run()` and raises a clear error if missing. This replaces the upfront
validation in `SchismModelConfig.validate()`.

### 1.2 Slim down `PreSCHISMStage`

**File:** `src/coastal_calibration/stages/schism.py`

After extraction, `PreSCHISMStage` retains only:

- Mesh partitioning (`partition_mesh`)
- `param.nml` patching for station output
- Critical file verification (`source.nc`, `param.nml`, `hgrid.gr3`)

Rename to better reflect its reduced scope (or keep name for backward compat but update
description to "Partition mesh and finalize inputs").

### 1.3 Remove `nwmReaches.csv` from required validation

**File:** `src/coastal_calibration/config/schema.py` (line ~384)

Move `"nwmReaches.csv"` out of the `required` list in `SchismModelConfig.validate()`.
The discharge stage handles its own validation.

### 1.4 Remove `element_areas.txt` from static symlink list

**File:** `src/coastal_calibration/schism_prep.py` -- `update_params()` (line ~732)

Remove `"element_areas.txt"` from the `static_files` list. The discharge stage
(`merge_source_sink`) reads it directly from `prebuilt_dir` (passed as `root_dir`
parameter). Confirm that `merge_source_sink`'s `root_dir` parameter can be pointed at
`prebuilt_dir` instead of `work_dir`.

Current call in `PreSCHISMStage`:

```python
merge_source_sink(work_dir=work_dir, root_dir=work_dir, prebuilt_dir=prebuilt_dir)
```

The `root_dir` is used solely for reading `element_areas.txt` (line 369). Change to:

```python
merge_source_sink(work_dir=work_dir, root_dir=prebuilt_dir, prebuilt_dir=prebuilt_dir)
```

Or better: rename `root_dir` to `element_area_dir` for clarity, or add an explicit
`element_area_file: Path` parameter.

### 1.5 Make elevation correction an explicit parameter

**Files:**

- `src/coastal_calibration/schism_prep.py` -- `make_tpxo_boundary()` and
    `make_stofs_boundary()`
- `src/coastal_calibration/stages/boundary.py` -- `TPXOBoundaryStage` and
    `STOFSBoundaryStage`

Current pattern (implicit):

```python
# Inside make_tpxo_boundary / make_stofs_boundary
correction_file = work_dir / "elevation_correction.csv"
if correction_file.exists():
    correct_elevation(output_file, correction_file)
```

New pattern (explicit):

```python
# Function signature adds optional parameter
def make_tpxo_boundary(
    ...,
    correction_file: Path | None = None,  # NEW
) -> Path:
    ...
    if correction_file is not None and correction_file.exists():
        correct_elevation(output_file, correction_file)
```

The stage classes resolve the path:

```python
# In TPXOBoundaryStage.run() / STOFSBoundaryStage.run()
correction_file = prebuilt_dir / "elevation_correction.csv"
make_tpxo_boundary(
    ...,
    correction_file=correction_file if correction_file.exists() else None,
)
```

Also remove `"elevation_correction.csv"` from the optional symlink list in
`update_params()` (line ~742) since it no longer needs to be in `work_dir`.

### 1.6 Update stage ordering

**File:** `src/coastal_calibration/config/schema.py` -- `SchismModelConfig.stage_order`

Current order (11 stages):

```
download -> forcing_prep -> forcing -> sflux -> params -> obs -> boundary -> prep -> run -> postprocess -> plot
```

New order (12 stages):

```
download -> forcing_prep -> forcing -> sflux -> params -> obs -> boundary -> discharge -> prep -> run -> postprocess -> plot
```

Insert `"schism_discharge"` between `"schism_boundary"` and `"schism_prep"`.

**File:** `src/coastal_calibration/config/schema.py` --
`SchismModelConfig.create_stages()`

Add the new stage instantiation:

```python
from coastal_calibration.stages import SchismDischargeStage

stages["schism_discharge"] = SchismDischargeStage(config, monitor)
```

### 1.7 Update `clean_run_directory`

**File:** `src/coastal_calibration/schism_prep.py` (line ~50)

Keep `nwmReaches.csv` in the cleanup list -- it is a generated (copied) file in
`work_dir` that should be cleaned between runs.

### 1.8 Update exports

**File:** `src/coastal_calibration/stages/__init__.py`

Add `SchismDischargeStage` to exports.

### 1.9 Update tests

**File:** `tests/common/test_schism_prep.py`

- Update `test_stage_chrtout_files` -- `nwmReaches.csv` copy is now in the discharge
    stage, not in `stage_chrtout_files` (or keep it there and have the stage call it).
- Add test for `SchismDischargeStage` that verifies it produces `source.nc`.
- Update `test_update_params` to verify `element_areas.txt` is NOT in the symlink list.

### 1.10 Update CHANGELOG.md

Add under `[Unreleased]` -> `Changed`:

- Extract discharge operations into dedicated `SchismDischargeStage`
- Make elevation correction an explicit parameter in boundary functions

______________________________________________________________________

## Phase 2: Integrate `schism/` Module into Workflow

**Goal:** Replace duplicated hgrid parsing in `stages/schism.py` with `NWMSCHISMProject`
from `schism/project_reader.py`. Optionally compute element areas from mesh instead of
reading the file.

### 2.1 Replace hgrid helpers in `SchismObservationStage`

**File:** `src/coastal_calibration/stages/schism.py`

Current standalone helpers (~130 lines):

- `_read_hgrid_header(hgrid_path)` -- lines 31-36
- `_read_node_coordinates(hgrid_path, n_nodes)` -- lines 39-49
- `_read_open_boundary_nodes(hgrid_path, n_nodes, n_elements)` -- ~80 lines

Replace with:

```python
from coastal_calibration.schism import NWMSCHISMProject

project = NWMSCHISMProject(prebuilt_dir, validate=False)
n_nodes = project.n_nodes
coords = project.nodes_coordinates  # (N, 2) array
boundaries = project.read_boundaries()  # BoundarySet
open_bnd_nodes = boundaries.open_boundaries  # list[list[int]]
```

Delete `_read_hgrid_header`, `_read_node_coordinates`, `_read_open_boundary_nodes`.

### 2.2 Optional: Compute element areas from mesh

**File:** `src/coastal_calibration/schism_prep.py` -- `merge_source_sink()` (line ~369)

Current:

```python
threshold = np.genfromtxt(str(root_dir / "element_areas.txt"))
```

Alternative using the new module:

```python
from coastal_calibration.schism import NWMSCHISMProject

project = NWMSCHISMProject(prebuilt_dir, validate=False)
threshold = project.element_areas
```

This eliminates the need for the pre-computed `element_areas.txt` file entirely for the
discharge pipeline. The file would only be needed by the subsetter.

**Trade-off:** Computing element areas from hgrid.gr3 takes time for large meshes. For
operational runs, reading the pre-computed file is faster. Consider:

- Accept both: try file first, fall back to computation
- Or keep the file approach for now and revisit when performance is measured

### 2.3 Add node-count validation to `correct_elevation`

**File:** `src/coastal_calibration/schism_prep.py` -- `correct_elevation()` (line ~766)

Optionally accept an `n_open_boundary_nodes` count to validate that the CSV row count
matches the total open boundary node count:

```python
boundaries = project.read_boundaries()
expected = boundaries.total_open_nodes
if len(elev_correct) != expected:
    raise ValueError(
        f"elevation_correction.csv has {len(elev_correct)} rows, "
        f"expected {expected} open boundary nodes"
    )
```

### 2.4 Update tests

- Update `tests/common/test_stages.py` if it tests hgrid parsing helpers.
- Ensure `tests/schism/` tests cover `NWMSCHISMProject` integration.

______________________________________________________________________

## Phase 3: Package Reorganization

**Goal:** Move model-specific code into `schism/` and `sfincs/` packages. Shared
infrastructure stays at the root level.

### 3.1 Target structure

```
src/coastal_calibration/
  __init__.py              # Keep lazy imports for backward compat
  cli.py                   # Shared
  runner.py                # Shared (model-agnostic orchestrator)
  downloader.py            # Shared
  coops_api.py             # Shared

  config/                  # Shared (ModelConfig ABC + subclasses)
    schema.py
    ...

  utils/                   # Shared (15 modules)
    logging.py, system.py, time.py, mpi.py, streamflow.py,
    workflow.py, raster.py, floodmap.py, copdem.py,
    esa_worldcover.py, gebco_wms.py, topobathy_noaa.py,
    topobathy_nws.py, _gdal.py

  plotting/                # Shared
  regridding/              # Shared
  tides/                   # Shared (usable by any coastal model)

  stages/                  # Shared base + download
    __init__.py            # Re-export all stages (backward compat)
    base.py                # WorkflowStage ABC
    download.py            # DownloadStage

  schism/                  # SCHISM workflow package
    __init__.py            # Public API
    constants.py           # <- exists
    project_reader.py      # <- exists
    subsetter.py           # <- exists
    prep.py                # <- from schism_prep.py
    sflux.py               # <- from root sflux.py
    stages.py              # <- merge of stages/schism.py
                           #    + stages/boundary.py
                           #    + stages/forcing.py

  sfincs/                  # SFINCS workflow package
    __init__.py            # Public API
    stages.py              # <- from stages/sfincs_build.py
    create.py              # <- from stages/sfincs_create.py + creator.py

  data_catalog/            # Keep shared or move to sfincs/ (TBD)
```

### 3.2 File moves

| Current path              | New path                   | Notes                           |
| ------------------------- | -------------------------- | ------------------------------- |
| `schism_prep.py`          | `schism/prep.py`           | All 15 functions move           |
| `sflux.py`                | `schism/sflux.py`          | SCHISM-only atmospheric forcing |
| `stages/schism.py`        | `schism/stages.py`         | 5 stage classes + helpers       |
| `stages/boundary.py`      | `schism/stages.py` (merge) | 4 SCHISM-only stage classes     |
| `stages/forcing.py`       | `schism/stages.py` (merge) | 3 SCHISM-only stage classes     |
| `stages/sfincs_build.py`  | `sfincs/stages.py`         | 14 classes                      |
| `stages/sfincs_create.py` | `sfincs/create.py`         | Creation pipeline stages        |
| `creator.py`              | `sfincs/create.py` (merge) | `SfincsCreator` class           |

### 3.3 Backward compatibility

**`stages/__init__.py`** continues to re-export everything from the new locations:

```python
# Backward compat -- import from new locations
from coastal_calibration.schism.stages import (
    BoundaryConditionStage,
    PreSCHISMStage,
    SCHISMRunStage,
    SchismDischargeStage,
    ...
)
from coastal_calibration.sfincs.stages import (
    SfincsDischargeStage,
    SfincsInitStage,
    ...
)
```

**`__init__.py`** (root) updates lazy import paths but keeps the same public names.

Old files (`schism_prep.py`, `sflux.py`, `creator.py`) can either:

- Be deleted (breaking change) -- preferred for cleanliness
- Become thin re-export shims with deprecation warnings

### 3.4 Import updates

All internal imports need updating. Key patterns:

```python
# Before
from coastal_calibration.schism_prep import make_discharge
from coastal_calibration.sflux import make_sflux

# After
from coastal_calibration.schism.prep import make_discharge
from coastal_calibration.schism.sflux import make_sflux
```

Since stages move into their model packages, `config/schema.py` import paths for
`create_stages()` change:

```python
# Before
from coastal_calibration.stages import PreSCHISMStage

# After
from coastal_calibration.schism.stages import PreSCHISMStage
```

### 3.5 Test updates

- Move/update test imports to match new paths
- `tests/common/test_schism_prep.py` -> tests `schism.prep` functions
- `tests/common/test_stages.py` -> verify re-exports still work

### 3.6 Update `__init__.py` lazy import registry

The root `__init__.py` has ~80 lazy import entries. Update source paths for moved
modules while keeping the same exported names.

______________________________________________________________________

## Key Files Reference

### Files to modify in Phase 1

- `src/coastal_calibration/stages/schism.py` -- Add `SchismDischargeStage`, slim
    `PreSCHISMStage`
- `src/coastal_calibration/stages/__init__.py` -- Export new stage
- `src/coastal_calibration/stages/boundary.py` -- Pass `correction_file` explicitly to
    boundary stages
- `src/coastal_calibration/schism_prep.py` -- Remove `element_areas.txt` from static
    symlinks, remove `elevation_correction.csv` from optional symlinks, add
    `correction_file` param to `make_tpxo_boundary`/`make_stofs_boundary`, update
    `merge_source_sink` `root_dir` usage
- `src/coastal_calibration/config/schema.py` -- Update `stage_order`, `create_stages`,
    `validate` (remove `nwmReaches.csv` from required)
- `tests/common/test_schism_prep.py` -- Update tests
- `CHANGELOG.md` -- Document changes

### Files to modify in Phase 2

- `src/coastal_calibration/stages/schism.py` -- Replace hgrid helpers with
    `NWMSCHISMProject`
- `src/coastal_calibration/schism_prep.py` -- Optional: compute element areas from mesh

### Files to move in Phase 3

- See table in section 3.2

### Key existing functions to reuse

- `NWMSCHISMProject` (`schism/project_reader.py`) -- mesh reading, element areas,
    boundaries
- `SCHISMFiles` constants (`schism/constants.py`) -- file name constants
- `WorkflowStage` (`stages/base.py`) -- base class for all stages
- `correct_elevation` (`schism_prep.py:766`) -- elevation correction function
- `make_discharge`, `run_combine_sink_source`, `merge_source_sink` (`schism_prep.py`) --
    discharge pipeline functions
- `stage_chrtout_files` (`schism_prep.py:87`) -- CHRTOUT file staging

______________________________________________________________________

## Verification

### Phase 1

```bash
pixi r lint
pixi r typecheck
pixi r -e test313-common test-common
pixi r -e test313-schism test-schism
pixi r -e test313 test
```

Verify:

- `SchismDischargeStage` produces `source.nc` when `nwmReaches.csv` exists in
    `prebuilt_dir`
- `PreSCHISMStage` no longer touches discharge files
- Boundary stages apply elevation correction from `prebuilt_dir` path, not `work_dir`
    symlink
- `element_areas.txt` is no longer symlinked by `update_params`
- `nwmReaches.csv` is no longer in the required files validation list
- Stage order includes `schism_discharge` between `schism_boundary` and `schism_prep`

### Phase 2

```bash
pixi r -e test313-common test-common
pixi r -e test313-schism test-schism
```

Verify:

- `SchismObservationStage` works with `NWMSCHISMProject` instead of standalone helpers
- Deleted helper functions are not imported anywhere

### Phase 3

```bash
pixi r -e test313 test
pixi r typecheck
pixi r lint
```

Verify:

- All imports resolve (both new paths and backward-compat re-exports)
- CLI still works: `pixi r -e dev python -m coastal_calibration.cli --help`
- Root `__init__.py` lazy imports all resolve
