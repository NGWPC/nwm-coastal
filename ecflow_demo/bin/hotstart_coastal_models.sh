#!/usr/bin/env bash
# hotstart_coastal_models.sh -- standalone SCHISM/SFINCS boundary/tide
# spin-up, run OUTSIDE ecflow.
#
# A real AnA cycle is only 3h (T-3 -> T0), which is too short for
# the STOFS tidal/surge boundary signal to physically propagate into and
# equilibrate across the SCHISM/SFINCS domains. The very first AnA cycle in
# a fresh chain therefore cold-starts with unrealistic internal
# hydrodynamics.
#
# This script runs a much longer (default 18h), STOFS-boundary-forcing-ONLY
# spin-up with each model's own physics ramp enabled, landing its output in 
# the `ana_<H-1>` directory/filename convention that a real AnA cycle would
# have produced. This way gen_configs_ana.ecf's existing, unmodified self-chain
# discovery logic picks it up for the first real AnA cycle (TARGET_CYCLE below),
# with no ecflow changes needed.
#
# Discharge, precip, and wind/pressure are all disabled via config
# overrides in gen_cycle_config.py's --run-type spinup (not by editing
# schism_sims/run.yaml or sfincs_sims/run.yaml - this way every other real 
# ana/sr cycle sharing those base templates is unaffected):
#   - SCHISM: discharge_file pointed at a deliberately nonexistent path
#     (skips schism_discharge AND the precip regridding that feeds it,
#     schism_forcing); include_wind=False (skips schism_sflux, sets
#     nws=0 in param.nml).
#   - SFINCS: include_precip/include_wind/include_pressure=False,
#     merge_discharge=False, discharge_locations_file=null.
#
# Usage:
#   hotstart_coastal_models.sh <TARGET_CYCLE YYYYMMDDHH> [SPINUP_HOURS] [RAMP_HOURS]
#
# TARGET_CYCLE is the first REAL AnA cycle this spin-up feeds (its own
# PREV_CYCLE is where the spin-up output lands). SPINUP_HOURS defaults to
# 18, RAMP_HOURS (each model's own physics-ramp period) defaults to
# SPINUP_HOURS/2.
set -euo pipefail

TARGET_CYCLE="${1:?Usage: hotstart_coastal_models.sh <TARGET_CYCLE YYYYMMDDHH> [SPINUP_HOURS] [RAMP_HOURS]}"
SPINUP_HOURS="${2:-18}"
RAMP_HOURS="${3:-$((SPINUP_HOURS / 2))}"

# NWM_COASTAL_ROOT/RUN_COASTAL_ROOT (not NWM_RTE_ROOT/RUN_NGEN_ROOT -- this
# script never touches troute/ngen forcing, STOFS boundary only) -- see
# ../README.md for what each should point to.
for _var in NWM_COASTAL_ROOT RUN_COASTAL_ROOT; do
  if [ -z "${!_var:-}" ]; then
    echo "ERROR: ${_var} is not set -- see ecflow_demo/README.md" >&2
    exit 1
  fi
done

NWM_COASTAL_PY="${NWM_COASTAL_ROOT}/nwm-coastal-py"
NWM_COASTAL_CLI="${NWM_COASTAL_ROOT}/nwm-coastal-cli"
GEN_SCRIPT="${NWM_COASTAL_ROOT}/ecflow_demo/bin/gen_cycle_config.py"
SCHISM_BASE_YAML="${RUN_COASTAL_ROOT}/schism_sims/run.yaml"
SFINCS_BASE_YAML="${RUN_COASTAL_ROOT}/sfincs_sims/run.yaml"
SCHISM_CYCLES_DIR="${RUN_COASTAL_ROOT}/schism_sims/cycles"
SFINCS_CYCLES_DIR="${RUN_COASTAL_ROOT}/sfincs_sims/cycles"

TARGET_DT="${TARGET_CYCLE:0:4}-${TARGET_CYCLE:4:2}-${TARGET_CYCLE:6:2} ${TARGET_CYCLE:8:2}:00:00"
SPINUP_END_DT=$(date -u -d "-3 hours ${TARGET_DT}" +"%Y-%m-%d %H:%M:%S")
SPINUP_END_STAMP=$(date -u -d "-3 hours ${TARGET_DT}" +"%Y%m%d%H")00
PREV_CYCLE=$(date -u -d "-1 hour ${TARGET_DT}" +"%Y%m%d%H")

echo "=== hotstart_coastal_models ==="
echo "TARGET_CYCLE=${TARGET_CYCLE}  SPINUP_HOURS=${SPINUP_HOURS}  RAMP_HOURS=${RAMP_HOURS}"
echo "Spin-up window ends at ${SPINUP_END_DT} (${SPINUP_HOURS}h run) -> lands in ana_${PREV_CYCLE}"
echo "STOFS boundary is the only data this script downloads/uses."
echo

# ---------------------------------------------------------------------
# 1. SFINCS spin-up run -> lands directly in ana_<PREV_CYCLE>. tspinup 
#    is in SECONDS (sfincs_input.f90: tspinup = t0 + tspinup).
# ---------------------------------------------------------------------
TSPINUP_SEC=$(( RAMP_HOURS * 3600 ))
SFINCS_OVERRIDES=$(python3 -c "import json; print(json.dumps({'tspinup': ${TSPINUP_SEC}}))")
SFINCS_CYCLE_DIR="${SFINCS_CYCLES_DIR}/ana_${PREV_CYCLE}"
echo "--- [1/2] SFINCS spin-up (tspinup=${TSPINUP_SEC}s) -> ${SFINCS_CYCLE_DIR} ---"
"${NWM_COASTAL_PY}" "${GEN_SCRIPT}" \
  --model sfincs --run-type spinup \
  --base-yaml "${SFINCS_BASE_YAML}" \
  --cycle "spinup_${SPINUP_END_STAMP}" \
  --start-date "${SPINUP_END_DT}" \
  --duration-hours "${SPINUP_HOURS}" \
  --extra-run-param-overrides "${SFINCS_OVERRIDES}" \
  --cycle-dir "${SFINCS_CYCLE_DIR}"
"${NWM_COASTAL_CLI}" run "${SFINCS_CYCLE_DIR}/run.yaml"
echo

# ---------------------------------------------------------------------
# 2. SCHISM spin-up run -> lands directly in ana_<PREV_CYCLE>.
#    This compiled SCHISM build's &OPT namelist (schism_init.F90) never
#    declares nramp/nrampbc at all -- ramping is controlled purely by
#    dramp/drampbc being > 0 (the separate boolean enable flags were
#    removed in this version; passing them aborts SCHISM at init with
#    "Cannot match namelist object name nramp"). Both dramp/drampbc
#    already exist in the base param.nml template, so only their values
#    need overriding. RAMP_HOURS/24 converts to SCHISM's
#    ramp-period-in-days convention.
# ---------------------------------------------------------------------
DRAMP_DAYS=$(python3 -c "print(${RAMP_HOURS}/24)")
SCHISM_OVERRIDES=$(python3 -c "
import json
print(json.dumps({'dramp': ${DRAMP_DAYS}, 'drampbc': ${DRAMP_DAYS}}))
")
SCHISM_CYCLE_DIR="${SCHISM_CYCLES_DIR}/ana_${PREV_CYCLE}"
echo "--- [2/2] SCHISM spin-up (dramp=${DRAMP_DAYS}d) -> ${SCHISM_CYCLE_DIR} ---"
"${NWM_COASTAL_PY}" "${GEN_SCRIPT}" \
  --model schism --run-type spinup \
  --base-yaml "${SCHISM_BASE_YAML}" \
  --cycle "spinup_${SPINUP_END_STAMP}" \
  --start-date "${SPINUP_END_DT}" \
  --duration-hours "${SPINUP_HOURS}" \
  --extra-run-param-overrides "${SCHISM_OVERRIDES}" \
  --cycle-dir "${SCHISM_CYCLE_DIR}"
"${NWM_COASTAL_CLI}" run "${SCHISM_CYCLE_DIR}/run.yaml"
echo

echo "=== hotstart_coastal_models complete: ana_${PREV_CYCLE} ready for ${TARGET_CYCLE}'s own AnA cycle ==="
