#!/usr/bin/env bash
# hotstart_coastal_models.sh -- standalone cold-start / bootstrap procedure,
# run OUTSIDE ecflow, to seed a predecessor hour's state so the live
# coastal_hourly suite's self-chaining trigger graph has something real to
# start from.
#
# Produces TWO independent pieces of state for PREV_CYCLE = TARGET_CYCLE-1h:
#
#   1. troute AnA-A bootstrap (step 0): a single cold-started
#      troute_ana_a.ecf-equivalent run, producing
#      region_ana_a_<PREV_CYCLE>/<VPU>/state_save/troute -- the ONLY troute
#      state gen_configs_ana.ecf's/schism_ana.ecf's/sfincs_ana.ecf's/
#      run_stofs_download_ana.ecf's shared bootstrap_check.h looks for, and
#      the same state TARGET_CYCLE's own real troute_ana_a/troute_ana_b
#      tasks load from. troute_ana_b is deliberately NOT mirrored here --
#      traced against every downstream consumer, nothing ever reads a
#      bootstrap-produced region_ana_b_<PREV_CYCLE>; TARGET_CYCLE's
#      troute_sr.ecf loads from that hour's own real region_ana_b_<TARGET_CYCLE>
#      instead, produced normally once the ring is seeded.
#
#   2. SCHISM/SFINCS spin-up (steps 1-2, unchanged from before): a real AnA
#      cycle is only 3h (T-3 -> T0), too short for the STOFS tidal/surge
#      boundary signal to physically propagate into and equilibrate across
#      the SCHISM/SFINCS domains. This runs a much longer (default 18h),
#      STOFS-boundary-forcing-ONLY spin-up with each model's own physics
#      ramp enabled, landing its output in the `ana_<PREV_CYCLE>`
#      directory/filename convention a real AnA cycle would have produced,
#      so gen_configs_ana.ecf's existing, unmodified self-chain discovery
#      logic picks it up for TARGET_CYCLE's own first real AnA cycle.
#
# Discharge, precip, and wind/pressure are all disabled for the SCHISM/SFINCS
# spin-up via config overrides in gen_cycle_config.py's --run-type spinup
# (not by editing schism_sims/run.yaml or sfincs_sims/run.yaml - this way
# every other real ana/sr cycle sharing those base templates is unaffected):
#   - SCHISM: discharge_file pointed at a deliberately nonexistent path
#     (skips schism_discharge AND the precip regridding that feeds it,
#     schism_forcing); include_wind=False (skips schism_sflux, sets
#     nws=0 in param.nml).
#   - SFINCS: include_precip/include_wind/include_pressure=False,
#     merge_discharge=False, discharge_locations_file=null.
#
# Usage:
#   hotstart_coastal_models.sh <TARGET_CYCLE YYYYMMDDHH> [SPINUP_HOURS] [RAMP_HOURS] \
#       [--coastal-only | --troute-only] [--dry-run]
#
# TARGET_CYCLE is the first REAL AnA cycle this bootstrap feeds (its own
# PREV_CYCLE is where all bootstrap output lands). SPINUP_HOURS defaults to
# 18, RAMP_HOURS (each model's own physics-ramp period) defaults to
# SPINUP_HOURS/2. --coastal-only / --troute-only run just one half (for
# isolated re-runs/debugging without redoing the expensive half).
# --dry-run prints every command that would run without executing any of
# them. Once this script completes, seed the live ecflow ring with
# server/seed_ring.sh <TARGET_CYCLE>.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE_EOF'
Usage: hotstart_coastal_models.sh <TARGET_CYCLE YYYYMMDDHH> [SPINUP_HOURS] [RAMP_HOURS] \
    [--coastal-only | --troute-only] [--dry-run]
USAGE_EOF
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi
TARGET_CYCLE="$1"
shift

SPINUP_HOURS=""
RAMP_HOURS=""
RUN_TROUTE=1
RUN_COASTAL=1
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --coastal-only) RUN_TROUTE=0 ;;
    --troute-only) RUN_COASTAL=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [ -z "${SPINUP_HOURS}" ]; then
        SPINUP_HOURS="${arg}"
      elif [ -z "${RAMP_HOURS}" ]; then
        RAMP_HOURS="${arg}"
      else
        echo "ERROR: unexpected argument: ${arg}" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [ "${RUN_TROUTE}" -eq 0 ] && [ "${RUN_COASTAL}" -eq 0 ]; then
  echo "ERROR: --coastal-only and --troute-only are mutually exclusive" >&2
  exit 1
fi

SPINUP_HOURS="${SPINUP_HOURS:-18}"
RAMP_HOURS="${RAMP_HOURS:-$((SPINUP_HOURS / 2))}"

# NWM_COASTAL_ROOT/RUN_COASTAL_ROOT are always required (SCHISM/SFINCS
# spin-up config generation and paths). NWM_RTE_ROOT/RUN_NGEN_ROOT are only
# needed for the troute bootstrap step, but are required unconditionally
# (not just under --coastal-only vs --troute-only) so a missing var is
# caught up front rather than only on a re-run with different flags -- see
# ../README.md for what each should point to.
for _var in NWM_COASTAL_ROOT RUN_COASTAL_ROOT NWM_RTE_ROOT RUN_NGEN_ROOT; do
  if [ -z "${!_var:-}" ]; then
    echo "ERROR: ${_var} is not set -- see forecast_demo/README.md" >&2
    exit 1
  fi
done

NWM_COASTAL_PY="${NWM_COASTAL_ROOT}/nwm-coastal-py"
NWM_COASTAL_CLI="${NWM_COASTAL_ROOT}/nwm-coastal-cli"
GEN_SCRIPT="${NWM_COASTAL_ROOT}/forecast_demo/bin/gen_cycle_config.py"
SCHISM_BASE_YAML="${RUN_COASTAL_ROOT}/schism_sims/run.yaml"
SFINCS_BASE_YAML="${RUN_COASTAL_ROOT}/sfincs_sims/run.yaml"
SCHISM_CYCLES_DIR="${RUN_COASTAL_ROOT}/schism_sims/cycles"
SFINCS_CYCLES_DIR="${RUN_COASTAL_ROOT}/sfincs_sims/cycles"

# VPU/NWM_RTE_DIR/EWTS_ENABLED/TROUTE_REGIONALIZATION_ROOT* mirror
# suite_def/coastal_hourly.def.template's own `edit` block exactly (see
# that file's header comment) -- VPU defaults to '03S', overridable via
# env, but there is no single shared source of truth between this script
# and the suite def today; keep both in sync by hand if VPU ever changes.
VPU="${VPU:-03S}"
NWM_RTE_DIR="${NWM_RTE_ROOT}"
EWTS_ENABLED="NO"
TROUTE_REGIONALIZATION_ROOT="${RUN_NGEN_ROOT}/regionalization"
TROUTE_REGIONALIZATION_ROOT_CONTAINER="/ngwpc/run_ngen/regionalization"

TARGET_DT="${TARGET_CYCLE:0:4}-${TARGET_CYCLE:4:2}-${TARGET_CYCLE:6:2} ${TARGET_CYCLE:8:2}:00:00"
SPINUP_END_DT=$(date -u -d "-3 hours ${TARGET_DT}" +"%Y-%m-%d %H:%M:%S")
SPINUP_END_STAMP=$(date -u -d "-3 hours ${TARGET_DT}" +"%Y%m%d%H")00
PREV_CYCLE=$(date -u -d "-1 hour ${TARGET_DT}" +"%Y%m%d%H")

run() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

echo "=== hotstart_coastal_models ==="
echo "TARGET_CYCLE=${TARGET_CYCLE}  SPINUP_HOURS=${SPINUP_HOURS}  RAMP_HOURS=${RAMP_HOURS}  VPU=${VPU}"
echo "PREV_CYCLE=${PREV_CYCLE}  RUN_TROUTE=${RUN_TROUTE}  RUN_COASTAL=${RUN_COASTAL}  DRY_RUN=${DRY_RUN}"
echo

# ---------------------------------------------------------------------
# 0. troute AnA-A bootstrap -> region_ana_a_<PREV_CYCLE>/<VPU>/state_save/troute
#    Mirrors troute_ana_a.ecf's LAUNCH block (see that file for the full
#    rationale on the -dt/-lb arithmetic and host-vs-container path split);
#    kept in sync manually since .ecf files can't be invoked standalone
#    (unresolved %VAR%/%include ecflow syntax). Cold start on purpose
#    (no -lsf) -- this run IS the bootstrap. Run first (cheap: ~1h of
#    simulated time) so a misconfigured env/docker/VPU fails in seconds,
#    before the much longer SCHISM/SFINCS spin-up below.
# ---------------------------------------------------------------------
if [ "${RUN_TROUTE}" -eq 1 ]; then
  echo "--- [0] troute AnA-A bootstrap -> region_ana_a_${PREV_CYCLE} ---"
  TROUTE_END_DT=$(date -u -d "-2 hours ${TARGET_DT}" +"%Y-%m-%d %H:%M:%S")
  RNAME_A="region_ana_a_${PREV_CYCLE}"
  SAVE_STATE_DIR_A="${TROUTE_REGIONALIZATION_ROOT}/${RNAME_A}/${VPU}/state_save"
  SAVE_STATE_DIR_A_CONTAINER="${TROUTE_REGIONALIZATION_ROOT_CONTAINER}/${RNAME_A}/${VPU}/state_save"

  run bash -c '
    set -euo pipefail
    cd "'"${NWM_RTE_DIR}"'"
    export EWTS_ENABLED="'"${EWTS_ENABLED}"'"
    export TARGET_IMAGE_NAME="${TARGET_IMAGE_NAME:-ngen_rte_ghcr}"
    source config.bashrc
    source run.sh
    TEST_VPU="vpu_'"${VPU}"'"
    TEST_FORM_ASSIGN_VPU="${INSTALLED_REGIONALIZATION_RESULTS}/${TEST_VPU}/formulation_assignment.csv"
    TEST_CAT_GRP_VPU="${INSTALLED_REGIONALIZATION_RESULTS}/${TEST_VPU}/catchment_groups.csv"
    docker_run python -um "ngen_rte.run_regionalization_standalone" -n 12 -faf "${TEST_FORM_ASSIGN_VPU}" -cgf "${TEST_CAT_GRP_VPU}" -fconfig "standard_ana" -dt "'"${TROUTE_END_DT}"'" -lb 120 -rname "'"${RNAME_A}"'" -v "'"${VPU}"'" --hydrofab_file "/ngwpc/run_ngen/data/hydrofabric/vpu_'"${VPU}"'.gpkg" -outfmt NetCDF -ss -ssd "'"${SAVE_STATE_DIR_A_CONTAINER}"'"
  ' "${NWM_RTE_DIR}/run.sh"

  if [ "${DRY_RUN}" -eq 0 ]; then
    [ -f "${SAVE_STATE_DIR_A}/troute" ] || { echo "ERROR: expected troute_ana_a bootstrap state missing: ${SAVE_STATE_DIR_A}/troute" >&2; exit 1; }
    echo "troute AnA-A bootstrap OK: ${SAVE_STATE_DIR_A}/troute"
  fi
  echo
else
  echo "--- [0] troute AnA-A bootstrap SKIPPED (--coastal-only) ---"
  echo
fi

if [ "${RUN_COASTAL}" -eq 1 ]; then
# ---------------------------------------------------------------------
# 1. SFINCS spin-up run -> lands directly in ana_<PREV_CYCLE>. tspinup 
#    is in SECONDS (sfincs_input.f90: tspinup = t0 + tspinup).
# ---------------------------------------------------------------------
TSPINUP_SEC=$(( RAMP_HOURS * 3600 ))
SFINCS_OVERRIDES=$(python3 -c "import json; print(json.dumps({'tspinup': ${TSPINUP_SEC}}))")
SFINCS_CYCLE_DIR="${SFINCS_CYCLES_DIR}/ana_${PREV_CYCLE}"
echo "--- [1/2] SFINCS spin-up (tspinup=${TSPINUP_SEC}s) -> ${SFINCS_CYCLE_DIR} ---"
run "${NWM_COASTAL_PY}" "${GEN_SCRIPT}" \
  --model sfincs --run-type spinup \
  --base-yaml "${SFINCS_BASE_YAML}" \
  --cycle "spinup_${SPINUP_END_STAMP}" \
  --start-date "${SPINUP_END_DT}" \
  --duration-hours "${SPINUP_HOURS}" \
  --extra-run-param-overrides "${SFINCS_OVERRIDES}" \
  --cycle-dir "${SFINCS_CYCLE_DIR}"
run "${NWM_COASTAL_CLI}" run "${SFINCS_CYCLE_DIR}/run.yaml"
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
run "${NWM_COASTAL_PY}" "${GEN_SCRIPT}" \
  --model schism --run-type spinup \
  --base-yaml "${SCHISM_BASE_YAML}" \
  --cycle "spinup_${SPINUP_END_STAMP}" \
  --start-date "${SPINUP_END_DT}" \
  --duration-hours "${SPINUP_HOURS}" \
  --extra-run-param-overrides "${SCHISM_OVERRIDES}" \
  --cycle-dir "${SCHISM_CYCLE_DIR}"
run "${NWM_COASTAL_CLI}" run "${SCHISM_CYCLE_DIR}/run.yaml"
echo

# ---------------------------------------------------------------------
# 3. gen_configs_ana.ecf's SCHISM hot-start discovery globs
#    hotstart_it=*.nc in PREV_CYCLE's outputs/ dir and picks the SMALLEST
#    iteration, assuming (correctly, for a normal 3h AnA run) that it has
#    only 2 checkpoints -- T-2 (smallest) and T0 (largest). A spin-up run
#    is much longer than 3h, so combine_hotstart() (schism/prep.py)
#    writes one checkpoint per simulated hour the whole way through --
#    the smallest here is NOT T-2, it's 1h into the spin-up, hours off
#    from what's actually needed.
#
# The spin-up's OWN final checkpoint (the largest iteration) is always
# the correct one instead: SPINUP_END_DT is defined above as exactly
# TARGET_CYCLE-3h == PREV_CYCLE's own T-2 by construction, and the
# spin-up's last hourly checkpoint lands at SPINUP_END_DT.
#
# Fix: relocate every OTHER checkpoint out of outputs/ (into a
# subdirectory, not deleted -- still fully recoverable) so the discovery
# glob only ever finds the one correct file. Non-destructive on purpose:
# combine_hotstart7's own per-rank inputs (hotstart_<rank>_<iter>.nc)
# are untouched either way, so even the *_it=N.nc files being moved
# could still be regenerated later if ever needed.
# ---------------------------------------------------------------------
if [ "${DRY_RUN}" -eq 0 ]; then
  SCHISM_OUTPUTS_DIR="${SCHISM_CYCLE_DIR}/run/outputs"
  INTERMEDIATE_DIR="${SCHISM_OUTPUTS_DIR}/spinup_intermediate_hotstarts"
  LATEST_ITERATION=$(
    for f in "${SCHISM_OUTPUTS_DIR}"/hotstart_it=*.nc; do
      [ -e "$f" ] || continue
      n=$(basename "$f" | sed -E 's/hotstart_it=([0-9]+)\.nc/\1/')
      echo "${n} ${f}"
    done | sort -n | tail -1 | awk '{print $2}'
  )
  if [ -n "${LATEST_ITERATION}" ]; then
    mkdir -p "${INTERMEDIATE_DIR}"
    moved=0
    for f in "${SCHISM_OUTPUTS_DIR}"/hotstart_it=*.nc; do
      [ -e "$f" ] || continue
      if [ "$f" != "${LATEST_ITERATION}" ]; then
        mv "$f" "${INTERMEDIATE_DIR}/"
        moved=$((moved + 1))
      fi
    done
    echo "Kept $(basename "${LATEST_ITERATION}") as the T-2 hot-start for ${TARGET_CYCLE}'s AnA cycle;"
    echo "moved ${moved} intermediate spin-up checkpoint(s) to ${INTERMEDIATE_DIR}"
  else
    echo "WARNING: no hotstart_it=*.nc found in ${SCHISM_OUTPUTS_DIR}; nothing to relocate" >&2
  fi
  echo
else
  echo "[dry-run] would relocate intermediate SCHISM hotstart checkpoints in ${SCHISM_CYCLE_DIR}/run/outputs"
  echo
fi
else
  echo "--- [1-2] SCHISM/SFINCS spin-up SKIPPED (--troute-only) ---"
  echo
fi

echo "=== hotstart_coastal_models complete for ${TARGET_CYCLE} ==="
[ "${RUN_TROUTE}" -eq 1 ] && echo "  troute:                  region_ana_a_${PREV_CYCLE} ready"
[ "${RUN_COASTAL}" -eq 1 ] && echo "  coastal (SCHISM/SFINCS): ana_${PREV_CYCLE} ready"
echo "Next: forecast_demo/server/seed_ring.sh ${TARGET_CYCLE}   (once the ecflow server is up and the suite is loaded+begun)"
