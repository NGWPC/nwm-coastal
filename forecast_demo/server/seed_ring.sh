#!/bin/bash
# seed_ring.sh -- seeds the live coastal_hourly suite's self-chaining
# trigger ring so it can start running forward from TARGET_CYCLE.
#
# The suite's cross-hour triggers form a true 24h ring (every hour's
# troute_ana_a/schism_ana/sfincs_ana triggers off the SAME task in the
# previous hour, h00's predecessor being ../h23) with no natural entry
# point on a fresh --begin, for any hour. Starting the ring requires
# manually force-completing PREV_CYCLE's (TARGET_CYCLE-1h's) three
# cross-hour-referenced nodes -- confirmed by grepping every "../h"
# reference in suite_def/coastal_hourly.def.template, exactly these three
# are ever referenced: troute_ana_a, schism_ana, sfincs_ana. No others
# need seeding (PREV_CYCLE's other tasks -- troute_ana_b, gen_configs_ana,
# run_stofs_download_ana, gen_configs_sr, run_stofs_download_sr, schism_sr,
# sfincs_sr -- aren't referenced by any trigger the next hour depends on).
#
# Forcing ecflow's own node status does NOT fabricate the on-disk state
# files bootstrap_check.h (ecf_home/bootstrap_check.h) and gen_configs_ana.ecf
# independently look for -- those two mechanisms are both necessary. This
# script verifies the real files are actually present before forcing
# anything, and fails loudly (not silently) if they're not -- run
# bin/hotstart_coastal_models.sh first to produce them.
#
# Usage: seed_ring.sh <TARGET_CYCLE YYYYMMDDHH> [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./ecf_env.sh

if [ $# -lt 1 ]; then
  echo "Usage: seed_ring.sh <TARGET_CYCLE YYYYMMDDHH> [--dry-run]" >&2
  exit 1
fi
TARGET_CYCLE="$1"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

for _var in NWM_RTE_ROOT RUN_NGEN_ROOT RUN_COASTAL_ROOT; do
  if [ -z "${!_var:-}" ]; then
    echo "ERROR: ${_var} is not set -- see forecast_demo/README.md" >&2
    exit 1
  fi
done

TARGET_DT="${TARGET_CYCLE:0:4}-${TARGET_CYCLE:4:2}-${TARGET_CYCLE:6:2} ${TARGET_CYCLE:8:2}:00:00"
PREV_CYCLE=$(date -u -d "-1 hour ${TARGET_DT}" +"%Y%m%d%H")
PREV_HH="${PREV_CYCLE:8:2}"
# VPU must match suite_def/coastal_hourly.def.template's own `edit VPU` --
# see hotstart_coastal_models.sh for the same note.
VPU="${VPU:-03S}"

TROUTE_REGIONALIZATION_ROOT="${RUN_NGEN_ROOT}/regionalization"
SCHISM_CYCLES_DIR="${RUN_COASTAL_ROOT}/schism_sims/cycles"
SFINCS_CYCLES_DIR="${RUN_COASTAL_ROOT}/sfincs_sims/cycles"

echo "=== seed_ring: TARGET_CYCLE=${TARGET_CYCLE} PREV_CYCLE=${PREV_CYCLE} DRY_RUN=${DRY_RUN} ==="
echo

# 1) ecflow server reachability -- fail loudly, don't let a connection
#    error surface as a cryptic ecflow_client stack trace later.
if ! ecflow_client --port="${ECF_PORT}" --host=localhost --ping >/dev/null 2>&1; then
  echo "ERROR: ecflow_server not reachable on port ${ECF_PORT} -- run server/start_server.sh + server/load_and_begin.sh first" >&2
  exit 1
fi
echo "[1/4] ecflow_server reachable on port ${ECF_PORT}"

# 2) The 3 target nodes must actually exist in the loaded suite.
for _task in troute_ana_a schism_ana sfincs_ana; do
  NODE="/coastal_hourly/cycle/h${PREV_HH}/${_task}"
  if ! ecflow_client --port="${ECF_PORT}" --host=localhost --get_state="${NODE}" >/dev/null 2>&1; then
    echo "ERROR: ${NODE} not found -- is the suite loaded+begun? (server/load_and_begin.sh)" >&2
    exit 1
  fi
done
echo "[2/4] all 3 target nodes exist under /coastal_hourly/cycle/h${PREV_HH}"

# 3) Real on-disk state must actually be present -- mirrors exactly what
#    bootstrap_check.h / gen_configs_ana.ecf's own consumers check for.
#    Forcing ecflow's status does not fabricate files; this is the
#    "fail loudly, not silently" guarantee. T_MINUS_3_STAMP is computed
#    relative to TARGET_DT (not PREV_DT) -- this is the same stamp
#    gen_configs_ana.ecf computes for TARGET_CYCLE's own warm-start lookup,
#    and hotstart_coastal_models.sh's spin-up lands its last checkpoint
#    exactly there by construction.
T_MINUS_3_STAMP=$(date -u -d "-3 hours ${TARGET_DT}" +"%Y%m%d.%H0000")
TROUTE_STATE_FILE="${TROUTE_REGIONALIZATION_ROOT}/region_ana_a_${PREV_CYCLE}/${VPU}/state_save/troute"
SCHISM_OUTPUTS_DIR="${SCHISM_CYCLES_DIR}/ana_${PREV_CYCLE}/run/outputs"
SFINCS_RST_FILE="${SFINCS_CYCLES_DIR}/ana_${PREV_CYCLE}/run/sfincs_model/sfincs.${T_MINUS_3_STAMP}.rst"

if [ ! -f "${TROUTE_STATE_FILE}" ]; then
  echo "ERROR: no troute_ana_a state at ${TROUTE_STATE_FILE} -- run bin/hotstart_coastal_models.sh ${TARGET_CYCLE} first" >&2
  exit 1
fi
if ! compgen -G "${SCHISM_OUTPUTS_DIR}/hotstart_it=*.nc" >/dev/null; then
  echo "ERROR: no SCHISM hotstart in ${SCHISM_OUTPUTS_DIR} -- run bin/hotstart_coastal_models.sh ${TARGET_CYCLE} first" >&2
  exit 1
fi
if [ ! -f "${SFINCS_RST_FILE}" ]; then
  echo "ERROR: no SFINCS restart at ${SFINCS_RST_FILE} -- run bin/hotstart_coastal_models.sh ${TARGET_CYCLE} first" >&2
  exit 1
fi
echo "[3/4] real on-disk state confirmed for PREV_CYCLE=${PREV_CYCLE}:"
echo "         ${TROUTE_STATE_FILE}"
echo "         ${SCHISM_OUTPUTS_DIR}/hotstart_it=*.nc"
echo "         ${SFINCS_RST_FILE}"
echo

# 4) Force-complete the 3 nodes so the ring has a real starting point.
if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[4/4] --dry-run: would force-complete:"
  for _task in troute_ana_a schism_ana sfincs_ana; do
    echo "         /coastal_hourly/cycle/h${PREV_HH}/${_task}"
  done
  exit 0
fi

for _task in troute_ana_a schism_ana sfincs_ana; do
  NODE="/coastal_hourly/cycle/h${PREV_HH}/${_task}"
  ecflow_client --port="${ECF_PORT}" --host=localhost --force=complete "${NODE}"
  echo "[4/4] forced complete: ${NODE}"
done

echo
echo "Ring seeded. ${TARGET_CYCLE}'s troute_ana_a/schism_ana/sfincs_ana can now trigger normally."
echo "Confirm with: ecflow_client --get_state=/coastal_hourly/cycle/h${TARGET_CYCLE:8:2}"
