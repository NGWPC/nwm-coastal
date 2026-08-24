#!/bin/bash
# Shared env for start_server.sh / stop_server.sh / load_and_begin.sh.
# source this, don't execute it directly.
#
# NWM_COASTAL_ROOT (required) is the only thing this file itself needs --
# everything else it derives is relative to it. The other pipeline roots
# (NWM_RTE_ROOT/RUN_NGEN_ROOT/RUN_COASTAL_ROOT) are required too, but only
# load_and_begin.sh actually needs them (for generating coastal_hourly.def
# from its template) -- see ../README.md for what each should point to.
if [ -z "${NWM_COASTAL_ROOT:-}" ]; then
  echo "ERROR: NWM_COASTAL_ROOT is not set -- see forecast_demo/README.md" >&2
  return 1 2>/dev/null || exit 1
fi

export ECF_PORT="${ECF_PORT:-39411}"   # arbitrary local port, avoids the ecflow-default 3141
export ECF_HOME="${NWM_COASTAL_ROOT}/forecast_demo/ecf_home"
# Where the ecflow 5.6.0 binaries live -- adjust if your install differs.
export PATH="${ECFLOW_BIN_DIR:-/contrib/software/ecflow/5.6.0/bin}:${PATH}"
