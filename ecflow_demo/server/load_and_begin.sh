#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./ecf_env.sh

for _var in NWM_RTE_ROOT RUN_NGEN_ROOT RUN_COASTAL_ROOT; do
  if [ -z "${!_var:-}" ]; then
    echo "ERROR: ${_var} is not set -- see ecflow_demo/README.md" >&2
    exit 1
  fi
done

TEMPLATE="${NWM_COASTAL_ROOT}/ecflow_demo/suite_def/coastal_hourly.def.template"
DEF="${NWM_COASTAL_ROOT}/ecflow_demo/suite_def/coastal_hourly.def"

# TODAY_YMD seeds the suite's `repeat date YMD` start bound (see the
# template's own comment at that line) -- computed fresh every run, not a
# fixed literal, since this script always does a `--load force` (below),
# which resets the repeat to whatever's in the def at load time regardless
# of what day it actually is. Real UTC date, matching the suite's own
# real-clock calendar (no `clock` directive means real clock is the
# default) -- keeps the two in sync.
export TODAY_YMD=$(date -u +%Y%m%d)

# coastal_hourly.def is generated, not tracked in git (see .gitignore) --
# the template is the real source of truth. Explicit variable list (not
# bare envsubst) so nothing else in the file that happens to look like
# "$FOO" gets touched.
envsubst '${NWM_COASTAL_ROOT} ${NWM_RTE_ROOT} ${RUN_NGEN_ROOT} ${RUN_COASTAL_ROOT} ${TODAY_YMD}' \
  < "${TEMPLATE}" > "${DEF}"

ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}" check_only
# force: this is meant to be safely re-runnable (e.g. re-seeding after a
# long suspension per the README's cold-start procedure), which means a
# suite of the same name is often already loaded -- without force, --load
# errors out on that instead of replacing it.
ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}" force
ecflow_client --port="${ECF_PORT}" --host=localhost --begin=coastal_hourly

echo "Suite loaded and begun. State:"
ecflow_client --port="${ECF_PORT}" --host=localhost --get_state=/coastal_hourly
