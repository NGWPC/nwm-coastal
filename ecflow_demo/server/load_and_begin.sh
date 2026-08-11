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

# coastal_hourly.def is generated, not tracked in git (see .gitignore) --
# the template is the real source of truth. Explicit variable list (not
# bare envsubst) so nothing else in the file that happens to look like
# "$FOO" gets touched.
envsubst '${NWM_COASTAL_ROOT} ${NWM_RTE_ROOT} ${RUN_NGEN_ROOT} ${RUN_COASTAL_ROOT}' \
  < "${TEMPLATE}" > "${DEF}"

ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}" check_only
ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}"
ecflow_client --port="${ECF_PORT}" --host=localhost --begin=coastal_hourly

echo "Suite loaded and begun. State:"
ecflow_client --port="${ECF_PORT}" --host=localhost --get_state=/coastal_hourly
