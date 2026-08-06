#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./ecf_env.sh

DEF="/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/suite_def/coastal_hourly.def"

ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}" check_only
ecflow_client --port="${ECF_PORT}" --host=localhost --load="${DEF}"
ecflow_client --port="${ECF_PORT}" --host=localhost --begin=coastal_hourly

echo "Suite loaded and begun. State:"
ecflow_client --port="${ECF_PORT}" --host=localhost --get_state=/coastal_hourly
