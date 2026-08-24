#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./ecf_env.sh

ecflow_client --port="${ECF_PORT}" --host=localhost --halt=yes
ecflow_client --port="${ECF_PORT}" --host=localhost --terminate=yes
