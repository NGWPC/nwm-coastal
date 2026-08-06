#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./ecf_env.sh

mkdir -p "${ECF_HOME}"

if ecflow_client --port="${ECF_PORT}" --host=localhost --ping >/dev/null 2>&1; then
  echo "ecflow_server already running on port ${ECF_PORT}"
  exit 0
fi

cd "${ECF_HOME}"     # server chdir's here anyway on startup; doing it explicitly
                      # keeps nohup's redirected file colocated too
nohup ecflow_server --port="${ECF_PORT}" >"${ECF_HOME}/ecflow_server.nohup.out" 2>&1 &
disown

sleep 2
ecflow_client --port="${ECF_PORT}" --host=localhost --ping

# ecflow_server starts in a HALTED state by default (does not process job
# submissions/dependencies until told to). --restart is the server-level
# unhalt; unrelated to --begin, which is a per-suite command run separately
# in load_and_begin.sh.
ecflow_client --port="${ECF_PORT}" --host=localhost --restart

echo "ecflow_server started on $(hostname):${ECF_PORT}  ECF_HOME=${ECF_HOME}"
