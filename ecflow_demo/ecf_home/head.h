set -e -u -x   # abort on error/unset var, print commands as executed

# Variables ecFlow substitutes at job-generation time (ecflow 5.6.0)
export ECF_NAME=%ECF_NAME%
export ECF_HOST=%ECF_HOST%
export ECF_PORT=%ECF_PORT%
export ECF_PASS=%ECF_PASS%
export ECF_TRYNO=%ECF_TRYNO%
export ECF_RID=$$              # no batch scheduler wrapping this job -> use the shell PID

export PATH="/contrib/software/ecflow/5.6.0/bin:${PATH}"

# Tell ecFlow this task has started
ecflow_client --init="${ECF_RID}"

# Error handler: report --abort to ecFlow instead of leaving the task
# stuck 'active' forever. Guarded against double-firing (ERR then EXIT).
_ERROR_HANDLED=0
ERROR() {
  local rc=$1
  set +e
  if [ "${_ERROR_HANDLED}" -eq 1 ]; then
    exit "${rc}"
  fi
  _ERROR_HANDLED=1
  trap - ERR EXIT
  ecflow_client --abort="task failed (rc=${rc})" || true
  exit "${rc}"
}
trap 'ERROR $?' ERR
trap 'ERROR $?' EXIT
