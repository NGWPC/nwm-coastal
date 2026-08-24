ecflow_client --complete   # tell ecFlow the task finished successfully
trap 0                     # clear all traps so ERROR() doesn't also fire on exit
exit 0
