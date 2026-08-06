#!/bin/bash
# Shared env for start_server.sh / stop_server.sh / load_and_begin.sh.
# source this, don't execute it directly.

export ECF_PORT="${ECF_PORT:-39411}"   # arbitrary local port, avoids the ecflow-default 3141
export ECF_HOME="/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home"
export PATH="/contrib/software/ecflow/5.6.0/bin:${PATH}"
