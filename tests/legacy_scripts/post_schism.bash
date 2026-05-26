#!/usr/bin/env bash

set -ex

#--------------------------------------------------------------
#
#This task runs SCHISM on the prepared coastal inputs, combines output and restarts, and writes output to:
#
#   $DATAexec/outputs
#
post_nwm_coastal() {

   if [[ ! -d $DATAexec/outputs ]]; then
       mkdir -p $DATAexec/outputs
   fi
   cd $DATAexec

   # if outputs/fatal.error size > 0, mark our status as failed
   if [ -s outputs/fatal.error ]; then
      echo "pschism failed. See $DATAexec/outputs/fatal.error file for more detail."
      exit 1
   fi

   cd ./outputs
   # combine hotstarts for analysis, or if running a chained reanalysis
   if [[ $LENGTH_HRS -lt 0 || "$CHAINED_REANALYSIS" != "" ]]; then
       # create the hotstart for the next AnA run
       let hotstart_it=18*${RESTART_WRITE_HR/#-}
       combine_hotstart7
   fi
   # remove traps so below cleanup doesn't error out if the
   # files don't exist
   trap 0
   cd ../
}
