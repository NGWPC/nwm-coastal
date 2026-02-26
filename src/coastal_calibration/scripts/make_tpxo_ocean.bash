#!/usr/bin/env bash

set -ex

make_tpxo_ocean() {
   #
   # create water level boundary file from TPXO data
   #

#   start_date=${FORCING_BEGIN_DATE:0:8}
#   start_hour=${FORCING_START_HOUR}

   PDY=${1:0:8}
   cyc=${1:8}
   export LENGTH_HRS=$2
   export OTPSnc_DIR=$3
   export NGEN_FORCING_DIR=$4
   export SCHISM_PARM_DIR=$5
   export COASTAL_DOMAIN=$6
   export TIME_STEP_IN_SECS=$7

#   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/contrib/software/gcc/8.5.0/lib64:/contrib/software/netcdf/4.7.4/lib

   # END_DATETIME is precomputed by the Python package
   python $TPXO_SCRIPTS_DIR/make_otps_input.py \
	  $SCHISM_PARM_DIR/$COASTAL_DOMAIN/open_bnds_hgrid.nc \
	  $PDY$cyc $END_DATETIME $TIME_STEP_IN_SECS  \
	  $DATAexec/otps_lat_lon_time.txt

   cp "$SCRIPTS_DIR/setup_tpxo.txt" $DATAexec/
   cp "$SCRIPTS_DIR/Model_tpxo10_atlas" $DATAexec/

   cd $DATAexec

#   ln -sf $OTPSnc_DIR/DATA ./
   ln -sf $NGWPC_COASTAL_PARM_DIR/TPXO10_atlas_v2_nc .

   $OTPSnc_DIR/predict_tide < setup_tpxo.txt

   python $TPXO_SCRIPTS_DIR/otps_to_open_bnds_hgrid.py  \
	  ./otps_out.txt                                                               \
          $SCHISM_PARM_DIR/$COASTAL_DOMAIN/open_bnds_hgrid.nc                          \
	  ./elev2D.th.nc

   local _correction_file=$DATAexec/elevation_correction.csv
   if [[ -f  ${_correction_file} ]]; then
       echo "Applying elevation datum correction to elev2D.th.nc file"
       python $COASTAL_SCRIPTS_DIR/correct_elevation.py \
       ./elev2D.th.nc ${_correction_file}
   fi
   cd -
}
