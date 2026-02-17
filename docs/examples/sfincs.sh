#!/usr/bin/env bash
#SBATCH --job-name=coastal_sfincs
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

# On the INT cluster, /ngwpc-coastal is not available, so
# on compute nodes and we need to use the one in /ngen-test.
# On the UAT cluster, remove paths and parm_dir
cat > "${CONFIG_FILE}" <<'EOF'
model: sfincs

simulation:
  start_date: 2020-05-11
  duration_hours: 168
  coastal_domain: atlgulf
  meteo_source: nwm_retro

boundary:
  source: tpxo

paths:
  parm_dir: /ngen-test/coastal/ngwpc-coastal

model_config:
  prebuilt_dir: /ngen-dev/taher.chegini/nwm-coastal-dev/docs/examples/texas
  include_noaa_gages: true
  forcing_to_mesh_offset_m: 0.171
  vdatum_mesh_to_msl_m: 0.171
  include_precip: true
  include_wind: true
  include_pressure: true
EOF

# For production only this line is needed
# coastal-calibration run "${CONFIG_FILE}"

# For running the dev version we use pixi.
# For production comment out these three lines.
# For making uv to work well on NSF mounted locations
# we need to set these envs
export UV_CACHE_DIR=$HOME/.uv-cache
export UV_LINK_MODE=copy
pixi r -e dev coastal-calibration run "${CONFIG_FILE}"

rm -f "${CONFIG_FILE}"
