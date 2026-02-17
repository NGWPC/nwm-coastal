#!/usr/bin/env bash
#SBATCH --job-name=coastal_schism
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 2
#SBATCH --ntasks-per-node=18
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

# On the INT cluster, /ngwpc-coastal is not available, so
# on compute nodes and we need to use the one in /ngen-test.
# On the UAT cluster, remove paths and parm_dir
cat > "${CONFIG_FILE}" <<'EOF'
model: schism

simulation:
  start_date: 2022-01-01
  duration_hours: 12
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs

paths:
  parm_dir: /ngen-test/coastal/ngwpc-coastal

model_config:
  include_noaa_gages: true
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
