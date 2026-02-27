#!/usr/bin/env bash
#SBATCH --job-name=coastal_schism
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 2
#SBATCH --ntasks-per-node=18
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

cat > "${CONFIG_FILE}" <<'EOF'
model: schism

simulation:
  start_date: 2022-01-01
  duration_hours: 12
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  include_noaa_gages: true
EOF

# For production only this line is needed
/ngen-test/coastal-calibration/coastal-calibration run "${CONFIG_FILE}"

# For running the dev version we use pixi.
# For production comment out these three lines.
# For making uv work well on NSF mounted locations
# we need to set these envs
# export UV_CACHE_DIR=/var/tmp/uv-cache
# export UV_LINK_MODE=copy
# pixi r -e dev coastal-calibration run "${CONFIG_FILE}"

rm -f "${CONFIG_FILE}"
