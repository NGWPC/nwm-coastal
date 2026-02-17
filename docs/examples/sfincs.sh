#!/usr/bin/env bash
#SBATCH --job-name=coastal_sfincs
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

cat > "${CONFIG_FILE}" <<'EOF'
model: sfincs

simulation:
  start_date: 2020-05-11
  duration_hours: 168
  coastal_domain: atlgulf
  meteo_source: nwm_retro

boundary:
  source: tpxo

model_config:
  prebuilt_dir: /absolute/path/to/prebuilt/sfincs/model
  include_noaa_gages: true
  forcing_to_mesh_offset_m: 0.171
  vdatum_mesh_to_msl_m: 0.171
  include_precip: true
  include_wind: true
  include_pressure: true
EOF

coastal-calibration run "${CONFIG_FILE}"
rm -f "${CONFIG_FILE}"
