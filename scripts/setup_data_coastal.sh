#!/bin/bash
# Download the data needed to run the NWM Coastal workflows.
#
# Requires AWS credentials with read access to s3://ngwpc-dev. Configure them
# (aws configure, SSO, or environment variables) before running.
#
# Downloads, from s3://ngwpc-dev/nwm-tools-data/:
#
#   esmf/esmf_mesh/NWM/domain       -> <run_ngen>/data/esmf_mesh/NWM/domain
#   coastal_data/esmf_domain_extract-> <run_ngen>/data/esmf_mesh/esmf_domain_extract
#   coastal_data/hydrofabric_copies -> <coastal_data>/hydrofabric_copies
#   coastal_data/TPXO10_atlas_v2_nc -> <coastal_data>/TPXO10_atlas_v2_nc
#   coastal_data/run_coastal        -> <run_coastal>   (contents at top level)
#
# Destinations default to siblings of the nwm-coastal checkout and can be set
# with RUN_NGEN_ROOT / RUN_COASTAL_ROOT, or entered when prompted.
#
# By default aws s3 sync re-downloads a file when the S3 object is newer than
# the local copy or the sizes differ. Pass --size-only to compare size alone, 
# or --no-overwrite to keep every file already on disk.
#
# Usage:
#   ./scripts/setup_data_coastal.sh
#   ./scripts/setup_data_coastal.sh --dry-run
#   ./scripts/setup_data_coastal.sh --yes
#   ./scripts/setup_data_coastal.sh --size-only
#   ./scripts/setup_data_coastal.sh --no-overwrite

set -euo pipefail

BUCKET="ngwpc-dev"
PREFIX="nwm-tools-data"

ASSUME_YES="false"
SYNC_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) ASSUME_YES="true" ;;
        --dry-run) SYNC_OPTS+=("--dryrun") ;;
        --size-only|--no-overwrite) SYNC_OPTS+=("$1") ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
    shift
done

function info () { echo "INFO: $*" >&2; }
function fatal () { echo "FATAL ERROR: $*" >&2; exit 1; }

# Echo a directory path: the env value if set, the user's answer if they are
# prompted, otherwise the default.
function prompt_dir () {
    local label="${1}" current="${2}" default="${3}" answer

    if [[ -n "${current}" ]]; then
        echo "${current}"
        return 0
    fi
    if [[ "${ASSUME_YES}" == "true" || ! -t 0 ]]; then
        echo "${default}"
        return 0
    fi

    read -r -p "Where should ${label} go? [${default}] " answer </dev/tty
    echo "${answer:-${default}}"
}

function sync_dir () {
    local src="s3://${BUCKET}/${PREFIX}/${1}/" dst="${2}/"
    info "Syncing ${src} -> ${dst}"
    mkdir -p "${dst}"
    aws s3 sync ${SYNC_OPTS[@]+"${SYNC_OPTS[@]}"} "${src}" "${dst}"
}

command -v aws >/dev/null 2>&1 || fatal "aws CLI not found. Install it: https://aws.amazon.com/cli/"

if ! aws sts get-caller-identity >/dev/null 2>&1; then
    fatal "No valid AWS credentials. Configure access to s3://${BUCKET} and re-run."
fi

NWM_COASTAL_ROOT="${NWM_COASTAL_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
SIBLING_ROOT="$(dirname "${NWM_COASTAL_ROOT}")"

RUN_NGEN_ROOT="$(prompt_dir "run_ngen" "${RUN_NGEN_ROOT:-}" "${SIBLING_ROOT}/run_ngen")"
RUN_COASTAL_ROOT="$(prompt_dir "run_coastal" "${RUN_COASTAL_ROOT:-}" "${SIBLING_ROOT}/run_coastal")"
COASTAL_DATA_DIR="$(prompt_dir "coastal_data" "" "${SIBLING_ROOT}/coastal_data")"

ESMF_MESH_DIR="${RUN_NGEN_ROOT}/data/esmf_mesh"

cat >&2 <<SUMMARY

Downloading from s3://${BUCKET}/${PREFIX}/ into:

  esmf/esmf_mesh/NWM/domain        -> ${ESMF_MESH_DIR}/NWM/domain
  coastal_data/esmf_domain_extract -> ${ESMF_MESH_DIR}/esmf_domain_extract
  coastal_data/hydrofabric_copies  -> ${COASTAL_DATA_DIR}/hydrofabric_copies
  coastal_data/TPXO10_atlas_v2_nc  -> ${COASTAL_DATA_DIR}/TPXO10_atlas_v2_nc
  coastal_data/run_coastal         -> ${RUN_COASTAL_ROOT}

SUMMARY

if [[ "${ASSUME_YES}" != "true" && -t 0 ]]; then
    read -r -p "Proceed? [Y/n] " reply </dev/tty
    [[ "${reply:-Y}" =~ ^[Yy]?$ ]] || fatal "Aborted."
fi

sync_dir "esmf/esmf_mesh/NWM/domain" "${ESMF_MESH_DIR}/NWM/domain"
sync_dir "coastal_data/esmf_domain_extract" "${ESMF_MESH_DIR}/esmf_domain_extract"
sync_dir "coastal_data/hydrofabric_copies" "${COASTAL_DATA_DIR}/hydrofabric_copies"
sync_dir "coastal_data/TPXO10_atlas_v2_nc" "${COASTAL_DATA_DIR}/TPXO10_atlas_v2_nc"
sync_dir "coastal_data/run_coastal" "${RUN_COASTAL_ROOT}"

cat >&2 <<DONE

Done. Data is in:

  ${ESMF_MESH_DIR}
  ${COASTAL_DATA_DIR}
  ${RUN_COASTAL_ROOT}

Point paths.tidal_atlas_dir at ${COASTAL_DATA_DIR}/TPXO10_atlas_v2_nc in any
run config that uses harmonic tides.
DONE
