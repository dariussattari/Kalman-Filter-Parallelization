#!/bin/bash
#SBATCH --job-name=ekf_mpi_vtune_r8only
#SBATCH --output=mpi/results/ekf_mpi_vtune_r8only_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0:10:00

set -uo pipefail   # NOT -e: we want to continue past nonzero `tail` etc.

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
fi
SCRIPT_DIR="${REPO_ROOT}/mpi"
DATA_BIN="${REPO_ROOT}/data/v101.bin"
cd "${SCRIPT_DIR}"

VTUNE_BIN=/shared/courseSharedFolders/161588outer/161588/spack/var/spack/environments/cs2050-cpu/.spack-env/view/vtune/2024.3/bin64/vtune
MPIRUN=/opt/amazon/openmpi/bin/mpirun
export HOME="/shared/home/${SLURM_JOB_USER:-$USER}"
export USER="${SLURM_JOB_USER:-$USER}"
export OMPI_MCA_rmaps_base_mapping_policy=core
export OMPI_MCA_hwloc_base_binding_policy=core

if [ ! -x ./ekf_mpi_prof ]; then
    /usr/bin/make prof
fi

RESULT_DIR="${SCRIPT_DIR}/results/vtune_r8_sw"
rm -rf "${RESULT_DIR}" "${RESULT_DIR}".*

collect_one () {
    local R=$1
    local TAG=$2
    local M=$3
    local DIR="${SCRIPT_DIR}/results/vtune_${TAG}_sw"
    rm -rf "${DIR}" "${DIR}".*
    echo
    echo "=== vtune hotspots(sw) ranks=${R} M=${M} (mpirun) ==="
    ${VTUNE_BIN} -collect hotspots \
        -knob sampling-mode=sw \
        -knob enable-stack-collection=false \
        -result-dir "${DIR}" \
        -finalization-mode=none \
        -- ${MPIRUN} -n ${R} --bind-to none --oversubscribe ./ekf_mpi_prof \
            --data "${DATA_BIN}" --M ${M} --seed 42 \
            --runs 3 --warmup 1 --quiet \
            --out "results/vtune_${TAG}_sw_runs.csv"

    local ACT=$(ls -d "${DIR}" "${DIR}".* 2>/dev/null | head -1)
    ${VTUNE_BIN} -finalize -result-dir "${ACT}" 2>&1 | tail -3
    ${VTUNE_BIN} -report summary -result-dir "${ACT}" \
        -format text -report-output "results/vtune_${TAG}_sw_summary.txt" 2>&1 | tail -3
    ${VTUNE_BIN} -report summary -result-dir "${ACT}" \
        -format csv -csv-delimiter=comma \
        -report-output "results/vtune_${TAG}_sw_summary.csv" 2>&1 | tail -3
    echo
    echo "===== ${TAG} SUMMARY ====="
    head -25 "results/vtune_${TAG}_sw_summary.txt"
}

# Both at M=2048 so the wall time is large enough for 100Hz sw sampling to
# converge on the actual ekf workload (~5–10 seconds each).
collect_one 4 r4 2048
collect_one 8 r8 2048
echo
echo "=== done ==="
ls -lh results/ | grep vtune
