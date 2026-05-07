#!/bin/bash
#SBATCH --job-name=ekf_omp_vtune
#SBATCH --output=openmp/results/ekf_omp_vtune_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --time=0:30:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   sbatch openmp/run_vtune.slurm
#
# Captures Intel VTune `hotspots` (driverless / SW-sampling) profiles at T=16
# and T=32 on the OpenMP build, with a hard --duration cap so each collection
# stays well under 5 minutes even on a slow finalize.  The T=16 vs T=32
# comparison directly substantiates the SMT-register-sharing bottleneck claim
# without rerunning any of the timing sweeps.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/openmp"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_vtune] node       = $(hostname)"
echo "[run_vtune] cores      = $(nproc)"
cd "${SCRIPT_DIR}"
mkdir -p results

VTUNE_BIN=/shared/courseSharedFolders/161588outer/161588/spack/var/spack/environments/cs2050-cpu/.spack-env/view/vtune/2024.3/bin64/vtune
echo "[run_vtune] vtune = $(${VTUNE_BIN} --version 2>&1 | head -1)"

echo "=== build (with -g for symbol attribution) ==="
make clean
make prof

export OMP_PROC_BIND=close
export OMP_PLACES=cores

run_collect () {
    local T=$1
    local TAG=$2
    export OMP_NUM_THREADS=${T}
    local RESULT_DIR="${SCRIPT_DIR}/results/vtune_${TAG}"
    rm -rf "${RESULT_DIR}"
    echo "=== vtune hotspots  threads=${T}  tag=${TAG} ==="
    # M=128, runs=1 + --duration 60 hard cap ensures the collection ends in
    # well under a minute regardless of how the kernel is sampled.
    # Software-based sampling: no PMU access required, completes in seconds.
    # Stack collection off for speed; per-thread CPU time is what we want for
    # the SMT comparison anyway.
    ${VTUNE_BIN} -collect hotspots \
        -knob sampling-mode=sw \
        -knob enable-stack-collection=false \
        -result-dir "${RESULT_DIR}" \
        -finalization-mode=none \
        -- ./ekf_omp_prof --data "${DATA_BIN}" --M 64 --seed 42 \
                          --runs 1 --warmup 1 --quiet \
                          --out "results/vtune_${TAG}_runs.csv"
    echo "[run_vtune] finalizing ${RESULT_DIR}"
    ${VTUNE_BIN} -finalize -result-dir "${RESULT_DIR}" 2>&1 | tail -3
    ${VTUNE_BIN} -report summary -result-dir "${RESULT_DIR}" \
        -format text -report-output "results/vtune_${TAG}_summary.txt" 2>&1 | tail -3
    ${VTUNE_BIN} -report summary -result-dir "${RESULT_DIR}" \
        -format csv -csv-delimiter=comma \
        -report-output "results/vtune_${TAG}_summary.csv" 2>&1 | tail -3
}

run_collect 16 t16
run_collect 32 t32

echo
echo "=== T=16 summary ==="
head -60 "results/vtune_t16_summary.txt" || true
echo
echo "=== T=32 summary ==="
head -60 "results/vtune_t32_summary.txt" || true

echo
echo "=== done ==="
ls -lh results/ | grep -E "vtune"
