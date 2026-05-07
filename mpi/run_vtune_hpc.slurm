#!/bin/bash
#SBATCH --job-name=ekf_mpi_vtune
#SBATCH --output=mpi/results/ekf_mpi_vtune_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0:30:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   sbatch mpi/run_vtune.slurm
#
# Captures Intel VTune `hpc-performance` profiles on rank 0 of two MPI
# configurations: a 4-rank job (within one NUMA socket) and an 8-rank job
# (crossing the inter-socket boundary).  The contrast directly substantiates
# the report's MPI 4->8 NUMA-boundary bottleneck claim with hardware-counter
# evidence (memory bandwidth, LLC miss rate) instead of inferring it from
# the strong-scaling slope alone.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/mpi"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_mpi_vtune] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_mpi_vtune] node       = $(hostname)"

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found." >&2
    exit 1
fi

VTUNE_BIN=/shared/courseSharedFolders/161588outer/161588/spack/var/spack/environments/cs2050-cpu/.spack-env/view/vtune/2024.3/bin64/vtune
echo "[run_mpi_vtune] vtune = $(${VTUNE_BIN} --version 2>&1 | head -1)"

# OpenMPI HOME workaround (same fix as run_mpi.slurm).
export HOME="/shared/home/${SLURM_JOB_USER:-$USER}"
export USER="${SLURM_JOB_USER:-$USER}"
export OMPI_MCA_rmaps_base_mapping_policy=core
export OMPI_MCA_hwloc_base_binding_policy=core

echo "=== build (with -g for symbol attribution) ==="
make clean
make prof

# Wrapper that VTune-attaches only to MPI rank 0 and runs the binary plain on
# all other ranks.  Avoids the "vtune doesn't know about MPI" problem and
# keeps timing on non-instrumented ranks honest.
WRAPPER="results/vtune_rank0_wrapper.sh"
cat > "${WRAPPER}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-\${SLURM_PROCID:-0}}}"
if [ "\$RANK" = "0" ]; then
    exec "${VTUNE_BIN}" -collect hpc-performance \\
        -result-dir "\${VTUNE_RESULT_DIR}" \\
        -finalization-mode=none \\
        -- "\$@"
else
    exec "\$@"
fi
EOF
chmod +x "${WRAPPER}"

run_collect () {
    local R=$1
    local TAG=$2
    local RESULT_DIR="${SCRIPT_DIR}/results/vtune_${TAG}"
    rm -rf "${RESULT_DIR}"
    export VTUNE_RESULT_DIR="${RESULT_DIR}"
    echo "=== vtune hpc-performance  ranks=${R}  tag=${TAG} ==="
    # M = 64 * R (matches weak-scaling cadence: 64 realizations per rank).
    local M=$((64 * R))
    srun --export=ALL,HOME=$HOME,VTUNE_RESULT_DIR=$VTUNE_RESULT_DIR \
         -n "${R}" --cpu-bind=cores \
         "${WRAPPER}" ./ekf_mpi_prof \
            --data "${DATA_BIN}" --M ${M} --seed 42 \
            --runs 2 --warmup 1 --quiet \
            --out "results/vtune_${TAG}_runs.csv"
    # Finalize result dir so we can run reports against it.
    ${VTUNE_BIN} -finalize -result-dir "${RESULT_DIR}" || true
    ${VTUNE_BIN} -report summary \
        -result-dir "${RESULT_DIR}" \
        -format text \
        -report-output "results/vtune_${TAG}_summary.txt"
    ${VTUNE_BIN} -report summary \
        -result-dir "${RESULT_DIR}" \
        -format csv -csv-delimiter=comma \
        -report-output "results/vtune_${TAG}_summary.csv"
    echo "[run_mpi_vtune] wrote ${RESULT_DIR}/, results/vtune_${TAG}_summary.{txt,csv}"
}

run_collect 4 r4
run_collect 8 r8

echo "=== summary excerpts ==="
for TAG in r4 r8; do
    echo "----- ${TAG} (rank 0) -----"
    head -80 "results/vtune_${TAG}_summary.txt"
    echo
done

echo "=== done ==="
ls -lh results/ | grep -E "vtune|^total"
