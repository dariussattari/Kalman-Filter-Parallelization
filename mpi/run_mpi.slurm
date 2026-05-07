#!/bin/bash
#SBATCH --job-name=ekf_mpi
#SBATCH --output=mpi/results/ekf_mpi_%j.out
#SBATCH --partition=general
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0:30:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   mkdir -p mpi/results
#   sbatch mpi/run_mpi.slurm
#
# We allocate 4 nodes x 16 tasks = 64 ranks total, then srun -n R uses just R
# of them at a time for each point in the rank sweep.

set -euo pipefail

# Locate repo root (mirrors run_serial.slurm / run_omp.slurm).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [ -z "${REPO_ROOT}" ] || [ ! -d "${REPO_ROOT}/data" ]; then
    echo "ERROR: cannot locate repo root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi
SCRIPT_DIR="${REPO_ROOT}/mpi"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_mpi] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_mpi] SCRIPT_DIR = ${SCRIPT_DIR}"
echo "[run_mpi] DATA_BIN   = ${DATA_BIN}"
echo "[run_mpi] alloc      = ${SLURM_NNODES} nodes, ${SLURM_NTASKS} tasks"
echo "[run_mpi] nodes      = ${SLURM_JOB_NODELIST}"

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found. Run python ${REPO_ROOT}/data/preprocess.py first." >&2
    exit 1
fi

echo "=== build ==="
make clean
make

# Open MPI on dynamically-allocated nodes refuses to initialise unless HOME
# resolves to a real path inside each spawned task — set it explicitly so
# srun-launched ranks inherit it.
export HOME="/shared/home/${SLURM_JOB_USER:-$USER}"
export USER="${SLURM_JOB_USER:-$USER}"

# Open MPI bindings: pin each rank to a physical core; equivalent
# to OMP_PROC_BIND=close for the OpenMP run.
export OMPI_MCA_rmaps_base_mapping_policy=core
export OMPI_MCA_hwloc_base_binding_policy=core

RANK_COUNTS=(1 2 4 8 16 32 64)

# ---- Strong scaling: fix M=4096, sweep ranks ----
rm -f results/strong.csv
for R in "${RANK_COUNTS[@]}"; do
    echo "=== strong  M=4096  ranks=$R ==="
    srun --export=ALL,HOME=$HOME -n "$R" --cpu-bind=cores ./ekf_mpi \
        --data "${DATA_BIN}" --M 4096 --seed 42 \
        --runs 5 --warmup 1 --out results/strong.csv
done

# ---- Weak scaling: fix M_per_rank = 64, scale total M with ranks ----
rm -f results/weak.csv
for R in "${RANK_COUNTS[@]}"; do
    M=$((64 * R))
    echo "=== weak    M=$M  ranks=$R ==="
    srun --export=ALL,HOME=$HOME -n "$R" --cpu-bind=cores ./ekf_mpi \
        --data "${DATA_BIN}" --M ${M} --seed 42 \
        --runs 5 --warmup 1 --out results/weak.csv
done

# ---- Golden checksum at the same M=128 used by serial/openmp ----
srun --export=ALL,HOME=$HOME -n 16 --cpu-bind=cores ./ekf_mpi \
    --data "${DATA_BIN}" --M 128 --seed 42 \
    --runs 1 --warmup 1 --golden results/golden.csv

# ---- Single-rank realization-0 trajectory for the cross-impl figure ----
srun --export=ALL,HOME=$HOME -n 1 ./ekf_mpi \
    --data "${DATA_BIN}" --M 1 --seed 42 \
    --runs 1 --warmup 0 --trajectory results/trajectory.csv

# ---- Validate vs serial within 1e-9 ----
echo "=== correctness check vs serial ==="
python3 "${REPO_ROOT}/common/check_golden.py" \
    --reference "${REPO_ROOT}/serial/results/golden.csv" \
    --candidate results/golden.csv \
    --tol 1e-9

# ---- Plots: MPI's own scaling, plus 3-way trajectory overlay ----
echo "=== plot scaling ==="
python3 plot_scaling.py \
    --strong results/strong.csv \
    --weak   results/weak.csv \
    --out-dir results/

echo "=== plot trajectory comparison (serial vs openmp vs mpi) ==="
python3 "${REPO_ROOT}/common/plot_trajectory_compare.py" \
    --traj serial=${REPO_ROOT}/serial/results/trajectory.csv \
    --traj openmp=${REPO_ROOT}/openmp/results/trajectory.csv \
    --traj mpi=results/trajectory.csv \
    --reference serial \
    --out-dir results/

echo "=== done ==="
ls -lh results/
