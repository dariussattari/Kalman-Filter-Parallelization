#!/bin/bash
#SBATCH --job-name=ekf_cuda
#SBATCH --output=cuda/results/ekf_cuda_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   mkdir -p cuda/results
#   sbatch cuda/run_cuda.slurm

set -euo pipefail

# Locate repo root (mirrors the other slurm scripts).
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
SCRIPT_DIR="${REPO_ROOT}/cuda"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_cuda] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_cuda] SCRIPT_DIR = ${SCRIPT_DIR}"
echo "[run_cuda] DATA_BIN   = ${DATA_BIN}"
echo "[run_cuda] node       = $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found. Run python ${REPO_ROOT}/data/preprocess.py first." >&2
    exit 1
fi

echo "=== build ==="
make clean
make

# ---- Throughput sweep: vary M, fixed block_size=128 ----
rm -f results/throughput.csv
for M in 64 256 1024 4096 16384 65536; do
    echo "=== throughput  M=${M}  block=128 ==="
    ./ekf_cuda --data "${DATA_BIN}" --M ${M} --seed 42 \
               --runs 5 --warmup 1 --block-size 128 \
               --out results/throughput.csv
done

# ---- Block-size sweep: vary block_size at fixed M=4096 ----
# Capped at 256: blocks of 512+ exceed the L4 register budget per SM (kernel
# uses ~96 regs/thread for x[9], P[81], plus predict/update workspaces) and
# fail with "too many resources requested for launch".  This itself is a
# useful finding — high per-thread state limits occupancy and block size.
rm -f results/blocksize.csv
for BS in 32 64 128 256; do
    echo "=== blocksize  M=4096  block=${BS} ==="
    ./ekf_cuda --data "${DATA_BIN}" --M 4096 --seed 42 \
               --runs 5 --warmup 1 --block-size ${BS} \
               --out results/blocksize.csv
done

# ---- Determinism check: re-run M=128 twice with same seed ----
./ekf_cuda --data "${DATA_BIN}" --M 128 --seed 42 \
           --runs 1 --warmup 1 --golden results/golden.csv
./ekf_cuda --data "${DATA_BIN}" --M 128 --seed 42 \
           --runs 1 --warmup 1 --golden results/golden_repeat.csv
echo "=== within-CUDA determinism (must be byte-identical) ==="
if diff -q results/golden.csv results/golden_repeat.csv > /dev/null; then
    echo "PASS: CUDA is deterministic across runs (same seed → same checksum)"
else
    echo "FAIL: CUDA results differ between runs"
    diff results/golden.csv results/golden_repeat.csv | head -10
fi
rm -f results/golden_repeat.csv

# ---- Single-realization trajectory for cross-impl figure ----
./ekf_cuda --data "${DATA_BIN}" --M 1 --seed 42 \
           --runs 1 --warmup 0 --trajectory results/trajectory.csv

# ---- Plots: CUDA's own scaling, plus 4-way trajectory overlay ----
echo "=== plot scaling ==="
python3 plot_scaling.py \
    --throughput results/throughput.csv \
    --blocksize  results/blocksize.csv \
    --out-dir    results/

echo "=== plot trajectory comparison (serial vs openmp vs mpi vs cuda) ==="
python3 "${REPO_ROOT}/common/plot_trajectory_compare.py" \
    --traj serial=${REPO_ROOT}/serial/results/trajectory.csv \
    --traj openmp=${REPO_ROOT}/openmp/results/trajectory.csv \
    --traj mpi=${REPO_ROOT}/mpi/results/trajectory.csv \
    --traj cuda=results/trajectory.csv \
    --reference serial \
    --out-dir results/

echo "=== done ==="
ls -lh results/
