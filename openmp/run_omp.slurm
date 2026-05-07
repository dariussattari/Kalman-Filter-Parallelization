#!/bin/bash
#SBATCH --job-name=ekf_omp
#SBATCH --output=openmp/results/ekf_omp_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#
# Submit from the repo root so --output above resolves correctly:
#   cd /shared/home/das1301/project-das1301
#   mkdir -p openmp/results       # only needed once
#   sbatch openmp/run_omp.slurm

set -euo pipefail

# Locate repo root (mirrors run_serial.slurm).  When sbatch runs the script,
# Slurm stages it into a read-only spool dir, so BASH_SOURCE points there;
# use SLURM_SUBMIT_DIR (set by Slurm to your submit cwd) instead.
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
SCRIPT_DIR="${REPO_ROOT}/openmp"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_omp] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_omp] SCRIPT_DIR = ${SCRIPT_DIR}"
echo "[run_omp] DATA_BIN   = ${DATA_BIN}"
echo "[run_omp] node       = $(hostname),  user = $(whoami)"

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found. Run python ${REPO_ROOT}/data/preprocess.py first." >&2
    exit 1
fi

echo "=== build ==="
make clean
make

# Pin threads to physical cores; avoids OS migration noise during timing.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

THREAD_COUNTS=(1 2 4 8 16 32)

# ---- Strong scaling: fix M=1024, sweep threads ----
rm -f results/strong.csv
for T in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$T
    echo "=== strong  M=1024  threads=$T ==="
    ./ekf_omp --data "${DATA_BIN}" --M 1024 --seed 42 \
              --runs 5 --warmup 1 --out results/strong.csv
done

# ---- Weak scaling: fix M_per_thread = 32, scale total M with thread count ----
rm -f results/weak.csv
for T in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$T
    M=$((32 * T))
    echo "=== weak    M=$M  threads=$T ==="
    ./ekf_omp --data "${DATA_BIN}" --M ${M} --seed 42 \
              --runs 5 --warmup 1 --out results/weak.csv
done

# ---- Golden checksum (same M=128 as serial for direct comparison) ----
export OMP_NUM_THREADS=16
./ekf_omp --data "${DATA_BIN}" --M 128 --seed 42 \
          --runs 1 --warmup 1 --golden results/golden.csv

# ---- Validate vs serial within 1e-9 relative ----
echo "=== correctness check vs serial ==="
python3 "${REPO_ROOT}/common/check_golden.py" \
    --reference "${REPO_ROOT}/serial/results/golden.csv" \
    --candidate results/golden.csv \
    --tol 1e-9

# ---- Plots ----
echo "=== plot scaling ==="
python3 plot_scaling.py \
    --strong results/strong.csv \
    --weak   results/weak.csv \
    --out-dir results/

echo "=== done ==="
ls -lh results/
