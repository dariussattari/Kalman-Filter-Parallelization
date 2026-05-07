#!/bin/bash
#SBATCH --job-name=ekf_jax
#SBATCH --output=additional/results/ekf_jax_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   mkdir -p additional/results
#   sbatch additional/run_jax.slurm
#
# Assumes JAX is already installed under ~/.local for python3.9
# (pip install --user "jax[cuda12]"==0.4.30).  Same L4 GPU as ekf_cuda
# so timings are directly comparable.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/additional"
DATA_BIN="${REPO_ROOT}/data/v101.bin"
PY=python3.9

echo "[run_jax] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_jax] SCRIPT_DIR = ${SCRIPT_DIR}"
echo "[run_jax] DATA_BIN   = ${DATA_BIN}"
echo "[run_jax] node       = $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found." >&2
    exit 1
fi

# Confirm the python and JAX backend we'll use.
${PY} -c "import jax; print('[run_jax] jax', jax.__version__, 'on', jax.default_backend())"

# ---- Throughput sweep: vary M, fixed JIT amortised over 5 timed runs ----
rm -f results/throughput.csv
for M in 64 256 1024 4096 16384 65536; do
    echo "=== throughput  M=${M} ==="
    ${PY} ekf_jax.py --data "${DATA_BIN}" --M ${M} --seed 42 \
        --runs 5 --warmup 1 \
        --out results/throughput.csv
done

# ---- Determinism check: re-run M=128 twice with same seed ----
${PY} ekf_jax.py --data "${DATA_BIN}" --M 128 --seed 42 \
    --runs 1 --warmup 1 --golden results/golden.csv
${PY} ekf_jax.py --data "${DATA_BIN}" --M 128 --seed 42 \
    --runs 1 --warmup 1 --golden results/golden_repeat.csv
echo "=== within-JAX determinism (must be byte-identical) ==="
if diff -q results/golden.csv results/golden_repeat.csv > /dev/null; then
    echo "PASS: JAX is deterministic across runs (same key → same output)"
else
    echo "FAIL: JAX golden files differ between runs"
    diff results/golden.csv results/golden_repeat.csv | head -10
fi
rm -f results/golden_repeat.csv

# ---- Single-realization trajectory for cross-impl figure ----
${PY} ekf_jax.py --data "${DATA_BIN}" --M 1 --seed 42 \
    --runs 1 --warmup 1 --trajectory results/trajectory.csv

# ---- Plot: 5-way trajectory overlay (serial / openmp / mpi / cuda / jax) ----
echo "=== plot trajectory comparison: 5 implementations ==="
${PY} "${REPO_ROOT}/common/plot_trajectory_compare.py" \
    --traj serial=${REPO_ROOT}/serial/results/trajectory.csv \
    --traj openmp=${REPO_ROOT}/openmp/results/trajectory.csv \
    --traj mpi=${REPO_ROOT}/mpi/results/trajectory.csv \
    --traj cuda=${REPO_ROOT}/cuda/results/trajectory.csv \
    --traj jax=results/trajectory.csv \
    --reference serial \
    --out-dir results/

# ---- Plot: cross-implementation inference comparison ----
echo "=== plot cross-implementation inference comparison ==="
${PY} "${REPO_ROOT}/common/plot_inference_compare.py" \
    --serial   ${REPO_ROOT}/serial/results/results.csv \
    --openmp   ${REPO_ROOT}/openmp/results/strong.csv \
    --openmp-weak ${REPO_ROOT}/openmp/results/weak.csv \
    --mpi      ${REPO_ROOT}/mpi/results/strong.csv \
    --mpi-weak    ${REPO_ROOT}/mpi/results/weak.csv \
    --cuda     ${REPO_ROOT}/cuda/results/throughput.csv \
    --jax      results/throughput.csv \
    --out-dir  results/

echo "=== done ==="
ls -lh results/
