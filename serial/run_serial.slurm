#!/bin/bash
#SBATCH --job-name=ekf_serial
#SBATCH --output=serial/results/ekf_serial_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#
# Submit from the repo root so --output above resolves correctly:
#   cd /shared/home/das1301/project-das1301
#   mkdir -p serial/results       # only needed once
#   sbatch serial/run_serial.slurm

# Serial baseline: build, sweep M, save scaling-vs-workload data, save the
# golden checksum and one trajectory for the report's "does it work?" figure.
#
# Submit from the repo root:    sbatch serial/run_serial.slurm
# (or anywhere from inside the repo; SLURM_SUBMIT_DIR is anchored on submission cwd)

set -euo pipefail

# When sbatch runs, Slurm stages the script into a read-only spool, so
# BASH_SOURCE points there.  Use SLURM_SUBMIT_DIR (the directory you ran
# sbatch from) to find the repo.  Falls back to the script's own location
# when run interactively.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    # SLURM_SUBMIT_DIR may be the repo root or a subdir; walk up to find data/.
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
SCRIPT_DIR="${REPO_ROOT}/serial"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_serial] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_serial] SCRIPT_DIR = ${SCRIPT_DIR}"
echo "[run_serial] DATA_BIN   = ${DATA_BIN}"
echo "[run_serial] node       = $(hostname),  user = $(whoami)"

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found. Run python ${REPO_ROOT}/data/preprocess.py first." >&2
    exit 1
fi

echo "=== build ==="
make clean
make

# Fresh results CSV each submission.
rm -f results/results.csv

echo "=== M sweep (warmup=1, runs=5 each) ==="
for M in 16 64 256 1024; do
    echo "--- M=${M} ---"
    ./ekf_serial \
        --data "${DATA_BIN}" \
        --M "${M}" \
        --seed 42 \
        --warmup 1 \
        --runs 5 \
        --out results/results.csv
done

echo "=== golden checksum + trajectory (M=128) ==="
./ekf_serial \
    --data "${DATA_BIN}" \
    --M 128 \
    --seed 42 \
    --warmup 1 \
    --runs 1 \
    --golden results/golden.csv \
    --trajectory results/trajectory.csv

echo "=== plot trajectory vs ground truth ==="
python3 plot_trajectory.py \
    --csv results/trajectory.csv \
    --out results/serial_trajectory.png

echo "=== done ==="
ls -lh results/
