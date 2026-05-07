#!/bin/bash
#SBATCH --job-name=ekf_cuda_prof
#SBATCH --output=cuda/results/ekf_cuda_prof_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   sbatch cuda/run_profile.slurm
#
# Captures (1) the ptxas register/shared-memory dump from a -Xptxas=-v build,
# (2) the resulting occupancy table at each block size, and (3) opportunistic
# Nsight Compute (`ncu`) metrics if the tool is present on the GPU node.
# Together these substantiate the report's claim that the CUDA kernel is
# register-pressure-limited (block sizes >=512 fail with "too many resources").

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    while [ -n "${REPO_ROOT}" ] && [ ! -d "${REPO_ROOT}/data" ]; do
        REPO_ROOT="${REPO_ROOT%/*}"
    done
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/cuda"
DATA_BIN="${REPO_ROOT}/data/v101.bin"

echo "[run_cuda_prof] REPO_ROOT  = ${REPO_ROOT}"
echo "[run_cuda_prof] node       = $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found." >&2
    exit 1
fi

echo "=== build with -Xptxas=-v ==="
# Use the system make (/usr/bin/make).  The spack-env make on $PATH was built
# for `linux-skylake_avx512` (login node) and SIGILLs on the GPU node's
# different CPU class (e.g. A10G hosts).  /usr/bin/make is portable.
MAKE=/usr/bin/make
$MAKE clean
# Tee stderr (where ptxas writes its summary) to results/ptxas.log; keep stdout
# on the screen.  `2> >(tee ...)` is bash-native.
$MAKE prof 2> >(tee results/ptxas.log >&2)

echo
echo "=== ptxas log ==="
grep -E "ptxas info" results/ptxas.log || true

echo
echo "=== occupancy table (derived from ptxas + L4 sm_89 limits) ==="
python3 occupancy_table.py \
    --ptxas-log results/ptxas.log \
    --output    results/occupancy_table.md \
    --arch      sm_89

echo
echo "=== probe for nsight tools on this GPU node ==="
HAS_NCU=0
HAS_NSYS=0
which ncu  >/dev/null 2>&1 && HAS_NCU=1  && echo "[run_cuda_prof] ncu  found: $(ncu --version  2>&1 | head -1)"
which nsys >/dev/null 2>&1 && HAS_NSYS=1 && echo "[run_cuda_prof] nsys found: $(nsys --version 2>&1 | head -1)"
if [ "$HAS_NCU" = "0" ] && [ "$HAS_NSYS" = "0" ]; then
    echo "[run_cuda_prof] Nsight tools not installed on this node — relying on ptxas + occupancy_table for evidence."
fi

if [ "$HAS_NCU" = "1" ]; then
    echo
    echo "=== ncu --set basic on ekf_cuda (M=4096, block=128) ==="
    ncu --set basic --target-processes all \
        --export results/ncu_block128 \
        --force-overwrite \
        ./ekf_cuda_prof --data "${DATA_BIN}" --M 4096 --seed 42 \
                        --runs 1 --warmup 0 --block-size 128 --quiet \
        2>&1 | tee results/ncu_block128.log || true
    # Extract the headline metrics into a small text summary.
    ncu --import results/ncu_block128.ncu-rep \
        --print-summary per-kernel \
        --csv > results/ncu_block128_summary.csv 2>/dev/null || true
fi

if [ "$HAS_NSYS" = "1" ]; then
    echo
    echo "=== nsys profile on ekf_cuda (M=4096, block=128) ==="
    nsys profile -t cuda,nvtx \
        -o results/nsys_block128 \
        --force-overwrite=true \
        ./ekf_cuda_prof --data "${DATA_BIN}" --M 4096 --seed 42 \
                        --runs 1 --warmup 0 --block-size 128 --quiet \
        2>&1 | tail -40 || true
fi

echo
echo "=== done ==="
ls -lh results/ | grep -E "ptxas|ncu|nsys|occupancy"
