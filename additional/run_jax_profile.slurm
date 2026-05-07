#!/bin/bash
#SBATCH --job-name=ekf_jax_prof
#SBATCH --output=additional/results/ekf_jax_prof_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#
# Submit from the repo root:
#   cd /shared/home/das1301/project-das1301
#   sbatch additional/run_jax_profile.slurm
#
# Captures a `jax.profiler.trace(...)` of the M=16,384 batch run on the L4
# GPU.  The resulting Perfetto trace contains per-op self-time and HBM
# allocation events, directly quantifying the noise-tensor + per-step cost
# the report attributes to JAX's pre-generated noise stream.

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

echo "[run_jax_prof] node = $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

cd "${SCRIPT_DIR}"
mkdir -p results

if [ ! -f "${DATA_BIN}" ]; then
    echo "ERROR: ${DATA_BIN} not found." >&2
    exit 1
fi

${PY} -c "import jax; print('[run_jax_prof] jax', jax.__version__, 'on', jax.default_backend())"

TRACE_DIR="results/jax_profile"
rm -rf "${TRACE_DIR}"
mkdir -p "${TRACE_DIR}"

echo "=== jax.profiler.trace at M=16384, runs=3 ==="
${PY} ekf_jax.py --data "${DATA_BIN}" --M 16384 --seed 42 \
    --runs 3 --warmup 1 \
    --profile-trace "${TRACE_DIR}"

echo
echo "=== trace artefacts ==="
find "${TRACE_DIR}" -type f -printf "%p\t%s bytes\n" | head -40

echo
echo "=== top ops by self-time (parsed from XPlane proto) ==="
${PY} - <<'PY'
import gzip
import json
import os
from pathlib import Path
TRACE_DIR = Path("results/jax_profile")
# JAX writes a gzipped trace.json.gz under plugins/profile/<host>/.
trace_files = list(TRACE_DIR.rglob("*.trace.json.gz"))
if not trace_files:
    print("no trace.json.gz found — trace may be in a different format")
    raise SystemExit(0)
print(f"found {len(trace_files)} trace file(s); summarising the largest one")
trace = max(trace_files, key=lambda p: p.stat().st_size)
print(f"  trace = {trace}  ({trace.stat().st_size} bytes)")
with gzip.open(trace, "rt") as f:
    data = json.load(f)
# Trace events: list of dicts with name / dur / cat.
events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X" and e.get("dur")]
print(f"  {len(events)} duration events")
# Aggregate self-time by event name (HLO op).
totals = {}
for e in events:
    name = e.get("name", "<anon>")
    totals[name] = totals.get(name, 0) + e["dur"]
top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:15]
print()
print("rank  self-time (ms)  op")
print("----  --------------  --------------------------------------------")
for i, (name, us) in enumerate(top, 1):
    print(f"{i:>4}  {us/1000.0:>14.2f}  {name[:60]}")
# Save to a CSV for the report.
with open("results/jax_profile_top_ops.csv", "w") as f:
    f.write("rank,self_time_ms,op\n")
    for i, (name, us) in enumerate(top, 1):
        clean = name.replace(",", " ").replace("\n", " ")
        f.write(f"{i},{us/1000.0:.3f},{clean}\n")
print()
print("wrote results/jax_profile_top_ops.csv")
PY

echo
echo "=== peak HBM (extrapolated from M=16384 measurement) ==="
${PY} - <<'PY'
# Each per-step noise sample is 6 doubles = 48 B.  At UPDATE_PERIOD=10 there
# are ceil(N/10) ~= 2870 update events, so noise stream per realization is
# 2870 * 48 = 137 KB.  Total noise tensor = M * 137 KB.
N_imu = 28712  # from data/v101.bin header
n_updates = (N_imu + 9) // 10
bytes_per_real = n_updates * 6 * 8   # 6 floats per update, 8 bytes each
for M in (16384, 32768, 65536):
    gb = bytes_per_real * M / 2**30
    print(f"  M={M:>6}  noise tensor = {gb:7.3f} GB ({bytes_per_real} B/real)")
print("  L4 HBM total = 24 GB; OOM expected somewhere in the M=32k -- M=64k range.")
PY

echo
echo "=== done ==="
ls -lh results/ | grep -E "jax_profile|^total"
