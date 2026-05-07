# Parallelizing the Extended Kalman Filter

**Author:** Darius B. Sattari · 7 May 2026

This repository implements a 9-state Extended Kalman Filter for micro-aerial-vehicle navigation on the EuRoC `V1_01_easy` flight, parallelised five different ways for a Monte-Carlo ensemble of `M` independent realizations:

| Part | Directory | Backend |
|---|---|---|
| 1 — Serial | [`serial/`](serial/) | C++17, single core |
| 2 — OpenMP | [`openmp/`](openmp/) | C++17 + `#pragma omp parallel for` |
| 3 — MPI | [`mpi/`](mpi/) | C++17 + Open MPI block-decomposition |
| 4 — CUDA | [`cuda/`](cuda/) | C++17 / `nvcc`, one realization per thread |
| 5 — Additional (JAX) | [`additional/`](additional/) | Python + `jax.jit` / `vmap` / `lax.scan` |
| 6 — Report | [`report/report.md`](report/report.md) | blog-style writeup |

All five drivers share a single header-only EKF math kernel ([`common/ekf_math.hpp`](common/ekf_math.hpp)) tagged `__host__ __device__`, so the CPU and GPU paths run byte-identical floating-point code. Per-realization checksums match across implementations to within `1e-9` (verified by [`common/check_golden.py`](common/check_golden.py)).

## How to reproduce

The expected reproduction flow on the cluster is:

```bash
# 1. Fetch + preprocess the EuRoC V1_01_easy dataset (one-time, ~10 min).
bash data/fetch_euroc.sh
python3 data/preprocess.py        # writes data/v101.bin

# 2. Run all five implementations end-to-end (each Slurm job ~5–30 min).
sbatch serial/run_serial.slurm
sbatch openmp/run_omp.slurm
sbatch mpi/run_mpi.slurm
sbatch cuda/run_cuda.slurm
sbatch additional/run_jax.slurm

# 3. Run the profilers (Part 6 — VTune for CPU, ptxas + jax.profiler for GPU).
sbatch openmp/run_vtune.slurm           # Intel VTune hotspots T=16 vs T=32
sbatch mpi/run_vtune_hpc.slurm          # VTune hpc-performance, 4-rank baseline
sbatch mpi/run_vtune_sw.slurm           # VTune hotspots, 4-rank vs 8-rank
sbatch cuda/run_profile.slurm           # nvcc -Xptxas=-v + occupancy table
sbatch additional/run_jax_profile.slurm # jax.profiler.trace
```

Each Slurm script writes its `results/` directory next to its sources and rebuilds the binary first (`make clean && make`). Plots regenerate as a final step in each script.

The full bottleneck analysis with profiler-derived numbers lives in [`report/report.md` § Profiling and bottleneck analysis](report/report.md).
