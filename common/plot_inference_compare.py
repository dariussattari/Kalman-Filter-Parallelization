#!/usr/bin/env python3
"""Cross-implementation inference comparison.

Aggregates the timed-run CSVs from all five implementations (serial, openmp,
mpi, cuda, jax), computes median wall time per (impl, M) combo, and produces
two complementary figures:

  inference_throughput.png   realizations / second  vs  M, log-log
                              shows where each impl saturates in batch mode

  inference_latency.png      ms / realization        vs  M, log-log
                              the "single-filter latency" view; lower is better
                              for on-board deployment

Each implementation is plotted at *its best parallel config* — i.e. the
result we'd actually ship:
    serial   1 thread / rank  (only choice)
    openmp   highest thread count present in the strong-scaling CSV
    mpi      highest rank count present in the strong-scaling CSV
    cuda     a single L4 GPU, block_size = 128
    jax      a single L4 GPU, with JIT compile cost amortised over 5 runs

Each --*-weak option (optional) lets us extend the OpenMP/MPI curves to small
M, since the strong-scaling CSV has fixed M and the weak-scaling CSV varies M.

Usage:
    python3 plot_inference_compare.py \\
        --serial path/to/serial/results.csv \\
        --openmp path/to/openmp/strong.csv \\
        [--openmp-weak path/to/openmp/weak.csv] \\
        --mpi    path/to/mpi/strong.csv \\
        [--mpi-weak path/to/mpi/weak.csv] \\
        --cuda   path/to/cuda/throughput.csv \\
        --jax    path/to/additional/throughput.csv \\
        --out-dir path/to/figures/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "serial": "#1f77b4",   # blue
    "openmp": "#2ca02c",   # green
    "mpi":    "#d62728",   # red
    "cuda":   "#9467bd",   # purple
    "jax":    "#ff7f0e",   # orange
}
MARKERS = {
    "serial": "o", "openmp": "s", "mpi": "^", "cuda": "D", "jax": "v",
}


def load_csv(path: str):
    """Returns (M[N], workers[N], wall_ms[N]) as int / int / float arrays."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1,
                     usecols=(1, 3, 5), dtype=np.float64)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]


def median_per_M_best_workers(M_arr, workers_arr, wall_arr):
    """For each M, take the row with the highest workers, then median over runs.

    Returns sorted (M[K], median_ms[K])."""
    out = {}
    for M in np.unique(M_arr):
        mask_m = M_arr == M
        ws = workers_arr[mask_m]
        if len(ws) == 0:
            continue
        best_w = ws.max()
        mask_w = mask_m & (workers_arr == best_w)
        out[int(M)] = float(np.median(wall_arr[mask_w]))
    Ms = np.array(sorted(out.keys()))
    ms = np.array([out[m] for m in Ms])
    return Ms, ms


def merge_strong_and_weak(strong_csv, weak_csv):
    """Combine two CSVs (e.g. openmp strong+weak) into one (Ms, ms) curve.

    Strong CSV usually contributes one M (the largest), weak CSV contributes
    several smaller Ms.  When both contain a given M, prefer strong (it ran
    at higher worker count).
    """
    strong_M, strong_w, strong_ms = load_csv(strong_csv)
    Ms_s, ms_s = median_per_M_best_workers(strong_M, strong_w, strong_ms)
    if weak_csv:
        weak_M, weak_w, weak_ms = load_csv(weak_csv)
        Ms_w, ms_w = median_per_M_best_workers(weak_M, weak_w, weak_ms)
        # Combine, prefer strong for duplicate M
        combined = {int(m): float(t) for m, t in zip(Ms_w, ms_w)}
        for m, t in zip(Ms_s, ms_s):
            combined[int(m)] = float(t)
        Ms = np.array(sorted(combined.keys()))
        ms = np.array([combined[m] for m in Ms])
        return Ms, ms
    return Ms_s, ms_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--openmp", required=True, help="strong-scaling CSV")
    ap.add_argument("--openmp-weak")
    ap.add_argument("--mpi",    required=True, help="strong-scaling CSV")
    ap.add_argument("--mpi-weak")
    ap.add_argument("--cuda",   required=True, help="throughput CSV")
    ap.add_argument("--jax",    required=True, help="throughput CSV")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Per-implementation (M, median wall ms) curves.
    curves = {}

    # Serial — only one config.
    sM, sw, sms = load_csv(args.serial)
    Ms, ms = median_per_M_best_workers(sM, sw, sms)
    curves["serial"] = (Ms, ms)

    # OpenMP — combine strong + weak for full M range.
    Ms, ms = merge_strong_and_weak(args.openmp, args.openmp_weak)
    curves["openmp"] = (Ms, ms)

    # MPI — same.
    Ms, ms = merge_strong_and_weak(args.mpi, args.mpi_weak)
    curves["mpi"] = (Ms, ms)

    # CUDA — only one config (block=128) in throughput.csv.
    cM, cw, cms = load_csv(args.cuda)
    Ms, ms = median_per_M_best_workers(cM, cw, cms)
    curves["cuda"] = (Ms, ms)

    # JAX — same.
    jM, jw, jms = load_csv(args.jax)
    Ms, ms = median_per_M_best_workers(jM, jw, jms)
    curves["jax"] = (Ms, ms)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: throughput (realizations/sec) vs M ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for impl, (Ms, ms) in curves.items():
        if len(Ms) == 0:
            continue
        rps = (Ms * 1000.0) / ms
        ax.plot(Ms, rps, marker=MARKERS[impl], color=COLORS[impl],
                lw=1.6, ms=7, label=impl)
        # Annotate peak throughput
        peak_i = int(np.argmax(rps))
        ax.annotate(f"{rps[peak_i]:.0f} r/s",
                    xy=(Ms[peak_i], rps[peak_i]),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color=COLORS[impl])
    ax.set_xlabel("number of concurrent realizations  M")
    ax.set_ylabel("throughput (realizations / second)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Cross-implementation throughput  (best config per impl)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    p = out / "inference_throughput.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # ---- Figure 2: ms per realization vs M (latency view) ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for impl, (Ms, ms) in curves.items():
        if len(Ms) == 0:
            continue
        per_m = ms / Ms
        ax.plot(Ms, per_m, marker=MARKERS[impl], color=COLORS[impl],
                lw=1.6, ms=7, label=impl)
        # Annotate the lowest per-realization cost
        best_i = int(np.argmin(per_m))
        ax.annotate(f"{per_m[best_i]:.3f} ms/r",
                    xy=(Ms[best_i], per_m[best_i]),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color=COLORS[impl])
    ax.set_xlabel("number of concurrent realizations  M")
    ax.set_ylabel("wall time per realization (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Per-realization latency  (lower is better; on-board metric)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    p = out / "inference_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # ---- Print a small summary table for the report ----
    print()
    print("Best performance per implementation:")
    print(f"  {'impl':<8} {'best M':>8} {'wall(ms)':>10} {'r/s':>10} {'ms/r':>10}")
    for impl, (Ms, ms) in curves.items():
        if len(Ms) == 0:
            continue
        rps = (Ms * 1000.0) / ms
        peak_i = int(np.argmax(rps))
        per_m = ms[peak_i] / Ms[peak_i]
        print(f"  {impl:<8} {int(Ms[peak_i]):>8} {ms[peak_i]:>10.1f} "
              f"{rps[peak_i]:>10.1f} {per_m:>10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
