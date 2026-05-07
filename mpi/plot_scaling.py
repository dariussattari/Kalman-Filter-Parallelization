#!/usr/bin/env python3
"""Strong + weak scaling figures from the MPI results CSVs.

Reads results/strong.csv (fixed M, varying ranks) and results/weak.csv
(M scaled with ranks), each with header
    impl,M,T,workers,run_idx,wall_time_ms,checksum
takes the median wall_time_ms across runs per (workers, M) combo, and writes:
    <out>/strong_scaling.png   speedup vs ranks, log-log, with ideal line
    <out>/weak_scaling.png     parallel efficiency vs ranks

Pure-numpy implementation — no pandas required.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def median_per_workers(csv_path: str):
    """Return three parallel arrays (workers, M, median_ms) sorted by workers."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                     usecols=(1, 3, 5), dtype=np.float64)
    workers = arr[:, 1].astype(int)
    M_col   = arr[:, 0].astype(int)
    wall    = arr[:, 2]

    unique_w = np.sort(np.unique(workers))
    out_w, out_M, out_med = [], [], []
    for w in unique_w:
        mask = workers == w
        out_w.append(int(w))
        out_M.append(int(np.median(M_col[mask])))
        out_med.append(float(np.median(wall[mask])))
    return np.array(out_w), np.array(out_M), np.array(out_med)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strong", required=True, help="strong-scaling CSV")
    ap.add_argument("--weak",   required=True, help="weak-scaling CSV")
    ap.add_argument("--out-dir", required=True, help="directory for PNG output")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Strong scaling ----
    workers_s, M_s, med_s = median_per_workers(args.strong)
    T1 = med_s[0]
    speedup = T1 / med_s
    M_strong = int(M_s[0])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(workers_s, speedup, "o-", label="measured", lw=1.5, ms=6, color="C2")
    ax.plot(workers_s, workers_s, "k--", alpha=0.4, label="ideal (linear)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(f"speedup vs 1 rank  ($T_1$ = {T1:.0f} ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(workers_s)
    ax.set_xticklabels([str(w) for w in workers_s])
    ax.set_title(f"MPI strong scaling  (M = {M_strong})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    for w, s in zip(workers_s, speedup):
        ax.annotate(f"{s:.1f}×", xy=(w, s), xytext=(4, -10),
                    textcoords="offset points", fontsize=8)
    fig.tight_layout()
    p = out / "strong_scaling.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # ---- Weak scaling ----
    workers_w, M_w, med_w = median_per_workers(args.weak)
    T1w = med_w[0]
    efficiency = T1w / med_w
    M_per_rank = M_w[0] // workers_w[0]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(workers_w, efficiency, "o-", label="measured", lw=1.5, ms=6, color="C2")
    ax.axhline(1.0, color="k", ls="--", alpha=0.4, label="ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("parallel efficiency  ($T_1 / T_N$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers_w)
    ax.set_xticklabels([str(w) for w in workers_w])
    ax.set_ylim(0, 1.2)
    ax.set_title(f"MPI weak scaling  (M per rank = {M_per_rank})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    for w, e in zip(workers_w, efficiency):
        ax.annotate(f"{e:.2f}", xy=(w, e), xytext=(4, -10),
                    textcoords="offset points", fontsize=8)
    fig.tight_layout()
    p = out / "weak_scaling.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
