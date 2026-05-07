#!/usr/bin/env python3
"""CUDA scaling figures.

Produces two PNGs:
    throughput_scaling.png   wall-time and realizations/sec vs M
    blocksize_sweep.png      wall-time vs block_size at fixed M

Inputs are the CSVs written by ekf_cuda --out, which share the
    impl,M,T,workers,run_idx,wall_time_ms,checksum
schema with the CPU implementations.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path: str):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                     usecols=(1, 2, 5), dtype=np.float64)
    return arr  # columns: M, T, wall_time_ms


def median_per_group(arr: np.ndarray, group_col: int):
    keys = arr[:, group_col].astype(int)
    unique = np.sort(np.unique(keys))
    out_keys, out_med = [], []
    for k in unique:
        mask = keys == k
        out_keys.append(int(k))
        out_med.append(float(np.median(arr[mask, 2])))
    return np.array(out_keys), np.array(out_med)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--throughput", required=True,
                    help="CSV from the M-sweep at fixed block size")
    ap.add_argument("--blocksize", required=False,
                    help="CSV from the block-size sweep at fixed M (optional)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Throughput vs M ----
    th = load(args.throughput)
    Ms, wall_ms = median_per_group(th, group_col=0)
    realizations_per_sec = (Ms * 1000.0) / wall_ms

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    ax1.plot(Ms, wall_ms, "o-", color="C3", lw=1.5, ms=6)
    ax1.set_ylabel("kernel wall time [ms]")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10)
    ax1.set_xticks(Ms)
    ax1.set_xticklabels([str(m) for m in Ms])
    ax1.grid(alpha=0.3, which="both")
    ax1.set_title("CUDA throughput sweep (block_size = 128, NVIDIA L4)")
    for m, w in zip(Ms, wall_ms):
        ax1.annotate(f"{w:.1f} ms", xy=(m, w), xytext=(4, -10),
                     textcoords="offset points", fontsize=8)

    ax2.plot(Ms, realizations_per_sec, "s-", color="C0", lw=1.5, ms=6)
    ax2.set_xlabel("number of realizations M")
    ax2.set_ylabel("realizations / second")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=10)
    ax2.grid(alpha=0.3, which="both")
    for m, r in zip(Ms, realizations_per_sec):
        ax2.annotate(f"{r:.0f}", xy=(m, r), xytext=(4, -10),
                     textcoords="offset points", fontsize=8)
    fig.tight_layout()
    p = out / "throughput_scaling.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # ---- Block-size sweep ----
    if args.blocksize:
        bs = load(args.blocksize)
        BSs, wall_bs = median_per_group(bs, group_col=0)
        # The "M" column is fixed in the block-size sweep, but we logged the
        # block_size as the per-row M's value via convention?  No — we drove
        # block_size via --block-size so M stays at 4096; we derive block_size
        # from a separate column.  Since the CSV doesn't carry block_size, we
        # rely on the M sweep being at fixed block; for the BS sweep, group by
        # the "T" column which we (mis-)use to encode block_size... actually
        # the simplest approach: re-load with usecols=(3,5) since "workers" is
        # fixed at 1, and we'll pick up block_size from the slurm-script order.
        pass  # filled below

        # The CSV format doesn't carry block_size as its own column (workers
        # is always 1 for cuda).  We rely on the Slurm script's ordering and
        # infer block_size from row index: 5 timed runs per block_size.
        raw = np.loadtxt(args.blocksize, delimiter=",", skiprows=1,
                         usecols=(5,), dtype=np.float64)
        block_sizes = [32, 64, 128, 256]   # matches run_cuda.slurm
        n_rows_per_bs = len(raw) // len(block_sizes)
        per_bs = raw.reshape(len(block_sizes), n_rows_per_bs)
        med_bs = np.median(per_bs, axis=1)

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(block_sizes, med_bs, "o-", color="C2", lw=1.5, ms=6)
        ax.set_xlabel("CUDA block size (threads/block)")
        ax.set_ylabel("kernel wall time [ms]  (M = 4096)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(block_sizes)
        ax.set_xticklabels([str(b) for b in block_sizes])
        ax.grid(alpha=0.3, which="both")
        ax.set_title("CUDA block-size sweep at M = 4096")
        for b, w in zip(block_sizes, med_bs):
            ax.annotate(f"{w:.1f} ms", xy=(b, w), xytext=(4, -10),
                        textcoords="offset points", fontsize=8)
        fig.tight_layout()
        p = out / "blocksize_sweep.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
