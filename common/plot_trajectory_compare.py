#!/usr/bin/env python3
"""Overlay multiple EKF trajectories against the EuRoC ground truth.

For each implementation (serial, openmp, mpi, cuda, ...) we plot:
    - position vs time on a 3-panel figure
    - position residual |est - serial_est| on a second figure
The expectation is that *every* implementation produces an estimate that
matches serial to within one ULP (~1e-15) — the residual figure proves that
parallelization does not degrade the filter.

Usage:
    python3 plot_trajectory_compare.py \\
        --traj serial=src/serial/results/trajectory.csv \\
        --traj openmp=src/openmp/results/trajectory.csv \\
        --reference serial \\
        --out-dir src/openmp/results/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_traj(path: str):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return {
        "t":   arr[:, 0],
        "gt_p":  arr[:, 1:4],   "est_p":  arr[:, 4:7],
        "gt_th": arr[:, 7:10],  "est_th": arr[:, 10:13],
    }


def parse_traj_arg(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--traj wants label=path, got {s!r}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", action="append", required=True, type=parse_traj_arg,
                    metavar="LABEL=PATH",
                    help="Repeat for each impl, e.g. --traj serial=path/to/serial.csv")
    ap.add_argument("--reference", default=None,
                    help="Label to use as the reference for residuals "
                         "(default: first --traj). Residuals are |other_est - ref_est|.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if len(args.traj) < 1:
        raise SystemExit("need at least one --traj")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trajs = {label: load_traj(path) for label, path in args.traj}
    labels = [label for label, _ in args.traj]
    ref_label = args.reference or labels[0]
    if ref_label not in trajs:
        raise SystemExit(f"reference label {ref_label!r} not among {labels}")

    # Sanity: every trajectory must share timestamps and ground truth.
    t = trajs[ref_label]["t"]
    for lbl in labels:
        if not np.array_equal(trajs[lbl]["t"], t):
            print(f"warning: {lbl} has different timestamps from {ref_label}; "
                  "skipping residual cross-check")
            return 1

    # ----- Figure 1: position-vs-time overlay (GT + every estimate) -----
    # When implementations overlap to numerical precision, a solid coloured
    # line on top hides every layer underneath.  We use markers (not dashes)
    # at staggered offsets so each implementation appears as a distinct
    # coloured dotted track that visually interlaces with the others.  GT is
    # drawn as a thick solid line behind everything.
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
    axis_labels = ("x [m]", "y [m]", "z [m]")
    colors = {"gt": "#222222"}
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(labels):
        colors[lbl] = cmap(i + 1)  # skip index 0 to keep impl colours distinct

    GT_LW       = 3.5
    MARKER_SIZE = 5.0
    MARKERS     = ["o", "s", "^", "D", "v"]
    n_impl      = len(labels)
    n_pts       = len(t)
    # Show ~80 markers across the time axis, offset between impls so they
    # form an interlaced rainbow rather than stacking on identical timestamps.
    every       = max(50, n_pts // 80)
    offsets     = [int(i * every / max(n_impl, 1)) for i in range(n_impl)]

    for ax, alabel, c in zip(axes, axis_labels, range(3)):
        ax.plot(t, trajs[ref_label]["gt_p"][:, c], color=colors["gt"],
                lw=GT_LW, alpha=1.0, zorder=1, label="ground truth")
        for i, lbl in enumerate(labels):
            ax.plot(t, trajs[lbl]["est_p"][:, c],
                    color=colors[lbl],
                    lw=0.0,                       # no connecting line
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=MARKER_SIZE,
                    markevery=(offsets[i], every),
                    markeredgecolor=colors[lbl],
                    markerfacecolor=colors[lbl],
                    alpha=0.9,
                    zorder=3 + i,
                    label=f"{lbl} estimate")
        ax.set_ylabel(alabel)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=len(labels) + 1)
    axes[-1].set_xlabel("time [s]")

    # Per-axis position RMSE for each impl, for the title.
    rmse_lines = []
    for lbl in labels:
        d = trajs[lbl]
        rmse = np.sqrt(np.mean((d["est_p"] - d["gt_p"]) ** 2, axis=0))
        rmse_lines.append(f"{lbl} RMSE = "
                          f"{rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f} m")
    fig.suptitle("EKF position vs ground truth — V1_01_easy\n" + "  |  ".join(rmse_lines),
                 fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p_out = out / "trajectory_compare.png"
    fig.savefig(p_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_out}")

    # ----- Figure 2: residuals between each impl and the reference -----
    others = [l for l in labels if l != ref_label]
    if not others:
        print("only one trajectory provided; skipping residual figure")
        return 0

    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    ref_est = trajs[ref_label]["est_p"]
    # Plain log y, fixed range.  Clip exact zeros to the floor so they show
    # up as a flat line at the bottom of the panel rather than vanishing.
    ULP   = 2.220446049250313e-16
    FLOOR = 1.0e-18
    YMAX  = 1.0e-8
    for ax, alabel, c in zip(axes, ("|x| [m]", "|y| [m]", "|z| [m]"), range(3)):
        for lbl in others:
            r = np.abs(trajs[lbl]["est_p"][:, c] - ref_est[:, c])
            r = np.where(r == 0.0, FLOOR, r)
            ax.plot(t, r, lw=0.9, alpha=0.9,
                    color=colors[lbl], label=f"|{lbl} − {ref_label}|")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.5, YMAX)
        ax.set_ylabel(alabel)
        ax.grid(alpha=0.3, which="both")
        ax.axhline(ULP,   color="grey", ls=":",  lw=0.8,
                   label="1 ULP (≈2.2×10⁻¹⁶)" if c == 0 else None)
        ax.axhline(1e-9,  color="red",  ls="--", lw=0.8, alpha=0.6,
                   label="project tolerance (1×10⁻⁹)" if c == 0 else None)
        ax.axhline(FLOOR, color="black", ls="-", lw=0.4, alpha=0.3,
                   label=f"floor (zeros clipped to {FLOOR:.0e})" if c == 0 else None)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    # Worst-case residual per impl across all axes/timesteps for the title.
    worst_lines = []
    for lbl in others:
        worst = float(np.max(np.abs(trajs[lbl]["est_p"] - ref_est)))
        worst_lines.append(f"max|{lbl} − {ref_label}| = {worst:.2e} m")
    fig.suptitle("Implementation-vs-reference residuals (positions)\n" +
                 "  |  ".join(worst_lines), fontsize=11)
    fig.tight_layout()
    p_out = out / "trajectory_residual.png"
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"wrote {p_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
