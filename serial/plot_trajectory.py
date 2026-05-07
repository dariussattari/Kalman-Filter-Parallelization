#!/usr/bin/env python3
"""Overlay the serial EKF estimate on top of the EuRoC ground truth.

Reads the trajectory CSV produced by ekf_serial --trajectory and writes a
3-panel position-vs-time PNG (and, for free, a 3-panel attitude figure) so
we can eyeball that the filter actually tracks the V1_01_easy flight.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="trajectory CSV from ekf_serial")
    ap.add_argument("--out", required=True, help="output PNG path (position figure)")
    ap.add_argument("--out-attitude", default=None,
                    help="optional output PNG path for attitude (default: derived from --out)")
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    t       = data[:, 0]
    gt_p    = data[:, 1:4]
    est_p   = data[:, 4:7]
    gt_th   = data[:, 7:10]
    est_th  = data[:, 10:13]

    rmse_p = np.sqrt(np.mean((est_p - gt_p) ** 2, axis=0))
    # Wrap angular residuals to [-pi, pi] before squaring, so RMSE doesn't get
    # blown up by the EuRoC roll signal naturally cycling through +/- pi.
    angle_diff = np.mod(est_th - gt_th + np.pi, 2 * np.pi) - np.pi
    rmse_t = np.sqrt(np.mean(angle_diff ** 2, axis=0))
    print(f"position RMSE per axis [m]   : {rmse_p}")
    print(f"attitude RMSE per axis [rad] : {rmse_t}")

    # ---- position figure ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, lbl, c in zip(axes, ("x [m]", "y [m]", "z [m]"), range(3)):
        ax.plot(t, gt_p[:, c], lw=1.0, label="ground truth")
        ax.plot(t, est_p[:, c], lw=0.8, label="EKF estimate", alpha=0.85)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Serial EKF vs ground truth (V1_01_easy) | RMSE_pos = "
                 f"{rmse_p[0]:.3f}, {rmse_p[1]:.3f}, {rmse_p[2]:.3f} m")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- attitude figure ----
    out_att = args.out_attitude
    if out_att is None:
        out_att = str(out).replace(".png", "_attitude.png")
        if out_att == str(out):
            out_att = str(out) + "_attitude.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, lbl, c in zip(axes, ("roll [rad]", "pitch [rad]", "yaw [rad]"), range(3)):
        ax.plot(t, gt_th[:, c], lw=1.0, label="ground truth")
        ax.plot(t, est_th[:, c], lw=0.8, label="EKF estimate", alpha=0.85)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Serial EKF vs ground truth (attitude) | RMSE_att = "
                 f"{rmse_t[0]:.3f}, {rmse_t[1]:.3f}, {rmse_t[2]:.3f} rad")
    fig.tight_layout()
    fig.savefig(out_att, dpi=150)
    plt.close(fig)
    print(f"wrote {out_att}")


if __name__ == "__main__":
    main()
