#!/usr/bin/env python3
"""Pack EuRoC V1_01_easy into a single float64 binary for the EKF kernel.

Output: data/v101.bin
    Line 1 (ASCII, ends with \\n):
        N_imu=<N> T_seconds=<T> dt_imu=<dt>
    Then three float64 arrays back-to-back, little-endian:
        imu_t  shape (N,)      seconds, t[0] = 0
        imu_uw shape (N, 6)    [omega_xyz, accel_xyz] in body frame, ASL ordering
        gt     shape (N, 6)    [p_xyz_world, theta_xyz_world] (small-angle Euler XYZ)

Also writes data/v101_trajectory.png as a sanity-check plot of GT position vs time.

The C++ loader reads the header line with std::getline, parses N, then reads
N + 6N + 6N = 13*N float64s in that order.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation, Slerp
except ImportError:
    sys.exit("scipy is required (pip install --user scipy)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_imu(csv_path: Path):
    # ASL columns: timestamp[ns], w_x, w_y, w_z, a_x, a_y, a_z
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    t_ns = arr[:, 0].astype(np.int64)
    uw = arr[:, 1:7].astype(np.float64)
    return t_ns, uw


def load_gt(csv_path: Path):
    # ASL columns: timestamp[ns], p_xyz, q_wxyz, v_xyz, b_w_xyz, b_a_xyz  (17 cols)
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    t_ns = arr[:, 0].astype(np.int64)
    p = arr[:, 1:4].astype(np.float64)
    q_wxyz = arr[:, 4:8].astype(np.float64)
    return t_ns, p, q_wxyz


def interp_position(t_query_ns, t_src_ns, p_src):
    out = np.empty((t_query_ns.size, 3), dtype=np.float64)
    tq = t_query_ns.astype(np.float64)
    ts = t_src_ns.astype(np.float64)
    for d in range(3):
        out[:, d] = np.interp(tq, ts, p_src[:, d])
    return out


def interp_attitude(t_query_ns, t_src_ns, q_wxyz_src):
    # scipy Rotation expects [x, y, z, w]
    q_xyzw = q_wxyz_src[:, [1, 2, 3, 0]]
    rot_src = Rotation.from_quat(q_xyzw)
    slerp = Slerp(t_src_ns.astype(np.float64), rot_src)
    rot_q = slerp(t_query_ns.astype(np.float64))
    # Uppercase "XYZ" = intrinsic Tait-Bryan convention.  Matches the
    # body_to_world(theta) = Rx(tx) * Ry(ty) * Rz(tz) formulation in
    # src/common/ekf_math.hpp.  Lowercase "xyz" would be extrinsic and
    # decompose as Rz*Ry*Rx, which would mismatch the C++ side.
    return rot_q.as_euler("XYZ", degrees=False)


def find_sequence_dir(start: Path) -> Path:
    """Locate the directory whose direct child is mav0/.

    Accepts any of these layouts (in priority order):
        <start>/mav0/...                    (mav0 dropped straight into raw/)
        <start>/V1_01_easy/mav0/...         (per-sequence zip extracted)
        <start>/<anything>/mav0/...         (some other naming)
    """
    if (start / "mav0" / "imu0" / "data.csv").is_file():
        return start
    for child in sorted(start.iterdir()):
        if child.is_dir() and (child / "mav0" / "imu0" / "data.csv").is_file():
            return child
    raise FileNotFoundError(
        f"could not find mav0/imu0/data.csv under {start} or any of its subdirectories.\n"
        f"hint: run `bash data/fetch_euroc.sh`, or upload the EuRoC mav0/ tree into {start}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sequence-dir",
                    default=str(Path(__file__).parent / "raw"),
                    help="Path under which mav0/ lives (auto-detected one level deep)")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "v101.bin"),
                    help="Output packed binary path")
    ap.add_argument("--plot",
                    default=str(Path(__file__).parent / "v101_trajectory.png"),
                    help="Output trajectory sanity plot path")
    args = ap.parse_args()

    try:
        seq = find_sequence_dir(Path(args.sequence_dir).resolve())
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"[preprocess] using sequence root: {seq}")
    imu_csv = seq / "mav0" / "imu0" / "data.csv"
    gt_csv = seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    if not gt_csv.is_file():
        sys.exit(
            f"missing GT file: {gt_csv}\n"
            f"the sequence at {seq} appears not to include state_groundtruth_estimate0/.\n"
            f"V1_01_easy from the ASL/Research-Collection sources should include it."
        )

    print(f"[preprocess] loading IMU : {imu_csv}")
    t_imu_ns, uw = load_imu(imu_csv)
    print(f"[preprocess] loading GT  : {gt_csv}")
    t_gt_ns, p_gt, q_gt_wxyz = load_gt(gt_csv)

    # Crop IMU to the GT-covered range so interpolation never extrapolates.
    lo, hi = t_gt_ns[0], t_gt_ns[-1]
    keep = (t_imu_ns >= lo) & (t_imu_ns <= hi)
    if not keep.all():
        print(f"[preprocess] cropping {(~keep).sum()} IMU samples outside GT range")
    t_imu_ns = t_imu_ns[keep]
    uw = uw[keep]

    print(f"[preprocess] interpolating GT onto {t_imu_ns.size} IMU timestamps")
    p_at_imu = interp_position(t_imu_ns, t_gt_ns, p_gt)
    theta_at_imu = interp_attitude(t_imu_ns, t_gt_ns, q_gt_wxyz)
    gt = np.concatenate([p_at_imu, theta_at_imu], axis=1)

    t_sec = (t_imu_ns - t_imu_ns[0]).astype(np.float64) * 1e-9
    N = int(t_sec.size)
    T = float(t_sec[-1])
    dt = float(np.median(np.diff(t_sec)))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] writing {out_path}")
    print(f"             N={N}, T={T:.3f}s, dt={dt*1e3:.3f}ms (~{1.0/dt:.1f} Hz)")

    header = f"N_imu={N} T_seconds={T:.6f} dt_imu={dt:.9f}\n".encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header)
        np.ascontiguousarray(t_sec, dtype=np.float64).tofile(f)
        np.ascontiguousarray(uw,    dtype=np.float64).tofile(f)
        np.ascontiguousarray(gt,    dtype=np.float64).tofile(f)

    expected_bytes = len(header) + 8 * N * (1 + 6 + 6)
    actual_bytes = out_path.stat().st_size
    if actual_bytes != expected_bytes:
        sys.exit(f"size mismatch: wrote {actual_bytes}, expected {expected_bytes}")

    plot_path = Path(args.plot).resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, lbl, col in zip(axes, ("x [m]", "y [m]", "z [m]"), range(3)):
        ax.plot(t_sec, p_at_imu[:, col], lw=0.8)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("EuRoC V1_01_easy ground-truth position (interpolated to IMU rate)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[preprocess] sanity plot -> {plot_path}")


if __name__ == "__main__":
    main()
