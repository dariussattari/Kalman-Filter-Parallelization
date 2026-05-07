#!/usr/bin/env python3
"""additional/ekf_jax.py — Part 5 deliverable: JAX implementation.

The same Monte-Carlo Extended Kalman Filter as serial/openmp/mpi/cuda, expressed
in pure functional JAX.  Strategy:
    - jax.lax.scan      iterates the time dimension (28k IMU steps / realization)
    - jax.vmap          parallelises across the M realizations
    - jax.jit           compiles the whole forward pass into one XLA fusion
                        which then runs on whatever backend JAX picks (GPU
                        when --gres=gpu:*, CPU otherwise).

Compared to ekf_cuda.cu, this is roughly 5x shorter line-for-line: no manual
matmul helpers, no manual 6x6 inverse, no host/device memory plumbing, no
launch-config tuning.  The trade-off is opaqueness: a ~3 s JIT compile cost on
the first call, and limited control over per-thread state placement.

Cross-implementation correctness: JAX's RNG is jax.random.PRNGKey, distinct
from std::mt19937_64 (used by serial/openmp/mpi) and cuRAND Philox (used by
CUDA), so checksum bit-identity is unattainable.  We verify correctness via
(1) within-JAX determinism (same key -> same output) and (2) trajectory RMSE
matching the CPU implementations within statistical bounds (~few %).

CLI mirrors the C++ drivers.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random

# Match the CPU implementations' float64.  XLA defaults to float32 otherwise.
jax.config.update("jax_enable_x64", True)


# =====================================================================
# Constants — kept identical to common/ekf_config.hpp
# =====================================================================
GRAVITY_Z = 9.80665
P0_POS = 1.0e-4
P0_VEL = 2.5e-1
P0_ATT = 1.0e-4
Q_POS  = 1.0e-8
Q_VEL  = 1.0e-4
Q_ATT  = 1.0e-7
SIGMA_MEAS_POS = 1.0e-2
SIGMA_MEAS_ATT = 1.0e-2
UPDATE_PERIOD  = 10


# =====================================================================
# EKF math — pure functions
# =====================================================================
def body_to_world(theta):
    """R = Rx(tx) * Ry(ty) * Rz(tz), 3x3."""
    sx, cx = jnp.sin(theta[0]), jnp.cos(theta[0])
    sy, cy = jnp.sin(theta[1]), jnp.cos(theta[1])
    sz, cz = jnp.sin(theta[2]), jnp.cos(theta[2])
    return jnp.array([
        [cy * cz,                  -cy * sz,                 sy],
        [cx * sz + sx * sy * cz,   cx * cz - sx * sy * sz,   -sx * cy],
        [sx * sz - cx * sy * cz,   sx * cz + cx * sy * sz,   cx * cy],
    ])


def dRa_dtheta(theta, a):
    """3x3 Jacobian of R(theta) @ a w.r.t. theta. Same formulas as ekf_math.hpp."""
    sx, cx = jnp.sin(theta[0]), jnp.cos(theta[0])
    sy, cy = jnp.sin(theta[1]), jnp.cos(theta[1])
    R = body_to_world(theta)
    v = R @ a
    col0 = jnp.array([0.0, -v[2], v[1]])
    col1 = jnp.array([cx * v[2] - sx * v[1], sx * v[0], -cx * v[0]])
    wx, wy, wz = sy, -sx * cy, cx * cy
    col2 = jnp.array([wy * v[2] - wz * v[1],
                      wz * v[0] - wx * v[2],
                      wx * v[1] - wy * v[0]])
    return jnp.stack([col0, col1, col2], axis=1)


def wrap_pi(a):
    return jnp.mod(a + jnp.pi, 2.0 * jnp.pi) - jnp.pi


_H = jnp.zeros((6, 9)).at[:3, :3].set(jnp.eye(3)).at[3:, 6:].set(jnp.eye(3))
_G = jnp.array([0.0, 0.0, GRAVITY_Z])


def predict(x, P, u, dt, Q):
    w       = u[:3]
    a_body  = u[3:]
    theta_p = x[6:9]
    R       = body_to_world(theta_p)
    a_world = R @ a_body - _G

    new_p     = x[:3] + x[3:6] * dt + 0.5 * a_world * dt * dt
    new_v     = x[3:6] + a_world * dt
    new_theta = wrap_pi(x[6:9] + w * dt)
    new_x     = jnp.concatenate([new_p, new_v, new_theta])

    dRa = dRa_dtheta(theta_p, a_body)
    F = jnp.eye(9)
    F = F.at[:3, 3:6].set(jnp.eye(3) * dt)
    F = F.at[:3, 6:9].set(0.5 * dt * dt * dRa)
    F = F.at[3:6, 6:9].set(dt * dRa)
    new_P = F @ P @ F.T + Q
    return new_x, new_P


def update(x, P, z, R_meas):
    z_pred = jnp.concatenate([x[:3], x[6:9]])
    y      = z - z_pred
    y      = jnp.concatenate([y[:3], wrap_pi(y[3:])])
    S      = _H @ P @ _H.T + R_meas
    # Solve S K^T = (P H^T)^T  ->  K = P H^T S^{-1}
    K_T    = jnp.linalg.solve(S, _H @ P)
    K      = K_T.T
    new_x  = x + K @ y
    new_x  = jnp.concatenate([new_x[:6], wrap_pi(new_x[6:9])])
    new_P  = (jnp.eye(9) - K @ _H) @ P
    return new_x, new_P


def initial_state(gt_row0):
    return jnp.array([gt_row0[0], gt_row0[1], gt_row0[2],
                      0.0,        0.0,        0.0,
                      gt_row0[3], gt_row0[4], gt_row0[5]])


def initial_P():
    return jnp.diag(jnp.array([P0_POS] * 3 + [P0_VEL] * 3 + [P0_ATT] * 3))


def constants_Q_R_sigma():
    Q       = jnp.diag(jnp.array([Q_POS] * 3 + [Q_VEL] * 3 + [Q_ATT] * 3))
    R_meas  = jnp.diag(jnp.array([SIGMA_MEAS_POS ** 2] * 3 + [SIGMA_MEAS_ATT ** 2] * 3))
    sigma   = jnp.array([SIGMA_MEAS_POS] * 3 + [SIGMA_MEAS_ATT] * 3)
    return Q, R_meas, sigma


# =====================================================================
# Per-realization forward pass — scanned over time
# =====================================================================
def _step(noise_stream, Q, R_meas, sigma_meas, carry, inputs):
    x, P, n_updates = carry
    u, gt_t, t, dt = inputs
    x_p, P_p = predict(x, P, u, dt, Q)
    do_update = jnp.equal(jnp.mod(t, UPDATE_PERIOD), 0)
    idx = jnp.minimum(n_updates, noise_stream.shape[0] - 1)
    z   = gt_t + sigma_meas * noise_stream[idx]
    x_u, P_u = update(x_p, P_p, z, R_meas)
    x_next = jnp.where(do_update, x_u, x_p)
    P_next = jnp.where(do_update, P_u, P_p)
    n_next = n_updates + jnp.where(do_update, 1, 0)
    return (x_next, P_next, n_next)


def run_one_final(imu_uw_seq, gt_seq, dt_seq, ts, noise_stream,
                  Q, R_meas, sigma_meas, x0, P0):
    """Returns ONLY the final state — no per-step trajectory accumulation.

    This is the memory-efficient path used by vmap over M realizations.
    """
    def step(carry, inputs):
        new_carry = _step(noise_stream, Q, R_meas, sigma_meas, carry, inputs)
        return new_carry, None    # `None` output → scan stores nothing per step

    init = (x0, P0, jnp.int32(0))
    (final_x, _, _), _ = lax.scan(step, init, (imu_uw_seq, gt_seq, ts, dt_seq))
    return final_x


def run_one_traj(imu_uw_seq, gt_seq, dt_seq, ts, noise_stream,
                 Q, R_meas, sigma_meas, x0, P0):
    """Returns final state + full per-step trajectory.  Use only at M=1."""
    def step(carry, inputs):
        new_carry = _step(noise_stream, Q, R_meas, sigma_meas, carry, inputs)
        return new_carry, new_carry[0]    # scan accumulates the state per step

    init = (x0, P0, jnp.int32(0))
    (final_x, _, _), traj = lax.scan(step, init, (imu_uw_seq, gt_seq, ts, dt_seq))
    traj = jnp.concatenate([x0[None, :], traj], axis=0)
    return final_x, traj


# Vectorize the *final-only* path across realizations — that's what the
# throughput sweep uses.  trajectory is computed separately at M=1 below.
_run_batch = jax.vmap(run_one_final,
    in_axes=(None, None, None, None, 0, None, None, None, None, None))
run_batch_jit = jax.jit(_run_batch)
run_traj_jit  = jax.jit(run_one_traj)


# =====================================================================
# CLI / driver
# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1,
                    help="number of warmup runs (the first one absorbs JIT compile)")
    ap.add_argument("--out")
    ap.add_argument("--golden")
    ap.add_argument("--trajectory")
    ap.add_argument("--profile-trace",
                    help="If set, wrap the timed runs in jax.profiler.trace(DIR) "
                         "to produce a Perfetto/TensorBoard trace tree under DIR.")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def load_binary(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii")
        # "N_imu=<N> T_seconds=<T> dt_imu=<dt>\n"
        parts = dict(p.split("=") for p in header.strip().split())
        N = int(parts["N_imu"])
        T_seconds = float(parts["T_seconds"])
        dt_imu    = float(parts["dt_imu"])
        imu_t   = np.frombuffer(f.read(N * 8), dtype=np.float64).copy()
        imu_uw  = np.frombuffer(f.read(N * 6 * 8), dtype=np.float64).reshape(N, 6).copy()
        gt      = np.frombuffer(f.read(N * 6 * 8), dtype=np.float64).reshape(N, 6).copy()
    return N, T_seconds, dt_imu, imu_t, imu_uw, gt


def main():
    a = parse_args()

    if not a.quiet:
        print(f"[ekf_jax] backend = {jax.default_backend()}  (devices: "
              f"{[d.platform for d in jax.devices()]})")

    # Load on host as numpy, then move to device once.
    N, T_sec, dt_imu, imu_t_np, imu_uw_np, gt_np = load_binary(a.data)
    if not a.quiet:
        print(f"[ekf_jax] loaded {a.data}: N={N}  T={T_sec:.3f}s  dt={dt_imu*1e3:.3f}ms")
        print(f"[ekf_jax] M={a.M}  seed={a.seed}  runs={a.runs} (warmup={a.warmup})  "
              f"update_period={UPDATE_PERIOD}")

    # Pre-compute step inputs (length N-1).
    imu_uw_seq = jnp.asarray(imu_uw_np[:-1])
    gt_seq     = jnp.asarray(gt_np[1:])
    dt_seq     = jnp.asarray(np.diff(imu_t_np))
    ts         = jnp.arange(1, N, dtype=jnp.int32)

    # Pre-generate noise per realization: shape (M, n_updates, 6).  Drawing on
    # the host avoids variability in scan-internal RNG state across vmap'd lanes.
    n_updates = (N + UPDATE_PERIOD - 1) // UPDATE_PERIOD
    master_key = random.PRNGKey(a.seed)
    keys = random.split(master_key, a.M)
    @jax.jit
    def gen_noise(keys):
        return jax.vmap(lambda k: random.normal(k, (n_updates, 6)))(keys)
    noise_stream = gen_noise(keys)

    Q, R_meas, sigma_meas = constants_Q_R_sigma()
    x0 = initial_state(gt_np[0])
    P0 = initial_P()

    # ---- Warmup runs (the first absorbs JIT compile) ----
    jit_compile_ms = None
    for w in range(a.warmup):
        t0 = time.perf_counter()
        final_x = run_batch_jit(
            imu_uw_seq, gt_seq, dt_seq, ts,
            noise_stream, Q, R_meas, sigma_meas, x0, P0)
        final_x.block_until_ready()
        elapsed = (time.perf_counter() - t0) * 1e3
        if w == 0:
            jit_compile_ms = elapsed
            if not a.quiet:
                print(f"[ekf_jax] JIT compile + first run = {elapsed:.1f} ms")
        else:
            if not a.quiet:
                print(f"[ekf_jax] warmup {w} = {elapsed:.1f} ms")

    # ---- Timed runs (final-state-only batch path; small memory footprint) ----
    wall_ms = []
    profile_ctx = (jax.profiler.trace(a.profile_trace, create_perfetto_link=False)
                   if a.profile_trace else None)
    if profile_ctx is not None:
        profile_ctx.__enter__()
        if not a.quiet:
            print(f"[ekf_jax] jax.profiler.trace -> {a.profile_trace}")
    try:
        for run in range(a.runs):
            t0 = time.perf_counter()
            final_x = run_batch_jit(
                imu_uw_seq, gt_seq, dt_seq, ts,
                noise_stream, Q, R_meas, sigma_meas, x0, P0)
            final_x.block_until_ready()
            ms = (time.perf_counter() - t0) * 1e3
            wall_ms.append(ms)
            # Total checksum: per-realization sum of final state, then sum over m
            per_m = np.asarray(jnp.sum(final_x, axis=1))
            total_checksum = float(np.sum(per_m))
            if not a.quiet:
                print(f"  run {run}  wall={ms:.2f} ms  checksum={total_checksum:.15g}")
    finally:
        if profile_ctx is not None:
            profile_ctx.__exit__(None, None, None)

    median_ms = float(np.median(wall_ms))
    if not a.quiet:
        print(f"[ekf_jax] median wall = {median_ms:.2f} ms over {a.runs} runs")

    # ---- Append rows to results CSV ----
    if a.out:
        out_path = Path(a.out)
        write_header = not out_path.exists()
        with open(out_path, "a") as f:
            if write_header:
                f.write("impl,M,T,workers,run_idx,wall_time_ms,checksum\n")
            for run, ms in enumerate(wall_ms):
                f.write(f"jax,{a.M},{N},1,{run},{ms:.6f},{total_checksum:.15g}\n")
        if not a.quiet:
            print(f"[ekf_jax] appended {len(wall_ms)} rows to {a.out}")

    # ---- Per-realization golden checksum ----
    if a.golden:
        with open(a.golden, "w") as f:
            f.write("m,checksum\n")
            for m in range(a.M):
                f.write(f"{m},{per_m[m]:.17g}\n")
        if not a.quiet:
            print(f"[ekf_jax] wrote {a.golden} ({a.M} rows)")

    # ---- Trajectory CSV (realization 0 only) — separate run on the M=1 path
    # so we don't blow up GPU memory accumulating a (M, N, 9) tensor.
    if a.trajectory:
        traj_noise = noise_stream[0]   # realization 0's noise stream
        _, traj_jax = run_traj_jit(
            imu_uw_seq, gt_seq, dt_seq, ts,
            traj_noise, Q, R_meas, sigma_meas, x0, P0)
        traj_jax.block_until_ready()
        traj_np = np.asarray(traj_jax)    # shape (N, 9): [p, v, theta]
        with open(a.trajectory, "w") as f:
            f.write("t,gt_x,gt_y,gt_z,est_x,est_y,est_z,gt_tx,gt_ty,gt_tz,est_tx,est_ty,est_tz\n")
            for t in range(N):
                f.write(f"{imu_t_np[t]:.6f},"
                        f"{gt_np[t,0]:.6f},{gt_np[t,1]:.6f},{gt_np[t,2]:.6f},"
                        f"{traj_np[t,0]:.6f},{traj_np[t,1]:.6f},{traj_np[t,2]:.6f},"
                        f"{gt_np[t,3]:.6f},{gt_np[t,4]:.6f},{gt_np[t,5]:.6f},"
                        f"{traj_np[t,6]:.6f},{traj_np[t,7]:.6f},{traj_np[t,8]:.6f}\n")
        if not a.quiet:
            print(f"[ekf_jax] wrote {a.trajectory} ({N} rows)")

    if jit_compile_ms is not None and not a.quiet:
        print(f"[ekf_jax] JIT compile cost (one-time) ≈ {jit_compile_ms:.0f} ms")
        print(f"[ekf_jax] amortised wall after JIT     ≈ {median_ms:.0f} ms / run")


if __name__ == "__main__":
    main()
