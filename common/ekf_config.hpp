// common/ekf_config.hpp
// Single source of truth for EKF tuning constants. All five implementations
// (serial, openmp, mpi, cuda, additional) include this so that initial state,
// process/measurement noise, and the noise-injection strategy stay byte-stable
// across paradigms — a prerequisite for the cross-implementation checksum
// validation tolerance of 1e-9.
//
// Defaults are tuned for the EuRoC V1_01_easy sequence at 200 Hz IMU rate.

#ifndef EKF_CONFIG_HPP
#define EKF_CONFIG_HPP

#include "ekf_math.hpp"

namespace ekf_cfg {

// ---- Initial covariance (diagonal). Position and attitude are well-known
// from the GT at t=0; velocity is unknown (we initialise it to zero). ----
constexpr double P0_POS  = 1.0e-4;   // (m)^2
constexpr double P0_VEL  = 2.5e-1;   // (m/s)^2  (sigma_v0 = 0.5 m/s)
constexpr double P0_ATT  = 1.0e-4;   // (rad)^2

// ---- Process noise (per-step diagonal Q). ----
constexpr double Q_POS   = 1.0e-8;
constexpr double Q_VEL   = 1.0e-4;   // dominant: drives IMU integration drift
constexpr double Q_ATT   = 1.0e-7;

// ---- Measurement noise (R) and matching noise std-devs we INJECT. ----
constexpr double SIGMA_MEAS_POS = 1.0e-2;   // 1 cm  std
constexpr double SIGMA_MEAS_ATT = 1.0e-2;   // ~0.6 deg std
constexpr double R_POS = SIGMA_MEAS_POS * SIGMA_MEAS_POS;
constexpr double R_ATT = SIGMA_MEAS_ATT * SIGMA_MEAS_ATT;

// ---- Measurement update cadence: do an update every UPDATE_PERIOD IMU steps.
// 1 = update every step; 10 = 20 Hz updates against 200 Hz IMU prediction.
// Larger values make the predict step do more visible work between corrections.
constexpr int UPDATE_PERIOD = 10;

EKF_HD inline void fill_initial_P(double* P) {
    ekf::mat_zero(P, 81);
    for (int i = 0; i < 3; ++i) P[i * 9 + i]              = P0_POS;
    for (int i = 0; i < 3; ++i) P[(3 + i) * 9 + (3 + i)]  = P0_VEL;
    for (int i = 0; i < 3; ++i) P[(6 + i) * 9 + (6 + i)]  = P0_ATT;
}

EKF_HD inline void fill_Q(double* Q) {
    ekf::mat_zero(Q, 81);
    for (int i = 0; i < 3; ++i) Q[i * 9 + i]              = Q_POS;
    for (int i = 0; i < 3; ++i) Q[(3 + i) * 9 + (3 + i)]  = Q_VEL;
    for (int i = 0; i < 3; ++i) Q[(6 + i) * 9 + (6 + i)]  = Q_ATT;
}

EKF_HD inline void fill_R(double* R) {
    ekf::mat_zero(R, 36);
    for (int i = 0; i < 3; ++i) R[i * 6 + i]              = R_POS;
    for (int i = 0; i < 3; ++i) R[(3 + i) * 6 + (3 + i)]  = R_ATT;
}

// Set initial state from the first ground-truth sample. Velocity = 0.
EKF_HD inline void fill_initial_state(double* x, const double* gt_row0) {
    x[0] = gt_row0[0]; x[1] = gt_row0[1]; x[2] = gt_row0[2];
    x[3] = 0.0;        x[4] = 0.0;        x[5] = 0.0;
    x[6] = gt_row0[3]; x[7] = gt_row0[4]; x[8] = gt_row0[5];
}

}  // namespace ekf_cfg

#endif  // EKF_CONFIG_HPP
