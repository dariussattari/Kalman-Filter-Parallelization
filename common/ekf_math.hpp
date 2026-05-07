// common/ekf_math.hpp
//
// Shared EKF math for the CS 2050 final project (Kalman filter on EuRoC MAV).
// Header-only, no dependencies beyond <cmath>. Compiles cleanly under g++ and
// nvcc; mark functions __host__ __device__ via the EKF_HD macro so the same
// source runs on the CPU drivers (serial / openmp / mpi) and the CUDA kernel.
//
// State layout (9-dim direct-state EKF):
//   x[0..2]  = position p_xyz   in world frame (m)
//   x[3..5]  = velocity v_xyz   in world frame (m/s)
//   x[6..8]  = attitude theta   intrinsic XYZ Euler angles (rad)
//              Matches scipy.spatial.transform.Rotation.from_euler("xyz", ...)
//
// IMU input layout (matches data/preprocess.py output):
//   u[0..2]  = angular rate omega_xyz in body frame (rad/s)
//   u[3..5]  = specific force a_xyz in body frame (m/s^2; sensed accel + R^T*g)
//
// Measurement layout:
//   z[0..2]  = position p_xyz in world frame (m)
//   z[3..5]  = attitude theta in world frame (rad, XYZ Euler)
//
// Covariance matrices are stored as flat row-major double arrays:
//   P  is 9x9  -> double[81]
//   Q  is 9x9  -> double[81]
//   R  is 6x6  -> double[36]   (measurement noise covariance)

#ifndef EKF_MATH_HPP
#define EKF_MATH_HPP

#include <cmath>

#ifdef __CUDACC__
  #define EKF_HD __host__ __device__
#else
  #define EKF_HD
#endif

namespace ekf {

constexpr int N_STATE = 9;
constexpr int N_MEAS  = 6;
constexpr int N_INPUT = 6;

// World-frame gravity. Sign convention: positive Z is up; gravity points down.
constexpr double GRAVITY_Z = 9.80665;

// =====================================================================
// Tiny linear-algebra helpers (row-major).  C must not alias A or B.
// =====================================================================

EKF_HD inline void mat_zero(double* M, int n) {
    for (int i = 0; i < n; ++i) M[i] = 0.0;
}

EKF_HD inline void mat_eye(double* M, int n) {
    for (int i = 0; i < n * n; ++i) M[i] = 0.0;
    for (int i = 0; i < n; ++i)     M[i * n + i] = 1.0;
}

EKF_HD inline void mat_copy(const double* src, double* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

// C[m x n] = A[m x k] * B[k x n]
EKF_HD inline void matmul(const double* A, const double* B, double* C,
                          int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) s += A[i * k + p] * B[p * n + j];
            C[i * n + j] = s;
        }
    }
}

// C[m x n] = A[m x k] * B^T, where B is stored as [n x k] row-major.
EKF_HD inline void matmul_AB_T(const double* A, const double* B, double* C,
                               int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) s += A[i * k + p] * B[j * k + p];
            C[i * n + j] = s;
        }
    }
}

EKF_HD inline void mat_add_n(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

EKF_HD inline void mat_sub_n(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) C[i] = A[i] - B[i];
}

// 6x6 inverse via Gauss-Jordan with partial pivoting. Returns false if singular.
// Workspace is a stack-resident 6x12 augmented matrix; no heap allocation.
EKF_HD inline bool inv6(const double* A_in, double* A_inv) {
    constexpr int N = 6;
    double M[N * 2 * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M[i * 2 * N + j]     = A_in[i * N + j];
            M[i * 2 * N + N + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        double pmax = std::fabs(M[col * 2 * N + col]);
        for (int r = col + 1; r < N; ++r) {
            double v = std::fabs(M[r * 2 * N + col]);
            if (v > pmax) { pmax = v; pivot = r; }
        }
        if (pmax < 1e-300) return false;
        if (pivot != col) {
            for (int j = 0; j < 2 * N; ++j) {
                double t = M[col * 2 * N + j];
                M[col * 2 * N + j] = M[pivot * 2 * N + j];
                M[pivot * 2 * N + j] = t;
            }
        }
        double pv = M[col * 2 * N + col];
        for (int j = 0; j < 2 * N; ++j) M[col * 2 * N + j] /= pv;
        for (int r = 0; r < N; ++r) {
            if (r == col) continue;
            double f = M[r * 2 * N + col];
            if (f == 0.0) continue;
            for (int j = 0; j < 2 * N; ++j) {
                M[r * 2 * N + j] -= f * M[col * 2 * N + j];
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_inv[i * N + j] = M[i * 2 * N + N + j];
        }
    }
    return true;
}

// Wrap an angle to [-pi, pi]. Used on the attitude residual so the EKF doesn't
// chase a 2*pi jump when measurement and state straddle the wrap.
EKF_HD inline double wrap_pi(double a) {
    constexpr double PI  = 3.141592653589793;
    constexpr double TAU = 6.283185307179586;
    double w = a + PI;
    w -= TAU * std::floor(w / TAU);
    return w - PI;
}

// =====================================================================
// Rotation: R(theta) = Rx(tx) * Ry(ty) * Rz(tz), row-major 3x3.
// Matches scipy Rotation.from_euler("xyz", theta).as_matrix().
// =====================================================================
EKF_HD inline void body_to_world(const double theta[3], double R[9]) {
    const double sx = std::sin(theta[0]), cx = std::cos(theta[0]);
    const double sy = std::sin(theta[1]), cy = std::cos(theta[1]);
    const double sz = std::sin(theta[2]), cz = std::cos(theta[2]);

    R[0] = cy * cz;                  R[1] = -cy * sz;                 R[2] = sy;
    R[3] = cx * sz + sx * sy * cz;   R[4] = cx * cz - sx * sy * sz;   R[5] = -sx * cy;
    R[6] = sx * sz - cx * sy * cz;   R[7] = sx * cz + cx * sy * sz;   R[8] = cx * cy;
}

// J[3x3] = d(R(theta) * a) / d(theta), via the intrinsic-axis chain rule:
//   col0 = skew(ex)         * (R*a),  ex      = [1, 0, 0]
//   col1 = skew(Rx*ey)      * (R*a),  Rx*ey   = [0, cx, sx]
//   col2 = skew(Rx*Ry*ez)   * (R*a),  Rx*Ry*ez = [sy, -sx*cy, cx*cy]
EKF_HD inline void dRa_dtheta(const double theta[3], const double a[3], double J[9]) {
    const double sx = std::sin(theta[0]), cx = std::cos(theta[0]);
    const double sy = std::sin(theta[1]), cy = std::cos(theta[1]);

    double R[9];
    body_to_world(theta, R);
    const double v0 = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
    const double v1 = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
    const double v2 = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];

    // skew(w)*v = [w1*v2 - w2*v1, w2*v0 - w0*v2, w0*v1 - w1*v0]
    // col0 (w = ex = [1,0,0])
    J[0] = 0.0;     J[3] = -v2;     J[6] =  v1;
    // col1 (w = [0, cx, sx])
    J[1] = cx * v2 - sx * v1;
    J[4] = sx * v0;
    J[7] = -cx * v0;
    // col2 (w = [sy, -sx*cy, cx*cy])
    const double wx = sy;
    const double wy = -sx * cy;
    const double wz =  cx * cy;
    J[2] = wy * v2 - wz * v1;
    J[5] = wz * v0 - wx * v2;
    J[8] = wx * v1 - wy * v0;
}

// =====================================================================
// EKF predict step.
// Modifies x[9] and P[81] in place.
//   x      : pre-step state (in)/post-step state (out)
//   P      : pre-step covariance (in)/post-step covariance (out)
//   u[6]   : IMU input (omega, accel) at this step
//   dt     : timestep (s)
//   Q[81]  : process-noise covariance (additive on the state)
// =====================================================================
EKF_HD inline void predict(double* x, double* P,
                           const double u[6], double dt,
                           const double* Q) {
    // ---- nonlinear state prediction ----
    const double* w = &u[0];   // omega
    const double* a = &u[3];   // accel (body frame)

    // Save theta_prev before mutating x; needed for the Jacobian below.
    const double theta_prev[3] = {x[6], x[7], x[8]};

    double R[9];
    body_to_world(theta_prev, R);

    // a_world = R * a_body - g (g = (0, 0, GRAVITY_Z))
    double aw0 = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
    double aw1 = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
    double aw2 = R[6] * a[0] + R[7] * a[1] + R[8] * a[2] - GRAVITY_Z;

    // p_new = p + v*dt + 0.5*aw*dt^2
    x[0] += x[3] * dt + 0.5 * aw0 * dt * dt;
    x[1] += x[4] * dt + 0.5 * aw1 * dt * dt;
    x[2] += x[5] * dt + 0.5 * aw2 * dt * dt;
    // v_new = v + aw*dt
    x[3] += aw0 * dt;
    x[4] += aw1 * dt;
    x[5] += aw2 * dt;
    // theta_new = theta + omega*dt, then wrap to [-pi, pi] so the update
    // step's wrapped innovation always sees the shorter angular path.
    x[6] = wrap_pi(x[6] + w[0] * dt);
    x[7] = wrap_pi(x[7] + w[1] * dt);
    x[8] = wrap_pi(x[8] + w[2] * dt);

    // ---- Jacobian F (9x9) of f w.r.t. x_prev ----
    // theta_prev was captured at the top of the function.
    double dRa[9];
    dRa_dtheta(theta_prev, a, dRa);

    double F[81];
    mat_eye(F, 9);
    // F[0:3, 3:6] = I*dt
    F[0 * 9 + 3] = dt; F[1 * 9 + 4] = dt; F[2 * 9 + 5] = dt;
    // F[0:3, 6:9] = 0.5 * dt^2 * dRa
    const double half_dt2 = 0.5 * dt * dt;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F[i * 9 + (6 + j)] = half_dt2 * dRa[i * 3 + j];
    // F[3:6, 6:9] = dt * dRa
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F[(3 + i) * 9 + (6 + j)] = dt * dRa[i * 3 + j];

    // ---- P = F P F^T + Q ----
    double FP[81], FPFT[81];
    matmul(F, P, FP, 9, 9, 9);
    matmul_AB_T(FP, F, FPFT, 9, 9, 9);
    mat_add_n(FPFT, Q, P, 81);
}

// =====================================================================
// EKF update step (linear measurement: pos and attitude observed directly).
// Modifies x[9] and P[81] in place.  Returns false if S is singular.
//   z[6]    : measurement (p, theta)
//   R[36]   : measurement-noise covariance (6x6)
// H is implicit:
//   H = [I_3  0_3  0_3 ]
//       [0_3  0_3  I_3 ]
// =====================================================================
EKF_HD inline bool update(double* x, double* P,
                          const double z[6], const double* R) {
    // Innovation y = z - H*x, with attitude residual wrapped to [-pi, pi].
    double y[6];
    y[0] = z[0] - x[0];
    y[1] = z[1] - x[1];
    y[2] = z[2] - x[2];
    y[3] = wrap_pi(z[3] - x[6]);
    y[4] = wrap_pi(z[4] - x[7]);
    y[5] = wrap_pi(z[5] - x[8]);

    // S = H P H^T + R, where H selects rows/cols (0,1,2) and (6,7,8) of P.
    // Build the 6x6 from those four 3x3 blocks of P.
    double S[36];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) S[i * 6 + j]             = P[i * 9 + j];           // top-left
        for (int j = 0; j < 3; ++j) S[i * 6 + (3 + j)]       = P[i * 9 + (6 + j)];     // top-right
        for (int j = 0; j < 3; ++j) S[(3 + i) * 6 + j]       = P[(6 + i) * 9 + j];     // bottom-left
        for (int j = 0; j < 3; ++j) S[(3 + i) * 6 + (3 + j)] = P[(6 + i) * 9 + (6 + j)]; // bottom-right
    }
    mat_add_n(S, R, S, 36);

    double S_inv[36];
    if (!inv6(S, S_inv)) return false;

    // K = P H^T S^-1.  P*H^T (9x6) is just columns (0,1,2,6,7,8) of P.
    double PHT[54];  // 9 x 6
    for (int i = 0; i < 9; ++i) {
        PHT[i * 6 + 0] = P[i * 9 + 0];
        PHT[i * 6 + 1] = P[i * 9 + 1];
        PHT[i * 6 + 2] = P[i * 9 + 2];
        PHT[i * 6 + 3] = P[i * 9 + 6];
        PHT[i * 6 + 4] = P[i * 9 + 7];
        PHT[i * 6 + 5] = P[i * 9 + 8];
    }

    double K[54];  // 9 x 6
    matmul(PHT, S_inv, K, 9, 6, 6);

    // x += K * y
    for (int i = 0; i < 9; ++i) {
        double s = 0.0;
        for (int j = 0; j < 6; ++j) s += K[i * 6 + j] * y[j];
        x[i] += s;
    }
    // Re-wrap attitude to [-pi, pi] after the additive update.
    x[6] = wrap_pi(x[6]);
    x[7] = wrap_pi(x[7]);
    x[8] = wrap_pi(x[8]);

    // P = (I - K H) P.  K*H is the 9x9 matrix whose columns 0..2 come from
    // K's first three columns and whose columns 6..8 come from K's last three.
    double KH[81];
    mat_zero(KH, 81);
    for (int i = 0; i < 9; ++i) {
        KH[i * 9 + 0] = K[i * 6 + 0];
        KH[i * 9 + 1] = K[i * 6 + 1];
        KH[i * 9 + 2] = K[i * 6 + 2];
        KH[i * 9 + 6] = K[i * 6 + 3];
        KH[i * 9 + 7] = K[i * 6 + 4];
        KH[i * 9 + 8] = K[i * 6 + 5];
    }
    double IKH[81];
    mat_eye(IKH, 9);
    mat_sub_n(IKH, KH, IKH, 81);

    double P_new[81];
    matmul(IKH, P, P_new, 9, 9, 9);
    mat_copy(P_new, P, 81);
    return true;
}

}  // namespace ekf

#endif  // EKF_MATH_HPP
