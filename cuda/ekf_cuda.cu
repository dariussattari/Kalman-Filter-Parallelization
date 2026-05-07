// cuda/ekf_cuda.cu
//
// Part 4 deliverable: CUDA-accelerated Monte-Carlo EKF.
//
// Strategy: one CUDA thread per Monte-Carlo realization.  All per-realization
// state (x[9], P[81], plus working buffers in predict/update) lives in the
// thread's stack — the compiler keeps the hot fields in registers and spills
// the rest to local memory on the L4 GPU's GDDR6.  The same predict()/update()
// from common/ekf_math.hpp are called: those functions are __host__ __device__
// via the EKF_HD macro, so the same source compiles for CPU and GPU.
//
// RNG choice: cuRAND's Philox4_32_10.  Unlike std::mt19937_64 (used by the CPU
// drivers), Philox is a counter-based generator with no GPU-friendly equivalent
// in libstdc++.  This means CUDA produces a different noise stream from
// serial/openmp/mpi → its golden checksum will NOT match the CPU 1e-9
// tolerance.  CUDA correctness is verified differently:
//   1. Within-CUDA determinism: same seed → byte-identical checksum (proved by
//      re-running, since cuRAND Philox is deterministic).
//   2. Trajectory RMSE vs ground truth matches the CPU implementations within
//      statistical tolerance (~few percent).
//   3. Visually, the CUDA estimate overlays serial/openmp/mpi on the
//      trajectory comparison plot.
//
// CLI: same as the CPU drivers.

#include "../common/ekf_math.hpp"
#include "../common/ekf_config.hpp"
#include "../common/io.hpp"
#include "../common/timing.hpp"
#include "../common/checksum.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do {                                                \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error %s:%d  %s -> %s\n",                 \
                     __FILE__, __LINE__, #expr, cudaGetErrorString(_err));   \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

namespace {

struct Args {
    std::string data;
    int    M     = 128;
    long   seed  = 42;
    int    runs  = 5;
    int    warmup = 1;
    std::string out;
    std::string golden;
    std::string trajectory;
    bool   quiet = false;
    int    block_size = 128;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if      (s == "--data")       a.data = next("--data");
        else if (s == "--M")          a.M = std::atoi(next("--M"));
        else if (s == "--seed")       a.seed = std::atol(next("--seed"));
        else if (s == "--runs")       a.runs = std::atoi(next("--runs"));
        else if (s == "--warmup")     a.warmup = std::atoi(next("--warmup"));
        else if (s == "--out")        a.out = next("--out");
        else if (s == "--golden")     a.golden = next("--golden");
        else if (s == "--trajectory") a.trajectory = next("--trajectory");
        else if (s == "--block-size") a.block_size = std::atoi(next("--block-size"));
        else if (s == "--quiet")      a.quiet = true;
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); std::exit(2); }
    }
    if (a.data.empty()) { std::fprintf(stderr, "--data is required\n"); std::exit(2); }
    if (a.M <= 0)       { std::fprintf(stderr, "--M must be positive\n"); std::exit(2); }
    return a;
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel: one thread per realization.
// ---------------------------------------------------------------------------
__global__ void ekf_kernel(
        const double* __restrict__ imu_t,    // [N]
        const double* __restrict__ imu_uw,   // [N x 6]
        const double* __restrict__ gt,       // [N x 6]
        const double* __restrict__ Q,        // [81]
        const double* __restrict__ R,        // [36]
        int N,
        int M,
        unsigned long long base_seed,
        double*       checksum_per_m,        // [M]
        double*       trajectory_out)        // [N x 12] or nullptr
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    double x[9], P[81];
    ekf_cfg::fill_initial_state(x, gt);
    ekf_cfg::fill_initial_P(P);

    // Per-realization, statistically independent stream via the cuRAND seed.
    curandStatePhilox4_32_10_t rng;
    curand_init(base_seed + (unsigned long long)m, 0ULL, 0ULL, &rng);

    const bool save_traj = (trajectory_out != nullptr) && (m == 0);
    if (save_traj) {
        for (int j = 0; j < 3; ++j) trajectory_out[j]      = gt[j];
        for (int j = 0; j < 3; ++j) trajectory_out[3 + j]  = x[j];
        for (int j = 0; j < 3; ++j) trajectory_out[6 + j]  = gt[3 + j];
        for (int j = 0; j < 3; ++j) trajectory_out[9 + j]  = x[6 + j];
    }

    for (int t = 1; t < N; ++t) {
        const double dt = imu_t[t] - imu_t[t - 1];
        ekf::predict(x, P, &imu_uw[(t - 1) * 6], dt, Q);

        if ((t % ekf_cfg::UPDATE_PERIOD) == 0) {
            double z[6];
            z[0] = gt[t * 6 + 0] + ekf_cfg::SIGMA_MEAS_POS * curand_normal_double(&rng);
            z[1] = gt[t * 6 + 1] + ekf_cfg::SIGMA_MEAS_POS * curand_normal_double(&rng);
            z[2] = gt[t * 6 + 2] + ekf_cfg::SIGMA_MEAS_POS * curand_normal_double(&rng);
            z[3] = gt[t * 6 + 3] + ekf_cfg::SIGMA_MEAS_ATT * curand_normal_double(&rng);
            z[4] = gt[t * 6 + 4] + ekf_cfg::SIGMA_MEAS_ATT * curand_normal_double(&rng);
            z[5] = gt[t * 6 + 5] + ekf_cfg::SIGMA_MEAS_ATT * curand_normal_double(&rng);
            ekf::update(x, P, z, R);
        }

        if (save_traj) {
            double* row = trajectory_out + static_cast<size_t>(t) * 12;
            for (int j = 0; j < 3; ++j) row[j]     = gt[t * 6 + j];
            for (int j = 0; j < 3; ++j) row[3 + j] = x[j];
            for (int j = 0; j < 3; ++j) row[6 + j] = gt[t * 6 + 3 + j];
            for (int j = 0; j < 3; ++j) row[9 + j] = x[6 + j];
        }
    }

    checksum_per_m[m] = checksum::of_state(x);
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    ekf_io::Dataset d = ekf_io::load(a.data);

    // Pick device 0 and report it.
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (!a.quiet) {
        std::printf("[ekf_cuda] loaded %s: N=%d  T=%.3fs  dt=%.3fms\n",
                    a.data.c_str(), d.N, d.T_seconds, d.dt_imu * 1e3);
        std::printf("[ekf_cuda] device     : %s  (CC %d.%d, %d SMs)\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        std::printf("[ekf_cuda] M=%d  seed=%ld  runs=%d (warmup=%d)  update_period=%d  block=%d\n",
                    a.M, a.seed, a.runs, a.warmup, ekf_cfg::UPDATE_PERIOD, a.block_size);
    }

    // Q, R prepared on host then copied to device.
    double Q_h[81], R_h[36];
    ekf_cfg::fill_Q(Q_h);
    ekf_cfg::fill_R(R_h);

    // Allocate device buffers.
    double *d_imu_t = nullptr, *d_imu_uw = nullptr, *d_gt = nullptr;
    double *d_Q = nullptr, *d_R = nullptr;
    double *d_checksum = nullptr, *d_trajectory = nullptr;
    CUDA_CHECK(cudaMalloc(&d_imu_t,   d.N           * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_imu_uw,  d.N * 6       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gt,      d.N * 6       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q,       81            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_R,       36            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_checksum, a.M          * sizeof(double)));
    if (!a.trajectory.empty()) {
        CUDA_CHECK(cudaMalloc(&d_trajectory,
                              static_cast<size_t>(d.N) * 12 * sizeof(double)));
    }

    // H2D copies (timed separately from the kernel).
    cudaEvent_t e_h2d_a, e_h2d_b;
    cudaEventCreate(&e_h2d_a); cudaEventCreate(&e_h2d_b);
    cudaEventRecord(e_h2d_a);
    CUDA_CHECK(cudaMemcpy(d_imu_t,  d.imu_t.data(),  d.N      * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_imu_uw, d.imu_uw.data(), d.N * 6  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gt,     d.gt.data(),     d.N * 6  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,      Q_h,             81       * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R,      R_h,             36       * sizeof(double), cudaMemcpyHostToDevice));
    cudaEventRecord(e_h2d_b);
    cudaEventSynchronize(e_h2d_b);
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, e_h2d_a, e_h2d_b);
    if (!a.quiet) std::printf("[ekf_cuda] H2D      : %.2f ms\n", h2d_ms);

    const int block_size = a.block_size;
    const int grid_size  = (a.M + block_size - 1) / block_size;

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start); cudaEventCreate(&e_stop);

    // Warmup runs (kernel-only timing).
    for (int w = 0; w < a.warmup; ++w) {
        ekf_kernel<<<grid_size, block_size>>>(
                d_imu_t, d_imu_uw, d_gt, d_Q, d_R,
                d.N, a.M, static_cast<unsigned long long>(a.seed),
                d_checksum, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Timed runs.
    std::vector<double> wall_ms;
    wall_ms.reserve(a.runs);
    std::vector<double> checksum_h(a.M);
    double last_total_checksum = 0.0;
    for (int run = 0; run < a.runs; ++run) {
        cudaEventRecord(e_start);
        ekf_kernel<<<grid_size, block_size>>>(
                d_imu_t, d_imu_uw, d_gt, d_Q, d_R,
                d.N, a.M, static_cast<unsigned long long>(a.seed),
                d_checksum,
                (run == 0) ? d_trajectory : nullptr);
        cudaEventRecord(e_stop);
        cudaEventSynchronize(e_stop);
        CUDA_CHECK(cudaGetLastError());
        float ms_f = 0.0f;
        cudaEventElapsedTime(&ms_f, e_start, e_stop);
        const double ms = static_cast<double>(ms_f);
        wall_ms.push_back(ms);
        CUDA_CHECK(cudaMemcpy(checksum_h.data(), d_checksum,
                              a.M * sizeof(double), cudaMemcpyDeviceToHost));
        last_total_checksum = checksum::total(checksum_h.data(), a.M);
        if (!a.quiet) {
            std::printf("  run %d  wall=%.2f ms  checksum=%.15g\n",
                        run, ms, last_total_checksum);
        }
    }

    std::vector<double> sorted = wall_ms;
    std::sort(sorted.begin(), sorted.end());
    const double median_ms = sorted[sorted.size() / 2];
    if (!a.quiet) {
        std::printf("[ekf_cuda] median wall = %.2f ms over %d runs\n",
                    median_ms, a.runs);
    }

    // ---- Append per-run rows to results CSV ----
    if (!a.out.empty()) {
        bool exists = false;
        if (std::FILE* ck = std::fopen(a.out.c_str(), "r")) { std::fclose(ck); exists = true; }
        std::FILE* f = std::fopen(a.out.c_str(), "a");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.out.c_str()); return 1; }
        if (!exists) {
            std::fprintf(f, "impl,M,T,workers,run_idx,wall_time_ms,checksum\n");
        }
        // workers = 1 (single GPU device); throughput is reflected by M.
        for (int run = 0; run < a.runs; ++run) {
            std::fprintf(f, "cuda,%d,%d,1,%d,%.6f,%.15g\n",
                         a.M, d.N, run, wall_ms[run], last_total_checksum);
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_cuda] appended %d rows to %s\n", a.runs, a.out.c_str());
    }

    // ---- Per-realization golden checksum CSV ----
    if (!a.golden.empty()) {
        std::FILE* f = std::fopen(a.golden.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.golden.c_str()); return 1; }
        std::fprintf(f, "m,checksum\n");
        for (int m = 0; m < a.M; ++m) {
            std::fprintf(f, "%d,%.17g\n", m, checksum_h[m]);
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_cuda] wrote %s (%d rows)\n", a.golden.c_str(), a.M);
    }

    // ---- Trajectory CSV (rank 0 of m=0) ----
    if (!a.trajectory.empty()) {
        std::vector<double> traj_h(static_cast<size_t>(d.N) * 12);
        CUDA_CHECK(cudaMemcpy(traj_h.data(), d_trajectory,
                              traj_h.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::FILE* f = std::fopen(a.trajectory.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.trajectory.c_str()); return 1; }
        std::fprintf(f, "t,gt_x,gt_y,gt_z,est_x,est_y,est_z,gt_tx,gt_ty,gt_tz,est_tx,est_ty,est_tz\n");
        for (int t = 0; t < d.N; ++t) {
            const double* row = traj_h.data() + static_cast<size_t>(t) * 12;
            std::fprintf(f, "%.6f", d.imu_t[t]);
            for (int j = 0; j < 12; ++j) std::fprintf(f, ",%.6f", row[j]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_cuda] wrote %s (%d rows)\n", a.trajectory.c_str(), d.N);
    }

    // ---- Cleanup ----
    cudaFree(d_imu_t); cudaFree(d_imu_uw); cudaFree(d_gt);
    cudaFree(d_Q); cudaFree(d_R);
    cudaFree(d_checksum);
    if (d_trajectory) cudaFree(d_trajectory);
    cudaEventDestroy(e_start); cudaEventDestroy(e_stop);
    cudaEventDestroy(e_h2d_a); cudaEventDestroy(e_h2d_b);

    return 0;
}
