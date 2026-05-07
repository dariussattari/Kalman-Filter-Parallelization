// serial/ekf_serial.cpp
//
// Part 1 deliverable: serial baseline of the multi-realization Monte-Carlo EKF.
//
// The outer loop runs M independent realizations (each with its own RNG seed
// = base_seed + m), generating a different stream of measurement noise. The
// inner time loop predicts at every IMU step (200 Hz) and corrects every
// UPDATE_PERIOD steps (default 10 -> 20 Hz). Final state checksums are summed
// in ascending-m order so the answer is bit-identical regardless of
// parallelization strategy in later parts.
//
// CLI:
//   --data PATH        path to packed binary from data/preprocess.py (required)
//   --M N              number of Monte-Carlo realizations (default 128)
//   --seed S           base RNG seed (default 42)
//   --runs R           number of timed runs (default 5)
//   --warmup W         number of warmup runs (default 1)
//   --out PATH         results CSV (one row per timed run, appended)
//   --golden PATH      per-realization checksum CSV (written once)
//   --trajectory PATH  per-step trajectory CSV for realization 0 (written once)
//   --quiet            suppress stdout progress

#include "../common/ekf_math.hpp"
#include "../common/ekf_config.hpp"
#include "../common/io.hpp"
#include "../common/timing.hpp"
#include "../common/checksum.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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
        else if (s == "--quiet")      a.quiet = true;
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); std::exit(2); }
    }
    if (a.data.empty()) { std::fprintf(stderr, "--data is required\n"); std::exit(2); }
    if (a.M <= 0)       { std::fprintf(stderr, "--M must be positive\n"); std::exit(2); }
    return a;
}

// Run M independent EKF realizations. Returns the per-realization final-state
// checksum array (size M). If trajectory_out is non-null, also fills that with
// per-step (gt_p, est_p, gt_theta, est_theta) for realization 0 — size N*12.
void run_ensemble(const ekf_io::Dataset& d, int M, long base_seed,
                  std::vector<double>& checksum_per_m,
                  std::vector<double>* trajectory_out) {
    const int N = d.N;
    const double* imu_uw = d.imu_uw.data();
    const double* gt     = d.gt.data();
    const double* imu_t  = d.imu_t.data();

    double Q[81], R[36];
    ekf_cfg::fill_Q(Q);
    ekf_cfg::fill_R(R);

    checksum_per_m.assign(M, 0.0);
    if (trajectory_out) trajectory_out->assign(static_cast<std::size_t>(N) * 12, 0.0);

    for (int m = 0; m < M; ++m) {
        double x[9], P[81];
        ekf_cfg::fill_initial_state(x, gt);    // gt[0..5] is GT at t=0
        ekf_cfg::fill_initial_P(P);

        std::mt19937_64 rng(static_cast<uint64_t>(base_seed) + static_cast<uint64_t>(m));
        std::normal_distribution<double> norm(0.0, 1.0);

        const bool save_traj = (trajectory_out != nullptr) && (m == 0);
        if (save_traj) {
            // t=0 row: estimate equals initial state which equals GT (no noise yet)
            double* row = trajectory_out->data();
            for (int j = 0; j < 3; ++j) row[j]      = gt[j];        // gt_p
            for (int j = 0; j < 3; ++j) row[3 + j]  = x[j];          // est_p
            for (int j = 0; j < 3; ++j) row[6 + j]  = gt[3 + j];     // gt_theta
            for (int j = 0; j < 3; ++j) row[9 + j]  = x[6 + j];      // est_theta
        }

        for (int t = 1; t < N; ++t) {
            // Predict: use IMU sample at the start of the interval [t-1, t].
            const double dt = imu_t[t] - imu_t[t - 1];
            ekf::predict(x, P, &imu_uw[(t - 1) * 6], dt, Q);

            // Update every UPDATE_PERIOD steps with a noisy measurement of (p, theta).
            if ((t % ekf_cfg::UPDATE_PERIOD) == 0) {
                double z[6];
                z[0] = gt[t * 6 + 0] + ekf_cfg::SIGMA_MEAS_POS * norm(rng);
                z[1] = gt[t * 6 + 1] + ekf_cfg::SIGMA_MEAS_POS * norm(rng);
                z[2] = gt[t * 6 + 2] + ekf_cfg::SIGMA_MEAS_POS * norm(rng);
                z[3] = gt[t * 6 + 3] + ekf_cfg::SIGMA_MEAS_ATT * norm(rng);
                z[4] = gt[t * 6 + 4] + ekf_cfg::SIGMA_MEAS_ATT * norm(rng);
                z[5] = gt[t * 6 + 5] + ekf_cfg::SIGMA_MEAS_ATT * norm(rng);
                ekf::update(x, P, z, R);
            }

            if (save_traj) {
                double* row = trajectory_out->data() + static_cast<std::size_t>(t) * 12;
                for (int j = 0; j < 3; ++j) row[j]     = gt[t * 6 + j];
                for (int j = 0; j < 3; ++j) row[3 + j] = x[j];
                for (int j = 0; j < 3; ++j) row[6 + j] = gt[t * 6 + 3 + j];
                for (int j = 0; j < 3; ++j) row[9 + j] = x[6 + j];
            }
        }
        checksum_per_m[m] = checksum::of_state(x);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    ekf_io::Dataset d = ekf_io::load(a.data);
    if (!a.quiet) {
        std::printf("[ekf_serial] loaded %s: N=%d  T=%.3fs  dt=%.3fms\n",
                    a.data.c_str(), d.N, d.T_seconds, d.dt_imu * 1e3);
        std::printf("[ekf_serial] M=%d  seed=%ld  runs=%d (warmup=%d)  update_period=%d\n",
                    a.M, a.seed, a.runs, a.warmup, ekf_cfg::UPDATE_PERIOD);
    }

    std::vector<double> checksum_per_m;
    std::vector<double> trajectory;
    std::vector<double>* traj_ptr = a.trajectory.empty() ? nullptr : &trajectory;

    // Warmup runs (not timed).
    for (int w = 0; w < a.warmup; ++w) {
        run_ensemble(d, a.M, a.seed, checksum_per_m, nullptr);
    }

    // Timed runs.
    std::vector<double> wall_ms;
    wall_ms.reserve(a.runs);
    double last_total_checksum = 0.0;
    for (int run = 0; run < a.runs; ++run) {
        timing::Stopwatch sw;
        sw.start();
        // Save trajectory only on the first timed run, only if requested.
        run_ensemble(d, a.M, a.seed, checksum_per_m,
                     (run == 0) ? traj_ptr : nullptr);
        const double ms = sw.elapsed_ms();
        wall_ms.push_back(ms);
        last_total_checksum = checksum::total(checksum_per_m.data(), a.M);
        if (!a.quiet) {
            std::printf("  run %d  wall=%.2f ms  checksum=%.15g\n",
                        run, ms, last_total_checksum);
        }
    }

    // Report median wall time.
    std::vector<double> sorted = wall_ms;
    std::sort(sorted.begin(), sorted.end());
    const double median_ms = sorted[sorted.size() / 2];
    if (!a.quiet) {
        std::printf("[ekf_serial] median wall = %.2f ms over %d runs\n",
                    median_ms, a.runs);
    }

    // Append per-run rows to results CSV.
    if (!a.out.empty()) {
        bool exists = false;
        if (std::FILE* ck = std::fopen(a.out.c_str(), "r")) { std::fclose(ck); exists = true; }
        std::FILE* f = std::fopen(a.out.c_str(), "a");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.out.c_str()); return 1; }
        if (!exists) {
            std::fprintf(f, "impl,M,T,workers,run_idx,wall_time_ms,checksum\n");
        }
        for (int run = 0; run < a.runs; ++run) {
            std::fprintf(f, "serial,%d,%d,1,%d,%.6f,%.15g\n",
                         a.M, d.N, run, wall_ms[run], last_total_checksum);
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_serial] appended %d rows to %s\n", a.runs, a.out.c_str());
    }

    // Per-realization golden checksum CSV.
    if (!a.golden.empty()) {
        std::FILE* f = std::fopen(a.golden.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.golden.c_str()); return 1; }
        std::fprintf(f, "m,checksum\n");
        for (int m = 0; m < a.M; ++m) {
            std::fprintf(f, "%d,%.17g\n", m, checksum_per_m[m]);
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_serial] wrote %s (%d rows)\n", a.golden.c_str(), a.M);
    }

    // Trajectory CSV (only if requested and we ran at least one timed run).
    if (!a.trajectory.empty() && !trajectory.empty()) {
        std::FILE* f = std::fopen(a.trajectory.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.trajectory.c_str()); return 1; }
        std::fprintf(f, "t,gt_x,gt_y,gt_z,est_x,est_y,est_z,gt_tx,gt_ty,gt_tz,est_tx,est_ty,est_tz\n");
        for (int t = 0; t < d.N; ++t) {
            const double* row = trajectory.data() + static_cast<std::size_t>(t) * 12;
            std::fprintf(f, "%.6f", d.imu_t[t]);
            for (int j = 0; j < 12; ++j) std::fprintf(f, ",%.6f", row[j]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        if (!a.quiet) std::printf("[ekf_serial] wrote %s (%d rows)\n", a.trajectory.c_str(), d.N);
    }

    return 0;
}
