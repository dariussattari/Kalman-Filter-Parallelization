// mpi/ekf_mpi.cpp
//
// Part 3 deliverable: MPI-parallel Monte-Carlo EKF (distributed memory).
//
// Communication strategy
// ----------------------
// The Monte-Carlo realizations are independent, so this is embarrassingly
// parallel.  We use *block decomposition*: rank r of R total ranks owns
// realizations m in [r*M/R, (r+1)*M/R).  Each rank reads the same packed
// binary v101.bin from the shared filesystem (small enough that broadcasting
// from rank 0 would be slower than a parallel read).
//
//   Per-step communication: NONE (no collectives inside the EKF loop)
//   Per-run synchronisation:
//       MPI_Barrier        — ensure all ranks start the timed region together
//       MPI_Reduce(MAX)    — collect parallel wall time
//       MPI_Gatherv        — collect per-realization checksums on rank 0
//                            (preserves m-ascending order for deterministic sum)
//   Total: 3 collectives per timed run, all O(R) latency.
//
// Determinism: per-realization seed is base_seed + m, independent of which
// rank computes m.  The total checksum is summed in ascending-m order on
// rank 0 after the gather, so rank count cannot change the floating-point
// bit pattern of the result.  → matches serial within 1e-9.
//
// CLI: same as ekf_serial / ekf_omp.

#include "../common/ekf_math.hpp"
#include "../common/ekf_config.hpp"
#include "../common/io.hpp"
#include "../common/timing.hpp"
#include "../common/checksum.hpp"

#include <mpi.h>

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

// Run realizations m in [m_start, m_end). Fills checksum_local[0..(m_end-m_start)-1].
// If trajectory_out is non-null AND m_start <= 0 < m_end (i.e. realization 0 is on
// this rank), fills trajectory_out for realization 0.
void run_ensemble_range(const ekf_io::Dataset& d,
                        int m_start, int m_end, long base_seed,
                        std::vector<double>& checksum_local,
                        std::vector<double>* trajectory_out) {
    const int N = d.N;
    const double* imu_uw = d.imu_uw.data();
    const double* gt     = d.gt.data();
    const double* imu_t  = d.imu_t.data();

    double Q[81], R[36];
    ekf_cfg::fill_Q(Q);
    ekf_cfg::fill_R(R);

    const int my_M = m_end - m_start;
    checksum_local.assign(my_M, 0.0);
    if (trajectory_out) trajectory_out->assign(static_cast<std::size_t>(N) * 12, 0.0);

    for (int m = m_start; m < m_end; ++m) {
        double x[9], P[81];
        ekf_cfg::fill_initial_state(x, gt);
        ekf_cfg::fill_initial_P(P);

        std::mt19937_64 rng(static_cast<uint64_t>(base_seed) + static_cast<uint64_t>(m));
        std::normal_distribution<double> norm(0.0, 1.0);

        const bool save_traj = (trajectory_out != nullptr) && (m == 0);
        if (save_traj) {
            double* row = trajectory_out->data();
            for (int j = 0; j < 3; ++j) row[j]      = gt[j];
            for (int j = 0; j < 3; ++j) row[3 + j]  = x[j];
            for (int j = 0; j < 3; ++j) row[6 + j]  = gt[3 + j];
            for (int j = 0; j < 3; ++j) row[9 + j]  = x[6 + j];
        }

        for (int t = 1; t < N; ++t) {
            const double dt = imu_t[t] - imu_t[t - 1];
            ekf::predict(x, P, &imu_uw[(t - 1) * 6], dt, Q);

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
        checksum_local[m - m_start] = checksum::of_state(x);
    }
}

// Block-decompose [0, M) over `nranks`. Rank r owns [start_of(r), start_of(r+1)).
inline int block_start(int r, int nranks, int M) { return (r * M) / nranks; }

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args a = parse(argc, argv);

    // Each rank loads independently from the shared filesystem.
    ekf_io::Dataset d = ekf_io::load(a.data);

    const int my_start = block_start(rank,     size, a.M);
    const int my_end   = block_start(rank + 1, size, a.M);
    const int my_M     = my_end - my_start;

    if (rank == 0 && !a.quiet) {
        std::printf("[ekf_mpi] loaded %s: N=%d  T=%.3fs  dt=%.3fms\n",
                    a.data.c_str(), d.N, d.T_seconds, d.dt_imu * 1e3);
        std::printf("[ekf_mpi] M=%d  seed=%ld  runs=%d (warmup=%d)  update_period=%d  ranks=%d\n",
                    a.M, a.seed, a.runs, a.warmup, ekf_cfg::UPDATE_PERIOD, size);
    }

    // Recv counts/displs for MPI_Gatherv (only meaningful on rank 0, but cheap to build everywhere).
    std::vector<int> recv_counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const int s = block_start(r,     size, a.M);
        const int e = block_start(r + 1, size, a.M);
        recv_counts[r] = e - s;
        displs[r]      = s;
    }

    std::vector<double> checksum_local;
    std::vector<double> checksum_global;
    if (rank == 0) checksum_global.assign(a.M, 0.0);

    std::vector<double> trajectory;
    std::vector<double>* traj_ptr = (rank == 0 && !a.trajectory.empty()) ? &trajectory : nullptr;

    // Warmup runs (not timed). Trajectory not saved during warmup.
    for (int w = 0; w < a.warmup; ++w) {
        run_ensemble_range(d, my_start, my_end, a.seed, checksum_local, nullptr);
    }

    // Timed runs.
    std::vector<double> wall_ms;
    wall_ms.reserve(a.runs);
    double last_total_checksum = 0.0;
    for (int run = 0; run < a.runs; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        timing::Stopwatch sw;
        sw.start();
        run_ensemble_range(d, my_start, my_end, a.seed, checksum_local,
                           (run == 0) ? traj_ptr : nullptr);
        const double local_ms = sw.elapsed_ms();

        // Parallel wall time = slowest rank's local time.
        double max_ms = 0.0;
        MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Collect per-realization checksums in ascending-m order.
        MPI_Gatherv(checksum_local.data(), my_M, MPI_DOUBLE,
                    rank == 0 ? checksum_global.data() : nullptr,
                    recv_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            wall_ms.push_back(max_ms);
            last_total_checksum = checksum::total(checksum_global.data(), a.M);
            if (!a.quiet) {
                std::printf("  run %d  wall=%.2f ms  checksum=%.15g\n",
                            run, max_ms, last_total_checksum);
            }
        }
    }

    // ---- Rank 0 writes all output files ----
    if (rank == 0) {
        std::vector<double> sorted = wall_ms;
        std::sort(sorted.begin(), sorted.end());
        const double median_ms = sorted[sorted.size() / 2];
        if (!a.quiet) {
            std::printf("[ekf_mpi] median wall = %.2f ms over %d runs\n",
                        median_ms, a.runs);
        }

        if (!a.out.empty()) {
            bool exists = false;
            if (std::FILE* ck = std::fopen(a.out.c_str(), "r")) { std::fclose(ck); exists = true; }
            std::FILE* f = std::fopen(a.out.c_str(), "a");
            if (!f) {
                std::fprintf(stderr, "cannot open %s\n", a.out.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (!exists) {
                std::fprintf(f, "impl,M,T,workers,run_idx,wall_time_ms,checksum\n");
            }
            for (int run = 0; run < a.runs; ++run) {
                std::fprintf(f, "mpi,%d,%d,%d,%d,%.6f,%.15g\n",
                             a.M, d.N, size, run, wall_ms[run], last_total_checksum);
            }
            std::fclose(f);
            if (!a.quiet) std::printf("[ekf_mpi] appended %d rows to %s\n", a.runs, a.out.c_str());
        }

        if (!a.golden.empty()) {
            std::FILE* f = std::fopen(a.golden.c_str(), "w");
            if (!f) {
                std::fprintf(stderr, "cannot open %s\n", a.golden.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::fprintf(f, "m,checksum\n");
            for (int m = 0; m < a.M; ++m) {
                std::fprintf(f, "%d,%.17g\n", m, checksum_global[m]);
            }
            std::fclose(f);
            if (!a.quiet) std::printf("[ekf_mpi] wrote %s (%d rows)\n", a.golden.c_str(), a.M);
        }

        if (!a.trajectory.empty() && !trajectory.empty()) {
            std::FILE* f = std::fopen(a.trajectory.c_str(), "w");
            if (!f) {
                std::fprintf(stderr, "cannot open %s\n", a.trajectory.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::fprintf(f, "t,gt_x,gt_y,gt_z,est_x,est_y,est_z,gt_tx,gt_ty,gt_tz,est_tx,est_ty,est_tz\n");
            for (int t = 0; t < d.N; ++t) {
                const double* row = trajectory.data() + static_cast<std::size_t>(t) * 12;
                std::fprintf(f, "%.6f", d.imu_t[t]);
                for (int j = 0; j < 12; ++j) std::fprintf(f, ",%.6f", row[j]);
                std::fprintf(f, "\n");
            }
            std::fclose(f);
            if (!a.quiet) std::printf("[ekf_mpi] wrote %s (%d rows)\n", a.trajectory.c_str(), d.N);
        }
    }

    MPI_Finalize();
    return 0;
}
