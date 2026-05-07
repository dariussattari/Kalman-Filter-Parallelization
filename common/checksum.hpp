// common/checksum.hpp
// Cross-implementation validation: each realization contributes a deterministic
// scalar (sum of its final state vector) and the per-realization values are
// summed in monotonically increasing m order to guarantee bit-stable output
// across serial/openmp/mpi/cuda when each implementation produces the same
// trajectory.

#ifndef EKF_CHECKSUM_HPP
#define EKF_CHECKSUM_HPP

#include <cstddef>

#ifndef EKF_HD
  #ifdef __CUDACC__
    #define EKF_HD __host__ __device__
  #else
    #define EKF_HD
  #endif
#endif

namespace checksum {

EKF_HD inline double of_state(const double* x, int n = 9) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

// Sum a per-realization checksum array in fixed (ascending m) order, so the
// answer is independent of how many threads / ranks / GPU blocks ran.
inline double total(const double* per_m, int M) {
    double s = 0.0;
    for (int m = 0; m < M; ++m) s += per_m[m];
    return s;
}

}  // namespace checksum

#endif  // EKF_CHECKSUM_HPP
