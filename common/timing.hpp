// common/timing.hpp
// Wall-time stopwatch over std::chrono::steady_clock. steady_clock (not
// high_resolution_clock) is used because some platforms alias the latter to a
// non-monotonic clock; for performance measurement we always want monotonic.

#ifndef EKF_TIMING_HPP
#define EKF_TIMING_HPP

#include <chrono>

namespace timing {

struct Stopwatch {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;

    void start() { t0 = clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

}  // namespace timing

#endif  // EKF_TIMING_HPP
