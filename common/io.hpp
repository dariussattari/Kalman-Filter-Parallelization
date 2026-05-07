// common/io.hpp
// Loader for the packed float64 binary produced by data/preprocess.py.
//
// File layout:
//   line 1 : ASCII "N_imu=<N> T_seconds=<T> dt_imu=<dt>\n"
//   bytes  : imu_t  [N]      float64
//            imu_uw [N x 6]  float64  (omega_xyz, accel_xyz)  body frame
//            gt     [N x 6]  float64  (p_xyz, theta_xyz)      world frame

#ifndef EKF_IO_HPP
#define EKF_IO_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ekf_io {

struct Dataset {
    int N = 0;
    double T_seconds = 0.0;
    double dt_imu = 0.0;
    std::vector<double> imu_t;        // size N
    std::vector<double> imu_uw;       // size N*6
    std::vector<double> gt;           // size N*6
};

inline Dataset load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);

    // Header line — at most 128 chars, terminated by '\n'.
    char header[256];
    if (!std::fgets(header, sizeof(header), f)) {
        std::fclose(f);
        throw std::runtime_error("cannot read header from " + path);
    }
    Dataset d;
    if (std::sscanf(header, "N_imu=%d T_seconds=%lf dt_imu=%lf",
                    &d.N, &d.T_seconds, &d.dt_imu) != 3) {
        std::fclose(f);
        throw std::runtime_error("bad header: " + std::string(header));
    }
    if (d.N <= 0) {
        std::fclose(f);
        throw std::runtime_error("non-positive N in header");
    }

    d.imu_t.resize(d.N);
    d.imu_uw.resize(d.N * 6);
    d.gt.resize(d.N * 6);

    auto must_read = [&](double* dst, std::size_t n) {
        std::size_t got = std::fread(dst, sizeof(double), n, f);
        if (got != n) {
            std::fclose(f);
            throw std::runtime_error("short read in " + path);
        }
    };
    must_read(d.imu_t.data(),  d.N);
    must_read(d.imu_uw.data(), d.N * 6);
    must_read(d.gt.data(),     d.N * 6);
    std::fclose(f);
    return d;
}

}  // namespace ekf_io

#endif  // EKF_IO_HPP
