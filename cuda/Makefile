# Build the CUDA Kalman-filter driver.
#   make            # release build (default)
#   make clean
#
# Targets the cluster's NVIDIA L4 GPU (sm_89, Ada Lovelace) but the same
# code compiles for any CC>=7.0 — adjust ARCH if running on a different GPU.

NVCC   ?= nvcc
ARCH   ?= sm_89
CXXSTD := -std=c++17
OPT    := -O3 --use_fast_math
INC    := -I../common
WARN   := -Xcompiler -Wall,-Wextra
LIBS   := -lcurand

NVCCFLAGS := $(CXXSTD) $(OPT) -arch=$(ARCH) $(INC) $(WARN) \
             --expt-relaxed-constexpr -lineinfo
# `prof` adds ptxas verbose output so per-thread register usage and shared-mem
# requirements are dumped at compile time — direct evidence for the report's
# CUDA register-pressure bottleneck claim, exactly the metric NCU's
# `launch__registers_per_thread` exposes at runtime.
NVCCFLAGS_PROF := $(NVCCFLAGS) -Xptxas=-v

SRC := ekf_cuda.cu
BIN := ekf_cuda

.PHONY: all prof clean
all: $(BIN)

$(BIN): $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
        ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) $(LIBS)

# Re-emit the binary with -Xptxas=-v so the register/shared-mem count lands in
# stderr; the slurm script tees stderr to results/ptxas.log.  Compile for
# sm_80 (compute capability 8.0) so the same binary works whether the
# scheduler hands us an A10G (sm_86) or an L4 (sm_89) — both forward-compat
# from sm_80 via PTX.  Register usage at sm_80 is within ~1 of sm_89 so the
# occupancy diagnosis carries over.  We pin -ccbin to /usr/bin/g++ because
# the spack-env gcc on $PATH was built for `linux-skylake_avx512` (login
# node) and SIGILLs on some GPU host CPUs (e.g. A10G hosts).
PROF_ARCH ?= sm_80
PROF_CCBIN ?= /usr/bin/g++
NVCCFLAGS_PROF_ARCH := $(CXXSTD) $(OPT) -arch=$(PROF_ARCH) $(INC) $(WARN) \
                      --expt-relaxed-constexpr -lineinfo -Xptxas=-v \
                      -ccbin $(PROF_CCBIN)
prof: $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
      ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(NVCC) $(NVCCFLAGS_PROF_ARCH) -o $(BIN)_prof $(SRC) $(LIBS)

clean:
	rm -f $(BIN) $(BIN)_prof
