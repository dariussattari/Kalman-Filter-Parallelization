# Build the MPI Kalman-filter driver.
#   make            # release build (default)
#   make debug      # -O0 -g for gdb
#   make clean

MPICXX   ?= mpicxx
CXXSTD   ?= -std=c++17
WARN     ?= -Wall -Wextra -Wpedantic
OPT      ?= -O3 -march=native -ffast-math
INC      ?= -I../common

CXXFLAGS_RELEASE := $(CXXSTD) $(WARN) $(OPT) $(INC)
CXXFLAGS_DEBUG   := $(CXXSTD) $(WARN) -O0 -g $(INC)
# Optimised build with debug symbols so VTune can attribute samples.
CXXFLAGS_PROF    := $(CXXSTD) $(WARN) $(OPT) -g -fno-omit-frame-pointer $(INC)

SRC := ekf_mpi.cpp
BIN := ekf_mpi

.PHONY: all debug prof clean
all: $(BIN)

$(BIN): $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
        ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(MPICXX) $(CXXFLAGS_RELEASE) -o $@ $(SRC)

debug: $(SRC)
	$(MPICXX) $(CXXFLAGS_DEBUG) -o $(BIN)_debug $(SRC)

prof: $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
      ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(MPICXX) $(CXXFLAGS_PROF) -o $(BIN)_prof $(SRC)

clean:
	rm -f $(BIN) $(BIN)_debug $(BIN)_prof
