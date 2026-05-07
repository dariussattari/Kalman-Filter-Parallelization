# Build the OpenMP Kalman-filter driver.
#   make            # release build (default)
#   make debug      # -O0 -g for gdb / sanitizers
#   make clean

CXX      ?= g++
CXXSTD   ?= -std=c++17
WARN     ?= -Wall -Wextra -Wpedantic
OPT      ?= -O3 -march=native -ffast-math -fopenmp
INC      ?= -I../common

CXXFLAGS_RELEASE := $(CXXSTD) $(WARN) $(OPT) $(INC)
CXXFLAGS_DEBUG   := $(CXXSTD) $(WARN) -O0 -g -fopenmp -fsanitize=address,undefined $(INC)
# Same optimised build but with -g so VTune can attribute samples to source lines.
CXXFLAGS_PROF    := $(CXXSTD) $(WARN) $(OPT) -g -fno-omit-frame-pointer $(INC)

SRC := ekf_omp.cpp
BIN := ekf_omp

.PHONY: all debug prof clean
all: $(BIN)

$(BIN): $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
        ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $(SRC)

debug: $(SRC)
	$(CXX) $(CXXFLAGS_DEBUG) -o $(BIN)_debug $(SRC)

prof: $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
      ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(CXX) $(CXXFLAGS_PROF) -o $(BIN)_prof $(SRC)

clean:
	rm -f $(BIN) $(BIN)_debug $(BIN)_prof
