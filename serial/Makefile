# Build the serial Kalman-filter baseline.
# Usage:
#   make            # release build (default)
#   make debug      # -O0 -g for gdb / sanitizers
#   make clean

CXX      ?= g++
CXXSTD   ?= -std=c++17
WARN     ?= -Wall -Wextra -Wpedantic
OPT      ?= -O3 -march=native -ffast-math
INC      ?= -I../common

CXXFLAGS_RELEASE := $(CXXSTD) $(WARN) $(OPT) $(INC)
CXXFLAGS_DEBUG   := $(CXXSTD) $(WARN) -O0 -g -fsanitize=address,undefined $(INC)

SRC := ekf_serial.cpp
BIN := ekf_serial

.PHONY: all debug clean
all: $(BIN)

$(BIN): $(SRC) ../common/ekf_math.hpp ../common/ekf_config.hpp \
        ../common/io.hpp ../common/timing.hpp ../common/checksum.hpp
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $(SRC)

debug: $(SRC)
	$(CXX) $(CXXFLAGS_DEBUG) -o $(BIN)_debug $(SRC)

clean:
	rm -f $(BIN) $(BIN)_debug
