# builds one perf_bench variant. the scripts call this as
#   make -C perf_bench EXTRA="-DBLOCK_M=64 ..." OUT=../bin/pb_something
# EXTRA carries the -D knobs for the configuration being measured, OUT is
# where the binary goes. every config is a full rebuild, since the -D flags
# change what the sources mean.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall
EXTRA    ?=
OUT      ?= ../bin/perf_bench

SRC := perf_bench.cpp \
       ../src/matmul_naive.cpp ../src/matmul_simd.cpp \
       ../src/matmul_prefetch.cpp ../src/matmul_optimized.cpp

.PHONY: all clean
all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p $(dir $(OUT))
	$(CXX) $(CXXFLAGS) $(EXTRA) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
