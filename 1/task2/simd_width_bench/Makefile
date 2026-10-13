# builds the SIMD width comparison binary for 2B.
# matmul_avx512.cpp needs -mavx512f just to compile. this CPU has no AVX-512,
# so bench.cpp checks at runtime and never calls it, it's only here so the
# 512-bit variant is reported as unsupported rather than missing.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall
OUT      := ../bin/simd_width_bench

OBJ := bench.o matmul_sse.o matmul_avx2.o matmul_avx512.o ../src/matmul_naive.o

.PHONY: all clean
all: $(OUT)

matmul_avx512.o: matmul_avx512.cpp
	$(CXX) $(CXXFLAGS) -mavx512f -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	@mkdir -p $(dir $(OUT))
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(OUT)

clean:
	rm -f $(OBJ) $(OUT)
