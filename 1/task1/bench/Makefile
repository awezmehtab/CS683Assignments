# report-only build, same flags as the graded Makefile
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall
TILE_H   ?= 64
TILE_W   ?= 256
SRC := bench.cpp conv_tile_param.cpp conv_simd128.cpp conv_simd_notile.cpp conv_simd_1ch.cpp \
       ../src/conv_naive.cpp ../src/conv_reorder.cpp ../src/conv_unroll.cpp \
       ../src/conv_simd.cpp ../src/conv_optimized.cpp

bench: $(SRC)
	$(CXX) $(CXXFLAGS) -DTILE_H=$(TILE_H) -DTILE_W=$(TILE_W) $^ -o bench
