// matmul_variants.h  ungraded SIMD-width comparison kernels (report plot 4 only)
//
// These are NOT part of the graded submission (src/matmul_simd.cpp stays AVX2-only, built
// with the pinned flags). Each variant below is compiled from its own .cpp with the ISA
// flags that variant needs (-msse4.2, -mavx2 -mfma, -mavx512f) so they can coexist in one
// benchmark binary; see simd_width_bench/Makefile.

#ifndef CS683_PA1_SIMD_WIDTH_VARIANTS_H
#define CS683_PA1_SIMD_WIDTH_VARIANTS_H

// 128-bit (SSE, 4 floats/vector).
void matmul_simd_sse(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc);

// 256-bit (AVX2 + FMA, 8 floats/vector) — same technique as src/matmul_simd.cpp.
void matmul_simd_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc);

// 512-bit (AVX-512F, 16 floats/vector). Only call this if __builtin_cpu_supports("avx512f")
// — it compiles on any machine but SIGILLs on hardware lacking AVX-512.
void matmul_simd_avx512(const float* A, const float* B, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc);

#endif  // CS683_PA1_SIMD_WIDTH_VARIANTS_H
