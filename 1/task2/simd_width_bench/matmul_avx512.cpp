// matmul_avx512.cpp  512-bit SIMD variant (AVX-512F, 16 floats/vector). Ungraded: report
// plot 4 only. Compiles on any machine but will SIGILL if executed on hardware without
// AVX-512F; callers must guard with __builtin_cpu_supports("avx512f"). Compiled with
// -mavx512f.

#include <immintrin.h>

#include "matmul_variants.h"

void matmul_simd_avx512(const float* A, const float* B, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m512 acc_md = _mm512_setzero_ps();
            int p = 0;
            for (; p + 16 <= K; p += 16) {
                __m512 a_md = _mm512_loadu_ps(&a[p]);
                __m512 b_md = _mm512_loadu_ps(&b[p]);
                acc_md = _mm512_fmadd_ps(a_md, b_md, acc_md);
            }
            float acc = _mm512_reduce_add_ps(acc_md);
            for (; p < K; ++p) acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
