// matmul_avx2.cpp  256-bit SIMD variant (AVX2 + FMA, 8 floats/vector). Ungraded: report
// plot 4 only. Same technique as src/matmul_simd.cpp, just renamed to coexist in this
// benchmark alongside the SSE and AVX-512 variants. Compiled with -mavx2 -mfma.

#include <immintrin.h>

#include "matmul_variants.h"

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                 // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);       // duplicate odd lanes
    __m128 sums = _mm_add_ps(lo, shuf);      // 2 partial sums (lanes 0, 2)
    shuf = _mm_movehl_ps(shuf, sums);        // move lane 2 down to lane 0
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_simd_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc_md = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8) {
                __m256 a_md = _mm256_loadu_ps(&a[p]);
                __m256 b_md = _mm256_loadu_ps(&b[p]);
                acc_md = _mm256_fmadd_ps(a_md, b_md, acc_md);
            }
            float acc = hsum256(acc_md);
            for (; p < K; ++p) acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
