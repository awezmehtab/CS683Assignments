// matmul_sse.cpp  128-bit SIMD variant (SSE4.2, 4 floats/vector). Ungraded: report plot 4 only.
// Compiled with -msse4.2 (no AVX/FMA) so the codegen is genuinely 128-bit.

#include <immintrin.h>

#include "matmul_variants.h"

static inline float hsum128(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // duplicate odd lanes (SSE3)
    __m128 sums = _mm_add_ps(v, shuf);       // 2 partial sums (lanes 0, 2)
    shuf = _mm_movehl_ps(shuf, sums);        // move lane 2 down to lane 0
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_simd_sse(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m128 acc_md = _mm_setzero_ps();
            int p = 0;
            for (; p + 4 <= K; p += 4) {
                __m128 a_md = _mm_loadu_ps(&a[p]);
                __m128 b_md = _mm_loadu_ps(&b[p]);
                acc_md = _mm_add_ps(acc_md, _mm_mul_ps(a_md, b_md));
            }
            float acc = hsum128(acc_md);
            for (; p < K; ++p) acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
