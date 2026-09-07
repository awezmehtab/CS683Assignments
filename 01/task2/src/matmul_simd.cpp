#include <immintrin.h>

#include "matmul.h"

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                    // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);          // duplicate odd lanes
    __m128 sums = _mm_add_ps(lo, shuf);         // 2 partial sums (in lanes 0,2)
    shuf = _mm_movehl_ps(shuf, sums);           // move lane 2 down to lane 0
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}


void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            // Multiple accumulators: FOUR independent FMA chains.
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;
            for (; p + 32 <= K; p += 32) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p + 0]),
                                       _mm256_loadu_ps(&b[p + 0]), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p + 8]),
                                       _mm256_loadu_ps(&b[p + 8]), acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p + 16]),
                                       _mm256_loadu_ps(&b[p + 16]), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p + 24]),
                                       _mm256_loadu_ps(&b[p + 24]), acc3);
            }

            // Fold the four chains together, then mop up whole vectors, then scalars.
            __m256 total = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                         _mm256_add_ps(acc2, acc3));
            for (; p + 8 <= K; p += 8) {
                total = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p]),
                                        _mm256_loadu_ps(&b[p]), total);
            }

            float acc = hsum256(total);
            for (; p < K; ++p) acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
