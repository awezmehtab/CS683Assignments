#include <algorithm>
#include <immintrin.h>

#include "matmul.h"

#ifndef BLOCK_M
#define BLOCK_M 64                          // rows of A/C per block
#endif
#ifndef BLOCK_N
#define BLOCK_N 64                          // rows of B / cols of C per block
#endif
#ifndef BLOCK_K
#define BLOCK_K 256                         // length of the K-slice per block
#endif

static constexpr int kBlockM = BLOCK_M;
static constexpr int kBlockN = BLOCK_N;
static constexpr int kBlockK = BLOCK_K;
static constexpr int kLine = 16;            // floats per 64B cache line — prefetch granularity

#ifndef PREFETCH_DIST
#define PREFETCH_DIST 64                    // floats ahead to prefetch
#endif
static constexpr int kPrefetchDist = PREFETCH_DIST;

// PREFETCH_DIST=0 is the "no software prefetch" baseline the speedups in 2A are
// measured against, so at 0 the prefetches have to compile away completely, not
// just become _mm_prefetch(&a[p + 0]).
#if PREFETCH_DIST > 0
#define PF_A(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0)
#define PF_B(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1)
#else
#define PF_A(ptr) ((void)0)
#define PF_B(ptr) ((void)0)
#endif

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // K-blocking splits each dot product into partial sums, so C can no longer be written
    // once — it has to be accumulated across K-blocks, which means starting from zero.
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[static_cast<long>(i) * ldc + j] = 0.0f;

    for (int kk = 0; kk < K; kk += kBlockK) {
        const int k_end = std::min(kk + kBlockK, K);
        for (int jj = 0; jj < N; jj += kBlockN) {
            const int j_end = std::min(jj + kBlockN, N);
            for (int ii = 0; ii < M; ii += kBlockM) {
                const int i_end = std::min(ii + kBlockM, M);

                for (int i = ii; i < i_end; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;

                    for (int j = jj; j < j_end; ++j) {
                        const float* b = B + static_cast<long>(j) * ldb;

                        float acc = 0.0f;
                        int p = kk;
                        for (; p + kLine <= k_end; p += kLine) {
                            PF_A(&a[p + kPrefetchDist]);
                            PF_B(&b[p + kPrefetchDist]);
                            for (int q = 0; q < kLine; ++q) acc += a[p + q] * b[p + q];
                        }
                        for (; p < k_end; ++p) acc += a[p] * b[p];
                        C[static_cast<long>(i) * ldc + j] += acc;
                    }
                }
            }
        }
    }
}
