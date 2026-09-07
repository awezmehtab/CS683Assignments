// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// Build is -O2 -fno-tree-vectorize -mavx2 -mfma (see Makefile), so register allocation is
// real and the micro-kernel is sized to FIT the 16 YMM registers:
//
//  1. 4x3 register tile: 12 accumulators + 4 A vectors, with the 3 B vectors folded into
//     FMA memory operands by the compiler. A 4x4 tile needs 16 acc + 8 operands = 24 live
//     vectors and spills inside the hot loop; 4x3 does not. 12 FMAs per 7 vector loads.
//
//  2. Cache blocking over M, N and K, K-block loop innermost, so each BLOCK_M x BLOCK_N tile
//     of C is zeroed once, accumulated across all K-blocks while cache resident, then never
//     touched again. Large BLOCK_K (default = whole K) also minimises the number of
//     horizontal-sum passes.
//
//  3. Horizontal sums batched four at a time with hadd (5 ops per 4 results instead of
//     ~7 ops per result), since with BLOCK_K = K they are the main non-FMA cost.
//
//  4. Software prefetch inside the kernel, one prefetch per 64B line per B row (T1: land in
//     L2 without evicting the A/C data that is reused from L1). A rows are reused across
//     every j of the i-tile and stay L1-resident, so they are not prefetched by default
//     (enable with -DPREFETCH_A to compare). -DNO_PREFETCH removes all software prefetches
//     for the Task 2A "without prefetching" baseline.
//
//  5. 1x4 fallback kernel for leftover rows so that small-M shapes (M = 1..3, i.e. the
//     token-generation matmuls in llama.cpp) still get 4 FMAs per A vector load.

#include <algorithm>
#include <immintrin.h>

#include "matmul.h"

namespace {

#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 48
#endif
#ifndef BLOCK_K
#define BLOCK_K 1024
#endif
static constexpr int kBlockM = BLOCK_M;
static constexpr int kBlockN = BLOCK_N;
static constexpr int kBlockK = BLOCK_K;

static constexpr int kMR = 4;    // A rows per micro-kernel call (hand-unrolled)
static constexpr int kNR = 3;    // B rows per micro-kernel call (hand-unrolled)

#ifndef PREFETCH_DIST
#define PREFETCH_DIST 64
#endif
static constexpr int kPrefetchDist = PREFETCH_DIST;

#ifndef PREFETCH_HINT_A
#define PREFETCH_HINT_A _MM_HINT_T0
#endif
#ifndef PREFETCH_HINT_B
#define PREFETCH_HINT_B _MM_HINT_T1
#endif

#if PREFETCH_DIST > 0
#define PF_A(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), PREFETCH_HINT_A)
#define PF_B(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), PREFETCH_HINT_B)
#else
#define PF_A(ptr) ((void)0)
#define PF_B(ptr) ((void)0)
#endif

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Returns [sum(v0), sum(v1), sum(v2), sum(v3)] in one __m128.
inline __m128 hsum4(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    __m256 t0 = _mm256_hadd_ps(v0, v1);
    __m256 t1 = _mm256_hadd_ps(v2, v3);
    __m256 t2 = _mm256_hadd_ps(t0, t1);
    return _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
}

// C[i..i+3][j..j+2] += A[i..i+3][k0..k1) . B[j..j+2][k0..k1)
inline void microkernel_4x3(const float* A, const float* B, float* C,
                            int i, int j, int k0, int k1,
                            int lda, int ldb, int ldc) {
    const float* a0 = A + static_cast<long>(i + 0) * lda;
    const float* a1 = A + static_cast<long>(i + 1) * lda;
    const float* a2 = A + static_cast<long>(i + 2) * lda;
    const float* a3 = A + static_cast<long>(i + 3) * lda;
    const float* b0 = B + static_cast<long>(j + 0) * ldb;
    const float* b1 = B + static_cast<long>(j + 1) * ldb;
    const float* b2 = B + static_cast<long>(j + 2) * ldb;

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

    int p = k0;
    // Main loop: 16 floats (one 64B line per row) per iteration, prefetch once per line.
    for (; p + 16 <= k1; p += 16) {
        PF_A(a0 + p + kPrefetchDist);
        PF_A(a1 + p + kPrefetchDist);
        PF_A(a2 + p + kPrefetchDist);
        PF_A(a3 + p + kPrefetchDist);
        PF_B(b0 + p + kPrefetchDist);
        PF_B(b1 + p + kPrefetchDist);
        PF_B(b2 + p + kPrefetchDist);

        __m256 vb0 = _mm256_loadu_ps(b0 + p);
        __m256 vb1 = _mm256_loadu_ps(b1 + p);
        __m256 vb2 = _mm256_loadu_ps(b2 + p);
        __m256 va = _mm256_loadu_ps(a0 + p);
        c00 = _mm256_fmadd_ps(va, vb0, c00); c01 = _mm256_fmadd_ps(va, vb1, c01); c02 = _mm256_fmadd_ps(va, vb2, c02);
        va = _mm256_loadu_ps(a1 + p);
        c10 = _mm256_fmadd_ps(va, vb0, c10); c11 = _mm256_fmadd_ps(va, vb1, c11); c12 = _mm256_fmadd_ps(va, vb2, c12);
        va = _mm256_loadu_ps(a2 + p);
        c20 = _mm256_fmadd_ps(va, vb0, c20); c21 = _mm256_fmadd_ps(va, vb1, c21); c22 = _mm256_fmadd_ps(va, vb2, c22);
        va = _mm256_loadu_ps(a3 + p);
        c30 = _mm256_fmadd_ps(va, vb0, c30); c31 = _mm256_fmadd_ps(va, vb1, c31); c32 = _mm256_fmadd_ps(va, vb2, c32);

        vb0 = _mm256_loadu_ps(b0 + p + 8);
        vb1 = _mm256_loadu_ps(b1 + p + 8);
        vb2 = _mm256_loadu_ps(b2 + p + 8);
        va = _mm256_loadu_ps(a0 + p + 8);
        c00 = _mm256_fmadd_ps(va, vb0, c00); c01 = _mm256_fmadd_ps(va, vb1, c01); c02 = _mm256_fmadd_ps(va, vb2, c02);
        va = _mm256_loadu_ps(a1 + p + 8);
        c10 = _mm256_fmadd_ps(va, vb0, c10); c11 = _mm256_fmadd_ps(va, vb1, c11); c12 = _mm256_fmadd_ps(va, vb2, c12);
        va = _mm256_loadu_ps(a2 + p + 8);
        c20 = _mm256_fmadd_ps(va, vb0, c20); c21 = _mm256_fmadd_ps(va, vb1, c21); c22 = _mm256_fmadd_ps(va, vb2, c22);
        va = _mm256_loadu_ps(a3 + p + 8);
        c30 = _mm256_fmadd_ps(va, vb0, c30); c31 = _mm256_fmadd_ps(va, vb1, c31); c32 = _mm256_fmadd_ps(va, vb2, c32);
    }
    // Leftover whole vector (8..15 floats remain).
    for (; p + 8 <= k1; p += 8) {
        const __m256 vb0 = _mm256_loadu_ps(b0 + p);
        const __m256 vb1 = _mm256_loadu_ps(b1 + p);
        const __m256 vb2 = _mm256_loadu_ps(b2 + p);
        __m256 va = _mm256_loadu_ps(a0 + p);
        c00 = _mm256_fmadd_ps(va, vb0, c00); c01 = _mm256_fmadd_ps(va, vb1, c01); c02 = _mm256_fmadd_ps(va, vb2, c02);
        va = _mm256_loadu_ps(a1 + p);
        c10 = _mm256_fmadd_ps(va, vb0, c10); c11 = _mm256_fmadd_ps(va, vb1, c11); c12 = _mm256_fmadd_ps(va, vb2, c12);
        va = _mm256_loadu_ps(a2 + p);
        c20 = _mm256_fmadd_ps(va, vb0, c20); c21 = _mm256_fmadd_ps(va, vb1, c21); c22 = _mm256_fmadd_ps(va, vb2, c22);
        va = _mm256_loadu_ps(a3 + p);
        c30 = _mm256_fmadd_ps(va, vb0, c30); c31 = _mm256_fmadd_ps(va, vb1, c31); c32 = _mm256_fmadd_ps(va, vb2, c32);
    }

    // Reduce the 12 accumulators in three batches of four: s[r*3 + c] = C[i+r][j+c].
    float s[12];
    _mm_storeu_ps(s + 0, hsum4(c00, c01, c02, c10));
    _mm_storeu_ps(s + 4, hsum4(c11, c12, c20, c21));
    _mm_storeu_ps(s + 8, hsum4(c22, c30, c31, c32));

    for (; p < k1; ++p) {
        const float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
        const float y0 = b0[p], y1 = b1[p], y2 = b2[p];
        s[0] += x0 * y0; s[1]  += x0 * y1; s[2]  += x0 * y2;
        s[3] += x1 * y0; s[4]  += x1 * y1; s[5]  += x1 * y2;
        s[6] += x2 * y0; s[7]  += x2 * y1; s[8]  += x2 * y2;
        s[9] += x3 * y0; s[10] += x3 * y1; s[11] += x3 * y2;
    }

    for (int r = 0; r < 4; ++r) {
        float* crow = C + static_cast<long>(i + r) * ldc + j;
        crow[0] += s[r * 3 + 0];
        crow[1] += s[r * 3 + 1];
        crow[2] += s[r * 3 + 2];
    }
}

// One A row against four B rows: used for leftover rows (M % 4) and small-M shapes.
inline void microkernel_1x4(const float* A, const float* B, float* C,
                            int i, int j, int k0, int k1,
                            int lda, int ldb, int ldc) {
    const float* a  = A + static_cast<long>(i) * lda;
    const float* b0 = B + static_cast<long>(j + 0) * ldb;
    const float* b1 = B + static_cast<long>(j + 1) * ldb;
    const float* b2 = B + static_cast<long>(j + 2) * ldb;
    const float* b3 = B + static_cast<long>(j + 3) * ldb;

    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    int p = k0;
    for (; p + 16 <= k1; p += 16) {
        PF_A(a + p + kPrefetchDist);
        PF_B(b0 + p + kPrefetchDist);
        PF_B(b1 + p + kPrefetchDist);
        PF_B(b2 + p + kPrefetchDist);
        PF_B(b3 + p + kPrefetchDist);
        __m256 va = _mm256_loadu_ps(a + p);
        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), c0);
        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), c1);
        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), c2);
        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), c3);
        va = _mm256_loadu_ps(a + p + 8);
        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p + 8), c0);
        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p + 8), c1);
        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p + 8), c2);
        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p + 8), c3);
    }
    for (; p + 8 <= k1; p += 8) {
        const __m256 va = _mm256_loadu_ps(a + p);
        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), c0);
        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), c1);
        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), c2);
        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), c3);
    }
    float s[4];
    _mm_storeu_ps(s, hsum4(c0, c1, c2, c3));
    for (; p < k1; ++p) {
        s[0] += a[p] * b0[p]; s[1] += a[p] * b1[p];
        s[2] += a[p] * b2[p]; s[3] += a[p] * b3[p];
    }
    float* crow = C + static_cast<long>(i) * ldc + j;
    crow[0] += s[0]; crow[1] += s[1]; crow[2] += s[2]; crow[3] += s[3];
}

// Single output: last resort for the final N % 4 columns of leftover rows.
inline void microkernel_1x1(const float* A, const float* B, float* C,
                            int i, int j, int k0, int k1,
                            int lda, int ldb, int ldc) {
    const float* a = A + static_cast<long>(i) * lda;
    const float* b = B + static_cast<long>(j) * ldb;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int p = k0;
    for (; p + 16 <= k1; p += 16) {
        PF_A(a + p + kPrefetchDist);
        PF_B(b + p + kPrefetchDist);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),     _mm256_loadu_ps(b + p),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8), _mm256_loadu_ps(b + p + 8), acc1);
    }
    for (; p + 8 <= k1; p += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
    }
    float acc = hsum256(_mm256_add_ps(acc0, acc1));
    for (; p < k1; ++p) acc += a[p] * b[p];
    C[static_cast<long>(i) * ldc + j] += acc;
}

// Sweep one leftover A row across [jj, j_end): 1x4 tiles, then 1x1 for the last N % 4.
inline void leftover_row(const float* A, const float* B, float* C,
                         int i, int jj, int j_end, int k0, int k1,
                         int lda, int ldb, int ldc) {
    int j = jj;
    for (; j + 4 <= j_end; j += 4) microkernel_1x4(A, B, C, i, j, k0, k1, lda, ldb, ldc);
    for (; j < j_end; ++j)         microkernel_1x1(A, B, C, i, j, k0, k1, lda, ldb, ldc);
}

}  // namespace

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    for (int jj = 0; jj < N; jj += kBlockN) {
        const int j_end = std::min(jj + kBlockN, N);
        for (int ii = 0; ii < M; ii += kBlockM) {
            const int i_end = std::min(ii + kBlockM, M);

            // Zero only this C tile; it is accumulated across the K-blocks below.
            for (int i = ii; i < i_end; ++i) {
                float* crow = C + static_cast<long>(i) * ldc;
                for (int j = jj; j < j_end; ++j) crow[j] = 0.0f;
            }

            for (int kk = 0; kk < K; kk += kBlockK) {
                const int k_end = std::min(kk + kBlockK, K);

                int i = ii;
                for (; i + kMR <= i_end; i += kMR) {
                    int j = jj;
                    for (; j + kNR <= j_end; j += kNR) {
                        microkernel_4x3(A, B, C, i, j, kk, k_end, lda, ldb, ldc);
                    }
                    // Trailing N % 3 columns of this 4-row tile.
                    for (; j < j_end; ++j) {
                        for (int r = 0; r < kMR; ++r) {
                            microkernel_1x1(A, B, C, i + r, j, kk, k_end, lda, ldb, ldc);
                        }
                    }
                }
                // Trailing M % 4 rows (and the whole matrix when M < 4).
                for (; i < i_end; ++i) {
                    leftover_row(A, B, C, i, jj, j_end, kk, k_end, lda, ldb, ldc);
                }
            }
        }
    }
}
