// conv_simd.cpp without the tiling, so 1D can measure SIMD on its own.
// report only.

#include <immintrin.h>

#include "convolution.h"

void conv_simd_notile(const float* in, float* out, const float* ker,
                      int H, int W, int K) {
    const int p  = K / 2;
    const int PW = W + 2 * p;

    const __m256 vzero = _mm256_setzero_ps();
    for (int oy = 0; oy < H; ++oy) {
        float* op = out + oy * W;
        for (int ox = 0; ox < W; ox += 8) _mm256_store_ps(op + ox, vzero);
    }

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const __m256 vw = _mm256_set1_ps(ker[ky * K + kx]);

            for (int oy = 0; oy < H; ++oy) {
                const float* ip = in + (oy + ky) * PW + kx;
                float*       op = out + oy * W;

                int ox = 0;
                for (; ox + 16 <= W; ox += 16) {
                    __m256 a0 = _mm256_load_ps(op + ox);
                    __m256 a1 = _mm256_load_ps(op + ox + 8);
                    const __m256 p0 = _mm256_loadu_ps(ip + ox);
                    const __m256 p1 = _mm256_loadu_ps(ip + ox + 8);
                    a0 = _mm256_fmadd_ps(p0, vw, a0);
                    a1 = _mm256_fmadd_ps(p1, vw, a1);
                    _mm256_store_ps(op + ox, a0);
                    _mm256_store_ps(op + ox + 8, a1);
                }
                for (; ox < W; ox += 8) {
                    __m256 a0 = _mm256_load_ps(op + ox);
                    const __m256 p0 = _mm256_loadu_ps(ip + ox);
                    a0 = _mm256_fmadd_ps(p0, vw, a0);
                    _mm256_store_ps(op + ox, a0);
                }
            }
        }
    }
}
