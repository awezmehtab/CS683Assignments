// 128-bit version of conv_simd.cpp, for the width comparison in 1C.
// same structure, half the vector width. report only.

#include <immintrin.h>

#include "convolution.h"

namespace {
constexpr int kTileH = 64;
constexpr int kTileW = 256;
}  // namespace

void conv_simd128(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p  = K / 2;
    const int PW = W + 2 * p;

    const __m128 vzero = _mm_setzero_ps();

    for (int ty = 0; ty < H; ty += kTileH) {
        const int y_end = (ty + kTileH < H) ? ty + kTileH : H;

        for (int tx = 0; tx < W; tx += kTileW) {
            const int x_end = (tx + kTileW < W) ? tx + kTileW : W;

            for (int oy = ty; oy < y_end; ++oy) {
                float* op = out + oy * W;
                for (int ox = tx; ox < x_end; ox += 4) {
                    _mm_store_ps(op + ox, vzero);
                }
            }

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const __m128 vw = _mm_set1_ps(ker[ky * K + kx]);

                    for (int oy = ty; oy < y_end; ++oy) {
                        const float* ip = in + (oy + ky) * PW + kx;
                        float*       op = out + oy * W;

                        int ox = tx;
                        for (; ox + 8 <= x_end; ox += 8) {
                            __m128 a0 = _mm_load_ps(op + ox);
                            __m128 a1 = _mm_load_ps(op + ox + 4);
                            const __m128 p0 = _mm_loadu_ps(ip + ox);
                            const __m128 p1 = _mm_loadu_ps(ip + ox + 4);
                            a0 = _mm_fmadd_ps(p0, vw, a0);
                            a1 = _mm_fmadd_ps(p1, vw, a1);
                            _mm_store_ps(op + ox, a0);
                            _mm_store_ps(op + ox + 4, a1);
                        }
                        for (; ox < x_end; ox += 4) {
                            __m128 a0 = _mm_load_ps(op + ox);
                            const __m128 p0 = _mm_loadu_ps(ip + ox);
                            a0 = _mm_fmadd_ps(p0, vw, a0);
                            _mm_store_ps(op + ox, a0);
                        }
                    }
                }
            }
        }
    }
}
