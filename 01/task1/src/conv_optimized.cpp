#include <immintrin.h>

#include "convolution.h"

namespace {

constexpr int kUnrollCols = 32;  // 4 vectors per iteration
constexpr int kPfDist = 1024;    // prefetch lookahead, in floats

// specialised for k = 3, all nine weights stay in registers
void conv3(const float* __restrict in, float* __restrict out,
           const float* __restrict ker, int H, int W) {
    const int PW = W + 2;  // p = 1

    const __m256 w00 = _mm256_set1_ps(ker[0]);
    const __m256 w01 = _mm256_set1_ps(ker[1]);
    const __m256 w02 = _mm256_set1_ps(ker[2]);
    const __m256 w10 = _mm256_set1_ps(ker[3]);
    const __m256 w11 = _mm256_set1_ps(ker[4]);
    const __m256 w12 = _mm256_set1_ps(ker[5]);
    const __m256 w20 = _mm256_set1_ps(ker[6]);
    const __m256 w21 = _mm256_set1_ps(ker[7]);
    const __m256 w22 = _mm256_set1_ps(ker[8]);

    for (int oy = 0; oy < H; ++oy) {
        const float* r0 = in + static_cast<long>(oy) * PW;
        const float* r1 = r0 + PW;
        const float* r2 = r1 + PW;
        float* op = out + static_cast<long>(oy) * W;

        int ox = 0;
        for (; ox + kUnrollCols <= W; ox += kUnrollCols) {
            // 32 floats is 2 cache lines, so we need 2 prefetches
            _mm_prefetch(reinterpret_cast<const char*>(r2 + ox + kPfDist),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(r2 + ox + kPfDist + 16),
                         _MM_HINT_T0);

            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(r0 + ox + 0), w00);
            __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(r0 + ox + 8), w00);
            __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(r0 + ox + 16), w00);
            __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(r0 + ox + 24), w00);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 1), w01, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 9), w01, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 17), w01, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 25), w01, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 2), w02, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 10), w02, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 18), w02, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 26), w02, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 0), w10, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 8), w10, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 16), w10, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 24), w10, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 1), w11, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 9), w11, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 17), w11, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 25), w11, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 2), w12, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 10), w12, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 18), w12, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 26), w12, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 0), w20, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 8), w20, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 16), w20, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 24), w20, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 1), w21, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 9), w21, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 17), w21, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 25), w21, a3);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 2), w22, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 10), w22, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 18), w22, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 26), w22, a3);

            _mm256_stream_ps(op + ox + 0, a0);
            _mm256_stream_ps(op + ox + 8, a1);
            _mm256_stream_ps(op + ox + 16, a2);
            _mm256_stream_ps(op + ox + 24, a3);
        }

        // W is a multiple of 8, so the tail is whole vectors
        for (; ox < W; ox += 8) {
            __m256 a = _mm256_mul_ps(_mm256_loadu_ps(r0 + ox + 0), w00);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 1), w01, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + ox + 2), w02, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 0), w10, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 1), w11, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + ox + 2), w12, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 0), w20, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 1), w21, a);
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + ox + 2), w22, a);
            _mm256_stream_ps(op + ox, a);
        }
    }
}

void convK(const float* __restrict in, float* __restrict out,
           const float* __restrict ker, int H, int W, int K) {
    const int p = K / 2;
    const int PW = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        const float* rbase = in + static_cast<long>(oy) * PW;
        float* op = out + static_cast<long>(oy) * W;

        int ox = 0;
        for (; ox + kUnrollCols <= W; ox += kUnrollCols) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* ip = rbase + static_cast<long>(ky) * PW + ox;
                _mm_prefetch(reinterpret_cast<const char*>(ip + kPfDist),
                             _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(ip + kPfDist + 16),
                             _MM_HINT_T0);
                const float* kp = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const __m256 vw = _mm256_set1_ps(kp[kx]);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip + kx + 0), vw, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ip + kx + 8), vw, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(ip + kx + 16), vw, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(ip + kx + 24), vw, a3);
                }
            }

            _mm256_stream_ps(op + ox + 0, a0);
            _mm256_stream_ps(op + ox + 8, a1);
            _mm256_stream_ps(op + ox + 16, a2);
            _mm256_stream_ps(op + ox + 24, a3);
        }

        for (; ox < W; ox += 8) {
            __m256 a = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* ip = rbase + static_cast<long>(ky) * PW + ox;
                const float* kp = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(ip + kx),
                                        _mm256_set1_ps(kp[kx]), a);
                }
            }
            _mm256_stream_ps(op + ox, a);
        }
    }
}

}  // namespace

void conv_optimized(const float* in, float* out, const float* ker, int H, int W,
                    int K) {
    if (K == 3) {
        conv3(in, out, ker, H, W);
    } else {
        convK(in, out, ker, H, W, K);
    }
    // the stream stores are weakly ordered, so fence before the caller reads
    _mm_sfence();
}
