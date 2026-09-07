// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled
    // implementation. conv_naive(in, out, ker, H, W, K);
    
    const int p = K / 2;
    const int PW = W + 2 * p;

    for (int i = 0; i < H * W; i++) out[i] = 0.0f;

    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            const float k = ker[ky * K + kx];
            for (int oy = 0; oy < H; oy++) {
                float* o = out + oy * W;
                const float* i = in + (oy + ky) * PW + kx;
                for (int ox = 0; ox < W; ox += 8) {
                    o[ox + 0] += i[ox + 0] * k;
                    o[ox + 1] += i[ox + 1] * k;
                    o[ox + 2] += i[ox + 2] * k;
                    o[ox + 3] += i[ox + 3] * k;
                    o[ox + 4] += i[ox + 4] * k;
                    o[ox + 5] += i[ox + 5] * k;
                    o[ox + 6] += i[ox + 6] * k;
                    o[ox + 7] += i[ox + 7] * k;
                }
            }
        }
    }
}
