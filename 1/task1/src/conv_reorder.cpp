// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered
    // implementation. conv_naive(in, out, ker, H, W, K);

    const int p = K / 2;
    const int PW = W + 2 * p;

    for (int i = 0; i < W * H; i++) out[i] = 0.0f;

    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            const float k = ker[ky * K + kx];
            for (int oy = 0; oy < H; oy++) {
                for (int ox = 0; ox < W; ox++) {
                    out[oy * W + ox] += in[(oy + ky) * PW + (ox + kx)] * k;
                }
            }
        }
    }
}
