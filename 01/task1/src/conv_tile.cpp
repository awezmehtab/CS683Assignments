#include "convolution.h"

namespace {
constexpr int kTileH = 64;
constexpr int kTileW = 256; // due to the unroll factor
}  // namespace

void conv_tile(const float* in, float* out, const float* ker, int H, int W,
               int K) {
    const int p = K / 2;
    const int PW = W + 2 * p;

    // tx, ty is one corner of tile in out
    for (int ty = 0; ty < H; ty += kTileH) {
        const int y_end = (ty + kTileH < H) ? ty + kTileH : H;

        for (int tx = 0; tx < W; tx += kTileW) {
            const int x_end = (tx + kTileW < W) ? tx + kTileW : W;

            for (int oy = ty; oy < y_end; ++oy) {
                float* op = out + oy * W;
                for (int ox = tx; ox < x_end; ++ox) op[ox] = 0.0f;
            }

            // this is similar to the previous loops, but it's for this
            // tile so it fully fits in L2 cache
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float w = ker[ky * K + kx];

                    for (int oy = ty; oy < y_end; ++oy) {
                        const float* ip = in + (oy + ky) * PW + kx;
                        float* op = out + oy * W;

                        for (int ox = tx; ox < x_end; ox += 4) {
                            op[ox + 0] += ip[ox + 0] * w;
                            op[ox + 1] += ip[ox + 1] * w;
                            op[ox + 2] += ip[ox + 2] * w;
                            op[ox + 3] += ip[ox + 3] * w;
                        }
                    }
                }
            }
        }
    }
}
