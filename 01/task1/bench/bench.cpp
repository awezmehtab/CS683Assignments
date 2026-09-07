// runs one kernel in a loop with no correctness check, so perf counts
// belong to that kernel alone. report only, not part of the graded build.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "convolution.h"
#include "utils.h"
#include <chrono>

void conv_tile_param(const float*, float*, const float*, int, int, int);
void conv_simd128(const float*, float*, const float*, int, int, int);
void conv_simd_notile(const float*, float*, const float*, int, int, int);
void conv_simd_1ch(const float*, float*, const float*, int, int, int);

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s stage H W K reps\n", argv[0]);
        return 2;
    }
    const std::string stage = argv[1];
    const int H = std::atoi(argv[2]), W = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]), reps = std::atoi(argv[5]);

    float* img = pa1::alloc_floats((size_t)H * W);
    pa1::fill_random(img, (size_t)H * W, 1234u);
    float* in = pa1::make_padded(img, H, W, K);
    float* ker = pa1::alloc_floats((size_t)K * K);
    pa1::fill_random(ker, (size_t)K * K, 99u);
    float* out = pa1::alloc_floats((size_t)H * W);

    ConvFn fn = nullptr;
    if      (stage == "naive")     fn = conv_naive;
    else if (stage == "reorder")   fn = conv_reorder;
    else if (stage == "unroll")    fn = conv_unroll;
    else if (stage == "tile")      fn = conv_tile_param;
    else if (stage == "simd")      fn = conv_simd;
    else if (stage == "simd128")   fn = conv_simd128;
    else if (stage == "simd_nt")   fn = conv_simd_notile;
    else if (stage == "simd_1ch")  fn = conv_simd_1ch;
    else if (stage == "optimized") fn = conv_optimized;
    else { std::fprintf(stderr, "unknown stage %s\n", stage.c_str()); return 2; }

    fn(in, out, ker, H, W, K);  // warm up

    if (argc > 6 && std::string(argv[6]) == "check") {
        float* ref = pa1::alloc_floats((size_t)H * W);
        conv_naive(in, ref, ker, H, W, K);
        std::printf("maxdiff %.3e\n", pa1::max_abs_diff(out, ref, H, W));
        pa1::free_floats(ref);
    }

    using clock = std::chrono::steady_clock;
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clock::now();
        fn(in, out, ker, H, W, K);
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    std::printf("%.4f\n", best);

    pa1::free_floats(img); pa1::free_floats(in);
    pa1::free_floats(ker); pa1::free_floats(out);
    return 0;
}
