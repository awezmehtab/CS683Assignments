// bench.cpp  standalone driver for report plot 4: "speedup vs. SIMD width".
//
// Ungraded. Compares matmul_naive against the 128/256/512-bit variants in this directory
// across a range of matrix sizes and writes a CSV for plotting.
//
//     ./bin/simd_width_bench                 # default size sweep
//     ./bin/simd_width_bench 128 256 512 1024 2048 4096   # custom sizes (M=N=K each)
//
// Isolate mode (for `perf stat -e instructions`, so the count reflects ~one kernel, not
// the whole sweep):
//
//     ./bin/simd_width_bench --only=avx2_256 --size=1024 [--reps=3]
//     perf stat -e instructions ./bin/simd_width_bench --only=avx2_256 --size=1024

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "matmul.h"
#include "matmul_variants.h"
#include "timer.h"
#include "utils.h"

namespace {

constexpr float kRelTol = 1e-4f;
constexpr int kWarmup = 1;
constexpr int kReps = 3;
constexpr unsigned kSeed = 1234u;

struct Variant {
    const char* name;
    MatMulFn fn;
    bool available;
};

bool check_and_time(MatMulFn fn, const float* A, const float* B, float* C, const float* ref,
                    int M, int N, int K, int lda, int ldb, int ldc, float tol, double* ms_out) {
    fn(A, B, C, M, N, K, lda, ldb, ldc);
    bool ok = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol;
    auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
    *ms_out = pa1::time_median_ms(run, kWarmup, kReps);
    return ok;
}

// Match a --only= name against a known variant (nullptr if none, e.g. "naive").
const Variant* find_variant(const char* name, bool avx512_ok) {
    static const Variant variants[] = {
        {"sse128", matmul_simd_sse, true},
        {"avx2_256", matmul_simd_avx2, true},
        {"avx512", matmul_simd_avx512, avx512_ok},
    };
    for (const Variant& v : variants)
        if (std::strcmp(v.name, name) == 0) return &v;
    return nullptr;
}

bool starts_with(const char* s, const char* prefix) {
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

// Isolate mode: run ONE variant on ONE size, `reps` times, nothing else timed or printed
// beyond a one-line summary. Meant to be wrapped in `perf stat -e instructions`.
int run_isolated(const char* only, int size, int reps) {
    const bool avx512_ok = __builtin_cpu_supports("avx512f");

    MatMulFn fn = nullptr;
    if (std::strcmp(only, "naive") == 0) {
        fn = matmul_naive;
    } else {
        const Variant* v = find_variant(only, avx512_ok);
        if (!v) {
            std::fprintf(stderr, "error: unknown --only=%s (naive|sse128|avx2_256|avx512)\n", only);
            return 1;
        }
        if (!v->available) {
            std::printf("size=%d variant=%s skipped=unsupported_on_this_cpu\n", size, only);
            return 0;
        }
        fn = v->fn;
    }

    const int M = size, N = size, K = size, lda = K, ldb = K, ldc = N;
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, kSeed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, kSeed + 1u);

    matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
    const float tol = kRelTol * (pa1::max_abs(ref, static_cast<std::size_t>(M) * N) + 1e-30f);

    bool ok = true;
    for (int r = 0; r < reps; ++r) {
        fn(A, B, C, M, N, K, lda, ldb, ldc);
        if (r == 0) ok = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol;
    }

    std::printf("size=%d variant=%s reps=%d correct=%s\n", size, only, reps,
               ok ? "yes" : "NO");

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    pa1::free_floats(ref);
    return ok ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    // --only=<variant> --size=<n> [--reps=<r>]  ->  isolate mode for `perf stat`.
    const char* only = nullptr;
    int size = 0;
    int reps = kReps;
    bool isolate_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (starts_with(argv[i], "--only=")) {
            only = argv[i] + 7;
            isolate_mode = true;
        } else if (starts_with(argv[i], "--size=")) {
            size = std::atoi(argv[i] + 7);
            isolate_mode = true;
        } else if (starts_with(argv[i], "--reps=")) {
            reps = std::atoi(argv[i] + 7);
        }
    }
    if (isolate_mode) {
        if (!only || size <= 0) {
            std::fprintf(stderr, "usage: %s --only=<naive|sse128|avx2_256|avx512> --size=<n> "
                                 "[--reps=<r>]\n", argv[0]);
            return 1;
        }
        return run_isolated(only, size, reps);
    }

    const bool avx512_ok = __builtin_cpu_supports("avx512f");
    std::printf("AVX-512F available on this machine: %s\n\n", avx512_ok ? "yes" : "no");

    std::vector<int> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    } else {
        sizes = {128, 256, 512, 1024, 2048};
    }

    std::FILE* csv = std::fopen("simd_width_results.csv", "w");
    if (csv) std::fprintf(csv, "size,variant,time_ms,gflops,speedup\n");

    std::printf("%-6s  %-8s  %-7s  %10s  %10s  %9s\n", "size", "variant", "correct",
               "time(ms)", "GFLOP/s", "speedup");
    std::printf("--------------------------------------------------------------\n");

    for (int n : sizes) {
        const int M = n, N = n, K = n, lda = K, ldb = K, ldc = N;
        float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
        float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
        float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
        float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
        pa1::fill_random(A, static_cast<std::size_t>(M) * K, kSeed);
        pa1::fill_random(B, static_cast<std::size_t>(N) * K, kSeed + 1u);

        matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
        const float tol = kRelTol * (pa1::max_abs(ref, static_cast<std::size_t>(M) * N) + 1e-30f);
        const double flops = pa1::matmul_flops(M, N, K);

        double naive_ms;
        check_and_time(matmul_naive, A, B, C, ref, M, N, K, lda, ldb, ldc, tol, &naive_ms);
        std::printf("%-6d  %-8s  %-7s  %10.3f  %10.2f  %9s\n", n, "naive", "-", naive_ms,
                   flops / (naive_ms * 1e6), "1.00x");
        if (csv) std::fprintf(csv, "%d,naive,%.6f,%.4f,1.0\n", n, naive_ms, flops / (naive_ms * 1e6));

        const Variant variants[] = {
            {"sse128", matmul_simd_sse, true},
            {"avx2_256", matmul_simd_avx2, true},
            {"avx512", matmul_simd_avx512, avx512_ok},
        };
        for (const Variant& v : variants) {
            if (!v.available) {
                std::printf("%-6d  %-8s  %-7s  %10s  %10s  %9s\n", n, v.name, "-", "n/a", "n/a",
                           "n/a (unsupported on this CPU)");
                continue;
            }
            double ms;
            bool ok = check_and_time(v.fn, A, B, C, ref, M, N, K, lda, ldb, ldc, tol, &ms);
            const double gflops = flops / (ms * 1e6);
            const double speedup = naive_ms / ms;
            std::printf("%-6d  %-8s  %-7s  %10.3f  %10.2f  %8.2fx\n", n, v.name,
                       ok ? "yes" : "NO", ms, gflops, speedup);
            if (csv)
                std::fprintf(csv, "%d,%s,%.6f,%.4f,%.4f\n", n, v.name, ms, gflops, speedup);
        }
        std::printf("\n");

        pa1::free_floats(A);
        pa1::free_floats(B);
        pa1::free_floats(C);
        pa1::free_floats(ref);
    }

    if (csv) {
        std::fclose(csv);
        std::printf("wrote simd_width_results.csv\n");
    }
    return 0;
}
