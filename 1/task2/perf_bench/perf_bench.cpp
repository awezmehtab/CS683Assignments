// perf_bench.cpp  single-stage isolate driver for the Task 2 experiments.  UNGRADED.
//
// The graded harness (src/main.cpp) always runs matmul_naive AND the stage you asked for
// in the same process, because it needs the naive result as a correctness reference and as
// the speedup baseline. That is exactly wrong for `perf stat`: the counters would include
// the naive run, which for a 1024^3 workload dwarfs everything the optimized kernel does.
//
// This driver runs ONE stage, `reps` times, and nothing else. Two extra properties make it
// usable as a perf target:
//
//   1. --reps=0 does the full setup (allocate, fill, zero) and then exits without calling
//      any kernel. Running the same command twice, once with --reps=R and once with
//      --reps=0, and subtracting the counters removes allocation/page-fault/RNG noise from
//      the measurement. scripts/common.sh::perf_delta does exactly that.
//
//   2. --verify is OFF by default. Verification needs a matmul_naive reference run, which
//      would land in the counters. Correctness is instead checked once per build config by
//      the scripts, at a small size, via a separate --verify invocation.
//
// Usage:
//   ./bin/perf_bench --stage=optimized --size=1024 --reps=3
//   ./bin/perf_bench --stage=optimized --size=1024 --reps=0          # setup-only baseline
//   ./bin/perf_bench --stage=optimized --size=257  --reps=1 --verify # correctness check
//   ./bin/perf_bench --stage=prefetch --M=512 --N=1024 --K=768 --reps=3
//
// Output is one machine-readable key=value line on stdout, e.g.
//   stage=optimized M=1024 N=1024 K=1024 reps=3 time_ms=12.345 gflops=173.98 correct=n/a

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

namespace {

constexpr float kRelTol = 1e-4f;
constexpr unsigned kDefSeed = 1234u;

struct Stage {
    const char* key;
    MatMulFn fn;
};

const Stage kStages[] = {
    {"naive", matmul_naive},
    {"simd", matmul_simd},
    {"prefetch", matmul_prefetch},
    {"optimized", matmul_optimized},
};
constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

MatMulFn find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(kStages[i].key, key) == 0) return kStages[i].fn;
    return nullptr;
}

bool starts_with(const char* s, const char* prefix) {
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --stage=<naive|simd|prefetch|optimized>\n"
                 "          (--size=N | --M=.. --N=.. --K=..)\n"
                 "          [--reps=R] [--warmup=W] [--seed=S] [--verify]\n"
                 "\n"
                 "  --reps=0   setup only, no kernel call (counter baseline to subtract)\n"
                 "  --verify   also run matmul_naive and compare (pollutes perf counters,\n"
                 "             so use it in a separate run from the measured one)\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    const char* stage_key = nullptr;
    int size = 0, M = 0, N = 0, K = 0;
    int reps = 3, warmup = 1;
    unsigned seed = kDefSeed;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (starts_with(a, "--stage=")) {
            stage_key = a + 8;
        } else if (starts_with(a, "--size=")) {
            size = std::atoi(a + 7);
        } else if (starts_with(a, "--M=")) {
            M = std::atoi(a + 4);
        } else if (starts_with(a, "--N=")) {
            N = std::atoi(a + 4);
        } else if (starts_with(a, "--K=")) {
            K = std::atoi(a + 4);
        } else if (starts_with(a, "--reps=")) {
            reps = std::atoi(a + 7);
        } else if (starts_with(a, "--warmup=")) {
            warmup = std::atoi(a + 9);
        } else if (starts_with(a, "--seed=")) {
            seed = static_cast<unsigned>(std::strtoul(a + 7, nullptr, 10));
        } else if (std::strcmp(a, "--verify") == 0) {
            verify = true;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a);
            usage(argv[0]);
            return 1;
        }
    }

    if (!stage_key) {
        std::fprintf(stderr, "error: --stage is required\n");
        usage(argv[0]);
        return 1;
    }
    MatMulFn fn = find_stage(stage_key);
    if (!fn) {
        std::fprintf(stderr, "error: unknown stage '%s'\n", stage_key);
        usage(argv[0]);
        return 1;
    }

    if (size > 0) {
        M = N = K = size;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "error: give --size=N, or all three of --M --N --K\n");
        usage(argv[0]);
        return 1;
    }
    if (reps < 0 || warmup < 0) {
        std::fprintf(stderr, "error: --reps and --warmup must be >= 0\n");
        return 1;
    }

    const int lda = K, ldb = K, ldc = N;

    // ---- setup: identical whether reps>0 or reps==0, so the subtraction is valid -------
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    if (!A || !B || !C) {
        std::fprintf(stderr, "error: allocation failed for M=%d N=%d K=%d\n", M, N, K);
        return 1;
    }
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);
    // Touch C so its pages are already faulted in before any timed region.
    for (std::size_t i = 0; i < static_cast<std::size_t>(M) * N; ++i) C[i] = 0.0f;

    const double flops = pa1::matmul_flops(M, N, K);
    double ms = 0.0;
    const char* correct = "n/a";

    // ---- the measured region ------------------------------------------------
    if (reps > 0) {
        auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
        ms = pa1::time_median_ms(run, warmup, reps);
    }

    // ---- optional correctness check (deliberately AFTER the timed region) ----
    if (verify) {
        float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
        if (!ref) {
            std::fprintf(stderr, "error: allocation failed for reference buffer\n");
            return 1;
        }
        // Poison C so a kernel that forgets to zero before accumulating cannot pass by luck.
        for (std::size_t i = 0; i < static_cast<std::size_t>(M) * N; ++i) C[i] = 12345.0f;
        fn(A, B, C, M, N, K, lda, ldb, ldc);
        matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
        const float tol = kRelTol * (pa1::max_abs(ref, static_cast<std::size_t>(M) * N) + 1e-30f);
        const bool ok = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol;
        correct = ok ? "yes" : "NO";
        pa1::free_floats(ref);
    }

    std::printf("stage=%s M=%d N=%d K=%d reps=%d warmup=%d time_ms=%.6f gflops=%.4f "
                "correct=%s\n",
                stage_key, M, N, K, reps, warmup, ms,
                (ms > 0.0) ? flops / (ms * 1e6) : 0.0, correct);

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    return (std::strcmp(correct, "NO") == 0) ? 2 : 0;
}
