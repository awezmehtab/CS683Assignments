// verify.cpp — standalone correctness check for matmul_prefetch against matmul_naive.
//
// Not part of the graded submission (only src/matmul_*.cpp are) — it's a dev tool that
// exercises the blocking cleanup paths on sizes that are NOT multiples of the tile sizes,
// which is where off-by-one errors in a blocked kernel actually show up.
//
// Build (from task2/):
//   g++ -O0 -std=c++17 -Iinclude verify.cpp src/matmul_naive.cpp src/matmul_prefetch.cpp -o bin/verify
//   ./bin/verify
//
// Exits 0 only if every size passes.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "matmul.h"

static constexpr float kAbsTol = 1e-3f;

// Square M=N=K sizes. The odd ones (7, 15, 100, 257) deliberately straddle the tile
// boundaries so the i/j/k cleanup tails get exercised.
static const int kSizes[] = {7, 8, 15, 16, 100, 257, 512};

static bool check_size(int n) {
    const int M = n, N = n, K = n;
    const int lda = K, ldb = K, ldc = N;

    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(N) * K);
    std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);
    std::vector<float> ref(static_cast<std::size_t>(M) * N, 0.0f);

    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : A) v = dist(rng);
    for (float& v : B) v = dist(rng);

    // Poison C first: a kernel that forgets to zero-initialize before accumulating with +=
    // would silently pass if C happened to arrive already zeroed.
    for (float& v : C) v = 12345.0f;

    matmul_naive(A.data(), B.data(), ref.data(), M, N, K, lda, ldb, ldc);
    matmul_prefetch(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < C.size(); ++i) {
        const float d = std::fabs(C[i] - ref[i]);
        if (d > max_diff) max_diff = d;
    }

    const bool ok = (max_diff <= kAbsTol);
    std::printf("%4dx%-4d x %-4d  %-4s  max_diff = %.3e  (tol %.1e)\n", M, N, K,
                ok ? "PASS" : "FAIL", max_diff, kAbsTol);
    return ok;
}

int main() {
    std::printf("matmul_prefetch vs matmul_naive\n");
    std::printf("---------------------------------------------------------------\n");
    bool all_ok = true;
    for (int n : kSizes) all_ok &= check_size(n);
    std::printf("---------------------------------------------------------------\n");
    std::printf("%s\n", all_ok ? "ALL PASS" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
