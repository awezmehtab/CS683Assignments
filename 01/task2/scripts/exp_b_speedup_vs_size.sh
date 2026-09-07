#!/usr/bin/env bash
# exp_b_speedup_vs_size.sh  EXPERIMENT B: speedup vs matrix size, all four stages.
#
# Deliverables 2A.1 (impact of matrix size), 2A.3 (execution time and speedup) and the
# final 2C comparison plot. Feeds plots/task2_speedup_vs_size.png.
#
# Sweeps M=N=K over $SIZES and times naive / simd / prefetch / optimized at each point,
# with everything pinned to one core and each stage measured in its own process so a stage
# never runs in a cache warmed by another stage.
#
# Timing only, so this one does NOT need perf. Cache counters for the same grid are
# experiment C.
#
# Usage:
#   ./exp_b_speedup_vs_size.sh
#   SIZES="128 256 512 1024 2048 3072" REPS=5 ./exp_b_speedup_vs_size.sh
#
# Runtime note: cost is dominated by matmul_naive, which is O(n^3) at roughly 1-2 GFLOP/s.
# 2048 costs about 10 s per naive repetition, 3072 about 35 s. The optimized stages are
# 20-40x faster and effectively free by comparison.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGES="${STAGES:-naive simd prefetch optimized}"
CSV="$RESULTS_DIR/b_speedup_vs_size.csv"

banner "EXPERIMENT B  speedup vs matrix size"
check_measurement_hygiene

log "building perf_bench (default knobs)..."
BIN="$(build_pb perf_bench)"

# One correctness check per stage up front. Doing it here rather than per size point keeps
# the naive reference run out of every measurement, and 257 straddles every block boundary.
info ""
info "correctness at ${VERIFY_SIZE}x${VERIFY_SIZE}x${VERIFY_SIZE} (tile-straddling):"
for stage in $STAGES; do
    [ "$stage" = "naive" ] && continue
    ok="$(verify_pb "$BIN" "$stage")"
    printf '  %-10s %s\n' "$stage" "$ok" >&2
    [ "$ok" = "NO" ] && die "$stage is incorrect; fix it before collecting numbers"
done

echo "size,stage,time_ms,gflops,speedup_vs_naive" > "$CSV"

printf '\n%-6s  %-10s  %12s  %10s  %10s\n' "size" "stage" "time(ms)" "GFLOP/s" "speedup" >&2
printf -- '------------------------------------------------------------\n' >&2

for n in $SIZES; do
    naive_ms=""
    for stage in $STAGES; do
        ms="$(run_time "$BIN" "$stage" "$n" "$n" "$n")"
        gflops="$(awk -v n="$n" -v ms="$ms" 'BEGIN { printf "%.4f", 2.0*n*n*n/(ms*1e6) }')"

        if [ "$stage" = "naive" ]; then
            naive_ms="$ms"
            speedup="1.0000"
        else
            speedup="$(ratio "$naive_ms" "$ms")"
        fi

        printf '%-6s  %-10s  %12.3f  %10.2f  %9.2fx\n' "$n" "$stage" "$ms" "$gflops" "$speedup" >&2
        echo "$n,$stage,$ms,$gflops,$speedup" >> "$CSV"
    done
    printf -- '------------------------------------------------------------\n' >&2
done

info ""
info "wrote $CSV"
info "plot with: python3 $SCRIPTS_DIR/plot_all.py"
