#!/usr/bin/env bash
# exp_c_cache_metrics.sh  EXPERIMENT C: L1D / L2 / LLC misses vs matrix size, per stage.
#
# Deliverable 2A.2 ("analyze CPU metrics: L1D cache misses, L2 cache misses, LLC misses"),
# across the same grid as experiment B so the two tables line up row for row.
#
# Every point is measured with perf_delta(), which runs the binary twice (once with $REPS
# kernel repetitions, once with --reps=0 doing setup only) and subtracts. Without that
# subtraction the page faults from first-touching an n^2-float working set and the mt19937
# fill sit inside every count, and at n=128 they dominate the kernel entirely.
#
# Reported per stage and size:
#   - raw miss counts at each level, and instructions/cycles, for the whole $REPS-rep region
#   - MPKI at each level (misses * 1000 / instructions), which is the size-independent form
#     and the one the assignment asks about
#   - IPC, which separates "fewer misses" from "the misses stopped mattering"
#
# Usage:
#   ./exp_c_cache_metrics.sh
#   SIZES="256 512 1024 2048" STAGES="naive optimized" ./exp_c_cache_metrics.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGES="${STAGES:-naive simd prefetch optimized}"
CSV="$RESULTS_DIR/c_cache_metrics.csv"

banner "EXPERIMENT C  cache metrics vs matrix size"

require_perf
resolve_all_events
check_measurement_hygiene

info ""
print_event_map >&2

log ""
log "building perf_bench (default knobs)..."
BIN="$(build_pb perf_bench)"

echo "size,stage,time_ms,instructions,cycles,ipc,l1d_miss,l2_miss,llc_miss,l1d_mpki,l2_mpki,llc_mpki" > "$CSV"

printf '\n%-6s  %-10s  %10s  %10s  %10s  %10s  %7s\n' \
       "size" "stage" "L1D MPKI" "L2 MPKI" "LLC MPKI" "time(ms)" "IPC" >&2
printf -- '--------------------------------------------------------------------------\n' >&2

for n in $SIZES; do
    for stage in $STAGES; do
        perf_delta "$BIN" "$stage" "$n" "$n" "$n" l1d_miss l2_miss llc_miss

        ipc="$(ratio "${PD[instructions]}" "${PD[cycles]}")"
        l1_mpki="$(mpki "${PD[l1d_miss]}" "${PD[instructions]}")"
        l2_mpki="$(mpki "${PD[l2_miss]}" "${PD[instructions]}")"
        llc_mpki="$(mpki "${PD[llc_miss]}" "${PD[instructions]}")"

        printf '%-6s  %-10s  %10s  %10s  %10s  %10.3f  %7s\n' \
               "$n" "$stage" "$l1_mpki" "$l2_mpki" "$llc_mpki" "$PD_TIME_MS" "$ipc" >&2
        echo "$n,$stage,$PD_TIME_MS,${PD[instructions]},${PD[cycles]},$ipc,${PD[l1d_miss]},${PD[l2_miss]},${PD[llc_miss]},$l1_mpki,$l2_mpki,$llc_mpki" >> "$CSV"
    done
    printf -- '--------------------------------------------------------------------------\n' >&2
done

info ""
info "wrote $CSV"
