#!/usr/bin/env bash
# exp_e_prefetch_level.sh  EXPERIMENT E: cache fill level (locality hint) sweep.
#
# Deliverable 2A.4 ("at what cache fill level do you achieve the maximum speedup?") and the
# 2C repeat. Feeds plots/task2_prefetch_level.png.
#
# Two passes, because the kernel uses TWO independent hints and collapsing them would hide
# the interesting result:
#
#   PASS 1 (tied):  both operands get the same hint. Four points, T0/T1/T2/NTA. This is the
#                   plot the assignment asks for, and the one whose answer is quotable as
#                   "the best cache fill level is X".
#
#   PASS 2 (grid):  the full 4x4 of PREFETCH_HINT_A x PREFETCH_HINT_B, at the single size
#                   given by GRID_SIZE. The kernel prefetches A and B for different reasons:
#                   A's rows are reused across the whole j-sweep of the current i-tile, so
#                   they want L1 (T0); B's rows are touched once per (i,j) and only need to
#                   have landed by the time they are consumed, so L2 (T1) is enough and
#                   avoids evicting the hotter A data from L1. Pass 2 tests whether that
#                   reasoning survives contact with the hardware, and it is what justifies
#                   the asymmetric default rather than just asserting it.
#
# The distance is held fixed at PF_DIST for the whole experiment (default: whatever
# experiment D found best; override with PF_DIST=<n>), so this sweep varies one thing.
#
# Usage:
#   ./exp_e_prefetch_level.sh
#   PF_DIST=64 SIZES="512 1024 2048" GRID_SIZE=1024 ./exp_e_prefetch_level.sh
#   SKIP_GRID=1 ./exp_e_prefetch_level.sh        # pass 1 only

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

HINTS="${HINTS:-_MM_HINT_T0 _MM_HINT_T1 _MM_HINT_T2 _MM_HINT_NTA}"
PF_DIST="${PF_DIST:-64}"
SIZES="${SIZES:-512 1024 2048}"
GRID_SIZE="${GRID_SIZE:-1024}"
STAGES="${STAGES:-prefetch optimized}"
SKIP_GRID="${SKIP_GRID:-0}"
CSV="$RESULTS_DIR/e_prefetch_level.csv"
GRID_CSV="$RESULTS_DIR/e_prefetch_level_grid.csv"

short() { sed 's/_MM_HINT_//' <<< "$1"; }

banner "EXPERIMENT E  cache fill level (T0 / T1 / T2 / NTA), distance fixed at $PF_DIST"

require_perf
resolve_all_events
check_measurement_hygiene

# The no-prefetch baseline again, so "speedup" here means the same thing it does in D.
log "building no-prefetch baseline..."
BASE_BIN="$(build_pb pb_d0 "-DPREFETCH_DIST=0")"
assert_noprefetch "$BASE_BIN"

# ------------------------------------------------------------------ pass 1 ----
echo "stage,size,hint,time_ms,speedup_vs_noprefetch,instructions,cycles,ipc,l1d_miss,l1d_mpki,l2_miss,l2_mpki,sw_prefetch,correct" > "$CSV"

for stage in $STAGES; do
    info ""
    info "--- pass 1 (both operands tied), stage: $stage ---"

    declare -A BINS=()
    for h in $HINTS; do
        s="$(short "$h")"
        log "building hint=$s ..."
        BINS[$h]="$(build_pb "pb_h${s}" "-DPREFETCH_DIST=$PF_DIST -DPREFETCH_HINT_A=$h -DPREFETCH_HINT_B=$h")"
    done

    printf '\n%-6s  %-6s  %10s  %9s  %10s  %10s  %7s\n' \
           "size" "hint" "time(ms)" "speedup" "L1D MPKI" "L2 MPKI" "IPC" >&2
    printf -- '--------------------------------------------------------------------\n' >&2

    for n in $SIZES; do
        base_ms="$(run_time "$BASE_BIN" "$stage" "$n" "$n" "$n")"
        for h in $HINTS; do
            s="$(short "$h")"
            bin="${BINS[$h]}"
            ok="$(verify_pb "$bin" "$stage")"
            [ "$ok" = "NO" ] && die "$stage incorrect at hint=$s"

            perf_delta "$bin" "$stage" "$n" "$n" "$n" l1d_miss l2_miss sw_prefetch

            speedup="$(ratio "$base_ms" "$PD_TIME_MS")"
            ipc="$(ratio "${PD[instructions]}" "${PD[cycles]}")"
            l1_mpki="$(mpki "${PD[l1d_miss]}" "${PD[instructions]}")"
            l2_mpki="$(mpki "${PD[l2_miss]}" "${PD[instructions]}")"

            printf '%-6s  %-6s  %10.3f  %8.3fx  %10s  %10s  %7s\n' \
                   "$n" "$s" "$PD_TIME_MS" "$speedup" "$l1_mpki" "$l2_mpki" "$ipc" >&2
            echo "$stage,$n,$s,$PD_TIME_MS,$speedup,${PD[instructions]},${PD[cycles]},$ipc,${PD[l1d_miss]},$l1_mpki,${PD[l2_miss]},$l2_mpki,${PD[sw_prefetch]},$ok" >> "$CSV"
        done
        printf -- '--------------------------------------------------------------------\n' >&2
    done
    unset BINS
done

info ""
info "wrote $CSV"

# ------------------------------------------------------------------ pass 2 ----
if [ "$SKIP_GRID" = "1" ]; then
    log "SKIP_GRID=1, skipping the 4x4 A-hint x B-hint grid"
    exit 0
fi

banner "EXPERIMENT E  pass 2: 4x4 A-hint x B-hint grid at size $GRID_SIZE"

echo "stage,size,hint_a,hint_b,time_ms,speedup_vs_noprefetch,l1d_miss,l1d_mpki,l2_miss,l2_mpki,correct" > "$GRID_CSV"

for stage in $STAGES; do
    info ""
    info "--- grid, stage: $stage, size $GRID_SIZE ---"
    base_ms="$(run_time "$BASE_BIN" "$stage" "$GRID_SIZE" "$GRID_SIZE" "$GRID_SIZE")"

    # Header row: B hints across the top.
    printf '\n%-8s' "A \\ B" >&2
    for hb in $HINTS; do printf '  %10s' "$(short "$hb")" >&2; done
    printf '\n' >&2

    for ha in $HINTS; do
        printf '%-8s' "$(short "$ha")" >&2
        for hb in $HINTS; do
            sa="$(short "$ha")"; sb="$(short "$hb")"
            bin="$(build_pb "pb_g${sa}_${sb}" \
                   "-DPREFETCH_DIST=$PF_DIST -DPREFETCH_HINT_A=$ha -DPREFETCH_HINT_B=$hb")"
            ok="$(verify_pb "$bin" "$stage")"
            [ "$ok" = "NO" ] && die "$stage incorrect at A=$sa B=$sb"

            perf_delta "$bin" "$stage" "$GRID_SIZE" "$GRID_SIZE" "$GRID_SIZE" l1d_miss l2_miss
            speedup="$(ratio "$base_ms" "$PD_TIME_MS")"
            l1_mpki="$(mpki "${PD[l1d_miss]}" "${PD[instructions]}")"
            l2_mpki="$(mpki "${PD[l2_miss]}" "${PD[instructions]}")"

            printf '  %9.3fx' "$speedup" >&2
            echo "$stage,$GRID_SIZE,$sa,$sb,$PD_TIME_MS,$speedup,${PD[l1d_miss]},$l1_mpki,${PD[l2_miss]},$l2_mpki,$ok" >> "$GRID_CSV"
        done
        printf '\n' >&2
    done
done

info ""
info "wrote $GRID_CSV"
info ""
info "best (A,B) hint pair per stage:"
awk -F, 'NR>1 { k = $1; if (!(k in best) || $6+0 > best[k]) { best[k] = $6+0; p[k] = $3 "/" $4 } }
         END { for (k in best) printf "  %-12s A/B=%-10s speedup=%.3fx\n", k, p[k], best[k] }' \
    "$GRID_CSV" | sort >&2
