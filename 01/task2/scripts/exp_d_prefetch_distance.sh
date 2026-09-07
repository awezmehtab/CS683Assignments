#!/usr/bin/env bash
# exp_d_prefetch_distance.sh  EXPERIMENT D: prefetch distance sweep.
#
# Deliverables 2A.3/2A.4 ("what is the best prefetch distance?") and, via STAGES, the 2C
# repeat on the combined kernel. Feeds plots/task2_prefetch_degree.png.
#
# Sweeps PREFETCH_DIST (in FLOATS ahead of the current position; 16 floats = one 64 B line)
# by rebuilding the kernel at each value, since the distance is a compile-time -D knob.
#
# THE BASELINE. The assignment defines
#     speedup = time without software prefetching / time with software prefetching
# so the denominator has to be the SAME kernel with prefetches removed, not matmul_simd.
# matmul_simd is un-blocked, so dividing by it would report the cache-blocking win as if it
# were a prefetching win. PREFETCH_DIST=0 compiles every _mm_prefetch away through the PF()
# macro (verified: zero prefetcht* instructions in the binary), which gives exactly that
# baseline. It is the dist=0 row in the CSV and the denominator of every speedup below.
#
# Counters collected per point:
#   l1d_miss     did the prefetch actually remove demand misses
#   sw_prefetch  how many prefetch instructions were dispatched (the PDF's
#                sw_prefetch_access; ls_pref_instr_disp.prefetch on AMD). Should be roughly
#                flat across distances, which is what proves the distance, and not the
#                prefetch COUNT, is the variable.
#   sw_prefetch_inef  (AMD only) prefetches whose line was already in L1, i.e. issued too
#                late to help. This is the direct evidence for the left edge of the curve:
#                a short distance shows high ineffective counts AND high L1D misses, which
#                separates "too late" from "too early / evicted before use" instead of
#                inferring the distinction from the shape of the timing curve alone.
#
# Usage:
#   ./exp_d_prefetch_distance.sh
#   DISTS="0 8 16 32 64 128" SIZES="512 1024" STAGES=optimized ./exp_d_prefetch_distance.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# 0 means "compiled out" (the baseline). The rest span from well under one cache line to
# several lines ahead, which brackets the useful range on any current core.
DISTS="${DISTS:-0 8 16 32 64 128 256 512}"
# The sizes here are a subset of experiment B's: the distance optimum only moves with the
# memory level being hit, so three points (L2-resident, borderline, DRAM-bound) show the
# trend without paying for the full sweep at every distance.
SIZES="${SIZES:-512 1024 2048}"
STAGES="${STAGES:-prefetch optimized}"
CSV="$RESULTS_DIR/d_prefetch_distance.csv"

banner "EXPERIMENT D  prefetch distance sweep"

require_perf
resolve_all_events
check_measurement_hygiene

[ -z "${EV[sw_prefetch_inef]}" ] && \
    log "note: no ineffective-prefetch counter on this CPU (AMD-only event); that column will be 0."

echo "stage,size,dist_floats,dist_bytes,time_ms,speedup_vs_noprefetch,instructions,cycles,ipc,l1d_miss,l1d_mpki,sw_prefetch,sw_prefetch_inef,inef_frac,correct" > "$CSV"

for stage in $STAGES; do
    info ""
    info "--- stage: $stage ---"

    # Build every distance once up front, so the sweep is pure measurement afterwards and a
    # rebuild never lands between two timed points.
    declare -A BINS=()
    for d in $DISTS; do
        log "building PREFETCH_DIST=$d ..."
        BINS[$d]="$(build_pb "pb_d${d}" "-DPREFETCH_DIST=$d")"
        [ "$d" = "0" ] && assert_noprefetch "${BINS[$d]}"
    done

    printf '\n%-6s  %-6s  %-8s  %10s  %9s  %10s  %12s  %8s\n' \
           "size" "dist" "bytes" "time(ms)" "speedup" "L1D MPKI" "sw_pf" "inef%" >&2
    printf -- '------------------------------------------------------------------------------\n' >&2

    for n in $SIZES; do
        base_ms=""
        for d in $DISTS; do
            bin="${BINS[$d]}"
            ok="$(verify_pb "$bin" "$stage")"
            [ "$ok" = "NO" ] && die "$stage incorrect at PREFETCH_DIST=$d"

            perf_delta "$bin" "$stage" "$n" "$n" "$n" l1d_miss sw_prefetch sw_prefetch_inef

            # dist=0 is the no-software-prefetch build and defines the denominator.
            if [ "$d" = "0" ]; then base_ms="$PD_TIME_MS"; fi
            speedup="$(ratio "$base_ms" "$PD_TIME_MS")"

            ipc="$(ratio "${PD[instructions]}" "${PD[cycles]}")"
            l1_mpki="$(mpki "${PD[l1d_miss]}" "${PD[instructions]}")"
            inef_frac="$(ratio "${PD[sw_prefetch_inef]}" "${PD[sw_prefetch]}")"
            bytes=$((d * 4))

            printf '%-6s  %-6s  %-8s  %10.3f  %8.3fx  %10s  %12s  %7s%%\n' \
                   "$n" "$d" "$bytes" "$PD_TIME_MS" "$speedup" "$l1_mpki" \
                   "${PD[sw_prefetch]}" \
                   "$(awk -v f="$inef_frac" 'BEGIN { printf "%.1f", f*100 }')" >&2

            echo "$stage,$n,$d,$bytes,$PD_TIME_MS,$speedup,${PD[instructions]},${PD[cycles]},$ipc,${PD[l1d_miss]},$l1_mpki,${PD[sw_prefetch]},${PD[sw_prefetch_inef]},$inef_frac,$ok" >> "$CSV"
        done
        printf -- '------------------------------------------------------------------------------\n' >&2
    done
    unset BINS
done

info ""
info "wrote $CSV"
info ""
info "best distance per (stage, size):"
awk -F, 'NR>1 && $3 != 0 { k = $1 "," $2; if (!(k in best) || $6+0 > best[k]) { best[k] = $6+0; d[k] = $3 } }
         END { for (k in best) printf "  %-22s dist=%-5s speedup=%.3fx\n", k, d[k], best[k] }' \
    "$CSV" | sort >&2
