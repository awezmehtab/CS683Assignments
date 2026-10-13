#!/usr/bin/env bash
# exp_h_block_size.sh  EXPERIMENT H: cache-blocking parameter tuning.
#
# Not a named PDF deliverable, but it does two jobs:
#   - supports "identify optimal parameters" (2A.4) for the blocking knobs, not just the
#     prefetch ones, and gives the report a cache-capacity argument backed by measurement
#   - finds the configuration to leave as the committed default in src/matmul_optimized.cpp,
#     which is what the graded speedup tier is computed from
#
# Sweeps BLOCK_M x BLOCK_N x BLOCK_K by rebuilding at each point. The working set of one
# block is roughly
#
#     (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K + BLOCK_M * BLOCK_N) * 4 bytes
#
# for the A panel, B panel and C tile respectively. The script prints that figure next to
# each result together with the machine's L1d and L2 sizes, so the winner can be explained
# by where its working set lands in the hierarchy rather than just declared empirically.
#
# Usage:
#   ./exp_h_block_size.sh
#   BM_LIST="16 32 64" BN_LIST="16 32 64" BK_LIST="128 256 512" TUNE_SIZE=2048 ./exp_h_block_size.sh
#
# Cost: one full rebuild per combination. The default grid is 3*3*3 = 27 builds per stage.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BM_LIST="${BM_LIST:-16 32 64 128}"
BN_LIST="${BN_LIST:-16 32 64 128}"
BK_LIST="${BK_LIST:-128 256 512}"
TUNE_SIZE="${TUNE_SIZE:-1024}"
STAGES="${STAGES:-optimized}"
PF_DIST="${PF_DIST:-}"     # empty = leave the source default alone
CSV="$RESULTS_DIR/h_block_size.csv"

banner "EXPERIMENT H  cache-block size tuning at ${TUNE_SIZE}^3"
check_measurement_hygiene

# Cache sizes, for the working-set-vs-capacity column.
L1D_KB="$(lscpu | sed -n 's/^L1d cache:[[:space:]]*\([0-9]*\) KiB.*/\1/p')"
L1D_INST="$(lscpu | sed -n 's/^L1d cache:.*(\([0-9]*\) instances).*/\1/p')"
L2_KB="$(lscpu | sed -n 's/^L2 cache:[[:space:]]*\([0-9]*\) \(KiB\|MiB\).*/\1/p')"
L2_UNIT="$(lscpu | sed -n 's/^L2 cache:[[:space:]]*[0-9]* \(KiB\|MiB\).*/\1/p')"
L2_INST="$(lscpu | sed -n 's/^L2 cache:.*(\([0-9]*\) instances).*/\1/p')"
[ "$L2_UNIT" = "MiB" ] && L2_KB=$((L2_KB * 1024))
PER_CORE_L1="$(( ${L1D_KB:-32} / ${L1D_INST:-1} ))"
PER_CORE_L2="$(( ${L2_KB:-512} / ${L2_INST:-1} ))"

info "per-core L1d = ${PER_CORE_L1} KiB, per-core L2 = ${PER_CORE_L2} KiB"
info "a block's working set should sit inside L2, with the hot A panel inside L1d"

echo "stage,size,block_m,block_n,block_k,working_set_kib,time_ms,gflops,correct" > "$CSV"

printf '\n%-10s  %5s  %5s  %5s  %12s  %10s  %10s\n' \
       "stage" "BM" "BN" "BK" "wset(KiB)" "time(ms)" "GFLOP/s" >&2
printf -- '----------------------------------------------------------------------\n' >&2

for stage in $STAGES; do
    for bm in $BM_LIST; do
        for bn in $BN_LIST; do
            for bk in $BK_LIST; do
                wset="$(awk -v m="$bm" -v n="$bn" -v k="$bk" \
                        'BEGIN { printf "%.1f", (m*k + n*k + m*n) * 4 / 1024.0 }')"

                extra="-DBLOCK_M=$bm -DBLOCK_N=$bn -DBLOCK_K=$bk"
                [ -n "$PF_DIST" ] && extra="$extra -DPREFETCH_DIST=$PF_DIST"

                bin="$(build_pb "pb_b${bm}_${bn}_${bk}" "$extra")"
                ok="$(verify_pb "$bin" "$stage")"
                if [ "$ok" = "NO" ]; then
                    warn "INCORRECT at BM=$bm BN=$bn BK=$bk, skipping"
                    echo "$stage,$TUNE_SIZE,$bm,$bn,$bk,$wset,,,NO" >> "$CSV"
                    continue
                fi

                ms="$(run_time "$bin" "$stage" "$TUNE_SIZE" "$TUNE_SIZE" "$TUNE_SIZE")"
                gflops="$(awk -v n="$TUNE_SIZE" -v ms="$ms" \
                          'BEGIN { printf "%.4f", 2.0*n*n*n/(ms*1e6) }')"

                printf '%-10s  %5s  %5s  %5s  %12s  %10.3f  %10.2f\n' \
                       "$stage" "$bm" "$bn" "$bk" "$wset" "$ms" "$gflops" >&2
                echo "$stage,$TUNE_SIZE,$bm,$bn,$bk,$wset,$ms,$gflops,$ok" >> "$CSV"

                rm -f "$bin"   # 27+ binaries at ~1 MB each, and none is needed again
            done
        done
    done
done

info ""
info "wrote $CSV"
info ""
info "best configuration per stage:"
awk -F, 'NR>1 && $9=="yes" && $7 != "" {
             if (!($1 in best) || $7+0 < best[$1]) { best[$1]=$7+0; cfg[$1]=$3"x"$4"x"$5; w[$1]=$6 }
         }
         END { for (s in best)
                   printf "  %-12s BLOCK_M x BLOCK_N x BLOCK_K = %-14s  %.3f ms  (working set %s KiB)\n",
                          s, cfg[s], best[s], w[s] }' "$CSV" >&2

info ""
info "to adopt: set the defaults in src/matmul_optimized.cpp, or pass the -D flags through"
info "          EXTRA when building perf_bench."
