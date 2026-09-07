#!/usr/bin/env bash
# exp_g_simd_width.sh  EXPERIMENT G: speedup and instruction count vs SIMD width.
#
# Deliverable 2B in full: impact of SIMD instruction width (2B.1), impact of matrix size in
# combination with width (2B.2), instruction count per width (2B.3), and speedup (2B.4).
#
# This one is mostly a driver for machinery that already exists in simd_width_bench/:
# separate translation units for the 128-bit (SSE), 256-bit (AVX2) and 512-bit (AVX-512)
# kernels, each compiled with only the ISA flags it needs and linked into one binary.
# Compiling an AVX-512 object on a machine without AVX-512 is fine; only EXECUTING the
# instructions would fault, and bench.cpp guards that call with __builtin_cpu_supports.
#
# What this script adds on top:
#   - pins the timing run to the same core as every other experiment, so the numbers are
#     comparable with B/C/D/E rather than being collected under different conditions
#   - normalises both CSVs into results/ alongside everything else
#   - states up front whether AVX-512 is measurable here, so the "512-bit" row in the report
#     is an explicit, justified n/a rather than a hole
#
# On a CPU without AVX-512 (all AMD Zen 3 included), reporting the 512-bit variant as
# unsupported IS the correct deliverable. The PDF asks for 512-bit "if available" (Task 1C)
# and "depending on your machine's capabilities" (Task 2B).
#
# Usage:
#   ./exp_g_simd_width.sh
#   SIZES="128 256 512 1024 2048" REPS=5 ./exp_g_simd_width.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BENCH_DIR="$TASK2_DIR/simd_width_bench"
TIME_CSV="$RESULTS_DIR/g_simd_width_time.csv"
INSTR_CSV="$RESULTS_DIR/g_simd_width_instructions.csv"

banner "EXPERIMENT G  SIMD width: 128 / 256 / 512-bit"

require_perf
check_measurement_hygiene

if has_flag avx512f; then
    info "AVX-512F: available, all three widths will be measured"
    VARIANTS="${VARIANTS:-naive sse128 avx2_256 avx512}"
else
    info "AVX-512F: NOT available on this CPU ($(cpu_model))"
    info "  The 512-bit variant will be reported as unsupported, which is the correct"
    info "  deliverable here, not a missing measurement."
    VARIANTS="${VARIANTS:-naive sse128 avx2_256}"
fi

log ""
log "building simd_width_bench..."
make -s -C "$BENCH_DIR" >/dev/null || die "simd_width_bench build failed"
BENCH="$BIN_DIR/simd_width_bench"
[ -x "$BENCH" ] || die "expected $BENCH after build"

# ------------------------------------------------- pass 1: time / speedup ----
# bench.cpp writes simd_width_results.csv into its working directory, so run it from
# results/ and let it land there directly.
info ""
info "pass 1: timing and speedup vs size"
( cd "$RESULTS_DIR" && "${PIN[@]}" "$BENCH" $SIZES )
if [ -f "$RESULTS_DIR/simd_width_results.csv" ]; then
    mv "$RESULTS_DIR/simd_width_results.csv" "$TIME_CSV"
    info "wrote $TIME_CSV"
else
    warn "simd_width_bench did not produce simd_width_results.csv"
fi

# --------------------------------------------- pass 2: instruction counts ----
# Deliverable 2B.3. Uses bench.cpp's --only isolate mode so the count reflects one variant
# rather than the whole sweep, and subtracts a --reps=0 baseline for the same reason
# perf_delta does: at 128^3 the setup would otherwise be a large fraction of the total.
info ""
info "pass 2: instruction count per variant (perf)"

echo "size,variant,reps,instructions,instructions_per_rep,cycles,ipc,correct" > "$INSTR_CSV"

printf '\n%-6s  %-9s  %16s  %16s  %8s  %8s\n' \
       "size" "variant" "instructions" "per-rep" "IPC" "correct" >&2
printf -- '----------------------------------------------------------------------------\n' >&2

# bench.cpp's isolate mode has no --reps=0 equivalent, so instead of subtracting a baseline
# we measure at two rep counts and take the slope. The difference between an R-rep run and a
# 1-rep run is exactly (R-1) kernel invocations, with identical setup in both, which cancels
# the setup out the same way.
R_HI="${R_HI:-4}"

instr_of() {  # instr_of <variant> <size> <reps>  ->  "instructions cycles correct"
    local v="$1" n="$2" r="$3" prog perfout instr cyc corr
    prog="$(mktemp)"
    perfout="$(perf stat -e instructions,cycles -x, -- \
               "${PIN[@]}" "$BENCH" --only="$v" --size="$n" --reps="$r" 2>&1 1>"$prog")" || {
        rm -f "$prog"; echo "0 0 error"; return 0; }
    instr="$(awk -F, '$3 ~ /^instructions/ { gsub(/[^0-9]/,"",$1); print $1; exit }' <<< "$perfout")"
    cyc="$(awk -F, '$3 ~ /^cycles/ { gsub(/[^0-9]/,"",$1); print $1; exit }' <<< "$perfout")"
    corr="$(sed -n 's/.*correct=\([a-zA-Z]*\).*/\1/p' "$prog")"
    grep -q 'skipped=unsupported' "$prog" && corr="unsupported"
    rm -f "$prog"
    echo "${instr:-0} ${cyc:-0} ${corr:-?}"
}

for n in $SIZES; do
    for v in $VARIANTS; do
        read -r hi_i hi_c corr <<< "$(instr_of "$v" "$n" "$R_HI")"
        if [ "$corr" = "unsupported" ]; then
            printf '%-6s  %-9s  %16s  %16s  %8s  %8s\n' "$n" "$v" "n/a" "n/a" "n/a" "unsupported" >&2
            echo "$n,$v,$R_HI,,,,,unsupported" >> "$INSTR_CSV"
            continue
        fi
        read -r lo_i lo_c _ <<< "$(instr_of "$v" "$n" 1)"

        per_rep="$(awk -v a="$hi_i" -v b="$lo_i" -v r="$R_HI" \
                   'BEGIN { printf "%.0f", (r > 1 ? (a-b)/(r-1) : a) }')"
        cyc_rep="$(awk -v a="$hi_c" -v b="$lo_c" -v r="$R_HI" \
                   'BEGIN { printf "%.0f", (r > 1 ? (a-b)/(r-1) : a) }')"
        ipc="$(ratio "$per_rep" "$cyc_rep")"

        printf '%-6s  %-9s  %16s  %16s  %8s  %8s\n' "$n" "$v" "$hi_i" "$per_rep" "$ipc" "$corr" >&2
        echo "$n,$v,$R_HI,$hi_i,$per_rep,$cyc_rep,$ipc,$corr" >> "$INSTR_CSV"
    done
    printf -- '----------------------------------------------------------------------------\n' >&2
done

info ""
info "wrote $INSTR_CSV"
