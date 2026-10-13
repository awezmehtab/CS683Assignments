#!/usr/bin/env bash
# run_perf_sweep.sh  Task 2B deliverable 3: instruction count vs. SIMD width & matrix size.
#
# Builds simd_width_bench, then for every (matrix size, SIMD width) combination runs the
# isolated single-variant mode under `perf stat -e instructions` and writes a CSV:
#
#   size,variant,reps,instructions,instructions_per_rep,correct
#
# Usage:
#   ./run_perf_sweep.sh
#   SIZES="128 256 512 1024" VARIANTS="sse128 avx2_256" REPS=5 ./run_perf_sweep.sh
#
# Requires `perf` with instruction-counting access (see the perf_event_paranoid note below
# if you get "No supported events found").

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SIZES=${SIZES:-"128 256 512 1024 2048"}
VARIANTS=${VARIANTS:-"naive sse128 avx2_256 avx512"}
REPS=${REPS:-5}
OUT_CSV=${OUT_CSV:-"instruction_counts.csv"}
BIN="../bin/simd_width_bench"

if ! command -v perf >/dev/null 2>&1; then
    echo "error: 'perf' not found on PATH. Install linux-tools for your kernel (e.g." >&2
    echo "  sudo apt install linux-tools-common linux-tools-\$(uname -r))." >&2
    exit 1
fi

echo "building simd_width_bench..."
make -s

# Sanity-check perf can actually count instructions before running the full sweep.
if ! perf stat -e instructions -- true >/tmp/perf_probe.$$ 2>&1; then
    echo "error: 'perf stat' failed a basic probe:" >&2
    cat /tmp/perf_probe.$$ >&2
    echo "" >&2
    echo "If you see 'No supported events found' / a perf_event_paranoid warning, either" >&2
    echo "run this script with sudo, or as root:" >&2
    echo "  sudo sysctl kernel.perf_event_paranoid=1" >&2
    rm -f /tmp/perf_probe.$$
    exit 1
fi
rm -f /tmp/perf_probe.$$

avx512_supported=0
if grep -qm1 avx512f /proc/cpuinfo 2>/dev/null; then
    avx512_supported=1
fi

echo "size,variant,reps,instructions,instructions_per_rep,correct" > "$OUT_CSV"

printf "%-6s  %-9s  %14s  %16s  %8s\n" "size" "variant" "instructions" "per-rep" "correct"
printf -- "----------------------------------------------------------------\n"

for size in $SIZES; do
    for variant in $VARIANTS; do
        if [ "$variant" = "avx512" ] && [ "$avx512_supported" -eq 0 ]; then
            printf "%-6s  %-9s  %14s  %16s  %8s\n" "$size" "$variant" "n/a" "n/a" "unsupported"
            echo "$size,$variant,$REPS,,,unsupported" >> "$OUT_CSV"
            continue
        fi

        prog_out=$(mktemp)
        perf_out=$(perf stat -e instructions -x',' -- "$BIN" --only="$variant" --size="$size" \
                   --reps="$REPS" 2>&1 1>"$prog_out") || {
            echo "warning: run failed for size=$size variant=$variant" >&2
            echo "$size,$variant,$REPS,,,error" >> "$OUT_CSV"
            rm -f "$prog_out"
            continue
        }

        instructions=$(echo "$perf_out" | grep -m1 instructions | cut -d',' -f1)
        correct=$(grep -o 'correct=[a-zA-Z]*' "$prog_out" | cut -d= -f2)
        rm -f "$prog_out"

        if [ -z "$instructions" ]; then
            printf "%-6s  %-9s  %14s  %16s  %8s\n" "$size" "$variant" "n/a" "n/a" "perf_error"
            echo "$size,$variant,$REPS,,,perf_error" >> "$OUT_CSV"
            continue
        fi

        per_rep=$(awk -v i="$instructions" -v r="$REPS" 'BEGIN { printf "%.0f", i / r }')
        printf "%-6s  %-9s  %14s  %16s  %8s\n" "$size" "$variant" "$instructions" "$per_rep" \
               "${correct:-?}"
        echo "$size,$variant,$REPS,$instructions,$per_rep,${correct:-?}" >> "$OUT_CSV"
    done
done

echo ""
echo "wrote $OUT_CSV"
