#!/usr/bin/env bash
# exp_i_llama.sh  EXPERIMENT I: the kernel inside a real llama.cpp inference.
#
# task2/README.md section 4. Clones a pinned llama.cpp (b5731), builds it stock, injects
# matmul_optimized into ggml's CPU F32 matmul, builds again, and runs the same prompt
# through both single-threaded, reporting whether the kernel was actually invoked, whether
# the generated text matches token for token, and the tokens/sec of each.
#
# The heavy lifting is llama/run_demo.sh. This wrapper adds:
#   - a prerequisite check up front, since the underlying script fails deep into a clone or
#     a cmake build otherwise
#   - REPEATS runs, because a single tokens/sec figure off one short generation is noise;
#     the median across runs is what belongs in the report
#   - parsing of the prompt-eval and generation-eval timings into a CSV
#
# READ THIS BEFORE INTERPRETING THE NUMBERS. Single-token generation calls matmul with M=1,
# which lands in the kernel's scalar/cleanup tail: correct, but not the register-tiled fast
# path, and with one row of A there is no reuse to block for anyway. The PROMPT EVAL
# matmuls are the batched ones where the 4x4 micro-kernel actually runs. So the prompt-eval
# row is the meaningful comparison against llama.cpp's native SGEMM, and the generation row
# is expected to be unremarkable. Reporting the generation number as if it measured the
# kernel would be wrong.
#
# Requires: git, cmake, a C/C++ toolchain, python3, curl, and network access.
#
# Usage:
#   ./exp_i_llama.sh
#   REPEATS=5 NGEN=128 PROMPT="Once upon a time" ./exp_i_llama.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REPEATS="${REPEATS:-3}"
LOG="$RESULTS_DIR/i_llama.log"
CSV="$RESULTS_DIR/i_llama.csv"

banner "EXPERIMENT I  matmul_optimized inside llama.cpp"

# ------------------------------------------------------------ preflight ----
missing=""
for t in git cmake python3 curl; do
    command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
done
[ -n "$missing" ] && die "missing required tools:$missing"

if ! curl -fsS --max-time 10 -o /dev/null https://github.com 2>/dev/null; then
    die "no network access to github.com. This experiment clones llama.cpp and downloads a model."
fi

info "running the demo $REPEATS time(s); the first includes the clone and two cmake builds"
: > "$LOG"

echo "run,build,prompt_eval_tps,gen_eval_tps,kernel_used,output_matches" > "$CSV"

for r in $(seq 1 "$REPEATS"); do
    info ""
    info "--- run $r/$REPEATS ---"
    # run_demo.sh is idempotent: it resets the llama.cpp tree to pristine before each
    # stock build, so repeated runs re-time rather than re-clone.
    if ! bash "$TASK2_DIR/llama/run_demo.sh" 2>&1 | tee -a "$LOG"; then
        die "run_demo.sh failed; see $LOG"
    fi

    kernel_used=no
    grep -q "student matmul_optimized was actually used" "$LOG" && kernel_used=yes
    matches=no
    grep -q "matches stock output token-for-token" "$LOG" && matches=yes

    # llama.cpp prints, on stderr, lines of the form
    #   llama_perf_context_print: prompt eval time = X ms / N tokens ( ... , T tokens per second)
    #   llama_perf_context_print:        eval time = X ms / N runs   ( ... , T tokens per second)
    parse_tps() {  # parse_tps <errfile> <prompt|gen>
        local f="$1" which="$2" pat
        [ "$which" = "prompt" ] && pat="prompt eval time" || pat="^[^:]*: *eval time"
        grep -E "$pat" "$f" 2>/dev/null | tail -1 \
            | sed -n 's/.*,[[:space:]]*\([0-9.]*\)[[:space:]]*tokens per second.*/\1/p'
    }

    for build in stock student; do
        errf="/tmp/mm_${build}.err"
        p_tps="$(parse_tps "$errf" prompt)"
        g_tps="$(parse_tps "$errf" gen)"
        echo "$r,$build,${p_tps:-},${g_tps:-},$kernel_used,$matches" >> "$CSV"
        printf '  %-8s prompt-eval %8s tok/s   gen-eval %8s tok/s\n' \
               "$build" "${p_tps:-n/a}" "${g_tps:-n/a}" >&2
    done
done

info ""
info "wrote $CSV and $LOG"
info ""
info "median across $REPEATS run(s):"
awk -F, 'NR>1 && $3 != "" { p[$2] = p[$2] " " $3 }
         NR>1 && $4 != "" { g[$2] = g[$2] " " $4 }
         function med(s,   a, n, i) {
             n = split(s, a, " ")
             if (n == 0) return "n/a"
             for (i = 1; i < n; i++) for (j = i+1; j <= n; j++)
                 if (a[j]+0 < a[i]+0) { t = a[i]; a[i] = a[j]; a[j] = t }
             return a[int((n+1)/2)]
         }
         END {
             printf "  %-8s prompt-eval %10s tok/s   gen-eval %10s tok/s\n", "stock",   med(p["stock"]),   med(g["stock"])
             printf "  %-8s prompt-eval %10s tok/s   gen-eval %10s tok/s\n", "student", med(p["student"]), med(g["student"])
             ps = med(p["stock"]) + 0; pt = med(p["student"]) + 0
             if (ps > 0 && pt > 0)
                 printf "\n  prompt-eval speedup vs llama.cpp native SGEMM: %.3fx\n", pt / ps
             print "\n  (the generation row runs the M=1 scalar tail, not the register-tiled"
             print "   fast path, so the prompt-eval row is the one to report)"
         }' "$CSV" >&2
