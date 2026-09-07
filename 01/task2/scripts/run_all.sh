#!/usr/bin/env bash
# run_all.sh  runs experiments A through I in dependency order, then plots everything.
#
# The order matters. D finds the best prefetch distance, and E/F both need that distance
# held fixed while they vary something else, so this script parses D's result and threads it
# into E and F rather than leaving the default in place. Running the scripts by hand in a
# different order is fine, but then pass PF_DIST=<n> to E and F yourself.
#
# Experiments that cannot run here are SKIPPED with a reason rather than failing the whole
# sweep, so a machine without root still produces everything except F.
#
# Usage:
#   ./run_all.sh                 # everything that can run
#   sudo -E ./run_all.sh         # including F (hardware prefetcher toggle)
#   SKIP="F I" ./run_all.sh      # skip named experiments
#   QUICK=1 ./run_all.sh         # small sizes / coarse sweeps, for a smoke test
#
# Total runtime at the defaults is dominated by matmul_naive in B and C and by the rebuild
# count in E's 4x4 grid and H's block sweep. Budget a couple of hours; QUICK=1 takes minutes.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SKIP="${SKIP:-}"
QUICK="${QUICK:-0}"

if [ "$QUICK" = "1" ]; then
    export SIZES="${SIZES:-128 256 512}"
    export REPS="${REPS:-2}"
    export DISTS="${DISTS:-0 16 64 256}"
    export BM_LIST="${BM_LIST:-32 64}" BN_LIST="${BN_LIST:-32 64}" BK_LIST="${BK_LIST:-128 256}"
    export TUNE_SIZE="${TUNE_SIZE:-512}"
    export GRID_SIZE="${GRID_SIZE:-512}"
    export REPEATS="${REPEATS:-1}"
    info "QUICK=1: reduced sizes and sweeps, for validating the pipeline rather than reporting"
fi

skipped() { grep -qw "$1" <<< "$SKIP"; }

run_exp() {  # run_exp <letter> <script> [precondition-message]
    local letter="$1" script="$2"
    if skipped "$letter"; then
        warn "skipping experiment $letter (in SKIP)"
        return 0
    fi
    info ""
    if "$SCRIPTS_DIR/$script"; then
        info "experiment $letter: done"
    else
        warn "experiment $letter FAILED (see the message above). Continuing with the rest."
        FAILED="$FAILED $letter"
    fi
}

FAILED=""
START="$(date +%s)"

run_exp A exp_a_env.sh
run_exp B exp_b_speedup_vs_size.sh
run_exp C exp_c_cache_metrics.sh
run_exp D exp_d_prefetch_distance.sh

# Thread D's answer into E and F. Picking the best distance for the LARGEST size measured,
# since that is the DRAM-bound case where prefetching actually matters; the small sizes are
# cache-resident and their optimum is mostly noise.
if [ -z "${PF_DIST:-}" ] && [ -f "$RESULTS_DIR/d_prefetch_distance.csv" ]; then
    PF_DIST="$(awk -F, '
        NR > 1 && $3 != 0 && $1 == "optimized" {
            if ($2 + 0 > maxn) maxn = $2 + 0
        }
        NR > 1 && $3 != 0 && $1 == "optimized" && $2 + 0 == maxn && $6 + 0 > best {
            best = $6 + 0; d = $3
        }
        END { if (d != "") print d }' "$RESULTS_DIR/d_prefetch_distance.csv")"
    if [ -n "$PF_DIST" ]; then
        export PF_DIST
        info ""
        info "carrying experiment D's best prefetch distance into E and F: PREFETCH_DIST=$PF_DIST"
    fi
fi

run_exp E exp_e_prefetch_level.sh

if skipped F; then
    warn "skipping experiment F (in SKIP)"
elif [ "$(id -u)" -ne 0 ]; then
    warn "skipping experiment F: needs root for the prefetch-control MSR."
    warn "  Run it separately with: sudo -E $SCRIPTS_DIR/exp_f_hw_prefetcher.sh"
    FAILED="$FAILED F(no-root)"
else
    run_exp F exp_f_hw_prefetcher.sh
fi

run_exp G exp_g_simd_width.sh
run_exp H exp_h_block_size.sh
run_exp I exp_i_llama.sh

# ---------------------------------------------------------------- plots ----
info ""
banner "plotting"
python3 "$SCRIPTS_DIR/plot_all.py" || warn "plotting failed"

# --------------------------------------------------------------- summary ---
ELAPSED=$(( $(date +%s) - START ))
banner "done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
info "CSVs:  $RESULTS_DIR"
info "plots: $PLOTS_DIR"
[ -n "$FAILED" ] && warn "did not complete:$FAILED"
info ""
info "the three figures the submission tree names explicitly:"
for f in task2_speedup_vs_size.png task2_prefetch_degree.png task2_prefetch_level.png; do
    if [ -f "$PLOTS_DIR/$f" ]; then
        info "  [ok]      $f"
    else
        warn "  [missing] $f"
    fi
done
