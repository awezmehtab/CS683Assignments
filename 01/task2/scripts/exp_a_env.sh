#!/usr/bin/env bash
# exp_a_env.sh  EXPERIMENT A: environment capture and perf event resolution.
#
# Not a measurement. Run this FIRST. It does three things:
#
#   1. Records the machine configuration the report has to state (CPU, cache sizes, ISA
#      support, governor, SMT) into results/env.txt.
#   2. Probes which perf events actually exist here and prints the LOGICAL -> ACTUAL
#      mapping. This matters because the assignment PDF names Intel events
#      (sw_prefetch_access) that do not exist on AMD, and the generic kernel aliases
#      (L1-dcache-load-misses) are unmapped on several AMD parts. The report needs to say
#      which counter each number came from.
#   3. Fails loudly if perf cannot open events at all, rather than letting every later
#      script emit empty columns.
#
# Usage:  ./exp_a_env.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$RESULTS_DIR/env.txt"

banner "EXPERIMENT A  environment and perf event resolution"

require_perf
resolve_all_events
check_measurement_hygiene

{
    echo "CS683 PA-1 Task 2  measurement environment"
    echo "captured: $(date -Is)"
    echo "host:     $(hostname)"
    echo
    echo "== CPU =="
    echo "vendor:      $VENDOR"
    echo "model:       $(cpu_model)"
    echo "cores:       $(nproc) logical"
    lscpu | grep -E '^(Thread|Core|Socket|CPU max|CPU min|Model name)' || true
    echo
    echo "== Cache hierarchy =="
    lscpu | grep -i cache || true
    echo
    echo "== Relevant ISA support =="
    for f in sse4_2 avx avx2 fma avx512f; do
        if has_flag "$f"; then echo "  $f: yes"; else echo "  $f: NO"; fi
    done
    echo
    echo "== Measurement configuration =="
    echo "pinned core:   $CORE"
    echo "governor:      $(cat /sys/devices/system/cpu/cpu"$CORE"/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "SMT:           $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"
    echo "reps/warmup:   $REPS / $WARMUP"
    echo "perf paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
    echo "perf version:  $(perf --version 2>/dev/null || echo unknown)"
    echo
    echo "== perf event resolution =="
    echo "The scripts address counters by LOGICAL name and resolve them by probing"
    echo "candidate vendor-specific names in order. Quote this table in the report."
    echo
    print_event_map
    echo
    echo "== Compiler =="
    g++ --version | head -1
    echo "pinned flags: -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma"
} | tee "$OUT"

# Fail early on the two events without which experiments C and D lose their point.
missing=""
[ -z "${EV[l1d_miss]}" ] && missing="$missing l1d_miss"
[ -z "${EV[sw_prefetch]}" ] && missing="$missing sw_prefetch"
if [ -n "$missing" ]; then
    warn "no usable counter for:$missing"
    warn "  Experiments C/D will still record timing, but their cache columns will be zero."
    warn "  Add a candidate name to _candidates() in common.sh if your CPU calls it something else."
fi

if ! has_flag avx512f; then
    log ""
    log "note: no AVX-512 on this CPU. Experiment G will report the 512-bit variant as"
    log "      'unsupported' rather than measuring it, which is the correct outcome to"
    log "      report for deliverable 2B, not a gap."
fi

info ""
info "wrote $OUT"
