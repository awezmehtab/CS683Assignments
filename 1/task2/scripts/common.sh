#!/usr/bin/env bash
# common.sh  shared plumbing for the Task 2 experiment scripts. Sourced, not executed.
#
# Provides:
#   - path/env setup and a pinned runner (taskset)
#   - CPU vendor detection and LOGICAL -> ACTUAL perf event resolution, by probing
#     candidate names, so the same scripts work on the AMD Zen 3 dev box and on an Intel
#     server without editing anything
#   - perf_delta(), which measures a kernel's counters with the process setup subtracted out
#   - build_pb(), which builds a perf_bench variant with a given set of -D overrides
#
# Env overrides (all optional):
#   CORE=2            physical core to pin every measurement to
#   REPS=3            timed repetitions per measurement
#   WARMUP=1          untimed warmup repetitions
#   SIZES="..."       matrix sizes (M=N=K) for the size sweeps
#   RESULTS_DIR=...   where CSVs land   (default task2/results)
#   PLOTS_DIR=...     where PNGs land   (default <repo>/plots)
#   PERF_CHUNK=4      programmable events per perf run, to avoid counter multiplexing

set -euo pipefail

# ---------------------------------------------------------------- paths ----
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
REPO_DIR="$(cd "$TASK2_DIR/.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-$TASK2_DIR/results}"
PLOTS_DIR="${PLOTS_DIR:-$REPO_DIR/plots}"
BIN_DIR="$TASK2_DIR/bin"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$BIN_DIR"

# ------------------------------------------------------------- defaults ----
CORE="${CORE:-2}"
REPS="${REPS:-3}"
WARMUP="${WARMUP:-1}"
SIZES="${SIZES:-128 256 512 1024 2048}"
PERF_CHUNK="${PERF_CHUNK:-4}"
# Small, tile-straddling size used for the per-configuration correctness check. 257 is
# prime, so it lands badly against every block size and exercises all the cleanup tails.
VERIFY_SIZE="${VERIFY_SIZE:-257}"

# --------------------------------------------------------------- output ----
c_bold=$'\033[1m'; c_dim=$'\033[2m'; c_red=$'\033[31m'; c_yel=$'\033[33m'; c_off=$'\033[0m'
[ -t 1 ] || { c_bold=""; c_dim=""; c_red=""; c_yel=""; c_off=""; }

log()  { printf '%s%s%s\n' "$c_dim" "$*" "$c_off" >&2; }
info() { printf '%s%s%s\n' "$c_bold" "$*" "$c_off" >&2; }
warn() { printf '%swarning:%s %s\n' "$c_yel" "$c_off" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

# ------------------------------------------------------------- pinning -----
# Every measurement runs on the same core so migrations and per-core frequency differences
# do not show up as "results". Without this, sweep points differ by more than the effect
# being measured.
if command -v taskset >/dev/null 2>&1; then
    PIN=(taskset -c "$CORE")
else
    warn "taskset not found; measurements will not be core-pinned and will be noisier"
    PIN=()
fi

# -------------------------------------------------------------- vendor -----
cpu_vendor() {
    if grep -qm1 '^vendor_id.*AuthenticAMD' /proc/cpuinfo; then echo amd
    elif grep -qm1 '^vendor_id.*GenuineIntel' /proc/cpuinfo; then echo intel
    else echo unknown; fi
}
VENDOR="$(cpu_vendor)"

cpu_model() { sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1; }

has_flag() { grep -qm1 "\b$1\b" /proc/cpuinfo; }

# ---------------------------------------------------------------- perf -----
have_perf() { command -v perf >/dev/null 2>&1; }

# True only if perf can actually open an event here. paranoid=4 (or a container without
# CAP_PERFMON) makes every event fail, and the failure mode is a silent empty column, so
# this is checked up front rather than discovered halfway through a sweep.
perf_works() {
    have_perf || return 1
    perf stat -e instructions -x, -- true >/dev/null 2>&1
}

require_perf() {
    have_perf || die "perf not found. Install linux-tools for your kernel."
    if ! perf_works; then
        die "perf cannot open events here (perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')).
       Fix with:  sudo sysctl kernel.perf_event_paranoid=1
       or re-run this script under sudo."
    fi
}

# event_supported <name>  ->  0 if perf can open it
event_supported() {
    perf stat -e "$1" -x, -- true >/dev/null 2>&1
}

# Candidate lists per LOGICAL event, most specific first. Resolution probes each in order
# and takes the first that opens, so an Intel box picks the Intel name and an AMD box picks
# the AMD one. The generic kernel aliases (L1-dcache-load-misses, LLC-load-misses) come last
# because on several AMD parts they are either unmapped or map to something misleading.
_candidates() {
    case "$1" in
    l1d_miss)
        echo "ls_mab_alloc.load_store_allocations
MEM_LOAD_RETIRED.L1_MISS
mem_load_retired.l1_miss
L1-dcache-load-misses
ls_mab_alloc.all_allocations" ;;
    l1d_access)
        echo "ls_dispatch.ld_st_dispatch
MEM_INST_RETIRED.ALL_LOADS
mem_inst_retired.all_loads
L1-dcache-loads" ;;
    l2_miss)
        echo "l2_cache_misses_from_dc_misses
l2_rqsts.miss
L2_RQSTS.MISS
l2_cache_req_stat.ls_rd_blk_c" ;;
    llc_miss)
        echo "ls_any_fills_from_sys.mem_io_local
LONGEST_LAT_CACHE.MISS
longest_lat_cache.miss
LLC-load-misses
cache-misses" ;;
    # The PDF names Intel's sw_prefetch_access. The AMD equivalent is ls_pref_instr_disp,
    # which counts software prefetch instructions dispatched.
    sw_prefetch)
        echo "ls_pref_instr_disp.prefetch
ls_pref_instr_disp
sw_prefetch_access.any
sw_prefetch_access.t0
SW_PREFETCH_ACCESS.T0" ;;
    # AMD-only, and the most informative counter in the whole distance sweep: prefetches
    # that found their line already in L1, i.e. issued too late to have done any good.
    sw_prefetch_inef)
        echo "ls_inef_sw_pref.data_pipe_sw_pf_dc_hit" ;;
    # HARDWARE prefetcher activity. Experiment F uses this to prove the MSR write actually
    # took effect: a count that drops to ~0 is direct evidence the prefetchers are off,
    # rather than trusting that the write landed.
    hwpf)
        echo "l2_cache_accesses_from_l2_hwpf
ls_mab_alloc.hardware_prefetcher_allocations
l2_rqsts.all_pf
L2_RQSTS.ALL_PF" ;;
    *) die "unknown logical event '$1'" ;;
    esac
}

declare -A _EVENT_CACHE=()

# resolve_event <logical>  ->  prints the actual event name, or nothing if unsupported.
resolve_event() {
    local logical="$1"
    if [ -n "${_EVENT_CACHE[$logical]+set}" ]; then
        printf '%s' "${_EVENT_CACHE[$logical]}"
        return 0
    fi
    local found="" cand
    while read -r cand; do
        [ -n "$cand" ] || continue
        if event_supported "$cand"; then found="$cand"; break; fi
    done <<< "$(_candidates "$logical")"
    # On hybrid Intel parts (P + E cores) a bare event name opens on both PMUs and comes
    # back as <not counted>, which every caller then parses as a zero. Address the P-core
    # PMU explicitly so the counters actually land.
    if [ -d /sys/devices/cpu_core ] && [ -n "$found" ] && [[ "$found" != */* ]]; then
        found="cpu_core/$found/"
    fi
    _EVENT_CACHE[$logical]="$found"
    printf '%s' "$found"
}

# Resolve every logical event this suite uses and report the mapping once per run, so the
# report can state exactly which counter produced each number.
declare -A EV=()
resolve_all_events() {
    local l
    for l in l1d_miss l1d_access l2_miss llc_miss sw_prefetch sw_prefetch_inef hwpf; do
        EV[$l]="$(resolve_event "$l")"
    done
}

print_event_map() {
    printf '%-18s  %s\n' "LOGICAL" "RESOLVED EVENT"
    printf -- '----------------------------------------------------------------\n'
    local l
    for l in l1d_miss l1d_access l2_miss llc_miss sw_prefetch sw_prefetch_inef hwpf; do
        printf '%-18s  %s\n' "$l" "${EV[$l]:-<unsupported on this CPU>}"
    done
}

# --------------------------------------------------------------- build -----
# build_pb <out_basename> [extra -D flags...]
# Builds a perf_bench variant into bin/<out_basename>. Each configuration is a full rebuild
# because the -D flags change the meaning of the sources; object reuse would be wrong.
build_pb() {
    local out="$1"; shift
    local extra="$*"
    make -s -C "$TASK2_DIR/perf_bench" EXTRA="$extra" OUT="$BIN_DIR/$out" >/dev/null \
        || die "build failed for $out (EXTRA='$extra')"
    printf '%s' "$BIN_DIR/$out"
}

# assert_noprefetch <binary>
#
# The PREFETCH_DIST=0 build is the denominator of every speedup in experiments D, E and F.
# If a knob in src/matmul_*.cpp loses its #ifndef guard, the source's own #define silently
# overrides the -D, the prefetches stay in the binary, and the "no software prefetch"
# baseline is not a baseline at all: every sweep point would compare a kernel against
# itself and report roughly 1.0x while the CSV labels the rows as different distances.
# That failure is invisible in the timings, so it is checked against the disassembly, which
# cannot be fooled by it.
assert_noprefetch() {
    local bin="$1" n
    if ! command -v objdump >/dev/null 2>&1; then
        warn "objdump not found; cannot verify the no-prefetch baseline is really prefetch-free"
        return 0
    fi
    n="$(objdump -d "$bin" | grep -cE '\bprefetch(t0|t1|t2|nta)\b' || true)"
    [ "$n" -eq 0 ] && { log "  baseline verified: 0 prefetch instructions in $(basename "$bin")"; return 0; }
    die "the PREFETCH_DIST=0 build still contains $n prefetch instruction(s).
       A knob in src/matmul_*.cpp is missing its '#ifndef NAME / #define NAME / #endif'
       guard, so -DPREFETCH_DIST=0 was overridden by the source's own #define. Every
       speedup measured against this baseline would be meaningless. Fix the guard first."
}

# verify_pb <binary> <stage>  ->  "yes" | "NO"
# One correctness check per build configuration, at a tile-straddling size, run separately
# from any measured run so the naive reference never lands in the counters.
verify_pb() {
    local bin="$1" stage="$2" out
    out="$("${PIN[@]}" "$bin" --stage="$stage" --size="$VERIFY_SIZE" --reps=1 --verify 2>/dev/null)" || true
    printf '%s' "$(sed -n 's/.*correct=\([a-zA-Z/]*\).*/\1/p' <<< "$out")"
}

# ------------------------------------------------------------ measuring ----
# run_time <binary> <stage> <M> <N> <K>  ->  prints median time in ms
run_time() {
    local bin="$1" stage="$2" M="$3" N="$4" K="$5" out
    out="$("${PIN[@]}" "$bin" --stage="$stage" --M="$M" --N="$N" --K="$K" \
           --reps="$REPS" --warmup="$WARMUP")" || die "run failed: $bin --stage=$stage"
    sed -n 's/.*time_ms=\([0-9.e+-]*\).*/\1/p' <<< "$out"
}

# _perf_raw <events_csv> <binary> <stage> <M> <N> <K> <reps>
# Emits "event<TAB>count" lines. Unsupported/not-counted events are emitted as 0.
_perf_raw() {
    local events="$1" bin="$2" stage="$3" M="$4" N="$5" K="$6" reps="$7"
    perf stat -e "$events" -x, --  \
        "${PIN[@]}" "$bin" --stage="$stage" --M="$M" --N="$N" --K="$K" \
        --reps="$reps" --warmup="$WARMUP" 2>&1 >/dev/null \
    | awk -F, '
        NF >= 3 {
            v = $1; ev = $3
            if (v ~ /not (counted|supported)/ || v == "") v = 0
            gsub(/[^0-9.]/, "", v)
            if (v == "") v = 0
            # perf echoes the event with its modifier attached (cpu_core/instructions/u),
            # but callers look it up either as the full pmu form or as the bare name. Emit
            # both, otherwise every lookup misses and silently reads as zero.
            full = ev
            sub(/\/[ukhIGD]+$/, "/", full)
            bare = full
            sub(/^[a-z_]+\//, "", bare)
            sub(/\/$/, "", bare)
            sub(/:[ukhIGD]+$/, "", bare)
            print full "\t" v
            if (bare != full) print bare "\t" v
        }'
}

# perf_delta <binary> <stage> <M> <N> <K> <logical events...>
#
# Runs the binary TWICE: once doing $REPS kernel repetitions, once with --reps=0 (setup
# only, no kernel call), and reports the difference. That subtraction is what makes the
# numbers attributable to the kernel: without it, page faults from touching a 1024^3
# working set and the mt19937 fill would sit inside every count, and would dominate at the
# small sizes where the kernel itself is only a few milliseconds.
#
# Results land in the global associative array PD, keyed by LOGICAL name, plus
# PD[instructions], PD[cycles], and PD_TIME_MS.
declare -A PD=()
PD_TIME_MS=0
perf_delta() {
    local bin="$1" stage="$2" M="$3" N="$4" K="$5"; shift 5
    local logicals=("$@")
    PD=()

    # instructions/cycles sit on fixed counters on both vendors, so they ride along in
    # every chunk for free and never cause multiplexing.
    local fixed="instructions,cycles"

    # Only chunk the programmable events. Requesting more than the core has PMCs makes perf
    # time-multiplex and scale the results, which quietly adds error to every ratio.
    local -a prog=()
    local l
    for l in "${logicals[@]}"; do
        [ -n "${EV[$l]:-}" ] && prog+=("$l")
        PD[$l]=0
    done
    PD[instructions]=0; PD[cycles]=0

    local i=0 n=${#prog[@]}
    local first_chunk=1
    while :; do
        local -a chunk=()
        local j
        for ((j = 0; j < PERF_CHUNK && i < n; j++, i++)); do chunk+=("${prog[$i]}"); done

        local evlist="$fixed"
        for l in "${chunk[@]}"; do evlist="$evlist,${EV[$l]}"; done

        local -A hot=() cold=()
        local ev cnt
        while IFS=$'\t' read -r ev cnt; do hot[$ev]="$cnt"; done \
            < <(_perf_raw "$evlist" "$bin" "$stage" "$M" "$N" "$K" "$REPS")
        while IFS=$'\t' read -r ev cnt; do cold[$ev]="$cnt"; done \
            < <(_perf_raw "$evlist" "$bin" "$stage" "$M" "$N" "$K" 0)

        # instructions/cycles are re-measured per chunk; keep the first chunk's values so
        # every derived MPKI uses one consistent instruction count.
        if [ "$first_chunk" -eq 1 ]; then
            PD[instructions]=$(_sub "${hot[instructions]:-0}" "${cold[instructions]:-0}")
            PD[cycles]=$(_sub "${hot[cycles]:-0}" "${cold[cycles]:-0}")
            first_chunk=0
        fi
        for l in "${chunk[@]}"; do
            PD[$l]=$(_sub "${hot[${EV[$l]}]:-0}" "${cold[${EV[$l]}]:-0}")
        done

        [ "$i" -ge "$n" ] && break
    done

    PD_TIME_MS="$(run_time "$bin" "$stage" "$M" "$N" "$K")"
}

# Clamp at 0: the setup-only baseline can occasionally exceed the measured run by a hair on
# a low-count event, and a negative miss count in a CSV is worse than a zero.
_sub() { awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; printf "%.0f", (d > 0 ? d : 0) }'; }

# mpki <misses> <instructions>
mpki() { awk -v m="$1" -v i="$2" 'BEGIN { printf "%.4f", (i > 0 ? m * 1000.0 / i : 0) }'; }

# ratio <a> <b>  ->  a/b, 0 when b is 0
ratio() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4f", (b > 0 ? a / b : 0) }'; }

# ------------------------------------------------------------- banners -----
banner() {
    info ""
    info "================================================================"
    info " $*"
    info "================================================================"
}

# Warn once about the things that quietly widen error bars on a shared machine.
check_measurement_hygiene() {
    local gov
    gov="$(cat /sys/devices/system/cpu/cpu"$CORE"/cpufreq/scaling_governor 2>/dev/null || true)"
    if [ -n "$gov" ] && [ "$gov" != "performance" ]; then
        warn "cpu$CORE governor is '$gov', not 'performance'. Frequency ramping will show up"
        warn "  as run-to-run variance. Fix: sudo cpupower frequency-set -g performance"
    fi
    if [ "$(nproc)" -gt 0 ] && grep -qm1 '^' /sys/devices/system/cpu/smt/control 2>/dev/null; then
        local smt; smt="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || true)"
        [ "$smt" = "on" ] && log "note: SMT is on. Pinned to cpu$CORE; its sibling may still add noise."
    fi
}
