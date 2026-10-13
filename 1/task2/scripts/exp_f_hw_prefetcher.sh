#!/usr/bin/env bash
# exp_f_hw_prefetcher.sh  EXPERIMENT F: hardware prefetchers on vs off.
#
# Deliverable 2A.5 ("enable and disable hardware prefetchers, and analyze their impact").
#
# Runs a 2x2:  {HW prefetch on, off} x {SW prefetch on, off}
#
# The point of the 2x2 rather than a simple on/off pair: software prefetching usually looks
# unimpressive on a modern core precisely BECAUSE the hardware prefetcher already caught the
# unit-stride access pattern the kernel has. Turning the hardware prefetchers off exposes
# how much work the software prefetches were actually doing. The interaction term is the
# result worth reporting, and you cannot see it from either variable alone.
#
# REQUIRES ROOT. It writes a model-specific register:
#
#   Intel   MSR 0x1A4 (MSR_MISC_FEATURE_CONTROL)
#             bit 0  L2 hardware prefetcher disable
#             bit 1  L2 adjacent cache line prefetcher disable
#             bit 2  DCU (L1 data) prefetcher disable
#             bit 3  DCU IP prefetcher disable
#           -> 0xF disables all four, 0x0 enables all four
#
#   AMD     MSR 0xC0000108 (prefetch control, family 17h/19h)
#             bit 0  L1 stream disable      bit 3  L2 stream disable
#             bit 1  L1 stride disable      bit 4  up/down disable
#             bit 2  L1 region disable
#           -> 0x1F disables all five, 0x0 enables all five
#
# The original value is read back and restored by an EXIT trap, so an interrupt or a failed
# build leaves the machine as it found it. The write is applied ONLY to the pinned core
# (wrmsr -p), not system-wide, since every measurement here is pinned anyway.
#
# The hwpf counter is recorded in every cell as proof the toggle landed: if the "off" rows
# do not show that count collapsing, the MSR write did not take effect (some BIOSes and
# most virtualised/cloud instances block it) and the rest of the table is meaningless.
#
# Setup:  sudo apt install msr-tools && sudo modprobe msr
# Usage:  sudo -E ./exp_f_hw_prefetcher.sh
#         sudo -E SIZES="1024 2048" PF_DIST=64 ./exp_f_hw_prefetcher.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PF_DIST="${PF_DIST:-64}"
SIZES="${SIZES:-512 1024 2048}"
STAGES="${STAGES:-simd prefetch optimized}"
CSV="$RESULTS_DIR/f_hw_prefetcher.csv"

banner "EXPERIMENT F  hardware prefetchers on vs off"

# ------------------------------------------------------------ preflight ----
[ "$(id -u)" -eq 0 ] || die "must run as root (MSR writes). Try: sudo -E $0"
require_perf
resolve_all_events

command -v rdmsr >/dev/null 2>&1 || die "rdmsr not found. Install with: apt install msr-tools"
command -v wrmsr >/dev/null 2>&1 || die "wrmsr not found. Install with: apt install msr-tools"

if [ ! -e /dev/cpu/"$CORE"/msr ]; then
    modprobe msr 2>/dev/null || true
    [ -e /dev/cpu/"$CORE"/msr ] || die "/dev/cpu/$CORE/msr missing. Try: modprobe msr"
fi

case "$VENDOR" in
intel) MSR=0x1A4;       DISABLE_VAL=0xF  ;;
amd)   MSR=0xC0000108;  DISABLE_VAL=0x1F ;;
*)     die "unknown CPU vendor '$VENDOR'; no known prefetch-control MSR" ;;
esac

log "vendor=$VENDOR  prefetch-control MSR=$MSR  disable-all value=$DISABLE_VAL  core=$CORE"

ORIG="$(rdmsr -p "$CORE" "$MSR" 2>/dev/null)" \
    || die "cannot read MSR $MSR on core $CORE. Virtualised hosts usually block this."
log "original MSR value: 0x$ORIG"

# Restore on ANY exit path, including Ctrl-C and a mid-sweep failure. Leaving a machine with
# its prefetchers disabled would silently skew every later measurement on it.
restore_msr() {
    wrmsr -p "$CORE" "$MSR" "0x$ORIG" 2>/dev/null \
        && log "restored MSR $MSR on core $CORE to 0x$ORIG" \
        || warn "FAILED to restore MSR $MSR on core $CORE. Set it back manually: wrmsr -p $CORE $MSR 0x$ORIG"
}
trap restore_msr EXIT INT TERM

set_hwpf() {  # set_hwpf on|off
    case "$1" in
    on)  wrmsr -p "$CORE" "$MSR" 0x0 ;;
    off) wrmsr -p "$CORE" "$MSR" "$DISABLE_VAL" ;;
    esac
    local now; now="$(rdmsr -p "$CORE" "$MSR")"
    log "  hwpf=$1  MSR now 0x$now"
}

# ---------------------------------------------------------------- build ----
log "building SW-prefetch-on (dist=$PF_DIST) and SW-prefetch-off (dist=0) variants..."
BIN_SWON="$(build_pb pb_swon "-DPREFETCH_DIST=$PF_DIST")"
BIN_SWOFF="$(build_pb pb_d0 "-DPREFETCH_DIST=0")"
assert_noprefetch "$BIN_SWOFF"

# --------------------------------------------------------------- sweep -----
echo "stage,size,hwpf,swpf,time_ms,instructions,cycles,ipc,l1d_miss,l1d_mpki,l2_miss,l2_mpki,llc_miss,llc_mpki,hwpf_events,sw_prefetch" > "$CSV"

printf '\n%-10s  %-6s  %-5s  %-5s  %10s  %10s  %10s  %14s\n' \
       "stage" "size" "hwpf" "swpf" "time(ms)" "L1D MPKI" "LLC MPKI" "hwpf events" >&2
printf -- '------------------------------------------------------------------------------------\n' >&2

for hw in on off; do
    set_hwpf "$hw"
    for stage in $STAGES; do
        for n in $SIZES; do
            for sw in on off; do
                if [ "$sw" = "on" ]; then bin="$BIN_SWON"; else bin="$BIN_SWOFF"; fi

                perf_delta "$bin" "$stage" "$n" "$n" "$n" l1d_miss l2_miss llc_miss hwpf sw_prefetch

                ipc="$(ratio "${PD[instructions]}" "${PD[cycles]}")"
                l1_mpki="$(mpki "${PD[l1d_miss]}" "${PD[instructions]}")"
                l2_mpki="$(mpki "${PD[l2_miss]}" "${PD[instructions]}")"
                llc_mpki="$(mpki "${PD[llc_miss]}" "${PD[instructions]}")"

                printf '%-10s  %-6s  %-5s  %-5s  %10.3f  %10s  %10s  %14s\n' \
                       "$stage" "$n" "$hw" "$sw" "$PD_TIME_MS" "$l1_mpki" "$llc_mpki" \
                       "${PD[hwpf]}" >&2
                echo "$stage,$n,$hw,$sw,$PD_TIME_MS,${PD[instructions]},${PD[cycles]},$ipc,${PD[l1d_miss]},$l1_mpki,${PD[l2_miss]},$l2_mpki,${PD[llc_miss]},$llc_mpki,${PD[hwpf]},${PD[sw_prefetch]}" >> "$CSV"
            done
        done
    done
    printf -- '------------------------------------------------------------------------------------\n' >&2
done

info ""
info "wrote $CSV"

# ------------------------------------------------------------ sanity -------
# If disabling the prefetchers did not move the hardware-prefetch counter, the MSR write was
# silently ignored and nothing in the table above means what it claims to.
awk -F, '
    NR > 1 { sum[$3] += $15; n[$3]++ }
    END {
        on = (n["on"] ? sum["on"] / n["on"] : 0)
        off = (n["off"] ? sum["off"] / n["off"] : 0)
        printf "\nhardware-prefetch events, mean per run:  hwpf=on %.0f   hwpf=off %.0f\n", on, off
        if (on > 0 && off > 0.25 * on)
            print "WARNING: the off rows did not collapse. The MSR write was likely ignored\n" \
                  "         (virtualised host, or BIOS lockout). Do not report this table."
        else if (on > 0)
            print "ok: the toggle took effect."
        else
            print "note: no hardware-prefetch counter on this CPU; toggle could not be verified."
    }' "$CSV" >&2

info ""
info "speedup from SW prefetch, split by hardware prefetcher state:"
awk -F, 'NR>1 { t[$1","$2","$3","$4] = $5 }
         END {
             for (k in t) {
                 split(k, f, ",")
                 if (f[4] == "on") {
                     off = f[1] "," f[2] "," f[3] ",off"
                     if (off in t && t[k] > 0)
                         printf "  %-10s n=%-5s hwpf=%-4s  swpf speedup = %.3fx\n",
                                f[1], f[2], f[3], t[off] / t[k]
                 }
             }
         }' "$CSV" | sort >&2
