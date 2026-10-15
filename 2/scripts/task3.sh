#!/usr/bin/env bash
# task 3 configs. usage: PREF=spp scripts/task3.sh [build|run|report|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
PREF=${PREF:-spp}
step=${1:-all}

LABELS=("${PREF}_lru" "${PREF}_dyn_nopf" "${PREF}_dyn" "${PREF}_dyn_pf2")

build() {
    scripts/build.sh "$PREF" lru "${PREF}_lru" | grep -E "error|Binary:"
    scripts/build.sh "$PREF" dyn_irrip "${PREF}_dyn" | grep -E "error|Binary:"
    L1D=$PREF scripts/variant.sh dyn_irrip "${PREF}_dyn_nopf" PF_AWARE=0
    L1D=$PREF scripts/variant.sh dyn_irrip "${PREF}_dyn_pf2" PF_INSERT_RRPV=2
}

report() {
    mkdir -p results plots
    scripts/parse.py results > results/summary.csv
    "$PYTHON" scripts/plot.py results/summary.csv "plots/task3_$PREF" \
        "baseline=LRU,dyn_irrip=Dyn-I-RRIP,${PREF}_lru=$PREF+LRU,${PREF}_dyn_nopf=$PREF+Dyn-I-RRIP,${PREF}_dyn=$PREF+PF-aware,${PREF}_dyn_pf2=$PREF+PF-aware(pf@2)" \
        ipc speedup l1d_load_mpki l1d_pf_accuracy l1d_coverage \
        l2c_load_mpki l2c_instr_load_mpki l2c_data_load_mpki
}

case $step in
    build) build ;;
    run) scripts/run.sh "${LABELS[@]}" ;;
    report) report ;;
    all) build; scripts/run.sh "${LABELS[@]}"; report ;;
    *) echo "usage: $0 [build|run|report|all]" >&2; exit 1 ;;
esac
