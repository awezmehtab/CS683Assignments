#!/usr/bin/env bash
# task 2 configs. usage: scripts/task2.sh [build|run|report|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
step=${1:-all}

IRRIP_VARIANTS=(
    "irrip_i1 INSTR_INSERT_RRPV=1"
    "irrip_noevict EVICT_DATA_FIRST=0"
    "irrip_d3 DATA_INSERT_RRPV=3"
)
# 2B threshold sweep: ratio only (no duel), data inserted at 2
THRESHOLDS=(10 25 40 55 70)
BASE_DYN="DUEL=0 DATA_INSERT_RRPV=2"

LABELS=(baseline rrip irrip irrip_i1 irrip_noevict irrip_d3)
for t in "${THRESHOLDS[@]}"; do LABELS+=("dyn_t$t"); done
LABELS+=(dyn_duel dyn_irrip)

build() {
    scripts/build.sh no lru baseline | grep -E "error|Binary:"
    for p in rrip irrip dyn_irrip; do
        scripts/build.sh no "$p" "$p" | grep -E "error|Binary:"
    done
    for v in "${IRRIP_VARIANTS[@]}"; do
        # shellcheck disable=SC2086
        scripts/variant.sh irrip $v
    done
    for t in "${THRESHOLDS[@]}"; do
        # shellcheck disable=SC2086
        scripts/variant.sh dyn_irrip "dyn_t$t" DYN_THRESHOLD_PCT="$t" $BASE_DYN
    done
    scripts/variant.sh dyn_irrip dyn_duel DUEL=1 DATA_INSERT_RRPV=2
}

report() {
    mkdir -p results plots
    scripts/parse.py results > results/summary.csv
    m="ipc speedup l2c_load_mpki l2c_instr_load_mpki l2c_data_load_mpki"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/plot.py results/summary.csv plots/task2a \
        baseline=LRU,rrip=SRRIP,irrip=I-RRIP $m
    "$PYTHON" scripts/plot.py results/summary.csv plots/task2a_variants \
        rrip=SRRIP,irrip=I-RRIP,irrip_i1=I-RRIP_i1,irrip_noevict=I-RRIP_noevict,irrip_d3=I-RRIP_d3 \
        $m
    "$PYTHON" scripts/plot.py results/summary.csv plots/task2b \
        baseline=LRU,rrip=SRRIP,irrip=I-RRIP,dyn_t25=Dyn-ratio,dyn_duel=Dyn+duel,dyn_irrip=Dyn+duel+d3 \
        $m
    "$PYTHON" scripts/plot.py results/summary.csv plots/task2b_threshold \
        dyn_t10=10%,dyn_t25=25%,dyn_t40=40%,dyn_t55=55%,dyn_t70=70% $m
}

case $step in
    build) build ;;
    run) scripts/run.sh "${LABELS[@]}" ;;
    report) report ;;
    all) build; scripts/run.sh "${LABELS[@]}"; report ;;
    *) echo "usage: $0 [build|run|report|all]" >&2; exit 1 ;;
esac
