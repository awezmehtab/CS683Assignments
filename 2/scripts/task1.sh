#!/usr/bin/env bash
# task 1 configs. usage: scripts/task1.sh [build|run|report|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
step=${1:-all}

SPP_VARIANTS=(
    "spp_unionly NU_CONF=4"
    "spp_uni2 UNI_CONF=2"
    "spp_fixed4 DEG_INIT=4 DEG_LO=4 DEG_MID=4 DEG_HI=4 DEG_MAX=4"
    "spp_aggr DEG_LO=2 DEG_MID=4 DEG_HI=6 DEG_MAX=8"
    "spp_cons ACC_LO=40 ACC_MID=60 ACC_HI=80"
)

LABELS=(baseline ip_stride_lru spp)
for v in "${SPP_VARIANTS[@]}"; do LABELS+=("${v%% *}"); done

build() {
    scripts/build.sh no lru baseline | grep -E "error|Binary:"
    scripts/build.sh ip_stride lru ip_stride_lru | grep -E "error|Binary:"
    scripts/build.sh spp lru spp | grep -E "error|Binary:"
    for v in "${SPP_VARIANTS[@]}"; do
        # shellcheck disable=SC2086
        KIND=l1d_pref scripts/variant.sh spp $v
    done
}

report() {
    mkdir -p results plots
    scripts/parse.py results > results/summary.csv
    m="ipc speedup l1d_load_mpki l1d_pf_accuracy l1d_coverage"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/plot.py results/summary.csv plots/task1 \
        baseline=No-prefetch,ip_stride_lru=IP-stride,spp=SPP $m
    "$PYTHON" scripts/plot.py results/summary.csv plots/task1_variants \
        spp=SPP,spp_unionly=uniform-only,spp_uni2=uni-conf-2,spp_fixed4=fixed-deg-4,spp_aggr=aggressive,spp_cons=conservative \
        $m
}

case $step in
    build) build ;;
    run) scripts/run.sh "${LABELS[@]}" ;;
    report) report ;;
    all) build; scripts/run.sh "${LABELS[@]}"; report ;;
    *) echo "usage: $0 [build|run|report|all]" >&2; exit 1 ;;
esac
