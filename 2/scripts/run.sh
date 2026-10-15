#!/usr/bin/env bash
# run bin/*-1core-<label> on every trace in parallel -> results/<label>/
# usage: scripts/run.sh <label>...
set -euo pipefail
cd "$(dirname "$0")/.."

TRACE_DIR=${TRACE_DIR:-Traces}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
JOBS=${JOBS:-$(( $(nproc) - 2 ))}
RESULTS=${RESULTS:-results}

[ $# -ge 1 ] || { echo "usage: $0 <label>..." >&2; exit 1; }

shopt -s nullglob
traces=("$TRACE_DIR"/*.xz "$TRACE_DIR"/*.gz)
[ ${#traces[@]} -gt 0 ] || { echo "no traces in $TRACE_DIR" >&2; exit 1; }

runs=()
for label in "$@"; do
    bins=(bin/*-1core-"$label")
    [ ${#bins[@]} -eq 1 ] || {
        echo "need exactly one bin/*-1core-$label" >&2; exit 1; }
    mkdir -p "$RESULTS/$label"
    for t in "${traces[@]}"; do
        out="$RESULTS/$label/$(basename "${t%.*}" .champsimtrace).txt"
        [ -s "$out" ] && grep -q "Region of Interest" "$out" && continue
        runs+=("${bins[0]}|$t|$out")
    done
done

echo "${#runs[@]} runs, $JOBS at a time"
[ ${#runs[@]} -gt 0 ] || exit 0

run_one() {
    IFS="|" read -r bin trace out <<< "$1"
    if "$bin" -warmup_instructions "$WARMUP" \
            -simulation_instructions "$SIM" \
            -traces "$trace" > "$out.tmp" 2>&1; then
        mv "$out.tmp" "$out" && echo "done   $out"
    else
        echo "FAILED $out (see $out.tmp)"
    fi
}
export -f run_one
export WARMUP SIM

printf '%s\n' "${runs[@]}" | xargs -r -d '\n' -P "$JOBS" -I{} \
    bash -c 'run_one "$1"' _ {}
