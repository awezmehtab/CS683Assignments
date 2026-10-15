#!/usr/bin/env bash
# build ChampSim with GCC 7 in a container (Ubuntu 18.04)
# usage: scripts/build.sh <L1D prefetcher> <L2C replacement> <binary name>
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-localhost/champsim-gcc7}
ENGINE=${ENGINE:-podman}

if ! "$ENGINE" image exists "$IMAGE" 2>/dev/null; then
    "$ENGINE" build -t "$IMAGE" -f scripts/Containerfile scripts >&2
fi

exec "$ENGINE" run --rm -v "$PWD":/champsim:Z -w /champsim "$IMAGE" \
    ./build_champsim.sh "$@"
