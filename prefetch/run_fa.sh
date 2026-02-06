#!/usr/bin/env bash
set -euo pipefail

# Host-side helper: run everything inside the already-running `fa` container.
# No rm is used; output is written into a timestamped folder.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

CONTAINER_DIR="/tmp/cutest_prefetch_run_${TS}"
OUT_HOST="${ROOT_DIR}/out/${TS}"

mkdir -p "${OUT_HOST}"

echo "[host] Copying ${ROOT_DIR} -> fa:${CONTAINER_DIR}"
docker exec fa bash -lc "mkdir -p '${CONTAINER_DIR}'"
docker cp "${ROOT_DIR}/." "fa:${CONTAINER_DIR}/"

echo "[fa] Building + running sweeps..."
docker exec fa bash -lc "cd '${CONTAINER_DIR}' && make -j && python3 scripts/run_all.py --device 0 --out-dir '${CONTAINER_DIR}/out/${TS}'"

echo "[host] Copying results back -> ${OUT_HOST}"
docker cp "fa:${CONTAINER_DIR}/out/${TS}/." "${OUT_HOST}/"

echo "Done. See: ${OUT_HOST}"

