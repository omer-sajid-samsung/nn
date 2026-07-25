#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/opt/ls/agentic_kv_bench/runs/qwen3-8b-awq_lmcache_sweep_20260724/c008"
CYCLES_DIR="${RUN_DIR}/amoprof"
OUTPUT_DIR="${CYCLES_DIR}/aggregated_full_sweep"
RUN_LABEL="qwen3-8b-awq_lmcache_sweep_20260724_c008"
SETUP_DETAILS="/opt/ls/setup_details.json"

mapfile -t CYCLES < <(ls -d "${CYCLES_DIR}"/service_cycle_* 2>/dev/null)

if [ "${#CYCLES[@]}" -lt 2 ]; then
    echo "error: found ${#CYCLES[@]} service_cycle_* dirs under ${CYCLES_DIR} — need at least 2" >&2
    exit 1
fi

echo "Aggregating ${#CYCLES[@]} cycles from ${CYCLES_DIR}"

amoprof aggregate \
    --run-dirs "${CYCLES[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-label "${RUN_LABEL}" \
    --setup-details "${SETUP_DETAILS}" \
    --combined-report

echo "Done. Combined report: ${OUTPUT_DIR}/${RUN_LABEL}/amoprof_combined.html"
