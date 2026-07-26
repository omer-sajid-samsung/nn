#!/usr/bin/env bash
# Additional proposal-relevant tests, run after run_proposal_gaps.sh finishes:
#   1. w2 revisit-thrash, heaviest cell (u=16), WITH real AMOProf telemetry
#      enabled -- the SudoKeepalive bug (bare `sudo -v` needing an interactive
#      password even with a cached ticket / command-scoped NOPASSWD) is fixed
#      in amoprof.py now, so this is the first run all session with real
#      blktrace/biosnoop/smart NAND-tier telemetry instead of --no-amoprof.
#   2. w6-bigctx-thrash: revisit thrash at ~20k tokens/prompt using the raised
#      234,640-token pool -- forces real eviction + SSD reads AT context
#      lengths near the proposal's 32k floor (w3-32k-headroom only proved the
#      context length boots; this proves eviction actually happens there too).
#   3. w5-otel-traces-sweep: the other two official example agentic traces
#      (code-review workflow, customer-support escalation), for dataset
#      diversity beyond the single multi-agent-research trace already run.
#   4. Aggregate reports.
set -uo pipefail

cd /opt/ls/akb/agentic-kv-bench

VLLM_VENV="/opt/ls/vllm/.venv/bin/activate"
MASTER_LOG="/opt/ls/agentic_kv_bench/additional_tests_$(date +%Y%m%d_%H%M%S).log"
LOCK="/opt/ls/agentic_kv_bench/.sweep.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run() {
    log "RUN: $*"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "EXIT: $?"
}

log "=== additional proposal-relevant tests starting ==="

if [ -e "$LOCK" ] && ! flock -n -x "$LOCK" -c true 2>/dev/null; then
    log "sweep lock held by another process — waiting for it to finish..."
    while ! flock -n -x "$LOCK" -c true 2>/dev/null; do
        sleep 30
    done
    log "lock free, proceeding"
fi

# --- 1. w2 revisit-thrash, heaviest cell, WITH real AMOProf (blktrace/biosnoop/smart) ---
run akb run --spec specs/w2_revisit_thrash.yaml --sweep-name w2-revisit-thrash-amoprof \
    --manage-vllm --vllm-venv "$VLLM_VENV" --vllm-boot-wait-s 300 \
    --input-tokens 6000 --unique-prompts 16 --num-requests 48 --concurrency 8 --settle-s 60

# --- 2. eviction at large (~20k) context, WITH real AMOProf too ---
run akb run --spec specs/w6_bigctx_thrash.yaml --sweep-name w6-bigctx-thrash \
    --manage-vllm --vllm-venv "$VLLM_VENV" --vllm-boot-wait-s 300 --settle-s 60

# --- 3. additional agentic traces ---
run akb sweep --sweep sweeps/w5_otel_traces_sweep.yaml \
    --manage-vllm --vllm-venv "$VLLM_VENV" --vllm-boot-wait-s 300 --no-amoprof

# --- 4. aggregate reports ---
TODAY=$(date +%Y%m%d)
for d in "${TODAY}_w2-revisit-thrash-amoprof" "${TODAY}_w6-bigctx-thrash" "${TODAY}_w5-otel-traces-sweep"; do
    sd="/opt/ls/agentic_kv_bench/runs/$d"
    if [ -d "$sd" ]; then
        run akb report --sweep-dir "$sd"
    fi
done

log "=== additional tests finished ==="
