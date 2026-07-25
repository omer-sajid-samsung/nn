#!/usr/bin/env bash
# Runs the proposal-gap-closing sweeps built this session, in order:
#   0. wait for the in-flight w3-32k-headroom sweep (started earlier, still
#      running under its own nohup) to release the sweep lock
#   1. w3-32k-headroom  (resume — skips any cells already OK)   [context length, item D]
#   2. w1-concurrency-sweep-full: 1/8/16/100/250                [concurrency,   item E]
#   3. w4-turns-sweep: 5/10/20/40 turns/session                 [turns/session, item F]
#   4. w5-otel-agentic: real multi-agent trace replay           [agentic dataset, item F]
#   5. akb report for every sweep dir touched above
#
# Each step launches vLLM itself via --manage-vllm (agentic_kv_bench/vllm_proc.py)
# using the vllm_launch: block in that step's spec, and stops it before the next
# step starts — so w1's max_num_seqs=256 and w3's raised gpu_memory_utilization
# don't leak into each other's server config.
#
# --no-amoprof: this box's sudo needs an interactive password for `sudo -v`
# specifically (confirmed again this session, same quirk as the first overnight
# run), so unattended AMOProf would hang waiting on stdin. Drop --no-amoprof
# (and run this attended, once, to type the password) if you want full AMOProf
# storage telemetry on these sweeps instead of just vLLM/LMCache telemetry.
set -uo pipefail

cd /opt/ls/akb/agentic-kv-bench

VLLM_VENV="/opt/ls/vllm/.venv/bin/activate"
MASTER_LOG="/opt/ls/agentic_kv_bench/proposal_gaps_$(date +%Y%m%d_%H%M%S).log"
LOCK="/opt/ls/agentic_kv_bench/.sweep.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run() {
    log "RUN: $*"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "EXIT: $?"
}

akb_common=(--manage-vllm --vllm-venv "$VLLM_VENV" --vllm-boot-wait-s 300 --no-amoprof)

log "=== proposal-gap sweep starting ==="

# --- 0. wait out any sweep already holding the lock (the manual w3-32k run) ---
if [ -e "$LOCK" ] && ! flock -n -x "$LOCK" -c true 2>/dev/null; then
    log "sweep lock held by another process — waiting for it to finish..."
    while ! flock -n -x "$LOCK" -c true 2>/dev/null; do
        sleep 30
    done
    log "lock free, proceeding"
fi

# --- 1. context-length headroom (resume: skips i16000/i24000 if already OK) ---
run akb sweep --sweep sweeps/w3_32k_sweep.yaml "${akb_common[@]}"

# --- 2. concurrency 1/8/16/100/250 ---
run akb sweep --sweep sweeps/w1_concurrency_sweep.yaml "${akb_common[@]}"

# --- 3. turns-per-session 5/10/20/40 ---
run akb sweep --sweep sweeps/w4_turns_sweep.yaml "${akb_common[@]}"

# --- 4. real agentic dataset via OTel trace replay ---
run akb run --spec specs/w5_otel_agentic.yaml --sweep-name w5-otel-agentic "${akb_common[@]}"

# --- 5. aggregate reports ---
TODAY=$(date +%Y%m%d)
for d in "${TODAY}_w3-32k-headroom" "${TODAY}_w1-concurrency-sweep-full" \
         "${TODAY}_w4-turns-sweep" "${TODAY}_w5-otel-agentic"; do
    sd="/opt/ls/agentic_kv_bench/runs/$d"
    if [ -d "$sd" ]; then
        run akb report --sweep-dir "$sd"
    fi
done

log "=== proposal-gap sweep finished ==="
