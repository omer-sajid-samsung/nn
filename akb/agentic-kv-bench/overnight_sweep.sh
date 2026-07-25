#!/usr/bin/env bash
# Overnight multi-workload sweep across w1-w4, sized for this box:
#   4x Tesla T4, GPU KV pool = 43,808 tokens, --max-model-len 16384.
# Not set -e: one bad run must not kill the rest of the night (matches the
# harness's own "a failed run is recorded and the sweep continues" design).
set -uo pipefail

cd /opt/ls/akb/agentic-kv-bench

VLLM_LOG="$LS/vllm/vllm_server.log"
MASTER_LOG="/opt/ls/agentic_kv_bench/overnight_sweep_$(date +%Y%m%d_%H%M%S).log"
SETTLE=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run() {
    log "RUN: $*"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "EXIT: $?"
}

log "=== overnight sweep starting ==="
akb preflight --vllm-log "$VLLM_LOG" 2>&1 | tee -a "$MASTER_LOG"

# -----------------------------------------------------------------------
# W1 — concurrency cell: N concurrent long sessions sharing the GPU.
# Aggregate live KV = concurrency x (4000+128) tokens; pool = 43808 ->
# expect IN_REGIME at low N, eviction once N*4128 > 43808 (N >= 11).
# -----------------------------------------------------------------------
run akb sweep --spec specs/w1_concurrency_cell.yaml --sweep-name w1-concurrency-sweep \
    --vllm-log "$VLLM_LOG" --concurrency-list "1 4 8 16" \
    --input-tokens 4000 --output-tokens 128 --unique-prompts 8 --num-requests 32 \
    --settle-s "$SETTLE"

# -----------------------------------------------------------------------
# W2 — revisit thrash: bounded unique-prompt pool > GPU pool, revisited.
# input=6000+128=6128 tokens/prompt; sweep unique_prompts across the
# pool ratio: u=4 (0.56x, IN_REGIME) .. u=16 (2.24x, heavy thrash).
# -----------------------------------------------------------------------
for u in 4 8 12 16; do
    run akb run --spec specs/w2_revisit_thrash.yaml --sweep-name w2-revisit-thrash-sweep \
        --vllm-log "$VLLM_LOG" \
        --input-tokens 6000 --output-tokens 128 --unique-prompts "$u" \
        --num-requests 48 --concurrency 8 --settle-s "$SETTLE"
done

# -----------------------------------------------------------------------
# W3 — single long context: one session, sweep prompt length up toward
# the server's max-model-len (16384), staying under it (input+output).
# -----------------------------------------------------------------------
for i in 2000 6000 10000 14000; do
    run akb run --spec specs/w3_single_longctx.yaml --sweep-name w3-single-longctx-sweep \
        --vllm-log "$VLLM_LOG" \
        --input-tokens "$i" --output-tokens 256 --unique-prompts 2 \
        --num-requests 4 --concurrency 1 --settle-s "$SETTLE"
done

# -----------------------------------------------------------------------
# W4 — agentic conversation: sweep the shared (env/tool) prefix length.
# dynamic suffix + 3 turns kept small so even s=8000 stays well under
# max-model-len as the session's context grows turn over turn.
# -----------------------------------------------------------------------
for s in 1000 2000 4000 8000; do
    run akb run --spec specs/w4_agentic_conversation.yaml --sweep-name w4-agentic-conversation-sweep \
        --vllm-log "$VLLM_LOG" \
        --input-tokens "$s" --output-tokens 150 \
        --set data.conversation_replay.num_conversations=8 \
        --set data.conversation_replay.dynamic_system_prompt_len.min=500 \
        --set data.conversation_replay.dynamic_system_prompt_len.max=500 \
        --set data.conversation_replay.dynamic_system_prompt_len.mean=500 \
        --set data.conversation_replay.turns_per_conversation.min=3 \
        --set data.conversation_replay.turns_per_conversation.max=3 \
        --set data.conversation_replay.turns_per_conversation.mean=3 \
        --set data.conversation_replay.input_tokens_per_turn.min=150 \
        --set data.conversation_replay.input_tokens_per_turn.max=350 \
        --set data.conversation_replay.input_tokens_per_turn.mean=250 \
        --num-requests 16 --concurrency 4 --settle-s "$SETTLE"
done

# -----------------------------------------------------------------------
# Aggregate reports
# -----------------------------------------------------------------------
TODAY=$(date +%Y%m%d)
for d in "${TODAY}_w1-concurrency-sweep" "${TODAY}_w2-revisit-thrash-sweep" \
         "${TODAY}_w3-single-longctx-sweep" "${TODAY}_w4-agentic-conversation-sweep"; do
    sd="/opt/ls/agentic_kv_bench/runs/$d"
    if [ -d "$sd" ]; then
        run akb report --sweep-dir "$sd"
    fi
done

log "=== overnight sweep finished ==="
