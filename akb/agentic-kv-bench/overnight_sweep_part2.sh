#!/usr/bin/env bash
# Part 2 of the overnight sweep: W2, W3, W4 (W1 already completed OK).
#
# Switched to --no-sudo here: this box's sudo requires an interactive
# password for `sudo -v` specifically (even with NOPASSWD:ALL and a fresh
# cached ticket -- confirmed by hand), so AMOProfService's SudoKeepalive
# hangs at the start of every single `akb run`/`akb sweep` invocation
# waiting on stdin that will never come unattended. --no-sudo drops the
# root-only collectors (blktrace/biosnoop) but keeps GPU/DRAM/vmstat/iostat
# and all vLLM/LMCache telemetry, which is what the verdict logic needs.
set -uo pipefail

cd /opt/ls/akb/agentic-kv-bench

VLLM_LOG="$LS/vllm/vllm_server.log"
MASTER_LOG="/opt/ls/agentic_kv_bench/overnight_sweep_part2_$(date +%Y%m%d_%H%M%S).log"
SETTLE=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run() {
    log "RUN: $*"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "EXIT: $?"
}

log "=== overnight sweep part 2 starting (W2, W3, W4; --no-sudo) ==="

# W2 — revisit thrash: sweep unique_prompts across the pool ratio.
for u in 4 8 12 16; do
    run akb run --spec specs/w2_revisit_thrash.yaml --sweep-name w2-revisit-thrash-sweep \
        --vllm-log "$VLLM_LOG" --no-sudo \
        --input-tokens 6000 --output-tokens 128 --unique-prompts "$u" \
        --num-requests 48 --concurrency 8 --settle-s "$SETTLE"
done

# W3 — single long context: sweep prompt length toward max-model-len.
for i in 2000 6000 10000 14000; do
    run akb run --spec specs/w3_single_longctx.yaml --sweep-name w3-single-longctx-sweep \
        --vllm-log "$VLLM_LOG" --no-sudo \
        --input-tokens "$i" --output-tokens 256 --unique-prompts 2 \
        --num-requests 4 --concurrency 1 --settle-s "$SETTLE"
done

# W4 — agentic conversation: sweep the shared (env/tool) prefix length.
for s in 1000 2000 4000 8000; do
    run akb run --spec specs/w4_agentic_conversation.yaml --sweep-name w4-agentic-conversation-sweep \
        --vllm-log "$VLLM_LOG" --no-sudo \
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

# Aggregate reports for all four sweeps (W1 included).
TODAY=$(date +%Y%m%d)
for d in "${TODAY}_w1-concurrency-sweep" "${TODAY}_w2-revisit-thrash-sweep" \
         "${TODAY}_w3-single-longctx-sweep" "${TODAY}_w4-agentic-conversation-sweep"; do
    sd="/opt/ls/agentic_kv_bench/runs/$d"
    if [ -d "$sd" ]; then
        run akb report --sweep-dir "$sd"
    fi
done

log "=== overnight sweep part 2 finished ==="
