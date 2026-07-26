#!/usr/bin/env bash
set -euo pipefail
cd "$LS/vllm"
source .venv/bin/activate

unset CUDA_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND
unset PYTORCH_CUDA_ALLOC_CONF
export LMCACHE_CONFIG_FILE="$LS/vllm/lmcache_disk_config.yaml"
export PYTHONHASHSEED=0
export VLLM_CACHE_ROOT=$LS/vllm-cache
export LMCACHE_INTERNAL_API_SERVER_ENABLED=true

if pgrep -f "vllm serve" > /dev/null; then
  echo "ERROR: vllm serve already running. Kill it first: pkill -9 -f 'vllm serve'"
  exit 1
fi

vllm serve Qwen/Qwen3-8B-AWQ \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8000 2>&1 | tee vllm_server.log
  # --no-enable-prefix-caching \