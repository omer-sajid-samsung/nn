set -euo pipefail
LS=/opt/ls
VLLM_DIR="$LS/vllm"
cd "$VLLM_DIR"
source .venv/bin/activate

# pinned to the combo you've already run successfully on T4
uv pip install "vllm==0.11.0" "lmcache==0.3.7" "transformers==4.57.0" requests

# sanity: torch must see the T4 and support sm_75
# python - <<'PY'
# import torch
# assert torch.cuda.is_available(), "torch can't see any GPU"
# archs = torch.cuda.get_arch_list()
# print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| archs:", archs)
# assert any("7.5" in a for a in archs), "sm_75 missing from this torch build"
# import vllm, lmcache
# print("vllm", vllm.__version__)
# PY

# lmcache disk config — KEEP max_local_cpu_size: 5.0, removing it crashes 0.3.7
mkdir -p "$LS/lmcache-disk-cache"
cat > lmcache_disk_config.yaml <<EOF
chunk_size: 256
local_cpu: false
max_local_cpu_size: 5.0
local_disk: "file://$LS/lmcache-disk-cache/"
max_local_disk_size: 20.0
EOF

cat > serve.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$VLLM_DIR"
source .venv/bin/activate

unset CUDA_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND
export LMCACHE_CONFIG_FILE="$VLLM_DIR/lmcache_disk_config.yaml"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export VLLM_CACHE_ROOT=$LS/vllm-cache


if pgrep -f "vllm serve" > /dev/null; then
  echo "ERROR: vllm serve already running. Kill it first: pkill -9 -f 'vllm serve'"
  exit 1
fi

vllm serve Qwen/Qwen3-8B-AWQ \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8000 2>&1 | tee vllm_server.log
EOF
chmod +x serve.sh

cat > test_lmcache_disk.py <<'PY'
import time
import requests

API_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3-8B-AWQ"

shared_context = "The quick brown fox jumps over the lazy dog. " * 400
question = "Summarize the sentence above in 5 words."

def ask(label):
    payload = {
        "model": MODEL,
        "prompt": shared_context + "\n\n" + question,
        "max_tokens": 30,
        "temperature": 0,
    }
    start = time.perf_counter()
    resp = requests.post(API_URL, json=payload)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"].strip()
    print(f"[{label}] {elapsed:.2f}s -> {text[:100]!r}")
    return elapsed

if __name__ == "__main__":
    cold = ask("COLD (first request)")
    time.sleep(1)
    warm = ask("WARM (same prompt again)")
    print(f"\n{cold / warm:.2f}x faster on the warm request")
PY

echo "==> done. tmux new -s vllm  →  bash $VLLM_DIR/serve.sh"