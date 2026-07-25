"""Pre-registered eviction prediction.

The harness's contract with physics: before every run we compute whether the
workload's unique KV working set SHOULD exceed the GPU KV pool, and we write
that prediction into the manifest. After the run, validate.py checks the
telemetry against it. Prediction != observation means harness bug (or a wrong
pool estimate), not interesting physics — and we say so in the verdict.

KV bytes/token (fp16/bf16):
    layers x 2 (K,V) x num_key_value_heads x head_dim x bytes_per_elem

Precedence for each quantity:
    explicit override (spec/CLI) > measured (server log / snapshots) > estimate
Every quantity records which source produced it — no silent estimates.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .naming import dig
from .util import log

DTYPE_BYTES = {
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4,
    "float8": 1, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1, "int4": 1,  # int4 packed is handled by the caller if needed
}


def _hf_config_path(model: str, local_only: bool) -> str | None:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=model, filename="config.json",
                               local_files_only=local_only)
    except Exception:
        return None


def _load_model_config(model: str, config_path: str | None = None,
                       network_timeout_s: float = 8.0) -> dict | None:
    """config.json from: explicit path > local HF cache > network (hard timeout).

    Offline bench boxes must not stall on HF's internal retry loop — the model
    is almost always already in the local HF cache there anyway.
    """
    if config_path:
        try:
            return json.loads(Path(config_path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not read model config at %s (%s)", config_path, e)
            return None
    cfg_path = _hf_config_path(model, local_only=True)
    if cfg_path is None:
        import logging as _logging
        import threading
        _logging.getLogger("huggingface_hub").setLevel(_logging.CRITICAL)  # no retry spam
        result: dict = {}
        def _fetch():
            result["path"] = _hf_config_path(model, local_only=False)
        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=network_timeout_s)
        cfg_path = result.get("path")
    if cfg_path is None:
        log.warning("config.json for %s unavailable (offline?); "
                    "pass prediction.model_config_path or kv_bytes_per_token", model)
        return None
    try:
        return json.loads(Path(cfg_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("could not parse config.json for %s (%s)", model, e)
        return None


def kv_bytes_per_token_from_config(model: str, kv_dtype: str = "float16",
                                   kv_bytes_override: int | None = None,
                                   config_path: str | None = None) -> tuple[int | None, str]:
    """Return (bytes_per_token, source). None when we can't know."""
    if kv_bytes_override:
        return int(kv_bytes_override), "override"
    cfg = _load_model_config(model, config_path)
    if cfg is None:
        return None, "unavailable"
    layers = cfg.get("num_hidden_layers")
    kv_heads = cfg.get("num_key_value_heads")
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        hidden = cfg.get("hidden_size")
        attn_heads = cfg.get("num_attention_heads")
        if hidden and attn_heads:
            head_dim = hidden // attn_heads
    if not (layers and kv_heads and head_dim):
        log.warning("config.json for %s lacks layers/kv_heads/head_dim; KV math unavailable", model)
        return None, "unavailable"
    bpe = DTYPE_BYTES.get(str(kv_dtype).lower())
    if bpe is None:
        raise ValueError(f"unknown kv_dtype {kv_dtype!r}; known: {sorted(DTYPE_BYTES)}")
    return int(layers) * 2 * int(kv_heads) * int(head_dim) * bpe, "hf_config"


_VLLM_POOL_PATTERNS = [
    re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE),
    re.compile(r"kv cache size[:\s]+([\d,]+)\s*tokens", re.IGNORECASE),
]


def pool_tokens_from_vllm_log(log_path: Path) -> tuple[int | None, str]:
    if not log_path or not log_path.exists():
        return None, "no_log"
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None, "no_log"
    for pat in _VLLM_POOL_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return int(matches[-1].replace(",", "")), "vllm_log"
    return None, "no_match"


def total_gpu_bytes() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
        return sum(int(line.strip()) for line in r.stdout.splitlines() if line.strip()) * 1024 ** 2
    except (OSError, subprocess.TimeoutExpired):
        return None


def pool_tokens_estimate(kv_bytes_per_token: int, gpu_mem_util: float,
                         weights_gb: float, reserve_gb: float = 4.0) -> tuple[int | None, str]:
    total = total_gpu_bytes()
    if total is None:
        return None, "no_nvidia_smi"
    pool_bytes = total * gpu_mem_util - weights_gb * 1024**3 - reserve_gb * 1024**3
    if pool_bytes <= 0:
        return None, "estimate_nonpositive"
    return int(pool_bytes / kv_bytes_per_token), "estimate"


def working_set_tokens(merged_driver_config: dict) -> tuple[int | None, str]:
    """Unique KV working set in tokens, per data generator type.

    This is the number that must exceed the pool for eviction/disk offload to
    engage. Formula depends on data.type and is documented per branch.
    """
    data = merged_driver_config.get("data", {}) or {}
    dtype = data.get("type")
    load = merged_driver_config.get("load", {}) or {}
    stages = load.get("stages") or []
    concurrent_sessions = None
    if stages and isinstance(stages, list):
        concurrent_sessions = stages[0].get("concurrent_sessions") or stages[0].get("concurrency_level")

    def dist_tokens(dist: dict | None, default: float) -> float:
        return float((dist or {}).get("mean", default))

    if dtype in ("random", "synthetic"):
        in_d, out_d = data.get("input_distribution") or {}, data.get("output_distribution") or {}
        unique = in_d.get("total_count")  # None => unbounded unique prompts
        if unique is None:
            return None, "unbounded_unique_prompts"
        per = dist_tokens(in_d, 512) + dist_tokens(out_d, 512)
        return int(unique * per), "random:total_count*(in+out)"

    if dtype == "shared_prefix":
        sp = data.get("shared_prefix") or {}
        def_len = lambda v: (v.get("mean") if isinstance(v, dict) else v) or 0
        sys_len, q_len, o_len = def_len(sp.get("system_prompt_len")), def_len(sp.get("question_len")), def_len(sp.get("output_len"))
        groups, per_group = int(sp.get("num_groups", 10)), int(sp.get("num_prompts_per_group", 10))
        # shared system prompts count once each; per-prompt suffix+output per user
        return int(groups * sys_len + groups * per_group * (q_len + o_len)), \
            "shared_prefix:G*sys+G*U*(q+out)"

    if dtype == "conversation_replay":
        cr = data.get("conversation_replay") or {}
        n = int(cr.get("num_conversations", 200))
        sys_len = int(cr.get("shared_system_prompt_len", 0))
        dyn = dist_tokens(cr.get("dynamic_system_prompt_len"), 0)
        turns = dist_tokens(cr.get("turns_per_conversation"), 1)
        tin = dist_tokens(cr.get("input_tokens_per_turn"), 0)
        tout = dist_tokens(cr.get("output_tokens_per_turn"), 0)
        # concurrent conversations' KV is live simultaneously; shared prompt cached once
        live = concurrent_sessions or n
        return int(sys_len + live * (dyn + turns * (tin + tout))), \
            "conversation:shared+concurrent*(dyn+turns*(in+out))"

    if dtype in ("otel_trace_replay", "weka_trace_replay"):
        return None, f"{dtype}:unknown(trace-dependent)"

    return None, f"{dtype}:no_formula"


@dataclass
class Prediction:
    kv_bytes_per_token: int | None = None
    kv_bytes_source: str = ""
    pool_tokens: int | None = None
    pool_source: str = ""
    working_set_tokens: int | None = None
    working_set_source: str = ""
    expect_eviction: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_bytes_source": self.kv_bytes_source,
            "pool_tokens": self.pool_tokens,
            "pool_source": self.pool_source,
            "working_set_tokens": self.working_set_tokens,
            "working_set_source": self.working_set_source,
            "expect_eviction": self.expect_eviction,
            "notes": self.notes,
        }


def build_prediction(spec_prediction: dict, merged_driver_config: dict,
                     vllm_log_path: Path | None = None) -> Prediction:
    p = Prediction()
    model = str(merged_driver_config.get("server", {}).get("model_name", ""))
    kv_dtype = spec_prediction.get("kv_dtype", "float16")

    p.kv_bytes_per_token, p.kv_bytes_source = kv_bytes_per_token_from_config(
        model, kv_dtype=kv_dtype, kv_bytes_override=spec_prediction.get("kv_bytes_per_token"),
        config_path=spec_prediction.get("model_config_path"))

    # pool: override > vllm log > estimate
    if spec_prediction.get("pool_tokens"):
        p.pool_tokens, p.pool_source = int(spec_prediction["pool_tokens"]), "override"
    else:
        pool, src = pool_tokens_from_vllm_log(vllm_log_path) if vllm_log_path else (None, "no_log")
        if pool is not None:
            p.pool_tokens, p.pool_source = pool, src
        elif p.kv_bytes_per_token:
            p.pool_tokens, p.pool_source = pool_tokens_estimate(
                p.kv_bytes_per_token,
                float(spec_prediction.get("gpu_mem_util", 0.7)),
                float(spec_prediction.get("weights_gb", 6.0)),
                float(spec_prediction.get("reserve_gb", 4.0)))

    ws, ws_src = working_set_tokens(merged_driver_config)
    if spec_prediction.get("working_set_tokens"):
        p.working_set_tokens, p.working_set_source = int(spec_prediction["working_set_tokens"]), "override"
    else:
        p.working_set_tokens, p.working_set_source = ws, ws_src

    expect = spec_prediction.get("expect_eviction", "auto")
    if str(expect).lower() == "auto":
        if p.working_set_tokens is not None and p.pool_tokens is not None:
            margin = float(spec_prediction.get("margin", 1.0))
            p.expect_eviction = p.working_set_tokens > p.pool_tokens * margin
        else:
            p.expect_eviction = None
            p.notes.append("expect_eviction=auto but working set or pool unknown")
    else:
        p.expect_eviction = bool(expect)

    if p.pool_source == "estimate":
        p.notes.append("pool is an ESTIMATE (nvidia-smi x gpu_mem_util - weights - reserve); "
                       "pass prediction.pool_tokens or --vllm-log for ground truth")
    if p.working_set_tokens is not None and p.pool_tokens is not None:
        p.notes.append(
            f"working_set/pool = {p.working_set_tokens / p.pool_tokens:.2f}x")
    return p
