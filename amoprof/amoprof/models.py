"""
models.py — Built-in model registry.

Stores the parameters needed to:
  1. Determine which memory tier(s) are exercised given user-supplied caps.
  2. Size KV$ tensors for a given context length.
  3. Select the right framework and dtype.

Users can also pass a custom model spec via --model-spec on the CLI.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Arch(str, Enum):
    DENSE = "dense"
    MOE   = "moe"


@dataclass
class ModelSpec:
    # ── Identity ────────────────────────────────────────────────────────────
    alias: str                  # short name used on CLI, e.g. "deepseek-v3"
    hf_id: str                  # HuggingFace model ID
    arch: Arch

    # ── Weight sizes ────────────────────────────────────────────────────────
    size_gb_fp16: float         # total model weight at FP16
    active_params_b: float      # active params per token (for MoE: expert subset)
    total_params_b: float

    # ── Transformer dimensions (for KV$ sizing) ──────────────────────────────
    num_layers: int
    num_kv_heads: int           # GQA / MQA / MLA heads
    head_dim: int               # dimension per head

    # ── Preferred inference settings ────────────────────────────────────────
    preferred_dtype: str        = "fp16"    # fp16 | fp8 | bf16
    min_tp: int                 = 1         # minimum tensor parallel degree
    recommended_tp: int         = 1
    gguf_quant: str             = "Q4_K_M"  # quantisation for llama.cpp

    # ── Optional GGUF path (set at runtime) ─────────────────────────────────
    gguf_path: Optional[str]    = None

    # ── Derived ─────────────────────────────────────────────────────────────
    def size_gb(self, dtype: str) -> float:
        """Weight size at a given dtype."""
        factors = {"fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
                   "fp8": 0.5, "int8": 0.5, "int4": 0.25}
        return self.size_gb_fp16 * factors.get(dtype, 1.0)

    def kv_size_gb(self, context_len: int, batch_size: int,
                   dtype: str = "fp16") -> float:
        """
        KV$ size (GB) for a single batch at the given context length.
        Formula: ctx × num_kv_heads × head_dim × 2 (K+V) × num_layers × bytes
        """
        bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2,
                          "fp8": 1, "int8": 1, "int4": 0.5}.get(dtype, 2)
        kv_bytes = (context_len * self.num_kv_heads * self.head_dim
                    * 2 * self.num_layers * bytes_per_elem * batch_size)
        return kv_bytes / (1024 ** 3)

    def kv_bytes_per_token(self, dtype: str | None = None) -> int:
        """
        KV$ bytes stored per token (single request, all layers).
        Formula: num_layers × num_kv_heads × head_dim × 2 (K+V) × bytes_per_elem

        This is the per-token allocation rate into the HBM KV pool:
          pool_fills_at = pool_capacity_gb × 1024^3 / kv_bytes_per_token / context_len
        """
        d = dtype or self.preferred_dtype
        bpe = {"fp32": 4, "fp16": 2, "bf16": 2,
               "fp8": 1, "fp8_e4m3": 1, "int8": 1, "int4": 0}.get(d, 2)
        return int(self.num_layers * self.num_kv_heads * self.head_dim * 2 * bpe)

    def kv_pool_stats(self, hbm_cap_gb: float,
                      context_len: int = 65536,
                      mem_fraction: float = 0.80) -> dict:
        """
        Compute KV$ pool capacity and thresholds for a given hardware config.

        Returns dict with:
          pool_capacity_gb       total HBM allocated to KV cache
          kv_per_request_gb      KV$ consumed per active request
          overflow_concurrency   concurrent requests before HiCache eviction starts
          token_capacity         max tokens the pool can hold
        """
        weights_gb     = self.size_gb(self.preferred_dtype)
        pool_gb        = max(0.0, hbm_cap_gb * mem_fraction - weights_gb)
        bpt            = self.kv_bytes_per_token()
        kv_req_gb      = self.kv_size_gb(context_len, 1, self.preferred_dtype)
        overflow_at    = int(pool_gb / max(kv_req_gb, 0.001))
        token_cap      = int(pool_gb * (1024 ** 3) / max(bpt, 1))
        return {
            "pool_capacity_gb":     round(pool_gb, 1),
            "weights_gb":           round(weights_gb, 1),
            "kv_per_request_gb":    round(kv_req_gb, 2),
            "overflow_concurrency": overflow_at,
            "token_capacity":       token_cap,
            "kv_bytes_per_token":   bpt,
        }

    def describe(self) -> str:
        return (f"{self.alias}  [{self.arch.value}]  "
                f"{self.total_params_b:.0f}B total  "
                f"{self.active_params_b:.0f}B active  "
                f"weights={self.size_gb_fp16:.0f}GB@FP16")


# ── Registry ───────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec):
    _REGISTRY[spec.alias.lower()] = spec
    # also register common short variants
    for variant in [spec.alias.lower().replace("-", "_"),
                    spec.alias.lower().replace("_", "-")]:
        _REGISTRY[variant] = spec


def get(name: str) -> Optional[ModelSpec]:
    return _REGISTRY.get(name.lower())


def list_models() -> list[ModelSpec]:
    seen = set()
    result = []
    for s in _REGISTRY.values():
        if s.alias not in seen:
            result.append(s)
            seen.add(s.alias)
    return sorted(result, key=lambda x: x.size_gb_fp16)


# ── Built-in models ────────────────────────────────────────────────────────────

register(ModelSpec(
    alias="gemma-3-27b",
    hf_id="google/gemma-3-27b-it",
    arch=Arch.DENSE,
    size_gb_fp16=54, active_params_b=27, total_params_b=27,
    num_layers=46, num_kv_heads=16, head_dim=256,
    preferred_dtype="fp16", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="qwen3-32b",
    hf_id="Qwen/Qwen3-32B",
    arch=Arch.DENSE,
    size_gb_fp16=64, active_params_b=32, total_params_b=32,
    num_layers=64, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="mistral-large-3",
    hf_id="mistralai/Mistral-Large-Instruct-2411",
    arch=Arch.DENSE,
    size_gb_fp16=246, active_params_b=123, total_params_b=123,
    num_layers=88, num_kv_heads=8, head_dim=128,
    preferred_dtype="bf16", recommended_tp=4,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="llama-4-maverick",
    hf_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    arch=Arch.MOE,
    size_gb_fp16=800, active_params_b=17, total_params_b=400,
    num_layers=48, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=8,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="qwen3-235b",
    hf_id="Qwen/Qwen3-235B-A22B",
    arch=Arch.MOE,
    size_gb_fp16=470, active_params_b=22, total_params_b=235,
    num_layers=94, num_kv_heads=4, head_dim=128,
    preferred_dtype="bf16", recommended_tp=8,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="deepseek-v3",
    hf_id="deepseek-ai/DeepSeek-V3",
    arch=Arch.MOE,
    size_gb_fp16=671, active_params_b=37, total_params_b=671,
    num_layers=61, num_kv_heads=128, head_dim=128,
    preferred_dtype="fp8", recommended_tp=8,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="deepseek-r1",
    hf_id="deepseek-ai/DeepSeek-R1",
    arch=Arch.MOE,
    size_gb_fp16=671, active_params_b=37, total_params_b=671,
    num_layers=61, num_kv_heads=128, head_dim=128,
    preferred_dtype="fp8", recommended_tp=8,
    gguf_quant="Q4_K_M",
))

# ── Nemotron family ────────────────────────────────────────────────────────────

register(ModelSpec(
    alias="nemotron-4-340b",
    hf_id="nvidia/Nemotron-4-340B-Instruct",
    arch=Arch.DENSE,
    size_gb_fp16=680, active_params_b=340, total_params_b=340,
    num_layers=96, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=8,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="nemotron-mini-4b",
    hf_id="nvidia/Nemotron-Mini-4B-Instruct",
    arch=Arch.DENSE,
    size_gb_fp16=8, active_params_b=4, total_params_b=4,
    num_layers=32, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp16", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

# ── DeepSeek-R1 distilled variants (fit on fewer GPUs) ────────────────────────

register(ModelSpec(
    alias="deepseek-r1-70b",
    hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    arch=Arch.DENSE,
    size_gb_fp16=140, active_params_b=70, total_params_b=70,
    num_layers=80, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=2,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="deepseek-r1-32b",
    hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    arch=Arch.DENSE,
    size_gb_fp16=64, active_params_b=32, total_params_b=32,
    num_layers=64, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="deepseek-r1-8b",
    hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    arch=Arch.DENSE,
    size_gb_fp16=16, active_params_b=8, total_params_b=8,
    num_layers=32, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp16", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

# ── Nemotron Super family (v1 and v1.5) ───────────────────────────────────────
# NAS-compressed from Llama-3.3-70B-Instruct. Reasoning ON/OFF via system prompt.
# Fits on 1× H100 80GB or 2× A100 80GB at fp8.
# On 8× A100 40GB (320GB): fits comfortably at fp8 (~49GB).

register(ModelSpec(
    alias="nemotron-super-49b",
    hf_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    arch=Arch.DENSE,
    size_gb_fp16=98, active_params_b=49, total_params_b=49,
    num_layers=80, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=2,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="nemotron-super-49b-v1.5",
    hf_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    arch=Arch.DENSE,
    size_gb_fp16=98, active_params_b=49, total_params_b=49,
    num_layers=80, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=2,
    gguf_quant="Q4_K_M",
))

# ── Nemotron 3 Nano family (hybrid Mamba-Transformer MoE) ────────────────────
# Novel hybrid architecture: Mamba-2 + Transformer MoE layers.
# 1M token context window. Reasoning ON/OFF via system prompt.
# nemotron-3-nano-30b: 30B total / 3.2B active — fits on 1× A100 40GB at fp8.
# nemotron-3-nano-4b:  4B  total / 4B  active — fits on edge devices.

register(ModelSpec(
    alias="nemotron-3-nano-30b",
    hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    arch=Arch.MOE,
    size_gb_fp16=60, active_params_b=3, total_params_b=30,
    num_layers=52, num_kv_heads=4, head_dim=128,
    preferred_dtype="fp8", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="nemotron-3-nano-4b",
    hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    arch=Arch.DENSE,
    size_gb_fp16=8, active_params_b=4, total_params_b=4,
    num_layers=32, num_kv_heads=4, head_dim=128,
    preferred_dtype="bf16", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

# ── Nemotron Super family (v1 and v1.5) ───────────────────────────────────────
# NAS-compressed from Llama-3.3-70B-Instruct. Reasoning ON/OFF via system prompt.
# Fits on 2× A100 40GB at fp8 (~49GB). 128K context window.

register(ModelSpec(
    alias="nemotron-super-49b",
    hf_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    arch=Arch.DENSE,
    size_gb_fp16=98, active_params_b=49, total_params_b=49,
    num_layers=80, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=2,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="nemotron-super-49b-v1.5",
    hf_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    arch=Arch.DENSE,
    size_gb_fp16=98, active_params_b=49, total_params_b=49,
    num_layers=80, num_kv_heads=8, head_dim=128,
    preferred_dtype="fp8", recommended_tp=2,
    gguf_quant="Q4_K_M",
))

# ── Nemotron 3 Nano family (hybrid Mamba-Transformer MoE, 1M ctx) ────────────
# Novel architecture: Mamba-2 + Transformer MoE layers. 1M token context.
# Reasoning ON/OFF via system prompt. 3.3x higher throughput than Qwen3-30B.
# nemotron-3-nano-30b: 30B total / 3.2B active — fits on 1× A100 40GB at fp8.
# nemotron-3-nano-4b:  4B  total            — edge device (Jetson/RTX).

register(ModelSpec(
    alias="nemotron-3-nano-30b",
    hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    arch=Arch.MOE,
    size_gb_fp16=60, active_params_b=3, total_params_b=30,
    num_layers=52, num_kv_heads=4, head_dim=128,
    preferred_dtype="fp8", recommended_tp=1,
    gguf_quant="Q4_K_M",
))

register(ModelSpec(
    alias="nemotron-3-nano-4b",
    hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    arch=Arch.DENSE,
    size_gb_fp16=8, active_params_b=4, total_params_b=4,
    num_layers=32, num_kv_heads=4, head_dim=128,
    preferred_dtype="bf16", recommended_tp=1,
    gguf_quant="Q4_K_M",
))
