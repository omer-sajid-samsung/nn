"""
plan.py — Derives a RunPlan from user-supplied arguments.

Given:
    model   — ModelSpec
    hbm_cap — HBM GB available for inference (e.g. 80 for 1x H100)
    dram_cap— DRAM GB available (e.g. 512)
    nvme_cap— NVMe GB available (e.g. 10240)
    ai_op   — one of: prefill | decode | weight-load | checkpoint |
                      kv-evict | mixed

Outputs a RunPlan that specifies:
    tier        — which memory tier is exercised (HBM / DRAM / SSD)
    framework   — which inference/training framework to use
    context_lens— list of context lengths to sweep
    batch_sizes — list of batch sizes to sweep
    io_pattern  — the expected I/O pattern on SSD
    ssd_dim     — the SSD dimension(s) this exercises
    dtype       — dtype selected for the run
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import ModelSpec, Arch


# ── Enums ──────────────────────────────────────────────────────────────────────

class Tier(str, Enum):
    HBM  = "HBM"
    DRAM = "DRAM"
    SSD  = "SSD"
    MULTI= "HBM+DRAM+SSD"   # mixed op spans all tiers

class AiOp(str, Enum):
    PREFILL     = "prefill"      # process input tokens → KV$ write
    DECODE      = "decode"       # generate output tokens → KV$ read
    WEIGHT_LOAD = "weight-load"  # cold-load weights from SSD → seq read BW
    CHECKPOINT  = "checkpoint"   # fine-tune, save checkpoints → write WAF
    KV_EVICT    = "kv-evict"     # force KV$ eviction to SSD → rand write IOPS
    MIXED       = "mixed"        # all of the above concurrently

class Framework(str, Enum):
    LLAMA_CPP      = "llama_cpp"
    VLLM           = "vllm"
    SGLANG         = "sglang"          # one-shot via sglang.bench_serving
    SGLANG_SERVER  = "sglang_server"   # full server lifecycle + rate sweep
    ACCELERATE     = "accelerate"
    FLEXGEN        = "flexgen"
    DEEPSPEED      = "deepspeed"
    FIO            = "fio"
    SWEBENCH       = "swebench"        # SWE-bench agent workload

IO_PATTERN: dict[AiOp, str] = {
    AiOp.PREFILL:     "Large sequential write  (K+V tensors per layer)",
    AiOp.DECODE:      "Small random read       (cold KV$ block per token)",
    AiOp.WEIGHT_LOAD: "Large sequential read   (weight shards from SSD)",
    AiOp.CHECKPOINT:  "Burst large write       (optimizer state + params)",
    AiOp.KV_EVICT:    "Small random write      (evicted KV$ blocks)",
    AiOp.MIXED:       "Concurrent R+W          (decode reads + ckpt writes)",
}

SSD_DIM: dict[AiOp, str] = {
    AiOp.PREFILL:     "Write BW  ·  WAF  ·  SLC drain",
    AiOp.DECODE:      "Random IOPS  ·  P99/P999 read latency  ·  KV$ miss stall",
    AiOp.WEIGHT_LOAD: "Sequential read BW  ·  SLC cache depth  ·  TLC fallback",
    AiOp.CHECKPOINT:  "Write BW  ·  WAF  ·  Endurance (DWPD)  ·  SLC drain",
    AiOp.KV_EVICT:    "Random write IOPS  ·  WAF  ·  Endurance",
    AiOp.MIXED:       "R+W interference  ·  GC pressure  ·  P99 read degradation",
}


# ── Capacity parsing ───────────────────────────────────────────────────────────

def parse_gb(s: str) -> float:
    """Parse '80GB', '512G', '10TB', '2048' → float GB."""
    s = s.strip().upper().replace(" ", "")
    if s.endswith("TB"):
        return float(s[:-2]) * 1024
    if s.endswith("GB") or s.endswith("G"):
        return float(s.rstrip("GB").rstrip("G"))
    if s.endswith("MB") or s.endswith("M"):
        return float(s.rstrip("MB").rstrip("M")) / 1024
    return float(s)   # assume GB if no suffix


# ── Tier derivation ────────────────────────────────────────────────────────────

def derive_tier(model: ModelSpec, hbm_gb: float, dram_gb: float,
                ai_op: AiOp, context_len: int, batch_size: int,
                dtype: str) -> Tier:
    """
    Determine which memory tier is exercised given the capacity constraints.

    Logic:
      weight-load / prefill:  tier where weights reside
      decode / kv-evict:      tier where KV$ blocks reside
      checkpoint:             always SSD (writes go to disk)
      mixed:                  spans all tiers
    """
    if ai_op == AiOp.CHECKPOINT:
        return Tier.SSD
    if ai_op == AiOp.MIXED:
        return Tier.MULTI

    weight_gb = model.size_gb(dtype)
    kv_gb     = model.kv_size_gb(context_len, batch_size, dtype)

    if ai_op in (AiOp.WEIGHT_LOAD,):
        # Where do weights live?
        if weight_gb <= hbm_gb:
            return Tier.HBM
        if weight_gb <= hbm_gb + dram_gb:
            return Tier.DRAM
        return Tier.SSD

    if ai_op in (AiOp.PREFILL, AiOp.DECODE, AiOp.KV_EVICT):
        # Where does KV$ live?
        hbm_for_kv = max(0.0, hbm_gb - weight_gb)   # HBM left after weights
        if kv_gb <= hbm_for_kv:
            return Tier.HBM
        dram_for_kv = max(0.0, dram_gb)
        if kv_gb <= hbm_for_kv + dram_for_kv:
            return Tier.DRAM
        return Tier.SSD

    return Tier.SSD


# ── Framework selection ────────────────────────────────────────────────────────

def select_framework(model: ModelSpec, tier: Tier,
                     ai_op: AiOp, weight_gb: float,
                     hbm_gb: float, dram_gb: float) -> Framework:
    """
    Pick the best framework for the tier + operation combination.
    SGLANG_SERVER is preferred for HBM-resident models (full rate sweep).
    SWEBENCH is only selected when explicitly requested via build_plan(benchmark=).
    """
    if ai_op == AiOp.WEIGHT_LOAD:
        return Framework.LLAMA_CPP

    if ai_op == AiOp.CHECKPOINT:
        return Framework.DEEPSPEED

    if tier == Tier.HBM:
        return Framework.SGLANG_SERVER   # full server lifecycle + rate sweep

    if tier == Tier.DRAM:
        if model.total_params_b >= 200:
            return Framework.FLEXGEN
        return Framework.ACCELERATE

    if tier == Tier.SSD:
        if ai_op in (AiOp.KV_EVICT, AiOp.DECODE):
            return Framework.ACCELERATE
        if ai_op == AiOp.PREFILL:
            return Framework.VLLM
        return Framework.ACCELERATE

    if tier == Tier.MULTI:
        return Framework.SGLANG_SERVER

    return Framework.VLLM


# ── Context length and batch size defaults ─────────────────────────────────────

_CTX_BY_OP: dict[AiOp, list[int]] = {
    AiOp.PREFILL:     [8192, 32768, 65536],
    AiOp.DECODE:      [8192, 32768, 65536],
    AiOp.WEIGHT_LOAD: [4096],               # minimal KV$, stress weight BW
    AiOp.CHECKPOINT:  [4096],
    AiOp.KV_EVICT:    [32768, 65536],
    AiOp.MIXED:       [8192, 32768],
}

_BS_BY_OP: dict[AiOp, list[int]] = {
    AiOp.PREFILL:     [1, 4],
    AiOp.DECODE:      [1, 4, 16],
    AiOp.WEIGHT_LOAD: [1],
    AiOp.CHECKPOINT:  [4],
    AiOp.KV_EVICT:    [1, 4],
    AiOp.MIXED:       [1],
}


# ── RunPlan ────────────────────────────────────────────────────────────────────

@dataclass
class RunPlan:
    # What the user asked for
    model: ModelSpec
    ai_op: AiOp
    hbm_cap_gb: float
    dram_cap_gb: float
    nvme_cap_gb: float

    # Derived
    tier: Tier
    framework: Framework
    dtype: str
    context_lens: list[int]
    batch_sizes: list[int]
    io_pattern: str
    ssd_dimensions: str

    # Offload dirs / device
    ssd_device: str        = "/dev/nvme0n1"
    hicache_storage_path: str = ""      # path passed to --file-storage-path as JSON
    ssd_path: str          = "/mnt/ssd"
    offload_dir: str       = "/mnt/ssd/amoprof_offload"
    gguf_path: str         = ""
    tensor_parallel: int   = 1

    # Derived summary
    weight_gb: float       = 0.0
    kv_gb_per_ctx: float   = 0.0   # at max context, batch=1

    def summary(self) -> str:
        lines = [
            f"  Model      : {self.model.describe()}",
            f"  AI Op      : {self.ai_op.value}",
            f"  Caps       : HBM={self.hbm_cap_gb:.0f}GB  "
                            f"DRAM={self.dram_cap_gb:.0f}GB  "
                            f"NVMe={self.nvme_cap_gb:.0f}GB",
            f"  Tier       : {self.tier.value}  "
                f"(weights={self.weight_gb:.1f}GB@{self.dtype}  "
                f"KV$≈{self.kv_gb_per_ctx:.1f}GB@max_ctx)",
            f"  Framework  : {self.framework.value}",
            f"  dtype      : {self.dtype}",
            f"  Contexts   : {self.context_lens}  tokens",
            f"  Batches    : {self.batch_sizes}",
            f"  I/O pattern: {self.io_pattern}",
            f"  SSD dims   : {self.ssd_dimensions}",
        ]
        return "\n".join(lines)


# ── Builder ────────────────────────────────────────────────────────────────────

def build_plan(
    model: ModelSpec,
    ai_op: AiOp,
    hbm_cap_gb: float,
    dram_cap_gb: float,
    nvme_cap_gb: float,
    ssd_device: str        = "/dev/nvme0n1",
    hicache_storage_path: str = "",     # path passed to --file-storage-path as JSON
    ssd_path: str          = "/mnt/ssd",
    context_lens: Optional[list[int]] = None,
    batch_sizes: Optional[list[int]]  = None,
    dtype: Optional[str]              = None,
    tensor_parallel: Optional[int]    = None,
) -> RunPlan:

    # dtype: user override > model preferred
    dtype = dtype or model.preferred_dtype

    # context / batch: user override > op default
    ctx_list = context_lens or _CTX_BY_OP[ai_op]
    bs_list  = batch_sizes  or _BS_BY_OP[ai_op]

    weight_gb    = model.size_gb(dtype)
    kv_max       = model.kv_size_gb(max(ctx_list), 1, dtype)
    tier         = derive_tier(model, hbm_cap_gb, dram_cap_gb,
                               ai_op, max(ctx_list), max(bs_list), dtype)
    framework    = select_framework(model, tier, ai_op, weight_gb,
                                    hbm_cap_gb, dram_cap_gb)
    tp           = tensor_parallel or model.recommended_tp

    offload_dir = f"{ssd_path}/amoprof_offload/{model.alias}"

    return RunPlan(
        model          = model,
        ai_op          = ai_op,
        hbm_cap_gb     = hbm_cap_gb,
        dram_cap_gb    = dram_cap_gb,
        nvme_cap_gb    = nvme_cap_gb,
        tier           = tier,
        framework      = framework,
        dtype          = dtype,
        context_lens   = ctx_list,
        batch_sizes    = bs_list,
        io_pattern     = IO_PATTERN[ai_op],
        ssd_dimensions = SSD_DIM[ai_op],
        ssd_device            = ssd_device,
        hicache_storage_path  = hicache_storage_path,
        ssd_path       = ssd_path,
        offload_dir    = offload_dir,
        gguf_path      = model.gguf_path or "",
        tensor_parallel= tp,
        weight_gb      = weight_gb,
        kv_gb_per_ctx  = kv_max,
    )
