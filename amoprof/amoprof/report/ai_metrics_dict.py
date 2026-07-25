"""
ai_metrics_dict.py  —  AMOprof Universal Metrics Dictionary  v2.0
===================================================================
Comprehensive semantic registry covering:
  Inference servers : SGLang, vLLM, TGI, TensorRT-LLM, Triton
  Benchmarks        : MLPerf, LLMPerf, HELM, vLLM-bench, LM-Eval,
                      SWEBench, LongBench, OpenLLM, MMLU/ARC/GSM8K
  GPU/HBM           : DCGM, nvidia-smi, ncu
  System memory     : AMDuProf, Intel PCM, node_exporter, procfs
  SSD/NVMe IO       : node_exporter, iostat, dstat, nvme-cli (SMART)
  Block layer       : blktrace, biosnoop, bpftrace, sysfs, procfs

Key API
-------
  resolve(raw_name)          -> MetricDef | None   any source, regex + alias scoring
  match_prometheus(name)     -> MetricDef | None   Prometheus-tuned alias
  discover(source)           -> {raw_name: MetricDef}  bulk-map a data source
  annotate_df(df, col)       -> DataFrame  add canonical/unit/formula columns
  get_category(cat)          -> [MetricDef]
  search(query)              -> [MetricDef]
  lookup(name)               -> MetricDef | None   exact canonical/alias match
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# ---------------------------------------------------------------------------
#  Core data model
# ---------------------------------------------------------------------------

@dataclass
class MetricDef:
    name:             str
    full_name:        str
    aliases:          list[str] = field(default_factory=list)
    regex_patterns:   list[str] = field(default_factory=list)
    category:         str = ""
    subcategory:      str = ""
    section:          str = ""
    unit:             str = ""
    higher_is_better: Optional[bool] = None
    normal_range:     Optional[tuple] = None
    description:      str = ""
    formula:          str = ""
    interpretation:   str = ""
    sources:          list[str] = field(default_factory=list)
    related:          list[str] = field(default_factory=list)
    profiler_field:   str = ""

    def match_score(self, raw: str) -> int:
        """Confidence score 0-100 for raw metric name vs this definition."""
        s = raw.lower().replace(":", "_").replace("-", "_").replace(".", "_")
        best = 0
        if s == self.name:
            return 100
        for a in self.aliases:
            al = a.lower().replace("-", "_").replace(":", "_").replace(".", "_")
            if s == al:
                best = max(best, 90)
            elif al in s or s in al:
                best = max(best, 55)
        for pat in self.regex_patterns:
            try:
                if re.search(pat, raw, re.I):
                    best = max(best, 75)
            except re.error:
                pass
        if self.name in s:
            best = max(best, 65)
        words = [w for w in re.split(r"[_\s]+", self.name) if len(w) > 2]
        if words and all(w in s for w in words):
            best = max(best, 52)
        return best

    def to_html_row(self, i: int = 0) -> str:
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        dir_arrow = ("↑" if self.higher_is_better
                     else "↓" if self.higher_is_better is False else "–")
        rng  = (f"{self.normal_range[0]}–{self.normal_range[1]}"
                if self.normal_range else "")
        ali  = ", ".join(self.aliases[:5]) + ("…" if len(self.aliases) > 5 else "")
        srcs = " · ".join(self.sources[:5]) + ("…" if len(self.sources) > 5 else "")
        fml  = ""
        if self.formula:
            body = self.formula.replace("<", "&lt;").replace(">", "&gt;")
            note = (f'<div style="font-size:9px;color:#475569;margin-top:4px;'
                    f'font-style:italic">{self.interpretation[:200]}'
                    + ("…" if len(self.interpretation) > 200 else "") + "</div>"
                    if self.interpretation else "")
            fml  = (
                '<details><summary style="cursor:pointer;font-size:9px;'
                'color:#4f46e5">📐 formula</summary>'
                '<div style="background:#f1f5f9;border-radius:4px;padding:6px 8px;'
                'margin-top:4px;font-size:9px;font-family:monospace;color:#1e3a5f;'
                'white-space:pre-wrap;max-width:360px">'
                + body + "</div>" + note + "</details>"
            )
        return (
            f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9">'
            f'<td style="padding:7px 10px;font-size:10px;white-space:nowrap">'
            f'<code style="background:#f1f5f9;padding:2px 5px;border-radius:3px">'
            f'{self.name}</code></td>'
            f'<td style="padding:7px 10px;font-size:11px;font-weight:600;'
            f'color:#1e3a5f">{self.full_name}</td>'
            f'<td style="padding:7px 10px;font-size:10px;color:#475569">{self.unit}</td>'
            f'<td style="padding:7px 10px;font-size:10px;text-align:center">{dir_arrow}</td>'
            f'<td style="padding:7px 10px;font-size:10px;color:#334155;max-width:280px">'
            f'{self.description[:140]}{"…" if len(self.description)>140 else ""}</td>'
            f'<td style="padding:7px 10px;font-size:9px;color:#64748b;'
            f'font-family:monospace">{ali}</td>'
            f'<td style="padding:7px 10px;font-size:9px;color:#475569">{srcs}</td>'
            f'<td style="padding:7px 10px;font-size:9px;color:#94a3b8;'
            f'font-family:monospace">{rng}</td>'
            f'<td style="padding:7px 10px">{fml}</td>'
            f'</tr>'
        )


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, MetricDef] = {}


def _r(name: str, full_name: str, **kw) -> MetricDef:
    m = MetricDef(name=name, full_name=full_name, **kw)
    REGISTRY[name] = m
    return m


# ===========================================================================
#  A. INFERENCE — LATENCY
# ===========================================================================

_r("ttft", "Time to First Token",
   aliases=["time_to_first_token", "TTFT", "first_token_latency", "prefill_latency",
             "ttft_ms", "server_ttft_ms", "time-to-first-token", "median_ttft_ms",
             "p50_ttft", "p90_ttft", "p99_ttft", "mean_ttft_ms",
             "first_token_ms", "nv_trt_llm_request_metrics_first_token_ms",
             "tgi_request_prefill_duration_seconds",
             "sglang_time_to_first_token_seconds"],
   regex_patterns=[r"time_to_first_token", r"first_token.*lat", r"\bttft\b",
                   r"prefill.*lat", r"prefill.*time", r"first.*tok"],
   category="inference", subcategory="latency", section="§A8",
   unit="ms", higher_is_better=False, normal_range=(100, 5000),
   description="Time from request arrival to first generated token. "
               "Includes queue wait + prefill compute + KV cache lookup. "
               "Reported by all major inference servers.",
   formula=(
       "SGLang  : rate(time_to_first_token_seconds_sum) / rate(_count) × 1000\n"
       "vLLM    : sglang_time_to_first_token_seconds histogram\n"
       "TGI     : tgi_request_prefill_duration_seconds\n"
       "TRT-LLM : nv_trt_llm_request_metrics_first_token_ms\n"
       "MLPerf  : first_token_latency_ms  (99th pct target)\n"
       "CSV     : mean_ttft_ms / p99_ttft_ms / first_token_latency_ms"
   ),
   interpretation=(
       "< 500ms: excellent. 500ms–2s: acceptable for long prompts. "
       "> 5s: prefill-limited. "
       "Floor = cached_tokens × kv_bytes_per_token / HBM_peak_bw × 1000."
   ),
   sources=["sglang", "vllm", "tgi", "tensorrt-llm", "triton",
             "mlperf", "llmperf", "vllm-bench"],
   related=["tpot", "e2e_latency", "cache_hit_rate", "prefill_throughput"],
   profiler_field="ttft_ms")

_r("tpot", "Time per Output Token (TPOT / ITL)",
   aliases=["inter_token_latency", "ITL", "itl", "tpot_ms", "token_latency",
             "time_per_output_token", "inter-token-latency", "server_itl_ms",
             "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms",
             "sglang_inter_token_latency_seconds",
             "nv_trt_llm_request_metrics_generate_queue_duration_ms"],
   regex_patterns=[r"inter_token_latency", r"\btpot\b", r"\bitl\b",
                   r"time_per_output_token", r"per.*token.*lat",
                   r"token.*latency", r"decode.*lat"],
   category="inference", subcategory="latency", section="§A9",
   unit="ms", higher_is_better=False, normal_range=(5, 100),
   description="Mean latency per generated token in decode phase. "
               "Hardware floor = model_bytes_per_GPU / HBM_peak_bw × 1000.",
   formula=(
       "SGLang : rate(inter_token_latency_seconds_sum) / rate(_count) × 1000\n"
       "vLLM   : sglang_inter_token_latency_seconds histogram\n"
       "Min    : (n_params × 2B_FP16 / n_gpus) / HBM_peak_GBs × 1000"
   ),
   interpretation=(
       "A100 FP16 70B/8GPU min ≈ 8.75 ms at 2 TB/s HBM. "
       "Observed >> min → compute overhead, batching, or KV pressure."
   ),
   sources=["sglang", "vllm", "tgi", "tensorrt-llm", "mlperf", "llmperf", "vllm-bench"],
   related=["ttft", "gen_throughput", "hbm_bw", "mfu"],
   profiler_field="tpot_ms")

_r("e2e_latency", "End-to-End Request Latency",
   aliases=["e2e_latency", "end_to_end_latency", "request_latency", "total_latency",
             "latency", "response_time", "e2e_ms", "server_e2e_ms",
             "mean_e2e_latency_ms", "p99_e2e_latency_ms",
             "sglang_e2e_request_latency_seconds",
             "nv_trt_llm_request_metrics_e2e_latency_ms",
             "tgi_request_duration_seconds"],
   regex_patterns=[r"e2e.*lat", r"end.*to.*end", r"request.*latency",
                   r"total.*latency", r"response.*time"],
   category="inference", subcategory="latency", section="§A5-7",
   unit="ms", higher_is_better=False, normal_range=(500, 60000),
   description="Total latency from submission to last output token. "
               "E2E = queue_ms + TTFT_ms + TPOT_ms × output_tokens.",
   formula=(
       "SGLang  : rate(e2e_request_latency_seconds_sum) / rate(_count) × 1000\n"
       "vLLM    : sglang_e2e_request_latency_seconds histogram\n"
       "TRT-LLM : nv_trt_llm_request_metrics_e2e_latency_ms\n"
       "TGI     : tgi_request_duration_seconds"
   ),
   sources=["sglang", "vllm", "tgi", "tensorrt-llm", "llmperf", "vllm-bench"],
   related=["ttft", "tpot", "queue_time"],
   profiler_field="sess_e2e_mean_s")

_r("queue_time", "Request Queue Wait Time",
   aliases=["queue_time", "queue_latency", "scheduling_latency", "wait_time", "queue_ms",
             "vllm:request_queue_time_seconds",
             "nv_trt_llm_request_metrics_context_queue_duration_ms"],
   regex_patterns=[r"queue_time", r"queue.*lat", r"sched.*lat", r"wait_time",
                   r"context_queue"],
   category="inference", subcategory="latency",
   unit="ms", higher_is_better=False, normal_range=(0, 500),
   description="Time request waits in scheduler queue before prefill starts.",
   formula=(
       "SGLang  : rate(queue_time_seconds_sum) / rate(_count) × 1000\n"
       "vLLM    : vllm:request_queue_time_seconds\n"
       "TRT-LLM : nv_trt_llm_request_metrics_context_queue_duration_ms"
   ),
   sources=["sglang", "vllm", "tensorrt-llm"],
   related=["ttft", "num_running_reqs"],
   profiler_field="sess_queue_mean_ms")


# ===========================================================================
#  A. INFERENCE — THROUGHPUT
# ===========================================================================

_r("throughput", "Request Throughput",
   aliases=["request_throughput", "rps", "qps", "requests_per_second",
             "req_per_sec", "request_rate", "arrival_rate",
             "nv_inference_request_success"],
   regex_patterns=[r"request.*throughput", r"\brps\b", r"\bqps\b",
                   r"requests_per_s", r"req_per_s"],
   category="inference", subcategory="throughput",
   unit="req/s", higher_is_better=True,
   description="Inference requests completed per second.",
   formula="Δsglang_num_requests_total / Δ(time_sec)\nvllm-bench CSV: request_throughput",
   sources=["sglang", "vllm", "tgi", "tensorrt-llm", "mlperf", "llmperf", "vllm-bench"],
   related=["gen_throughput", "tpot", "ttft"],
   profiler_field="gen_tp_peak")

_r("gen_throughput", "Generation Token Throughput",
   aliases=["token_throughput", "output_throughput", "gen_throughput", "decode_throughput",
             "tokens_per_second", "tps", "gen_tps", "output_token_throughput",
             "mean_output_throughput_token_per_s",
             "output_token_throughput_token_per_s"],
   regex_patterns=[r"gen_throughput", r"output.*token.*throughput",
                   r"token.*per.*sec", r"tok.*per.*s\b",
                   r"generation.*token.*rate", r"decode.*throughput"],
   category="inference", subcategory="throughput",
   unit="tok/s", higher_is_better=True, normal_range=(10, 2000),
   description="Output tokens generated per second across all concurrent requests. "
               "Upper bound ≈ HBM_peak_bw / model_bytes_per_GPU.",
   formula=(
       "SGLang    : sglang_gen_throughput  (rolling 1s)\n"
       "vLLM-bench: output_throughput column\n"
       "LLMPerf   : mean_output_throughput_token_per_s"
   ),
   sources=["sglang", "vllm", "tgi", "tensorrt-llm", "mlperf", "llmperf", "vllm-bench"],
   related=["tpot", "hbm_bw", "batch_size"],
   profiler_field="gen_tp_peak")

_r("prefill_throughput", "Prefill / Input Token Throughput",
   aliases=["input_token_throughput", "prompt_throughput", "prefill_tps",
             "input_throughput", "prompt_tokens_per_sec"],
   regex_patterns=[r"input.*token.*throughput", r"prompt.*throughput",
                   r"prefill.*throughput", r"input.*tok.*rate"],
   category="inference", subcategory="throughput",
   unit="tok/s", higher_is_better=True,
   description="Input (prompt) tokens processed per second.",
   formula="Δ(prompt_tokens_total) / Δ(time_sec)\nvllm-bench CSV: input_throughput",
   sources=["sglang", "vllm", "vllm-bench", "mlperf"],
   related=["ttft", "gen_throughput", "cache_hit_rate"],
   profiler_field="")


# ===========================================================================
#  A. INFERENCE — CONCURRENCY
# ===========================================================================

_r("num_running_reqs", "Active Concurrent Sessions",
   aliases=["num_running_reqs", "running_requests", "active_requests",
             "concurrent_requests", "running_batch_size",
             "sglang_num_running_reqs", "vllm:scheduler_running",
             "tgi_batch_current_size",
             "nv_trt_llm_inflight_batcher_requests_active"],
   regex_patterns=[r"num_running_reqs", r"running.*req", r"active.*req",
                   r"inflight.*active", r"batch.*current.*size"],
   category="inference", subcategory="concurrency", section="§A1",
   unit="sessions", higher_is_better=None,
   description="Requests currently in prefill or decode phase.",
   formula=(
       "SGLang  : sglang_num_running_reqs\n"
       "vLLM    : sglang_num_running_reqs\n"
       "TGI     : tgi_batch_current_size\n"
       "TRT-LLM : nv_trt_llm_inflight_batcher_requests_active"
   ),
   sources=["sglang", "vllm", "tgi", "tensorrt-llm"],
   related=["num_queue_reqs", "tpot", "throughput"],
   profiler_field="sess_max_concurrent")

_r("num_queue_reqs", "Queued Requests",
   aliases=["num_queue_reqs", "queued_requests", "pending_requests",
             "sglang_num_queue_reqs", "vllm:scheduler_waiting",
             "nv_trt_llm_inflight_batcher_requests_paused"],
   regex_patterns=[r"num_queue_reqs", r"queued.*req", r"waiting.*req",
                   r"requests_waiting", r"pending.*req"],
   category="inference", subcategory="concurrency", section="§A2",
   unit="sessions", higher_is_better=False, normal_range=(0, 50),
   description="Requests waiting for GPU capacity.",
   formula=(
       "SGLang  : sglang_num_queue_reqs\n"
       "vLLM    : sglang_num_queue_reqs\n"
       "TRT-LLM : nv_trt_llm_inflight_batcher_requests_paused"
   ),
   sources=["sglang", "vllm", "tensorrt-llm"],
   related=["num_running_reqs", "queue_time"],
   profiler_field="sess_queued_mean")

_r("num_preemptions", "KV Preemption / Swap Count",
   aliases=["num_preemptions", "preemptions_total", "swap_count",
             "sglang_num_preemptions_total (if exported)", "vllm:num_requests_swapped"],
   regex_patterns=[r"preempt", r"num_requests_swapped"],
   category="inference", subcategory="concurrency",
   unit="count", higher_is_better=False,
   description="vLLM: requests preempted/swapped to CPU due to KV memory pressure. "
               "High value → KV pool too small.",
   formula="sglang_num_preemptions_total (if exported)  (cumulative)",
   sources=["vllm"],
   related=["kv_pool_fill", "num_running_reqs"],
   profiler_field="")


# ===========================================================================
#  A. INFERENCE — KV CACHE
# ===========================================================================

_r("cache_hit_rate", "KV Cache Hit Rate",
   aliases=["cache_hit_rate", "kv_cache_hit", "prefix_cache_hit", "radix_cache_hit",
             "cache_efficiency", "kv_hit_rate",
             "vllm:prefix_cache_hit_rate", "prefix_cache_hit_rate"],
   regex_patterns=[r"cache_hit_rate", r"prefix_cache_hit", r"kv.*hit",
                   r"radix.*hit", r"token_reuse"],
   category="inference", subcategory="kv_cache",
   unit="%", higher_is_better=True, normal_range=(50, 99),
   description="Fraction of prefill tokens served from KV cache (no recomputation). "
               "High hit rate → less TTFT, fewer GPU FLOPs.",
   formula=(
       "SGLang : sglang_cache_hit_rate  (0–1)\n"
       "vLLM   : vllm:prefix_cache_hit_rate\n"
       "Manual : prefill_cache_tokens / (cache_tokens + compute_tokens)"
   ),
   sources=["sglang", "vllm", "tgi"],
   related=["ttft", "kv_pool_fill", "hicache_fill"],
   profiler_field="cache_hit_pct")

_r("kv_pool_fill", "KV Pool Utilisation",
   aliases=["token_usage", "kv_usage", "kv_pool_fill", "full_token_usage",
             "kv_cache_utilization", "gpu_cache_usage_perc",
             "sglang_kv_used_tokens / configured KV token capacity", "vllm:cpu_cache_usage_perc",
             "nv_trt_llm_kv_cache_block_manager_used_num_blocks"],
   regex_patterns=[r"gpu_cache_usage", r"kv.*util", r"token_usage",
                   r"full_token_usage", r"kv.*pool.*fill", r"kv.*block.*used"],
   category="inference", subcategory="kv_cache",
   unit="%", higher_is_better=None, normal_range=(10, 85),
   description="Fraction of L1 HBM KV token pool occupied. >85% → offload/eviction begins.",
   formula=(
       "SGLang  : sglang_full_token_usage  (0–1)\n"
       "vLLM    : sglang_kv_used_tokens / configured KV token capacity  (0–1)\n"
       "TRT-LLM : used_num_blocks / max_num_blocks"
   ),
   sources=["sglang", "vllm", "tensorrt-llm"],
   related=["hicache_fill", "kvb_offload_rate", "kv_bytes_per_token"],
   profiler_field="kv_pool_fill_pct")

_r("kv_bytes_per_token", "KV Cache Bytes per Token",
   aliases=["kv_bytes_per_token", "kv_footprint", "kv_memory_per_token",
             "tokens_per_kv_block", "page_size",
             "nv_trt_llm_kv_cache_block_manager_tokens_per_block"],
   regex_patterns=[r"kv.*bytes.*token", r"tokens.*per.*block",
                   r"kv_block.*token", r"cache.*byte.*token"],
   category="inference", subcategory="kv_cache", section="§A17",
   unit="bytes", higher_is_better=False,
   description="Memory per KV token. Determined by model architecture (layers, heads, dtype).",
   formula=(
       "GQA/MHA: 2 × n_layers × n_kv_heads × head_dim × dtype_bytes\n"
       "MLA    : 2 × n_layers × latent_dim × dtype_bytes\n"
       "Llama-70B FP16 GQA: 2×80×8×128×2 = 327,680 bytes"
   ),
   sources=["derived", "sglang_model_config", "tensorrt-llm"],
   related=["kv_pool_fill", "hbm_used"],
   profiler_field="kv_bytes_per_token")

_r("hicache_fill", "HiCache L2 DRAM Fill (SGLang)",
   aliases=["hicache_fill", "hicache_host_fill", "l2_kv_fill",
             "hicache_host_used_tokens", "kv_l2_fill"],
   regex_patterns=[r"hicache_host_used", r"hicache.*fill", r"l2.*kv.*fill"],
   category="inference", subcategory="kv_cache", section="§B",
   unit="%", higher_is_better=None,
   description="SGLang: fraction of L2 hicache DRAM KV pool in use.",
   formula="sglang_hicache_host_used_tokens / hicache_host_total_tokens × 100",
   sources=["sglang"],
   related=["kv_pool_fill", "kvb_offload_rate"],
   profiler_field="hicache_fill_pct")


# ===========================================================================
#  A. INFERENCE — KV BLOCK MOVEMENT
# ===========================================================================

_r("kvb_offload_rate", "KV Block Offload Rate L1→L2",
   aliases=["backuped_tokens_total", "kv_offload_rate", "kvb_backup_rate",
             "backup_bandwidth", "l1_to_l2_rate"],
   regex_patterns=[r"backuped_tokens", r"backup.*rate", r"offload.*kv",
                   r"backup_bandwidth"],
   category="inference", subcategory="kv_movement", section="§B22",
   unit="tok/s", higher_is_better=False,
   description="Rate KV tokens offload from L1 HBM to L2 DRAM hicache. "
               "Non-zero → KV pool pressure.",
   formula=(
       "rate(sglang_backuped_tokens_total[W])\n"
       "offload_BW_Bps = rate × kv_bytes_per_token"
   ),
   sources=["sglang"],
   related=["kv_pool_fill", "dram_bw"],
   profiler_field="kvb_offload_tok_rate")

_r("kvb_onboard_rate", "KV Block Onboard Rate L2→L1",
   aliases=["prefetched_tokens_total", "kv_onboard_rate", "kv_load_back",
             "load_back_bandwidth", "l2_to_l1_rate"],
   regex_patterns=[r"prefetched_tokens", r"load_back", r"onboard.*kv",
                   r"load_back_bandwidth"],
   category="inference", subcategory="kv_movement", section="§B25",
   unit="tok/s", higher_is_better=None,
   description="Rate KV tokens prefetched/onboarded from the backing tier. This is the logical L3 read metric; load_back is diagnostic-only.",
   formula="rate(sglang_prefetched_tokens_total[W])",
   sources=["sglang"],
   related=["kvb_offload_rate", "cache_hit_rate"],
   profiler_field="kvb_onboard_tok_rate")

_r("kv_evict_rate", "KV Token Eviction Rate",
   aliases=["evicted_tokens_total", "kv_evict", "eviction_rate"],
   regex_patterns=[r"evicted_tokens", r"eviction.*rate", r"kv.*evict"],
   category="inference", subcategory="kv_movement", section="§B21",
   unit="tok/s", higher_is_better=False,
   description="Rate KV tokens are permanently evicted from all cache tiers.",
   formula="rate(sglang_evicted_tokens_total[W])",
   sources=["sglang"],
   related=["kvb_offload_rate", "kv_pool_fill"],
   profiler_field="kvb_evicted_tokens_total")


# ===========================================================================
#  A. INFERENCE — TOKENS & MISC
# ===========================================================================

_r("prompt_tokens", "Prompt Input Token Count",
   aliases=["prompt_tokens", "input_tokens", "prompt_length", "request_prompt_tokens",
             "average_prompt_len", "p90_prompt_len", "num_prompt_tokens",
             "sglang_prompt_tokens_total deltas or histogram when exported", "tgi_request_input_length"],
   regex_patterns=[r"prompt_token", r"input_token", r"prompt_len",
                   r"input_len", r"prompt.*tokens"],
   category="inference", subcategory="tokens",
   unit="tokens", higher_is_better=None,
   description="Input prompt token count per request.",
   formula="sglang_prompt_tokens_total deltas or histogram when exported histogram\nCSV: average_prompt_len / p90_prompt_len",
   sources=["sglang", "vllm", "tgi", "vllm-bench", "llmperf"],
   related=["ttft", "cache_hit_rate"],
   profiler_field="sess_input_tok_mean")

_r("generation_tokens", "Generation Output Token Count",
   aliases=["generation_tokens", "output_tokens", "output_length",
             "request_generation_tokens", "average_output_len",
             "sglang_generation_tokens_total deltas or histogram when exported", "tgi_request_generated_tokens"],
   regex_patterns=[r"generation_token", r"output_token", r"output_len",
                   r"generated_token", r"completion_token"],
   category="inference", subcategory="tokens",
   unit="tokens", higher_is_better=None,
   description="Output token count per request.",
   formula="sglang_generation_tokens_total deltas or histogram when exported\nCSV: average_output_len",
   sources=["sglang", "vllm", "tgi", "vllm-bench", "llmperf"],
   related=["tpot", "e2e_latency"],
   profiler_field="sess_output_tok_mean")

_r("spec_decode_acceptance", "Speculative Decoding Acceptance Rate",
   aliases=["spec_decode_draft_acceptance_rate", "draft_acceptance",
             "SGLang speculative decode acceptance metric, when exported", "spec_decode_efficiency"],
   regex_patterns=[r"spec_decode.*accept", r"draft.*accept", r"speculative.*accept"],
   category="inference", subcategory="spec_decoding",
   unit="%", higher_is_better=True, normal_range=(50, 95),
   description="Fraction of draft tokens accepted by verifier. Higher → more speedup.",
   formula="SGLang speculative decode acceptance metric, when exported  (0–1)",
   sources=["vllm", "sglang"],
   related=["gen_throughput", "tpot"],
   profiler_field="")

_r("mfu", "Model FLOPs Utilisation",
   aliases=["MFU", "model_flops_utilization", "flop_efficiency"],
   regex_patterns=[r"\bMFU\b", r"model_flop.*util", r"flop.*efficiency"],
   category="inference", subcategory="efficiency",
   unit="%", higher_is_better=True, normal_range=(10, 70),
   description="Fraction of theoretical peak FLOPs used. LLM decode: typically 10–30%.",
   formula="MFU = actual_flops / (peak_flops × time)\nProxy: TPOT_min / TPOT_observed",
   sources=["derived", "nanoGPT", "palm", "megatron"],
   related=["tpot", "gpu_util", "hbm_bw"],
   profiler_field="")


# ===========================================================================
#  B. GPU METRICS
# ===========================================================================

_r("gpu_util", "GPU SM Utilisation",
   aliases=["gpu_util", "gpu_utilization", "sm_utilization", "DCGM_FI_DEV_GPU_UTIL",
             "DCGM_FI_PROF_GR_ENGINE_ACTIVE", "utilization.gpu", "gpu_active"],
   regex_patterns=[r"DCGM_FI_DEV_GPU_UTIL", r"GR_ENGINE_ACTIVE",
                   r"gpu_util", r"sm_util", r"utilization\.gpu"],
   category="gpu", subcategory="utilisation",
   unit="%", higher_is_better=True, normal_range=(30, 100),
   description="Fraction of time ≥1 SM warp active. 100% ≠ peak perf — also check HBM BW.",
   formula=(
       "DCGM_FI_DEV_GPU_UTIL  (0–100)\n"
       "nvidia-smi: utilization.gpu\n"
       "DCGM_FI_PROF_GR_ENGINE_ACTIVE  (0–1, more precise)"
   ),
   sources=["dcgm", "nvidia-smi", "nvml", "ncu"],
   related=["hbm_bw", "tpot", "gpu_power"],
   profiler_field="gpu_util_mean")

_r("hbm_bw", "HBM Bandwidth Utilisation",
   aliases=["hbm_bw", "DCGM_FI_PROF_DRAM_ACTIVE", "hbm_bandwidth",
             "gpu_memory_bandwidth", "hbm_bw_util", "DRAM_ACTIVE"],
   regex_patterns=[r"DCGM_FI_PROF_DRAM_ACTIVE", r"hbm.*bw",
                   r"hbm.*util", r"DRAM_ACTIVE"],
   category="gpu", subcategory="memory_bandwidth", section="§F74-78",
   unit="%", higher_is_better=None, normal_range=(30, 90),
   description="Fraction of HBM cycles with outstanding transactions. "
               "A100: 2 TB/s peak. Est_BW = DRAM_ACTIVE × 2000 GB/s.",
   formula="DCGM_FI_PROF_DRAM_ACTIVE  (0–1)\nEst_BW_GBs = DRAM_ACTIVE × 2000",
   sources=["dcgm", "ncu"],
   related=["tpot", "gpu_util", "hbm_used"],
   profiler_field="hbm_bw_util_pct")

_r("hbm_used", "HBM Memory Used",
   aliases=["hbm_used", "fb_used", "gpu_memory_used", "vram_used",
             "DCGM_FI_DEV_FB_USED", "memory.used"],
   regex_patterns=[r"FB_USED", r"memory_used", r"vram_used",
                   r"hbm.*used", r"memory\.used"],
   category="gpu", subcategory="memory", section="§F74",
   unit="MiB", higher_is_better=None,
   description="GPU HBM in use: weights + KV pool + activations.",
   formula="DCGM_FI_DEV_FB_USED  [MiB]\nnvidia-smi: memory.used",
   sources=["dcgm", "nvidia-smi"],
   related=["kv_pool_fill", "hbm_bw"],
   profiler_field="hbm_used_mb_mean")

_r("gpu_power", "GPU Power Draw",
   aliases=["gpu_power", "power_draw", "power_usage", "DCGM_FI_DEV_POWER_USAGE",
             "power.draw", "ipmi_power_watts", "ipmi_dcmi_power_consumption_watts"],
   regex_patterns=[r"DCGM.*POWER", r"power_usage", r"power_draw",
                   r"gpu.*power", r"ipmi.*power"],
   category="gpu", subcategory="power",
   unit="W", higher_is_better=None, normal_range=(100, 400),
   description="GPU power consumption. A100 TDP: 400W. Idle: ~50–60W.",
   formula=(
       "DCGM_FI_DEV_POWER_USAGE  [W]\n"
       "nvidia-smi: power.draw\n"
       "IPMI: ipmi_power_watts / ipmi_dcmi_power_consumption_watts"
   ),
   sources=["dcgm", "nvidia-smi", "ipmi"],
   related=["gpu_util", "hbm_bw"],
   profiler_field="gpu_power_mean_w")

_r("gpu_temp", "GPU Temperature",
   aliases=["gpu_temp", "DCGM_FI_DEV_GPU_TEMP", "temperature.gpu"],
   regex_patterns=[r"DCGM.*TEMP", r"gpu.*temp", r"temperature\.gpu"],
   category="gpu", subcategory="thermal",
   unit="°C", higher_is_better=False, normal_range=(40, 83),
   description="A100 thermal throttle threshold: 83°C.",
   formula="DCGM_FI_DEV_GPU_TEMP  [°C]",
   sources=["dcgm", "nvidia-smi"],
   related=["gpu_power"],
   profiler_field="")

_r("tensor_active", "Tensor Core Pipeline Activity",
   aliases=["tensor_active", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "pipe_tensor_active"],
   regex_patterns=[r"PIPE_TENSOR_ACTIVE", r"tensor_active", r"tensor.*util"],
   category="gpu", subcategory="utilisation",
   unit="%", higher_is_better=True, normal_range=(10, 80),
   description="Fraction of cycles tensor cores are active. Low in decode, high in prefill.",
   formula="DCGM_FI_PROF_PIPE_TENSOR_ACTIVE  (0–1)",
   sources=["dcgm"],
   related=["gpu_util", "mfu"],
   profiler_field="")

_r("sm_clock", "SM Clock",
   aliases=["sm_clock", "DCGM_FI_DEV_SM_CLOCK", "clocks.current.graphics"],
   regex_patterns=[r"DCGM.*SM_CLOCK", r"sm_clock", r"clocks.*graphics"],
   category="gpu", subcategory="clock",
   unit="MHz", higher_is_better=True, normal_range=(900, 1410),
   description="A100 SM clock. Base: 765 MHz, boost: 1410 MHz.",
   formula="DCGM_FI_DEV_SM_CLOCK  [MHz]",
   sources=["dcgm", "nvidia-smi"],
   related=["gpu_util", "gpu_power"],
   profiler_field="")

_r("pcie_bw", "PCIe Bandwidth",
   aliases=["pcie_bw", "DCGM_FI_PROF_PCIE_RX_BYTES", "DCGM_FI_PROF_PCIE_TX_BYTES"],
   regex_patterns=[r"PCIE.*BYTES", r"pcie.*bw", r"pcie.*transfer"],
   category="gpu", subcategory="interconnect",
   unit="MB/s", higher_is_better=None,
   description="PCIe host↔GPU rate. A100 PCIe 4.0 ×16 = 64 GB/s.",
   formula="DCGM_FI_PROF_PCIE_RX_BYTES + DCGM_FI_PROF_PCIE_TX_BYTES",
   sources=["dcgm"],
   related=["hbm_bw"],
   profiler_field="")


# ===========================================================================
#  C. SYSTEM MEMORY
# ===========================================================================

_r("dram_bw", "System DRAM Bandwidth",
   aliases=["dram_bw", "dram_bandwidth", "memory_bandwidth", "mem_bw",
             "dram_read_bw", "dram_write_bw", "dram_total_bw",
             "Total Mem Bw", "MemRdBw", "MemWrBw"],
   regex_patterns=[r"dram.*bw", r"mem.*bandwidth", r"Total.*Mem.*Bw",
                   r"\bMemRdBw\b", r"\bMemWrBw\b"],
   category="memory", subcategory="bandwidth", section="§F82-83",
   unit="GB/s", higher_is_better=None, normal_range=(10, 200),
   description="System DRAM BW. AMD EPYC 7742 peak: 204.8 GB/s. "
               "Includes KV$B DMA, model loader, OS page cache.",
   formula=(
       "AMDuProf : amduprof_pcm_raw.csv → 'Total Mem Bw (GB/s)'\n"
       "Intel PCM: READ + WRITE columns\n"
       "node_exporter: does not provide BW directly — use AMDuProf"
   ),
   sources=["amduprof", "intel-pcm", "perf", "pmu"],
   related=["hbm_bw", "kvb_offload_rate"],
   profiler_field="dram_total_bw_gbs")

_r("dram_free", "System DRAM Available",
   aliases=["dram_free", "mem_available", "MemAvailable", "memory_available",
             "node_memory_MemAvailable_bytes"],
   regex_patterns=[r"MemAvailable", r"memory_available",
                   r"mem_free", r"node_memory_Mem"],
   category="memory", subcategory="capacity", section="§F79-81",
   unit="GB", higher_is_better=True,
   description="Available system memory (includes reclaimable page cache).",
   formula="node_memory_MemAvailable_bytes / 1e9",
   sources=["node_exporter", "procfs"],
   related=["hicache_fill", "kvb_offload_rate"],
   profiler_field="")

_r("page_faults", "Page Faults/s",
   aliases=["page_faults", "pgfault", "pgmajfault", "minor_faults",
             "node_vmstat_pgfault", "node_vmstat_pgmajfault"],
   regex_patterns=[r"pgfault", r"pgmajfault", r"page_fault", r"vmstat.*pgf"],
   category="memory", subcategory="virtual_memory", section="§F86-87",
   unit="/s", higher_is_better=False,
   description="Minor (no disk) and major (disk needed) page faults. "
               "Major faults should be zero on DGX.",
   formula=(
       "rate(node_vmstat_pgfault[W])     minor\n"
       "rate(node_vmstat_pgmajfault[W])  major"
   ),
   sources=["node_exporter", "procfs", "vmstat"],
   related=["dram_bw", "numa_migrations"],
   profiler_field="")

_r("numa_migrations", "NUMA Page Migrations/s",
   aliases=["numa_migrations", "numa_pages_migrated",
             "node_vmstat_numa_pages_migrated"],
   regex_patterns=[r"numa.*migrat", r"numa_pages_migrated"],
   category="memory", subcategory="numa",
   unit="/s", higher_is_better=False,
   description="NUMA-crossed page migrations. High rate → remote memory latency penalty.",
   formula="rate(node_vmstat_numa_pages_migrated[W])",
   sources=["node_exporter", "numastat"],
   related=["dram_bw"],
   profiler_field="")


# ===========================================================================
#  D. SSD / NVMe IO
# ===========================================================================

_r("ssd_rd_bw", "SSD Read Bandwidth",
   aliases=["ssd_rd_bw", "nvme_read_bw", "disk_read_bw", "read_bandwidth_mbs",
             "rkB/s", "rMB/s", "rkb_s", "r_mb_s"],
   regex_patterns=[r"disk_read_bytes_total", r"read.*bw", r"rkb.*s\b",
                   r"rmb.*s\b", r"read_bytes_total"],
   category="ssd_io", subcategory="bandwidth", section="§C32",
   unit="MB/s", higher_is_better=True,
   description="L3 (local storage) read throughput. KV$B onboard reads are random 128KB → BW << sequential peak.",
   formula=(
       "node_exporter: rate(node_disk_read_bytes_total{device=nvme*}[W]) / 1e6\n"
       "iostat       : rkB/s or rMB/s\n"
       "blktrace     : sum(size_bytes[R]) / duration / 1e6"
   ),
   sources=["node_exporter", "iostat", "dstat", "blktrace", "biosnoop", "sysfs"],
   related=["kvb_onboard_rate", "ssd_rd_iops"],
   profiler_field="ssd_rd_bw_mbs_mean")

_r("ssd_wr_bw", "SSD Write Bandwidth",
   aliases=["ssd_wr_bw", "nvme_write_bw", "disk_write_bw", "write_bandwidth_mbs",
             "wkB/s", "wMB/s", "wkb_s", "w_mb_s"],
   regex_patterns=[r"disk_written_bytes_total", r"written.*bytes_total",
                   r"write.*bw", r"wkb.*s\b", r"wmb.*s\b"],
   category="ssd_io", subcategory="bandwidth", section="§D47",
   unit="MB/s", higher_is_better=None,
   description="NVMe write BW. KV$B offload: write_bw ≈ offload_rate × kv_block_MB.",
   formula=(
       "node_exporter: rate(node_disk_written_bytes_total{device=nvme*}[W]) / 1e6\n"
       "iostat       : wkB/s or wMB/s"
   ),
   sources=["node_exporter", "iostat", "dstat", "blktrace", "biosnoop"],
   related=["kvb_offload_rate", "ssd_wr_iops"],
   profiler_field="ssd_wr_bw_mbs_mean")

_r("ssd_wr_iops", "SSD Write IOPS",
   aliases=["ssd_wr_iops", "nvme_write_iops", "disk_write_iops",
             "w/s", "wr_ios_rate", "nvme_wr_iops"],
   regex_patterns=[r"writes_completed_total", r"write.*iops", r"\bw_s\b"],
   category="ssd_io", subcategory="iops",
   unit="IOPS", higher_is_better=None,
   description="NVMe write IOPS. Each write = one KV block offload (128KB). "
               "Zero expected when KV fits in HBM+DRAM.",
   formula=(
       "rate(node_disk_writes_completed_total{device=nvme*}[W])\n"
       "iostat: w/s\nblktrace: count(W events) / duration"
   ),
   sources=["node_exporter", "iostat", "blktrace", "biosnoop"],
   related=["ssd_wr_bw", "kvb_offload_rate"],
   profiler_field="nvme_wr_iops_mean")

_r("ssd_rd_iops", "SSD Read IOPS",
   aliases=["ssd_rd_iops", "nvme_read_iops", "disk_read_iops", "r/s"],
   regex_patterns=[r"reads_completed_total", r"read.*iops", r"\br_s\b"],
   category="ssd_io", subcategory="iops",
   unit="IOPS", higher_is_better=True,
   description="NVMe read IOPS. Each 128KB read = one KV block onboard.",
   formula=(
       "rate(node_disk_reads_completed_total{device=nvme*}[W])\n"
       "iostat: r/s"
   ),
   sources=["node_exporter", "iostat", "blktrace", "biosnoop"],
   related=["ssd_rd_bw", "kvb_onboard_rate"],
   profiler_field="")

_r("ssd_rd_lat", "SSD Read Latency",
   aliases=["ssd_rd_lat", "nvme_read_latency", "r_await", "read_latency"],
   regex_patterns=[r"read_time_seconds_total", r"\br_await\b", r"read.*latency"],
   category="ssd_io", subcategory="latency",
   unit="ms", higher_is_better=False, normal_range=(0.05, 5.0),
   description="Average read request latency.",
   formula=(
       "Δ(node_disk_read_time_seconds_total) / Δ(reads_completed) × 1000\n"
       "iostat: r_await"
   ),
   sources=["node_exporter", "iostat", "biosnoop"],
   related=["ssd_rd_iops"],
   profiler_field="nvme_rd_lat_mean_ms")

_r("ssd_wr_lat", "SSD Write Latency",
   aliases=["ssd_wr_lat", "nvme_write_latency", "w_await", "write_latency"],
   regex_patterns=[r"write_time_seconds_total", r"\bw_await\b", r"write.*latency"],
   category="ssd_io", subcategory="latency",
   unit="ms", higher_is_better=False, normal_range=(0.05, 10.0),
   description="Average write request latency.",
   formula=(
       "Δ(node_disk_write_time_seconds_total) / Δ(writes_completed) × 1000\n"
       "iostat: w_await"
   ),
   sources=["node_exporter", "iostat", "biosnoop"],
   related=["ssd_wr_iops", "kvb_offload_rate"],
   profiler_field="nvme_wr_lat_mean_ms")

_r("ssd_io_util", "SSD IO Utilisation",
   aliases=["ssd_io_util", "disk_util", "io_util", "nvme_util",
             "%util", "nvme_io_util"],
   regex_patterns=[r"io_time_seconds_total", r"disk.*util", r"io.*util"],
   category="ssd_io", subcategory="utilisation",
   unit="%", higher_is_better=None, normal_range=(0, 80),
   description="Fraction of time NVMe had outstanding IO. 100% = saturated.",
   formula=(
       "Δ(node_disk_io_time_seconds_total) / Δ(time_sec) × 100\n"
       "iostat: %util"
   ),
   sources=["node_exporter", "iostat"],
   related=["ssd_rd_iops", "ssd_wr_iops"],
   profiler_field="nvme_io_util_pct")

_r("ssd_queue_depth", "SSD Queue Depth",
   aliases=["ssd_queue_depth", "io_queue", "aqu-sz", "avgqu-sz",
             "inflight", "io_now"],
   regex_patterns=[r"disk_io_now", r"queue_depth", r"\baqu\b", r"\binflight\b"],
   category="ssd_io", subcategory="queueing",
   unit="requests", higher_is_better=None, normal_range=(1, 32),
   description="Mean requests in-flight. iostat: aqu-sz (time-averaged).",
   formula="node_disk_io_now  (instantaneous)\niostat: aqu-sz",
   sources=["node_exporter", "iostat"],
   related=["ssd_io_util"],
   profiler_field="nvme_inflight_mean")

_r("trim_iops", "Trim / Discard IOPS",
   aliases=["trim_iops", "discard_iops", "nvme_trim", "d/s",
             "node_disk_discards_completed_total"],
   regex_patterns=[r"discards_completed", r"discard.*iops", r"trim.*iops"],
   category="ssd_io", subcategory="trim", section="§E61-63",
   unit="IOPS", higher_is_better=None,
   description="TRIM/discard IOPS. In KV$B: proxy for evicted block rate.",
   formula=(
       "rate(node_disk_discards_completed_total{device=nvme*}[W])\n"
       "blktrace: count(D events) / duration"
   ),
   sources=["node_exporter", "iostat", "blktrace"],
   related=["kv_evict_rate", "smart_wear"],
   profiler_field="trim_iops_mean")

_r("smart_wear", "SSD Endurance Used",
   aliases=["smart_wear", "percent_used", "percentage_used",
             "endurance_used", "nvme_wear"],
   regex_patterns=[r"percent.*used", r"percentage_used",
                   r"endurance_used", r"smart.*wear"],
   category="ssd_io", subcategory="health",
   unit="%", higher_is_better=False, normal_range=(0, 50),
   description="Cumulative % of rated NAND endurance consumed (SMART).",
   formula=(
       "nvme smart-log → percentage_used\n"
       "JSONL poll: outer.stdout → inner.percentage_used"
   ),
   sources=["nvme-cli", "smartctl"],
   related=["smart_spare", "ssd_wr_bw"],
   profiler_field="nvme_smart_wear_pct")

_r("smart_spare", "SSD Available Spare",
   aliases=["smart_spare", "avail_spare", "available_spare"],
   regex_patterns=[r"avail_spare", r"available_spare"],
   category="ssd_io", subcategory="health",
   unit="%", higher_is_better=True, normal_range=(25, 100),
   description="Spare NAND blocks remaining for bad-block replacement.",
   formula="nvme smart-log → avail_spare",
   sources=["nvme-cli"],
   related=["smart_wear"],
   profiler_field="nvme_smart_avail_spare")

_r("smart_data_written", "Lifetime Bytes Written (SMART)",
   aliases=["data_units_written", "smart_bytes_written", "host_write_bytes"],
   regex_patterns=[r"data_units_written", r"smart.*written"],
   category="ssd_io", subcategory="endurance",
   unit="GB", higher_is_better=None,
   description="Bytes written since manufacture. Delta = bytes written this run.",
   formula="Δ(data_units_written) × 512 × 1000  [bytes]",
   sources=["nvme-cli"],
   related=["smart_wear", "ssd_wr_bw"],
   profiler_field="nvme_bytes_written_total")


# ===========================================================================
#  E. BLOCK IO — blktrace / biosnoop / iostat / sysfs
# ===========================================================================

_r("blk_iat", "IO Inter-Arrival Time",
   aliases=["inter_arrival_time", "iat", "io_iat", "inter_arrival_sec",
             "inter_arrival"],
   regex_patterns=[r"inter_arrival", r"\biat\b", r"arrival.*time"],
   category="block_io", subcategory="temporal", section="§C34 §D49",
   unit="µs", higher_is_better=None,
   description="Time between consecutive IOs. <1ms → burst. "
               "Source: interarrival_distribution.csv.",
   formula=(
       "IAT_i = ts_i - ts_{i-1}  [µs]  (same op, sorted by timestamp)\n"
       "Source: interarrival_distribution.csv → inter_arrival_sec"
   ),
   sources=["blktrace", "biosnoop", "bpftrace"],
   related=["ssd_rd_iops", "blk_burst_shape"],
   profiler_field="ssd_rd_iat_mean_us")

_r("blk_alignment", "Block IO KV / Page Alignment",
   aliases=["io_alignment", "kv_block_aligned", "page_aligned",
             "not_kv_block_aligned", "kv_aligned"],
   regex_patterns=[r"kv.*aligned", r"page_aligned",
                   r"not_kv_block_aligned", r"io.*align"],
   category="block_io", subcategory="access_pattern", section="§C40-41",
   unit="%", higher_is_better=True,
   description="Fraction of IOs aligned to KV block (128KB) and page (4KB). "
               "Misalignment → read-modify-write. KV$B: always 100%.",
   formula=(
       "kvb_aligned = (offset % 131072 == 0) AND (size % 131072 == 0)\n"
       "Source: request_size_distribution.csv / summary.json"
   ),
   sources=["blktrace", "biosnoop"],
   related=["blk_req_size"],
   profiler_field="ssd_rd_kvb_aligned_pct")

_r("blk_seq_ratio", "Sequential IO Ratio",
   aliases=["sequential_ratio", "seq_pct", "sequential_access",
             "sequential_count", "random_count"],
   regex_patterns=[r"seq.*pct", r"sequential.*ratio",
                   r"access_pattern", r"sequential_count"],
   category="block_io", subcategory="access_pattern", section="§C42 §D57",
   unit="%", higher_is_better=None,
   description="Fraction of IOs that are sequential. KV$B: 0% (all random).",
   formula=(
       "seq_i = (sector_i == prev_sector + prev_nr_sectors)\n"
       "Source: request_size_distribution.csv → access_area=='sequential'"
   ),
   sources=["blktrace", "biosnoop"],
   related=["blk_req_size"],
   profiler_field="ssd_rd_seq_pct")

_r("blk_req_size", "IO Request Size",
   aliases=["request_size", "io_size", "request_size_bytes",
             "size_bytes", "nsectors"],
   regex_patterns=[r"request_size", r"req_size", r"io_size",
                   r"\bnsectors\b"],
   category="block_io", subcategory="size", section="§C38-39",
   unit="KB", higher_is_better=None,
   description="Size per IO request. KV$B: always 128KB (one block).",
   formula=(
       "size_bytes = nsectors × 512  [blktrace]\n"
       "Source: request_size_distribution.csv"
   ),
   sources=["blktrace", "biosnoop", "bpftrace"],
   related=["blk_alignment"],
   profiler_field="ssd_rd_size_mean_kb")

_r("blk_hot_cold", "LBA Hot/Cold Distribution",
   aliases=["hot_regions", "lba_distribution", "access_skew",
             "gini_bytes", "skewness_bytes", "top_1pct_byte_share"],
   regex_patterns=[r"hot_region", r"gini_bytes", r"skewness_bytes",
                   r"access_skew", r"top.*pct.*share"],
   category="block_io", subcategory="access_pattern", section="§C36-37",
   unit="", higher_is_better=None,
   description="LBA access distribution. Gini=0: uniform; Gini→1: one hot region. "
               "Source: hot_regions_overall.csv, access_skew_summary.csv.",
   formula=(
       "region_id = floor(sector × 512 / kv_block_bytes)\n"
       "Gini = (2×Σ(rank×bytes)/(N×total)) - (N+1)/N\n"
       "Source: access_skew_summary.csv → gini_bytes, skewness_bytes"
   ),
   sources=["blktrace", "biosnoop"],
   related=["blk_seq_ratio"],
   profiler_field="ssd_rd_gini")

_r("blk_bw_degradation", "IO Bandwidth Degradation Ratio",
   aliases=["bw_degradation", "bandwidth_degradation",
             "degradation_ratio_late_over_early", "degradation_ratio"],
   regex_patterns=[r"bw_degrad", r"bandwidth_degradation",
                   r"degradation_ratio", r"late.*early"],
   category="block_io", subcategory="temporal", section="§C45 §D60",
   unit="ratio", higher_is_better=True, normal_range=(0.9, 1.05),
   description="BW(last 10%) / BW(first 10%). <1.0 → thermal or write-cliff degradation.",
   formula=(
       "early_bw = mean BW over first 10% of trace\n"
       "late_bw  = mean BW over last  10% of trace\n"
       "ratio    = late_bw / early_bw\n"
       "Source: bandwidth_degradation.csv → degradation_ratio_late_over_early"
   ),
   sources=["blktrace", "biosnoop"],
   related=["ssd_rd_bw", "smart_wear"],
   profiler_field="ssd_rd_bw_degradation")

_r("blk_per_stream_bw", "Per-Stream IO Bandwidth",
   aliases=["stream_bandwidth", "per_session_bw", "bandwidth_mib_s",
             "stream_bw", "bandwidth_per_stream"],
   regex_patterns=[r"bandwidth_per_stream", r"stream.*bw",
                   r"bandwidth_mib_s"],
   category="block_io", subcategory="bandwidth", section="§C",
   unit="MiB/s", higher_is_better=True,
   description="BW per inference stream (trace_window:pid = one session). "
               "Source: bandwidth_per_stream.csv.",
   formula=(
       "stream_bw = stream_total_bytes / stream_duration_sec  [MiB/s]\n"
       "stream_id = trace_window_id : pid"
   ),
   sources=["blktrace"],
   related=["ssd_rd_bw", "blk_iat"],
   profiler_field="ssd_rd_bw_per_stream_mean_mbs")

_r("blk_burst_shape", "IO Burst Temporal Shape",
   aliases=["burst_shape", "burst_temporal", "burst_windows",
             "burst_count", "burst_dynamics"],
   regex_patterns=[r"burst_temporal", r"burst_window", r"burst_count"],
   category="block_io", subcategory="temporal", section="§C35 §D50",
   unit="", higher_is_better=None,
   description="Temporal IO burst clustering. "
               "Source: burst_temporal_windows.csv.",
   formula=(
       "burst_start = min(ts) where IAT < threshold (e.g. 1ms)\n"
       "burst_bytes = sum(size_bytes in window)\n"
       "Source: burst_temporal_windows.csv"
   ),
   sources=["blktrace", "biosnoop"],
   related=["blk_iat", "ssd_rd_iops"],
   profiler_field="")

_r("iostat_await", "iostat IO Await",
   aliases=["await", "r_await", "w_await", "average_wait"],
   regex_patterns=[r"\br_await\b", r"\bw_await\b", r"\bawait\b"],
   category="block_io", subcategory="iostat",
   unit="ms", higher_is_better=False, normal_range=(0.05, 10),
   description="Average IO wait time (queue + service) from iostat -x.",
   formula="iostat -xm: r_await (read), w_await (write) columns",
   sources=["iostat"],
   related=["ssd_rd_lat"],
   profiler_field="")

_r("biosnoop_latency", "biosnoop Per-Request Latency",
   aliases=["biosnoop_lat", "bio_latency", "bpf_io_lat"],
   regex_patterns=[r"biosnoop.*lat", r"bio.*latency", r"bpf.*io.*lat"],
   category="block_io", subcategory="biosnoop",
   unit="µs", higher_is_better=False,
   description="Per-request kernel block-layer latency via eBPF. "
               "More accurate than iostat (per-request, not averaged).",
   formula=(
       "sudo biosnoop -d nvme2n1 > trace.csv\n"
       "CSV: TIME, COMM, PID, TYPE, DEV, SECTOR, BYTES, LAT(ms)"
   ),
   sources=["biosnoop", "bcc", "bpftrace"],
   related=["ssd_rd_lat", "blk_iat"],
   profiler_field="")


# ===========================================================================
#  F. CROSS-LAYER KPIs
# ===========================================================================

_r("ssd_rd_per_output_tok", "SSD Read Bytes per Output Token",
   aliases=["ssd_rd_per_output_token", "bytes_read_per_token",
             "kv_read_cost_per_token"],
   regex_patterns=[r"ssd.*per.*token", r"read.*per.*output.*tok"],
   category="cross_layer", subcategory="kpi", section="§G88",
   unit="B/tok", higher_is_better=False,
   description="SSD bytes read per generated output token. "
               "= kv_bytes_per_token when decode is fully SSD-bound.",
   formula=(
       "rate(node_disk_read_bytes_total) / rate(sglang_generation_tokens_total)"
   ),
   sources=["derived"],
   related=["kv_bytes_per_token", "ssd_rd_bw"],
   profiler_field="ssd_rd_bytes_per_output_tok")

_r("throughput_bottleneck", "Throughput Bottleneck Source",
   aliases=["bottleneck", "limiting_factor", "throughput_limit"],
   regex_patterns=[r"throughput_bottleneck", r"limiting.*factor"],
   category="cross_layer", subcategory="kpi", section="§G93",
   unit="", higher_is_better=None,
   description="Whether throughput is limited by GPU or SSD bandwidth. "
               "min(GPU tok/s, SSD_BW / kv_bytes_per_token).",
   formula=(
       "ssd_tok_ceiling = ssd_read_bw_Bps / kv_bytes_per_token\n"
       "bottleneck = 'GPU' if gen_throughput < ssd_tok_ceiling else 'SSD'"
   ),
   sources=["derived"],
   related=["gen_throughput", "ssd_rd_bw"],
   profiler_field="throughput_bottleneck_src")


# ===========================================================================
#  G. BENCHMARK METRICS
# ===========================================================================

_r("mlperf_ttft", "MLPerf First Token Latency",
   aliases=["first_token_latency_ms", "MLPerf_TTFT", "mlperf_first_token"],
   regex_patterns=[r"mlperf.*ttft", r"first_token_latency_ms"],
   category="benchmark", subcategory="mlperf",
   unit="ms", higher_is_better=False,
   description="MLPerf Server: 99th pct TTFT must meet target. "
               "Offline: output_token_throughput is primary.",
   formula="loadgen: 99th pct first_token_latency_ms",
   sources=["mlperf", "loadgen"],
   related=["ttft"],
   profiler_field="")

_r("vllm_bench_throughput", "vLLM Benchmark Throughput",
   aliases=["request_throughput", "output_throughput", "input_throughput"],
   regex_patterns=[r"output_throughput.*tok", r"input_throughput.*tok"],
   category="benchmark", subcategory="vllm_bench",
   unit="tok/s", higher_is_better=True,
   description="vLLM benchmark_serving.py output: "
               "request_throughput (req/s), input/output_throughput (tok/s).",
   formula=(
       "output_throughput = total_output_tokens / total_time\n"
       "CSV cols: request_throughput, input_throughput, output_throughput"
   ),
   sources=["vllm-bench"],
   related=["gen_throughput", "ttft"],
   profiler_field="")

_r("vllm_bench_mean_ttft", "vLLM Benchmark Mean TTFT",
   aliases=["mean_ttft_ms", "p99_ttft_ms", "median_ttft_ms",
             "mean_itl_ms", "p99_tpot_ms"],
   regex_patterns=[r"mean_ttft", r"p99_ttft", r"median_ttft",
                   r"mean_itl", r"p99_tpot"],
   category="benchmark", subcategory="vllm_bench",
   unit="ms", higher_is_better=False,
   description="TTFT / ITL statistics from vLLM benchmark output.",
   formula="CSV cols: mean_ttft_ms, median_ttft_ms, p99_ttft_ms",
   sources=["vllm-bench", "llmperf"],
   related=["ttft"],
   profiler_field="")

_r("llmperf_throughput", "LLMPerf Token Throughput",
   aliases=["llmperf_throughput", "mean_output_throughput_token_per_s",
             "output_token_throughput_token_per_s"],
   regex_patterns=[r"mean_output_throughput_token",
                   r"output_token_throughput_token_per_s"],
   category="benchmark", subcategory="llmperf",
   unit="tok/s", higher_is_better=True,
   description="LLMPerf output token throughput.",
   formula="total_output_tokens / total_time",
   sources=["llmperf"],
   related=["gen_throughput"],
   profiler_field="")

_r("accuracy", "Task Accuracy",
   aliases=["acc", "accuracy", "acc_norm", "exact_match", "pass_at_1",
             "mean_win_rate", "average_score", "resolved", "pass_rate"],
   regex_patterns=[r"\bacc\b", r"\bacc_norm\b", r"exact_match",
                   r"pass_at_\d", r"mean_win_rate", r"average_score"],
   category="benchmark", subcategory="accuracy",
   unit="%", higher_is_better=True, normal_range=(0, 100),
   description="Task accuracy metric — varies by benchmark: "
               "acc/acc_norm (LM-Eval), pass@k (code), "
               "resolved (SWEBench), average_score (OpenLLM).",
   formula=(
       "LM-Eval  : acc = correct / total\n"
       "SWEBench : resolved_instances / total_instances\n"
       "OpenLLM  : mean(ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K)"
   ),
   sources=["lm-eval", "swebench", "openllm", "helm", "mmlu", "arc", "gsm8k"],
   related=["ttft", "gen_throughput"],
   profiler_field="")

_r("perplexity", "Perplexity",
   aliases=["perplexity", "ppl", "word_perplexity", "byte_perplexity",
             "bits_per_byte"],
   regex_patterns=[r"\bppl\b", r"perplexity", r"bits_per_byte"],
   category="benchmark", subcategory="accuracy",
   unit="", higher_is_better=False,
   description="Language model perplexity. Lower = better. "
               "PPL = exp(-1/N × Σ log p(token_i)).",
   formula="PPL = exp(-1/N × Σ log p(token_i))",
   sources=["lm-eval", "helm"],
   related=["accuracy"],
   profiler_field="")

_r("swebench_resolved", "SWEBench Resolved Rate",
   aliases=["resolved", "swebench_score", "resolved_instances",
             "unresolved_instances", "total_instances"],
   regex_patterns=[r"swebench.*resolv", r"resolv.*instanc"],
   category="benchmark", subcategory="swebench",
   unit="%", higher_is_better=True,
   description="Fraction of SWEBench instances correctly resolved. "
               "High cache_hit_rate expected (repeated codebase context).",
   formula="resolved_instances / total_instances × 100",
   sources=["swebench", "sweagent"],
   related=["cache_hit_rate", "ttft", "accuracy"],
   profiler_field="")

_r("longbench_score", "LongBench Score",
   aliases=["longbench_score", "long_context_score", "context_length",
             "rouge_l"],
   regex_patterns=[r"longbench", r"long.*context.*score"],
   category="benchmark", subcategory="longbench",
   unit="", higher_is_better=True,
   description="LongBench task score (F1/accuracy/ROUGE-L). "
               "Tests long-context up to 200K tokens.",
   formula="Task-dependent: F1, ROUGE-L, accuracy",
   sources=["longbench"],
   related=["cache_hit_rate", "kv_bytes_per_token"],
   profiler_field="")

_r("helm_tokens_per_s", "HELM Tokens per Second",
   aliases=["helm_tokens_per_s", "HELM_throughput", "inference_runtime",
             "num_prompt_tokens", "num_completion_tokens"],
   regex_patterns=[r"helm.*token", r"helm.*throughput"],
   category="benchmark", subcategory="helm",
   unit="tok/s", higher_is_better=True,
   description="HELM throughput: (prompt + completion tokens) / inference_runtime.",
   formula="(num_prompt_tokens + num_completion_tokens) / inference_runtime",
   sources=["helm", "crfm"],
   related=["gen_throughput"],
   profiler_field="")

_r("sharegpt_avg_len", "Dataset Average Sequence Length",
   aliases=["avg_seq_len", "mean_seq_len", "average_prompt_len",
             "average_output_len", "p90_prompt_len", "p99_output_len",
             "average_input_tokens"],
   regex_patterns=[r"avg.*prompt.*len", r"avg.*output.*len",
                   r"average.*input.*token", r"p9\d.*prompt.*len"],
   category="benchmark", subcategory="dataset",
   unit="tokens", higher_is_better=None,
   description="Mean input/output token length. "
               "Affects TTFT (longer input) and E2E (longer output).",
   formula="CSV: average_prompt_len, average_output_len, p90_prompt_len",
   sources=["sharegpt", "vllm-bench", "llmperf", "alpaca"],
   related=["ttft", "tpot", "cache_hit_rate"],
   profiler_field="")


# ---------------------------------------------------------------------------
#  RESOLUTION API
# ---------------------------------------------------------------------------

def resolve(raw_name: str, threshold: int = 40) -> Optional[MetricDef]:
    """
    Resolve ANY metric name from ANY source to its canonical MetricDef.

    Uses a scored matching strategy:
      100 - exact canonical name
       90 - exact alias
       75 - regex pattern match
       65 - canonical name is substring of raw
       55 - all significant words of canonical name appear in raw
       52 - partial alias substring match

    Args:
        raw_name  : any string — Prometheus metric, CSV column, JSONL key,
                    blktrace CSV header, iostat column, etc.
        threshold : minimum score to accept (default 40)

    Returns:
        MetricDef | None
    """
    best_m, best_s = None, 0
    for m in REGISTRY.values():
        s = m.match_score(raw_name)
        if s > best_s:
            best_s = s
            best_m = m
    return best_m if best_s >= threshold else None


def match_prometheus(metric_name: str) -> Optional[MetricDef]:
    """Resolve a Prometheus metric name. Alias for resolve() with threshold=50."""
    return resolve(metric_name, threshold=50)


def discover(source, threshold: int = 40) -> dict[str, MetricDef]:
    """
    Bulk-map all keys/columns in a data source to MetricDef entries.

    Accepts:
      - dict              {metric_name: value}
      - pandas DataFrame  (uses .columns)
      - list[str]         of metric names

    Returns:
        {raw_name: MetricDef}  for every resolvable metric
    """
    try:
        import pandas as _pd
        if isinstance(source, _pd.DataFrame):
            names = list(source.columns)
        elif isinstance(source, dict):
            names = list(source.keys())
        elif isinstance(source, (list, tuple)):
            names = [str(x) for x in source]
        else:
            names = list(str(source).split())
    except ImportError:
        names = list(source.keys()) if hasattr(source, "keys") else []

    return {name: m for name in names
            if (m := resolve(str(name), threshold)) is not None}


def annotate_df(df, col: str = "metric", threshold: int = 40):
    """
    Add annotation columns to a DataFrame containing metric names.

    New columns added:
      canonical, full_name, category, subcategory, unit, higher_better,
      description (120 chars), formula (150 chars), sources, related,
      profiler_field
    """
    import pandas as _pd
    rows = []
    for name in df[col]:
        m = resolve(str(name), threshold)
        rows.append({
            "canonical":       m.name               if m else "",
            "full_name":       m.full_name          if m else "",
            "category":        m.category           if m else "",
            "subcategory":     m.subcategory        if m else "",
            "unit":            m.unit               if m else "",
            "higher_better":   m.higher_is_better   if m else None,
            "description":     m.description[:120]  if m else "",
            "formula":         m.formula[:150]       if m else "",
            "sources":         "|".join(m.sources)  if m else "",
            "related":         "|".join(m.related)  if m else "",
            "profiler_field":  m.profiler_field      if m else "",
        })
    return _pd.concat([df.reset_index(drop=True),
                       _pd.DataFrame(rows)], axis=1)


def get_category(cat: str) -> list[MetricDef]:
    """Return all metrics in a category."""
    return [m for m in REGISTRY.values() if m.category == cat]


def search(query: str) -> list[MetricDef]:
    """Full-text search across name, aliases, description, sources."""
    q = query.lower()
    scored = []
    for m in REGISTRY.values():
        s = (5 * int(q in m.name)
             + 4 * int(q in m.full_name.lower())
             + 3 * int(any(q in a.lower() for a in m.aliases))
             + 2 * int(q in m.category)
             + 1 * int(q in m.description.lower())
             + 1 * int(any(q in src.lower() for src in m.sources)))
        if s:
            scored.append((s, m))
    return [m for _, m in sorted(scored, reverse=True)]


def lookup(name: str) -> Optional[MetricDef]:
    """Exact canonical name or alias lookup (case-insensitive)."""
    nl = name.lower().replace("-", "_")
    if nl in REGISTRY:
        return REGISTRY[nl]
    for m in REGISTRY.values():
        if any(a.lower().replace("-", "_") == nl for a in m.aliases):
            return m
    return None


def print_registry_summary():
    from collections import Counter
    cats = Counter(m.category for m in REGISTRY.values())
    print(f"\n{'='*64}")
    print(f"  AMOprof Metrics Dictionary v2.0  —  {len(REGISTRY)} definitions")
    print(f"  Servers  : SGLang · vLLM · TGI · TensorRT-LLM · Triton")
    print(f"  Benchmarks: MLPerf · LLMPerf · SWEBench · LongBench · HELM")
    print(f"             LM-Eval · OpenLLM · vLLM-bench · ShareGPT")
    print(f"  System   : DCGM · iostat · blktrace · biosnoop · nvme-cli")
    print(f"{'='*64}")
    for cat, count in sorted(cats.items()):
        names = [m.name for m in REGISTRY.values() if m.category == cat]
        print(f"  {cat:<15}: {count:3d}  {names[:6]}{'…' if len(names) > 6 else ''}")


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_registry_summary()
    print("\n── resolve() examples ──────────────────────────────────────────")
    tests = [
        # SGLang Prometheus
        ("sglang_time_to_first_token_seconds_sum",   "ttft"),
        ("sglang_gen_throughput",                    "gen_throughput"),
        ("sglang_cache_hit_rate",                    "cache_hit_rate"),
        ("sglang_num_running_reqs",                  "num_running_reqs"),
        ("sglang_full_token_usage",                  "kv_pool_fill"),
        ("sglang_backuped_tokens_total",             "kvb_offload_rate"),
        ("sglang_prefetched_tokens_total",           "kvb_onboard_rate"),
        ("sglang_evicted_tokens_total",              "kv_evict_rate"),
        # vLLM Prometheus
        ("sglang_num_running_reqs",                "num_running_reqs"),
        ("sglang_kv_used_tokens / configured KV token capacity",                "kv_pool_fill"),
        ("sglang_time_to_first_token_seconds",         "ttft"),
        ("sglang_num_queue_reqs",                "num_queue_reqs"),
        ("vllm:prefix_cache_hit_rate",               "cache_hit_rate"),
        ("sglang_num_preemptions_total (if exported)",               "num_preemptions"),
        ("SGLang speculative decode acceptance metric, when exported",   "spec_decode_acceptance"),
        # TGI
        ("tgi_request_duration_seconds",             "e2e_latency"),
        ("tgi_request_prefill_duration_seconds",     "ttft"),
        ("tgi_batch_current_size",                   "num_running_reqs"),
        # TRT-LLM
        ("nv_trt_llm_request_metrics_first_token_ms","ttft"),
        ("nv_trt_llm_request_metrics_e2e_latency_ms","e2e_latency"),
        ("nv_trt_llm_inflight_batcher_requests_active","num_running_reqs"),
        ("nv_trt_llm_kv_cache_block_manager_used_num_blocks","kv_pool_fill"),
        # DCGM / GPU
        ("DCGM_FI_PROF_DRAM_ACTIVE",                "hbm_bw"),
        ("DCGM_FI_DEV_GPU_UTIL",                    "gpu_util"),
        ("DCGM_FI_DEV_FB_USED",                     "hbm_used"),
        ("DCGM_FI_DEV_POWER_USAGE",                 "gpu_power"),
        # node_exporter / system
        ("node_disk_written_bytes_total",            "ssd_wr_bw"),
        ("node_disk_read_bytes_total",               "ssd_rd_bw"),
        ("node_disk_writes_completed_total",         "ssd_wr_iops"),
        ("node_disk_io_time_seconds_total",          "ssd_io_util"),
        ("node_disk_discards_completed_total",       "trim_iops"),
        ("ipmi_power_watts",                         "gpu_power"),
        # iostat columns
        ("r_await",                                  "ssd_rd_lat"),
        ("w_await",                                  "ssd_wr_lat"),
        ("percent_used",                             "smart_wear"),
        ("avail_spare",                              "smart_spare"),
        # blktrace analysis CSVs
        ("degradation_ratio_late_over_early",        "blk_bw_degradation"),
        ("bandwidth_mib_s",                          "blk_per_stream_bw"),
        ("gini_bytes",                               "blk_hot_cold"),
        ("skewness_bytes",                           "blk_hot_cold"),
        ("inter_arrival_sec",                        "blk_iat"),
        ("not_kv_block_aligned_count",               "blk_alignment"),
        # vllm-bench / llmperf CSV columns
        ("mean_ttft_ms",                             "vllm_bench_mean_ttft"),
        ("output_throughput",                        "vllm_bench_throughput"),
        ("mean_output_throughput_token_per_s",       "llmperf_throughput"),
        ("average_prompt_len",                       "sharegpt_avg_len"),
        # SWEBench
        ("resolved",                                 "swebench_resolved"),
    ]

    ok, fail = 0, 0
    for raw, expected in tests:
        m = resolve(raw)
        got = m.name if m else "NO_MATCH"
        status = "✅" if got == expected else f"❌ got={got}"
        if got == expected:
            ok += 1
        else:
            fail += 1
        print(f"  {status}  {raw[:50]:<52} → {got}")

    print(f"\n  {ok}/{ok+fail} passed")
