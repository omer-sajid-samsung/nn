"""
bench_swebench.py — SWE-bench workload runner for AMOprof.

Improvements in this version
─────────────────────────────
1.  TTFT and TPOT measured from SSE stream timestamps.
2.  SGLang /metrics polled per instance for prefill/decode tok/s,
    KV$ hit rate, and AI op type classification.
3.  Per-instance HBM and DRAM snapshots (nvidia-smi + /proc/meminfo).
4.  Complete file content in prompt — fetches from GitHub raw API at
    base_commit so the model can actually read the code it must fix.
5.  Patch truncation fix — max_tokens=8192 default, stream collects
    until [DONE] with wall-clock deadline only as safety net.
6.  Resolved parsing reads JSON report from run_evaluation, not stdout grep.
"""

from __future__ import annotations

import http.client
import threading
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

log = logging.getLogger("amoprof.swebench")

DATASET_IDS = {
    "lite":     "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full":     "princeton-nlp/SWE-bench",
    # SWE-bench Pro (Scale AI, 2025) — 731 public instances, GPL repos,
    # contamination-resistant. Docker images on Docker Hub: jefzda/sweap-images.
    # Requires separate setup — see SWEBenchHarness._run_docker_eval_pro()
    "pro":      "ScaleAI/SWE-bench_Pro",
}

# Splits that use SWE-bench Pro's Docker Hub images instead of epoch-research ghcr.io
PRO_SPLITS = {"pro"}

# ── Docker image pull cache — skip re-pulling already-local images ────────────
_PULLED_IMAGES: set = set()
_PULLED_IMAGES_LOCK = threading.Lock()

def _ensure_image(image: str, timeout: int = 300) -> bool:
    """
    Pull Docker image only if not already present locally.

    Avoids N concurrent `docker pull` calls for the same image when
    inference_concurrency=32 — Docker Hub rate-limits concurrent registry
    queries even for cached images.  A single pull per unique image tag
    per process is sufficient.

    Returns True if image is available (already local or pull succeeded).
    """
    with _PULLED_IMAGES_LOCK:
        if image in _PULLED_IMAGES:
            return True           # already pulled this session

    # Check if image exists locally first (fast — no network)
    try:
        check = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=5)
        if check.returncode == 0:
            with _PULLED_IMAGES_LOCK:
                _PULLED_IMAGES.add(image)
            return True
    except Exception:
        pass

    # Not local — pull once (serialised via lock to avoid concurrent pulls)
    with _PULLED_IMAGES_LOCK:
        if image in _PULLED_IMAGES:    # re-check: another thread may have pulled
            return True
        try:
            log.info(f"docker pull {image}")
            pull = subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=timeout)
            if pull.returncode == 0:
                _PULLED_IMAGES.add(image)
                return True
            log.warning(f"docker pull failed for {image}: {pull.stderr[:200]}")
            return False
        except Exception as e:
            log.warning(f"docker pull error for {image}: {e}")
            return False


# ── swebench harness capability probe (cached, thread-safe) ───────────────────
_HARNESS_FLAGS: dict | None = None
_HARNESS_FLAGS_LOCK = threading.Lock()

def _harness_supports(flag: str) -> bool:
    """
    Return True if the installed swebench.harness.run_evaluation accepts `flag`.

    Thread-safe: uses a lock so only ONE subprocess is spawned even when called
    concurrently from a ThreadPoolExecutor with inference_concurrency > 1.
    Without the lock, N concurrent threads all see _HARNESS_FLAGS is None and
    each spawns a separate `python -m swebench.harness.run_evaluation --help`
    subprocess — causing N×15s of wasted startup overhead per concurrency sweep level.
    """
    global _HARNESS_FLAGS
    if _HARNESS_FLAGS is not None:          # fast path — no lock needed after init
        return _HARNESS_FLAGS.get(flag, False)
    with _HARNESS_FLAGS_LOCK:
        if _HARNESS_FLAGS is not None:      # re-check inside lock
            return _HARNESS_FLAGS.get(flag, False)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "swebench.harness.run_evaluation", "--help"],
                capture_output=True, text=True, timeout=15)
            help_text = r.stdout + r.stderr
            _HARNESS_FLAGS = {
                "--output_dir":         "--output_dir"         in help_text,
                "--report_path":        "--report_path"        in help_text,
                "--instance_image_tag": "--instance_image_tag" in help_text,
            }
            log.info(f"swebench harness flags: {_HARNESS_FLAGS}")
        except Exception as e:
            log.warning(f"Could not probe swebench harness flags: {e}")
            _HARNESS_FLAGS = {"--output_dir": True, "--report_path": False,
                               "--instance_image_tag": False}
    return _HARNESS_FLAGS.get(flag, False)


PROMPT_WITH_CODE = """\
You are an expert software engineer fixing a real GitHub issue.

<issue>
{problem_statement}
</issue>

<source_files>
{file_contents}
</source_files>

<failing_tests>
{test_contents}
</failing_tests>

<instructions>
The failing tests above show exactly what behavior needs to change.
Generate a complete unified git diff patch that makes those tests pass.
The patch will be applied with: patch -p1 < fix.patch

Rules:
1. Start immediately with "diff --git a/path b/path" — no preamble, no explanation
2. Use exact line numbers from the source files shown above
3. Include 3 unchanged context lines before and after each changed block
4. Make the minimal change needed to fix the issue — do not refactor unrelated code
5. If multiple files need changes, include all of them in the same patch
6. End the patch cleanly — do not truncate mid-hunk
7. No markdown code fences, no trailing explanation
</instructions>
"""

PROMPT_FILES_ONLY = """\
You are an expert software engineer fixing a real GitHub issue.

<issue>
{problem_statement}
</issue>

<relevant_files>
{file_context}
</relevant_files>

<failing_tests>
{test_contents}
</failing_tests>

Generate a complete unified git diff patch that resolves the issue and makes
the failing tests pass. Start immediately with "diff --git a/..." — no preamble,
no markdown fences. Include 3 context lines per hunk. Do NOT truncate.
"""

REASONING_SYSTEM = (
    "You are an expert software engineer. "
    "Carefully read the issue and the provided source files. "
    "Think through the root cause step by step in your reasoning. "
    "Then output a complete, correct unified git diff patch. "
    "Requirements:\n"
    "- Start immediately with 'diff --git a/...' — no preamble\n"
    "- Include EVERY changed file in the diff\n"
    "- Include 3 lines of unchanged context before and after each hunk\n"
    "- The patch must pass: git apply --check\n"
    "- Do NOT truncate — output the full patch even if it is long\n"
    "- No markdown fences, no explanation after the patch"
)


@dataclass
class SWEBenchResult:
    run_id:          str   = ""
    timestamp:       str   = ""
    model:           str   = ""
    split:           str   = "lite"
    instance_id:     str   = ""
    repo:            str   = ""
    instance_index:  int   = 0

    prompt_tokens:   int   = 0
    output_tokens:   int   = 0
    prompt_has_code: bool  = False
    num_turns:       int   = 1

    # ── SGLang /metrics — AI operation ──────────────────────────────────────
    ai_op_type:            str   = ""
    ai_op_prefill_tok_s:   float = 0.0
    ai_op_decode_tok_s:    float = 0.0
    kv_cache_hit_rate_pct: float = 0.0   # from sglang:cache_hit_rate
    num_running_req_mean:  float = 0.0

    # ── SGLang /metrics — KV$ pool state ──────────────────────────────────
    token_usage_peak:          float = 0.0   # sglang:token_usage peak (%)
    kv_pool_capacity_tokens:   int   = 0     # sglang:max_total_num_tokens
    kv_used_tokens_peak:       int   = 0     # sglang:num_used_tokens peak

    # ── SGLang /metrics — KV$ tier occupancy (L1/L2/L3) ──────────────────
    # Maps directly to the three memory tiers in the stack-cake diagram
    kv_l1_device_tokens:    int   = 0   # tokens in GPU HBM (L1)
    kv_l2_host_tokens:      int   = 0   # tokens in CPU DRAM staging (L2)
    kv_l3_storage_tokens:   int   = 0   # tokens on SSD cold store (L3)
    hicache_host_used_tokens:int  = 0   # sglang:hicache_host_used_tokens
    hicache_host_total_tokens:int = 0   # sglang:hicache_host_total_tokens
    hicache_host_fill_pct:  float = 0.0 # L2 staging fill %

    # ── SGLang /metrics — KV$ movement (token-level eviction/restore) ────
    kv_evicted_tokens:      int   = 0   # tokens evicted GPU→CPU this instance
    kv_restored_tokens:     int   = 0   # tokens restored CPU→GPU this instance
    kv_prefetched_tokens:   int   = 0   # tokens prefetched

    # ── SGLang /metrics — HiCache operation latency ───────────────────────
    hicache_eviction_ms:    float = 0.0  # mean time per HBM→CPU eviction (ms)
    hicache_load_back_ms:   float = 0.0  # mean time per CPU→HBM restore (ms)
    hicache_queue_time_ms:  float = 0.0  # mean time spent in request queue (ms)

    # ── SGLang /metrics — AI op breakdown (ground truth) ─────────────────
    rt_prefill_compute_tokens: int   = 0    # tokens needing real prefill (KV$ writes)
    rt_prefill_cache_tokens:   int   = 0    # tokens served from cache (KV$ hits)
    rt_decode_tokens:          int   = 0    # decode tokens (KV$ reads per step)
    cache_hit_rate_realtime_pct: float = 0.0  # realtime cache hits / prefill total
    new_token_ratio_mean:      float = 0.0  # fraction of tokens not cached

    # ── SGLang /metrics — Server-side latency (ground truth histograms) ───
    server_ttft_ms:         float = 0.0   # sglang:time_to_first_token_seconds mean
    server_itl_ms:          float = 0.0   # sglang:inter_token_latency_seconds mean
    server_e2e_ms:          float = 0.0   # sglang:e2e_request_latency_seconds mean

    # ── SGLang /metrics — Queue and throughput ────────────────────────────
    num_queue_reqs_peak:    int   = 0     # sglang:num_queue_reqs peak
    decode_sum_seq_lens:    int   = 0     # total context length in decode (KV$ demand)
    utilization_mean:       float = 0.0  # sglang:utilization mean (%)

    # Inference latency (measured from SSE stream)
    ttft_ms:       float = 0.0
    tpot_mean_ms:  float = 0.0
    tpot_p99_ms:   float = 0.0
    total_time_s:  float = 0.0
    tok_per_s:     float = 0.0

    # ── HBM (GPU memory) per instance ──────────────────────────────────────
    # nvidia-smi snapshots before/after the full instance (all turns)
    hbm_used_gb_start: float = 0.0   # before turn 1
    hbm_used_gb_end:   float = 0.0   # after last turn
    hbm_delta_gb:      float = 0.0   # net KV$ growth across all turns
    hbm_util_pct:      float = 0.0   # mean across turns
    hbm_peak_gb:       float = 0.0   # peak HBM observed (max across turns)

    # Per-AI-op HBM breakdown (turn-level means)
    # Prefill writes the KV$ for the new tokens; decode reads all of it per step
    hbm_prefill_delta_gb:  float = 0.0  # HBM increase during prefill phase (KV$ written)
    hbm_decode_delta_gb:   float = 0.0  # HBM change during decode (should be ~0: KV$ already allocated)
    hbm_kv_pool_fill_pct:  float = 0.0  # KV$ pool fill % (from token_usage metric)
    hbm_kv_evicted_gb:     float = 0.0  # estimated GB evicted to SSD this instance

    # ── DRAM (host memory) per instance ──────────────────────────────────
    dram_used_gb_start: float = 0.0
    dram_used_gb_end:   float = 0.0
    dram_delta_gb:      float = 0.0   # DRAM growth from HiCache staging buffers
    dram_util_pct:      float = 0.0
    dram_hicache_staging_gb: float = 0.0  # estimated HiCache DRAM staging (dram_delta when SSD active)

    # ── Per-turn memory timeline (serialized as JSON string) ─────────────
    # Each element: {turn, prompt_tok, output_tok, ttft_ms, tpot_ms,
    #                hbm_start_gb, hbm_end_gb, hbm_delta_gb,
    #                dram_delta_gb, kv_hit_pct, ssd_read_mb, ssd_write_mb,
    #                ai_op, resolved}
    turn_timeline: str = ""   # JSON — empty string when num_turns=1

    resolved:           bool  = False
    resolved_in_loop:   bool  = False   # agent called finish() before max_steps
    patch_generated:    bool  = False
    patch_applied:      bool  = False
    patch_complete:     bool  = False
    tests_passed:       int   = 0
    tests_failed:       int   = 0
    agent_steps_taken:  int   = 0       # tool calls consumed; 0 in non-agent mode

    # ── A3 OS/Block Layer (L3=SSD) — iostat + biolatency ────────────────────
    read_bw_mb_mean:    float = 0.0
    write_bw_mb_mean:   float = 0.0
    read_iops_mean:     float = 0.0
    write_iops_mean:    float = 0.0
    r_await_ms_p99:     float = 0.0
    r_await_ms_p999:    float = 0.0
    w_await_ms_mean:    float = 0.0
    avgqu_sz_mean:      float = 0.0   # A2 io_uring queue depth
    util_pct_mean:      float = 0.0
    # biolatency (eBPF histogram)
    bio_lat_p50_us:     float = 0.0
    bio_lat_p99_us:     float = 0.0
    bio_lat_p999_us:    float = 0.0

    # ── A1 SSD Hardware — SMART extended ─────────────────────────────────────
    waf:                float = 0.0
    host_written_gb:    float = 0.0
    nand_written_gb:    float = 0.0
    ssd_lifetime_tbw:   float = 0.0   # cumulative TBW (lifetime)
    ssd_dwpd_est:       float = 0.0   # estimated DWPD this run
    temp_peak_c:        int   = 0
    ssd_media_errors:   int   = 0
    # HiCache KV$ cold store on SSD
    hicache_size_gb:    float = 0.0
    hicache_file_count: int   = 0

    # ── A2 io_uring / NVMe Driver (sysfs) ────────────────────────────────────
    nvme_inflight_mean: float = 0.0
    nvme_inflight_peak: int   = 0
    nvme_rd_lat_ms_sysfs: float = 0.0
    nvme_wr_lat_ms_sysfs: float = 0.0
    nvme_nr_requests:   int   = 0
    nvme_scheduler:     str   = ""

    # ── A3 OS/Memory Manager (L2=DRAM) — vmstat ──────────────────────────────
    page_faults_per_s:      float = 0.0
    major_faults_per_s:     float = 0.0
    swap_pages_total:       int   = 0
    page_cache_reads_per_s: float = 0.0
    numa_migrations_per_s:  float = 0.0
    tlb_remote_miss_per_s:  float = 0.0
    hugepages_used:         int   = 0

    # ── A2 GPU Driver — NVLink + PCIe ─────────────────────────────────────────
    nvlink_tx_gb_s:         float = 0.0
    nvlink_rx_gb_s:         float = 0.0
    pcie_tx_gb_s:           float = 0.0
    pcie_rx_gb_s:           float = 0.0
    pcie_link_gen:          int   = 0
    pcie_theoretical_gbps:  float = 0.0
    # CUDA kernel-level
    cuda_sm_active_mean_pct:     float = 0.0
    cuda_sm_active_min_pct:      float = 0.0
    cuda_tensor_active_mean_pct: float = 0.0
    cuda_tensor_active_min_pct:  float = 0.0
    cuda_dram_active_mean_pct:   float = 0.0
    cuda_hbm_bw_read_gb_s:       float = 0.0
    cuda_hbm_bw_write_gb_s:      float = 0.0
    cuda_sm_clock_mhz:           float = 0.0
    cuda_sm_occupancy_mean_pct:  float = 0.0
    cuda_fp16_active_mean_pct:   float = 0.0
    cuda_throttled_pct:          float = 0.0
    cuda_source:                 str   = ""

    # ── A5 Application Layer — request latency distribution ───────────────────
    req_lat_p99_ms:         float = 0.0
    req_lat_p999_ms:        float = 0.0

    # ── Deep profiler metrics (optional — zero when profiler not available) ────

    # ncu: attention kernel roofline
    ncu_attention_available:    bool  = False
    ncu_attention_kernel_count: int   = 0
    ncu_dram_read_gb:           float = 0.0   # GB read by attention kernels
    ncu_dram_write_gb:          float = 0.0
    ncu_l2_hit_rate_pct:        float = 0.0
    ncu_arith_intensity:        float = 0.0   # FLOP/byte roofline
    ncu_sm_eff_pct:             float = 0.0
    ncu_duration_us_mean:       float = 0.0
    ncu_duration_us_p99:        float = 0.0

    # nsys: CUDA timeline
    nsys_available:             bool  = False
    nsys_cuda_api_calls:        int   = 0
    nsys_kernel_count:          int   = 0
    nsys_memcpy_h2d_gb:         float = 0.0
    nsys_memcpy_d2h_gb:         float = 0.0
    nsys_gpu_active_pct:        float = 0.0
    nsys_kernel_top5:           str   = ""    # JSON

    # perf stat: L3 miss rate
    perf_available:             bool  = False
    perf_l3_miss_count:         int   = 0
    perf_all_loads_count:       int   = 0
    perf_l3_miss_rate_pct:      float = 0.0   # key DRAM pressure signal
    perf_l3_miss_per_s:         float = 0.0

    # pcm-memory: socket-level DRAM bandwidth
    pcm_available:              bool  = False
    pcm_dram_read_gb_s:         float = 0.0
    pcm_dram_write_gb_s:        float = 0.0
    pcm_dram_total_gb_s:        float = 0.0
    pcm_dram_read_gb_s_peak:    float = 0.0
    pcm_source:                 str   = ""    # "pcm-memory"|"imc_pmu"|"unavailable"

    # bpftrace: page faults + mmap/malloc
    bpf_available:              bool  = False
    bpf_page_faults_total:      int   = 0
    bpf_major_faults_total:     int   = 0
    bpf_mmap_calls:             int   = 0
    bpf_mmap_bytes_gb:          float = 0.0
    bpf_malloc_calls:           int   = 0
    bpf_malloc_bytes_gb:        float = 0.0
    bpf_read_lat_p99_us:        float = 0.0
    bpf_write_lat_p99_us:       float = 0.0

    # PyTorch Profiler: operator-level CUDA time
    torch_prof_available:       bool  = False
    torch_prof_cuda_time_ms:    float = 0.0
    torch_prof_cpu_time_ms:     float = 0.0
    torch_prof_memory_alloc_mb: float = 0.0
    torch_prof_kernel_count:    int   = 0
    torch_prof_top_ops:         str   = ""    # JSON

    # VTune: DRAM BW + NUMA locality
    vtune_available:            bool  = False
    vtune_dram_bw_gb_s:         float = 0.0
    vtune_numa_local_access_pct:float = 0.0
    vtune_l3_bound_pct:         float = 0.0
    vtune_mem_bound_pct:        float = 0.0
    vtune_ipc:                  float = 0.0
    vtune_hotspot_fn:           str   = ""

    raw_sglang_metrics_json:    str   = ""
    raw_iostat_metrics_json:    str   = ""
    raw_biolat_metrics_json:    str   = ""
    raw_vmstat_metrics_json:    str   = ""
    raw_nvlink_metrics_json:    str   = ""
    raw_nvme_driver_metrics_json:str  = ""
    raw_cuda_metrics_json:      str   = ""
    raw_perf_metrics_json:      str   = ""
    raw_pcm_metrics_json:       str   = ""
    raw_bpf_metrics_json:       str   = ""

    success: bool  = True
    notes:   str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── SGLang /metrics polling ────────────────────────────────────────────────────

def _fetch_metrics_once(port: int, debug: bool = False,
                        debug_path: str | None = None,
                        host: str = "127.0.0.1") -> dict:
    """
    Fetch one Prometheus snapshot from SGLang /metrics.

    Correctly parses:
      - Plain gauges:      sglang:gen_throughput 3.05
      - Labeled metrics:   sglang:cached_tokens_total{source="device"} 12345
      - Histogram sums:    sglang:eviction_duration_seconds_sum 0.042
      - Histogram counts:  sglang:eviction_duration_seconds_count 7

    Returns a flat dict where labeled metrics get key metric_name[label=value].
    Counter/histogram pairs are preserved for delta computation in the sampler.
    When debug is enabled, scrape diagnostics are added to the result and also
    appended to debug_path when provided.
    """
    result = {}
    url = f"http://{host}:{port}/metrics"
    scrape_bytes = 0
    metric_lines = 0
    matched_lines = 0
    ttft_lines = 0
    debug_lines: list[str] = []

    def _dbg(msg: str) -> None:
        if debug:
            debug_lines.append(msg)

    _dbg(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] fetch url={url}")
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            raw = r.read().decode("utf-8", errors="replace")
            scrape_bytes = len(raw.encode("utf-8", errors="replace"))
            lines = raw.splitlines()
            _dbg(f"scrape_success=1 scrape_bytes={scrape_bytes} total_lines={len(lines)}")
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                metric_lines += 1
                if "time_to_first_token" in line:
                    ttft_lines += 1
                    if ttft_lines <= 5:
                        _dbg(f"ttft_line={line}")
                # Parse: metric_name{k="v",...} value OR metric_name value.
                # Accept both SGLang styles: sglang:foo and sglang_foo.
                m = re.match(
                    r'^\s*((?:sglang[:_])\w+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)', line)
                if not m:
                    continue
                matched_lines += 1
                try:
                    name = m.group(1)
                    # Normalize to the colon style used by the rest of this module.
                    if name.startswith("sglang_"):
                        name = "sglang:" + name[len("sglang_"):]
                    labels = m.group(2) or ""
                    value = float(m.group(3))
                except ValueError:
                    continue

                _PRIORITY_LABELS = {"mode", "source", "type", "status",
                                    "backend", "direction", "phase"}
                if labels:
                    label_pairs = re.findall(r'(\w+)="([^"]+)"', labels)
                    label_dict = dict(label_pairs)
                    keyed = False
                    for _pl in _PRIORITY_LABELS:
                        if _pl in label_dict:
                            result[f'{name}[{_pl}={label_dict[_pl]}]'] = value
                            keyed = True
                            break
                    if not keyed and label_pairs:
                        result[f'{name}[{label_pairs[0][0]}={label_pairs[0][1]}]'] = value
                    result.setdefault(name, value)
                else:
                    result[name] = value

    except Exception as e:
        result["sglang:scrape_success"] = 0.0
        result["sglang:scrape_error"] = str(e)
        _dbg(f"scrape_success=0 error={e!r}")
    else:
        result["sglang:scrape_success"] = 1.0

    result["sglang:scrape_bytes"] = float(scrape_bytes)
    result["sglang:scrape_metric_lines"] = float(metric_lines)
    result["sglang:scrape_matched_lines"] = float(matched_lines)
    result["sglang:scrape_ttft_lines"] = float(ttft_lines)
    if debug:
        _dbg(f"parsed metric_lines={metric_lines} matched_lines={matched_lines} ttft_lines={ttft_lines}")
        for key in [
            "sglang:time_to_first_token_seconds_sum",
            "sglang:time_to_first_token_seconds_count",
            "sglang:e2e_request_latency_seconds_sum",
            "sglang:e2e_request_latency_seconds_count",
            "sglang:realtime_tokens_total[mode=decode]",
            "sglang:realtime_tokens_total[mode=prefill_compute]",
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
        ]:
            if key in result:
                _dbg(f"parsed {key}={result[key]}")

    # Supplement with /get_server_info
    try:
        with urllib.request.urlopen(
                f"http://{host}:{port}/get_server_info", timeout=2) as r:
            info = json.loads(r.read())
            for src_key, dst_key in [
                ("cache_hit_rate", "sglang:cache_hit_rate"),
                ("kv_cache_usage", "sglang:token_usage"),
            ]:
                if src_key in info:
                    result[dst_key] = float(info[src_key])
            if debug:
                _dbg("get_server_info_success=1")
    except Exception as e:
        if debug:
            _dbg(f"get_server_info_success=0 error={e!r}")

    if debug and debug_path:
        try:
            with open(debug_path, "a", encoding="utf-8") as f:
                for msg in debug_lines:
                    f.write(msg + "\n")
                f.write("---\n")
        except Exception:
            pass

    return result


def _histogram_mean(samples: list[dict], prefix: str) -> float:
    """
    Compute mean from a Prometheus histogram _sum / _count pair.
    Returns 0.0 if data unavailable.
    """
    sum_key   = f"{prefix}_sum"
    count_key = f"{prefix}_count"
    first = samples[0]  if samples else {}
    last  = samples[-1] if samples else {}
    d_sum   = max(last.get(sum_key,   0.0) - first.get(sum_key,   0.0), 0.0)
    d_count = max(last.get(count_key, 0.0) - first.get(count_key, 0.0), 0.0)
    return round(d_sum / d_count, 6) if d_count > 0 else 0.0


class SGLangMetricsSampler:
    """
    Polls SGLang /metrics every `interval_s` while a request is in flight.

    Collects ALL meaningful metrics from the /metrics endpoint as seen in
    a real SGLang 0.5.9 deployment, including:

    KV$ tier metrics (direct from SGLang):
      evicted_tokens_total    — tokens evicted GPU→CPU (L1→L2, cumulative counter)
      load_back_tokens_total  — tokens restored CPU→GPU (L2→L1, cumulative counter)
      cached_tokens[source=device]  — tokens in GPU KV cache (L1 occupancy)
      cached_tokens[source=host]    — tokens in CPU staging (L2 occupancy)
      cached_tokens[source=storage] — tokens on SSD (L3 occupancy)
      hicache_host_used_tokens      — host KV cache used tokens
      hicache_host_total_tokens     — host KV cache capacity

    HiCache latency:
      eviction_duration_seconds — mean time per HBM→CPU eviction event
      load_back_duration_seconds— mean time per CPU→HBM restore event

    AI op classification (using realtime_tokens_total{mode=...}):
      prefill_compute tokens  — tokens that needed actual prefill (new KV$ write)
      prefill_cache tokens    — tokens served from RadixAttention cache (KV$ hit)
      decode tokens           — decode phase tokens (KV$ read-all per step)

    Server-side latency histograms (ground truth, vs our stream measurements):
      time_to_first_token_seconds  — server-measured TTFT
      inter_token_latency_seconds  — server-measured ITL/TPOT
      e2e_request_latency_seconds  — server-measured E2E
      queue_time_seconds           — time spent queued before prefill

    Queue and throughput:
      num_running_reqs     — active requests (gauge)
      num_queue_reqs       — queued requests (gauge)
      decode_sum_seq_lens  — total decode context length (KV$ demand proxy)
      gen_throughput       — decode tokens/s (gauge)
      new_token_ratio      — fraction of tokens not served from cache
      utilization          — overall server utilisation

    Pool capacity:
      max_total_num_tokens — KV pool capacity in tokens
      num_used_tokens      — tokens currently in KV pool
      token_usage          — pool fill ratio (0..1)
    """

    def __init__(self, port: int, interval_s: float = 1.0,
                 host: str = "127.0.0.1",
                 debug: bool = False, debug_path: str | None = None):
        self.port     = port
        self.host     = host
        self.interval = interval_s
        self.debug    = bool(debug)
        self.debug_path = debug_path
        self._samples: list[dict] = []
        self._stop    = threading.Event()
        self._thread  = None
        self._t_start = 0.0
        self._t_end   = 0.0

    @property
    def prometheus_url(self) -> str:
        return f"http://{self.host}:{self.port}/metrics"

    @property
    def raw_samples(self) -> list[dict]:
        """All collected Prometheus scrape snapshots, in chronological order.
        Each snapshot is {metric_key: float, 'ts': epoch_seconds}.
        Used by amoprof.writer to emit sglang_timeseries.csv."""
        return list(self._samples)

    @property
    def elapsed_s(self) -> float:
        if self._t_end > 0 and self._t_start > 0:
            return self._t_end - self._t_start
        return 0.0

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._t_start = time.time()
        s = _fetch_metrics_once(self.port, self.debug, self.debug_path, host=self.host)
        if s:
            s = dict(s)
            s["ts"] = time.time()
            self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = _fetch_metrics_once(self.port, self.debug, self.debug_path, host=self.host)
            if s:
                s = dict(s)
                s["ts"] = time.time()
                self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        s = _fetch_metrics_once(self.port, self.debug, self.debug_path, host=self.host)
        if s:
            s = dict(s)
            s["ts"] = time.time()
            self._samples.append(s)
        self._t_end = time.time()
        if self._thread:
            self._thread.join(timeout=3)

        if not self._samples:
            return self._empty()

        first  = self._samples[0]
        last   = self._samples[-1]
        n      = len(self._samples)

        def peak(k):
            return max((s.get(k, 0.0) for s in self._samples), default=0.0)
        def mn(k):
            vs = [s.get(k, 0.0) for s in self._samples]
            return round(sum(vs) / max(len(vs), 1), 4)
        def delta(k):
            return max(last.get(k, 0.0) - first.get(k, 0.0), 0.0)

        # ── KV$ tier occupancy ────────────────────────────────────────────────
        # Gauges — current state
        kv_pool_capacity_tokens  = peak("sglang:max_total_num_tokens")
        kv_used_tokens_peak      = peak("sglang:num_used_tokens")
        kv_token_usage_peak      = peak("sglang:token_usage")
        hicache_host_used        = peak("sglang:hicache_host_used_tokens")
        hicache_host_total       = peak("sglang:hicache_host_total_tokens")

        # Labeled cache tier occupancy (tokens per tier)
        kv_l1_device_tokens = peak("sglang:cached_tokens_total[source=device]")
        kv_l2_host_tokens   = peak("sglang:cached_tokens_total[source=host]")
        kv_l3_storage_tokens= peak("sglang:cached_tokens_total[source=storage]")

        # ── KV$ eviction / restore (counters — use deltas) ───────────────────
        evicted_delta   = delta("sglang:evicted_tokens_total")
        load_back_delta = delta("sglang:load_back_tokens_total")
        prefetched_delta= delta("sglang:prefetched_tokens_total")

        # ── HiCache latency histograms ────────────────────────────────────────
        eviction_mean_s  = _histogram_mean(self._samples,
                                           "sglang:eviction_duration_seconds")
        load_back_mean_s = _histogram_mean(self._samples,
                                           "sglang:load_back_duration_seconds")
        queue_time_mean_s= _histogram_mean(self._samples,
                                           "sglang:queue_time_seconds")

        # ── Server-side latency histograms ────────────────────────────────────
        ttft_mean_s   = _histogram_mean(self._samples,
                                        "sglang:time_to_first_token_seconds")
        itl_mean_s    = _histogram_mean(self._samples,
                                        "sglang:inter_token_latency_seconds")
        e2e_mean_s    = _histogram_mean(self._samples,
                                        "sglang:e2e_request_latency_seconds")

        # ── Realtime tokens by mode (counters) ───────────────────────────────
        # These are the ground-truth AI operation breakdown
        rt_prefill_compute = delta("sglang:realtime_tokens_total[mode=prefill_compute]")
        rt_prefill_cache   = delta("sglang:realtime_tokens_total[mode=prefill_cache]")
        rt_decode          = delta("sglang:realtime_tokens_total[mode=decode]")
        rt_total = rt_prefill_compute + rt_prefill_cache + rt_decode

        # Cache hit rate from realtime_tokens: cache_hit / (compute + cache)
        prefill_total = rt_prefill_compute + rt_prefill_cache
        cache_hit_rate_realtime = (
            round(rt_prefill_cache / prefill_total * 100, 1)
            if prefill_total > 0 else 0.0)

        # ── AI op classification (improved) ──────────────────────────────────
        # Use realtime_tokens_total as ground truth when available;
        # fall back to gen_throughput gauge for live classification.
        dc_peak  = peak("sglang:gen_throughput")
        n_run_pk = peak("sglang:num_running_reqs")
        n_run_mn = mn("sglang:num_running_reqs")
        new_tok_ratio = mn("sglang:new_token_ratio")   # 1.0 = no cache, 0.0 = all cached

        if rt_total > 0:
            # Ground-truth classification from token counters
            if rt_prefill_compute > rt_decode * 2:
                op = "prefill"
            elif rt_decode > 0 and dc_peak < 15:
                op = "reasoning"       # R1/QwQ: long decode at low tok/s
            elif rt_decode > rt_prefill_compute:
                op = "decode"
            else:
                op = "mixed"
        elif n_run_pk == 0:
            op = "idle"
        elif dc_peak > 0:
            op = "reasoning" if dc_peak < 15 else "decode"
        else:
            op = "prefill" if n_run_pk > 0 else "idle"

        # ── Throughput and queue ──────────────────────────────────────────────
        decode_sum_seq_lens = peak("sglang:decode_sum_seq_lens")
        n_queue_peak        = peak("sglang:num_queue_reqs")
        utilization         = mn("sglang:utilization")
        kv_cache_hit_rate   = last.get("sglang:cache_hit_rate",
                              last.get("sglang:cache_hit_rate[cache=0]", 0.0))

        return {
            # ── Current 5 fields (unchanged interface) ────────────────────────
            "ai_op_type":              op,
            "ai_op_prefill_tok_s":     round(rt_prefill_compute / max(n, 1), 2),
            "ai_op_decode_tok_s":      round(dc_peak, 2),
            "kv_cache_hit_rate_pct":   round(kv_cache_hit_rate * 100, 1),
            "num_running_req_mean":    round(n_run_mn, 2),

            # ── NEW: KV$ pool state ───────────────────────────────────────────
            "token_usage_peak":        round(kv_token_usage_peak * 100, 1),
            "kv_pool_capacity_tokens": int(kv_pool_capacity_tokens),
            "kv_used_tokens_peak":     int(kv_used_tokens_peak),

            # ── NEW: KV$ tier breakdown (L1/L2/L3 occupancy) ─────────────────
            "kv_l1_device_tokens":     int(kv_l1_device_tokens),
            "kv_l2_host_tokens":       int(kv_l2_host_tokens),
            "kv_l3_storage_tokens":    int(kv_l3_storage_tokens),
            "hicache_host_used_tokens":int(hicache_host_used),
            "hicache_host_total_tokens":int(hicache_host_total),
            "hicache_host_fill_pct":   round(hicache_host_used / max(hicache_host_total, 1) * 100, 1),

            # ── NEW: KV$ eviction / restore (token counts this instance) ─────
            "kv_evicted_tokens":       int(evicted_delta),
            "kv_restored_tokens":      int(load_back_delta),
            "kv_prefetched_tokens":    int(prefetched_delta),

            # ── NEW: HiCache operation latency ────────────────────────────────
            "hicache_eviction_ms":     round(eviction_mean_s  * 1000, 2),
            "hicache_load_back_ms":    round(load_back_mean_s * 1000, 2),
            "hicache_queue_time_ms":   round(queue_time_mean_s * 1000, 2),

            # ── NEW: AI op breakdown (token-level ground truth) ───────────────
            "rt_prefill_compute_tokens": int(rt_prefill_compute),  # new KV$ writes
            "rt_prefill_cache_tokens":   int(rt_prefill_cache),    # KV$ cache hits
            "rt_decode_tokens":          int(rt_decode),           # KV$ read-all ops
            "cache_hit_rate_realtime_pct": cache_hit_rate_realtime,
            "new_token_ratio_mean":      round(new_tok_ratio, 3),

            # ── NEW: Server-side latency histograms (ground truth) ────────────
            "server_ttft_ms":          round(ttft_mean_s  * 1000, 2),
            "server_itl_ms":           round(itl_mean_s   * 1000, 2),
            "server_e2e_ms":           round(e2e_mean_s   * 1000, 2),

            # ── NEW: Queue and throughput ──────────────────────────────────────
            "num_queue_reqs_peak":     int(n_queue_peak),
            "decode_sum_seq_lens":     int(decode_sum_seq_lens),
            "utilization_mean":        round(utilization * 100, 1),

            "num_samples":             n,
        }

    def _empty(self) -> dict:
        return {k: v for k, v in {
            "ai_op_type": "unknown", "ai_op_prefill_tok_s": 0.0,
            "ai_op_decode_tok_s": 0.0, "kv_cache_hit_rate_pct": 0.0,
            "num_running_req_mean": 0.0, "token_usage_peak": 0.0,
            "kv_pool_capacity_tokens": 0, "kv_used_tokens_peak": 0,
            "kv_l1_device_tokens": 0, "kv_l2_host_tokens": 0,
            "kv_l3_storage_tokens": 0, "hicache_host_used_tokens": 0,
            "hicache_host_total_tokens": 0, "hicache_host_fill_pct": 0.0,
            "kv_evicted_tokens": 0, "kv_restored_tokens": 0, "kv_prefetched_tokens": 0,
            "hicache_eviction_ms": 0.0, "hicache_load_back_ms": 0.0, "hicache_queue_time_ms": 0.0,
            "rt_prefill_compute_tokens": 0, "rt_prefill_cache_tokens": 0, "rt_decode_tokens": 0,
            "cache_hit_rate_realtime_pct": 0.0, "new_token_ratio_mean": 0.0,
            "server_ttft_ms": 0.0, "server_itl_ms": 0.0, "server_e2e_ms": 0.0,
            "num_queue_reqs_peak": 0, "decode_sum_seq_lens": 0, "utilization_mean": 0.0,
            "num_samples": 0,
        }.items()}


# ── Memory snapshots ──────────────────────────────────────────────────────────

def _hbm_snapshot() -> tuple[float, float]:
    """(used_gb_all_gpus, util_pct) from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        used = cap = 0.0
        for line in out.strip().splitlines():
            p = line.split(",")
            if len(p) == 2:
                used += float(p[0].strip())
                cap  += float(p[1].strip())
        return round(used/1024, 2), round(used/max(cap,1)*100, 1)
    except Exception:
        return 0.0, 0.0


def _dram_snapshot() -> tuple[float, float]:
    """(used_gb, util_pct) from /proc/meminfo."""
    try:
        info: dict[str, float] = {}
        for line in open("/proc/meminfo").read().splitlines():
            p = line.split()
            if len(p) >= 2:
                info[p[0].rstrip(":")] = int(p[1]) / 1024 / 1024
        total   = info.get("MemTotal", 0)
        used    = max(0, total - info.get("MemFree",0)
                      - info.get("Buffers",0) - info.get("Cached",0))
        return round(used, 2), round(used/max(total,1)*100, 1)
    except Exception:
        return 0.0, 0.0


# ── Prompt builder ────────────────────────────────────────────────────────────

def _extract_affected_files(patch: str) -> list[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            p = line.split()
            if len(p) >= 3:
                f = p[2].lstrip("b/")
                if f not in files:
                    files.append(f)
    return files


def _fetch_github_files(repo: str, commit: str, files: list[str]) -> str:
    """
    Fetch file contents from GitHub raw API at base_commit.

    Authentication:
      Set GITHUB_TOKEN env var for 5000 req/hr (vs 60/hr unauthenticated).
      export GITHUB_TOKEN=ghp_your_token_here

    Includes up to 6 files × 500 lines each ≈ 8000–12000 tokens.
    """
    parts   = []
    base    = f"https://raw.githubusercontent.com/{repo}/{commit}"
    token   = os.environ.get("GITHUB_TOKEN", "")
    headers = {"User-Agent": "amoprof/1.0"}
    if token:
        headers["Authorization"] = f"token {token}"

    for fname in files[:6]:
        try:
            req = urllib.request.Request(f"{base}/{fname}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    content  = r.read().decode("utf-8", errors="replace")
                    lines    = content.splitlines()
                    total    = len(lines)
                    sample   = "\n".join(lines[:500])
                    truncnote = f"  [showing first 500 of {total} lines]" if total > 500 else ""
                    parts.append(
                        f"### {fname}{truncnote}\n```python\n{sample}\n```")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                log.warning(
                    f"GitHub rate limit fetching {fname}. "
                    f"Set GITHUB_TOKEN env var: export GITHUB_TOKEN=ghp_xxx")
            elif e.code == 404:
                log.debug(f"GitHub 404 for {fname} at commit {commit[:8]}")
            else:
                log.debug(f"GitHub fetch {fname}: HTTP {e.code}")
        except Exception as e:
            log.debug(f"GitHub fetch {fname}: {e}")

    return "\n\n".join(parts)


def _fetch_test_files(repo: str, commit: str, test_patch: str) -> str:
    """
    Fetch test files from GitHub at base_commit using the test_patch field.
    Showing the model the failing tests is the single biggest accuracy boost
    because it tells the model exactly what behaviour needs to change.
    """
    test_files = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git a/"):
            p = line.split()
            if len(p) >= 3:
                f = p[2].lstrip("b/")
                if ("test" in f.lower() or "spec" in f.lower()) \
                        and f not in test_files:
                    test_files.append(f)

    if not test_files:
        return ""

    token   = os.environ.get("GITHUB_TOKEN", "")
    headers = {"User-Agent": "amoprof/1.0"}
    if token:
        headers["Authorization"] = f"token {token}"
    base  = f"https://raw.githubusercontent.com/{repo}/{commit}"
    parts = []

    for fname in test_files[:3]:
        try:
            req = urllib.request.Request(f"{base}/{fname}", headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                if r.status == 200:
                    lines = r.read().decode("utf-8", errors="replace").splitlines()[:200]
                    parts.append(f"### {fname}\n```python\n" + "\n".join(lines) + "\n```")
        except Exception as e:
            log.debug(f"Test file fetch {fname}: {e}")

    return "\n\n".join(parts)


def _build_prompt(instance: dict) -> tuple[str, bool]:
    """
    Build the model prompt. Returns (prompt_text, has_actual_code).

    Content included (in order of priority):
      1. Source files at base_commit  — what the model must modify
      2. Test files at base_commit    — what the model must make pass
      3. Fallback: filenames + patch structure when GitHub fetch fails
    """
    problem    = instance.get("problem_statement", "")
    hints      = instance.get("hints_text", "")
    patch      = instance.get("patch", "")
    test_patch = instance.get("test_patch", "")
    repo       = instance.get("repo", "")
    commit     = instance.get("base_commit", "")

    if hints:
        problem = problem.strip() + "\n\nHints:\n" + hints.strip()

    affected = _extract_affected_files(patch)

    # Fetch test files (used by all paths)
    tc = _fetch_test_files(repo, commit, test_patch) if (repo and commit) else ""
    test_ctx = tc or "(test files not available — set GITHUB_TOKEN env var)"

    # Source 1: stored file_contents field (some dataset versions)
    stored = instance.get("file_contents", {})
    if isinstance(stored, dict) and stored:
        fc = "\n\n".join(
            f"### {fn}\n```python\n{ct[:3000]}\n```"
            for fn, ct in list(stored.items())[:6])
        return PROMPT_WITH_CODE.format(
            problem_statement=problem[:5000],
            file_contents=fc[:10000],
            test_contents=test_ctx), True

    # Source 2: GitHub raw API
    if repo and commit and affected:
        fc = _fetch_github_files(repo, commit, affected)
        if fc:
            return PROMPT_WITH_CODE.format(
                problem_statement=problem[:5000],
                file_contents=fc[:10000],
                test_contents=test_ctx), True

    # Source 3: fallback — filenames + patch structure hint
    if affected:
        file_ctx = "Relevant files:\n" + "".join(f"  - {f}\n" for f in affected[:8])
        hdrs = "\n".join(
            l for l in patch.splitlines()
            if l.startswith(("diff --git", "---", "+++", "@@")))[:2000]
        if hdrs:
            file_ctx += f"\nPatch structure hint:\n{hdrs}"
    else:
        file_ctx = "(no file context available)"

    return PROMPT_FILES_ONLY.format(
        problem_statement=problem[:5000],
        file_context=file_ctx[:3000],
        test_contents=test_ctx), False


# ── Streaming model call with TTFT / TPOT measurement ─────────────────────────

def _call_model(model_hf_id: str, port: int, prompt: str,
                max_tokens: int = 8192, timeout: int = 900,
                is_reasoning: bool = False,
                messages_override: list | None = None,
                host: str = "127.0.0.1",
                ) -> tuple[str, int, int, float, float, list[float]]:
    """
    Stream /v1/chat/completions and record per-token timestamps.

    messages_override: full conversation history for multi-turn use.
      If provided, it is sent as-is and prompt/is_reasoning are ignored.

    Returns: (text, prompt_tokens, output_tokens, duration_s, ttft_ms, itl_ms_list)
    """
    if messages_override is not None:
        messages = messages_override
    else:
        messages = []
        if is_reasoning:
            messages.append({"role": "system", "content": REASONING_SYSTEM})
        messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model":       model_hf_id,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "stream":      True,
    }).encode()

    t0           = time.perf_counter()
    deadline     = time.time() + timeout
    chunks:      list[str]   = []
    token_times: list[float] = []
    prompt_tokens = output_tokens = 0

    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        if resp.status != 200:
            err = resp.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"Server HTTP {resp.status}: {err}")

        buf  = b""
        done = False
        while not done:
            if time.time() > deadline:
                log.warning(
                    f"Wall-clock deadline {timeout}s — "
                    f"{len(token_times)} tokens so far. Returning partial.")
                break
            conn.sock.settimeout(30)
            try:
                raw = resp.read(8192)
            except (TimeoutError, OSError):
                break
            if not raw:
                break
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line == b"data: [DONE]":
                    done = True
                    break
                if line.startswith(b"data: "):
                    try:
                        ev    = json.loads(line[6:])
                        delta = ev.get("choices",[{}])[0].get("delta",{})
                        now   = time.perf_counter()

                        # Qwen3/QwQ reasoning models stream a `reasoning_content`
                        # field (the <think>...</think> block) before any `content`
                        # arrives.  We must record the first token time regardless
                        # of which field carries it, otherwise token_times stays
                        # empty until visible content begins and ttft_ms is 0.
                        txt = delta.get("content") or ""
                        reasoning_txt = delta.get("reasoning_content") or ""

                        if txt:
                            chunks.append(txt)
                            token_times.append(now)
                        elif reasoning_txt and not token_times:
                            # First token of the thinking phase — record for TTFT
                            # but do not add to visible chunks.
                            token_times.append(now)

                        usage = ev.get("usage") or {}
                        if usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                        if usage.get("completion_tokens"):
                            output_tokens = usage["completion_tokens"]
                    except (json.JSONDecodeError, KeyError):
                        pass
        conn.close()

    except (ConnectionRefusedError, OSError) as e:
        raise RuntimeError(
            f"Cannot reach server on port {port}: {e}\n"
            f"Start SGLang first or omit --swebench-server-port.") from e

    dur  = round(time.perf_counter() - t0, 3)
    text = "".join(chunks)

    if output_tokens == 0:
        output_tokens = len(token_times) or max(1, len(text.split()))
    if prompt_tokens == 0 and text:
        prompt_tokens = max(1, len(prompt) // 4)

    ttft_ms = round((token_times[0] - t0) * 1000, 1) if token_times else 0.0
    itl_ms  = [round((token_times[i] - token_times[i-1]) * 1000, 2)
               for i in range(1, len(token_times))]

    return text, prompt_tokens, output_tokens, dur, ttft_ms, itl_ms


# ── Patch utilities ───────────────────────────────────────────────────────────

def _extract_test_ids(instance: dict) -> list[str]:
    """
    Extract the specific test node IDs that must pass from the instance.

    SWE-bench datasets provide FAIL_TO_PASS: the exact pytest node IDs
    (e.g. "tests/test_foo.py::TestBar::test_baz") that should flip from
    failing to passing when the correct patch is applied.

    Falls back to extracting test file paths from test_patch if the field
    is missing (older dataset versions).
    """
    # Preferred: explicit test node IDs
    ftp = instance.get("FAIL_TO_PASS", [])
    if isinstance(ftp, str):
        import json as _j
        try:
            ftp = _j.loads(ftp)
        except Exception:
            ftp = [ftp] if ftp.strip() else []
    if ftp:
        return list(ftp)

    # Fallback: parse test files from test_patch diff header
    # Format: "diff --git a/<path> b/<path>" — use the b/ side (destination)
    test_files = []
    for line in instance.get("test_patch", "").splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split()
            if len(parts) >= 4:
                # parts[3] is "b/<path>" — strip the "b/" prefix
                f = parts[3][2:] if parts[3].startswith("b/") else parts[3]
            elif len(parts) == 3:
                # Unusual: only one path listed — strip either a/ or b/
                f = parts[2]
                if f.startswith("b/"):
                    f = f[2:]
                elif f.startswith("a/"):
                    f = f[2:]
            else:
                continue
            if ("test" in f.lower() or "spec" in f.lower()) and f not in test_files:
                test_files.append(f)
    return test_files


def _run_tests_in_docker(
    instance:   dict,
    patch_path: "Path",
    work_dir:   "Path",
    image:      str,
    timeout:    int = 180,
) -> tuple[bool, str]:
    """
    Apply patch inside the per-instance Docker container and run the
    FAIL_TO_PASS tests. Returns (patch_applied: bool, test_output: str).

    This is used during the agent loop (between turns) to give the model
    real assertion failures and tracebacks rather than just "patch applies
    cleanly." The same Docker image is reused for final scoring via
    run_evaluation — this call is read-only (container is discarded after).

    Container layout (swebench standard):
      /testbed        — checked-out repo at base_commit
      /patch/fix.patch — mounted read-only from host
    """
    iid        = instance.get("instance_id", "unknown")
    test_ids   = _extract_test_ids(instance)
    patch_text = patch_path.read_text() if patch_path.exists() else ""

    if not patch_text.strip():
        return False, "No patch to apply."

    # Build the in-container shell script:
    #   1. Apply the patch (git apply from /testbed)
    #   2. Run only the FAIL_TO_PASS tests with short tracebacks
    #   3. Emit a clear separator so we can parse applied vs output
    test_args = " ".join(test_ids[:8]) if test_ids else "tests/"
    lines = [
        "set -e",
        "cd /testbed",
        "echo === APPLYING PATCH ===",
        "git apply /patch/fix.patch 2>&1 && echo PATCH_APPLIED_OK"
        " || (echo PATCH_APPLY_FAILED; git apply --reject /patch/fix.patch 2>&1; exit 1)",
        "echo === RUNNING TESTS ===",
        f"python -m pytest {test_args} --tb=short -x --no-header -q 2>&1 | head -200",
    ]
    script = "\n".join(lines) + "\n"

    script_path = work_dir / "run_tests.sh"
    script_path.write_text(script)

    # Ensure image is local before running (uses cached pull, no repeat pulls)
    _ensure_image(image, timeout=300)

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",          # no outbound network inside container
        "-v", f"{patch_path.parent}:/patch:ro",
        "-v", f"{script_path}:/run_tests.sh:ro",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "--memory", "4g",
        "--cpus", "2",
        image,
        "bash", "/run_tests.sh",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout)
        output = result.stdout + result.stderr
        (work_dir / "docker_test_output.txt").write_text(output)
        applied  = "PATCH_APPLIED_OK" in output
        return applied, output[-4000:]   # trim to ~4K chars for model context
    except subprocess.TimeoutExpired:
        return True, f"Tests timed out after {timeout}s."
    except FileNotFoundError:
        return False, "docker not found — is Docker running?"
    except Exception as e:
        return False, f"Docker test run error: {e}"



def _run_turns(
    server_mid: str, port: int,
    instance:   dict,
    num_turns:  int,
    max_tokens: int,
    call_timeout: int,
    is_reasoning: bool,
    work_dir:   "Path",
    use_docker: bool = False,
    docker_image: str = "",
    docker_timeout: int = 180,
    docker_in_loop: bool = False,
) -> dict:
    """
    Multi-turn agent loop for SWE-bench.

    Turn structure:
      Turn 1:  prefill(problem + source_files + test_files) → decode(patch)
      Turn 2:  prefill(history + test_failure_output)       → decode(revised_patch)
      Turn N:  prefill(full_history)                        → decode(final_patch)

    Why num_turns affects memory tiers:
    ─────────────────────────────────────────────────────────────────────
    Turn 1:  KV$ = kv(prompt_tokens)           ← baseline allocation
    Turn 2:  KV$ = kv(prompt + patch1 + fail)  ← larger: prefix grows
    Turn N:  KV$ = kv(sum of all turns)        ← largest prefix

    SGLang's RadixAttention reuses the common prefix across turns,
    so from turn 2 onward the incremental KV$ cost is only the NEW
    tokens in that turn, not the full history. This means:
      - kv_cache_hit_rate_pct rises significantly after turn 1
      - HBM pressure grows sub-linearly (prefix cached)
      - SSD eviction pressure rises at lower concurrency than single-turn

    AI op → memory op mapping per turn:
    ─────────────────────────────────────────────────────────────────────
    PREFILL (each new turn):
      ai_op    : prefill
      hbm_op   : KV$ write  (new tokens appended to KV cache)
      dram_op  : minimal     (weights stay in HBM, no host transfer)
      ssd_op   : eviction write IF pool was already full
      pattern  : sequential large write (layer-by-layer K+V tensors)
      size     : new_tokens × kv_bytes_per_token

    DECODE (each token in the patch):
      ai_op    : decode (or "reasoning" for R1 CoT)
      hbm_op   : KV$ read-all (ENTIRE context re-read for attention)
      dram_op  : HiCache staging if SSD restore active
      ssd_op   : random read IF evicted blocks needed
      pattern  : random small read (32K-256K per KV$ miss)
      size     : variable per miss, ~10 GB at 65K ctx

    KV$ EVICTION (when pool overflows):
      trigger  : new prefill cannot fit; HiCache evicts cold blocks
      hbm_op   : HBM → DRAM transfer (DMA)
      dram_op  : temporary staging buffer
      ssd_op   : write to /mnt/sglang_dv3 (HiCache file backend)
      pattern  : random write, 32K-256K blocks
      syscall  : pwrite64() via mmap or write()

    KV$ RESTORE (cache miss during decode):
      trigger  : attention needs a block that was evicted
      ssd_op   : pread64() from /mnt/sglang_dv3
      dram_op  : staging buffer (DRAM as L2 cache for SSD)
      hbm_op   : DMA upload back to GPU HBM
      latency  : ~1-10 ms per restore → visible as TPOT spike

    Returns dict with all turn-level metrics merged into per-instance aggregates.
    """
    import json as _json

    prompt, has_code = _build_prompt(instance)
    messages = []  # OpenAI-format conversation history

    # Add system prompt for reasoning models
    if is_reasoning:
        messages.append({"role": "system", "content": REASONING_SYSTEM})

    # First user message = the bug description + code
    messages.append({"role": "user", "content": prompt})

    turns_data = []
    total_pt = total_ot = 0
    total_dur = total_ttft = 0.0
    all_itl: list[float] = []

    # Accumulators for per-op HBM tracking
    hbm_prefill_deltas = []
    hbm_decode_deltas  = []
    kv_fill_pcts       = []
    ssd_reads_mb       = []
    ssd_writes_mb      = []

    final_patch = ""

    for turn_idx in range(num_turns):
        turn_num = turn_idx + 1

        # HBM snapshot before this turn's prefill
        hbm_before_turn, _ = _hbm_snapshot()

        # Call model — streaming with per-token timestamps
        try:
            gen_text, pt, ot, dur, ttft_ms, itl_ms = _call_model(
                server_mid, port, "",   # prompt passed via messages
                max_tokens=max_tokens,
                timeout=call_timeout,
                is_reasoning=is_reasoning,
                messages_override=messages,
            )
        except RuntimeError as e:
            log.error(f"  [turn {turn_num}] {e}")
            break

        # HBM snapshot after prefill (approximated as after first token)
        hbm_after_prefill, _ = _hbm_snapshot()
        hbm_prefill_delta = round(hbm_after_prefill - hbm_before_turn, 2)
        hbm_prefill_deltas.append(hbm_prefill_delta)

        # HBM snapshot after decode (end of turn)
        hbm_after_decode, _ = _hbm_snapshot()
        hbm_decode_delta = round(hbm_after_decode - hbm_after_prefill, 2)
        hbm_decode_deltas.append(hbm_decode_delta)

        total_pt  += pt
        total_ot  += ot
        total_dur += dur
        total_ttft += ttft_ms
        all_itl.extend(itl_ms)

        tpot_mean = round(statistics.mean(itl_ms), 2) if itl_ms else 0.0

        # Save patch from this turn
        clean = _extract_patch(gen_text)
        if _validate_patch(clean):
            final_patch = clean
        patch_path = work_dir / f"model_turn{turn_num}.patch"
        patch_path.write_text(clean or gen_text)
        (work_dir / f"model_turn{turn_num}_raw.txt").write_text(gen_text)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": gen_text})

        # If not the last turn: run tests to get failure feedback for next turn.
        # With use_docker=True we run the real test suite inside the per-instance
        # Docker container and feed actual assertion failures / tracebacks back to
        # the model — this is what makes the agent loop meaningful.
        # Without Docker we fall back to git apply --check (syntax only).
        test_feedback = ""
        turn_resolved = False
        turn_applied  = False
        if turn_num < num_turns and _validate_patch(clean):
            if use_docker and docker_image and docker_in_loop:
                # Real in-loop evaluation: apply patch + run FAIL_TO_PASS tests
                log.info(f"  [turn {turn_num}] running tests in Docker ({docker_image[:60]})")
                turn_applied, raw_output = _run_tests_in_docker(
                    instance, patch_path, work_dir, docker_image,
                    timeout=docker_timeout)

                if not turn_applied:
                    test_feedback = (
                        "Your patch failed to apply with git apply:\n"
                        + raw_output[:2000]
                        + "\n\nCheck line numbers and context lines carefully."
                    )
                elif "passed" in raw_output.lower() and "failed" not in raw_output.lower():
                    turn_resolved = True
                    test_feedback = (
                        "All FAIL_TO_PASS tests now pass. "
                        "Verify no PASS_TO_PASS tests were broken."
                    )
                    log.info(f"  [turn {turn_num}] tests PASSED — stopping early")
                else:
                    # Extract the most useful part: the FAILED block + short traceback
                    lines = raw_output.splitlines()
                    # Find FAILED lines and surrounding context
                    relevant = []
                    for i, line in enumerate(lines):
                        if any(kw in line for kw in
                               ("FAILED", "ERROR", "AssertionError", "assert ",
                                "E   ", "FAIL:", "error:", "TypeError",
                                "AttributeError", "short test summary")):
                            relevant.extend(lines[max(0,i-2):min(len(lines),i+6)])
                    # Deduplicate while preserving order
                    seen = set()
                    deduped = []
                    for l in relevant:
                        if l not in seen:
                            seen.add(l)
                            deduped.append(l)
                    test_feedback = "\n".join(deduped[:80]) or raw_output[:2000]

                log.info(
                    f"  [turn {turn_num}] docker test: "
                    f"applied={turn_applied} resolved={turn_resolved} "
                    f"output_len={len(raw_output)}")
            else:
                # Fallback: syntax check only (no Docker)
                try:
                    proc = subprocess.run(
                        ["git", "apply", "--check", str(patch_path)],
                        capture_output=True, text=True, timeout=30, cwd=work_dir)
                    if proc.returncode != 0:
                        test_feedback = "Patch failed to apply:\n" + proc.stderr[:500]
                        turn_applied  = False
                    else:
                        test_feedback = (
                            "Patch applies cleanly (syntax OK). "
                            "Tests were not run — pass --swebench-docker to get "
                            "real test feedback in the agent loop.")
                        turn_applied = True
                except Exception as e:
                    test_feedback = f"Could not verify patch: {e}"

            if turn_resolved:
                # Tests already pass — no point spending more turns
                break

            # Craft feedback prompt that gives the model exactly what it needs:
            # its own patch + the real failure output + a clear instruction.
            test_ids_str = ", ".join(_extract_test_ids(instance)[:5]) or "the failing tests"
            next_prompt = (
                f"Your turn {turn_num} patch:\n"
                f"```diff\n{clean[:1500]}\n```\n\n"
                f"Test results:\n"
                f"```\n{test_feedback[:2500]}\n```\n\n"
                f"The tests that must pass are: {test_ids_str}\n\n"
                f"Provide a corrected unified diff patch that makes those tests pass. "
                f"Start directly with 'diff --git a/...' — no explanation."
            )
            messages.append({"role": "user", "content": next_prompt})

        # Record turn metrics
        turns_data.append({
            "turn":        turn_num,
            "prompt_tok":  pt,
            "output_tok":  ot,
            "ttft_ms":     ttft_ms,
            "tpot_ms":     tpot_mean,
            "hbm_before_gb": hbm_before_turn,
            "hbm_after_gb":  hbm_after_decode,
            "hbm_prefill_delta_gb": hbm_prefill_delta,
            "hbm_decode_delta_gb":  hbm_decode_delta,
            "ai_op":         "reasoning" if dur > 60 and ot < 200 else "decode",
            "patch_valid":   _validate_patch(clean),
            "turn_resolved": turn_resolved,
            "turn_applied":  turn_applied,
        })

        log.info(
            f"  [turn {turn_num}/{num_turns}] "
            f"pt={pt:,} ot={ot:,} "
            f"ttft={ttft_ms:.0f}ms tpot={tpot_mean:.1f}ms "
            f"HBM_prefill_delta={hbm_prefill_delta:+.1f}GB "
            f"HBM_decode_delta={hbm_decode_delta:+.1f}GB "
            f"patch={'yes' if _validate_patch(clean) else 'no'}")

    tpot_mean_all = round(statistics.mean(all_itl), 2) if all_itl else 0.0
    tpot_p99_all  = 0.0
    if all_itl:
        s = sorted(all_itl)
        tpot_p99_all = round(s[min(int(len(s)*.99), len(s)-1)], 2)

    return {
        "final_patch":         final_patch,
        "prompt":              prompt,
        "has_code":            has_code,
        "total_prompt_tokens": total_pt,
        "total_output_tokens": total_ot,
        "total_duration_s":    round(total_dur, 3),
        "ttft_ms":             round(total_ttft / max(len(turns_data), 1), 1),
        "tpot_mean_ms":        tpot_mean_all,
        "tpot_p99_ms":         tpot_p99_all,
        "num_turns_completed": len(turns_data),
        "turn_timeline":       _json.dumps(turns_data),
        "resolved_in_loop":    any(td.get("turn_resolved", False) for td in turns_data),
        "hbm_prefill_delta_gb": round(statistics.mean(hbm_prefill_deltas), 3) if hbm_prefill_deltas else 0.0,
        "hbm_decode_delta_gb":  round(statistics.mean(hbm_decode_deltas),  3) if hbm_decode_deltas  else 0.0,
    }


def _validate_patch(patch_text: str) -> bool:
    return bool(patch_text.strip()) and (
        "diff --git" in patch_text or
        ("--- " in patch_text and "+++ " in patch_text))


def _check_patch_complete(patch_text: str) -> bool:
    """True when patch ends cleanly (not mid-hunk or mid-line)."""
    lines = [l for l in patch_text.splitlines() if l.strip()]
    if not lines:
        return False
    last = lines[-1]
    return not (last.startswith("@@") or len(last) > 250)


def _extract_patch(raw_text: str) -> str:
    """
    Extract just the unified diff from model output.
    Strips preamble, markdown fences, and trailing explanation.

    Handles common R1/DeepSeek output patterns:
      - Clean:    diff --git ...
      - Fenced:   ```diff\ndiff --git ...\n```
      - Prefixed: "Here is the fix:\n\ndiff --git ..."
      - Suffixed: "diff --git ...\n\nThis patch fixes..."
    """
    import re as _re

    text = raw_text.strip()

    # Strip markdown code fences
    fence = _re.search(r"```(?:diff|patch)?[ \t]*\n(.*?)```",
                       text, _re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Find where the actual diff starts (first line starting with diff/---/+++)
    diff_start = len(text)
    for marker in ["diff --git ", "--- a/", "--- "]:
        idx = text.find(marker)
        if idx == -1:
            continue
        # Walk back to start of this line
        sol = text.rfind("\n", 0, idx)
        sol = sol + 1 if sol != -1 else 0
        if sol < diff_start:
            diff_start = sol

    if diff_start == len(text):
        return text   # no diff markers found

    patch = text[diff_start:]

    # Strip trailing non-diff content after the last hunk
    DIFF_PREFIXES = ("diff --git", "---", "+++", "@@", "+", "-", " ", "\\")
    lines   = patch.splitlines()
    end_idx = len(lines)
    in_hunk = False
    for i, line in enumerate(lines):
        if line.startswith(DIFF_PREFIXES):
            in_hunk = True
            end_idx = i + 1
        elif in_hunk and line.strip() == "":
            continue   # blank line between hunks — keep going
        elif in_hunk and not line.startswith(DIFF_PREFIXES):
            # Non-diff text after a hunk = trailing explanation
            end_idx = i
            break

    return "\n".join(lines[:end_idx]).strip()


def _run_docker_eval_pro(instance: dict, patch_path: Path, work_dir: Path,
                         run_id: str, workers: int = 1) -> tuple[bool, bool, int, int]:
    """
    Run SWE-bench Pro evaluation using Scale AI's Docker Hub images.

    SWE-bench Pro differences vs regular SWE-bench:
      - Dataset:     ScaleAI/SWE-bench_Pro (HuggingFace)
      - Docker Hub:  jefzda/sweap-images (not ghcr.io/epoch-research)
      - Image tag:   from instance["dockerhub_tag"] field in the dataset
      - Eval script: swe_bench_pro_eval.py (from scaleapi/SWE-bench_Pro-os)
        OR standard swebench.harness.run_evaluation with --instance_image_tag

    Setup (one-time):
      pip install swebench datasets
      docker login  # Docker Hub login (public images, no auth needed)

    Returns (patch_applied, resolved, tests_passed, tests_failed).
    """
    if not _validate_patch(patch_path.read_text()):
        return False, False, 0, 0

    iid        = instance["instance_id"]
    docker_tag = instance.get("dockerhub_tag", "")
    report_dir = work_dir / "eval_report"
    report_dir.mkdir(exist_ok=True)

    if not docker_tag:
        log.warning(
            f"No dockerhub_tag for {iid}. "
            f"Make sure dataset was loaded from ScaleAI/SWE-bench_Pro.")
        return True, False, 0, 0

    # Ensure image is available locally (pulls once, skips if already present)
    image = f"jefzda/sweap-images:{docker_tag}"
    _ensure_image(image)

    pred_path = work_dir / "predictions.jsonl"
    pred_path.write_text(json.dumps({
        "instance_id":        iid,
        "model_patch":        patch_path.read_text(),
        "model_name_or_path": run_id,
    }) + "\n")

    # Build harness command, probing for the args the installed version accepts.
    # swebench arg surface changed across versions:
    #   <2.1  --output_dir exists, --instance_image_tag does NOT exist
    #   >=2.1 --output_dir may be --report_path in some forks
    #   >=3.0 --instance_image_tag added
    # We probe with --help and fall back gracefully.
    base_cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name",     "ScaleAI/SWE-bench_Pro",
        "--split",            "test",
        "--instance_ids",     iid,
        "--predictions_path", str(pred_path),
        "--max_workers",      str(workers),
        "--run_id",           run_id,
    ]

    # --output_dir vs --report_path
    if _harness_supports("--output_dir"):
        base_cmd += ["--output_dir", str(report_dir)]
    elif _harness_supports("--report_path"):
        base_cmd += ["--report_path", str(report_dir)]

    # --instance_image_tag (only in newer swebench builds)
    if docker_tag and _harness_supports("--instance_image_tag"):
        base_cmd += ["--instance_image_tag", docker_tag]

    cmd = base_cmd
    env = {**os.environ, "DOCKER_IMAGE_PREFIX": "jefzda/sweap-images"}

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, env=env)
        out = result.stdout + result.stderr
        (work_dir / "harness_output.txt").write_text(out)

        if result.returncode != 0:
            log.warning(
                f"Pro eval exit={result.returncode} for {iid}. "
                f"Output: {out[:300]}")

        # Parse JSON report
        resolved = False
        tests_passed = tests_failed = 0
        for rfile in report_dir.rglob("*.json"):
            try:
                data  = json.loads(rfile.read_text())
                inner = data.get(iid, data) if isinstance(data, dict) else {}
                resolved = bool(inner.get("resolved", False))
                ts = inner.get("tests_status", {})
                if isinstance(ts, dict):
                    tests_passed = len(ts.get("PASSED", []))
                    tests_failed = len(ts.get("FAILED", []))
                break
            except Exception:
                pass

        if not resolved and "1 resolved" in out.lower():
            resolved = True

        applied = (result.returncode == 0 or tests_passed > 0
                   or tests_failed > 0)
        return applied, resolved, tests_passed, tests_failed

    except subprocess.TimeoutExpired:
        log.warning(f"Pro Docker eval timed out for {iid}")
        return True, False, 0, 0
    except Exception as e:
        log.warning(f"Pro Docker eval error for {iid}: {e}")
        return True, False, 0, 0


def _run_docker_eval(instance: dict, patch_path: Path, work_dir: Path,
                     split: str, run_id: str,
                     workers: int = 1) -> tuple[bool, bool, int, int]:
    """
    Run SWE-bench Docker harness. Returns (applied, resolved, passed, failed).
    workers: number of parallel test runners inside the harness (speeds up
             multi-file test suites; keep at 1 for single-instance eval).
    """
    if not _validate_patch(patch_path.read_text()):
        return False, False, 0, 0

    iid        = instance["instance_id"]
    report_dir = work_dir / "eval_report"
    report_dir.mkdir(exist_ok=True)

    pred_path = work_dir / "predictions.jsonl"
    pred_path.write_text(json.dumps({
        "instance_id":        iid,
        "model_patch":        patch_path.read_text(),
        "model_name_or_path": run_id,
    }) + "\n")

    base_cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name",     DATASET_IDS.get(split, DATASET_IDS["lite"]),
        "--split",            "test",
        "--instance_ids",     iid,
        "--predictions_path", str(pred_path),
        "--max_workers",      str(workers),
        "--run_id",           run_id,
    ]
    if _harness_supports("--output_dir"):
        base_cmd += ["--output_dir", str(report_dir)]
    elif _harness_supports("--report_path"):
        base_cmd += ["--report_path", str(report_dir)]

    cmd = base_cmd
    env = {**os.environ,
           "SWEBENCH_IMAGE_ORG":    "epoch-research",
           "SWEBENCH_IMAGE_PREFIX": "swe-bench.eval"}

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env)
        out = result.stdout + result.stderr
        (work_dir / "harness_output.txt").write_text(out)

        if result.returncode != 0:
            log.warning(
                f"run_evaluation exit={result.returncode} for {iid}. "
                f"If image missing: docker pull "
                f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}:latest\n"
                f"{out[:300]}")

        # Parse JSON report written by run_evaluation
        resolved = False
        tests_passed = tests_failed = 0
        for rfile in report_dir.rglob("*.json"):
            try:
                data  = json.loads(rfile.read_text())
                inner = data.get(iid, data) if isinstance(data, dict) else {}
                resolved = bool(inner.get("resolved", False))
                ts = inner.get("tests_status", {})
                if isinstance(ts, dict):
                    tests_passed = len(ts.get("PASSED", []))
                    tests_failed = len(ts.get("FAILED", []))
                elif isinstance(inner.get("n_passed"), int):
                    tests_passed = inner.get("n_passed", 0)
                    tests_failed = inner.get("n_failed", 0)
                break
            except Exception:
                pass

        # Fallback stdout grep
        if not resolved and "1 resolved" in out.lower():
            resolved = True

        applied = (result.returncode == 0 or "Applied patch" in out
                   or tests_passed > 0 or tests_failed > 0)
        return applied, resolved, tests_passed, tests_failed

    except subprocess.TimeoutExpired:
        log.warning(f"Docker eval timed out for {iid}")
        return True, False, 0, 0
    except Exception as e:
        log.warning(f"Docker eval error for {iid}: {e}")
        return True, False, 0, 0


# ── Server health ─────────────────────────────────────────────────────────────

def _wait_server_healthy(port: int, host: str = "127.0.0.1", timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://{host}:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ── Main harness ──────────────────────────────────────────────────────────────

class SWEBenchHarness:

    def __init__(self, plan, run_id: str, output_dir: Path,
                 split: str = "lite",
                 max_instances: Optional[int] = 50,
                 server_port: Optional[int] = None,
                 server_host: str = "127.0.0.1",
                 use_docker: bool = False,
                 max_tokens: int = 8192,
                 call_timeout: int = 900,
                 workers: int = 1,
                 inference_concurrency: int = 1,
                 num_turns: int = 1,
                 docker_in_loop: bool = False,
                 agent_backend: str = "none",
                 max_agent_steps: int = 30,
                 context_len: int = 65536,
                 instances_path: Optional[str] = None,
                 sweagent_default_config: Optional[str] = None,
                 sweagent_config_path: Optional[str] = None,
                 sweagent_model_name: Optional[str] = None,
                 sweagent_model_api_base: Optional[str] = None,
                 sweagent_model_api_key: str = "EMPTY",
                 sweagent_max_input_tokens: int = 50000,
                 sweagent_num_workers: int = 1,
                 sweagent_instances_type: str = "file",
                 sweagent_redo_existing: bool = True,
                 sweagent_shuffle: bool = False):
        self.plan          = plan
        self.run_id        = run_id
        self.output_dir    = output_dir
        self.split         = split
        self.max_instances = max_instances
        self.server_port   = server_port
        self.server_host   = server_host
        self.use_docker    = use_docker
        self.max_tokens    = max_tokens
        self.call_timeout  = call_timeout
        self.workers               = workers
        self.inference_concurrency = inference_concurrency
        self.num_turns             = num_turns
        self.docker_in_loop        = docker_in_loop
        self.agent_backend         = agent_backend
        self.max_agent_steps       = max_agent_steps
        self.context_len           = context_len
        self.instances_path        = instances_path
        self.sweagent_default_config = sweagent_default_config
        self.sweagent_config_path  = sweagent_config_path
        self.sweagent_model_name   = sweagent_model_name
        self.sweagent_model_api_base = sweagent_model_api_base
        self.sweagent_model_api_key = sweagent_model_api_key
        self.sweagent_max_input_tokens = sweagent_max_input_tokens
        self.sweagent_num_workers  = sweagent_num_workers
        self.sweagent_instances_type = sweagent_instances_type
        self.sweagent_redo_existing = sweagent_redo_existing
        self.sweagent_shuffle      = sweagent_shuffle
        output_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self) -> list[dict]:
        def _normalize_instance(rec: dict) -> dict:
            out = dict(rec or {})
            if not out.get("repo") and out.get("repo_name"):
                out["repo"] = out.get("repo_name")
            image_name = out.get("image_name", "") or ""
            if image_name and not out.get("dockerhub_tag") and ":" in image_name:
                out["dockerhub_tag"] = image_name.split(":", 1)[1]
            return out

        if self.instances_path:
            src = Path(self.instances_path)
            if not src.exists():
                raise RuntimeError(f"Instances file not found: {src}")
            text = src.read_text(encoding="utf-8")
            suffix = src.suffix.lower()
            instances: list[dict] = []
            try:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(text)
                    if isinstance(data, list):
                        instances = [_normalize_instance(x) for x in data if isinstance(x, dict)]
                    elif isinstance(data, dict):
                        for key in ("instances", "data", "items"):
                            if isinstance(data.get(key), list):
                                instances = [_normalize_instance(x) for x in data[key] if isinstance(x, dict)]
                                break
                elif suffix == ".jsonl":
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            instances.append(_normalize_instance(json.loads(line)))
                elif suffix == ".json":
                    data = json.loads(text)
                    if isinstance(data, list):
                        instances = [_normalize_instance(x) for x in data if isinstance(x, dict)]
                    elif isinstance(data, dict):
                        for key in ("instances", "data", "items"):
                            if isinstance(data.get(key), list):
                                instances = [_normalize_instance(x) for x in data[key] if isinstance(x, dict)]
                                break
            except Exception as e:
                raise RuntimeError(f"Failed to parse instances file {src}: {e}")
            if not instances:
                raise RuntimeError(f"No instances found in {src}")
            if self.max_instances:
                instances = instances[:self.max_instances]
            for i, inst in enumerate(instances):
                if isinstance(inst, dict):
                    inst["__amoprof_index"] = i
                    inst["__amoprof_instances_path"] = str(src)
            log.info(f"Loaded {len(instances)} instances from file {src}")
            return instances

        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("pip install datasets")
        ds = load_dataset(DATASET_IDS.get(self.split, DATASET_IDS["lite"]),
                          split="test")
        instances = [_normalize_instance(x) for x in list(ds)]
        if self.max_instances:
            instances = instances[:self.max_instances]
        log.info(f"Loaded {len(instances)} instances (split={self.split})")
        return instances

    def _launch_server(self, log_dir: Path) -> tuple[object, int]:
        from .bench_sglang import SGLangServer
        p   = self.plan
        srv = SGLangServer(
            model_hf_id     = p.model.hf_id,
            dtype           = p.dtype,
            gpu_memory_util = min(p.hbm_cap_gb / 80.0, 0.95),
            tensor_parallel = p.tensor_parallel,
            port            = 30000,
        )
        srv.start(log_dir)
        if not srv.wait_healthy(timeout_s=300):
            srv.stop()
            raise RuntimeError(
                "SGLang server failed. Check: " +
                str(log_dir / "sglang_server.log"))
        return srv, 30000

    def run_all(self) -> list[SWEBenchResult]:
        from .collectors import (IostatMonitor, NvmeSmartMonitor, PowerMonitor,
                                  VmstatMonitor, NvlinkPcieMonitor, NvmeDriverMonitor,
                                  SsdHardwareMonitor, BiolatencyCollector,
                                  RequestLatencyTracker, CudaKernelMonitor,
                                  NcuAttentionCollector, NsysTraceCollector,
                                  PerfStatCollector, PcmMemoryCollector,
                                  BpftraceCollector, TorchProfilerCollector,
                                  VtuneCollector)

        p         = self.plan
        instances = self._load_dataset()

        server = None
        port   = self.server_port
        if port is None:
            server, port = self._launch_server(self.output_dir / "server")
        else:
            log.info(f"Checking server on port {port} ...")
            if not _wait_server_healthy(port, host=self.server_host, timeout_s=10):
                raise RuntimeError(
                    f"No server on port {port}. "
                    f"Start SGLang or omit --swebench-server-port.")
            log.info("Server healthy")

        # Fetch exact model ID registered on the server
        try:
            with urllib.request.urlopen(
                    f"http://{self.server_host}:{port}/v1/models", timeout=5) as r:
                ms = json.loads(r.read()).get("data", [])
                server_mid = ms[0]["id"] if ms else p.model.hf_id
        except Exception:
            server_mid = p.model.hf_id
        log.info(f"Server model ID: {server_mid}")

        is_reasoning = any(x in server_mid.lower()
                           for x in ["r1", "r1-distill", "qwq"])

        smart    = NvmeSmartMonitor(p.ssd_device)
        power    = PowerMonitor()
        ssd_hw   = SsdHardwareMonitor(p.ssd_device)
        req_lat  = RequestLatencyTracker()
        smart.start(); power.start(); ssd_hw.start()

        # ── Deep profilers (run-level, attach to the model server process) ────
        # These are expensive and attach to the live server PID, so we start
        # them once per harness run rather than per instance.
        _server_pid = 0
        try:
            # Find the SGLang server PID via lsof on the model server port
            lsof_out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], text=True, timeout=5)
            _server_pid = int(lsof_out.strip().splitlines()[0])
        except Exception:
            pass

        ncu_col    = NcuAttentionCollector(_server_pid, work_dir=self.output_dir)
        nsys_col   = NsysTraceCollector(_server_pid, work_dir=self.output_dir,
                                        capture_duration_s=min(30, self.call_timeout // 4))
        vtune_col  = VtuneCollector(_server_pid, work_dir=self.output_dir,
                                    duration_s=min(30, self.call_timeout // 4))
        torch_col  = TorchProfilerCollector(port, output_dir=self.output_dir,
                                            duration_s=min(30, self.call_timeout // 4))
        if _server_pid:
            ncu_col.start()
            nsys_col.start()
            vtune_col.start()
            torch_col.start()
        else:
            log.warning("Could not resolve server PID — ncu/nsys/vtune/torch profilers disabled")

        # kv_pool_stats for pool_gb used inside _process_one
        _kv_stats = p.model.kv_pool_stats(
            hbm_cap_gb   = p.hbm_cap_gb,
            context_len  = self.context_len,
            mem_fraction = 0.80,
        )
        _pool_gb = _kv_stats['pool_capacity_gb']

        results: list[SWEBenchResult] = []
        results_lock = threading.Lock()
        idx_counter  = [0]

        def _process_one(instance: dict) -> "SWEBenchResult":
            with results_lock:
                my_idx = idx_counter[0]
                idx_counter[0] += 1
            n_total = len(instances)
            iid  = instance.get("instance_id", f"instance_{my_idx}")
            repo = instance.get("repo", "")
            log.info(f"[{my_idx+1}/{n_total}]  {iid}  ({repo})")

            work_dir = self.output_dir / iid.replace("/", "_")
            work_dir.mkdir(exist_ok=True)

            hbm_s, hbm_util   = _hbm_snapshot()
            dram_s, dram_util = _dram_snapshot()

            # Per-instance collectors wrapping the full turn loop
            iostat   = IostatMonitor(p.ssd_device)
            sampler  = SGLangMetricsSampler(port, interval_s=1.0)
            biolat   = BiolatencyCollector(p.ssd_device)
            vmstat   = VmstatMonitor()
            nvlink   = NvlinkPcieMonitor()
            nvme_drv  = NvmeDriverMonitor(p.ssd_device)
            cuda_kern = CudaKernelMonitor(interval_s=0.5)
            # Per-instance deep profilers
            perf_col  = PerfStatCollector(_server_pid)
            pcm_col   = PcmMemoryCollector(interval_s=1.0)
            # bpf: for agent backends (sweagent/mini), the run can last 30-60 min.
            # Use a large cap (call_timeout * max_steps) so bpf runs the full
            # duration and is stopped by bpf_col.stop() → SIGINT, not by timeout.
            _bpf_dur = (self.call_timeout * self.max_agent_steps
                        if self.agent_backend != "none"
                        else max(5, self.call_timeout // 6))
            bpf_col   = BpftraceCollector(_server_pid,
                                          duration_s=_bpf_dur,
                                          work_dir=work_dir)
            iostat.start()
            sampler.start()
            biolat.start()
            vmstat.start()
            nvlink.start()
            nvme_drv.start()
            cuda_kern.start()
            if _server_pid:
                perf_col.start()
                pcm_col.start()
                bpf_col.start()

            # Run agent loop (num_turns=1 → single call, identical to old behaviour)
            # Determine the Docker image for this instance (needed for in-loop testing)
            _docker_image = ""
            if self.use_docker:
                if self.split in PRO_SPLITS:
                    _docker_image = "jefzda/sweap-images:" + instance.get("dockerhub_tag","")
                else:
                    _docker_image = (
                        f"ghcr.io/epoch-research/swe-bench.eval.x86_64."
                        f"{instance.get('instance_id','')}:latest"
                    )

            if self.agent_backend != "none" and _docker_image:
                # Dispatch to the selected agent backend.
                # Wrapped in try/except so a subprocess crash or timeout in
                # sweagent/mini doesn't kill the entire instance — we fall
                # back to a single-shot _run_turns instead.
                from ._agent import dispatch_agent
                try:
                    turn_result = dispatch_agent(
                        backend        = self.agent_backend,
                        server_mid     = server_mid,
                        port           = port,
                        instance       = instance,
                        max_tokens     = self.max_tokens,
                        call_timeout   = self.call_timeout,
                        is_reasoning   = is_reasoning,
                        work_dir       = work_dir,
                        docker_image   = _docker_image,
                        max_steps      = self.max_agent_steps,
                        step_timeout   = max(120, self.call_timeout // 3),
                        hbm_snapshot_fn= _hbm_snapshot,
                        fallback_fn    = _run_turns,
                        split          = self.split,
                        sweagent_options = {
                            "default_config": self.sweagent_default_config,
                            "config_path": self.sweagent_config_path,
                            "model_name": self.sweagent_model_name,
                            "api_base": self.sweagent_model_api_base,
                            "api_key": self.sweagent_model_api_key,
                            "max_input_tokens": self.sweagent_max_input_tokens,
                            "num_workers": self.sweagent_num_workers,
                            "instances_type": self.sweagent_instances_type,
                            "redo_existing": self.sweagent_redo_existing,
                            "shuffle": self.sweagent_shuffle,
                            "instances_path": self.instances_path,
                        },
                    )
                except Exception as _agent_exc:
                    log.error(f"  [{iid}] dispatch_agent raised: {_agent_exc} — "
                              f"falling back to single-shot _run_turns")
                    turn_result = _run_turns(
                        server_mid, port, instance,
                        num_turns=1, max_tokens=self.max_tokens,
                        call_timeout=self.call_timeout,
                        is_reasoning=is_reasoning,
                        work_dir=work_dir)
            else:
                turn_result = _run_turns(
                    server_mid, port, instance,
                    num_turns      = self.num_turns,
                    max_tokens     = self.max_tokens,
                    call_timeout   = self.call_timeout,
                    is_reasoning   = is_reasoning,
                    work_dir       = work_dir,
                    use_docker     = self.use_docker,
                    docker_image   = _docker_image,
                    docker_timeout = max(120, self.call_timeout // 3),
                    docker_in_loop = self.docker_in_loop,
                )

            iostat.stop()
            sg_m      = sampler.stop()
            bio_m     = biolat.stop()
            vmstat_m  = vmstat.stop()
            nvlink_m  = nvlink.stop()
            nvme_drv_m= nvme_drv.stop()
            cuda_m    = cuda_kern.stop()
            io_m      = iostat.summarise()
            # Per-instance deep profilers
            perf_m    = perf_col.stop()  if _server_pid else {}
            pcm_m     = pcm_col.stop()   if _server_pid else {}
            bpf_m     = bpf_col.stop()   if _server_pid else {}
            req_lat.record(turn_result["total_duration_s"])

            # Unpack turn results
            clean_patch     = turn_result["final_patch"]
            has_code        = turn_result["has_code"]
            pt              = turn_result["total_prompt_tokens"]
            ot              = turn_result["total_output_tokens"]
            dur             = turn_result["total_duration_s"]
            ttft_ms         = turn_result["ttft_ms"]
            tpot_mean       = turn_result["tpot_mean_ms"]
            tpot_p99        = turn_result["tpot_p99_ms"]
            completed_turns = turn_result["num_turns_completed"]
            turn_timeline   = turn_result["turn_timeline"]

            # Agent-mode extra metrics
            _resolved_in_loop  = bool(turn_result.get("resolved_in_loop", False))
            _agent_steps_taken = int(turn_result.get("num_turns_completed", 0)) \
                                 if self.agent_backend != "none" else 0

            # Per-op HBM breakdown
            hbm_prefill_delta = turn_result.get("hbm_prefill_delta_gb", 0.0)
            hbm_decode_delta  = turn_result.get("hbm_decode_delta_gb",  0.0)
            kv_pool_fill      = round(sg_m.get("token_usage_peak", 0.0), 1)

            # Save final patch
            patch_path = work_dir / "model.patch"
            patch_path.write_text(clean_patch or "")
            is_complete = _check_patch_complete(clean_patch)

            # Take end-of-instance memory snapshots BEFORE computing deltas
            hbm_e, _  = _hbm_snapshot()
            dram_e, _ = _dram_snapshot()

            # Now safe to compute deltas using both start and end values
            pool_gb       = _pool_gb  # from plan.model.kv_pool_stats() above
            kv_evicted_gb = round(max(0.0, (hbm_s + abs(hbm_prefill_delta)) - pool_gb), 2)
            dram_hicache_gb = round(max(0.0, dram_e - dram_s), 2) \
                              if io_m.get("write_bw_mb_mean", 0) > 0.1 else 0.0

            ai_op    = sg_m["ai_op_type"]
            pf_tok_s = sg_m["ai_op_prefill_tok_s"]
            dc_tok_s = sg_m["ai_op_decode_tok_s"]
            kv_hit   = sg_m["kv_cache_hit_rate_pct"]
            n_req    = sg_m["num_running_req_mean"]

            applied = resolved = False
            tp = tf = 0
            if self.use_docker and _validate_patch(clean_patch):
                if self.split in PRO_SPLITS:
                    applied, resolved, tp, tf = _run_docker_eval_pro(
                        instance, patch_path, work_dir,
                        self.run_id, workers=self.workers)
                else:
                    applied, resolved, tp, tf = _run_docker_eval(
                        instance, patch_path, work_dir,
                        self.split, self.run_id,
                        workers=self.workers)
            elif _validate_patch(clean_patch):
                applied = True

            r = SWEBenchResult(
                run_id          = self.run_id,
                timestamp       = datetime.now().isoformat(timespec="seconds"),
                model           = p.model.alias,
                split           = self.split,
                instance_id     = iid,
                repo            = repo,
                instance_index  = my_idx,
                prompt_tokens   = pt,
                output_tokens   = ot,
                prompt_has_code = has_code,
                num_turns       = completed_turns,
                ai_op_type            = ai_op,
                ai_op_prefill_tok_s   = pf_tok_s,
                ai_op_decode_tok_s    = dc_tok_s,
                kv_cache_hit_rate_pct = kv_hit,
                num_running_req_mean  = n_req,
                # KV$ pool
                token_usage_peak          = sg_m.get("token_usage_peak",          0.0),
                kv_pool_capacity_tokens   = sg_m.get("kv_pool_capacity_tokens",   0),
                kv_used_tokens_peak       = sg_m.get("kv_used_tokens_peak",       0),
                # KV$ tier breakdown
                kv_l1_device_tokens       = sg_m.get("kv_l1_device_tokens",       0),
                kv_l2_host_tokens         = sg_m.get("kv_l2_host_tokens",         0),
                kv_l3_storage_tokens      = sg_m.get("kv_l3_storage_tokens",      0),
                hicache_host_used_tokens  = sg_m.get("hicache_host_used_tokens",  0),
                hicache_host_total_tokens = sg_m.get("hicache_host_total_tokens", 0),
                hicache_host_fill_pct     = sg_m.get("hicache_host_fill_pct",     0.0),
                # KV$ movement
                kv_evicted_tokens         = sg_m.get("kv_evicted_tokens",         0),
                kv_restored_tokens        = sg_m.get("kv_restored_tokens",        0),
                kv_prefetched_tokens      = sg_m.get("kv_prefetched_tokens",      0),
                # HiCache latency
                hicache_eviction_ms       = sg_m.get("hicache_eviction_ms",       0.0),
                hicache_load_back_ms      = sg_m.get("hicache_load_back_ms",      0.0),
                hicache_queue_time_ms     = sg_m.get("hicache_queue_time_ms",     0.0),
                # AI op breakdown
                rt_prefill_compute_tokens    = sg_m.get("rt_prefill_compute_tokens",    0),
                rt_prefill_cache_tokens      = sg_m.get("rt_prefill_cache_tokens",      0),
                rt_decode_tokens             = sg_m.get("rt_decode_tokens",             0),
                cache_hit_rate_realtime_pct  = sg_m.get("cache_hit_rate_realtime_pct",  0.0),
                new_token_ratio_mean         = sg_m.get("new_token_ratio_mean",         0.0),
                # Server-side latency
                server_ttft_ms            = sg_m.get("server_ttft_ms",            0.0),
                server_itl_ms             = sg_m.get("server_itl_ms",             0.0),
                server_e2e_ms             = sg_m.get("server_e2e_ms",             0.0),
                # Queue
                num_queue_reqs_peak       = sg_m.get("num_queue_reqs_peak",       0),
                decode_sum_seq_lens       = sg_m.get("decode_sum_seq_lens",       0),
                utilization_mean          = sg_m.get("utilization_mean",          0.0),
                ttft_ms       = ttft_ms,
                tpot_mean_ms  = tpot_mean,
                tpot_p99_ms   = tpot_p99,
                total_time_s  = round(dur, 3),
                tok_per_s     = round(ot / max(dur, 0.001), 2),
                hbm_used_gb_start    = hbm_s,
                hbm_used_gb_end      = hbm_e,
                hbm_delta_gb         = round(hbm_e - hbm_s, 2),
                hbm_util_pct         = hbm_util,
                hbm_peak_gb          = round(max(hbm_s, hbm_e), 2),
                hbm_prefill_delta_gb = hbm_prefill_delta,
                hbm_decode_delta_gb  = hbm_decode_delta,
                hbm_kv_pool_fill_pct = kv_pool_fill,
                hbm_kv_evicted_gb    = kv_evicted_gb,
                dram_used_gb_start   = dram_s,
                dram_used_gb_end     = dram_e,
                dram_delta_gb        = round(dram_e - dram_s, 2),
                dram_util_pct        = dram_util,
                dram_hicache_staging_gb = dram_hicache_gb,
                turn_timeline        = turn_timeline,
                resolved        = resolved,
                resolved_in_loop = _resolved_in_loop,
                agent_steps_taken = _agent_steps_taken,
                patch_generated = bool(clean_patch.strip()),
                patch_applied   = applied,
                patch_complete  = is_complete,
                tests_passed    = tp,
                tests_failed    = tf,
                # A3 OS/Block Layer
                read_bw_mb_mean    = io_m.get("read_bw_mb_mean",    0.0),
                write_bw_mb_mean   = io_m.get("write_bw_mb_mean",   0.0),
                read_iops_mean     = io_m.get("read_iops_mean",     0.0),
                write_iops_mean    = io_m.get("write_iops_mean",    0.0),
                r_await_ms_p99     = io_m.get("r_await_ms_p99",     0.0),
                r_await_ms_p999    = io_m.get("r_await_ms_p999",    0.0),
                w_await_ms_mean    = io_m.get("w_await_ms_mean",    0.0),
                avgqu_sz_mean      = io_m.get("avgqu_sz_mean",      0.0),
                util_pct_mean      = io_m.get("util_pct_mean",      0.0),
                bio_lat_p50_us     = bio_m.get("read_lat_p50_us",   0.0),
                bio_lat_p99_us     = bio_m.get("read_lat_p99_us",   0.0),
                bio_lat_p999_us    = bio_m.get("read_lat_p999_us",  0.0),
                # A2 io_uring/NVMe Driver (populated from run-level nvme_drv_m in summary)
                nvme_inflight_mean = nvme_drv_m.get("nvme_inflight_mean", 0.0),
                nvme_inflight_peak = nvme_drv_m.get("nvme_inflight_peak", 0),
                nvme_rd_lat_ms_sysfs = nvme_drv_m.get("nvme_rd_lat_ms_mean", 0.0),
                nvme_wr_lat_ms_sysfs = nvme_drv_m.get("nvme_wr_lat_ms_mean", 0.0),
                nvme_nr_requests   = nvme_drv_m.get("nvme_nr_requests", 0),
                nvme_scheduler     = nvme_drv_m.get("nvme_scheduler", ""),
                # A3 OS/Memory Manager
                page_faults_per_s      = vmstat_m.get("page_faults_per_s",      0.0),
                major_faults_per_s     = vmstat_m.get("major_faults_per_s",     0.0),
                swap_pages_total       = vmstat_m.get("swap_pages_total",        0),
                page_cache_reads_per_s = vmstat_m.get("page_cache_reads_per_s", 0.0),
                numa_migrations_per_s  = vmstat_m.get("numa_migrations_per_s",  0.0),
                tlb_remote_miss_per_s  = vmstat_m.get("tlb_remote_miss_per_s",  0.0),
                hugepages_used         = vmstat_m.get("hugepages_used",          0),
                # A2 GPU Driver
                nvlink_tx_gb_s        = nvlink_m.get("nvlink_tx_gb_s",       0.0),
                nvlink_rx_gb_s        = nvlink_m.get("nvlink_rx_gb_s",       0.0),
                pcie_tx_gb_s          = nvlink_m.get("pcie_tx_gb_s",         0.0),
                pcie_rx_gb_s          = nvlink_m.get("pcie_rx_gb_s",         0.0),
                pcie_link_gen         = nvlink_m.get("pcie_link_gen",        0),
                pcie_theoretical_gbps = nvlink_m.get("pcie_theoretical_gbps",0.0),
                # A1 CUDA kernel-level metrics
                cuda_sm_active_mean_pct     = cuda_m.get("sm_active_mean_pct",     0.0),
                cuda_sm_active_min_pct      = cuda_m.get("sm_active_min_pct",      0.0),
                cuda_tensor_active_mean_pct = cuda_m.get("tensor_active_mean_pct", 0.0),
                cuda_tensor_active_min_pct  = cuda_m.get("tensor_active_min_pct",  0.0),
                cuda_dram_active_mean_pct   = cuda_m.get("dram_active_mean_pct",   0.0),
                cuda_hbm_bw_read_gb_s       = cuda_m.get("hbm_bw_read_gb_s_mean", 0.0),
                cuda_hbm_bw_write_gb_s      = cuda_m.get("hbm_bw_write_gb_s_mean",0.0),
                cuda_sm_clock_mhz           = cuda_m.get("sm_clock_mhz_mean",      0.0),
                cuda_sm_occupancy_mean_pct  = cuda_m.get("sm_occupancy_mean_pct",  0.0),
                cuda_fp16_active_mean_pct   = cuda_m.get("fp16_active_mean_pct",   0.0),
                cuda_throttled_pct          = cuda_m.get("throttled_pct",          0.0),
                cuda_source                 = cuda_m.get("cuda_source",            ""),
                # A5 Application Layer — req latency (single instance = same as total_time_s)
                req_lat_p99_ms  = round(turn_result["total_duration_s"] * 1000, 1),
                req_lat_p999_ms = round(turn_result["total_duration_s"] * 1000, 1),
                # ── Deep profiler metrics (per-instance) ──────────────────────
                # perf stat — L3 miss rate
                perf_available         = perf_m.get("perf_available",         False),
                perf_l3_miss_count     = perf_m.get("perf_l3_miss_count",     0),
                perf_all_loads_count   = perf_m.get("perf_all_loads_count",   0),
                perf_l3_miss_rate_pct  = perf_m.get("perf_l3_miss_rate_pct",  0.0),
                perf_l3_miss_per_s     = perf_m.get("perf_l3_miss_per_s",     0.0),
                # pcm-memory — socket DRAM BW
                pcm_available          = pcm_m.get("pcm_available",           False),
                pcm_dram_read_gb_s     = pcm_m.get("pcm_dram_read_gb_s",      0.0),
                pcm_dram_write_gb_s    = pcm_m.get("pcm_dram_write_gb_s",     0.0),
                pcm_dram_total_gb_s    = pcm_m.get("pcm_dram_total_gb_s",     0.0),
                pcm_dram_read_gb_s_peak= pcm_m.get("pcm_dram_read_gb_s_peak", 0.0),
                pcm_source             = pcm_m.get("pcm_source",              ""),
                # bpftrace — page faults, mmap, malloc, I/O latency
                bpf_available          = bpf_m.get("bpf_available",           False),
                bpf_page_faults_total  = bpf_m.get("bpf_page_faults_total",   0),
                bpf_major_faults_total = bpf_m.get("bpf_major_faults_total",  0),
                bpf_mmap_calls         = bpf_m.get("bpf_mmap_calls",          0),
                bpf_mmap_bytes_gb      = bpf_m.get("bpf_mmap_bytes_gb",       0.0),
                bpf_malloc_calls       = bpf_m.get("bpf_malloc_calls",        0),
                bpf_malloc_bytes_gb    = bpf_m.get("bpf_malloc_bytes_gb",     0.0),
                bpf_read_lat_p99_us    = bpf_m.get("bpf_read_lat_p99_us",     0.0),
                bpf_write_lat_p99_us   = bpf_m.get("bpf_write_lat_p99_us",    0.0),
                raw_sglang_metrics_json     = json.dumps(sg_m, sort_keys=True),
                raw_iostat_metrics_json     = json.dumps(io_m, sort_keys=True),
                raw_biolat_metrics_json     = json.dumps(bio_m, sort_keys=True),
                raw_vmstat_metrics_json     = json.dumps(vmstat_m, sort_keys=True),
                raw_nvlink_metrics_json     = json.dumps(nvlink_m, sort_keys=True),
                raw_nvme_driver_metrics_json= json.dumps(nvme_drv_m, sort_keys=True),
                raw_cuda_metrics_json       = json.dumps(cuda_m, sort_keys=True),
                raw_perf_metrics_json       = json.dumps(perf_m, sort_keys=True),
                raw_pcm_metrics_json        = json.dumps(pcm_m, sort_keys=True),
                raw_bpf_metrics_json        = json.dumps(bpf_m, sort_keys=True),
                raw_traj_stats_json        = str(turn_result.get("raw_traj_stats_json", "") or ""),
                success = True,
                notes   = (
                    f"turns={completed_turns}  "
                    f"patch={'yes' if applied else 'no'}  "
                    f"complete={'yes' if is_complete else 'truncated'}  "
                    f"code={'yes' if has_code else 'no'}  "
                    f"docker={'yes' if self.use_docker else 'no'}"
                ),
            )
            (work_dir / "result.json").write_text(json.dumps(r.to_dict(), indent=2))

            trunc = "" if is_complete else " [TRUNCATED]"
            log.info(
                f"  {'ok' if resolved else ('~' if applied else 'x')}  "
                f"prompt={pt:,}tok  gen={ot:,}tok{trunc}  "
                f"{dur:.1f}s  TTFT={ttft_ms:.0f}ms  TPOT={tpot_mean:.1f}ms  "
                f"op={ai_op}  KV$={kv_hit:.0f}%  "
                f"HBM={hbm_s:.0f}->{hbm_e:.0f}GB(d{r.hbm_delta_gb:+.1f})  "
                f"DRAM={dram_s:.0f}->{dram_e:.0f}GB(d{r.dram_delta_gb:+.1f})")
            return r

        # ── Dispatch: sequential (concurrency=1) or concurrent ────────────────
        # Why concurrency matters for SSD I/O:
        #   KV$ per request at 32K ctx ~ 5 GB
        #   HBM KV pool on 8xA100 40GB  ~ 256 GB
        #   Need 52 concurrent requests to overflow -> HiCache writes to SSD
        #   concurrency=16  -> ~80 GB KV$ in flight  (partial pressure)
        #   concurrency=32  -> ~160 GB KV$ in flight (strong pressure)
        #   concurrency=52  -> ~260 GB KV$ in flight (overflow -> SSD writes)
        concurrency = max(1, self.inference_concurrency)
        if concurrency > 1:
            log.info(
                f"inference_concurrency={concurrency} -- "
                f"~{concurrency * 5} GB KV$ in HBM simultaneously. "
                f"HiCache eviction starts when pool exceeds ~256 GB.")

        try:
            if concurrency == 1:
                for inst in instances:
                    results.append(_process_one(inst))
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = {pool.submit(_process_one, inst): inst
                            for inst in instances}
                    for fut in as_completed(futs):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            iid = futs[fut].get("instance_id", "?")
                            log.error(f"Instance {iid} failed: {e}")
        finally:
            smart_m   = smart.stop()
            power_m   = power.stop()
            ssd_hw_m  = ssd_hw.stop()
            req_lat_m = req_lat.summarise()
            # Stop run-level deep profilers (attached to server PID for full run)
            ncu_m   = ncu_col.stop()   if _server_pid else {}
            nsys_m  = nsys_col.stop()  if _server_pid else {}
            vtune_m = vtune_col.stop() if _server_pid else {}
            torch_m = torch_col.stop() if _server_pid else {}
            if server:
                server.stop()

        if results:
            # These are run-level (not per-instance) metrics from monitors that
            # span the whole harness run.  Stamp every result so that:
            #   - concurrent runs (ThreadPoolExecutor) don't lose data when
            #     results[-1] happens to be a non-deterministic completion order
            #   - sweep aggregation (mn/pk) works correctly across all instances
            _run_level = dict(
                waf              = smart_m.get("waf",              0.0),
                host_written_gb  = smart_m.get("host_written_gb",  0.0),
                temp_peak_c      = smart_m.get("temp_peak_c",      0),
                nand_written_gb  = ssd_hw_m.get("ssd_nand_written_gb",  0.0),
                ssd_lifetime_tbw = ssd_hw_m.get("ssd_lifetime_tbw",     0.0),
                ssd_dwpd_est     = ssd_hw_m.get("ssd_dwpd_est",         0.0),
                ssd_media_errors = ssd_hw_m.get("ssd_media_errors",     0),
                hicache_size_gb  = ssd_hw_m.get("hicache_size_gb",      0.0),
                hicache_file_count = ssd_hw_m.get("hicache_file_count", 0),
                # ── Run-level deep profiler metrics ───────────────────────────
                # ncu: attention kernel roofline (one capture per run)
                ncu_attention_available    = ncu_m.get("ncu_attention_available",    False),
                ncu_attention_kernel_count = ncu_m.get("ncu_attention_kernel_count", 0),
                ncu_dram_read_gb           = ncu_m.get("ncu_dram_read_gb",           0.0),
                ncu_dram_write_gb          = ncu_m.get("ncu_dram_write_gb",          0.0),
                ncu_l2_hit_rate_pct        = ncu_m.get("ncu_l2_hit_rate_pct",        0.0),
                ncu_arith_intensity        = ncu_m.get("ncu_arith_intensity",        0.0),
                ncu_sm_eff_pct             = ncu_m.get("ncu_sm_eff_pct",             0.0),
                ncu_duration_us_mean       = ncu_m.get("ncu_duration_us_mean",       0.0),
                ncu_duration_us_p99        = ncu_m.get("ncu_duration_us_p99",        0.0),
                # nsys: CUDA timeline
                nsys_available             = nsys_m.get("nsys_available",            False),
                nsys_cuda_api_calls        = nsys_m.get("nsys_cuda_api_calls",       0),
                nsys_kernel_count          = nsys_m.get("nsys_kernel_count",         0),
                nsys_memcpy_h2d_gb         = nsys_m.get("nsys_memcpy_h2d_gb",        0.0),
                nsys_memcpy_d2h_gb         = nsys_m.get("nsys_memcpy_d2h_gb",        0.0),
                nsys_gpu_active_pct        = nsys_m.get("nsys_gpu_active_pct",       0.0),
                nsys_kernel_top5           = nsys_m.get("nsys_kernel_top5",          ""),
                # vtune: DRAM BW + NUMA locality
                vtune_available            = vtune_m.get("vtune_available",          False),
                vtune_dram_bw_gb_s         = vtune_m.get("vtune_dram_bw_gb_s",       0.0),
                vtune_numa_local_access_pct= vtune_m.get("vtune_numa_local_access_pct", 0.0),
                vtune_l3_bound_pct         = vtune_m.get("vtune_l3_bound_pct",       0.0),
                vtune_mem_bound_pct        = vtune_m.get("vtune_mem_bound_pct",      0.0),
                vtune_ipc                  = vtune_m.get("vtune_ipc",               0.0),
                vtune_hotspot_fn           = vtune_m.get("vtune_hotspot_fn",         ""),
                # torch profiler: operator-level CUDA time
                torch_prof_available       = torch_m.get("torch_prof_available",     False),
                torch_prof_cuda_time_ms    = torch_m.get("torch_prof_cuda_time_ms",  0.0),
                torch_prof_cpu_time_ms     = torch_m.get("torch_prof_cpu_time_ms",   0.0),
                torch_prof_memory_alloc_mb = torch_m.get("torch_prof_memory_alloc_mb", 0.0),
                torch_prof_kernel_count    = torch_m.get("torch_prof_kernel_count",  0),
                torch_prof_top_ops         = torch_m.get("torch_prof_top_ops",       ""),
            )
            for r in results:
                for k, v in _run_level.items():
                    setattr(r, k, v)
            # A5 request latency P99/P999 across all instances
            for r in results:
                r.req_lat_p99_ms  = req_lat_m.get("req_lat_p99_ms",  0.0)
                r.req_lat_p999_ms = req_lat_m.get("req_lat_p999_ms", 0.0)

        n     = len(results)
        n_res = sum(r.resolved        for r in results)
        n_p   = sum(r.patch_generated for r in results)
        n_c   = sum(r.patch_complete  for r in results)
        n_cd  = sum(r.prompt_has_code for r in results)

        summary = {
            "run_id":              self.run_id,
            "model":               p.model.alias,
            "split":               self.split,
            "total_instances":     n,
            "resolved":            n_res,
            "resolved_pct":        round(n_res/max(n,1)*100, 2),
            "patch_generated_pct": round(n_p  /max(n,1)*100, 2),
            "patch_complete_pct":  round(n_c  /max(n,1)*100, 2),
            "prompt_had_code_pct": round(n_cd /max(n,1)*100, 2),
            "avg_ttft_ms":         round(sum(r.ttft_ms      for r in results)/max(n,1), 1),
            "avg_tpot_ms":         round(sum(r.tpot_mean_ms for r in results)/max(n,1), 1),
            "waf":                 smart_m.get("waf", 0.0),
            "host_written_gb":     smart_m.get("host_written_gb", 0.0),
            "temp_peak_c":         smart_m.get("temp_peak_c", 0),
            "total_energy_wh":     power_m.get("total_system_energy_wh", 0.0),
        }
        log.info(
            f"\nSWE-bench {self.split}:  resolved={n_res}/{n}  "
            f"complete_patches={n_c}/{n}  with_code={n_cd}/{n}  "
            f"avg_TTFT={summary['avg_ttft_ms']}ms  "
            f"avg_TPOT={summary['avg_tpot_ms']}ms")

        (self.output_dir / "swebench_summary.json").write_text(
            json.dumps(summary, indent=2))
        return results


# ── Concurrency sweep ─────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    """One row in the concurrency sweep results CSV."""
    inference_concurrency: int   = 0
    # Inference latency
    ttft_mean_ms:          float = 0.0
    ttft_p99_ms:           float = 0.0
    tpot_mean_ms:          float = 0.0
    tpot_p99_ms:           float = 0.0
    # Throughput
    throughput_req_s:      float = 0.0
    throughput_tok_s:      float = 0.0
    # Token counts (mean per instance)
    prompt_tokens_mean:    float = 0.0
    output_tokens_mean:    float = 0.0
    # HBM
    hbm_used_gb_mean:      float = 0.0
    hbm_util_pct_mean:     float = 0.0
    hbm_delta_gb_mean:     float = 0.0
    # DRAM
    dram_used_gb_mean:     float = 0.0
    dram_util_pct_mean:    float = 0.0
    dram_delta_gb_mean:    float = 0.0
    # SSD
    ssd_read_bw_mb_mean:   float = 0.0
    ssd_write_bw_mb_mean:  float = 0.0
    ssd_read_iops_mean:    float = 0.0
    ssd_r_await_p99_ms:    float = 0.0
    ssd_util_pct_mean:     float = 0.0
    # SGLang
    kv_cache_hit_rate_pct: float = 0.0
    ai_op_decode_tok_s:    float = 0.0
    # Outcomes
    instances_run:         int   = 0
    patch_generated_pct:   float = 0.0
    # KV$ pool semantics (derived from model architecture + HBM config)
    kv_pool_capacity_gb:   float = 0.0   # HBM allocated to KV cache (GB)
    kv_weights_gb:         float = 0.0   # model weights footprint (GB)
    kv_per_request_gb:     float = 0.0   # KV$ consumed per active request (GB)
    kv_fill_pct_mean:      float = 0.0   # mean KV pool fill % (from token_usage)
    kv_fill_pct_peak:      float = 0.0   # peak KV pool fill % across instances
    kv_overflow_at:        int   = 0     # concurrency where eviction starts
    kv_eviction_mb_s:      float = 0.0   # KV$ eviction rate to SSD = ssd_write_bw
    kv_restore_mb_s:       float = 0.0   # KV$ restore rate from SSD = ssd_read_bw
    kv_miss_penalty_ms:    float = 0.0   # TPOT overhead vs baseline (concurrency=1)
    kv_tokens_capacity:    int   = 0     # max tokens pool can hold
    kv_bytes_per_token:    int   = 0     # bytes per token in KV cache

    # ── A5 Application ────────────────────────────────────────────────────────
    total_time_s_mean:     float = 0.0
    tok_per_s_mean:        float = 0.0
    req_lat_p99_ms:        float = 0.0
    req_lat_p999_ms:       float = 0.0
    resolved_pct:          float = 0.0

    # ── HBM extended ──────────────────────────────────────────────────────────
    hbm_peak_gb_mean:             float = 0.0
    hbm_prefill_delta_gb_mean:    float = 0.0
    hbm_decode_delta_gb_mean:     float = 0.0
    hbm_kv_evicted_gb_mean:       float = 0.0

    # ── DRAM extended ─────────────────────────────────────────────────────────
    dram_hicache_staging_gb_mean: float = 0.0

    # ── SSD extended ──────────────────────────────────────────────────────────
    ssd_write_iops_mean:   float = 0.0
    ssd_w_await_ms_mean:   float = 0.0
    ssd_avgqu_sz_mean:     float = 0.0
    bio_lat_p50_us_mean:   float = 0.0
    bio_lat_p99_us_mean:   float = 0.0
    bio_lat_p999_us_mean:  float = 0.0

    # ── AI op classification & phase distribution ────────────────────────────
    # Counts of instances by dominant AI op type
    op_prefill_count:      int   = 0   # instances where prefill dominated
    op_decode_count:       int   = 0   # instances where decode dominated
    op_reasoning_count:    int   = 0   # instances where R1/QwQ reasoning dominated
    op_mixed_count:        int   = 0   # instances with mixed prefill+decode

    # ── Prefill phase metrics (KV$ write path) ────────────────────────────────
    # Prefill: reads input tokens → computes KV$ → writes KV$ to HBM
    #   SSD write = eviction when HBM pool overflows during prefill
    pf_hbm_delta_gb_mean:        float = 0.0  # mean HBM increase per prefill (KV$ written)
    pf_rt_compute_tokens_mean:   float = 0.0  # mean tokens that required new KV$ computation
    pf_rt_cache_tokens_mean:     float = 0.0  # mean tokens served from existing KV$ (cache hit)
    pf_cache_hit_pct:            float = 0.0  # prefill cache hit rate (rt_cache / total_prefill)
    pf_ssd_eviction_gb_mean:     float = 0.0  # mean GB evicted to SSD triggered by prefill overflow
    pf_ssd_eviction_bw_mb_s:     float = 0.0  # write BW at instances where prefill dominated

    # ── Decode phase metrics (KV$ read path) ──────────────────────────────────
    # Decode: each output token re-reads ENTIRE context KV$ for attention
    #   SSD read = restore when KV$ block was evicted during earlier prefill
    dc_hbm_delta_gb_mean:        float = 0.0  # mean HBM change during decode (should be ~0)
    dc_rt_decode_tokens_mean:    float = 0.0  # mean decode tokens generated
    dc_kv_restored_tokens_mean:  float = 0.0  # mean KV$ tokens restored from SSD per instance
    dc_ssd_restore_gb_mean:      float = 0.0  # mean GB restored from SSD triggered by decode
    dc_ssd_restore_bw_mb_s:      float = 0.0  # read BW at instances where decode dominated
    dc_miss_penalty_ms:          float = 0.0  # mean TPOT overhead from SSD KV$ restores
    dc_kv_read_per_step_gb:      float = 0.0  # theoretical KV$ read per decode step (ctx × bpt)

    # ── AI op throughput ───────────────────────────────────────────────────────
    op_prefill_tok_s_mean:       float = 0.0  # mean prefill throughput (tok/s)
    op_decode_tok_s_mean:        float = 0.0  # mean decode throughput (tok/s)
    op_prefill_to_decode_ratio:  float = 0.0  # prefill_tok_s / decode_tok_s (>1 = prefill-heavy)

    # ── SGLang extended ───────────────────────────────────────────────────────
    ai_op_prefill_tok_s:          float = 0.0
    cache_hit_rate_realtime_pct:  float = 0.0
    kv_l1_device_tokens_mean:     float = 0.0
    kv_l2_host_tokens_mean:       float = 0.0
    kv_l3_storage_tokens_mean:    float = 0.0
    kv_evicted_tokens_mean:       float = 0.0
    kv_restored_tokens_mean:      float = 0.0
    server_ttft_ms_mean:          float = 0.0
    server_itl_ms_mean:           float = 0.0
    hicache_eviction_ms_mean:     float = 0.0
    hicache_load_back_ms_mean:    float = 0.0
    token_usage_peak_mean:        float = 0.0
    num_queue_reqs_peak_mean:     float = 0.0
    utilization_mean:             float = 0.0

    # ── A2 NVMe Driver ────────────────────────────────────────────────────────
    nvme_inflight_mean:    float = 0.0
    nvme_inflight_peak:    int   = 0
    nvme_rd_lat_ms_sysfs:  float = 0.0

    # ── A3 OS / vmstat ────────────────────────────────────────────────────────
    page_faults_per_s:     float = 0.0
    major_faults_per_s:    float = 0.0
    numa_migrations_per_s: float = 0.0
    hugepages_used:        int   = 0

    # ── A2 GPU Driver / NVLink ────────────────────────────────────────────────
    nvlink_tx_gb_s:        float = 0.0
    nvlink_rx_gb_s:        float = 0.0
    pcie_tx_gb_s:          float = 0.0
    pcie_rx_gb_s:          float = 0.0

    # ── Endurance ─────────────────────────────────────────────────────────────
    waf:                   float = 0.0
    host_written_gb:       float = 0.0
    nand_written_gb:       float = 0.0
    ssd_lifetime_tbw:      float = 0.0
    ssd_dwpd_est:          float = 0.0
    temp_peak_c:           int   = 0

    # ── A1 CUDA kernel-level ─────────────────────────────────────────────────
    cuda_sm_active_mean_pct:     float = 0.0
    cuda_sm_active_min_pct:      float = 0.0
    cuda_tensor_active_mean_pct: float = 0.0
    cuda_tensor_active_min_pct:  float = 0.0
    cuda_dram_active_mean_pct:   float = 0.0
    cuda_hbm_bw_read_gb_s:       float = 0.0
    cuda_hbm_bw_write_gb_s:      float = 0.0
    cuda_sm_clock_mhz:           float = 0.0
    cuda_sm_occupancy_mean_pct:  float = 0.0
    cuda_fp16_active_mean_pct:   float = 0.0
    cuda_throttled_pct:          float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _ai_op_phase_fields(results: list, kv_bpt: int) -> dict:
    """
    Compute AI op phase breakdown fields from a list of SWEBenchResult.

    Prefill = KV$ write path:  input tokens → compute attention → write KV$ to HBM
      SSD impact: if HBM pool overflows during prefill, cold blocks are evicted (writes)
      Metric: rt_prefill_compute_tokens  → KV$ bytes written = tokens × kv_bytes_per_token
              kv_evicted_tokens          → tokens evicted to SSD during this instance

    Decode = KV$ read path: each output token re-reads ENTIRE context KV$ for attention
      SSD impact: if KV$ blocks were evicted, decode stalls to restore them (reads)
      Metric: rt_decode_tokens           → decode steps taken
              kv_restored_tokens         → tokens restored from SSD (each restore = SSD read)
    """
    n = max(len(results), 1)

    def mn(attr):
        return round(sum(getattr(r, attr, 0) or 0 for r in results) / n, 3)

    # Op type distribution
    op_counts = {"prefill": 0, "decode": 0, "reasoning": 0, "mixed": 0, "idle": 0}
    for r in results:
        raw_op = getattr(r, "ai_op_type", "") or ""
        key    = raw_op.split("_")[0] if "_" in raw_op else raw_op
        if key not in op_counts:
            key = "mixed"   # unknown / empty type → mixed
        op_counts[key] = op_counts[key] + 1

    # Prefill phase
    pf_compute = mn("rt_prefill_compute_tokens")
    pf_cache   = mn("rt_prefill_cache_tokens")
    pf_total   = pf_compute + pf_cache
    pf_hit_pct = round(pf_cache / pf_total * 100, 1) if pf_total > 0 else 0.0
    pf_evicted_gb = round(mn("kv_evicted_tokens") * kv_bpt / (1024**3), 4)
    pf_hbm_delta  = mn("hbm_prefill_delta_gb")

    # Decode phase
    dc_tokens    = mn("rt_decode_tokens")
    dc_restored  = mn("kv_restored_tokens")
    dc_restore_gb= round(dc_restored * kv_bpt / (1024**3), 4)
    dc_hbm_delta = mn("hbm_decode_delta_gb")

    # Theoretical KV$ read per decode step = context_len × kv_bytes_per_token.
    # prompt_tokens=1 occurs when GITHUB_TOKEN is unset (minimal fallback prompt).
    # In that case, use hbm_prefill_delta (≈ KV$ written = ctx × kv_bpt) to
    # back-calculate context_len. Floor at 4096 to avoid divide-by-near-zero.
    pf_delta_gb = mn("hbm_prefill_delta_gb")
    pt          = int(mn("prompt_tokens"))
    if pf_delta_gb > 0.1:
        approx_ctx = max(4096, int(pf_delta_gb * (1024**3) / max(kv_bpt, 1)))
    else:
        approx_ctx = max(4096, pt or 32768)
    dc_read_per_step_gb = round(approx_ctx * kv_bpt / (1024**3), 3)

    # Phase-split BW (instances dominated by prefill vs decode)
    pf_results = [r for r in results if "prefill" in getattr(r, "ai_op_type", "")]
    dc_results = [r for r in results if "decode" in getattr(r, "ai_op_type", "")
                  or "reasoning" in getattr(r, "ai_op_type", "")]
    def _mn_subset(lst, attr):
        return round(sum(getattr(r, attr, 0) or 0 for r in lst) / max(len(lst), 1), 3)

    pf_tok_s = mn("ai_op_prefill_tok_s")
    dc_tok_s = mn("ai_op_decode_tok_s")

    return {
        "op_prefill_count":           op_counts.get("prefill", 0),
        "op_decode_count":            op_counts.get("decode", 0),
        "op_reasoning_count":         op_counts.get("reasoning", 0),
        "op_mixed_count":             op_counts.get("mixed", 0),
        "pf_hbm_delta_gb_mean":       pf_hbm_delta,
        "pf_rt_compute_tokens_mean":  pf_compute,
        "pf_rt_cache_tokens_mean":    pf_cache,
        "pf_cache_hit_pct":           pf_hit_pct,
        "pf_ssd_eviction_gb_mean":    pf_evicted_gb,
        "pf_ssd_eviction_bw_mb_s":    _mn_subset(pf_results, "write_bw_mb_mean") if pf_results else 0.0,
        "dc_hbm_delta_gb_mean":       dc_hbm_delta,
        "dc_rt_decode_tokens_mean":   dc_tokens,
        "dc_kv_restored_tokens_mean": dc_restored,
        "dc_ssd_restore_gb_mean":     dc_restore_gb,
        "dc_ssd_restore_bw_mb_s":     _mn_subset(dc_results, "read_bw_mb_mean")  if dc_results else 0.0,
        "dc_miss_penalty_ms":         mn("kv_miss_penalty_ms") if hasattr(results[0] if results else object(), "kv_miss_penalty_ms") else 0.0,
        "dc_kv_read_per_step_gb":     dc_read_per_step_gb,
        "op_prefill_tok_s_mean":      pf_tok_s,
        "op_decode_tok_s_mean":       dc_tok_s,
        "op_prefill_to_decode_ratio": round(pf_tok_s / dc_tok_s, 3) if dc_tok_s > 0 else 0.0,
    }


def run_concurrency_sweep(
    plan,
    run_id: str,
    output_dir: Path,
    concurrency_levels: list[int],
    instances_per_level: int = 20,
    server_port: int = 30000,
    use_docker: bool = False,
    max_tokens: int = 8192,
    call_timeout: int = 900,
    workers: int = 1,
    context_len: int = 65536,
    mem_fraction: float = 0.80,
    split: str = "pro",
    agent_backend: str = "none",
    max_agent_steps: int = 30,
    instances_path: Optional[str] = None,
    sweagent_default_config: Optional[str] = None,
    sweagent_config_path: Optional[str] = None,
    sweagent_model_name: Optional[str] = None,
    sweagent_model_api_base: Optional[str] = None,
    sweagent_model_api_key: str = "EMPTY",
    sweagent_max_input_tokens: int = 50000,
    sweagent_num_workers: int = 1,
    sweagent_instances_type: str = "file",
    sweagent_redo_existing: bool = False,
    sweagent_shuffle: bool = False,
) -> list[SweepPoint]:
    """
    Run SWE-bench at multiple inference_concurrency levels and collect
    per-level aggregate metrics for comparison.

    For each concurrency level:
      - Sends `instances_per_level` SWE-bench instances concurrently
      - Aggregates TTFT, TPOT, throughput, HBM, DRAM, SSD metrics
      - Writes per-level CSV row

    This directly answers: how does concurrency affect each memory tier?

    Args:
        concurrency_levels:  list of concurrency values to sweep, e.g. [1,4,8,16,32,52]
        instances_per_level: instances to run at each level (20 is enough for stable stats)
    """
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results: list[SweepPoint] = []

    # ── KV$ pool stats (fixed for this run) ─────────────────────────────────
    kv_stats = plan.model.kv_pool_stats(
        hbm_cap_gb   = plan.hbm_cap_gb,
        context_len  = context_len,
        mem_fraction = mem_fraction,
    )
    log.info(
        f"KV$ pool: capacity={kv_stats['pool_capacity_gb']:.0f} GB  "
        f"weights={kv_stats['weights_gb']:.0f} GB  "
        f"per_request={kv_stats['kv_per_request_gb']:.1f} GB  "
        f"overflow_at=concurrency≥{kv_stats['overflow_concurrency']}")
    log.info(
        f"Concurrency sweep: levels={concurrency_levels}  "
        f"instances_per_level={instances_per_level}")

    baseline_tpot: float = 0.0   # TPOT at concurrency=1, for miss penalty calc

    for concurrency in concurrency_levels:
        log.info(f"\n{'='*60}")
        log.info(f"  Concurrency level: {concurrency}")
        log.info(f"  Expected KV$ in HBM: ~{concurrency * 10} GB at 65K ctx")
        spill = "YES" if concurrency >= 26 else "no"
        log.info(f"  HiCache SSD eviction: {spill}")
        log.info(f"{'='*60}")

        level_dir = output_dir / f"concurrency_{concurrency:03d}"
        harness = SWEBenchHarness(
            plan                  = plan,
            run_id                = f"{run_id}_c{concurrency}",
            output_dir            = level_dir,
            split                 = split,
            max_instances         = instances_per_level,
            server_port           = server_port,
            use_docker            = use_docker,
            max_tokens            = max_tokens,
            call_timeout          = call_timeout,
            workers               = workers,
            inference_concurrency = concurrency,
            agent_backend         = agent_backend,
            max_agent_steps       = max_agent_steps,
            context_len           = context_len,
            instances_path        = instances_path,
            sweagent_default_config = sweagent_default_config,
            sweagent_config_path  = sweagent_config_path,
            sweagent_model_name   = sweagent_model_name,
            sweagent_model_api_base = sweagent_model_api_base,
            sweagent_model_api_key = sweagent_model_api_key,
            sweagent_max_input_tokens = sweagent_max_input_tokens,
            sweagent_num_workers  = sweagent_num_workers,
            sweagent_instances_type = sweagent_instances_type,
            sweagent_redo_existing = sweagent_redo_existing,
            sweagent_shuffle      = sweagent_shuffle,
        )

        results = harness.run_all()
        if not results:
            log.warning(f"No results at concurrency={concurrency}, skipping")
            continue

        n = len(results)
        def mn(attr): return round(sum(getattr(r, attr) for r in results) / max(n, 1), 3)
        def pk(attr): return round(max(getattr(r, attr) for r in results), 3)

        # Throughput: instances per second over wall-clock span
        total_time = sum(r.total_time_s for r in results)
        # Effective wall time = total_time / concurrency (parallel execution)
        wall_time  = max(total_time / max(concurrency, 1), 0.001)
        req_per_s  = round(n / wall_time, 3)
        tok_per_s  = round(sum(r.output_tokens for r in results) / max(wall_time, 0.001), 1)

        # KV$ fill %:
        #   Prefer token_usage_peak from SGLang /metrics — it is the actual fraction
        #   of the KV token pool that is occupied (0.0–1.0). This is accurate even
        #   though SGLang pre-allocates the full HBM pool at startup (so hbm_used_gb
        #   is constant at ~weights+pool regardless of how many requests are in flight).
        #   Fall back to HBM-based estimate when SGLang metrics are unavailable.
        pool_gb    = kv_stats["pool_capacity_gb"]
        weights_gb = kv_stats["weights_gb"]
        tu_mean = mn("token_usage_peak")
        tu_peak = pk("token_usage_peak")
        if tu_mean > 0:
            fill_mean = round(tu_mean * 100, 1)
            fill_peak = round(tu_peak * 100, 1)
        else:
            kv_used_mean = max(0.0, mn("hbm_used_gb_end") - weights_gb)
            kv_used_peak = max(0.0, pk("hbm_used_gb_end") - weights_gb)
            fill_mean = round(kv_used_mean / max(pool_gb, 0.001) * 100, 1)
            fill_peak = round(kv_used_peak / max(pool_gb, 0.001) * 100, 1)

        tpot_now = mn("tpot_mean_ms")
        if concurrency == 1 or baseline_tpot == 0:
            baseline_tpot = tpot_now
        miss_penalty = round(max(0.0, tpot_now - baseline_tpot), 2)

        sp = SweepPoint(
            inference_concurrency = concurrency,
            ttft_mean_ms          = mn("ttft_ms"),
            ttft_p99_ms           = pk("ttft_ms"),
            tpot_mean_ms          = tpot_now,
            tpot_p99_ms           = pk("tpot_p99_ms"),
            throughput_req_s      = req_per_s,
            throughput_tok_s      = tok_per_s,
            prompt_tokens_mean    = mn("prompt_tokens"),
            output_tokens_mean    = mn("output_tokens"),
            hbm_used_gb_mean      = mn("hbm_used_gb_end"),
            hbm_util_pct_mean     = mn("hbm_util_pct"),
            hbm_delta_gb_mean     = mn("hbm_delta_gb"),
            dram_used_gb_mean     = mn("dram_used_gb_end"),
            dram_util_pct_mean    = mn("dram_util_pct"),
            dram_delta_gb_mean    = mn("dram_delta_gb"),
            ssd_read_bw_mb_mean   = mn("read_bw_mb_mean"),
            ssd_write_bw_mb_mean  = mn("write_bw_mb_mean"),
            ssd_read_iops_mean    = mn("read_iops_mean"),
            ssd_r_await_p99_ms    = pk("r_await_ms_p99"),
            ssd_util_pct_mean     = mn("util_pct_mean"),
            kv_cache_hit_rate_pct = mn("kv_cache_hit_rate_pct"),
            ai_op_decode_tok_s    = mn("ai_op_decode_tok_s"),
            instances_run         = n,
            patch_generated_pct   = round(
                sum(r.patch_generated for r in results) / max(n, 1) * 100, 1),
            # KV$ pool semantics
            kv_pool_capacity_gb  = kv_stats["pool_capacity_gb"],
            kv_weights_gb        = kv_stats["weights_gb"],
            kv_per_request_gb    = kv_stats["kv_per_request_gb"],
            kv_fill_pct_mean     = fill_mean,
            kv_fill_pct_peak     = fill_peak,
            kv_overflow_at       = kv_stats["overflow_concurrency"],
            kv_eviction_mb_s     = mn("write_bw_mb_mean"),   # same as SSD write BW
            kv_restore_mb_s      = mn("read_bw_mb_mean"),    # same as SSD read BW
            kv_miss_penalty_ms   = miss_penalty,
            kv_tokens_capacity   = kv_stats["token_capacity"],
            kv_bytes_per_token   = kv_stats["kv_bytes_per_token"],
            total_time_s_mean    = mn("total_time_s"),
            tok_per_s_mean       = mn("tok_per_s"),
            req_lat_p99_ms       = pk("req_lat_p99_ms"),
            req_lat_p999_ms      = pk("req_lat_p999_ms"),
            resolved_pct         = round(sum(r.resolved for r in results) / max(n, 1) * 100, 1),
            hbm_peak_gb_mean             = mn("hbm_peak_gb"),
            hbm_prefill_delta_gb_mean    = mn("hbm_prefill_delta_gb"),
            hbm_decode_delta_gb_mean     = mn("hbm_decode_delta_gb"),
            hbm_kv_evicted_gb_mean       = mn("hbm_kv_evicted_gb"),
            dram_hicache_staging_gb_mean = mn("dram_hicache_staging_gb"),
            ssd_write_iops_mean   = mn("write_iops_mean"),
            ssd_w_await_ms_mean   = mn("w_await_ms_mean"),
            ssd_avgqu_sz_mean     = mn("avgqu_sz_mean"),
            bio_lat_p50_us_mean   = mn("bio_lat_p50_us"),
            bio_lat_p99_us_mean   = mn("bio_lat_p99_us"),
            bio_lat_p999_us_mean  = mn("bio_lat_p999_us"),
            ai_op_prefill_tok_s   = mn("ai_op_prefill_tok_s"),
            cache_hit_rate_realtime_pct = mn("cache_hit_rate_realtime_pct"),
            kv_l1_device_tokens_mean = mn("kv_l1_device_tokens"),
            kv_l2_host_tokens_mean   = mn("kv_l2_host_tokens"),
            kv_l3_storage_tokens_mean= mn("kv_l3_storage_tokens"),
            kv_evicted_tokens_mean   = mn("kv_evicted_tokens"),
            kv_restored_tokens_mean  = mn("kv_restored_tokens"),
            server_ttft_ms_mean      = mn("server_ttft_ms"),
            server_itl_ms_mean       = mn("server_itl_ms"),
            hicache_eviction_ms_mean = mn("hicache_eviction_ms"),
            hicache_load_back_ms_mean= mn("hicache_load_back_ms"),
            token_usage_peak_mean    = mn("token_usage_peak"),
            num_queue_reqs_peak_mean = mn("num_queue_reqs_peak"),
            utilization_mean         = mn("utilization_mean"),
            nvme_inflight_mean       = mn("nvme_inflight_mean"),
            nvme_inflight_peak       = int(max(getattr(r, "nvme_inflight_peak", 0) for r in results)),
            nvme_rd_lat_ms_sysfs     = mn("nvme_rd_lat_ms_sysfs"),
            page_faults_per_s        = mn("page_faults_per_s"),
            major_faults_per_s       = mn("major_faults_per_s"),
            numa_migrations_per_s    = mn("numa_migrations_per_s"),
            hugepages_used           = mn("hugepages_used"),
            nvlink_tx_gb_s           = mn("nvlink_tx_gb_s"),
            nvlink_rx_gb_s           = mn("nvlink_rx_gb_s"),
            pcie_tx_gb_s             = mn("pcie_tx_gb_s"),
            pcie_rx_gb_s             = mn("pcie_rx_gb_s"),
            waf             = results[-1].waf,
            host_written_gb = results[-1].host_written_gb,
            nand_written_gb = results[-1].nand_written_gb,
            ssd_lifetime_tbw = results[-1].ssd_lifetime_tbw,
            ssd_dwpd_est    = results[-1].ssd_dwpd_est,
            temp_peak_c     = results[-1].temp_peak_c,
            cuda_sm_active_mean_pct     = mn("cuda_sm_active_mean_pct"),
            cuda_sm_active_min_pct      = mn("cuda_sm_active_min_pct"),
            cuda_tensor_active_mean_pct = mn("cuda_tensor_active_mean_pct"),
            cuda_tensor_active_min_pct  = mn("cuda_tensor_active_min_pct"),
            cuda_dram_active_mean_pct   = mn("cuda_dram_active_mean_pct"),
            cuda_hbm_bw_read_gb_s       = mn("cuda_hbm_bw_read_gb_s"),
            cuda_hbm_bw_write_gb_s      = mn("cuda_hbm_bw_write_gb_s"),
            cuda_sm_clock_mhz           = mn("cuda_sm_clock_mhz"),
            cuda_sm_occupancy_mean_pct  = mn("cuda_sm_occupancy_mean_pct"),
            cuda_fp16_active_mean_pct   = mn("cuda_fp16_active_mean_pct"),
            cuda_throttled_pct          = mn("cuda_throttled_pct"),
            raw_instance_metrics_json   = json.dumps([r.to_dict() for r in results], sort_keys=True),
            raw_metric_vectors_json     = json.dumps({
                "ttft_ms": [r.ttft_ms for r in results],
                "tpot_mean_ms": [r.tpot_mean_ms for r in results],
                "prompt_tokens": [r.prompt_tokens for r in results],
                "output_tokens": [r.output_tokens for r in results],
                "hbm_used_gb_end": [r.hbm_used_gb_end for r in results],
                "dram_used_gb_end": [r.dram_used_gb_end for r in results],
                "read_bw_mb_mean": [r.read_bw_mb_mean for r in results],
                "write_bw_mb_mean": [r.write_bw_mb_mean for r in results],
                "perf_l3_miss_count": [r.perf_l3_miss_count for r in results],
                "pcm_dram_total_gb_s": [r.pcm_dram_total_gb_s for r in results],
                "bpf_page_faults_total": [r.bpf_page_faults_total for r in results],
            }, sort_keys=True),
            **_ai_op_phase_fields(results, kv_stats["kv_bytes_per_token"]),
        )
        sweep_results.append(sp)

        log.info(
            f"  concurrency={concurrency}  "
            f"TTFT={sp.ttft_mean_ms:.0f}ms  TPOT={sp.tpot_mean_ms:.1f}ms  "
            f"req/s={sp.throughput_req_s:.2f}  tok/s={sp.throughput_tok_s:.0f}  "
            f"HBM={sp.hbm_used_gb_mean:.0f}GB({sp.hbm_util_pct_mean:.0f}%)  "
            f"SSD_rBW={sp.ssd_read_bw_mb_mean:.1f}MB/s  "
            f"SSD_wBW={sp.ssd_write_bw_mb_mean:.1f}MB/s")

    # Write sweep CSV
    csv_path = output_dir / "concurrency_sweep.csv"
    if sweep_results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_results[0].to_dict().keys()))
            w.writeheader()
            w.writerows(r.to_dict() for r in sweep_results)
        log.info(f"\nSweep CSV: {csv_path}")

    return sweep_results


# ── Turns sweep ───────────────────────────────────────────────────────────────

@dataclass
class TurnsSweepPoint:
    """One row per num_turns level in the turns sweep CSV.

    Interpretation of num_turns depends on agent_mode:
      agent_backend="none"  ->  num_turns is the number of multi-turn LLM exchanges
                            (model sees test feedback and retries the patch each turn).
      agent_backend!="none" ->  num_turns is the max_agent_steps cap; the real
                            tool-use agent loop runs but is limited to this many
                            bash/str_replace/view_file calls.  agent_steps_cap
                            records that cap explicitly so the CSV is unambiguous.
    """
    num_turns:             int   = 0
    agent_backend:         str   = "none"  # "none"|"amoprof"|"sweagent"|"mini"
    agent_steps_cap:       int   = 0      # max_agent_steps cap (== num_turns when agent_mode)

    # Latency
    ttft_mean_ms:          float = 0.0
    tpot_mean_ms:          float = 0.0
    tpot_p99_ms:           float = 0.0
    total_time_s_mean:     float = 0.0

    # Throughput
    throughput_req_s:      float = 0.0
    throughput_tok_s:      float = 0.0

    # Token counts (mean per instance)
    prompt_tokens_mean:    float = 0.0
    output_tokens_mean:    float = 0.0

    # HBM — overall and per-op breakdown
    hbm_used_gb_mean:      float = 0.0
    hbm_util_pct_mean:     float = 0.0
    hbm_delta_gb_mean:     float = 0.0
    hbm_peak_gb_mean:      float = 0.0
    hbm_prefill_delta_gb_mean: float = 0.0   # KV$ written per prefill phase
    hbm_decode_delta_gb_mean:  float = 0.0   # HBM change during decode (~0 expected)
    hbm_kv_pool_fill_pct_mean: float = 0.0   # KV pool fill % (from token_usage)
    hbm_kv_evicted_gb_mean:    float = 0.0   # estimated eviction per instance

    # DRAM
    dram_used_gb_mean:     float = 0.0
    dram_delta_gb_mean:    float = 0.0
    dram_hicache_staging_gb_mean: float = 0.0  # HiCache DRAM staging buffer

    # SSD
    ssd_read_bw_mb_mean:   float = 0.0
    ssd_write_bw_mb_mean:  float = 0.0
    ssd_read_iops_mean:    float = 0.0
    ssd_util_pct_mean:     float = 0.0

    # SGLang
    kv_cache_hit_rate_pct: float = 0.0   # rises with more turns (prefix reuse)
    ai_op_decode_tok_s:    float = 0.0

    # Outcomes
    instances_run:         int   = 0
    resolved_pct:          float = 0.0
    patch_generated_pct:   float = 0.0

    # ── SSD extended ──────────────────────────────────────────────────────────
    ssd_write_iops_mean:   float = 0.0
    ssd_avgqu_sz_mean:     float = 0.0
    bio_lat_p50_us_mean:   float = 0.0
    bio_lat_p99_us_mean:   float = 0.0
    bio_lat_p999_us_mean:  float = 0.0

    # ── AI op classification & phase distribution ────────────────────────────
    # Counts of instances by dominant AI op type
    op_prefill_count:      int   = 0   # instances where prefill dominated
    op_decode_count:       int   = 0   # instances where decode dominated
    op_reasoning_count:    int   = 0   # instances where R1/QwQ reasoning dominated
    op_mixed_count:        int   = 0   # instances with mixed prefill+decode

    # ── Prefill phase metrics (KV$ write path) ────────────────────────────────
    # Prefill: reads input tokens → computes KV$ → writes KV$ to HBM
    #   SSD write = eviction when HBM pool overflows during prefill
    pf_hbm_delta_gb_mean:        float = 0.0  # mean HBM increase per prefill (KV$ written)
    pf_rt_compute_tokens_mean:   float = 0.0  # mean tokens that required new KV$ computation
    pf_rt_cache_tokens_mean:     float = 0.0  # mean tokens served from existing KV$ (cache hit)
    pf_cache_hit_pct:            float = 0.0  # prefill cache hit rate (rt_cache / total_prefill)
    pf_ssd_eviction_gb_mean:     float = 0.0  # mean GB evicted to SSD triggered by prefill overflow
    pf_ssd_eviction_bw_mb_s:     float = 0.0  # write BW at instances where prefill dominated

    # ── Decode phase metrics (KV$ read path) ──────────────────────────────────
    # Decode: each output token re-reads ENTIRE context KV$ for attention
    #   SSD read = restore when KV$ block was evicted during earlier prefill
    dc_hbm_delta_gb_mean:        float = 0.0  # mean HBM change during decode (should be ~0)
    dc_rt_decode_tokens_mean:    float = 0.0  # mean decode tokens generated
    dc_kv_restored_tokens_mean:  float = 0.0  # mean KV$ tokens restored from SSD per instance
    dc_ssd_restore_gb_mean:      float = 0.0  # mean GB restored from SSD triggered by decode
    dc_ssd_restore_bw_mb_s:      float = 0.0  # read BW at instances where decode dominated
    dc_miss_penalty_ms:          float = 0.0  # mean TPOT overhead from SSD KV$ restores
    dc_kv_read_per_step_gb:      float = 0.0  # theoretical KV$ read per decode step (ctx × bpt)

    # ── AI op throughput ───────────────────────────────────────────────────────
    op_prefill_tok_s_mean:       float = 0.0  # mean prefill throughput (tok/s)
    op_decode_tok_s_mean:        float = 0.0  # mean decode throughput (tok/s)
    op_prefill_to_decode_ratio:  float = 0.0  # prefill_tok_s / decode_tok_s (>1 = prefill-heavy)

    # ── AI op classification & phase distribution ────────────────────────────
    op_prefill_count:      int   = 0
    op_decode_count:       int   = 0
    op_reasoning_count:    int   = 0
    op_mixed_count:        int   = 0
    # Prefill phase metrics
    pf_hbm_delta_gb_mean:        float = 0.0
    pf_rt_compute_tokens_mean:   float = 0.0
    pf_rt_cache_tokens_mean:     float = 0.0
    pf_cache_hit_pct:            float = 0.0
    pf_ssd_eviction_gb_mean:     float = 0.0
    # Decode phase metrics
    dc_hbm_delta_gb_mean:        float = 0.0
    dc_rt_decode_tokens_mean:    float = 0.0
    dc_kv_restored_tokens_mean:  float = 0.0
    dc_ssd_restore_gb_mean:      float = 0.0
    dc_miss_penalty_ms:          float = 0.0
    dc_kv_read_per_step_gb:      float = 0.0
    # AI op throughput
    op_prefill_tok_s_mean:       float = 0.0
    op_decode_tok_s_mean:        float = 0.0
    op_prefill_to_decode_ratio:  float = 0.0

    # ── SGLang extended ───────────────────────────────────────────────────────
    ai_op_prefill_tok_s:          float = 0.0
    cache_hit_rate_realtime_pct:  float = 0.0
    kv_l1_device_tokens_mean:     float = 0.0
    kv_l2_host_tokens_mean:       float = 0.0
    kv_l3_storage_tokens_mean:    float = 0.0
    kv_evicted_tokens_mean:       float = 0.0
    kv_restored_tokens_mean:      float = 0.0
    server_ttft_ms_mean:          float = 0.0
    server_itl_ms_mean:           float = 0.0
    hicache_eviction_ms_mean:     float = 0.0
    hicache_load_back_ms_mean:    float = 0.0
    token_usage_peak_mean:        float = 0.0
    num_queue_reqs_peak_mean:     float = 0.0
    utilization_mean:             float = 0.0

    # ── A2 NVMe Driver ────────────────────────────────────────────────────────
    nvme_inflight_mean:    float = 0.0
    nvme_inflight_peak:    int   = 0
    nvme_rd_lat_ms_sysfs:  float = 0.0

    # ── A3 OS / vmstat ────────────────────────────────────────────────────────
    page_faults_per_s:     float = 0.0
    major_faults_per_s:    float = 0.0
    numa_migrations_per_s: float = 0.0
    hugepages_used:        int   = 0

    # ── A2 GPU Driver / NVLink ────────────────────────────────────────────────
    nvlink_tx_gb_s:        float = 0.0
    nvlink_rx_gb_s:        float = 0.0
    pcie_tx_gb_s:          float = 0.0
    pcie_rx_gb_s:          float = 0.0

    # ── Application ───────────────────────────────────────────────────────────
    tok_per_s_mean:        float = 0.0
    req_lat_p99_ms:        float = 0.0
    req_lat_p999_ms:       float = 0.0

    # ── Endurance ─────────────────────────────────────────────────────────────
    waf:                   float = 0.0
    host_written_gb:       float = 0.0
    nand_written_gb:       float = 0.0
    ssd_lifetime_tbw:      float = 0.0
    ssd_dwpd_est:          float = 0.0
    temp_peak_c:           int   = 0

    # ── A1 CUDA kernel-level ─────────────────────────────────────────────────
    cuda_sm_active_mean_pct:     float = 0.0
    cuda_sm_active_min_pct:      float = 0.0
    cuda_tensor_active_mean_pct: float = 0.0
    cuda_tensor_active_min_pct:  float = 0.0
    cuda_dram_active_mean_pct:   float = 0.0
    cuda_hbm_bw_read_gb_s:       float = 0.0
    cuda_hbm_bw_write_gb_s:      float = 0.0
    cuda_sm_clock_mhz:           float = 0.0
    cuda_sm_occupancy_mean_pct:  float = 0.0
    cuda_fp16_active_mean_pct:   float = 0.0
    cuda_throttled_pct:          float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_turns_sweep(
    plan,
    run_id: str,
    output_dir: Path,
    turns_levels: list[int],
    instances_per_level: int = 10,
    server_port: int = 30000,
    use_docker: bool = False,
    docker_in_loop: bool = False,
    max_tokens: int = 8192,
    call_timeout: int = 900,
    workers: int = 1,
    inference_concurrency: int = 1,
    agent_backend: str = "none",
    max_agent_steps: int = 30,
    split: str = "pro",
    instances_path: Optional[str] = None,
    sweagent_default_config: Optional[str] = None,
    sweagent_config_path: Optional[str] = None,
    sweagent_model_name: Optional[str] = None,
    sweagent_model_api_base: Optional[str] = None,
    sweagent_model_api_key: str = "EMPTY",
    sweagent_max_input_tokens: int = 50000,
    sweagent_num_workers: int = 1,
    sweagent_instances_type: str = "file",
    sweagent_redo_existing: bool = False,
    sweagent_shuffle: bool = False,
) -> list[TurnsSweepPoint]:
    """
    Run SWE-bench at multiple num_turns levels and collect per-level metrics.

    Behaviour differs by agent_mode:

    NON-AGENT MODE (agent_mode=False):
    ─────────────────────────────────
    num_turns controls how many times the model sees test-execution feedback
    and retries its patch.  Each turn: model generates patch → Docker applies
    and runs tests → failure output fed back → model tries again.

    turns=1:  Single prefill + decode
              KV$ hit rate ≈ 0  (no prefix to reuse)
              HBM delta = kv(prompt_tokens)

    turns=2:  Second turn reuses turn-1 prefix via RadixAttention
              KV$ hit rate ≈ 40-60%  (all of turn 1 is cached)
              HBM delta per turn decreases  (incremental tokens only)
              Cumulative KV$ = sum of all turns' unique tokens

    turns=3+: Hit rate continues rising  (larger shared prefix)
              Per-turn TTFT falls  (prefix hit → fewer tokens to prefill)
              SSD pressure rises  (larger total KV$ → more eviction at fixed concurrency)
              TPOT rises  (larger context = more KV$ to read per decode step)

    AGENT MODE (agent_mode=True):
    ─────────────────────────────
    num_turns is remapped to max_agent_steps — the cap on tool calls the agent
    may make (bash / str_replace / view_file / finish).  The real SWE-bench
    agent loop runs inside a Docker container; the model explores the repo,
    edits files, and verifies with tests on each step.

    steps=5:  Short trajectory.  Context stays small (~10–25K tokens).
              KV$ footprint per instance is low → overflow threshold is high.
              Agent may not finish; finish() often not called within the cap.

    steps=15: Medium trajectory.  Context grows as bash output accumulates.
              RadixAttention reuses the shared system+problem prefix across steps.
              Increasing KV$ hit rate as the conversation grows.

    steps=30: Long trajectory.  Context can reach 50–100K tokens if bash
              output is verbose.  High KV$ pressure at moderate concurrency.
              SSD eviction likely if concurrency > overflow_concurrency.

    In agent mode the CSV column `agent_steps_cap` records the cap value
    explicitly; `num_turns` is kept for backward-compat but equals agent_steps_cap.
    Use `agent_steps_taken` on per-instance SWEBenchResult for actual steps used.

    AI op → memory tier mapping across turns/steps:
    ────────────────────────────────────────────────
    TURN/STEP 1 PREFILL:
      hbm_op:  write  N_tokens × kv_bytes_per_token to KV$ pool
      dram_op: minimal
      ssd_op:  idle (unless pool already full)

    TURN/STEP N PREFILL (N>1):
      hbm_op:  write  only NEW tokens (RadixAttention reuses prefix)
      kv_hit:  (cumulative_prefix_tokens / total_tokens) × 100%
      dram_op: minimal
      ssd_op:  eviction write if pool overflows

    DECODE (all turns/steps):
      hbm_op:  read   entire KV$ for each output token
      dram_op: HiCache staging (SSD→DRAM→HBM path when evicted)
      ssd_op:  random read on KV$ miss

    The key insight:  more turns/steps = larger effective context per instance
    = more KV$ in HBM = lower overflow threshold = more SSD I/O per instance.
    """
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results: list[TurnsSweepPoint] = []

    log.info(f"Turns sweep: levels={turns_levels}  instances_per_level={instances_per_level}")

    for num_turns in turns_levels:
        log.info(f"\n{'='*60}")
        if agent_backend != "none":
            log.info(f"  agent_steps_cap = {num_turns}  (max tool calls per instance)")
            log.info(f"  Est. context growth: ~{num_turns * 2}K–{num_turns * 5}K tokens "
                     f"(varies with bash output verbosity)")
            log.info(f"  RadixAttention prefix reuse: yes (system+problem shared across steps)")
        else:
            log.info(f"  num_turns = {num_turns}")
            log.info(f"  Expected KV$ growth per instance: "
                     f"~{num_turns * 10:.0f} GB cumulative (prompt-mode estimate)")
            log.info(f"  RadixAttention prefix reuse: {'yes' if num_turns > 1 else 'no'}")
        log.info(f"{'='*60}")

        level_dir = output_dir / f"turns_{num_turns:02d}"
        level_max_agent_steps = num_turns if agent_backend != "none" else max_agent_steps
        harness = SWEBenchHarness(
            plan                  = plan,
            run_id                = f"{run_id}_t{num_turns}",
            output_dir            = level_dir,
            split                 = "pro",
            max_instances         = instances_per_level,
            server_port           = server_port,
            use_docker            = use_docker,
            max_tokens            = max_tokens,
            call_timeout          = call_timeout,
            workers               = workers,
            inference_concurrency = inference_concurrency,
            num_turns             = num_turns,
            docker_in_loop        = docker_in_loop,
            agent_backend         = agent_backend,
            max_agent_steps       = level_max_agent_steps,
            context_len           = 65536,
            instances_path        = instances_path,
            sweagent_default_config = sweagent_default_config,
            sweagent_config_path  = sweagent_config_path,
            sweagent_model_name   = sweagent_model_name,
            sweagent_model_api_base = sweagent_model_api_base,
            sweagent_model_api_key = sweagent_model_api_key,
            sweagent_max_input_tokens = sweagent_max_input_tokens,
            sweagent_num_workers  = sweagent_num_workers,
            sweagent_instances_type = sweagent_instances_type,
            sweagent_redo_existing = sweagent_redo_existing,
            sweagent_shuffle      = sweagent_shuffle,
        )

        results = harness.run_all()
        if not results:
            log.warning(f"No results at num_turns={num_turns}")
            continue

        n = len(results)
        def mn(attr): return round(sum(getattr(r, attr, 0) or 0 for r in results) / max(n, 1), 3)
        def pk(attr): return round(max((getattr(r, attr, 0) or 0) for r in results), 3)

        wall_time = sum(r.total_time_s for r in results) / max(inference_concurrency, 1)
        req_per_s = round(n / max(wall_time, 0.001), 3)
        tok_per_s = round(sum(r.output_tokens for r in results) / max(wall_time, 0.001), 1)

        sp = TurnsSweepPoint(
            num_turns             = num_turns,
            agent_backend         = agent_backend,
            agent_steps_cap       = level_max_agent_steps if agent_backend != "none" else 0,
            ttft_mean_ms          = mn("ttft_ms"),
            tpot_mean_ms          = mn("tpot_mean_ms"),
            tpot_p99_ms           = pk("tpot_p99_ms"),
            total_time_s_mean     = mn("total_time_s"),
            throughput_req_s      = req_per_s,
            throughput_tok_s      = tok_per_s,
            prompt_tokens_mean    = mn("prompt_tokens"),
            output_tokens_mean    = mn("output_tokens"),
            hbm_used_gb_mean      = mn("hbm_used_gb_end"),
            hbm_util_pct_mean     = mn("hbm_util_pct"),
            hbm_delta_gb_mean     = mn("hbm_delta_gb"),
            hbm_peak_gb_mean      = mn("hbm_peak_gb"),
            hbm_prefill_delta_gb_mean  = mn("hbm_prefill_delta_gb"),
            hbm_decode_delta_gb_mean   = mn("hbm_decode_delta_gb"),
            hbm_kv_pool_fill_pct_mean  = mn("hbm_kv_pool_fill_pct"),
            hbm_kv_evicted_gb_mean     = mn("hbm_kv_evicted_gb"),
            dram_used_gb_mean     = mn("dram_used_gb_end"),
            dram_delta_gb_mean    = mn("dram_delta_gb"),
            dram_hicache_staging_gb_mean = mn("dram_hicache_staging_gb"),
            ssd_read_bw_mb_mean   = mn("read_bw_mb_mean"),
            ssd_write_bw_mb_mean  = mn("write_bw_mb_mean"),
            ssd_read_iops_mean    = mn("read_iops_mean"),
            ssd_util_pct_mean     = mn("util_pct_mean"),
            kv_cache_hit_rate_pct = mn("kv_cache_hit_rate_pct"),
            ai_op_decode_tok_s    = mn("ai_op_decode_tok_s"),
            instances_run         = n,
            resolved_pct          = round(sum(r.resolved for r in results) / max(n,1) * 100, 1),
            patch_generated_pct   = round(sum(r.patch_generated for r in results) / max(n,1) * 100, 1),
            # SSD extended
            ssd_write_iops_mean   = mn("write_iops_mean"),
            ssd_avgqu_sz_mean     = mn("avgqu_sz_mean"),
            bio_lat_p50_us_mean   = mn("bio_lat_p50_us"),
            bio_lat_p99_us_mean   = mn("bio_lat_p99_us"),
            bio_lat_p999_us_mean  = mn("bio_lat_p999_us"),
            # SGLang extended
            ai_op_prefill_tok_s          = mn("ai_op_prefill_tok_s"),
            cache_hit_rate_realtime_pct  = mn("cache_hit_rate_realtime_pct"),
            kv_l1_device_tokens_mean     = mn("kv_l1_device_tokens"),
            kv_l2_host_tokens_mean       = mn("kv_l2_host_tokens"),
            kv_l3_storage_tokens_mean    = mn("kv_l3_storage_tokens"),
            kv_evicted_tokens_mean       = mn("kv_evicted_tokens"),
            kv_restored_tokens_mean      = mn("kv_restored_tokens"),
            server_ttft_ms_mean          = mn("server_ttft_ms"),
            server_itl_ms_mean           = mn("server_itl_ms"),
            hicache_eviction_ms_mean     = mn("hicache_eviction_ms"),
            hicache_load_back_ms_mean    = mn("hicache_load_back_ms"),
            token_usage_peak_mean        = mn("token_usage_peak"),
            num_queue_reqs_peak_mean     = mn("num_queue_reqs_peak"),
            utilization_mean             = mn("utilization_mean"),
            # NVMe Driver
            nvme_inflight_mean    = mn("nvme_inflight_mean"),
            nvme_inflight_peak    = int(pk("nvme_inflight_peak")),
            nvme_rd_lat_ms_sysfs  = mn("nvme_rd_lat_ms_sysfs"),
            # vmstat
            page_faults_per_s     = mn("page_faults_per_s"),
            major_faults_per_s    = mn("major_faults_per_s"),
            numa_migrations_per_s = mn("numa_migrations_per_s"),
            hugepages_used        = int(mn("hugepages_used")),
            # NVLink / PCIe
            nvlink_tx_gb_s        = mn("nvlink_tx_gb_s"),
            nvlink_rx_gb_s        = mn("nvlink_rx_gb_s"),
            pcie_tx_gb_s          = mn("pcie_tx_gb_s"),
            pcie_rx_gb_s          = mn("pcie_rx_gb_s"),
            # Application
            tok_per_s_mean        = mn("tok_per_s"),
            req_lat_p99_ms        = results[-1].req_lat_p99_ms,
            req_lat_p999_ms       = results[-1].req_lat_p999_ms,
            # Endurance
            waf              = results[-1].waf,
            host_written_gb  = results[-1].host_written_gb,
            nand_written_gb  = results[-1].nand_written_gb,
            ssd_lifetime_tbw = results[-1].ssd_lifetime_tbw,
            ssd_dwpd_est     = results[-1].ssd_dwpd_est,
            temp_peak_c      = results[-1].temp_peak_c,
            cuda_sm_active_mean_pct     = mn("cuda_sm_active_mean_pct"),
            cuda_sm_active_min_pct      = mn("cuda_sm_active_min_pct"),
            cuda_tensor_active_mean_pct = mn("cuda_tensor_active_mean_pct"),
            cuda_tensor_active_min_pct  = mn("cuda_tensor_active_min_pct"),
            cuda_dram_active_mean_pct   = mn("cuda_dram_active_mean_pct"),
            cuda_hbm_bw_read_gb_s       = mn("cuda_hbm_bw_read_gb_s"),
            cuda_hbm_bw_write_gb_s      = mn("cuda_hbm_bw_write_gb_s"),
            cuda_sm_clock_mhz           = mn("cuda_sm_clock_mhz"),
            cuda_sm_occupancy_mean_pct  = mn("cuda_sm_occupancy_mean_pct"),
            cuda_fp16_active_mean_pct   = mn("cuda_fp16_active_mean_pct"),
            cuda_throttled_pct          = mn("cuda_throttled_pct"),
            **_ai_op_phase_fields(results, plan.model.kv_bytes_per_token()),
        )
        sweep_results.append(sp)

        _level_label = (f"steps_cap={num_turns}" if agent_backend != "none"
                        else f"turns={num_turns}")
        log.info(
            f"  {_level_label}  "
            f"TTFT={sp.ttft_mean_ms:.0f}ms  TPOT={sp.tpot_mean_ms:.1f}ms  "
            f"HBM_prefill_delta={sp.hbm_prefill_delta_gb_mean:+.2f}GB  "
            f"KV$_hit={sp.kv_cache_hit_rate_pct:.0f}%  "
            f"SSD_wBW={sp.ssd_write_bw_mb_mean:.1f}MB/s")

    csv_path = output_dir / "turns_sweep.csv"
    if sweep_results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_results[0].to_dict().keys()))
            w.writeheader()
            w.writerows(r.to_dict() for r in sweep_results)
        log.info(f"\nTurns sweep CSV: {csv_path}")

    return sweep_results


# ════════════════════════════════════════════════════════════════════════════
# Context-length sweep
# ════════════════════════════════════════════════════════════════════════════
#
# Varies context_len while holding concurrency fixed.
# Key insight: KV$ per request = kv_bytes_per_token × context_len
#              overflow_concurrency = pool_gb × 1024³ / kv_bytes_per_request
# At short contexts (8K) the pool can hold many more requests before overflow.
# At long contexts (128K) the pool fills at very low concurrency.


@dataclass
class ContextSweepPoint:
    """One row in the context-length sweep CSV."""
    context_len:           int   = 0    # context length in tokens (x-axis)
    inference_concurrency: int   = 0    # fixed concurrency used for this level

    # ── KV$ pool maths (derived from model + context_len) ─────────────────
    kv_per_request_gb:     float = 0.0  # KV$ consumed per active request at this ctx
    kv_bytes_per_token:    int   = 0    # bytes per token (constant for a model)
    kv_pool_capacity_gb:   float = 0.0  # HBM × mem_fraction − weights
    overflow_concurrency:  int   = 0    # pool / kv_per_request_gb — at this ctx
    pool_fill_pct:         float = 0.0  # (concurrency × kv_per_req) / pool × 100
    ssd_expected:          bool  = False # True when concurrency ≥ overflow_concurrency

    # ── Inference latency ─────────────────────────────────────────────────
    ttft_mean_ms:          float = 0.0  # scales with context (more tokens to prefill)
    ttft_p99_ms:           float = 0.0
    tpot_mean_ms:          float = 0.0  # constant per token IF KV$ fits in HBM
    tpot_p99_ms:           float = 0.0  # rises when SSD fetches stall decode
    kv_miss_penalty_ms:    float = 0.0  # TPOT overhead from SSD KV$ restore

    # ── HBM tier ──────────────────────────────────────────────────────────
    hbm_used_gb_mean:      float = 0.0
    hbm_util_pct_mean:     float = 0.0
    hbm_prefill_delta_gb_mean: float = 0.0  # grows linearly with context_len
    hbm_kv_evicted_gb_mean:    float = 0.0  # >0 only when pool_fill > 100%

    # ── DRAM tier ─────────────────────────────────────────────────────────
    dram_used_gb_mean:          float = 0.0
    dram_hicache_staging_gb_mean: float = 0.0

    # ── SSD tier ──────────────────────────────────────────────────────────
    ssd_read_bw_mb_mean:   float = 0.0  # KV$ restore rate — 0 below overflow
    ssd_write_bw_mb_mean:  float = 0.0  # KV$ eviction rate — 0 below overflow
    ssd_read_iops_mean:    float = 0.0
    ssd_r_await_p99_ms:    float = 0.0
    ssd_util_pct_mean:     float = 0.0
    bio_lat_p99_us_mean:   float = 0.0

    # ── AI op phase breakdown ─────────────────────────────────────────────
    op_prefill_count:             int   = 0
    op_decode_count:              int   = 0
    pf_hbm_delta_gb_mean:         float = 0.0
    pf_rt_compute_tokens_mean:    float = 0.0
    pf_cache_hit_pct:             float = 0.0
    pf_ssd_eviction_gb_mean:      float = 0.0
    dc_rt_decode_tokens_mean:     float = 0.0
    dc_kv_restored_tokens_mean:   float = 0.0
    dc_ssd_restore_gb_mean:       float = 0.0
    dc_kv_read_per_step_gb:       float = 0.0  # context_len × kv_bpt — key scaling metric

    # ── Throughput ────────────────────────────────────────────────────────
    throughput_req_s:      float = 0.0
    throughput_tok_s:      float = 0.0
    prompt_tokens_mean:    float = 0.0
    output_tokens_mean:    float = 0.0

    # ── Endurance ─────────────────────────────────────────────────────────
    waf:                   float = 0.0
    host_written_gb:       float = 0.0
    temp_peak_c:           int   = 0
    instances_run:         int   = 0
    resolved_pct:          float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_context_sweep(
    plan,
    run_id: str,
    output_dir: Path,
    context_lengths: list[int],
    instances_per_level: int = 10,
    inference_concurrency: int = 8,
    server_port: int = 30000,
    use_docker: bool = False,
    max_tokens: int = 8192,
    call_timeout: int = 900,
    workers: int = 1,
    mem_fraction: float = 0.80,
    docker_in_loop: bool = False,
    agent_backend: str = "none",
    max_agent_steps: int = 30,
    instances_path: Optional[str] = None,
    sweagent_default_config: Optional[str] = None,
    sweagent_config_path: Optional[str] = None,
    sweagent_model_name: Optional[str] = None,
    sweagent_model_api_base: Optional[str] = None,
    sweagent_model_api_key: str = "EMPTY",
    sweagent_max_input_tokens: int = 50000,
    sweagent_num_workers: int = 1,
    sweagent_instances_type: str = "file",
    sweagent_redo_existing: bool = False,
    sweagent_shuffle: bool = False,
) -> list[ContextSweepPoint]:
    """
    Run SWE-bench at multiple context_lengths and collect per-level metrics.

    The key insight this sweep reveals:
      - kv_per_request_gb = kv_bytes_per_token × context_len / 1024³
      - overflow_concurrency = pool_gb / kv_per_request_gb
      - At short contexts (8K): overflow at ~160 concurrent requests
      - At long contexts (128K): overflow at ~9 concurrent requests
      - SSD I/O therefore starts at MUCH lower concurrency for long-context workloads

    SGLang context_len is set per-request via max_tokens in the prompt, NOT via
    the server --max-model-len parameter.  Each level simply truncates/pads
    the SWE-bench prompt to the target length before calling the model.

    Args:
        context_lengths: list of context lengths to sweep, e.g. [8192, 32768, 65536, 131072]
        inference_concurrency: fixed concurrency used at every level
    """
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results: list[ContextSweepPoint] = []

    model = plan.model
    kv_bpt = model.kv_bytes_per_token()
    pool_gb = max(0.0, plan.hbm_cap_gb * mem_fraction - model.size_gb(model.preferred_dtype))

    log.info(
        f"Context sweep: lengths={context_lengths}  concurrency={inference_concurrency}  "
        f"pool={pool_gb:.0f}GB  kv_bpt={kv_bpt} bytes")

    # Shared endurance monitors (run across whole sweep)
    from .collectors import NvmeSmartMonitor, PowerMonitor, SsdHardwareMonitor
    smart   = NvmeSmartMonitor(plan.ssd_device)
    power   = PowerMonitor()
    ssd_hw  = SsdHardwareMonitor(plan.ssd_device)
    smart.start(); power.start(); ssd_hw.start()

    baseline_tpot: float = 0.0

    try:
        for context_len in context_lengths:
            kv_req_gb = model.kv_size_gb(context_len, 1, model.preferred_dtype)
            overflow_at = int(pool_gb / max(kv_req_gb, 0.001))
            pool_fill_pct = round(inference_concurrency * kv_req_gb / max(pool_gb, 0.001) * 100, 1)
            ssd_expected = inference_concurrency >= overflow_at

            log.info(
                f"  context={context_len:,}  kv_per_req={kv_req_gb:.1f}GB  "
                f"overflow_at={overflow_at}  pool_fill={pool_fill_pct:.0f}%  "
                f"ssd_expected={ssd_expected}")

            # Run instances with this context length
            harness = SWEBenchHarness(
                plan                  = plan,
                run_id                = f"{run_id}_ctx{context_len}",
                output_dir            = output_dir / f"ctx_{context_len}",
                split                 = "pro",
                max_instances         = instances_per_level,
                server_port           = server_port,
                use_docker            = use_docker,
                max_tokens            = min(max_tokens, context_len),
                call_timeout          = call_timeout,
                inference_concurrency = inference_concurrency,
                workers               = workers,
                num_turns             = 1,
                docker_in_loop        = docker_in_loop,
                agent_backend         = agent_backend,
                max_agent_steps       = max_agent_steps,
                context_len           = context_len,
                instances_path        = instances_path,
                sweagent_default_config = sweagent_default_config,
                sweagent_config_path  = sweagent_config_path,
                sweagent_model_name   = sweagent_model_name,
                sweagent_model_api_base = sweagent_model_api_base,
                sweagent_model_api_key = sweagent_model_api_key,
                sweagent_max_input_tokens = sweagent_max_input_tokens,
                sweagent_num_workers  = sweagent_num_workers,
                sweagent_instances_type = sweagent_instances_type,
                sweagent_redo_existing = sweagent_redo_existing,
                sweagent_shuffle      = sweagent_shuffle,
            )
            results = harness.run_all()
            n = max(len(results), 1)

            def mn(attr):
                return round(sum(getattr(r, attr, 0) or 0 for r in results) / n, 3)
            def pk(attr):
                return round(max((getattr(r, attr, 0) or 0) for r in results), 3)

            tpot_now = mn("tpot_mean_ms")
            if baseline_tpot == 0 and tpot_now > 0:
                baseline_tpot = tpot_now
            miss_penalty = round(max(0.0, tpot_now - baseline_tpot), 2)

            ai_op_fields = _ai_op_phase_fields(results, kv_bpt)
            dc_read_per_step_gb = round(context_len * kv_bpt / (1024**3), 3)

            # Throughput
            wall_time = sum(getattr(r, "total_time_s", 0) for r in results)
            req_per_s  = round(n / max(wall_time / inference_concurrency, 0.001), 3)
            tok_per_s  = round(mn("output_tokens") * n / max(wall_time / inference_concurrency, 0.001), 1)

            sp = ContextSweepPoint(
                context_len           = context_len,
                inference_concurrency = inference_concurrency,
                kv_per_request_gb     = round(kv_req_gb, 3),
                kv_bytes_per_token    = kv_bpt,
                kv_pool_capacity_gb   = round(pool_gb, 1),
                overflow_concurrency  = overflow_at,
                pool_fill_pct         = pool_fill_pct,
                ssd_expected          = ssd_expected,
                ttft_mean_ms          = mn("ttft_ms"),
                ttft_p99_ms           = pk("ttft_ms"),
                tpot_mean_ms          = tpot_now,
                tpot_p99_ms           = pk("tpot_p99_ms"),
                kv_miss_penalty_ms    = miss_penalty,
                hbm_used_gb_mean      = mn("hbm_used_gb_end"),
                hbm_util_pct_mean     = mn("hbm_util_pct"),
                hbm_prefill_delta_gb_mean  = mn("hbm_prefill_delta_gb"),
                hbm_kv_evicted_gb_mean     = mn("hbm_kv_evicted_gb"),
                dram_used_gb_mean          = mn("dram_used_gb_end"),
                dram_hicache_staging_gb_mean = mn("dram_hicache_staging_gb"),
                ssd_read_bw_mb_mean   = mn("read_bw_mb_mean"),
                ssd_write_bw_mb_mean  = mn("write_bw_mb_mean"),
                ssd_read_iops_mean    = mn("read_iops_mean"),
                ssd_r_await_p99_ms    = pk("r_await_ms_p99"),
                ssd_util_pct_mean     = mn("util_pct_mean"),
                bio_lat_p99_us_mean   = mn("bio_lat_p99_us"),
                dc_kv_read_per_step_gb= dc_read_per_step_gb,
                throughput_req_s      = req_per_s,
                throughput_tok_s      = tok_per_s,
                prompt_tokens_mean    = mn("prompt_tokens"),
                output_tokens_mean    = mn("output_tokens"),
                instances_run         = n,
                resolved_pct          = round(sum(r.resolved for r in results) / n * 100, 1),
                op_prefill_count      = ai_op_fields["op_prefill_count"],
                op_decode_count       = ai_op_fields["op_decode_count"],
                pf_hbm_delta_gb_mean      = ai_op_fields["pf_hbm_delta_gb_mean"],
                pf_rt_compute_tokens_mean = ai_op_fields["pf_rt_compute_tokens_mean"],
                pf_cache_hit_pct          = ai_op_fields["pf_cache_hit_pct"],
                pf_ssd_eviction_gb_mean   = ai_op_fields["pf_ssd_eviction_gb_mean"],
                dc_rt_decode_tokens_mean  = ai_op_fields["dc_rt_decode_tokens_mean"],
                dc_kv_restored_tokens_mean= ai_op_fields["dc_kv_restored_tokens_mean"],
                dc_ssd_restore_gb_mean    = ai_op_fields["dc_ssd_restore_gb_mean"],
            )
            sweep_results.append(sp)
            log.info(
                f"  ctx={context_len:,}  TTFT={sp.ttft_mean_ms:.0f}ms  TPOT={sp.tpot_mean_ms:.0f}ms  "
                f"pool_fill={sp.pool_fill_pct:.0f}%  SSD_r={sp.ssd_read_bw_mb_mean:.1f}MB/s  "
                f"SSD_w={sp.ssd_write_bw_mb_mean:.1f}MB/s  miss_pen={sp.kv_miss_penalty_ms:.0f}ms")

    finally:
        smart_m  = smart.stop()
        power_m  = power.stop()
        ssd_hw_m = ssd_hw.stop()
        if sweep_results:
            sweep_results[-1].waf            = smart_m.get("waf", 0.0)
            sweep_results[-1].host_written_gb= smart_m.get("host_written_gb", 0.0)
            sweep_results[-1].temp_peak_c    = smart_m.get("temp_peak_c", 0)

    csv_path = output_dir / "context_sweep.csv"
    if sweep_results:
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=list(sweep_results[0].to_dict().keys()))
            w.writeheader()
            w.writerows(r.to_dict() for r in sweep_results)
        log.info(f"\nContext sweep CSV: {csv_path}")

    return sweep_results


# ════════════════════════════════════════════════════════════════════════════
# Batch-size sweep
# ════════════════════════════════════════════════════════════════════════════
#
# Varies the number of instances submitted simultaneously per burst batch.
# Unlike the concurrency sweep (steady-state concurrent requests), the batch
# sweep submits exactly batch_size requests at once, waits for ALL of them
# to complete, then submits the next batch.
#
# This characterises bursty AI workloads where requests arrive in waves:
#   batch_size=1  : sequential, no parallelism
#   batch_size=4  : small wave — brief HBM fill spike then drain
#   batch_size=32 : large wave — sustained pool fill, heavy SSD I/O per batch
#
# Key observable: HBM peak per batch (not steady-state), inter-batch cooling,
# and whether the pool overflows within a single batch burst.


@dataclass
class BatchSweepPoint:
    """One row in the batch-size sweep CSV."""
    batch_size:            int   = 0    # requests submitted simultaneously (x-axis)
    inference_concurrency: int   = 0    # same as batch_size for this sweep

    # ── KV$ pool analysis per batch ───────────────────────────────────────
    kv_per_request_gb:     float = 0.0
    kv_pool_capacity_gb:   float = 0.0
    pool_fill_at_batch_pct:float = 0.0  # batch_size × kv_per_req / pool × 100
    batch_overflows_pool:  bool  = False # True when entire batch can't fit in HBM

    # ── Burst latency (wall clock for the entire batch to complete) ────────
    batch_wall_time_s_mean:float = 0.0  # mean wall-clock time per batch
    batch_wall_time_s_p99: float = 0.0  # P99 wall-clock time per batch
    ttft_mean_ms:          float = 0.0  # TTFT rises as more requests compete for prefill
    ttft_p99_ms:           float = 0.0
    tpot_mean_ms:          float = 0.0
    tpot_p99_ms:           float = 0.0
    kv_miss_penalty_ms:    float = 0.0

    # ── HBM: peak vs steady-state ──────────────────────────────────────────
    hbm_peak_gb_mean:       float = 0.0  # peak HBM during the batch burst
    hbm_used_gb_mean:       float = 0.0  # mean HBM (lower — averages over burst+idle)
    hbm_util_pct_mean:      float = 0.0
    hbm_prefill_delta_gb_mean:  float = 0.0  # KV$ written per prefill phase
    hbm_kv_evicted_gb_mean:     float = 0.0  # KV$ evicted during this batch

    # ── DRAM tier ─────────────────────────────────────────────────────────
    dram_hicache_staging_gb_mean: float = 0.0

    # ── SSD tier ──────────────────────────────────────────────────────────
    ssd_read_bw_mb_mean:   float = 0.0
    ssd_write_bw_mb_mean:  float = 0.0
    ssd_read_iops_mean:    float = 0.0
    ssd_r_await_p99_ms:    float = 0.0
    ssd_util_pct_mean:     float = 0.0
    bio_lat_p99_us_mean:   float = 0.0

    # ── AI op phase breakdown ─────────────────────────────────────────────
    op_prefill_count:            int   = 0
    op_decode_count:             int   = 0
    pf_hbm_delta_gb_mean:        float = 0.0
    pf_rt_compute_tokens_mean:   float = 0.0
    pf_cache_hit_pct:            float = 0.0
    pf_ssd_eviction_gb_mean:     float = 0.0
    dc_rt_decode_tokens_mean:    float = 0.0
    dc_kv_restored_tokens_mean:  float = 0.0
    dc_ssd_restore_gb_mean:      float = 0.0
    dc_miss_penalty_ms:          float = 0.0

    # ── Throughput ────────────────────────────────────────────────────────
    throughput_req_s:      float = 0.0  # effective requests/s including inter-batch idle
    throughput_tok_s:      float = 0.0
    prompt_tokens_mean:    float = 0.0
    output_tokens_mean:    float = 0.0

    # ── Endurance ─────────────────────────────────────────────────────────
    waf:                   float = 0.0
    host_written_gb:       float = 0.0
    temp_peak_c:           int   = 0
    batches_run:           int   = 0    # number of batches completed
    instances_run:         int   = 0
    resolved_pct:          float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_batch_sweep(
    plan,
    run_id: str,
    output_dir: Path,
    batch_sizes: list[int],
    total_instances: int = 50,
    context_len: int = 65536,
    server_port: int = 30000,
    use_docker: bool = False,
    max_tokens: int = 8192,
    call_timeout: int = 900,
    workers: int = 1,
    mem_fraction: float = 0.80,
    docker_in_loop: bool = False,
    agent_backend: str = "none",
    max_agent_steps: int = 30,
    instances_path: Optional[str] = None,
    sweagent_default_config: Optional[str] = None,
    sweagent_config_path: Optional[str] = None,
    sweagent_model_name: Optional[str] = None,
    sweagent_model_api_base: Optional[str] = None,
    sweagent_model_api_key: str = "EMPTY",
    sweagent_max_input_tokens: int = 50000,
    sweagent_num_workers: int = 1,
    sweagent_instances_type: str = "file",
    sweagent_redo_existing: bool = False,
    sweagent_shuffle: bool = False,
) -> list[BatchSweepPoint]:
    """
    Run SWE-bench with bursting batches of batch_size instances.

    Unlike concurrency sweep (steady-state), this submits exactly batch_size
    requests simultaneously, waits for ALL to finish (synchronous batch),
    then submits the next batch.  This reveals:

      - HBM peak pressure within a burst (vs steady-state average)
      - Whether a single batch overflows the KV$ pool
      - SSD I/O per batch burst (event-driven rather than continuous)
      - TTFT degradation within a batch as requests compete for prefill slots
      - Inter-batch "cooldown" — KV$ eviction pattern

    Args:
        batch_sizes:      list of batch sizes to sweep, e.g. [1, 4, 8, 16, 32]
        total_instances:  total instances to process per level (split into batches)
        context_len:      fixed context length for this sweep
    """
    import csv as _csv
    import math as _math
    from concurrent.futures import ThreadPoolExecutor, as_completed
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results: list[BatchSweepPoint] = []

    model  = plan.model
    kv_bpt = model.kv_bytes_per_token()
    pool_gb = max(0.0, plan.hbm_cap_gb * mem_fraction - model.size_gb(model.preferred_dtype))
    kv_req_gb = model.kv_size_gb(context_len, 1, model.preferred_dtype)

    log.info(
        f"Batch sweep: sizes={batch_sizes}  total_instances={total_instances}  "
        f"pool={pool_gb:.0f}GB  kv_per_req={kv_req_gb:.1f}GB")

    from .collectors import NvmeSmartMonitor, PowerMonitor
    smart = NvmeSmartMonitor(plan.ssd_device)
    power = PowerMonitor()
    smart.start(); power.start()

    baseline_tpot: float = 0.0

    try:
        for batch_size in batch_sizes:
            pool_fill_pct = round(batch_size * kv_req_gb / max(pool_gb, 0.001) * 100, 1)
            overflows     = pool_fill_pct > 100.0
            n_batches     = max(1, _math.ceil(total_instances / batch_size))

            log.info(
                f"  batch_size={batch_size}  pool_fill={pool_fill_pct:.0f}%  "
                f"overflows={overflows}  n_batches={n_batches}")

            all_results  = []
            batch_times  = []

            for batch_idx in range(n_batches):
                # Run one synchronous batch
                harness = SWEBenchHarness(
                    plan           = plan,
                    run_id         = f"{run_id}_bs{batch_size}_b{batch_idx}",
                    output_dir     = output_dir / f"bs_{batch_size}" / f"batch_{batch_idx}",
                    split          = "pro",
                    max_instances  = batch_size,
                    server_port    = server_port,
                    use_docker     = use_docker,
                    max_tokens     = max_tokens,
                    call_timeout   = call_timeout,
                    inference_concurrency = batch_size,  # all batch members sent together
                    workers        = workers,
                    num_turns      = 1,
                    docker_in_loop = docker_in_loop,
                    agent_backend  = agent_backend,
                    max_agent_steps= max_agent_steps,
                    context_len    = context_len,
                    instances_path        = instances_path,
                    sweagent_default_config = sweagent_default_config,
                    sweagent_config_path  = sweagent_config_path,
                    sweagent_model_name   = sweagent_model_name,
                    sweagent_model_api_base = sweagent_model_api_base,
                    sweagent_model_api_key = sweagent_model_api_key,
                    sweagent_max_input_tokens = sweagent_max_input_tokens,
                    sweagent_num_workers  = sweagent_num_workers,
                    sweagent_instances_type = sweagent_instances_type,
                    sweagent_redo_existing = sweagent_redo_existing,
                    sweagent_shuffle      = sweagent_shuffle,
                )
                t0 = time.time()
                batch_results = harness.run_all()
                batch_wall = round(time.time() - t0, 2)
                batch_times.append(batch_wall)
                all_results.extend(batch_results)
                log.info(f"    batch {batch_idx+1}/{n_batches}: {len(batch_results)} instances  {batch_wall:.1f}s")

            n = max(len(all_results), 1)
            def mn(attr):
                return round(sum(getattr(r, attr, 0) or 0 for r in all_results) / n, 3)
            def pk(attr):
                return round(max((getattr(r, attr, 0) or 0) for r in all_results), 3)

            tpot_now = mn("tpot_mean_ms")
            if baseline_tpot == 0 and tpot_now > 0:
                baseline_tpot = tpot_now
            miss_penalty = round(max(0.0, tpot_now - baseline_tpot), 2)

            import statistics as _stats
            batch_wall_mean = round(_stats.mean(batch_times), 2)
            batch_wall_p99  = round(sorted(batch_times)[min(int(len(batch_times) * 0.99), len(batch_times)-1)], 2)
            total_wall = sum(batch_times)
            req_per_s = round(n / max(total_wall, 0.001), 3)
            tok_per_s = round(mn("output_tokens") * n / max(total_wall, 0.001), 1)

            ai_op_fields = _ai_op_phase_fields(all_results, kv_bpt)

            sp = BatchSweepPoint(
                batch_size             = batch_size,
                inference_concurrency  = batch_size,
                kv_per_request_gb      = round(kv_req_gb, 3),
                kv_pool_capacity_gb    = round(pool_gb, 1),
                pool_fill_at_batch_pct = pool_fill_pct,
                batch_overflows_pool   = overflows,
                batch_wall_time_s_mean = batch_wall_mean,
                batch_wall_time_s_p99  = batch_wall_p99,
                ttft_mean_ms           = mn("ttft_ms"),
                ttft_p99_ms            = pk("ttft_ms"),
                tpot_mean_ms           = tpot_now,
                tpot_p99_ms            = pk("tpot_p99_ms"),
                kv_miss_penalty_ms     = miss_penalty,
                hbm_peak_gb_mean       = mn("hbm_peak_gb"),
                hbm_used_gb_mean       = mn("hbm_used_gb_end"),
                hbm_util_pct_mean      = mn("hbm_util_pct"),
                hbm_prefill_delta_gb_mean  = mn("hbm_prefill_delta_gb"),
                hbm_kv_evicted_gb_mean     = mn("hbm_kv_evicted_gb"),
                dram_hicache_staging_gb_mean = mn("dram_hicache_staging_gb"),
                ssd_read_bw_mb_mean    = mn("read_bw_mb_mean"),
                ssd_write_bw_mb_mean   = mn("write_bw_mb_mean"),
                ssd_read_iops_mean     = mn("read_iops_mean"),
                ssd_r_await_p99_ms     = pk("r_await_ms_p99"),
                ssd_util_pct_mean      = mn("util_pct_mean"),
                bio_lat_p99_us_mean    = mn("bio_lat_p99_us"),
                op_prefill_count       = ai_op_fields["op_prefill_count"],
                op_decode_count        = ai_op_fields["op_decode_count"],
                pf_hbm_delta_gb_mean       = ai_op_fields["pf_hbm_delta_gb_mean"],
                pf_rt_compute_tokens_mean  = ai_op_fields["pf_rt_compute_tokens_mean"],
                pf_cache_hit_pct           = ai_op_fields["pf_cache_hit_pct"],
                pf_ssd_eviction_gb_mean    = ai_op_fields["pf_ssd_eviction_gb_mean"],
                dc_rt_decode_tokens_mean   = ai_op_fields["dc_rt_decode_tokens_mean"],
                dc_kv_restored_tokens_mean = ai_op_fields["dc_kv_restored_tokens_mean"],
                dc_ssd_restore_gb_mean     = ai_op_fields["dc_ssd_restore_gb_mean"],
                dc_miss_penalty_ms         = ai_op_fields["dc_miss_penalty_ms"],
                throughput_req_s       = req_per_s,
                throughput_tok_s       = tok_per_s,
                prompt_tokens_mean     = mn("prompt_tokens"),
                output_tokens_mean     = mn("output_tokens"),
                batches_run            = n_batches,
                instances_run          = n,
                resolved_pct           = round(sum(r.resolved for r in all_results) / n * 100, 1),
            )
            sweep_results.append(sp)
            log.info(
                f"  bs={batch_size}  TTFT={sp.ttft_mean_ms:.0f}ms  TPOT={sp.tpot_mean_ms:.0f}ms  "
                f"pool_fill={pool_fill_pct:.0f}%  SSD_r={sp.ssd_read_bw_mb_mean:.1f}MB/s")

    finally:
        smart_m = smart.stop()
        power_m = power.stop()
        if sweep_results:
            sweep_results[-1].waf             = smart_m.get("waf", 0.0)
            sweep_results[-1].host_written_gb = smart_m.get("host_written_gb", 0.0)
            sweep_results[-1].temp_peak_c     = smart_m.get("temp_peak_c", 0)

    csv_path = output_dir / "batch_sweep.csv"
    if sweep_results:
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(sweep_results[0].to_dict().keys()))
            w.writeheader()
            w.writerows(r.to_dict() for r in sweep_results)
        log.info(f"\nBatch sweep CSV: {csv_path}")

    return sweep_results
