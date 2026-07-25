"""
bench_vllm.py — vLLM server metrics adapter for AMOprof.

vLLM exposes Prometheus metrics on /metrics. This module polls that endpoint
and normalizes the metric names into the canonical sglang:* keys that the rest
of AMOprof (writer.py, report/amoprof.py, cli.py) already consumes for SGLang.
This lets vLLM runs — including LMCache SSD-offload configurations — reuse the
existing AMOprof pipeline with minimal changes.

Canonical mapping
─────────────────
vllm:num_requests_running              → sglang:num_running_reqs
vllm:num_requests_waiting              → sglang:num_queue_reqs
vllm:kv_cache_usage_perc               → sglang:token_usage
vllm:prompt_tokens_total               → sglang:prompt_tokens_total
vllm:generation_tokens_total           → sglang:generation_tokens_total
vllm:prefix_cache_hit_rate             → sglang:cache_hit_rate
vllm:num_preemptions_total             → sglang:num_preemptions_total
vllm:time_to_first_token_seconds_*     → sglang:time_to_first_token_seconds_*
vllm:inter_token_latency_seconds_*     → sglang:inter_token_latency_seconds_*
vllm:e2e_request_latency_seconds_*     → sglang:e2e_request_latency_seconds_*
vllm:request_queue_time_seconds_*      → sglang:queue_time_seconds_*

Raw vllm:* keys are also kept in each scrape snapshot so the original data is
preserved for debugging.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger("amoprof.vllm")


# ── Mapping from vLLM / LMCache metric names to canonical SGLang-style keys ───
# Only the base name is mapped; suffixes like _sum / _count / _bucket are kept.
_VLLM_TO_SGLANG_BASE: dict[str, str | list[str]] = {
    "vllm:num_requests_running":     "sglang:num_running_reqs",
    "vllm:num_requests_waiting":     "sglang:num_queue_reqs",
    "vllm:kv_cache_usage_perc":      "sglang:token_usage",
    "vllm:prompt_tokens_total":      ["sglang:prompt_tokens_total",
                                        "sglang:realtime_tokens_total[mode=prefill_compute]"],
    "vllm:generation_tokens_total":  ["sglang:generation_tokens_total",
                                        "sglang:realtime_tokens_total[mode=decode]"],
    "vllm:prefix_cache_hit_rate":    "sglang:cache_hit_rate",
    "vllm:num_preemptions_total":    "sglang:num_preemptions_total",
    "vllm:time_to_first_token_seconds":     "sglang:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds":     "sglang:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds":     "sglang:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds":      "sglang:queue_time_seconds",
    # LMCache lookup hit rate is the user-visible KV-cache hit ratio (tokens
    # found / tokens checked). retrieve_hit_rate is almost always 1.0 because
    # LMCache only retrieves what it already found, so we keep it as a
    # secondary/debug metric.
    "lmcache:lookup_hit_rate":       "sglang:cache_hit_rate",
    "lmcache:retrieve_hit_rate":     "sglang:cache_hit_rate_retrieve",
}

# Label keys we want to preserve in the flattened key name.
_PRIORITY_LABELS = {"engine", "model_name", "finished_reason", "worker_id"}

# Prometheus counter-name suffixes. Metrics ending with these are summed across
# LMCache workers; everything else is averaged.
_COUNTER_SUFFIXES = ("_total", "_created", "_count", "_sum", "_bucket")

# Suffix for per-worker raw LMCache keys stored in each scrape snapshot.
_LMCACHE_WORKER_KEY_SUFFIX = "_lmcache_worker_raw"


def _map_metric_name(name: str) -> list[str]:
    """Return the canonical sglang:* name(s) if known; otherwise empty list."""
    # Strip any trailing _sum / _count / _bucket / _created suffix to find base.
    base = name
    for suffix in ("_sum", "_count", "_bucket", "_created"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    mapped = _VLLM_TO_SGLANG_BASE.get(base)
    if mapped is None:
        return []
    if isinstance(mapped, str):
        mapped = [mapped]
    # Re-attach suffix.
    suffix = name[len(base):]
    return [m + suffix for m in mapped]


def _label_key(labels: str) -> str:
    """Build a compact label suffix from a Prometheus label string."""
    if not labels:
        return ""
    label_pairs = re.findall(r'(\w+)="([^"]+)"', labels)
    kept = [(k, v) for k, v in label_pairs if k in _PRIORITY_LABELS]
    if not kept and label_pairs:
        kept = [label_pairs[0]]
    return ",".join(f'{k}="{v}"' for k, v in kept)


def _ingest_metric(raw_name: str, labels: str, value: float,
                   result: dict[str, float]) -> None:
    """Store one Prometheus metric (raw + canonical aliases) in result."""
    label_suffix = _label_key(labels)
    raw_key = f'{raw_name}[{label_suffix}]' if label_suffix else raw_name
    result[raw_key] = value

    for mapped_name in _map_metric_name(raw_name):
        if mapped_name == raw_name:
            continue
        mapped_key = f'{mapped_name}[{label_suffix}]' if label_suffix else mapped_name
        result[mapped_key] = value
        # Also emit a label-free alias for keys the operation classifier and
        # some report code expect unlabeled.
        result[mapped_name] = value


def _is_counter(name: str) -> bool:
    """Heuristic: Prometheus counters end with known suffixes."""
    return any(name.endswith(s) for s in _COUNTER_SUFFIXES)


def _discover_lmcache_worker_ports(host: str, start_port: int,
                                   max_workers: int = 16,
                                   timeout: float = 2.0) -> list[int]:
    """
    LMCache binds its internal API server as:
      start_port       -> scheduler (no lmcache:* metrics)
      start_port + 1   -> worker 0
      start_port + 2   -> worker 1
      ...
    Probe worker ports until one does not respond or has no lmcache metrics.
    """
    worker_ports: list[int] = []
    for offset in range(1, max_workers + 1):
        port = start_port + offset
        url = f"http://{host}:{port}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
                if "lmcache:" in raw:
                    worker_ports.append(port)
                else:
                    # Port responded but isn't an LMCache worker; stop scanning.
                    break
        except Exception:
            break
    return worker_ports


def _scrape_lmcache_worker(host: str, port: int,
                           timeout: float = 2.0) -> dict[str, float]:
    """Scrape a single LMCache worker /metrics endpoint.

    Returns unlabeled base metric names (e.g. lmcache:retrieve_hit_rate) so
    that values from multiple workers can be summed or averaged.
    """
    url = f"http://{host}:{port}/metrics"
    metrics: dict[str, float] = {}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            for line in raw.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                m = re.match(
                    r'^\s*([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)',
                    line,
                )
                if not m:
                    continue
                try:
                    raw_name = m.group(1)
                    value = float(m.group(3))
                except ValueError:
                    continue
                if not raw_name.startswith("lmcache:"):
                    continue
                # Keep only unlabeled base names for cross-worker aggregation.
                metrics[raw_name] = value
    except Exception:
        return {}
    return metrics


def _aggregate_lmcache_metrics(worker_results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics from multiple LMCache workers."""
    if not worker_results:
        return {}

    all_names = set()
    for wr in worker_results:
        all_names.update(wr.keys())

    aggregated: dict[str, float] = {}
    for name in all_names:
        values = [wr[name] for wr in worker_results if name in wr]
        if not values:
            continue
        if _is_counter(name):
            aggregated[name] = sum(values)
        else:
            aggregated[name] = sum(values) / len(values)

    # Compute overall hit rates from summed counters when available. This is
    # more accurate than averaging per-worker gauges.
    requested = aggregated.get("lmcache:num_requested_tokens_total", 0.0)
    hit = aggregated.get("lmcache:num_hit_tokens_total", 0.0)
    if requested > 0:
        aggregated["lmcache:retrieve_hit_rate"] = hit / requested

    lookup_tokens = aggregated.get("lmcache:num_lookup_tokens_total", 0.0)
    lookup_hits = aggregated.get("lmcache:num_lookup_hits_total", 0.0)
    if lookup_tokens > 0:
        aggregated["lmcache:lookup_hit_rate"] = lookup_hits / lookup_tokens

    return aggregated


def _parse_vllm_cache_config(raw_lines: list[str]) -> dict[str, float]:
    """
    Parse vllm:cache_config_info to extract num_gpu_blocks and block_size.
    Returns a dict that may contain num_gpu_blocks, block_size, and
    kv_pool_capacity_tokens.
    """
    info: dict[str, float] = {}
    for line in raw_lines:
        if not line.strip() or line.startswith("#"):
            continue
        m = re.match(
            r'^\s*vllm:cache_config_info\{([^}]*)\}\s+([\d.eE+\-]+)',
            line,
        )
        if not m:
            continue
        labels = m.group(1)
        value = float(m.group(2))
        if value != 1.0:
            continue
        pairs = dict(re.findall(r'(\w+)="([^"]+)"', labels))
        try:
            if "num_gpu_blocks" in pairs and pairs["num_gpu_blocks"] not in ("", "None"):
                info["num_gpu_blocks"] = float(pairs["num_gpu_blocks"])
            if "block_size" in pairs and pairs["block_size"] not in ("", "None"):
                info["block_size"] = float(pairs["block_size"])
        except ValueError:
            continue
        break
    if "num_gpu_blocks" in info and "block_size" in info:
        info["kv_pool_capacity_tokens"] = info["num_gpu_blocks"] * info["block_size"]
    return info


def _fetch_metrics_once(port: int, host: str = "127.0.0.1",
                        lmcache_port: Optional[int] = None,
                        lmcache_host: Optional[str] = None,
                        lmcache_bytes_per_token: Optional[float] = None,
                        lmcache_max_disk_gb: Optional[float] = None,
                        debug: bool = False,
                        debug_path: Optional[str] = None) -> dict:
    """Fetch one snapshot from vLLM /metrics (and optionally LMCache /metrics)
    and normalize to sglang-style keys."""
    url = f"http://{host}:{port}/metrics"
    result: dict[str, float] = {}
    debug_lines: list[str] = []
    scrape_bytes = 0
    metric_lines = 0
    matched_lines = 0
    raw_lines: list[str] = []

    def _dbg(msg: str) -> None:
        if debug:
            debug_lines.append(msg)

    _dbg(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] fetch url={url}")
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            raw = r.read().decode("utf-8", errors="replace")
            scrape_bytes = len(raw.encode("utf-8", errors="replace"))
            raw_lines = raw.splitlines()
            lines = raw_lines
            _dbg(f"scrape_success=1 scrape_bytes={scrape_bytes} total_lines={len(lines)}")
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                metric_lines += 1
                m = re.match(
                    r'^\s*([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)',
                    line,
                )
                if not m:
                    continue
                matched_lines += 1
                try:
                    raw_name = m.group(1)
                    labels = m.group(2) or ""
                    value = float(m.group(3))
                except ValueError:
                    continue

                # Keep vLLM / SGLang / LMCache metrics; drop starlette/fastapi/python process metrics.
                if not (raw_name.startswith("vllm:") or
                        raw_name.startswith("sglang:") or
                        raw_name.startswith("lmcache:")):
                    continue

                _ingest_metric(raw_name, labels, value, result)

    except Exception as e:
        result["sglang:scrape_success"] = 0.0
        result["sglang:scrape_error"] = str(e)
        _dbg(f"scrape_success=0 error={e!r}")
    else:
        result["sglang:scrape_success"] = 1.0

    # Derive vLLM KV-cache capacity from cache_config_info so downstream code
    # can compute kv_pool_capacity_tokens / kv_used_tokens_peak.
    cache_info = _parse_vllm_cache_config(raw_lines)
    if "kv_pool_capacity_tokens" in cache_info:
        result["sglang:kv_pool_capacity_tokens"] = cache_info["kv_pool_capacity_tokens"]

    # Optionally scrape LMCache's internal /metrics endpoints. LMCache exposes
    # per-worker metrics on start_port+1, start_port+2, ... when its internal
    # API server is enabled (internal_api_server_enabled: true in LMCache config).
    lmcache_workers_scraped = 0
    if lmcache_port:
        lmcache_host = lmcache_host or host
        worker_ports = _discover_lmcache_worker_ports(lmcache_host, lmcache_port)
        _dbg(f"lmcache_discovered_ports={worker_ports}")
        worker_results: list[dict[str, float]] = []
        for wp in worker_ports:
            wr = _scrape_lmcache_worker(lmcache_host, wp)
            if wr:
                worker_results.append(wr)
                # Preserve per-worker raw snapshot for debugging.
                result[f"sglang:lmcache_worker_{wp}{_LMCACHE_WORKER_KEY_SUFFIX}"] = 1.0

        if worker_results:
            aggregated = _aggregate_lmcache_metrics(worker_results)
            for name, value in aggregated.items():
                _ingest_metric(name, "", value, result)

            # Convert LMCache disk usage (bytes) to tokens if a bytes-per-token
            # factor is provided or can be inferred.
            local_storage_bytes = aggregated.get("lmcache:local_storage_usage", 0.0)
            if local_storage_bytes > 0 and lmcache_bytes_per_token and lmcache_bytes_per_token > 0:
                storage_tokens = local_storage_bytes / lmcache_bytes_per_token
                result["sglang:kv_l3_storage_tokens"] = storage_tokens
                result["sglang:hicache_host_used_tokens"] = storage_tokens
                if lmcache_max_disk_gb and lmcache_max_disk_gb > 0:
                    max_bytes = lmcache_max_disk_gb * 1024 * 1024 * 1024
                    result["sglang:hicache_host_total_tokens"] = max_bytes / lmcache_bytes_per_token
                    result["sglang:hicache_host_fill_pct"] = min(local_storage_bytes / max_bytes, 1.0) * 100.0

            lmcache_workers_scraped = len(worker_results)
            result["sglang:lmcache_scrape_success"] = 1.0
            result["sglang:lmcache_workers_scraped"] = float(lmcache_workers_scraped)
            result["sglang:lmcache_retrieve_hit_rate"] = aggregated.get("lmcache:retrieve_hit_rate", 0.0)
            result["sglang:lmcache_lookup_hit_rate"] = aggregated.get("lmcache:lookup_hit_rate", 0.0)
        else:
            result["sglang:lmcache_scrape_success"] = 0.0
            result["sglang:lmcache_scrape_error"] = "no LMCache worker ports responded"
            _dbg("lmcache_scrape_success=0 error=no_worker_ports_responded")

    result["sglang:scrape_bytes"] = float(scrape_bytes)
    result["sglang:scrape_metric_lines"] = float(metric_lines)
    result["sglang:scrape_matched_lines"] = float(matched_lines)

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
    """Mean from histogram _sum / _count pair."""
    sum_key = f"{prefix}_sum"
    count_key = f"{prefix}_count"
    first = samples[0] if samples else {}
    last = samples[-1] if samples else {}
    d_sum = max(last.get(sum_key, 0.0) - first.get(sum_key, 0.0), 0.0)
    d_count = max(last.get(count_key, 0.0) - first.get(count_key, 0.0), 0.0)
    return round(d_sum / d_count, 6) if d_count > 0 else 0.0


class VLLMMetricsSampler:
    """Polls vLLM /metrics and normalizes keys to the SGLang-style schema."""

    def __init__(self, port: int, interval_s: float = 1.0,
                 host: str = "127.0.0.1",
                 lmcache_port: Optional[int] = None,
                 lmcache_host: Optional[str] = None,
                 lmcache_bytes_per_token: Optional[float] = None,
                 lmcache_max_disk_gb: Optional[float] = None,
                 debug: bool = False, debug_path: Optional[str] = None):
        self.port = port
        self.host = host
        self.lmcache_port = lmcache_port
        self.lmcache_host = lmcache_host or host
        self.lmcache_bytes_per_token = lmcache_bytes_per_token
        self.lmcache_max_disk_gb = lmcache_max_disk_gb
        self.interval = interval_s
        self.debug = bool(debug)
        self.debug_path = debug_path
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start = 0.0
        self._t_end = 0.0

    @property
    def prometheus_url(self) -> str:
        return f"http://{self.host}:{self.port}/metrics"

    @property
    def raw_samples(self) -> list[dict]:
        return list(self._samples)

    @property
    def elapsed_s(self) -> float:
        if self._t_end > 0 and self._t_start > 0:
            return self._t_end - self._t_start
        return 0.0

    def _fetch(self) -> dict:
        return _fetch_metrics_once(
            self.port, self.host,
            lmcache_port=self.lmcache_port,
            lmcache_host=self.lmcache_host,
            lmcache_bytes_per_token=self.lmcache_bytes_per_token,
            lmcache_max_disk_gb=self.lmcache_max_disk_gb,
            debug=self.debug,
            debug_path=self.debug_path,
        )

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._t_start = time.time()
        s = self._fetch()
        if s:
            s = dict(s)
            s["ts"] = time.time()
            self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = self._fetch()
            if s:
                s = dict(s)
                s["ts"] = time.time()
                self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        s = self._fetch()
        if s:
            s = dict(s)
            s["ts"] = time.time()
            self._samples.append(s)
        self._t_end = time.time()
        if self._thread:
            self._thread.join(timeout=3)

        if not self._samples:
            return self._empty()

        first = self._samples[0]
        last = self._samples[-1]
        n = len(self._samples)

        def peak(k):
            return max((s.get(k, 0.0) for s in self._samples), default=0.0)

        def mn(k):
            vs = [s.get(k, 0.0) for s in self._samples]
            return round(sum(vs) / max(len(vs), 1), 4)

        def delta(k):
            return max(last.get(k, 0.0) - first.get(k, 0.0), 0.0)

        # Core gauges
        n_running_mean = mn("sglang:num_running_reqs")
        n_running_peak = peak("sglang:num_running_reqs")
        n_queue_peak = peak("sglang:num_queue_reqs")
        kv_usage_peak = peak("sglang:token_usage")
        kv_usage_mean = mn("sglang:token_usage")
        cache_hit_rate = last.get("sglang:cache_hit_rate", 0.0)

        # Prefer LMCache window-accurate lookup hit rate from counter deltas.
        # lookup hit rate = tokens found / tokens checked, which is the
        # user-visible cache ratio. retrieve counters always yield ~100%.
        d_lookup_tokens = delta("lmcache:num_lookup_tokens_total")
        d_lookup_hits = delta("lmcache:num_lookup_hits_total")
        if d_lookup_tokens > 0:
            cache_hit_rate = d_lookup_hits / d_lookup_tokens

        # Fall back to vLLM prefix cache counters if no LMCache lookup data.
        if cache_hit_rate == 0.0:
            q = delta("vllm:prefix_cache_queries_total")
            h = delta("vllm:prefix_cache_hits_total")
            if q > 0:
                cache_hit_rate = h / q

        # Counters
        prompt_tok = delta("sglang:prompt_tokens_total")
        gen_tok = delta("sglang:generation_tokens_total")
        preemptions = delta("sglang:num_preemptions_total")

        # Server-side latency histograms (seconds → ms)
        ttft_mean_s = _histogram_mean(self._samples, "sglang:time_to_first_token_seconds")
        itl_mean_s = _histogram_mean(self._samples, "sglang:inter_token_latency_seconds")
        e2e_mean_s = _histogram_mean(self._samples, "sglang:e2e_request_latency_seconds")
        queue_mean_s = _histogram_mean(self._samples, "sglang:queue_time_seconds")

        # AI op classification: vLLM doesn't expose realtime token modes, so we
        # infer from running queue / generation counters.
        if n_running_peak == 0 and n_queue_peak == 0:
            op = "idle"
        elif prompt_tok > gen_tok * 2:
            op = "prefill"
        elif gen_tok > 0:
            # vLLM does not split reasoning vs decode; call it decode.
            op = "decode"
        else:
            op = "active"

        # KV-cache capacity / usage from vLLM cache_config_info.
        kv_pool_capacity = int(last.get("sglang:kv_pool_capacity_tokens", 0.0))
        kv_used_tokens_peak = round(kv_usage_peak * kv_pool_capacity, 0) if kv_pool_capacity > 0 else 0.0

        # LMCache-derived L3 storage / HiCache fields.
        kv_l3_storage_tokens = last.get("sglang:kv_l3_storage_tokens", 0.0)
        hicache_host_used_tokens = last.get("sglang:hicache_host_used_tokens", 0.0)
        hicache_host_total_tokens = last.get("sglang:hicache_host_total_tokens", 0.0)
        hicache_host_fill_pct = last.get("sglang:hicache_host_fill_pct", 0.0)

        # LMCache counter deltas for movement/eviction metrics.
        kv_evicted_tokens = int(delta("lmcache:local_cpu_evict_keys_count_total"))
        kv_restored_tokens = int(delta("lmcache:num_hit_tokens_total"))
        kv_prefetched_tokens = int(delta("lmcache:num_lookup_tokens_total"))

        # New-token ratio: decode tokens / total generated+prefill tokens.
        total_tok = prompt_tok + gen_tok
        new_token_ratio = gen_tok / total_tok if total_tok > 0 else 0.0

        return {
            "ai_op_type":              op,
            "ai_op_prefill_tok_s":     round(prompt_tok / max(n, 1), 2),
            "ai_op_decode_tok_s":      round(gen_tok / max(n, 1), 2),
            "kv_cache_hit_rate_pct":   round(cache_hit_rate * 100, 1),
            "num_running_req_mean":    round(n_running_mean, 2),

            "token_usage_peak":        round(kv_usage_peak * 100, 1),
            "kv_pool_capacity_tokens": kv_pool_capacity,
            "kv_used_tokens_peak":     kv_used_tokens_peak,

            "kv_l1_device_tokens":     int(kv_used_tokens_peak),
            "kv_l2_host_tokens":       0,  # local_cpu is false in this LMCache config
            "kv_l3_storage_tokens":    int(kv_l3_storage_tokens),
            "hicache_host_used_tokens": int(hicache_host_used_tokens),
            "hicache_host_total_tokens": int(hicache_host_total_tokens),
            "hicache_host_fill_pct":   round(hicache_host_fill_pct, 2),

            "kv_evicted_tokens":       kv_evicted_tokens,
            "kv_restored_tokens":      kv_restored_tokens,
            "kv_prefetched_tokens":    kv_prefetched_tokens,

            "hicache_eviction_ms":     0.0,  # not exposed by LMCache metrics
            "hicache_load_back_ms":    0.0,  # not exposed by LMCache metrics
            "hicache_queue_time_ms":   round(queue_mean_s * 1000, 2),

            "rt_prefill_compute_tokens": int(prompt_tok),
            "rt_prefill_cache_tokens":   0,
            "rt_decode_tokens":          int(gen_tok),
            "cache_hit_rate_realtime_pct": round(cache_hit_rate * 100, 1),
            "new_token_ratio_mean":      round(new_token_ratio * 100, 2),

            "server_ttft_ms":          round(ttft_mean_s * 1000, 2),
            "server_itl_ms":           round(itl_mean_s * 1000, 2),
            "server_e2e_ms":           round(e2e_mean_s * 1000, 2),

            "num_queue_reqs_peak":     int(n_queue_peak),
            "decode_sum_seq_lens":     0,
            "utilization_mean":        round(n_running_mean * 10, 1),  # proxy
            "num_preemptions":         int(preemptions),

            "num_samples":             n,
        }

    def _empty(self) -> dict:
        return {
            "ai_op_type": "unknown", "ai_op_prefill_tok_s": 0.0,
            "ai_op_decode_tok_s": 0.0, "kv_cache_hit_rate_pct": 0.0,
            "num_running_req_mean": 0.0, "token_usage_peak": 0.0,
            "kv_pool_capacity_tokens": 0, "kv_used_tokens_peak": 0,
            "kv_l1_device_tokens": 0, "kv_l2_host_tokens": 0,
            "kv_l3_storage_tokens": 0, "hicache_host_used_tokens": 0,
            "hicache_host_total_tokens": 0, "hicache_host_fill_pct": 0.0,
            "kv_evicted_tokens": 0, "kv_restored_tokens": 0, "kv_prefetched_tokens": 0,
            "hicache_eviction_ms": 0.0, "hicache_load_back_ms": 0.0,
            "hicache_queue_time_ms": 0.0,
            "rt_prefill_compute_tokens": 0, "rt_prefill_cache_tokens": 0,
            "rt_decode_tokens": 0, "cache_hit_rate_realtime_pct": 0.0,
            "new_token_ratio_mean": 0.0,
            "server_ttft_ms": 0.0, "server_itl_ms": 0.0, "server_e2e_ms": 0.0,
            "num_queue_reqs_peak": 0, "decode_sum_seq_lens": 0,
            "utilization_mean": 0.0, "num_preemptions": 0,
            "num_samples": 0,
        }
