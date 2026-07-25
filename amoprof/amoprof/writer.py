"""
amoprof.writer.py — Build amoprof-compatible files from AMOprof collector outputs.

amoprof (bundled at amoprof.tools.amoprof) reads a `raw/` directory with these
canonical files:

    sglang_timeseries.csv      one row per scrape, metrics as columns
                               key format: metric_name  or  metric_name[label=val,...]
    sglang_summary.json        scalar summary (TTFT, TPOT, cache hit, ...)
    gpu_timeseries.csv         time_sec, gpu_idx, gpu_util, mem_used, mem_total,
                               power, temp_c
    gpu_summary.json
    vmstat_timeseries.csv      time_sec, pgfault, pgmajfault, pgpgin, pgpgout,
                               pswpin, pswpout, swap_free_mb, mem_avail_mb
    vmstat_summary.json
    power_timeseries.csv       time_sec, source, gpu_power
    nvme_driver_timeseries.csv time_sec + iostat-style fields
    amduprof_pcm_raw.csv       passthrough (already in this format)
    blkparse_events.generated.csv  produced by BlktraceCollector

AMOprof's per-collector CSVs (`<name>_timeseries.csv`) follow a different
schema (single flattened row layout with `time_sec`/`collector` columns).
This module synthesizes the canonical amoprof files from the per-collector
summaries and raw JSONL streams.
"""
from __future__ import annotations

import csv as _csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("amoprof.amoprof.writer")


# ─────────────────────────────────────────────────────────────────────────────
#  SGLang time-series + summary
# ─────────────────────────────────────────────────────────────────────────────
def write_sglang_files(samples: list[dict],
                       raw_dir: Path,
                       prometheus_source: str = "",
                       elapsed_s: float = 0.0,
                       model_name: str = "unknown",
                       t0: float = 0.0,
                       server_type: str = "sglang") -> tuple[Path, Path]:
    """
    Write `sglang_timeseries.csv` and `sglang_summary.json` from a list of
    Prometheus scrape snapshots.

    Each `samples[i]` is a dict {metric_key: float, ...} as returned by
    `_fetch_metrics_once` (or `_scrape_once` in amoprof). It can also have
    a 'ts' (epoch seconds) key.

    Args:
        samples: list of scrape dicts in chronological order
        raw_dir: directory to write outputs
        prometheus_source: URL string for sglang_summary.json
        elapsed_s: total collection duration
        model_name: model name to embed in summary
        t0: shared time origin (epoch seconds) used by ALL collectors.
            When provided (> 0), time_sec column is computed as
            sample.ts - t0 so SGLang timestamps align with gpu/nvme/vmstat
            timeseries on the same X-axis. When 0, falls back to the
            first-sample anchor for backward compatibility.
        server_type: "sglang" or "vllm" — recorded in the summary so the
            report knows which inference server produced the data.

    Returns: (timeseries_csv_path, summary_json_path)
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts_path = raw_dir / "sglang_timeseries.csv"
    js_path = raw_dir / "sglang_summary.json"

    if not samples:
        # Write empty stubs so amoprof doesn't error out
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("time_sec\n")
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump({"sglang_available": False,
                       "server_type": server_type,
                       "prometheus_source": prometheus_source,
                       "collection_elapsed_s": elapsed_s}, f, indent=2)
        return ts_path, js_path

    # Time origin: prefer the shared run-wide t0 so SGLang timestamps align
    # with all other collectors' time_sec values on the same axis. Fall back
    # to the first sample's ts if t0 wasn't supplied (legacy callers).
    if t0 and t0 > 0:
        base_ts = float(t0)
    else:
        base_ts = float(samples[0].get("ts", 0.0)) or float(samples[0].get("_ts", 0.0))
        if not base_ts:
            base_ts = 0.0

    # Build the column set: union of all keys across samples, excluding 'ts'.
    all_keys: set[str] = set()
    for s in samples:
        all_keys.update(k for k in s.keys() if k not in ("ts", "_ts"))
    sorted_keys = sorted(all_keys)

    # Write CSV: time_sec, iteration, <key1>, <key2>, ...
    with open(ts_path, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["time_sec", "iteration"] + sorted_keys)
        for i, s in enumerate(samples):
            ts = float(s.get("ts", s.get("_ts", 0.0)))
            t_sec = round(ts - base_ts, 3) if base_ts else round(i * 1.0, 3)
            row = [t_sec, i]
            for k in sorted_keys:
                v = s.get(k)
                if v is None:
                    row.append("")
                else:
                    row.append(v)
            w.writerow(row)

    # Build summary
    summary = _derive_sglang_summary(samples, sorted_keys, prometheus_source,
                                     elapsed_s, model_name, server_type)
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return ts_path, js_path


def _find_col(cols: Iterable[str], partial: str) -> str | None:
    """Pick the first column whose name contains `partial`."""
    for c in cols:
        if partial in c:
            return c
    return None


def _find_col_with_label(cols: Iterable[str], partial: str, label_part: str) -> str | None:
    for c in cols:
        if partial in c and label_part in c:
            return c
    return None


def _delta(samples: list[dict], col: str) -> float:
    if not col:
        return 0.0
    vals = [float(s[col]) for s in samples if col in s and s[col] is not None]
    if len(vals) < 2:
        return 0.0
    return vals[-1] - vals[0]


def _last(samples: list[dict], col: str) -> float:
    if not col:
        return 0.0
    vals = [float(s[col]) for s in samples if col in s and s[col] is not None]
    if not vals:
        return 0.0
    return vals[-1]


def _mean(samples: list[dict], col: str) -> float:
    if not col:
        return 0.0
    vals = [float(s[col]) for s in samples if col in s and s[col] is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _max(samples: list[dict], col: str) -> float:
    if not col:
        return 0.0
    vals = [float(s[col]) for s in samples if col in s and s[col] is not None]
    if not vals:
        return 0.0
    return max(vals)


def _cache_hit_gauge_stats(samples: list[dict], col: str | None) -> dict:
    """Return script-compatible cache-hit gauge stats in percent."""
    out = {"overall_pct": 0.0, "active_pct": 0.0, "peak_pct": 0.0,
           "total_samples": 0, "active_samples": 0}
    if not col:
        return out
    vals = []
    for smp in samples:
        if col in smp and smp[col] is not None:
            try:
                v = float(smp[col])
                vals.append(v * 100.0 if v <= 1.0 else v)
            except Exception:
                pass
    if not vals:
        return out
    active = [v for v in vals if v > 0]
    out["overall_pct"] = sum(vals) / len(vals)
    out["peak_pct"] = max(vals)
    out["total_samples"] = len(vals)
    out["active_samples"] = len(active)
    out["active_pct"] = (sum(active) / len(active)) if active else 0.0
    return out


def _derive_sglang_summary(samples: list[dict],
                            cols: list[str],
                            prometheus_source: str,
                            elapsed_s: float,
                            model_name: str,
                            server_type: str = "sglang") -> dict:
    """Replicates amoprof's sglang_summary derivation from the timeseries."""
    # Find canonical columns (may have label suffixes like [mode="decode"])
    ttft_sum = _find_col(cols, "time_to_first_token_seconds_sum")
    ttft_cnt = _find_col(cols, "time_to_first_token_seconds_count")
    itl_sum  = _find_col(cols, "inter_token_latency_seconds_sum")
    itl_cnt  = _find_col(cols, "inter_token_latency_seconds_count")
    e2e_sum  = _find_col(cols, "e2e_request_latency_seconds_sum")
    e2e_cnt  = _find_col(cols, "e2e_request_latency_seconds_count")
    gen_tp   = _find_col(cols, "gen_throughput")
    cache    = _find_col(cols, "cache_hit_rate")
    hi_used  = _find_col(cols, "hicache_host_used_tokens")
    hi_total = _find_col(cols, "hicache_host_total_tokens")
    kv_used  = _find_col(cols, "kv_used_tokens") or _find_col(cols, "token_usage")
    kv_avail = _find_col(cols, "kv_available_tokens")
    decode_tok = _find_col_with_label(cols, "realtime_tokens_total", "mode=decode") \
                 or _find_col_with_label(cols, "realtime_tokens_total", 'mode="decode"')
    pfc_tok    = _find_col_with_label(cols, "realtime_tokens_total", "mode=prefill_compute") \
                 or _find_col_with_label(cols, "realtime_tokens_total", 'mode="prefill_compute"')
    pfc_cache  = _find_col_with_label(cols, "realtime_tokens_total", "mode=prefill_cache") \
                 or _find_col_with_label(cols, "realtime_tokens_total", 'mode="prefill_cache"')
    evicted    = _find_col(cols, "evicted_tokens_total")
    backuped   = _find_col(cols, "backuped_tokens_total")
    loadback   = _find_col(cols, "load_back_tokens_total")
    gen_tok    = _find_col(cols, "generation_tokens_total")

    d_ttft_sum = _delta(samples, ttft_sum) if ttft_sum else 0
    d_ttft_cnt = _delta(samples, ttft_cnt) if ttft_cnt else 0
    d_itl_sum  = _delta(samples, itl_sum)  if itl_sum  else 0
    d_itl_cnt  = _delta(samples, itl_cnt)  if itl_cnt  else 0
    d_e2e_sum  = _delta(samples, e2e_sum)  if e2e_sum  else 0
    d_e2e_cnt  = _delta(samples, e2e_cnt)  if e2e_cnt  else 0

    ttft_ms = (d_ttft_sum / d_ttft_cnt * 1000) if d_ttft_cnt > 0 else 0
    tpot_ms = (d_itl_sum  / d_itl_cnt  * 1000) if d_itl_cnt  > 0 else 0
    e2e_ms  = (d_e2e_sum  / d_e2e_cnt  * 1000) if d_e2e_cnt  > 0 else 0

    # Fallback for single-request runs where counters didn't tick
    if ttft_ms == 0 and ttft_sum and ttft_cnt:
        last_cnt = _last(samples, ttft_cnt)
        if last_cnt > 0:
            ttft_ms = _last(samples, ttft_sum) / last_cnt * 1000
    if tpot_ms == 0 and itl_sum and itl_cnt:
        last_cnt = _last(samples, itl_cnt)
        if last_cnt > 0:
            tpot_ms = _last(samples, itl_sum) / last_cnt * 1000
    if e2e_ms == 0 and e2e_sum and e2e_cnt:
        last_cnt = _last(samples, e2e_cnt)
        if last_cnt > 0:
            e2e_ms = _last(samples, e2e_sum) / last_cnt * 1000

    # Second fallback: when histogram sum/count counters give 0 (e.g. SGLang
    # versions that emit empty sum/count but accumulate bucket counts), use
    # the last-scrape p50 from the bucket-derived quantile (which is computed
    # by Prometheus and still meaningful). The interactive report's
    # "Server E2E (Prom)" tile already uses this — keep mean-tile in sync.
    def _hist_p50_ms(metric_base: str) -> float:
        # Look for a percentiles-source column like
        # "sglang:<base>_seconds_bucket{le=...}" via histogram_quantile precomputed
        # at scrape time, or a precomputed *_p50_seconds column.
        for q_col in (
            f"{metric_base}_seconds_p50",
            f"{metric_base}_p50_seconds",
            f"{metric_base}_seconds_quantile0.5",
        ):
            c = _find_col(cols, q_col)
            if c:
                v = _last(samples, c)
                if v and v > 0:
                    return float(v) * 1000  # seconds → ms
        return 0.0
    if ttft_ms == 0:
        ttft_ms = _hist_p50_ms("time_to_first_token") or _hist_p50_ms("ttft")
    if tpot_ms == 0:
        tpot_ms = _hist_p50_ms("inter_token_latency") or _hist_p50_ms("itl")
    if e2e_ms == 0:
        e2e_ms = _hist_p50_ms("e2e_request_latency") or _hist_p50_ms("request_latency")

    # Production cache-hit methodology for the selected report window.
    # Primary: token-weighted counter ratio. The raw gauge is retained only
    # as a time-weighted diagnostic because SGLang overwrites it frequently.
    gstats = _cache_hit_gauge_stats(samples, cache)
    def _delta_col(c):
        return _delta(samples, c) if c else 0.0
    def _pct(num, den):
        return max(0.0, min(100.0, (num / den * 100.0))) if den and den > 0 else 0.0
    d_rt_cache = _delta_col(pfc_cache)
    d_rt_compute = _delta_col(pfc_tok)
    cache_prefill_token_weighted = _pct(d_rt_cache, d_rt_cache + d_rt_compute)
    cached_tok = _find_col(cols, "cached_tokens_total")
    prompt_tok = _find_col(cols, "prompt_tokens_total")
    d_cached = _delta_col(cached_tok)
    d_prompt = _delta_col(prompt_tok)
    cache_token_weighted = _pct(d_cached, d_prompt)
    cache_cached_prompt = cache_token_weighted
    req_hit_col = (_find_col(cols, "request_cache_hit_total") or _find_col(cols, "cache_hit_request_total") or
                   _find_col(cols, "request_cache_hits_total") or _find_col(cols, "cache_hit_requests_total"))
    req_total_col = _find_col(cols, "request_total") or _find_col(cols, "requests_total")
    d_req_hit = _delta_col(req_hit_col)
    d_req_total = _delta_col(req_total_col)
    cache_request_weighted = _pct(d_req_hit, d_req_total)
    cache_time_weighted = gstats["overall_pct"]
    cache_effective = _pct(d_cached + d_rt_cache, d_cached + d_rt_cache + d_rt_compute)
    if cache_token_weighted > 0:
        cache_pct = cache_token_weighted
        cache_method = "selected_window_token_weighted_cached_tokens_over_prompt_tokens"
    elif cache_request_weighted > 0:
        cache_pct = cache_request_weighted
        cache_method = "selected_window_request_weighted_request_cache_hit_over_request_total"
    elif gstats["total_samples"]:
        cache_pct = cache_time_weighted
        cache_method = "selected_window_time_weighted_avg_over_time_sglang_cache_hit_rate"
    elif cache_effective > 0:
        cache_pct = cache_effective
        cache_method = "combined_cached_plus_prefill_cache_prometheus_fallback"
    else:
        cache_pct = cache_prefill_token_weighted
        cache_method = "prefill_realtime_tokens_delta_prometheus_fallback" if cache_prefill_token_weighted > 0 else "none"

    hi_used_last = _last(samples, hi_used) if hi_used else 0
    hi_total_last = _last(samples, hi_total) if hi_total else 0
    hi_fill = (hi_used_last / hi_total_last * 100) if hi_total_last > 0 else 0

    return {
        "model_name":                   model_name,
        "server_type":                  server_type,
        "server_ttft_ms":               round(ttft_ms, 2),
        "server_itl_ms":                round(tpot_ms, 2),
        "server_e2e_ms":                round(e2e_ms,  2),
        "server_ttft_ms_method":         "delta_sglang_time_to_first_token_seconds_sum_over_count",
        "server_itl_ms_method":          "delta_sglang_inter_token_latency_seconds_sum_over_count",
        "server_e2e_ms_method":          "delta_sglang_e2e_request_latency_seconds_sum_over_count",
        "ai_op_decode_tok_s":           round(_mean(samples, gen_tp), 3) if gen_tp else 0,
        "cache_hit_rate_realtime_pct":  round(cache_pct, 2),
        "cache_hit_calc_method":         cache_method,
        "cache_hit_token_weighted_pct":  round(cache_token_weighted, 2),
        "cache_hit_prefill_token_weighted_pct": round(cache_prefill_token_weighted, 2),
        "cache_hit_time_weighted_pct": round(cache_time_weighted, 2),
        "cache_hit_request_weighted_pct": round(cache_request_weighted, 2),
        "cache_hit_cached_prompt_pct":   round(cache_cached_prompt, 2),
        "cache_hit_effective_prompt_pct": round(cache_effective, 2),
        "cache_hit_token_weighted_numerator_tokens": round(d_cached, 3),
        "cache_hit_token_weighted_denominator_tokens": round(d_prompt, 3),
        "cache_hit_request_weighted_numerator_requests": round(d_req_hit, 3),
        "cache_hit_request_weighted_denominator_requests": round(d_req_total, 3),
        "cache_hit_gauge_overall_pct":   round(gstats["overall_pct"], 2),
        "cache_hit_gauge_active_pct":    round(gstats["active_pct"], 2),
        "cache_hit_gauge_peak_pct":      round(gstats["peak_pct"], 2),
        "cache_hit_gauge_total_samples": gstats["total_samples"],
        "cache_hit_gauge_active_samples": gstats["active_samples"],
        "hicache_host_fill_pct":        round(hi_fill, 2),
        "hicache_host_used_tokens":     hi_used_last,
        "hicache_host_total_tokens":    hi_total_last,
        "rt_decode_tokens":             int(_last(samples, decode_tok) or _delta(samples, gen_tok)),
        "rt_prefill_compute_tokens":    int(_last(samples, pfc_tok)) if pfc_tok else 0,
        "rt_prefill_cache_tokens":      int(_last(samples, pfc_cache)) if pfc_cache else 0,
        "kv_used_tokens":               int(_last(samples, kv_used)) if kv_used else 0,
        "kv_available_tokens":          int(_last(samples, kv_avail)) if kv_avail else 0,
        "evicted_tokens_total":         int(_delta(samples, evicted)) if evicted else 0,
        "backuped_tokens_total":        int(_delta(samples, backuped)) if backuped else 0,
        "load_back_tokens_total":       int(_delta(samples, loadback)) if loadback else 0,
        "prometheus_source":            prometheus_source,
        "collection_elapsed_s":         round(elapsed_s, 2),
        "collection_samples":           len(samples),
        # Epoch of the first SGLang scrape. Recorded so `amoprof retime` can
        # re-anchor a SGLang timeseries CSV against the run-wide t0_epoch
        # stored in summary.json::meta, fixing the offset that older versions
        # of this writer introduced when it used its own anchor.
        "first_sample_epoch":           float(samples[0].get("ts",
                                                samples[0].get("_ts", 0.0))),
        "last_sample_epoch":            float(samples[-1].get("ts",
                                                samples[-1].get("_ts", 0.0))),
        "sglang_available":             True,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GPU / power / vmstat / nvme_driver timeseries
# ─────────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_gpu_timeseries(raw_dir: Path, t0: float) -> Path | None:
    """
    Build `gpu_timeseries.csv` from gpu_timeseries.jsonl.

    Expected output columns: time_sec, gpu_idx, gpu_util, mem_used, mem_total,
                             power, temp_c
    """
    src = raw_dir / "gpu_timeseries.jsonl"
    if not src.exists():
        return None
    rows = _read_jsonl(src)
    if not rows:
        return None
    dst = raw_dir / "gpu_timeseries.csv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["time_sec", "gpu_idx", "gpu_util",
                    "mem_used", "mem_total", "power", "temp_c"])
        for r in rows:
            ts = float(r.get("time_sec") or r.get("_ts") or r.get("ts") or 0.0)
            time_sec = round(ts - t0, 3) if ts > 1e9 else round(ts, 3)
            # GpuMonitor may sample per-GPU as a sub-record list or flat per-row.
            # Handle both.
            sub = r.get("per_gpu") or [r]
            if not isinstance(sub, list):
                sub = [r]
            for g in sub:
                gpu_idx = g.get("gpu_idx", g.get("gpu_id", 0))
                util    = g.get("gpu_util", g.get("util_gpu", g.get("utilization_gpu", 0)))
                # Try a broader set of mem_used keys, in priority order.
                # The canonical Prometheus collector writes `mem_used` (MB);
                # local-collect's JSONL writer typically uses `mem_used_mb`;
                # some older sources use `memory_used` or `fb_used`. Default
                # to 0 only when none of these are present.
                m_used  = g.get("mem_used",
                          g.get("mem_used_mb",
                          g.get("memory_used",
                          g.get("fb_used",
                          g.get("hbm_used_mb",
                          g.get("FB_USED", 0))))))
                m_total = g.get("mem_total",
                          g.get("mem_total_mb",
                          g.get("memory_total",
                          g.get("fb_total",
                          g.get("hbm_total_mb", 0)))))
                power   = g.get("power_w", g.get("power", g.get("power_draw", 0)))
                temp    = g.get("temp_c", g.get("temperature", g.get("gpu_temp", 0)))
                w.writerow([time_sec, gpu_idx, util, m_used, m_total, power, temp])
    return dst


def write_power_timeseries(raw_dir: Path, t0: float) -> Path | None:
    """Build `power_timeseries.csv` from power_timeseries.jsonl.

    PowerMonitor logs one row per GPU per sample, so multiple JSONL rows share
    the same timestamp. Sum the per-GPU `gpu_power` values into a single
    `gpu_power` column representing total system GPU draw per timestamp, which
    is what the report chart_power/chart_gpu_timeline both expect.
    """
    src = raw_dir / "power_timeseries.jsonl"
    if not src.exists():
        return None
    rows = _read_jsonl(src)
    if not rows:
        return None
    dst = raw_dir / "power_timeseries.csv"
    # Group rows by (time_bucket, source) and sum gpu_power across GPUs
    from collections import defaultdict
    grouped: dict[tuple[float, str], float] = defaultdict(float)
    order: list[tuple[float, str]] = []
    seen = set()
    for r in rows:
        ts = float(r.get("time_sec") or r.get("_ts") or r.get("ts") or 0.0)
        time_sec = round(ts - t0, 3) if ts > 1e9 else round(ts, 3)
        # Quantize to interval bucket to merge per-GPU readings that share a
        # sample window. nvidia-smi --loop-ms emits all GPUs in quick succession
        # with sub-second jitter, so round to nearest 0.5s.
        bucket = round(time_sec * 2) / 2.0
        src_tag = r.get("source", "nvidia-smi")
        # For non-GPU sources (e.g. ipmi system_power), keep their own value
        gp = (r.get("gpu_power_total_w")
              or r.get("total_power_w")
              or r.get("gpu_power")
              or r.get("power_w")
              or r.get("system_power")
              or 0)
        try:
            gp = float(gp)
        except (TypeError, ValueError):
            gp = 0.0
        key = (bucket, src_tag)
        if key not in seen:
            order.append(key); seen.add(key)
        grouped[key] += gp
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["time_sec", "source", "gpu_power"])
        for key in order:
            w.writerow([key[0], key[1], round(grouped[key], 2)])
    return dst


def write_vmstat_timeseries(raw_dir: Path, t0: float) -> Path | None:
    """
    Build `vmstat_timeseries.csv` from vmstat_timeseries.jsonl or swap_timeseries.csv.

    Columns: time_sec, pgfault, pgmajfault, pgpgin, pgpgout, pswpin, pswpout,
             swap_free_mb, mem_avail_mb
    """
    # Prefer the swap_timeseries.csv produced by SwapStormMonitor (real per-sample
    # rates), and add static swap_free_mb / mem_avail_mb from /proc/meminfo if
    # available. Otherwise fall back to vmstat_timeseries.jsonl.
    swap_csv = raw_dir / "swap_timeseries.csv"
    src_jsonl = raw_dir / "vmstat_timeseries.jsonl"

    dst = raw_dir / "vmstat_timeseries.csv"

    if swap_csv.exists():
        # Convert SwapStormMonitor CSV into amoprof's vmstat_timeseries layout.
        # SwapStormMonitor CSV columns:
        #   ts, pswpin_per_s, pswpout_per_s, pgmajfault_per_s,
        #   pgpgin_per_s, pgpgout_per_s, oom_kills_cum
        with open(swap_csv, encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8", newline="") as fout:
            r = _csv.DictReader(fin)
            w = _csv.writer(fout)
            w.writerow(["time_sec", "pgfault", "pgmajfault",
                        "pgpgin", "pgpgout", "pswpin", "pswpout",
                        "swap_free_mb", "mem_avail_mb"])
            for row in r:
                try:
                    ts = float(row["ts"])
                except (ValueError, KeyError):
                    continue
                time_sec = round(ts - t0, 3) if ts > 1e9 else round(ts, 3)
                w.writerow([time_sec,
                            0,  # pgfault not in swap CSV; left zero
                            row.get("pgmajfault_per_s", 0),
                            row.get("pgpgin_per_s", 0),
                            row.get("pgpgout_per_s", 0),
                            row.get("pswpin_per_s", 0),
                            row.get("pswpout_per_s", 0),
                            0,  # swap_free_mb - filled in from /proc/meminfo below
                            0])
        return dst

    if src_jsonl.exists():
        rows = _read_jsonl(src_jsonl)
        if rows:
            with open(dst, "w", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["time_sec", "pgfault", "pgmajfault",
                            "pgpgin", "pgpgout", "pswpin", "pswpout",
                            "swap_free_mb", "mem_avail_mb"])
                for r in rows:
                    ts = float(r.get("time_sec") or r.get("_ts") or r.get("ts") or 0.0)
                    time_sec = round(ts - t0, 3) if ts > 1e9 else round(ts, 3)
                    w.writerow([time_sec,
                                r.get("pgfault", 0),
                                r.get("pgmajfault", 0),
                                r.get("pgpgin", 0),
                                r.get("pgpgout", 0),
                                r.get("pswpin", 0),
                                r.get("pswpout", 0),
                                r.get("swap_free_mb", 0),
                                r.get("mem_avail_mb", 0)])
            return dst
    return None



def write_queue_depth_sources_timeseries(raw_dir: Path, t0: float) -> Path | None:
    """
    Merge every sampled queue-depth source into one canonical CSV.

    Sources:
      • queue_depth_sysfs_timeseries.jsonl / queue_depth_sources_timeseries.csv
        from BlockQueueDepthCollector
      • iostat_timeseries.jsonl avgqu/aqu-sz
      • nvme_driver_timeseries.jsonl /sys/block inflight
      • blktrace_analysis/queue_depth_timeseries.csv when exact Q→C analysis exists

    Output columns are intentionally superset-style. Reports should prefer exact
    blktrace Q→C, then iostat aqu-sz, then sysfs weighted_qd/inflight.
    """
    dst = raw_dir / "queue_depth_sources_timeseries.csv"
    rows: list[dict] = []

    def _float(v, default=0.0):
        try:
            if v is None or v == "":
                return default
            return float(v)
        except Exception:
            return default

    def _t(row):
        ts = _float(row.get("time_sec"), None)
        if ts is not None:
            return round(ts, 6)
        ts_abs = _float(row.get("ts", row.get("_ts", 0)), 0)
        return round(ts_abs - t0, 6) if ts_abs > 1e9 else round(ts_abs, 6)

    # Exact Q→C blktrace analysis, if already generated.
    qd_exact = raw_dir / "blktrace_analysis" / "queue_depth_timeseries.csv"
    if qd_exact.exists() and qd_exact.stat().st_size > 0:
        try:
            with qd_exact.open(encoding="utf-8") as fh:
                rr = _csv.DictReader(fh)
                for r in rr:
                    qd = _float(r.get("qd_total", r.get("queue_depth", 0)))
                    rows.append({
                        "time_sec": _float(r.get("t_sec", r.get("time_sec", 0))),
                        "source": "blktrace_q_to_c_exact",
                        "qd": qd,
                        "qd_best_effort": qd,
                        "qd_read": _float(r.get("qd_read", 0)),
                        "qd_write": _float(r.get("qd_write", 0)),
                    })
        except Exception:
            pass

    # Always-on sysfs collector derived CSV.
    sysfs_csv = raw_dir / "queue_depth_sources_timeseries.csv"
    if sysfs_csv.exists() and sysfs_csv.stat().st_size > 0 and sysfs_csv != dst:
        pass
    # Avoid self-read loop: if the BlockQueueDepthCollector wrote dst, read jsonl instead.
    sysfs_jsonl = raw_dir / "queue_depth_sysfs_timeseries.jsonl"
    for r in (_read_jsonl(sysfs_jsonl) if sysfs_jsonl.exists() else []):
        rows.append({
            "time_sec": _t(r),
            "source": "sysfs_weighted_qd_inflight",
            "qd": max(_float(r.get("weighted_qd")), _float(r.get("qd_best_effort")), _float(r.get("stat_inflight"))),
            "qd_best_effort": max(_float(r.get("weighted_qd")), _float(r.get("qd_best_effort")), _float(r.get("stat_inflight"))),
            "weighted_qd": _float(r.get("weighted_qd")),
            "inflight": _float(r.get("stat_inflight", r.get("sysfs_inflight", 0))),
            "inflight_reads": _float(r.get("sysfs_inflight_reads", 0)),
            "inflight_writes": _float(r.get("sysfs_inflight_writes", 0)),
            "io_util_pct": _float(r.get("io_util_pct", 0)),
            "rd_iops": _float(r.get("rd_iops", 0)),
            "wr_iops": _float(r.get("wr_iops", 0)),
            "rd_bw_mbs": _float(r.get("rd_bw_mbs", 0)),
            "wr_bw_mbs": _float(r.get("wr_bw_mbs", 0)),
        })

    # iostat avg queue size. This is a kernel-averaged queue length over the interval.
    for r in (_read_jsonl(raw_dir / "iostat_timeseries.jsonl") if (raw_dir / "iostat_timeseries.jsonl").exists() else []):
        qd = _float(r.get("avgqu", r.get("aqu-sz", r.get("aqu_sz", 0))))
        rows.append({
            "time_sec": _t(r),
            "source": "iostat_aqu_sz",
            "qd": qd,
            "qd_best_effort": qd,
            "io_util_pct": _float(r.get("util", r.get("%util", 0))),
            "rd_iops": _float(r.get("riops", r.get("r/s", 0))),
            "wr_iops": _float(r.get("wiops", r.get("w/s", 0))),
            "rd_bw_mbs": _float(r.get("rMBs", 0)),
            "wr_bw_mbs": _float(r.get("wMBs", 0)),
        })

    # NvmeDriverMonitor instantaneous /sys/block in-flight samples.
    for r in (_read_jsonl(raw_dir / "nvme_driver_timeseries.jsonl") if (raw_dir / "nvme_driver_timeseries.jsonl").exists() else []):
        qd = _float(r.get("inflight", 0))
        rows.append({
            "time_sec": _t(r),
            "source": "sysfs_stat_inflight",
            "qd": qd,
            "qd_best_effort": qd,
            "inflight": qd,
            "weighted_io_ms": _float(r.get("weighted_io_ms", 0)),
            "io_util_pct": _float(r.get("util", r.get("io_util_pct", 0))),
        })

    if not rows:
        return None

    rows.sort(key=lambda r: (_float(r.get("time_sec", 0)), str(r.get("source", ""))))
    fields = ["time_sec", "source", "qd", "qd_best_effort", "weighted_qd", "inflight",
              "inflight_reads", "inflight_writes", "qd_read", "qd_write", "io_util_pct",
              "rd_iops", "wr_iops", "rd_bw_mbs", "wr_bw_mbs", "weighted_io_ms"]
    extra = sorted({k for r in rows for k in r.keys()} - set(fields))
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return dst

def write_nvme_driver_timeseries(raw_dir: Path, t0: float) -> Path | None:
    """
    Build `nvme_driver_timeseries.csv` from iostat output preferentially,
    falling back to nvme_driver_timeseries.jsonl.

    Column contract (matches amoprof's extract_metrics nv_df2 reader):
        time_sec, rd_iops, wr_iops, rd_bw_mbs, wr_bw_mbs,
        rd_lat_ms, wr_lat_ms, io_util_pct, inflight,
        disc_ios, disc_sectors
    """
    iostat_jsonl = raw_dir / "iostat_timeseries.jsonl"
    drv_jsonl    = raw_dir / "nvme_driver_timeseries.jsonl"
    discard_csv  = raw_dir / "discard_timeseries.csv"

    dst = raw_dir / "nvme_driver_timeseries.csv"

    iostat_rows = _read_jsonl(iostat_jsonl) if iostat_jsonl.exists() else []
    drv_rows    = _read_jsonl(drv_jsonl)    if drv_jsonl.exists()    else []

    if not iostat_rows and not drv_rows and not discard_csv.exists():
        return None

    primary = iostat_rows or drv_rows
    if not primary:
        return None
    # NvmeDriverMonitor stores cumulative counters (rd_ios, wr_ios, rd_ms,
    # wr_ms, io_ms). When that's the source, derive per-window rates from
    # consecutive samples. IostatMonitor already stores rates (riops, wiops,
    # rMBs, wMBs, rawt, wawt, util), so skip derivation for it.
    is_iostat_src = bool(iostat_rows)
    if not is_iostat_src and drv_rows:
        derived = []
        prev = None
        for r in drv_rows:
            cur = dict(r)
            if prev is not None:
                dt = max(float(cur.get("ts", cur.get("_ts", 0))) -
                         float(prev.get("ts", prev.get("_ts", 0))), 1e-6)
                d_rd  = max(float(cur.get("rd_ios", 0)) - float(prev.get("rd_ios", 0)), 0)
                d_wr  = max(float(cur.get("wr_ios", 0)) - float(prev.get("wr_ios", 0)), 0)
                d_rms = max(float(cur.get("rd_ms",  0)) - float(prev.get("rd_ms",  0)), 0)
                d_wms = max(float(cur.get("wr_ms",  0)) - float(prev.get("wr_ms",  0)), 0)
                d_iom = max(float(cur.get("io_ms",  0)) - float(prev.get("io_ms",  0)), 0)
                cur["riops"] = round(d_rd / dt, 2)
                cur["wiops"] = round(d_wr / dt, 2)
                # avg completion latency per IO over this window
                cur["rawt"]  = round(d_rms / max(d_rd, 1), 3)
                cur["wawt"]  = round(d_wms / max(d_wr, 1), 3)
                # device-busy fraction (io_ms / wall_ms; clamp to 100)
                cur["util"]  = round(min(d_iom / (dt * 10), 100), 2)
                # BW unavailable from /sys/block/<dev>/stat (no byte counters
                # in pre-4.18 layout); leave to be filled by blktrace synthesis
                cur.setdefault("rMBs", 0)
                cur.setdefault("wMBs", 0)
            else:
                cur.setdefault("riops", 0); cur.setdefault("wiops", 0)
                cur.setdefault("rawt", 0);  cur.setdefault("wawt", 0)
                cur.setdefault("util", 0);  cur.setdefault("rMBs", 0)
                cur.setdefault("wMBs", 0)
            derived.append(cur); prev = r
        primary = derived

    # Load discard rates indexed by approximate time
    discards: dict[float, tuple[float, float]] = {}
    if discard_csv.exists():
        try:
            with open(discard_csv, encoding="utf-8") as fh:
                r = _csv.DictReader(fh)
                for row in r:
                    try:
                        ts = float(row["ts"])
                    except (KeyError, ValueError):
                        continue
                    t_sec = round(ts - t0, 1) if ts > 1e9 else round(ts, 1)
                    discards[t_sec] = (float(row.get("discard_iops", 0)),
                                       float(row.get("discard_mb_s", 0)) * 1024 * 2)
                                       # sectors = MB/s * 1024 KB/MB * 2 sectors/KB
        except Exception:
            pass

    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["time_sec",
                    "rd_iops", "wr_iops",
                    "rd_bw_mbs", "wr_bw_mbs",
                    "rd_lat_ms", "wr_lat_ms",
                    "io_util_pct", "inflight",
                    "disc_ios", "disc_sectors"])
        for r in primary:
            ts = float(r.get("time_sec") or r.get("_ts") or r.get("ts") or 0.0)
            time_sec = round(ts - t0, 3) if ts > 1e9 else round(ts, 3)
            # Field aliases — IostatMonitor uses rMBs/wMBs/riops/wiops/rawt/wawt/util,
            # NvmeDriverMonitor uses rd_ios/wr_ios/rd_ms/wr_ms/inflight/io_ms.
            # Accept all forms so the canonical CSV is populated either way.
            rd_iops = r.get("riops", r.get("r_iops", r.get("rd_iops", 0)))
            wr_iops = r.get("wiops", r.get("w_iops", r.get("wr_iops", 0)))
            rd_bw   = r.get("rMBs",  r.get("r_mb_s", r.get("rd_bw_mbs", 0)))
            wr_bw   = r.get("wMBs",  r.get("w_mb_s", r.get("wr_bw_mbs", 0)))
            rd_lat  = r.get("rawt",  r.get("r_await_ms", r.get("rd_lat_ms", r.get("r_await", 0))))
            wr_lat  = r.get("wawt",  r.get("w_await_ms", r.get("wr_lat_ms", r.get("w_await", 0))))
            util    = r.get("util",  r.get("util_pct", r.get("io_util_pct", r.get("%util", 0))))
            infl    = r.get("avgqu", r.get("aqu-sz", r.get("aqu_sz", r.get("queue_depth",
                            r.get("inflight", 0)))))
            disc    = discards.get(round(time_sec, 1), (0, 0))
            w.writerow([time_sec, rd_iops, wr_iops, rd_bw, wr_bw,
                        rd_lat, wr_lat, util, infl, disc[0], disc[1]])
    return dst


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry — called from cli.py after all collectors stop
# ─────────────────────────────────────────────────────────────────────────────
def write_amoprof_files(raw_dir: Path,
                        t0: float,
                        sglang_samples: list[dict] | None = None,
                        sglang_source: str = "",
                        sglang_elapsed_s: float = 0.0,
                        sglang_model: str = "unknown",
                        server_type: str = "sglang") -> dict[str, Path]:
    """
    Convenience driver — writes all amoprof-compatible files from a raw/ dir.

    Returns dict of {filename: Path} for what was actually written.
    """
    out: dict[str, Path] = {}

    if sglang_samples:
        ts, js = write_sglang_files(sglang_samples, raw_dir,
                                    prometheus_source=sglang_source,
                                    elapsed_s=sglang_elapsed_s,
                                    model_name=sglang_model,
                                    t0=t0,
                                    server_type=server_type)
        out["sglang_timeseries.csv"] = ts
        out["sglang_summary.json"] = js

    p = write_gpu_timeseries(raw_dir, t0)
    if p: out["gpu_timeseries.csv"] = p
    p = write_power_timeseries(raw_dir, t0)
    if p: out["power_timeseries.csv"] = p
    p = write_vmstat_timeseries(raw_dir, t0)
    if p: out["vmstat_timeseries.csv"] = p
    p = write_nvme_driver_timeseries(raw_dir, t0)
    if p: out["nvme_driver_timeseries.csv"] = p
    p = write_queue_depth_sources_timeseries(raw_dir, t0)
    if p: out["queue_depth_sources_timeseries.csv"] = p

    # Mirror amduprof_pcm_raw.csv as .txt — amoprof's loader loads .csv files
    # as DataFrames first, then only loads text if no DataFrame exists with the
    # same stem. The DRAM BW parser needs the raw text, so we create a sibling
    # .txt copy that the loader picks up correctly.
    amd_csv = raw_dir / "amduprof_pcm_raw.csv"
    amd_txt = raw_dir / "amduprof_pcm_raw.txt"
    if amd_csv.exists() and not amd_txt.exists():
        try:
            amd_txt.write_text(amd_csv.read_text(errors="replace"), encoding="utf-8")
            out["amduprof_pcm_raw.txt"] = amd_txt
        except Exception as e:
            log.debug("amoprof_writer: amduprof txt mirror failed: %s", e)

    # gpu_summary.json / vmstat_summary.json — derive from collector summaries
    sum_path = raw_dir.parent / "summary.json"
    if sum_path.exists():
        try:
            with open(sum_path) as f:
                full = json.load(f)
            csum = full.get("summary", {})
            if "gpu" in csum:
                _write_json(raw_dir / "gpu_summary.json", csum["gpu"])
                out["gpu_summary.json"] = raw_dir / "gpu_summary.json"
            if "vmstat" in csum:
                _write_json(raw_dir / "vmstat_summary.json", csum["vmstat"])
                out["vmstat_summary.json"] = raw_dir / "vmstat_summary.json"
        except Exception as e:
            log.debug("amoprof.writer: summary read failed: %s", e)

    return out


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
