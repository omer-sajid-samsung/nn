
"""Shared AMOprof report KPI calculations.

This module is the single source of truth for common report tiles used by
Executive, Interactive, and End Report.  Report renderers should not recompute
these independently because small differences in fallback order, rounding, or
label names create confusing cross-tab mismatches.
"""
from __future__ import annotations

import csv
import html as _html
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        try:
            x = float(v)
        except Exception:
            m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(v).replace(",", ""))
            if not m:
                return default
            x = float(m.group(0))
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _read_json(path: Path) -> dict:
    try:
        if path.exists() and path.stat().st_size > 0:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Some regression fixtures and a few collector logs can contain literal
        # backslash-n sequences after shell/json escaping. Treat them as lines
        # when there are no real line breaks beyond the final terminator.
        if "\\n" in text and text.count("\n") <= 1:
            text = text.replace("\\n", "\n")
        return list(csv.DictReader(text.splitlines()))
    except Exception:
        return []



def _amoprof_resolve_raw_dir(raw_dir: Path) -> Path:
    """Resolve analyzer input to the actual raw metrics directory.

    `collect --output-dir X` writes user-friendly summary/copy files in X and
    the full collector payload in X/raw.  Executive/Interactive/Static must use
    X/raw when present, even if X also contains copied sglang/gpu summaries.
    The older resolver returned X whenever X had *any* key file; that made
    Executive miss Intel PCM files under X/raw and report DRAM PMU missing while
    End Report had DRAM sections populated.
    """
    p = Path(raw_dir)
    child = p / "raw"
    raw_marker_files = (
        "all_timeseries.csv", "setup_details.json", "server_info.json",
        "pcm_summary.json", "pcm_timeseries.csv", "pcm_memory_raw.csv",
        "amduprof_pcm_summary.json", "amduprof_pcm_timeseries.csv",
        "amduprof_pcm_raw.csv", "dram_summary.json", "dram_timeseries.csv",
        "sglang_summary.json", "sglang_timeseries.csv",
        "gpu_summary.json", "gpu_timeseries.csv",
    )
    try:
        # If the caller already handed us a raw/ directory, do not descend.
        if p.name == "raw":
            return p
        if child.is_dir() and any((child / k).exists() for k in raw_marker_files):
            return child
        # Legacy wrappers sometimes add metrics_run_*/raw under the supplied dir.
        candidates = []
        for pat in ("metrics_run_*/raw", "*/metrics_run_*/raw", "*/raw"):
            for cand in p.glob(pat):
                if cand.is_dir() and any((cand / k).exists() for k in raw_marker_files):
                    try:
                        mt = max((cand / k).stat().st_mtime for k in raw_marker_files if (cand / k).exists())
                    except Exception:
                        mt = cand.stat().st_mtime
                    candidates.append((mt, cand))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    except Exception:
        pass
    return p


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _metric_stem_norm(name: str) -> str:
    """Normalize only the metric stem, stripping Prometheus label suffixes.

    Examples:
      sglang_gen_throughput[engine_type=unified] -> sglanggenthroughput
      sglang:gen_throughput{engine_type="unified"} -> sglanggenthroughput
    """
    stem = re.split(r"[\[{]", str(name), maxsplit=1)[0]
    return _norm_name(stem)


def _find_col(rows: list[dict[str, Any]], *partials: str) -> str:
    """Find a metric column without loose substring matching.

    Exact/suffix matching is performed on the metric stem only. This preserves
    v1.39.48's regression guard against helper/debug substring matches while
    correctly handling Prometheus-labelled columns such as
    `sglang_gen_throughput[engine_type=unified]`.
    """
    if not rows:
        return ""
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    wanted = [_norm_name(p) for p in partials if p]

    for k in keys:
        nk = _metric_stem_norm(k)
        if any(nk == w for w in wanted):
            return k

    for k in keys:
        nk = _metric_stem_norm(k)
        if any(w and nk.endswith(w) for w in wanted):
            return k
    return ""


def _delta(rows: list[dict[str, Any]], *partials: str) -> float:
    col = _find_col(rows, *partials)
    if not col or not rows:
        return 0.0
    first = None
    last = None
    for r in rows:
        x = _safe_float(r.get(col), None)  # type: ignore[arg-type]
        if x is None:
            continue
        if first is None:
            first = x
        last = x
    if first is None or last is None:
        return 0.0
    d = float(last) - float(first)
    return d if d > 0 else 0.0


def _ratio_delta_ms(rows: list[dict[str, Any]], sum_name: str, count_name: str) -> float:
    ds = _delta(rows, sum_name)
    dc = _delta(rows, count_name)
    if ds > 0 and dc > 0:
        return ds / dc * 1000.0
    return 0.0


def _active_values(rows: list[dict[str, Any]], *partials: str) -> list[float]:
    col = _find_col(rows, *partials)
    vals: list[float] = []
    if not col:
        return vals
    for r in rows:
        x = _safe_float(r.get(col), 0.0)
        if x > 0:
            vals.append(x)
    return vals


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def _pct_json_value(pct_ts: dict, metric_key: str, percentile: str = "p50") -> float:
    """Return active-window average of percentile samples.

    These samples are generated from Prometheus histogram_quantile when the
    histogram is available. They are preferred over ratio fallback p50 values.
    """
    try:
        block = (pct_ts or {}).get(metric_key) or {}
        vals = block.get(percentile) or []
        clean = []
        for v in vals:
            x = _safe_float(v, 0.0)
            if x > 0:
                clean.append(x)
        return _mean(clean)
    except Exception:
        return 0.0


def _summary_first(summary: dict, *keys: str) -> float:
    for k in keys:
        v = _safe_float(summary.get(k), 0.0)
        if v > 0:
            return v
    return 0.0


def _duration_s(raw_dir: Path, rows: list[dict[str, Any]], summary: dict) -> float:
    sj = _read_json(raw_dir / "summary.json")
    meta = sj.get("meta") if isinstance(sj.get("meta"), dict) else {}
    for v in [meta.get("duration_s"), summary.get("duration_s"), summary.get("elapsed_s")]:
        x = _safe_float(v, 0.0)
        if x > 0:
            return x
    # Try timestamp/time_sec range in rows.
    for cands in [("time_sec",), ("timestamp",), ("ts",)]:
        col = _find_col(rows, *cands)
        if col:
            vals = [_safe_float(r.get(col), 0.0) for r in rows]
            vals = [v for v in vals if v > 0]
            if len(vals) >= 2 and max(vals) > min(vals):
                return max(vals) - min(vals)
    return 0.0



def _rows_from_any(rows: Any) -> list[dict[str, Any]]:
    """Return row dictionaries from either CSV rows or a pandas DataFrame."""
    if rows is None:
        return []
    try:
        # pandas.DataFrame path without importing pandas into this small helper.
        if hasattr(rows, "to_dict"):
            out = rows.to_dict("records")
            return out if isinstance(out, list) else []
    except Exception:
        pass
    if isinstance(rows, list):
        return rows
    return []


def _key_has_parts(key: str, metric: str, parts: tuple[str, ...]) -> bool:
    stem = _metric_stem_norm(key)
    wanted = _norm_name(metric)
    if not (stem == wanted or stem.endswith(wanted)):
        return False
    nk = _norm_name(key)
    return all(_norm_name(p) in nk for p in parts if p)


def _find_labelled_col(rows: list[dict[str, Any]], metric: str, *label_parts: str) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    # Label-specific match first.
    for k in keys:
        if _key_has_parts(k, metric, tuple(label_parts)):
            return k
    # If no labels requested, use the generic helper.
    if not label_parts:
        return _find_col(rows, metric)
    return ""


def _delta_col(rows: list[dict[str, Any]], col: str) -> float:
    if not col or not rows:
        return 0.0
    first = None
    last = None
    for r in rows:
        x = _safe_float(r.get(col), None)  # type: ignore[arg-type]
        if x is None:
            continue
        if first is None:
            first = x
        last = x
    if first is None or last is None:
        return 0.0
    d = float(last) - float(first)
    return d if d > 0 else 0.0


def _delta_labelled(rows: list[dict[str, Any]], metric: str, *label_parts: str) -> float:
    col = _find_labelled_col(rows, metric, *label_parts)
    return _delta_col(rows, col)


def _walk_json_dicts(obj: Any):
    """Yield dictionaries in a decoded JSON object, depth-first."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json_dicts(v)


def _discover_request_cache_from_json(raw_dir: Path) -> dict[str, float | str]:
    """Aggregate request/response JSON cache-hit fields when available.

    SGLang request responses may carry `meta_info.cached_tokens` and
    `meta_info.prompt_tokens`, where prompt_tokens is the uncached prompt work
    requiring prefill.  OpenAI-compatible responses may carry
    `usage.prompt_tokens_details.cached_tokens` and `usage.prompt_tokens`, where
    prompt_tokens is total prompt tokens.  We prefer request JSON because it is
    the closest semantic match to the benchmark's per-run cache hit rate.
    """
    out = {
        "meta_cached_tokens": 0.0,
        "meta_uncached_prompt_tokens": 0.0,
        "meta_records": 0.0,
        "usage_cached_tokens": 0.0,
        "usage_total_prompt_tokens": 0.0,
        "usage_records": 0.0,
        "source": "",
    }
    if not raw_dir or not raw_dir.exists():
        return out
    skip_names = {
        "sglang_summary.json", "gpu_summary.json", "summary.json", "common_kpis.json",
        "pcm_summary.json", "pcm_memory_summary.json", "dram_summary.json",
        "smart_summary.json", "setup_details.json", "server_info.json",
        "sglang_percentiles_timeseries.json", "amoprof_source_policy.json",
    }
    patterns = ["*response*.json", "*request*.json", "*result*.json", "*bench*.json", "*.jsonl"]
    candidates: list[Path] = []
    try:
        for pat in patterns:
            for fp in raw_dir.glob(pat):
                if fp.is_file() and fp.name not in skip_names and fp not in candidates:
                    candidates.append(fp)
        # Also allow one level under common benchmark/output dirs.
        for sub in ("responses", "requests", "benchmark", "bench", "output", "outputs"):
            d = raw_dir / sub
            if d.is_dir():
                for fp in d.glob("*.json*"):
                    if fp.is_file() and fp not in candidates:
                        candidates.append(fp)
    except Exception:
        candidates = []
    for fp in candidates[:256]:
        try:
            if fp.stat().st_size <= 0 or fp.stat().st_size > 100 * 1024 * 1024:
                continue
            objs = []
            if fp.suffix.lower() == ".jsonl":
                for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        objs.append(json.loads(line))
                    except Exception:
                        continue
            else:
                objs = [json.loads(fp.read_text(encoding="utf-8", errors="replace"))]
            for obj in objs:
                for d in _walk_json_dicts(obj):
                    mi = d.get("meta_info") if isinstance(d.get("meta_info"), dict) else None
                    if mi is not None and ("cached_tokens" in mi) and ("prompt_tokens" in mi):
                        c = _safe_float(mi.get("cached_tokens"), 0.0)
                        u = _safe_float(mi.get("prompt_tokens"), 0.0)
                        if c >= 0 and u >= 0 and (c + u) > 0:
                            out["meta_cached_tokens"] += c
                            out["meta_uncached_prompt_tokens"] += u
                            out["meta_records"] += 1
                            out["source"] = str(fp)
                    usage = d.get("usage") if isinstance(d.get("usage"), dict) else None
                    if usage is not None:
                        ptd = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
                        c = _safe_float(ptd.get("cached_tokens"), 0.0)
                        total = _safe_float(usage.get("prompt_tokens"), 0.0)
                        if c > 0 and total > 0:
                            out["usage_cached_tokens"] += c
                            out["usage_total_prompt_tokens"] += total
                            out["usage_records"] += 1
                            out["source"] = str(fp)
        except Exception:
            continue
    return out


def _discover_benchmark_cache_hit(raw_dir: Path) -> float:
    """Return benchmark aggregate cache_hit_rate percent if a summary exists."""
    if not raw_dir or not raw_dir.exists():
        return 0.0
    names = [
        "bench_summary.json", "benchmark_summary.json", "sglang_bench_summary.json",
        "sglang_bench.json", "bench_serving_output.json", "lc_bm_results.json",
    ]
    for name in names:
        p = raw_dir / name
        if not p.exists() or p.stat().st_size <= 0:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            vals = []
            for d in _walk_json_dicts(obj):
                if "cache_hit_rate" in d:
                    vals.append(_safe_float(d.get("cache_hit_rate"), 0.0))
            vals = [v for v in vals if v > 0]
            if vals:
                # JSON benchmark uses fraction 0..1 in SGLang; tolerate percent.
                v = vals[-1]
                return (v * 100.0) if v <= 1.5 else v
        except Exception:
            continue
    return 0.0


def compute_cache_hit_kpis(raw_dir: Path | None = None,
                           rows: Any | None = None,
                           summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical cache-hit calculation used by all report tabs.

    Primary KPI semantics:
      cache-served prompt/prefill tokens /
      (cache-served prompt/prefill tokens + compute-served prompt/prefill tokens)

    Source priority:
      1. Request/response JSON meta_info.cached_tokens / (cached + prompt_tokens)
         where meta_info.prompt_tokens is uncached prefill work.
      2. OpenAI-compatible usage.prompt_tokens_details.cached_tokens /
         usage.prompt_tokens where prompt_tokens is total prompt tokens.
      3. Benchmark aggregate cache_hit_rate if supplied.
      4. Prometheus realtime prefill modes:
         Δrealtime_tokens_total{mode=prefill_cache} /
         (Δprefill_cache + Δprefill_compute).
      5. Diagnostics/fallbacks only: request counters, cached/prompt counter,
         and sglang_cache_hit_rate gauge.
    """
    rd = _amoprof_resolve_raw_dir(Path(raw_dir)) if raw_dir is not None else None
    summary = summary or (_read_json(rd / "sglang_summary.json") if rd is not None else {})
    rr = _rows_from_any(rows)
    if not rr and rd is not None:
        rr = _read_csv_rows(rd / "sglang_timeseries.csv")

    # Request/response JSON exact semantics.
    req = _discover_request_cache_from_json(rd) if rd is not None else {}
    meta_cached = _safe_float(req.get("meta_cached_tokens") if req else 0.0, 0.0)
    meta_uncached = _safe_float(req.get("meta_uncached_prompt_tokens") if req else 0.0, 0.0)
    meta_pct = (meta_cached / (meta_cached + meta_uncached) * 100.0) if (meta_cached + meta_uncached) > 0 else 0.0

    usage_cached = _safe_float(req.get("usage_cached_tokens") if req else 0.0, 0.0)
    usage_total = _safe_float(req.get("usage_total_prompt_tokens") if req else 0.0, 0.0)
    usage_pct = (usage_cached / usage_total * 100.0) if usage_total > 0 else 0.0

    bench_pct = _discover_benchmark_cache_hit(rd) if rd is not None else 0.0

    # Prometheus realtime prefill modes: benchmark-equivalent when request JSON
    # is unavailable.
    d_rt_cache = (_delta_labelled(rr, "sglang_realtime_tokens_total", "mode=prefill_cache") or
                  _delta_labelled(rr, "realtime_tokens_total", "mode=prefill_cache") or
                  _delta_labelled(rr, "sglang:realtime_tokens_total", "mode=prefill_cache"))
    d_rt_compute = (_delta_labelled(rr, "sglang_realtime_tokens_total", "mode=prefill_compute") or
                    _delta_labelled(rr, "realtime_tokens_total", "mode=prefill_compute") or
                    _delta_labelled(rr, "sglang:realtime_tokens_total", "mode=prefill_compute"))
    prefill_pct = (d_rt_cache / (d_rt_cache + d_rt_compute) * 100.0) if (d_rt_cache + d_rt_compute) > 0 else 0.0

    # Diagnostics.  Do not use cached/prompt as primary because prompt_tokens_total
    # may mean total prompt tokens, while cached_tokens_total and realtime prefill
    # counters can represent different token streams across SGLang versions.
    d_cached = _delta(rr, "sglang_cached_tokens_total") or _delta(rr, "cached_tokens_total")
    d_prompt = _delta(rr, "sglang_prompt_tokens_total") or _delta(rr, "prompt_tokens_total")
    cached_prompt_pct = (d_cached / d_prompt * 100.0) if d_prompt > 0 else 0.0

    effective_prompt_pct = 0.0
    if (d_cached + d_rt_cache + d_rt_compute) > 0:
        effective_prompt_pct = ((d_cached + d_rt_cache) / (d_cached + d_rt_cache + d_rt_compute)) * 100.0

    d_req_hit = (_delta(rr, "sglang_request_cache_hit_total") or _delta(rr, "request_cache_hit_total") or
                 _delta(rr, "cache_hit_request_total") or _delta(rr, "request_cache_hits_total") or
                 _delta(rr, "cache_hit_requests_total"))
    d_req_total = (_delta(rr, "sglang_request_total") or _delta(rr, "request_total") or _delta(rr, "requests_total"))
    request_pct = (d_req_hit / d_req_total * 100.0) if d_req_total > 0 else 0.0

    ch_vals_all: list[float] = []
    ch_vals_active: list[float] = []
    ch_col = _find_col(rr, "sglang_cache_hit_rate") or _find_col(rr, "cache_hit_rate")
    if ch_col:
        for r in rr:
            v = _safe_float(r.get(ch_col), 0.0)
            # normalize fractions to percent later by looking at max.
            ch_vals_all.append(v)
            if v > 0:
                ch_vals_active.append(v)
    scale = 100.0 if ch_vals_all and max(ch_vals_all) <= 1.5 else 1.0
    gauge_timeline_pct = _mean(ch_vals_all) * scale if ch_vals_all else 0.0
    gauge_active_pct = _mean(ch_vals_active) * scale if ch_vals_active else 0.0
    gauge_peak_pct = max(ch_vals_all) * scale if ch_vals_all else 0.0

    candidates = [
        (meta_pct, "request_response_meta_info_cached_over_cached_plus_uncached_prompt_tokens"),
        (usage_pct, "openai_usage_cached_tokens_over_total_prompt_tokens"),
        (bench_pct, "benchmark_summary_cache_hit_rate"),
        (prefill_pct, "prefill_token_work_avoidance_prefill_cache_over_prefill_cache_plus_compute"),
    ]
    primary = 0.0
    method = "unavailable"
    for val, meth in candidates:
        val = max(0.0, min(100.0, _safe_float(val, 0.0)))
        if val > 0:
            primary = val
            method = meth
            break
    if primary <= 0:
        # Fallbacks are intentionally labelled diagnostics; they are better than
        # rendering 0, but not semantically benchmark-comparable.
        for val, meth in [
            (request_pct, "diagnostic_request_weighted_request_cache_hit_over_request_total"),
            (cached_prompt_pct, "diagnostic_cached_tokens_over_prompt_tokens_total"),
            (gauge_timeline_pct, "diagnostic_time_weighted_avg_over_time_sglang_cache_hit_rate"),
            (gauge_active_pct, "diagnostic_active_mean_sglang_cache_hit_rate_idle_excluded"),
            (effective_prompt_pct, "diagnostic_combined_cached_plus_prefill_cache_over_cached_plus_prefill_cache_plus_compute"),
        ]:
            val = max(0.0, min(100.0, _safe_float(val, 0.0)))
            if val > 0:
                primary = val
                method = meth
                break

    return {
        "cache_hit_pct": round(primary, 6),
        "cache_hit_rate_realtime_pct": round(primary, 6),
        "cache_hit_primary_pct": round(primary, 6),
        "cache_hit_calc_method": method,
        "cache_hit_method": method,
        "cache_hit_primary_formula": "cache_served_prefill_prompt_tokens / (cache_served_prefill_prompt_tokens + compute_served_prefill_prompt_tokens)",
        "cache_hit_request_json_pct": round(meta_pct, 6),
        "cache_hit_request_json_cached_tokens": round(meta_cached, 3),
        "cache_hit_request_json_uncached_prompt_tokens": round(meta_uncached, 3),
        "cache_hit_openai_usage_pct": round(usage_pct, 6),
        "cache_hit_openai_usage_cached_tokens": round(usage_cached, 3),
        "cache_hit_openai_usage_total_prompt_tokens": round(usage_total, 3),
        "cache_hit_benchmark_pct": round(bench_pct, 6),
        "cache_hit_prefill_token_weighted_pct": round(max(0.0, min(100.0, prefill_pct)), 6),
        "cache_hit_prefill_cache_tokens": round(d_rt_cache, 3),
        "cache_hit_prefill_compute_tokens": round(d_rt_compute, 3),
        "cache_hit_token_weighted_pct": round(max(0.0, min(100.0, cached_prompt_pct)), 6),
        "cache_hit_cached_prompt_pct": round(max(0.0, min(100.0, cached_prompt_pct)), 6),
        "cache_hit_effective_prompt_pct": round(max(0.0, min(100.0, effective_prompt_pct)), 6),
        "cache_hit_request_weighted_pct": round(max(0.0, min(100.0, request_pct)), 6),
        "cache_hit_time_weighted_pct": round(max(0.0, min(100.0, gauge_timeline_pct)), 6),
        "cache_hit_gauge_overall_pct": round(max(0.0, min(100.0, gauge_timeline_pct)), 6),
        "cache_hit_gauge_active_pct": round(max(0.0, min(100.0, gauge_active_pct)), 6),
        "cache_hit_gauge_peak_pct": round(max(0.0, min(100.0, gauge_peak_pct)), 6),
        "cache_hit_gauge_total_samples": len(ch_vals_all),
        "cache_hit_gauge_active_samples": len(ch_vals_active),
        "cache_hit_token_weighted_numerator_tokens": round(d_cached, 3),
        "cache_hit_token_weighted_denominator_tokens": round(d_prompt, 3),
        "cache_hit_request_weighted_numerator_requests": round(d_req_hit, 3),
        "cache_hit_request_weighted_denominator_requests": round(d_req_total, 3),
        "cache_hit_json_source": str(req.get("source", "")) if req else "",
    }

def compute_common_kpis(raw_dir: Path) -> dict[str, Any]:
    """Compute canonical common KPI values once for all report tabs."""
    raw_dir = _amoprof_resolve_raw_dir(Path(raw_dir))
    rows = _read_csv_rows(raw_dir / "sglang_timeseries.csv")
    pct_ts = _read_json(raw_dir / "sglang_percentiles_timeseries.json")
    summary = _read_json(raw_dir / "sglang_summary.json")

    out: dict[str, Any] = {
        "source": "amoprof.report.common_kpis.compute_common_kpis",
        "raw_dir": str(raw_dir),
    }

    # Selected-window mean latencies from cumulative counter deltas.
    out["ttft_ms"] = (
        _ratio_delta_ms(rows, "sglang_time_to_first_token_seconds_sum", "sglang_time_to_first_token_seconds_count")
        or _ratio_delta_ms(rows, "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count")
        or _summary_first(summary, "server_ttft_ms", "ttft_ms", "mean_ttft_ms")
    )
    out["tpot_ms"] = (
        _ratio_delta_ms(rows, "sglang_inter_token_latency_seconds_sum", "sglang_inter_token_latency_seconds_count")
        or _ratio_delta_ms(rows, "inter_token_latency_seconds_sum", "inter_token_latency_seconds_count")
        or _summary_first(summary, "server_itl_ms", "server_tpot_ms", "tpot_ms", "itl_ms", "mean_tpot_ms")
    )
    out["e2e_ms"] = (
        _ratio_delta_ms(rows, "sglang_e2e_request_latency_seconds_sum", "sglang_e2e_request_latency_seconds_count")
        or _ratio_delta_ms(rows, "e2e_request_latency_seconds_sum", "e2e_request_latency_seconds_count")
        or _summary_first(summary, "server_e2e_ms", "e2e_ms", "mean_e2e_ms")
    )

    # Percentile values from histogram_quantile-generated percentile file first.
    out["ttft_p50_ms"] = (
        _pct_json_value(pct_ts, "ttft", "p50")
        or _summary_first(summary, "server_ttft_p50_ms", "ttft_p50_ms", "sess_ttft_p50_ms")
    )
    out["tpot_p50_ms"] = (
        _pct_json_value(pct_ts, "itl", "p50")
        or _pct_json_value(pct_ts, "tpot", "p50")
        or _summary_first(summary, "server_itl_p50_ms", "server_tpot_p50_ms", "tpot_p50_ms", "sess_tpot_p50_ms")
    )
    out["e2e_p50_ms"] = (
        _pct_json_value(pct_ts, "e2e", "p50")
        or _summary_first(summary, "server_e2e_p50_ms", "e2e_p50_ms", "sess_e2e_p50_ms")
    )

    # Throughput: keep mean / p50 / peak from one canonical source.
    #
    # Prefer active `sglang_gen_throughput[...]` samples when present because
    # those are the actual selected-window decode-throughput trace used in the
    # interactive chart. Summary fields are fallback only. Peak is never allowed
    # to be below mean or p50.
    tp_vals = (
        _active_values(rows, "sglang_gen_throughput")
        or _active_values(rows, "gen_throughput")
        or _active_values(rows, "generation_throughput")
    )
    if tp_vals:
        out["throughput_mean"] = _mean(tp_vals)
        out["throughput_p50"] = _median(tp_vals)
        out["throughput_peak"] = max(tp_vals)
        out["throughput_method"] = "active_sglang_gen_throughput_samples"
    else:
        tp_summary_mean = _summary_first(summary, "gen_tp_active_mean", "gen_tp_mean", "throughput_mean", "ai_op_decode_tok_s")
        tp_summary_p50 = _summary_first(summary, "gen_tp_active_p50", "gen_tp_p50", "throughput_p50")
        tp_summary_peak = _summary_first(summary, "gen_tp_peak", "throughput_peak", "peak_throughput_tok_s")
        out["throughput_mean"] = tp_summary_mean
        out["throughput_p50"] = tp_summary_p50 or tp_summary_mean
        out["throughput_peak"] = tp_summary_peak
        out["throughput_method"] = "summary_fallback"

    if out["throughput_mean"] <= 0:
        dur = _duration_s(raw_dir, rows, summary)
        gen_delta = _delta(rows, "sglang_generation_tokens_total") or _delta(rows, "generation_tokens_total")
        if gen_delta > 0 and dur > 0:
            out["throughput_mean"] = gen_delta / dur
            out["throughput_p50"] = out["throughput_p50"] or out["throughput_mean"]
            out["throughput_method"] = "generation_token_counter_fallback"

    out["throughput_peak"] = max(out["throughput_peak"], out["throughput_mean"], out["throughput_p50"])


    # Cache hit: single canonical function used by Executive, Interactive, and End Report.
    # The reader-facing primary KPI is cache-served prefill/prompt tokens divided
    # by total cache-served + compute-served prefill/prompt tokens.
    out.update(compute_cache_hit_kpis(raw_dir=raw_dir, rows=rows, summary=summary))

    # Round only at render time; keep numeric values here.
    return out


def _value_text(key: str, value: float) -> str:
    if key in {"cache_hit_pct"}:
        return f"{float(value):.1f}"
    if key.startswith("tpot") or key.startswith("throughput"):
        return f"{float(value):.1f}"
    # TTFT/E2E can be very large; use integer ms for cross-tab exactness.
    return f"{float(value):.0f}"


def _unit_for_key(key: str) -> str:
    if key in {"cache_hit_pct"}:
        return "%"
    if key.startswith("throughput"):
        return "tok/s"
    if key.startswith("tpot"):
        return "ms"
    if key.startswith("ttft") or key.startswith("e2e"):
        return "ms"
    return ""


def _note_for_key(key: str, kpis: dict[str, Any]) -> str:
    if key == "cache_hit_pct":
        return str(kpis.get("cache_hit_method") or "Primary cache-hit KPI: cache-served prefill/prompt tokens over cache+compute prefill/prompt tokens")
    if key.endswith("_p50_ms"):
        return "Canonical common KPI: histogram percentile when available; shared by Executive, Interactive, and End Report"
    if key.endswith("_ms"):
        return "Canonical common KPI: selected-window Δsum/Δcount; shared by Executive, Interactive, and End Report"
    if key.startswith("throughput"):
        return "Canonical common KPI: active throughput samples/counter fallback; shared by Executive, Interactive, and End Report"
    return "Canonical common KPI shared by Executive, Interactive, and End Report"


def _replace_kpi_value_by_label(report_html: str, label_variants: list[str], key: str,
                                value: float, kpis: dict[str, Any],
                                canonical_label: str | None = None) -> str:
    out = report_html
    unit = _unit_for_key(key)
    val_txt = _value_text(key, value)
    note = _note_for_key(key, kpis)
    unit_span = f'<span style="font-size:13px;color:#94a3b8;margin-left:4px">{_html.escape(unit)}</span>' if unit else ""

    for label in label_variants:
        repl_label = canonical_label or label

        # Standard KPI card/tile with semantic classes: Executive + Interactive.
        pat = re.compile(
            r'(<div[^>]*class=["\'][^"\']*\bkpi-label\b[^"\']*["\'][^>]*>)' + re.escape(label) + r'(</div>\s*'
            r'<div[^>]*class=["\'][^"\']*\bkpi-value\b[^"\']*["\'][^>]*>)(.*?)(</div>\s*'
            r'<div[^>]*class=["\'][^"\']*\bkpi-note\b[^"\']*["\'][^>]*>)(.*?)(</div>)',
            re.S | re.I,
        )

        def repl(m):
            return f"{m.group(1)}{repl_label}{m.group(2)}{val_txt} {unit}{m.group(4)}{_html.escape(note)}{m.group(6)}"

        out = pat.sub(repl, out)

        # Inline styled End Report cards. Some tiles (notably Throughput peak)
        # have no note in the raw static report, so the note block must be
        # optional and must not consume the next KPI card.
        pat2 = re.compile(
            r'(<div\s+style=["\'][^"\']*background:#ffffff;[^"\']*min-width:120px[^"\']*["\']>\s*'
            r'<div\s+style=["\'][^"\']*font-size:10px;[^"\']*text-transform:uppercase[^"\']*["\']>)'
            + re.escape(label) +
            r'(</div>\s*<div\s+style=["\'][^"\']*font-size:20px;[^"\']*["\']>)(.*?)(</div>)'
            r'(\s*<div\s+style=["\'][^"\']*font-size:10px;[^"\']*["\']>.*?</div>)?(\s*</div>)',
            re.S | re.I,
        )

        def repl2(m):
            note_div = f'<div style="font-size:10px;color:#64748b;margin-top:2px">{_html.escape(note)}</div>'
            return f"{m.group(1)}{repl_label}{m.group(2)}{val_txt}{unit_span}{m.group(4)}{note_div}{m.group(6)}"

        out = pat2.sub(repl2, out)

    return out


def _insert_missing_throughput_peak_tile(report_html: str, kpis: dict[str, Any]) -> str:
    """Add a Throughput peak tile to Executive/Interactive if only mean/p50 exist."""
    if "Throughput peak" in report_html:
        return report_html
    peak = _safe_float(kpis.get("throughput_peak"), 0.0)
    if peak <= 0:
        return report_html
    val_txt = _value_text("throughput_peak", peak)
    unit = _unit_for_key("throughput_peak")
    note = _html.escape(_note_for_key("throughput_peak", kpis))

    # Executive card style.
    pat_exec = re.compile(
        r'(<div class="kpi">\s*<div class="kpi-label">Throughput p50</div>\s*'
        r'<div class="kpi-value">.*?</div>\s*<div class="kpi-note">.*?</div>\s*</div>)',
        re.S,
    )
    exec_tile = (
        f'<div class="kpi"><div class="kpi-label">Throughput peak</div>'
        f'<div class="kpi-value">{val_txt} {unit}</div><div class="kpi-note">{note}</div></div>'
    )
    if pat_exec.search(report_html):
        return pat_exec.sub(r'\1' + exec_tile, report_html, count=1)

    # Interactive tile style.
    pat_int = re.compile(
        r'(<div class="kpi-tile"[^>]*>\s*<div class="kpi-label">Throughput p50</div>\s*'
        r'<div class="kpi-value"[^>]*>.*?</div>\s*<div class="kpi-note">.*?</div>\s*</div>)',
        re.S,
    )
    int_tile = (
        f'<div class="kpi-tile" title="Canonical common KPI peak throughput">'
        f'<div class="kpi-label">Throughput peak</div>'
        f'<div class="kpi-value" style="color:#0f172a">{val_txt}<span class="kpi-unit">{unit}</span></div>'
        f'<div class="kpi-note">{note}</div></div>'
    )
    if pat_int.search(report_html):
        return pat_int.sub(r'\1' + int_tile, report_html, count=1)

    return report_html


def apply_common_kpis_to_html(report_html: str, raw_dir: Path | None = None,
                              kpis: dict[str, Any] | None = None) -> str:
    """Apply canonical common KPI values to a report tab's HTML."""
    if not report_html:
        return report_html
    if kpis is None:
        if raw_dir is None:
            return report_html
        kpis = compute_common_kpis(Path(raw_dir))
    specs = [
        ("cache_hit_pct", ["Cache hit", "Cache Hit"], "Cache Hit"),
        ("ttft_ms", ["TTFT mean"], "TTFT mean"),
        ("ttft_p50_ms", ["TTFT p50"], "TTFT p50"),
        ("tpot_ms", ["TPOT / ITL mean", "TPOT mean", "ITL mean"], "TPOT / ITL mean"),
        ("tpot_p50_ms", ["TPOT / ITL p50", "TPOT p50", "ITL p50"], "TPOT / ITL p50"),
        ("e2e_ms", ["E2E mean"], "E2E mean"),
        ("e2e_p50_ms", ["E2E p50"], "E2E p50"),
        ("throughput_mean", ["Throughput mean"], "Throughput mean"),
        ("throughput_p50", ["Throughput p50"], "Throughput p50"),
        ("throughput_peak", ["Throughput peak", "Peak throughput", "Peak"], "Throughput peak"),
    ]
    out = report_html
    for key, labels, canonical in specs:
        v = _safe_float(kpis.get(key), 0.0)
        if v > 0:
            out = _replace_kpi_value_by_label(out, labels, key, v, kpis, canonical)
    out = _insert_missing_throughput_peak_tile(out, kpis)
    return out


def write_common_kpis_json(raw_dir: Path, kpis: dict[str, Any] | None = None) -> Path:
    raw_dir = _amoprof_resolve_raw_dir(Path(raw_dir))
    kpis = kpis or compute_common_kpis(raw_dir)
    out = raw_dir / "common_kpis.json"
    try:
        out.write_text(json.dumps(kpis, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return out
