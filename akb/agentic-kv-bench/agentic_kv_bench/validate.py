"""Post-run validation: did the run actually enter the regime we predicted?

Three independent signals:
  1. GPU KV pressure — max(vllm:kv_cache_usage_perc) from our scrapes.
  2. LMCache activity — numeric counters that moved between before/after
     snapshots (evict/retrieve/store/hit/miss keys).
  3. amoprof output non-empty (the storage-side ground truth).

Verdicts:
  CONFIRMED           predicted eviction, pressure high, LMCache moved
  LIKELY              predicted eviction, pressure high, LMCache silent/unknown
  NOT_OBSERVED        predicted eviction but pressure stayed low  -> harness bug
                      or pool misestimate; investigate, don't report
  UNEXPECTED_PRESSURE no eviction predicted but pressure hit the ceiling
  IN_REGIME           no eviction predicted, none observed (clean control run)
  INDETERMINATE       missing telemetry; say so rather than guess

The verdict NEVER changes the run's exit status. A NOT_OBSERVED run still
produced valid telemetry; the verdict is a signpost for the human reading it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .naming import RunLayout
from .util import log

_PRESSURE_RE = re.compile(
    r"^vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)\{[^}]*\}\s+([0-9.eE+-]+)\s*$",
    re.MULTILINE)
_ACTIVITY_KEYS = ("evict", "retriev", "lookup", "store", "hit", "miss", "usage", "size")

HIGH_PRESSURE = 0.95
LOW_PRESSURE = 0.90


def max_kv_pressure(vllm_metrics_path: Path) -> float | None:
    if not vllm_metrics_path.exists():
        return None
    try:
        text = vllm_metrics_path.read_text(errors="replace")
    except OSError:
        return None
    vals = [float(m) for m in _PRESSURE_RE.findall(text)]
    return max(vals) if vals else None


def _flatten_numbers(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_numbers(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def _extract_prom_numbers(text: str) -> dict:
    """name{labels} value  ->  name value (labels collapsed; last sample wins)."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def _snapshot_counters(path: Path) -> dict:
    """All numeric counters from a snapshot, regardless of endpoint format."""
    if not path.exists():
        return {}
    try:
        snap = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    counters = {}
    for ep, result in (snap.get("endpoints") or {}).items():
        body = result.get("body")
        if body is None:
            continue
        try:
            parsed = json.loads(body)
            counters.update(_flatten_numbers(parsed, prefix=ep))
        except (json.JSONDecodeError, TypeError):
            counters.update(_extract_prom_numbers(body))
    return counters


def lmcache_activity(before_path: Path, after_path: Path) -> dict:
    """Counters that increased between snapshots, filtered to activity-ish keys."""
    before, after = _snapshot_counters(before_path), _snapshot_counters(after_path)
    if not before or not after:
        return {"available": False, "deltas": {}}
    deltas = {}
    for key, after_v in after.items():
        delta = after_v - before.get(key, 0.0)
        if delta > 0 and any(tok in key.lower() for tok in _ACTIVITY_KEYS):
            deltas[key] = delta
    return {"available": True, "deltas": deltas}


def extract_perf_summary(report_dir: Path) -> dict:
    """Pull the headline latency numbers out of inference-perf's summary report."""
    summary_path = report_dir / "summary_lifecycle_metrics.json"
    if not summary_path.exists():
        matches = list(report_dir.rglob("*summary_lifecycle_metrics*.json"))
        if not matches:
            return {}
        summary_path = matches[0]
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    successes = data.get("successes") or {}
    latency = successes.get("latency") or {}

    def ms(block):
        b = latency.get(block) or {}
        return {k: round(v * 1000, 1) for k, v in b.items()
                if k in ("mean", "median", "p90", "p99") and isinstance(v, (int, float))}

    count = successes.get("count")
    return {
        "benchmark_time_s": round(data.get("benchmark_time_seconds") or 0, 1),
        "requests": count.get("count") if isinstance(count, dict) else count,
        "failures": (data.get("failures") or {}).get("count"),
        "throughput": successes.get("throughput") or {},
        "ttft_ms": ms("time_to_first_token"),
        "tpot_ms": ms("time_per_output_token"),
        "itl_ms": ms("inter_token_latency"),
        "request_latency_ms": ms("request_latency"),
    }


def validate_run(layout: RunLayout, prediction: dict, driver_status: int) -> dict:
    pressure = max_kv_pressure(layout.vllm_metrics)
    activity = lmcache_activity(layout.lmcache_before, layout.lmcache_after)
    amoprof_files = [p for p in layout.amoprof_out.rglob("*") if p.is_file()] if layout.amoprof_out.exists() else []
    perf = extract_perf_summary(layout.report)

    expected = prediction.get("expect_eviction")
    moved = bool(activity.get("deltas"))
    if expected is None:
        verdict = "INDETERMINATE"
        reason = "no prediction (pool or working set unknown)"
    elif pressure is None:
        verdict = "INDETERMINATE"
        reason = "no vllm:kv_cache_usage_perc/gpu_cache_usage_perc samples in telemetry"
    elif expected and pressure >= HIGH_PRESSURE and moved:
        verdict = "CONFIRMED"
        reason = f"predicted eviction; peak KV pressure {pressure:.0%}; LMCache counters moved"
    elif expected and pressure >= HIGH_PRESSURE:
        verdict = "LIKELY"
        reason = f"predicted eviction; peak KV pressure {pressure:.0%}; LMCache counters silent/unavailable"
    elif expected and pressure < LOW_PRESSURE:
        verdict = "NOT_OBSERVED"
        reason = (f"eviction predicted ({prediction.get('working_set_tokens')} tokens working set vs "
                  f"{prediction.get('pool_tokens')} pool) but peak KV pressure only {pressure:.0%} — "
                  "harness bug or pool misestimate; investigate before reporting")
    elif not expected and pressure >= HIGH_PRESSURE:
        verdict = "UNEXPECTED_PRESSURE"
        reason = f"no eviction predicted but peak KV pressure {pressure:.0%} — check pool estimate"
    else:
        verdict = "IN_REGIME"
        reason = f"no eviction predicted, peak KV pressure {pressure:.0%} (clean control)"

    summary = {
        "verdict": verdict,
        "verdict_reason": reason,
        "driver_status": driver_status,
        "max_kv_pressure": pressure,
        "lmcache_activity": activity,
        "amoprof_files": len(amoprof_files),
        "perf": perf,
        "prediction": prediction,
    }
    layout.run_summary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log.info("verdict: %s — %s", verdict, reason)
    if not amoprof_files:
        log.warning("amoprof output dir is empty — check logs/amoprof_service.log")
    return summary
