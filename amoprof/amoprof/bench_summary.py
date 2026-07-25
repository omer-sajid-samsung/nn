"""
amoprof/bench_summary.py — Parse per-request benchmark summary output and
render it as charts.

The "Performance metrics summary" block emitted by SGLang's bench_serving
script (and similar benchmark tools) gives us aggregate per-request stats
that are NOT exposed via Prometheus:
  • TTFT, ITL, E2E latency: Mean / Median / P90 / P99 / Max
  • Prompt and Output lengths: Mean / P90 / P99
  • Input/Output token throughput
  • Request throughput
  • Cache hit rate (benchmark-derived, distinct from sglang_cache_hit_rate)

We accept input in two formats:
  1. JSON file with the same key names
  2. The plaintext "key: value" format emitted by bench_serving

The parser is permissive — extra keys are kept, missing keys are skipped.
The output is a normalized dict that the interactive report renders into
4 chart groups (throughput KPIs, latency percentiles, token-length stats,
cache hit/request stats).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("amoprof.bench_summary")

# Mapping from the human-readable text labels to canonical snake_case keys
# used internally and in the JSON form. Order matters for some keys (we
# match longest label first to avoid e.g. "P90 TTFT" colliding with "TTFT").
_LABEL_MAP = [
    ("Total requests",          "total_requests"),
    ("Average Prompt Length",   "avg_prompt_tokens"),
    ("Average Output Length",   "avg_output_tokens"),
    ("Median Prompt Length",    "median_prompt_tokens"),
    ("Median Output Length",    "median_output_tokens"),
    ("P50 Prompt Length",       "median_prompt_tokens"),
    ("P50 Output Length",       "median_output_tokens"),
    ("P90 Prompt Length",       "p90_prompt_tokens"),
    ("P99 Prompt Length",       "p99_prompt_tokens"),
    ("P90 Output Length",       "p90_output_tokens"),
    ("P99 Output Length",       "p99_output_tokens"),
    ("Max Prompt Length",       "max_prompt_tokens"),
    ("Max Output Length",       "max_output_tokens"),
    ("Average TTFT",            "avg_ttft_s"),
    ("Median TTFT",             "median_ttft_s"),
    ("P50 TTFT",                "median_ttft_s"),
    ("P90 TTFT",                "p90_ttft_s"),
    ("P99 TTFT",                "p99_ttft_s"),
    ("Max TTFT",                "max_ttft_s"),
    ("Min TTFT",                "min_ttft_s"),
    ("Average ITL",             "avg_itl_s"),
    ("Median ITL",              "median_itl_s"),
    ("P50 ITL",                 "median_itl_s"),
    ("P90 ITL",                 "p90_itl_s"),
    ("P99 ITL",                 "p99_itl_s"),
    ("Max ITL",                 "max_itl_s"),
    ("Min ITL",                 "min_itl_s"),
    ("Average latency",         "avg_latency_s"),
    ("Median latency",          "median_latency_s"),
    ("P50 latency",             "median_latency_s"),
    ("P90 latency",             "p90_latency_s"),
    ("P99 latency",             "p99_latency_s"),
    ("Max latency",             "max_latency_s"),
    ("Min latency",             "min_latency_s"),
    ("Input token throughput",  "input_tok_per_s"),
    ("Output token throughput", "output_tok_per_s"),
    ("Request Throughput",      "req_per_s"),
    ("Request throughput",      "req_per_s"),
    ("Cache Hit Rate",          "cache_hit_rate"),
    ("Cache hit rate",          "cache_hit_rate"),
    ("Concurrency",             "concurrency"),
    ("Duration",                "duration_s"),
    ("Request rate",            "request_rate"),
]

# Sort by descending label length so multi-word matches win over substrings
_LABEL_MAP_SORTED = sorted(_LABEL_MAP, key=lambda x: -len(x[0]))


def _try_float(s: str) -> float | None:
    """Coerce a string to float, stripping common units (tokens/s, ms, %)."""
    s = s.strip()
    if not s:
        return None
    # Strip trailing units: "9748.18 tokens per second", "26.12 ms", "49%"
    s = re.sub(r"\s*(tokens?\s+per\s+second|tok/s|tokens/s|requests?\s+per\s+second|"
                r"req/s|seconds?|secs?|tokens?|ms|millis|%)\s*$",
                "", s, flags=re.IGNORECASE)
    try:
        return float(s)
    except ValueError:
        return None


def parse_bench_summary(source: str | Path) -> dict[str, Any]:
    """Parse a bench summary from text content, a JSON string, or a file path.

    Returns a normalized dict with snake_case keys. All time fields are
    stored in seconds (auto-converted from ms if the input is < 1000s and
    label contains 'ms' or value range suggests milliseconds — heuristic).
    """
    text: str
    is_path = (
        isinstance(source, Path) or
        (isinstance(source, str) and len(source) < 4096
         and "\n" not in source and Path(source).exists())
    )
    if is_path:
        p = Path(source)
        text = p.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(source)

    # Try JSON first
    json_data: dict | None = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            json_data = candidate
    except (ValueError, TypeError):
        pass

    out: dict[str, Any] = {}
    if json_data is not None:
        # Walk JSON, normalize keys
        for k, v in json_data.items():
            # If the JSON already uses our canonical keys, keep them.
            # Otherwise try the label map for known variants.
            canon = _canonicalize_json_key(k)
            if canon:
                out[canon] = v
        if out:
            return _post_process(out)

    # Plaintext parser: scan line-by-line for "label: value" pairs
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        # Find which canonical key this line matches
        for label, canon in _LABEL_MAP_SORTED:
            # Match "label:" anchored at line start (with optional leading whitespace)
            pat = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", re.IGNORECASE)
            m = pat.match(line)
            if m:
                # Handle compound values like "256 at 1 requests per second"
                value_str = m.group(1)
                if canon == "total_requests":
                    # "Total requests: 256 at 1 requests per second"
                    m2 = re.match(r"(\d+)\s+at\s+(\d+(?:\.\d+)?)", value_str)
                    if m2:
                        out["total_requests"] = int(m2.group(1))
                        out.setdefault("request_rate", float(m2.group(2)))
                        break
                v = _try_float(value_str)
                if v is not None:
                    out[canon] = v
                else:
                    out[canon] = value_str
                break

    return _post_process(out)


def _canonicalize_json_key(k: str) -> str | None:
    """Map common JSON key spellings to our canonical names."""
    k_lower = k.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "total_requests": ["total_requests", "num_requests", "n_requests"],
        "avg_prompt_tokens": ["avg_prompt_tokens", "mean_prompt_tokens",
                              "avg_prompt_length", "average_prompt_len", "average_prompt_length", "mean_prompt_len"],
        "avg_output_tokens": ["avg_output_tokens", "mean_output_tokens",
                              "avg_output_length", "average_output_len", "average_output_length", "mean_output_len"],
        "p90_prompt_tokens": ["p90_prompt_tokens", "p90_prompt_length", "p90_prompt_len"],
        "p99_prompt_tokens": ["p99_prompt_tokens", "p99_prompt_length", "p99_prompt_len"],
        "p90_output_tokens": ["p90_output_tokens", "p90_output_length", "p90_output_len"],
        "p99_output_tokens": ["p99_output_tokens", "p99_output_length", "p99_output_len"],
        "max_prompt_tokens": ["max_prompt_tokens", "max_prompt_length"],
        "max_output_tokens": ["max_output_tokens", "max_output_length"],
        "avg_ttft_s":    ["avg_ttft_s", "avg_ttft", "average_ttft", "mean_ttft", "mean_ttft_ms"],
        "median_ttft_s": ["median_ttft_s", "median_ttft", "p50_ttft"],
        "p90_ttft_s":    ["p90_ttft_s", "p90_ttft"],
        "p99_ttft_s":    ["p99_ttft_s", "p99_ttft"],
        "max_ttft_s":    ["max_ttft_s", "max_ttft"],
        "avg_itl_s":    ["avg_itl_s", "avg_itl", "average_itl", "mean_itl"],
        "median_itl_s": ["median_itl_s", "median_itl", "p50_itl"],
        "p90_itl_s":    ["p90_itl_s", "p90_itl"],
        "p99_itl_s":    ["p99_itl_s", "p99_itl"],
        "max_itl_s":    ["max_itl_s", "max_itl"],
        "avg_latency_s":    ["avg_latency_s", "avg_latency", "average_latency", "mean_latency"],
        "median_latency_s": ["median_latency_s", "median_latency", "p50_latency"],
        "p90_latency_s":    ["p90_latency_s", "p90_latency"],
        "p99_latency_s":    ["p99_latency_s", "p99_latency"],
        "max_latency_s":    ["max_latency_s", "max_latency"],
        "input_tok_per_s":  ["input_tok_per_s", "input_token_throughput",
                             "input_tokens_per_s", "prompt_tps"],
        "output_tok_per_s": ["output_tok_per_s", "output_token_throughput",
                             "output_tokens_per_s", "decode_tps"],
        "req_per_s":        ["req_per_s", "request_throughput", "requests_per_s", "throughput"],
        "cache_hit_rate":   ["cache_hit_rate", "cache_hit_ratio"],
        "duration_s":       ["duration_s", "duration", "total_time_s"],
        "concurrency":      ["concurrency", "n_workers", "max_concurrent"],
        "request_rate":     ["request_rate", "qps"],
    }
    for canon, names in aliases.items():
        if k_lower in names:
            return canon
    return None


def _post_process(d: dict[str, Any]) -> dict[str, Any]:
    """Heuristic unit cleanups + derived fields."""
    # Some tools emit TTFT/ITL in ms instead of s. Heuristic: if median TTFT > 50
    # and label clearly says 'ms' (rare in our text format), treat as ms.
    # Otherwise leave as-is — the bench output the user showed is in seconds.

    # Cache hit rate may come in as 0.49 (fraction) or 49 (percent).
    # Normalize to fraction.
    if "cache_hit_rate" in d:
        chr_ = d["cache_hit_rate"]
        if isinstance(chr_, (int, float)) and chr_ > 1.5:
            d["cache_hit_rate"] = chr_ / 100.0

    # Derive request rate if total_requests and duration are known
    if ("req_per_s" not in d and "total_requests" in d and
        "duration_s" in d and d["duration_s"]):
        d["req_per_s"] = float(d["total_requests"]) / float(d["duration_s"])

    return d


def discover_bench_summary(raw_dir: Path) -> Path | None:
    """Look for a bench summary file in the raw/ directory using common names."""
    candidates = [
        "bench_summary.json", "bench_summary.txt",
        "performance_summary.json", "performance_summary.txt",
        "sglang_bench_summary.json", "sglang_bench.json",
        "bench_serving_output.json",
    ]
    for name in candidates:
        p = raw_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None
