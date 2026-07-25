"""
amoprof/percentiles.py — Fetch true per-request P50/P90/P99 latencies and
throughputs from a Prometheus server using histogram_quantile() against the
SGLang histogram bucket metrics.

WHY THIS EXISTS

SGLang exposes Prometheus histograms for the metrics that matter most:
  - sglang_time_to_first_token_seconds_bucket    (TTFT)
  - sglang_inter_token_latency_seconds_bucket    (ITL)
  - sglang_e2e_request_latency_seconds_bucket    (E2E)
  - sglang_generation_tokens_histogram_bucket    (tokens/req)
  - sglang_prompt_tokens_histogram_bucket        (prompt tokens/req)

These bucket counters preserve the full per-request distribution. By issuing
`histogram_quantile(q, sum by (le) (rate(<metric>[<window>])))` against
Prometheus, we recover P50 / P90 / P99 / Max for each metric exactly the way
bench_serving's bench_serving.py computes them — these are the same numbers
the user sees in their "Performance metrics summary" output.

WHAT THIS MODULE PROVIDES

Two public functions:

  fetch_percentile_summary(prom_url, start, end, step_s, instance=...)
      Returns a single dict of point-in-time aggregate percentiles for the
      entire window, suitable for merging into sglang_summary.json:
          {
            "server_ttft_p50_ms": 26100.0,
            "server_ttft_p90_ms": 37800.0,
            "server_ttft_p99_ms": 49200.0,
            ...
          }

  fetch_percentile_timeseries(prom_url, start, end, step_s, ...)
      Returns a per-tick timeseries of [P50, P90, P99] for each metric.
      Used by the interactive report to draw evolution-over-time charts:
          {
            "ttft_ms": {"time_sec": [...], "p50": [...], "p90": [...], "p99": [...]},
            "itl_ms":  {...},
            "e2e_ms":  {...},
          }

Both functions degrade gracefully — if the Prometheus server is unreachable,
or the histogram buckets aren't being scraped, they return {}.

The window for `rate()` defaults to 1 minute, matching what bench_serving
uses internally. For longer collection windows, callers can pass a larger
`rate_window_s` to smooth tail noise.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("amoprof.percentiles")


# Map: bench-summary canonical key → (metric base name, scale factor, "label")
# The "label" is used in chart titles and is just human-friendly.
HISTOGRAM_METRICS = [
    # canonical key, bucket metric base alias(es), scale, unit, label.
    # SGLang versions differ on token histogram names.  Latency histograms are
    # usually `sglang_*_seconds_bucket`; token histograms have been observed as
    # both `sglang_prompt_tokens_histogram_bucket` and
    # `sglang_prompt_tokens_bucket`.  Try all aliases before giving up.
    ("ttft",  "sglang_time_to_first_token_seconds", 1000.0, "ms", "TTFT"),
    ("itl",   "sglang_inter_token_latency_seconds", 1000.0, "ms", "ITL"),
    ("e2e",   "sglang_e2e_request_latency_seconds", 1000.0, "ms", "E2E"),
    ("prompt_tokens", ("sglang_prompt_tokens_histogram", "sglang_prompt_tokens"), 1.0, "tok", "Prompt tokens"),
    ("output_tokens", ("sglang_generation_tokens_histogram", "sglang_generation_tokens", "sglang_output_tokens", "sglang_completion_tokens"), 1.0, "tok", "Output tokens"),
]


def _metric_bases(metric_base: str | tuple | list) -> list[str]:
    if isinstance(metric_base, (tuple, list)):
        return [str(x) for x in metric_base if x]
    return [str(metric_base)]

PERCENTILES = [
    ("p50", 0.50),
    ("p90", 0.90),
    ("p99", 0.99),
]


def _http_get_json(url: str, params: dict, timeout: float = 30.0) -> dict | None:
    """Fetch URL with params, return parsed JSON or None on any error."""
    try:
        import requests
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("HTTP fetch failed for %s: %s", url, e)
        return None


def _build_label_selector(instance: str = "", job: str = "",
                            extra_labels: dict[str, str] | None = None) -> str:
    """Build PromQL label selector like {instance=\"X\",job=\"Y\"}.

    Returns "" if no filters. Caller is responsible for placing this
    inside the metric selector: `metric{label_selector}`.
    """
    parts: list[str] = []
    if instance:
        parts.append(f'instance="{instance}"')
    if job:
        parts.append(f'job="{job}"')
    for k, v in (extra_labels or {}).items():
        parts.append(f'{k}="{v}"')
    return "{" + ",".join(parts) + "}" if parts else ""



def _list_prom_metrics(prom_url: str) -> set[str]:
    data = _http_get_json(f"{prom_url.rstrip('/')}/api/v1/label/__name__/values", {}, timeout=10.0)
    if data and data.get("status") == "success":
        try:
            return set(str(x) for x in data.get("data", []))
        except Exception:
            return set()
    return set()


def _resolve_metric_base(base: str, available: set[str]) -> str:
    if not available:
        return base
    if base + "_bucket" in available:
        return base
    if base.startswith("sglang_"):
        colon = "sglang:" + base[len("sglang_"):]
        if colon + "_bucket" in available:
            return colon
    if base.startswith("sglang:"):
        under = "sglang_" + base[len("sglang:"):]
        if under + "_bucket" in available:
            return under
    return base


def _quantile_query_range(prom_url: str, metric_base: str, q: float,
                          start: float, end: float, step_s: int,
                          rate_window_s: int,
                          label_selector: str = "") -> dict | None:
    """Issue histogram_quantile() against query_range, return raw response.

    PromQL form:
        histogram_quantile(q,
          sum by (le) (rate(<metric_base>_bucket<labels>[<window>s])))
    """
    # `le` is the implicit bucket-upper-bound label on Prometheus histograms.
    # We sum across all dimensions OTHER than `le` so we get one combined
    # distribution across e.g. all engine types / model names.
    query = (
        f"histogram_quantile({q}, "
        f"sum by (le) "
        f"(rate({metric_base}_bucket{label_selector}[{rate_window_s}s])))"
    )
    return _http_get_json(f"{prom_url.rstrip('/')}/api/v1/query_range",
                           {"query": query, "start": start, "end": end,
                            "step": step_s})


def _quantile_query_instant_over_range(
        prom_url: str, metric_base: str, q: float,
        start: float, end: float, label_selector: str = "") -> float | None:
    """Compute a single aggregate quantile over the ENTIRE window.

    Uses an instant `histogram_quantile()` against the increase across the
    full window — this is the closest analog to what bench_serving emits in
    its per-run "P99 TTFT: 49.21" line.
    """
    window_s = max(int(end - start), 1)
    query = (
        f"histogram_quantile({q}, "
        f"sum by (le) "
        f"(increase({metric_base}_bucket{label_selector}[{window_s}s])))"
    )
    data = _http_get_json(f"{prom_url.rstrip('/')}/api/v1/query",
                           {"query": query, "time": end})
    if not data or data.get("status") != "success":
        return None
    results = data.get("data", {}).get("result", [])
    if not results:
        return None
    try:
        val = float(results[0]["value"][1])
        return val if val == val else None  # filter NaN
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def fetch_percentile_summary(prom_url: str, start: float, end: float,
                              instance: str = "", job: str = "",
                              extra_labels: dict[str, str] | None = None,
                              ) -> dict[str, float]:
    """Compute P50/P90/P99 for the FULL window. Returns flat dict like:
        {"server_ttft_p50_ms": 26100.0, "server_ttft_p90_ms": 37800.0, ...}

    Empty dict if Prometheus is unreachable or no histogram data exists.
    """
    label_selector = _build_label_selector(instance, job, extra_labels)
    available_metrics = _list_prom_metrics(prom_url)
    out: dict[str, float] = {}
    for canon, base_aliases, scale, unit, _label in HISTOGRAM_METRICS:
        bases = _metric_bases(base_aliases)
        for p_label, q in PERCENTILES:
            v = None
            for base in bases:
                base = _resolve_metric_base(base, available_metrics)
                v = _quantile_query_instant_over_range(
                    prom_url, base, q, start, end, label_selector)
                if v is not None:
                    break
            if v is None:
                continue
            # Key naming matches existing convention:
            #   server_ttft_p99_ms, server_e2e_p50_ms, etc.
            if canon in ("ttft", "itl", "e2e"):
                key = f"server_{canon}_{p_label}_ms"
            elif canon == "prompt_tokens":
                key = f"prompt_tokens_{p_label}"
            elif canon == "output_tokens":
                key = f"output_tokens_{p_label}"
            else:
                key = f"{canon}_{p_label}"
            out[key] = round(v * scale, 2)
        # Also fetch the mean for comparison
        # (sum of _sum increase) / (sum of _count increase) over window
        window_s = max(int(end - start), 1)
        for base in bases:
            base = _resolve_metric_base(base, available_metrics)
            mean_query = (
                f"sum(increase({base}_sum{label_selector}[{window_s}s])) "
                f"/ "
                f"sum(increase({base}_count{label_selector}[{window_s}s]))"
            )
            data = _http_get_json(
                f"{prom_url.rstrip('/')}/api/v1/query",
                {"query": mean_query, "time": end})
            if data and data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                if results:
                    try:
                        val = float(results[0]["value"][1])
                        if val == val:  # filter NaN
                            if canon in ("ttft", "itl", "e2e"):
                                out[f"server_{canon}_mean_ms"] = round(val * scale, 2)
                            else:
                                out[f"{canon}_mean"] = round(val * scale, 2)
                            break
                    except (ValueError, KeyError, IndexError, TypeError):
                        pass
    return out


def fetch_percentile_timeseries(prom_url: str, start: float, end: float,
                                 step_s: int = 30,
                                 rate_window_s: int | None = None,
                                 instance: str = "", job: str = "",
                                 extra_labels: dict[str, str] | None = None,
                                 ) -> dict[str, dict]:
    """Pull per-tick P50/P90/P99 timeseries from Prometheus.

    Returns a dict like:
        {
          "ttft": {
            "label": "TTFT", "unit": "ms",
            "time_sec": [0, 30, 60, ...],
            "p50": [...], "p90": [...], "p99": [...]
          },
          "itl":  {...}, "e2e": {...},
          "prompt_tokens": {...}, "output_tokens": {...},
        }

    rate_window_s defaults to max(60, step_s * 4) for stable smoothing.
    """
    if rate_window_s is None:
        rate_window_s = max(60, step_s * 4)
    label_selector = _build_label_selector(instance, job, extra_labels)
    out: dict[str, dict] = {}

    for canon, base_aliases, scale, unit, label in HISTOGRAM_METRICS:
        bases = _metric_bases(base_aliases)
        block: dict[str, Any] = {
            "label": label, "unit": unit, "metric_base": bases[0],
            "time_sec": [], "p50": [], "p90": [], "p99": [],
            "rate_window_s": int(rate_window_s),
            "rate_window": f"{int(rate_window_s)}s",
            "step_s": int(step_s),
            "method": (
                "histogram_quantile(q, sum by (le) "
                f"(rate(<metric>_bucket[{int(rate_window_s)}s])))"
            ),
        }
        # Pull each percentile timeseries. For each percentile, try all known
        # metric-name aliases before treating it as unavailable.
        any_data = False
        ts_set: list[float] | None = None
        used_base = None
        for p_label, q in PERCENTILES:
            data = None
            for base in bases:
                data = _quantile_query_range(
                    prom_url, base, q, start, end, step_s,
                    rate_window_s, label_selector)
                if data and data.get("status") == "success" and data.get("data", {}).get("result", []):
                    used_base = base
                    break
            if not data or data.get("status") != "success":
                continue
            results = data.get("data", {}).get("result", [])
            if not results:
                continue
            # We summed over `le` already, so there should be exactly 1 series
            values = results[0].get("values", [])
            if not values:
                continue
            # Each value is [unix_ts, "float_str"]
            ts_list = [float(v[0]) - start for v in values]
            try:
                val_list = [float(v[1]) * scale if v[1] not in ("NaN", "+Inf", "-Inf") else None
                            for v in values]
            except (ValueError, TypeError):
                continue
            if ts_set is None:
                ts_set = ts_list
                block["time_sec"] = ts_list
            block[p_label] = val_list
            any_data = True
        if any_data:
            if used_base:
                block["metric_base"] = used_base
            out[canon] = block
    return out
