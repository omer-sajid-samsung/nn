"""
amoprof/bench_lc_results.py — Parse and render a long-context benchmark
results JSON file (lc_bm_results.json format) as a self-contained HTML
benchmark summary report.

File format
───────────
A JSON array of records, each representing one metric for one request-rate run:

  {
    "MT_REQUEST_RATE": "20",       // offered request rate (req/s)
    "metric":  "average_ttft",     // metric name (see METRIC_META below)
    "n":        1,                 // number of repetitions in this measurement
    "n_failed": 0,                 // failed requests
    "mean":     6.548...,          // mean value across repetitions
    "stddev":   0.0,               // std deviation (0 when n=1)
    "p50":      6.548...,          // 50th percentile
    "p95":      6.548...,          // 95th percentile
    "p99":      6.548...,          // 99th percentile
    "ci95_low": null,              // 95% confidence interval lower bound
    "ci95_high": null              // 95% confidence interval upper bound
  }

Aggregation across rates
────────────────────────
Each MT_REQUEST_RATE value represents a separate load-level run. The report
shows all load levels together as a sweep (latency vs load, throughput vs load)
to characterise the system's behaviour under increasing request pressure.

Usage (module)
──────────────
    from amoprof.bench_lc_results import parse_lc_results, build_lc_report
    records = parse_lc_results(Path("lc_bm_results.json"))
    html    = build_lc_report(records, title="LC Benchmark — DeepSeek-R1-70B")
    Path("lc_report.html").write_text(html)

Usage (CLI)
───────────
    amoprof bench-lc --input lc_bm_results.json --output lc_report.html
"""
from __future__ import annotations

import json
import math
import html as _html
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Metric metadata: display name, unit, description ────────────────────────
METRIC_META: Dict[str, Dict[str, str]] = {
    # Latency — TTFT (Time To First Token)
    "average_ttft":  {"label": "Avg TTFT",    "unit": "s",       "group": "TTFT",       "dir": "lower"},
    "median_ttft":   {"label": "Median TTFT", "unit": "s",       "group": "TTFT",       "dir": "lower"},
    "p90_ttft":      {"label": "P90 TTFT",    "unit": "s",       "group": "TTFT",       "dir": "lower"},
    "p99_ttft":      {"label": "P99 TTFT",    "unit": "s",       "group": "TTFT",       "dir": "lower"},
    "max_ttft":      {"label": "Max TTFT",    "unit": "s",       "group": "TTFT",       "dir": "lower"},
    # Latency — ITL / TPOT (Inter-Token Latency)
    "average_itl":   {"label": "Avg ITL",     "unit": "s",       "group": "ITL",        "dir": "lower"},
    "median_itl":    {"label": "Median ITL",  "unit": "s",       "group": "ITL",        "dir": "lower"},
    "p90_itl":       {"label": "P90 ITL",     "unit": "s",       "group": "ITL",        "dir": "lower"},
    "p99_itl":       {"label": "P99 ITL",     "unit": "s",       "group": "ITL",        "dir": "lower"},
    "max_itl":       {"label": "Max ITL",     "unit": "s",       "group": "ITL",        "dir": "lower"},
    # Latency — End-to-End
    "average_latency": {"label": "Avg E2E",   "unit": "s",       "group": "E2E",        "dir": "lower"},
    "median_latency":  {"label": "Median E2E","unit": "s",       "group": "E2E",        "dir": "lower"},
    "p90_latency":     {"label": "P90 E2E",   "unit": "s",       "group": "E2E",        "dir": "lower"},
    "p99_latency":     {"label": "P99 E2E",   "unit": "s",       "group": "E2E",        "dir": "lower"},
    "max_latency":     {"label": "Max E2E",   "unit": "s",       "group": "E2E",        "dir": "lower"},
    # Throughput
    "throughput":              {"label": "Req throughput",    "unit": "req/s",  "group": "Throughput", "dir": "higher"},
    "output_token_throughput": {"label": "Output tok/s",      "unit": "tok/s",  "group": "Throughput", "dir": "higher"},
    "input_token_throughput":  {"label": "Input tok/s",       "unit": "tok/s",  "group": "Throughput", "dir": "higher"},
    # Cache / misc
    "cache_hit_rate":   {"label": "Cache hit",      "unit": "%",      "group": "Cache",  "dir": "higher"},
    # Request shape
    "average_output_len": {"label": "Avg output len",  "unit": "tok",  "group": "Shape",  "dir": None},
    "average_prompt_len": {"label": "Avg prompt len",  "unit": "tok",  "group": "Shape",  "dir": None},
    "p90_output_len":     {"label": "P90 output len",  "unit": "tok",  "group": "Shape",  "dir": None},
    "p90_prompt_len":     {"label": "P90 prompt len",  "unit": "tok",  "group": "Shape",  "dir": None},
    "p99_output_len":     {"label": "P99 output len",  "unit": "tok",  "group": "Shape",  "dir": None},
    "p99_prompt_len":     {"label": "P99 prompt len",  "unit": "tok",  "group": "Shape",  "dir": None},
}

# Which metrics to show in the summary KPI grid (most important)
KPI_METRICS = [
    "average_ttft", "median_ttft", "p99_ttft",
    "average_itl",  "median_itl",  "p99_itl",
    "average_latency", "p99_latency",
    "throughput", "output_token_throughput",
    "cache_hit_rate",
]

# Metrics to plot as line charts (one line per metric, x = request rate)
CHART_GROUPS = [
    ("TTFT vs Offered Load",       ["average_ttft", "median_ttft", "p90_ttft", "p99_ttft"]),
    ("ITL / TPOT vs Offered Load", ["average_itl",  "median_itl",  "p90_itl",  "p99_itl"]),
    ("E2E Latency vs Offered Load",["average_latency","median_latency","p90_latency","p99_latency"]),
    ("Throughput vs Offered Load", ["throughput", "output_token_throughput"]),
    ("Cache Hit Rate vs Offered Load", ["cache_hit_rate"]),
]


# ── Parsing ──────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def parse_lc_results(path: Path) -> List[Dict]:
    """Load and validate a lc_bm_results.json file.  Returns the raw list."""
    raw = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")
    required = {"MT_REQUEST_RATE", "metric", "mean"}
    for i, rec in enumerate(raw):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} missing keys: {missing}")
    return raw


def aggregate_records(records: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Return a nested dict:
        pivot[rate_str][metric_name] → record dict

    When multiple records exist for the same (rate, metric) — e.g. repeated
    runs stored as separate rows — the mean of the means is used.
    """
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    raw_rec: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    for rec in records:
        rate   = str(rec.get("MT_REQUEST_RATE", "?"))
        metric = str(rec.get("metric", "?"))
        val    = _safe_float(rec.get("mean"))
        acc[rate][metric].append(val)
        raw_rec[rate][metric] = rec   # keep last for metadata

    pivot: Dict[str, Dict[str, Dict]] = {}
    for rate in acc:
        pivot[rate] = {}
        for metric in acc[rate]:
            vals = acc[rate][metric]
            base = dict(raw_rec[rate][metric])
            base["mean"] = sum(vals) / len(vals)
            base["_n_rates"] = len(vals)
            pivot[rate][metric] = base

    return pivot


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _fmt(v: float, unit: str) -> str:
    """Format a value with its unit in a human-readable way."""
    if unit == "s":
        if v >= 60:
            return f"{v/60:.1f} min"
        if v >= 1:
            return f"{v:.2f} s"
        return f"{v*1000:.0f} ms"
    if unit == "%":
        return f"{v*100:.1f}%"
    if unit in ("tok", "tok/s", "req/s"):
        if v >= 1e6:
            return f"{v/1e6:.2f}M {unit}"
        if v >= 1e3:
            return f"{v/1e3:.1f}K {unit}"
        return f"{v:.2f} {unit}"
    return f"{v:.3g} {unit}"


def _delta_badge(base_val: float, cur_val: float, direction: Optional[str]) -> str:
    """Return a coloured Δ% badge relative to the lowest-rate baseline."""
    if base_val == 0 or direction is None:
        return ""
    delta = (cur_val - base_val) / abs(base_val) * 100
    if abs(delta) < 1:
        return '<span class="badge neutral">≈0%</span>'
    sign = "+" if delta > 0 else ""
    # For lower-is-better: positive delta is bad; for higher-is-better: positive is good
    good = (delta < 0) if direction == "lower" else (delta > 0)
    cls  = "good" if good else "bad"
    return f'<span class="badge {cls}">{sign}{delta:.0f}%</span>'


def _plotly_line_chart(chart_id: str, title: str,
                        rates: List[str], series: List[Dict]) -> str:
    """Build a Plotly line chart spec as an inline script block."""
    colours = ["#6366f1","#22c55e","#f59e0b","#ef4444","#a78bfa","#34d399"]
    traces = []
    for i, s in enumerate(series):
        traces.append({
            "type": "scatter", "mode": "lines+markers",
            "name": s["name"],
            "x": [int(r) for r in rates],
            "y": s["values"],
            "line": {"color": colours[i % len(colours)], "width": 2},
            "marker": {"size": 7},
            "hovertemplate": f"<b>{_html.escape(s['name'])}</b><br>Rate: %{{x}} req/s<br>Value: %{{y:.3f}} {s.get('unit','')}<extra></extra>",
        })
    layout = {
        "title": {"text": title, "font": {"size": 14, "color": "#f1f5f9"}},
        "paper_bgcolor": "#1e293b", "plot_bgcolor": "#1e293b",
        "font": {"color": "#e2e8f0", "size": 12},
        "xaxis": {
            "title": "Offered request rate (req/s)",
            "gridcolor": "#334155", "color": "#e2e8f0", "zerolinecolor": "#475569",
            "tickmode": "array", "tickvals": [int(r) for r in rates],
        },
        "yaxis": {
            "title": series[0].get("unit", "") if len(series) == 1 else "",
            "gridcolor": "#334155", "color": "#e2e8f0", "zerolinecolor": "#475569",
        },
        "legend": {"orientation": "h", "y": -0.28, "font": {"color": "#e2e8f0"}},
        "margin": {"l": 55, "r": 15, "t": 45, "b": 80},
        "height": 310,
    }
    data_json = json.dumps({"data": traces, "layout": layout})
    return (f'<div id="{chart_id}" class="chart-box"></div>\n'
            f'<script>Plotly.newPlot("{chart_id}",'
            f' {data_json}.data, {data_json}.layout,'
            f' {{"responsive":true,"displayModeBar":false}});</script>')


# ── Main report builder ───────────────────────────────────────────────────────

def build_lc_report(records: List[Dict],
                     title: str = "Long-Context Benchmark Summary",
                     run_label: str = "",
                     extra_meta: Optional[Dict] = None) -> str:
    """Build a self-contained HTML benchmark report from parsed lc_bm_results records."""
    pivot = aggregate_records(records)
    rates = sorted(pivot.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    baseline_rate = rates[0]

    n_failed_total = sum(
        _safe_float(rec.get("n_failed", 0))
        for rate_data in pivot.values()
        for rec in rate_data.values()
    )

    # ── KPI summary table ─────────────────────────────────────────────────────
    kpi_header = ("<tr><th>Metric</th>"
                  + "".join(f"<th>Rate = {r} req/s</th>" for r in rates)
                  + "</tr>")
    kpi_rows = ""
    for metric in KPI_METRICS:
        meta = METRIC_META.get(metric, {"label": metric, "unit": "", "dir": None})
        base_val = _safe_float(pivot.get(baseline_rate, {}).get(metric, {}).get("mean"))
        row = f"<tr><td class='mname'>{_html.escape(meta['label'])}</td>"
        for ri, rate in enumerate(rates):
            rec = pivot.get(rate, {}).get(metric)
            if rec is None:
                row += "<td>—</td>"
                continue
            val  = _safe_float(rec.get("mean"))
            disp = _fmt(val, meta["unit"])
            badge = _delta_badge(base_val, val, meta["dir"]) if ri > 0 else ""
            row += f"<td>{disp} {badge}</td>"
        row += "</tr>"
        kpi_rows += row

    # ── Plotly charts ─────────────────────────────────────────────────────────
    charts_html = ""
    chart_idx   = 0
    for chart_title, metric_names in CHART_GROUPS:
        series = []
        for metric in metric_names:
            meta   = METRIC_META.get(metric, {"label": metric, "unit": "", "dir": None})
            values = [_safe_float(pivot.get(r, {}).get(metric, {}).get("mean")) for r in rates]
            if any(v > 0 for v in values):
                # Convert cache_hit_rate fraction → percentage for display
                if meta["unit"] == "%":
                    values = [v * 100 for v in values]
                series.append({"name": meta["label"], "values": values, "unit": meta["unit"]})
        if series:
            charts_html += _plotly_line_chart(
                f"lc-chart-{chart_idx}", chart_title, rates, series)
            chart_idx += 1

    # ── Saturation analysis ───────────────────────────────────────────────────
    sat_rows = ""
    # Achieved vs offered
    for rate in rates:
        offered  = int(rate)
        achieved = _safe_float(pivot.get(rate, {}).get("throughput", {}).get("mean"))
        sat_pct  = achieved / offered * 100 if offered > 0 else 0
        ttft_avg = _safe_float(pivot.get(rate, {}).get("average_ttft", {}).get("mean"))
        itl_avg  = _safe_float(pivot.get(rate, {}).get("average_itl",  {}).get("mean"))
        cache    = _safe_float(pivot.get(rate, {}).get("cache_hit_rate", {}).get("mean"))
        n_fail   = _safe_float(pivot.get(rate, {}).get("throughput", {}).get("n_failed", 0))
        # Saturation indicator
        if sat_pct >= 90:   sat_label = '<span class="badge good">✓ Healthy</span>'
        elif sat_pct >= 50: sat_label = '<span class="badge warn">~ Partial</span>'
        else:                sat_label = '<span class="badge bad">✗ Saturated</span>'
        sat_rows += (
            f"<tr>"
            f"<td><b>{offered} req/s</b></td>"
            f"<td>{achieved:.2f} req/s</td>"
            f"<td>{sat_pct:.1f}% {sat_label}</td>"
            f"<td>{_fmt(ttft_avg, 's')}</td>"
            f"<td>{_fmt(itl_avg, 's')}</td>"
            f"<td>{cache*100:.1f}%</td>"
            f"</tr>"
        )

    # ── Request shape summary ─────────────────────────────────────────────────
    # Use first rate (all rates have same prompt/output distributions)
    r0   = pivot.get(baseline_rate, {})
    avg_prompt = _safe_float(r0.get("average_prompt_len", {}).get("mean"))
    p90_prompt = _safe_float(r0.get("p90_prompt_len",    {}).get("mean"))
    p99_prompt = _safe_float(r0.get("p99_prompt_len",    {}).get("mean"))
    avg_out    = _safe_float(r0.get("average_output_len",{}).get("mean"))
    p90_out    = _safe_float(r0.get("p90_output_len",    {}).get("mean"))
    p99_out    = _safe_float(r0.get("p99_output_len",    {}).get("mean"))

    extra_html = ""
    if extra_meta:
        for k, v in extra_meta.items():
            extra_html += f"<br><b>{_html.escape(str(k))}</b>: {_html.escape(str(v))}"

    # ── Assemble ──────────────────────────────────────────────────────────────
    title_esc = _html.escape(title)
    label_esc = _html.escape(run_label) if run_label else ""
    n_rates   = len(rates)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title_esc}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *{{box-sizing:border-box;}}
  body{{margin:0;padding:20px 24px 40px;background:#0f172a;color:#e2e8f0;
       font-family:Inter,system-ui,sans-serif;line-height:1.6;font-size:14px;}}
  .wrap{{max-width:1380px;margin:0 auto;}}
  h1{{font-size:22px;margin:0 0 4px;color:#f8fafc;letter-spacing:-.4px;}}
  .subtitle{{color:#94a3b8;font-size:13px;margin-bottom:18px;}}
  .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;
          padding:16px 18px;margin:12px 0;}}
  .card h2{{font-size:14px;margin:0 0 12px;color:#f1f5f9;display:flex;
             align-items:center;gap:8px;}}
  .tag{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:8px;
         background:#312e81;color:#a5b4fc;}}

  /* KPI table */
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th{{background:#0f172a;color:#94a3b8;text-align:left;padding:8px 10px;
       border-bottom:1px solid #334155;font-size:11px;text-transform:uppercase;
       letter-spacing:.4px;font-weight:700;}}
  td{{padding:8px 10px;border-bottom:1px solid #1e3a5b;color:#e2e8f0;vertical-align:middle;}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:rgba(99,102,241,0.06);}}
  .mname{{color:#94a3b8;white-space:nowrap;font-size:12px;}}

  /* Badges */
  .badge{{display:inline-block;border-radius:5px;padding:1px 6px;font-size:10.5px;
          font-weight:800;margin-left:4px;letter-spacing:.3px;}}
  .badge.good{{background:#064e3b;color:#bbf7d0;border:1px solid #047857;}}
  .badge.bad {{background:#7f1d1d;color:#fca5a5;border:1px solid #b91c1c;}}
  .badge.warn{{background:#78350f;color:#fde68a;border:1px solid #b45309;}}
  .badge.neutral{{background:#1e293b;color:#94a3b8;border:1px solid #475569;}}

  /* Saturation table */
  .sat-table td:first-child{{font-weight:700;color:#f1f5f9;}}

  /* Charts */
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
  @media(max-width:1000px){{.charts-grid{{grid-template-columns:1fr;}}}}
  .chart-box{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:10px;}}

  /* Shape grid */
  .shape-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;}}
  .shape-card{{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px 14px;}}
  .shape-label{{color:#94a3b8;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;font-weight:700;}}
  .shape-value{{color:#f8fafc;font-size:18px;font-weight:800;margin-top:4px;}}
  .shape-note{{color:#64748b;font-size:11px;margin-top:2px;}}
</style></head>
<body><div class="wrap">

<h1>📊 {title_esc}</h1>
<div class="subtitle">
  Long-context benchmark · {n_rates} load levels tested
  ({', '.join(f"{r} req/s" for r in rates)})
  · Δ% relative to lowest load ({baseline_rate} req/s baseline){extra_html}
  {"· <b>" + label_esc + "</b>" if label_esc else ""}
</div>

<div class="card">
  <h2>⚡ Load-Level Saturation Analysis</h2>
  <table class="sat-table">
    <thead><tr>
      <th>Offered load</th>
      <th>Achieved throughput</th>
      <th>Saturation</th>
      <th>Avg TTFT</th>
      <th>Avg ITL (TPOT)</th>
      <th>Cache hit</th>
    </tr></thead>
    <tbody>{sat_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>📐 Request Shape (constant across load levels)</h2>
  <div class="shape-grid">
    <div class="shape-card">
      <div class="shape-label">Avg prompt length</div>
      <div class="shape-value">{avg_prompt:,.0f} tok</div>
      <div class="shape-note">P90 {p90_prompt:,.0f} · P99 {p99_prompt:,.0f}</div>
    </div>
    <div class="shape-card">
      <div class="shape-label">Avg output length</div>
      <div class="shape-value">{avg_out:.0f} tok</div>
      <div class="shape-note">P90 {p90_out:.0f} · P99 {p99_out:.0f}</div>
    </div>
    <div class="shape-card">
      <div class="shape-label">Prompt/output ratio</div>
      <div class="shape-value">{avg_prompt/max(avg_out,1):.0f}:1</div>
      <div class="shape-note">Long-context, short response</div>
    </div>
    <div class="shape-card">
      <div class="shape-label">Load levels tested</div>
      <div class="shape-value">{n_rates}</div>
      <div class="shape-note">{', '.join(rates)} req/s offered</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>📋 Full KPI Table <span class="tag">Δ vs {baseline_rate} req/s baseline</span></h2>
  <table>
    <thead>{kpi_header}</thead>
    <tbody>{kpi_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>📈 Latency & Throughput vs Offered Load</h2>
  <div class="charts-grid">
    {charts_html}
  </div>
</div>

</div></body></html>"""


# ── Standalone entry point ────────────────────────────────────────────────────

def main_bench_lc(args) -> int:
    """CLI entry point for `amoprof bench-lc`."""
    from pathlib import Path as _P
    import logging
    log = logging.getLogger("amoprof.bench_lc")

    input_path = _P(args.input).expanduser().resolve()
    if not input_path.exists():
        log.error("bench-lc: input file not found: %s", input_path)
        return 2

    out_path = (_P(args.output).expanduser().resolve()
                if getattr(args, "output", None)
                else input_path.parent / (input_path.stem + "_report.html"))

    log.info("bench-lc: parsing %s", input_path)
    try:
        records = parse_lc_results(input_path)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("bench-lc: failed to parse %s: %s", input_path, e)
        return 2

    pivot = aggregate_records(records)
    rates = sorted(pivot.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    log.info("bench-lc: %d records, %d rate levels (%s), %d metrics/rate",
             len(records), len(rates), ", ".join(rates),
             len(next(iter(pivot.values()))))

    title     = getattr(args, "title", "") or "Long-Context Benchmark Summary"
    run_label = getattr(args, "run_label", "") or ""

    html = build_lc_report(records, title=title, run_label=run_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("bench-lc: wrote %s (%d KB)", out_path, len(html) // 1024)
    print(f"\nBenchmark report → {out_path}")
    return 0
