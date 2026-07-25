"""
amoprof/comparator.py — Compare metrics across multiple run directories.

Each run can have:
  • Its own local raw/ directory
  • Its own Prometheus time window (start/end)
  • A short human-readable label

The comparison report is a single self-contained HTML file with:
  1. Summary table    — all KPIs side-by-side, delta % vs the baseline
  2. Bar charts       — Plotly grouped bars for every metric family
  3. Radar chart      — normalised 0-100 score per metric for each run
  4. Bottleneck table — per-layer score for each run
  5. Recommendations  — which run is "better" per metric and why

Usage (module):
    from amoprof.comparator import compare_runs
    compare_runs(runs=[...], out_html=Path("comparison.html"))

Usage (CLI):
    amoprof compare \
      --run "Baseline:./run_c1:1778871540:1778877078" \
      --run "Optimised:./run_c4:1778881540:1778887078" \
      --output-dir ./comparison
"""
from __future__ import annotations

import csv
import html as _html
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("amoprof.comparator")


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    """Specification for one run to compare."""
    label: str
    raw_dir: Path
    t_start: float = 0.0
    t_end:   float = 0.0


@dataclass
class RunMetrics:
    """Key scalar metrics extracted from one run."""
    label: str
    # Latency / throughput
    ttft_ms:          float = 0.0
    tpot_ms:          float = 0.0
    e2e_ms:           float = 0.0
    throughput_tok_s: float = 0.0
    cache_hit_pct:    float = 0.0
    cache_hit_tw_pct: float = 0.0
    # GPU / HBM
    gpu_util_pct:     float = 0.0
    gpu_util_peak:    float = 0.0
    hbm_used_gb:      float = 0.0
    hbm_pct:          float = 0.0
    gpu_power_w:      float = 0.0
    dcgm_active_pct:  float = 0.0
    # NVMe
    nvme_rd_bw_mbs:   float = 0.0
    nvme_wr_bw_mbs:   float = 0.0
    nvme_rd_iops:     float = 0.0
    nvme_util_pct:    float = 0.0
    nvme_rd_lat_ms:   float = 0.0
    nvme_wr_lat_ms:   float = 0.0
    read_total_gb:    float = 0.0
    write_total_gb:   float = 0.0
    # DRAM
    dram_total_bw_gbs: float = 0.0
    dram_rd_bw_gbs:    float = 0.0
    dram_wr_bw_gbs:    float = 0.0
    dram_peak_bw_gbs:  float = 0.0
    # Context info
    model:    str = ""
    runtime:  str = ""
    gpu_desc: str = ""
    tp_size:  str = ""
    context:  str = ""
    hicache:  str = ""
    duration_min: float = 0.0
    window_str: str = ""
    # Raw sources available
    has_blktrace: bool = False
    has_dram:     bool = False
    has_gpu:      bool = False
    raw_dir:      str  = ""   # for display in the comparison table


def _sf(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _pick(d: Dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", "unknown"):
            return d[k]
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if str(k).lower() in low and low[str(k).lower()] not in (None, "", "unknown"):
            return low[str(k).lower()]
    return None


def _read_json(path: Path) -> Dict:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        pass
    return {}


def _read_csv_col(path: Path, col: str) -> List[float]:
    """Return all non-zero finite values of `col` from a CSV."""
    if not path.exists():
        return []
    out = []
    try:
        with open(path, encoding="utf-8", newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                v = _sf(row.get(col))
                if v > 0:
                    out.append(v)
    except Exception:
        pass
    return out


# ─── Metric extraction ───────────────────────────────────────────────────────

def extract_metrics(spec: RunSpec) -> RunMetrics:
    """Extract all comparable scalar metrics from a run's raw/ directory."""
    raw = spec.raw_dir
    m   = RunMetrics(label=spec.label, raw_dir=str(raw))

    setup  = _read_json(raw / "setup_details.json") or _read_json(raw / "setup.json")
    sglang = _read_json(raw / "sglang_summary.json")
    gpu    = _read_json(raw / "gpu_summary.json")
    smart  = _read_json(raw / "smart_summary.json")
    blk    = _read_json(raw / "summary.json")

    # Context
    m.model    = str(_pick(setup, "Model", "model", "model_name") or
                     _pick(sglang, "model_name", "model") or "unknown")
    m.runtime  = str(_pick(setup, "Runtime", "runtime") or "SGLang")
    m.gpu_desc = str(_pick(setup, "GPU", "gpu") or "unknown")
    m.tp_size  = str(_pick(setup, "TP size", "tp_size") or "?")
    m.context  = str(_pick(setup, "Context length", "context_length", "max_model_len") or "?")
    m.hicache  = str(_pick(setup, "HiCache", "enable_hierarchical_cache") or "?")

    dur_s = _sf(_pick(blk, "duration_s", "run_duration_s")) or \
            _sf(_pick(sglang, "duration_s"))
    if spec.t_start > 0 and spec.t_end > 0:
        dur_s = spec.t_end - spec.t_start
    m.duration_min = dur_s / 60.0

    if spec.t_start > 1_000_000_000:
        from datetime import datetime, timezone, timedelta
        t0 = datetime.fromtimestamp(spec.t_start, tz=timezone.utc)
        t1 = datetime.fromtimestamp(spec.t_end,   tz=timezone.utc) if spec.t_end else None
        m.window_str = t0.strftime("%Y-%m-%d %H:%M UTC")
        if t1:
            m.window_str += f" → {t1.strftime('%H:%M UTC')}"

    # Latency / throughput
    m.ttft_ms          = _sf(_pick(sglang, "server_ttft_ms", "ttft_ms", "mean_ttft_ms"))
    m.tpot_ms          = _sf(_pick(sglang, "server_itl_ms", "tpot_ms", "itl_ms", "mean_tpot_ms"))
    m.e2e_ms           = _sf(_pick(sglang, "server_e2e_ms", "e2e_ms", "mean_e2e_ms"))
    m.cache_hit_pct    = _sf(_pick(sglang, "cache_hit_pct", "cache_hit_active_avg_pct",
                                    "cache_hit_rate_pct", "cache_hit_rate_realtime_pct"))
    m.cache_hit_tw_pct = _sf(_pick(sglang, "cache_hit_token_weighted_pct",
                                    "token_weighted_cache_hit_pct"))
    m.throughput_tok_s = _sf(_pick(sglang, "gen_tp_peak", "gen_tp_active_mean",
                                    "throughput_tok_s", "ai_op_decode_tok_s"))

    # GPU
    m.gpu_util_pct   = _sf(_pick(gpu, "gpu_util_mean", "gpu_util_pct_mean", "util_mean"))
    m.gpu_util_peak  = _sf(_pick(gpu, "gpu_util_peak", "gpu_util_pct_peak"))
    m.hbm_used_gb    = _sf(_pick(gpu, "hbm_used_gb_mean", "hbm_gb", "mem_used_gb_mean"))
    if m.hbm_used_gb > 1000:
        m.hbm_used_gb /= 1024.0
    m.hbm_pct        = _sf(_pick(gpu, "hbm_util_pct_mean", "hbm_pct", "mem_util_pct_mean"))
    m.gpu_power_w    = _sf(_pick(gpu, "power_peak_w", "gpu_power_peak_w"))
    m.dcgm_active_pct = _sf(_pick(gpu, "dcgm_hbm_bw_active_pct", "dcgm_active_pct"))
    m.has_gpu        = bool(gpu)

    # NVMe
    m.nvme_rd_bw_mbs = _sf(_pick(blk, "nvme_rd_bw_mbs_mean", "read_bw_mb_s_mean"))
    m.nvme_wr_bw_mbs = _sf(_pick(blk, "nvme_wr_bw_mbs_mean", "write_bw_mb_s_mean"))
    m.nvme_rd_iops   = _sf(_pick(blk, "nvme_rd_iops_mean", "read_iops_mean"))
    m.nvme_util_pct  = _sf(_pick(blk, "nvme_io_util_pct", "nvme_util_pct"))
    m.nvme_rd_lat_ms = _sf(_pick(blk, "nvme_rd_lat_ms_mean", "read_lat_ms_mean"))
    m.nvme_wr_lat_ms = _sf(_pick(blk, "nvme_wr_lat_ms_mean", "write_lat_ms_mean"))
    m.read_total_gb  = _sf(_pick(blk, "nvme_read_total_gb", "read_gb_total", "read_GB_total"))
    m.write_total_gb = _sf(_pick(blk, "nvme_write_total_gb", "write_gb_total"))

    # Fallback from nvme_driver_timeseries
    nvme_ts = raw / "nvme_driver_timeseries.csv"
    if m.nvme_rd_bw_mbs == 0:
        vals = _read_csv_col(nvme_ts, "rd_bw_mbs")
        if vals: m.nvme_rd_bw_mbs = sum(vals) / len(vals)
    if m.nvme_util_pct == 0:
        vals = _read_csv_col(nvme_ts, "io_util_pct")
        if vals: m.nvme_util_pct = sum(vals) / len(vals)
    if m.nvme_rd_lat_ms == 0:
        vals = _read_csv_col(nvme_ts, "rd_lat_ms")
        if vals: m.nvme_rd_lat_ms = sum(vals) / len(vals)

    m.has_blktrace = any((raw / f).exists() for f in (
        "request_size_distribution.csv", "hot_regions_overall.csv"))

    # DRAM
    for dram_file in ("amduprof_pcm_timeseries.csv", "pcm_timeseries.csv"):
        dp = raw / dram_file
        if dp.exists():
            for total_col in ("dram_total_gb_s", "total_bw_gbps", "total_gb_s"):
                vals = _read_csv_col(dp, total_col)
                if vals:
                    m.dram_total_bw_gbs = sum(vals) / len(vals)
                    m.dram_peak_bw_gbs  = max(vals)
                    m.has_dram = True
                    break
            if m.has_dram:
                for rd_col in ("dram_read_gb_s", "read_bw_gbps", "read_gb_s"):
                    v = _read_csv_col(dp, rd_col)
                    if v: m.dram_rd_bw_gbs = sum(v) / len(v); break
                for wr_col in ("dram_write_gb_s", "write_bw_gbps", "write_gb_s"):
                    v = _read_csv_col(dp, wr_col)
                    if v: m.dram_wr_bw_gbs = sum(v) / len(v); break
                break

    # Clamp all float fields to >= 0 to prevent negative values
    # (e.g. -1 sentinel values in DCGM, or misconfigured Prometheus metrics)
    # reaching bar charts or delta cells.
    for _attr in ("ttft_ms", "tpot_ms", "e2e_ms", "throughput_tok_s",
                  "cache_hit_pct", "cache_hit_tw_pct", "gpu_util_pct",
                  "gpu_util_peak", "hbm_used_gb", "hbm_pct", "gpu_power_w",
                  "dcgm_active_pct", "nvme_rd_bw_mbs", "nvme_wr_bw_mbs",
                  "nvme_rd_iops", "nvme_util_pct", "nvme_rd_lat_ms",
                  "nvme_wr_lat_ms", "read_total_gb", "write_total_gb",
                  "dram_total_bw_gbs", "dram_rd_bw_gbs", "dram_wr_bw_gbs",
                  "dram_peak_bw_gbs"):
        v = getattr(m, _attr, 0.0)
        if v < 0:
            setattr(m, _attr, 0.0)

    log.info("Extracted metrics for '%s': TTFT=%.1fs TPOT=%.0fms "
             "NVMe=%.0f MB/s GPU=%.1f%%",
             m.label, m.ttft_ms / 1000, m.tpot_ms,
             m.nvme_rd_bw_mbs, m.gpu_util_pct)
    return m


# ─── HTML report builder ──────────────────────────────────────────────────────

def _delta_cell(base: float, cur: float, lower_is_better: bool = False,
                fmt: str = ".1f", unit: str = "") -> str:
    """Return a <td> with delta % colouring relative to base."""
    # Treat zero current as N/A (missing data, not genuinely zero performance)
    if cur == 0:
        return "<td class='neutral'><span style='color:#475569'>N/A</span></td>"
    if base == 0:
        return f"<td class='neutral'>{cur:{fmt}}{unit}</td>"
    delta = (cur - base) / abs(base) * 100
    # Suppress tiny deltas (< 1%) as noise
    good = (delta < -1) if lower_is_better else (delta > 1)
    bad  = (delta >  1) if lower_is_better else (delta < -1)
    cls  = "good" if good else ("bad" if bad else "neutral")
    sign = "+" if delta >= 0 else ""
    return (f"<td class='{cls}'>{cur:{fmt}}{unit} "
            f"<span class='delta'>({sign}{delta:.1f}%)</span></td>")


def build_comparison_html(
    metrics: List[RunMetrics],
    title: str = "AMOprof Comparison Report",
) -> str:
    """Return a self-contained comparison HTML from a list of RunMetrics."""

    if len(metrics) < 2:
        raise ValueError("Need at least 2 runs to compare")

    baseline = metrics[0]
    n        = len(metrics)
    labels   = [_html.escape(m.label) for m in metrics]
    colours  = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444",
                "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

    # ── Plotly chart helper ────────────────────────────────────────────────
    def bar_chart(chart_id: str, title_str: str,
                  series: List[tuple[str, List[float]]],
                  y_label: str = "", note: str = "") -> str:
        traces = []
        for i, (name, vals) in enumerate(series):
            traces.append({
                "type": "bar", "name": name,
                "x": labels, "y": vals,
                "marker": {"color": colours[i % len(colours)]},
                "text": [f"{v:.1f}" for v in vals],
                "textposition": "outside",
            })
        layout = {
            "title": {"text": title_str, "font": {"size": 14, "color": "#f1f5f9"}},
            "paper_bgcolor": "#1e293b", "plot_bgcolor": "#1e293b",
            "font": {"color": "#e2e8f0", "size": 12},
            "barmode": "group",
            "yaxis": {"title": y_label, "gridcolor": "#334155",
                      "color": "#e2e8f0", "zerolinecolor": "#475569"},
            "xaxis": {"color": "#e2e8f0", "gridcolor": "#334155"},
            "legend": {"orientation": "h", "y": -0.3,
                       "font": {"color": "#e2e8f0"}},
            "margin": {"l": 55, "r": 15, "t": 45, "b": 70},
            "height": 300,
        }
        data_json = json.dumps({"data": traces, "layout": layout})
        note_html = f'<div class="chart-note">{_html.escape(note)}</div>' if note else ""
        return (f'<div class="chart-wrap">'
                f'<div id="{chart_id}"></div>{note_html}</div>'
                f'<script>Plotly.newPlot("{chart_id}",'
                f'{data_json}.data, {data_json}.layout,'
                f'{{"responsive":true,"displayModeBar":false}});</script>')

    def radar_chart() -> str:
        """Radar chart: normalised 0–100 score for each run on 8 axes."""
        # Higher = better for all axes (some are inverted)
        theta = ["Throughput", "Cache Hit", "GPU Util",
                 "HBM Headroom", "NVMe Latency\n(inv)", "DRAM Util\n(inv headroom)",
                 "TTFT\n(inv)", "TPOT\n(inv)"]
        traces = []
        for i, m in enumerate(metrics):
            # Normalise each dimension to 0-100 relative to max across all runs
            def norm_hi(vals: List[float]) -> float:
                mx = max(vals) if vals and max(vals) > 0 else 1
                return (m_vals / mx * 100) if (m_vals := vals[i]) else 0

            all_tp  = [x.throughput_tok_s for x in metrics]
            all_ch  = [x.cache_hit_pct for x in metrics]
            all_gu  = [x.gpu_util_pct for x in metrics]
            all_hbm = [max(100 - x.hbm_pct, 0) for x in metrics]   # headroom (clipped)
            all_nll = [1 / max(x.nvme_rd_lat_ms, 0.001) * 100 for x in metrics]  # inv latency
            all_dram= [100 - min(x.dram_total_bw_gbs / 204.8 * 100, 100) for x in metrics]  # headroom
            all_ttft= [1 / max(x.ttft_ms, 1) * 1e6 for x in metrics]   # inv
            all_tpot= [1 / max(x.tpot_ms, 1) * 1e6 for x in metrics]   # inv

            def _norm(vals: List[float], idx: int) -> float:
                mx = max(vals) if vals and max(vals) > 0 else 1
                return round(vals[idx] / mx * 100, 1)

            r = [
                _norm(all_tp,   i), _norm(all_ch,   i), _norm(all_gu,  i),
                _norm(all_hbm,  i), _norm(all_nll,  i), _norm(all_dram, i),
                _norm(all_ttft, i), _norm(all_tpot,  i),
            ]
            r.append(r[0])  # close loop
            traces.append({
                "type": "scatterpolar",
                "r": r, "theta": theta + [theta[0]],
                "name": m.label,
                "fill": "toself",
                "line": {"color": colours[i % len(colours)], "width": 2},
                "fillcolor": colours[i % len(colours)].replace("#", "rgba(") + ",0.15)",
            })
        layout = {
            "polar": {
                "radialaxis": {"visible": True, "range": [0, 105],
                               "color": "#475569", "gridcolor": "#334155"},
                "angularaxis": {"color": "#e2e8f0", "gridcolor": "#334155"},
                "bgcolor": "#1e293b",
            },
            "paper_bgcolor": "#1e293b",
            "font": {"color": "#e2e8f0", "size": 11},
            "legend": {"orientation": "h", "y": -0.15,
                       "font": {"color": "#e2e8f0"}},
            "margin": {"l": 40, "r": 40, "t": 10, "b": 60},
            "height": 380,
            "title": {"text": "Normalised performance profile (higher = better)",
                      "font": {"size": 13, "color": "#94a3b8"}},
        }
        data_json = json.dumps({"data": traces, "layout": layout})
        return (f'<div id="radar-chart"></div>'
                f'<script>Plotly.newPlot("radar-chart",'
                f'{data_json}.data, {data_json}.layout,'
                f'{{"responsive":true,"displayModeBar":false}});</script>')

    # ── KPI comparison table ───────────────────────────────────────────────
    def kpi_rows() -> str:
        rows_data = [
            # (label, attr, unit, lower_is_better, fmt)
            ("TTFT (s)",         "ttft_ms",          "s",    True,  ".1f", 1/1000),
            ("TPOT (ms/tok)",    "tpot_ms",          " ms",  True,  ".0f", 1.0),
            ("E2E latency (s)",  "e2e_ms",           "s",    True,  ".1f", 1/1000),
            ("Throughput (tok/s)","throughput_tok_s","tok/s", False, ".1f", 1.0),
            ("Cache hit % (gauge)","cache_hit_pct",  "%",    False, ".1f", 1.0),
            ("Cache hit % (token-wt)","cache_hit_tw_pct","%",False, ".1f", 1.0),
            ("GPU util %",       "gpu_util_pct",     "%",    False, ".1f", 1.0),
            ("HBM used (GB)",    "hbm_used_gb",      " GB",  True,  ".1f", 1.0),
            ("HBM fill %",       "hbm_pct",          "%",    True,  ".1f", 1.0),
            ("GPU power (W)",    "gpu_power_w",      " W",   True,  ".0f", 1.0),
            ("L3 (local storage) read throughput (MB/s)","nvme_rd_bw_mbs", " MB/s",False, ".0f", 1.0),
            ("NVMe read IOPS",   "nvme_rd_iops",     " IOPS",False, ".0f", 1.0),
            ("NVMe read lat (ms)","nvme_rd_lat_ms",  " ms",  True,  ".2f", 1.0),
            ("NVMe device util%","nvme_util_pct",    "%",    True,  ".1f", 1.0),
            ("SSD reads total (GB)","read_total_gb", " GB",  False, ".0f", 1.0),
            ("DRAM total BW (GB/s)","dram_total_bw_gbs"," GB/s",False,".1f",1.0),
        ]
        header = "<tr><th>Metric</th>" + "".join(
            f"<th>{l}</th>" for l in labels) + "</tr>"
        body = ""
        for row_label, attr, unit, lib, fmt, scale in rows_data:
            vals = [getattr(m, attr) * scale for m in metrics]
            if all(v == 0 for v in vals):
                continue
            cells = ""
            for j, v in enumerate(vals):
                if j == 0:
                    cells += f"<td class='baseline'>{v:{fmt}}{unit}</td>"
                else:
                    cells += _delta_cell(vals[0] / scale * scale,
                                          v, lib, fmt, unit)
            body += f"<tr><td class='metric-name'>{_html.escape(row_label)}</td>{cells}</tr>"
        return f"<table class='kpi-table'><thead>{header}</thead><tbody>{body}</tbody></table>"

    # ── Configuration table ────────────────────────────────────────────────
    def config_table() -> str:
        fields = [
            ("Model",    [m.model for m in metrics]),
            ("Runtime",  [m.runtime for m in metrics]),
            ("GPU",      [m.gpu_desc for m in metrics]),
            ("TP size",  [m.tp_size for m in metrics]),
            ("Context",  [m.context for m in metrics]),
            ("HiCache",  [m.hicache for m in metrics]),
            ("Duration", [f"{m.duration_min:.0f} min" if m.duration_min else "?" for m in metrics]),
            ("Window",   [m.window_str or "—" for m in metrics]),
        ]
        header = "<tr><th>Parameter</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
        rows = ""
        for fname, fvals in fields:
            # Highlight differences
            cells = ""
            all_same = len(set(str(v) for v in fvals)) == 1
            for v in fvals:
                cls = "" if all_same else " class='diff'"
                cells += f"<td{cls}>{_html.escape(str(v))}</td>"
            rows += f"<tr><td class='metric-name'>{_html.escape(fname)}</td>{cells}</tr>"
        return f"<table class='kpi-table'><thead>{header}</thead><tbody>{rows}</tbody></table>"

    # ── Winner badges ──────────────────────────────────────────────────────
    def winner_summary() -> str:
        categories = [
            ("Lowest TTFT",       "ttft_ms",          True),
            ("Lowest TPOT",       "tpot_ms",          True),
            ("Best throughput",   "throughput_tok_s",  False),
            ("Best cache hit",    "cache_hit_pct",     False),
            ("Lowest HBM fill",   "hbm_pct",           True),
            ("Highest NVMe BW",   "nvme_rd_bw_mbs",    False),
            ("Lowest NVMe lat",   "nvme_rd_lat_ms",    True),
            ("Lowest DRAM BW",    "dram_total_bw_gbs", True),  # lower = less L2 pressure
        ]
        html_out = '<div class="winner-grid">'
        for cat, attr, lower_is_better in categories:
            vals = [(getattr(m, attr), m.label) for m in metrics if getattr(m, attr) > 0]
            if not vals:
                continue
            best_val, best_label = min(vals) if lower_is_better else max(vals)
            esc_label = _html.escape(best_label)
            esc_cat   = _html.escape(cat)
            esc_val   = _html.escape(f"{best_val:.1f}")
            html_out += (
                f'<div class="winner-card">'
                f'<div class="winner-cat">{esc_cat}</div>'
                f'<div class="winner-label">🏆 {esc_label}</div>'
                f'<div class="winner-val">{esc_val}</div>'
                f'</div>')
        return html_out + '</div>'

    # ── Build chart blocks ─────────────────────────────────────────────────
    charts_html = ""
    charts_html += bar_chart("ch-latency", "Latency comparison",
        [("TTFT (s)",      [m.ttft_ms / 1000 for m in metrics]),
         ("TPOT (ms×10)",  [max(m.tpot_ms, 0) / 100  for m in metrics]),
         ("E2E (s)",       [m.e2e_ms  / 1000 for m in metrics])],
        y_label="seconds / ×10ms",
        note="Lower is better. TPOT scaled ÷100 to share axis.")

    charts_html += bar_chart("ch-throughput", "Throughput & cache hit",
        [("Throughput (tok/s)", [m.throughput_tok_s  for m in metrics]),
         ("Cache hit % (gauge)", [m.cache_hit_pct    for m in metrics]),
         ("Cache hit % (tw)",    [m.cache_hit_tw_pct for m in metrics])],
        y_label="tok/s or %",
        note="Higher is better.")

    charts_html += bar_chart("ch-gpu", "GPU utilisation & HBM",
        [("GPU util %",  [m.gpu_util_pct  for m in metrics]),
         ("HBM fill %",  [m.hbm_pct       for m in metrics]),
         ("DCGM active%",[m.dcgm_active_pct for m in metrics])],
        y_label="%",
        note="GPU util higher is better; HBM fill lower is better.")

    charts_html += bar_chart("ch-nvme", "NVMe storage",
        [("Read BW (MB/s)",  [m.nvme_rd_bw_mbs  for m in metrics]),
         ("Read IOPS",       [m.nvme_rd_iops     for m in metrics]),
         ("Device util %",   [m.nvme_util_pct    for m in metrics]),
         ("Read lat (ms×100)",[m.nvme_rd_lat_ms * 100 for m in metrics])],
        y_label="MB/s or IOPS or %",
        note="Read BW and IOPS higher = more KV load-back. Latency lower is better.")

    if any(m.dram_total_bw_gbs > 0 for m in metrics):
        charts_html += bar_chart("ch-dram", "System DRAM bandwidth",
            [("Total BW (GB/s)", [m.dram_total_bw_gbs for m in metrics]),
             ("Read BW (GB/s)",  [m.dram_rd_bw_gbs    for m in metrics]),
             ("Write BW (GB/s)", [m.dram_wr_bw_gbs    for m in metrics])],
            y_label="GB/s",
            note="AMD EPYC PMU. Lower total = less L2 DRAM pressure.")

    charts_html += bar_chart("ch-ssd-total", "SSD I/O volume",
        [("Reads (GB)",  [m.read_total_gb  for m in metrics]),
         ("Writes (GB)", [m.write_total_gb for m in metrics])],
        y_label="GB",
        note="Total bytes read/written to NVMe over the collection window.")

    # ── Full HTML ──────────────────────────────────────────────────────────
    run_rows_html = "".join(
        f"<tr><td>{i}</td><td><b>{_html.escape(m.label)}</b></td>"
        f"<td>{_html.escape(str(m.raw_dir))}</td>"
        f"<td>{_html.escape(m.window_str or '—')}</td>"
        f"<td>{m.duration_min:.0f} min</td></tr>"
        for i, m in enumerate(metrics)
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *{{box-sizing:border-box;}}
  body{{margin:0;padding:20px 24px 40px;background:#0f172a;color:#e2e8f0;
       font-family:Inter,system-ui,sans-serif;line-height:1.6;font-size:13px;}}
  .wrap{{max-width:1400px;margin:0 auto;}}
  h1{{font-size:22px;margin:0 0 4px;color:#f8fafc;letter-spacing:-.4px;}}
  .subtitle{{color:#94a3b8;font-size:13px;margin-bottom:18px;}}
  .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;
          padding:16px 18px;margin:12px 0;}}
  .card h2{{font-size:14px;margin:0 0 12px;color:#f1f5f9;display:flex;
             align-items:center;gap:8px;}}
  .tag{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:8px;
         background:#312e81;color:#a5b4fc;}}
  .kpi-table{{width:100%;border-collapse:collapse;font-size:12px;}}
  .kpi-table th{{background:#0f172a;color:#94a3b8;text-align:left;
                  padding:8px 10px;border-bottom:1px solid #334155;
                  font-size:11px;text-transform:uppercase;letter-spacing:.4px;}}
  .kpi-table td{{padding:8px 10px;border-bottom:1px solid #1e3a5b;
                  vertical-align:middle;}}
  .kpi-table tr:last-child td{{border-bottom:none;}}
  .kpi-table tr:hover td{{background:rgba(99,102,241,0.06);}}
  .metric-name{{color:#94a3b8;white-space:nowrap;font-size:12px;}}
  .baseline{{color:#a5b4fc;font-weight:700;}}
  .good{{color:#4ade80;font-weight:700;}}
  .bad {{color:#f87171;font-weight:700;}}
  .neutral{{color:#e2e8f0;}}
  .diff{{color:#fbbf24;}}
  .delta{{font-size:10px;font-weight:600;opacity:0.85;}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
  @media(max-width:900px){{.two-col{{grid-template-columns:1fr;}}}}
  .chart-wrap{{background:#1e293b;border:1px solid #334155;border-radius:10px;
                padding:12px;margin:6px 0;}}
  .chart-note{{font-size:11px;color:#475569;margin-top:6px;font-style:italic;}}
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
  @media(max-width:1000px){{.charts-grid{{grid-template-columns:1fr;}}}}
  .winner-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;}}
  .winner-card{{background:#111827;border:1px solid #334155;border-radius:10px;
                 padding:12px;text-align:center;}}
  .winner-cat{{color:#94a3b8;font-size:10.5px;text-transform:uppercase;
                letter-spacing:.5px;font-weight:700;}}
  .winner-label{{color:#f1f5f9;font-size:13px;font-weight:800;margin:5px 0 3px;}}
  .winner-val{{color:#64748b;font-size:11px;}}
  table.runs{{width:100%;border-collapse:collapse;font-size:12px;}}
  table.runs th{{background:#0f172a;color:#64748b;text-align:left;
                  padding:7px 10px;border-bottom:1px solid #334155;
                  font-size:11px;text-transform:uppercase;}}
  table.runs td{{padding:7px 10px;border-bottom:1px solid #1e3a5b;color:#e2e8f0;}}
  .legend-note{{font-size:11px;color:#475569;margin-top:8px;}}
</style></head>
<body><div class="wrap">

<h1>📊 AMOprof Comparison Report</h1>
<div class="subtitle">{n} runs compared · baseline: <b>{_html.escape(baseline.label)}</b>
 · delta % relative to baseline · <span style="color:#4ade80">green = improvement</span>,
 <span style="color:#f87171">red = regression</span></div>

<div class="card">
  <h2>🏃 Runs compared</h2>
  <table class="runs">
    <thead><tr><th>#</th><th>Label</th><th>Raw directory</th>
               <th>Window</th><th>Duration</th></tr></thead>
    <tbody>{run_rows_html}</tbody>
  </table>
  <div class="legend-note">
    Delta % is relative to run 0 ({_html.escape(baseline.label)}).
    Green = improvement over baseline; Red = regression.
  </div>
</div>

<div class="card">
  <h2>🏆 Best in class</h2>
  {winner_summary()}
</div>

<div class="card">
  <h2>⚙️ Configuration</h2>
  {config_table()}
</div>

<div class="card">
  <h2>📐 KPI comparison <span class="tag">DELTA vs BASELINE</span></h2>
  {kpi_rows()}
</div>

<div class="card">
  <h2>🕸️ Performance radar</h2>
  {radar_chart()}
</div>

<div class="card">
  <h2>📈 Charts</h2>
  <div class="charts-grid">
    {charts_html}
  </div>
</div>

</div></body></html>"""


# ─── Entry point ─────────────────────────────────────────────────────────────

def compare_runs(
    runs: List[RunSpec],
    out_html: Path,
    title: str = "",
) -> Path:
    """Extract metrics from all run specs and write a comparison HTML report."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = [extract_metrics(spec) for spec in runs]
    html = build_comparison_html(
        all_metrics,
        title=title or f"AMOprof Comparison — {len(runs)} runs",
    )
    out_html.write_text(html, encoding="utf-8")
    log.info("Wrote comparison report: %s (%d KB)", out_html, len(html) // 1024)
    return out_html
