"""
amoprof/report/interactive.py — Generate an interactive HTML report with
hover tooltips, using Plotly.js loaded from CDN.

Unlike the bundled static report which renders matplotlib PNGs, this report
uses Plotly.js for every time-series chart so that hovering any data point
shows the exact value at that timestamp. Charts are emitted as inline
<div> + JSON traces — Plotly.js is loaded once from a CDN.

Designed to consume the same `raw/` directory layout that the static report
uses, so the same collection pipeline feeds both.

Usage:
    from amoprof.report.interactive import build_report
    build_report(raw_dir=Path("/path/to/raw"),
                 out_html=Path("/path/to/report.html"))
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from amoprof.report.l3_backend import resolve_l3_backend, reconcile_l3_io
except Exception:
    try:
        from l3_backend import resolve_l3_backend, reconcile_l3_io  # type: ignore
    except Exception:
        resolve_l3_backend = None  # type: ignore
        reconcile_l3_io = None  # type: ignore

try:
    from amoprof.report.common_kpis import compute_common_kpis, apply_common_kpis_to_html, write_common_kpis_json, compute_cache_hit_kpis
except Exception:
    try:
        from common_kpis import compute_common_kpis, apply_common_kpis_to_html, write_common_kpis_json, compute_cache_hit_kpis  # type: ignore
    except Exception:
        compute_common_kpis = None  # type: ignore
        apply_common_kpis_to_html = None  # type: ignore
        write_common_kpis_json = None  # type: ignore
        compute_cache_hit_kpis = None  # type: ignore

# Plotly.js version pinned to a known-stable release with broad browser support.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _amoprof_storage_label_cleanup(html_text: str, l3_backend_class: str = "") -> str:
    """Normalize visible L3/L3.5 labels consistently.

    Report convention:
      * L3   = local SSD/NVMe/file-backed cache tier.
      * L3.5 = AI Memory Node / remote/shared/disaggregated backing tier, or
               unresolved logical SGLang backing-tier movement when no local
               SSD/block mapping exists.

    The resolver passes backend_class='local_ssd' only for local SSD.  All other
    classes use L3.5 wording so a remote AI Memory Node run is never shown as
    local SSD L3.  Base64 image payloads are protected before text replacement.
    """
    if not html_text:
        return html_text

    data_uri_re = re.compile(r'data:image/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\s]+')
    protected_payloads: list[str] = []

    def _stash_data_uri(match: re.Match) -> str:
        protected_payloads.append(match.group(0))
        return f"__AMOPROF_DATA_URI_{len(protected_payloads) - 1}__"

    html_text = data_uri_re.sub(_stash_data_uri, html_text)

    is_local = (str(l3_backend_class or "").lower() == "local_ssd")
    tier = "L3" if is_local else "L3.5"
    tier_long = "L3 (SSD/local storage)" if is_local else "L3.5 (AI Memory Node / remote storage)"

    # Collapse prior accidental repeated L3.5 decorations.
    html_text = re.sub(
        r'(L3\.5 \(AI Memory Node / remote storage\))(?:\.5 \(AI Memory Node / remote storage\))+',
        r'\1', html_text)

    if is_local:
        replacements = {
            "L3.5 (AI Memory Node / remote storage)": tier_long,
            "L3 (AI Memory Node / remote storage)": tier_long,
            "L3 (AI Memory Node)": tier_long,
            "L3 (local storage)": tier,
            "L3 local storage": tier,
            "L3 local-storage": tier,
            "L3 local SSD": "L3 SSD",
            "L3 storage": "L3 storage",
            "L3 Storage": "L3 storage",
            "KV$ SSD IO": "KV$ L3 I/O",
            "KV$ SSD I/O": "KV$ L3 I/O",
            "KV$ SSD": "KV$ L3",
            "NVMe read throughput": "L3 read throughput",
            "NVMe Read Throughput": "L3 read throughput",
            "NVMe read BW": "L3 read BW",
            "NVMe Read BW": "L3 read BW",
        }
    else:
        replacements = {
            "L3.5 (AI Memory Node / remote storage)": tier_long,
            "L3 (AI Memory Node / remote storage)": tier_long,
            "L3 (AI Memory Node)": tier_long,
            "L3 (local storage)": tier_long,
            "L3 local storage": tier_long,
            "L3 local-storage": tier_long,
            "L3 local SSD": tier_long,
            "L3 storage / blktrace": "L3.5 / backing-tier telemetry",
            "L3 storage block telemetry": "L3.5 backing-tier telemetry",
            "L3 storage block-device": "L3.5 backing-tier device",
            "L3 storage QUEUE": "L3.5 QUEUE",
            "L3 storage Queue": "L3.5 Queue",
            "L3 storage queue": "L3.5 queue",
            "L3 storage": "L3.5 storage",
            "L3 Storage": "L3.5 storage",
            "KV$ L3 IO": "KV$ L3.5 I/O",
            "KV$ L3 I/O": "KV$ L3.5 I/O",
            "KV$ L3 ": "KV$ L3.5 ",
            "KV$ L3\n": "KV$ L3.5\n",
            "Logical KV movement — L3": "Logical KV movement — L3.5",
            "Physical L3 block": "Physical L3.5 block",
            "Physical L3.5 block R / W total": "Physical L3.5 block R / W total",
            "NVMe read throughput": "L3.5 read throughput",
            "NVMe Read Throughput": "L3.5 read throughput",
            "NVMe read BW": "L3.5 read BW",
            "NVMe Read BW": "L3.5 read BW",
        }
    for a, b in replacements.items():
        html_text = html_text.replace(a, b)

    # Clean up any doubled L3.5 token from overlapping replacements.
    html_text = html_text.replace("L3.5.5", "L3.5")
    html_text = html_text.replace("L3.5.5", "L3.5")

    for i, payload in enumerate(protected_payloads):
        html_text = html_text.replace(f"__AMOPROF_DATA_URI_{i}__", payload)
    return html_text


# ─── Run-wide wall-clock anchor ──────────────────────────────────────────────
# Set once by build_report() from summary.json "t0_epoch" (Unix timestamp of
# the collection-window start).  All chart functions read this automatically
# via _ts_to_dates() and _layout() so every chart's X axis is anchored to the
# same real wall-clock time — matching the --start / --end the user specified.
#
# When 0 (default), charts fall back to elapsed time from 0.
_T0_EPOCH: float = 0.0


# ─── Data loading ────────────────────────────────────────────────────────────
def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def _read_intel_pcm_memory_raw_summary(raw_dir: Path) -> dict:
    """Return Intel PCM DRAM summary from raw/pcm_memory_raw.csv if present."""
    p = Path(raw_dir) / "pcm_memory_raw.csv"
    if not p.exists() or p.stat().st_size <= 0:
        return {}
    try:
        rows = list(csv.reader(p.open(errors="replace")))
    except Exception:
        return {}
    if len(rows) < 3:
        return {}
    group, names = rows[0], rows[1]
    idx_read = idx_write = idx_mem = None
    for i, (g, n) in enumerate(zip(group, names)):
        gl = str(g or "").strip().lower(); nl = str(n or "").strip().lower()
        if gl == "system" and nl == "read": idx_read = i
        elif gl == "system" and nl == "write": idx_write = i
        elif gl == "system" and nl == "memory": idx_mem = i
    if idx_read is None or idx_write is None:
        reads = [i for i,n in enumerate(names) if str(n).strip().lower() == "mem read (mb/s)"]
        writes = [i for i,n in enumerate(names) if str(n).strip().lower() == "mem write (mb/s)"]
        if reads and writes:
            idx_read, idx_write = reads[-1], writes[-1]
    def num(x):
        try: return float(str(x).strip().replace(",", ""))
        except Exception: return None
    rd=[]; wr=[]; total=[]
    for r in rows[2:]:
        if idx_read is None or idx_write is None or len(r) <= max(idx_read, idx_write):
            continue
        a=num(r[idx_read]); b=num(r[idx_write])
        if a is None or b is None:
            continue
        ag=a/1024.0; bg=b/1024.0
        if idx_mem is not None and len(r)>idx_mem and num(r[idx_mem]) is not None:
            tg=num(r[idx_mem])/1024.0
        else:
            tg=ag+bg
        if tg>0:
            rd.append(ag); wr.append(bg); total.append(tg)
    if not total:
        return {}
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    return {
        "pcm_available": True, "pcm_source": "intel-pcm/pcm-memory/raw-reparse",
        "dram_collector": "intel-pcm/pcm-memory/raw-reparse",
        "pcm_raw_path": str(p), "pcm_samples": len(total), "pcm_nonzero_samples": len(total),
        "dram_read_gb_s_mean": mean(rd), "dram_write_gb_s_mean": mean(wr), "dram_total_gb_s_mean": mean(total),
        "pcm_dram_read_gb_s": mean(rd), "pcm_dram_write_gb_s": mean(wr), "pcm_dram_total_gb_s": mean(total),
        "dram_read_gb_s_peak": max(rd), "dram_write_gb_s_peak": max(wr), "dram_total_gb_s_peak": max(total),
        "pcm_dram_read_gb_s_peak": max(rd), "pcm_dram_write_gb_s_peak": max(wr), "pcm_dram_total_gb_s_peak": max(total),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        try:
            import re as _re
            txt = str(v).replace(",", "")
            m = _re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", txt)
            return float(m.group(0)) if m else default
        except Exception:
            return default


def _col(rows: list[dict], key: str) -> list[float]:
    return [_to_float(r.get(key)) for r in rows]


def _find_col(rows: list[dict], partial: str) -> str | None:
    """First column name containing `partial`. Useful for label-suffixed
    Prometheus metric column names like `sglang_cache_hit_rate`; older colon-style aliases are normalized internally when present."""
    if not rows:
        return None
    for c in rows[0].keys():
        if partial in c:
            return c
    return None




def _ratio_delta_ms_from_rows(rows: list[dict], sum_partial: str, count_partial: str) -> float:
    """Selected-window mean latency from cumulative Prometheus counters."""
    if not rows:
        return 0.0
    sum_col = _find_col(rows, sum_partial)
    cnt_col = _find_col(rows, count_partial)
    if not sum_col or not cnt_col:
        return 0.0
    try:
        a_sum = _to_float(rows[0].get(sum_col)); b_sum = _to_float(rows[-1].get(sum_col))
        a_cnt = _to_float(rows[0].get(cnt_col)); b_cnt = _to_float(rows[-1].get(cnt_col))
        ds = b_sum - a_sum; dc = b_cnt - a_cnt
        return (ds / dc * 1000.0) if ds > 0 and dc > 0 else 0.0
    except Exception:
        return 0.0



def _pct_ts_latency_ms(pct_ts: dict, metric_key: str, percentile: str = "p50") -> float:
    """Representative selected-window latency from percentile timeseries."""
    try:
        block = (pct_ts or {}).get(metric_key) or {}
        vals = block.get(percentile) or []
        clean = []
        for v in vals:
            try:
                x = float(v)
                if x == x and x > 0:
                    clean.append(x)
            except Exception:
                pass
        return (sum(clean) / len(clean)) if clean else 0.0
    except Exception:
        return 0.0


def _active_p50_from_rows(rows: list[dict], partial: str) -> float:
    """Median of nonzero samples for a timeseries column matched by substring."""
    try:
        col = _find_col(rows, partial)
        vals = []
        if col:
            for r in rows or []:
                try:
                    v = _to_float(r.get(col))
                    if v == v and v > 0:
                        vals.append(v)
                except Exception:
                    pass
        if vals:
            vals.sort(); n = len(vals); mid = n // 2
            return float(vals[mid] if n % 2 else (vals[mid-1] + vals[mid]) / 2.0)
    except Exception:
        pass
    return 0.0

def _active_gpu_util_from_rows(rows: list[dict]) -> tuple[float, float, str]:
    """Return active/nonzero mean GPU util, peak, and method note."""
    if not rows:
        return 0.0, 0.0, ""
    col = _find_col(rows, "gpu_util") or _find_col(rows, "DCGM_FI_DEV_GPU_UTIL")
    if not col:
        return 0.0, 0.0, ""
    vals = [_to_float(r.get(col)) for r in rows]
    active = [v for v in vals if v > 0]
    if active:
        return sum(active) / len(active), max(vals), "active mean from gpu_timeseries"
    return 0.0, max(vals) if vals else 0.0, ""

def _time_axis_config(ts_seconds: list[float] | None = None,
                       max_ts: float | None = None,
                       t0_epoch: float | None = None) -> dict:
    """Return Plotly xaxis config.

    Uses the module-level _T0_EPOCH when t0_epoch is not explicitly given.
    When t0 > 0 labels show real wall-clock datetimes matching --start/--end.
    When t0 == 0 labels show elapsed time (HH:MM:SS / MM:SS / SS).
    """
    t0 = _T0_EPOCH if t0_epoch is None else t0_epoch
    if max_ts is None:
        if ts_seconds:
            try:
                max_ts = max(ts_seconds)
            except (ValueError, TypeError):
                max_ts = 0
        else:
            max_ts = 0

    cfg = {
        "gridcolor": "#cbd5e1",
        "zerolinecolor": "#94a3b8",
        "linecolor": "#475569",
        "tickcolor": "#475569",
        "tickfont": {"size": 12, "color": "#0f172a"},
        "type": "date",
    }

    if t0 > 0:
        cfg["tickformat"] = "%H:%M:%S"
        cfg["title"] = {"text": "Wall-clock time (UTC)", "font": {"size": 13}}
        cfg["hoverformat"] = "%Y-%m-%d %H:%M:%S UTC"
    elif max_ts >= 3600:
        cfg["tickformat"] = "%H:%M:%S"
        cfg["title"] = {"text": "Elapsed time (HH:MM:SS)", "font": {"size": 13}}
    elif max_ts >= 60:
        cfg["tickformat"] = "%M:%S"
        cfg["title"] = {"text": "Elapsed time (MM:SS)", "font": {"size": 13}}
    else:
        cfg["tickformat"] = "%Ss"
        cfg["title"] = {"text": "Elapsed time (sec)", "font": {"size": 13}}

    return cfg


def _ts_to_dates(ts_seconds: list[float],
                  t0_epoch: float | None = None) -> list[str]:
    """Convert elapsed-seconds-from-t0 into ISO 8601 datetime strings for Plotly.

    Uses the module-level _T0_EPOCH (set by build_report from summary.json)
    when t0_epoch is not explicitly given.

    When the resolved t0 is 0, anchors at 1970-01-01 so tick labels show
    elapsed time (00:00:00 .. HH:MM:SS).

    When t0 > 0 (a real Unix timestamp), anchors at the actual wall-clock
    start of the collection window so every chart aligns with the
    --start / --end timestamps the user specified.
    """
    from datetime import datetime, timezone, timedelta
    t0 = _T0_EPOCH if t0_epoch is None else t0_epoch
    base = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=t0)
    out: list[str] = []
    for t in ts_seconds:
        try:
            sec = float(t)
        except (TypeError, ValueError):
            sec = 0.0
        if sec < 0:
            sec = 0.0
        dt = base + timedelta(seconds=sec)
        out.append(dt.strftime("%Y-%m-%dT%H:%M:%S.") +
                   f"{dt.microsecond // 1000:03d}")
    return out


# ─── Chart builders — each returns a Plotly figure-spec dict ────────────────
def _layout(title: str, ylabel: str = "", ylabel2: str = "",
            height: int = 460, ts: list[float] | None = None,
            t0_epoch: float | None = None) -> dict:
    """Default Plotly layout.

    Uses the module-level _T0_EPOCH when t0_epoch is not explicitly given.
    When t0 > 0 the X axis shows real wall-clock datetimes so every chart
    lines up with the --start / --end window the user specified.
    """
    t0 = _T0_EPOCH if t0_epoch is None else t0_epoch
    layout = {
        "title": {
            "text": title,
            "font": {"size": 15, "color": "#0f172a",
                     "family": "Inter, system-ui, sans-serif"},
            "x": 0.02, "y": 0.97, "xanchor": "left", "yanchor": "top",
        },
        "height": height,
        # Bottom margin needs room for the x-axis tick labels AND the legend
        # below them. The legend is anchored at y=-0.18 (below chart) and can
        # wrap to 2-3 rows for charts with many traces (e.g. 8-GPU chart).
        # 130px holds 3 legend rows + tick labels comfortably.
        "margin": {"l": 64, "r": 80, "t": 60, "b": 130},
        "plot_bgcolor": "#f8fafc",
        "paper_bgcolor": "#ffffff",
        "font": {"color": "#0f172a", "size": 13, "family": "Inter, system-ui, sans-serif"},
        "xaxis": (_time_axis_config(ts, t0_epoch=t0) if ts is not None else {
            "title": {"text": "Wall-clock time (UTC)" if t0 > 0 else "Time",
                      "font": {"size": 13}, "standoff": 15},
            "gridcolor": "#cbd5e1", "zerolinecolor": "#94a3b8",
            "linecolor": "#475569", "tickcolor": "#475569",
            "tickfont": {"size": 12, "color": "#0f172a"},
            "type": "date",
            "automargin": True,
        }),
        "yaxis": {
            "title": {"text": ylabel, "font": {"size": 13}, "standoff": 10},
            "gridcolor": "#cbd5e1", "zerolinecolor": "#94a3b8",
            "linecolor": "#475569", "tickcolor": "#475569",
            "tickfont": {"size": 12, "color": "#0f172a"},
            "automargin": True,
        },
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": "#1e293b", "bordercolor": "#0f172a",
                        "font": {"color": "#f1f5f9", "size": 13}},
        # Legend: horizontal row(s) below the x-axis. Plotly wraps automatically
        # when items don't fit on one row. Use xanchor="left" so the legend
        # left-aligns with the plot area. Anchor at top so the legend grows
        # downward (the chart's bottom margin reserves space for it).
        # y=-0.18 leaves ~30px gap between x-axis tick labels and legend.
        "legend": {"orientation": "h",
                    "yanchor": "top", "y": -0.18,
                    "xanchor": "left", "x": 0,
                    "font": {"size": 11, "color": "#0f172a"},
                    "bgcolor": "rgba(248,250,252,0.92)",
                    "bordercolor": "#cbd5e1", "borderwidth": 1,
                    "itemwidth": 40,
                    "itemsizing": "constant",
                    "tracegroupgap": 8,
                    "entrywidth": 0,
                    "entrywidthmode": "pixels"},
    }
    if ylabel2:
        layout["yaxis2"] = {
            "title": {"text": ylabel2, "font": {"size": 13}},
            "overlaying": "y", "side": "right",
            "gridcolor": "transparent", "linecolor": "#475569",
            "tickfont": {"size": 12, "color": "#0f172a"},
        }
    return layout


def _fit_plotly_legend(fig: dict | None) -> dict | None:
    """Normalize Plotly legend sizing so labels remain readable.

    v1.39.61 makes the right-side legend the default for multi-trace charts
    and reserves enough right margin for long trace labels.
    """
    if not fig:
        return fig
    data = fig.get("data") or []
    n = len([t for t in data if t is not None and t.get("showlegend", True) is not False])
    layout = fig.setdefault("layout", {})
    margin = dict(layout.get("margin") or {})
    margin["l"] = max(int(margin.get("l") or 0), 70)
    legend = dict(layout.get("legend") or {})

    if n >= 2:
        legend.update({"orientation": "v", "yanchor": "top", "y": 1.0,
                       "xanchor": "left", "x": 1.03})
        margin["r"] = max(int(margin.get("r") or 0), 320 if n <= 4 else 360)
        margin["b"] = max(int(margin.get("b") or 0), 110)
        layout["height"] = max(int(layout.get("height") or 460), 540 if n <= 4 else 580)
    else:
        legend.update({"orientation": "h", "yanchor": "top", "y": -0.14,
                       "xanchor": "left", "x": 0})
        margin["b"] = max(int(margin.get("b") or 0), 120)
        margin["r"] = max(int(margin.get("r") or 0), 90)

    legend.update({
        "font": {"size": 11 if n <= 4 else 10, "color": "#0f172a"},
        "bgcolor": "rgba(248,250,252,0.98)",
        "bordercolor": "#cbd5e1", "borderwidth": 1,
        "itemsizing": "constant",
        "tracegroupgap": 6,
        "itemwidth": 30,
    })
    legend.pop("entrywidth", None)
    legend.pop("entrywidthmode", None)
    layout["margin"] = margin
    layout["legend"] = legend
    layout["autosize"] = True
    return fig


def _chart_sglang_throughput(sglang: list[dict], bench: dict | None = None, sglang_summary: dict | None = None) -> dict | None:
    """Chart: token throughput + cache hit rate from SGLang/Prometheus.

    Prometheus-only reports must not render benchmark fallback lines.  If the
    explicit `sglang_gen_throughput` gauge is absent/all-zero, derive a flat throughput
    line from SGLang cumulative token counters that were already summarized in
    sglang_summary.json. Cache-hit reporting now prefers selected-window counter-derived
    token/request ratios; the raw `sglang_cache_hit_rate` gauge is plotted only as a
    time-weighted diagnostic because it is overwritten frequently.
    """
    sglang_summary = sglang_summary or {}
    if sglang:
        ts = _col(sglang, "time_sec")
    else:
        dur = float(sglang_summary.get("collection_elapsed_s") or 60.0)
        ts = [0.0, max(dur, 1.0)]
    if not ts:
        return None
    xs = _ts_to_dates(ts)
    gen_col = _find_col(sglang, "gen_throughput") if sglang else None
    cache_col = _find_col(sglang, "cache_hit_rate") if sglang else None

    traces = []
    y_tp = _col(sglang, gen_col) if gen_col else []
    if not any(v > 0 for v in y_tp):
        prom_tps = 0.0
        for key in ("ai_op_decode_tok_s", "gen_tp_active_mean", "gen_tp_mean", "decode_tok_s", "gen_tp_peak"):
            try:
                prom_tps = float(sglang_summary.get(key, 0) or 0)
                if prom_tps > 0:
                    break
            except Exception:
                prom_tps = 0.0
        y_tp = [prom_tps for _ in ts] if prom_tps > 0 else []
        tp_name = "Prometheus-derived output throughput (tok/s)"
    else:
        tp_name = "Throughput (tok/s)"
    if y_tp:
        traces.append({
            "x": xs, "y": y_tp,
            "type": "scatter", "mode": "lines", "name": tp_name,
            "line": {"color": "#22d3ee", "width": 2.2},
            "fill": "tozeroy", "fillcolor": "rgba(34,211,238,0.15)",
            "hovertemplate": "<b>%{y:.2f}</b> tok/s<extra></extra>",
        })

    cache_vals = []
    cache_name = "Cache hit %"
    if cache_col:
        # Keep an all-zero cache-hit gauge visible.  Dropping the trace when
        # every sample is 0 made users think the cache-hit chart regressed or
        # disappeared even though SGLang reported a valid cold-cache series.
        cache_vals = [v * 100 if v <= 1 else v for v in _col(sglang, cache_col)]
        cache_name = "Cache hit % (SGLang gauge)"
    if not cache_vals:
        prom_hit = None
        for key in ("cache_hit_token_weighted_pct", "cache_hit_request_weighted_pct", "cache_hit_time_weighted_pct", "cache_hit_rate_realtime_pct", "cache_hit_effective_prompt_pct", "cache_hit_cached_prompt_pct", "cache_hit_pct"):
            if key in sglang_summary:
                try:
                    prom_hit = float(sglang_summary.get(key, 0) or 0)
                    # Prefer a positive fallback, but preserve an explicit 0 if
                    # it is the only vetted cache-hit value available.
                    if prom_hit > 0:
                        break
                except Exception:
                    prom_hit = None
        if prom_hit is not None:
            cache_vals = [prom_hit for _ in ts]
            cache_name = "Cache hit % (summary fallback)"
    if cache_vals:
        traces.append({
            "x": xs, "y": cache_vals,
            "type": "scatter", "mode": "lines", "name": cache_name,
            "line": {"color": "#a78bfa", "width": 2.6, "dash": "dot"},
            "yaxis": "y2",
            "hovertemplate": "<b>%{y:.1f}%</b> hit<extra></extra>",
        })
    if not traces:
        return None
    return {"data": traces,
            "layout": _layout("SGLang Token Throughput & Cache Hit Rate",
                              "Throughput (tok/s)", "Cache hit %", ts=ts)}

def _derive_nvme_columns(rows: list[dict]) -> list[dict]:
    """Derive per-sample rate columns that are missing from the raw CSV.

    The nvme_driver_timeseries.csv written by the Prometheus loader contains
    only *cumulative* kernel counters (rd_ios, wr_ios, rd_ms, wr_ms, io_ms,
    rd_bytes, wr_bytes).  The static report derives rate columns inside
    extract_metrics() but does not save them back to the CSV, so the
    interactive report sees all-zero values when it tries to read rd_iops,
    rd_bw_mbs, etc.

    This function adds the missing columns in-place, exactly mirroring the
    derivation logic in amoprof.py::extract_metrics():

      rd_iops    = Δrd_ios  / Δtime_sec          (reads completed per sec)
      wr_iops    = Δwr_ios  / Δtime_sec          (writes completed per sec)
      rd_bw_mbs  = Δrd_bytes / Δtime_sec / 2²⁰  (read MB/s from byte counter)
      wr_bw_mbs  = Δwr_bytes / Δtime_sec / 2²⁰  (write MB/s from byte counter)
      rd_lat_ms  = Δrd_ms   / Δrd_ios            (avg read latency ms)
      wr_lat_ms  = Δwr_ms   / Δwr_ios            (avg write latency ms)
      io_util_pct= Δio_ms   / (Δtime_sec × 1000) × 100  (% time device was busy)

    If the columns are already present (iostat path writes them directly),
    they are left untouched so this function is safe to call unconditionally.

    Returns the same list with new dict keys added (mutates dicts in-place).
    """
    if not rows:
        return rows

    # Check what's already present
    existing = set(rows[0].keys())
    needs_iops   = "rd_iops" not in existing and "wr_iops" not in existing
    needs_bw     = "rd_bw_mbs" not in existing and "wr_bw_mbs" not in existing
    needs_lat    = "rd_lat_ms" not in existing and "wr_lat_ms" not in existing
    needs_util   = "io_util_pct" not in existing
    has_ios      = "rd_ios" in existing
    has_bytes    = "rd_bytes" in existing
    has_ms       = "rd_ms" in existing
    has_io_ms    = "io_ms" in existing

    if not (needs_iops or needs_bw or needs_lat or needs_util):
        return rows  # everything already present

    for i, row in enumerate(rows):
        if i == 0:
            # First row: no prior sample to diff against — set all to 0
            if needs_iops:
                row["rd_iops"] = row.get("rd_iops", 0.0)
                row["wr_iops"] = row.get("wr_iops", 0.0)
            if needs_bw:
                row["rd_bw_mbs"] = row.get("rd_bw_mbs", 0.0)
                row["wr_bw_mbs"] = row.get("wr_bw_mbs", 0.0)
            if needs_lat:
                row["rd_lat_ms"] = row.get("rd_lat_ms", 0.0)
                row["wr_lat_ms"] = row.get("wr_lat_ms", 0.0)
            if needs_util:
                row["io_util_pct"] = row.get("io_util_pct", 0.0)
            continue

        prev = rows[i - 1]
        dt = _to_float(row.get("time_sec")) - _to_float(prev.get("time_sec"))
        if dt <= 0:
            dt = 1.0

        if needs_iops and has_ios:
            d_rd = max(_to_float(row.get("rd_ios")) - _to_float(prev.get("rd_ios")), 0)
            d_wr = max(_to_float(row.get("wr_ios")) - _to_float(prev.get("wr_ios")), 0)
            row["rd_iops"] = round(d_rd / dt, 2)
            row["wr_iops"] = round(d_wr / dt, 2)

        if needs_bw and has_bytes:
            d_rdb = max(_to_float(row.get("rd_bytes")) - _to_float(prev.get("rd_bytes")), 0)
            d_wrb = max(_to_float(row.get("wr_bytes")) - _to_float(prev.get("wr_bytes")), 0)
            row["rd_bw_mbs"] = round(d_rdb / dt / (1024 * 1024), 3)
            row["wr_bw_mbs"] = round(d_wrb / dt / (1024 * 1024), 3)
        elif needs_bw and needs_iops and has_ios:
            # No byte counters: estimate from IOPS × a typical NVMe block size.
            # We use the actual KV block size if we can get it, else 128 KB
            # (typical large-IO average for KV workloads).
            _bsz_mb = 0.128  # 128 KB default
            row["rd_bw_mbs"] = round(_to_float(row.get("rd_iops", 0)) * _bsz_mb, 3)
            row["wr_bw_mbs"] = round(_to_float(row.get("wr_iops", 0)) * _bsz_mb, 3)

        if needs_lat and has_ms and has_ios:
            d_rdm = max(_to_float(row.get("rd_ms")) - _to_float(prev.get("rd_ms")), 0)
            d_wrm = max(_to_float(row.get("wr_ms")) - _to_float(prev.get("wr_ms")), 0)
            d_rd  = max(_to_float(row.get("rd_ios")) - _to_float(prev.get("rd_ios")), 0)
            d_wr  = max(_to_float(row.get("wr_ios")) - _to_float(prev.get("wr_ios")), 0)
            row["rd_lat_ms"] = round(d_rdm / d_rd, 3) if d_rd > 0 else 0.0
            row["wr_lat_ms"] = round(d_wrm / d_wr, 3) if d_wr > 0 else 0.0

        if needs_util and has_io_ms:
            d_iomsec = max(_to_float(row.get("io_ms")) - _to_float(prev.get("io_ms")), 0)
            # io_ms is accumulated milliseconds the device was busy;
            # utilisation = busy_ms / wall_ms * 100
            row["io_util_pct"] = round(min(d_iomsec / (dt * 1000) * 100, 100), 2)

    return rows



    """Chart: token throughput + cache hit rate."""
    if not sglang:
        return None
    gen_col = _find_col(sglang, "gen_throughput")
    cache_col = _find_col(sglang, "cache_hit_rate")
    if not gen_col:
        return None
    ts = _col(sglang, "time_sec")
    xs = _ts_to_dates(ts)
    traces = []
    traces.append({
        "x": xs, "y": _col(sglang, gen_col),
        "type": "scatter", "mode": "lines", "name": "Throughput (tok/s)",
        "line": {"color": "#22d3ee", "width": 2.2},
        "fill": "tozeroy", "fillcolor": "rgba(34,211,238,0.15)",
        "hovertemplate": "<b>%{y:.1f}</b> tok/s<extra></extra>",
    })
    if cache_col:
        cache_vals = [v * 100 if v <= 1 else v for v in _col(sglang, cache_col)]
        traces.append({
            "x": xs, "y": cache_vals,
            "type": "scatter", "mode": "lines", "name": "Cache hit %",
            "line": {"color": "#a78bfa", "width": 2, "dash": "dot"},
            "yaxis": "y2",
            "hovertemplate": "<b>%{y:.1f}%</b> hit<extra></extra>",
        })
    return {"data": traces,
            "layout": _layout("SGLang Token Throughput & Cache Hit Rate",
                              "Throughput (tok/s)", "Cache hit %", ts=ts)}


def _chart_gpu(gpu: list[dict],
               gpu_summary: dict | None = None,
               raw_dir: Path | None = None) -> dict | None:
    """Chart: per-GPU utilization + memory used.

    Robust against the gpu_timeseries.csv ↔ gpu_summary.json divergence
    bug observed in v56 reports (chart's mem trace all-zero while summary's
    hbm_used_mb_mean correctly read ~21 GB). Three defensive layers:

    1. Try multiple candidate column names for the mem column. Different
       collector paths (Prometheus vs local) and DCGM exporter versions
       use different names: `mem_used` (canonical), `mem_used_mb`,
       `hbm_used_mb`, `mem_used_gb` (gets ×1024'd to MB).

    2. If the resolved column is all-zero (or absent) but the gpu_summary
       file has a non-zero `hbm_used_mb_mean`, render the summary value
       as a flat trend line and prepend a clear warning annotation. This
       prevents the chart from silently showing 0 MB when we know it's
       wrong.

    3. If neither timeseries nor summary has data, log a warning and
       still render the util/power traces — never silently produce a
       broken chart.
    """
    # v1.39.109: make Interactive and End Report consume exactly the same
    # canonical GPU/HBM payload builder.  The previous Interactive-only chart
    # parsed gpu_timeseries.csv independently and could disagree with the End
    # Report on source selection, timestamp normalization, HBM column aliases,
    # active-GPU filtering, and y-axis visible set.  Prefer the shared builder;
    # keep the legacy implementation below only as a defensive fallback.
    if gpu:
        try:
            import pandas as _pd
            from amoprof.report.amoprof import Metrics as _Metrics, canonical_gpu_plotly_figure as _canonical_gpu_plotly_figure
            _m = _Metrics()
            _m.gpu_ts = _pd.DataFrame(gpu)
            if raw_dir is not None:
                setattr(_m, "_amoprof_raw_dir", str(Path(raw_dir)))
                # Source policy can live beside raw/ in flat output-dir mode.
                for _pol in (Path(raw_dir) / "amoprof_source_policy.json", Path(raw_dir).parent / "amoprof_source_policy.json"):
                    try:
                        if _pol.exists() and _pol.stat().st_size > 0:
                            with open(_pol) as _fh:
                                setattr(_m, "_amoprof_source_policy", json.load(_fh))
                            break
                    except Exception:
                        pass
            try:
                _ts = [_to_float(r.get("time_sec")) for r in gpu if r.get("time_sec") is not None]
                _ts = [x for x in _ts if x == x and x >= 0]
                if _ts:
                    _m.run_duration_s = max(_ts)
            except Exception:
                pass
            try:
                if gpu_summary:
                    _cap = _to_float(gpu_summary.get("hbm_total_mb") or gpu_summary.get("hbm_capacity_mb") or gpu_summary.get("fb_total_mb"))
                    if _cap and _cap > 0:
                        _m.hbm_total_mb = _cap
            except Exception:
                pass
            _fig = _canonical_gpu_plotly_figure(
                _m, points_per_gpu=900,
                title="GPU Utilization & HBM Memory — canonical payload shared with End Report"
            )
            if _fig:
                return _fig
        except Exception:
            pass

    if not gpu:
        return None
    # Candidate mem-column names in priority order. First match wins.
    # Units are tracked so we can normalize everything to MB.
    mem_candidates = [
        ("mem_used", 1.0),         # canonical (Prom & local), MB
        ("mem_used_mb", 1.0),      # alt naming
        ("hbm_used_mb", 1.0),      # legacy
        ("mem_used_gb", 1024.0),   # GB → MB
        ("hbm_used_gb", 1024.0),   # GB → MB
    ]
    # Detect which candidate the rows actually carry. Probe the first
    # non-empty row.
    mem_key, mem_scale = "mem_used", 1.0
    for r in gpu:
        for cand, scale in mem_candidates:
            v = r.get(cand)
            if v not in (None, "", "nan", "NaN"):
                mem_key, mem_scale = cand, scale
                break
        else:
            continue
        break

    by_gpu: dict[str, dict] = {}
    for r in gpu:
        idx = str(r.get("gpu_idx", "0"))
        d = by_gpu.setdefault(idx, {"t": [], "util": [], "mem": [], "power": []})
        d["t"].append(_to_float(r.get("time_sec")))
        d["util"].append(_to_float(r.get("gpu_util")))
        d["mem"].append(_to_float(r.get(mem_key)) * mem_scale)
        d["power"].append(_to_float(r.get("power")))
    colors = ["#22d3ee", "#a78bfa", "#f97316", "#22c55e",
              "#ef4444", "#3b82f6", "#eab308", "#ec4899"]
    traces = []
    # Keep the same active-device display semantics as the static End Report.
    # A GPU can have HBM/model memory resident while doing no work; using HBM
    # alone as the activity signal left all-zero GPU util traces in the chart.
    # Prefer compute-active util>0 devices; if util is unavailable for all GPUs,
    # fall back to memory-active rows so HBM-only collections still render.
    def _has_util_activity(d: dict) -> bool:
        try:
            return any(_to_float(v) > 0 for v in d.get("util", []))
        except Exception:
            return False

    def _has_mem_activity(d: dict) -> bool:
        try:
            return any(_to_float(v) > 0 for v in d.get("mem", []))
        except Exception:
            return False

    util_active_ids = {idx for idx, d in by_gpu.items() if _has_util_activity(d)}
    mem_active_ids = {idx for idx, d in by_gpu.items() if _has_mem_activity(d)}
    active_ids = util_active_ids or mem_active_ids
    plot_items = [(idx, d) for idx, d in by_gpu.items() if (not active_ids or idx in active_ids)]
    plot_items = sorted(plot_items, key=lambda x: int(x[0]) if x[0].isdigit() else 99)

    # Collect all timestamps for axis-format selection
    all_ts: list[float] = []
    for _, d in plot_items:
        all_ts.extend(d["t"])
    for i, (idx, d) in enumerate(plot_items):
        traces.append({
            "x": _ts_to_dates(d["t"]), "y": d["util"],
            "type": "scatter", "mode": "lines+markers", "name": f"GPU{idx} util",
            "line": {"color": colors[i % len(colors)], "width": 1.8},
            "marker": {"size": 4},
            "hovertemplate": f"<b>GPU{idx}</b>: %{{y:.1f}}%<extra></extra>",
        })
    if plot_items:
        try:
            # Overlay the mean of the same visible GPUs so the End Report and
            # Interactive view remain readable even when several util traces
            # sit at 98-100% and overlap.
            n = len(plot_items[0][1]["t"])
            mean_util = [sum(d["util"][i] for _, d in plot_items) / max(len(plot_items), 1) for i in range(n)]
            traces.append({
                "x": _ts_to_dates(plot_items[0][1]["t"]), "y": mean_util,
                "type": "scatter", "mode": "lines", "name": "Active GPU mean",
                "line": {"color": "#0f172a", "width": 2.8, "dash": "dash"},
                "hovertemplate": "<b>%{y:.1f}%</b> active mean<extra></extra>",
            })
        except Exception:
            pass
    # Aggregate mem on secondary axis over the same visible/active GPU set.
    mem_annotation = None
    if plot_items:
        n = len(plot_items[0][1]["t"])
        avg_mem = [sum(d["mem"][i] for _, d in plot_items) / len(plot_items)
                   for i in range(n)]
        avg_ts = plot_items[0][1]["t"]

        # Sanity check: is the mem series all-zero?
        mem_all_zero = (not avg_mem) or (max(avg_mem) == 0.0)
        if mem_all_zero:
            # Try to recover the mean from the summary file. The chart
            # then renders as a flat line at that value, with an
            # annotation explaining the recovery.
            summary_mb = 0.0
            if gpu_summary:
                summary_mb = float(gpu_summary.get("hbm_used_mb_mean", 0) or 0)
                if summary_mb == 0:
                    summary_gb = float(gpu_summary.get("hbm_used_gb_mean", 0) or 0)
                    summary_mb = summary_gb * 1024.0
            if summary_mb > 0:
                avg_mem = [summary_mb] * n
                mem_annotation = (
                    f"⚠ Per-sample HBM data was unavailable in "
                    f"gpu_timeseries.csv; showing summary mean "
                    f"({summary_mb/1024:.1f} GB) as a flat reference. "
                    f"Re-collect with the DCGM exporter exposing the "
                    f"FB_USED metric per GPU to see the time-series.")
            else:
                # No CSV data, no summary fallback — at least make it visible
                # that this is broken rather than silently rendering zeros.
                mem_annotation = (
                    "⚠ HBM memory data not captured for this run "
                    "(DCGM_FI_DEV_FB_USED unavailable). Verify the DCGM "
                    "exporter is collecting per-GPU memory metrics.")

        traces.append({
            "x": _ts_to_dates(avg_ts), "y": avg_mem,
            "type": "scatter", "mode": "lines",
            "name": "Mean mem used (MB, visible GPUs)",
            "line": {"color": "#7c3aed", "width": 2.5, "dash": "dashdot"},
            "yaxis": "y2",
            "hovertemplate": "<b>%{y:.0f}</b> MB used (mean)<extra></extra>",
        })
    layout = _layout("GPU Utilization & HBM Memory (per GPU)",
                     "Utilization %", "HBM used (MB)",
                     height=480, ts=all_ts)
    # Attach the recovery annotation in the chart's layout so it's visible
    # in both interactive and embedded contexts.
    if mem_annotation:
        layout.setdefault("annotations", []).append({
            "text": mem_annotation,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 1.08,
            "xanchor": "center", "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 11, "color": "#dc2626"},
            "bgcolor": "#fef3c7",
            "bordercolor": "#fbbf24", "borderwidth": 1,
        })
    return {"data": traces, "layout": layout}


def _chart_power(power: list[dict], gpu: list[dict] | None = None) -> dict | None:
    """System GPU power chart.

    Primary source: power_timeseries.csv (already summed across GPUs).
    Fallback: gpu_timeseries.csv per-GPU 'power' column summed by timestamp,
    used when the primary source is missing or all-zero.
    """
    ts: list[float] = []
    pw: list[float] = []
    if power:
        ts = _col(power, "time_sec")
        pw = _col(power, "gpu_power")
    # If primary source absent or all-zero, synthesize from per-GPU gpu_ts
    if (not pw or not any(pw)) and gpu:
        try:
            from collections import defaultdict
            sums: dict[float, float] = defaultdict(float)
            for r in gpu:
                t = _to_float(r.get("time_sec"))
                p = _to_float(r.get("power"))
                if t is not None and p:
                    sums[t] += p
            if sums:
                ts = sorted(sums.keys())
                pw = [sums[t] for t in ts]
        except Exception:
            pass
    if not pw or not any(pw):
        return None
    xs = _ts_to_dates(ts)
    return {
        "data": [{
            "x": xs, "y": pw,
            "type": "scatter", "mode": "lines",
            "name": "Total GPU power (W)",
            "line": {"color": "#fb923c", "width": 2.2},
            "fill": "tozeroy", "fillcolor": "rgba(251,146,60,0.2)",
            "hovertemplate": "<b>%{y:.0f}</b> W total<extra></extra>",
        }],
        "layout": _layout("System GPU Power Draw", "Watts (sum of 8 GPUs)", ts=ts)
    }


def _chart_swap_storm(vmstat: list[dict]) -> dict | None:
    if not vmstat:
        return None
    ts = _col(vmstat, "time_sec")
    xs = _ts_to_dates(ts)
    # Convert cumulative counters to per-second rates
    def _rate(col: str) -> list[float]:
        v = _col(vmstat, col)
        if len(v) < 2: return []
        rates = [0.0]
        for i in range(1, len(v)):
            dt = ts[i] - ts[i-1]
            rates.append((v[i] - v[i-1]) / dt if dt > 0 else 0)
        return rates
    swpin  = _rate("pswpin")
    swpout = _rate("pswpout")
    pgmaj  = _rate("pgmajfault")
    # Render even when swap is zero — major faults and the zero-swap confirmation
    # are both valuable. Previously this returned None if swap counters were
    # all zero, making the chart silently disappear from the interactive report
    # while the static report still showed a placeholder.
    has_any_data = bool(any(swpin) or any(swpout) or any(pgmaj))
    # Build the chart — always render (even if all-zero) so the interactive
    # report matches the static report's coverage. Add a green annotation
    # confirming no swap occurred when swap counters are all-zero.
    layout = _layout("Swap Storm — Page Swap & Major Fault Rates",
                      "Pages / sec", "Major faults / sec", height=480, ts=ts)
    if not any(swpin) and not any(swpout):
        # Add a note that swap did not occur — this is a healthy state
        layout.setdefault("annotations", []).append({
            "text": "✅ No swap I/O detected — kernel memory was not under pressure",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "xanchor": "center",
            "showarrow": False,
            "font": {"size": 14, "color": "#22c55e"},
        })
    return {
        "data": [
            {"x": xs, "y": swpin,  "type": "scatter", "mode": "lines",
             "name": "Swap-in pages/s",
             "line": {"color": "#3b82f6", "width": 2},
             "hovertemplate": "<b>%{y:.0f}</b> pages/s in<extra></extra>"},
            {"x": xs, "y": swpout, "type": "scatter", "mode": "lines",
             "name": "Swap-out pages/s",
             "line": {"color": "#ef4444", "width": 2},
             "fill": "tozeroy", "fillcolor": "rgba(239,68,68,0.15)",
             "hovertemplate": "<b>%{y:.0f}</b> pages/s out<extra></extra>"},
            {"x": xs, "y": pgmaj, "type": "scatter", "mode": "lines",
             "name": "Major faults/s", "yaxis": "y2",
             "line": {"color": "#a16207", "width": 1.5, "dash": "dot"},
             "hovertemplate": "<b>%{y:.1f}</b> major/s<extra></extra>"},
        ],
        "layout": layout,
    }


def _chart_nvme_iops(nvme: list[dict], label: str = "Host Block I/O") -> dict | None:
    if not nvme:
        return None
    ts = _col(nvme, "time_sec")
    xs = _ts_to_dates(ts)
    rd = _col(nvme, "rd_iops")
    wr = _col(nvme, "wr_iops")
    disc = _col(nvme, "disc_ios")
    traces = [
        {"x": xs, "y": rd, "type": "scatter", "mode": "lines",
         "name": "Read IOPS",
         "line": {"color": "#22c55e", "width": 1.8},
         "hovertemplate": "<b>%{y:.0f}</b> rd IOPS<extra></extra>"},
        {"x": xs, "y": wr, "type": "scatter", "mode": "lines",
         "name": "Write IOPS",
         "line": {"color": "#f43f5e", "width": 1.8},
         "fill": "tozeroy", "fillcolor": "rgba(244,63,94,0.12)",
         "hovertemplate": "<b>%{y:.0f}</b> wr IOPS<extra></extra>"},
    ]
    if any(disc):
        traces.append({
            "x": xs, "y": disc, "type": "scatter", "mode": "lines",
            "name": "Discard IOPS", "yaxis": "y2",
            "line": {"color": "#eab308", "width": 1.5, "dash": "dash"},
            "hovertemplate": "<b>%{y:.0f}</b> trim IOPS<extra></extra>",
        })
    return {"data": traces,
            "layout": _layout(f"{label} Rate (Read / Write / Trim)",
                              "IOPS", "Trim IOPS" if any(disc) else "", ts=ts)}


def _chart_nvme_bw(nvme: list[dict], label: str = "Host Block I/O") -> dict | None:
    if not nvme:
        return None
    ts = _col(nvme, "time_sec")
    xs = _ts_to_dates(ts)
    rd = _col(nvme, "rd_bw_mbs")
    wr = _col(nvme, "wr_bw_mbs")
    return {
        "data": [
            {"x": xs, "y": rd, "type": "scatter", "mode": "lines",
             "name": "Read MB/s",
             "line": {"color": "#22c55e", "width": 2},
             "hovertemplate": "<b>%{y:.1f}</b> MB/s read<extra></extra>"},
            {"x": xs, "y": wr, "type": "scatter", "mode": "lines",
             "name": "Write MB/s",
             "line": {"color": "#f43f5e", "width": 2},
             "fill": "tozeroy", "fillcolor": "rgba(244,63,94,0.12)",
             "hovertemplate": "<b>%{y:.1f}</b> MB/s write<extra></extra>"},
        ],
        "layout": _layout(f"{label} Bandwidth (Read / Write)", "MB / sec", ts=ts)
    }


def _chart_nvme_latency(nvme: list[dict], label: str = "Host Block I/O") -> dict | None:
    if not nvme:
        return None
    ts = _col(nvme, "time_sec")
    xs = _ts_to_dates(ts)
    rd = _col(nvme, "rd_lat_ms")
    wr = _col(nvme, "wr_lat_ms")
    util = _col(nvme, "io_util_pct")
    rd_bw = _col(nvme, "rd_bw_mbs")
    wr_bw = _col(nvme, "wr_bw_mbs")
    rd_iops = _col(nvme, "rd_iops")
    wr_iops = _col(nvme, "wr_iops")

    has_latency_or_util = any(v and v > 0 for v in (rd + wr + util))
    if has_latency_or_util:
        return {
            "data": [
                {"x": xs, "y": rd, "type": "scatter", "mode": "lines",
                 "name": "Read latency (ms)",
                 "line": {"color": "#3b82f6", "width": 1.8},
                 "hovertemplate": "<b>%{y:.3f}</b> ms<extra></extra>"},
                {"x": xs, "y": wr, "type": "scatter", "mode": "lines",
                 "name": "Write latency (ms)",
                 "line": {"color": "#ef4444", "width": 1.8},
                 "hovertemplate": "<b>%{y:.3f}</b> ms<extra></extra>"},
                {"x": xs, "y": util, "type": "scatter", "mode": "lines",
                 "name": "Device util %", "yaxis": "y2",
                 "line": {"color": "#94a3b8", "width": 1.4, "dash": "dot"},
                 "hovertemplate": "<b>%{y:.1f}</b>%% util<extra></extra>"},
            ],
            "layout": _layout(f"{label} Latency & Device Busy-Time",
                              "Latency (ms)", "Util %", ts=ts)
        }

    # Fallback: read/write activity exists, but the source file did not include
    # latency/busy counters (rd_ms/wr_ms/io_ms, iostat await/busy-time, or blktrace
    # latency).  Do not render an empty chart; show the available activity and
    # make the missing latency provenance explicit in the title/legend.
    has_activity = any(v and v > 0 for v in (rd_bw + wr_bw + rd_iops + wr_iops))
    if not has_activity:
        return None
    data = [
        {"x": xs, "y": rd_bw, "type": "scatter", "mode": "lines",
         "name": "Read MB/s (latency unavailable)",
         "line": {"color": "#22c55e", "width": 2},
         "hovertemplate": "<b>%{y:.2f}</b> MB/s read<extra></extra>"},
        {"x": xs, "y": wr_bw, "type": "scatter", "mode": "lines",
         "name": "Write MB/s (latency unavailable)",
         "line": {"color": "#f43f5e", "width": 2},
         "hovertemplate": "<b>%{y:.2f}</b> MB/s write<extra></extra>"},
        {"x": xs, "y": rd_iops, "type": "scatter", "mode": "lines",
         "name": "Read IOPS", "yaxis": "y2",
         "line": {"color": "#16a34a", "width": 1.4, "dash": "dot"},
         "hovertemplate": "<b>%{y:.0f}</b> read IOPS<extra></extra>"},
        {"x": xs, "y": wr_iops, "type": "scatter", "mode": "lines",
         "name": "Write IOPS", "yaxis": "y2",
         "line": {"color": "#be123c", "width": 1.4, "dash": "dot"},
         "hovertemplate": "<b>%{y:.0f}</b> write IOPS<extra></extra>"},
    ]
    layout = _layout(f"{label} Activity Fallback — Latency/Utilization Counters Missing",
                     "MB/s", "IOPS", ts=ts)
    layout.setdefault("annotations", []).append({
        "text": "Read/write activity exists, but latency/busy fields are unavailable or zero. Showing BW/IOPS fallback instead of an empty latency chart.",
        "xref": "paper", "yref": "paper", "x": 0.0, "y": 1.08,
        "xanchor": "left", "showarrow": False,
        "font": {"size": 11, "color": "#64748b"},
    })
    return {"data": data, "layout": layout}


def _chart_qd_timeseries(qd_ts: list[dict]) -> dict | None:
    """Interactive time-series of NVMe queue depth (from blktrace Q→C walk).

    qd_ts rows are: t_sec, qd_total, qd_read, qd_write — produced by
    blktrace_analyzer._compute_queue_depth_timeseries.

    Two stacked traces (read + write) so a viewer can see whether saturation
    is read-driven (HiCache load-back) or write-driven (KV offload). The
    total trace sits on top as a thin overlay line for at-a-glance peak QD.
    """
    if not qd_ts:
        return None
    ts = [_to_float(r.get("t_sec")) or 0 for r in qd_ts]
    qd_t = [_to_float(r.get("qd_total")) or 0 for r in qd_ts]
    qd_r = [_to_float(r.get("qd_read"))  or 0 for r in qd_ts]
    qd_w = [_to_float(r.get("qd_write")) or 0 for r in qd_ts]
    layout = dict(_layout("NVMe Queue Depth Over Time", "Outstanding I/Os", ts=ts))
    layout["hovermode"] = "x unified"
    return {
        "data": [
            {"x": _ts_to_dates(ts), "y": qd_r, "type": "scatter", "mode": "lines",
             "name": "Read QD",   "stackgroup": "qd",
             "line": {"color": "#3b82f6", "width": 0},
             "fillcolor": "rgba(59,130,246,0.55)",
             "hovertemplate": "<b>%{y:.0f}</b> reads in-flight<extra></extra>"},
            {"x": _ts_to_dates(ts), "y": qd_w, "type": "scatter", "mode": "lines",
             "name": "Write QD",  "stackgroup": "qd",
             "line": {"color": "#ef4444", "width": 0},
             "fillcolor": "rgba(239,68,68,0.55)",
             "hovertemplate": "<b>%{y:.0f}</b> writes in-flight<extra></extra>"},
            {"x": _ts_to_dates(ts), "y": qd_t, "type": "scatter", "mode": "lines",
             "name": "Total QD (peak per bucket)",
             "line": {"color": "#7c3aed", "width": 2.2},
             "hovertemplate": "<b>%{y:.0f}</b> total outstanding<extra></extra>"},
        ],
        "layout": layout,
    }


def _chart_qd_distribution(qd_dist: list[dict]) -> dict | None:
    """Binned time-weighted QD histogram with meaningful risk legend."""
    if not qd_dist:
        return None
    def _risk(q: int) -> tuple[str, str]:
        if q <= 1: return ("Idle", "#0ea5e9")
        if q <= 31: return ("Healthy", "#22c55e")
        if q <= 127: return ("Deep queue", "#f59e0b")
        return ("Saturation risk", "#ef4444")
    bucket_pct: dict[str, float] = {}
    bucket_meta: dict[str, tuple[str, str]] = {}
    for r in qd_dist:
        q = int(_to_float(r.get("qd_value")) or 0)
        pct = _to_float(r.get("pct_of_run")) or 0.0
        legend_name, color = _risk(q)
        if q <= 1:
            xlab = str(q)
        elif q < 32:
            lo = max(2, (q // 4) * 4); hi = lo + 3; xlab = f"{lo}–{hi}"
        elif q < 128:
            lo = (q // 16) * 16; hi = lo + 15; xlab = f"{lo}–{hi}"
        else:
            lo = 1
            while lo * 2 <= q: lo *= 2
            hi = lo * 2 - 1; xlab = f"{lo}–{hi}"
        key = f"{xlab}|{legend_name}"
        bucket_pct[key] = bucket_pct.get(key, 0.0) + pct
        bucket_meta[key] = (legend_name, color)
    def _lo(key: str) -> int:
        m = re.match(r"(\d+)", key)
        return int(m.group(1)) if m else 0
    traces = []
    for legend_name, color in [("Idle","#0ea5e9"),("Healthy","#22c55e"),("Deep queue","#f59e0b"),("Saturation risk","#ef4444")]:
        xs, ys = [], []
        for key in sorted(bucket_pct, key=_lo):
            name, _ = bucket_meta[key]
            if name == legend_name:
                xs.append(key.split("|",1)[0]); ys.append(bucket_pct[key])
        if xs:
            traces.append({"x": xs, "y": ys, "type": "bar", "name": legend_name,
                           "marker": {"color": color},
                           "hovertemplate": "<b>QD bucket %{x}</b><br>%{y:.2f}% of trace time<extra></extra>"})
    layout = dict(_layout("Queue Depth Distribution (time-weighted, binned)", "% of trace time", height=430))
    layout["xaxis"] = {"type": "category", "title": {"text": "Queue depth bucket (outstanding I/Os)", "standoff": 12},
                       "gridcolor": "#cbd5e1", "zerolinecolor": "#94a3b8", "linecolor": "#475569", "tickcolor": "#475569",
                       "tickfont": {"size": 11, "color": "#0f172a"}, "automargin": True}
    layout["barmode"] = "stack"
    layout["hovermode"] = "closest"
    return _fit_plotly_legend({"data": traces, "layout": layout})



def _chart_kv_tiers(sglang: list[dict]) -> dict | None:
    if not sglang:
        return None
    ts = _col(sglang, "time_sec")
    xs = _ts_to_dates(ts)
    dev_col = _find_col(sglang, "cached_tokens_total[source=device]")
    host_col = _find_col(sglang, "cached_tokens_total[source=host]")
    storage_col = _find_col(sglang, "cached_tokens_total[source=storage]")
    traces = []
    if dev_col:
        traces.append({"x": xs, "y": _col(sglang, dev_col),
                       "type": "scatter", "mode": "lines", "stackgroup": "kv",
                       "name": "L1 Device (HBM)",
                       "line": {"color": "#7c3aed", "width": 0},
                       "fillcolor": "rgba(124,58,237,0.7)",
                       "hovertemplate": "<b>L1</b>: %{y:.0f} tok<extra></extra>"})
    if host_col:
        traces.append({"x": xs, "y": _col(sglang, host_col),
                       "type": "scatter", "mode": "lines", "stackgroup": "kv",
                       "name": "L2 Host (DRAM)",
                       "line": {"color": "#22d3ee", "width": 0},
                       "fillcolor": "rgba(34,211,238,0.65)",
                       "hovertemplate": "<b>L2</b>: %{y:.0f} tok<extra></extra>"})
    if storage_col:
        traces.append({"x": xs, "y": _col(sglang, storage_col),
                       "type": "scatter", "mode": "lines", "stackgroup": "kv",
                       "name": "L3 (local storage)",
                       "line": {"color": "#f97316", "width": 0},
                       "fillcolor": "rgba(249,115,22,0.6)",
                       "hovertemplate": "<b>L3</b>: %{y:.0f} tok<extra></extra>"})

    # Many SGLang builds do not export tier-labeled cached_tokens_total.
    # Fallback to movement counters only when reliable L3 (local storage) evidence exists.
    # backuped_tokens_total or prefetched_tokens_total prove L3 (local storage) use;
    # load_back-only (L2→L1 restore) and evicted-only are ambiguous/cache-pressure signals and
    # must not create L3 (local storage) charts.
    if not traces:
        bk_col = _find_col(sglang, "backuped_tokens_total")
        pf_col = _find_col(sglang, "prefetched_tokens_total")
        lb_col = _find_col(sglang, "load_back_tokens_total")
        ev_col = _find_col(sglang, "evicted_tokens_total")
        bk = _col(sglang, bk_col) if bk_col else [0.0] * len(ts)
        pf = _col(sglang, pf_col) if pf_col else [0.0] * len(ts)
        lb = _col(sglang, lb_col) if lb_col else [0.0] * len(ts)
        ev = _col(sglang, ev_col) if ev_col else [0.0] * len(ts)
        bk_delta = max((bk[-1] - bk[0]) if len(bk) > 1 else 0.0, 0.0)
        pf_delta = max((pf[-1] - pf[0]) if len(pf) > 1 else 0.0, 0.0)
        l3_reliable = (bk_delta > 0 or pf_delta > 0)
        if l3_reliable:
            resident = [max(b - p, 0.0) for b, p in zip(bk, pf)]
            if any(resident):
                traces.append({"x": xs, "y": resident, "type": "scatter",
                               "mode": "lines", "name": "Approx L3 (local storage) resident tokens",
                               "fill": "tozeroy", "fillcolor": "rgba(249,115,22,0.18)",
                               "line": {"color": "#f97316", "width": 2},
                               "hovertemplate": "<b>%{y:.0f}</b> approx resident tok<extra></extra>"})
            if any(bk):
                traces.append({"x": xs, "y": bk, "type": "scatter",
                               "mode": "lines", "name": "Cumulative L3 backup tokens",
                               "line": {"color": "#3b82f6", "width": 1.6, "dash": "dash"},
                               "hovertemplate": "<b>%{y:.0f}</b> backup tok<extra></extra>"})
            if any(pf):
                traces.append({"x": xs, "y": pf, "type": "scatter",
                               "mode": "lines", "name": "Cumulative L3 prefetched tokens",
                               "line": {"color": "#f97316", "width": 1.6, "dash": "dot"},
                               "hovertemplate": "<b>%{y:.0f}</b> prefetched tok<extra></extra>"})
            if any(lb):
                traces.append({"x": xs, "y": lb, "type": "scatter",
                               "mode": "lines", "name": "Cumulative L2→L1 load-back restore tokens (diagnostic, not SSD/L3 bytes)",
                               "line": {"color": "#22c55e", "width": 1.8},
                               "hovertemplate": "<b>%{y:.0f}</b> L2→L1 load-back restore tokens<extra></extra>"})
            if any(ev):
                traces.append({"x": xs, "y": ev, "type": "scatter",
                               "mode": "lines", "name": "Cumulative evicted tokens",
                               "line": {"color": "#ef4444", "width": 1.6, "dash": "dot"},
                               "hovertemplate": "<b>%{y:.0f}</b> evicted tok<extra></extra>"})
    if not traces:
        return None
    return {"data": traces,
            "layout": _layout("KV Cache Tier / L3 (local storage) movement Occupancy",
                              "Tokens", height=480, ts=ts)}


def _chart_eviction(sglang: list[dict]) -> dict | None:
    if not sglang:
        return None
    ts = _col(sglang, "time_sec")
    xs = _ts_to_dates(ts)
    ev_col = _find_col(sglang, "evicted_tokens_total")
    lb_col = _find_col(sglang, "load_back_tokens_total")
    bk_col = _find_col(sglang, "backuped_tokens_total")
    if not (ev_col or lb_col or bk_col):
        return None
    def _rate(col: str) -> list[float]:
        if not col: return [0.0] * len(ts)
        v = _col(sglang, col)
        out = [0.0]
        for i in range(1, len(v)):
            dt = ts[i] - ts[i-1]
            out.append(max((v[i] - v[i-1]) / dt, 0) if dt > 0 else 0)
        return out
    traces = []
    if ev_col:
        traces.append({"x": xs, "y": _rate(ev_col),
                       "type": "scatter", "mode": "lines", "name": "Evicted tok/s",
                       "line": {"color": "#ef4444", "width": 2},
                       "fill": "tozeroy", "fillcolor": "rgba(239,68,68,0.15)",
                       "hovertemplate": "<b>%{y:.0f}</b> tok/s evicted<extra></extra>"})
    if lb_col:
        traces.append({"x": xs, "y": _rate(lb_col),
                       "type": "scatter", "mode": "lines", "name": "L2→L1 load-back restore tok/s",
                       "line": {"color": "#22c55e", "width": 2},
                       "hovertemplate": "<b>%{y:.0f}</b> tok/s loaded<extra></extra>"})
    if bk_col:
        traces.append({"x": xs, "y": _rate(bk_col),
                       "type": "scatter", "mode": "lines", "name": "Backup tok/s",
                       "line": {"color": "#3b82f6", "width": 1.6, "dash": "dot"},
                       "hovertemplate": "<b>%{y:.0f}</b> tok/s backup<extra></extra>"})
    return {"data": traces,
            "layout": _layout("KV Cache Eviction / L2→L1 Load-back / Backup Rates",
                              "Tokens / sec", ts=ts)}


def _kv_bytes_per_token_for_model(model_name: str, kv_dtype: str) -> float:
    """Estimate KV cache bytes per token from model architecture.

    Mirrored from amoprof.KV_ARCH_DB / combined._kv_bytes_per_token_kb so
    the interactive tab can derive byte-level L3 (local storage) BW from token-level SGLang
    counters without depending on the static-report module.
    """
    arch_db = {
        "gpt-oss-120b": (36, 8, 64, "GQA"),
        "gpt-oss-20b":  (24, 8, 64, "GQA"),
        "deepseek-r1-distill-llama-70b": (80, 8, 128, "GQA"),
        "deepseek-r1-distill-qwen-32b":  (64, 8, 128, "GQA"),
        "deepseek-r1-distill-qwen-14b":  (48, 8, 128, "GQA"),
        "deepseek-v3":      (61, 128, 512, "MLA"),
        "qwen3-235b":       (94, 4, 512, "MLA"),
        "llama-3.1-405b":  (126, 8, 128, "GQA"),
        "llama-3.1-70b":    (80, 8, 128, "GQA"),
        "llama-3-70b":      (80, 8, 128, "GQA"),
    }
    ml = (model_name or "").lower()
    arch = None
    for key, val in arch_db.items():
        if key in ml:
            arch = val; break
    if arch is None:
        for hint, val in [("671b",(61,128,512,"MLA")),
                          ("405b",(126,8,128,"GQA")),
                          ("235b",(94,4,512,"MLA")),
                          ("120b",(36,8,64,"GQA")),
                          ("72b",(80,8,128,"GQA")),
                          ("70b",(80,8,128,"GQA")),
                          ("32b",(64,8,128,"GQA")),
                          ("20b",(24,8,64,"GQA")),
                          ("14b",(48,8,128,"GQA")),
                          ("13b",(40,8,128,"GQA")),
                          ("8b",(32,8,128,"GQA")),
                          ("7b",(28,4,128,"GQA")),
                          ("3b",(32,8,128,"GQA"))]:
            if hint in ml:
                arch = val; break
    if arch is None:
        arch = (80, 8, 128, "GQA")
    n_layers, n_kv_heads, head_dim, attn = arch
    dtype_bytes = 1 if (kv_dtype or "").lower().startswith(("fp8","int8","int4")) else 2
    if attn == "MLA":
        return n_layers * head_dim * dtype_bytes * 2  # bytes, not KB
    return n_layers * n_kv_heads * head_dim * 2 * dtype_bytes



def _delta_from_ts(rows: list[dict], partial: str) -> float:
    """Return selected-window positive delta for a cumulative counter column."""
    col = _find_col(rows, partial) if rows else None
    if not col:
        return 0.0
    vals = [_to_float(r.get(col), float("nan")) for r in rows]
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return 0.0
    return max(vals[-1] - vals[0], 0.0)


def _summary_or_ts_delta(summary: dict, rows: list[dict], *keys: str) -> float:
    """Prefer summary totals, then selected-window timeseries deltas."""
    for k in keys:
        v = _to_float(summary.get(k), 0.0) if summary else 0.0
        if v > 0:
            return v
    for k in keys:
        v = _delta_from_ts(rows, k)
        if v > 0:
            return v
    return 0.0


def _block_gb_from_summary(summary: dict, direction: str) -> float:
    """Physical local-block GB from blktrace/iostat summaries only."""
    if not summary:
        return 0.0
    if direction == "read":
        gb_keys = ("nvme_read_total_gb", "read_gb_total", "read_GB_total", "blktrace_read_gb", "read_gb")
        byte_keys = ("read_bytes_total", "rd_bytes_total", "blktrace_read_bytes", "read_bytes")
    else:
        gb_keys = ("nvme_write_total_gb", "write_gb_total", "write_GB_total", "blktrace_write_gb", "write_gb")
        byte_keys = ("write_bytes_total", "wr_bytes_total", "blktrace_write_bytes", "write_bytes")
    for k in gb_keys:
        v = _to_float(summary.get(k), 0.0)
        if v > 0:
            return v
    for k in byte_keys:
        v = _to_float(summary.get(k), 0.0)
        if v > 0:
            return v / (1024**3)
    return 0.0


def _chart_l3_consistency(sglang_read_gb: float, sglang_write_gb: float,
                          block_read_gb: float, block_write_gb: float) -> dict:
    return {
        "data": [
            {"x": ["Read", "Write"], "y": [sglang_read_gb, sglang_write_gb],
             "type": "bar", "name": "SGLang logical L3", "hovertemplate": "%{x}: %{y:.3f} GB<extra></extra>"},
            {"x": ["Read", "Write"], "y": [block_read_gb, block_write_gb],
             "type": "bar", "name": "Local block physical", "hovertemplate": "%{x}: %{y:.3f} GB<extra></extra>"},
        ],
        "layout": {**_layout("L3 consistency: SGLang logical vs local-source physical bytes", "GB", height=380),
                   "barmode": "group",
                   "xaxis": {"type": "category", "title": {"text": "Direction"},
                             "gridcolor": "#cbd5e1", "linecolor": "#475569",
                             "tickfont": {"size": 12, "color": "#0f172a"}, "automargin": True}},
    }


def _l3_consistency_card(status: str, note: str, backend_display: str, backend_evidence: str,
                         sglang_read_gb: float, sglang_write_gb: float,
                         block_read_gb: float, block_write_gb: float,
                         chart_id: str = "ch_l3_consistency") -> str:
    fig = _fit_plotly_legend(_chart_l3_consistency(sglang_read_gb, sglang_write_gb, block_read_gb, block_write_gb))
    fig_json = json.dumps(fig, default=str)
    return f"""
<section class="card">
  <h2>L3 Consistency Check — SGLang vs local sources</h2>
  <div class="caption"><b>Status:</b> {status}<br>
  <b>Resolved backend:</b> {backend_display} &nbsp;·&nbsp; <b>Evidence:</b> {backend_evidence or 'not available'}<br>
  <b>Rule:</b> SGLang counters are logical KV movement. blktrace/blkparse/iostat are physical local-block bytes. Visualizations compare them side-by-side; they are not merged unless backend is resolved as L3 local SSD and the physical capture is complete.<br>
  <b>Note:</b> {note}</div>
  <div id="{chart_id}" class="plot"></div>
  <table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:13px">
    <tr><th style="text-align:left;padding:6px;border-bottom:1px solid #334155">Direction</th><th style="text-align:right;padding:6px;border-bottom:1px solid #334155">SGLang logical GB</th><th style="text-align:right;padding:6px;border-bottom:1px solid #334155">Local physical GB</th></tr>
    <tr><td style="padding:6px">Read: prefetched only (L2→L1 load_back restore diagnostic)</td><td style="text-align:right;padding:6px">{sglang_read_gb:.3f}</td><td style="text-align:right;padding:6px">{block_read_gb:.3f}</td></tr>
    <tr><td style="padding:6px">Write: backuped</td><td style="text-align:right;padding:6px">{sglang_write_gb:.3f}</td><td style="text-align:right;padding:6px">{block_write_gb:.3f}</td></tr>
  </table>
  <script>
    (function() {{
      var fig = {fig_json};
      Plotly.newPlot('{chart_id}', fig.data, fig.layout,
        {{responsive:true, displayModeBar:true, displaylogo:false,
          modeBarButtonsToRemove:['select2d','lasso2d']}});
    }})();
  </script>
</section>"""

def _chart_l3_prom_bandwidth(sglang: list[dict],
                              kv_bytes_per_token: float) -> dict | None:
    """Interactive estimated L3 (local storage) bandwidth in MB/s (Prometheus-derived).

    Multiplies per-second backuped/prefetched token rates by kv_bytes_per_token
    to estimate logical L3 write/read movement. load_back_tokens_total is plotted
    only as a diagnostic upper bound and is not counted as L3 local SSD bytes.
    This is the Prometheus-only logical KV movement view; blktrace/iostat remain
    the source of truth for physical SSD bytes when a concrete L3 device is mapped.

    Caveat: byte estimate is logical KV movement, not raw device bytes.

    Returns None when SGLang token counters aren't present or
    kv_bytes_per_token isn't set.
    """
    if not sglang or kv_bytes_per_token <= 0:
        return None
    ts = _col(sglang, "time_sec")
    if not ts:
        return None
    lb_col = _find_col(sglang, "load_back_tokens_total")
    bk_col = _find_col(sglang, "backuped_tokens_total")
    pf_col = _find_col(sglang, "prefetched_tokens_total")
    # Do not render L3 bandwidth from load_back-only. Require backup or prefetch.
    _bk_v = _col(sglang, bk_col) if bk_col else []
    _pf_v = _col(sglang, pf_col) if pf_col else []
    _bk_delta = max((_bk_v[-1] - _bk_v[0]) if len(_bk_v) > 1 else 0.0, 0.0)
    _pf_delta = max((_pf_v[-1] - _pf_v[0]) if len(_pf_v) > 1 else 0.0, 0.0)
    if _bk_delta <= 0 and _pf_delta <= 0:
        return None

    def _rate_mbs(col: str) -> list[float]:
        if not col: return [0.0] * len(ts)
        v = _col(sglang, col)
        out = [0.0]
        for i in range(1, len(v)):
            dt = ts[i] - ts[i-1]
            tok_per_s = max((v[i] - v[i-1]) / dt, 0) if dt > 0 else 0
            # tok/s × bytes/tok / 1e6 = MB/s
            out.append(tok_per_s * kv_bytes_per_token / 1e6)
        return out

    xs = _ts_to_dates(ts)
    traces = []
    if pf_col:
        rd_mbs = _rate_mbs(pf_col)
        traces.append({
            "x": xs, "y": rd_mbs,
            "type": "scatter", "mode": "lines", "name": "L3 prefetch/read MB/s (est)",
            "line": {"color": "#22c55e", "width": 2.2},
            "fill": "tozeroy", "fillcolor": "rgba(34,197,94,0.15)",
            "hovertemplate": "<b>%{y:.1f}</b> MB/s estimated prefetch/read<extra></extra>",
        })
    if lb_col:
        traces.append({
            "x": xs, "y": _rate_mbs(lb_col),
            "type": "scatter", "mode": "lines", "name": "L2→L1 load-back restore MB/s (diagnostic)",
            "line": {"color": "#94a3b8", "width": 1.5, "dash": "dot"},
            "hovertemplate": "<b>%{y:.1f}</b> MB/s L2→L1 restore; not SSD/L3 bytes<extra></extra>",
        })
    if bk_col:
        wr_mbs = _rate_mbs(bk_col)
        traces.append({
            "x": xs, "y": wr_mbs,
            "type": "scatter", "mode": "lines", "name": "L3 (local storage) write MB/s (est)",
            "line": {"color": "#3b82f6", "width": 2.2},
            "fill": "tozeroy", "fillcolor": "rgba(59,130,246,0.15)",
            "hovertemplate": "<b>%{y:.1f}</b> MB/s estimated write<extra></extra>",
        })
    return {
        "data": traces,
        "layout": _layout(
            f"L3 (local storage) cache Bandwidth — Prometheus-derived  "
            f"(× {kv_bytes_per_token/1024:.0f} KB/tok)",
            "MB/s (estimated)", ts=ts),
    }


def _chart_l3_prom_cumulative(sglang: list[dict],
                               kv_bytes_per_token: float) -> dict | None:
    """Cumulative L3 read/write bytes over time (Prometheus-derived).

    Useful for sizing the endurance footprint of an L3 (local storage) backend over a
    representative window. Plots cumulative tokens × bytes/tok / 1e9 = GB.
    """
    if not sglang or kv_bytes_per_token <= 0:
        return None
    ts = _col(sglang, "time_sec")
    if not ts:
        return None
    lb_col = _find_col(sglang, "load_back_tokens_total")
    bk_col = _find_col(sglang, "backuped_tokens_total")
    pf_col = _find_col(sglang, "prefetched_tokens_total")
    _bk_v = _col(sglang, bk_col) if bk_col else []
    _pf_v = _col(sglang, pf_col) if pf_col else []
    _bk_delta = max((_bk_v[-1] - _bk_v[0]) if len(_bk_v) > 1 else 0.0, 0.0)
    _pf_delta = max((_pf_v[-1] - _pf_v[0]) if len(_pf_v) > 1 else 0.0, 0.0)
    if _bk_delta <= 0 and _pf_delta <= 0:
        return None

    def _cum_gb(col: str) -> list[float]:
        if not col: return [0.0] * len(ts)
        v = _col(sglang, col)
        v0 = v[0] if v else 0
        return [max((vi - v0), 0) * kv_bytes_per_token / 1e9 for vi in v]

    xs = _ts_to_dates(ts)
    traces = []
    if pf_col:
        traces.append({
            "x": xs, "y": _cum_gb(pf_col),
            "type": "scatter", "mode": "lines", "name": "L3 prefetch/read (cumulative GB)",
            "line": {"color": "#22c55e", "width": 2.2},
            "hovertemplate": "<b>%{y:.2f}</b> GB cumulative prefetch/read<extra></extra>",
        })
    if lb_col:
        traces.append({
            "x": xs, "y": _cum_gb(lb_col),
            "type": "scatter", "mode": "lines", "name": "L2→L1 load-back restore (diagnostic, not SSD/L3 bytes)",
            "line": {"color": "#94a3b8", "width": 1.5, "dash": "dot"},
            "hovertemplate": "<b>%{y:.2f}</b> GB L2→L1 restore; not SSD/L3 bytes<extra></extra>",
        })
    if bk_col:
        traces.append({
            "x": xs, "y": _cum_gb(bk_col),
            "type": "scatter", "mode": "lines", "name": "L3 (local storage) writes (cumulative GB)",
            "line": {"color": "#3b82f6", "width": 2.2},
            "hovertemplate": "<b>%{y:.2f}</b> GB cumulative write<extra></extra>",
        })
    return {
        "data": traces,
        "layout": _layout(
            "L3 (local storage) cache Traffic — Cumulative Bytes (Prometheus-derived)",
            "Cumulative GB (estimated)", ts=ts),
    }


def _chart_request_size_dist(reqsize: list[dict]) -> dict | None:
    """Histogram: read/write/trim sizes by bucket."""
    if not reqsize:
        return None
    buckets = ["≤4KB", "4-16KB", "16-64KB", "64-256KB", "256KB-1MB", ">1MB"]
    counts: dict[str, dict[str, int]] = {b: {"read": 0, "write": 0, "trim": 0}
                                          for b in buckets}
    def _bucket_from_size(sz):
        try: sz = float(sz)
        except Exception: return ""
        if sz <= 4*1024: return "≤4KB"
        if sz <= 16*1024: return "4-16KB"
        if sz <= 64*1024: return "16-64KB"
        if sz <= 256*1024: return "64-256KB"
        if sz <= 1024*1024: return "256KB-1MB"
        return ">1MB"
    def _op_norm(op):
        o = str(op).lower()
        if o.startswith('r'): return 'read'
        if o.startswith('w'): return 'write'
        if o.startswith(('t','d')): return 'trim'
        return o
    for r in reqsize:
        bucket = r.get("size_bucket", "") or _bucket_from_size(r.get("size_bytes", r.get("bytes", r.get("size", 0))))
        op = _op_norm(r.get("op", r.get("type", r.get("operation", ""))))
        if bucket in counts and op in counts[bucket]:
            counts[bucket][op] += 1
    if not any(sum(counts[b].values()) for b in buckets):
        return None
    return {
        "data": [
            {"x": buckets, "y": [counts[b]["read"]  for b in buckets],
             "type": "bar", "name": "Read",
             "marker": {"color": "#22c55e", "line": {"color": "#0f172a", "width": 1}},
             "hovertemplate": "<b>Read</b> %{x}: %{y:,} requests<extra></extra>"},
            {"x": buckets, "y": [counts[b]["write"] for b in buckets],
             "type": "bar", "name": "Write",
             "marker": {"color": "#f43f5e", "line": {"color": "#0f172a", "width": 1}},
             "hovertemplate": "<b>Write</b> %{x}: %{y:,} requests<extra></extra>"},
            {"x": buckets, "y": [counts[b]["trim"]  for b in buckets],
             "type": "bar", "name": "Trim",
             "marker": {"color": "#eab308", "line": {"color": "#0f172a", "width": 1}},
             "hovertemplate": "<b>Trim</b> %{x}: %{y:,} requests<extra></extra>"},
        ],
        "layout": {**_layout("Request Size Distribution by Operation",
                              "Number of requests"),
                   "barmode": "group",
                   "xaxis": {"title": {"text": "Request size bucket"},
                             "tickfont": {"size": 12, "color": "#0f172a"}}}
    }


def _chart_iat_dist(iat: list[dict]) -> dict | None:
    if not iat:
        return None
    by_op: dict[str, dict[str, int]] = {}
    bins_order: list[str] = []
    for r in iat:
        op = r.get("op", "")
        label = r.get("bin_label", "")
        cnt = int(_to_float(r.get("count", 0)))
        by_op.setdefault(op, {})[label] = cnt
        if label not in bins_order:
            bins_order.append(label)
    if not by_op:
        return None
    palette = {"read": "#22c55e", "write": "#f43f5e", "trim": "#eab308"}
    traces = []
    for op, bins in by_op.items():
        traces.append({
            "x": bins_order,
            "y": [bins.get(b, 0) for b in bins_order],
            "type": "bar", "name": op,
            "marker": {"color": palette.get(op, "#94a3b8"),
                        "line": {"color": "#0f172a", "width": 1}},
            "hovertemplate": f"<b>{op}</b> %{{x}}: %{{y:,}} reqs<extra></extra>"
        })
    return {"data": traces,
            "layout": {**_layout("Inter-Arrival Time Distribution",
                                 "Number of requests"),
                       "barmode": "group",
                       "xaxis": {"title": {"text": "IAT bucket"},
                                 "tickfont": {"size": 12, "color": "#0f172a"}}}}


def _chart_temporal_rwt(tp: list[dict]) -> dict | None:
    if not tp:
        return None
    ts = _col(tp, "window_start_sec") or _col(tp, "time_sec") or _col(tp, "start")
    def _first(cols):
        for c in cols:
            vals = _col(tp, c)
            if vals and any(vals):
                return vals
        return [0.0] * len(tp)
    rd_raw = _first(["read", "read_bytes", "rd_bytes"]); wr_raw = _first(["write", "write_bytes", "wr_bytes"]); tr_raw = _first(["trim", "trim_bytes", "discard", "discard_bytes"])
    scale = 1e6 if max(rd_raw + wr_raw + tr_raw + [0]) > 1e4 else 1.0
    rd = [v / scale for v in rd_raw]; wr = [v / scale for v in wr_raw]; tr = [v / scale for v in tr_raw]
    if not any(rd) and not any(wr) and not any(tr):
        return None
    return {
        "data": [
            {"x": ts, "y": rd, "type": "scatter", "mode": "lines",
             "name": "Read MB/window", "stackgroup": "rwt",
             "fillcolor": "rgba(34,197,94,0.6)", "line": {"color": "#22c55e"},
             "hovertemplate": "<b>%{y:.1f}</b> MB read<extra></extra>"},
            {"x": ts, "y": wr, "type": "scatter", "mode": "lines",
             "name": "Write MB/window", "stackgroup": "rwt",
             "fillcolor": "rgba(244,63,94,0.55)", "line": {"color": "#f43f5e"},
             "hovertemplate": "<b>%{y:.1f}</b> MB write<extra></extra>"},
            {"x": ts, "y": tr, "type": "scatter", "mode": "lines",
             "name": "Trim MB/window", "stackgroup": "rwt",
             "fillcolor": "rgba(234,179,8,0.5)", "line": {"color": "#eab308"},
             "hovertemplate": "<b>%{y:.1f}</b> MB trim<extra></extra>"},
        ],
        "layout": _layout("R/W/T Bytes per 10-sec Window (stacked)",
                          "MB per window", height=460)
    }


def _chart_bw_per_stream(bps: list[dict]) -> dict | None:
    if not bps:
        return None
    palette = {"read": "#22c55e", "write": "#f43f5e", "trim": "#eab308"}
    traces = []
    for op in ("read", "write", "trim"):
        rows = [r for r in bps if r.get("op") == op]
        if not rows: continue
        rows.sort(key=lambda r: _to_float(r.get("bandwidth_mib_s")), reverse=True)
        rows = rows[:20]
        labels = [r.get("stream", "?") for r in rows]
        bws = [_to_float(r.get("bandwidth_mib_s")) for r in rows]
        bytes_ = [_to_float(r.get("bytes")) / 1e6 for r in rows]
        traces.append({
            "y": labels, "x": bws,
            "type": "bar", "orientation": "h", "name": op.title(),
            "marker": {"color": palette.get(op, "#94a3b8"),
                        "line": {"color": "#0f172a", "width": 1}},
            "customdata": bytes_,
            "hovertemplate": (f"<b>{op}</b><br>"
                              "Stream: %{y}<br>"
                              "Bandwidth: <b>%{x:.2f} MiB/s</b><br>"
                              "Total: %{customdata:.1f} MB<extra></extra>"),
        })
    if not traces:
        return None
    return {"data": traces,
            "layout": {**_layout("Bandwidth per Stream (top 20 per op)",
                                 "Stream (pid:comm)", height=580),
                       "xaxis": {"title": {"text": "MiB / sec"},
                                 "tickfont": {"size": 12, "color": "#0f172a"}},
                       "barmode": "group"}}


def _chart_dram_bw_from_timeseries(rows: list[dict]) -> dict | None:
    """Build DRAM BW chart from collector-normalized PCM timeseries CSV.

    Accepts both modern column names:
      time_sec, dram_total_gb_s, dram_read_gb_s, dram_write_gb_s
    and older aliases:
      total_bw_gbs/total_gb_s, read_bw_gbs/read_gb_s, write_bw_gbs/write_gb_s.
    """
    if not rows:
        return None

    def _pick_col(candidates):
        keys = set()
        for r in rows[:5]:
            keys.update(r.keys())
        for c in candidates:
            if c in keys:
                return c
        # fallback: substring match
        lower = {str(k).lower(): k for k in keys}
        for c in candidates:
            cl = c.lower()
            for lk, orig in lower.items():
                if cl in lk:
                    return orig
        return None

    total_col = _pick_col(["dram_total_gb_s", "total_bw_gbs", "total_gb_s", "pcm_dram_total_gb_s", "Total Mem Bw"])
    read_col  = _pick_col(["dram_read_gb_s", "read_bw_gbs", "read_gb_s", "pcm_dram_read_gb_s", "Total Mem RdBw"])
    write_col = _pick_col(["dram_write_gb_s", "write_bw_gbs", "write_gb_s", "pcm_dram_write_gb_s", "Total Mem WrBw"])
    if not (total_col or read_col or write_col):
        return None

    ts = _col(rows, "time_sec") if "time_sec" in rows[0] else [i * 1.2 for i in range(len(rows))]
    rd_bw = _col(rows, read_col) if read_col else [0.0 for _ in rows]
    wr_bw = _col(rows, write_col) if write_col else [0.0 for _ in rows]
    if total_col:
        tot_bw = _col(rows, total_col)
    else:
        tot_bw = [r + w for r, w in zip(rd_bw, wr_bw)]

    # Drop all-zero parser artifacts.
    if max([abs(v) for v in tot_bw] or [0.0]) <= 0:
        return None

    xs = _ts_to_dates(ts)
    return {
        "data": [
            {"x": xs, "y": tot_bw, "type": "scatter", "mode": "lines",
             "name": "Total (GB/s)",
             "line": {"color": "#a78bfa", "width": 2.2},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s total<extra></extra>"},
            {"x": xs, "y": rd_bw, "type": "scatter", "mode": "lines",
             "name": "Read (GB/s)",
             "line": {"color": "#22c55e", "width": 1.8},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s read<extra></extra>"},
            {"x": xs, "y": wr_bw, "type": "scatter", "mode": "lines",
             "name": "Write (GB/s)",
             "line": {"color": "#f43f5e", "width": 1.8},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s write<extra></extra>"},
        ],
        "layout": _layout("System DRAM Bandwidth (AMDuProf PCM)",
                          "GB / sec", ts=ts)
    }


def _chart_dram_bw_from_summary(summary: dict, duration_s: float | None = None) -> dict | None:
    """Build a DRAM BW chart from collector JSON summaries.

    Intel PCM often writes raw/pcm_summary.json with valid PMU BW even when
    the Executive/Interactive CSV loaders do not see a usable timeseries.  The
    static End Report already uses these summaries; the interactive report must
    do the same so tabs do not disagree.
    """
    if not isinstance(summary, dict) or not summary:
        return None

    def pick(keys):
        for k in keys:
            if k in summary:
                v = _to_float(summary.get(k), 0.0)
                if v > 0:
                    return v
        # case-insensitive fallback
        lower = {str(k).lower(): k for k in summary.keys()}
        for k in keys:
            lk = lower.get(str(k).lower())
            if lk is not None:
                v = _to_float(summary.get(lk), 0.0)
                if v > 0:
                    return v
        return 0.0

    read = pick([
        "dram_read_gb_s_mean", "pcm_dram_read_gb_s", "pcm_dram_read_gb_s_mean",
        "dram_read_bw_gbs", "dram_read_bw_gbps", "dram_read_bw_gbps_mean",
        "read_gb_s_mean", "read_bw_gbps", "amduprof_dram_read_bw_gbps_mean",
    ])
    write = pick([
        "dram_write_gb_s_mean", "pcm_dram_write_gb_s", "pcm_dram_write_gb_s_mean",
        "dram_write_bw_gbs", "dram_write_bw_gbps", "dram_write_bw_gbps_mean",
        "write_gb_s_mean", "write_bw_gbps", "amduprof_dram_write_bw_gbps_mean",
    ])
    total = pick([
        "dram_total_gb_s_mean", "pcm_dram_total_gb_s", "pcm_dram_total_gb_s_mean",
        "dram_total_bw_gbs", "dram_total_bw_gbps", "dram_total_bw_gbps_mean",
        "total_gb_s_mean", "total_bw_gbps", "amduprof_dram_total_bw_gbps_mean",
    ])
    if total <= 0 and (read > 0 or write > 0):
        total = read + write
    if read <= 0 and total > 0 and write > 0:
        read = max(total - write, 0.0)
    if write <= 0 and total > 0 and read > 0:
        write = max(total - read, 0.0)
    if total <= 0 and read <= 0 and write <= 0:
        return None
    dur = float(duration_s or 0.0)
    if dur <= 0:
        dur = pick(["pcm_duration_s", "amduprof_pcm_duration_s", "dram_duration_s", "duration_s"]) or 60.0
    rows = [
        {"time_sec": 0.0, "dram_read_gb_s": read, "dram_write_gb_s": write, "dram_total_gb_s": total},
        {"time_sec": max(dur, 1.0), "dram_read_gb_s": read, "dram_write_gb_s": write, "dram_total_gb_s": total},
    ]
    fig = _chart_dram_bw_from_timeseries(rows)
    if fig:
        fig["layout"]["title"]["text"] = "System DRAM Bandwidth (CPU PMU summary fallback)"
    return fig


def _chart_dram_bw_from_kv_activity(sglang: list[dict], kv_bytes_per_token: float) -> dict | None:
    """Fallback DRAM/KV movement chart from SGLang token counters.

    This is not a CPU PMU bandwidth counter.  It estimates host-memory traffic
    caused by KV movement when AMDuProf/Intel PCM was not collected:
      • write/offload  = backuped_tokens_total × KV bytes/token
      • read/restore   = load_back_tokens_total × KV bytes/token
      • L3 prefetch    = prefetched_tokens_total × KV bytes/token (shown as a
        separate read-side component when present)
    """
    if not sglang or kv_bytes_per_token <= 0:
        return None
    ts = _col(sglang, "time_sec")
    if len(ts) < 2:
        return None
    bk_col = _find_col(sglang, "backuped_tokens_total")
    lb_col = _find_col(sglang, "load_back_tokens_total")
    pf_col = _find_col(sglang, "prefetched_tokens_total")
    if not (bk_col or lb_col or pf_col):
        return None

    def _rate_gbs(col: str | None) -> list[float]:
        if not col:
            return [0.0] * len(ts)
        v = _col(sglang, col)
        out = [0.0]
        for i in range(1, len(v)):
            dt = ts[i] - ts[i-1]
            tok_s = max((v[i] - v[i-1]) / dt, 0.0) if dt > 0 else 0.0
            out.append(tok_s * kv_bytes_per_token / 1e9)
        return out

    rd = _rate_gbs(lb_col)
    wr = _rate_gbs(bk_col)
    pf = _rate_gbs(pf_col)
    total = [r + w + p for r, w, p in zip(rd, wr, pf)]
    if max(total or [0.0]) <= 0:
        return None
    xs = _ts_to_dates(ts)
    return {
        "data": [
            {"x": xs, "y": total, "type": "scatter", "mode": "lines",
             "name": "Estimated total KV movement (GB/s)",
             "line": {"color": "#a78bfa", "width": 2.2},
             "hovertemplate": "<b>%{y:.3f}</b> GB/s estimated KV movement<extra></extra>"},
            {"x": xs, "y": rd, "type": "scatter", "mode": "lines",
             "name": "L2→L1 load_back restore read (GB/s est)",
             "line": {"color": "#22c55e", "width": 1.8},
             "hovertemplate": "<b>%{y:.3f}</b> GB/s estimated restore read<extra></extra>"},
            {"x": xs, "y": wr, "type": "scatter", "mode": "lines",
             "name": "KV backup/offload write (GB/s est)",
             "line": {"color": "#f43f5e", "width": 1.8},
             "hovertemplate": "<b>%{y:.3f}</b> GB/s estimated backup write<extra></extra>"},
            {"x": xs, "y": pf, "type": "scatter", "mode": "lines",
             "name": "L3 prefetch/onboard write-to-L2 component (GB/s est)",
             "line": {"color": "#3b82f6", "width": 1.4, "dash": "dot"},
             "hovertemplate": "<b>%{y:.3f}</b> GB/s estimated prefetch component<extra></extra>"},
        ],
        "layout": _layout("System DRAM Bandwidth — SGLang KV movement fallback",
                          "GB / sec", ts=ts),
    }


def _chart_dram_bw(amduprof_txt: Path) -> dict | None:
    """Parse AMDuProf PCM raw text/CSV and build DRAM BW chart.

    This intentionally mirrors the static report parser instead of using
    pandas.read_csv(), because AMDuProfPcm output is a multi-section CSV-like
    text file and normal CSV tokenization often fails with variable-width
    headers.  The parser searches the DF METRICS header containing Total Mem Bw,
    Total Mem RdBw and Total Mem WrBw, resolves the real column indices, and
    then reads only numeric rows from that section.
    """
    raw_path: Path | None = None
    candidates = [
        amduprof_txt,
        amduprof_txt.with_suffix(".csv"),
        amduprof_txt.with_suffix(".txt"),
        amduprof_txt.parent / "amduprof_pcm_raw.csv",
        amduprof_txt.parent / "amduprof_pcm_raw.txt",
        amduprof_txt.parent / "amduprof_pcm_raw",
    ]
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.stat().st_size > 0:
            raw_path = candidate
            break
    if raw_path is None:
        return None

    def _nums(parts):
        out = []
        for p in parts:
            try:
                out.append(float(str(p).strip()))
            except Exception:
                out.append(0.0)
        return out

    rd_bw, wr_bw, tot_bw = [], [], []
    col_total: int | None = None
    col_rd:    int | None = None
    col_wr:    int | None = None
    in_data = False

    for line in raw_path.read_text(errors="replace").splitlines():
        ll = line.strip().rstrip(",")
        if not ll:
            continue

        # Header detection — accept both AMDuProf long names and short aliases.
        if ("Total Mem Bw" in ll or "Mem Bw" in ll) and ("RdBw" in ll or "WrBw" in ll):
            cols = [c.strip() for c in ll.split(",")]
            col_total = col_rd = col_wr = None
            for i, c in enumerate(cols):
                cl = c.lower().strip()
                if col_total is None and "total mem bw" in cl and "rdbw" not in cl and "wrbw" not in cl:
                    col_total = i
                elif col_rd is None and ("total mem rdbw" in cl or cl == "rdbw" or "mem rdbw" in cl):
                    col_rd = i
                elif col_wr is None and ("total mem wrbw" in cl or cl == "wrbw" or "mem wrbw" in cl):
                    col_wr = i
            if col_total is not None or col_rd is not None or col_wr is not None:
                in_data = True
            continue

        if not in_data:
            continue

        # Data rows in AMDuProf PCM sections normally start with RecordId or timestamp.
        # Stop when a new non-numeric section begins.
        first = ll.split(",", 1)[0].strip()
        try:
            float(first)
        except Exception:
            in_data = False
            continue

        parts = _nums(ll.split(","))
        n = len(parts)
        i_t = col_total if col_total is not None and col_total < n else None
        i_r = col_rd    if col_rd    is not None and col_rd    < n else None
        i_w = col_wr    if col_wr    is not None and col_wr    < n else None
        if i_t is None and i_r is None and i_w is None:
            continue
        r = parts[i_r] if i_r is not None else 0.0
        w = parts[i_w] if i_w is not None else 0.0
        t = parts[i_t] if i_t is not None else (r + w)
        if t != 0 or r != 0 or w != 0:
            tot_bw.append(t); rd_bw.append(r); wr_bw.append(w)

    if not tot_bw:
        return None

    interval_s = 1.2
    ts_elapsed = [i * interval_s for i in range(len(tot_bw))]
    xs = _ts_to_dates(ts_elapsed)
    return {
        "data": [
            {"x": xs, "y": tot_bw, "type": "scatter", "mode": "lines",
             "name": "Total (GB/s)",
             "line": {"color": "#a78bfa", "width": 2.2},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s total<extra></extra>"},
            {"x": xs, "y": rd_bw, "type": "scatter", "mode": "lines",
             "name": "Read (GB/s)",
             "line": {"color": "#22c55e", "width": 1.8},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s read<extra></extra>"},
            {"x": xs, "y": wr_bw, "type": "scatter", "mode": "lines",
             "name": "Write (GB/s)",
             "line": {"color": "#f43f5e", "width": 1.8},
             "hovertemplate": "<b>%{y:.2f}</b> GB/s write<extra></extra>"},
        ],
        "layout": _layout("System DRAM Bandwidth (AMDuProf PCM)",
                          "GB / sec", ts=ts_elapsed)
    }


def _chart_hot_regions(hr: list[dict],
                       lba_dist: list[dict] | None = None,
                       ssd_capacity_gb: float = 0.0,
                       ssd_used_gb: float = 0.0,
                       fs_partition_start_gb: float = 0.0) -> dict | None:
    """SSD I/O distribution across LBA space + hot regions overlay.

    Shows where on the disk this run's I/O landed and how spread out it was.

    Important — TWO different "GB" quantities visible in this section:
      • Bars (and "X GB" in legend) = bytes accessed during the collect window
        only. Limited by `summary.json :: duration_sec` (your run = 1096 s,
        148 GB writes). Does NOT include I/O before tracing started.
      • Dashed `df filesystem-used` line = files on the mount AT ANALYZE TIME,
        cumulative across all previous runs. The bars will typically be far
        below this line because pre-existing KV blocks aren't re-written.

    The hot-regions CSV only keeps the top 50 buckets per op (~15% of bytes
    on a write-heavy KV-offload workload). This chart shows two layers:

      1. **I/O distribution this window** (background bars) — every LBA
         bucket that saw traffic, binned into ~120 equal-width slots across
         the device's full capacity. Y axis = bytes (log) accessed per slot.
         Answers "how spread out was THIS run's I/O".
      2. **Hot regions** (foreground diamonds) — top 30 hottest buckets per
         op from hot_regions_overall.csv.

    X axis spans 0 → `ssd_capacity_gb` so position matches the SSD's physical
    layout. Dotted vertical line at capacity; dashed line at df-used as
    cumulative-state context.
    """
    if not hr and not lba_dist:
        return None

    palette = {"read": "#22c55e", "write": "#f43f5e", "trim": "#eab308"}
    traces: list[dict] = []

    # ── Layer 1: full distribution, binned ──────────────────────────────────
    if lba_dist:
        # Determine X-axis range: prefer the device capacity (so the chart
        # spans the physical SSD), else the max observed LBA.
        max_lba_observed_b = max(
            (_to_float(r.get("lba_bucket_end")) or _to_float(r.get("lba_bucket_start")) or 0)
            for r in lba_dist
        )
        max_x_gb = max(ssd_capacity_gb, max_lba_observed_b / 1e9, 1.0)

        # Bin into ~120 equal-width slots
        N_BINS = 120
        bin_size_gb = max_x_gb / N_BINS

        from collections import defaultdict
        per_op_bins: dict[str, list[float]] = defaultdict(lambda: [0.0] * N_BINS)
        per_op_total: dict[str, float] = defaultdict(float)
        for r in lba_dist:
            op = (r.get("op") or "").lower()
            if op not in palette:
                continue
            lba_start_gb = _to_float(r.get("lba_bucket_start")) / 1e9
            bytes_ = _to_float(r.get("bytes")) or 0
            if bytes_ <= 0:
                continue
            bin_idx = min(int(lba_start_gb / bin_size_gb), N_BINS - 1)
            per_op_bins[op][bin_idx] += bytes_
            per_op_total[op] += bytes_

        bin_centers_gb = [(i + 0.5) * bin_size_gb for i in range(N_BINS)]
        for op in ("read", "write", "trim"):
            bins = per_op_bins.get(op)
            if not bins or per_op_total[op] <= 0:
                continue
            total_gb = per_op_total[op] / 1e9
            buckets_used = sum(1 for v in bins if v > 0)
            coverage_pct = buckets_used / N_BINS * 100.0
            traces.append({
                "x": bin_centers_gb,
                # Replace empty bins with None so the log y-axis doesn't render
                # them as -infinity bars (those produce "NaN GB" hover tooltips
                # on some browsers when the user hovers over the bin center).
                # None makes Plotly skip the bar entirely — no visual artifact,
                # no spurious tooltip.
                "y": [(v / 1e6) if v > 0 else None for v in bins],  # MB
                "type": "bar",
                "name": f"{op.title()} traffic ({total_gb:.1f} GB written in window, touches {coverage_pct:.0f}% of device)",
                "marker": {"color": palette[op], "opacity": 0.45,
                            "line": {"color": palette[op], "width": 0}},
                "width": bin_size_gb * 0.95,
                "offsetgroup": op,
                "hovertemplate": (
                    f"<b>{op.title()} distribution</b><br>"
                    "LBA range: %{x:.1f} GB ± "
                    f"{bin_size_gb/2:.1f} GB<br>"
                    "Bytes in bin: <b>%{y:.2f} MB</b><extra></extra>"),
                "legendgroup": "dist",
            })

    # ── Layer 2: hot regions on top (top 30 per op) ─────────────────────────
    if hr:
        for op in ("read", "write", "trim"):
            rows = [r for r in hr if (r.get("op") or "").lower() == op][:30]
            if not rows:
                continue
            traces.append({
                "x": [_to_float(r.get("lba_bucket_start")) / 1e9 for r in rows],
                "y": [(_to_float(r.get("bytes")) or 0) / 1e6 for r in rows],
                "type": "scatter", "mode": "markers",
                "name": f"{op.title()} hot regions (top 30)",
                "marker": {"color": palette.get(op), "size": 11, "opacity": 0.9,
                            "symbol": "diamond",
                            "line": {"color": "#0f172a", "width": 1.2}},
                "hovertemplate": (
                    f"<b>{op.title()} hot region</b><br>"
                    "LBA: %{x:.3f} GB<br>"
                    "Bytes: <b>%{y:.1f} MB</b><extra></extra>"),
                "legendgroup": "hot",
            })

    if not traces:
        return None

    # ── Layout with optional vlines for capacity and used-space ─────────────
    layout = dict(_layout("SSD I/O Distribution Across LBA Space",
                          "Bytes accessed in bin (MB, log)", height=520))
    # X-axis: span the SSD capacity, label in GB
    max_x_gb = max(ssd_capacity_gb,
                    (max((_to_float(r.get("lba_bucket_start")) for r in (lba_dist or [])), default=0)
                     / 1e9), 1.0)
    # CRITICAL: _layout() defaults xaxis type to "date" because most amoprof
    # charts are time-series. This chart's X axis is GB-of-LBA-offset, so we
    # MUST force the type back to "linear". Without this, Plotly treats x
    # values as Unix-epoch milliseconds — 0 becomes Dec 31 1969 and the
    # hovertemplate's "%{x:.1f} GB" formats a date as a number giving NaN GB.
    layout["xaxis"] = {
        "type": "linear",
        "title": {"text": (f"Disk LBA offset (GB) — device capacity "
                            f"{ssd_capacity_gb:.0f} GB" if ssd_capacity_gb > 0
                            else "Disk LBA offset (GB)"),
                   "standoff": 15},
        "range": [0, max_x_gb * 1.02],
        "gridcolor": "#cbd5e1", "zerolinecolor": "#94a3b8",
        "linecolor": "#475569", "tickcolor": "#475569",
        "tickfont": {"size": 12, "color": "#0f172a"},
        "automargin": True,
    }
    # Y-axis: log scale to make sparse activity visible alongside hot peaks.
    # Replace the default linear y-axis from _layout(); explicit type avoids
    # any chance of inheriting an unexpected type.
    layout["yaxis"] = {
        "type": "log",
        "title": {"text": "Bytes per bin (MB, log)", "standoff": 10},
        "gridcolor": "#cbd5e1", "zerolinecolor": "#94a3b8",
        "linecolor": "#475569", "tickcolor": "#475569",
        "tickfont": {"size": 12, "color": "#0f172a"},
        "automargin": True,
    }
    # hovermode default from _layout is "x unified" which is wrong for a
    # non-time axis (it lines all traces up on x; here we want per-trace).
    layout["hovermode"] = "closest"
    layout["barmode"] = "overlay"
    shapes = []
    annotations = []
    # The df-used number is FS-relative (GB used inside the partition).
    # Plot it in device-absolute LBA = partition_start + fs_used so the
    # dashed line lines up correctly with where the FS data actually lives.
    df_used_lba_gb = fs_partition_start_gb + ssd_used_gb
    if ssd_used_gb > 0 and df_used_lba_gb < max_x_gb:
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": df_used_lba_gb, "x1": df_used_lba_gb, "y0": 0, "y1": 1,
            "line": {"color": "#0ea5e9", "width": 2, "dash": "dash"},
        })
        _df_label = (f"df at analyze time: {ssd_used_gb:.0f} GB FS-used "
                      f"(at device LBA {df_used_lba_gb:.0f} GB)"
                      if fs_partition_start_gb > 0
                      else f"df at analyze time: {ssd_used_gb:.0f} GB filesystem-used "
                            "(cumulative across runs, not just this window)")
        annotations.append({
            "x": df_used_lba_gb, "y": 1.02, "xref": "x", "yref": "paper",
            "text": _df_label,
            "showarrow": False, "yanchor": "bottom",
            "font": {"size": 10, "color": "#0ea5e9"},
            "bgcolor": "rgba(14,165,233,0.08)",
            "bordercolor": "#0ea5e9", "borderwidth": 1, "borderpad": 3,
        })
    # Partition-start marker (only when traced device is whole-disk and FS
    # lives on a partition — i.e. fs_partition_start_gb > 0). Helps explain
    # why hot-LBA cluster is centered around the partition's logical zero
    # rather than the device's physical zero.
    if fs_partition_start_gb > 0 and fs_partition_start_gb < max_x_gb:
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": fs_partition_start_gb, "x1": fs_partition_start_gb,
            "y0": 0, "y1": 1,
            "line": {"color": "#16a34a", "width": 2, "dash": "dashdot"},
        })
        annotations.append({
            "x": fs_partition_start_gb, "y": 0.88, "xref": "x", "yref": "paper",
            "text": f"FS partition start ({fs_partition_start_gb:.0f} GB)",
            "showarrow": False, "yanchor": "top",
            "font": {"size": 10, "color": "#16a34a"},
            "bgcolor": "rgba(22,163,74,0.08)",
            "bordercolor": "#16a34a", "borderwidth": 1, "borderpad": 3,
        })
    if ssd_capacity_gb > 0:
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": ssd_capacity_gb, "x1": ssd_capacity_gb, "y0": 0, "y1": 1,
            "line": {"color": "#475569", "width": 2, "dash": "dot"},
        })
        annotations.append({
            "x": ssd_capacity_gb, "y": 0.95, "xref": "x", "yref": "paper",
            "text": f"capacity {ssd_capacity_gb:.0f} GB",
            "showarrow": False, "yanchor": "top", "xanchor": "right",
            "font": {"size": 11, "color": "#475569"},
        })
    if shapes:
        layout["shapes"] = shapes
    if annotations:
        layout["annotations"] = annotations
    return {"data": traces, "layout": layout}


# ─── KPI tile builder ────────────────────────────────────────────────────────
def _chart_percentile_timeseries(pct_ts: dict, metric_key: str) -> dict | None:
    """Time-series chart: how P50/P90/P99 evolve across the Prometheus window.

    Consumes the `sglang_percentiles_timeseries.json` file written by
    amoprof.percentiles.fetch_percentile_timeseries.

    metric_key is one of: "ttft", "itl", "e2e", "prompt_tokens", "output_tokens".

    Returns None if the metric isn't present (e.g. histogram buckets weren't
    scraped, or analyze ran without --prometheus).
    """
    if not pct_ts:
        return None
    block = pct_ts.get(metric_key)
    if not block or not block.get("time_sec"):
        return None
    ts = block["time_sec"]
    xs = _ts_to_dates(ts)
    traces: list[dict] = []
    # Filter out None values per-series so Plotly doesn't draw gaps incorrectly
    def _clean_series(ys: list) -> list:
        return [None if (y is None or (isinstance(y, float) and y != y)) else float(y)
                for y in ys]
    series_specs = [
        ("p99", "P99", "#ef4444", 2.0, "solid"),
        ("p90", "P90", "#f59e0b", 1.8, "solid"),
        ("p50", "P50 (median)", "#22d3ee", 1.6, "dot"),
    ]
    for key, label, color, width, dash in series_specs:
        ys = block.get(key)
        if not ys:
            continue
        traces.append({
            "x": xs, "y": _clean_series(ys),
            "type": "scatter", "mode": "lines",
            "name": label,
            "line": {"color": color, "width": width, "dash": dash},
            "connectgaps": False,
            "hovertemplate": (f"<b>{label}</b>: "
                              f"%{{y:,.2f}} {block.get('unit', '')}"
                              f"<extra></extra>"),
        })
    if not traces:
        return None
    unit_label = block.get("unit", "")
    metric_label = block.get("label", metric_key.upper())
    rate_window_s = block.get("rate_window_s")
    rate_window_txt = f", rate window={int(rate_window_s)}s" if rate_window_s else ""
    return {
        "data": traces,
        "layout": _layout(
            f"{metric_label} — P50 / P90 / P99 over time (Prometheus histogram_quantile{rate_window_txt})",
            f"{unit_label} (log scale)",
            height=480, ts=ts),
        # Override Y to log; tail latencies span 1-2 decades
        "_yaxis_override": {"type": "log", "tickformat": ",.0f"},
    }



def _fig_has_numeric_y(fig: dict | None) -> bool:
    """True when at least one trace has a finite positive y value."""
    if not fig:
        return False
    for tr in (fig.get("data") or []):
        for y in (tr.get("y") or []):
            try:
                v = float(y)
                if math.isfinite(v) and v > 0:
                    return True
            except Exception:
                continue
    return False


def _first_positive_from_summary(summary: dict | None, keys: list[str]) -> float:
    summary = summary or {}
    for k in keys:
        try:
            v = float(summary.get(k, 0) or 0)
            if math.isfinite(v) and v > 0:
                return v
        except Exception:
            pass
    return 0.0


def _apply_yaxis_override(spec: dict | None) -> dict | None:
    """Apply ._yaxis_override into the layout's yaxis if present."""
    if not spec:
        return spec
    override = spec.pop("_yaxis_override", None)
    if override and "layout" in spec and "yaxis" in spec["layout"]:
        spec["layout"]["yaxis"].update(override)
    elif override and "layout" in spec:
        spec["layout"]["yaxis"] = override
    return spec


def _chart_bench_latency_percentiles(bench: dict) -> dict | None:
    """Grouped bar chart: TTFT / ITL / E2E latency at Mean, P50, P90, P99, Max.

    This is the headline view of per-request latency from a benchmark
    summary. Y-axis uses log scale because P99/Max often dwarfs Mean
    (e.g. user's data: avg TTFT 26s, max TTFT 50s — only ~2× spread —
    but ITL avg 1.9s vs max 38s is 20× spread).
    """
    if not bench:
        return None
    # (metric, [Mean, P50, P90, P99, Max] keys)
    series = [
        ("TTFT (s)",  ["avg_ttft_s", "median_ttft_s", "p90_ttft_s", "p99_ttft_s", "max_ttft_s"]),
        ("ITL (s)",   ["avg_itl_s",  "median_itl_s",  "p90_itl_s",  "p99_itl_s",  "max_itl_s"]),
        ("E2E (s)",   ["avg_latency_s", "median_latency_s", "p90_latency_s",
                       "p99_latency_s", "max_latency_s"]),
    ]
    stat_labels = ["Mean", "P50", "P90", "P99", "Max"]
    stat_colors = ["#22d3ee", "#a78bfa", "#f59e0b", "#fb923c", "#ef4444"]
    have_any = False
    traces = []
    metric_names = [s[0] for s in series]
    for i, stat_label in enumerate(stat_labels):
        ys = []
        for _, keys in series:
            v = bench.get(keys[i])
            ys.append(float(v) if v is not None else 0.0)
            if v is not None:
                have_any = True
        traces.append({
            "x": metric_names, "y": ys,
            "type": "bar", "name": stat_label,
            "marker": {"color": stat_colors[i],
                        "line": {"color": "#0f172a", "width": 1}},
            "hovertemplate": (f"<b>{stat_label}</b><br>"
                              "%{x}: <b>%{y:.3f} sec</b><extra></extra>"),
        })
    if not have_any:
        return None
    return {
        "data": traces,
        "layout": {**_layout("Per-Request Latency — Distribution Across Percentiles",
                              "seconds (log scale)", height=480),
                   "barmode": "group",
                   "yaxis": {"type": "log",
                             "title": {"text": "seconds (log scale)",
                                        "font": {"size": 13}},
                             "gridcolor": "#cbd5e1",
                             "tickfont": {"size": 12, "color": "#0f172a"}},
                   "xaxis": {"title": {"text": "Latency metric",
                                       "font": {"size": 13}},
                             "tickfont": {"size": 12, "color": "#0f172a"}}}
    }



def _chart_token_length_fallback(sglang: list[dict], metric_key: str,
                                 sglang_summary: dict | None = None,
                                 pct_ts: dict | None = None) -> dict | None:
    """Fallback token-length-over-time chart.

    Used when token histogram buckets are missing or exported as all-null
    traces. It prefers per-interval counter deltas and falls back to a flat
    session/summary percentile line so the chart is visibly populated.
    """
    sglang_summary = sglang_summary or {}
    pct_ts = pct_ts or {}
    ts = _col(sglang, "time_sec") if sglang else []
    if not ts:
        block = pct_ts.get(metric_key) or pct_ts.get("input_tokens") or pct_ts.get("generation_tokens") or {}
        ts = block.get("time_sec") or [0.0, 60.0]
    if not ts:
        return None

    if metric_key in ("prompt_tokens", "input_tokens"):
        tok_col = _find_col(sglang, "prompt_tokens_total")
        title = "Prompt/Input token length distribution over time"
        color = "#3b82f6"
        mean_keys = ["sess_input_tok_mean","input_tok_mean","input_tokens_mean","prompt_tokens_mean","avg_prompt_tokens","avg_input_tokens","mean_prompt_tokens"]
        p50_keys = ["sess_input_tok_p50","input_tok_p50","prompt_tokens_p50","p50_prompt_tokens","median_prompt_tokens"]
        p90_keys = ["sess_input_tok_p90","input_tok_p90","prompt_tokens_p90","p90_prompt_tokens"]
        p99_keys = ["sess_input_tok_p99","input_tok_p99","prompt_tokens_p99","p99_prompt_tokens"]
    else:
        tok_col = _find_col(sglang, "generation_tokens_total") or _find_col(sglang, "output_tokens_total")
        title = "Output token length distribution over time"
        color = "#22c55e"
        mean_keys = ["sess_output_tok_mean","output_tok_mean","output_tokens_mean","generation_tokens_mean","avg_output_tokens","mean_output_tokens"]
        p50_keys = ["sess_output_tok_p50","output_tok_p50","output_tokens_p50","p50_output_tokens","median_output_tokens"]
        p90_keys = ["sess_output_tok_p90","output_tok_p90","output_tokens_p90","p90_output_tokens"]
        p99_keys = ["sess_output_tok_p99","output_tok_p99","output_tokens_p99","p99_output_tokens"]

    req_col = (_find_col(sglang, "num_requests_total") or _find_col(sglang, "request_total") or
               _find_col(sglang, "requests_total") or _find_col(sglang, "e2e_request_latency_seconds_count") or
               _find_col(sglang, "time_to_first_token_seconds_count"))

    xs = _ts_to_dates(ts)
    traces = []
    if tok_col and req_col and sglang:
        toks = _col(sglang, tok_col)
        reqs = _col(sglang, req_col)
        ys = []
        last_t = last_r = None
        for t, r in zip(toks, reqs):
            if last_t is None:
                ys.append(None)
            else:
                dtok = max(float(t) - float(last_t), 0.0)
                dreq = max(float(r) - float(last_r), 0.0)
                ys.append(dtok / dreq if dreq > 0 else None)
            last_t, last_r = t, r
        if any((v or 0) > 0 for v in ys):
            traces.append({"x": xs, "y": ys, "type": "scatter", "mode": "lines+markers",
                           "name": "Mean tokens/request (counter delta)",
                           "line": {"color": color, "width": 2.2},
                           "hovertemplate": "<b>%{y:,.0f}</b> tokens/request<extra></extra>"})

    if not traces:
        p50 = _first_positive_from_summary(sglang_summary, p50_keys)
        p90 = _first_positive_from_summary(sglang_summary, p90_keys)
        p99 = _first_positive_from_summary(sglang_summary, p99_keys)
        mean = _first_positive_from_summary(sglang_summary, mean_keys)
        if p50 <= 0 and mean > 0: p50 = mean
        if p90 <= 0 and p50 > 0: p90 = max(p50, mean)
        if p99 <= 0 and p90 > 0: p99 = p90
        for name, val, col, dash in [
            ("P99 summary fallback", p99, "#ef4444", "solid"),
            ("P90 summary fallback", p90, "#f59e0b", "solid"),
            ("P50 summary fallback", p50, "#22d3ee", "dot"),
        ]:
            if val > 0:
                traces.append({"x": xs, "y": [val for _ in xs], "type": "scatter", "mode": "lines",
                               "name": name, "line": {"color": col, "width": 1.8, "dash": dash},
                               "hovertemplate": f"<b>{name}</b>: %{{y:,.0f}} tokens<extra></extra>"})

    if not traces:
        return None
    fig = {"data": traces,
           "layout": _layout(title + " (fallback when histogram buckets are empty)",
                             "tokens/request", height=480, ts=ts)}
    fig["layout"]["annotations"] = [{
        "text": "Fallback view: token histogram buckets were missing/all-null; using counter deltas or session summary values.",
        "xref": "paper", "yref": "paper", "x": 0, "y": 1.08, "showarrow": False,
        "font": {"size": 11, "color": "#64748b"}, "align": "left"}]
    return _fit_plotly_legend(fig)



def _chart_bench_token_lengths(bench: dict) -> dict | None:
    """Grouped bar chart: prompt vs output token lengths at Mean, P90, P99, Max."""
    if not bench:
        return None
    series = [
        ("Prompt tokens", ["avg_prompt_tokens", "p90_prompt_tokens",
                            "p99_prompt_tokens", "max_prompt_tokens"]),
        ("Output tokens", ["avg_output_tokens", "p90_output_tokens",
                            "p99_output_tokens", "max_output_tokens"]),
    ]
    stat_labels = ["Mean", "P90", "P99", "Max"]
    stat_colors = ["#22c55e", "#f59e0b", "#fb923c", "#ef4444"]
    have_any = False
    traces = []
    metric_names = [s[0] for s in series]
    for i, stat_label in enumerate(stat_labels):
        ys = []
        for _, keys in series:
            v = bench.get(keys[i])
            ys.append(float(v) if v is not None else 0.0)
            if v is not None:
                have_any = True
        traces.append({
            "x": metric_names, "y": ys,
            "type": "bar", "name": stat_label,
            "marker": {"color": stat_colors[i],
                        "line": {"color": "#0f172a", "width": 1}},
            "hovertemplate": (f"<b>{stat_label}</b><br>"
                              "%{x}: <b>%{y:,.0f} tokens</b><extra></extra>"),
        })
    if not have_any:
        return None
    return {
        "data": traces,
        "layout": {**_layout("Prompt & Output Token Length Distribution",
                              "tokens", height=460),
                   "barmode": "group",
                   "yaxis": {"type": "log",
                             "title": {"text": "tokens (log scale)",
                                        "font": {"size": 13}},
                             "gridcolor": "#cbd5e1",
                             "tickfont": {"size": 12, "color": "#0f172a"}},
                   "xaxis": {"title": {"text": "Token sequence",
                                       "font": {"size": 13}},
                             "tickfont": {"size": 12, "color": "#0f172a"}}}
    }


def _chart_bench_throughput_breakdown(bench: dict) -> dict | None:
    """Side-by-side bars for input vs output tok/s, plus request rate.

    Shows the asymmetry typical of long-prompt/short-output workloads
    (user's data: 9748 in tok/s vs 6.82 out tok/s — 1430× ratio).
    """
    if not bench:
        return None
    in_tps  = bench.get("input_tok_per_s")
    out_tps = bench.get("output_tok_per_s")
    if in_tps is None and out_tps is None:
        return None
    traces = [{
        "x": ["Input tok/s", "Output tok/s"],
        "y": [float(in_tps or 0), float(out_tps or 0)],
        "type": "bar",
        "marker": {
            "color": ["#3b82f6", "#22c55e"],
            "line": {"color": "#0f172a", "width": 1.5},
        },
        "text": [f"{float(in_tps or 0):,.1f}",
                  f"{float(out_tps or 0):,.2f}"],
        "textposition": "outside",
        "textfont": {"size": 14, "color": "#0f172a"},
        "hovertemplate": ("<b>%{x}</b>: <b>%{y:,.2f}</b> tokens/sec<extra></extra>"),
        "showlegend": False,
    }]
    ratio = ""
    if in_tps and out_tps and float(out_tps) > 0:
        ratio = f" — input is {float(in_tps)/float(out_tps):.0f}× output"
    return {
        "data": traces,
        "layout": {**_layout("Token Throughput — Input vs Output" + ratio,
                              "tokens/sec (log scale)", height=320),
                   "yaxis": {"type": "log",
                             "title": {"text": "tokens / sec (log scale)",
                                        "font": {"size": 13}},
                             "gridcolor": "#cbd5e1",
                             "tickfont": {"size": 12, "color": "#0f172a"}},
                   "xaxis": {"tickfont": {"size": 13, "color": "#0f172a"}}}
    }


def _chart_bench_latency_ranges(bench: dict) -> dict | None:
    """Box-and-whisker style chart: visualize the spread per metric.

    Plotly's 'box' trace type can be fed pre-computed percentiles via the
    q1/median/q3/lowerfence/upperfence properties — perfect for our case
    where we have summary stats but not the underlying samples.
    """
    if not bench:
        return None
    boxes = []
    palette = {"TTFT": "#22d3ee", "ITL": "#a78bfa", "E2E latency": "#f59e0b"}
    for metric, prefix in [("TTFT", "ttft"), ("ITL", "itl"),
                            ("E2E latency", "latency")]:
        med  = bench.get(f"median_{prefix}_s")
        p90  = bench.get(f"p90_{prefix}_s")
        p99  = bench.get(f"p99_{prefix}_s")
        mx   = bench.get(f"max_{prefix}_s")
        avg  = bench.get(f"avg_{prefix}_s")
        if med is None or p90 is None or p99 is None:
            continue
        # Use median as the "center", P90 as Q3, P99 as upperfence, Max as outlier.
        # Q1 is approximated as max(0, median - (P90 - median)) for visual symmetry
        # — we don't have the actual Q1 but this gives a reasonable lower whisker.
        q1_approx = max(0.0, 2 * float(med) - float(p90))
        boxes.append({
            "name": metric,
            "type": "box",
            "q1": [q1_approx],
            "median": [float(med)],
            "q3": [float(p90)],
            "lowerfence": [q1_approx * 0.5],
            "upperfence": [float(p99)],
            "mean": [float(avg)] if avg is not None else None,
            "sd":   [0],
            "marker": {"color": palette.get(metric, "#94a3b8"),
                        "line": {"color": "#0f172a", "width": 1}},
            "boxpoints": False,
            "hovertemplate": (f"<b>{metric}</b><br>"
                              f"Mean: {avg or 0:.2f}s · P50: {med:.2f}s · "
                              f"P90: {p90:.2f}s · P99: {p99:.2f}s · "
                              f"Max: {mx or 0:.2f}s<extra></extra>"),
        })
    if not boxes:
        return None
    return {
        "data": boxes,
        "layout": {**_layout(
            "Latency Spread (Box = P50→P90, Whiskers = ~Q1→P99)",
            "seconds (log scale)", height=460),
                   "yaxis": {"type": "log",
                             "title": {"text": "seconds (log scale)",
                                        "font": {"size": 13}},
                             "gridcolor": "#cbd5e1",
                             "tickfont": {"size": 12, "color": "#0f172a"}}}
    }



def _kpi_tile(label: str, value: str, unit: str = "",
              note: str = "", color: str = "#0f172a") -> str:
    return f"""
<div class="kpi-tile" title="{note}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color}">{value}<span class="kpi-unit">{unit}</span></div>
  {f'<div class="kpi-note">{note}</div>' if note else ''}
</div>"""


def _section(title: str, chart_id: str, fig: dict | None,
             note: str = "", formula: str = "") -> str:
    """Render an interactive chart section.

    Empty sections (fig is None) are wrapped in a collapsed <details> with
    a single-line muted summary. The user can expand to see the explanation
    of why no data is available, but the section doesn't take up vertical
    real estate by default. This addresses long pages with many "requires
    blktrace / requires AMDuProf / no L3 (local storage) activity" sections that scroll
    interesting content off-screen.

    Sections WITH data render expanded as before — no extra clicks needed.
    """
    if fig is None:
        # Empty section: collapsed by default. The summary row stays visible
        # so the user knows the section exists and can be expanded.
        return f"""
<section class="card empty-card">
  <details class="empty-details">
    <summary class="empty-summary">
      <span class="empty-icon">⊘</span>
      <span class="empty-title">{title}</span>
      <span class="empty-hint">no data — click to expand</span>
    </summary>
    <div class="no-data">No data available for this chart.<br>
      <small>{note or 'Check that the corresponding collector ran during capture.'}</small>
      {f'<details class="fml"><summary>📐 Formula & Metrics Used</summary><pre>{formula}</pre></details>' if formula else ''}
    </div>
  </details>
</section>"""
    fig = _fit_plotly_legend(fig)
    fig_json = json.dumps(fig, default=str)
    return f"""
<section class="card">
  <h2>{title}</h2>
  <div id="{chart_id}" class="plot"></div>
  {f'<div class="caption">{note}</div>' if note else ''}
  {f'<details class="fml" style="margin-top:10px"><summary>📐 Formula & Metrics Used</summary><pre>{formula}</pre></details>' if formula else ''}
  <script>
    (function() {{
      var fig = {fig_json};
      Plotly.newPlot('{chart_id}', fig.data, fig.layout,
        {{responsive:true, displayModeBar:true, displaylogo:false,
          modeBarButtonsToRemove:['select2d','lasso2d']}});
    }})();
  </script>
</section>"""

def _layer_header(layer_id: str, title: str, note: str, color: str) -> str:
    return f"""
<section class="card layer-header" id="{layer_id}" style="background:{color};color:#fff;border-color:{color}">
  <h2 style="color:#fff;border-bottom:1px solid rgba(255,255,255,.35)">{title}</h2>
  <div style="font-size:13px;opacity:.95">{note}</div>
</section>"""



def _amoprof_parse_launch_arg(launch: str, *names: str) -> str:
    if not launch:
        return ""
    for name in names:
        pat = r'(?<!\S)' + re.escape(name) + r'(?:=|\s+)(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))'
        m = re.search(pat, str(launch))
        if m:
            return next((g for g in m.groups() if g), "").strip()
    return ""


def _amoprof_missing(v) -> bool:
    return v is None or str(v).strip() in ("", "?", "unknown", "None", "null", "N/A")


def _amoprof_augment_setup_from_launch(setup: dict) -> dict:
    if not isinstance(setup, dict):
        setup = {}
    out = dict(setup)
    launch = str(out.get("Launch command") or out.get("launch_command") or out.get("Command") or out.get("Reference launch command") or "")
    if launch and _amoprof_missing(out.get("Launch command")):
        out["Launch command"] = launch
    mapping = [
        ("Model", ("--model", "--model-path")),
        ("TP size", ("--tensor-parallel-size", "--tp-size", "--tp")),
        ("DP size", ("--data-parallel-size", "--dp-size", "--dp")),
        ("mem-fraction-static", ("--mem-fraction-static",)),
        ("Page size", ("--page-size",)),
        ("KV cache dtype", ("--kv-cache-dtype",)),
        ("Chunked prefill size", ("--chunked-prefill-size",)),
        ("HiCache size", ("--hicache-size",)),
        ("HiCache IO backend", ("--hicache-io-backend",)),
        ("HiCache mem layout", ("--hicache-mem-layout",)),
        ("HiCache storage backend", ("--hicache-storage-backend",)),
        ("HiCache write policy", ("--hicache-write-policy",)),
        ("HiCache storage prefetch policy", ("--hicache-storage-prefetch-policy",)),
        ("Attention backend", ("--attention-backend",)),
        ("CUDA graph max batch size", ("--cuda-graph-max-bs",)),
        ("Host", ("--host",)),
        ("Port", ("--port",)),
    ]
    lower_keys = {str(k).lower(): k for k in out.keys()}
    for label, flags in mapping:
        val = _amoprof_parse_launch_arg(launch, *flags)
        existing_key = lower_keys.get(label.lower())
        if val and (existing_key is None or _amoprof_missing(out.get(existing_key))):
            out[label] = val
    return out


def _setup_section(setup: dict) -> str:
    import html as _html
    setup = _amoprof_augment_setup_from_launch(setup or {})
    rows = []
    for k, v in setup.items():
        key_l = str(k).lower()
        if key_l == "instance":
            continue
        klass = "setup-row-defaulted" if key_l in ("default profile", "defaulted fields") else ""
        rows.append(
            f'<tr class="{klass}"><td class="setup-key">{_html.escape(str(k))}</td>'
            f'<td class="setup-value">{_html.escape(str(v))}</td></tr>'
        )
    return f"""
<details class="card setup-panel setup-panel-collapsible">
  <summary>⚙️ Setup / Server Configuration <span class="setup-summary-hint">collapsed by default</span></summary>
  <div class="setup-grid">
    <div><table class="setup-table"><tbody>{''.join(rows)}</tbody></table></div>
    <div class="setup-info-box">
      <div class="setup-info-title">How setup fields were populated</div>
      <p><b>Auto-detected:</b> read from <code>sglang_summary.json</code>, Prometheus labels, and cache config labels when present.</p>
      <p><b>Manual override:</b> <code>setup_details.json</code> or <code>setup.json</code> values override auto-detection.</p>
      <p><b>Defaults:</b> remaining missing SGLang launch parameters use the built-in DeepSeek-R1-70B HiCache reference profile and are listed under <code>Defaulted fields</code>.</p>
    </div>
  </div>
</details>"""


def _stack_overview_section() -> str:
    return """
<section class="card" style="border-left:4px solid #0f766e">
  <h2>🍰 AI Workload Stack — Layered Memory-Tier View</h2>
  <div class="stack-cake">
    <div class="stack-row full a5"><b>A5 · Application Layer</b><span>benchmarks, prompt/output mix, TTFT, TPOT, P99 request latency</span></div>
    <div class="stack-row pair"><div class="a4"><b>A4 · Inference Runtime</b><span>SGLang scheduling, batching, KV$ eviction/L2→L1 load-back</span></div><div class="model"><b>AI Models</b><span>model config, dtype, context length, KV$/token</span></div></div>
    <div class="stack-row trio"><div class="l1"><b>A3/A2/A1 · L1 HBM</b><span>CUDA kernels, GPU driver, HBM capacity/BW</span></div><div class="l2"><b>A3/A2/A1 · L2 DRAM</b><span>OS memory manager, page faults, swap, DRAM BW</span></div><div class="l3"><b>A3/A2/A1 · L3 (local storage)</b><span>OS block layer, NVMe driver, NAND, blktrace/biosnoop</span></div></div>
  </div>
  <div class="caption">Report sections below are ordered to match this layered view.</div>
</section>"""


# ─── Main entry ──────────────────────────────────────────────────────────────


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


def build_report(raw_dir: Path, out_html: Path,
                 run_label: str = "") -> Path:
    """Build the interactive HTML report from a raw/ directory.

    Returns the path to the written HTML file.
    """
    global _T0_EPOCH
    raw_dir = _amoprof_resolve_raw_dir(Path(raw_dir))
    out_html = Path(out_html)

    # Load all data
    sglang_ts   = _read_csv(raw_dir / "sglang_timeseries.csv")
    sglang_sum  = _read_json(raw_dir / "sglang_summary.json")
    gpu_ts      = _read_csv(raw_dir / "gpu_timeseries.csv")
    gpu_sum     = _read_json(raw_dir / "gpu_summary.json")
    power_ts    = _read_csv(raw_dir / "power_timeseries.csv")
    vmstat_ts   = _read_csv(raw_dir / "vmstat_timeseries.csv")
    nvme_ts     = _read_csv(raw_dir / "nvme_driver_timeseries.csv")
    # The blktrace analyzer (v1.38.15+) writes its analysis CSVs to the
    # raw_dir/blktrace_analysis/ subdir so they don't clobber the collect-time
    # summary.json. Prefer that subdir; fall back to raw_dir for older runs.
    _ba_dir = raw_dir / "blktrace_analysis"
    def _ba_csv(name: str) -> list[dict]:
        p_sub = _ba_dir / name
        if p_sub.exists() and p_sub.stat().st_size > 0:
            return _read_csv(p_sub)
        return _read_csv(raw_dir / name)
    # Derive per-sample rate columns (rd_iops, wr_iops, rd_bw_mbs, wr_bw_mbs,
    # rd_lat_ms, wr_lat_ms, io_util_pct) from the raw cumulative counters that
    # the Prometheus loader writes.  If these columns are already present
    # (e.g. from a live iostat collection), _derive_nvme_columns() is a no-op.
    nvme_ts = _derive_nvme_columns(nvme_ts)
    # If the NVMe driver / iostat timeseries is empty or all-zero (a common
    # case when /sys/block stats weren't captured but blktrace was), synthesize
    # a per-window NVMe timeseries from the blktrace temporal pattern so the
    # NVMe BW / IOPS / utilization charts still reflect the real device I/O.
    def _nvme_ts_has_data(rows: list[dict]) -> bool:
        if not rows:
            return False
        for r in rows:
            for k in ("rd_bw_mbs", "wr_bw_mbs", "rd_iops", "wr_iops"):
                try:
                    if float(r.get(k, 0) or 0) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False
    if not _nvme_ts_has_data(nvme_ts):
        _tp_rows = _ba_csv("temporal_read_write_trim_pattern.csv")
        if _tp_rows:
            _WIN = 10.0  # temporal pattern window seconds (blktrace_analyzer.WINDOW_S)
            _synth: list[dict] = []
            for _r in _tp_rows:
                try:
                    _t0 = float(_r.get("window_start_sec", 0) or 0)
                    _t1 = float(_r.get("window_end_sec", _t0 + _WIN) or (_t0 + _WIN))
                    _dur = max(_t1 - _t0, 1e-6)
                    _rb = float(_r.get("read", 0) or 0)
                    _wb = float(_r.get("write", 0) or 0)
                    _tb = float(_r.get("trim", 0) or 0)
                    _rc = float(_r.get("read_count", 0) or 0)
                    _wc = float(_r.get("write_count", 0) or 0)
                    _tc = float(_r.get("trim_count", 0) or 0)
                    _busy = 1.0 if (_rb + _wb + _tb) > 0 else 0.0
                    _synth.append({
                        "time_sec":   round(_t0, 1),
                        "rd_bw_mbs":  round(_rb / 1e6 / _dur, 3),
                        "wr_bw_mbs":  round(_wb / 1e6 / _dur, 3),
                        "rd_iops":    round(_rc / _dur, 2),
                        "wr_iops":    round(_wc / _dur, 2),
                        "disc_ios":   round(_tc / _dur, 4),
                        "io_util_pct": round(_busy * 100.0, 1),
                        # latency unknown from temporal pattern; leave 0
                        "rd_lat_ms":  0.0,
                        "wr_lat_ms":  0.0,
                    })
                except (TypeError, ValueError):
                    continue
            if _synth:
                nvme_ts = _synth
    smart_sum   = _read_json(raw_dir / "smart_summary.json")
    reqsize     = _ba_csv("request_size_distribution.csv")
    iat         = _ba_csv("interarrival_distribution.csv")
    temporal    = _ba_csv("temporal_read_write_trim_pattern.csv")
    bw_stream   = _ba_csv("bandwidth_per_stream.csv")
    hot_regions = _ba_csv("hot_regions_overall.csv")
    # Complete per-bucket LBA distribution (every bucket that saw traffic).
    # Written by blktrace_analyzer v1.38.29+; reports generated against
    # older runs degrade gracefully — the chart shows only hot regions.
    lba_dist = _ba_csv("lba_distribution_full.csv")
    # Queue-depth analysis (v40+) — time-series + time-weighted histogram.
    # Empty list when the run pre-dates QD analysis; sections degrade gracefully.
    qd_ts_data   = _ba_csv("queue_depth_timeseries.csv")
    qd_dist_data = _ba_csv("queue_depth_distribution.csv")
    # Fallback: when exact blktrace Q→C queue-depth files are missing, use
    # nvme_driver_timeseries inflight/aqu-sz so the AI-stack layer pressure
    # section does not render empty. This is coarser than event-level QD but
    # valid for sampled device pressure.
    qd_fallback_note = ""
    if not qd_ts_data and nvme_ts:
        _fallback = []
        for _r in nvme_ts:
            _t = _to_float(_r.get("time_sec")) or 0
            _qd = (_to_float(_r.get("inflight")) or _to_float(_r.get("queue_depth")) or
                   _to_float(_r.get("aqu_sz")) or _to_float(_r.get("avgqu")) or 0)
            if _qd > 0:
                _fallback.append({"t_sec": _t, "qd_total": _qd, "qd_read": 0, "qd_write": 0,
                                  "source": "iostat/sysfs inflight fallback"})
        if _fallback:
            qd_ts_data = _fallback
            qd_fallback_note = "Fallback from nvme_driver_timeseries inflight / iostat aqu-sz because exact blktrace Q→C queue-depth files are missing."
    # Capacity/used context for the SSD I/O Distribution chart
    _ssd_cap_gb  = _to_float(smart_sum.get("nvme_device_capacity_gb")) or 0.0
    _ssd_used_gb = _to_float(smart_sum.get("hicache_fs_used_gb")) or \
                   _to_float(smart_sum.get("hicache_size_gb")) or 0.0
    # blktrace analysis summary lives in the subdir; merge with collect summary.
    bt_summary  = _read_json(raw_dir / "summary.json")
    _ba_summary = _read_json(_ba_dir / "summary.json")
    if _ba_summary:
        # Fill NVMe BW/IOPS/bytes from blktrace analysis when absent.
        for _k, _v in _ba_summary.items():
            bt_summary.setdefault(_k, _v)
    manual_setup = _read_json(raw_dir / "setup_details.json") or _read_json(raw_dir / "setup.json")

    # ── Set module-level wall-clock anchor ────────────────────────────────────
    # t0_epoch is the Unix timestamp of the collection-window start, stored in
    # summary.json during both live collect and Prometheus analyze runs.
    # All _ts_to_dates() calls pick this up automatically so every chart's
    # X axis is anchored to the same real wall-clock time — consistent with
    # the --start / --end the user specified on the command line.
    _t0_candidates = [
        bt_summary.get("t0_epoch"),           # live collect primary
        sglang_sum.get("t0_epoch"),            # sglang_summary fallback
        sglang_sum.get("start_time"),          # Prometheus analyze fallback
        bt_summary.get("start_time"),
        bt_summary.get("prom_start"),
    ]
    _T0_EPOCH = 0.0
    for _t0c in _t0_candidates:
        try:
            v = float(_t0c)
            if v > 1_000_000_000:   # sanity: must be a plausible Unix timestamp (> year 2001)
                _T0_EPOCH = v
                break
        except (TypeError, ValueError):
            pass
    setup = {
        "Model": sglang_sum.get("model_name", "unknown"),
        "Runtime": sglang_sum.get("runtime", sglang_sum.get("server_type", "SGLang")),
        "Instance": sglang_sum.get("instance", sglang_sum.get("prom_instance", "unknown")),
        "Job": sglang_sum.get("job", "unknown"),
        "Engine": sglang_sum.get("engine_type", "unknown"),
        "TP size": sglang_sum.get("tp_size", sglang_sum.get("tensor_parallel_size", "unknown")),
        "dtype": sglang_sum.get("dtype", "unknown"),
        "KV dtype": sglang_sum.get("kv_cache_dtype", "unknown"),
        "Context length": sglang_sum.get("context_length", sglang_sum.get("max_context_len", "unknown")),
        "Page size": sglang_sum.get("page_size", sglang_sum.get("sglang_page_size", "unknown")),
        "Num pages/tokens": sglang_sum.get("num_pages", sglang_sum.get("max_total_num_tokens", "unknown")),
        "HiCache": sglang_sum.get("hicache_size", sglang_sum.get("hicache_mem_layout", "unknown")),
    }

    # Merge observed server info derived from Prometheus metric labels (model,
    # instance, TP/PP/EP, page size). These overwrite only missing/unknown
    # fields so the setup table reflects the real server, not a default profile.
    _prom_server_info = sglang_sum.get("prometheus_server_info")
    if isinstance(_prom_server_info, dict):
        for _k, _v in _prom_server_info.items():
            cur = setup.get(_k)
            if cur is None or str(cur).strip().lower() in ("", "unknown", "none", "n/a", "0"):
                setup[_k] = str(_v)

    def _is_missing_setup_value(v):
        if v is None:
            return True
        try:
            sv = str(v).strip()
        except Exception:
            return True
        return sv == "" or sv.lower() in {"unknown", "none", "nan", "null", "n/a", "—", "-"}

    def _parse_launch_command_to_setup(cmd: str) -> dict:
        if not cmd:
            return {}
        try:
            import shlex as _shlex_setup
            toks = _shlex_setup.split(str(cmd))
        except Exception:
            toks = str(cmd).split()
        out = {"Reference launch command": str(cmd)}
        mapping = {
            "--model-path": "Model path", "--tp-size": "TP size", "--dp-size": "DP size",
            "--port": "Port", "--host": "Host", "--attention-backend": "Attention backend",
            "--mem-fraction-static": "Mem fraction static", "--hicache-size": "HiCache size",
            "--hicache-storage-backend": "HiCache storage backend",
            "--hicache-write-policy": "HiCache write policy", "--reasoning-parser": "Reasoning parser",
            "--tool-call-parser": "Tool call parser", "--dtype": "dtype",
            "--kv-cache-dtype": "KV dtype", "--context-length": "Context length",
            "--page-size": "Page size", "--schedule-policy": "Schedule policy",
        }
        bool_flags = {
            "--enable-hierarchical-cache": ("Hierarchical cache", "enabled"),
            "--trust-remote-code": ("Trust remote code", "true"),
            "--enable-metrics": ("Metrics enabled", "true"),
            "--enable-cache-report": ("Cache report enabled", "true"),
        }
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in bool_flags:
                k, v = bool_flags[t]; out[k] = v; i += 1; continue
            if t in mapping and i + 1 < len(toks):
                out[mapping[t]] = toks[i+1]; i += 2; continue
            i += 1
        if "Model path" in out and _is_missing_setup_value(out.get("Model")):
            out["Model"] = out["Model path"]
        return out

    if isinstance(manual_setup, dict):
        setup.update({str(k): str(v) for k, v in manual_setup.items()})
        launch_cmd = (setup.get("Launch command") or setup.get("launch_command") or
                      setup.get("SGLang launch command") or setup.get("sglang_launch_command"))
        if launch_cmd:
            for k, v in _parse_launch_command_to_setup(launch_cmd).items():
                if _is_missing_setup_value(setup.get(k)):
                    setup[k] = str(v)

    sglang_default_setup = {
        "Model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Model path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Runtime": "SGLang",
        "Host": "0.0.0.0",
        "Port": "30000",
        "TP size": "8",
        "DP size": "1",
        "Attention backend": "triton",
        "Mem fraction static": "0.8",
        "Hierarchical cache": "enabled",
        "HiCache size": "50",
        "HiCache storage backend": "file",
        "HiCache write policy": "write_through",
        "Trust remote code": "true",
        "Metrics enabled": "true",
        "Cache report enabled": "true",
        "Reasoning parser": "deepseek-r1",
        "Tool call parser": "deepseekv3",
        "Reference launch command": "python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp-size 8 --dp-size 1 --port 30000 --host 0.0.0.0 --attention-backend triton --mem-fraction-static 0.8 --enable-hierarchical-cache --hicache-size 50 --trust-remote-code --enable-metrics --enable-cache-report --reasoning-parser deepseek-r1 --tool-call-parser deepseekv3 --hicache-storage-backend file --hicache-write-policy write_through",
    }
    # Suppress model-specific fields of the fabricated profile when a real model
    # was observed (e.g. from Prometheus labels) and it is not the DeepSeek
    # reference model — otherwise the setup table shows DeepSeek's launch
    # command / TP size / parsers for an unrelated run.
    _default_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    _obs_model = setup.get("Model") or setup.get("Model path")
    _model_observed = bool(_obs_model and not _is_missing_setup_value(_obs_model)
                           and str(_obs_model) != _default_id)
    _model_specific = {
        "Model", "Model path", "TP size", "DP size", "Attention backend",
        "Reasoning parser", "Tool call parser", "Reference launch command",
        "HiCache size", "Mem fraction static",
    }
    defaulted = []
    for k, v in sglang_default_setup.items():
        if _model_observed and k in _model_specific:
            continue
        if _is_missing_setup_value(setup.get(k)):
            setup[k] = v
            defaulted.append(k)
    # Cross-fill: when the real Model is provided but Model path is missing
    # (or vice-versa), propagate the populated one rather than letting the
    # hardcoded DeepSeek default win. Without this the setup table shows
    # "Model: gpt-oss-120b" alongside "Model path: deepseek-ai/...".
    real_model = setup.get("Model")
    real_path  = setup.get("Model path")
    default_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    if real_model and not _is_missing_setup_value(real_model) and real_model != default_id:
        # Real Model exists; if Model path is the stale DeepSeek default, replace it.
        if real_path == default_id or _is_missing_setup_value(real_path):
            setup["Model path"] = real_model
            if "Model path" in defaulted:
                defaulted.remove("Model path")
    elif real_path and not _is_missing_setup_value(real_path) and real_path != default_id:
        if real_model == default_id or _is_missing_setup_value(real_model):
            setup["Model"] = real_path
            if "Model" in defaulted:
                defaulted.remove("Model")
    if defaulted:
        setup["Default profile"] = "SGLang DeepSeek-R1-70B HiCache reference launch; defaulted fields are assumed, not observed"
        setup["Defaulted fields"] = ", ".join(defaulted)

    # Prefer a user-facing application/benchmark label from setup_details.json;
    # do not show Prometheus instance as a setup row.
    app_bench = (setup.get("Application / Benchmark") or setup.get("Application") or
                 setup.get("Benchmark") or setup.get("benchmark") or setup.get("application") or
                 setup.get("Instance") or sglang_sum.get("instance") or "unknown")
    setup["Application / Benchmark"] = app_bench
    setup.pop("Instance", None)

    # ── Bench summary (per-request benchmark output) ────────────────────────
    # Auto-discover bench_summary.{json,txt} in raw/; the parser handles
    # both the SGLang bench_serving plaintext and a JSON alias form.
    bench: dict = {}
    try:
        from ..bench_summary import discover_bench_summary, parse_bench_summary
        bench_path = discover_bench_summary(raw_dir)
        if bench_path:
            bench = parse_bench_summary(bench_path)
    except Exception:
        bench = {}
    # Prometheus-only report mode: benchmark summaries are validation inputs for
    # humans, not data sources for tiles/charts.  Keep this empty so the report
    # does not create separate benchmark sections or benchmark fallback lines.
    bench = {}

    # Percentile timeseries from Prometheus (P50/P90/P99 over time).
    # Written by amoprof.prometheus_source.fetch_from_prometheus when
    # --prometheus is supplied to `analyze`. Falls back to empty dict
    # otherwise (collect-only runs don't have histogram_quantile data).
    pct_ts: dict = {}
    try:
        pct_ts_path = raw_dir / "sglang_percentiles_timeseries.json"
        if pct_ts_path.exists() and pct_ts_path.stat().st_size > 0:
            import json as _json
            pct_ts = _json.loads(pct_ts_path.read_text())
    except Exception:
        pct_ts = {}

    def _setup_value(*keys, default="—"):
        for k in keys:
            v = setup.get(k)
            if not _is_missing_setup_value(v):
                return str(v)
        return default

    def _setup_float(*keys, default=0.0) -> float:
        for k in keys:
            v = setup.get(k)
            if not _is_missing_setup_value(v):
                try:
                    return float(str(v).replace(",", "").split()[0])
                except Exception:
                    pass
        return float(default)

    def _setup_gpu_count() -> float:
        """Active inference GPU count for HBM capacity.

        Prefer TP × DP from setup_details because report HBM capacity should
        reflect the GPUs participating in the model placement, not necessarily
        every physical GPU present in the host. Fallback to GPU Count / parsed
        GPU description when TP is absent.
        """
        tp = _setup_float("TP size", "tp_size", "tensor_parallel_size", "TP", "tp", default=0.0)
        dp = _setup_float("DP size", "dp_size", "data_parallel_size", "DP", "dp", default=1.0) or 1.0
        if tp > 0:
            return max(tp * dp, 1.0)
        c = _setup_float("GPU Count", "gpu_count", "num_gpus", "Number of GPUs", default=0.0)
        if c > 0:
            return c
        try:
            import re as _re
            desc = str(_setup_value("GPU", "gpu", default=""))
            m = _re.search(r"(\d+)\s*[xX]", desc)
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return 1.0

    def _active_sglang_throughput() -> tuple[str, str]:
        """Return display value/note for throughput KPI.

        Prefer explicit summary fields. If those are absent or zero, compute
        the script-compatible active-window mean from sglang_timeseries.csv
        gen_throughput. If there are no SGLang samples in the selected
        Prometheus window, show N/A instead of a misleading 0.0 tok/s.
        """
        for key in ("ai_op_decode_tok_s", "gen_tp_active_mean", "gen_tp_mean", "decode_tok_s", "output_token_throughput"):
            try:
                v = float(sglang_sum.get(key, 0) or 0)
                if v > 0:
                    return f"{v:.1f}", f"From sglang_summary.json:{key}"
            except Exception:
                pass
        gen_col = _find_col(sglang_ts, "gen_throughput") if sglang_ts else None
        if gen_col:
            vals = [_to_float(r.get(gen_col)) for r in sglang_ts]
            active = [v for v in vals if v > 0]
            if active:
                return f"{(sum(active) / len(active)):.1f}", f"Active mean from {len(active)}/{len(vals)} gen_throughput samples"
            if vals:
                return "0.0", "SGLang gen_throughput samples were present but all zero"
        for key in ("gen_tp_peak", "rt_decode_tokens"):
            try:
                if key == "rt_decode_tokens":
                    dur = float(sglang_sum.get("collection_elapsed_s") or 0)
                    v = float(sglang_sum.get(key, 0) or 0) / max(dur, 1.0)
                else:
                    v = float(sglang_sum.get(key, 0) or 0)
                if v > 0:
                    return f"{v:.1f}", f"Prometheus-derived from sglang_summary.json:{key}"
            except Exception:
                pass
        return "N/A", "No SGLang throughput gauge/counter samples in this Prometheus window"

    # KPI tiles
    kpis = []
    if sglang_sum:
        _ttft_delta = _ratio_delta_ms_from_rows(sglang_ts, "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count")
        _itl_delta  = _ratio_delta_ms_from_rows(sglang_ts, "inter_token_latency_seconds_sum", "inter_token_latency_seconds_count")
        _e2e_delta  = _ratio_delta_ms_from_rows(sglang_ts, "e2e_request_latency_seconds_sum", "e2e_request_latency_seconds_count")
        _ttft_pct = _pct_ts_latency_ms(pct_ts, "ttft", "p50")
        _itl_pct  = _pct_ts_latency_ms(pct_ts, "itl", "p50")
        _e2e_pct  = _pct_ts_latency_ms(pct_ts, "e2e", "p50")
        _ttft_p50_ms = _ttft_pct or float(sglang_sum.get('server_ttft_p50_ms', 0) or 0)
        _itl_p50_ms  = _itl_pct  or float(sglang_sum.get('server_itl_p50_ms', 0) or sglang_sum.get('server_tpot_p50_ms', 0) or 0)
        _e2e_p50_ms  = _e2e_pct  or float(sglang_sum.get('server_e2e_p50_ms', 0) or 0)
        _tp_p50 = _active_p50_from_rows(sglang_ts, "gen_throughput") or float(sglang_sum.get('gen_tp_p50', sglang_sum.get('gen_tp_active_p50', 0)) or 0)
        _ttft_ms_kpi = _ttft_delta or _ttft_pct or (sglang_sum.get('server_ttft_ms') or sglang_sum.get('server_ttft_p50_ms') or 0)
        _itl_ms_kpi  = _itl_delta  or _itl_pct  or (sglang_sum.get('server_itl_ms') or sglang_sum.get('server_itl_p50_ms') or 0)
        _e2e_ms_kpi  = _e2e_delta  or _e2e_pct  or (sglang_sum.get('server_e2e_ms') or sglang_sum.get('server_e2e_p50_ms') or 0)
        _lat_note = "Selected-window Δsum/Δcount" if (_ttft_delta or _itl_delta or _e2e_delta) else ("Selected-window percentile p50" if (_ttft_pct or _itl_pct or _e2e_pct) else "From summary")
        _tp_mean_val, _tp_mean_note = _active_sglang_throughput()
        kpis += [
            _kpi_tile("Model", _setup_value("Model", "Model path")[-40:],
                      note="From setup/server configuration"),
            _kpi_tile("TTFT mean", f"{float(_ttft_ms_kpi or 0):.0f}", "ms", _lat_note),
            _kpi_tile("TTFT p50", f"{float(_ttft_p50_ms or _ttft_ms_kpi or 0):.0f}", "ms", "Median · histogram_quantile/percentile timeseries"),
            _kpi_tile("TPOT / ITL mean", f"{float(_itl_ms_kpi or 0):.1f}", "ms", _lat_note),
            _kpi_tile("TPOT / ITL p50", f"{float(_itl_p50_ms or _itl_ms_kpi or 0):.1f}", "ms", "Median · histogram_quantile/percentile timeseries"),
            _kpi_tile("E2E mean", f"{float(_e2e_ms_kpi or 0):.0f}", "ms", _lat_note),
            _kpi_tile("E2E p50", f"{float(_e2e_p50_ms or _e2e_ms_kpi or 0):.0f}", "ms", "Median · histogram_quantile/percentile timeseries"),
            _kpi_tile("Throughput mean", _tp_mean_val, "tok/s" if _tp_mean_val != "N/A" else "", _tp_mean_note),
            _kpi_tile("Throughput p50", f"{float(_tp_p50 or 0):.1f}" if _tp_p50 else "N/A", "tok/s" if _tp_p50 else "", "Median of active sglang_gen_throughput samples"),
            _kpi_tile("Cache hit", f"{float(sglang_sum.get('cache_hit_pct', sglang_sum.get('cache_hit_rate_realtime_pct', 0)) or 0):.1f}",
                      "%", f"primary={sglang_sum.get('cache_hit_calc_method', 'canonical')} · prefill={float(sglang_sum.get('cache_hit_prefill_token_weighted_pct', 0) or 0):.1f}% · cached/prompt={float(sglang_sum.get('cache_hit_token_weighted_pct', 0) or 0):.1f}%"),
        ]
    if gpu_sum:
        _gpu_active_mean, _gpu_active_peak, _gpu_note = _active_gpu_util_from_rows(gpu_ts)
        _gpu_mean_show = _gpu_active_mean if _gpu_active_mean > 0 else float(gpu_sum.get('gpu_util_mean', 0) or 0)
        _gpu_peak_show = max(_gpu_active_peak, float(gpu_sum.get('gpu_util_peak', 0) or 0))
        kpis += [
            _kpi_tile("GPU util", f"{_gpu_mean_show:.1f}", "%",
                      (f"Peak {_gpu_peak_show:.0f}% · {_gpu_note}" if _gpu_note else f"Peak {_gpu_peak_show:.0f}%")),
            (lambda _gpu_n, _per_gb, _per_total: _kpi_tile(
                "HBM used (all GPUs)",
                f"{_per_gb*_gpu_n:.1f} / {_per_total*_gpu_n:.0f}",
                "GB",
                f"{((_per_gb*_gpu_n)/max(_per_total*_gpu_n,1)*100):.0f}% full · {int(_gpu_n)} GPU(s) · {_per_gb:.1f} GB/GPU avg"
            ))(_setup_gpu_count(), float(gpu_sum.get('hbm_used_mb_mean', 0) or 0)/1024.0,
               float(_setup_float('GPU Memory per Device GB', 'GPU memory per device GB', 'HBM per GPU GB', default=float(gpu_sum.get('hbm_total_gb', 40.0) or 40.0)))),
            _kpi_tile("GPU power", f"{gpu_sum.get('power_w_peak', 0):.0f}", "W",
                      "Peak per-GPU draw"),
        ]
    # Capacity tiles from setup_details/raw summaries. These are capacity
    # views, separate from bandwidth/IOPS.
    # DRAM/L2 capacity should reflect the configured L2 cache allocation when
    # setup_details provides HiCache size per GPU. This is more accurate for
    # cache-sizing reports than showing the whole host DRAM DIMM capacity.
    gpu_count_for_l2 = _setup_gpu_count() or 1.0
    hicache_per_gpu_gb = _setup_float(
        "HiCache size per GPU GB", "HiCache DRAM per GPU GB",
        "HiCache size GB per GPU", "hicache_size_per_gpu_gb",
        "hicache_dram_per_gpu_gb", "HiCache size", "hicache_size",
        default=0.0)
    hicache_total_gb = _setup_float(
        "HiCache total GB", "HiCache DRAM total GB", "L2 DRAM cache capacity GB",
        "L2 cache capacity GB", "hicache_total_gb", "l2_dram_cache_capacity_gb",
        default=0.0)
    if hicache_total_gb <= 0 and hicache_per_gpu_gb > 0:
        hicache_total_gb = hicache_per_gpu_gb * max(gpu_count_for_l2, 1.0)
    dram_total_gb = hicache_total_gb or _setup_float("Host DRAM GB", "DRAM capacity GB", "DRAM total GB", default=0.0)
    dram_used_gb = _setup_float("Host DRAM used GB", "DRAM used GB", default=0.0)
    _l2_used_source = "setup/static"
    _l2_used_variation_pct = 0.0
    # Prefer the run-local SGLang L2/HiCache token gauge.  In combined reports
    # the Executive and End Report paths already use this signal, but the
    # Interactive tab was previously looking only in sglang_summary.json.  Many
    # Prometheus-only runs have the gauge in sglang_timeseries.csv only, which
    # made this tile show "— / 100 GB" even while the other tabs showed
    # ~46.6 / 100 GB.
    try:
        def _active_mean_ts_value(*partials: str) -> tuple[float, str, float]:
            for part in partials:
                col = _find_col(sglang_ts, part) if sglang_ts else None
                if not col:
                    continue
                vals = [_to_float(r.get(col), float("nan")) for r in sglang_ts]
                vals = [v for v in vals if v == v]
                active = [v for v in vals if v > 0]
                if active:
                    mean = float(sum(active) / len(active))
                    mn = min(active); mx = max(active)
                    variation = ((mx - mn) / mean * 100.0) if mean > 0 and mx >= mn else 0.0
                    return mean, "selected-window active mean", variation
                if vals:
                    return float(vals[-1]), "selected-window last sample", 0.0
            return 0.0, "missing", 0.0

        _l2_used_tokens, _l2_used_source, _l2_used_variation_pct = _active_mean_ts_value(
            "sglang_hicache_host_used_tokens", "hicache_host_used_tokens")
        _l2_used_dynamic = (_l2_used_tokens > 0 and _l2_used_variation_pct > 1.0)
        if _l2_used_tokens <= 0:
            _l2_used_source = "sglang_hicache_host_used_tokens missing from selected-window timeseries"
        elif not _l2_used_dynamic:
            _l2_used_source = "sglang_hicache_host_used_tokens present but flat/static in selected window"
        _kv_kb = _setup_float("KV bytes per token KB", "KV_bytes_per_token_KB", "kv_bytes_per_token_kb", default=0.0)
        if _kv_kb <= 0:
            # Fall back to the same model-profile estimator used by the L3
            # bandwidth charts.  It returns bytes/token, so convert to KiB/token.
            _model_for_kv = (_setup_value("Model", "Model path", default="")
                             or sglang_sum.get("model_name", ""))
            _kv_dtype_for_kv = (_setup_value("KV dtype", "kv_cache_dtype", default="")
                                or sglang_sum.get("kv_cache_dtype", "fp16") or "fp16")
            _kv_kb = _kv_bytes_per_token_for_model(str(_model_for_kv), str(_kv_dtype_for_kv)) / 1024.0
        if _l2_used_tokens > 0 and _kv_kb > 0:
            _tp_for_kv = max(_setup_float("TP size", "tp_size", default=1.0), 1.0)
            _full_gb = _l2_used_tokens * _kv_kb / (1024 * 1024)
            _tp_gb = _l2_used_tokens * (_kv_kb / _tp_for_kv) / (1024 * 1024)
            if dram_total_gb > 0 and _full_gb > dram_total_gb * 1.05 and _tp_gb <= dram_total_gb * 1.05:
                dram_used_gb = _tp_gb
            elif dram_total_gb > 0 and _full_gb > dram_total_gb * 1.05:
                # Keep the tile truthful: avoid impossible >100% values.
                # If a setup value was supplied, keep it; otherwise mark unknown.
                dram_used_gb = dram_used_gb if dram_used_gb > 0 else 0.0
            else:
                dram_used_gb = _full_gb
    except Exception:
        pass
    if dram_total_gb > 0:
        _dram_pct = dram_used_gb / max(dram_total_gb, 1) * 100.0 if dram_used_gb > 0 else 0.0
        if dram_used_gb > 0 and hicache_total_gb > 0:
            if _l2_used_variation_pct > 1.0:
                _dram_note = f"{_dram_pct:.0f}% full · selected-window mean residency (sglang_hicache_host_used_tokens varied {_l2_used_variation_pct:.1f}%)"
            else:
                _dram_note = f"{_dram_pct:.0f}% full · flat residency snapshot from sglang_hicache_host_used_tokens; not a DRAM bandwidth/activity metric"
            _dram_value = f"{dram_used_gb:.1f} / {dram_total_gb:.0f}"
            _dram_label = "DRAM used"
        elif dram_used_gb > 0:
            _dram_note = f"{_dram_pct:.0f}% full · host DRAM"
            _dram_value = f"{dram_used_gb:.1f} / {dram_total_gb:.0f}"
            _dram_label = "DRAM used"
        else:
            _dram_note = (f"runtime residency unavailable · {_l2_used_source}" if hicache_total_gb > 0
                          else "host DRAM total · used capacity unavailable")
            _dram_value = f"— / {dram_total_gb:.0f}"
            _dram_label = "DRAM residency"
        kpis.append(_kpi_tile(_dram_label, _dram_value, "GB", _dram_note))

    l3_total_gb = _setup_float("L3 (local storage) capacity GB", "L3 Capacity GB", "L3 total capacity GB", "l3_storage_capacity_gb", default=0.0)
    l3_used_gb = _setup_float("L3 (local storage) used GB", "L3 used capacity GB", "l3_storage_used_gb", default=0.0)
    if smart_sum:
        l3_total_gb = float(smart_sum.get('hicache_fs_total_gb', l3_total_gb) or l3_total_gb)
        l3_used_gb = float(smart_sum.get('hicache_fs_used_gb', l3_used_gb) or l3_used_gb)
    if l3_total_gb > 0:
        kpis.append(_kpi_tile("L3 (local storage) used",
                              (f"{l3_used_gb:.1f} / {l3_total_gb:.0f}" if l3_used_gb > 0 else f"— / {l3_total_gb:.0f}"),
                              "GB",
                              (f"{l3_used_gb/max(l3_total_gb,1)*100:.0f}% full" if l3_used_gb > 0 else "used capacity unavailable")))

    if smart_sum and str(_setup_value("L3 storage type", "L3 Storage Type", default="")).lower() in {"nvme", "nvme ssd", "ssd"}:
        kpis += [
            _kpi_tile("L3 (local storage) device", str(smart_sum.get('model', '—'))[:24], "",
                      f"Temp {smart_sum.get('temperature_c', 0)}°C  "
                      f"WAF {smart_sum.get('waf', 0):.2f}"),
        ]
    if bt_summary:
        kpis += [
            _kpi_tile("Block IOs", f"{bt_summary.get('total_events', 0):,}", "",
                      f"R={bt_summary.get('read_events',0):,}  "
                      f"W={bt_summary.get('write_events',0):,}  "
                      f"T={bt_summary.get('trim_events',0):,}"),
        ]

    def _canonical_kpi_sort_key(tile_html: str) -> int:
        txt = re.sub(r"<[^>]+>", " ", tile_html).lower()
        order = [
            ("cache hit", 10), ("ttft mean", 20), ("ttft p50", 21),
            ("tpot / itl mean", 30), ("tpot mean", 30), ("itl mean", 30),
            ("tpot / itl p50", 31), ("tpot p50", 31), ("itl p50", 31),
            ("e2e mean", 40), ("e2e p50", 41),
            ("throughput mean", 50), ("throughput p50", 51),
            ("gpu util", 60), ("hbm used", 70), ("hbm util", 70),
            ("gpu power", 80), ("dram used", 90), ("dram residency", 90), ("dram bw", 91),
            ("l3 (local storage) used", 100), ("l3 (local storage) read", 101),
            ("l3 (local storage) write", 102), ("logical kv movement", 103),
            ("physical l3", 104), ("model", 900), ("server", 910), ("bench", 920),
        ]
        for key, idx in order:
            if key in txt:
                return idx
        return 800

    # Prometheus-derived per-request percentiles (server-side ground truth)
    # — present when analyze ran with --prometheus AND the SGLang scrape
    # exposes histogram bucket metrics.
    if sglang_sum and sglang_sum.get("server_ttft_p99_ms") is not None:
        p50 = sglang_sum.get("server_ttft_p50_ms", 0)
        p99 = sglang_sum.get("server_ttft_p99_ms", 0)
        kpis.append(_kpi_tile(
            "Server TTFT (Prom)", f"{p50/1000:.1f}",
            "s P50", f"P99 {p99/1000:.1f}s · histogram_quantile"))
    if sglang_sum and sglang_sum.get("server_itl_p99_ms") is not None:
        p50 = sglang_sum.get("server_itl_p50_ms", 0)
        p99 = sglang_sum.get("server_itl_p99_ms", 0)
        kpis.append(_kpi_tile(
            "Server ITL (Prom)", f"{p50:.0f}",
            "ms P50", f"P99 {p99:.0f}ms · histogram_quantile"))
    if sglang_sum and sglang_sum.get("server_e2e_p99_ms") is not None:
        p50 = sglang_sum.get("server_e2e_p50_ms", 0)
        p99 = sglang_sum.get("server_e2e_p99_ms", 0)
        kpis.append(_kpi_tile(
            "Server E2E (Prom)", f"{p50/1000:.1f}",
            "s P50", f"P99 {p99/1000:.1f}s · histogram_quantile"))

    # Bench-summary KPIs (per-request percentile stats)
    if bench:
        if bench.get("total_requests") is not None:
            kpis.append(_kpi_tile(
                "Bench requests", f"{int(bench['total_requests']):,}",
                "", f"at {bench.get('request_rate', 0):.1f} req/s sent · "
                f"{bench.get('req_per_s', 0):.2f} req/s served"))
        if bench.get("avg_prompt_tokens") is not None:
            kpis.append(_kpi_tile(
                "Avg prompt", f"{int(bench['avg_prompt_tokens']):,}",
                "tok", f"P99 {int(bench.get('p99_prompt_tokens', 0)):,}"))
        if bench.get("avg_output_tokens") is not None:
            kpis.append(_kpi_tile(
                "Avg output", f"{int(bench['avg_output_tokens']):,}",
                "tok", f"P99 {int(bench.get('p99_output_tokens', 0)):,}"))
        if bench.get("avg_ttft_s") is not None:
            kpis.append(_kpi_tile(
                "Bench TTFT", f"{bench['avg_ttft_s']:.1f}",
                "s", f"P99 {bench.get('p99_ttft_s', 0):.1f}s"))
        if bench.get("avg_itl_s") is not None:
            kpis.append(_kpi_tile(
                "Bench ITL", f"{bench['avg_itl_s']*1000:.0f}",
                "ms", f"P99 {bench.get('p99_itl_s', 0)*1000:.0f}ms"))
        if bench.get("input_tok_per_s") is not None:
            kpis.append(_kpi_tile(
                "Input TPS", f"{bench['input_tok_per_s']:,.0f}",
                "tok/s", "Aggregate prompt token rate"))
        if bench.get("output_tok_per_s") is not None:
            kpis.append(_kpi_tile(
                "Output TPS", f"{bench['output_tok_per_s']:,.2f}",
                "tok/s", "Aggregate decode token rate"))
        if bench.get("cache_hit_rate") is not None:
            kpis.append(_kpi_tile(
                "Bench cache hit", f"{bench['cache_hit_rate']*100:.1f}",
                "%", "From bench summary (post-run aggregate)"))

    kpis = sorted(kpis, key=_canonical_kpi_sort_key)

    formula_token_cache = """Throughput mean = active_mean(sglang_gen_throughput)
Throughput p50 = median(nonzero_samples(sglang_gen_throughput))
Fallback throughput = Δsglang_generation_tokens_total / Δtime_sec
Primary cache hit = cache-served prompt/prefill tokens / (cache-served + compute-served prompt/prefill tokens) × 100
Request JSON exact = Σmeta_info.cached_tokens / Σ(meta_info.cached_tokens + meta_info.prompt_tokens) × 100
OpenAI usage exact = Σusage.prompt_tokens_details.cached_tokens / Σusage.prompt_tokens × 100
Prometheus fallback = Δsglang_realtime_tokens_total{mode="prefill_cache"} / (Δprefill_cache + Δprefill_compute) × 100
Cached/prompt diagnostic = Δsglang_cached_tokens_total / Δsglang_prompt_tokens_total × 100
Gauge diagnostics = avg/active/peak sglang_cache_hit_rate samples"""
    formula_kv_tiers = """L1/HBM tokens = sglang_kv_used_tokens or sglang_token_usage-derived fields
L2/host cache occupancy = sglang_hicache_host_used_tokens / sglang_hicache_host_total_tokens × 100
L3 (local storage) movement = Δsglang_backuped_tokens_total (write/offload) and Δsglang_prefetched_tokens_total (read/onboard)
Diagnostic-only counters = Δsglang_load_back_tokens_total (hierarchy restore upper bound) and Δsglang_evicted_tokens_total (cache pressure)
Bytes estimate = token_delta × KV bytes per token from setup/model profile; load_back is L2→L1 restore and is not counted as SSD/L3 bytes"""
    formula_latency_pct = """P50/P90/P99 = histogram_quantile(q, rate(metric_bucket[window]))
TTFT metric = sglang_time_to_first_token_seconds_bucket
ITL metric = sglang_inter_token_latency_seconds_bucket
E2E metric = sglang_e2e_request_latency_seconds_bucket
Prompt/output token percentile metrics use prompt/generation token histogram buckets when exported."""
    formula_l3_bw = """L3 (local storage) read/onboard tokens = Δsglang_prefetched_tokens_total
L3 (local storage) write/offload tokens = Δsglang_backuped_tokens_total
L2→L1 load-back restore diagnostic = Δsglang_load_back_tokens_total × KV_bytes_per_token (L2→L1 restore diagnostic; not SSD/L3 bytes)
Estimated read MB/s = prefetched_tokens × KV_bytes_per_token / Δtime_sec / 2^20
Estimated write MB/s = backuped_tokens × KV_bytes_per_token / Δtime_sec / 2^20
These are SGLang/Mooncake tier movement estimates, not block-device measurements."""
    formula_host_io = """Host block I/O is derived from node-exporter/iostat-style counters, not SGLang. It is not treated as L3 SSD I/O unless setup_details resolves a concrete L3 local SSD device or mount path. df/cache-used is capacity footprint, not window I/O:
Read IOPS = rate(node_disk_reads_completed_total) or rate(read_ios_total)
Write IOPS = rate(node_disk_writes_completed_total) or rate(write_ios_total)
Read MB/s = rate(node_disk_read_bytes_total) / 2^20
Write MB/s = rate(node_disk_written_bytes_total) / 2^20
VM activity = rate(node_vmstat_pgfault), rate(node_vmstat_pgmajfault), rate(node_vmstat_pswpin), rate(node_vmstat_pswpout)"""

    sections_html = []
    sections_html.append(_setup_section(setup))
    sections_html.append(_stack_overview_section())
    sections_html.append(_layer_header("layer-a5", "A5 · Application Layer", "Benchmark workload, request mix, prompt/output token distributions, TTFT/TPOT and tail latency.", "#24577f"))
    if bench:
        sections_html.append(_section(
            "Bench Summary — Per-Request Latency Percentiles",
            "ch_bench_lat", _chart_bench_latency_percentiles(bench),
            "Mean/P50/P90/P99/Max for TTFT, ITL (inter-token), and E2E "
            "latency. Y-axis is log scale because P99 typically dwarfs Mean."))
        sections_html.append(_section(
            "Bench Summary — Latency Spread (Box Plot)",
            "ch_bench_box", _chart_bench_latency_ranges(bench),
            "Box shows P50→P90 range; whiskers extend toward Q1-ish (lower) "
            "and P99 (upper). Hover for the exact percentiles."))
        sections_html.append(_section(
            "Bench Summary — Token Length Distribution",
            "ch_bench_tokens", _chart_bench_token_lengths(bench),
            "Distribution of prompt vs output token counts across "
            "the benchmark. Tells you whether your workload is "
            "long-prompt/short-output or vice versa."))
        sections_html.append(_section(
            "Bench Summary — Token Throughput (In vs Out)",
            "ch_bench_tps", _chart_bench_throughput_breakdown(bench),
            "Aggregate input and output token rates. A large input:output "
            "ratio (10× or more) indicates a prefill-dominated workload "
            "where most compute goes into processing prompts."))

    # Per-request latency percentile timeseries — only present when analyze
    # was run with --prometheus and the SGLang scrape includes histogram
    # bucket metrics. Each chart shows how P50, P90, P99 evolve through
    # the run, which is impossible to see from summary numbers alone.
    if pct_ts:
        chart_specs = [
            ("ttft",   "Server TTFT — P50 / P90 / P99 Time Series",
             "True per-request percentiles via histogram_quantile() over "
             "sglang_time_to_first_token_seconds_bucket using the configured "
             "Prometheus rate window from --prom-rate-window. Watch for P99 "
             "spikes that indicate occasional cold-cache prefills or "
             "queue contention."),
            ("itl",    "Server ITL (Inter-Token Latency) — P50 / P90 / P99",
             "Per-request inter-token latency percentiles. P99 ITL spikes "
             "usually correlate with HBM bandwidth contention or KV-cache "
             "eviction storms; sustained high P50 indicates the model is "
             "memory-bandwidth bound."),
            ("e2e",    "Server E2E Latency — P50 / P90 / P99",
             "Total request latency percentiles. The P99/P50 ratio tells "
             "you how heavy-tailed your latency distribution is — high "
             "ratios mean tail-latency-sensitive workloads will suffer."),
            ("prompt_tokens",  "Prompt Token Length Distribution Over Time",
             "How prompt length distribution shifts during the run. If P99 "
             "rises sharply mid-run, you're likely entering a long-context "
             "phase that will stress KV cache."),
            ("output_tokens", "Output Token Length Distribution Over Time",
             "How output length distribution shifts. Useful for spotting "
             "phase transitions between summarization (short outputs) and "
             "generation (long outputs) workloads."),
        ]
        for metric_key, title, body in chart_specs:
            chart_spec = _chart_percentile_timeseries(pct_ts, metric_key)
            if metric_key == "prompt_tokens":
                alt = _chart_percentile_timeseries(pct_ts, "input_tokens")
                if not _fig_has_numeric_y(chart_spec):
                    chart_spec = alt if _fig_has_numeric_y(alt) else _chart_token_length_fallback(
                        sglang_ts, "prompt_tokens", sglang_sum, pct_ts)
            if metric_key == "output_tokens":
                alt = _chart_percentile_timeseries(pct_ts, "generation_tokens")
                if not _fig_has_numeric_y(chart_spec):
                    chart_spec = alt if _fig_has_numeric_y(alt) else _chart_token_length_fallback(
                        sglang_ts, "output_tokens", sglang_sum, pct_ts)
            chart_spec = _apply_yaxis_override(chart_spec)
            if chart_spec is None:
                continue
            sections_html.append(_section(title, f"ch_pct_{metric_key}", chart_spec, body, formula_latency_pct))

    sections_html.append(_layer_header("layer-a4", "A4 · Inference Runtime + AI Model", "SGLang runtime behavior, model/cache configuration, token throughput, cache-hit, tier residency and KV eviction/L2→L1 load-back.", "#317bb8"))
    sections_html.append(_section("Token Throughput & Cache Hit Rate",
                                   "ch_thru",
                                   _chart_sglang_throughput(sglang_ts, None, sglang_sum),
                                   "SGLang Prometheus scrape — hover any point for values",
                                   formula_token_cache))
    sections_html.append(_section("KV Cache Tier Occupancy",
                                   "ch_kv_tiers", _chart_kv_tiers(sglang_ts),
                                   "L1=Device(HBM), L2=Host(DRAM), L3 (local storage) — stacked",
                                   formula_kv_tiers))
    sections_html.append(_section("KV Cache Eviction / L2→L1 Load-back / Backup Rates",
                                   "ch_evict", _chart_eviction(sglang_ts),
                                   "SGLang hierarchical-cache movement counters",
                                   formula_kv_tiers))

    # ── L3 (local storage) cache bandwidth (Prometheus-derived) ───────────────────────────
    # When the operator runs in Prom-only mode (no blktrace), the only L3
    # bandwidth signal available is the per-second delta of SGLang's
    # cumulative token counters multiplied by KV-bytes-per-token. These two
    # charts surface that signal in the interactive tab so users can see when
    # L3 (local storage) was active, identify burst patterns, and read endurance footprint
    # off the cumulative trace. Both gracefully no-op if the counters are
    # zero or absent.
    _kv_b_pt = _kv_bytes_per_token_for_model(
        sglang_sum.get("model_name", ""),
        sglang_sum.get("kv_cache_dtype", "fp16"))

    # Single source of truth for L3 consistency visuals in the interactive report.
    # SGLang logical movement is kept separate from physical local-source bytes.
    _sg_backup_tok = _summary_or_ts_delta(sglang_sum, sglang_ts, "backuped_tokens_total", "kvb_backuped_tokens_total")
    _sg_load_tok = _summary_or_ts_delta(sglang_sum, sglang_ts, "load_back_tokens_total", "kvb_loadback_tokens_total")
    _sg_pref_tok = _summary_or_ts_delta(sglang_sum, sglang_ts, "prefetched_tokens_total", "kvb_prefetched_tokens_total")
    _l3_sg_write_gb = (_sg_backup_tok * _kv_b_pt) / (1024**3) if _kv_b_pt > 0 else 0.0
    _l3_sg_read_gb = (_sg_pref_tok * _kv_b_pt) / (1024**3) if _kv_b_pt > 0 else 0.0
    _l3_sg_loadback_upper_gb = (_sg_load_tok * _kv_b_pt) / (1024**3) if _kv_b_pt > 0 else 0.0
    _l3_block_read_gb = _block_gb_from_summary(bt_summary, "read")
    _l3_block_write_gb = _block_gb_from_summary(bt_summary, "write")
    if (_l3_block_read_gb <= 0 or _l3_block_write_gb <= 0) and _ba_summary:
        _l3_block_read_gb = _l3_block_read_gb or _block_gb_from_summary(_ba_summary, "read")
        _l3_block_write_gb = _l3_block_write_gb or _block_gb_from_summary(_ba_summary, "write")
    _l3_res = resolve_l3_backend(setup, str(setup.get("Launch command") or setup.get("launch_command") or "")) if resolve_l3_backend else None
    _host_block_is_l3 = bool(_l3_res and getattr(_l3_res, "backend_class", "") == "local_ssd" and getattr(_l3_res, "has_local_block_mapping", False))
    _host_block_label = "L3 (local storage) physical block I/O" if _host_block_is_l3 else "Host/OS block I/O — not KV block L3"
    _host_block_note = (
        "Physical block-device counters are mapped to the resolved L3 (local storage) device/path."
        if _host_block_is_l3 else
        "Host block-device counters are not interpreted as KV block/L3 traffic because setup_details does not resolve this device as the L3 local SSD L3 backend. Read/write activity can appear here without latency/utilization if the capture lacks rd_ms/wr_ms/io_ms or iostat await/busy-time fields; in that case the latency section shows an activity fallback. Use the SGLang KV block movement chart and L3 consistency card for logical L3 direction."
    )
    if reconcile_l3_io and _l3_res:
        _l3_recon = reconcile_l3_io(_l3_res,
            sglang_write_gb=_l3_sg_write_gb, sglang_read_gb=_l3_sg_read_gb,
            block_write_gb=_l3_block_write_gb, block_read_gb=_l3_block_read_gb,
            blktrace_available=bool(_l3_block_read_gb > 0 or _l3_block_write_gb > 0 or lba_dist or temporal))
        sections_html.append(_l3_consistency_card(
            _l3_recon.display_status, _l3_recon.note,
            _l3_res.display_name, _l3_res.evidence,
            _l3_sg_read_gb, _l3_sg_write_gb, _l3_block_read_gb, _l3_block_write_gb))

    sections_html.append(_section(
        "L3 (local storage) cache Bandwidth — Prometheus-derived (MB/s)",
        "ch_l3_bw",
        _chart_l3_prom_bandwidth(sglang_ts, _kv_b_pt),
        f"Estimated from SGLang backuped/prefetched token counters × "
        f"{_kv_b_pt/1024:.0f} KB/tok (model arch). Useful work, not raw "
        f"device bytes. load_back is shown as L2→L1 restore diagnostic-only and is not counted as SSD/L3 read bytes bytes."))
    sections_html.append(_section(
        "L3 (local storage) cache Traffic — Cumulative GB (Prometheus-derived)",
        "ch_l3_cum",
        _chart_l3_prom_cumulative(sglang_ts, _kv_b_pt),
        "Endurance footprint over the collection window. Use for SSD TBW "
        "sizing when extrapolating to a longer production timeframe."))

    sections_html.append(_layer_header("layer-l1", "A3/A2/A1 · L1 HBM / CUDA / GPU", "GPU kernel/runtime layer, HBM occupancy, HBM bandwidth pressure, GPU power and utilization.", "#2f8a4e"))
    sections_html.append(_section("GPU Utilization & HBM (per GPU)",
                                   "ch_gpu", _chart_gpu(gpu_ts, gpu_summary=gpu_sum, raw_dir=raw_dir),
                                   "Per-GPU lines — click legend entries to isolate one GPU"))
    sections_html.append(_section("Total GPU Power Draw",
                                   "ch_power", _chart_power(power_ts, gpu_ts)))

    sections_html.append(_layer_header("layer-l2", "A3/A2/A1 · L2 DRAM / OS Memory Manager", "Host DRAM and OS memory behavior: page faults, swap activity, NUMA effects and DRAM bandwidth.", "#7a3f97"))
    sections_html.append(_section("Swap Storm — Page Activity",
                                   "ch_swap", _chart_swap_storm(vmstat_ts),
                                   "Per-second rates derived from /proc/vmstat cumulative counters"))
    # DRAM BW: prefer normalized AMDuProf PCM timeseries when present;
    # otherwise parse the raw multi-section AMDuProf CSV/TXT exactly like the
    # static report. This keeps the interactive tab consistent with the static tab.
    _dram_ts_rows = (_read_csv(raw_dir / "amduprof_pcm_timeseries.csv") or
                     _read_csv(raw_dir / "pcm_timeseries.csv") or
                     _read_csv(raw_dir / "pcm_memory_timeseries.csv"))
    _dram_path = (raw_dir / "amduprof_pcm_raw.txt"
                  if (raw_dir / "amduprof_pcm_raw.txt").exists()
                     and (raw_dir / "amduprof_pcm_raw.txt").stat().st_size > 0
                  else raw_dir / "amduprof_pcm_raw.csv")
    _dram_summary_fig = None
    for _dram_summary_name in ("pcm_summary.json", "pcm_memory_summary.json",
                               "amduprof_pcm_summary.json", "dram_summary.json"):
        _dram_summary_fig = _chart_dram_bw_from_summary(_read_json(raw_dir / _dram_summary_name), None)
        if _dram_summary_fig:
            break
    if not _dram_summary_fig:
        _dram_summary_fig = _chart_dram_bw_from_summary(_read_intel_pcm_memory_raw_summary(raw_dir), None)
    _dram_fig = (_chart_dram_bw_from_timeseries(_dram_ts_rows) or
                 _dram_summary_fig or
                 _chart_dram_bw(_dram_path) or
                 _chart_dram_bw_from_kv_activity(sglang_ts, _kv_b_pt))
    _dram_note = ("From --enable-dram: AMD uProf PCM on AMD or Intel PCM/perf IMC on Intel. "
                  "If PMU data is missing, AMOprof falls back to an estimated KV-movement bandwidth from SGLang backup/load_back/prefetch counters; that fallback is useful for shape/correlation, not physical DRAM-channel saturation.")
    sections_html.append(_section("System DRAM Bandwidth (CPU PMU)",
                                   "ch_dram",
                                   _dram_fig,
                                   _dram_note))

    sections_html.append(_layer_header("layer-l3", "A3/A2/A1 · L3 (local storage) / host block layer", "SGLang KV block movement is logical L3/backing-tier traffic. Host block charts are physical device telemetry and are treated as L3 only when setup_details resolves a concrete L3 local SSD device/path.", "#a45714"))
    sections_html.append(_section(f"{_host_block_label} IOPS (Read / Write / Trim)",
                                   "ch_nvme_iops", _chart_nvme_iops(nvme_ts, _host_block_label),
                                   _host_block_note))
    sections_html.append(_section(f"{_host_block_label} Bandwidth",
                                   "ch_nvme_bw", _chart_nvme_bw(nvme_ts, _host_block_label),
                                   _host_block_note))
    sections_html.append(_section(f"{_host_block_label} Latency & Device Busy-Time",
                                   "ch_nvme_lat", _chart_nvme_latency(nvme_ts, _host_block_label),
                                   _host_block_note))
    # ── Queue depth section (only when blktrace QD data is present) ────────
    # Two charts: stacked R/W in-flight over time, then the time-weighted
    # histogram showing what fraction of the run was spent at each QD value.
    if qd_ts_data:
        _qd_title = "NVMe Queue Depth Over Time" if not qd_fallback_note else "NVMe Queue Depth Over Time — iostat/sysfs fallback"
        _qd_note = (("Exact queue depth derived from blktrace Q (queued) → C (completed) events. "
                     "Stacked area shows read vs write in-flight count; purple line is total. ")
                    if not qd_fallback_note else
                    (qd_fallback_note + " The purple line is sampled total in-flight/aqu-sz; read/write split is unavailable in fallback mode. "))
        sections_html.append(_section(
            _qd_title,
            "ch_qd_ts",
            _chart_qd_timeseries(qd_ts_data),
            (_qd_note +
             "<b>QD = 1</b> means the device serves requests one at a time (latency-bound). "
             "<b>QD ≥ 32 sustained</b> means the device queue is the bottleneck.")))
    if qd_dist_data:
        sections_html.append(_section(
            "Queue Depth Distribution (time-weighted)",
            "ch_qd_dist",
            _chart_qd_distribution(qd_dist_data),
            ("Histogram is time-weighted: bar height = fraction of trace time spent at "
             "that QD value (NOT count of events). Blue: idle (QD ≤ 1) · "
             "green: healthy parallel I/O · amber: deep queue · red: saturation. "
             "Read left-to-right to find where the device actually spends its time.")))
    sections_html.append(_section("Request Size Distribution",
                                   "ch_reqsize", _chart_request_size_dist(reqsize),
                                   "From blktrace request_size_distribution.csv — bucketed by IO size"))
    sections_html.append(_section("Inter-Arrival Time Distribution",
                                   "ch_iat", _chart_iat_dist(iat),
                                   "Time between consecutive requests per op"))
    sections_html.append(_section("R/W/T Bytes per 10-sec Window (stacked)",
                                   "ch_rwt", _chart_temporal_rwt(temporal)))
    sections_html.append(_section("Bandwidth per Stream (top 20)",
                                   "ch_bps", _chart_bw_per_stream(bw_stream),
                                   "Each stream = (pid, comm) — sorted by bandwidth"))
    # Coverage warning surfaced inline with the SSD I/O chart, so a reader
    # who jumps directly to it sees the "this is under-counted" caveat next
    # to the data being doubted.
    _ba_cov_warning = (_ba_summary or {}).get("coverage_warning", "")
    _ba_kernel_gb   = _to_float((_ba_summary or {}).get("kernel_wr_gb_delta")) or 0.0
    _ba_capture_pct = (_to_float((_ba_summary or {}).get("captured_vs_kernel_ratio")) or 0.0) * 100
    # Write churn = total write traffic during the window / current FS-used at
    # analyze time. Ratio > 1 means the workload was overwriting/deleting (the
    # common HiCache eviction pattern). Ratio < 1 means writes accumulated
    # without compaction. Surface this number explicitly so the reader doesn't
    # have to mentally compare bar-totals against df.
    _total_write_gb_from_dist = sum(
        (_to_float(r.get("bytes")) or 0)
        for r in (lba_dist or [])
        if (r.get("op") or "").lower() == "write"
    ) / 1e9
    _churn_ratio = (_total_write_gb_from_dist / _ssd_used_gb) if _ssd_used_gb > 0 else 0
    _churn_line = ""
    if _total_write_gb_from_dist > 0 and _ssd_used_gb > 0:
        _churn_line = (
            f"<b>This run:</b> {_total_write_gb_from_dist:.0f} GB of writes "
            f"during the trace window vs <b>{_ssd_used_gb:.0f} GB of files</b> "
            f"on disk at analyze time"
            + (f" — write churn ratio <b>{_churn_ratio:.1f}×</b> "
                f"(workload overwrites/deletes {_churn_ratio:.1f}× as many bytes "
                f"as it leaves behind)." if _churn_ratio > 1.2
                else f" — workload accumulating without major churn "
                     f"({_churn_ratio:.2f}× ratio).")
            + "<br>"
        )
    # Device geometry surfaced by _refresh_smart_capacity — explains why
    # hot LBAs can sit far above the df-used line when blktrace traces the
    # whole device but the FS lives on a partition.
    _traced_device     = smart_sum.get("traced_device", "") or ""
    _fs_backing_dev    = smart_sum.get("fs_backing_device", "") or ""
    _fs_part_start_gb  = _to_float(smart_sum.get("fs_partition_start_gb")) or 0.0
    _lba_offset_warn   = smart_sum.get("lba_offset_warning", "") or ""

    _device_geom_line = ""
    if _traced_device or _fs_backing_dev:
        _device_geom_line = (
            f"<strong>Trace target:</strong> /dev/{_traced_device or '?'} &nbsp;·&nbsp; "
            f"<strong>FS backing device:</strong> /dev/{_fs_backing_dev or '?'}"
            + (f" (partition starts at LBA {_fs_part_start_gb:.0f} GB on device)"
                if _fs_part_start_gb > 0 else "")
            + "<br>"
        )

    _ssd_chart_caption = (
        _device_geom_line +
        _churn_line +
        "Bars (log scale) show <b>total bytes written or read per ~120 equal-width "
        "LBA bins during the collection window</b>. This is cumulative I/O traffic, "
        "not current file allocation: a workload that creates and overwrites or "
        "deletes KV blocks will rack up many TB of writes even if df shows only a "
        "few hundred GB of files at the end. "
        "Diamonds mark the 30 hottest 16 MB buckets per op. "
        "Dotted line = device capacity. The dashed line at 'df' marks the "
        "filesystem byte total at analyze time — it's a different quantity than "
        "the bars: <b>df = files that currently exist</b>, <b>bars = bytes written "
        "during the trace</b>. They can disagree in either direction. "
        "If the bars sum to MORE than df, the workload was churning (writing then "
        "deleting/overwriting). If the bars sum to LESS, pre-existing files weren't "
        "re-touched during this window."
    )
    if _lba_offset_warn:
        _ssd_chart_caption = (
            f"<div style='background:#fef3c7;border-left:4px solid #f59e0b;"
            f"padding:8px 10px;margin-bottom:8px;border-radius:4px'>"
            f"⚠ <strong>LBA addressing mismatch:</strong> {_lba_offset_warn} "
            f"Subtract {_fs_part_start_gb:.0f} GB from hot-LBA values to get "
            f"file-offset within the FS.</div>"
        ) + _ssd_chart_caption
    if _ba_cov_warning:
        _ssd_chart_caption = (
            f"⚠ <strong>Coverage warning:</strong> blktrace captured only "
            f"{_ba_capture_pct:.0f}% of the bytes the kernel's /sys/block counter "
            f"observed during this trace ({_ba_kernel_gb:.0f} GB at the kernel "
            f"vs the bar totals below). The shape below is real but the "
            f"magnitudes are under-counted. See the top finding in the "
            f"executive summary for the cause.<br><br>"
        ) + _ssd_chart_caption
    _ssd_chart_caption = (
        "<strong>Consistency rule:</strong> this chart is physical local-block telemetry from blktrace/blkparse "
        "completion events only. SGLang backuped/prefetched counters are logical L3 KV movement; load_back is L2→L1 restore diagnostic-only; "
        "Executive and End Report reconcile them separately, so this LBA distribution must not be used to "
        "override SGLang logical directionality when blktrace is partial/missing.<br><br>"
    ) + _ssd_chart_caption
    _lba_section_title = ("L3 SSD I/O Distribution Across LBA Space"
                          if (_l3_res is not None and getattr(_l3_res, "backend_class", "") == "local_ssd")
                          else "SSD I/O Distribution Across LBA Space")
    sections_html.append(_section(_lba_section_title,
                                   "ch_hot",
                                   _chart_hot_regions(hot_regions,
                                                       lba_dist=lba_dist,
                                                       ssd_capacity_gb=_ssd_cap_gb,
                                                       ssd_used_gb=_ssd_used_gb,
                                                       fs_partition_start_gb=_fs_part_start_gb),
                                   _ssd_chart_caption))

    sections_html.append(_layer_header("layer-cross", "Cross-layer Diagnosis", "Use these charts together to connect A5 request symptoms back to A4 runtime behavior and A1–A3 memory/storage tiers.", "#0f172a"))

    # ─── Render full HTML ────────────────────────────────────────────────────
    # Cross-layer correlation and setup-aware launch guidance.
    _corr_sections = []
    try:
        _corr_sections.append("""
<section class="card">
  <h2>🔗 Cross-layer correlation — why utilization can be low while latency is high</h2>
  <p>
    Based on the SGLang launch command and setup details, this section explains why GPU/HBM/DRAM/L3
    utilization may be below 100% while TTFT/TPOT are high. Low utilization can mean the pipeline is
    waiting on prefill, scheduling, cache movement, request concurrency, or tier transitions rather than
    saturating a hardware counter.
  </p>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr><td style="font-weight:700;width:190px;padding:7px;border-bottom:1px solid #e2e8f0">Latency</td><td style="padding:7px;border-bottom:1px solid #e2e8f0">Use the KPI tiles above for canonical TTFT, TPOT/ITL, E2E, and throughput. These are the user-visible symptoms.</td></tr>
    <tr><td style="font-weight:700;padding:7px;border-bottom:1px solid #e2e8f0">Token movement</td><td style="padding:7px;border-bottom:1px solid #e2e8f0">Eviction/backup pressure explains L3 writes and cache churn; prefetch plus L2→L1 load_back restore pressure matters when it aligns with latency spikes. load_back remains a diagnostic upper bound, not physical SSD bytes.</td></tr>
    <tr><td style="font-weight:700;padding:7px;border-bottom:1px solid #e2e8f0">Compute and GPU</td><td style="padding:7px;border-bottom:1px solid #e2e8f0">GPU utilization can be low when prefill, scheduling, cache movement, or low concurrency leaves gaps between kernels. This is pipeline starvation, not necessarily low workload intensity.</td></tr>
    <tr><td style="font-weight:700;padding:7px;border-bottom:1px solid #e2e8f0">HBM</td><td style="padding:7px;border-bottom:1px solid #e2e8f0">HBM used is residency/capacity, not a target that must reach 100%. Headroom may be reserved for activations, CUDA graphs, and safety margin.</td></tr>
    <tr><td style="font-weight:700;padding:7px;border-bottom:1px solid #e2e8f0">DRAM/L3</td><td style="padding:7px;border-bottom:1px solid #e2e8f0">Low DRAM BW indicates L2 headroom. L3 advisory QD/busy-time is busy-time/pressure and must be correlated with latency, bandwidth, and exact Q→C evidence before calling L3 saturation.</td></tr>
  </table>
</section>
<section class="card">
  <h2>🚀 Setup-aware SGLang launch tuning</h2>
  <p>Recommendations are based on the setup_details launch command and observed report metrics.</p>
  <ul style="margin-left:18px;color:#334155">
    <li><b>HBM KV pool:</b> if HBM headroom exists, A/B test higher <code>--mem-fraction-static</code> values such as 0.85, then 0.88/0.90 only if SGLang logs show safe available GPU memory.</li>
    <li><b>FP8 KV:</b> keep or enable <code>--kv-cache-dtype fp8_e5m2</code> to reduce KV bytes/token and L3 spill pressure.</li>
    <li><b>Chunked prefill:</b> keep or enable <code>--chunked-prefill-size 4096</code>; A/B 2048/4096/8192 for TTFT vs throughput.</li>
    <li><b>L2 DRAM:</b> when DRAM BW is low, test increasing <code>--hicache-size</code> before assuming SSD hardware is the bottleneck.</li>
    <li><b>L3 local storage:</b> use an explicit <code>--file-storage-path</code> on the NVMe mount and direct/page_first_direct for cleaner L3 correlation.</li>
  </ul>
</section>
""")
    except Exception:
        _corr_sections = []


    head_title = f"AMOprof Report — {run_label}" if run_label else "AMOprof Report"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{head_title}</title>
  <script src="{PLOTLY_CDN}" charset="utf-8"></script>
  <style>
    *,*::before,*::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px;
      font-family: Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      color: #0f172a;
      background: #e2e8f0;
      line-height: 1.55;
    }}
    header {{
      max-width: 1400px; margin: 0 auto 24px auto;
      padding: 24px 28px;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f1f5f9;
      border-radius: 14px;
      box-shadow: 0 4px 12px rgba(15,23,42,0.18);
    }}
    header h1 {{ margin: 0 0 6px 0; font-size: 24px; letter-spacing: -0.3px; }}
    header .sub {{ color: #94a3b8; font-size: 13px; }}
    .kpi-grid {{
      max-width: 1400px; margin: 0 auto 24px auto;
      display: grid; gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .kpi-tile {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.06);
      transition: transform 0.12s, box-shadow 0.12s;
    }}
    .kpi-tile:hover {{
      transform: translateY(-1px);
      box-shadow: 0 3px 10px rgba(15,23,42,0.12);
    }}
    .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
                  color: #64748b; margin-bottom: 4px; font-weight: 600; }}
    .kpi-value {{ font-size: 22px; font-weight: 700; color: #0f172a; }}
    .kpi-unit  {{ font-size: 13px; font-weight: 500; color: #475569; margin-left: 4px; }}
    .kpi-note  {{ font-size: 11px; color: #64748b; margin-top: 3px; }}
    section.card {{
      max-width: 1400px; margin: 0 auto 18px auto;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 18px 22px 22px 22px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }}
    section.card h2 {{
      margin: 0 0 12px 0;
      font-size: 16px;
      color: #0f172a;
      letter-spacing: -0.2px;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 8px;
    }}
    .plot {{ width: 100%; min-height: 360px; }}
    .caption {{ margin-top: 10px; font-size: 12px; color: #64748b; font-style: italic; }}
    .no-data {{
      padding: 32px; text-align: center; color: #64748b;
      background: #f1f5f9; border-radius: 8px; font-size: 14px;
    }}

    /* Collapsed-empty section styling. The whole <section.card.empty-card>
       collapses to a single muted row when the section has no data, so
       long-tail "requires blktrace" / "Prom-only" sections don't push
       interesting content off-screen. Click anywhere on the summary row
       to expand and read the explanation. */
    section.card.empty-card {{
      padding: 0;
      background: #f8fafc;
      border-color: #e2e8f0;
    }}
    .empty-details {{ margin: 0; }}
    .empty-summary {{
      display: flex; align-items: center; gap: 10px;
      padding: 10px 16px;
      cursor: pointer; user-select: none;
      list-style: none;  /* hide default disclosure triangle on most browsers */
      font-size: 13px; color: #64748b;
      border-radius: 8px;
      transition: background 80ms ease-in-out;
    }}
    .empty-summary::-webkit-details-marker {{ display: none; }}
    .empty-summary:hover {{ background: #f1f5f9; }}
    .empty-details[open] .empty-summary {{
      border-bottom: 1px solid #e2e8f0;
      border-radius: 8px 8px 0 0;
    }}
    .empty-icon {{
      font-size: 16px; color: #94a3b8;
      transition: transform 120ms ease-in-out;
    }}
    .empty-details[open] .empty-icon {{ transform: rotate(45deg); }}
    .empty-title {{
      flex: 1; font-weight: 600; color: #475569;
    }}
    .empty-hint {{
      font-size: 11px; color: #94a3b8; font-style: italic;
    }}
    .empty-details[open] .empty-hint {{ display: none; }}
    .empty-details .no-data {{
      margin: 0; border-radius: 0 0 8px 8px;
      background: #f8fafc;
    }}
    footer {{
      max-width: 1400px; margin: 24px auto 0 auto;
      text-align: center; color: #64748b; font-size: 12px;
      padding: 16px;
    }}
    /* Hover hint banner */
    .hint {{
      background: #fef3c7; border-left: 4px solid #f59e0b;
      color: #78350f; padding: 10px 14px; border-radius: 6px;
      margin: 0 auto 18px auto; max-width: 1400px; font-size: 13px;
    }}

    .setup-panel {{
      background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%) !important;
      border: 1px solid #a5b4fc !important;
      border-left: 6px solid #4f46e5 !important;
      color: #0f172a !important;
      box-shadow: 0 4px 14px rgba(79,70,229,0.12) !important;
    }}
    .setup-panel h2 {{ color: #1e1b4b !important; }}
    .setup-panel-collapsible summary {{ cursor:pointer; list-style:none; font-size:18px; font-weight:800; color:#1e1b4b !important; display:flex; align-items:center; justify-content:space-between; gap:12px; }}
    .setup-panel-collapsible summary::-webkit-details-marker {{ display:none; }}
    .setup-panel-collapsible summary:before {{ content:'▶'; font-size:12px; color:#4f46e5; margin-right:4px; }}
    .setup-panel-collapsible[open] summary:before {{ content:'▼'; }}
    .setup-panel-collapsible:not([open]) .setup-grid {{ display:none; }}
    .setup-summary-hint {{ font-size:11px; color:#64748b; font-weight:600; }}
    .setup-grid {{ display:grid; grid-template-columns: 1.35fr .85fr; gap:16px; align-items:start; }}
    .setup-table {{ width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border-radius:10px; border:1px solid #c7d2fe; background:#ffffff; }}
    .setup-table tr:nth-child(even) td {{ background:#f8fafc; }}
    .setup-table tr:nth-child(odd) td {{ background:#ffffff; }}
    .setup-table tr:hover td {{ background:#dbeafe !important; }}
    .setup-key {{ padding:9px 12px; font-weight:800; color:#1e293b !important; width:34%; border-bottom:1px solid #e2e8f0; vertical-align:top; }}
    .setup-value {{ padding:9px 12px; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; color:#0f172a !important; border-bottom:1px solid #e2e8f0; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }}
    .setup-row-defaulted .setup-key, .setup-row-defaulted .setup-value {{ background:#fff7ed !important; color:#7c2d12 !important; }}
    .setup-info-box {{ background:#ffffff !important; border:1px solid #c7d2fe; border-radius:10px; padding:14px 16px; font-size:12px; color:#334155 !important; line-height:1.65; }}
    .setup-info-box p {{ margin:0 0 9px 0; }}
    .setup-info-box code {{ background:#eef2ff !important; color:#3730a3 !important; padding:1px 4px; border-radius:4px; }}
    .setup-info-title {{ font-size:12px; font-weight:800; color:#312e81; text-transform:uppercase; letter-spacing:.5px; margin-bottom:8px; }}
    .stack-cake {{ display:grid; gap:8px; margin-top:8px; }}
    .stack-row {{ display:grid; gap:8px; }}
    .stack-row.full {{ grid-template-columns:1fr; }}
    .stack-row.pair {{ grid-template-columns:1.6fr 1fr; }}
    .stack-row.trio {{ grid-template-columns:1fr 1fr 1fr; }}
    .stack-row > div, .stack-row.full {{ border-radius:10px; padding:14px 16px; color:#fff; }}
    .stack-row b {{ display:block; font-size:15px; margin-bottom:4px; }}
    .stack-row span {{ font-size:11px; opacity:.92; }}
    .a5 {{ background:#24577f; }} .a4 {{ background:#317bb8; }} .model {{ background:#b89a05; }}
    .l1 {{ background:#2f8a4e; }} .l2 {{ background:#7a3f97; }} .l3 {{ background:#a45714; }}
    @media(max-width:768px) {{ .setup-grid,.stack-row.pair,.stack-row.trio {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>📊 AMOprof — AI Memory &amp; IO Profiler · Interactive Report</h1>
    <div class="sub">
      Model: {_setup_value('Model', 'Model path')}
      &nbsp;·&nbsp; Duration: {sglang_sum.get('collection_elapsed_s', 0):.0f} sec
      &nbsp;·&nbsp; Source: {sglang_sum.get('prometheus_source','offline')}
    </div>
  </header>

  <div class="hint">
    💡 <b>Tip:</b> Hover any chart for exact values. Click a legend entry to hide/show traces.
    Drag to zoom; double-click to reset.
  </div>

  <div class="kpi-grid">
    {''.join(kpis)}
  </div>

  {''.join(_corr_sections)}
  {''.join(sections_html)}

  <footer>
    Generated by AMOprof · static charts available via <code>--report</code> · interactive via <code>--interactive</code>
  </footer>
</body>
</html>"""
    try:
        if compute_common_kpis is not None and apply_common_kpis_to_html is not None:
            _common_kpis = compute_common_kpis(raw_dir)
            if write_common_kpis_json is not None:
                write_common_kpis_json(raw_dir, _common_kpis)
            html = apply_common_kpis_to_html(html, raw_dir, _common_kpis)
    except Exception:
        pass
    html = _amoprof_storage_label_cleanup(
        html, getattr(_l3_res, "backend_class", "") if _l3_res is not None else "")
    out_html.write_text(html, encoding="utf-8")
    return out_html
