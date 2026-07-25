"""
amoprof/aggregator.py — Aggregate multiple local run directories into a
single merged raw/ directory for a given Prometheus time window.

Merging strategy per file type
───────────────────────────────
Time-series CSVs (sglang_timeseries, gpu_timeseries, nvme_driver_timeseries,
vmstat_timeseries, power_timeseries, amduprof_pcm_timeseries):
  • Row-union:  rows from all run dirs are concatenated and re-sorted by
    time_sec.  Rows whose time_sec falls outside [start, end] are dropped.
  • Duplicate timestamps: the row from the run dir that has more non-zero
    columns wins (prefer denser data).

Histogram/distribution CSVs (request_size_distribution,
interarrival_distribution, temporal_read_write_trim_pattern,
bandwidth_per_stream, hot_regions_overall):
  • Column-sum: numeric cells are summed across all run dirs.  String keys
    (bucket labels, pid/comm) are kept from the first file that has them.

Summary JSONs (sglang_summary, gpu_summary, smart_summary, summary):
  • Best-of merge: numeric values are averaged (for rates/latencies) or
    summed (for totals).  The merge heuristic is keyed on suffix:
      *_total / *_count / *_ios → sum
      everything else → mean (or max for peak_* / max_* keys)

setup_details.json:
  • First non-empty file wins.

Per-event CSVs (blkparse_events.generated.csv, biosnoop_events.csv,
biosnoop_events_all.csv):
  • Row-union keyed on `ts` (no de-dup — many events legitimately share a
    timestamp). See _merge_event_csvs.

Other files (*.txt, *.raw):
  • First non-empty file wins.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger("amoprof.aggregator")

# ─── CSV helpers ─────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, encoding="utf-8", newline="", errors="replace") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        log.debug("_read_csv %s: %s", path, e)
        return []


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Ensure all keys from all rows are present
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _safe_float(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _parse_event_ts(value: Any, t0_epoch: float = 0.0) -> float:
    """Convert an event timestamp to a sortable float.

    Supports:
      * blkparse-style relative seconds (float)
      * biosnoop-style clock strings ("HH:MM:SS.mmm")
    When t0_epoch is known, relative timestamps are made absolute and
    clock strings are resolved to absolute Unix epochs.
    """
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0

    # Numeric relative seconds (blkparse)
    try:
        f = float(s)
        if math.isfinite(f):
            if t0_epoch > 1_000_000_000 and abs(f) < 1_000_000_000:
                return f + t0_epoch
            return f
    except (TypeError, ValueError):
        pass

    # Clock string (biosnoop). Anchor to the date of t0_epoch when available.
    base = None
    if t0_epoch > 1_000_000_000:
        try:
            base = datetime.fromtimestamp(t0_epoch, tz=timezone.utc)
        except Exception:
            pass

    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt).time()
            if base is not None:
                dt = datetime.combine(base.date(), t, tzinfo=timezone.utc)
                return dt.timestamp()
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        except ValueError:
            continue
    return 0.0


def _merge_event_csvs(
    all_rows: list[list[dict]],
    t0_epochs: list[float],
    t_start: float,
    t_end: float,
    time_col: str = "ts",
) -> list[dict]:
    """Row-union event CSVs (blkparse, biosnoop) and re-normalise timestamps.

    Unlike regular time-series, event logs must not be de-duplicated by
    timestamp — many events can legitimately share the same ts.  Rows are
    filtered to [t_start, t_end] and renormalised to elapsed seconds from the
    earliest retained source epoch.
    """
    base_ts = 0.0
    if t_start > 1_000_000_000:
        base_ts = t_start
    else:
        base_ts = min((t for t in t0_epochs if t > 1_000_000_000), default=0.0)

    combined: list[dict] = []
    for src_idx, rows in enumerate(all_rows):
        t0 = t0_epochs[src_idx] if src_idx < len(t0_epochs) else 0.0
        for row in rows:
            ts_abs = _parse_event_ts(row.get(time_col), t0)
            if t_start > 0 and ts_abs < t_start:
                continue
            if t_end > 0 and ts_abs > t_end:
                continue
            new_row = dict(row)
            if base_ts > 0 and ts_abs > 1_000_000_000:
                new_row[time_col] = f"{ts_abs - base_ts:.6f}"
            combined.append(new_row)

    combined.sort(key=lambda r: _safe_float(_parse_event_ts(r.get(time_col), 0.0)))
    return combined


# ─── Time-series union ───────────────────────────────────────────────────────

def _merge_timeseries(
    all_rows: list[list[dict]],
    t_start: float,
    t_end: float,
    time_col: str = "time_sec",
    t0_epochs: list[float] | None = None,
) -> list[dict]:
    """
    Concatenate rows from multiple sources, filter to [t_start, t_end],
    sort by time_sec, and deduplicate by keeping the row with the most
    non-zero numeric columns per timestamp.

    When t0_epochs is provided (one Unix epoch per source), each source's
    elapsed time_sec is converted to an absolute timestamp before merging.
    This prevents rows from different runs being silently dropped when they
    share the same relative timestamps (e.g. both start at t=0, t=30, ...).
    After merging, timestamps are re-normalised relative to the earliest row.
    """
    combined: dict[str, dict] = {}  # ts_str → best row

    def _nonzero_count(row: dict) -> int:
        return sum(1 for v in row.values() if _is_numeric(str(v)) and _safe_float(str(v)) != 0)

    for src_idx, source_rows in enumerate(all_rows):
        # Offset: if t0_epochs provided, convert relative time_sec → absolute epoch
        t0_off = (t0_epochs[src_idx]
                  if t0_epochs and src_idx < len(t0_epochs)
                  and t0_epochs[src_idx] > 1_000_000_000
                  else 0.0)
        for row in source_rows:
            ts = _safe_float(row.get(time_col, ""))
            abs_ts = ts + t0_off if t0_off > 0 else ts

            # Filter to window.  If a per-source t0_epoch is known, compare
            # using the reconstructed absolute epoch.  The old mixed condition
            # (abs_ts > end AND relative_ts > end) accidentally kept rows from
            # later service cycles because their relative time_sec was small.
            compare_ts = abs_ts if t0_off > 0 or abs_ts > 1_000_000_000 else ts
            if t_start > 0 and compare_ts < t_start:
                continue
            if t_end > 0 and compare_ts > t_end:
                continue

            # Key on absolute ts so different-origin rows don't collide
            key = f"{abs_ts:.3f}"
            # Keep the row, storing absolute ts for later re-normalisation
            candidate = dict(row)
            if t0_off > 0:
                candidate[time_col] = str(abs_ts)
            if key not in combined or _nonzero_count(candidate) > _nonzero_count(combined[key]):
                combined[key] = candidate

    if not combined:
        return []

    result = sorted(combined.values(),
                    key=lambda r: _safe_float(r.get(time_col, "0")))

    # Re-normalise: subtract the earliest timestamp so elapsed starts at 0
    first_ts = _safe_float(result[0].get(time_col, "0"))
    if first_ts > 1_000_000_000:
        # Absolute epochs — normalise to elapsed
        for row in result:
            row[time_col] = str(round(_safe_float(row.get(time_col, "0")) - first_ts, 3))
    return result


def _normalize_time_sec(
    rows: list[dict],
    t0_epoch: float,
    time_col: str = "time_sec",
) -> list[dict]:
    """
    Ensure time_sec is elapsed seconds since t0_epoch.
    If values look like raw Unix timestamps (> 1e9), subtract t0_epoch.
    Also clamps any negative elapsed values to 0.
    """
    if not rows:
        return rows
    first = _safe_float(rows[0].get(time_col, "0"))
    if first > 1_000_000_000 and t0_epoch > 0:
        for row in rows:
            elapsed = _safe_float(row.get(time_col, "0")) - t0_epoch
            row[time_col] = str(round(max(elapsed, 0.0), 3))
    return rows


# ─── Histogram / distribution sum ───────────────────────────────────────────

def _merge_histograms(
    all_rows: list[list[dict]],
    key_col: str,
) -> list[dict]:
    """Sum numeric columns across all sources, keyed on key_col."""
    acc: dict[str, dict] = {}
    key_order: list[str] = []

    for source_rows in all_rows:
        for row in source_rows:
            k = str(row.get(key_col, ""))
            if k not in acc:
                acc[k] = {c: v for c, v in row.items()}
                key_order.append(k)
            else:
                for col, val in row.items():
                    if col == key_col:
                        continue
                    if _is_numeric(str(val)) and _is_numeric(str(acc[k].get(col, "0"))):
                        acc[k][col] = str(
                            round(_safe_float(acc[k][col]) + _safe_float(val), 6))
    return [acc[k] for k in key_order if k in acc]


# ─── Summary JSON merge ──────────────────────────────────────────────────────

def _merge_summaries(summaries: list[dict]) -> dict:
    """
    Best-of merge for scalar summary JSONs.

    Heuristic per key:
      key ends with _total / _count / _ios / _bytes / _gb → sum
      key starts with peak_ / max_ / contains _max_ → max
      everything else (rates, latencies, percentages) → mean of non-zero values
    """
    if not summaries:
        return {}

    all_keys: list[str] = []
    seen: set[str] = set()
    for s in summaries:
        for k in s:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    result: dict = {}
    for key in all_keys:
        vals = [s[key] for s in summaries if key in s and s[key] is not None]
        if not vals:
            continue

        # Non-numeric: keep first non-empty non-None
        numeric_vals = [_safe_float(v) for v in vals
                        if v is not None and _is_numeric(str(v))]
        if not numeric_vals:
            # Keep first non-empty, non-None value
            non_null = [v for v in vals if v is not None and str(v) not in ("", "null")]
            result[key] = non_null[0] if non_null else None
            continue

        kl = key.lower()
        if any(kl.endswith(s) for s in ("_total", "_count", "_ios",
                                          "_bytes", "_gb", "_tb",
                                          "_reads", "_writes", "_trims")):
            result[key] = round(sum(numeric_vals), 4)
        elif kl.startswith("peak_") or kl.startswith("max_") or "_max_" in kl:
            result[key] = round(max(numeric_vals), 4)
        else:
            nz = [v for v in numeric_vals if v != 0]
            result[key] = round(sum(nz) / len(nz), 4) if nz else 0

    return result


# ─── Main aggregation function ───────────────────────────────────────────────

def aggregate_run_dirs(
    run_dirs: list[Path],
    output_dir: Path,
    t_start: float = 0.0,
    t_end: float = 0.0,
    run_label: str = "",
) -> Path:
    """Merge multiple run directories into a single synthetic raw/ directory.

    Args:
        run_dirs:   List of run directories (each must contain a raw/ sub-dir).
        output_dir: Parent directory for the output merged_run_<ts>/ directory.
        t_start:    Prometheus window start as Unix epoch (0 = no filter).
        t_end:      Prometheus window end as Unix epoch (0 = no filter).
        run_label:  Optional label for the merged run.

    Returns:
        Path to the newly created merged run directory.
    """
    ts_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")
    label    = run_label or f"merged_{ts_tag}"
    out_run  = output_dir / label
    out_raw  = out_run / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    t0_epoch = t_start if t_start > 1_000_000_000 else 0.0
    n        = len(run_dirs)
    log.info("Aggregating %d run directories → %s", n, out_run)
    for i, rd in enumerate(run_dirs):
        log.info("  [%d/%d] %s", i + 1, n, rd)

    raw_dirs = []
    for rd in run_dirs:
        raw_candidate = rd / "raw"
        raw_dirs.append(raw_candidate if raw_candidate.is_dir() else rd)

    # Extract per-source t0_epoch values so _merge_timeseries can convert
    # relative time_sec → absolute before deduplication.  This prevents rows
    # from two runs that both start at t=0 from colliding and being dropped.
    #
    # summary.json location depends on how the run was created:
    #   live collect:  run_dir/summary.json  (parent of raw_dir)
    #   analyze/prom:  raw_dir/summary.json
    # We check both, preferring the parent-level file.
    t0_epochs: list[float] = []
    for rd in raw_dirs:
        t0 = 0.0
        # Check parent directory first (live collect path)
        for candidate_dir in (rd.parent, rd):
            meta = _read_json(candidate_dir / "summary.json")
            # summary.json may be {"meta": {"t0_epoch": ...}} or flat {"t0_epoch": ...}
            t0_cand = _safe_float(meta.get("t0_epoch") or meta.get("start_time") or
                                   (meta.get("meta") or {}).get("t0_epoch") or 0)
            if t0_cand > 1_000_000_000:
                t0 = t0_cand
                break
        t0_epochs.append(t0)
        if t0 > 0:
            log.debug("Source %s: t0_epoch=%.0f", rd, t0)
        else:
            log.warning("Source %s: t0_epoch not found in summary.json — "
                        "timeseries rows from different runs may collide during merge. "
                        "Ensure summary.json exists in either %s or %s.",
                        rd, rd, rd.parent)

    # ── Time-series CSVs ─────────────────────────────────────────────────────
    TIMESERIES_FILES = [
        "sglang_timeseries.csv",
        "gpu_timeseries.csv",
        "nvme_driver_timeseries.csv",
        "nvme_driver_timeseries_extra.csv",
        "vmstat_timeseries.csv",
        "power_timeseries.csv",
        "dram_timeseries.csv",
        "amduprof_pcm_timeseries.csv",
        "pcm_timeseries.csv",
        "iostat_timeseries.csv",
        "all_timeseries.csv",
        "queue_depth_sources_timeseries.csv",
    ]
    for fname in TIMESERIES_FILES:
        all_rows = [_read_csv(rd / fname) for rd in raw_dirs]
        merged   = _merge_timeseries(all_rows, t_start, t_end, t0_epochs=t0_epochs)
        merged   = _normalize_time_sec(merged, t0_epoch)
        if merged:
            _write_csv(merged, out_raw / fname)
            log.info("  %-42s %6d rows (from %d sources)",
                     fname, len(merged),
                     sum(1 for r in all_rows if r))

    # ── Per-event CSVs (row-union by ts, no de-dup — see _merge_event_csvs) ──
    # blkparse_events.generated.csv is the actual filename written by the
    # blktrace collector (collectors_extra.py); biosnoop_events.csv/
    # biosnoop_events_all.csv are the actual biosnoop outputs. Merging these
    # correctly (instead of dropping them) is what lets _run_amoprof() detect
    # blkparse_events.generated.csv in the merged raw/ and automatically
    # re-run blktrace_analyzer to produce request_size_distribution.csv,
    # bandwidth_per_stream.csv, hot_regions_overall.csv,
    # temporal_read_write_trim_pattern.csv, etc. (cli.py:_run_amoprof).
    EVENT_FILES = {
        "blkparse_events.generated.csv": "ts",
        "biosnoop_events.csv": "ts",
        "biosnoop_events_all.csv": "ts",
    }
    for fname, time_col in EVENT_FILES.items():
        all_rows = [_read_csv(rd / fname) for rd in raw_dirs]
        merged   = _merge_event_csvs(all_rows, t0_epochs, t_start, t_end, time_col=time_col)
        if merged:
            _write_csv(merged, out_raw / fname)
            log.info("  %-42s %6d rows (event union from %d sources)",
                     fname, len(merged),
                     sum(1 for r in all_rows if r))

    # ── Histogram / distribution CSVs ────────────────────────────────────────
    HISTOGRAM_FILES: dict[str, str] = {
        "request_size_distribution.csv": "bucket",
        "interarrival_distribution.csv": "bucket",
        "temporal_read_write_trim_pattern.csv": "window_s",
        "bandwidth_per_stream.csv": "pid",
        "hot_regions_overall.csv": "lba_gb_start",
    }
    for fname, key_col in HISTOGRAM_FILES.items():
        # blktrace_analyzer.py writes these into a dedicated blktrace_analysis/
        # subdir (cli.py: "Write blktrace analysis to a dedicated subdir so it
        # never overwrites the collect-time summary.json"), not raw/ root.
        all_rows = [_read_csv(rd / "blktrace_analysis" / fname) for rd in raw_dirs]
        merged   = _merge_histograms(all_rows, key_col)
        if merged:
            _write_csv(merged, out_raw / "blktrace_analysis" / fname)
            log.info("  %-42s %6d rows (summed)", fname, len(merged))

    # ── Summary JSONs ─────────────────────────────────────────────────────────
    SUMMARY_FILES = [
        "sglang_summary.json",
        "gpu_summary.json",
        "smart_summary.json",
        "bpf_summary.json",
        "nvme_smart.json",
        "blktrace_summary.json",
        "pcm_summary.json",
        "iostat_summary.json",
        "dram_summary.json",
        "vmstat_summary.json",
        "biosnoop_summary.json",
    ]
    for fname in SUMMARY_FILES:
        summaries = [_read_json(rd / fname) for rd in raw_dirs]
        summaries = [s for s in summaries if s]
        if summaries:
            merged_j = _merge_summaries(summaries)
            (out_raw / fname).write_text(
                json.dumps(merged_j, indent=2), encoding="utf-8")
            log.info("  %-42s merged (%d sources)", fname, len(summaries))

    # ── summary.json (the run-level metadata) ────────────────────────────────
    meta_list = [_read_json(rd / "summary.json") for rd in raw_dirs]
    meta_list = [m for m in meta_list if m]
    meta_merged = _merge_summaries(meta_list)
    # Override timing fields with the explicit window.
    # Also write t0_epoch at the top level (not just inside "meta") so that
    # analysis tools that read {"t0_epoch": ...} directly can find it.
    resolved_t0 = t0_epoch if t0_epoch > 1_000_000_000 else min(
        (t for t in t0_epochs if t > 1_000_000_000), default=0.0)
    if resolved_t0 > 0:
        meta_merged["t0_epoch"]   = resolved_t0
        meta_merged["start_time"] = resolved_t0
        meta_merged["prom_start"] = resolved_t0
    if t0_epoch > 1_000_000_000:
        meta_merged["t0_epoch"]   = t0_epoch
        meta_merged["start_time"] = t0_epoch
        meta_merged["prom_start"] = t0_epoch
    if t_end > 0:
        meta_merged["end_time"]   = t_end
        meta_merged["prom_end"]   = t_end
    if t_start > 0 and t_end > 0:
        meta_merged["duration_s"] = round(t_end - t_start, 1)
        meta_merged["run_duration_s"] = meta_merged["duration_s"]
    meta_merged["merged_from"] = [str(rd) for rd in run_dirs]
    meta_merged["merged_at"]   = datetime.now(tz=timezone.utc).isoformat()
    meta_merged["n_sources"]   = n
    (out_raw / "summary.json").write_text(
        json.dumps(meta_merged, indent=2), encoding="utf-8")
    log.info("  %-42s written", "summary.json")

    # ── setup_details.json — first non-empty wins ────────────────────────────
    for fname in ("setup_details.json", "setup.json"):
        for rd in raw_dirs:
            j = _read_json(rd / fname)
            if j:
                (out_raw / "setup_details.json").write_text(
                    json.dumps(j, indent=2), encoding="utf-8")
                log.info("  %-42s from %s", "setup_details.json", rd)
                break

    # ── DRAM PMU raw files — first non-empty wins ────────────────────────────
    # amduprof_pcm_raw.csv/.txt: AMD uProf output. pcm_memory_raw.csv: Intel
    # PCM's actual output filename (PcmMemoryCollector in collectors.py).
    for fname in ("amduprof_pcm_raw.csv", "amduprof_pcm_raw.txt", "pcm_memory_raw.csv"):
        for rd in raw_dirs:
            src = rd / fname
            if src.exists() and src.stat().st_size > 0:
                shutil.copy2(src, out_raw / fname)
                log.info("  %-42s from %s", fname, rd)
                break

    # ── sglang_percentiles_timeseries.json ───────────────────────────────────
    for fname in ("sglang_percentiles_timeseries.json",):
        # Each file is a list of {ts, p50, p90, p99} dicts — union and sort
        all_pct: list[dict] = []
        for rd in raw_dirs:
            src = rd / fname
            if src.exists():
                try:
                    data = json.loads(src.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            ts = _safe_float(item.get("ts", 0))
                            if (t_start == 0 or ts >= t_start) and \
                               (t_end   == 0 or ts <= t_end):
                                if t0_epoch > 0 and ts > 1_000_000_000:
                                    item = dict(item)
                                    item["ts"] = round(ts - t0_epoch, 3)
                                all_pct.append(item)
                except Exception:
                    pass
        if all_pct:
            all_pct.sort(key=lambda d: _safe_float(d.get("ts", 0)))
            (out_raw / fname).write_text(
                json.dumps(all_pct, indent=2), encoding="utf-8")
            log.info("  %-42s %6d points (union)", fname, len(all_pct))

    log.info("Aggregation complete: %s", out_raw)
    return out_run
