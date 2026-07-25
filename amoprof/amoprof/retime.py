"""
amoprof/retime.py — Re-derive the time_sec column in raw/*_timeseries.csv
files for run-dirs collected with an old writer that misaligned the SGLang
chart against the rest of the timeseries.

Background. AMOprof before v1.12 had a writer bug: SGLang's `time_sec=0`
anchored at the first SGLang scrape, while gpu/nvme/vmstat/power anchored
at the run-wide `t0` (set at collect start). The first SGLang scrape
typically arrives 2-15 sec after `t0` (Prometheus scrape interval), so the
same wall-clock moment landed at different X positions across charts.

This module rewrites `sglang_timeseries.csv` by adding the correct offset
to every `time_sec` value, in place. The original is backed up with a
`.preretime` suffix.

Inference paths (in order of preference):

  1. **Recorded values**. v1.12+ writers store:
       - `summary.json :: meta.t0_epoch`              (run-wide anchor)
       - `sglang_summary.json :: first_sample_epoch`  (SGLang's old anchor)
     Then offset = first_sample_epoch - t0_epoch.

  2. **Manual offset**. User passes `--sglang-offset-s N`. Useful for
     legacy run-dirs that lack the recorded fields.

  3. **Heuristic**. We compare the earliest non-zero timestamp in
     gpu_timeseries against SGLang's t=0. If the GPU collector ran for a
     short interval before SGLang, gpu_timeseries[0].time_sec is roughly
     the SGLang offset (e.g. GPU starts at t=0, SGLang's first scrape is
     12s later → SGLang t=0 should become t=12, offset=12).

Usage:
    amoprof retime --run-dir <existing_dir>
    amoprof retime --run-dir <dir> --sglang-offset-s 12
    amoprof retime --run-dir <dir> --dry-run     # show what would change
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger("amoprof.retime")


def _read_json(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _infer_offset_from_recorded(run_dir: Path) -> tuple[float | None, str]:
    """Look for t0_epoch + first_sample_epoch and compute offset.

    Returns (offset_sec, reason_string).
    """
    summary = _read_json(run_dir / "summary.json")
    sg_summary = _read_json(run_dir / "raw" / "sglang_summary.json")
    if not summary or not sg_summary:
        return None, ("missing summary.json or raw/sglang_summary.json")
    meta = summary.get("meta", {})
    t0 = float(meta.get("t0_epoch", 0) or 0)
    first_sg = float(sg_summary.get("first_sample_epoch", 0) or 0)
    if t0 <= 0:
        return None, "summary.json::meta.t0_epoch not recorded (legacy run-dir)"
    if first_sg <= 0:
        return None, "sglang_summary.json::first_sample_epoch not recorded"
    offset = first_sg - t0
    return offset, (
        f"recorded: t0_epoch={t0:.1f}, first_sample_epoch={first_sg:.1f}, "
        f"offset = first - t0 = {offset:+.3f}s"
    )


def _infer_offset_heuristic(run_dir: Path) -> tuple[float | None, str]:
    """Compare gpu_timeseries[0].time_sec vs sglang_timeseries[0].time_sec.

    In legacy run-dirs, SGLang starts at 0 and GPU starts at ~0. If the
    GPU collector ran-for-N-seconds before SGLang's first scrape, GPU's
    first row has time_sec close to 0 too, but the *N-th* GPU row matches
    SGLang's t=0. We can't recover this directly without the epochs.

    A weaker heuristic: if SGLang has fewer samples than GPU over the
    same total elapsed time, the difference suggests the offset. This
    is unreliable — surface it as informational only.
    """
    raw = run_dir / "raw"
    gpu_csv = raw / "gpu_timeseries.csv"
    sg_csv  = raw / "sglang_timeseries.csv"
    if not gpu_csv.exists() or not sg_csv.exists():
        return None, "missing gpu_timeseries.csv or sglang_timeseries.csv"
    try:
        with open(gpu_csv) as f:
            gpu_max = max((float(r["time_sec"]) for r in csv.DictReader(f)
                            if r.get("time_sec")), default=0.0)
        with open(sg_csv) as f:
            sg_max = max((float(r["time_sec"]) for r in csv.DictReader(f)
                            if r.get("time_sec")), default=0.0)
    except (KeyError, ValueError, OSError) as e:
        return None, f"failed to scan timeseries: {e}"
    if gpu_max <= 0 or sg_max <= 0:
        return None, "one of the timeseries appears empty"
    diff = gpu_max - sg_max
    if diff <= 0:
        return None, (f"heuristic inconclusive: gpu_max={gpu_max:.1f}s "
                        f"<= sglang_max={sg_max:.1f}s")
    return diff, (
        f"heuristic: gpu_max={gpu_max:.1f}s, sglang_max={sg_max:.1f}s, "
        f"offset estimate = {diff:+.1f}s (unreliable — verify manually)"
    )


def _rewrite_csv_with_offset(csv_path: Path, offset_s: float,
                               backup_suffix: str = ".preretime") -> tuple[int, int]:
    """Add `offset_s` to every `time_sec` value in csv_path, in place.

    Backs up the original to <csv_path><backup_suffix>. Returns
    (rows_written, rows_skipped_due_to_missing_column).
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    backup = csv_path.with_suffix(csv_path.suffix + backup_suffix)
    if not backup.exists():
        shutil.copyfile(csv_path, backup)
        log.info("backup → %s (%d bytes)", backup.name, backup.stat().st_size)

    with open(backup, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "time_sec" not in reader.fieldnames:
            raise ValueError(f"{csv_path.name} has no time_sec column")
        rows = list(reader)
        fields = list(reader.fieldnames)

    written = 0
    skipped = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            ts = row.get("time_sec", "")
            if ts == "" or ts is None:
                w.writerow(row); skipped += 1; continue
            try:
                row["time_sec"] = f"{float(ts) + offset_s:.3f}"
            except (ValueError, TypeError):
                skipped += 1
            else:
                written += 1
            w.writerow(row)
    return written, skipped


def retime_run_dir(run_dir: Path,
                    sglang_offset_s: float | None = None,
                    dry_run: bool = False,
                    use_heuristic: bool = False) -> int:
    """Public entry point. Returns 0 on success, non-zero on failure."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        log.error("%s is not a directory", run_dir)
        return 2
    raw = run_dir / "raw"
    if not raw.is_dir():
        log.error("%s/raw not found — not an AMOprof run dir", run_dir)
        return 2
    sg_csv = raw / "sglang_timeseries.csv"
    if not sg_csv.exists():
        log.error("%s not found — nothing to retime", sg_csv)
        return 2

    # 1. Decide on offset
    if sglang_offset_s is not None:
        offset = float(sglang_offset_s)
        log.info("Using manual offset: %+.3fs", offset)
    else:
        offset, why = _infer_offset_from_recorded(run_dir)
        if offset is None:
            log.warning("Cannot infer offset from recorded values: %s", why)
            if use_heuristic:
                offset, why = _infer_offset_heuristic(run_dir)
                if offset is None:
                    log.error("Heuristic also failed: %s", why)
                    log.error("Pass --sglang-offset-s <N> manually, or re-collect "
                              "with v1.12+ to get t0_epoch recorded for next time.")
                    return 3
                log.warning("Applying heuristic offset: %s", why)
            else:
                log.error("This run-dir was likely collected with a pre-v1.12 "
                          "writer that didn't record t0_epoch. Options:")
                log.error("  1. Pass --sglang-offset-s <seconds> manually")
                log.error("  2. Pass --use-heuristic to estimate from "
                          "gpu_timeseries.csv (unreliable — verify by eye)")
                log.error("  3. Re-collect with v1.12+ (recommended)")
                return 3
        else:
            log.info("Inferred offset: %s", why)

    if abs(offset) < 0.001:
        log.info("Offset is ~0 — sglang_timeseries.csv is already aligned. "
                 "No changes needed.")
        return 0

    # 2. Show what will change
    with open(sg_csv, encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    if not sample:
        log.error("sglang_timeseries.csv is empty")
        return 4
    head = sample[:3]
    tail = sample[-3:]
    log.info("Current sglang_timeseries.csv (head):")
    for r in head:
        log.info("  time_sec=%s", r.get("time_sec"))
    log.info("Current sglang_timeseries.csv (tail):")
    for r in tail:
        log.info("  time_sec=%s", r.get("time_sec"))
    log.info("After retime: every time_sec will be shifted by %+.3fs", offset)
    log.info("  e.g. head[0]: %s → %.3f",
              head[0].get("time_sec"),
              float(head[0].get("time_sec", 0)) + offset)
    log.info("  e.g. tail[-1]: %s → %.3f",
              tail[-1].get("time_sec"),
              float(tail[-1].get("time_sec", 0)) + offset)

    if dry_run:
        log.info("--dry-run: no files modified.")
        return 0

    # 3. Apply
    try:
        n_w, n_skip = _rewrite_csv_with_offset(sg_csv, offset)
        log.info("Rewrote %s: %d rows shifted, %d skipped",
                 sg_csv.name, n_w, n_skip)
    except Exception as e:
        log.error("retime failed: %s", e)
        return 5

    # 4. Update meta in summary.json so it doesn't get retimed again
    meta_path = run_dir / "summary.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            meta = data.setdefault("meta", {})
            applied = meta.setdefault("retime_applied", [])
            applied.append({
                "file": "raw/sglang_timeseries.csv",
                "offset_s": offset,
                "source": ("recorded" if sglang_offset_s is None
                            else "manual"),
            })
            meta_path.write_text(json.dumps(data, indent=2, default=str))
            log.info("Recorded retime in summary.json::meta.retime_applied")
        except Exception as e:
            log.warning("could not update summary.json: %s", e)

    log.info("✓ Retime complete. The SGLang chart in any regenerated report "
              "will now align with gpu/nvme/vmstat timelines.")
    log.info("  To regenerate the report:")
    log.info("    amoprof analyze --run-dir %s --interactive-report",
              run_dir)
    return 0
