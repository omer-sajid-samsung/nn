"""
blktrace_service.py - Periodic incremental blktrace parser for the AMOprof service.

BlktraceServicePoller runs as a background thread alongside an active
BlktraceCollector.  Every --blkparse-interval-s seconds it:

  1. Runs blkparse against the currently accumulated binary files.
  2. Writes ALL events from those files to a fresh CSV.
  3. Computes summary metrics for rows [cursor : total], i.e. only the
     new events since the last interval.
  4. Pushes those interval metrics into the supplied MetricsStore under
     "blktrace_interval".

When the parsed event count reaches max_csv_events the poller performs a
rotation:
  - The existing binary trace files are deleted (blktrace continues running
    and immediately starts writing new per-CPU relay files).
  - The CSV is cleared and the cursor is reset to zero.
  - The next interval parses only the freshly written binaries.

This means the service runs indefinitely without growing unbounded on disk.
"""

from __future__ import annotations

import csv as _csv
import logging
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("amoprof.blktrace_service")


# ---------------------------------------------------------------------------
# Metrics computation from parsed rows
# ---------------------------------------------------------------------------

def _summarise_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """
    Compute aggregate metrics from a list of blkparse CSV rows.

    Expected columns: ts, pid, action, rwbs, op, sector, nsectors,
                      size_bytes, comm, dev, cpu
    Returns a flat dict of numeric metrics suitable for Prometheus.
    """
    if not rows:
        return {"blktrace_interval_events": 0}

    read_sizes:    list[float] = []
    write_sizes:   list[float] = []
    discard_sizes: list[float] = []
    read_sectors:  list[int]   = []
    write_sectors: list[int]   = []

    for r in rows:
        op = r.get("op", "?")
        try:
            size_b = float(r.get("size_bytes", 0) or 0)
            nsec   = int(r.get("nsectors", 0) or 0)
        except (ValueError, TypeError):
            size_b = 0.0
            nsec   = 0
        if op == "R":
            read_sizes.append(size_b)
            read_sectors.append(nsec)
        elif op == "W":
            write_sizes.append(size_b)
            write_sectors.append(nsec)
        elif op == "D":
            discard_sizes.append(size_b)

    def _mb(sizes):      return round(sum(sizes) / (1024 ** 2), 3)
    def _mean_kb(sizes): return round(statistics.mean(sizes) / 1024, 2) if sizes else 0.0
    def _p99_kb(sizes):
        if not sizes:
            return 0.0
        xs = sorted(sizes)
        return round(xs[int(len(xs) * 0.99)] / 1024, 2)

    m: dict[str, Any] = {
        "blktrace_interval_events":            len(rows),
        "blktrace_interval_read_ops":          len(read_sizes),
        "blktrace_interval_write_ops":         len(write_sizes),
        "blktrace_interval_discard_ops":       len(discard_sizes),
        "blktrace_interval_read_mb":           _mb(read_sizes),
        "blktrace_interval_write_mb":          _mb(write_sizes),
        "blktrace_interval_discard_mb":        _mb(discard_sizes),
        "blktrace_interval_read_mean_io_kb":   _mean_kb(read_sizes),
        "blktrace_interval_write_mean_io_kb":  _mean_kb(write_sizes),
        "blktrace_interval_read_p99_io_kb":    _p99_kb(read_sizes),
        "blktrace_interval_write_p99_io_kb":   _p99_kb(write_sizes),
    }

    all_sectors = read_sectors + write_sectors
    if all_sectors:
        aligned = sum(1 for s in all_sectors if s % 8 == 0)
        m["blktrace_interval_4k_aligned_ratio"] = round(
            aligned / len(all_sectors), 4)
    else:
        m["blktrace_interval_4k_aligned_ratio"] = 0.0

    return m


# ---------------------------------------------------------------------------
# BlktraceServicePoller
# ---------------------------------------------------------------------------

class BlktraceServicePoller:
    """
    Background thread that periodically runs blkparse against live blktrace
    binary files and pushes interval metrics into a MetricsStore.

    Parameters
    ----------
    trace_prefix : Path
        Prefix passed to blkparse -i (e.g. .../blktrace_data/trace).
        Binary files are expected at <prefix>.blktrace.N.
    csv_path : Path
        Where to write/overwrite the parsed event CSV each cycle.
    store : Any
        MetricsStore instance — receives updates under "blktrace_interval".
    blkparse_bin : str
        blkparse executable name or path.
    interval_s : float
        How often (seconds) to re-run blkparse and refresh metrics.
    use_sudo : bool
        Whether to prefix blkparse with sudo -n.
    max_csv_events : int
        When the cumulative parsed event count reaches this threshold the
        poller rotates: binary files are pruned, CSV is cleared, cursor is
        reset to zero, and collection continues from fresh binaries.
    """

    _LINE_RE = re.compile(
        r"^\s*(?P<major>\d+),(?P<minor>\d+)\s+(?P<cpu>\d+)\s+\d+\s+"
        r"(?P<ts>[\d.]+)\s+(?P<pid>\d+)\s+"
        r"(?P<action>[A-Z])\s+(?P<rwbs>[A-Z]*)\s*"
        r"(?P<sector>\d+)\s+\+\s+(?P<nsec>\d+)"
    )
    _LINE_RE_OLD = re.compile(
        r"^\s*(?P<ts>[\d.]+)\s+(?P<pid>\d+)\s+"
        r"(?P<action>[A-Z])\s+(?P<rwbs>[A-Z]+)\s+"
        r"(?P<major>\d+),(?P<minor>\d+)\s+"
        r"(?P<sector>\d+)\s+\+\s+(?P<nsec>\d+)"
    )

    def __init__(
        self,
        trace_prefix: Path,
        csv_path: Path,
        store: Any,
        blkparse_bin: str = "blkparse",
        interval_s: float = 10.0,
        use_sudo: bool = True,
        max_csv_events: int = 5_000_000,
    ) -> None:
        self.trace_prefix   = Path(trace_prefix)
        self.csv_path       = Path(csv_path)
        self.store          = store
        self.blkparse_bin   = blkparse_bin
        self.interval_s     = interval_s
        self.use_sudo       = use_sudo
        self.max_csv_events = max_csv_events

        self._stop         = threading.Event()
        self._thread: threading.Thread | None = None
        # _last_row is the CSV row index of the last event already processed.
        # It resets to 0 after each rotation.
        self._last_row     = 0
        self._parse_count  = 0   # monotonically increasing across rotations
        self._rotation_count = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="amoprof-blktrace-poller",
            daemon=True,
        )
        self._thread.start()
        log.info("BlktraceServicePoller started (interval=%.0fs, cap=%d events)",
                 self.interval_s, self.max_csv_events)

    def stop(self) -> None:
        """Signal the poller to stop and wait for the final parse to finish."""
        self._stop.set()
        if self._thread:
            # Allow enough time for one last blkparse run
            self._thread.join(timeout=self.interval_s + 60)

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self.interval_s):
            try:
                self._parse_and_update()
            except Exception as exc:
                log.warning("BlktraceServicePoller: parse cycle error: %s", exc)
        # One final parse after stop() is called so the last interval's
        # data makes it into the store before the blktrace collector is stopped.
        try:
            self._parse_and_update()
        except Exception as exc:
            log.warning("BlktraceServicePoller: final parse error: %s", exc)

    def _parse_and_update(self) -> None:
        """
        Core work unit: run blkparse, compute interval delta, push to store.
        Triggers a rotation when the cumulative event count hits the cap.
        """
        trace_dir = self.trace_prefix.parent
        bins = sorted(trace_dir.glob("trace.blktrace.*"))
        if not bins:
            log.debug("BlktraceServicePoller: no binary files yet — skipping")
            return

        if not self._which(self.blkparse_bin):
            log.warning("BlktraceServicePoller: %s not found — skipping",
                        self.blkparse_bin)
            return

        total_events = self._run_blkparse(bins)
        if total_events < 0:
            return  # parse failed; keep old metrics

        # ── Check for rotation ───────────────────────────────────────────────
        # If total_events has reached or exceeded the cap it means the CSV
        # was truncated during this run.  Rotate before advancing the cursor
        # so the next interval starts clean.
        needs_rotation = (total_events >= self.max_csv_events)

        # Compute interval metrics from new rows only
        new_rows = self._read_csv_slice(self._last_row, total_events)
        interval_metrics = _summarise_rows(new_rows)
        interval_metrics["blktrace_interval_parse_number"]   = self._parse_count
        interval_metrics["blktrace_interval_rotation_count"] = self._rotation_count
        interval_metrics["blktrace_interval_cursor_start"]   = self._last_row
        interval_metrics["blktrace_interval_cursor_end"]     = total_events

        self.store.update("blktrace_interval", interval_metrics)

        log.info(
            "BlktraceServicePoller: parse #%d — %d total events, "
            "%d new this interval (rows %d→%d)%s",
            self._parse_count, total_events, len(new_rows),
            self._last_row, total_events,
            " [ROTATING]" if needs_rotation else "",
        )

        self._last_row = total_events
        self._parse_count += 1

        if needs_rotation:
            self._rotate(bins)

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _rotate(self, bins: list[Path]) -> None:
        """
        Prune processed binary files and reset the CSV + cursor.

        blktrace keeps running and writes new per-CPU relay files as soon as
        the old ones are removed.  We delete the binaries we have already
        fully parsed, truncate the CSV, and reset _last_row so the next
        interval accumulates from a clean slate.
        """
        self._rotation_count += 1
        log.info(
            "BlktraceServicePoller: rotation #%d — pruning %d binary file(s), "
            "resetting CSV cursor",
            self._rotation_count, len(bins),
        )

        # Delete binary trace files
        pruned = 0
        pruned_bytes = 0
        for p in bins:
            try:
                pruned_bytes += p.stat().st_size
                p.unlink()
                pruned += 1
            except Exception as exc:
                log.warning("BlktraceServicePoller: could not remove %s: %s", p, exc)

        # Clear the CSV so the next parse writes a fresh file
        try:
            self.csv_path.write_text("", encoding="utf-8")
        except Exception as exc:
            log.warning("BlktraceServicePoller: could not clear CSV: %s", exc)

        # Reset cursor — next _run_blkparse will start from row 0
        self._last_row = 0

        log.info(
            "BlktraceServicePoller: rotation complete — "
            "removed %d file(s) (%.1f MB), cursor reset",
            pruned, pruned_bytes / (1024 ** 2),
        )

    # ── blkparse runner ───────────────────────────────────────────────────────

    def _run_blkparse(self, bins: list[Path]) -> int:
        """
        Run blkparse against *bins*, write events to self.csv_path.

        Returns total event rows written (0..max_csv_events), or -1 on error.
        The CSV is always overwritten from scratch so _last_row is a stable
        row-index into it.
        """
        cmd = [self.blkparse_bin, "-i", str(self.trace_prefix)]
        if self.use_sudo and os.geteuid() != 0:
            cmd = ["sudo", "-n"] + cmd

        events    = 0
        unmatched = 0
        _fmt       = "A"
        _fmt_locked = False

        try:
            with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
            ) as proc, open(self.csv_path, "w", encoding="utf-8", newline="") as fp:

                w = _csv.writer(fp)
                w.writerow(["ts", "pid", "action", "rwbs", "op",
                             "sector", "nsectors", "size_bytes",
                             "comm", "dev", "cpu"])

                for raw in proc.stdout:
                    m = None
                    if _fmt == "A":
                        m = self._LINE_RE.match(raw)
                        if m is None and not _fmt_locked:
                            m2 = self._LINE_RE_OLD.match(raw)
                            if m2 and m2.group("sector"):
                                _fmt = "B"
                                _fmt_locked = True
                                m = m2
                    else:
                        m = self._LINE_RE_OLD.match(raw)

                    if not m:
                        unmatched += 1
                        continue

                    d    = m.groupdict()
                    rwbs = d.get("rwbs") or ""
                    cpu  = d.get("cpu") or "0"
                    op   = ("D" if "D" in rwbs
                            else "W" if "W" in rwbs
                            else "R" if "R" in rwbs
                            else "?")
                    nsec = int(d["nsec"])
                    try:
                        w.writerow([
                            d["ts"], d["pid"], d["action"], rwbs, op,
                            d["sector"], nsec, nsec * 512,
                            d.get("comm") or "",
                            f"{d['major']},{d['minor']}",
                            cpu,
                        ])
                        events += 1
                    except Exception:
                        pass

                    if events >= self.max_csv_events:
                        # Cap reached — drain stdout so blkparse can exit cleanly,
                        # then break.  Rotation will fire after this run returns.
                        log.debug("BlktraceServicePoller: cap=%d reached, draining",
                                  self.max_csv_events)
                        try:
                            proc.stdout.read()
                        except Exception:
                            pass
                        break

                # Wait for blkparse to finish; timeout scales with binary size
                _mb = sum(p.stat().st_size for p in bins
                          if p.exists()) / (1024 * 1024)
                _wait = max(30, min(int(_mb) + 10, 300))
                try:
                    proc.wait(timeout=_wait)
                except subprocess.TimeoutExpired:
                    log.warning("BlktraceServicePoller: blkparse timeout (%ds) — killing",
                                _wait)
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass

                rc = proc.returncode if proc.returncode is not None else 0
                if rc != 0:
                    err = ""
                    try:
                        err = (proc.stderr.read() or "").strip().splitlines()
                        err = err[-1] if err else f"rc={rc}"
                    except Exception:
                        err = f"rc={rc}"
                    log.error("BlktraceServicePoller: blkparse failed: %s", err[:200])
                    return -1

        except Exception as exc:
            log.warning("BlktraceServicePoller: blkparse error: %s", exc)
            return -1

        log.debug("BlktraceServicePoller: blkparse produced %d events "
                  "(%d unmatched lines)", events, unmatched)
        return events

    # ── CSV reader ────────────────────────────────────────────────────────────

    def _read_csv_slice(self, start_row: int, end_row: int) -> list[dict[str, str]]:
        """
        Read data rows [start_row, end_row) from self.csv_path.
        Row indices are 0-based and exclude the header.
        Skips to start_row without loading the entire file into memory.
        """
        if not self.csv_path.exists() or start_row >= end_row:
            return []
        rows: list[dict[str, str]] = []
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < start_row:
                        continue
                    if i >= end_row:
                        break
                    rows.append(row)
        except Exception as exc:
            log.warning("BlktraceServicePoller: CSV slice read error: %s", exc)
        return rows

    # ── Utility ───────────────────────────────────────────────────────────────

    def _which(self, name: str) -> bool:
        try:
            subprocess.check_output(
                ["which", name], stderr=subprocess.DEVNULL, timeout=3)
            return True
        except Exception:
            return False
