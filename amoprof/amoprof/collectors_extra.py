"""
Additional per-request and event-level collectors for AMOprof.

Adds:
  - BlktraceCollector       per-request block I/O events (R/W/D) via blktrace/blkparse
  - BiosnoopCollector       per-I/O events with PID/process attribution via biosnoop-bpfcc
  - DiscardStatsMonitor     time-series of /proc/diskstats discard counters (TRIM)
  - SwapStormMonitor        time-series of /proc/vmstat swap pages (pswpin/pswpout)

These supply the data needed by:
  §C  KV$ L3 Read Workload Characteristics    (blktrace per-request)
  §D  KV$ L3 Write Workload Characteristics   (blktrace per-request)
  §E  TRIM / Discard Events                    (DiscardStatsMonitor + blktrace rwbs=D)
  §H  NVMe IO Deep Profiling                   (blktrace + biosnoop)
  §I  Per-Stream Bandwidth                     (biosnoop PID attribution)
  §G  Swap Storm Analysis                      (SwapStormMonitor time-series)
"""
from __future__ import annotations

import csv as _csv
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("amoprof.extra")


# ─────────────────────────────────────────────────────────────────────────────
#  BlktraceCollector
# ─────────────────────────────────────────────────────────────────────────────
class BlktraceCollector:
    """
    Capture per-request block I/O events with `blktrace`.

    Writes one binary file per CPU under `<work_dir>/blktrace_data/`:
        trace.blktrace.0
        trace.blktrace.1
        ...
    These can be analyzed offline at any time with:
        blkparse -i <work_dir>/blktrace_data/trace -O -d <work_dir>/blktrace.bin

    After collection stops, this collector also runs `blkparse` to generate
    `<work_dir>/blkparse_events.generated.csv` in the column layout amoprof
    expects:
        ts,pid,action,rwbs,sector,nsectors,size_bytes,comm,dev

    Why binary capture rather than live `blkparse` piping:
      • blktrace prefers writing its raw rings to disk; piping serializes them
        through one process and drops events when the inference workload spikes
      • The binaries are the canonical artifact — they can be re-parsed with
        different blkparse options later (e.g. -O to show all event types)
      • Failed CSV parsing in analyze does not lose the source data
    """

    def __init__(self, device: str, duration_s: int = 60,
                 work_dir: "Path | str" = ".",
                 blktrace_bin: str = "blktrace",
                 blkparse_bin: str = "blkparse",
                 use_sudo: bool = True,
                 parse_after: bool = True,
                 max_csv_events: int | None = None,
                 buffer_kb: int = 16384,
                 num_buffers: int = 4,
                 filename_suffix: str = ""):
        self._filename_suffix = filename_suffix or ""
        self.device = device
        self.duration_s = max(int(duration_s), 1)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.blktrace_bin = blktrace_bin
        self.blkparse_bin = blkparse_bin
        self.use_sudo = use_sudo
        self.parse_after = parse_after
        self.max_csv_events = (int(max_csv_events) if max_csv_events not in (None, "", 0, "0") else None)
        self.buffer_kb = max(int(buffer_kb or 0), 0)
        self.num_buffers = max(int(num_buffers or 0), 0)

        _sfx = ("_" + self._filename_suffix) if self._filename_suffix else ""
        self._trace_dir = self.work_dir / ("blktrace_data" + _sfx)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        # Use a fixed file prefix so blkparse later finds `trace.blktrace.N`.
        self._trace_prefix = self._trace_dir / "trace"
        self._csv_path = self.work_dir / ("blkparse_events.generated" + _sfx + ".csv")

        self._proc = None
        self._reason = ""
        self._events = 0
        self._t0 = 0.0
        self._launch_cmd: list[str] = []
        self._launch_rc: int | None = None
        self._launch_stderr: str = ""

    # ── blkparse output format detection ─────────────────────────────────────
    # FORMAT A — standard (default, all modern versions):
    #   MAJOR,MINOR  CPU  SEQNO  TS  PID  ACTION  RWBS  SECTOR + NSEC [COMM]
    # FORMAT B — legacy / "old" format (blkparse -O on some distros):
    #   TS  PID  ACTION  RWBS  MAJOR,MINOR  SECTOR + NSEC [COMM]
    # The -O flag meaning is ambiguous across versions; removing it guarantees
    # FORMAT A. The regex below handles both so that existing data still parses.
    _LINE_RE = re.compile(
        r"^\s*(?P<major>\d+),(?P<minor>\d+)\s+(?P<cpu>\d+)\s+\d+\s+"
        r"(?P<ts>[\d.]+)\s+(?P<pid>\d+)\s+"
        r"(?P<action>[A-Z])\s+(?P<rwbs>[A-Z]*)\s*"
        r"(?P<sector>\d+)\s+\+\s+(?P<nsec>\d+)"
        r"(?:\s+\[(?P<comm>[^\]]+)\])?"
    )
    _LINE_RE_OLD = re.compile(
        r"^\s*(?P<ts>[\d.]+)\s+(?P<pid>\d+)\s+"
        r"(?P<action>[A-Z])\s+(?P<rwbs>[A-Z]*)\s+"
        r"(?P<major>\d+),(?P<minor>\d+)\s+"
        r"(?P<sector>\d+)\s+\+\s+(?P<nsec>\d+)"
        r"(?:\s+\[(?P<comm>[^\]]+)\])?"
    )

    def _which(self, name: str) -> bool:
        try:
            subprocess.check_output(["which", name], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return Path(name).exists() and os.access(name, os.X_OK)

    @staticmethod
    def _stderr_is_ebusy(stderr_text: str) -> bool:
        t = (stderr_text or "").lower()
        return ("blktracesetup" in t and ("16/device" in t or "resource busy" in t or "device or resource busy" in t))

    def _kill_existing_trace(self) -> tuple[int, str]:
        cmd = [self.blktrace_bin, "-d", self.device, "-k"]
        if self.use_sudo and os.geteuid() != 0:
            cmd = ["sudo", "-n"] + cmd
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            msg = (r.stderr or r.stdout or "").strip()
            return r.returncode, f"{' '.join(cmd)} rc={r.returncode} {msg[:200]}"
        except subprocess.TimeoutExpired:
            return -1, f"{' '.join(cmd)} timeout"
        except Exception as e:
            return -2, f"{' '.join(cmd)} {e!r}"

    def _launch_blktrace_process(self, cmd: list[str]) -> subprocess.Popen:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def start(self):
        self._events = 0
        self._reason = ""
        # Snapshot baseline /sys/block/<dev>/stat counters. Used in stop() to
        # compute kernel-side ground-truth deltas during the trace window:
        #   - wr_sectors_delta → bytes written (cross-check vs blktrace bytes)
        #   - wr_merges_delta  → how many BIOs the block layer coalesced into
        #     larger I/Os before dispatching to the device. High merge counts
        #     mean blktrace's *captured ops* is lower than the *requests
        #     submitted* by the FS, which is healthy behaviour (XFS
        #     extent-based allocation triggers this much more than ext4).
        self._wr_sectors_start = 0
        self._rd_sectors_start = 0
        self._wr_merges_start  = 0
        self._rd_merges_start  = 0
        self._wr_ios_start     = 0
        self._rd_ios_start     = 0
        try:
            dev_base = re.sub(r"p\d+$", "", Path(self.device).name)
            stat_path = Path(f"/sys/block/{dev_base}/stat")
            if stat_path.exists():
                parts = stat_path.read_text().split()
                if len(parts) >= 7:
                    # /sys/block/<dev>/stat column layout:
                    #  0=rd_ios 1=rd_merges 2=rd_sectors 3=rd_ticks
                    #  4=wr_ios 5=wr_merges 6=wr_sectors 7=wr_ticks
                    self._rd_ios_start     = int(parts[0])
                    self._rd_merges_start  = int(parts[1])
                    self._rd_sectors_start = int(parts[2])
                    self._wr_ios_start     = int(parts[4])
                    self._wr_merges_start  = int(parts[5])
                    self._wr_sectors_start = int(parts[6])
        except Exception:
            pass
        if not self._which(self.blktrace_bin):
            self._reason = (f"{self.blktrace_bin} not found (apt: blktrace)")
            log.warning("BlktraceCollector: %s", self._reason)
            return
        if not Path(self.device).exists():
            self._reason = f"device not found: {self.device}"
            log.warning("BlktraceCollector: %s", self._reason)
            return

        # Binary capture mode: blktrace writes <prefix>.blktrace.N files per CPU.
        # No piping to blkparse here — we parse offline in stop() or analyze.
        cmd = [self.blktrace_bin,
               "-d", self.device,
               "-D", str(self._trace_dir),
               "-o", "trace",
               "-w", str(self.duration_s)]
        # High-I/O inference workloads can overflow default blktrace buffers.
        # Use larger buffers by default; user can override via CLI.
        if self.buffer_kb > 0:
            cmd.extend(["-b", str(self.buffer_kb)])
        if self.num_buffers > 0:
            cmd.extend(["-n", str(self.num_buffers)])
        if self.use_sudo and os.geteuid() != 0:
            cmd = ["sudo", "-n"] + cmd
        self._launch_cmd = list(cmd)
        try:
            (self.work_dir / "blktrace_command.txt").write_text(
                " ".join(cmd) + "\n", encoding="utf-8")
        except Exception:
            pass
        try:
            self._proc = self._launch_blktrace_process(cmd)
        except FileNotFoundError as e:
            self._reason = f"failed to launch blktrace: {e}"
            log.error("BlktraceCollector: %s", self._reason)
            return
        except Exception as e:
            self._reason = f"failed to launch blktrace: {e}"
            log.error("BlktraceCollector: %s", self._reason)
            return

        self._t0 = time.time()

        # blktrace can fail immediately. If it fails with EBUSY, recover once by
        # clearing the existing trace on exactly this device and relaunching.
        time.sleep(0.5)
        if self._proc.poll() is not None:
            self._launch_rc = self._proc.returncode
            try:
                self._launch_stderr = self._proc.stderr.read() if self._proc.stderr else ""
            except Exception:
                self._launch_stderr = ""

            if self._stderr_is_ebusy(self._launch_stderr):
                k_rc, k_msg = self._kill_existing_trace()
                try:
                    (self.work_dir / "blktrace_ebusy_recovery.txt").write_text(
                        k_msg + "\n", encoding="utf-8")
                except Exception:
                    pass
                if k_rc == 0:
                    log.warning("BlktraceCollector: EBUSY on %s; cleared existing trace and retrying", self.device)
                    try:
                        self._proc = self._launch_blktrace_process(cmd)
                        time.sleep(0.5)
                        if self._proc.poll() is None:
                            self._launch_rc = None
                            self._launch_stderr = ""
                        else:
                            self._launch_rc = self._proc.returncode
                            self._launch_stderr = self._proc.stderr.read() if self._proc.stderr else ""
                    except Exception as e:
                        self._launch_rc = -3
                        self._launch_stderr = repr(e)
                else:
                    self._launch_stderr = (self._launch_stderr or "") + "\n" + k_msg

        if self._proc.poll() is not None:
            self._launch_rc = self._proc.returncode
            try:
                self._launch_stderr = self._proc.stderr.read() if self._proc.stderr else self._launch_stderr
            except Exception:
                pass
            tail = (self._launch_stderr or "").strip().splitlines()
            tail_s = tail[-1] if tail else "no stderr"
            if self._stderr_is_ebusy(self._launch_stderr):
                self._reason = (
                    f"blktrace exited immediately with EBUSY after cleanup attempt rc={self._launch_rc}: "
                    f"{tail_s[:240]}. Stop the previous blktrace/AMOprof run or run "
                    f"`blktrace -d {self.device} -k`; Command: {' '.join(self._launch_cmd)}"
                )
            else:
                self._reason = (
                    f"blktrace exited immediately rc={self._launch_rc}: {tail_s[:240]}. "
                    f"Command: {' '.join(self._launch_cmd)}"
                )
            try:
                (self.work_dir / "blktrace_launch_stderr.txt").write_text(
                    self._launch_stderr or "", encoding="utf-8")
            except Exception:
                pass
            log.error("BlktraceCollector: %s", self._reason)
            return
        else:
            # Early file visibility is diagnostic only. On some systems blktrace
            # is alive and attached before trace.blktrace.* files are visible or
            # non-empty; the authoritative validation happens in stop(), after
            # blktrace flushes output. Do not emit a WARNING here because it can
            # look like a failure even when collection is healthy.
            try:
                bins = sorted(self._trace_dir.glob("trace.blktrace.*"))

                debugfs_mounted = False
                tracefs_mounted = False
                try:
                    mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore")
                    debugfs_mounted = (" /sys/kernel/debug " in mounts and "debugfs" in mounts)
                    tracefs_mounted = (" /sys/kernel/tracing " in mounts and "tracefs" in mounts)
                except Exception:
                    pass

                status = {
                    "blktrace_pid": getattr(self._proc, "pid", None),
                    "blktrace_running": bool(self._proc and self._proc.poll() is None),
                    "device": self.device,
                    "trace_dir": str(self._trace_dir),
                    "trace_prefix": str(self._trace_prefix),
                    "trace_file_count_after_start": len(bins),
                    "trace_files_seen_after_start": [str(x) for x in bins],
                    "debugfs_mounted": debugfs_mounted,
                    "tracefs_mounted": tracefs_mounted,
                    "note": (
                        "No trace.blktrace.* files immediately after launch is not an error. "
                        "AMOprof validates trace files at stop() after blktrace flushes output."
                        if not bins else
                        "Trace files were visible immediately after launch."
                    ),
                }
                try:
                    (self.work_dir / "blktrace_start_status.json").write_text(
                        json.dumps(status, indent=2), encoding="utf-8")
                except Exception:
                    pass

                if not bins:
                    log.info(
                        "BlktraceCollector: process is running; no trace.blktrace.* files "
                        "visible yet under %s. Diagnostic only; final validation occurs at stop().",
                        self._trace_dir)
            except Exception:
                pass

    def _run_blkparse_to_csv(self) -> int:
        """Run blkparse against the binary trace files and write CSV.
           Returns event count written. Returns -1 on parse failure.

           Captures both stdout (parsed events) and stderr (errors) so silent
           failures (events=0 with no reason) are surfaced.
        """
        if not self._which(self.blkparse_bin):
            self._reason = (self._reason + "; blkparse not found").lstrip("; ")
            log.warning("BlktraceCollector: blkparse not available — binaries kept at %s",
                        self._trace_dir)
            return -1
        # Check binary files exist
        bins = sorted(self._trace_dir.glob("trace.blktrace.*"))
        if not bins:
            self._reason = "no trace.blktrace.* binaries produced"
            log.warning("BlktraceCollector: %s", self._reason)
            return -1
        # `-O` says "do not output 'X--RWBS---' classified events" (notification
        # filtering). The default output format is what our regex parses.
        cmd = [self.blkparse_bin, "-i", str(self._trace_prefix)]
        log.info("BlktraceCollector: parsing %d binaries (%d MB total) "
                 "with: %s",
                 len(bins),
                 sum(p.stat().st_size for p in bins) // (1024*1024),
                 " ".join(cmd))
        events = 0
        unmatched_sample: list[str] = []
        try:
            # Capture stderr too — when blkparse fails it writes to stderr
            # and our previous DEVNULL silently swallowed the error, leaving
            # users with events=0 and no diagnostic.
            with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) as proc, \
                 open(self._csv_path, "w", encoding="utf-8", newline="") as fp:
                w = _csv.writer(fp)
                w.writerow(["ts", "pid", "action", "rwbs", "op",
                            "sector", "nsectors", "size_bytes",
                            "comm", "dev", "cpu"])
                if proc.stdout is None:
                    return -1
                lines_read = 0
                _fmt = "A"
                _fmt_locked = False
                pending_op: dict[tuple[str, str, str], str] = {}
                for raw in proc.stdout:
                    lines_read += 1
                    m = None
                    if _fmt == "A":
                        m = self._LINE_RE.match(raw)
                        if m is None and not _fmt_locked:
                            m2 = self._LINE_RE_OLD.match(raw)
                            if m2 and m2.group("sector"):
                                _fmt = "B"; _fmt_locked = True; m = m2
                    else:
                        m = self._LINE_RE_OLD.match(raw)
                    if not m:
                        # Capture a small sample of unmatched lines for diagnostics
                        if len(unmatched_sample) < 5 and raw.strip():
                            # Skip obvious non-event lines (CPU summary, "Input file")
                            stripped = raw.strip()
                            if not (stripped.startswith("CPU") or
                                    stripped.startswith("Input") or
                                    stripped.startswith("Total") or
                                    stripped.startswith("Throughput") or
                                    stripped.startswith("Events") or
                                    "events queued" in stripped.lower()):
                                unmatched_sample.append(stripped[:140])
                        continue
                    d = m.groupdict()
                    rwbs = d.get("rwbs") or ""
                    cpu_val = d.get("cpu") or "0"
                    if "D" in rwbs:
                        op = "D"
                    elif "W" in rwbs:
                        op = "W"
                    elif "R" in rwbs:
                        op = "R"
                    else:
                        op = "?"
                    nsec = int(d["nsec"])
                    req_key = (d.get("pid") or "0", d.get("sector") or "0", str(nsec))
                    action = d.get("action") or ""
                    if action != "C" and op in ("R", "W", "D"):
                        pending_op[req_key] = op
                    elif action == "C" and op not in ("R", "W", "D"):
                        op = pending_op.pop(req_key, "?")
                    try:
                        w.writerow([
                            d["ts"], d["pid"], action, rwbs, op,
                            d["sector"], nsec, nsec * 512,
                            d.get("comm") or "",
                            f"{d['major']},{d['minor']}",
                            cpu_val,
                        ])
                        events += 1
                        if self.max_csv_events is not None and events >= self.max_csv_events:
                            log.warning("BlktraceCollector: capping CSV at %d events because --blktrace-max-events was set; "
                                        "raw binaries at %s preserve full trace",
                                        self.max_csv_events, self._trace_dir)
                            # Drain remaining stdout so blkparse can exit cleanly.
                            # Without this, the pipe buffer fills, blkparse blocks
                            # on write(), and proc.wait() times out.
                            log.info("BlktraceCollector: draining remaining blkparse "
                                     "output (CSV capped) ...")
                            try:
                                if proc.stdout:
                                    proc.stdout.read()   # drain to EOF
                            except Exception:
                                pass
                            break
                    except Exception:
                        pass
                # Wait for blkparse to finish. Use a generous timeout scaled to
                # trace size: 1s per MB of binary data, minimum 60s, max 600s.
                _trace_mb = sum(
                    p.stat().st_size for p in self._trace_dir.glob("trace.blktrace.*")
                    if p.exists()
                ) / (1024 * 1024)
                _wait_s = max(60, min(int(_trace_mb), 600))
                stderr_output = ""
                try:
                    proc.wait(timeout=_wait_s)
                except subprocess.TimeoutExpired:
                    log.warning("BlktraceCollector: blkparse still running after %ds "
                                "— killing", _wait_s)
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                try:
                    stderr_output = proc.stderr.read() if proc.stderr else ""
                except Exception:
                    pass
                rc = proc.returncode if proc.returncode is not None else 0

            if rc != 0:
                # blkparse failed — report it instead of silently returning 0
                err_tail = (stderr_output or "").strip().splitlines()
                err_msg = err_tail[-1] if err_tail else f"rc={rc}"
                self._reason = f"blkparse exited rc={rc}: {err_msg[:200]}"
                log.error("BlktraceCollector: %s", self._reason)
                return -1

            if events == 0 and lines_read > 0:
                # blkparse produced output but our regex matched nothing
                self._reason = (
                    f"blkparse produced {lines_read} lines but regex matched 0 events; "
                    f"format may have changed. Sample unmatched lines: "
                    + " | ".join(unmatched_sample[:3])
                )[:400]
                log.error("BlktraceCollector: %s", self._reason)
                return -1

            if events == 0 and lines_read == 0:
                # blkparse rc=0 but no stdout — unusual
                err_tail = (stderr_output or "").strip().splitlines()
                err_msg = err_tail[-1] if err_tail else "no stderr"
                self._reason = f"blkparse produced no output (rc=0); stderr: {err_msg[:200]}"
                log.error("BlktraceCollector: %s", self._reason)
                return -1

            log.info("BlktraceCollector: parsed %d events from %d stdout lines",
                     events, lines_read)
        except subprocess.TimeoutExpired as e:
            self._reason = f"blkparse hung beyond 30s after stdout closed: {e}"
            log.error("BlktraceCollector: %s", self._reason)
            return -1
        except Exception as e:
            self._reason = f"blkparse parse error: {e}"
            log.warning("BlktraceCollector: %s", self._reason)
            return -1
        return events

    def stop(self) -> dict:
        # `blktrace -w` self-terminates after the duration, but if AMOprof stops
        # collection early we send SIGINT for a clean shutdown. On Ctrl-C,
        # keep this bounded so the user does not wait on a 4-hour blktrace -w.
        interrupted = bool(getattr(self, "_amoprof_interrupted", False))
        stop_timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (3.0 if interrupted else 10.0))
        stderr_text = ""
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGINT)
                except Exception:
                    try: self._proc.terminate()
                    except Exception: pass
                try:
                    self._proc.wait(timeout=stop_timeout)
                except Exception:
                    try: self._proc.kill()
                    except Exception: pass
            # Drain stderr — blktrace writes its per-CPU summary and any
            # dropped-event warnings there. Without this, we lose the only
            # signal that distinguishes "I traced everything correctly" from
            # "the kernel buffer overran and I silently dropped 75% of writes".
            try:
                if self._proc.stderr is not None:
                    stderr_text = self._proc.stderr.read() or ""
            except Exception:
                stderr_text = ""

        # Parse blktrace's per-CPU + dropped-events summary out of stderr.
        # Typical format on completion:
        #   === nvme7n1 ===
        #     CPU  0:  234567 events, 27 KiB data
        #     ...
        #     Total: 20733434 events (dropped 0), 3645 MiB data
        dropped_events = 0
        reported_events = 0
        per_cpu_dropped: dict[int, int] = {}
        try:
            # Per-CPU "dropped" warnings (one per overflowing CPU) — these
            # appear AS THEY HAPPEN, not just in the final summary.
            for m in re.finditer(r"CPU\s+(\d+).*?dropped\s+(\d+)\s+events",
                                  stderr_text, re.IGNORECASE):
                per_cpu_dropped[int(m.group(1))] = int(m.group(2))
            # Final summary "Total: N events (dropped M), ..."
            m_tot = re.search(r"Total[:\s]+([\d,]+)\s+events\s*\(dropped\s+([\d,]+)\)",
                               stderr_text, re.IGNORECASE)
            if m_tot:
                reported_events = int(m_tot.group(1).replace(",", ""))
                dropped_events = int(m_tot.group(2).replace(",", ""))
            elif per_cpu_dropped:
                dropped_events = sum(per_cpu_dropped.values())
        except Exception:
            pass

        bins = sorted(self._trace_dir.glob("trace.blktrace.*"))
        bin_count = len(bins)
        bin_bytes = sum(p.stat().st_size for p in bins) if bins else 0

        if bin_count == 0:
            reason = self._reason or "no trace.blktrace.* binaries produced"
            if not reason and self._launch_rc is not None:
                reason = f"blktrace exited immediately rc={self._launch_rc}"
            return {"blktrace_available": False,
                    "blktrace_reason": reason,
                    "blktrace_events": 0,
                    "blktrace_device": self.device,
                    "blktrace_binary_dir": str(self._trace_dir),
                    "blktrace_command": " ".join(self._launch_cmd) if self._launch_cmd else "",
                    "blktrace_launch_rc": self._launch_rc,
                    "blktrace_launch_stderr_tail": (self._launch_stderr or "")[-2000:],
                    "blktrace_stderr_tail": (stderr_text or self._launch_stderr or "")[-2000:]}

        events_csv = -1
        interrupted_raw_only = False
        if self.parse_after and not interrupted:
            events_csv = self._run_blkparse_to_csv()
            self._events = max(events_csv, 0)
        elif interrupted:
            interrupted_raw_only = True
            self._reason = (
                (self._reason + "; ") if self._reason else ""
            ) + "interrupted: skipped blkparse for fast shutdown; raw trace binaries kept for later analysis"
            try:
                (self.work_dir / "blktrace_interrupt_fast_stop.txt").write_text(
                    self._reason + "\n", encoding="utf-8")
            except Exception:
                pass
            self._events = 0

        # ── Coverage sanity: did we plausibly catch all of the device's I/O? ─
        # Cross-check captured byte total against /sys/block/<dev>/stat's
        # sectors_written DELTA over the trace window. This is the kernel's
        # own write counter, independent of blktrace's ring buffer. If blktrace
        # captured far fewer bytes than this counter saw, events were dropped
        # or the trace was on the wrong device.
        #
        # We also capture the merges delta, which lets the analyzer answer:
        # "did blktrace see fewer ops because events were dropped, or because
        # the block layer merged adjacent I/Os into bigger ones?" — the latter
        # is healthy behaviour and very common on XFS.
        coverage: dict = {}
        try:
            dev_base = re.sub(r"p\d+$", "", Path(self.device).name)
            stat_path = Path(f"/sys/block/{dev_base}/stat")
            if stat_path.exists():
                parts = stat_path.read_text().split()
                if len(parts) >= 8:
                    rd_ios_now     = int(parts[0])
                    rd_merges_now  = int(parts[1])
                    rd_sectors_now = int(parts[2])
                    wr_ios_now     = int(parts[4])
                    wr_merges_now  = int(parts[5])
                    wr_sectors_now = int(parts[6])

                    def _dlt(now, start_attr):
                        return max(now - getattr(self, start_attr, 0), 0)

                    rd_ios_delta     = _dlt(rd_ios_now,     "_rd_ios_start")
                    rd_merges_delta  = _dlt(rd_merges_now,  "_rd_merges_start")
                    rd_sectors_delta = _dlt(rd_sectors_now, "_rd_sectors_start")
                    wr_ios_delta     = _dlt(wr_ios_now,     "_wr_ios_start")
                    wr_merges_delta  = _dlt(wr_merges_now,  "_wr_merges_start")
                    wr_sectors_delta = _dlt(wr_sectors_now, "_wr_sectors_start")
                    wr_delta_gb      = round(wr_sectors_delta * 512 / (1024**3), 2)
                    rd_delta_gb      = round(rd_sectors_delta * 512 / (1024**3), 2)

                    coverage["sys_block_wr_sectors_delta"] = wr_sectors_delta
                    coverage["sys_block_wr_gb_delta"]      = wr_delta_gb
                    coverage["sys_block_wr_sectors_now"]   = wr_sectors_now
                    coverage["sys_block_rd_sectors_delta"] = rd_sectors_delta
                    coverage["sys_block_rd_gb_delta"]      = rd_delta_gb
                    coverage["sys_block_rd_ios_delta"]     = rd_ios_delta
                    coverage["sys_block_wr_ios_delta"]     = wr_ios_delta
                    coverage["sys_block_rd_merges_delta"]  = rd_merges_delta
                    coverage["sys_block_wr_merges_delta"]  = wr_merges_delta

                    # Merge ratio = merged_ops / (merged_ops + dispatched_ops).
                    # 0.0 means every BIO went straight to the device (no merging).
                    # 0.5 means the block layer halved the BIO count via coalescing.
                    # Values >0.3 on writes are typical for XFS-with-extent-allocation
                    # under contiguous-write workloads.
                    rd_total = rd_merges_delta + rd_ios_delta
                    wr_total = wr_merges_delta + wr_ios_delta
                    coverage["sys_block_rd_merge_ratio"] = round(
                        rd_merges_delta / rd_total, 3) if rd_total > 0 else 0
                    coverage["sys_block_wr_merge_ratio"] = round(
                        wr_merges_delta / wr_total, 3) if wr_total > 0 else 0

                    # Average bytes per *dispatched* I/O = how big a BIO the
                    # device actually saw. Useful for answering "why are op
                    # counts low" — if avg_io is large the FS merged a lot.
                    coverage["sys_block_avg_wr_io_kb"] = round(
                        wr_sectors_delta * 512 / 1024 / wr_ios_delta, 1
                    ) if wr_ios_delta > 0 else 0
                    coverage["sys_block_avg_rd_io_kb"] = round(
                        rd_sectors_delta * 512 / 1024 / rd_ios_delta, 1
                    ) if rd_ios_delta > 0 else 0
        except Exception:
            pass

        # ── Build coverage warning from BOTH signals ────────────────────────
        # (a) blktrace's own dropped-events counter from stderr, and
        # (b) the discrepancy between captured bytes and /sys/block delta.
        coverage_warning = ""
        warnings_list: list[str] = []
        if dropped_events > 0 and reported_events > 0:
            drop_pct = dropped_events / (dropped_events + reported_events) * 100
            if drop_pct >= 0.1:
                warnings_list.append(
                    f"blktrace dropped {dropped_events:,} events "
                    f"({drop_pct:.2f}% of total) — kernel ring buffer overran. "
                    f"Increase blktrace -b buffer size or trace fewer concurrent collectors.")
        # Compare captured byte total against /sys/block delta. Need to wait
        # until the analyzer runs to know captured bytes precisely; for now
        # store the kernel-side ground truth so the writer/report can flag
        # the discrepancy after blktrace analysis completes.
        kernel_wr_gb = coverage.get("sys_block_wr_gb_delta", 0)
        # The check happens in two places:
        #  1. Here, using just bin file size as a rough proxy (each event ≈
        #     24 bytes in blktrace binary form, plus IO data, so binary_bytes
        #     ~= 24 × events). Catches gross device mismatches early.
        if kernel_wr_gb > 10 and bin_bytes > 0:
            # Each blktrace event averages ~24 bytes binary; we don't know
            # the byte-traffic-per-event ratio without parsing, so just compare
            # orders of magnitude. If kernel wrote >10× more than blktrace
            # captured (after parsing), something is wrong.
            pass  # detailed check happens after blkparse runs; see writer
        coverage_warning = " ".join(warnings_list)

        return {
            "blktrace_available": True,
            "blktrace_binary_dir": str(self._trace_dir),
            "blktrace_binary_files": bin_count,
            "blktrace_binary_bytes": bin_bytes,
            "blktrace_csv":      str(self._csv_path) if events_csv > 0 else "",
            "blktrace_events":   self._events,
            "blktrace_interrupted_raw_only": interrupted_raw_only,
            "blktrace_fast_stop": interrupted,
            "blktrace_device":   self.device,
            "blktrace_buffer_kb": self.buffer_kb,
            "blktrace_num_buffers": self.num_buffers,
            "blktrace_duration_s": round(time.time() - self._t0, 2),
            "blktrace_csv_event_cap": self.max_csv_events,
            "blktrace_csv_capped": bool(self.max_csv_events is not None and self._events >= self.max_csv_events),
            "blktrace_reason":   self._reason,
            "blktrace_command": " ".join(self._launch_cmd) if self._launch_cmd else "",
            "blktrace_launch_rc": self._launch_rc,
            "blktrace_launch_stderr_tail": (self._launch_stderr or "")[-2000:],
            # NEW — diagnostic fields for the "blktrace says X GB but df says Y GB" puzzle
            "blktrace_events_reported_by_blktrace": reported_events,
            "blktrace_dropped_events": dropped_events,
            "blktrace_per_cpu_dropped": per_cpu_dropped,
            "blktrace_coverage_warning": coverage_warning,
            "blktrace_stderr_tail": (stderr_text or "")[-2000:],
            **coverage,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BiosnoopCollector — per-IO event w/ PID/comm via BCC biosnoop
# ─────────────────────────────────────────────────────────────────────────────
class BiosnoopCollector:
    """
    Run BCC biosnoop to capture per-I/O events with PID + command attribution.

    Important behavior:
      • biosnoop's -t means "print timestamps"; it is NOT a duration option.
        AMOprof controls duration by starting biosnoop at collect start and
        stopping it at collect stop.
      • stdout is captured both as a raw log and parsed CSV.
      • stderr/command/summary are persisted so empty CSVs are diagnosable.
      • all-device biosnoop output is retained in biosnoop_events_all.csv, while
        biosnoop_events.csv contains the target-device/partition-filtered view.
    """

    def __init__(self, duration_s: int = 60,
                 work_dir: "Path | str" = ".",
                 device: str | None = None,
                 binary: str = "biosnoop",
                 use_sudo: bool = True,
                 filename_suffix: str = ""):
        self.duration_s = max(int(duration_s), 1)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.device_name = os.path.basename(device) if device else None
        self.binary = binary
        self.use_sudo = use_sudo

        self._proc = None
        self._reader = None
        self._stderr_reader = None
        _sfx = ("_" + filename_suffix) if filename_suffix else ""
        self._csv_path = self.work_dir / ("biosnoop_events" + _sfx + ".csv")
        self._csv_all_path = self.work_dir / ("biosnoop_events_all" + _sfx + ".csv")
        self._raw_path = self.work_dir / ("biosnoop_raw" + _sfx + ".log")
        self._stderr_path = self.work_dir / ("biosnoop_stderr" + _sfx + ".txt")
        self._cmd_path = self.work_dir / ("biosnoop_command" + _sfx + ".txt")
        self._summary_path = self.work_dir / ("biosnoop_summary" + _sfx + ".json")

        self._csv_fp = None
        self._csv_all_fp = None
        self._raw_fp = None
        self._stderr_fp = None
        self._csv_w = None
        self._csv_all_w = None
        self._events = 0
        self._events_all = 0
        self._lines_seen = 0
        self._lines_unparsed = 0
        self._headers_seen = 0
        self._reason = ""
        self._t0 = 0.0
        self._found = None
        self._cmd = []
        self._header_cols = None

    def _which(self, name: str) -> bool:
        try:
            subprocess.check_output(["which", name], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _biosnoop_supports_t(self, binary: str) -> bool:
        """Return True only when this biosnoop build supports -t.

        Some bpfcc-tools builds already print TIME(s) by default and expose only
        [-h] [-Q]. Passing -t to those builds makes biosnoop exit with
        "unrecognized arguments: -t". AMOprof must auto-detect.
        """
        try:
            r = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=5)
            help_txt = (r.stdout or "") + "\n" + (r.stderr or "")
        except Exception:
            try:
                r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
                help_txt = (r.stdout or "") + "\n" + (r.stderr or "")
            except Exception:
                return False
        return bool(re.search(r"(^|[\s,\[])-t([\s,\],]|$)", help_txt))

    def _build_biosnoop_cmd(self, binary: str) -> list[str]:
        cmd = [binary]
        if self._biosnoop_supports_t(binary):
            cmd.append("-t")
        return cmd

    @staticmethod
    def _norm_col(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(x).lower())

    def _device_matches(self, disk: str) -> bool:
        """Accept whole-device and partition names.

        Examples:
          device nvme7n1 matches nvme7n1 and nvme7n1p1
          device sda     matches sda and sda1
        """
        if not self.device_name:
            return True
        d = str(disk or "")
        base = self.device_name
        if d == base:
            return True
        if base.startswith("nvme") and d.startswith(base + "p"):
            return True
        if re.match(r"^[a-z]+[a-z]$", base) and d.startswith(base) and d[len(base):].isdigit():
            return True
        return False

    def _open_outputs(self):
        header = ["ts", "comm", "pid", "disk", "op", "sector", "size_bytes", "latency_ms"]
        self._csv_fp = open(self._csv_path, "w", encoding="utf-8", newline="")
        self._csv_all_fp = open(self._csv_all_path, "w", encoding="utf-8", newline="")
        self._raw_fp = open(self._raw_path, "w", encoding="utf-8")
        self._stderr_fp = open(self._stderr_path, "w", encoding="utf-8")
        self._csv_w = _csv.writer(self._csv_fp)
        self._csv_all_w = _csv.writer(self._csv_all_fp)
        self._csv_w.writerow(header)
        self._csv_all_w.writerow(header)

    def _write_summary(self, extra: dict | None = None):
        data = {
            "biosnoop_available": bool(self._events > 0),
            "biosnoop_events": int(self._events),
            "biosnoop_events_all_devices": int(self._events_all),
            "biosnoop_lines_seen": int(self._lines_seen),
            "biosnoop_lines_unparsed": int(self._lines_unparsed),
            "biosnoop_headers_seen": int(self._headers_seen),
            "biosnoop_csv": str(self._csv_path),
            "biosnoop_all_devices_csv": str(self._csv_all_path),
            "biosnoop_raw_log": str(self._raw_path),
            "biosnoop_stderr": str(self._stderr_path),
            "biosnoop_command": str(self._cmd_path),
            "biosnoop_duration_s": round(time.time() - self._t0, 2) if self._t0 else 0,
            "biosnoop_filter_disk": self.device_name or "(all)",
            "biosnoop_reason": self._reason,
            "biosnoop_binary": self._found,
            "biosnoop_cmd": self._cmd,
            "note": (
                "biosnoop_events.csv is filtered to the requested SSD/device and matching partitions; "
                "biosnoop_events_all.csv keeps the unfiltered all-device stream."
            ),
        }
        if extra:
            data.update(extra)
        try:
            self._summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        return data

    def start(self):
        self._events = 0
        self._events_all = 0
        self._lines_seen = 0
        self._lines_unparsed = 0
        self._headers_seen = 0
        self._reason = ""

        candidates = ["biosnoop", "/usr/sbin/biosnoop", self.binary, "biosnoop-bpfcc",
                      "/usr/sbin/biosnoop-bpfcc", "/usr/share/bcc/tools/biosnoop"]
        found = None
        for c in candidates:
            if Path(c).exists() and os.access(c, os.X_OK):
                found = c
                break
            if self._which(c):
                found = c
                break
        self._found = found
        if not found:
            self._reason = "biosnoop not installed (apt: bpfcc-tools)"
            log.warning("BiosnoopCollector: %s", self._reason)
            self._write_summary({"biosnoop_available": False})
            return

        # Biosnoop variants differ: some accept -t, while others print TIME(s)
        # by default and reject -t. Auto-detect and never pass a duration arg.
        cmd = self._build_biosnoop_cmd(found)
        if self.use_sudo and os.geteuid() != 0:
            cmd = ["sudo", "-n", "-E"] + cmd
        self._cmd = cmd
        try:
            self._cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        except Exception:
            pass

        self._open_outputs()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
        except FileNotFoundError as e:
            self._reason = f"failed to launch biosnoop: {e}"
            log.warning("BiosnoopCollector: %s", self._reason)
            self._write_summary({"biosnoop_available": False})
            return
        except Exception as e:
            self._reason = f"failed to launch biosnoop: {e}"
            log.warning("BiosnoopCollector: %s", self._reason)
            self._write_summary({"biosnoop_available": False})
            return

        self._t0 = time.time()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()
        self._write_summary({"biosnoop_started": True, "biosnoop_pid": getattr(self._proc, "pid", None)})

    def _stderr_loop(self):
        if self._proc is None or self._proc.stderr is None:
            return
        for raw in self._proc.stderr:
            try:
                if self._stderr_fp:
                    self._stderr_fp.write(raw)
                    self._stderr_fp.flush()
            except Exception:
                pass

    def _parse_with_header(self, cols: list[str]):
        hdr = self._header_cols or []
        idx = {self._norm_col(h): i for i, h in enumerate(hdr)}
        def pick(names, default=None):
            for n in names:
                j = idx.get(self._norm_col(n))
                if j is not None and j < len(cols):
                    return cols[j]
            return default
        return [
            pick(["TIME(s)", "TIME", "TS"], cols[0] if cols else ""),
            pick(["COMM", "CMD", "PROCESS"], ""),
            pick(["PID"], ""),
            pick(["DISK", "DEV", "DEVICE"], ""),
            pick(["T", "TYPE", "OP", "RWBS"], ""),
            pick(["SECTOR", "LBA"], ""),
            pick(["BYTES", "SIZE"], ""),
            pick(["LAT(ms)", "LAT", "LATENCY", "LATENCY(ms)"], ""),
        ]

    def _parse_line(self, line: str):
        cols = line.split()
        if not cols:
            return None

        # Header variants commonly contain PID/DISK/SECTOR/BYTES/LAT.
        joined_norm = " ".join(cols).lower()
        if ("pid" in joined_norm and "disk" in joined_norm and
            ("sector" in joined_norm or "bytes" in joined_norm)):
            self._header_cols = cols
            self._headers_seen += 1
            return "header"

        if len(cols) < 7:
            self._lines_unparsed += 1
            return None

        # Header-driven path first.
        if self._header_cols:
            row = self._parse_with_header(cols)
            if row[2] and row[3] and row[6]:
                return row

        # Fallback for BCC biosnoop -t:
        # TIME(s) COMM PID DISK T SECTOR BYTES LAT(ms)
        if len(cols) >= 8:
            return [cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]]

        # Fallback for no timestamp:
        # COMM PID DISK T SECTOR BYTES LAT(ms)
        if len(cols) >= 7:
            return ["", cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6]]

        self._lines_unparsed += 1
        return None

    def _read_loop(self):
        if self._proc is None or self._proc.stdout is None:
            return
        for raw in self._proc.stdout:
            line = raw.rstrip("\n")
            if self._raw_fp:
                try:
                    self._raw_fp.write(raw)
                    self._raw_fp.flush()
                except Exception:
                    pass
            if not line:
                continue
            self._lines_seen += 1
            parsed = self._parse_line(line)
            if parsed is None or parsed == "header":
                continue
            try:
                ts, comm, pid, disk, op, sect, size, lat = parsed
                self._csv_all_w.writerow([ts, comm, pid, disk, op, sect, size, lat])
                self._events_all += 1
                if self._device_matches(disk):
                    self._csv_w.writerow([ts, comm, pid, disk, op, sect, size, lat])
                    self._events += 1
                if self._events_all % 1000 == 0:
                    try:
                        self._csv_all_fp.flush()
                        self._csv_fp.flush()
                    except Exception:
                        pass
            except Exception:
                self._lines_unparsed += 1

    def stop(self) -> dict:
        if self._proc is not None and self._proc.poll() is None:
            # SIGINT is friendlier to BCC Python tools than SIGTERM and gives
            # them a chance to unwind probes and flush stdout/stderr.
            try:
                self._proc.send_signal(signal.SIGINT)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

        if self._reader is not None:
            self._reader.join(timeout=10)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=5)

        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

        for fp in (self._csv_fp, self._csv_all_fp, self._raw_fp, self._stderr_fp):
            if fp is not None:
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass

        rc = None
        if self._proc is not None:
            try:
                rc = self._proc.poll()
            except Exception:
                rc = None

        if self._events == 0:
            if self._events_all > 0:
                self._reason = (
                    f"biosnoop captured {self._events_all} events on other disks/devices, "
                    f"but none matched requested device filter {self.device_name!r}; "
                    f"inspect {self._csv_all_path.name} and check DISK names/partitions."
                )
            else:
                stderr_tail = ""
                try:
                    stderr_tail = self._stderr_path.read_text(encoding="utf-8", errors="ignore")[-1200:]
                except Exception:
                    pass
                self._reason = self._reason or (
                    "no biosnoop events captured; possible causes: no block I/O during window, "
                    "BCC/kernel probe attach failure, unsupported biosnoop binary, or permissions/CAP_BPF/CAP_SYS_ADMIN. "
                    f"stderr_tail={stderr_tail!r}"
                )

        summary = self._write_summary({
            "biosnoop_available": bool(self._events > 0),
            "biosnoop_returncode": rc,
        })

        if self._events == 0:
            log.warning("BiosnoopCollector: %s", self._reason)
            return {
                "biosnoop_available": False,
                "biosnoop_reason": self._reason,
                "biosnoop_events": 0,
                "biosnoop_events_all_devices": self._events_all,
                "biosnoop_summary": str(self._summary_path),
                "biosnoop_stderr": str(self._stderr_path),
                "biosnoop_raw_log": str(self._raw_path),
            }

        return {
            "biosnoop_available": True,
            "biosnoop_events": self._events,
            "biosnoop_events_all_devices": self._events_all,
            "biosnoop_csv": str(self._csv_path),
            "biosnoop_all_devices_csv": str(self._csv_all_path),
            "biosnoop_summary": str(self._summary_path),
            "biosnoop_duration_s": round(time.time() - self._t0, 2),
            "biosnoop_filter_disk": self.device_name or "(all)",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DiscardStatsMonitor — TRIM/discard time-series from /proc/diskstats
# ─────────────────────────────────────────────────────────────────────────────
class DiscardStatsMonitor:
    """
    Time-series of TRIM/discard activity from /proc/diskstats.

    /proc/diskstats columns (Linux ≥4.18, fields 14-17 are discard fields):
        major minor name
        14: discards completed
        15: discards merged
        16: sectors discarded
        17: time spent discarding (ms)

    Plus we mirror the read/write columns (4-13) so we can compute the
    read/write/discard ratio in one place.

    Output: <work_dir>/discard_timeseries.csv per-sample row.
    Summary: discard_iops, discard_bw_mb_s, discard_present (bool).
    """

    def __init__(self, device: str, interval_s: float = 1.0,
                 work_dir: "Path | str" = ".", filename_suffix: str = ""):
        self.device_name = os.path.basename(device)
        self.interval = max(float(interval_s), 0.1)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread = None
        self._samples: list[dict] = []
        _sfx = ("_" + filename_suffix) if filename_suffix else ""
        self._csv_path = self.work_dir / ("discard_timeseries" + _sfx + ".csv")

    def _read_diskstats(self) -> dict | None:
        try:
            with open("/proc/diskstats") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 14:
                        continue
                    if p[2] != self.device_name:
                        continue
                    # Reads:  p[3]=completed, p[5]=sectors, p[6]=ms
                    # Writes: p[7]=completed, p[9]=sectors, p[10]=ms
                    # Discards: p[14]=completed, p[15]=merged, p[16]=sectors, p[17]=ms
                    rec = {
                        "ts":              time.time(),
                        "rd_completed":    int(p[3]),
                        "rd_sectors":      int(p[5]),
                        "rd_ms":           int(p[6]),
                        "wr_completed":    int(p[7]),
                        "wr_sectors":      int(p[9]),
                        "wr_ms":           int(p[10]),
                    }
                    if len(p) >= 18:
                        rec["disc_completed"] = int(p[14])
                        rec["disc_merged"]    = int(p[15])
                        rec["disc_sectors"]   = int(p[16])
                        rec["disc_ms"]        = int(p[17])
                    else:
                        rec["disc_completed"] = 0
                        rec["disc_merged"]    = 0
                        rec["disc_sectors"]   = 0
                        rec["disc_ms"]        = 0
                    return rec
        except Exception as e:
            log.debug("DiscardStatsMonitor: %s", e)
        return None

    def start(self):
        self._stop.clear()
        self._samples.clear()
        s = self._read_diskstats()
        if s is not None:
            self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = self._read_diskstats()
            if s is not None:
                self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        s = self._read_diskstats()
        if s is not None:
            self._samples.append(s)

        if len(self._samples) < 2:
            return {"discard_available": False,
                    "discard_reason": f"no diskstats samples for {self.device_name}"}

        # Write per-interval rate CSV
        rates = []
        with open(self._csv_path, "w", encoding="utf-8", newline="") as fp:
            w = _csv.writer(fp)
            w.writerow([
                "ts", "rd_iops", "rd_mb_s",
                       "wr_iops", "wr_mb_s",
                       "discard_iops", "discard_mb_s", "discard_avg_ms",
            ])
            for i in range(1, len(self._samples)):
                a, b = self._samples[i-1], self._samples[i]
                dt = max(b["ts"] - a["ts"], 1e-6)
                rd_iops = max(b["rd_completed"]   - a["rd_completed"],   0) / dt
                wr_iops = max(b["wr_completed"]   - a["wr_completed"],   0) / dt
                dc_iops = max(b["disc_completed"] - a["disc_completed"], 0) / dt
                rd_mb_s = max(b["rd_sectors"]     - a["rd_sectors"],     0) * 512 / 1024 / 1024 / dt
                wr_mb_s = max(b["wr_sectors"]     - a["wr_sectors"],     0) * 512 / 1024 / 1024 / dt
                dc_mb_s = max(b["disc_sectors"]   - a["disc_sectors"],   0) * 512 / 1024 / 1024 / dt
                dc_dms  = max(b["disc_ms"]        - a["disc_ms"],        0)
                dc_avg  = (dc_dms / max(b["disc_completed"] - a["disc_completed"], 1)
                           if (b["disc_completed"] - a["disc_completed"]) > 0 else 0.0)
                rates.append({
                    "rd_iops": rd_iops, "wr_iops": wr_iops, "dc_iops": dc_iops,
                    "rd_mb_s": rd_mb_s, "wr_mb_s": wr_mb_s, "dc_mb_s": dc_mb_s,
                })
                w.writerow([round(b["ts"], 3),
                            round(rd_iops, 2), round(rd_mb_s, 3),
                            round(wr_iops, 2), round(wr_mb_s, 3),
                            round(dc_iops, 2), round(dc_mb_s, 3),
                            round(dc_avg, 3)])

        def _mean(key):
            vals = [r[key] for r in rates]
            return sum(vals) / max(len(vals), 1) if vals else 0.0

        def _peak(key):
            vals = [r[key] for r in rates]
            return max(vals) if vals else 0.0

        # Total discards (sectors → bytes)
        total_disc_sectors = max(self._samples[-1]["disc_sectors"]
                                 - self._samples[0]["disc_sectors"], 0)
        total_disc_bytes = total_disc_sectors * 512
        total_disc_ops = max(self._samples[-1]["disc_completed"]
                             - self._samples[0]["disc_completed"], 0)
        discard_present = total_disc_ops > 0

        return {
            "discard_available":   True,
            "discard_present":     discard_present,
            "discard_csv":         str(self._csv_path),
            "discard_total_ops":   total_disc_ops,
            "discard_total_bytes": total_disc_bytes,
            "discard_iops_mean":   round(_mean("dc_iops"), 2),
            "discard_iops_peak":   round(_peak("dc_iops"), 2),
            "discard_bw_mb_s_mean": round(_mean("dc_mb_s"), 3),
            "discard_bw_mb_s_peak": round(_peak("dc_mb_s"), 3),
            # rd/wr summaries for completeness — analysis layer may use either
            # these or iostat's; here we expose them as discard collector by-product.
            "diskstats_rd_iops_mean": round(_mean("rd_iops"), 2),
            "diskstats_wr_iops_mean": round(_mean("wr_iops"), 2),
            "diskstats_rd_mb_s_mean": round(_mean("rd_mb_s"), 3),
            "diskstats_wr_mb_s_mean": round(_mean("wr_mb_s"), 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SwapStormMonitor — per-second time-series of swap pages from /proc/vmstat
# ─────────────────────────────────────────────────────────────────────────────
class SwapStormMonitor:
    """
    Sample /proc/vmstat at fixed interval and emit a time-series CSV of
    swap-related metrics. VmstatMonitor reports aggregates (first→last delta);
    this collector adds the per-sample rates needed to plot a swap storm chart.

    Output: <work_dir>/swap_timeseries.csv with columns:
        ts, pswpin_per_s, pswpout_per_s, pgmajfault_per_s,
            pgpgin_per_s,  pgpgout_per_s, oom_kills_cum
    """

    KEYS = ("pswpin", "pswpout", "pgmajfault", "pgpgin", "pgpgout", "oom_kill")

    def __init__(self, interval_s: float = 1.0,
                 work_dir: "Path | str" = "."):
        self.interval = max(float(interval_s), 0.1)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread = None
        self._samples: list[dict] = []
        self._csv_path = self.work_dir / "swap_timeseries.csv"

    def _read(self) -> dict:
        out: dict = {"ts": time.time()}
        try:
            with open("/proc/vmstat") as f:
                for line in f:
                    p = line.split()
                    if len(p) == 2 and p[0] in self.KEYS:
                        out[p[0]] = int(p[1])
        except Exception as e:
            log.debug("SwapStormMonitor read: %s", e)
        return out

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._samples.append(self._read())
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._samples.append(self._read())

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._samples.append(self._read())

        if len(self._samples) < 2:
            return {"swap_storm_available": False,
                    "swap_storm_reason": "no vmstat samples"}

        rates = []
        with open(self._csv_path, "w", encoding="utf-8", newline="") as fp:
            w = _csv.writer(fp)
            w.writerow(["ts", "pswpin_per_s", "pswpout_per_s",
                        "pgmajfault_per_s", "pgpgin_per_s", "pgpgout_per_s",
                        "oom_kills_cum"])
            base_oom = self._samples[0].get("oom_kill", 0)
            for i in range(1, len(self._samples)):
                a, b = self._samples[i-1], self._samples[i]
                dt = max(b["ts"] - a["ts"], 1e-6)
                def _r(k):
                    return max(b.get(k, 0) - a.get(k, 0), 0) / dt
                row = {
                    "pswpin": _r("pswpin"),
                    "pswpout": _r("pswpout"),
                    "pgmajfault": _r("pgmajfault"),
                    "pgpgin": _r("pgpgin"),
                    "pgpgout": _r("pgpgout"),
                }
                rates.append(row)
                w.writerow([round(b["ts"], 3),
                            round(row["pswpin"], 2),
                            round(row["pswpout"], 2),
                            round(row["pgmajfault"], 2),
                            round(row["pgpgin"], 2),
                            round(row["pgpgout"], 2),
                            max(b.get("oom_kill", 0) - base_oom, 0)])

        def _mean(k): return sum(r[k] for r in rates) / max(len(rates), 1)
        def _peak(k): return max((r[k] for r in rates), default=0.0)

        # Swap storm is detected when peak pswpout exceeds a threshold.
        # We expose the raw rates and let analysis decide; default threshold
        # for "storm present" is 100 pages/s (= 400 KB/s of swap activity).
        STORM_THRESHOLD = 100.0
        swap_storm = (_peak("pswpout") > STORM_THRESHOLD
                      or _peak("pswpin") > STORM_THRESHOLD)

        return {
            "swap_storm_available":  True,
            "swap_storm_present":    bool(swap_storm),
            "swap_storm_csv":        str(self._csv_path),
            "swap_in_per_s_mean":    round(_mean("pswpin"), 2),
            "swap_in_per_s_peak":    round(_peak("pswpin"), 2),
            "swap_out_per_s_mean":   round(_mean("pswpout"), 2),
            "swap_out_per_s_peak":   round(_peak("pswpout"), 2),
            "major_faults_per_s_peak": round(_peak("pgmajfault"), 2),
        }
