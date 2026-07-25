"""
service.py - AMOprof collection service with Prometheus /metrics endpoint.

Provides:
  MetricsStore       -- thread-safe store for latest collected metrics
  PrometheusServer   -- HTTP server exposing /metrics in Prometheus text format
  run_service        -- main loop: build collectors, cycle, update store
  AVAILABLE_COLLECTORS -- registry of all collector names the service supports

Usage:
  amoprof service --metrics-port 9101 --collectors gpu,dram,vmstat --interval-s 5
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("amoprof.service")


# ---------------------------------------------------------------------------
# Service-mode local persistence helpers
# ---------------------------------------------------------------------------

def _float_or_none(value: Any) -> float | None:
    """Convert a string/float/None value to float or None."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _detect_cpu_vendor() -> str:
    """Return a coarse CPU vendor string for DRAM PMU backend selection."""
    text = ""
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
    except Exception:
        text = platform.processor().lower()
    if "authenticamd" in text or "amd" in text:
        return "amd"
    if "genuineintel" in text or "intel" in text:
        return "intel"
    return "unknown"


def _resolve_dram_tool(args: Any) -> str:
    """Resolve --enable-dram backend: auto -> AMDuProf on AMD, Intel PCM on Intel."""
    tool = str(getattr(args, "dram_tool", "auto") or "auto").strip().lower()
    if tool in {"none", "off", "disabled"}:
        return "none"
    if tool in {"amduprof", "amd", "amduprof-pcm"}:
        return "amduprof"
    if tool in {"intel", "intel-pcm", "pcm", "pcm-memory"}:
        return "intel-pcm"
    if tool in {"perf", "perf-imc", "imc"}:
        return "perf-imc"
    vendor = _detect_cpu_vendor()
    if vendor == "amd":
        return "amduprof"
    if vendor == "intel":
        return "intel-pcm"
    return "intel-pcm"


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _iso_utc(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _collector_series_rows(obj: Any, t0_epoch: float) -> list[dict[str, Any]]:
    """Extract and normalize per-sample rows from a collector object.

    Each persisted row carries both absolute epoch time and relative time_sec.
    This is what lets offline service-mode data be filtered by an arbitrary
    wall-clock window without depending on Prometheus retention.
    """
    raw: list[Any] = []
    if hasattr(obj, "samples") and isinstance(getattr(obj, "samples"), list):
        raw = getattr(obj, "samples") or []
    elif hasattr(obj, "_samples") and isinstance(getattr(obj, "_samples"), list):
        raw = getattr(obj, "_samples") or []

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            d = dict(item)
        elif isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], dict):
            tag, ts, payload = item
            d = {"tag": tag, "ts": ts, **payload}
        else:
            continue

        ts_val = d.get("timestamp_epoch", d.get("ts", d.get("_ts")))
        try:
            ts_f = float(ts_val)
        except Exception:
            try:
                rel = float(d.get("time_sec", idx))
            except Exception:
                rel = float(idx)
            ts_f = float(t0_epoch) + rel

        if ts_f > 1_000_000_000:
            d["timestamp_epoch"] = round(ts_f, 6)
            d["ts"] = round(ts_f, 6)
            d["time_sec"] = round(ts_f - float(t0_epoch), 6)
        else:
            rel = ts_f
            d["time_sec"] = round(rel, 6)
            d["timestamp_epoch"] = round(float(t0_epoch) + rel, 6)
            d["ts"] = d["timestamp_epoch"]
        d["timestamp_utc"] = _iso_utc(float(d["timestamp_epoch"]))
        d["iteration"] = idx
        rows.append(d)
    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ["time_sec", "timestamp_epoch", "timestamp_utc", "ts", "iteration", "collector"]
    for k in preferred:
        for row in rows:
            if k in row and k not in seen:
                keys.append(k); seen.add(k)
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")


def _persist_service_cycle(
    *,
    cycle_dir: Path,
    raw_dir: Path,
    cycle: int,
    enabled: set[str],
    collectors: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    raw_paths: dict[str, str],
    start_epoch: float,
    end_epoch: float,
    interval_s: float,
    scrape_s: float,
    args: Any,
) -> None:
    """Write a service cycle as a normal local AMOprof run directory."""
    all_rows: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        summary_path = raw_dir / f"{name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        raw_paths[f"{name}_summary_json"] = str(summary_path)

        obj = collectors.get(name)
        if obj is None:
            continue
        rows = _collector_series_rows(obj, start_epoch)
        for row in rows:
            row["collector"] = name
        if rows:
            _write_rows_jsonl(raw_dir / f"{name}_timeseries.jsonl", rows)
            _write_rows_csv(raw_dir / f"{name}_timeseries.csv", rows)
            raw_paths[f"{name}_timeseries_jsonl"] = str(raw_dir / f"{name}_timeseries.jsonl")
            raw_paths[f"{name}_timeseries_csv"] = str(raw_dir / f"{name}_timeseries.csv")
            all_rows.extend(rows)

    if all_rows:
        all_rows = sorted(all_rows, key=lambda r: (float(r.get("time_sec", 0.0)), str(r.get("collector", ""))))
        _write_rows_jsonl(raw_dir / "all_timeseries.jsonl", all_rows)
        _write_rows_csv(raw_dir / "all_timeseries.csv", all_rows)
        raw_paths["all_timeseries_jsonl"] = str(raw_dir / "all_timeseries.jsonl")
        raw_paths["all_timeseries_csv"] = str(raw_dir / "all_timeseries.csv")

    # Create AMOprof-compatible canonical files/mirrors when possible.  This
    # keeps service_cycle_* directories analyzable with the same code path as
    # one-shot collect directories.
    try:
        from .writer import write_amoprof_files
        sglang_obj = collectors.get("sglang")
        sglang_samples = getattr(sglang_obj, "raw_samples", []) if sglang_obj is not None else []
        sglang_source = getattr(sglang_obj, "prometheus_url", "") if sglang_obj is not None else ""
        sglang_elapsed = float(getattr(sglang_obj, "elapsed_s", 0.0) or 0.0) if sglang_obj is not None else 0.0
        _server_type = "vllm" if (sglang_obj is not None and
                                  sglang_obj.__class__.__name__ == "VLLMMetricsSampler") else "sglang"
        written = write_amoprof_files(raw_dir, start_epoch,
                                       sglang_samples=sglang_samples,
                                       sglang_source=sglang_source,
                                       sglang_elapsed_s=sglang_elapsed,
                                       sglang_model=str(getattr(args, "model", "unknown")),
                                       server_type=_server_type)
        for fname, path in written.items():
            raw_paths[f"amoprof_{Path(fname).stem}"] = str(path)
    except Exception as exc:
        log.debug("service: canonical writer skipped: %s", exc)

    meta = {
        "run_id": cycle_dir.name,
        "label": cycle_dir.name,
        "service_mode": True,
        "service_cycle": cycle,
        "timestamp": _iso_utc(end_epoch),
        "t0_epoch": start_epoch,
        "start_time": start_epoch,
        "t_end_epoch": end_epoch,
        "end_time": end_epoch,
        "start_time_utc": _iso_utc(start_epoch),
        "end_time_utc": _iso_utc(end_epoch),
        "duration_s": round(end_epoch - start_epoch, 3),
        "requested_duration_s": scrape_s,
        "interval_s": interval_s,
        "collectors": sorted(enabled),
        "ssd_device": getattr(args, "ssd_device", ""),
        "ssd_devices": list(getattr(args, "ssd_devices", []) or
                            ([getattr(args, "ssd_device", "")] if getattr(args, "ssd_device", "") else [])),
        "hicache_path": getattr(args, "hicache_path", ""),
        "sglang_port": getattr(args, "sglang_port", "") or "",
        "vllm_port": getattr(args, "vllm_port", "") or "",
        "vllm_host": getattr(args, "vllm_host", "") or "",
    }
    payload = {"meta": meta, "summary": summaries, "raw_paths": raw_paths}
    (cycle_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    # Duplicate a compact meta file into raw/ because older aggregate/analyze
    # paths sometimes look in raw/summary.json first.
    (raw_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")

ALWAYS_COLLECTORS: list[str] = [
    "iostat", "smart", "ssd_hw", "vmstat", "nvlink_pcie",
    "nvme_driver", "gpu", "dram", "power", "pcm", "discard", "swap_storm",
]

OPTIONAL_COLLECTORS: list[str] = [
    "biolatency", "blktrace", "biosnoop", "amduprof_pcm",
    "sglang", "perf", "bpf", "nsys",
]

AVAILABLE_COLLECTORS: list[str] = ALWAYS_COLLECTORS + OPTIONAL_COLLECTORS


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------

class MetricsStore:
    """Thread-safe store for the latest scrape of each collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._last_scrape: float = 0.0
        self._scrape_count: int = 0

    def update(self, collector_name: str, summary: dict[str, Any]) -> None:
        with self._lock:
            self._data[collector_name] = dict(summary)
            self._last_scrape = time.time()
            self._scrape_count += 1

    def snapshot(self) -> tuple[dict[str, dict[str, Any]], float, int]:
        """Return (data_copy, last_scrape_epoch, scrape_count)."""
        with self._lock:
            return (
                {k: dict(v) for k, v in self._data.items()},
                self._last_scrape,
                self._scrape_count,
            )


# ---------------------------------------------------------------------------
# Prometheus text-format serialisation
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Convert an arbitrary string to a valid Prometheus metric name segment."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")


def _to_prometheus_text(store: MetricsStore, service_start_epoch: float) -> str:
    """
    Render the MetricsStore snapshot as Prometheus exposition text format.

    Only numeric (int/float) and boolean values are emitted.
    Metric names: amoprof_<collector>_<key>
    """
    data, last_scrape, scrape_count = store.snapshot()
    lines: list[str] = [
        "# HELP amoprof_service_up 1 if the collection service is running",
        "# TYPE amoprof_service_up gauge",
        "amoprof_service_up 1",
        "# HELP amoprof_service_start_time_seconds Unix epoch when the service started",
        "# TYPE amoprof_service_start_time_seconds gauge",
        f"amoprof_service_start_time_seconds {service_start_epoch:.3f}",
        "# HELP amoprof_service_last_scrape_time_seconds Unix epoch of the last completed collection cycle",
        "# TYPE amoprof_service_last_scrape_time_seconds gauge",
        f"amoprof_service_last_scrape_time_seconds {last_scrape:.3f}",
        "# HELP amoprof_service_scrape_total Total number of completed collection cycles",
        "# TYPE amoprof_service_scrape_total counter",
        f"amoprof_service_scrape_total {scrape_count}",
    ]
    for collector_name, summary in sorted(data.items()):
        safe_coll = _safe_name(collector_name)
        for key, val in sorted(summary.items()):
            if isinstance(val, bool):
                val = 1 if val else 0
            if not isinstance(val, (int, float)):
                continue
            if val != val:   # NaN guard
                continue
            metric_name = f"amoprof_{safe_coll}_{_safe_name(key)}"
            lines += [
                f"# HELP {metric_name} {collector_name}: {key}",
                f"# TYPE {metric_name} gauge",
                f"{metric_name} {val}",
            ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP handler and server
# ---------------------------------------------------------------------------

def _make_handler(store: MetricsStore, service_start_epoch: float):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/metrics", "/metrics/"):
                body = _to_prometheus_text(store, service_start_epoch)
                self._respond(200, "text/plain; version=0.0.4; charset=utf-8", body)
            elif self.path in ("/healthz", "/health"):
                self._respond(200, "text/plain", "ok\n")
            else:
                self._respond(404, "text/plain", "Not found\n")

        def _respond(self, code: int, content_type: str, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt, *args):
            pass  # suppress default access log

    return _Handler


class PrometheusServer:
    """Runs the /metrics HTTP server in a background daemon thread."""

    def __init__(self, store: MetricsStore, port: int, host: str = "0.0.0.0",
                 service_start_epoch: float = 0.0) -> None:
        self._store = store
        self._port = port
        self._host = host
        self._start_epoch = service_start_epoch
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self._store, self._start_epoch)
        self._server = HTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="amoprof-prometheus",
            daemon=True,
        )
        self._thread.start()
        log.info("Prometheus /metrics listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Collector factory
# ---------------------------------------------------------------------------

def _build_collectors(
    args: Any,
    raw_dir: Path,
    enabled: set[str],
    interval_s: float,
    duration_s: float,
    use_sudo: bool,
) -> dict[str, Any]:
    """Instantiate only the collectors present in *enabled*."""
    from .collectors import (
        AMDuProfPcmMemoryCollector,
        BiolatencyCollector,
        BpftraceCollector,
        DramMonitor,
        GpuMonitor,
        IostatMonitor,
        NvlinkPcieMonitor,
        NvmeDriverMonitor,
        NvmeSmartMonitor,
        PcmMemoryCollector,
        PerfStatCollector,
        PowerMonitor,
        SsdHardwareMonitor,
        VmstatMonitor,
    )
    from .collectors_extra import (
        BiosnoopCollector,
        BlktraceCollector,
        DiscardStatsMonitor,
        SwapStormMonitor,
    )

    from .cli import _resolve_ssd_devices, _ssd_device_slug

    ssd_raw       = getattr(args, "ssd_device", "/dev/nvme0n1")
    ssd_devices   = _resolve_ssd_devices(ssd_raw) or ["/dev/nvme0n1"]
    args.ssd_devices = ssd_devices
    args.ssd_device  = ssd_devices[0]
    ssd_device    = ssd_devices[0]
    hicache_path = getattr(args, "hicache_path", "/mnt/sglang_dv3")
    pid          = getattr(args, "pid",          None)
    sglang_port  = getattr(args, "sglang_port",  None)
    sglang_host  = getattr(args, "sglang_host",  "127.0.0.1")
    vllm_port    = getattr(args, "vllm_port",    None)
    vllm_host    = getattr(args, "vllm_host",    "127.0.0.1")
    dram_tool    = _resolve_dram_tool(args)

    factories: dict[str, Any] = {
        "vmstat":       lambda: VmstatMonitor(interval_s),
        "nvlink_pcie":  lambda: NvlinkPcieMonitor(max(interval_s, 1.0)),
        "gpu":          lambda: GpuMonitor(interval_s),
        "dram":         lambda: DramMonitor(interval_s),
        "power":        lambda: PowerMonitor(interval_s, use_sudo=use_sudo),
        "pcm":          lambda: PcmMemoryCollector(
            interval_s,
            binary=(None if dram_tool == "perf-imc" else getattr(args, "intel_pcm_memory_bin", None)),
            force_perf_imc=(dram_tool == "perf-imc"),
            use_sudo=use_sudo,
            work_dir=raw_dir,
        ),
        "swap_storm":   lambda: SwapStormMonitor(interval_s, work_dir=raw_dir),
        "amduprof_pcm": lambda: AMDuProfPcmMemoryCollector(
            duration_s=getattr(args, "amduprof_duration_s", None) or duration_s,
            interval_s=interval_s,
            binary=getattr(args, "amduprof_pcm_bin",
                           "/opt/AMDuProf_5.2-606/bin/AMDuProfPcm"),
            output_csv=str(raw_dir / "amduprof_pcm_raw.csv"),
            extra_args=getattr(args, "amduprof_extra_arg", []) or [],
            use_sudo=use_sudo,
        ),
    }
    # Per-SSD factories. Primary device (index 0) keeps canonical keys; extra
    # devices get "__<slug>" suffixed collector keys and filename suffixes so
    # their outputs never overwrite each other.
    def _mk_iostat(dev):        return lambda: IostatMonitor(dev, interval_s)
    def _mk_smart(dev):         return lambda: NvmeSmartMonitor(dev, poll_s=max(int(interval_s), 1), use_sudo=use_sudo)
    def _mk_ssd_hw(dev):        return lambda: SsdHardwareMonitor(dev, hicache_path=hicache_path, use_sudo=use_sudo)
    def _mk_biolat(dev):        return lambda: BiolatencyCollector(dev, duration_s=max(int(duration_s), 1), use_sudo=use_sudo)
    def _mk_nvme_drv(dev):      return lambda: NvmeDriverMonitor(dev, interval_s)
    def _mk_discard(dev, sfx):  return lambda: DiscardStatsMonitor(dev, interval_s, work_dir=raw_dir, filename_suffix=sfx)
    def _mk_blktrace(dev, sfx): return lambda: BlktraceCollector(
        dev, duration_s=max(int(duration_s), 1), work_dir=raw_dir,
        blktrace_bin=getattr(args, "blktrace_bin", "blktrace"),
        blkparse_bin=getattr(args, "blkparse_bin", "blkparse"),
        use_sudo=use_sudo, filename_suffix=sfx,
    )
    def _mk_biosnoop(dev, sfx): return lambda: BiosnoopCollector(
        duration_s=max(int(duration_s), 1), work_dir=raw_dir, device=dev,
        binary=getattr(args, "biosnoop_bin", "biosnoop"),
        use_sudo=use_sudo, filename_suffix=sfx,
    )
    for _idx, _dev in enumerate(ssd_devices):
        _slug = _ssd_device_slug(_dev)
        _ksfx = "" if _idx == 0 else ("__" + _slug)
        _fsfx = "" if _idx == 0 else _slug
        factories["iostat" + _ksfx]            = _mk_iostat(_dev)
        factories["smart" + _ksfx]             = _mk_smart(_dev)
        factories["ssd_hw" + _ksfx]            = _mk_ssd_hw(_dev)
        factories["biolatency" + _ksfx]        = _mk_biolat(_dev)
        factories["nvme_driver" + _ksfx]       = _mk_nvme_drv(_dev)
        factories["discard" + _ksfx]           = _mk_discard(_dev, _fsfx)
        factories["blktrace" + _ksfx]          = _mk_blktrace(_dev, _fsfx)
        factories["biosnoop" + _ksfx]          = _mk_biosnoop(_dev, _fsfx)

    if sglang_port is not None:
        from .bench_swebench import SGLangMetricsSampler
        factories["sglang"] = lambda: SGLangMetricsSampler(
            sglang_port, interval_s,
            host=sglang_host,
            debug=bool(getattr(args, "debug_sglang", False)),
            debug_path=str(raw_dir / "sglang_debug.log"),
        )
    if vllm_port is not None:
        if sglang_port is not None:
            log.warning("service: both --sglang-port and --vllm-port set; using vLLM sampler")
        from .bench_vllm import VLLMMetricsSampler
        factories["sglang"] = lambda: VLLMMetricsSampler(
            vllm_port, interval_s,
            host=vllm_host,
            lmcache_port=getattr(args, "lmcache_port", None),
            lmcache_host=getattr(args, "lmcache_host", "127.0.0.1"),
            lmcache_bytes_per_token=_float_or_none(getattr(args, "lmcache_bytes_per_token", None)),
            lmcache_max_disk_gb=_float_or_none(getattr(args, "lmcache_max_disk_gb", None)),
            debug=bool(getattr(args, "debug_vllm", False)),
            debug_path=str(raw_dir / "vllm_debug.log"),
        )

    if pid is not None:
        factories["perf"] = lambda: PerfStatCollector(pid, use_sudo=use_sudo)
        factories["bpf"]  = lambda: BpftraceCollector(
            pid, duration_s=max(int(duration_s), 1),
            work_dir=raw_dir, use_sudo=use_sudo,
        )

    # Expand SSD-scoped collector types (iostat/smart/ssd_hw/biolatency/
    # nvme_driver/discard/blktrace/biosnoop) to per-device keys for devices 2+.
    _ssd_scoped = {"iostat", "smart", "ssd_hw", "biolatency", "nvme_driver",
                   "discard", "blktrace", "biosnoop"}
    _expanded_enabled = set(enabled)
    if len(ssd_devices) > 1:
        for _base in list(enabled):
            if _base in _ssd_scoped:
                for _dev in ssd_devices[1:]:
                    _expanded_enabled.add(_base + "__" + _ssd_device_slug(_dev))

    built: dict[str, Any] = {}
    for name in sorted(_expanded_enabled):
        if name not in factories:
            log.warning("service: unknown collector %r - skipped", name)
            continue
        try:
            built[name] = factories[name]()
        except Exception as exc:
            log.warning("service: failed to build %r: %s - skipped", name, exc)

    return built


# ---------------------------------------------------------------------------
# Service main loop
# ---------------------------------------------------------------------------

def run_service(args: Any) -> int:
    """
    Long-running collection service.

    Each cycle (length = scrape_duration_s):
      1. Build fresh collector instances for the enabled set.
      2. Start them all.
      3. Sleep for scrape_duration_s (interruptible).
      4. Stop each collector, push its summary into MetricsStore.
      5. Repeat until SIGINT / SIGTERM.

    The Prometheus HTTP server runs throughout in a background thread and
    always serves the most recently completed cycle's data.
    """
    import signal as _signal
    from .blktrace_service import BlktraceServicePoller

    scrape_s:   float = float(getattr(args, "scrape_duration_s", 30.0))
    port:       int   = int(getattr(args,   "metrics_port",      9101))
    host:       str   = getattr(args,       "metrics_host",      "0.0.0.0")
    use_sudo:   bool  = not getattr(args,   "no_sudo",           False)
    interval_s: float = float(getattr(args, "interval_s",        1.0))
    blkparse_interval_s: float = float(getattr(args, "blkparse_interval_s", 10.0))

    # ---- resolve enabled collectors ----------------------------------------
    collectors_arg: str = (getattr(args, "collectors", "") or "").strip()
    if collectors_arg.lower() in ("", "all"):
        enabled: set[str] = set(ALWAYS_COLLECTORS)
    else:
        enabled = {c.strip() for c in collectors_arg.split(",") if c.strip()}
        unknown = enabled - set(AVAILABLE_COLLECTORS)
        if unknown:
            log.warning("service: unrecognised collector(s) %s - ignored",
                        sorted(unknown))
            enabled -= unknown

    if getattr(args, "enable_blktrace",     False): enabled.add("blktrace")
    if getattr(args, "enable_biosnoop",     False): enabled.add("biosnoop")

    # Generic service-mode DRAM PMU switch.  This mirrors collect-mode
    # --enable-dram so offline service cycles can contain timestamped CPU DRAM
    # bandwidth rows even when no Prometheus server is available.
    if getattr(args, "enable_amduprof_pcm", False):
        setattr(args, "enable_dram", True)
        if not getattr(args, "dram_tool", "auto") or getattr(args, "dram_tool", "auto") == "auto":
            setattr(args, "dram_tool", "amduprof")
    if getattr(args, "enable_dram", False):
        _dram_tool = _resolve_dram_tool(args)
        if _dram_tool == "amduprof":
            enabled.add("amduprof_pcm")
        elif _dram_tool in {"intel-pcm", "perf-imc"}:
            enabled.add("pcm")
        elif _dram_tool != "none":
            log.warning("service: unsupported --dram-tool %r; DRAM PMU collector disabled", _dram_tool)

    if getattr(args, "enable_all",          False):
        enabled |= {"blktrace", "biosnoop"}
        _dram_tool_all = _resolve_dram_tool(args)
        if _dram_tool_all == "amduprof":
            enabled.add("amduprof_pcm")
        elif _dram_tool_all in {"intel-pcm", "perf-imc"}:
            enabled.add("pcm")
    if getattr(args, "sglang_port",         None) is not None: enabled.add("sglang")
    if getattr(args, "vllm_port",           None) is not None: enabled.add("sglang")
    if getattr(args, "pid",                 None) is not None: enabled |= {"perf", "bpf"}
    if getattr(args, "enable_nsys",         False): enabled.add("nsys")

    output_base = Path(
        getattr(args, "output_dir",
                os.environ.get("AMOPROF_OUTPUT_DIR", "./amoprof_results"))
    ).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    store = MetricsStore()
    service_start = time.time()

    prom = PrometheusServer(store, port=port, host=host,
                            service_start_epoch=service_start)
    prom.start()

    sep = "=" * 68
    print(
        f"\n{sep}\n"
        f"  AMOprof collection service started\n"
        f"  Collectors     : {', '.join(sorted(enabled))}\n"
        f"  Scrape window  : {scrape_s:.0f} s\n"
        f"  Poll interval  : {interval_s:.1f} s\n"
        f"  Metrics URL    : http://{host}:{port}/metrics\n"
        f"  Health URL     : http://{host}:{port}/healthz\n"
        f"  Output base    : {output_base}\n"
        f"{sep}\n",
        flush=True,
    )

    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        log.info("service: received signal %s - stopping", sig)
        stop_event.set()

    _signal.signal(_signal.SIGINT,  _handle_signal)
    _signal.signal(_signal.SIGTERM, _handle_signal)

    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        cycle_start_epoch = time.time()
        cycle_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(cycle_start_epoch))
        cycle_dir = output_base / f"service_cycle_{cycle_ts}"
        raw_dir = cycle_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        log.info("service: cycle %d starting (%s; epoch=%.3f)", cycle, cycle_ts, cycle_start_epoch)

        collectors = _build_collectors(
            args=args,
            raw_dir=raw_dir,
            enabled=enabled,
            interval_s=interval_s,
            duration_s=scrape_s,
            use_sudo=use_sudo,
        )

        started: set[str] = set()
        for name, c in collectors.items():
            try:
                c.start()
                started.add(name)
            except Exception as exc:
                log.warning("service cycle %d: start %r failed: %s", cycle, name, exc)

        # If blktrace is running, start the incremental blkparse poller.
        # One poller per blktrace collector so multi-SSD runs each parse
        # their own binary trace stream.
        blktrace_pollers: list[BlktraceServicePoller] = []
        for _bt_name, _bt_coll in collectors.items():
            if not (_bt_name == "blktrace" or _bt_name.startswith("blktrace__")):
                continue
            if _bt_name not in started or _bt_coll is None:
                continue
            _trace_prefix = getattr(_bt_coll, "_trace_prefix", None)
            _csv_path     = getattr(_bt_coll, "_csv_path", None)
            _blkparse_bin = getattr(_bt_coll, "blkparse_bin", "blkparse")
            if _trace_prefix is None or _csv_path is None:
                continue
            _poller = BlktraceServicePoller(
                trace_prefix=_trace_prefix,
                csv_path=_csv_path,
                store=store,
                blkparse_bin=_blkparse_bin,
                interval_s=blkparse_interval_s,
                use_sudo=use_sudo,
            )
            _poller.start()
            blktrace_pollers.append(_poller)
            log.info("service: blktrace poller started for %s (interval=%.0fs)",
                     _bt_name, blkparse_interval_s)

        stop_event.wait(timeout=scrape_s)

        # Stop the blktrace pollers first so each does a final parse before
        # the blktrace binary collectors are stopped and files are closed.
        for _poller in blktrace_pollers:
            try:
                _poller.stop()
            except Exception as _e:
                log.warning("service: blktrace poller stop failed: %s", _e)
        if blktrace_pollers:
            log.info("service: %d blktrace poller(s) stopped", len(blktrace_pollers))

        cycle_end_epoch = time.time()
        cycle_summaries: dict[str, dict[str, Any]] = {}
        raw_paths: dict[str, str] = {}
        for name, c in collectors.items():
            if name not in started:
                continue
            try:
                result = c.stop()
            except Exception as exc:
                result = {f"{name}_available": False, f"{name}_error": str(exc)}
            if result is None:
                result = {f"{name}_available": False}
            result = dict(result)
            result.setdefault("collection_start_time_seconds", round(cycle_start_epoch, 6))
            result.setdefault("collection_end_time_seconds", round(cycle_end_epoch, 6))
            result.setdefault("collection_duration_s", round(cycle_end_epoch - cycle_start_epoch, 6))
            cycle_summaries[name] = result
            store.update(name, result)

        try:
            _persist_service_cycle(
                cycle_dir=cycle_dir, raw_dir=raw_dir, cycle=cycle, enabled=enabled,
                collectors=collectors, summaries=cycle_summaries, raw_paths=raw_paths,
                start_epoch=cycle_start_epoch, end_epoch=cycle_end_epoch,
                interval_s=interval_s, scrape_s=scrape_s, args=args,
            )
        except Exception as exc:
            log.warning("service: failed to persist timestamped local cycle %d: %s", cycle, exc)

        log.info("service: cycle %d complete, store covers %d collector(s); local raw=%s",
                 cycle, len(store.snapshot()[0]), raw_dir)

        if stop_event.is_set():
            break

        # brief gap so successive cycles never share the same timestamp
        stop_event.wait(timeout=0.5)

    prom.stop()
    log.info("service: shut down after %d cycle(s)", cycle)
    print(f"\nAMOprof service stopped after {cycle} cycle(s).\n", flush=True)
    return 0
