"""
collectors.py — Background metric collectors.
Unchanged from the original design; consolidated here as part of the package.
"""

from __future__ import annotations
import logging
import os, re, json, time, threading, subprocess, statistics, csv
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger("amoprof.collectors")


# ── sudo helpers ─────────────────────────────────────────────────────────────
# Some system-level collectors need elevated privileges on production nodes
# (nvme SMART/admin commands, eBPF/BCC tools, perf PID attach, bpftrace,
# AMDuProf MSR/PMU reads, and IPMI on many systems).  Use non-interactive sudo
# so AMOprof never hangs waiting for a password prompt.  Root runs commands
# directly.
def _sudo_cmd(cmd, use_sudo: bool = True, preserve_env: bool = False):
    if use_sudo and os.geteuid() != 0:
        return ["sudo", "-n"] + (["-E"] if preserve_env else []) + list(cmd)
    return list(cmd)

def _sudo_check_output(cmd, use_sudo: bool = True, preserve_env: bool = False, **kwargs):
    return subprocess.check_output(_sudo_cmd(cmd, use_sudo, preserve_env), **kwargs)

def _sudo_run(cmd, use_sudo: bool = True, preserve_env: bool = False, **kwargs):
    return subprocess.run(_sudo_cmd(cmd, use_sudo, preserve_env), **kwargs)

def _sudo_popen(cmd, use_sudo: bool = True, preserve_env: bool = False, **kwargs):
    return subprocess.Popen(_sudo_cmd(cmd, use_sudo, preserve_env), **kwargs)


# ── IostatMonitor ─────────────────────────────────────────────────────────────

class IostatMonitor:
    def __init__(self, device: str, interval: float = 1.0):
        self.device   = os.path.basename(device)
        self.interval = interval
        self.samples: list[dict] = []
        self._proc  = None
        self._thread= None
        self._stop  = threading.Event()
        self._cmd   = []        # ← add
        self._stderr = ""       # ← add


    def start(self):
        self._stop.clear(); self.samples.clear()
        self._proc = subprocess.Popen(
            ["iostat", "-x", "-d", str(self.interval), self.device],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()


    def _f(self, s):
        try: return float(s.replace(",", "."))
        except: return 0.0

    def _read(self):
        hdrs = []
        for line in self._proc.stdout:
            if self._stop.is_set(): break
            line = line.strip()
            if not line: continue
            if line.startswith("Device"): hdrs = line.split(); continue
            if self.device not in line: continue
            parts = line.split()
            if len(parts) != len(hdrs): continue
            d = dict(zip(hdrs, parts))
            # iostat may report BW in either MB/s or kB/s depending on version /
            # locale. Detect which key is present (substring match against keys,
            # not against d itself — `"rMB" in d` checks key equality, not
            # substring) and convert kB→MB when needed.
            rMB_key = next((k for k in d if k.startswith("rMB")), None)
            rkB_key = next((k for k in d if k.startswith("rkB")), None)
            wMB_key = next((k for k in d if k.startswith("wMB")), None)
            wkB_key = next((k for k in d if k.startswith("wkB")), None)
            r_bw = self._f(d.get(rMB_key, "0")) if rMB_key else \
                   self._f(d.get(rkB_key, "0")) / 1024.0
            w_bw = self._f(d.get(wMB_key, "0")) if wMB_key else \
                   self._f(d.get(wkB_key, "0")) / 1024.0
            self.samples.append({
                "ts":    time.time(),
                "rMBs":  r_bw,
                "wMBs":  w_bw,
                "riops": self._f(d.get("r/s",    d.get("tps","0"))),
                "wiops": self._f(d.get("w/s",    "0")),
                "rawt":  self._f(d.get("r_await",d.get("await","0"))),
                "wawt":  self._f(d.get("w_await","0")),
                "avgqu": self._f(d.get("avgqu-sz",d.get("aqu-sz","0"))),
                "util":  self._f(d.get("%util","0")),
            })

    def stop(self):
        self._stop.set()
        _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or 5.0)
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=min(_timeout, 5.0))
            except Exception:
                try: self._proc.kill()
                except Exception:
                    self._stderr = "ISSUE 1"
            # Drain stderr so we can see WHY iostat produced nothing
            try:
                if self._proc.stderr is not None:
                    self._stderr = self._proc.stderr.read() or ""
            except Exception:
                self._stderr = "ISSUE 1"
                pass
        if self._thread: self._thread.join(timeout=min(_timeout, 3.0))
        if self._stderr:
            log.warning("IostatMonitor: stderr: %s", self._stderr[-500:])
        return self.summarise()

    def summarise(self) -> dict:
        if not self.samples:
            return {"iostat_available": False}
        def m(k):  return round(statistics.mean(s[k] for s in self.samples), 3)
        def pk(k): return round(max(s[k] for s in self.samples), 3)
        def p(k, pct):
            xs = sorted(s[k] for s in self.samples)
            return round(xs[int(len(xs)*pct/100)], 3) if xs else 0.0
        return {
            "iostat_available":   True,
            "read_bw_mb_mean":    m("rMBs"),  "read_bw_mb_peak":   pk("rMBs"),
            "write_bw_mb_mean":   m("wMBs"),  "write_bw_mb_peak":  pk("wMBs"),
            "read_iops_mean":     m("riops"), "read_iops_peak":    pk("riops"),
            "write_iops_mean":    m("wiops"), "write_iops_peak":   pk("wiops"),
            "r_await_ms_mean":    m("rawt"),
            "r_await_ms_p99":     p("rawt", 99),
            "r_await_ms_p999":    p("rawt", 99.9),
            "w_await_ms_mean":    m("wawt"),  "w_await_ms_p99":    p("wawt",99),
            "avgqu_sz_mean":      m("avgqu"), "avgqu_sz_p50":      p("avgqu", 50),
            "avgqu_sz_p95":       p("avgqu", 95), "avgqu_sz_p99":      p("avgqu", 99),
            "avgqu_sz_peak":      pk("avgqu"), "util_pct_mean":       m("util"),
            "util_pct_peak":      pk("util"),
            "queue_depth_source": "iostat_aqu_sz_or_avgqu_sz",
            "iostat_samples":     len(self.samples),
        }


# ── NvmeSmartMonitor ──────────────────────────────────────────────────────────

class NvmeSmartMonitor:
    def __init__(self, device: str, poll_s: int = 15, use_sudo: bool = True):
        self.device = device; self.poll_s = poll_s; self.use_sudo = use_sudo
        self._start = {}; self._samples = []
        self._thread= None; self._stop = threading.Event()

    def _smart(self) -> dict:
        try:
            return json.loads(_sudo_check_output(
                ["nvme","smart-log",self.device,"-o","json"], self.use_sudo,
                stderr=subprocess.DEVNULL, timeout=10))
        except: return {}

    def _host_gb(self, s): return s.get("data_units_written",0)*512*1024/(1024**3)
    def _nand_gb(self, s): return (s["nand_bytes_written"]/(1024**3)
                                   if "nand_bytes_written" in s else self._host_gb(s))
    def _temp(self, s):    t=s.get("temperature",0); return int(t-273) if t>200 else int(t)

    def start(self):
        self._stop.clear(); self._start = self._smart()
        self._samples = [("start", time.time(), self._start)]
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.poll_s):
            self._samples.append(("poll", time.time(), self._smart()))

    def stop(self) -> dict:
        self._stop.set()
        if self._thread: self._thread.join(timeout=5)
        end = self._smart(); self._samples.append(("end", time.time(), end))
        if not self._start: return {}
        host  = max(0.0, self._host_gb(end) - self._host_gb(self._start))
        nand  = max(0.0, self._nand_gb(end) - self._nand_gb(self._start))
        temps = [self._temp(s) for _,_,s in self._samples if s]
        return {
            "host_written_gb": round(host, 3),
            "nand_written_gb": round(nand, 3),
            "waf":             round(nand/host, 3) if host > 0.01 else 0.0,
            "temp_start_c":    self._temp(self._start),
            "temp_end_c":      self._temp(end),
            "temp_peak_c":     max(temps) if temps else 0,
        }


# ── BiolatencyCollector ────────────────────────────────────────────────────────

class BiolatencyCollector:
    def __init__(self, device: str, duration_s: int = 60, use_sudo: bool = True):
        self.device=os.path.basename(device); self.duration_s=duration_s; self.use_sudo=use_sudo
        self._proc=None; self._out=""; self._done=threading.Event()

    def start(self):
        self._done.clear(); self._out=""
        try:
            self._proc = _sudo_popen(
                ["biolatency-bpfcc","-d",self.device,str(self.duration_s)], self.use_sudo,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except FileNotFoundError:
            self._done.set()

    def _read(self):
        if self._proc: self._out,_=self._proc.communicate()
        self._done.set()

    def stop(self) -> dict:
        _interrupted = bool(getattr(self, "_amoprof_interrupted", False))
        _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (3.0 if _interrupted else self.duration_s + 5))
        if self._proc and self._proc.poll() is None:
            try: self._proc.terminate()
            except Exception: pass
        self._done.wait(timeout=_timeout)
        if self._proc and self._proc.poll() is None:
            try: self._proc.kill()
            except Exception: pass
        buckets=[]
        for line in self._out.splitlines():
            m=re.match(r"\s*(\d+)\s*->\s*(\d+)\s*:\s*(\d+)",line)
            if m:
                lo,hi,cnt=int(m.group(1)),int(m.group(2)),int(m.group(3))
                buckets.extend([(lo+hi)/2]*cnt)
        if not buckets: return {"biolatency_available":False}
        buckets.sort(); n=len(buckets)
        return {"biolatency_available":True,
                "read_lat_p50_us":  round(buckets[int(n*0.50)],1),
                "read_lat_p99_us":  round(buckets[int(n*0.99)],1),
                "read_lat_p999_us": round(buckets[min(int(n*0.999),n-1)],1),
                "read_lat_samples": n}


# ── GpuMonitor ────────────────────────────────────────────────────────────────


class PowerMonitor:
    """
    Records system power consumption from every available source during a
    workload run and returns a unified power summary.

    Sources probed (in priority order per domain):

    GPU power
      • nvidia-smi  per-GPU power.draw + power.limit  (primary)
      • dcgmi dmon  per-GPU power counters            (DGX fallback)

    CPU / package power  (RAPL — Running Average Power Limit)
      • /sys/class/powercap/intel-rapl/*/energy_uj    (Intel, read directly)
      • /sys/class/powercap/intel-rapl-mmio/*/energy_uj (alternate path)
      • perf stat -e power/energy-pkg/               (fallback)

    DRAM power  (RAPL)
      • /sys/class/powercap/intel-rapl/*/intel-rapl:*/energy_uj  (sub-domain)

    System / PSU power
      • ipmitool dcmi power reading                  (BMC, most accurate)
      • ipmitool sdr type "Power Supply"             (PSU rails fallback)

    Metrics reported
    ─────────────────────────────────────────────────────────────────────
    gpu_power_w_mean / _peak / _total_wh             per-GPU average / peak / energy
    gpu_power_all_w_mean                             sum across all GPUs
    cpu_package_power_w_mean / _total_wh             CPU package (RAPL)
    dram_power_w_mean / _total_wh                    DRAM subsystem (RAPL)
    system_power_w_mean / _peak / _total_wh          BMC/IPMI total chassis
    total_system_power_w_mean                        gpu + cpu + dram combined estimate
    power_efficiency_tok_per_wh                      tokens / watt-hour (if tok count known)
    power_sources                                    comma list of active sources
    """

    _RAPL_BASE = "/sys/class/powercap"

    def __init__(self, interval_s: float = 1.0, use_sudo: bool = True):
        self.interval    = interval_s
        self.use_sudo   = use_sudo
        self.samples: list[dict] = []
        self._t_start    = 0.0
        self._t_end      = 0.0
        self._rapl_start: dict[str, int] = {}
        self._rapl_end:   dict[str, int] = {}
        self._thread     = None
        self._stop       = threading.Event()
        self._sources: list[str] = []

    # ── Source detection ──────────────────────────────────────────────────────

    def _has_nvidia(self) -> bool:
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _has_dcgmi(self) -> bool:
        try:
            subprocess.check_output(["dcgmi", "version"], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _has_ipmi(self) -> bool:
        try:
            _sudo_check_output(
                ["ipmitool", "dcmi", "power", "reading"], self.use_sudo,
                stderr=subprocess.DEVNULL, timeout=5)
            return True
        except Exception:
            return False

    def _rapl_paths(self) -> dict[str, str]:
        """Return {domain_label: energy_uj_path} for all readable RAPL domains."""
        paths: dict[str, str] = {}
        for base in [self._RAPL_BASE + "/intel-rapl",
                     self._RAPL_BASE + "/intel-rapl-mmio"]:
            base_p = Path(base)
            if not base_p.exists():
                continue
            for zone in sorted(base_p.iterdir()):
                ename = zone / "name"
                epath = zone / "energy_uj"
                if not epath.exists():
                    continue
                try:
                    label = ename.read_text().strip() if ename.exists() else zone.name
                    paths[label] = str(epath)
                except Exception:
                    pass
                # sub-domains (DRAM, uncore, etc.)
                for sub in zone.iterdir():
                    sub_e = sub / "energy_uj"
                    sub_n = sub / "name"
                    if sub_e.exists():
                        try:
                            slabel = (sub_n.read_text().strip()
                                      if sub_n.exists() else sub.name)
                            paths[f"{label}/{slabel}"] = str(sub_e)
                        except Exception:
                            pass
        return paths

    def _read_rapl_uj(self, paths: dict[str, str]) -> dict[str, int]:
        out = {}
        for label, path in paths.items():
            try:
                out[label] = int(Path(path).read_text().strip())
            except Exception:
                out[label] = 0
        return out

    def _rapl_to_watts(self, start: dict[str, int], end: dict[str, int],
                       elapsed_s: float) -> dict[str, float]:
        """Convert RAPL energy delta (µJ) to average power (W)."""
        watts: dict[str, float] = {}
        if elapsed_s <= 0:
            return watts
        for k in start:
            if k in end:
                delta_uj = end[k] - start[k]
                # Handle counter wrap (max ~2^32 µJ on older CPUs)
                if delta_uj < 0:
                    try:
                        max_uj = int(Path(str(Path(self._rapl_paths().get(k,""))
                                            .parent / "max_energy_range_uj")
                                         ).read_text().strip())
                        delta_uj += max_uj
                    except Exception:
                        delta_uj = abs(delta_uj)
                watts[k] = round(delta_uj / 1e6 / elapsed_s, 2)
        return watts

    # ── nvidia-smi polling (background thread) ────────────────────────────────

    def _poll_nvidia(self):
        """Poll all GPUs: index,power.draw,power.limit per sample."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={int(self.interval * 1000)}",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        self.samples.append({
                            "ts":         time.time(),
                            "source":     "nvidia-smi",
                            "gpu_index":  int(parts[0]),
                            "gpu_power":  float(parts[1]) if parts[1] != "N/A" else 0.0,
                            "gpu_limit":  float(parts[2]) if parts[2] != "N/A" else 0.0,
                            "gpu_temp_c": float(parts[3]) if len(parts) > 3 and parts[3] != "N/A" else 0.0,
                        })
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass

    def _poll_ipmi(self):
        """Poll IPMI system power every ~5 seconds (slow command)."""
        while not self._stop.wait(5.0):
            try:
                out = _sudo_check_output(
                    ["ipmitool", "dcmi", "power", "reading"], self.use_sudo,
                    stderr=subprocess.DEVNULL, timeout=8, text=True)
                for line in out.splitlines():
                    # "Instantaneous power reading:                   450 Watts"
                    if "Instantaneous" in line and "Watts" in line:
                        w = float(re.search(r"([\d.]+)\s+Watts", line).group(1))
                        self.samples.append({
                            "ts": time.time(), "source": "ipmi", "system_power": w})
                        break
            except Exception:
                break   # stop polling if IPMI becomes unavailable

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._sources.clear()
        self._t_start = time.time()

        # Snapshot RAPL energy counters before workload
        self._rapl_paths_map = self._rapl_paths()
        if self._rapl_paths_map:
            self._rapl_start = self._read_rapl_uj(self._rapl_paths_map)
            self._sources.append("rapl")

        threads = []

        if self._has_nvidia():
            self._sources.append("nvidia-smi")
            t = threading.Thread(target=self._poll_nvidia, daemon=True)
            t.start(); threads.append(t)

        if self._has_ipmi():
            self._sources.append("ipmi")
            t = threading.Thread(target=self._poll_ipmi, daemon=True)
            t.start(); threads.append(t)

        self._threads = threads

    def stop(self, tokens_generated: int = 0) -> dict:
        self._stop.set()
        for t in getattr(self, "_threads", []):
            t.join(timeout=10)

        self._t_end = time.time()
        elapsed_s = max(self._t_end - self._t_start, 0.001)

        result: dict = {
            "power_sources":            ",".join(self._sources) or "none",
            "power_elapsed_s":          round(elapsed_s, 2),
        }

        # ── GPU power (nvidia-smi) ────────────────────────────────────────
        gpu_samples = [s for s in self.samples if s.get("source") == "nvidia-smi"]
        if gpu_samples:
            # Group by GPU index
            gpu_indices = sorted(set(s["gpu_index"] for s in gpu_samples))
            per_gpu_mean = []
            per_gpu_peak = []
            per_gpu_limit = []
            per_gpu_temp  = []
            for idx in gpu_indices:
                gs = [s["gpu_power"] for s in gpu_samples if s["gpu_index"] == idx]
                ts = [s["gpu_temp_c"] for s in gpu_samples if s["gpu_index"] == idx]
                ls = [s["gpu_limit"]  for s in gpu_samples if s["gpu_index"] == idx]
                if gs:
                    per_gpu_mean.append(round(statistics.mean(gs), 2))
                    per_gpu_peak.append(round(max(gs), 2))
                    per_gpu_limit.append(round(statistics.mean(ls), 2))
                    per_gpu_temp.append(round(max(ts), 1) if ts else 0.0)

            all_gpu_power = [s["gpu_power"] for s in gpu_samples]
            gpu_all_mean  = round(sum(
                statistics.mean(s["gpu_power"] for s in gpu_samples
                                if s["gpu_index"] == idx)
                for idx in gpu_indices), 2)

            result.update({
                "gpu_count":                  len(gpu_indices),
                "gpu_power_w_mean":           round(statistics.mean(all_gpu_power), 2),
                "gpu_power_w_peak":           round(max(all_gpu_power), 2),
                "gpu_power_all_w_mean":       gpu_all_mean,
                "gpu_power_limit_w":          round(statistics.mean(per_gpu_limit), 2) if per_gpu_limit else 0.0,
                "gpu_utilisation_pct":        round(gpu_all_mean / max(sum(per_gpu_limit), 1) * 100, 1),
                "gpu_temp_peak_c":            max(per_gpu_temp) if per_gpu_temp else 0.0,
                # Energy = mean_power_W × duration_h
                "gpu_energy_wh":              round(gpu_all_mean * elapsed_s / 3600, 4),
                "gpu_power_per_gpu_mean_w":   per_gpu_mean,
                "gpu_power_per_gpu_peak_w":   per_gpu_peak,
            })

        # ── CPU + DRAM power (RAPL) ───────────────────────────────────────
        if self._rapl_paths_map:
            self._rapl_end = self._read_rapl_uj(self._rapl_paths_map)
            rapl_w = self._rapl_to_watts(self._rapl_start, self._rapl_end, elapsed_s)

            # Aggregate: "package-N" → CPU, "dram" → DRAM
            cpu_watts  = [w for k, w in rapl_w.items()
                          if "package" in k.lower() and "dram" not in k.lower()]
            dram_watts = [w for k, w in rapl_w.items() if "dram" in k.lower()]

            cpu_total  = round(sum(cpu_watts),  2)
            dram_total = round(sum(dram_watts), 2)

            result.update({
                "cpu_package_power_w_mean":   cpu_total,
                "cpu_package_energy_wh":      round(cpu_total * elapsed_s / 3600, 4),
                "dram_power_w_mean":          dram_total,
                "dram_energy_wh":             round(dram_total * elapsed_s / 3600, 4),
                "rapl_domains":               list(rapl_w.keys()),
                "rapl_power_w":               {k: v for k, v in rapl_w.items()},
            })

        # ── System power (IPMI) ───────────────────────────────────────────
        ipmi_samples = [s["system_power"] for s in self.samples
                        if s.get("source") == "ipmi"]
        if ipmi_samples:
            sys_mean = round(statistics.mean(ipmi_samples), 2)
            sys_peak = round(max(ipmi_samples), 2)
            result.update({
                "system_power_w_mean":        sys_mean,
                "system_power_w_peak":        sys_peak,
                "system_energy_wh":           round(sys_mean * elapsed_s / 3600, 4),
            })

        # ── Combined total estimate ───────────────────────────────────────
        # Prefer IPMI (measures everything). Fallback: GPU + CPU + DRAM.
        if ipmi_samples:
            total_w = result.get("system_power_w_mean", 0.0)
        else:
            total_w = round(
                result.get("gpu_power_all_w_mean", 0.0) +
                result.get("cpu_package_power_w_mean", 0.0) +
                result.get("dram_power_w_mean", 0.0), 2)

        result["total_system_power_w_mean"] = total_w
        result["total_system_energy_wh"]    = round(total_w * elapsed_s / 3600, 4)

        # ── Power efficiency: tokens per watt-hour ────────────────────────
        if tokens_generated > 0 and total_w > 0:
            energy_wh = total_w * elapsed_s / 3600
            result["power_efficiency_tok_per_wh"] = (
                round(tokens_generated / energy_wh, 2) if energy_wh > 0 else 0.0)
        else:
            result["power_efficiency_tok_per_wh"] = 0.0

        return result


# ── DramMonitor ───────────────────────────────────────────────────────────────

class DramMonitor:
    """
    Polls host DRAM usage from /proc/meminfo every interval_s seconds.
    Reports:
      dram_used_gb_mean / peak   — average and peak used DRAM
      dram_free_gb_mean          — average free DRAM
      dram_util_pct_mean         — used / total %
      dram_available_gb_mean     — MemAvailable (excludes cache/buffers)
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        self.samples: list[dict] = []
        self._thread = None
        self._stop   = threading.Event()
        self._t0     = 0.0

    def _read_meminfo(self) -> dict:
        try:
            info = {}
            for line in open("/proc/meminfo").read().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = int(parts[1]) / 1024 / 1024   # kB → GB
                    info[key] = round(val, 3)
            total     = info.get("MemTotal",     0)
            free      = info.get("MemFree",      0)
            available = info.get("MemAvailable", free)
            buffers   = info.get("Buffers",      0)
            cached    = info.get("Cached",       0)
            used      = total - free - buffers - cached
            return {
                "total_gb":     round(total, 2),
                "used_gb":      round(max(used, 0), 2),
                "free_gb":      round(free, 2),
                "available_gb": round(available, 2),
                "util_pct":     round(max(used, 0) / max(total, 1) * 100, 1),
            }
        except Exception:
            return {}

    def _stamp_sample(self, sample: dict) -> dict:
        """Attach absolute and run-relative timestamps to one DRAM sample."""
        ts = time.time()
        sample = dict(sample)
        sample["ts"] = round(ts, 6)
        sample["timestamp_epoch"] = round(ts, 6)
        sample["time_sec"] = round(ts - (self._t0 or ts), 6)
        return sample

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._t0 = time.time()
        # Capture baseline
        s = self._read_meminfo()
        if s:
            self.samples.append(self._stamp_sample(s))
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = self._read_meminfo()
            if s:
                self.samples.append(self._stamp_sample(s))

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"dram_available": False}
        def m(k):  return round(statistics.mean(s[k] for s in self.samples if k in s), 2)
        def pk(k): return round(max(s[k] for s in self.samples if k in s), 2)
        first = self.samples[0]
        last  = self.samples[-1]
        return {
            "dram_available":        True,
            "dram_total_gb":         first.get("total_gb", 0),
            "dram_used_gb_mean":     m("used_gb"),
            "dram_used_gb_peak":     pk("used_gb"),
            "dram_free_gb_mean":     m("free_gb"),
            "dram_available_gb_mean":m("available_gb"),
            "dram_util_pct_mean":    m("util_pct"),
            "dram_util_pct_peak":    pk("util_pct"),
            "dram_delta_gb":         round(last.get("used_gb", 0) -
                                          first.get("used_gb", 0), 2),
            "dram_samples":          len(self.samples),
        }


# ── Enhanced GpuMonitor with per-GPU HBM detail ───────────────────────────────

class GpuMonitor:
    """
    Polls nvidia-smi for per-GPU:
      utilisation (%), HBM used/total (MB), power draw (W), temperature (°C)
    Aggregates to mean/peak across all GPUs.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        self.samples: list[dict] = []
        self._thread = None
        self._stop   = threading.Event()

    def _has_gpu(self) -> bool:
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _gpu_count(self) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True)
            return len(out.strip().splitlines())
        except Exception:
            return 0

    def start(self):
        if not self._has_gpu():
            return
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={int(self.interval * 1000)}",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 7:
                    try:
                        self.samples.append({
                            "gpu_idx":   int(p[0]),
                            "gpu_util":  float(p[1]),
                            "mem_util":  float(p[2]),
                            "mem_used":  float(p[3]),
                            "mem_total": float(p[4]),
                            "power":     float(p[5]) if p[5] != "N/A" else 0.0,
                            "temp_c":    float(p[6]) if p[6] != "N/A" else 0.0,
                        })
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"gpu_available": False}

        def m(k):  return round(statistics.mean(s[k] for s in self.samples), 2)
        def pk(k): return round(max(s[k] for s in self.samples), 2)

        gpu_indices = sorted(set(s["gpu_idx"] for s in self.samples))
        n_gpus      = len(gpu_indices)

        # Per-GPU peak HBM used (MB) and mean utilisation
        per_hbm_peak = []
        per_gpu_util = []
        for idx in gpu_indices:
            gs = [s for s in self.samples if s["gpu_idx"] == idx]
            if gs:
                per_hbm_peak.append(round(max(s["mem_used"] for s in gs), 0))
                per_gpu_util.append(round(statistics.mean(s["gpu_util"] for s in gs), 1))

        all_mem_used  = [s["mem_used"]  for s in self.samples]
        all_mem_total = [s["mem_total"] for s in self.samples]
        total_hbm_gb  = round(all_mem_total[0] / 1024 * n_gpus, 1) if all_mem_total else 0
        peak_hbm_used_all_gb = round(sum(per_hbm_peak) / 1024, 2) if per_hbm_peak else 0

        return {
            "gpu_available":          True,
            "gpu_count":              n_gpus,
            # Utilisation
            "gpu_util_mean":          m("gpu_util"),
            "gpu_util_peak":          pk("gpu_util"),
            "gpu_util_per_gpu":       per_gpu_util,
            # HBM (GPU memory)
            "hbm_util_mean":          m("mem_util"),
            "hbm_used_mb_mean":       m("mem_used"),
            "hbm_used_mb_peak":       pk("mem_used"),
            "hbm_used_gb_peak_all":   peak_hbm_used_all_gb,
            "hbm_total_mb_per_gpu":   round(statistics.mean(all_mem_total), 0) if all_mem_total else 0,
            "hbm_total_gb_all":       total_hbm_gb,
            "hbm_util_pct_mean":      round(m("mem_used") / max(statistics.mean(all_mem_total), 1) * 100, 1),
            "hbm_per_gpu_peak_mb":    per_hbm_peak,
            # Power
            "power_w_mean":           m("power"),
            "power_w_peak":           pk("power"),
            "power_all_gpus_w_mean":  round(m("power") * n_gpus, 1),
            # Temperature
            "gpu_temp_mean_c":        m("temp_c"),
            "gpu_temp_peak_c":        pk("temp_c"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer-complete collectors — fills gaps identified in the stack-cake diagram
# ═══════════════════════════════════════════════════════════════════════════════


# ── A3 OS/Memory Manager (L2=DRAM) ───────────────────────────────────────────
# Metrics: page_fault_rate, TLB miss, NUMA faults, swap rate, huge pages,
#          KV$ swap rate
# Source:  /proc/vmstat (delta between samples)

class VmstatMonitor:
    """
    Polls /proc/vmstat for OS/Memory Manager metrics (A3 L2=DRAM layer).

    Captures per-interval deltas for:
      pgfault          — minor page faults (KV$ mmap accesses)
      pgmajfault       — major page faults (page not in RAM → disk fetch)
      pswpin/pswpout   — swap in/out pages (KV$ overflow to swap)
      pgpgin/pgpgout   — page cache reads/writes (filesystem I/O through cache)
      numa_pages_migrated — NUMA page migrations (cross-socket KV$ movement)
      nr_tlb_remote_cache_miss — TLB misses (indirect measure of KV$ scatter)
    """

    KEYS = [
        "pgfault", "pgmajfault",
        "pswpin",  "pswpout",
        "pgpgin",  "pgpgout",
        "numa_pages_migrated", "numa_hint_faults",
        "nr_tlb_remote_cache_miss",
        "oom_kill",
    ]

    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        self._samples: list[dict] = []
        self._thread  = None
        self._stop    = threading.Event()

    def _read(self) -> dict:
        out = {}
        try:
            for line in open("/proc/vmstat").read().splitlines():
                p = line.split()
                if len(p) == 2 and p[0] in self.KEYS:
                    out[p[0]] = int(p[1])
        except Exception:
            pass
        return out

    def start(self):
        self._stop.clear()
        self._samples.clear()
        s = self._read()
        s["_ts"] = time.time()   # always record timestamp
        self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = self._read()
            s["_ts"] = time.time()
            self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        s = self._read()
        s["_ts"] = time.time()
        self._samples.append(s)

        if len(self._samples) < 2 or not any(
                k in self._samples[0] for k in ("pgfault", "pswpin")):
            return {"vmstat_available": False}

        first, last = self._samples[0], self._samples[-1]
        dt = max(last["_ts"] - first["_ts"], 0.001)

        def rate(k):
            delta = last.get(k, 0) - first.get(k, 0)
            return round(max(delta, 0) / dt, 2)

        def total(k):
            return max(last.get(k, 0) - first.get(k, 0), 0)

        # Huge pages from /proc/meminfo
        huge_total = huge_free = huge_size_kb = 0
        try:
            for line in open("/proc/meminfo").read().splitlines():
                p = line.split()
                if p[0] == "HugePages_Total:":  huge_total  = int(p[1])
                if p[0] == "HugePages_Free:":   huge_free   = int(p[1])
                if p[0] == "Hugepagesize:":      huge_size_kb = int(p[1])
        except Exception:
            pass

        return {
            "vmstat_available":          True,
            # A3 OS/Memory Manager — page fault metrics
            "page_faults_per_s":         rate("pgfault"),
            "major_faults_per_s":        rate("pgmajfault"),
            "page_faults_total":         total("pgfault"),
            "major_faults_total":        total("pgmajfault"),
            # Swap (KV$ swap rate proxy)
            "swap_in_per_s":             rate("pswpin"),
            "swap_out_per_s":            rate("pswpout"),
            "swap_pages_total":          total("pswpin") + total("pswpout"),
            # Page cache I/O
            "page_cache_reads_per_s":    rate("pgpgin"),
            "page_cache_writes_per_s":   rate("pgpgout"),
            # NUMA
            "numa_migrations_per_s":     rate("numa_pages_migrated"),
            "numa_hint_faults_per_s":    rate("numa_hint_faults"),
            # TLB
            "tlb_remote_miss_per_s":     rate("nr_tlb_remote_cache_miss"),
            # OOM
            "oom_kills":                 total("oom_kill"),
            # Huge pages (static snapshot)
            "hugepages_total":           huge_total,
            "hugepages_free":            huge_free,
            "hugepages_used":            huge_total - huge_free,
            "hugepage_size_kb":          huge_size_kb,
        }


# ── A2 GPU Driver / Runtime — NVLink + PCIe ────────────────────────────────────
# Metrics: NVLink BW, PCIe H2D/D2H transfer rate
# Source:  nvidia-smi nvlink + dcgmi (fallback: nvidia-smi dmon)

class NvlinkPcieMonitor:
    """
    Samples NVLink bandwidth and PCIe host↔device transfer rates.

    NVLink BW:  nvidia-smi nvlink --query-nvlink=nvlink.tx.bytes,nvlink.rx.bytes
    PCIe BW:    nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current
                or dcgmi dmon -e 1009,1010 (PCIe TX/RX bytes)

    Populates A2 GPU Driver layer of the stack-cake.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval = interval_s
        self._samples: list[dict] = []
        self._thread  = None
        self._stop    = threading.Event()
        self._has_nvlink = None

    def _check_nvlink(self) -> bool:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "nvlink", "--status", "-i", "0"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            return "Active" in out or "Inactive" in out
        except Exception:
            return False

    def _sample_nvlink(self) -> dict:
        """Query NVLink TX/RX bytes per GPU via nvidia-smi."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "nvlink",
                 "--query-nvlink=nvlink.tx.bytes,nvlink.rx.bytes",
                 "--format=csv,noheader,nounits", "-i", "0"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            tx_total = rx_total = 0
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        tx_total += int(parts[0])
                        rx_total += int(parts[1])
                    except ValueError:
                        pass
            return {"nvlink_tx_bytes": tx_total, "nvlink_rx_bytes": rx_total,
                    "_ts": time.time()}
        except Exception:
            return {}

    def _sample_pcie(self) -> dict:
        """Query PCIe TX/RX via dcgmi dmon, fallback to nvidia-smi."""
        # Try dcgmi first (field IDs 1009=PCIe TX bytes, 1010=PCIe RX bytes)
        try:
            out = subprocess.check_output(
                ["dcgmi", "dmon", "-e", "1009,1010", "-c", "1"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            tx = rx = 0
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        tx += int(parts[1])
                        rx += int(parts[2])
                    except (ValueError, IndexError):
                        pass
            if tx > 0 or rx > 0:
                return {"pcie_tx_bytes_per_s": tx, "pcie_rx_bytes_per_s": rx}
        except Exception:
            pass
        # Fallback: nvidia-smi pcie link info (static, not per-interval BW)
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            gens = widths = []
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 2:
                    try:
                        gens.append(int(p[0]))
                        widths.append(int(p[1]))
                    except ValueError:
                        pass
            if gens:
                # Theoretical BW: Gen5 x16 = 64 GB/s, Gen4 x16 = 32 GB/s
                gen_bw = {1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
                bw_per_gpu = [gen_bw.get(g, 0) * w / 16 for g, w in zip(gens, widths)]
                return {
                    "pcie_link_gen":       gens[0] if gens else 0,
                    "pcie_link_width":     widths[0] if widths else 0,
                    "pcie_theoretical_gbps": round(sum(bw_per_gpu) / max(len(bw_per_gpu), 1), 1),
                }
        except Exception:
            pass
        return {}

    def start(self):
        self._has_nvlink = self._check_nvlink()
        self._stop.clear()
        self._samples.clear()
        s = {}
        if self._has_nvlink:
            s.update(self._sample_nvlink())
        s.update(self._sample_pcie())
        s["_ts"] = time.time()
        self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = {}
            if self._has_nvlink:
                s.update(self._sample_nvlink())
            s.update(self._sample_pcie())
            s["_ts"] = time.time()
            self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        if len(self._samples) < 2:
            return {"nvlink_available": self._has_nvlink or False}

        first, last = self._samples[0], self._samples[-1]
        dt = max(last["_ts"] - first["_ts"], 0.001)

        result = {"nvlink_available": bool(self._has_nvlink)}

        if self._has_nvlink and "nvlink_tx_bytes" in last:
            tx_delta = last.get("nvlink_tx_bytes", 0) - first.get("nvlink_tx_bytes", 0)
            rx_delta = last.get("nvlink_rx_bytes", 0) - first.get("nvlink_rx_bytes", 0)
            result["nvlink_tx_gb_s"] = round(max(tx_delta, 0) / dt / 1e9, 2)
            result["nvlink_rx_gb_s"] = round(max(rx_delta, 0) / dt / 1e9, 2)
            result["nvlink_total_bw_gb_s"] = round(result["nvlink_tx_gb_s"] +
                                                    result["nvlink_rx_gb_s"], 2)

        # PCIe
        if "pcie_tx_bytes_per_s" in last:
            result["pcie_tx_gb_s"] = round(last.get("pcie_tx_bytes_per_s", 0) / 1e9, 3)
            result["pcie_rx_gb_s"] = round(last.get("pcie_rx_bytes_per_s", 0) / 1e9, 3)
        if "pcie_link_gen" in last:
            result["pcie_link_gen"]           = last["pcie_link_gen"]
            result["pcie_link_width"]         = last["pcie_link_width"]
            result["pcie_theoretical_gbps"]   = last.get("pcie_theoretical_gbps", 0)

        return result


# ── A2 io_uring / NVMe Driver — queue depth, P999, inflight ───────────────────
# Metrics: SQ/CQ inflight, queue depth util, P999 latency, IOPS vs QD curve
# Source:  /sys/block/<dev>/stat, /sys/block/<dev>/queue/

class NvmeDriverMonitor:
    """
    Reads NVMe driver-level metrics from sysfs (A2 io_uring/NVMe Driver layer).

    /sys/block/nvme0n1/stat:
      field 0  = reads completed
      field 3  = read ms total
      field 4  = writes completed
      field 7  = write ms total
      field 8  = I/Os currently in flight (queue depth)
      field 9  = ms spent doing I/Os (utilisation)
      field 10 = weighted ms spent doing I/Os (time_in_queue; avg QD source)

    Also reads queue parameters:
      /sys/block/<dev>/queue/nr_requests  — queue depth limit
      /sys/block/<dev>/queue/scheduler    — current I/O scheduler
      /sys/block/<dev>/inflight           — reads/writes in flight
    """

    def __init__(self, device: str, interval_s: float = 1.0):
        self.dev      = os.path.basename(device)
        self.interval = interval_s
        self._samples: list[dict] = []
        self._thread  = None
        self._stop    = threading.Event()

    def _read_stat(self) -> dict:
        try:
            raw = open(f"/sys/block/{self.dev}/stat").read().split()
            if len(raw) < 11:
                return {}
            return {
                "_ts":        time.time(),
                "rd_ios":     int(raw[0]),
                "rd_ms":      int(raw[3]),
                "wr_ios":     int(raw[4]),
                "wr_ms":      int(raw[7]),
                "inflight":       int(raw[8]),
                "io_ms":          int(raw[9]),
                "weighted_io_ms": int(raw[10]) if len(raw) > 10 else 0,
            }
        except Exception:
            return {}

    def _read_queue_params(self) -> dict:
        out = {}
        base = f"/sys/block/{self.dev}/queue"
        for f, key in [("nr_requests", "nr_requests"),
                       ("scheduler",   "scheduler"),
                       ("rotational",  "rotational"),
                       ("physical_block_size", "physical_block_size_b"),
                       ("logical_block_size",  "logical_block_size_b")]:
            try:
                val = open(f"{base}/{f}").read().strip()
                out[key] = int(val) if val.isdigit() else val
            except Exception:
                pass
        # Inflight (reads in-flight, writes in-flight)
        try:
            parts = open(f"/sys/block/{self.dev}/inflight").read().split()
            if len(parts) >= 2:
                out["inflight_reads"]  = int(parts[0])
                out["inflight_writes"] = int(parts[1])
        except Exception:
            pass
        return out

    def start(self):
        self._stop.clear()
        self._samples.clear()
        s = self._read_stat()
        if s:
            self._samples.append(s)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            s = self._read_stat()
            if s:
                self._samples.append(s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        s = self._read_stat()
        if s:
            self._samples.append(s)

        qp = self._read_queue_params()

        if len(self._samples) < 2:
            return {"nvme_driver_available": False, **qp}

        first, last = self._samples[0], self._samples[-1]
        dt = max(last["_ts"] - first["_ts"], 0.001)

        rd_ios_delta = max(last["rd_ios"] - first["rd_ios"], 0)
        wr_ios_delta = max(last["wr_ios"] - first["wr_ios"], 0)
        rd_ms_delta  = max(last["rd_ms"]  - first["rd_ms"],  0)
        wr_ms_delta  = max(last["wr_ms"]  - first["wr_ms"],  0)
        io_ms_delta  = max(last["io_ms"]  - first["io_ms"],  0)

        # Average completion latency from sysfs (ms per IO)
        rd_lat_ms = round(rd_ms_delta / max(rd_ios_delta, 1), 2)
        wr_lat_ms = round(wr_ms_delta / max(wr_ios_delta, 1), 2)

        # IOPS from sysfs (crosscheck with iostat)
        rd_iops_sysfs = round(rd_ios_delta / dt, 1)
        wr_iops_sysfs = round(wr_ios_delta / dt, 1)

        # Device utilisation % from sysfs (ms spent / wall ms)
        util_pct = round(min(io_ms_delta / (dt * 10), 100), 1)

        # Mean inflight queue depth
        inflight_vals = [s["inflight"] for s in self._samples if "inflight" in s]
        inflight_mean = round(statistics.mean(inflight_vals), 2) if inflight_vals else 0.0
        inflight_peak = max(inflight_vals) if inflight_vals else 0
        inflight_sorted = sorted(inflight_vals)
        def _pct(vals, pct):
            if not vals:
                return 0.0
            idx = min(max(int(len(vals) * pct / 100.0), 0), len(vals)-1)
            return round(vals[idx], 2)

        # /sys/block/<dev>/stat column 10 is a time-weighted queue depth source:
        # avg_qd = Δweighted_io_ms / Δwall_ms.
        weighted_delta_ms = max(float(last.get("weighted_io_ms", 0)) - float(first.get("weighted_io_ms", 0)), 0.0)
        weighted_qd_mean = round(weighted_delta_ms / max(dt * 1000.0, 1e-6), 3)

        return {
            "nvme_driver_available":  True,
            # A2 io_uring/NVMe Driver metrics
            "nvme_rd_iops_sysfs":     rd_iops_sysfs,
            "nvme_wr_iops_sysfs":     wr_iops_sysfs,
            "nvme_rd_lat_ms_mean":    rd_lat_ms,
            "nvme_wr_lat_ms_mean":    wr_lat_ms,
            "nvme_inflight_mean":     inflight_mean,
            "nvme_inflight_p50":      _pct(inflight_sorted, 50),
            "nvme_inflight_p95":      _pct(inflight_sorted, 95),
            "nvme_inflight_p99":      _pct(inflight_sorted, 99),
            "nvme_inflight_peak":     inflight_peak,
            "nvme_weighted_qd_mean":  weighted_qd_mean,
            "nvme_util_pct_sysfs":    util_pct,
            "queue_depth_sources":    "sysfs_stat_inflight,sysfs_stat_weighted_io_ms,sysfs_inflight,sysfs_queue",
            # Queue parameters (A2 layer config)
            "nvme_nr_requests":       qp.get("nr_requests", 0),
            "nvme_scheduler":         qp.get("scheduler", "unknown"),
            "nvme_physical_block_b":  qp.get("physical_block_size_b", 0),
            "nvme_logical_block_b":   qp.get("logical_block_size_b", 0),
            "nvme_inflight_reads":    qp.get("inflight_reads", 0),
            "nvme_inflight_writes":   qp.get("inflight_writes", 0),
        }



# ── Unified block queue-depth collector ──────────────────────────────────────

class BlockQueueDepthCollector:
    """
    Capture queue-depth related metrics from every cheap local kernel source.

    Sources sampled every interval:
      • /sys/block/<dev>/stat field 9  -> instantaneous in-flight I/O
      • /sys/block/<dev>/stat field 10 -> cumulative device busy ticks
      • /sys/block/<dev>/stat field 11 -> cumulative weighted time in queue
      • /sys/block/<dev>/inflight      -> read/write in-flight split
      • /sys/block/<dev>/queue/*       -> queue limits/config
      • /proc/diskstats                -> same kernel counters via proc fallback

    Exact event-level queue depth still comes from blktrace Q→C analysis when
    --enable-blktrace is used. This collector provides always-on sampled
    advisory sampled queue pressure so the report has data even when exact blktrace Q/C parsing is
    missing or intentionally disabled. Cumulative sysfs fields are used only as run-window deltas, never as absolute point values.
    """

    def __init__(self, device: str, interval_s: float = 1.0, work_dir: "Path | str" = ".",
                 filename_suffix: str = ""):
        self.device = device
        self.dev = os.path.basename(device)
        # If the user passes a partition, queue stats live under the parent disk.
        self.base_dev = re.sub(r"p\d+$", "", self.dev)
        self.interval = max(float(interval_s or 1.0), 0.1)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[dict] = []
        self._derived: list[dict] = []
        self._thread = None
        self._stop = threading.Event()
        _sfx = ("_" + filename_suffix) if filename_suffix else ""
        self._csv_path = self.work_dir / ("queue_depth_sources_timeseries" + _sfx + ".csv")

    def _read_queue_params(self) -> dict:
        out = {}
        qdir = Path(f"/sys/block/{self.base_dev}/queue")
        for name in (
            "nr_requests", "scheduler", "nomerges", "rq_affinity",
            "rotational", "max_sectors_kb", "max_hw_sectors_kb",
            "logical_block_size", "physical_block_size", "minimum_io_size",
            "optimal_io_size", "io_poll", "io_poll_delay",
        ):
            p = qdir / name
            try:
                v = p.read_text().strip()
                out[f"queue_{name}"] = int(v) if re.fullmatch(r"-?\d+", v or "") else v
            except Exception:
                pass
        return out

    def _nr_hw_queues(self) -> int:
        try:
            mq = Path(f"/sys/block/{self.base_dev}/mq")
            return len([p for p in mq.iterdir() if p.is_dir()]) if mq.exists() else 1
        except Exception:
            return 1

    def _fallback_qd_plausibility_limit(self, row: dict | None = None) -> float:
        """Conservative sanity bound for sampled/fallback queue-depth values.

        This is not a saturation threshold. It only prevents cumulative sysfs
        counter/unit mistakes from being promoted into QD values.  Exact
        blktrace Q->C analysis does not use this helper.
        """
        row = row or {}
        try:
            nr_req = float(row.get("queue_nr_requests", 0) or 0)
        except Exception:
            nr_req = 0.0
        nr_hwq = float(row.get("nr_hw_queues", 0) or self._nr_hw_queues() or 1)
        # Many NVMe drivers expose nr_requests per hardware queue. Allow headroom
        # for transient samples but reject values orders-of-magnitude above the
        # configured queueing capacity.
        base = nr_req * max(nr_hwq, 1.0) if nr_req > 0 else 4096.0
        return max(128.0, min(base * 4.0, 262144.0))

    def _read_inflight(self) -> dict:
        out = {}
        for dev in (self.dev, self.base_dev):
            p = Path(f"/sys/block/{dev}/inflight")
            if p.exists():
                try:
                    vals = p.read_text().split()
                    if len(vals) >= 2:
                        out["sysfs_inflight_reads"] = int(vals[0])
                        out["sysfs_inflight_writes"] = int(vals[1])
                        out["sysfs_inflight_total_split"] = int(vals[0]) + int(vals[1])
                        out["sysfs_inflight_source_dev"] = dev
                        return out
                except Exception:
                    pass
        return out

    def _read_stat_file(self) -> dict:
        for dev in (self.dev, self.base_dev):
            p = Path(f"/sys/block/{dev}/stat")
            if p.exists():
                try:
                    raw = p.read_text().split()
                    if len(raw) >= 11:
                        return {
                            "stat_source": f"/sys/block/{dev}/stat",
                            "rd_ios": int(raw[0]),
                            "rd_merges": int(raw[1]),
                            "rd_sectors": int(raw[2]),
                            "rd_ms": int(raw[3]),
                            "wr_ios": int(raw[4]),
                            "wr_merges": int(raw[5]),
                            "wr_sectors": int(raw[6]),
                            "wr_ms": int(raw[7]),
                            "stat_inflight": int(raw[8]),
                            "io_ticks_ms": int(raw[9]),
                            "weighted_io_ms": int(raw[10]),
                            "discard_ios": int(raw[11]) if len(raw) > 11 else 0,
                            "discard_sectors": int(raw[13]) if len(raw) > 13 else 0,
                            "discard_ms": int(raw[14]) if len(raw) > 14 else 0,
                        }
                except Exception:
                    pass
        return {}

    def _read_proc_diskstats(self) -> dict:
        try:
            with open("/proc/diskstats") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 14 and parts[2] in {self.dev, self.base_dev}:
                        return {
                            "proc_diskstats_dev": parts[2],
                            "proc_rd_ios": int(parts[3]),
                            "proc_rd_sectors": int(parts[5]),
                            "proc_rd_ms": int(parts[6]),
                            "proc_wr_ios": int(parts[7]),
                            "proc_wr_sectors": int(parts[9]),
                            "proc_wr_ms": int(parts[10]),
                            "proc_inflight": int(parts[11]) if len(parts) > 11 else 0,
                            "proc_io_ticks_ms": int(parts[12]) if len(parts) > 12 else 0,
                            "proc_weighted_io_ms": int(parts[13]) if len(parts) > 13 else 0,
                        }
        except Exception:
            pass
        return {}

    def _read(self) -> dict:
        row = {"ts": time.time(), "device": self.device, "dev": self.dev, "base_dev": self.base_dev}
        row.update(self._read_stat_file())
        row.update(self._read_inflight())
        row.update(self._read_proc_diskstats())
        row.update(self._read_queue_params())
        return row

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._derived.clear()
        self._samples.append(self._read())
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._samples.append(self._read())

    def _derive_rows(self) -> list[dict]:
        rows: list[dict] = []
        for i in range(1, len(self._samples)):
            a, b = self._samples[i-1], self._samples[i]
            dt = max(float(b.get("ts", 0)) - float(a.get("ts", 0)), 1e-6)
            wall_ms = dt * 1000.0
            d_io_ms = max(float(b.get("io_ticks_ms", b.get("proc_io_ticks_ms", 0))) -
                          float(a.get("io_ticks_ms", a.get("proc_io_ticks_ms", 0))), 0.0)
            d_weighted_ms = max(float(b.get("weighted_io_ms", b.get("proc_weighted_io_ms", 0))) -
                                float(a.get("weighted_io_ms", a.get("proc_weighted_io_ms", 0))), 0.0)
            d_rd_ios = max(float(b.get("rd_ios", b.get("proc_rd_ios", 0))) -
                           float(a.get("rd_ios", a.get("proc_rd_ios", 0))), 0.0)
            d_wr_ios = max(float(b.get("wr_ios", b.get("proc_wr_ios", 0))) -
                           float(a.get("wr_ios", a.get("proc_wr_ios", 0))), 0.0)
            d_rd_sec = max(float(b.get("rd_sectors", b.get("proc_rd_sectors", 0))) -
                           float(a.get("rd_sectors", a.get("proc_rd_sectors", 0))), 0.0)
            d_wr_sec = max(float(b.get("wr_sectors", b.get("proc_wr_sectors", 0))) -
                           float(a.get("wr_sectors", a.get("proc_wr_sectors", 0))), 0.0)
            d_rd_ms = max(float(b.get("rd_ms", b.get("proc_rd_ms", 0))) -
                          float(a.get("rd_ms", a.get("proc_rd_ms", 0))), 0.0)
            d_wr_ms = max(float(b.get("wr_ms", b.get("proc_wr_ms", 0))) -
                          float(a.get("wr_ms", a.get("proc_wr_ms", 0))), 0.0)
            sysfs_inflight = float(b.get("stat_inflight", b.get("proc_inflight", 0)) or 0)
            weighted_qd_raw = d_weighted_ms / max(wall_ms, 1e-6)
            row = {
                "ts": b.get("ts", 0),
                "device": self.device,
                "dev": self.dev,
                "base_dev": self.base_dev,
                "source": "sysfs_run_delta_advisory",
                "sysfs_inflight": sysfs_inflight,
                "sysfs_inflight_reads": float(b.get("sysfs_inflight_reads", 0) or 0),
                "sysfs_inflight_writes": float(b.get("sysfs_inflight_writes", 0) or 0),
                # Keep raw weighted QD for audit only. It is a per-run delta:
                # delta(weighted_io_ms) / delta(wall_ms). Never use the
                # cumulative weighted_io_ms absolute value as QD.
                "weighted_qd_raw": round(weighted_qd_raw, 4),
                "weighted_qd": round(weighted_qd_raw, 4),
                "d_weighted_io_ms": round(d_weighted_ms, 3),
                "d_io_ticks_ms": round(d_io_ms, 3),
                "wall_ms": round(wall_ms, 3),
                "io_util_pct": round(min(d_io_ms / max(wall_ms, 1e-6) * 100.0, 100.0), 3),
                "rd_iops": round(d_rd_ios / dt, 3),
                "wr_iops": round(d_wr_ios / dt, 3),
                "rd_bw_mbs": round(d_rd_sec * 512.0 / 1024.0 / 1024.0 / dt, 6),
                "wr_bw_mbs": round(d_wr_sec * 512.0 / 1024.0 / 1024.0 / dt, 6),
                "rd_lat_ms": round(d_rd_ms / max(d_rd_ios, 1.0), 4) if d_rd_ios > 0 else 0.0,
                "wr_lat_ms": round(d_wr_ms / max(d_wr_ios, 1.0), 4) if d_wr_ios > 0 else 0.0,
                "queue_nr_requests": b.get("queue_nr_requests", 0),
                "nr_hw_queues": b.get("nr_hw_queues", 1),
                "queue_scheduler": b.get("queue_scheduler", ""),
                "queue_nomerges": b.get("queue_nomerges", ""),
                "queue_max_sectors_kb": b.get("queue_max_sectors_kb", 0),
                "stat_start_weighted_io_ms": a.get("weighted_io_ms", a.get("proc_weighted_io_ms", 0)),
                "stat_end_weighted_io_ms": b.get("weighted_io_ms", b.get("proc_weighted_io_ms", 0)),
                "stat_start_io_ticks_ms": a.get("io_ticks_ms", a.get("proc_io_ticks_ms", 0)),
                "stat_end_io_ticks_ms": b.get("io_ticks_ms", b.get("proc_io_ticks_ms", 0)),
            }
            plaus_limit = self._fallback_qd_plausibility_limit(row)
            weighted_valid = 0.0 <= weighted_qd_raw <= plaus_limit
            inflight_valid = 0.0 <= sysfs_inflight <= plaus_limit
            row["fallback_qd_plausibility_limit"] = round(plaus_limit, 3)
            row["weighted_qd_valid"] = int(bool(weighted_valid))
            row["sysfs_inflight_valid"] = int(bool(inflight_valid))
            # Conservative best-effort QD for fallback reports:
            #   1. prefer instantaneous in-flight from /sys/block/<dev>/stat
            #      because it is not cumulative and cannot include previous runs;
            #   2. use weighted_qd only when it is a plausible run-window delta.
            # Fallback QD remains advisory and must not by itself prove saturation.
            if inflight_valid:
                row["qd_best_effort"] = round(sysfs_inflight, 4)
                row["qd_best_effort_source"] = "sysfs_stat_inflight_instantaneous"
                row["qd_valid"] = 1
            elif weighted_valid:
                row["qd_best_effort"] = round(weighted_qd_raw, 4)
                row["qd_best_effort_source"] = "sysfs_weighted_io_ms_run_delta_advisory"
                row["qd_valid"] = 1
            else:
                row["qd_best_effort"] = 0.0
                row["qd_best_effort_source"] = "invalid_sysfs_counter_delta_rejected"
                row["qd_valid"] = 0
            rows.append(row)
        return rows

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._samples.append(self._read())
        self._derived = self._derive_rows()

        # Write a canonical, multi-source queue-depth timeseries immediately so
        # analyze/report can consume it without needing to know collector internals.
        if self._derived:
            import csv as _csv
            fields = sorted({k for row in self._derived for k in row.keys()})
            with open(self._csv_path, "w", encoding="utf-8", newline="") as fp:
                w = _csv.DictWriter(fp, fieldnames=fields)
                w.writeheader()
                for row in self._derived:
                    w.writerow(row)

        valid_rows = [r for r in self._derived if int(float(r.get("qd_valid", 0) or 0)) == 1]
        qd_vals = [float(r.get("qd_best_effort", 0)) for r in valid_rows]
        weighted_vals = [float(r.get("weighted_qd", 0)) for r in self._derived
                         if int(float(r.get("weighted_qd_valid", 0) or 0)) == 1]
        inflight_vals = [float(r.get("sysfs_inflight", 0)) for r in self._derived
                         if int(float(r.get("sysfs_inflight_valid", 0) or 0)) == 1]
        util_vals = [float(r.get("io_util_pct", 0)) for r in self._derived]
        invalid_qd_rows = max(len(self._derived) - len(valid_rows), 0)
        def _mean(vals):
            return round(statistics.mean(vals), 4) if vals else 0.0
        def _pct(vals, pct):
            if not vals: return 0.0
            xs = sorted(vals)
            idx = min(max(int(len(xs) * pct / 100.0), 0), len(xs)-1)
            return round(xs[idx], 4)
        def _peak(vals):
            return round(max(vals), 4) if vals else 0.0

        return {
            "queue_depth_available": bool(qd_vals),
            "queue_depth_csv": str(self._csv_path) if self._derived else "",
            "queue_depth_samples": len(qd_vals),
            "queue_depth_rows_total": len(self._derived),
            "queue_depth_invalid_rows": invalid_qd_rows,
            "queue_depth_sources": "advisory_sysfs_run_delta_and_inflight",
            "queue_depth_validity": "run_window_delta_only_not_accumulated; fallback_advisory_not_saturation_proof",
            "queue_depth_mean": _mean(qd_vals),
            "queue_depth_p50": _pct(qd_vals, 50),
            "queue_depth_p95": _pct(qd_vals, 95),
            "queue_depth_p99": _pct(qd_vals, 99),
            "queue_depth_peak": _peak(qd_vals),
            "weighted_qd_mean": _mean(weighted_vals),
            "weighted_qd_peak": _peak(weighted_vals),
            "inflight_mean": _mean(inflight_vals),
            "inflight_peak": _peak(inflight_vals),
            "io_util_pct_mean": _mean(util_vals),
            "io_util_pct_peak": _peak(util_vals),
        }

# ── A1 L3:SSD Hardware — extended SMART + TBW + DWPD + KV$ cold store ─────────

class SsdHardwareMonitor:
    """
    Extended NVMe SMART metrics for A1 L3:SSD Hardware layer.

    Adds to NvmeSmartMonitor:
      - TBW (terabytes written, from data_units_written)
      - DWPD estimate (host_written / (TBW_rated * days))
      - P999 read latency (from nvme latency stats if available)
      - KV$ cold store size (du -sh of HiCache directory)
      - HiCache file count (number of KV$ block files)
    """

    def __init__(self, device: str, hicache_path: str = "/mnt/sglang_dv3",
                 rated_tbw: float = 7300.0, use_sudo: bool = True):
        self.device       = device
        self.use_sudo    = use_sudo
        self.hicache_path = hicache_path
        self.rated_tbw    = rated_tbw   # rated TBW for DWPD calc (Dell CM7 = ~7.3 PBW)
        self._start: dict = {}
        self.nvme_capacity_gb = self._read_device_capacity_gb()

    def _read_device_capacity_gb(self) -> float:
        """Read NVMe device capacity in GB from /sys/block/<dev>/size.

        Falls back to `nvme id-ctrl ... tnvmcap` if the /sys path is missing
        (e.g. partition device passed in — strip the pN suffix and retry).
        Returns 0.0 if neither source works, so the collector keeps running.
        """
        try:
            dev_name = os.path.basename(self.device)
            # /sys/block has whole-device entries only — strip pN suffix
            dev_base = re.sub(r"p\d+$", "", dev_name)
            size_path = f"/sys/block/{dev_base}/size"
            if os.path.exists(size_path):
                with open(size_path) as f:
                    sectors = int(f.read().strip())
                # /sys/block exposes 512-byte sector count regardless of the
                # device's physical sector size, so 512 is correct here.
                return round(sectors * 512 / (1024 ** 3), 2)
        except Exception:
            pass
        # Fallback: nvme id-ctrl tnvmcap (total NVM capacity in bytes)
        try:
            out = _sudo_check_output(
                ["nvme", "id-ctrl", self.device, "-o", "json"], self.use_sudo,
                stderr=subprocess.DEVNULL, timeout=10)
            j = json.loads(out)
            tn = j.get("tnvmcap", 0)
            if tn:
                return round(int(tn) / (1024 ** 3), 2)
        except Exception:
            pass
        return 0.0

    def _smart(self) -> dict:
        try:
            return json.loads(_sudo_check_output(
                ["nvme", "smart-log", self.device, "-o", "json"], self.use_sudo,
                stderr=subprocess.DEVNULL, timeout=10))
        except Exception:
            return {}

    def _latency_stats(self) -> dict:
        """nvme latency-stats — P999 read latency histogram."""
        try:
            raw = _sudo_check_output(
                ["nvme", "latency-stats", self.device], self.use_sudo,
                stderr=subprocess.DEVNULL, text=True, timeout=10)
            # Parse read latency buckets to estimate P999
            buckets = []
            in_read = False
            for line in raw.splitlines():
                if "Read" in line and "Latency" in line:
                    in_read = True
                if in_read and re.match(r"\s*\d+", line):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            lat_us = int(parts[0])
                            count  = int(parts[1])
                            buckets.extend([lat_us] * count)
                        except ValueError:
                            pass
            if buckets:
                buckets.sort()
                n = len(buckets)
                return {
                    "nvme_rd_lat_p50_us_fw":  buckets[int(n * 0.50)],
                    "nvme_rd_lat_p99_us_fw":  buckets[int(n * 0.99)],
                    "nvme_rd_lat_p999_us_fw": buckets[min(int(n * 0.999), n - 1)],
                }
        except Exception:
            pass
        return {}

    def _hicache_stats(self) -> dict:
        """Measure KV$ cold store occupancy on the SSD."""
        path = self.hicache_path
        out  = {}
        try:
            # File count
            fc = len(list(Path(path).iterdir())) if Path(path).exists() else 0
            out["hicache_file_count"] = fc
            # Directory size (bytes)
            result = subprocess.run(
                ["du", "-sb", path],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
                out["hicache_size_gb"] = round(size_bytes / (1024 ** 3), 3)
            else:
                out["hicache_size_gb"] = 0.0
        except Exception:
            out["hicache_file_count"] = -1
            out["hicache_size_gb"]    = 0.0
        return out

    def start(self):
        self._start = self._smart()

    def stop(self) -> dict:
        end = self._smart()
        if not self._start:
            return {}

        def gb(s, k):
            return s.get(k, 0) * 512 * 1024 / (1024 ** 3)

        host_written_gb = max(0.0, gb(end, "data_units_written") -
                                   gb(self._start, "data_units_written"))
        nand_written_gb = max(0.0, (end.get("nand_bytes_written", 0) -
                                    self._start.get("nand_bytes_written", 0)) / (1024 ** 3))
        waf = round(nand_written_gb / max(host_written_gb, 0.001), 3) if host_written_gb > 0.01 else 0.0

        # Cumulative TBW from SMART (total lifetime host writes)
        lifetime_host_tb = round(gb(end, "data_units_written") / 1024, 2)
        # DWPD estimate: (host_written_gb_this_run / run_hours) / (rated_tbw_tb * 1024 / 365)
        # Simplified: host_written_gb / rated_TBW_per_day_gb
        rated_gb_per_day = self.rated_tbw * 1024 / 365
        run_hours = 1.0   # approximation; real value needs wall-clock from caller
        dwpd_est  = round(host_written_gb / max(rated_gb_per_day / 24 * run_hours, 0.001), 4)

        temps = []
        for s in [self._start, end]:
            t = s.get("temperature", 0)
            temps.append(int(t - 273) if t > 200 else int(t))

        result = {
            # A1 SSD Hardware layer
            "ssd_host_written_gb":   round(host_written_gb, 3),
            "ssd_nand_written_gb":   round(nand_written_gb, 3),
            "ssd_waf":               waf,
            "ssd_lifetime_tbw":      lifetime_host_tb,
            "ssd_dwpd_est":          dwpd_est,
            "ssd_temp_start_c":      temps[0],
            "ssd_temp_end_c":        temps[1],
            "ssd_temp_peak_c":       max(temps),
            "ssd_power_cycles":      end.get("power_cycles", 0),
            "ssd_unsafe_shutdowns":  end.get("unsafe_shutdowns", 0),
            "ssd_media_errors":      end.get("media_errors", 0),
        }
        result.update(self._latency_stats())
        result.update(self._hicache_stats())
        # NVMe device total capacity
        if self.nvme_capacity_gb > 0:
            result["nvme_device_capacity_gb"] = self.nvme_capacity_gb
        # HiCache filesystem df stats
        try:
            _df = subprocess.run(
                ["df", "-B1", "--output=size,used,avail", self.hicache_path],
                capture_output=True, text=True, timeout=5)
            if _df.returncode == 0:
                _ln = []
                for _l in _df.stdout.strip().splitlines():
                    _ps = _l.split()
                    if _ps and _ps[0].isdigit():
                        _ln.append(_l)
                if _ln:
                    _p = _ln[0].split()
                    if len(_p) >= 3:
                        _G = 1024 ** 3
                        result["hicache_fs_total_gb"] = round(int(_p[0]) / _G, 2)
                        result["hicache_fs_used_gb"]  = round(int(_p[1]) / _G, 2)
                        result["hicache_fs_avail_gb"] = round(int(_p[2]) / _G, 2)
                        result["hicache_fs_used_pct"] = round(
                            int(_p[1]) / max(int(_p[0]), 1) * 100, 1)
        except Exception:
            pass
        return result


# ── A5 Application Layer — request latency P99 across instances ────────────────

class RequestLatencyTracker:
    """
    Tracks per-request total latency (wall-clock) to compute P99 across
    a batch of instances — the A5 Application Layer metric.

    Usage:
        tracker = RequestLatencyTracker()
        tracker.record(total_time_s)
        ...
        result = tracker.summarise()
        # result["req_lat_p99_ms"], result["req_lat_p999_ms"]
    """

    def __init__(self):
        self._samples: list[float] = []

    def record(self, duration_s: float):
        self._samples.append(duration_s * 1000)   # store in ms

    def summarise(self) -> dict:
        if not self._samples:
            return {}
        s = sorted(self._samples)
        n = len(s)
        return {
            "req_lat_mean_ms":  round(statistics.mean(s), 1),
            "req_lat_p50_ms":   round(s[int(n * 0.50)], 1),
            "req_lat_p90_ms":   round(s[int(n * 0.90)], 1),
            "req_lat_p99_ms":   round(s[min(int(n * 0.99), n - 1)], 1),
            "req_lat_p999_ms":  round(s[min(int(n * 0.999), n - 1)], 1),
            "req_lat_max_ms":   round(s[-1], 1),
            "req_count":        n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA Kernel Monitor — SM / Tensor Core / HBM BW metrics
# ═══════════════════════════════════════════════════════════════════════════════
#
# Collects kernel-level GPU metrics that reveal the compute/memory/storage
# interaction at the hardware pipeline level.
#
# Data sources (tried in priority order):
#   1. dcgmi dmon  — DCGM profiling counters (most accurate, needs dcgmd daemon)
#   2. nvidia-smi  — SM clock, throttle reasons, basic utilisation
#
# Key metrics and what they mean for AMOprof:
#   sm_active_pct      — fraction of time ≥1 warp is active per SM
#                        DROPS sharply during SSD KV$ restore stalls
#   sm_occupancy_pct   — active warps / max warps per SM
#                        low = memory-bound; high = compute-bound
#   tensor_active_pct  — tensor core pipe utilisation (0=idle, 100=saturated)
#                        high during GEMM (prefill attention); low during decode
#   dram_active_pct    — HBM read+write active cycles ratio
#                        high during prefill KV$ writes; lower during decode
#   fp16_active_pct    — FP16 arithmetic pipe utilisation
#   hbm_bw_read_gb_s   — actual HBM read BW (derived: dram_active × peak_bw/2)
#   hbm_bw_write_gb_s  — actual HBM write BW
#   sm_clock_mhz       — current SM frequency (drops on throttle)
#   mem_clock_mhz      — current HBM frequency
#   throttled_pct      — fraction of samples where hw_slowdown was active
#   pcie_tx_gb_s       — PCIe TX BW (KV$ staging: HBM → DRAM → NVMe)
#   pcie_rx_gb_s       — PCIe RX BW (KV$ restore: NVMe → DRAM → HBM)
#
# SSD stall signature:
#   kv_miss_penalty_ms ↑  →  sm_active_pct ↓  →  tensor_active_pct → 0
#   This is the direct hardware evidence of storage bottlenecking compute.
#
# A100 SXM4 peak bandwidths (reference):
#   HBM2e:  2 TB/s (read+write combined)
#   NVLink: 600 GB/s (8 GPUs, bidirectional)
#   PCIe 4: ~32 GB/s per GPU slot


# DCGM field IDs used
_DCGM_FIELDS = {
    1002: "sm_active",
    1003: "sm_occupancy",
    1004: "tensor_active",
    1005: "dram_active",
    1008: "fp16_active",
    1009: "pcie_tx_bytes",
    1010: "pcie_rx_bytes",
    1011: "nvlink_tx_bytes",
    1012: "nvlink_rx_bytes",
}

# A100 SXM4 peak HBM2e bandwidth (TB/s → GB/s)
_A100_HBM_PEAK_GB_S = 2000.0   # 2 TB/s


class CudaKernelMonitor:
    """
    Polls CUDA kernel-level metrics via DCGM and nvidia-smi.

    Runs a background thread that samples every `interval_s` seconds.
    Aggregates mean/peak/min across the sampling window on stop().

    Usage:
        mon = CudaKernelMonitor(interval_s=0.5)
        mon.start()
        # ... run inference ...
        metrics = mon.stop()
        # metrics["sm_active_mean_pct"], metrics["tensor_active_min_pct"], ...
    """

    def __init__(self, interval_s: float = 0.5, gpu_ids: list[int] | None = None):
        self.interval   = interval_s
        self.gpu_ids    = gpu_ids      # None = all GPUs
        self.samples: list[dict] = []
        self._thread    = None
        self._stop_evt  = threading.Event()
        self._has_dcgm  = self._check_dcgm()
        self._gpu_count = self._count_gpus()

    def _check_dcgm(self) -> bool:
        try:
            subprocess.check_output(
                ["dcgmi", "dmon", "--help"], stderr=subprocess.DEVNULL, timeout=3)
            return True
        except Exception:
            return False

    def _count_gpus(self) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            return len(out.strip().splitlines())
        except Exception:
            return 0

    def start(self):
        if self._gpu_count == 0:
            return
        self._stop_evt.clear()
        self.samples.clear()
        self._thread = threading.Thread(
            target=self._poll_dcgm if self._has_dcgm else self._poll_nvidiasmi,
            daemon=True)
        self._thread.start()

    # ── DCGM polling ──────────────────────────────────────────────────────────

    def _poll_dcgm(self):
        """
        Poll DCGM profiling counters via dcgmi dmon.

        dcgmi dmon output (per GPU per line):
            # Entity  sm_active  sm_occ  tensor  dram  fp16  pcie_tx  pcie_rx  nvl_tx  nvl_rx
              GPU 0   0.712      0.481   0.603   0.698 0.421 1234567  987654   456789  321098
        """
        field_ids = ",".join(str(f) for f in _DCGM_FIELDS)
        gpu_arg   = []
        if self.gpu_ids:
            gpu_arg = ["-i", ",".join(str(g) for g in self.gpu_ids)]

        cmd = (["dcgmi", "dmon",
                "-e", field_ids,
                "-d", str(max(int(self.interval * 1000), 100))]
               + gpu_arg)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)

            for line in proc.stdout:
                if self._stop_evt.is_set():
                    proc.terminate()
                    break
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Entity"):
                    continue
                # Parse: "GPU 0  val1  val2  ..."
                parts = line.split()
                if len(parts) < 3 or parts[0] != "GPU":
                    continue
                try:
                    gpu_idx = int(parts[1])
                    vals    = [float(v) for v in parts[2:]]
                    keys    = list(_DCGM_FIELDS.values())
                    sample  = {"gpu_idx": gpu_idx, "source": "dcgm"}
                    for i, key in enumerate(keys):
                        if i < len(vals):
                            sample[key] = vals[i]
                    # Derive HBM bandwidth from dram_active ratio
                    if "dram_active" in sample:
                        da = sample["dram_active"]
                        # dram_active is 0-1 ratio of HBM cycles in use
                        # Split evenly between read and write as approximation
                        sample["hbm_bw_read_gb_s"]  = round(da * _A100_HBM_PEAK_GB_S * 0.5, 1)
                        sample["hbm_bw_write_gb_s"] = round(da * _A100_HBM_PEAK_GB_S * 0.5, 1)
                    # Convert PCIe/NVLink byte counters to GB/s (delta approach)
                    self.samples.append(sample)
                except (ValueError, IndexError):
                    continue
        except FileNotFoundError:
            # dcgmi disappeared — fall back silently
            self._poll_nvidiasmi()

    # ── nvidia-smi fallback polling ───────────────────────────────────────────

    def _poll_nvidiasmi(self):
        """
        Fallback: poll nvidia-smi for SM clock, memory clock, throttle reasons,
        and coarse utilisation. Runs when DCGM is unavailable.
        """
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,"
            "utilization.gpu,utilization.memory,"
            "clocks.current.sm,clocks.current.memory,"
            "clocks.throttle_reasons.hw_slowdown,"
            "clocks.throttle_reasons.sw_thermal_slowdown",
            "--format=csv,noheader,nounits",
            f"--loop-ms={max(int(self.interval * 1000), 200)}",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:
                if self._stop_evt.is_set():
                    proc.terminate()
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    # throttle reasons come back as "Active"/"Not Active"
                    def _is_active(s):
                        return 1 if "active" in s.lower() and "not" not in s.lower() else 0

                    self.samples.append({
                        "gpu_idx":      int(parts[0]),
                        "source":       "nvidia-smi",
                        # Map to same key names as DCGM path so stop() is uniform
                        "sm_active":    float(parts[1]) / 100.0,   # pct → ratio
                        "dram_active":  float(parts[2]) / 100.0,
                        "sm_clock_mhz": float(parts[3]) if parts[3] != "N/A" else 0.0,
                        "mem_clock_mhz":float(parts[4]) if parts[4] != "N/A" else 0.0,
                        "hw_slowdown":  _is_active(parts[5]),
                        "sw_thermal":   _is_active(parts[6]),
                        # Approximate HBM BW from mem_util
                        "hbm_bw_read_gb_s":  round(float(parts[2]) / 100 * _A100_HBM_PEAK_GB_S * 0.5, 1),
                        "hbm_bw_write_gb_s": round(float(parts[2]) / 100 * _A100_HBM_PEAK_GB_S * 0.5, 1),
                    })
                except (ValueError, IndexError):
                    continue
        except FileNotFoundError:
            pass

    # ── Aggregation ───────────────────────────────────────────────────────────

    def stop(self) -> dict:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)

        if not self.samples:
            return {
                "cuda_kernel_available": False,
                "cuda_source":           "none",
            }

        source = self.samples[0].get("source", "unknown")

        def _agg(key: str):
            """Return (mean, peak, min) for a metric key across all GPUs."""
            vals = [s[key] for s in self.samples if key in s]
            if not vals:
                return 0.0, 0.0, 0.0
            return (round(statistics.mean(vals), 4),
                    round(max(vals), 4),
                    round(min(vals), 4))

        def m(k): return _agg(k)[0]
        def pk(k): return _agg(k)[1]
        def mn_v(k): return _agg(k)[2]

        # Convert raw DCGM ratios (0-1) to percentages for readability
        def pct(k):   return round(m(k)   * 100, 2)
        def pct_pk(k): return round(pk(k) * 100, 2)
        def pct_mn(k): return round(mn_v(k) * 100, 2)

        # PCIe byte counter → GB/s requires delta; accumulate total if available
        def _bytes_to_gbs(key: str) -> float:
            """Approximate GB/s from sum of sampled byte counts / elapsed time."""
            vals = [s[key] for s in self.samples if key in s and s[key] > 0]
            if not vals:
                return 0.0
            total_bytes = max(vals) - min(vals)  # counter delta
            elapsed_s   = len(vals) * self.interval
            return round(total_bytes / max(elapsed_s, 0.001) / 1e9, 3)

        throttled_count = sum(1 for s in self.samples if s.get("hw_slowdown", 0))
        throttled_pct   = round(throttled_count / max(len(self.samples), 1) * 100, 1)

        result = {
            "cuda_kernel_available": True,
            "cuda_source":           source,
            "n_gpus":                len(set(s["gpu_idx"] for s in self.samples)),
            "n_samples":             len(self.samples),

            # ── SM metrics ──────────────────────────────────────────────────
            # sm_active: fraction of time ≥1 warp active on an SM
            #   high (70-90%) = compute-bound (good)
            #   low  (<20%)   = GPU stalled — likely waiting for HBM or SSD data
            "sm_active_mean_pct":    pct("sm_active"),
            "sm_active_peak_pct":    pct_pk("sm_active"),
            "sm_active_min_pct":     pct_mn("sm_active"),   # min reveals stall depth

            # sm_occupancy: warps in flight / max warps per SM
            "sm_occupancy_mean_pct": pct("sm_occupancy"),
            "sm_occupancy_peak_pct": pct_pk("sm_occupancy"),

            # ── Tensor core metrics ─────────────────────────────────────────
            # tensor_active high = GEMM-heavy (prefill attention, MLP layers)
            # tensor_active low  = decode autoregressive step (memory-bound)
            "tensor_active_mean_pct": pct("tensor_active"),
            "tensor_active_peak_pct": pct_pk("tensor_active"),
            "tensor_active_min_pct":  pct_mn("tensor_active"),

            # ── FP16 pipe ────────────────────────────────────────────────────
            "fp16_active_mean_pct":  pct("fp16_active"),
            "fp16_active_peak_pct":  pct_pk("fp16_active"),

            # ── HBM bandwidth metrics ────────────────────────────────────────
            # dram_active: HBM bus busy ratio
            "dram_active_mean_pct":  pct("dram_active"),
            "dram_active_peak_pct":  pct_pk("dram_active"),
            "hbm_bw_read_gb_s_mean": round(m("hbm_bw_read_gb_s"), 1),
            "hbm_bw_read_gb_s_peak": round(pk("hbm_bw_read_gb_s"), 1),
            "hbm_bw_write_gb_s_mean":round(m("hbm_bw_write_gb_s"), 1),
            "hbm_bw_write_gb_s_peak":round(pk("hbm_bw_write_gb_s"), 1),

            # ── Clock metrics ────────────────────────────────────────────────
            "sm_clock_mhz_mean":     round(m("sm_clock_mhz"), 0),
            "sm_clock_mhz_min":      round(mn_v("sm_clock_mhz"), 0),  # min = throttle
            "mem_clock_mhz_mean":    round(m("mem_clock_mhz"), 0),

            # ── Throttle ─────────────────────────────────────────────────────
            "throttled_pct":         throttled_pct,
            "thermal_throttle_pct":  round(
                sum(1 for s in self.samples if s.get("sw_thermal", 0))
                / max(len(self.samples), 1) * 100, 1),

            # ── PCIe / NVLink (DCGM only) ────────────────────────────────────
            "pcie_tx_gb_s":          _bytes_to_gbs("pcie_tx_bytes"),
            "pcie_rx_gb_s":          _bytes_to_gbs("pcie_rx_bytes"),
            "nvlink_tx_gb_s":        _bytes_to_gbs("nvlink_tx_bytes"),
            "nvlink_rx_gb_s":        _bytes_to_gbs("nvlink_rx_bytes"),
        }

        return result


# ════════════════════════════════════════════════════════════════════════════════
# ── NcuAttentionCollector ──────────────────────────────────────────────────────
#
# Wraps `ncu --set full --kernel-name regex:attention` against a target PID.
# Captures roofline metrics for attention kernels:
#   - arithmetic intensity (FLOP/byte)
#   - DRAM read/write bytes
#   - L2 hit rate
#   - SM efficiency
#
# Usage:
#   col = NcuAttentionCollector(server_pid)
#   col.start()
#   # ... run one SGLang request batch ...
#   m = col.stop()    # blocks until ncu finishes; returns aggregated dict
#
# Implementation note:
#   ncu must attach to the already-running SGLang server process via --target-processes
#   all and --pid. On DGX/HPC systems SYS_ADMIN is required; ncu will return an error
#   and available=False if permissions are insufficient.
# ════════════════════════════════════════════════════════════════════════════════

class NcuAttentionCollector:
    """
    Runs ``ncu --set full --kernel-name regex:attention`` against a live
    SGLang server process and extracts per-kernel roofline metrics.

    Metrics collected
    -----------------
    ncu_attention_available   bool   — False if ncu absent or permission denied
    ncu_attention_kernel_count int   — number of attention kernel launches captured
    ncu_dram_read_gb          float  — total DRAM bytes read by attention kernels (GB)
    ncu_dram_write_gb         float  — total DRAM bytes written (GB)
    ncu_l2_hit_rate_pct       float  — L2 cache hit rate across attention kernels (%)
    ncu_arith_intensity       float  — mean arithmetic intensity (FLOP/byte)
    ncu_sm_eff_pct            float  — mean SM active cycles / elapsed cycles (%)
    ncu_duration_us_mean      float  — mean kernel duration in µs
    ncu_duration_us_p99       float  — p99 kernel duration in µs
    ncu_raw_csv               str    — path to raw ncu CSV output file
    """

    # ncu metric names for --metrics flag (subset of --set full)
    _METRICS = ",".join([
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "l2cache__hit_rate.pct",
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
        "gpu__time_duration.sum",
    ])

    def __init__(self, server_pid: int, work_dir: "Path | None" = None,
                 capture_count: int = 100):
        self.server_pid    = server_pid
        self.work_dir      = work_dir
        self.capture_count = capture_count   # max kernel launches to capture
        self._proc         = None
        self._csv_path     = None
        self._thread       = None
        self._result: dict = {}

    def _ncu_available(self) -> bool:
        try:
            subprocess.run(["ncu", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def start(self):
        if not self._ncu_available():
            self._result = {"ncu_attention_available": False,
                            "ncu_attention_reason": "ncu not found"}
            return
        import tempfile
        td = self.work_dir or Path(tempfile.mkdtemp(prefix="amoprof_ncu_"))
        self._csv_path = td / "ncu_attention.csv"
        cmd = [
            "ncu",
            "--target-processes", "all",
            "--pid",              str(self.server_pid),
            "--kernel-name",      "regex:attention|flash_attn|fmha",
            "--launch-count",     str(self.capture_count),
            "--set",              "full",
            "--metrics",          self._METRICS,
            "--csv",
            "--log-file",         str(self._csv_path),
            "--clock-control",    "none",   # don't lock clocks — live server
            # no executable — attaches to running PID
        ]
        try:
            self._proc = _sudo_popen(
                cmd, self.use_sudo, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True,
                cwd=str(td))
        except Exception as e:
            self._result = {"ncu_attention_available": False,
                            "ncu_attention_reason": str(e)}

    def stop(self) -> dict:
        if self._result:          # early-exit set in start()
            return self._result
        if not self._proc:
            return {"ncu_attention_available": False,
                    "ncu_attention_reason": "ncu not started"}
        try:
            _, stderr = self._proc.communicate(timeout=120)
            if self._proc.returncode != 0:
                reason = (stderr or "")[:200]
                if "permission" in reason.lower() or "sys_admin" in reason.lower():
                    return {"ncu_attention_available": False,
                            "ncu_attention_reason": "permission denied (need SYS_ADMIN)"}
                return {"ncu_attention_available": False,
                        "ncu_attention_reason": reason}
        except subprocess.TimeoutExpired:
            self._proc.kill()
            return {"ncu_attention_available": False,
                    "ncu_attention_reason": "ncu timed out after 120s"}

        return self._parse_csv()

    def _parse_csv(self) -> dict:
        if not self._csv_path or not self._csv_path.exists():
            return {"ncu_attention_available": False,
                    "ncu_attention_reason": "no csv output"}
        try:
            import csv as _csv
            rows = []
            with open(self._csv_path, newline="") as f:
                # ncu CSV has a multi-line preamble; skip until "ID," header
                reader = None
                for line in f:
                    if line.startswith("\"ID\"") or line.startswith("ID,"):
                        import io
                        remainder = line + f.read()
                        reader = _csv.DictReader(io.StringIO(remainder))
                        rows = list(reader)
                        break

            if not rows:
                return {"ncu_attention_available": False,
                        "ncu_attention_reason": "csv empty or unparseable"}

            def _gb(metric):
                vals = []
                for r in rows:
                    v = r.get(metric, "").replace(",", "").strip()
                    try: vals.append(float(v))
                    except: pass
                return round(sum(vals) / 1e9, 4) if vals else 0.0

            def _mean_pct(metric):
                vals = []
                for r in rows:
                    v = r.get(metric, "").replace(",", "").strip()
                    try: vals.append(float(v))
                    except: pass
                return round(statistics.mean(vals), 2) if vals else 0.0

            def _dur_us(rows_):
                vals = []
                for r in rows_:
                    v = r.get("gpu__time_duration.sum", "").replace(",","").strip()
                    try: vals.append(float(v) / 1000.0)  # ns → µs
                    except: pass
                return vals

            dram_r_gb  = _gb("dram__bytes_read.sum")
            dram_w_gb  = _gb("dram__bytes_write.sum")
            l2_hit     = _mean_pct("l2cache__hit_rate.pct")
            sm_eff     = _mean_pct("sm__cycles_active.avg.pct_of_peak_sustained_elapsed")
            dur_us     = _dur_us(rows)
            dur_mean   = round(statistics.mean(dur_us), 2) if dur_us else 0.0
            dur_p99    = round(sorted(dur_us)[int(len(dur_us)*0.99)] if dur_us else 0.0, 2)

            # Arithmetic intensity: total FLOP / total bytes
            fp16_vals  = []
            for r in rows:
                v = r.get("smsp__sass_thread_inst_executed_op_fp16_pred_on.sum","").replace(",","")
                try: fp16_vals.append(float(v) * 2)   # 2 FLOP/FMA
                except: pass
            total_flop  = sum(fp16_vals)
            total_bytes = (dram_r_gb + dram_w_gb) * 1e9
            arith_int   = round(total_flop / max(total_bytes, 1), 3)

            return {
                "ncu_attention_available":   True,
                "ncu_attention_kernel_count": len(rows),
                "ncu_dram_read_gb":           dram_r_gb,
                "ncu_dram_write_gb":          dram_w_gb,
                "ncu_l2_hit_rate_pct":        l2_hit,
                "ncu_sm_eff_pct":             sm_eff,
                "ncu_arith_intensity":        arith_int,
                "ncu_duration_us_mean":       dur_mean,
                "ncu_duration_us_p99":        dur_p99,
                "ncu_raw_csv":                str(self._csv_path),
            }
        except Exception as e:
            return {"ncu_attention_available": False,
                    "ncu_attention_reason": f"parse error: {e}"}


# ════════════════════════════════════════════════════════════════════════════════
# ── NsysTraceCollector ─────────────────────────────────────────────────────────
#
# Wraps ``nsys profile --trace=cuda,nvtx`` for a time-bounded capture window.
# Since SGLang is already running, this attaches as a system-wide trace for a
# fixed duration, then exports to SQLite and queries the CUDA API timeline.
# ════════════════════════════════════════════════════════════════════════════════

class NsysTraceCollector:
    """
    Captures a time-bounded Nsight Systems trace (CUDA + NVTX) and extracts:

    nsys_available              bool
    nsys_cuda_api_calls         int    — total CUDA API calls in window
    nsys_kernel_count           int    — unique kernel launches
    nsys_memcpy_h2d_gb          float  — Host→Device transfers (GB)
    nsys_memcpy_d2h_gb          float  — Device→Host transfers (GB)
    nsys_memset_gb              float  — cudaMemset volume (GB)
    nsys_kernel_top5            str    — JSON: top-5 kernels by total duration
    nsys_gpu_active_pct         float  — % of trace window with ≥1 kernel on GPU
    nsys_report_path            str    — path to .nsys-rep file
    nsys_sqlite_path            str    — path to exported .sqlite file
    """

    def __init__(self, server_pid: int, capture_duration_s: int = 30,
                 work_dir: "Path | None" = None):
        self.server_pid        = server_pid
        self.capture_duration  = capture_duration_s
        self.work_dir          = work_dir
        self._proc             = None
        self._rep_path: "Path | None" = None
        self._result: dict     = {}

    def _nsys_available(self) -> bool:
        try:
            subprocess.run(["nsys", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def start(self):
        if not self._nsys_available():
            self._result = {"nsys_available": False,
                            "nsys_reason": "nsys not found"}
            return
        import tempfile
        td = self.work_dir or Path(tempfile.mkdtemp(prefix="amoprof_nsys_"))
        self._rep_path = td / "amoprof_nsys.nsys-rep"
        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            "--duration",        str(self.capture_duration),
            "--pid",             str(self.server_pid),
            "--output",          str(self._rep_path.with_suffix("")),
            "--force-overwrite=true",
            "--export",          "sqlite",
            "--stop-on-exit=true",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, cwd=str(td))
        except Exception as e:
            self._result = {"nsys_available": False, "nsys_reason": str(e)}

    def stop(self) -> dict:
        if self._result:
            return self._result
        if not self._proc:
            return {"nsys_available": False, "nsys_reason": "not started"}
        try:
            _interrupted = bool(getattr(self, "_amoprof_interrupted", False))
            _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (8.0 if _interrupted else self.capture_duration + 30))
            if _interrupted and self._proc.poll() is None:
                try: self._proc.terminate()
                except Exception: pass
            _, stderr = self._proc.communicate(timeout=_timeout)
            if self._proc.returncode not in (0, 1):
                return {"nsys_available": False,
                        "nsys_reason": (stderr or "")[:200]}
        except subprocess.TimeoutExpired:
            self._proc.kill()
            return {"nsys_available": False, "nsys_reason": "nsys timed out during interrupt fast-stop" if bool(getattr(self, "_amoprof_interrupted", False)) else "nsys timed out"}

        return self._parse_sqlite()

    def _parse_sqlite(self) -> dict:
        sq = self._rep_path.with_suffix(".sqlite") if self._rep_path else None
        if not sq or not sq.exists():
            # Try nsys export manually
            try:
                subprocess.run(
                    ["nsys", "export", "--type=sqlite",
                     str(self._rep_path),
                     "--output", str(sq)],
                    capture_output=True, timeout=60)
            except Exception:
                pass
        if not sq or not sq.exists():
            return {"nsys_available": False, "nsys_reason": "sqlite export failed",
                    "nsys_report_path": str(self._rep_path or "")}
        try:
            import sqlite3
            con = sqlite3.connect(str(sq))
            cur = con.cursor()

            # Total CUDA API calls
            try:
                cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
                api_calls = cur.fetchone()[0]
            except Exception:
                api_calls = 0

            # Kernel count + top-5 by total duration
            try:
                cur.execute("""
                    SELECT k.shortName,
                           COUNT(*) as launches,
                           SUM(e.end - e.start) as total_ns
                    FROM   CUPTI_ACTIVITY_KIND_KERNEL e
                    JOIN   StringIds k ON k.id = e.shortName
                    GROUP  BY k.shortName
                    ORDER  BY total_ns DESC
                    LIMIT  5
                """)
                top5 = [{"name": r[0], "launches": r[1],
                          "total_ms": round(r[2]/1e6, 2)}
                        for r in cur.fetchall()]
                cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
                kernel_count = cur.fetchone()[0]
            except Exception:
                top5, kernel_count = [], 0

            # MemCpy volumes
            def _memcpy_gb(copy_kind_int):
                try:
                    cur.execute(
                        "SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                        "WHERE copyKind=?", (copy_kind_int,))
                    v = cur.fetchone()[0] or 0
                    return round(v / 1e9, 4)
                except Exception:
                    return 0.0
            h2d_gb = _memcpy_gb(1)   # copyKind=1 → H2D
            d2h_gb = _memcpy_gb(2)   # copyKind=2 → D2H

            # cudaMemset
            try:
                cur.execute("SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET")
                memset_gb = round((cur.fetchone()[0] or 0) / 1e9, 4)
            except Exception:
                memset_gb = 0.0

            # GPU active %: fraction of trace window with ≥1 kernel active
            try:
                cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
                trace_start, trace_end = cur.fetchone()
                trace_dur = max((trace_end or 0) - (trace_start or 0), 1)
                cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
                kernel_total_ns = cur.fetchone()[0] or 0
                gpu_active_pct = round(min(kernel_total_ns / trace_dur * 100, 100), 1)
            except Exception:
                gpu_active_pct = 0.0

            con.close()
            return {
                "nsys_available":      True,
                "nsys_cuda_api_calls": api_calls,
                "nsys_kernel_count":   kernel_count,
                "nsys_memcpy_h2d_gb":  h2d_gb,
                "nsys_memcpy_d2h_gb":  d2h_gb,
                "nsys_memset_gb":      memset_gb,
                "nsys_kernel_top5":    json.dumps(top5),
                "nsys_gpu_active_pct": gpu_active_pct,
                "nsys_report_path":    str(self._rep_path),
                "nsys_sqlite_path":    str(sq),
            }
        except Exception as e:
            return {"nsys_available": False, "nsys_reason": f"sqlite parse: {e}",
                    "nsys_report_path": str(self._rep_path or "")}


# ════════════════════════════════════════════════════════════════════════════════
# ── PerfStatCollector ─────────────────────────────────────────────────────────
#
# Wraps ``perf stat -e mem_load_retired.l3_miss,mem_inst_retired.all_loads``
# against the SGLang server PID for the duration of one inference batch.
# ════════════════════════════════════════════════════════════════════════════════

class PerfStatCollector:
    """
    Attaches ``perf stat`` to the running SGLang server PID and captures
    L3 miss rate and total memory load instruction counts for the window.

    Metrics
    -------
    perf_available              bool
    perf_l3_miss_count          int    — L3 miss events in window
    perf_all_loads_count        int    — total memory load instructions
    perf_l3_miss_rate_pct       float  — l3_miss / all_loads × 100
    perf_l3_miss_per_s          float  — L3 misses/second
    perf_duration_s             float  — measurement window duration
    perf_raw_output             str    — raw perf stat stderr text
    """

    _EVENTS = "mem_load_retired.l3_miss,mem_inst_retired.all_loads"

    def __init__(self, server_pid: int, use_sudo: bool = True):
        self.server_pid = server_pid
        self.use_sudo  = use_sudo
        self._proc      = None
        self._t0: float = 0.0
        self._result: dict = {}

    def _perf_available(self) -> bool:
        try:
            r = subprocess.run(["perf", "stat", "--help"],
                               capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def start(self):
        if not self._perf_available():
            self._result = {"perf_available": False,
                            "perf_reason": "perf not found"}
            return
        cmd = [
            "perf", "stat",
            "-e",  self._EVENTS,
            "-p",  str(self.server_pid),
            "--field-separator", ",",
        ]
        try:
            # perf stat writes metrics to stderr when interrupted
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, cwd=str(td))
            self._t0 = time.time()
        except Exception as e:
            self._result = {"perf_available": False, "perf_reason": str(e)}

    def stop(self) -> dict:
        if self._result:
            return self._result
        if not self._proc:
            return {"perf_available": False, "perf_reason": "not started"}
        elapsed = max(time.time() - self._t0, 0.001)
        try:
            self._proc.send_signal(__import__("signal").SIGINT)
            _, stderr = self._proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            _, stderr = self._proc.communicate()
        except Exception as e:
            return {"perf_available": False, "perf_reason": str(e)}

        return self._parse(stderr or "", elapsed)

    def _parse(self, raw: str, elapsed_s: float) -> dict:
        """
        Parse CSV perf stat output.  The --field-separator , format is:
          value,unit,event,run_time_ns,pct_running,...
        """
        l3_miss = all_loads = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Also handle the plain human-readable format as fallback
            if "mem_load_retired.l3_miss" in line:
                m = re.search(r"([\d,]+)\s+mem_load_retired\.l3_miss", line)
                if m:
                    try: l3_miss = int(m.group(1).replace(",", ""))
                    except: pass
                # CSV path: first field is value
                parts = line.split(",")
                if parts:
                    try: l3_miss = int(parts[0].replace(",","").strip())
                    except: pass
            if "mem_inst_retired.all_loads" in line:
                m = re.search(r"([\d,]+)\s+mem_inst_retired\.all_loads", line)
                if m:
                    try: all_loads = int(m.group(1).replace(",", ""))
                    except: pass
                parts = line.split(",")
                if parts:
                    try: all_loads = int(parts[0].replace(",","").strip())
                    except: pass

        if l3_miss == 0 and all_loads == 0 and "not supported" in raw.lower():
            return {"perf_available": False,
                    "perf_reason": "PMU events not supported on this CPU"}

        miss_rate = round(l3_miss / max(all_loads, 1) * 100, 4)
        return {
            "perf_available":        True,
            "perf_l3_miss_count":    l3_miss,
            "perf_all_loads_count":  all_loads,
            "perf_l3_miss_rate_pct": miss_rate,
            "perf_l3_miss_per_s":    round(l3_miss / max(elapsed_s, 0.001), 1),
            "perf_duration_s":       round(elapsed_s, 3),
            "perf_raw_output":       raw[-2000:],
        }


# ════════════════════════════════════════════════════════════════════════════════
# ── PcmMemoryCollector ─────────────────────────────────────────────────────────
#
# Reads socket-level DRAM bandwidth via Intel PCM (pcm-memory).
# Falls back to reading /sys/devices/uncore_imc_*/events/cas_count_read if
# pcm-memory is unavailable (works on modern kernels without PCM binary).
# ════════════════════════════════════════════════════════════════════════════════

class PcmMemoryCollector:
    """
    Tracks socket-level DRAM bandwidth using Intel PCM or IMC PMU counters.

    Metrics
    -------
    pcm_available               bool
    pcm_dram_read_gb_s          float  — mean DRAM read BW across all sockets (GB/s)
    pcm_dram_write_gb_s         float  — mean DRAM write BW across all sockets (GB/s)
    pcm_dram_total_gb_s         float  — read + write
    pcm_dram_read_gb_s_peak     float  — peak in window
    pcm_dram_write_gb_s_peak    float  — peak in window
    pcm_samples                 int
    pcm_source                  str    — "pcm-memory" | "imc_pmu" | "unavailable"
    """

    def __init__(self, interval_s: float = 1.0, binary: str | None = None,
                 force_perf_imc: bool = False, use_sudo: bool = True,
                 work_dir: "Path | str | None" = None):
        self.interval  = interval_s
        self.binary    = binary
        self.force_perf_imc = force_perf_imc
        self.use_sudo  = use_sudo
        self.work_dir  = Path(work_dir) if work_dir is not None else None
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[dict] = []
        self._proc      = None
        self._thread    = None
        self._stop      = threading.Event()
        self._source    = "unavailable"
        self._t0        = 0.0
        self._raw_path: "Path | None" = (self.work_dir / "pcm_memory_raw.csv" if self.work_dir is not None else None)
        self._raw_fh = None
        self._command = ""
        self._reason = ""
        self._stderr_path: "Path | None" = (self.work_dir / "pcm_memory_stderr.txt" if self.work_dir is not None else None)
        self._stderr_thread = None
        self._stderr_lines: list[str] = []

    # ── PCM binary path ───────────────────────────────────────────────────────

    def _pcm_path(self) -> "str | None":
        candidates = []
        if self.binary:
            candidates.append(self.binary)
        candidates += ["pcm-memory", "pcm-memory.x", "/usr/local/bin/pcm-memory",
                       "/usr/local/sbin/pcm-memory", "/opt/intel/pcm/bin/pcm-memory",
                       "/opt/intel/pcm/pcm-memory"]
        seen = set()
        for p in candidates:
            if not p or p in seen:
                continue
            seen.add(p)
            try:
                subprocess.run([p, "--help"], capture_output=True, timeout=3)
                return p
            except Exception:
                pass
        return None

    # ── IMC PMU fallback via perf ─────────────────────────────────────────────

    def _imc_perf_available(self) -> bool:
        """Check if perf can access uncore_imc events."""
        try:
            r = subprocess.run(
                ["perf", "list", "uncore_imc"],
                capture_output=True, text=True, timeout=5)
            return "cas_count" in r.stdout.lower() or "cas_count" in r.stderr.lower()
        except Exception:
            return False

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._t0 = time.time()
        pcm = None if self.force_perf_imc else self._pcm_path()
        if pcm:
            self._source = "intel-pcm/pcm-memory"
            self._start_pcm(pcm)
        elif self._imc_perf_available():
            self._source = "imc_pmu"
            self._thread = threading.Thread(
                target=self._poll_imc_pmu, daemon=True)
            self._thread.start()
        else:
            self._source = "unavailable"

    def _start_pcm(self, pcm_bin: str):
        """Launch Intel pcm-memory in CSV mode and parse its streaming output."""
        cmd = [pcm_bin, str(self.interval), "-csv"]
        self._command = " ".join(str(x) for x in _sudo_cmd(cmd, self.use_sudo))
        try:
            self._proc = _sudo_popen(
                cmd, use_sudo=self.use_sudo, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                cwd=str(self.work_dir) if self.work_dir is not None else None)
            self._thread = threading.Thread(
                target=self._read_pcm, daemon=True)
            self._thread.start()
            self._stderr_thread = threading.Thread(
                target=self._read_pcm_stderr, daemon=True)
            self._stderr_thread.start()
        except Exception as e:
            self._source = "unavailable"
            self._reason = f"failed to launch pcm-memory: {e}"

    def _read_pcm_stderr(self):
        """Capture pcm-memory's own stderr — otherwise a real failure (e.g. no
        MSR/PCI access to program the uncore/iMC counters, common on
        non-metal EC2 instances) produces zero stdout rows with no clue why."""
        if not self._proc or not self._proc.stderr:
            return
        fh = None
        try:
            if self._stderr_path is not None:
                fh = open(self._stderr_path, "w", encoding="utf-8", buffering=1)
        except Exception:
            fh = None
        try:
            for line in self._proc.stderr:
                if fh:
                    try:
                        fh.write(line)
                    except Exception:
                        pass
                line = line.strip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 50:
                        self._stderr_lines.pop(0)
        except Exception:
            pass
        finally:
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    def _stderr_tail(self, max_chars: int = 500) -> str:
        text = " | ".join(self._stderr_lines)
        return text[-max_chars:] if text else ""

    @staticmethod
    def _num(v):
        try:
            t = str(v).strip().strip('"').replace(",", "")
            if not t or t.upper() in {"NA", "N/A", "-", "--"}:
                return None
            return float(t)
        except Exception:
            return None

    @staticmethod
    def _bw_scale_from_header(h: str) -> float:
        x = str(h).lower()
        if "kb/s" in x or "kbytes/s" in x or "kbyte/s" in x:
            return 1.0 / (1024 * 1024)
        if "mb/s" in x or "mbytes/s" in x or "mbyte/s" in x:
            return 1.0 / 1024
        if "gb/s" in x or "gbytes/s" in x or "gbyte/s" in x:
            return 1.0
        # Intel pcm-memory CSV emits MB/s for the unqualified System Read/Write/Memory
        # columns in common two-row-header output.  Older AMOprof treated these
        # unqualified columns as GB/s or lost them entirely; default to MB/s here
        # unless a unit is explicitly present.
        return 1.0 / 1024

    @staticmethod
    def _csv_split(line: str, delim: str) -> list[str]:
        try:
            return [x.strip().strip('"') for x in next(csv.reader([line], delimiter=delim, skipinitialspace=True))]
        except Exception:
            return [x.strip().strip('"') for x in str(line).split(delim)]

    @staticmethod
    def _unique_headers(cols: list[str]) -> list[str]:
        seen = {}
        out = []
        for c in cols:
            base = (str(c).strip() or "col")
            n = seen.get(base, 0)
            seen[base] = n + 1
            out.append(base if n == 0 else f"{base}.{n+1}")
        return out

    @classmethod
    def _combine_pcm_headers(cls, group_line: str | None, header_line: str, delim: str) -> list[str]:
        names = cls._csv_split(header_line, delim)
        groups = cls._csv_split(group_line, delim) if group_line else []
        if len(groups) != len(names):
            return cls._unique_headers(names)
        out = []
        for g, h in zip(groups, names):
            g = str(g or '').strip()
            h = str(h or '').strip()
            # pcm-memory often has a first row like: ,,SKT0,...,SKT1,...,System,System,System
            # and a second row like: Date,Time,Ch0Read,...,Read,Write,Memory.
            # Preserve both levels so System Read/Write can be selected and duplicate
            # per-socket column names do not overwrite each other.
            if g and h and h.lower() not in {"date", "time"}:
                out.append(f"{g} {h}")
            else:
                out.append(h or g)
        return cls._unique_headers(out)

    @classmethod
    def parse_pcm_memory_raw_csv(cls, path: "str | Path", *, source: str = "intel-pcm/pcm-memory",
                                 start_ts: float | None = None) -> list[dict]:
        """Parse Intel pcm-memory CSV, including two-row SKT/System headers.

        Returns AMOprof-normalized samples in GB/s.  This is used both by the
        live collector and by the analyzer to repair older raw directories where
        pcm_timeseries.csv was written as zeros even though pcm_memory_raw.csv
        contains real System Read/Write/Memory values.
        """
        try:
            lines = Path(path).read_text(errors="replace").splitlines()
        except Exception:
            return []
        prev = None
        header = None
        delim = ","
        out: list[dict] = []
        t0 = start_ts or time.time()
        for line in lines:
            if not str(line).strip():
                continue
            d = ";" if line.count(";") >= line.count(",") else ","
            if header is None:
                if cls._looks_like_pcm_header(line, d):
                    delim = d
                    header = cls._combine_pcm_headers(prev, line, delim)
                else:
                    prev = line
                continue
            parts = cls._csv_split(line, delim)
            if len(parts) < len(header):
                continue
            if len(parts) > len(header):
                parts = parts[:len(header)]
            row = dict(zip(header, parts))
            dummy = cls()
            read_gb_s, write_gb_s, total_gb_s, schema = dummy._extract_pcm_bw(row)
            if total_gb_s <= 0:
                continue
            ts = time.time()
            # Prefer PCM Date/Time when present for stable offline analysis.
            try:
                import datetime as _dt
                date = row.get("Date") or row.get("date")
                tm = row.get("Time") or row.get("time")
                if date and tm:
                    dt = _dt.datetime.fromisoformat(str(date).strip() + "T" + str(tm).strip())
                    ts = dt.replace(tzinfo=_dt.timezone.utc).timestamp()
            except Exception:
                pass
            out.append({
                "ts": round(float(ts), 6),
                "timestamp_epoch": round(float(ts), 6),
                "timestamp_utc": cls._utc(float(ts)),
                "time_sec": round(max(float(ts) - float(t0), 0.0), 6),
                "read": round(read_gb_s, 6),
                "write": round(write_gb_s, 6),
                "dram_read_gb_s": round(read_gb_s, 6),
                "dram_write_gb_s": round(write_gb_s, 6),
                "dram_total_gb_s": round(total_gb_s, 6),
                "pcm_dram_read_gb_s": round(read_gb_s, 6),
                "pcm_dram_write_gb_s": round(write_gb_s, 6),
                "pcm_dram_total_gb_s": round(total_gb_s, 6),
                "dram_source": source,
                "pcm_parse_schema": schema,
            })
        if out:
            base = float(out[0].get("timestamp_epoch", out[0].get("ts", 0)) or 0)
            for i, r in enumerate(out):
                r["iteration"] = i
                te = float(r.get("timestamp_epoch", r.get("ts", base)) or base)
                r["time_sec"] = round(max(te - base, 0.0), 6)
        return out

    @staticmethod
    def _utc(ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return ""

    @staticmethod
    def _looks_like_pcm_header(line: str, delim: str) -> bool:
        low = line.lower()
        if delim not in line:
            return False
        # Intel PCM often prints banner / socket topology lines before the CSV
        # header. The real memory CSV header contains at least read/write BW
        # terms and usually System/Socket/SKT columns.
        has_read = any(x in low for x in ("read", "rdbw", "read bw", "mem read"))
        has_write = any(x in low for x in ("write", "wrbw", "write bw", "mem write"))
        has_mem = any(x in low for x in ("memory", "mem", "dram", "socket", "skt", "system"))
        return has_read and has_write and has_mem

    def _extract_pcm_bw(self, row: dict[str, str]) -> tuple[float, float, float, str]:
        """Return read/write/total GB/s from one Intel PCM row.

        Prefer System/Total memory columns when present.  If only per-socket
        columns exist, sum those.  This avoids double-counting rows that expose
        both per-socket and already-aggregated System columns.
        """
        entries = []
        for k, v in row.items():
            lk = str(k).strip().strip('"').lower().replace("_", " ").replace("-", " ")
            fv = self._num(v)
            if fv is None:
                continue
            if not any(x in lk for x in ("read", "write", "rdbw", "wrbw", "memory", "mem", "dram")):
                continue
            if not any(x in lk for x in ("read", "write", "rdbw", "wrbw")):
                continue
            direction = "read" if ("read" in lk or "rdbw" in lk or "rd bw" in lk) else "write"
            scale = self._bw_scale_from_header(k)
            is_system = any(x in lk for x in ("system", "total", "all sockets", "aggregate"))
            is_socket = any(x in lk for x in ("socket", "skt", "package", "node", "channel", "imc"))
            entries.append((direction, fv * scale, is_system, is_socket, lk))

        # Prefer aggregate System/Total columns if available.
        use = [e for e in entries if e[2]] or [e for e in entries if e[3]] or entries
        read = sum(e[1] for e in use if e[0] == "read")
        write = sum(e[1] for e in use if e[0] == "write")
        total = read + write
        source = "system_total_columns" if any(e[2] for e in use) else ("per_socket_sum" if any(e[3] for e in use) else "best_effort_columns")
        return read, write, total, source

    def _read_pcm(self):
        """Parse Intel pcm-memory CSV robustly across PCM versions."""
        if not self._proc:
            return
        if self._raw_path is not None:
            try:
                self._raw_fh = open(self._raw_path, "w", encoding="utf-8", buffering=1)
            except Exception:
                self._raw_fh = None
        header = None
        prev_header_line = None
        try:
            iterator = self._proc.stdout or []
            for line in iterator:
                if self._raw_fh:
                    try:
                        self._raw_fh.write(line)
                    except Exception:
                        pass
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                delim = ";" if line.count(";") >= line.count(",") else ","
                if header is None:
                    if not self._looks_like_pcm_header(line, delim):
                        prev_header_line = line
                        continue
                    self._pcm_delim = delim
                    header = self._combine_pcm_headers(prev_header_line, line, delim)
                    continue
                delim = getattr(self, "_pcm_delim", ";")
                parts = self._csv_split(line, delim)
                if len(parts) < len(header):
                    continue
                if len(parts) > len(header):
                    parts = parts[:len(header)]
                row = dict(zip(header, parts))
                try:
                    read_gb_s, write_gb_s, total_gb_s, schema = self._extract_pcm_bw(row)
                    if total_gb_s <= 0:
                        continue
                    ts = time.time()
                    self._samples.append({
                        "ts": round(ts, 6),
                        "timestamp_epoch": round(ts, 6),
                        "timestamp_utc": self._utc(ts),
                        "time_sec": round(ts - (self._t0 or ts), 6),
                        "read": round(read_gb_s, 6), "write": round(write_gb_s, 6),
                        "dram_read_gb_s": round(read_gb_s, 6),
                        "dram_write_gb_s": round(write_gb_s, 6),
                        "dram_total_gb_s": round(total_gb_s, 6),
                        "pcm_dram_read_gb_s": round(read_gb_s, 6),
                        "pcm_dram_write_gb_s": round(write_gb_s, 6),
                        "pcm_dram_total_gb_s": round(total_gb_s, 6),
                        "dram_source": self._source,
                        "pcm_parse_schema": schema,
                    })
                except Exception as e:
                    self._reason = f"pcm row parse failed: {e}"
                    pass
        finally:
            if self._raw_fh:
                try:
                    self._raw_fh.close()
                except Exception:
                    pass
                self._raw_fh = None

    def _poll_imc_pmu(self):
        """
        Poll IMC CAS counters via perf stat --all-sockets every interval_s.
        CAS_COUNT_RD * 64 bytes / interval → GB/s.
        """
        cmd = [
            "perf", "stat", "-a",
            "-e", "uncore_imc/cas_count_read/,uncore_imc/cas_count_write/",
            "-I", str(int(self.interval * 1000)),
            "--field-separator", ",",
            "sleep", "86400",
        ]
        try:
            proc = _sudo_popen(
                cmd, use_sudo=self.use_sudo, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True,
                cwd=str(self.work_dir) if self.work_dir is not None else None)
            for line in proc.stderr:
                if self._stop.is_set():
                    proc.terminate()
                    break
                # format: timestamp,count,unit,event,duration,...
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    count = int(parts[1].replace(",","").strip())
                    event = parts[3].strip().lower()
                    dt    = self.interval
                    bw    = count * 64 / 1e9 / max(dt, 0.001)
                    ts = time.time()
                    base = {
                        "ts": round(ts, 6),
                        "timestamp_epoch": round(ts, 6),
                        "timestamp_utc": self._utc(ts),
                        "time_sec": round(ts - (self._t0 or ts), 6),
                        "dram_source": self._source,
                        "pcm_parse_schema": "perf_uncore_imc_cas_count",
                    }
                    if "read" in event:
                        self._samples.append({**base, "read": round(bw,6), "write": 0,
                                              "dram_read_gb_s": round(bw,6), "dram_write_gb_s": 0,
                                              "dram_total_gb_s": round(bw,6),
                                              "pcm_dram_read_gb_s": round(bw,6), "pcm_dram_write_gb_s": 0,
                                              "pcm_dram_total_gb_s": round(bw,6)})
                    elif "write" in event:
                        if self._samples:
                            self._samples[-1]["write"] = round(bw,6)
                            self._samples[-1]["dram_write_gb_s"] = round(bw,6)
                            self._samples[-1]["pcm_dram_write_gb_s"] = round(bw,6)
                            total = self._samples[-1].get("dram_read_gb_s", 0) + bw
                            self._samples[-1]["dram_total_gb_s"] = round(total,6)
                            self._samples[-1]["pcm_dram_total_gb_s"] = round(total,6)
                        else:
                            self._samples.append({**base, "read": 0, "write": round(bw,6),
                                                  "dram_read_gb_s": 0, "dram_write_gb_s": round(bw,6),
                                                  "dram_total_gb_s": round(bw,6),
                                                  "pcm_dram_read_gb_s": 0, "pcm_dram_write_gb_s": round(bw,6),
                                                  "pcm_dram_total_gb_s": round(bw,6)})
                except Exception:
                    pass
        except Exception as e:
            self._reason = f"perf imc polling failed: {e}"

    def stop(self) -> dict:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try: self._proc.wait(timeout=5)
            except Exception: pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)

        if self._source == "unavailable" or not self._samples:
            return {"pcm_available":  False,
                    "pcm_source":     self._source,
                    "pcm_reason":     self._reason or self._stderr_tail() or "no Intel PCM/perf IMC samples",
                    "pcm_samples":    0,
                    "pcm_raw_path":   str(self._raw_path) if self._raw_path is not None else "",
                    "pcm_stderr_path": str(self._stderr_path) if self._stderr_path is not None else ""}

        reads  = [float(s.get("dram_read_gb_s", s.get("read", 0)) or 0) for s in self._samples]
        writes = [float(s.get("dram_write_gb_s", s.get("write", 0)) or 0) for s in self._samples]
        totals = [float(s.get("dram_total_gb_s", r + w) or 0) for s, r, w in zip(self._samples, reads, writes)]

        def mn(xs): return round(statistics.mean(xs), 6) if xs else 0.0
        def pk(xs): return round(max(xs), 6) if xs else 0.0
        nonzero = sum(1 for x in totals if abs(float(x or 0)) > 0)
        elapsed = max(time.time() - (self._t0 or time.time()), 0.001)

        return {
            "pcm_available":            bool(nonzero),
            "pcm_source":               self._source,
            "dram_collector":           self._source,
            "pcm_dram_read_gb_s":       mn(reads),
            "pcm_dram_write_gb_s":      mn(writes),
            "pcm_dram_total_gb_s":      mn(totals),
            "dram_read_gb_s_mean":      mn(reads),
            "dram_write_gb_s_mean":     mn(writes),
            "dram_total_gb_s_mean":     mn(totals),
            "pcm_dram_read_gb_s_peak":  pk(reads),
            "pcm_dram_write_gb_s_peak": pk(writes),
            "pcm_dram_total_gb_s_peak": pk(totals),
            "dram_read_gb_s_peak":      pk(reads),
            "dram_write_gb_s_peak":     pk(writes),
            "dram_total_gb_s_peak":     pk(totals),
            "pcm_samples":              len(self._samples),
            "pcm_nonzero_samples":      nonzero,
            "pcm_duration_s":           round(elapsed, 3),
            "pcm_raw_path":             str(self._raw_path) if self._raw_path is not None else "",
            "pcm_stderr_path":          str(self._stderr_path) if self._stderr_path is not None else "",
            "pcm_reason":               "" if nonzero else (self._reason or self._stderr_tail() or "all Intel PCM samples were zero"),
        }


# ════════════════════════════════════════════════════════════════════════════════

# ── AMDuProfPcmMemoryCollector ────────────────────────────────────────────────
class AMDuProfPcmMemoryCollector:
    """Capture DRAM read/write bandwidth and transactions with AMD uProf PCM."""
    def __init__(self, duration_s: float = 10.0, interval_s: float = 1.0,
                 binary: str = "/opt/AMDuProf_5.2-606/bin/AMDuProfPcm",
                 output_csv: str | None = None, extra_args: list[str] | None = None,
                 use_sudo: bool = True):
        self.duration_s = max(float(duration_s), 1.0)
        self.interval_s = max(float(interval_s), 0.1)
        self.binary = binary
        self.output_csv = output_csv
        self.extra_args = extra_args or []
        self.use_sudo = use_sudo
        self.samples: list[dict] = []
        self._proc = None
        self._thread = None
        self._result: dict = {}
        self._raw_output = ""
        self._csv_path: str | None = None
        self._command: str = ""
        self._t0 = 0.0

    def start(self):
        self.samples.clear(); self._result = {}; self._raw_output = ""; self._t0 = time.time()
        if not (self.binary and Path(self.binary).exists() and os.access(self.binary, os.X_OK)):
            self._result = {"amduprof_pcm_available": False, "amduprof_pcm_reason": f"AMDuProfPcm not found or not executable: {self.binary}"}
            return
        out_csv = Path(self.output_csv) if self.output_csv else Path(f"/tmp/amoprof_amduprof_pcm_{int(self._t0)}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # AMDuProf 5.2-606 rejects -i (interval) in -m memory mode with
        # "Error: Failed to process args." It does accept -d (duration in
        # seconds) and auto-adjusts the sampling interval to its minimum
        # (~1200ms). Use -d so the binary self-terminates after our
        # collection window, matching the working command pattern:
        #     AMDuProfPcm -m memory -a -d <secs> -o <path> --msr
        # AMDuProf complains if the output file already exists, so unlink
        # first.
        try:
            out_csv.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self._csv_path = str(out_csv)
        duration_secs = max(int(self.duration_s), 2)
        # Reset/clean AMDuProfPcm internal state before capture.  This avoids
        # stale counters/state from a previous run contaminating the DRAM-BW
        # timeseries.  Do not duplicate -r if the user already supplied it via
        # --amduprof-extra-arg.
        reset_args = [] if any(str(a) == "-r" for a in self.extra_args) else ["-r"]
        cmd = [self.binary] + reset_args + ["-m", "memory", "-a", "--msr",
               "-d", str(duration_secs), "-o", str(out_csv)] + self.extra_args
        # AMDuProfPcm normally needs MSR access.  If amoprof is not running as
        # root, run through sudo -n so collection either succeeds non-
        # interactively or fails clearly instead of hanging for a password.
        run_cmd = _sudo_cmd(cmd, self.use_sudo)
        self._command = " ".join(str(x) for x in run_cmd)
        try:
            self._proc = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          cwd=str(out_csv.parent))
            self._thread = threading.Thread(target=self._wait_and_parse, daemon=True); self._thread.start()
        except Exception as e:
            self._result = {"amduprof_pcm_available": False, "amduprof_pcm_reason": str(e), "amduprof_pcm_command": " ".join(str(x) for x in run_cmd)}

    @staticmethod
    def _f(v):
        try:
            t = str(v).strip().replace(",", "")
            if not t or t.upper() in {"NA", "N/A", "-"}: return None
            return float(t)
        except Exception: return None

    @staticmethod
    def _gb_s_scale(h: str) -> float:
        x = h.lower()
        if "/s" not in x and "bandwidth" not in x and "bw" not in x: return 1.0
        if "kb" in x: return 1.0/(1024*1024)
        if "mb" in x: return 1.0/1024
        if "gb" in x: return 1.0
        if "byte" in x or "b/s" in x: return 1.0/(1024**3)
        return 1.0

    def _wait_and_parse(self):
        if not self._proc: return
        try:
            # AMDuProfPcm is launched with ``-d <duration>`` and normally
            # self-terminates after flushing its CSV.  Do not kill it exactly
            # at duration_s; that can truncate the DF METRICS section and leave
            # AMOprof with an empty/all-zero DRAM result even though the same
            # command works manually.  Wait for natural completion, then use a
            # bounded terminate/kill sequence only as a final fallback.
            try:
                out, _ = self._proc.communicate(timeout=max(self.duration_s + 30.0, 35.0))
                self._raw_output = out or ""
            except subprocess.TimeoutExpired:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                try:
                    out, _ = self._proc.communicate(timeout=10)
                    self._raw_output = out or ""
                except subprocess.TimeoutExpired:
                    try:
                        self._proc.kill()
                        out, _ = self._proc.communicate(timeout=5)
                        self._raw_output = out or ""
                    except Exception:
                        pass
        except Exception as e:
            self._result = {"amduprof_pcm_available": False, "amduprof_pcm_reason": f"runtime failed: {e}"}
        try: self._parse_csv_file()
        except Exception as e:
            self._result = {"amduprof_pcm_available": False, "amduprof_pcm_reason": f"parse failed: {e}", "amduprof_pcm_raw_output": self._raw_output[-4000:]}

    def _parse_csv_file(self):
        import csv as _csv
        if not self._csv_path or not Path(self._csv_path).exists():
            self._result = {"amduprof_pcm_available": False, "amduprof_pcm_reason": "output CSV not found", "amduprof_pcm_raw_output": self._raw_output[-4000:]}; return
        text = Path(self._csv_path).read_text(errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines: return
        header_idx = 0
        # AMDuProf PCM headers vary by platform/version. Common 5.2 memory mode
        # columns include compact names such as:
        #   Total Mem Bw (GB/s), Total Mem RdBw (GB/s), Total Mem WrBw (GB/s)
        # which do not contain the literal strings \"read\" or \"write\".
        for i, ln in enumerate(lines):
            low = ln.lower()
            has_delim = ("," in ln or ";" in ln)
            has_mem = any(x in low for x in ("memory", "dram", "ddr", "mem"))
            has_bw = any(x in low for x in ("bw", "bandwidth", "/s", "gb/s", "mb/s"))
            has_rw = any(x in low for x in ("read", "write", "rdbw", "wrbw", "rd bw", "wr bw", "rdb/s", "wrb/s"))
            if has_delim and has_mem and has_bw and has_rw:
                header_idx = i; break
        body = "\n".join(lines[header_idx:])
        try: dialect = _csv.Sniffer().sniff(body[:4096], delimiters=",;")
        except Exception: dialect = _csv.excel
        reader = _csv.DictReader(body.splitlines(), dialect=dialect)
        rows=[]; t0=self._t0 or time.time()

        def norm_name(name: str) -> str:
            return " ".join(str(name).strip().lower().replace("_", " " ).replace("-", " " ).split())

        for idx,row in enumerate(reader):
            sample={
                "ts": round(t0 + idx*self.interval_s, 6),
                "timestamp_epoch": round(t0 + idx*self.interval_s, 6),
                "time_sec": round(idx*self.interval_s, 6),
            }
            read=write=total=rtxn=wtxn=0.0
            saw_total=False
            for k,v in row.items():
                if not k: continue
                key=str(k).strip(); lk=norm_name(key); fv=self._f(v); sample[f"raw::{key}"]=v
                if fv is None: continue

                # Exact/common AMDuProf PCM aliases first. Your observed output uses:
                #   Total Mem Bw (GB/s), Total Mem RdBw (GB/s), Total Mem WrBw (GB/s)
                is_total_mem_read_bw = (
                    "total mem rdbw" in lk or "total mem rd bw" in lk or
                    "total memory rdbw" in lk or "total memory read" in lk or
                    "total mem read" in lk or "mem rdbw" in lk or "mem rd bw" in lk
                )
                is_total_mem_write_bw = (
                    "total mem wrbw" in lk or "total mem wr bw" in lk or
                    "total memory wrbw" in lk or "total memory write" in lk or
                    "total mem write" in lk or "mem wrbw" in lk or "mem wr bw" in lk
                )
                is_total_mem_bw = (
                    ("total mem bw" in lk or "total memory bw" in lk or "total mem bandwidth" in lk)
                    and not is_total_mem_read_bw and not is_total_mem_write_bw
                )

                if is_total_mem_read_bw:
                    read += fv*self._gb_s_scale(key); continue
                if is_total_mem_write_bw:
                    write += fv*self._gb_s_scale(key); continue
                if is_total_mem_bw:
                    total += fv*self._gb_s_scale(key); saw_total=True; continue

                is_mem=any(x in lk for x in ("dram","ddr","memory","mem","channel","socket"))
                is_read=("read" in lk or " rdbw" in lk or " rd bw" in lk or lk.endswith("rdbw"))
                is_write=("write" in lk or " wrbw" in lk or " wr bw" in lk or lk.endswith("wrbw"))
                is_bw=any(x in lk for x in ("/s","bw","bandwidth","bytes/s","mb/s","gb/s"))
                is_txn=any(x in lk for x in ("txn","trans","transaction","cas","count"))
                if is_mem and is_read and is_bw: read += fv*self._gb_s_scale(key)
                elif is_mem and is_write and is_bw: write += fv*self._gb_s_scale(key)
                elif is_mem and is_read and is_txn: rtxn += fv
                elif is_mem and is_write and is_txn: wtxn += fv

            total_bw = total if saw_total else (read + write)
            sample.update({
                "dram_read_gb_s": round(read,6),
                "dram_write_gb_s": round(write,6),
                "dram_total_gb_s": round(total_bw,6),
                "dram_read_transactions": round(rtxn,6),
                "dram_write_transactions": round(wtxn,6),
                "dram_total_transactions": round(rtxn+wtxn,6),
                "dram_read_gb_est": round(read*self.interval_s,6),
                "dram_write_gb_est": round(write*self.interval_s,6),
                "dram_total_gb_est": round(total_bw*self.interval_s,6),
                "amduprof_pcm_parse_schema": "amduprof_5_total_mem_bw_aliases",
                "dram_transactions_supported": bool(rtxn or wtxn),
            })
            rows.append(sample)
        self.samples=rows

    def stop(self) -> dict:
        _interrupted = bool(getattr(self, "_amoprof_interrupted", False))
        _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (5.0 if _interrupted else self.duration_s + 35))
        if _interrupted and self._proc and self._proc.poll() is None:
            try: self._proc.terminate()
            except Exception: pass
        if self._thread: self._thread.join(timeout=_timeout)
        if self._result: return self._result
        if self._proc and self._proc.poll() is None:
            try: self._proc.terminate()
            except Exception: pass
            try: self._proc.wait(timeout=min(_timeout, 5.0))
            except Exception:
                try: self._proc.kill()
                except Exception: pass
        if not self.samples:
            return {"amduprof_pcm_available": False, "amduprof_pcm_reason": "no samples", "amduprof_pcm_csv_path": self._csv_path or "", "amduprof_pcm_command": self._command, "amduprof_pcm_raw_output": self._raw_output[-4000:]}
        reads=[float(s.get("dram_read_gb_s",0) or 0) for s in self.samples]
        writes=[float(s.get("dram_write_gb_s",0) or 0) for s in self.samples]
        totals=[float(s.get("dram_total_gb_s",0) or 0) for s in self.samples]
        rtxn=[float(s.get("dram_read_transactions",0) or 0) for s in self.samples]
        wtxn=[float(s.get("dram_write_transactions",0) or 0) for s in self.samples]
        def mn(xs): return round(statistics.mean(xs),6) if xs else 0.0
        def pk(xs): return round(max(xs),6) if xs else 0.0
        elapsed=max(time.time()-(self._t0 or time.time()),0.001)
        return {
            "amduprof_pcm_available": True,
            "amduprof_pcm_source": self.binary,
            "amduprof_pcm_csv_path": self._csv_path or "",
            "amduprof_pcm_command": self._command,
            "amduprof_pcm_samples": len(self.samples),
            "dram_read_gb_s_mean": mn(reads),
            "dram_write_gb_s_mean": mn(writes),
            "dram_total_gb_s_mean": mn(totals),
            "dram_read_gb_s_peak": pk(reads),
            "dram_write_gb_s_peak": pk(writes),
            "dram_total_gb_s_peak": pk(totals),
            "dram_read_transactions_total": round(sum(rtxn),6),
            "dram_write_transactions_total": round(sum(wtxn),6),
            "dram_total_transactions_total": round(sum(rtxn)+sum(wtxn),6),
            "dram_read_gb_est": round(sum(reads)*self.interval_s,6),
            "dram_write_gb_est": round(sum(writes)*self.interval_s,6),
            "dram_total_gb_est": round(sum(totals)*self.interval_s,6),
            "amduprof_pcm_duration_s": round(elapsed,3),
            "amduprof_pcm_raw_output": self._raw_output[-4000:],
        }

# ── NcuHbmReadWriteCollector ──────────────────────────────────────────────────
class NcuHbmReadWriteCollector:
    """Capture GPU HBM/DRAM read and write bytes with NVIDIA Nsight Compute.

    In launch mode this collector is also the workload launcher. Output is streamed
    to a live log file while being buffered for later NCU CSV parsing, so long-running
    servers such as SGLang show startup logs immediately.
    """
    DEFAULT_METRICS = "dram__bytes_read.sum,dram__bytes_write.sum"

    def __init__(self, command: str | None = None, duration_s: float = 10.0,
                 binary: str = "ncu", metrics: str | None = None,
                 mode: str = "launch", extra_args: list[str] | None = None,
                 output_csv: str | None = None, live_log_path: str | None = None,
                 stream_output: bool = True):
        self.command = command
        self.duration_s = max(float(duration_s), 1.0)
        self.binary = binary
        self.metrics = metrics or self.DEFAULT_METRICS
        self.mode = mode
        self.extra_args = extra_args or []
        self.output_csv = output_csv
        self.live_log_path = live_log_path
        self.stream_output = stream_output
        self.samples = []
        self._proc = None
        self._thread = None
        self._result = {}
        self._raw_output = ""
        self._t0 = 0.0
        self.command_line = ""

    def start(self):
        self.samples.clear()
        self._result = {}
        self._raw_output = ""
        self._t0 = time.time()
        try:
            subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=5)
        except Exception:
            self._result = {"ncu_hbm_available": False, "ncu_hbm_reason": f"ncu not found: {self.binary}"}
            return

        cmd = [self.binary, "--metrics", self.metrics, "--csv", "--page", "raw", "--target-processes", "all"] + self.extra_args
        if self.mode == "attach":
            cmd += ["--mode", "attach"]
        elif self.mode == "launch":
            if not self.command:
                self._result = {"ncu_hbm_available": False, "ncu_hbm_reason": "--ncu-command is required for launch mode"}
                return
            import shlex
            cmd += shlex.split(self.command)
        else:
            self._result = {"ncu_hbm_available": False, "ncu_hbm_reason": f"unsupported mode: {self.mode}"}
            return

        self.command_line = " ".join(cmd)
        try:
            if self.live_log_path:
                Path(self.live_log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.live_log_path).write_text("# AMOprof NCU launched command\n" + self.command_line + "\n\n", encoding="utf-8")
            _cwd = None
            try:
                if self.output_csv:
                    _cwd = str(Path(self.output_csv).parent)
                elif self.live_log_path:
                    _cwd = str(Path(self.live_log_path).parent)
            except Exception:
                _cwd = None
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=_cwd)
            self._thread = threading.Thread(target=self._wait_and_parse, daemon=True)
            self._thread.start()
        except Exception as e:
            self._result = {"ncu_hbm_available": False, "ncu_hbm_reason": str(e), "ncu_hbm_command": self.command_line}

    @staticmethod
    def _f(v):
        try:
            return float(str(v).strip().replace(",", ""))
        except Exception:
            return None

    @staticmethod
    def _unit_to_bytes(v, unit):
        u = (unit or "").strip().lower()
        if u in {"kbyte", "kbytes", "kb"}:
            return v * 1024
        if u in {"mbyte", "mbytes", "mb"}:
            return v * 1024**2
        if u in {"gbyte", "gbytes", "gb"}:
            return v * 1024**3
        return v

    def _wait_and_parse(self):
        if not self._proc:
            return
        chunks = []
        live_fh = None
        try:
            if self.live_log_path:
                live_fh = open(self.live_log_path, "a", encoding="utf-8", buffering=1)
            deadline = time.time() + self.duration_s + 60
            assert self._proc.stdout is not None
            while True:
                if time.time() > deadline and self._proc.poll() is None:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                line = self._proc.stdout.readline()
                if line:
                    chunks.append(line)
                    if live_fh:
                        live_fh.write(line)
                    if self.stream_output:
                        try:
                            print(line, end="")
                        except Exception:
                            pass
                elif self._proc.poll() is not None:
                    break
                else:
                    time.sleep(0.05)
            try:
                rest = self._proc.stdout.read() if self._proc.stdout else ""
                if rest:
                    chunks.append(rest)
                    if live_fh:
                        live_fh.write(rest)
                    if self.stream_output:
                        try:
                            print(rest, end="")
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception as e:
            chunks.append(f"\n[AMOprof NCU reader error] {e}\n")
        finally:
            try:
                if live_fh:
                    live_fh.close()
            except Exception:
                pass
        self._raw_output = "".join(chunks)
        if self.output_csv:
            try:
                Path(self.output_csv).write_text(self._raw_output, encoding="utf-8")
            except Exception:
                pass
        try:
            self._parse(self._raw_output)
        except Exception as e:
            self._result = {
                "ncu_hbm_available": False,
                "ncu_hbm_reason": f"parse failed: {e}",
                "ncu_hbm_command": self.command_line,
                "ncu_hbm_live_log_path": self.live_log_path or "",
                "ncu_hbm_raw_output": self._raw_output[-4000:],
            }

    def _parse(self, raw):
        import csv as _csv
        lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith("==")]
        text = "\n".join(lines)
        grouped = {}
        t0 = self._t0 or time.time()
        try:
            parsed = list(_csv.DictReader(text.splitlines())) if text else []
        except Exception:
            parsed = []
        for r in parsed:
            metric = (r.get("Metric Name") or r.get("Metric") or r.get("Name") or "").strip()
            if metric not in {"dram__bytes_read.sum", "dram__bytes_write.sum"}:
                continue
            val = self._f(r.get("Metric Value") or r.get("Value") or r.get(metric))
            if val is None:
                continue
            unit = r.get("Metric Unit") or r.get("Unit") or "byte"
            key = r.get("Kernel Name") or r.get("Kernel") or r.get("ID") or f"kernel_{len(grouped)}"
            d = grouped.setdefault(str(key), {"ts": round(t0 + len(grouped), 6), "kernel_name": str(key)})
            if metric.endswith("read.sum"):
                d["hbm_read_bytes"] = d.get("hbm_read_bytes", 0.0) + self._unit_to_bytes(val, unit)
            else:
                d["hbm_write_bytes"] = d.get("hbm_write_bytes", 0.0) + self._unit_to_bytes(val, unit)
        rows = list(grouped.values())
        if not rows:
            read_vals = []
            write_vals = []
            for line in raw.splitlines():
                if "dram__bytes_read.sum" in line or "dram__bytes_write.sum" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line.replace(",", ""))
                    if not nums:
                        continue
                    if "dram__bytes_read.sum" in line:
                        read_vals.append(float(nums[-1]))
                    else:
                        write_vals.append(float(nums[-1]))
            for i in range(max(len(read_vals), len(write_vals))):
                rows.append({
                    "ts": round(t0 + i, 6),
                    "kernel_name": f"kernel_{i}",
                    "hbm_read_bytes": read_vals[i] if i < len(read_vals) else 0.0,
                    "hbm_write_bytes": write_vals[i] if i < len(write_vals) else 0.0,
                })
        for d in rows:
            rb = float(d.get("hbm_read_bytes", 0) or 0)
            wb = float(d.get("hbm_write_bytes", 0) or 0)
            d["hbm_total_bytes"] = rb + wb
            d["hbm_read_gb"] = round(rb / (1024**3), 6)
            d["hbm_write_gb"] = round(wb / (1024**3), 6)
            d["hbm_total_gb"] = round((rb + wb) / (1024**3), 6)
        self.samples = rows

    def stop(self):
        _interrupted = bool(getattr(self, "_amoprof_interrupted", False))
        _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (5.0 if _interrupted else self.duration_s + 70))
        if _interrupted and self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=_timeout)
        if self._result:
            return self._result
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=min(_timeout, 5.0))
            except Exception:
                try: self._proc.kill()
                except Exception: pass
        elapsed = max(time.time() - (self._t0 or time.time()), 0.001)
        if not self.samples:
            return {
                "ncu_hbm_available": False,
                "ncu_hbm_reason": "no samples parsed",
                "ncu_hbm_command": self.command_line,
                "ncu_hbm_live_log_path": self.live_log_path or "",
                "ncu_hbm_raw_output": self._raw_output[-4000:],
            }
        rb = sum(float(s.get("hbm_read_bytes", 0) or 0) for s in self.samples)
        wb = sum(float(s.get("hbm_write_bytes", 0) or 0) for s in self.samples)
        return {
            "ncu_hbm_available": True,
            "ncu_hbm_mode": self.mode,
            "ncu_hbm_metrics": self.metrics,
            "ncu_hbm_command": self.command_line,
            "ncu_hbm_live_log_path": self.live_log_path or "",
            "ncu_hbm_kernel_samples": len(self.samples),
            "hbm_read_bytes_total": int(rb),
            "hbm_write_bytes_total": int(wb),
            "hbm_total_bytes_total": int(rb + wb),
            "hbm_read_gb_total": round(rb / (1024**3), 6),
            "hbm_write_gb_total": round(wb / (1024**3), 6),
            "hbm_total_gb_total": round((rb + wb) / (1024**3), 6),
            "hbm_read_gb_s_est": round(rb / (1024**3) / elapsed, 6),
            "hbm_write_gb_s_est": round(wb / (1024**3) / elapsed, 6),
            "ncu_hbm_duration_s": round(elapsed, 3),
            "ncu_hbm_raw_output": self._raw_output[-4000:],
        }
# ── BpftraceCollector ─────────────────────────────────────────────────────────
#
# Runs bpftrace programs to capture page faults, mmap/malloc activity, and
# I/O latency histograms for the SGLang server PID.
# ════════════════════════════════════════════════════════════════════════════════

class BpftraceCollector:
    """
    Attaches bpftrace programs to the SGLang server PID and collects:

    bpf_available               bool
    bpf_page_faults_total       int    — minor + major page faults
    bpf_major_faults_total      int    — major faults (page not in RAM)
    bpf_mmap_calls              int    — mmap() call count
    bpf_mmap_bytes_gb           float  — total bytes requested via mmap (GB)
    bpf_malloc_calls            int    — malloc() call count (uprobe on libc)
    bpf_malloc_bytes_gb         float  — total bytes allocated (GB)
    bpf_read_lat_p99_us         float  — read() call latency p99 (µs)
    bpf_write_lat_p99_us        float  — write() call latency p99 (µs)
    bpf_duration_s              float  — actual monitoring window
    """

    # bpftrace program: page faults + mmap + read/write latency for target PID
    _BPF_PROG = """
BEGIN {{ printf("bpf_start\\n"); }}

/* page faults */
software:page-faults:{pid}
{{
    @pf_minor = count();
}}

software:major-faults:{pid}
{{
    @pf_major = count();
}}

/* mmap syscall */
tracepoint:syscalls:sys_enter_mmap
/ pid == {pid} /
{{
    @mmap_calls = count();
    @mmap_bytes = @mmap_bytes + args->len;
}}

/* malloc via uprobe (best-effort) */
uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc
/ pid == {pid} /
{{
    @malloc_calls = count();
    @malloc_bytes = @malloc_bytes + arg0;
}}

/* read/write latency */
tracepoint:syscalls:sys_enter_read  / pid == {pid} / {{ @rd_start[tid] = nsecs; }}
tracepoint:syscalls:sys_exit_read
/ pid == {pid} && @rd_start[tid] /
{{
    @rd_lat_us = hist((nsecs - @rd_start[tid]) / 1000);
    delete(@rd_start[tid]);
}}

tracepoint:syscalls:sys_enter_write / pid == {pid} / {{ @wr_start[tid] = nsecs; }}
tracepoint:syscalls:sys_exit_write
/ pid == {pid} && @wr_start[tid] /
{{
    @wr_lat_us = hist((nsecs - @wr_start[tid]) / 1000);
    delete(@wr_start[tid]);
}}

END
{{
    printf("bpf_pf_minor %d\\n",    @pf_minor);
    printf("bpf_pf_major %d\\n",    @pf_major);
    printf("bpf_mmap_calls %d\\n",  @mmap_calls);
    printf("bpf_mmap_bytes %d\\n",  @mmap_bytes);
    printf("bpf_malloc_calls %d\\n",@malloc_calls);
    printf("bpf_malloc_bytes %d\\n",@malloc_bytes);
    print(@rd_lat_us);
    print(@wr_lat_us);
}}
"""

    def __init__(self, server_pid: int, duration_s: int = 30,
                 work_dir: "Path | None" = None, use_sudo: bool = True):
        self.server_pid = server_pid
        self.use_sudo  = use_sudo
        self.duration   = duration_s
        self.work_dir   = work_dir
        self._proc      = None
        self._t0: float = 0.0
        self._result: dict = {}

    def _bpftrace_available(self) -> bool:
        try:
            subprocess.run(["bpftrace", "--version"],
                           capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def start(self):
        if not self._bpftrace_available():
            self._result = {"bpf_available": False,
                            "bpf_reason": "bpftrace not found"}
            return
        prog = self._BPF_PROG.replace("{pid}", str(self.server_pid))
        import tempfile
        td = self.work_dir or Path(tempfile.mkdtemp(prefix="amoprof_bpf_"))
        prog_file = td / "monitor.bt"
        prog_file.write_text(prog)

        cmd = ["bpftrace", "-e", prog,
               "--unsafe",           # allow uprobe on libc malloc
               "-c", f"sleep {self.duration}"]
        # Alternatively run with timeout via the prog's interval timer
        cmd = ["timeout", str(self.duration + 5),
               "bpftrace", str(prog_file)]
        try:
            self._proc = _sudo_popen(
                cmd, self.use_sudo, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
                cwd=str(td))
            self._t0 = time.time()
        except Exception as e:
            self._result = {"bpf_available": False, "bpf_reason": str(e)}

    def stop(self) -> dict:
        if self._result:
            return self._result
        if not self._proc:
            return {"bpf_available": False, "bpf_reason": "not started"}
        elapsed = round(time.time() - self._t0, 2)
        try:
            # Send SIGINT so bpftrace prints END block
            import signal
            self._proc.send_signal(signal.SIGINT)
            stdout, stderr = self._proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            stdout, stderr = self._proc.communicate()
        except Exception as e:
            return {"bpf_available": False, "bpf_reason": str(e)}

        return self._parse(stdout or "", stderr or "", elapsed)

    def _parse(self, stdout: str, stderr: str, elapsed_s: float) -> dict:
        if "permission denied" in stderr.lower() or "operation not permitted" in stderr.lower():
            return {"bpf_available": False,
                    "bpf_reason": "permission denied (need CAP_BPF or root)"}

        combined = stdout + "\n" + stderr

        def _int(key: str) -> int:
            m = re.search(rf"{key}\s+(\d+)", combined)
            return int(m.group(1)) if m else 0

        def _p99_from_hist(section_marker: str) -> float:
            """
            Extract p99 from bpftrace hist() output.  Format:
              [32K, 64K)    42 |@@@|
            We accumulate counts and find the bucket at 99th percentile.
            """
            in_section = False
            buckets: list[tuple[float, int]] = []
            for line in combined.splitlines():
                if section_marker in line:
                    in_section = True
                    continue
                if in_section:
                    if not line.strip():
                        break
                    # match "[lo, hi)  count"
                    m = re.match(r"\s*\[([^,]+),\s*([^\)]+)\)\s+(\d+)", line)
                    if m:
                        try:
                            lo = float(m.group(1).replace("K","e3").replace("M","e6"))
                            cnt = int(m.group(3))
                            buckets.append((lo, cnt))
                        except Exception:
                            pass
                    elif buckets:
                        break

            if not buckets:
                return 0.0
            total = sum(c for _, c in buckets)
            if total == 0:
                return 0.0
            target = total * 0.99
            cumul  = 0
            for lo, cnt in buckets:
                cumul += cnt
                if cumul >= target:
                    return round(lo, 2)
            return round(buckets[-1][0], 2)

        pf_minor     = _int("bpf_pf_minor")
        pf_major     = _int("bpf_pf_major")
        mmap_calls   = _int("bpf_mmap_calls")
        mmap_bytes   = _int("bpf_mmap_bytes")
        malloc_calls = _int("bpf_malloc_calls")
        malloc_bytes = _int("bpf_malloc_bytes")
        rd_p99       = _p99_from_hist("@rd_lat_us")
        wr_p99       = _p99_from_hist("@wr_lat_us")

        return {
            "bpf_available":         True,
            "bpf_page_faults_total": pf_minor + pf_major,
            "bpf_major_faults_total":pf_major,
            "bpf_mmap_calls":        mmap_calls,
            "bpf_mmap_bytes_gb":     round(mmap_bytes / 1e9, 4),
            "bpf_malloc_calls":      malloc_calls,
            "bpf_malloc_bytes_gb":   round(malloc_bytes / 1e9, 4),
            "bpf_read_lat_p99_us":   rd_p99,
            "bpf_write_lat_p99_us":  wr_p99,
            "bpf_duration_s":        elapsed_s,
            "bpf_raw_output":        combined[-4000:],
        }


# ════════════════════════════════════════════════════════════════════════════════
# ── TorchProfilerCollector ────────────────────────────────────────────────────
#
# Triggers PyTorch Profiler via the SGLang server's admin/debug endpoint
# (or falls back to scraping torch_profiler JSON from a known output dir).
# Extracts operator-level CUDA time, memory allocated, and kernel mapping.
# ════════════════════════════════════════════════════════════════════════════════

class TorchProfilerCollector:
    """
    Triggers a short PyTorch Profiler trace on the running SGLang server and
    parses the resulting Chrome trace JSON (torch.profiler exports this format).

    Two activation paths:
      1. POST /debug/torch_profiler_start + /debug/torch_profiler_stop
         (available in SGLang ≥ 0.4 when launched with --enable-torch-profiler)
      2. Drop-file activation: write a sentinel file that the server polls.
         (fallback for servers without the debug endpoint)

    Metrics
    -------
    torch_prof_available         bool
    torch_prof_top_ops           str    — JSON list of top-10 ops by CUDA time
    torch_prof_cuda_time_ms      float  — total CUDA time in profile window (ms)
    torch_prof_cpu_time_ms       float  — total CPU time in profile window (ms)
    torch_prof_memory_alloc_mb   float  — peak memory allocated during window (MB)
    torch_prof_kernel_count      int    — unique CUDA kernel types
    torch_prof_trace_path        str    — path to Chrome trace JSON
    """

    def __init__(self, server_port: int, output_dir: "Path | None" = None,
                 duration_s: int = 30, server_host: str = "127.0.0.1"):
        self.server_port = server_port
        self.server_host = server_host
        self.output_dir  = output_dir
        self.duration    = duration_s
        self._trace_path: "Path | None" = None
        self._result: dict = {}
        self._started    = False

    def _post(self, path: str, body: dict = {}) -> bool:
        import urllib.request as _req
        try:
            data = json.dumps(body).encode()
            req  = _req.Request(
                f"http://{self.server_host}:{self.server_port}{path}",
                data=data,
                headers={"Content-Type": "application/json"})
            with _req.urlopen(req, timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self):
        # Try SGLang debug endpoint first
        if self._post("/debug/torch_profiler_start",
                      {"duration_ms": self.duration * 1000}):
            self._started = True
            log.debug("TorchProfilerCollector: started via /debug endpoint")
        else:
            # Fallback: write sentinel file to SGLang profiler watch dir
            import tempfile
            td = self.output_dir or Path(tempfile.mkdtemp(prefix="amoprof_torch_"))
            sentinel = td / "start_profiler"
            sentinel.touch()
            self._started = True
            log.debug("TorchProfilerCollector: started via sentinel file")

    def stop(self) -> dict:
        if not self._started:
            return {"torch_prof_available": False,
                    "torch_prof_reason": "not started"}

        # Tell server to stop and flush
        trace_info = {}
        if self._post("/debug/torch_profiler_stop"):
            try:
                import urllib.request as _req
                with _req.urlopen(
                        f"http://{self.server_host}:{self.server_port}"
                        f"/debug/torch_profiler_result",
                        timeout=15) as r:
                    trace_info = json.loads(r.read())
            except Exception:
                pass

        # Find the trace file
        trace_path = None
        if trace_info.get("trace_path"):
            trace_path = Path(trace_info["trace_path"])
        else:
            # Search common SGLang profiler output locations
            search_dirs = [
                Path(".") / "torch_profiler",
                Path("/tmp") / "torch_profiler",
            ]
            if self.output_dir:
                search_dirs.insert(0, self.output_dir)
            for d in search_dirs:
                if not d.exists():
                    continue
                candidates = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    trace_path = candidates[-1]
                    break

        if not trace_path or not trace_path.exists():
            return {"torch_prof_available": False,
                    "torch_prof_reason": "no trace file found"}

        self._trace_path = trace_path
        return self._parse_trace(trace_path)

    def _parse_trace(self, path: Path) -> dict:
        try:
            with open(path) as f:
                trace = json.load(f)

            events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

            # Collect CUDA kernel events (cat="kernel" or "gpu_memcpy")
            cuda_events  = [e for e in events
                            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
            cpu_events   = [e for e in events
                            if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function")]

            total_cuda_us = sum(e.get("dur", 0) for e in cuda_events)
            total_cpu_us  = sum(e.get("dur", 0) for e in cpu_events)

            # Group kernels by name and sum duration
            from collections import defaultdict
            kernel_totals: dict = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
            for e in cuda_events:
                name = e.get("name", "unknown")
                kernel_totals[name]["count"]  += 1
                kernel_totals[name]["dur_us"] += e.get("dur", 0)

            top10 = sorted(kernel_totals.items(),
                           key=lambda x: x[1]["dur_us"], reverse=True)[:10]
            top10_json = json.dumps([
                {"name": k, "count": v["count"],
                 "dur_ms": round(v["dur_us"]/1000, 2)}
                for k, v in top10])

            # Memory allocations from aten::malloc events
            mem_events = [e for e in events
                          if "malloc" in e.get("name","").lower()
                          and e.get("args",{}).get("bytes")]
            peak_alloc_mb = sum(e["args"]["bytes"] for e in mem_events) / 1e6

            return {
                "torch_prof_available":       True,
                "torch_prof_top_ops":         top10_json,
                "torch_prof_cuda_time_ms":    round(total_cuda_us / 1000, 2),
                "torch_prof_cpu_time_ms":     round(total_cpu_us  / 1000, 2),
                "torch_prof_memory_alloc_mb": round(peak_alloc_mb, 2),
                "torch_prof_kernel_count":    len(kernel_totals),
                "torch_prof_trace_path":      str(path),
            }
        except Exception as e:
            return {"torch_prof_available": False,
                    "torch_prof_reason": f"parse error: {e}",
                    "torch_prof_trace_path": str(path)}


# ── VtuneCollector ────────────────────────────────────────────────────────────

class VtuneCollector:
    """
    Runs Intel VTune ``memory-access`` analysis on the SGLang server PID.

    Captures:
    vtune_available             bool
    vtune_dram_bw_gb_s          float  — DRAM bandwidth (GB/s) averaged
    vtune_numa_local_access_pct float  — % memory accesses to local NUMA node
    vtune_l3_bound_pct          float  — % of stall cycles due to L3 misses
    vtune_mem_bound_pct         float  — % of stall cycles due to DRAM latency
    vtune_ipc                   float  — instructions per cycle
    vtune_hotspot_fn            str    — top memory-bound function name
    vtune_report_dir            str    — path to vtune result directory
    """

    def __init__(self, server_pid: int, duration_s: int = 30,
                 work_dir: "Path | None" = None, use_sudo: bool = True):
        self.server_pid = server_pid
        self.use_sudo  = use_sudo
        self.duration   = duration_s
        self.work_dir   = work_dir
        self._proc      = None
        self._result_dir: "Path | None" = None
        self._result: dict = {}

    def _vtune_available(self) -> bool:
        for vtune in ["vtune", "/opt/intel/oneapi/vtune/latest/bin64/vtune"]:
            try:
                subprocess.run([vtune, "--version"],
                               capture_output=True, timeout=5)
                return True
            except Exception:
                pass
        return False

    def _vtune_bin(self) -> str:
        for vtune in ["vtune", "/opt/intel/oneapi/vtune/latest/bin64/vtune"]:
            try:
                subprocess.run([vtune, "--version"],
                               capture_output=True, timeout=5)
                return vtune
            except Exception:
                pass
        return "vtune"

    def start(self):
        if not self._vtune_available():
            self._result = {"vtune_available": False,
                            "vtune_reason": "vtune not found"}
            return
        import tempfile
        td = self.work_dir or Path(tempfile.mkdtemp(prefix="amoprof_vtune_"))
        self._result_dir = td / "vtune_result"
        cmd = [
            self._vtune_bin(),
            "-collect",     "memory-access",
            "-target-pid",  str(self.server_pid),
            "-duration",    str(self.duration),
            "-result-dir",  str(self._result_dir),
            "-quiet",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, cwd=str(td))
        except Exception as e:
            self._result = {"vtune_available": False,
                            "vtune_reason": str(e)}

    def stop(self) -> dict:
        if self._result:
            return self._result
        if not self._proc:
            return {"vtune_available": False, "vtune_reason": "not started"}
        try:
            _interrupted = bool(getattr(self, "_amoprof_interrupted", False))
            _timeout = float(getattr(self, "_amoprof_stop_timeout_s", 0.0) or (8.0 if _interrupted else self.duration + 60))
            if _interrupted and self._proc.poll() is None:
                try: self._proc.terminate()
                except Exception: pass
            _, stderr = self._proc.communicate(timeout=_timeout)
            if self._proc.returncode != 0:
                return {"vtune_available": False,
                        "vtune_reason": (stderr or "")[:200]}
        except subprocess.TimeoutExpired:
            self._proc.kill()
            return {"vtune_available": False, "vtune_reason": "vtune timed out during interrupt fast-stop" if bool(getattr(self, "_amoprof_interrupted", False)) else "vtune timed out"}

        return self._parse_report()

    def _parse_report(self) -> dict:
        if not self._result_dir or not self._result_dir.exists():
            return {"vtune_available": False,
                    "vtune_reason": "no result directory"}
        try:
            # Export summary CSV using vtune -report
            csv_path = self._result_dir / "summary.csv"
            r = subprocess.run(
                [self._vtune_bin(), "-report", "summary",
                 "-result-dir", str(self._result_dir),
                 "-format", "csv",
                 "-report-output", str(csv_path),
                 "-quiet"],
                capture_output=True, text=True, timeout=60)

            # Also export hotspots
            hot_path = self._result_dir / "hotspots.csv"
            subprocess.run(
                [self._vtune_bin(), "-report", "hotspots",
                 "-result-dir", str(self._result_dir),
                 "-format", "csv",
                 "-report-output", str(hot_path),
                 "-quiet"],
                capture_output=True, text=True, timeout=60)

            metrics = {
                "vtune_available":             True,
                "vtune_report_dir":            str(self._result_dir),
                "vtune_dram_bw_gb_s":          0.0,
                "vtune_numa_local_access_pct": 0.0,
                "vtune_l3_bound_pct":          0.0,
                "vtune_mem_bound_pct":         0.0,
                "vtune_ipc":                   0.0,
                "vtune_hotspot_fn":            "",
            }

            # Parse summary CSV
            if csv_path.exists():
                import csv as _csv
                with open(csv_path) as f:
                    for row in _csv.DictReader(f):
                        name = row.get("Metric Name", "").lower()
                        val  = row.get("Metric Value","0").replace(",","").strip()
                        try:
                            fval = float(val)
                        except Exception:
                            continue
                        if "dram bandwidth" in name:
                            metrics["vtune_dram_bw_gb_s"] = round(fval, 2)
                        elif "ipc" in name:
                            metrics["vtune_ipc"] = round(fval, 3)
                        elif "l3 bound" in name or "l3bound" in name:
                            metrics["vtune_l3_bound_pct"] = round(fval, 2)
                        elif "memory bound" in name or "dram bound" in name:
                            metrics["vtune_mem_bound_pct"] = round(fval, 2)
                        elif "numa" in name and "local" in name:
                            metrics["vtune_numa_local_access_pct"] = round(fval, 2)

            # Parse hotspots CSV — first function with highest "Memory Bound"
            if hot_path.exists():
                import csv as _csv
                with open(hot_path) as f:
                    rows = list(_csv.DictReader(f))
                if rows:
                    metrics["vtune_hotspot_fn"] = rows[0].get("Function","")

            return metrics

        except Exception as e:
            return {"vtune_available": False,
                    "vtune_reason": f"report parse: {e}",
                    "vtune_report_dir": str(self._result_dir or "")}
