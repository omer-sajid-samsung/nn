"""
bench_sglang.py — SGLang server benchmark runner.

Aligns AMOprof with industry-standard SGLang serving benchmarks.

What this covers
────────────────
1. Server launch  — sglang.launch_server with capacity flags
2. Health polling — waits until /health returns 200 (up to 5 min)
3. Model name resolution — fetches actual registered model name from
   /v1/models so bench_serving --model matches exactly
4. Throughput sweep — sglang.bench_serving at multiple request rates
5. Metric extraction — parses bench_serving stdout for tok/s, TTFT,
   ITL (TPOT), E2E latency, KV$ cache hit rate
6. SSD metric overlay — IostatMonitor + NvmeSmartMonitor run during
   the entire sweep
7. Fallback — if bench_serving fails, runs a direct HTTP sweep via
   the OpenAI-compatible endpoint and computes metrics manually

CLI usage
─────────
  amoprof --model qwen3-32b \\
         --model-hf-id QuantTrio/MiniMax-M2-AWQ \\
         --hbm-cap 320GB --dram-cap 512GB --nvme-cap 10TB \\
         --benchmark sglang \\
         --sglang-port 30000 \\
         --request-rates inf,8,2 \\
         --context-lens 4096,16384 \\
         --num-prompts 50
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("amoprof.sglang")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SGLangResult:
    run_id: str              = ""
    timestamp: str           = ""
    model: str               = ""
    server_model_id: str     = ""    # actual model name registered on server
    context_len: int         = 0
    output_len: int          = 64
    request_rate: float      = 0.0
    num_prompts: int         = 200
    dtype: str               = ""
    tensor_parallel: int     = 1
    sglang_version: str      = ""

    output_tok_throughput: float  = 0.0
    request_throughput:    float  = 0.0
    total_tokens:          int    = 0

    ttft_mean_ms:   float = 0.0
    ttft_p50_ms:    float = 0.0
    ttft_p99_ms:    float = 0.0
    ttft_p999_ms:   float = 0.0

    itl_mean_ms:    float = 0.0
    itl_p50_ms:     float = 0.0
    itl_p99_ms:     float = 0.0
    itl_p999_ms:    float = 0.0

    e2e_mean_ms:    float = 0.0
    e2e_p50_ms:     float = 0.0
    e2e_p99_ms:     float = 0.0

    kv_cache_hit_rate_pct: float = 0.0

    read_bw_mb_mean:   float = 0.0
    write_bw_mb_mean:  float = 0.0
    read_iops_mean:    float = 0.0
    write_iops_mean:   float = 0.0
    r_await_ms_p99:    float = 0.0
    util_pct_mean:     float = 0.0

    waf:               float = 0.0
    temp_peak_c:       int   = 0

    gpu_power_all_w_mean:       float = 0.0
    total_system_energy_wh:     float = 0.0
    power_efficiency_tok_per_wh: float = 0.0

    # ── HBM (GPU memory) usage ────────────────────────────────────────────
    gpu_count:              int   = 0
    hbm_util_pct_mean:      float = 0.0   # % of HBM in use (mean over run)
    hbm_used_gb_peak_all:   float = 0.0   # peak GB used across all GPUs
    hbm_total_gb_all:       float = 0.0   # total HBM across all GPUs
    gpu_temp_peak_c:        float = 0.0

    # ── DRAM (host memory) usage ──────────────────────────────────────────
    dram_used_gb_mean:      float = 0.0   # host DRAM mean used (GB)
    dram_used_gb_peak:      float = 0.0   # host DRAM peak used (GB)
    dram_delta_gb:          float = 0.0   # DRAM change start→end (GB)
    dram_util_pct_mean:     float = 0.0   # host DRAM utilisation %
    dram_total_gb:          float = 0.0   # total host DRAM (GB)

    # ── NVMe spill diagnosis ──────────────────────────────────────────────
    nvme_spill_expected:    bool  = False  # True when KV$ should overflow to SSD
    nvme_spill_reason:      str   = ""     # explanation of why NVMe is/isn't active

    success: bool = False
    method:  str  = ""    # "bench_serving" or "direct_http"
    notes:   str  = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Server lifecycle ──────────────────────────────────────────────────────────

class SGLangServer:

    def __init__(self, model_hf_id: str, dtype: str, gpu_memory_util: float,
                 tensor_parallel: int, port: int = 30000,
                 extra_args: Optional[list[str]] = None,
                 hicache_storage_path: str = ""):
        self.model                = model_hf_id
        self.dtype                = dtype
        self.gpu_mem_util         = gpu_memory_util
        self.tp                   = tensor_parallel
        self.port                 = port
        self.extra_args           = extra_args or []
        self.hicache_storage_path = hicache_storage_path
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None

    def start(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "sglang_server.log"
        # Build HiCache storage path arg.
        # --file-storage-path expects JSON: {"path": "/mnt/..."}
        # A bare directory string is silently ignored → falls back to /tmp/hicache
        hicache_args = []
        hp = self.hicache_storage_path.strip()
        if hp:
            import json as _json, os as _os
            if not _os.path.isdir(hp):
                log.warning(
                    f"HiCache path '{hp}' is not a mounted directory. "
                    f"SGLang will fall back to /tmp/hicache. "
                    f"Ensure the NVMe partition is mounted before launching.")
            else:
                hicache_args = [
                    "--enable-hierarchical-cache",
                    "--hicache-storage-backend", "file",
                    "--file-storage-path", _json.dumps({"path": hp}),
                    "--enable-metrics",
                ]
                log.info(f"HiCache enabled → {hp}  (JSON: {hicache_args[-1]})")

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path",          self.model,
            "--dtype",               self.dtype,
            "--mem-fraction-static", f"{self.gpu_mem_util:.2f}",
            "--tp-size",             str(self.tp),
            "--port",                str(self.port),
            "--host",                "0.0.0.0",
        ] + hicache_args + self.extra_args
        log.info(f"Launching SGLang server: port={self.port} tp={self.tp}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=open(self._log_path, "w"),
            stderr=subprocess.STDOUT,
        )

    def wait_healthy(self, timeout_s: int = 300) -> bool:
        url      = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                log.error("SGLang server exited — check: " + str(self._log_path))
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        log.info(f"SGLang server healthy (port {self.port})")
                        return True
            except Exception:
                pass
            time.sleep(3)
        log.error(f"SGLang server not healthy after {timeout_s}s")
        return False

    def get_model_id(self) -> str:
        """
        Fetch the exact model name the server registered under.
        This is what bench_serving --model must receive.
        """
        try:
            url = f"http://127.0.0.1:{self.port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                models = data.get("data", [])
                if models:
                    model_id = models[0].get("id", "")
                    log.info(f"Server model ID: {model_id}")
                    return model_id
        except Exception as e:
            log.warning(f"Could not fetch model ID from /v1/models: {e}")
        return self.model

    def get_sglang_version(self) -> str:
        try:
            url = f"http://127.0.0.1:{self.port}/get_server_info"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                return data.get("version", "")
        except Exception:
            pass
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import sglang; print(sglang.__version__)"],
                capture_output=True, text=True, timeout=5)
            return result.stdout.strip()
        except Exception:
            return ""

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            log.info("SGLang server stopped")


# ── bench_serving output parser ───────────────────────────────────────────────

def _parse_bench_output(text: str) -> dict:
    """
    Parse sglang.bench_serving stdout.
    Handles both old and new SGLang output formats.
    """
    def grab(pattern: str, default: float = 0.0) -> float:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else default

    return {
        # Throughput
        "output_tok_throughput": grab(r"[Oo]utput token throughput.*?([\d.]+)"),
        "request_throughput":    grab(r"[Rr]equest throughput.*?([\d.]+)"),
        # TTFT
        "ttft_mean_ms":   grab(r"[Mm]ean\s+TTFT.*?([\d.]+)"),
        "ttft_p50_ms":    grab(r"[Mm]edian\s+TTFT.*?([\d.]+)"),
        "ttft_p99_ms":    grab(r"[Pp]99\s+TTFT.*?([\d.]+)"),
        "ttft_p999_ms":   grab(r"[Pp]99\.9\s+TTFT.*?([\d.]+)"),
        # ITL (TPOT)
        "itl_mean_ms":    grab(r"[Mm]ean\s+ITL.*?([\d.]+)"),
        "itl_p50_ms":     grab(r"[Mm]edian\s+ITL.*?([\d.]+)"),
        "itl_p99_ms":     grab(r"[Pp]99\s+ITL.*?([\d.]+)"),
        "itl_p999_ms":    grab(r"[Pp]99\.9\s+ITL.*?([\d.]+)"),
        # E2E
        "e2e_mean_ms":    grab(r"[Mm]ean\s+[Ee]2[Ee].*?([\d.]+)"),
        "e2e_p50_ms":     grab(r"[Mm]edian\s+[Ee]2[Ee].*?([\d.]+)"),
        "e2e_p99_ms":     grab(r"[Pp]99\s+[Ee]2[Ee].*?([\d.]+)"),
        # Cache
        "kv_cache_hit_rate_pct": grab(r"[Cc]ache hit rate.*?([\d.]+)"),
    }


# ── Direct HTTP sweep (fallback when bench_serving fails) ─────────────────────

def _direct_http_sweep(model_id: str, port: int, ctx: int, output_len: int,
                       num_prompts: int, rate: float) -> dict:
    """
    Fallback: send requests directly to /v1/chat/completions and compute
    TTFT + TPOT manually from timing.

    Uses streaming so TTFT = time to first chunk, TPOT = mean inter-chunk gap.
    """
    import queue as qmod

    base_url = f"http://127.0.0.1:{port}"
    prompt   = "Summarize the following: " + ("word " * max(1, ctx // 2))
    ttfts, tpots, e2es = [], [], []
    tokens_total = 0

    # Rate limiting
    interval = 1.0 / rate if rate != float("inf") else 0.0

    def send_one() -> tuple[float, float, float, int]:
        """Returns (ttft_s, mean_itl_s, e2e_s, tokens)."""
        payload = json.dumps({
            "model":      model_id,
            "messages":   [{"role": "user", "content": prompt[:4000]}],
            "max_tokens": output_len,
            "temperature": 0.0,
            "stream":     True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"})

        t_start    = time.perf_counter()
        t_first    = None
        chunk_times = []
        n_tokens   = 0

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            else:
                                chunk_times.append(now)
                            n_tokens += 1
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"Request error: {e}")
            return 0.0, 0.0, 0.0, 0

        t_end = time.perf_counter()
        ttft  = (t_first - t_start) if t_first else (t_end - t_start)
        itls  = [chunk_times[i] - chunk_times[i-1]
                 for i in range(1, len(chunk_times))]
        mean_itl = statistics.mean(itls) if itls else 0.0
        e2e      = t_end - t_start
        return ttft, mean_itl, e2e, n_tokens

    t_run_start = time.time()
    for i in range(num_prompts):
        t_req = time.time()
        ttft, itl, e2e, n = send_one()
        if ttft > 0:
            ttfts.append(ttft * 1000)
            tpots.append(itl  * 1000)
            e2es.append( e2e  * 1000)
            tokens_total += n
        # Rate limiting
        if interval > 0:
            elapsed = time.time() - t_req
            sleep_t = max(0, interval - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    total_s = time.time() - t_run_start

    def pct(lst, p):
        if not lst: return 0.0
        s = sorted(lst)
        return round(s[min(int(len(s)*p/100), len(s)-1)], 2)

    if not ttfts:
        return {}

    return {
        "output_tok_throughput": round(tokens_total / max(total_s, 0.001), 2),
        "request_throughput":    round(len(ttfts) / max(total_s, 0.001), 3),
        "ttft_mean_ms":  round(statistics.mean(ttfts), 2),
        "ttft_p50_ms":   pct(ttfts, 50),
        "ttft_p99_ms":   pct(ttfts, 99),
        "ttft_p999_ms":  pct(ttfts, 99.9),
        "itl_mean_ms":   round(statistics.mean(tpots), 2) if tpots else 0.0,
        "itl_p50_ms":    pct(tpots, 50),
        "itl_p99_ms":    pct(tpots, 99),
        "itl_p999_ms":   pct(tpots, 99.9),
        "e2e_mean_ms":   round(statistics.mean(e2es), 2),
        "e2e_p50_ms":    pct(e2es, 50),
        "e2e_p99_ms":    pct(e2es, 99),
        "kv_cache_hit_rate_pct": 0.0,   # not available via HTTP endpoint
    }


# ── Main benchmark class ──────────────────────────────────────────────────────

class SGLangBenchmark:

    def __init__(self, plan, run_id: str, output_dir: Path,
                 port: int = 30000,
                 request_rates: Optional[list[float]] = None,
                 num_prompts: int = 200,
                 output_len: int = 64):
        self.plan          = plan
        self.run_id        = run_id
        self.output_dir    = output_dir
        self.port          = port
        self.request_rates = request_rates or [float("inf"), 8, 2]
        self.num_prompts   = num_prompts
        self.output_len    = output_len
        output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[SGLangResult]:
        from .collectors import IostatMonitor, NvmeSmartMonitor, PowerMonitor, GpuMonitor, DramMonitor
        p = self.plan

        server = SGLangServer(
            model_hf_id     = p.model.hf_id,
            dtype           = p.dtype,
            gpu_memory_util = min(p.hbm_cap_gb / 80.0, 0.95),
            tensor_parallel = p.tensor_parallel,
            port            = self.port,
        )

        log_dir = self.output_dir / "server"
        server.start(log_dir)
        if not server.wait_healthy(timeout_s=300):
            server.stop()
            log.error("SGLang server did not start — aborting benchmark")
            return []

        # Fetch the ACTUAL registered model name — this is what bench_serving needs
        server_model_id = server.get_model_id()
        sglang_ver      = server.get_sglang_version()
        log.info(f"Server model ID: '{server_model_id}'  SGLang: {sglang_ver}")

        # Determine whether bench_serving module is available
        bs_available = self._check_bench_serving()
        log.info(f"bench_serving available: {bs_available}")

        smart = NvmeSmartMonitor(p.ssd_device)
        power = PowerMonitor()
        gpu   = GpuMonitor()
        dram  = DramMonitor()
        smart.start(); power.start(); gpu.start(); dram.start()

        all_results: list[SGLangResult] = []
        try:
            for ctx in p.context_lens:
                for rate in self.request_rates:
                    result = self._run_one(
                        server_model_id, sglang_ver,
                        ctx, rate, bs_available)
                    all_results.append(result)
                    ok = "✓" if result.success else "✗"
                    log.info(
                        f"  {ok} ctx={ctx:,}  "
                        f"rate={'inf' if rate == float('inf') else f'{rate:.0f}'} req/s  "
                        f"method={result.method}  "
                        f"tok/s={result.output_tok_throughput:.1f}  "
                        f"TTFT_p99={result.ttft_p99_ms:.1f}ms  "
                        f"ITL_p99={result.itl_p99_ms:.1f}ms"
                    )
        finally:
            smart_m = smart.stop()
            power_m = power.stop()
            gpu_m   = gpu.stop()
            dram_m  = dram.stop()
            server.stop()

        # Overlay SMART + power + HBM + DRAM onto all results
        total_toks   = sum(r.total_tokens for r in all_results)
        hbm_used_gb  = gpu_m.get("hbm_used_gb_peak_all", 0.0)
        hbm_total_gb = gpu_m.get("hbm_total_gb_all", 0.0)
        weight_gb    = p.model.size_gb(p.dtype)

        # NVMe spill diagnosis
        hbm_headroom = max(hbm_total_gb - weight_gb, 0)
        if hbm_headroom > 50:
            spill_expected = False
            spill_reason = (
                f"Model weights={weight_gb:.0f}GB, "
                f"HBM total={hbm_total_gb:.0f}GB, "
                f"headroom={hbm_headroom:.0f}GB for KV$. "
                f"KV$ fits in HBM — no DRAM/NVMe spill. "
                f"To force NVMe I/O: use --ai-op kv-evict, or "
                f"run deepseek-r1 (336GB fp8) which exceeds your HBM."
            )
        elif hbm_headroom > 4:
            spill_expected = False
            spill_reason = (
                f"HBM headroom={hbm_headroom:.0f}GB. "
                f"KV$ may partially spill to DRAM at high batch/ctx. "
                f"Increase --context-lens or --num-prompts to fill KV$ pool."
            )
        else:
            spill_expected = True
            spill_reason = (
                f"HBM headroom={hbm_headroom:.0f}GB — KV$ should spill. "
                f"Check SSD device path with: iostat -x 1 {p.ssd_device}"
            )

        log.info(f"NVMe spill diagnosis: {spill_reason}")

        for r in all_results:
            r.waf                        = smart_m.get("waf", 0.0)
            r.temp_peak_c                = smart_m.get("temp_peak_c", 0)
            r.gpu_power_all_w_mean       = power_m.get("gpu_power_all_w_mean",
                                           gpu_m.get("power_all_gpus_w_mean", 0.0))
            r.total_system_energy_wh     = power_m.get("total_system_energy_wh", 0.0)
            ewh = r.total_system_energy_wh
            r.power_efficiency_tok_per_wh = (
                round(total_toks / ewh, 2) if ewh > 0 else 0.0)
            # HBM
            r.gpu_count             = gpu_m.get("gpu_count", 0)
            r.hbm_util_pct_mean     = gpu_m.get("hbm_util_pct_mean", 0.0)
            r.hbm_used_gb_peak_all  = hbm_used_gb
            r.hbm_total_gb_all      = hbm_total_gb
            r.gpu_temp_peak_c       = gpu_m.get("gpu_temp_peak_c", 0.0)
            # DRAM
            r.dram_used_gb_mean     = dram_m.get("dram_used_gb_mean", 0.0)
            r.dram_used_gb_peak     = dram_m.get("dram_used_gb_peak", 0.0)
            r.dram_delta_gb         = dram_m.get("dram_delta_gb", 0.0)
            r.dram_util_pct_mean    = dram_m.get("dram_util_pct_mean", 0.0)
            r.dram_total_gb         = dram_m.get("dram_total_gb", 0.0)
            # Spill
            r.nvme_spill_expected   = spill_expected
            r.nvme_spill_reason     = spill_reason

        (self.output_dir / "sglang_results.json").write_text(
            json.dumps([r.to_dict() for r in all_results], indent=2))

        return all_results

    def _check_bench_serving(self) -> bool:
        """Return True if sglang.bench_serving is importable."""
        result = subprocess.run(
            [sys.executable, "-c", "import sglang.bench_serving"],
            capture_output=True, text=True, timeout=10)
        return result.returncode == 0

    def _run_one(self, server_model_id: str, sglang_ver: str,
                 ctx: int, rate: float, try_bench_serving: bool) -> SGLangResult:
        from .collectors import IostatMonitor
        p        = self.plan
        rate_str = "inf" if rate == float("inf") else f"{rate:.1f}"
        work_dir = self.output_dir / f"ctx{ctx}_rate{rate_str}"
        work_dir.mkdir(exist_ok=True)

        iostat = IostatMonitor(p.ssd_device)
        iostat.start()

        metrics = {}
        method  = "none"
        success = False

        # ── Try 1: sglang.bench_serving ─────────────────────────────────────
        if try_bench_serving:
            metrics, method, success = self._try_bench_serving(
                server_model_id, ctx, rate, work_dir)

        # ── Try 2: direct HTTP with streaming ───────────────────────────────
        if not success:
            log.info(f"  Falling back to direct HTTP sweep "
                     f"(ctx={ctx}, rate={rate_str})")
            t0 = time.time()
            metrics = _direct_http_sweep(
                server_model_id, self.port, ctx,
                self.output_len, self.num_prompts, rate)
            dur = round(time.time() - t0, 2)
            if metrics.get("output_tok_throughput", 0) > 0:
                method  = "direct_http"
                success = True
            else:
                method  = "direct_http"
                success = False

        iostat.stop()
        io_m = iostat.summarise()

        result = SGLangResult(
            run_id          = self.run_id,
            timestamp       = datetime.now().isoformat(timespec="seconds"),
            model           = p.model.alias,
            server_model_id = server_model_id,
            context_len     = ctx,
            output_len      = self.output_len,
            request_rate    = rate if rate != float("inf") else -1,
            num_prompts     = self.num_prompts,
            dtype           = p.dtype,
            tensor_parallel = p.tensor_parallel,
            sglang_version  = sglang_ver,

            output_tok_throughput = metrics.get("output_tok_throughput", 0.0),
            request_throughput    = metrics.get("request_throughput",    0.0),
            total_tokens = int(metrics.get("output_tok_throughput", 0) * 60),

            ttft_mean_ms  = metrics.get("ttft_mean_ms",  0.0),
            ttft_p50_ms   = metrics.get("ttft_p50_ms",   0.0),
            ttft_p99_ms   = metrics.get("ttft_p99_ms",   0.0),
            ttft_p999_ms  = metrics.get("ttft_p999_ms",  0.0),
            itl_mean_ms   = metrics.get("itl_mean_ms",   0.0),
            itl_p50_ms    = metrics.get("itl_p50_ms",    0.0),
            itl_p99_ms    = metrics.get("itl_p99_ms",    0.0),
            itl_p999_ms   = metrics.get("itl_p999_ms",   0.0),
            e2e_mean_ms   = metrics.get("e2e_mean_ms",   0.0),
            e2e_p50_ms    = metrics.get("e2e_p50_ms",    0.0),
            e2e_p99_ms    = metrics.get("e2e_p99_ms",    0.0),
            kv_cache_hit_rate_pct = metrics.get("kv_cache_hit_rate_pct", 0.0),

            read_bw_mb_mean  = io_m.get("read_bw_mb_mean",  0.0),
            write_bw_mb_mean = io_m.get("write_bw_mb_mean", 0.0),
            read_iops_mean   = io_m.get("read_iops_mean",   0.0),
            write_iops_mean  = io_m.get("write_iops_mean",  0.0),
            r_await_ms_p99   = io_m.get("r_await_ms_p99",   0.0),
            util_pct_mean    = io_m.get("util_pct_mean",    0.0),

            success = success,
            method  = method,
            notes   = f"server_model={server_model_id}",
        )
        (work_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2))
        return result

    def _probe_bench_serving_args(self) -> dict:
        """
        Run bench_serving --help and parse which arguments it accepts.
        Returns a dict of supported arg names so we build the right command.
        """
        try:
            r = subprocess.run(
                [sys.executable, "-m", "sglang.bench_serving", "--help"],
                capture_output=True, text=True, timeout=10)
            help_text = r.stdout + r.stderr
        except Exception:
            return {}

        supported = {}
        supported["base_url"]     = "--base-url"     in help_text
        supported["host_port"]    = "--host"         in help_text and "--port" in help_text
        supported["backend"]      = "--backend"      in help_text
        supported["input_len"]    = "--input-len"    in help_text
        supported["output_len"]   = "--output-len"   in help_text
        supported["max_tokens"]   = "--max-tokens"   in help_text
        supported["request_rate"] = "--request-rate" in help_text
        supported["num_prompts"]  = "--num-prompts"  in help_text
        supported["dataset_name"] = "--dataset-name" in help_text

        log.debug(f"bench_serving supports: {supported}")
        return supported

    def _try_bench_serving(self, server_model_id: str, ctx: int,
                           rate: float, work_dir: Path
                           ) -> tuple[dict, str, bool]:
        """
        Build the correct bench_serving command for the installed SGLang version
        by probing --help first, then run it.
        """
        rate_str = "inf" if rate == float("inf") else f"{rate}"
        base_url = f"http://127.0.0.1:{self.port}"
        sup      = self._probe_bench_serving_args()

        if not sup:
            log.warning("Could not probe bench_serving args — skipping")
            return {}, "bench_serving_skipped", False

        # ── Build command from supported args ─────────────────────────────────
        cmd = [sys.executable, "-m", "sglang.bench_serving"]

        # Server address
        if sup.get("base_url"):
            cmd += ["--base-url", base_url]
        elif sup.get("host_port"):
            cmd += ["--host", "127.0.0.1", "--port", str(self.port)]
        else:
            log.warning("bench_serving has neither --base-url nor --host — skipping")
            return {}, "bench_serving_skipped", False

        # Backend (optional)
        if sup.get("backend"):
            cmd += ["--backend", "sglang"]

        # Model
        cmd += ["--model", server_model_id]

        # Number of prompts
        if sup.get("num_prompts"):
            cmd += ["--num-prompts", str(self.num_prompts)]

        # Context / output length — arg names vary by version
        if sup.get("input_len"):
            cmd += ["--input-len", str(ctx)]
        # some versions use --dataset-name random with --random-input-len
        elif sup.get("dataset_name"):
            cmd += ["--dataset-name", "random",
                    "--random-input-len", str(ctx),
                    "--random-output-len", str(self.output_len)]

        if sup.get("output_len") and not sup.get("dataset_name"):
            cmd += ["--output-len", str(self.output_len)]

        # Request rate
        if sup.get("request_rate") and rate != float("inf"):
            cmd += ["--request-rate", rate_str]

        log.info(f"bench_serving cmd: {' '.join(cmd[2:])}")
        (work_dir / "bench_cmd.txt").write_text(" ".join(cmd))

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600)
            bench_out = proc.stdout + proc.stderr
            (work_dir / "bench_output.txt").write_text(bench_out)

            if proc.returncode != 0:
                err_lines = [l for l in bench_out.splitlines()
                             if any(k in l.lower()
                                    for k in ["error","traceback","unrecognized","usage"])]
                log.warning(
                    f"bench_serving exit={proc.returncode}:\n" +
                    "\n".join(err_lines[:10]))
                return {}, "bench_serving_failed", False

            metrics = _parse_bench_output(bench_out)
            if metrics.get("output_tok_throughput", 0) > 0:
                log.info("bench_serving succeeded")
                return metrics, "bench_serving", True

            log.warning(
                f"bench_serving returned 0 tok/s. Output preview:\n{bench_out[:400]}")
            return {}, "bench_serving_zero", False

        except subprocess.TimeoutExpired:
            log.warning("bench_serving timed out after 600s")
            return {}, "bench_serving_timeout", False
        except Exception as e:
            log.warning(f"bench_serving exception: {e}")
            return {}, "bench_serving_error", False
