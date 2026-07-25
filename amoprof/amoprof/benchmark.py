"""
benchmark.py — Runs one RunPlan through the full collect → run → collect cycle
and returns a list of ResultRow objects, one per (context_len × batch_size).
"""

from __future__ import annotations
import json, time, socket, platform
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .plan       import RunPlan
from .runner     import WorkloadRunner, RunResult
from .collectors import (IostatMonitor, NvmeSmartMonitor,
                          BiolatencyCollector, GpuMonitor, PowerMonitor)


# ── Result row — one CSV row ───────────────────────────────────────────────────

@dataclass
class ResultRow:
    # Identity
    run_id: str        = ""
    timestamp: str     = ""
    hostname: str      = ""
    os_kernel: str     = ""

    # Experiment spec
    model:       str   = ""
    arch:        str   = ""
    hbm_cap_gb:  float = 0.0
    dram_cap_gb: float = 0.0
    nvme_cap_gb: float = 0.0
    ai_op:       str   = ""
    tier:        str   = ""
    framework:   str   = ""
    dtype:       str   = ""
    context_len: int   = 0
    batch_size:  int   = 0
    tensor_parallel: int = 1

    # Derived sizing
    weight_gb:        float = 0.0
    kv_size_gb:       float = 0.0
    io_pattern:       str   = ""
    ssd_dimensions:   str   = ""

    # Run outcome
    success:          bool  = False
    duration_s:       float = 0.0
    throughput_tok_s: float = 0.0
    ttft_ms:          float = 0.0
    tokens_generated: int   = 0
    notes:            str   = ""

    # SSD iostat
    read_bw_mb_mean:   float = 0.0
    read_bw_mb_peak:   float = 0.0
    write_bw_mb_mean:  float = 0.0
    write_bw_mb_peak:  float = 0.0
    read_iops_mean:    float = 0.0
    read_iops_peak:    float = 0.0
    write_iops_mean:   float = 0.0
    write_iops_peak:   float = 0.0
    r_await_ms_mean:   float = 0.0
    r_await_ms_p99:    float = 0.0
    r_await_ms_p999:   float = 0.0
    w_await_ms_mean:   float = 0.0
    w_await_ms_p99:    float = 0.0
    avgqu_sz_mean:     float = 0.0
    util_pct_mean:     float = 0.0
    iostat_samples:    int   = 0

    # Endurance / WAF / SMART
    host_written_gb:   float = 0.0
    nand_written_gb:   float = 0.0
    waf:               float = 0.0
    temp_start_c:      int   = 0
    temp_end_c:        int   = 0
    temp_peak_c:       int   = 0

    # Latency histogram (biolatency)
    read_lat_p50_us:      float = 0.0
    read_lat_p99_us:      float = 0.0
    read_lat_p999_us:     float = 0.0
    biolatency_available: bool  = False

    # GPU / HBM utilisation
    gpu_available:     bool  = False
    gpu_util_mean:     float = 0.0
    gpu_util_peak:     float = 0.0
    hbm_util_mean:     float = 0.0
    hbm_used_mb_peak:  float = 0.0

    # ── Power — GPU (nvidia-smi) ──────────────────────────────────────────
    gpu_count:                   int   = 0
    gpu_power_w_mean:            float = 0.0   # per-GPU mean (W)
    gpu_power_w_peak:            float = 0.0   # per-GPU peak (W)
    gpu_power_all_w_mean:        float = 0.0   # all GPUs combined (W)
    gpu_power_limit_w:           float = 0.0   # TDP limit per GPU (W)
    gpu_utilisation_pct:         float = 0.0   # power draw / TDP %
    gpu_temp_peak_c:             float = 0.0   # peak GPU temperature (°C)
    gpu_energy_wh:               float = 0.0   # GPU energy consumed (Wh)

    # ── Power — CPU package (Intel RAPL) ─────────────────────────────────
    cpu_package_power_w_mean:    float = 0.0   # all sockets combined (W)
    cpu_package_energy_wh:       float = 0.0   # CPU energy (Wh)

    # ── Power — DRAM subsystem (Intel RAPL) ──────────────────────────────
    dram_power_w_mean:           float = 0.0   # DRAM subsystem (W)
    dram_energy_wh:              float = 0.0   # DRAM energy (Wh)

    # ── Power — System / PSU (IPMI DCMI) ─────────────────────────────────
    system_power_w_mean:         float = 0.0   # chassis mean (W)
    system_power_w_peak:         float = 0.0   # chassis peak (W)
    system_energy_wh:            float = 0.0   # chassis energy (Wh)

    # ── Power — Combined totals ───────────────────────────────────────────
    total_system_power_w_mean:   float = 0.0   # best estimate total (W)
    total_system_energy_wh:      float = 0.0   # best estimate total (Wh)
    power_sources:               str   = ""    # active source list
    power_elapsed_s:             float = 0.0   # measurement window (s)

    # ── Power efficiency ──────────────────────────────────────────────────
    power_efficiency_tok_per_wh: float = 0.0   # tokens / Wh

    # Derived SSD
    ssd_efficiency_pct: float = 0.0   # observed BW as % of fio ceiling

    def to_dict(self) -> dict:
        return asdict(self)


# ── Benchmark orchestrator ─────────────────────────────────────────────────────

class Benchmark:

    def __init__(self, plan: RunPlan, run_id: str, output_dir: Path,
                 fio_ceiling_bw: float = 0.0):
        self.plan        = plan
        self.run_id      = run_id
        self.output_dir  = output_dir
        self.fio_ceiling = fio_ceiling_bw
        self._hostname   = socket.gethostname()
        self._kernel     = platform.release()
        output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[ResultRow]:
        rows = []
        for ctx in self.plan.context_lens:
            for bs in self.plan.batch_sizes:
                rows.append(self._run_one(ctx, bs))
        return rows

    def _run_one(self, ctx: int, bs: int) -> ResultRow:
        p        = self.plan
        work_dir = self.output_dir / f"ctx{ctx}_bs{bs}"

        row = ResultRow(
            run_id         = self.run_id,
            timestamp      = datetime.now().isoformat(timespec="seconds"),
            hostname       = self._hostname,
            os_kernel      = self._kernel,
            model          = p.model.alias,
            arch           = p.model.arch.value,
            hbm_cap_gb     = p.hbm_cap_gb,
            dram_cap_gb    = p.dram_cap_gb,
            nvme_cap_gb    = p.nvme_cap_gb,
            ai_op          = p.ai_op.value,
            tier           = p.tier.value,
            framework      = p.framework.value,
            dtype          = p.dtype,
            context_len    = ctx,
            batch_size     = bs,
            tensor_parallel= p.tensor_parallel,
            weight_gb      = p.weight_gb,
            kv_size_gb     = round(p.model.kv_size_gb(ctx, bs, p.dtype), 3),
            io_pattern     = p.io_pattern,
            ssd_dimensions = p.ssd_dimensions,
        )

        # ── Start all collectors ──────────────────────────────────────────
        iostat  = IostatMonitor(p.ssd_device)
        smart   = NvmeSmartMonitor(p.ssd_device)
        bio     = BiolatencyCollector(p.ssd_device,
                                      duration_s=min(ctx // 512 + 30, 120))
        gpu     = GpuMonitor()
        power   = PowerMonitor()

        iostat.start(); smart.start(); bio.start(); gpu.start(); power.start()

        # ── Run workload ──────────────────────────────────────────────────
        runner = WorkloadRunner(p, work_dir)
        result: RunResult = runner.run(ctx, bs)

        # ── Stop all collectors ───────────────────────────────────────────
        iostat.stop()
        smart_m = smart.stop()
        bio_m   = bio.stop()
        gpu_m   = gpu.stop()
        power_m = power.stop(tokens_generated=result.tokens_generated)
        io_m    = iostat.summarise()

        # ── Populate run outcome ──────────────────────────────────────────
        row.success          = result.success
        row.duration_s       = result.duration_s
        row.throughput_tok_s = result.throughput_tok_s
        row.ttft_ms          = result.ttft_ms
        row.tokens_generated = result.tokens_generated
        row.notes            = result.notes

        # ── Merge scalar metrics from all collectors ──────────────────────
        for src in (io_m, smart_m, bio_m):
            for k, v in src.items():
                if isinstance(v, (int, float, bool, str)) and hasattr(row, k):
                    setattr(row, k, v)

        # GPU utilisation
        if gpu_m.get("gpu_available"):
            row.gpu_available    = True
            row.gpu_util_mean    = gpu_m.get("gpu_util_mean",    0.0)
            row.gpu_util_peak    = gpu_m.get("gpu_util_peak",    0.0)
            row.hbm_util_mean    = gpu_m.get("hbm_util_mean",    0.0)
            row.hbm_used_mb_peak = gpu_m.get("hbm_used_mb_peak", 0.0)

        # Power metrics (scalar fields only — lists/dicts go into JSON snapshot)
        _POWER_SCALAR_FIELDS = {
            "gpu_count", "gpu_power_w_mean", "gpu_power_w_peak",
            "gpu_power_all_w_mean", "gpu_power_limit_w",
            "gpu_utilisation_pct", "gpu_temp_peak_c", "gpu_energy_wh",
            "cpu_package_power_w_mean", "cpu_package_energy_wh",
            "dram_power_w_mean", "dram_energy_wh",
            "system_power_w_mean", "system_power_w_peak", "system_energy_wh",
            "total_system_power_w_mean", "total_system_energy_wh",
            "power_sources", "power_elapsed_s",
            "power_efficiency_tok_per_wh",
        }
        for k in _POWER_SCALAR_FIELDS:
            v = power_m.get(k)
            if v is not None and isinstance(v, (int, float, bool, str)):
                setattr(row, k, v)

        # SSD efficiency vs fio ceiling
        bw = row.read_bw_mb_mean or row.write_bw_mb_mean
        if self.fio_ceiling > 0 and bw > 0:
            row.ssd_efficiency_pct = round(bw / self.fio_ceiling * 100, 1)

        # Persist full JSON snapshot (includes per-GPU lists, RAPL detail, etc.)
        snap_data = row.to_dict()
        snap_data["_power_detail"] = {
            k: v for k, v in power_m.items()
            if not isinstance(v, (int, float, bool, str))
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "result.json").write_text(
            json.dumps(snap_data, indent=2, default=str))

        return row
