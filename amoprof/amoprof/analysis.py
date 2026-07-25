"""
analysis.py — AMOprof scenario-based metric mapper and comparison engine.

Eight storage/memory characterisation scenarios, each mapping raw collector
metrics into derived KPIs and producing per-scenario comparison tables.

Scenarios
─────────
  S1  Read vs Write Ratio          — KV$ restore vs eviction balance
  S2  Latency vs Capacity          — latency cliff as capacity fills
  S3  IO Pattern & Size            — sequential vs random, effective IO size
  S4  IO Locality                  — KV$ cache hit rate and tier distribution
  S5  IO Frequency                 — IOPS, request rate, engine utilisation
  S6  Concurrency                  — all dimensions vs inference_concurrency
  S7  Sustainability & Endurance   — WAF, TBW, DWPD, lifetime projection
  S8  Device Degradation           — latency / WAF growth as wear accumulates

Usage
─────
  from amoprof.analysis import ScenarioAnalyzer
  az = ScenarioAnalyzer.from_csv("concurrency_sweep.csv")
  report = az.full_report()          # dict[scenario_id → ScenarioResult]
  az.print_report(report)
  az.write_csv(report, "analysis/")
"""

from __future__ import annotations

import csv
import io
import math
import statistics
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Colour codes for terminal output ──────────────────────────────────────────
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_R  = "\033[91m"   # red
_B  = "\033[94m"   # blue
_W  = "\033[97m"   # white/bold
_RE = "\033[0m"    # reset
_NO_COLOR = False  # set True to disable


def _c(text: str, code: str) -> str:
    return text if _NO_COLOR else f"{code}{text}{_RE}"


# ── Row type ───────────────────────────────────────────────────────────────────
Row = dict[str, float | int | str]


def _f(row: Row, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, default)
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


# ── Per-scenario result ────────────────────────────────────────────────────────
@dataclass
class ScenarioResult:
    scenario_id:   str
    scenario_name: str
    description:   str
    rows:          list[dict]          # one dict per sweep point (derived KPIs)
    summary:       dict[str, Any]      # scalar summary across all rows
    observations:  list[str]           # human-readable findings
    warnings:      list[str]           # anomalies / threshold breaches


# ── Main analyser ──────────────────────────────────────────────────────────────
def _classify_bottleneck(pf_tok_s: float, dc_tok_s: float,
                         pf_hbm_gb: float = 0.0,
                         dc_miss_ms: float = 0.0,
                         dc_tokens: float = 0.0) -> str:
    """
    Classify the compute bottleneck using available signals.

    Primary signal (when SGLang realtime_tokens_total is populated):
      pf_tok_s >> dc_tok_s  → prefill-bound
      dc_tok_s >> pf_tok_s  → decode-bound

    Fallback signals (when tok/s are zero due to metrics unavailability):
      pf_hbm_gb > 0 AND dc_miss_ms == 0  → prefill-bound (writes, no misses)
      dc_miss_ms > 0                      → decode_bound  (decode stalls)
      dc_tokens >> 0 AND pf_hbm_gb ≈ 0   → decode_bound  (output >> input)
      all zero                             → unknown (not 'balanced')

    SWE-bench is structurally decode-heavy: prompt ~4K tokens, output ~300-800
    tokens, but each output token re-reads the full KV$ context for attention.
    The total KV$ read traffic (dc_total_kv_read_gb) >> KV$ write traffic.
    """
    # Primary: throughput ratio
    if pf_tok_s > 0 or dc_tok_s > 0:
        if pf_tok_s > 0 and dc_tok_s == 0:
            return "prefill_bound"
        if dc_tok_s > 0 and pf_tok_s == 0:
            return "decode_bound"
        if pf_tok_s < dc_tok_s * 0.5:
            return "prefill_bound"
        if dc_tok_s < pf_tok_s * 0.5:
            return "decode_bound"
        return "balanced"

    # Fallback: structural signals
    if dc_miss_ms > 0.5:
        # Decode stalls on KV$ misses — definitely decode-bound
        return "decode_bound"
    if dc_tokens > 50:
        # Long output sequences — decode-dominated by token count
        return "decode_bound"
    if pf_hbm_gb > 0.5 and dc_miss_ms == 0:
        # Significant prefill KV$ writes, no decode stalls
        return "prefill_bound"
    if pf_hbm_gb > 0 and dc_tokens == 0:
        return "prefill_bound"

    return "unknown"   # not 'balanced' — we simply don't have the data


class ScenarioAnalyzer:
    """
    Load a concurrency_sweep.csv or turns_sweep.csv produced by AMOprof and
    compute all eight scenario analyses.

    The pivot key is auto-detected:
      'inference_concurrency'  → concurrency sweep
      'num_turns'              → turns sweep
    """

    # ── Thresholds (configurable) ──────────────────────────────────────────
    THRESHOLDS = {
        "kv_fill_pct_overflow":    100.0,   # % — pool overflow, SSD I/O starts
        "kv_fill_pct_pressure":     80.0,   # % — high pressure warning
        "ssd_util_saturation":      90.0,   # % — SSD near saturation
        "ssd_util_high":            60.0,   # % — elevated SSD load
        "bio_lat_p99_warn_us":     500.0,   # µs — P99 block latency warning
        "bio_lat_p99_crit_us":    2000.0,   # µs — P99 block latency critical
        "kv_miss_penalty_warn_ms":  50.0,   # ms — TPOT miss penalty warning
        "kv_miss_penalty_crit_ms": 200.0,   # ms — TPOT miss penalty critical
        "waf_warn":                  1.5,   # WAF warning
        "waf_crit":                  3.0,   # WAF critical
        "temp_warn_c":              65,     # °C  — SSD temp warning
        "temp_crit_c":              75,     # °C  — SSD temp critical
        "rw_ratio_warn":             2.0,   # R/W > 2 → read-dominated
        "lifetime_pct_warn":        10.0,   # % — <10% rated TBW remaining
    }

    def __init__(self, rows: list[Row], source_path: str = ""):
        self.rows        = rows
        self.source_path = source_path
        # detect pivot key
        if rows and "num_turns" in rows[0]:
            self.pivot = "num_turns"
        else:
            self.pivot = "inference_concurrency"
        self._pivot_vals = [_f(r, self.pivot) for r in rows]

    # ── Constructors ──────────────────────────────────────────────────────────
    @classmethod
    def from_csv(cls, path: str | Path) -> "ScenarioAnalyzer":
        path = Path(path)
        rows: list[Row] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: _safe_num(v) for k, v in row.items()})
        return cls(rows, str(path))

    @classmethod
    def from_sweep_points(cls, points: list) -> "ScenarioAnalyzer":
        """Accept a list of SweepPoint or TurnsSweepPoint dataclass instances."""
        import dataclasses
        rows = [dataclasses.asdict(p) for p in points]
        return cls(rows)

    # ── Public API ────────────────────────────────────────────────────────────
    def full_report(self) -> dict[str, ScenarioResult]:
        return {
            "S1": self.s1_rw_ratio(),
            "S2": self.s2_latency_vs_capacity(),
            "S3": self.s3_io_pattern_size(),
            "S4": self.s4_io_locality(),
            "S5": self.s5_io_frequency(),
            "S6": self.s6_concurrency(),
            "S7": self.s7_sustainability(),
            "S8": self.s8_degradation(),
        }

    # ── S1: Read vs Write Ratio ───────────────────────────────────────────────
    def s1_rw_ratio(self) -> ScenarioResult:
        """
        KV$ restore (L3 reads) vs KV$ eviction (SSD writes).

        Below overflow_concurrency: both are 0.  Above: eviction writes cold
        blocks to SSD; decode fetches them back as reads.  In steady-state the
        read and write rates should be roughly equal (write-once / read-many).
        A high read:write ratio means blocks are being fetched repeatedly
        (thrashing).  A ratio < 1 means evictions outpace restores (dropping
        blocks that are never needed again — wasted write amplification).
        """
        T = self.THRESHOLDS
        derived = []
        for r in self.rows:
            rd = _f(r, "ssd_read_bw_mb_mean")
            wr = _f(r, "ssd_write_bw_mb_mean")
            rd_iops = _f(r, "ssd_read_iops_mean")
            wr_iops = _f(r, "ssd_write_iops_mean")
            total_bw = rd + wr

            rw_bw_ratio  = rd / wr  if wr > 0.1 else (None if rd < 0.1 else float("inf"))
            rw_iop_ratio = rd_iops / wr_iops if wr_iops > 1 else None

            # Token-level: restored / evicted  (should be ≤ 1; >1 = same block fetched multiple times)
            ev = _f(r, "kv_evicted_tokens_mean")
            rs = _f(r, "kv_restored_tokens_mean")
            token_rr = rs / ev if ev > 0 else None

            # HBM: prefill writes KV$; decode reads it every step
            pf_gb = _f(r, "hbm_prefill_delta_gb_mean")
            dc_gb = _f(r, "hbm_decode_delta_gb_mean")

            # SSD active?
            active = total_bw > 0.5

            derived.append({
                self.pivot:           _f(r, self.pivot),
                "ssd_read_bw":        round(rd, 2),
                "ssd_write_bw":       round(wr, 2),
                "total_ssd_bw":       round(total_bw, 2),
                "rw_bw_ratio":        round(rw_bw_ratio, 3) if rw_bw_ratio is not None else "N/A (no SSD I/O)",
                "rw_iop_ratio":       round(rw_iop_ratio, 3) if rw_iop_ratio is not None else "N/A",
                "kv_token_rr":        round(token_rr, 3) if token_rr is not None else "N/A",
                "hbm_prefill_gb":     round(pf_gb, 3),
                "hbm_decode_gb":      round(dc_gb, 3),
                "hbm_rw_ratio":       round(pf_gb / dc_gb, 2) if dc_gb > 0.01 else "N/A",
                "ssd_active":         active,
                "kv_eviction_mb_s":   round(_f(r, "kv_eviction_mb_s"), 2),
                "kv_restore_mb_s":    round(_f(r, "kv_restore_mb_s"), 2),
            })

        # Summary
        active_rows = [d for d in derived if d["ssd_active"]]
        ratios = [d["rw_bw_ratio"] for d in active_rows if isinstance(d["rw_bw_ratio"], float)]
        mean_rr = statistics.mean(ratios) if ratios else 0.0
        max_total = max((d["total_ssd_bw"] for d in derived), default=0.0)

        observations, warnings = [], []
        if not active_rows:
            observations.append("SSD is idle across all levels — concurrency is below overflow threshold.")
        else:
            observations.append(f"SSD I/O active from level {active_rows[0][self.pivot]} onward.")
            if mean_rr > T["rw_ratio_warn"]:
                observations.append(f"Mean R/W ratio = {mean_rr:.2f} — read-dominated; blocks fetched multiple times (thrashing).")
                warnings.append(f"High R/W ratio {mean_rr:.2f} suggests KV$ pool too small for working set.")
            elif 0.3 < mean_rr <= T["rw_ratio_warn"]:
                observations.append(f"Mean R/W ratio = {mean_rr:.2f} — balanced eviction/restore pattern (healthy).")
            elif mean_rr > 0:
                observations.append(f"Mean R/W ratio = {mean_rr:.2f} — write-dominated; evicting more than restoring (cold KV$ blocks wasted).")
            observations.append(f"Peak total SSD BW = {max_total:.1f} MB/s.")

        return ScenarioResult(
            scenario_id="S1", scenario_name="Read vs Write Ratio",
            description="KV$ restore (read) vs eviction (write) balance across Memory and SSD tiers.",
            rows=derived, summary={"mean_rw_ratio": round(mean_rr, 3), "max_total_ssd_bw_mb_s": round(max_total, 1)},
            observations=observations, warnings=warnings)

    # ── S2: Latency vs Capacity ───────────────────────────────────────────────
    def s2_latency_vs_capacity(self) -> ScenarioResult:
        """
        Latency cliff as Memory/SSD capacity fills.

        Three latency dimensions:
          - HBM fill % → kv_miss_penalty_ms (TPOT overhead from SSD fetches)
          - SSD util % → bio_lat_p99/p999_us (block layer latency at saturation)
          - DRAM fill % → dram_hicache_staging (buffering pressure)

        The HBM latency cliff is sharp: near-zero miss penalty below overflow,
        then linear-to-superlinear growth above it.  The SSD latency cliff
        appears later at ~80–90% device utilisation.
        """
        T = self.THRESHOLDS
        derived = []
        for r in self.rows:
            kv_fill  = _f(r, "kv_fill_pct_mean")
            ssd_util = _f(r, "ssd_util_pct_mean")
            hbm_util = _f(r, "hbm_util_pct_mean")
            dram_util= _f(r, "dram_util_pct_mean")
            miss_pen = _f(r, "kv_miss_penalty_ms")
            bio_p99  = _f(r, "bio_lat_p99_us_mean")
            bio_p999 = _f(r, "bio_lat_p999_us_mean")
            tpot_p99 = _f(r, "tpot_p99_ms")
            ttft     = _f(r, "ttft_mean_ms")
            hicache_load = _f(r, "hicache_load_back_ms_mean")
            hicache_evict= _f(r, "hicache_eviction_ms_mean")

            # Latency budget breakdown (where does TPOT come from?)
            ssd_contrib_pct = round(miss_pen / tpot_p99 * 100, 1) if tpot_p99 > 0 else 0.0

            derived.append({
                self.pivot:               _f(r, self.pivot),
                "kv_fill_pct":            round(kv_fill, 1),
                "ssd_util_pct":           round(ssd_util, 1),
                "hbm_util_pct":           round(hbm_util, 1),
                "dram_util_pct":          round(dram_util, 1),
                "kv_miss_penalty_ms":     round(miss_pen, 2),
                "ssd_pct_of_tpot":        ssd_contrib_pct,
                "bio_lat_p99_us":         round(bio_p99, 1),
                "bio_lat_p999_us":        round(bio_p999, 1),
                "hicache_load_ms":        round(hicache_load, 2),
                "hicache_evict_ms":       round(hicache_evict, 2),
                "tpot_p99_ms":            round(tpot_p99, 1),
                "ttft_ms":                round(ttft, 1),
                "nvme_sysfs_lat_ms":      round(_f(r, "nvme_rd_lat_ms_sysfs"), 3),
                "latency_regime": (
                    "no_ssd_io"    if kv_fill < 95  else
                    "light_evict"  if kv_fill < 120 else
                    "heavy_evict"  if kv_fill < 200 else
                    "saturated"
                ),
            })

        # Latency cliff detection: find the fill % where miss_penalty first exceeds warning
        cliff_fill = None
        for d in derived:
            if isinstance(d["kv_miss_penalty_ms"], float) and d["kv_miss_penalty_ms"] >= T["kv_miss_penalty_warn_ms"]:
                cliff_fill = d["kv_fill_pct"]
                break

        peak_bio_p999 = max((d["bio_lat_p999_us"] for d in derived), default=0.0)
        peak_miss_pen = max((d["kv_miss_penalty_ms"] for d in derived
                             if isinstance(d["kv_miss_penalty_ms"], float)), default=0.0)
        peak_ssd_contrib = max((d["ssd_pct_of_tpot"] for d in derived), default=0.0)

        observations, warnings = [], []
        if cliff_fill:
            observations.append(f"Latency cliff at KV$ fill ≈ {cliff_fill:.0f}% — miss penalty exceeds {T['kv_miss_penalty_warn_ms']} ms.")
        else:
            observations.append("No significant latency cliff observed (KV$ pool not heavily pressured).")
        observations.append(f"Peak TPOT component from SSD: {peak_ssd_contrib:.1f}% of tpot_p99.")
        if peak_miss_pen >= T["kv_miss_penalty_crit_ms"]:
            warnings.append(f"Critical: peak miss penalty {peak_miss_pen:.1f} ms — SSD dominates decode latency.")
        if peak_bio_p999 >= T["bio_lat_p99_crit_us"]:
            warnings.append(f"Critical: P999 block latency {peak_bio_p999:.0f} µs — SSD likely queuing.")

        return ScenarioResult(
            scenario_id="S2", scenario_name="Latency vs Capacity",
            description="Latency cliff as KV$ pool, DRAM staging, and SSD capacity fill.",
            rows=derived,
            summary={"latency_cliff_kv_fill_pct": cliff_fill,
                     "peak_miss_penalty_ms": round(peak_miss_pen, 2),
                     "peak_ssd_pct_of_tpot": round(peak_ssd_contrib, 1),
                     "peak_bio_p999_us": round(peak_bio_p999, 1)},
            observations=observations, warnings=warnings)

    # ── S3: IO Pattern & Size ─────────────────────────────────────────────────
    def s3_io_pattern_size(self) -> ScenarioResult:
        """
        Effective IO block size and queue depth characterise the access pattern.

        KV$ eviction/restore uses random small writes/reads (32K–256K per block).
        This is the worst case for NVMe SSDs that prefer large sequential IO.

        Derived:
          avg_read_io_kb  = read_bw_MB * 1024 / read_iops
          avg_write_io_kb = write_bw_MB * 1024 / write_iops
          queue_depth     = ssd_avgqu_sz_mean  (from iostat)
          pattern_type    = "random"  if avg_io < 128 KB
                          = "mixed"   if 128 KB ≤ avg_io < 512 KB
                          = "seq"     if avg_io ≥ 512 KB
        """
        derived = []
        for r in self.rows:
            rd_bw   = _f(r, "ssd_read_bw_mb_mean")
            wr_bw   = _f(r, "ssd_write_bw_mb_mean")
            rd_iops = _f(r, "ssd_read_iops_mean")
            wr_iops = _f(r, "ssd_write_iops_mean")
            qd      = _f(r, "ssd_avgqu_sz_mean")
            inflight= _f(r, "nvme_inflight_mean")
            inflight_peak = _f(r, "nvme_inflight_peak")

            avg_rd_kb = round(rd_bw * 1024 / rd_iops, 1) if rd_iops > 0 else 0.0
            avg_wr_kb = round(wr_bw * 1024 / wr_iops, 1) if wr_iops > 0 else 0.0

            def _pattern(kb: float) -> str:
                if kb == 0:    return "idle"
                if kb < 128:   return "random_small"
                if kb < 512:   return "mixed"
                return "sequential"

            ssd_active = (rd_bw + wr_bw) > 0.5
            derived.append({
                self.pivot:         _f(r, self.pivot),
                "avg_read_io_kb":   avg_rd_kb,
                "avg_write_io_kb":  avg_wr_kb,
                "read_pattern":     _pattern(avg_rd_kb),
                "write_pattern":    _pattern(avg_wr_kb),
                "queue_depth":      round(qd, 2),
                "nvme_inflight":    round(inflight, 1),
                "nvme_inflight_peak": int(inflight_peak),
                "ssd_read_iops":    round(rd_iops, 0),
                "ssd_write_iops":   round(wr_iops, 0),
                "ssd_active":       ssd_active,
                # KV$ block size theory vs actual
                # Theoretical: 80 layers × 8 heads × 128 dim × 2 × 1 byte = 163840 B = 160 KB per token-row
                # But HiCache writes slabs, not individual token rows — actual size depends on slab granularity
                "theoretical_kv_block_kb": round(_f(r, "kv_bytes_per_token") / 1024, 1),
            })

        active = [d for d in derived if d["ssd_active"]]
        read_patterns  = [d["read_pattern"]  for d in active]
        write_patterns = [d["write_pattern"] for d in active]
        dominant_read  = max(set(read_patterns),  key=read_patterns.count)  if read_patterns  else "idle"
        dominant_write = max(set(write_patterns), key=write_patterns.count) if write_patterns else "idle"
        avg_rd = statistics.mean(d["avg_read_io_kb"]  for d in active) if active else 0.0
        avg_wr = statistics.mean(d["avg_write_io_kb"] for d in active) if active else 0.0
        max_qd = max((d["queue_depth"] for d in derived), default=0.0)
        max_inf= max((d["nvme_inflight_peak"] for d in derived), default=0)

        observations, warnings = [], []
        if not active:
            observations.append("SSD idle — IO pattern analysis not applicable below overflow threshold.")
        else:
            observations.append(f"Dominant read pattern:  {dominant_read}  (mean IO size {avg_rd:.0f} KB).")
            observations.append(f"Dominant write pattern: {dominant_write} (mean IO size {avg_wr:.0f} KB).")
            observations.append(f"Peak queue depth: {max_qd:.1f}  |  Peak NVMe in-flight: {max_inf}.")
            if "random" in dominant_read:
                warnings.append(
                    f"Random read IO ({avg_rd:.0f} KB avg) — SSD random-read IOPS is the bottleneck, "
                    "not sequential BW. Consider larger HiCache block size.")
            if max_qd > 32:
                warnings.append(f"Queue depth {max_qd:.1f} exceeds typical NVMe sweet-spot (8–16) — I/O scheduler may be limiting throughput.")

        return ScenarioResult(
            scenario_id="S3", scenario_name="IO Pattern & Size",
            description="Effective IO block size, queue depth, and sequential vs random characterisation.",
            rows=derived,
            summary={"dominant_read_pattern": dominant_read, "dominant_write_pattern": dominant_write,
                     "mean_read_io_kb": round(avg_rd, 1), "mean_write_io_kb": round(avg_wr, 1),
                     "max_queue_depth": round(max_qd, 1), "max_nvme_inflight": max_inf},
            observations=observations, warnings=warnings)

    # ── S4: IO Locality ───────────────────────────────────────────────────────
    def s4_io_locality(self) -> ScenarioResult:
        """
        KV$ cache hit rate and tier distribution — how well the working set
        fits in fast memory tiers.

        Locality metrics:
          kv_cache_hit_rate_pct       — prefix cache (RadixAttention) hits
          cache_hit_rate_realtime_pct — realtime: cache / (cache + compute)
          kv_l1/l2/l3_tokens          — tier occupancy (HBM / DRAM / SSD)
          l1_pct                      — fraction of tokens in HBM (warm)
          l2_pct                      — fraction in DRAM staging
          l3_pct                      — fraction on SSD (cold)

        High l3_pct means most tokens are cold — high miss rate expected.
        High cache_hit_rate with low l3_pct means good locality.
        """
        derived = []
        for r in self.rows:
            l1 = _f(r, "kv_l1_device_tokens_mean")
            l2 = _f(r, "kv_l2_host_tokens_mean")
            l3 = _f(r, "kv_l3_storage_tokens_mean")
            total_tok = l1 + l2 + l3
            cap = _f(r, "kv_tokens_capacity")

            l1_pct = round(l1 / total_tok * 100, 1) if total_tok > 0 else 0.0
            l2_pct = round(l2 / total_tok * 100, 1) if total_tok > 0 else 0.0
            l3_pct = round(l3 / total_tok * 100, 1) if total_tok > 0 else 0.0
            pool_fill = round(total_tok / cap * 100, 1) if cap > 0 else 0.0

            hit_prefix  = _f(r, "kv_cache_hit_rate_pct")
            hit_rt      = _f(r, "cache_hit_rate_realtime_pct")
            evicted_gb  = _f(r, "hbm_kv_evicted_gb_mean")
            pf_gb       = _f(r, "hbm_prefill_delta_gb_mean")

            # Effective reuse ratio: if cache hit is high, each prefill KV$ write
            # is used many times in decode before eviction
            reuse_ratio = round(hit_prefix / max(1, 100 - hit_prefix), 2)

            # Miss cost: fraction of KV$ that had to be fetched from SSD
            miss_rate = max(0.0, 100.0 - hit_prefix)

            derived.append({
                self.pivot:                _f(r, self.pivot),
                "kv_cache_hit_pct":        round(hit_prefix, 1),
                "kv_rt_hit_pct":           round(hit_rt, 1),
                "miss_rate_pct":           round(miss_rate, 1),
                "l1_hbm_pct":              l1_pct,
                "l2_dram_pct":             l2_pct,
                "l3_ssd_pct":              l3_pct,
                "total_tokens_M":          round(total_tok / 1e6, 3),
                "pool_fill_pct":           pool_fill,
                "evicted_gb_per_instance": round(evicted_gb, 3),
                "reuse_ratio":             reuse_ratio,
                "locality_score": (
                    "excellent" if l3_pct < 5   and hit_prefix > 50 else
                    "good"      if l3_pct < 20  and hit_prefix > 20 else
                    "poor"      if l3_pct > 40                       else
                    "moderate"
                ),
            })

        peak_l3 = max((d["l3_ssd_pct"] for d in derived), default=0.0)
        peak_hit = max((d["kv_cache_hit_pct"] for d in derived), default=0.0)
        mean_miss = statistics.mean(d["miss_rate_pct"] for d in derived) if derived else 0.0

        observations, warnings = [], []
        observations.append(f"Peak RadixAttention cache hit rate: {peak_hit:.1f}%.")
        observations.append(f"Peak L3 (SSD) token fraction: {peak_l3:.1f}% of active KV$ tokens.")
        if peak_hit < 5 and len(self.rows) > 1:
            observations.append("Very low cache hit rate — single-turn workload or diverse problem set (no prefix reuse).")
        if peak_l3 > 30:
            warnings.append(f"L3 Storage holds {peak_l3:.0f}% of tokens — cold-tier fetch dominates. "
                             "Increase HBM pool or reduce concurrency.")
        if mean_miss > 50:
            warnings.append(f"Mean miss rate {mean_miss:.1f}% — most prefill tokens are computed rather than cached. "
                             "Consider prompt prefix caching strategies.")

        return ScenarioResult(
            scenario_id="S4", scenario_name="IO Locality",
            description="KV$ cache hit rate and L1/L2/L3 tier distribution — working set fit in fast memory.",
            rows=derived,
            summary={"peak_cache_hit_pct": round(peak_hit, 1),
                     "peak_l3_ssd_pct": round(peak_l3, 1),
                     "mean_miss_rate_pct": round(mean_miss, 1)},
            observations=observations, warnings=warnings)

    # ── S5: IO Frequency ──────────────────────────────────────────────────────
    def s5_io_frequency(self) -> ScenarioResult:
        """
        IOPS, request rate, and engine utilisation.

        IO frequency determines whether the SSD is bandwidth-bound or
        IOPS-bound.  KV$ restore is IOPS-intensive (many small random reads).

        Derived:
          total_iops     = read_iops + write_iops
          iops_per_req   = total_iops / throughput_req_s
          engine_util    = utilization_mean (SGLang)
          prefill_util   = prefill_tok_s / (prefill_tok_s + decode_tok_s)
        """
        derived = []
        for r in self.rows:
            rd_iops = _f(r, "ssd_read_iops_mean")
            wr_iops = _f(r, "ssd_write_iops_mean")
            req_s   = _f(r, "throughput_req_s")
            tok_s   = _f(r, "throughput_tok_s")
            pf_tok  = _f(r, "ai_op_prefill_tok_s")
            dc_tok  = _f(r, "ai_op_decode_tok_s")
            util    = _f(r, "utilization_mean")
            q_peak  = _f(r, "num_queue_reqs_peak_mean")
            inflight= _f(r, "nvme_inflight_mean")

            total_iops  = rd_iops + wr_iops
            iops_per_req = round(total_iops / req_s, 1) if req_s > 0 else 0.0
            total_ai     = pf_tok + dc_tok
            pf_util_pct  = round(pf_tok / total_ai * 100, 1) if total_ai > 0 else 0.0

            # IOPS-bound vs BW-bound: if avg IO size < 64 KB the device is IOPS-limited
            rd_bw = _f(r, "ssd_read_bw_mb_mean")
            avg_io_kb = round(rd_bw * 1024 / rd_iops, 1) if rd_iops > 0 else 0.0
            bound = "iops_bound" if 0 < avg_io_kb < 64 else ("bw_bound" if avg_io_kb >= 64 else "idle")

            derived.append({
                self.pivot:         _f(r, self.pivot),
                "total_iops":       round(total_iops, 0),
                "read_iops":        round(rd_iops, 0),
                "write_iops":       round(wr_iops, 0),
                "iops_per_request": iops_per_req,
                "throughput_req_s": round(req_s, 3),
                "throughput_tok_s": round(tok_s, 1),
                "engine_util":      round(util, 3),
                "prefill_util_pct": pf_util_pct,
                "queue_peak_reqs":  round(q_peak, 1),
                "nvme_inflight":    round(inflight, 1),
                "avg_io_kb":        avg_io_kb,
                "bound_type":       bound,
            })

        max_iops = max((d["total_iops"] for d in derived), default=0.0)
        peak_util = max((d["engine_util"] for d in derived), default=0.0)
        bounds = [d["bound_type"] for d in derived if d["bound_type"] != "idle"]
        dominant_bound = max(set(bounds), key=bounds.count) if bounds else "idle"

        observations, warnings = [], []
        observations.append(f"Peak total SSD IOPS: {max_iops:,.0f}.")
        observations.append(f"Peak engine utilisation: {peak_util:.1%}.")
        observations.append(f"Dominant IO bottleneck type: {dominant_bound}.")
        if dominant_bound == "iops_bound":
            warnings.append("IOPS-bound workload — random 4K–64K reads are saturating the NVMe controller's queue depth.")

        return ScenarioResult(
            scenario_id="S5", scenario_name="IO Frequency",
            description="IOPS, request rate, SGLang engine utilisation, and IOPS/BW bottleneck classification.",
            rows=derived,
            summary={"max_total_iops": round(max_iops, 0), "peak_engine_util": round(peak_util, 3),
                     "dominant_bound": dominant_bound},
            observations=observations, warnings=warnings)

    # ── S6: Concurrency Effect ────────────────────────────────────────────────
    def s6_concurrency(self) -> ScenarioResult:
        """
        How every tier responds to increasing inference_concurrency.

        Finds:
          - overflow point (concurrency where SSD I/O begins)
          - throughput peak (concurrency of max tok/s)
          - latency inflection (concurrency where P99 TPOT growth accelerates)
        """
        derived = []
        baseline_tpot = None
        for r in self.rows:
            conc   = _f(r, self.pivot)
            tpot   = _f(r, "tpot_mean_ms")
            tpot_p99 = _f(r, "tpot_p99_ms")
            if baseline_tpot is None and tpot > 0:
                baseline_tpot = tpot
            miss   = _f(r, "kv_miss_penalty_ms")
            kv_fill= _f(r, "kv_fill_pct_mean")
            hbm_gb = _f(r, "hbm_used_gb_mean")
            dram_gb= _f(r, "dram_hicache_staging_gb_mean")
            rd_bw  = _f(r, "ssd_read_bw_mb_mean")
            wr_bw  = _f(r, "ssd_write_bw_mb_mean")
            tok_s  = _f(r, "throughput_tok_s")
            overflow_at = _f(r, "kv_overflow_at")

            tpot_growth_pct = round((tpot / baseline_tpot - 1) * 100, 1) if baseline_tpot else 0.0

            derived.append({
                self.pivot:             conc,
                "kv_fill_pct":          round(kv_fill, 1),
                "ssd_active":           kv_fill >= 98,
                "hbm_used_gb":          round(hbm_gb, 1),
                "dram_staging_gb":      round(dram_gb, 3),
                "ssd_read_bw_mb_s":     round(rd_bw, 1),
                "ssd_write_bw_mb_s":    round(wr_bw, 1),
                "tpot_mean_ms":         round(tpot, 1),
                "tpot_p99_ms":          round(tpot_p99, 1),
                "kv_miss_penalty_ms":   round(miss, 1),
                "tpot_growth_pct":      tpot_growth_pct,
                "throughput_tok_s":     round(tok_s, 1),
                "overflow_at":          int(overflow_at) if overflow_at else "?",
                "phase": (
                    "below_overflow"  if kv_fill < 95  else
                    "at_overflow"     if kv_fill < 110 else
                    "heavy_eviction"  if kv_fill < 200 else
                    "saturated"
                ),
            })

        overflow_rows = [d for d in derived if d["ssd_active"]]
        overflow_conc = overflow_rows[0][self.pivot] if overflow_rows else None
        peak_tok_row  = max(derived, key=lambda d: d["throughput_tok_s"]) if derived else {}
        peak_tok_conc = peak_tok_row.get(self.pivot)
        peak_tok_s    = peak_tok_row.get("throughput_tok_s", 0)
        max_miss      = max((d["kv_miss_penalty_ms"] for d in derived), default=0.0)

        observations, warnings = [], []
        if overflow_conc:
            observations.append(f"SSD I/O begins at {self.pivot}={overflow_conc} (KV$ pool overflow).")
        else:
            observations.append("KV$ pool does not overflow across tested concurrency range.")
        if peak_tok_conc:
            observations.append(
                f"Throughput peak: {peak_tok_s:.1f} tok/s at {self.pivot}={peak_tok_conc}.")
        if max_miss > 0:
            observations.append(f"Maximum TPOT miss penalty: {max_miss:.1f} ms ({_format_pct(max_miss, _f(derived[-1], 'tpot_p99_ms'))} of final P99 TPOT).")
        if overflow_conc and peak_tok_conc and float(peak_tok_conc) > float(overflow_conc):
            observations.append("Throughput continues rising past overflow — parallelism benefit outweighs latency cost.")
        elif overflow_conc and peak_tok_conc and float(peak_tok_conc) <= float(overflow_conc):
            warnings.append("Throughput peaks before overflow — high concurrency hurts more than it helps.")

        return ScenarioResult(
            scenario_id="S6", scenario_name="Concurrency Effect",
            description="All Memory and SSD tier metrics as function of inference_concurrency.",
            rows=derived,
            summary={"ssd_io_starts_at": overflow_conc, "throughput_peak_at": peak_tok_conc,
                     "peak_tok_s": round(peak_tok_s, 1), "max_miss_penalty_ms": round(max_miss, 1)},
            observations=observations, warnings=warnings)

    # ── S7: Sustainability & Endurance ────────────────────────────────────────
    def s7_sustainability(self) -> ScenarioResult:
        """
        How long the SSD can sustain this workload before rated endurance is
        reached.

        Key metrics:
          waf              — write amplification (NAND written / host written)
          host_written_gb  — cumulative host bytes written this run
          nand_written_gb  — actual NAND wear
          ssd_lifetime_tbw — device rated TBW endurance
          ssd_dwpd_est     — drive writes per day at current rate

        Derived:
          lifetime_remaining_pct = (1 - nand_written / (ssd_lifetime_tbw*1024)) * 100
          write_rate_gb_s        = kv_eviction_mb_s / 1024  (per second)
          days_to_exhaust        = lifetime_remaining_gb / write_rate_gb_s / 86400
        """
        T = self.THRESHOLDS
        derived = []
        for r in self.rows:
            waf        = _f(r, "waf")
            host_gb    = _f(r, "host_written_gb")
            nand_gb    = _f(r, "nand_written_gb")
            tbw_tb     = _f(r, "ssd_lifetime_tbw")
            tbw_gb     = tbw_tb * 1024 if tbw_tb > 0 else 0.0
            dwpd       = _f(r, "ssd_dwpd_est")
            temp       = _f(r, "temp_peak_c")
            evict_mb_s = _f(r, "kv_eviction_mb_s")
            bio_p999   = _f(r, "bio_lat_p999_us_mean")
            ssd_util   = _f(r, "ssd_util_pct_mean")

            life_rem_pct = round((1 - nand_gb / tbw_gb) * 100, 2) if tbw_gb > 0 else None
            write_rate_gb_s = evict_mb_s / 1024
            # Seconds until TBW exhausted at current write rate (accounting for WAF)
            effective_rate = write_rate_gb_s * max(waf, 1.0)
            days_to_exhaust = None
            if tbw_gb > 0 and effective_rate > 0 and nand_gb < tbw_gb:
                remaining_gb = tbw_gb - nand_gb
                days_to_exhaust = round(remaining_gb / effective_rate / 86400, 1)

            derived.append({
                self.pivot:              _f(r, self.pivot),
                "waf":                   round(waf, 3),
                "host_written_gb":       round(host_gb, 3),
                "nand_written_gb":       round(nand_gb, 3),
                "tbw_rated_tb":          round(tbw_tb, 1),
                "lifetime_remaining_pct": life_rem_pct,
                "days_to_tbw_exhaust":   days_to_exhaust,
                "ssd_dwpd_est":          round(dwpd, 2),
                "temp_peak_c":           int(temp),
                "eviction_rate_mb_s":    round(evict_mb_s, 2),
                "ssd_util_pct":          round(ssd_util, 1),
                "bio_lat_p999_us":       round(bio_p999, 1),
                "endurance_status": (
                    "critical" if (life_rem_pct is not None and life_rem_pct < T["lifetime_pct_warn"])
                               or waf > T["waf_crit"]
                    else "warning" if waf > T["waf_warn"] or temp >= T["temp_warn_c"]
                    else "ok"
                ),
            })

        max_waf  = max((d["waf"] for d in derived), default=0.0)
        max_temp = max((d["temp_peak_c"] for d in derived), default=0)
        min_life = min((d["lifetime_remaining_pct"] for d in derived
                        if d["lifetime_remaining_pct"] is not None), default=None)
        min_days = min((d["days_to_tbw_exhaust"] for d in derived
                        if d["days_to_tbw_exhaust"] is not None), default=None)

        observations, warnings = [], []
        if max_waf > 0:
            observations.append(f"Peak WAF: {max_waf:.2f} — every GB written by host causes {max_waf:.2f} GB NAND wear.")
        if min_life is not None:
            observations.append(f"Minimum remaining endurance: {min_life:.1f}% of rated TBW.")
        if min_days is not None:
            observations.append(f"At peak write rate, TBW exhausted in {min_days:.0f} days.")
        if max_waf > T["waf_crit"]:
            warnings.append(f"Critical WAF {max_waf:.2f} — SSD will wear out {max_waf:.1f}x faster than rated.")
        elif max_waf > T["waf_warn"]:
            warnings.append(f"Elevated WAF {max_waf:.2f} (threshold {T['waf_warn']}) — SSD wearing faster than rated. Monitor closely.")
        if max_temp >= T["temp_crit_c"]:
            warnings.append(f"Critical SSD temperature {max_temp}°C — thermal throttling imminent.")
        elif max_temp >= T["temp_warn_c"]:
            warnings.append(f"High SSD temperature {max_temp}°C — monitor for throttling.")

        return ScenarioResult(
            scenario_id="S7", scenario_name="Sustainability & Endurance",
            description="WAF, TBW consumption rate, DWPD, temperature, and projected lifetime.",
            rows=derived,
            summary={"max_waf": round(max_waf, 3), "max_temp_c": max_temp,
                     "min_lifetime_pct": min_life, "min_days_to_tbw": min_days},
            observations=observations, warnings=warnings)

    # ── S8: Device Degradation ────────────────────────────────────────────────
    def s8_degradation(self) -> ScenarioResult:
        """
        Correlation between accumulated wear (host_written_gb, nand_written_gb,
        WAF) and observed latency/performance metrics.

        In a single run the wear accumulated is small, so this scenario is most
        useful when comparing multiple runs from different time periods or when
        a long endurance run is analyzed.  Within a single concurrency sweep,
        rows are sorted by host_written_gb to show the latency trend.

        Degradation signature:
          - Increasing bio_lat_p99/p999 as wear accumulates
          - WAF creep (SLC cache drain → going direct to TLC/QLC)
          - Rising ssd_r_await_p99 as WA forces more read retries
          - TPOT P99 degrading as SSD restore latency worsens
        """
        # Sort by host_written_gb so wear is the x-axis
        sorted_rows = sorted(self.rows, key=lambda r: _f(r, "host_written_gb"))

        derived = []
        base_bio_p99  = None
        base_ssd_lat  = None
        base_tpot_p99 = None
        for r in sorted_rows:
            host_gb  = _f(r, "host_written_gb")
            nand_gb  = _f(r, "nand_written_gb")
            waf      = _f(r, "waf")
            temp     = _f(r, "temp_peak_c")
            bio_p99  = _f(r, "bio_lat_p99_us_mean")
            bio_p999 = _f(r, "bio_lat_p999_us_mean")
            ssd_lat  = _f(r, "ssd_r_await_p99_ms")
            miss_pen = _f(r, "kv_miss_penalty_ms")
            tpot_p99 = _f(r, "tpot_p99_ms")
            tbw_tb   = _f(r, "ssd_lifetime_tbw")
            wear_pct = round(nand_gb / (tbw_tb * 1024) * 100, 4) if tbw_tb > 0 else 0.0

            if base_bio_p99  is None and bio_p99  > 0: base_bio_p99  = bio_p99
            if base_ssd_lat  is None and ssd_lat  > 0: base_ssd_lat  = ssd_lat
            if base_tpot_p99 is None and tpot_p99 > 0: base_tpot_p99 = tpot_p99

            bio_degradation_pct  = round((bio_p99  / base_bio_p99  - 1) * 100, 1) if base_bio_p99  and bio_p99  else 0.0
            ssd_degradation_pct  = round((ssd_lat  / base_ssd_lat  - 1) * 100, 1) if base_ssd_lat  and ssd_lat  else 0.0
            tpot_degradation_pct = round((tpot_p99 / base_tpot_p99 - 1) * 100, 1) if base_tpot_p99 and tpot_p99 else 0.0

            derived.append({
                self.pivot:                   _f(r, self.pivot),
                "host_written_gb":            round(host_gb, 3),
                "nand_written_gb":            round(nand_gb, 3),
                "wear_pct_of_tbw":            wear_pct,
                "waf":                        round(waf, 3),
                "temp_c":                     int(temp),
                "bio_lat_p99_us":             round(bio_p99, 1),
                "bio_lat_p999_us":            round(bio_p999, 1),
                "ssd_r_await_p99_ms":         round(ssd_lat, 3),
                "kv_miss_penalty_ms":         round(miss_pen, 1),
                "tpot_p99_ms":                round(tpot_p99, 1),
                "bio_degradation_pct":        bio_degradation_pct,
                "ssd_lat_degradation_pct":    ssd_degradation_pct,
                "tpot_degradation_pct":       tpot_degradation_pct,
                "degradation_visible": (
                    bio_degradation_pct > 10 or
                    ssd_degradation_pct > 10 or
                    tpot_degradation_pct > 5
                ),
            })

        max_bio_deg  = max((d["bio_degradation_pct"] for d in derived), default=0.0)
        max_tpot_deg = max((d["tpot_degradation_pct"] for d in derived), default=0.0)
        max_host_gb  = max((d["host_written_gb"] for d in derived), default=0.0)
        any_degraded = any(d["degradation_visible"] for d in derived)

        observations, warnings = [], []
        observations.append(f"Total host bytes written this run: {max_host_gb:.2f} GB.")
        if any_degraded:
            observations.append(f"Latency degradation detected: bio P99 +{max_bio_deg:.1f}%, TPOT P99 +{max_tpot_deg:.1f}%.")
            warnings.append("Performance degradation correlated with write accumulation — possible SLC cache drain.")
        else:
            observations.append("No significant latency degradation within this run's wear range.")
            observations.append("For degradation trends, compare multiple runs at different lifetime wear levels.")

        return ScenarioResult(
            scenario_id="S8", scenario_name="Device Degradation",
            description="Latency and WAF trends correlated with accumulated SSD wear.",
            rows=derived,
            summary={"max_host_written_gb": round(max_host_gb, 2),
                     "max_bio_p99_degradation_pct": round(max_bio_deg, 1),
                     "max_tpot_degradation_pct": round(max_tpot_deg, 1),
                     "degradation_detected": any_degraded},
            observations=observations, warnings=warnings)

    # ── Report rendering ──────────────────────────────────────────────────────
    def print_report(self, report: dict[str, ScenarioResult], width: int = 120):
        """Print all 8 scenarios to stdout in a readable tabular format."""
        print(_c("═" * width, _W))
        print(_c("  AMOprof — Storage & Memory Scenario Analysis", _W))
        if self.source_path:
            print(_c(f"  Source: {self.source_path}", _B))
        print(_c(f"  Pivot:  {self.pivot}  |  Rows: {len(self.rows)}", _B))
        print(_c("═" * width, _W))

        for sc_id, res in report.items():
            self._print_scenario(res, width)

    def _print_scenario(self, res: ScenarioResult, width: int):
        bar = "─" * width
        print(f"\n{_c(bar, _B)}")
        print(_c(f"  {res.scenario_id}  {res.scenario_name}", _W))
        print(f"  {res.description}")
        print(_c(bar, _B))

        if res.rows:
            _print_table(res.rows, max_col_w=18, pivot=self.pivot)

        print()
        print(_c("  Summary:", _W))
        for k, v in res.summary.items():
            print(f"    {k:<40} {v}")

        if res.observations:
            print()
            print(_c("  Observations:", _W))
            for obs in res.observations:
                print(f"    • {obs}")

        if res.warnings:
            print()
            print(_c("  Warnings:", _Y))
            for w in res.warnings:
                print(f"    {_c('⚠', _Y)}  {w}")

    def write_csv(self, report: dict[str, ScenarioResult], out_dir: str | Path):
        """Write one CSV per scenario into out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for sc_id, res in report.items():
            if not res.rows:
                continue
            fname = out_dir / f"{sc_id}_{res.scenario_name.lower().replace(' ', '_').replace('&','and')}.csv"
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res.rows[0].keys()))
                w.writeheader()
                w.writerows(res.rows)
            written.append(str(fname))
        return written

    def write_summary_csv(self, report: dict[str, ScenarioResult], path: str | Path):
        """One-row-per-scenario summary CSV."""
        path = Path(path)
        rows = []
        for sc_id, res in report.items():
            row = {"scenario_id": sc_id, "scenario_name": res.scenario_name}
            row.update({f"summary_{k}": v for k, v in res.summary.items()})
            row["warnings"] = " | ".join(res.warnings)
            rows.append(row)
        if not rows:
            return
        all_keys = list({k: None for row in rows for k in row}.keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in all_keys})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_num(v: str) -> float | int | str:
    """Convert CSV string to number where possible."""
    try:
        f = float(v)
        return int(f) if f == int(f) and abs(f) < 1e9 else f
    except (ValueError, TypeError):
        return v


def _format_pct(part: float, total: float) -> str:
    if total > 0:
        return f"{part / total * 100:.1f}%"
    return "N/A"


def _print_table(rows: list[dict], max_col_w: int = 18, pivot: str = "inference_concurrency"):
    if not rows:
        return
    keys = list(rows[0].keys())
    # Always put pivot first
    if pivot in keys:
        keys = [pivot] + [k for k in keys if k != pivot]

    # Compute column widths
    widths = {}
    for k in keys:
        w = min(max(len(str(k)), max((len(str(r.get(k, ""))) for r in rows), default=0)), max_col_w)
        widths[k] = w

    # Header
    header = "  " + "  ".join(str(k)[:widths[k]].ljust(widths[k]) for k in keys)
    print(_c(header, _B))
    print("  " + "  ".join("-" * widths[k] for k in keys))

    for row in rows:
        cells = []
        for k in keys:
            v = row.get(k, "")
            s = str(v)[:widths[k]].ljust(widths[k])
            # Colour coding for key fields
            if k in ("endurance_status", "latency_regime", "bound_type", "locality_score"):
                if v in ("critical", "saturated", "iops_bound"):      s = _c(s, _R)
                elif v in ("warning", "heavy_eviction", "bw_bound"):  s = _c(s, _Y)
                elif v in ("ok", "good", "excellent", "no_ssd_io"):   s = _c(s, _G)
            cells.append(s)
        print("  " + "  ".join(cells))


# ════════════════════════════════════════════════════════════════════════════
# Extended analyser — S9, S10, S11
# ════════════════════════════════════════════════════════════════════════════

class ExtendedScenarioAnalyzer(ScenarioAnalyzer):
    """
    Adds three scenarios beyond the base eight:
      S9  AI Op Phase Classification  — prefill vs decode metric attribution
      S10 Batch Size Effect           — burst batch vs steady-state
      S11 Context Length Effect       — KV$/request scaling across context lengths
    """

    def full_report(self) -> dict[str, ScenarioResult]:
        report = super().full_report()
        report["S9"]  = self.s9_ai_op_phase()
        report["S10"] = self.s10_batch_effect()
        report["S11"] = self.s11_context_effect()
        return report

    # ── S9: AI Op Phase Classification ───────────────────────────────────────
    def s9_ai_op_phase(self) -> ScenarioResult:
        """
        Separate Memory and SSD metrics by AI operation phase.

        Every inference request passes through two phases:

          PREFILL  — reads prompt tokens, computes attention K and V, writes
                     KV tensors to HBM pool.  This is the KV$ WRITE path.
                     If the pool is full, old KV$ blocks are evicted to SSD
                     (write I/O triggered by prefill overflow).

          DECODE   — generates output tokens one at a time.  Each token
                     re-reads the ENTIRE context KV$ to compute attention.
                     This is the KV$ READ path.  If any block was evicted
                     during prefill, decode must restore it from SSD first
                     (read I/O triggered by decode miss).

        Derived KPIs:
          pf_kv_write_gb     = rt_prefill_compute_tokens × kv_bytes_per_token
          pf_ssd_eviction_gb = kv_evicted_tokens × kv_bytes_per_token
          pf_cache_hit_pct   = rt_prefill_cache / (rt_prefill_cache + rt_prefill_compute)
          dc_kv_read_per_step_gb = context_len × kv_bytes_per_token   (theoretical)
          dc_ssd_restore_gb  = kv_restored_tokens × kv_bytes_per_token
          dc_miss_rate_pct   = dc_ssd_restore_gb / (dc_kv_read_per_step_gb × dc_tokens) × 100
        """
        derived = []
        for r in self.rows:
            pv = _f(r, self.pivot)
            bpt = max(_f(r, "kv_bytes_per_token"), 1)

            # Prefill
            pf_compute = _f(r, "pf_rt_compute_tokens_mean")
            pf_cache   = _f(r, "pf_rt_cache_tokens_mean")
            pf_total   = pf_compute + pf_cache
            pf_kv_write_gb   = round(pf_compute * bpt / (1024**3), 4)
            pf_evict_gb      = _f(r, "pf_ssd_eviction_gb_mean")
            pf_hit_pct       = _f(r, "pf_cache_hit_pct")
            pf_hbm_delta     = _f(r, "pf_hbm_delta_gb_mean")
            pf_ssd_evict_bw  = _f(r, "pf_ssd_eviction_bw_mb_s") if "pf_ssd_eviction_bw_mb_s" in r else _f(r, "ssd_write_bw_mb_mean")

            # Decode
            dc_tokens        = _f(r, "dc_rt_decode_tokens_mean")
            dc_restore_gb    = _f(r, "dc_ssd_restore_gb_mean")
            dc_read_step_gb  = _f(r, "dc_kv_read_per_step_gb")
            dc_miss_pen      = _f(r, "dc_miss_penalty_ms") or _f(r, "kv_miss_penalty_ms")
            dc_hbm_delta     = _f(r, "dc_hbm_delta_gb_mean")
            dc_restore_bw    = _f(r, "dc_ssd_restore_bw_mb_s") if "dc_ssd_restore_bw_mb_s" in r else _f(r, "ssd_read_bw_mb_mean")

            # Total theoretical KV$ reads during decode
            dc_total_read_gb = round(dc_read_step_gb * max(dc_tokens, 1), 4)
            dc_miss_rate_pct = round(dc_restore_gb / max(dc_total_read_gb, 0.001) * 100, 2) if dc_total_read_gb > 0 else 0.0

            # Op mix
            n_total = sum([
                _f(r, "op_prefill_count"), _f(r, "op_decode_count"),
                _f(r, "op_reasoning_count"), _f(r, "op_mixed_count")
            ]) or 1
            pf_pct = round(_f(r, "op_prefill_count")   / n_total * 100, 1)
            dc_pct = round(_f(r, "op_decode_count")    / n_total * 100, 1)
            rs_pct = round(_f(r, "op_reasoning_count") / n_total * 100, 1)
            mx_pct = round(_f(r, "op_mixed_count")     / n_total * 100, 1)

            # Prefill/decode throughput ratio
            pf_tok_s = _f(r, "op_prefill_tok_s_mean") or _f(r, "ai_op_prefill_tok_s")
            dc_tok_s = _f(r, "op_decode_tok_s_mean")  or _f(r, "ai_op_decode_tok_s")
            ratio    = round(pf_tok_s / dc_tok_s, 2) if dc_tok_s > 0 else 0.0

            derived.append({
                self.pivot:               pv,
                # Op distribution
                "op_prefill_pct":         pf_pct,
                "op_decode_pct":          dc_pct,
                "op_reasoning_pct":       rs_pct,
                "op_mixed_pct":           mx_pct,
                # Prefill KV$ write path
                "pf_kv_written_gb":       round(pf_kv_write_gb, 4),
                "pf_cache_hit_pct":       round(pf_hit_pct, 1),
                "pf_hbm_delta_gb":        round(pf_hbm_delta, 3),
                "pf_ssd_eviction_gb":     round(pf_evict_gb, 4),
                "pf_ssd_eviction_bw_mb_s":round(pf_ssd_evict_bw, 2),
                # Decode KV$ read path
                "dc_tokens_mean":         round(dc_tokens, 1),
                "dc_kv_read_per_step_gb": round(dc_read_step_gb, 3),
                "dc_total_kv_read_gb":    round(dc_total_read_gb, 3),
                "dc_ssd_restore_gb":      round(dc_restore_gb, 4),
                "dc_ssd_restore_bw_mb_s": round(dc_restore_bw, 2),
                "dc_miss_rate_pct":       dc_miss_rate_pct,
                "dc_miss_penalty_ms":     round(dc_miss_pen, 2),
                "dc_hbm_delta_gb":        round(dc_hbm_delta, 3),
                # Throughput balance
                "pf_tok_s":               round(pf_tok_s, 1),
                "dc_tok_s":               round(dc_tok_s, 1),
                "pf_dc_ratio":            ratio,
                "bottleneck": _classify_bottleneck(
                    pf_tok_s, dc_tok_s,
                    _f(r, "pf_hbm_delta_gb"),
                    _f(r, "dc_miss_penalty_ms") or _f(r, "kv_miss_penalty_ms"),
                    _f(r, "dc_tokens_mean") or _f(r, "output_tokens_mean"),
                ),
            })

        # Summary
        max_pf_evict  = max((d["pf_ssd_eviction_gb"] for d in derived), default=0.0)
        max_dc_restore= max((d["dc_ssd_restore_gb"]   for d in derived), default=0.0)
        max_dc_miss   = max((d["dc_miss_rate_pct"]    for d in derived), default=0.0)
        mean_pf_hit   = _smean([d["pf_cache_hit_pct"] for d in derived])
        bottlenecks   = [d["bottleneck"] for d in derived]
        dom_bottleneck= max(set(bottlenecks), key=bottlenecks.count) if bottlenecks else "unknown"

        observations, warnings = [], []
        observations.append(
            f"Dominant bottleneck: {dom_bottleneck}  "
            + _ratio_range(derived, "pf_dc_ratio"))
        observations.append(
            f"Mean prefill KV$ cache hit rate: {mean_pf_hit:.1f}% "
            f"— {'high reuse (multi-turn or shared prefix)' if mean_pf_hit > 30 else 'low reuse (single-turn diverse prompts)'}.")
        observations.append(
            f"Peak prefill SSD eviction: {max_pf_evict:.4f} GB/instance — "
            f"{'pool overflowing during prefill' if max_pf_evict > 0.01 else 'pool fits all KV$ during prefill'}.")
        observations.append(
            f"Peak decode SSD restore: {max_dc_restore:.4f} GB/instance — "
            f"{'KV$ misses stalling decode' if max_dc_restore > 0.01 else 'all decode KV$ served from HBM'}.")
        if max_dc_miss > 1.0:
            warnings.append(
                f"Decode KV$ miss rate {max_dc_miss:.1f}% — {max_dc_miss:.0f}% of decode attention reads "
                f"must fetch from SSD. This directly causes kv_miss_penalty_ms inflation.")
        if dom_bottleneck == "prefill_bound":
            warnings.append(
                "Prefill-bound: prefill throughput << decode throughput. "
                "Each new request stalls the decode pipeline while its KV$ is computed. "
                "Consider chunked-prefill or increased HBM mem_fraction.")

        return ScenarioResult(
            scenario_id="S9", scenario_name="AI Op Phase Classification",
            description=(
                "Prefill (KV$ write) vs Decode (KV$ read) metric attribution. "
                "Shows which AI operation drives Memory and SSD pressure."
            ),
            rows=derived,
            summary={
                "dominant_bottleneck":       dom_bottleneck,
                "mean_pf_cache_hit_pct":     round(mean_pf_hit, 1),
                "peak_pf_ssd_eviction_gb":   round(max_pf_evict, 4),
                "peak_dc_ssd_restore_gb":    round(max_dc_restore, 4),
                "peak_dc_miss_rate_pct":     round(max_dc_miss, 2),
            },
            observations=observations, warnings=warnings)

    # ── S10: Batch Size Effect ────────────────────────────────────────────────
    def s10_batch_effect(self) -> ScenarioResult:
        """
        How burst batch size affects Memory and SSD tiers.

        Works on batch_sweep.csv (pivot = batch_size) OR on a concurrency
        sweep interpreted as batch-mode (each concurrency level treated as
        a synchronous burst batch of that size).

        Key question: does the HBM KV$ pool overflow during a batch burst?
          pool_fill_at_burst_pct = batch_size × kv_per_request_gb / pool_gb × 100
          overflow_during_burst  = pool_fill_at_burst_pct > 100

        At small batch sizes the burst fits in HBM entirely (no SSD I/O).
        At large batch sizes the burst overflows — SSD sees burst writes
        during prefill and burst reads during decode, then silence between
        batches (bursty rather than continuous I/O pattern).
        """
        # Detect pivot: batch_size (batch_sweep.csv) or inference_concurrency
        pivot_col = "batch_size" if any("batch_size" in str(r) for r in self.rows) else self.pivot

        derived = []
        for r in self.rows:
            pv      = _f(r, pivot_col)
            kv_req  = _f(r, "kv_per_request_gb")
            pool_gb = _f(r, "kv_pool_capacity_gb")
            fill_pct= _f(r, "pool_fill_at_batch_pct") or (
                round(pv * kv_req / max(pool_gb, 0.001) * 100, 1) if pool_gb > 0 else 0.0)
            overflows = _f(r, "batch_overflows_pool") or (fill_pct > 100)

            hbm_peak   = _f(r, "hbm_peak_gb_mean")  or _f(r, "hbm_used_gb_mean")
            hbm_steady = _f(r, "hbm_used_gb_mean")
            hbm_burst_overhead = round(hbm_peak - hbm_steady, 2) if hbm_peak > hbm_steady else 0.0

            batch_wall  = _f(r, "batch_wall_time_s_mean")
            rd_bw       = _f(r, "ssd_read_bw_mb_mean")
            wr_bw       = _f(r, "ssd_write_bw_mb_mean")
            ssd_active  = (rd_bw + wr_bw) > 0.5

            # Effective throughput per batch slot
            tpot        = _f(r, "tpot_mean_ms")
            tok_s       = _f(r, "throughput_tok_s")
            ttft        = _f(r, "ttft_mean_ms")

            derived.append({
                pivot_col:                pv,
                "burst_pool_fill_pct":    round(fill_pct, 1),
                "batch_overflows_hbm":    bool(overflows),
                "hbm_peak_gb":            round(hbm_peak, 2),
                "hbm_steady_state_gb":    round(hbm_steady, 2),
                "hbm_burst_overhead_gb":  hbm_burst_overhead,
                "dram_staging_gb":        round(_f(r, "dram_hicache_staging_gb_mean"), 3),
                "ssd_read_bw_mb_s":       round(rd_bw, 2),
                "ssd_write_bw_mb_s":      round(wr_bw, 2),
                "ssd_active":             ssd_active,
                "bio_lat_p99_us":         round(_f(r, "bio_lat_p99_us_mean"), 1),
                "ttft_mean_ms":           round(ttft, 1),
                "tpot_mean_ms":           round(tpot, 1),
                "kv_miss_penalty_ms":     round(_f(r, "kv_miss_penalty_ms"), 2),
                "throughput_tok_s":       round(tok_s, 1),
                "batch_wall_s":           round(batch_wall, 2),
                # Phase breakdown
                "pf_ssd_eviction_gb":     round(_f(r, "pf_ssd_eviction_gb_mean"), 4),
                "dc_ssd_restore_gb":      round(_f(r, "dc_ssd_restore_gb_mean"), 4),
                "io_pattern": (
                    "idle"              if not ssd_active else
                    "burst_then_idle"   if batch_wall > 0 else
                    "continuous"
                ),
            })

        overflow_sizes = [d[pivot_col] for d in derived if d["batch_overflows_hbm"]]
        first_overflow = overflow_sizes[0] if overflow_sizes else None
        peak_tok_row   = max(derived, key=lambda d: d["throughput_tok_s"]) if derived else {}
        peak_tok_bs    = peak_tok_row.get(pivot_col)

        observations, warnings = [], []
        if first_overflow:
            observations.append(
                f"HBM KV$ pool first overflows at batch_size={first_overflow} "
                f"— SSD I/O begins here.")
        else:
            observations.append(
                "All tested batch sizes fit within the HBM KV$ pool — no SSD I/O during burst.")
        if peak_tok_bs:
            observations.append(
                f"Throughput peak at batch_size={peak_tok_bs} "
                f"({peak_tok_row.get('throughput_tok_s', 0):.1f} tok/s).")

        burst_rows = [d for d in derived if d["batch_overflows_hbm"]]
        if burst_rows:
            max_burst_ssd = max((d["ssd_write_bw_mb_s"] + d["ssd_read_bw_mb_s"] for d in burst_rows), default=0.0)
            observations.append(
                f"Peak burst SSD BW: {max_burst_ssd:.1f} MB/s "
                f"(bursty pattern — high peak, idle between batches).")
            if max_burst_ssd > 100:
                warnings.append(
                    f"Peak burst SSD BW {max_burst_ssd:.1f} MB/s may saturate the device. "
                    f"Consider smaller batch_size or longer context to spread I/O.")

        return ScenarioResult(
            scenario_id="S10", scenario_name="Batch Size Effect",
            description=(
                "How burst batch size affects HBM peak fill, SSD I/O onset, "
                "and throughput. Burst batches create bursty SSD I/O vs continuous."
            ),
            rows=derived,
            summary={
                "first_overflow_at_batch": first_overflow,
                "peak_throughput_batch":   peak_tok_bs,
                "peak_tok_s":              round(peak_tok_row.get("throughput_tok_s", 0), 1),
            },
            observations=observations, warnings=warnings)

    # ── S11: Context Length Effect ────────────────────────────────────────────
    def s11_context_effect(self) -> ScenarioResult:
        """
        How context length changes KV$/request and therefore overflow_concurrency.

        kv_per_request_gb = kv_bytes_per_token × context_len / 1024³

        This is a linear relationship: doubling context doubles KV$/request.
        Since overflow_concurrency = pool / kv_per_request, doubling context
        halves the concurrency at which SSD I/O begins.

        At the DGX setup (pool=186 GB, kv_bpt=163840 bytes):
          ctx=  8K:  kv/req=1.25 GB  overflow_at=148 concurrent
          ctx= 32K:  kv/req=5.00 GB  overflow_at= 37 concurrent
          ctx= 65K:  kv/req=10.0 GB  overflow_at= 18 concurrent
          ctx=128K:  kv/req=20.0 GB  overflow_at=  9 concurrent

        Works on context_sweep.csv (pivot = context_len) OR on any sweep
        that includes kv_per_request_gb and context-varying data.

        Also shows:
          - TTFT scaling with context (more prefill tokens → longer TTFT)
          - TPOT sensitivity to context (decode KV$ read per step grows)
          - pf_hbm_delta scaling (more KV$ written per request)
          - dc_kv_read_per_step_gb = context × kv_bpt (each decode step reads more)
        """
        pivot_col = "context_len" if any("context_len" in str(r.get("context_len","")) for r in self.rows) else self.pivot

        derived = []
        for r in self.rows:
            pv          = _f(r, pivot_col)
            kv_req      = _f(r, "kv_per_request_gb")
            pool_gb     = _f(r, "kv_pool_capacity_gb")
            bpt         = max(_f(r, "kv_bytes_per_token"), 1)
            conc        = _f(r, "inference_concurrency")
            overflow_at = _f(r, "overflow_concurrency") or (
                int(pool_gb / max(kv_req, 0.001)) if pool_gb > 0 and kv_req > 0 else 0)
            fill_pct    = _f(r, "pool_fill_pct") or (
                round(conc * kv_req / max(pool_gb, 0.001) * 100, 1) if pool_gb > 0 else 0.0)

            # KV$ read demand per decode step
            dc_read_step = _f(r, "dc_kv_read_per_step_gb") or round(pv * bpt / (1024**3), 3)

            ttft        = _f(r, "ttft_mean_ms")
            tpot        = _f(r, "tpot_mean_ms")
            miss_pen    = _f(r, "kv_miss_penalty_ms")
            pf_hbm      = _f(r, "hbm_prefill_delta_gb_mean")
            evict_gb    = _f(r, "hbm_kv_evicted_gb_mean") or _f(r, "pf_ssd_eviction_gb_mean")
            restore_gb  = _f(r, "dc_ssd_restore_gb_mean")
            rd_bw       = _f(r, "ssd_read_bw_mb_mean")
            wr_bw       = _f(r, "ssd_write_bw_mb_mean")

            # Scaling ratios (relative to smallest context in this sweep)
            derived.append({
                pivot_col:                  int(pv) if pv else 0,
                "inference_concurrency":    int(conc),
                "kv_per_request_gb":        round(kv_req, 3),
                "kv_pool_capacity_gb":      round(pool_gb, 1),
                "overflow_concurrency":     int(overflow_at),
                "pool_fill_pct":            round(fill_pct, 1),
                "ssd_io_active":            fill_pct >= 98,
                "dc_kv_read_per_step_gb":   round(dc_read_step, 3),
                "pf_hbm_delta_gb":          round(pf_hbm, 3),
                "hbm_kv_evicted_gb":        round(evict_gb, 4),
                "dc_ssd_restore_gb":        round(restore_gb, 4),
                "ssd_read_bw_mb_s":         round(rd_bw, 2),
                "ssd_write_bw_mb_s":        round(wr_bw, 2),
                "ttft_mean_ms":             round(ttft, 1),
                "tpot_mean_ms":             round(tpot, 1),
                "kv_miss_penalty_ms":       round(miss_pen, 2),
                "throughput_tok_s":         round(_f(r, "throughput_tok_s"), 1),
                "pf_cache_hit_pct":         round(_f(r, "pf_cache_hit_pct"), 1),
                "context_regime": (
                    "short"    if pv <= 16384 else
                    "medium"   if pv <= 65536 else
                    "long"
                ),
            })

        # Compute scaling ratios relative to smallest context
        if derived:
            baseline = derived[0]
            baseline_kv  = max(baseline["kv_per_request_gb"], 0.001)
            baseline_ttft = max(baseline["ttft_mean_ms"], 1)
            baseline_tpot = max(baseline["tpot_mean_ms"], 1)
            for d in derived:
                d["kv_scaling_ratio"]   = round(d["kv_per_request_gb"] / baseline_kv, 2)
                d["ttft_scaling_ratio"] = round(d["ttft_mean_ms"] / baseline_ttft, 2)
                d["tpot_scaling_ratio"] = round(d["tpot_mean_ms"] / baseline_tpot, 2)

        ssd_active_rows = [d for d in derived if d["ssd_io_active"]]
        first_ssd_ctx   = ssd_active_rows[0][pivot_col] if ssd_active_rows else None

        # TTFT vs context correlation
        ctx_vals  = [d[pivot_col] for d in derived if d["ttft_mean_ms"] > 0]
        ttft_vals = [d["ttft_mean_ms"] for d in derived if d["ttft_mean_ms"] > 0]
        ttft_per_ktok = 0.0
        if len(ctx_vals) >= 2:
            # Linear slope: ms per 1K tokens
            ttft_per_ktok = round(
                (ttft_vals[-1] - ttft_vals[0]) / max((ctx_vals[-1] - ctx_vals[0]) / 1000, 1), 2)

        observations, warnings = [], []
        ctx_range = f"{int(derived[0][pivot_col]):,}–{int(derived[-1][pivot_col]):,}" if derived else "?"
        scale = derived[-1].get("kv_scaling_ratio", 1) if derived else 1
        observations.append(
            f"Context range: {ctx_range} tokens. "
            f"KV$/request scales {scale:.1f}× "
            f"from shortest to longest context.")
        observations.append(
            f"TTFT scales ~{ttft_per_ktok:.1f} ms per 1K context tokens "
            f"(linear with prompt length).")
        if first_ssd_ctx:
            observations.append(
                f"SSD I/O begins at context_len={first_ssd_ctx:,} tokens "
                f"(pool overflows at this context with concurrency={derived[0]['inference_concurrency']}).")
        else:
            observations.append(
                f"No SSD I/O across tested context range at concurrency={derived[0]['inference_concurrency'] if derived else '?'}.")

        if ssd_active_rows:
            max_miss = max((d["kv_miss_penalty_ms"] for d in ssd_active_rows), default=0.0)
            if max_miss > 100:
                warnings.append(
                    f"At long contexts the miss penalty reaches {max_miss:.0f} ms — "
                    f"each decode step stalls waiting for SSD restore of large KV$ blocks.")
        if derived and derived[-1].get("kv_scaling_ratio", 1) > 8:
            last = derived[-1]
            warnings.append(
                f"Context {last.get(pivot_col, 0):,} uses {last.get('kv_per_request_gb', 0):.1f} GB KV$/request — "
                f"overflow_concurrency drops to {last.get('overflow_concurrency', '?')}."
                f" Even light concurrency will hit SSD at this context length.")

        return ScenarioResult(
            scenario_id="S11", scenario_name="Context Length Effect",
            description=(
                "How context_len changes KV$/request, overflow_concurrency, TTFT scaling, "
                "and SSD I/O onset across Memory and SSD tiers."
            ),
            rows=derived,
            summary={
                "context_range":            f"{int(derived[0][pivot_col]):,}–{int(derived[-1][pivot_col]):,}" if derived else "?",
                "kv_scaling_ratio_max":     derived[-1].get("kv_scaling_ratio", 1) if derived else 0,
                "ttft_ms_per_1k_tokens":    ttft_per_ktok,
                "first_ssd_ctx":            first_ssd_ctx,
            },
            observations=observations, warnings=warnings)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _smean(vals: list) -> float:
    nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.mean(nums) if nums else 0.0


def _ratio_range(rows: list, key: str) -> str:
    """Return '(ratio range: lo–hi)' string, or empty string when all values are 0/missing."""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r.get(key, 0) > 0]
    if not vals:
        return "(ratio data not available — run with current amoprof for phase metrics)"
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return f"(prefill/decode ratio: {lo:.2f})"
    return f"(prefill/decode ratio range: {lo:.2f}–{hi:.2f})"
