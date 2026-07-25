"""KV load-back pattern classification helpers.

This module is intentionally dependency-light so report writers can use it to
avoid over-calling SSD saturation when the evidence points to small-block KV
load-back amplification instead.
"""

def classify_kv_loadback_pattern(
    *,
    physical_read_bw_mbs: float = 0.0,
    physical_read_bw_target_mbs: float = 7000.0,
    physical_read_gb: float = 0.0,
    physical_write_gb: float = 0.0,
    read_mb_per_output_token: float = 0.0,
    small_read_pct_16_64kb: float = 0.0,
    gpu_util_pct: float = 0.0,
    dram_bw_pct: float = 0.0,
    exact_qd_available: bool = False,
    exact_qd_p95: float = 0.0,
    await_ms: float = 0.0,
) -> dict:
    """Classify whether storage is truly saturated or load-back amplified.

    Returns a dict with:
      - classification: true_ssd_saturation | kv_loadback_amplification | inconclusive
      - confidence: low | medium | high
      - reasons: list[str]
    """
    reasons = []
    bw_pct = (physical_read_bw_mbs / physical_read_bw_target_mbs * 100.0) if physical_read_bw_target_mbs else 0.0
    read_dominant = physical_read_gb > max(physical_write_gb * 10.0, 1.0)
    small_reads = small_read_pct_16_64kb >= 50.0
    low_bw = bw_pct < 20.0
    heavy_per_token = read_mb_per_output_token >= 8.0
    host_gpu_not_saturated = gpu_util_pct < 70.0 and dram_bw_pct < 30.0

    if exact_qd_available and exact_qd_p95 >= 32 and await_ms >= 1.0 and not low_bw:
        return {
            "classification": "true_ssd_saturation",
            "confidence": "high",
            "reasons": ["exact blktrace queue depth is elevated", "await latency is elevated", "bandwidth is not trivially low"],
        }

    if read_dominant:
        reasons.append("physical reads dominate writes")
    if small_reads:
        reasons.append("dominant read size is small/medium 16–64 KB")
    if low_bw:
        reasons.append(f"read bandwidth is low vs target ({bw_pct:.1f}% of configured peak)")
    if heavy_per_token:
        reasons.append("high read MB per output token")
    if host_gpu_not_saturated:
        reasons.append("GPU and DRAM are not saturated")

    if len(reasons) >= 3:
        return {"classification": "kv_loadback_amplification", "confidence": "high", "reasons": reasons}
    if len(reasons) >= 2:
        return {"classification": "kv_loadback_amplification", "confidence": "medium", "reasons": reasons}
    return {"classification": "inconclusive", "confidence": "low", "reasons": reasons}
