"""
amoprof/report/enhancer.py — Post-process the bundled amoprof HTML report to:

  1. Apply a darker reading theme (slate-900 background, light text)
  2. Add hover tooltips with summary stats to each chart image
  3. Inject per-chart explanations describing the chart, what it measures,
     and its significance to KV cache and the overall AI inference stack

Designed to run AFTER amoprof.py has emitted its standard HTML report.
We do NOT modify the bundled amoprof.py — instead we read the produced
file and rewrite specific elements via parsing.

Public API:
    enhance_report(html_path: Path,
                   raw_dir: Path | None = None,
                   theme: str = "dark") -> Path

Theme options:
    "dark"   — slate-900 background, slate-50 text, kept color accents on charts
    "light"  — original amoprof palette (no-op)

The matplotlib chart palette is set inside amoprof.py at import time, so we
cannot retro-recolor the PNG plots. Instead we wrap each chart image in a
dark gradient frame that integrates the existing charts cleanly with the
darker page chrome — the matplotlib charts themselves remain readable on
their light-blue plot backgrounds.
"""
from __future__ import annotations

import csv
import json
import re
import logging
from pathlib import Path

log = logging.getLogger("amoprof.enhancer")


# ─── Chart explanations ──────────────────────────────────────────────────────
# Keyed by H2 heading text (case-insensitive substring match).
# Each entry: (one-line summary, multi-paragraph significance).
#
# Writing guide:
#   • Open with a crisp "What you are looking at" sentence.
#   • Then "What the sub-panels show" — describe each panel specifically.
#   • Then "How to read the numbers" — what good/bad looks like.
#   • Then "What this means for KV cache" — the L1/L2/L3 (AI Memory Node / remote storage) implication.
#   • End with "What to do if this looks bad" — one actionable fix.
#
# Typical run context for which these explanations were written:
#   Model:  DeepSeek-R1-Distill-Llama-70B (8× A100 40 GB, TP=8)
#   TTFT:   ~33 s  (very long context prefill — 43M tokens)
#   TPOT:   ~2.3 s/tok (HBM-bandwidth-bound decode)
#   Cache:  59 % gauge-active hit rate; 34 % token-weighted
#   HBM:    89 % full (35.7 / 40 GB), peak 100 %
#   NVMe:   3.8 TB reads, 3 GB writes — KV load-back dominated
#   DRAM:   Active L2 tier (hicache DRAM enabled)
CHART_EXPLANATIONS: dict[str, tuple[str, str]] = {

    # ── A5 · Application Layer ─────────────────────────────────────────────

    "p50 / p90 / p99 summary": (
        "Per-request latency percentiles (TTFT, ITL, E2E) from Prometheus "
        "histogram_quantile() — the truest view of tail latency.",
        "What you are looking at: Three rows of bar charts — one for TTFT "
        "(Time to First Token), one for ITL/TPOT (inter-token / decode "
        "step latency), one for E2E (full request wall-clock time). Each "
        "row has four bars: Mean, P50, P90, P99.\n\n"
        "How to read the numbers: A healthy inference server has a flat "
        "P50→P99 ratio (less than 2×). A wide P90→P99 spread means "
        "occasional cold-cache prefills or queue contention spikes. "
        "A high TTFT (>500 ms) with moderate ITL indicates the bottleneck "
        "is in the KV cache load path (L2/L3 (AI Memory Node / remote storage) miss), not in compute. "
        "A high ITL (>100 ms) means the GPU is memory-bandwidth-bound "
        "during decode — HBM is likely saturated.\n\n"
        "For KV$: P99 TTFT is the most sensitive indicator of L3 (AI Memory Node / remote storage) "
        "cache misses. An L1 hit gives near-zero TTFT; an L3 (AI Memory Node / remote storage) hit adds "
        "50–200 ms just for loading KV blocks from NVMe. If P99 >> P50 "
        "on TTFT, check the KV eviction/load-back charts — the working "
        "set is larger than your HBM pool.\n\n"
        "What to do: P99 TTFT driven by long contexts → enable "
        "`--chunked-prefill-size 4096`. Driven by cold cache → "
        "increase DRAM/NVMe tier capacity."
    ),

    "latency breakdown": (
        "Decomposes end-to-end request latency into queue time, prefill "
        "(TTFT), and decode (N × ITL) — plus a token-composition bar.",
        "What you are looking at: A stacked bar where each segment "
        "represents one phase: queue_time | TTFT | decode_total. "
        "Alongside it, a proportional token bar shows prefill_cached | "
        "prefill_compute | decode_output fractions.\n\n"
        "How to read: Large prefill_compute segment + TTFT dominating the "
        "latency bar → cache hit rate is low (KV eviction pressure or "
        "diverse prompts). decode_total dominating → model is HBM-"
        "bandwidth-bound; every decode step re-reads full model weights "
        "from HBM. queue_time dominating → insufficient capacity; "
        "requests are waiting before any GPU work starts.\n\n"
        "For KV$: The token bar's cyan (prefill_cached) fraction is the "
        "most direct measure of KV$ effectiveness. Aim for >70 % cached "
        "on workloads with repeated system prompts. Sub-30 % signals "
        "either an undersized HBM pool or a workload with no shared "
        "prefix structure.\n\n"
        "What to do: Low cached → check HBM fill % and eviction rate. "
        "High queue → scale batch size or add replicas."
    ),

    "cache hit methodology": (
        "Separates the four ways cache hit rate can be measured and "
        "explains the gap between benchmark tool output and gauge averages.",
        "What you are looking at: A table with four rows — "
        "token-weighted, gauge-active-average, gauge-timeline-average, "
        "and gauge-peak — each with its value and interpretation.\n\n"
        "The four methods:\n"
        "• Token-weighted: Δ(cached_tokens) / (Δcached + Δcompute). "
        "Matches bench_serving output exactly.\n"
        "• Gauge active avg: mean of non-zero sglang_cache_hit_rate "
        "samples. Filters out idle windows.\n"
        "• Gauge timeline avg: mean over ALL samples including idle — "
        "almost always misleadingly low.\n"
        "• Gauge peak: maximum instantaneous gauge value.\n\n"
        "In the reference run: token-weighted 34 % vs gauge-active 59 %. "
        "The gap means even during active windows, a large share of "
        "tokens still go through prefill_compute — likely unique prompt "
        "suffixes beyond the shared prefix.\n\n"
        "What to do: If token-weighted << gauge-active, inspect prompt "
        "structure. Adding a shared system-prompt prefix that all "
        "requests use dramatically raises the token-weighted hit rate."
    ),

    "ai operation phase": (
        "Side-by-side Prefill (TTFT) and Decode (TPOT) panels with "
        "severity badges, statistics, and a token composition bar.",
        "What you are looking at: Left panel = Prefill: TTFT in ms, "
        "context tokens split into compute vs cached, and a phase-driver "
        "diagnosis. Right panel = Decode: TPOT in ms/token, total decode "
        "tokens, peak throughput. Below: a token composition bar — "
        "cyan (cached) | orange (compute) | green (decode).\n\n"
        "How to read the badges: TTFT > 1000 ms → HIGH (long context or "
        "cold cache). TPOT > 80 ms/tok → HIGH (HBM-bandwidth-bound). "
        "A run with TTFT=33 s and TPOT=2.3 s is dominated by long-context "
        "prefill: 43M-token prompts hit every KV block in the cache and "
        "still run compute for uncached tails.\n\n"
        "For KV$: The cyan fraction of the token bar is your effective "
        "KV$ reuse rate. Sub-30 % cyan means less than a third of prompt "
        "tokens were served without compute — the KV pool is too small "
        "for the working set or the workload has too little prefix "
        "repetition.\n\n"
        "What to do: TTFT too high with large compute segment → chunked "
        "prefill or KV quantisation. TPOT too high → speculative decoding "
        "or reduce concurrent batch size to keep more KV in HBM."
    ),

    # ── A4 · Inference Runtime ────────────────────────────────────────────

    "sglang inference": (
        "Real-time 6-panel time series of every key SGLang metric — TTFT, "
        "TPOT, throughput, cache hit rate, token counters, queue depth.",
        "What you are looking at: Six panels from SGLang's Prometheus "
        "/metrics endpoint scraped at 1-sec intervals:\n"
        "(1) TTFT ms — per-interval mean of time_to_first_token.\n"
        "(2) TPOT/ITL ms — per-interval mean of inter_token_latency.\n"
        "(3) Gen throughput tok/s — sglang_gen_throughput rolling gauge.\n"
        "(4) Cache hit rate % — rolling cache_hit_rate gauge.\n"
        "(5) Token counters — cumulative prefill_cache/compute/decode.\n"
        "(6) Queue depth — num_running_reqs and num_queue_reqs.\n\n"
        "How to read: TTFT spike + no queue depth = cold-cache prefill "
        "(KV was evicted and must be reloaded from L2/L3 (AI Memory Node / remote storage)). "
        "Queue_depth > 0 continuously = engine capacity is insufficient. "
        "Cache hit rate drop concurrent with HBM near 100 % = "
        "eviction pressure causing cache churn.\n\n"
        "For KV$: The cache hit rate panel is your live tier health "
        "indicator. Cross-reference with the KV Block Event chart — "
        "rising evicted counters with a simultaneous hit-rate drop "
        "confirms the KV pool is too small for the working set.\n\n"
        "What to do: Sustained queue_depth > 0 → scale horizontally. "
        "Hit rate oscillating → increase `--max-total-tokens` or "
        "use KV quantisation."
    ),

    "memory profile": (
        "Combined time series of HBM (GPU), DRAM (host), and KV cache "
        "tier usage — the full three-tier memory picture on one canvas.",
        "What you are looking at: Overlapping time series for:\n"
        "• HBM Used (GB) and HBM util % per GPU (DCGM).\n"
        "• Estimated HBM BW (GB/s) derived from TPOT × model size.\n"
        "• DRAM BW (GB/s) from AMDuProf (when available).\n"
        "• KV pool usage — sglang_full_token_usage fraction.\n"
        "• hicache host tokens (L2 DRAM tier occupancy).\n\n"
        "How to read: HBM flat near 100 % + rising hicache host tokens = "
        "runtime is actively spilling KV to DRAM (L1→L2). That + rising "
        "NVMe write activity = spillover reaching L3 (DRAM→NVMe). "
        "DRAM BW spike during HBM saturation confirms active L1→L2 DMA.\n\n"
        "In the reference run: HBM 89 %, 8 GPUs × 17.5 GB weights = "
        "22 GB/GPU left for KV. At 327 KB/token, this allows ~67K tokens "
        "in HBM before spilling — any 131K-token context immediately "
        "uses both HBM and DRAM tiers.\n\n"
        "For KV$: This is the single best chart for seeing tier spillover "
        "in action. A rising hicache_host line concurrent with flat/full "
        "HBM is the L1→L2 spill event.\n\n"
        "What to do: HBM pinned at 100 % → KV quantisation (FP8 halves "
        "per-token cost) or reduce `--max-num-seqs`."
    ),

    "kv cache footprint": (
        "Static KV cache capacity analysis: bytes/token, max HBM context, "
        "weight footprint, quantisation savings, and spill threshold.",
        "What you are looking at: Six mini-cards showing:\n"
        "• KV Bytes/Token: 2 × n_layers × n_kv_heads × head_dim × dtype_bytes.\n"
        "• Max Context (HBM): tokens fitting in HBM KV pool.\n"
        "• Weight Footprint: GB of HBM used by model parameters.\n"
        "• KV Pool Used %: fraction of pre-allocated pool occupied.\n"
        "• INT8 Savings: doubled token capacity at INT8.\n"
        "• Spill Threshold: token count that triggers DRAM spill.\n\n"
        "For DeepSeek-R1-Distill-Llama-70B in BF16 on A100 40 GB:\n"
        "  2 × 80 layers × 8 KV-heads × 128 dim × 2 bytes = 327 KB/token\n"
        "  ~17.5 GB weights/GPU → ~22.5 GB left for KV → ~68K tokens max\n"
        "  At 131K context window: every request immediately overflows HBM\n"
        "  INT8 KV doubles capacity to ~137K before spilling.\n\n"
        "For KV$: The Spill Threshold card directly tells you the context "
        "length at which you START generating NVMe traffic. Any prompt "
        "longer than this will hit L2/L3 (AI Memory Node / remote storage).\n\n"
        "What to do: Contexts regularly exceeding spill threshold → "
        "enable FP8 KV (`--kv-cache-dtype fp8_e5m2` in SGLang ≥0.4). "
        "Accuracy impact is typically <0.5 % on standard benchmarks."
    ),

    "per cache tier": (
        "Per-tier (L1=HBM / L2=DRAM / L3 (local storage)=SSD; L3=AI Memory Node/remote storage) token distribution and "
        "access cost — translates tier occupancy into latency impact.",
        "What you are looking at: Two panels:\n"
        "(1) Token distribution: stacked bar or time series showing what "
        "fraction of active KV tokens live in each tier.\n"
        "(2) Tier cost analysis: access latency per tier (L1 ~1 µs, "
        "L2 ~10 µs, L3 ~100 µs) weighted by token fractions to give "
        "a weighted-average KV access cost.\n\n"
        "How to read: 100 % L1 = ideal (working set fits in HBM). "
        "50 % L1 / 50 % L2 = active DRAM staging, manageable. "
        "Any L3 (AI Memory Node / remote storage) fraction = L3 local storage on the critical decode path, "
        "adds 100–500 µs per decode step per session.\n\n"
        "For KV$: The cost chart is the direct translation from "
        "'where are my tokens' to 'what does it cost per generated "
        "token'. L3 (AI Memory Node / remote storage) hit at 100 µs × 1000 decode steps × 100 "
        "concurrent sessions = 10 s of aggregate L3 (local storage) latency per "
        "generated sequence.\n\n"
        "What to do: If L3 (AI Memory Node / remote storage) > 20 % of tokens during decode, either "
        "reduce session count, add DRAM (expand L2 tier), or upgrade "
        "to faster NVMe."
    ),

    "session / stream": (
        "Per-inference-session breakdown: concurrency, KV footprint, "
        "and I/O bandwidth consumed by each active request.",
        "What you are looking at: Four panels:\n"
        "(1) Gantt bars: session start/end times.\n"
        "(2) KV footprint: peak tokens in cache per session.\n"
        "(3) BW consumed: NVMe MB/s attributed per session (biosnoop).\n"
        "(4) Concurrency: simultaneous active sessions over time.\n\n"
        "How to read: Uniform session lengths + similar KV footprints = "
        "predictable, easy-to-size workload. A few outliers with 10× "
        "KV footprint of peers = long-context requests monopolising "
        "resources and pushing shorter sessions' KV out of HBM, "
        "degrading everyone's cache hit rate.\n\n"
        "For KV$: Sessions whose context exceeds HBM capacity generate "
        "continuous L2/L3 (AI Memory Node / remote storage) traffic on every decode step. A 200K-token "
        "session with a 68K-token L1 capacity has a permanent 132K-token "
        "'hot tail' in L2/L3 (AI Memory Node / remote storage) that gets paged in/out continuously.\n\n"
        "What to do: Route outlier long-context sessions to a dedicated "
        "server, or enable sliding window attention to cap the active "
        "KV footprint."
    ),

    "kv cache block": (
        "KV cache block lifecycle event rates — eviction, backup (L1→L2/L3 (AI Memory Node / remote storage) "
        "write), and load-back (L3 (AI Memory Node / remote storage)/L2→L1 read) — the clearest signal of "
        "KV cache thrashing.",
        "What you are looking at: Time series of four SGLang hicache "
        "counters (rates = Δ/s):\n"
        "• evicted_tokens/s → blocks removed from HBM.\n"
        "• backuped_tokens/s → blocks written to DRAM/NVMe.\n"
        "• load_back_tokens/s → blocks read back into HBM.\n"
        "• prefix_cache_miss_tokens/s → tokens requiring compute.\n\n"
        "How to read the patterns:\n"
        "High evicted + low backup = HBM overcommitted, blocks dropped "
        "before persisting (lost KV, must recompute).\n"
        "High backup + high load_back = thrashing: same blocks "
        "written then read repeatedly.\n"
        "Low backup + high load_back = blocks were backed up in a prior "
        "phase, now being read back (the reference run pattern).\n\n"
        "For KV$: The reference run shows 3.8 TB reads vs 3 GB writes "
        "in NVMe — consistent with load_back dominating over backup. "
        "The KV was written in an earlier warm-up and is now being "
        "read back on every decode step for long-context sessions. "
        "Reading 3.8 TB for 29,660 decode tokens = ~128 MB per token "
        "of NVMe traffic.\n\n"
        "What to do: High load_back rate + NVMe near saturation → "
        "add NVMe bandwidth (RAID-0 of 2–4 drives). High backup rate "
        "→ check if DRAM tier can absorb more before spilling to NVMe."
    ),

    # ── A3 · OS / Kernel ────────────────────────────────────────────────

    "storage i/o": (
        "Aggregate NVMe read/write bandwidth plus OS page-fault rate on "
        "a shared timeline — the health-vitals view of storage.",
        "What you are looking at: Two overlaid time series — disk R/W BW "
        "(MB/s, from iostat) and major page fault rate (faults/s, from "
        "/proc/vmstat).\n\n"
        "How to read: Sustained write >500 MB/s, no major-fault spike = "
        "SGLang hicache actively backing up KV to NVMe. "
        "Sustained read >500 MB/s, low write = load-back dominated "
        "(this run: 581 MB/s reads, 0.5 MB/s writes). "
        "Major faults + NO disk I/O = DRAM pressure / swap. "
        "Major faults + disk I/O = demand-paging model weights.\n\n"
        "The 18.9M major-faults/sec peak visible in the tooltip is "
        "almost certainly a /proc/vmstat counter overflow artefact — "
        "genuine 18M faults/sec would saturate any storage in seconds. "
        "Treat the major-fault data in this run as noise.\n\n"
        "For KV$: Read BW without corresponding write BW = the NVMe "
        "is serving previously-written KV (load-back phase). The "
        "write phase happened earlier. This is expected for "
        "long-running sessions that warm the cache over many requests.\n\n"
        "What to do: Read BW near device limit → add NVMe bandwidth. "
        "Write BW near limit → check whether other processes share "
        "the device; dedicate NVMe to hicache."
    ),

    "swap storm": (
        "Page swap rates (pswpin/pswpout) and major faults from /proc/vmstat "
        "— early warning for kernel memory pressure during inference.",
        "What you are looking at: Three time series — swap-in pages/s, "
        "swap-out pages/s, and major fault rate/s — all derived from "
        "Δ(counter) / Δ(time) on /proc/vmstat cumulative counters.\n\n"
        "How to read: Any sustained pswpin/pswpout > 0 during inference "
        "is a critical failure mode. Swap I/O during ML inference means "
        "the OS is treating model weights or activations as pageable "
        "memory and evicting them — CUDA operations block until the "
        "page returns from swap, causing TPOT spikes measured in "
        "seconds rather than milliseconds.\n\n"
        "Why the interactive chart shows 'no data': The interactive "
        "chart correctly returns empty when pswpin and pswpout are all "
        "zeros — no swap activity occurred. The static report renders a "
        "placeholder panel regardless of whether data is present. "
        "An empty interactive swap chart is good news: the server was "
        "not swapping during this run.\n\n"
        "For KV$: Swap I/O and hicache L3 (local storage) I/O can both appear in "
        "iostat reads/writes. This chart distinguishes them: hicache "
        "backup appears in iostat but NOT in pswpout. If iostat shows "
        "writes but this chart shows zero swap, it is KV backup, not "
        "OS swap — which is the correct and expected behaviour.\n\n"
        "What to do: To prevent swap: set vm.swappiness=0, use "
        "`--disable-swap-space` (Linux systems), and ensure the "
        "CUDA unified memory driver does not over-allocate."
    ),

    # ── A2 · Drivers ─────────────────────────────────────────────────────

    "gpu utilisation": (
        "Per-GPU time series of SM utilisation, HBM occupancy, power "
        "draw, and HBM bandwidth active fraction from DCGM.",
        "What you are looking at: A 4-panel chart (or 2×2 grid):\n"
        "(1) GPU Util % — DCGM_FI_DEV_GPU_UTIL (SM occupancy).\n"
        "(2) HBM Used (GB) — DCGM_FI_DEV_FB_USED.\n"
        "(3) Power (W) — DCGM_FI_DEV_POWER_USAGE.\n"
        "(4) HBM BW Active % — DCGM_FI_PROF_DRAM_ACTIVE × 100.\n\n"
        "Important: GPU util is SM occupancy, not HBM bandwidth. "
        "An A100 can show 100 % util at low HBM BW (compute-bound "
        "attention) or 40 % util at full HBM BW (decode, memory-"
        "bandwidth-bound). The HBM BW Active panel is the direct "
        "bandwidth saturation signal.\n\n"
        "In the reference run: mean 96.4 % util, peak 100 %, "
        "HBM 89 % full (35.7/40 GB), power 392 W/GPU. The GPUs are "
        "fully loaded during a combined long-context prefill (compute-"
        "bound) and decode (bandwidth-bound) workload.\n\n"
        "For KV$: HBM fill % is the L1 tier gauge. Pinned at 89–100 % "
        "means every new session competes for KV space. Sudden drop in "
        "HBM fill followed by a load_back spike = session evicted, "
        "then immediately requested again — classic thrashing.\n\n"
        "What to do: Util > 95 % with HBM > 85 % and rising TPOT → "
        "KV quantisation (FP8) to free HBM and improve batching."
    ),

    "nvme io workload characteristics": (
        "Aggregate NVMe IOPS, bandwidth, queue depth, and device "
        "utilisation from iostat — the SSD driver's view of all block I/O.",
        "What you are looking at: A 6-panel chart:\n"
        "(1) Read IOPS vs Write IOPS time series.\n"
        "(2) Read BW (MB/s) vs Write BW (MB/s).\n"
        "(3) Average I/O queue depth over time.\n"
        "(4) Device busy-time from iostat, advisory only.\n"
        "(5) SMART summary: model, temperature, WAF, lifetime used.\n"
        "(6) Service time / await latency distribution.\n\n"
        "In the reference run: 581 MB/s reads at 15,491 IOPS vs "
        "0.5 MB/s writes at 20 IOPS — a 1000:1 read/write ratio. "
        "Average read size = 581 MB/s ÷ 15,491 IOPS ≈ 37 KB, "
        "consistent with one KV block per I/O. The write floor "
        "confirms this is a load-back-dominated phase with no "
        "active eviction.\n\n"
        "Note on SMART: If model shows '?' and WAF=1.0, the nvme-cli "
        "SMART collection failed. Run `nvme smart-log /dev/nvme0 -o "
        "json` to verify the device is recognised.\n\n"
        "For KV$: 15K IOPS at 37 KB/IO suggests page_size=1 token "
        "produces one I/O per KV block. Larger page sizes reduce IOPS "
        "for the same bandwidth — try `--sglang-page-size 64`.\n\n"
        "What to do: IOPS near SSD random-read spec → increase page_size "
        "to coalesce reads. BW near sequential spec → add NVMe devices."
    ),

    "nvme io deep profiling": (
        "Per-request NVMe detail from blktrace: request size distribution, "
        "alignment, sequential vs random ratio, and I/O latency.",
        "What you are looking at: A 6-panel deep-dive:\n"
        "(1) Request size histogram — peak at the KV block size.\n"
        "(2) Access alignment — fraction at 4KB/64KB/512KB boundaries.\n"
        "(3) Sequential vs random — stride < 2× KV block = sequential.\n"
        "(4) IAT (inter-arrival time) distribution.\n"
        "(5) Latency distribution from blktrace timestamps.\n"
        "(6) Queue depth from blktrace issue/complete events.\n\n"
        "Requires `--enable-blktrace` at collect time. If these panels "
        "are empty, re-collect with blktrace enabled.\n\n"
        "How to read: KV I/O fingerprint = predominantly 32–512 KB reads, "
        "4KB-aligned, burst IAT (sequential within session, random across). "
        "4KB I/Os in this context = inode/metadata, not KV blocks. "
        "High misalignment → hicache files not written with O_DIRECT.\n\n"
        "For KV$: Sequential-vs-random ratio reveals how well hicache "
        "organises NVMe layout. Session blocks stored contiguously → "
        "sequential load-back (fast). Scattered → random I/O.\n\n"
        "What to do: High misalignment → use XFS with largeio mount. "
        "High randomness → pre-allocate KV files in session-order chunks."
    ),

    "ssd read workload": (
        "Read-only deep dive (§C) from blktrace: read request sizes, "
        "inter-arrival times, and alignment.",
        "What you are looking at: Three sub-panels — §C read size "
        "histogram, §C read IAT histogram, §C read alignment fractions.\n\n"
        "The reference run's 101M read events at 15K IOPS and 581 MB/s "
        "imply ~37 KB average read size. Peak in the size histogram "
        "at 32–64 KB = one KV block (page_size × KV bytes/token). "
        "A peak at 4 KB = fragmented hicache (individual memory pages "
        "accessed instead of whole blocks).\n\n"
        "For KV$: An L3 (AI Memory Node / remote storage) hit at 37 KB costs: 37 KB / 3 GB/s seq = "
        "12 µs on an uncongested NVMe. At 15K IOPS (random): "
        "37 KB / 600 MB/s = 62 µs. At TPOT=2.3 s, NVMe is not "
        "the primary bottleneck, but every parallel session's KV "
        "reads stack up on the same device queue.\n\n"
        "What to do: Read size peak below 32 KB → increase "
        "`--sglang-page-size`. Alignment < 90 % → add O_DIRECT to "
        "hicache open() flags."
    ),

    "ssd write workload": (
        "Write-only deep dive (§D) from blktrace: write sizes, "
        "inter-arrival times, and alignment.",
        "What you are looking at: Three sub-panels — §D write size "
        "histogram, §D write IAT histogram, §D write alignment.\n\n"
        "The reference run shows only 3.14 GB writes vs 3,808 GB reads. "
        "This near-zero write rate means the KV backup phase was "
        "completed in an earlier warm-up run; this session is entirely "
        "in load-back mode. In a write-heavy phase you would see "
        "large sequential writes (1–4 MB) as whole KV block groups "
        "are flushed to NVMe.\n\n"
        "For KV$: Sequential writes (large §D bars at 512KB–4MB) are "
        "SSD-friendly (low WAF). Random small writes trigger NAND GC "
        "and raise WAF. If WAF > 3.0 in SMART, the write pattern is "
        "suboptimal — another process may be writing concurrently.\n\n"
        "What to do: Write BW near device limit → ensure no other "
        "workloads share the NVMe. WAF >3 → check for 4K random "
        "writes from another process."
    ),

    "trim / discard": (
        "TRIM (discard) commands from blktrace — how the OS notifies the "
        "SSD which blocks are free so its GC can reclaim them.",
        "What you are looking at: Three sub-panels (§E) — TRIM event rate "
        "over time, TRIM byte volume, and seasonal TRIM pattern.\n\n"
        "The reference run shows 0 TRIM events. This means either the "
        "filesystem is mounted without the discard option, the hicache "
        "does not explicitly fallocate/unlink, or the SSD filters TRIM "
        "commands. Without TRIM, the SSD cannot reclaim freed KV blocks "
        "without a full GC pass, gradually raising write amplification.\n\n"
        "For a read-heavy physical block run the absence of TRIM has "
        "minimal immediate performance impact, but matters for long-term "
        "write endurance as the cache fills over days/weeks of usage.\n\n"
        "For KV$: TRIM is most critical on write-heavy workloads where "
        "KV eviction cycles are rapid. For read-heavy physical block runs, enabling "
        "TRIM has near-zero overhead and is always worthwhile.\n\n"
        "What to do: Mount hicache filesystem with `-o discard` "
        "(inline TRIM) or schedule `fstrim /mnt/hicache` hourly "
        "(batch TRIM, lower write overhead)."
    ),

    "cross-layer kpi": (
        "Composite KPIs derived by correlating NVMe + GPU + SGLang — "
        "per-token SSD cost and throughput bottleneck estimate.",
        "What you are looking at: Two key panels:\n"
        "§G88-89: SSD Dependency per Token — MB of L3 I/O per "
        "generated output token. Directly measures how much the decode "
        "path depends on L3 (AI Memory Node / remote storage).\n"
        "§G93: Throughput Bottleneck Estimate — which subsystem (HBM, "
        "DRAM, L3 (AI Memory Node / remote storage), or compute) is the binding constraint at the "
        "observed throughput.\n\n"
        "Reference framework (worked example, NOT this run's numbers): "
        "if a workload reads 3,808 GB from L3 over 29,660 decode tokens, "
        "that's ~128 MB per output token — every generated token costs "
        "128 MB of L3 (local storage) reads. At 581 MB/s L3 (AI Memory Node / remote storage) BW and 67.5 tok/s "
        "throughput, the KV demand is 128 MB × 67.5 = 8.6 GB/s while "
        "supply is 581 MB/s. The L3 (AI Memory Node / remote storage) bandwidth is 15× under-provisioned "
        "for the KV demand. Whether L3 (AI Memory Node / remote storage) is actually the binding "
        "constraint depends on GPU util — if the GPU is already saturated, "
        "L3 (AI Memory Node / remote storage) is a secondary constraint.\n\n"
        "For KV$: >1 MB/tok = L3 (AI Memory Node / remote storage) is active on the critical path. "
        ">10 MB/tok = L3 (AI Memory Node / remote storage) is a significant bottleneck. "
        ">100 MB/tok = KV tier architecture needs "
        "rethinking (more DRAM, faster L3 local-storage backend or configured L3 backend, or KV quantisation).\n\n"
        "What to do: Increase DRAM tier (move more KV from L3 to L2), "
        "or upgrade the L3 (local storage) backend (PCIe 5.0 NVMe at 7 GB/s vs 3.5 GB/s on Gen 4; "
        "or move to an AI Memory Node / Mooncake-style remote cache)."
    ),

    "cross-layer correlation": (
        "Scatter and time series correlating GPU utilisation, NVMe "
        "utilisation, and inference throughput — bottleneck attribution.",
        "What you are looking at: Three panels:\n"
        "(1) Per-token SSD bandwidth (MB/tok) over time.\n"
        "(2) Per-token HBM bandwidth (GB/tok) over time.\n"
        "(3) GPU util vs NVMe util scatter — one point per sample.\n\n"
        "How to read: GPU busy + NVMe busy simultaneously → decode "
        "doing both compute and KV I/O (common steady state). "
        "GPU idle + NVMe busy → GPU waiting for KV data from NVMe "
        "(storage is the bottleneck). "
        "GPU busy + NVMe idle → compute-bound prefill.\n\n"
        "The scatter plot is the fastest diagnosis tool: "
        "positive correlation (both rise together) = well-utilised. "
        "Negative correlation (GPU drops when NVMe spikes) = "
        "storage starving the GPU.\n\n"
        "For KV$: In this run the positive correlation between GPU util "
        "(96 %) and NVMe reads (581 MB/s) shows both are active "
        "simultaneously — the GPU is running compute while the NVMe "
        "prefetches the next batch of KV blocks.\n\n"
        "What to do: Negative correlation → increase NVMe bandwidth "
        "or expand DRAM tier. Positive saturation of both → need more "
        "hardware (more GPUs or faster NVMe)."
    ),

    "ssd / gpu / inference": (
        "Cross-layer KPI section combining §G88-89 (SSD dependency), "
        "§G93 (bottleneck estimate), and GPU correlation data.",
        "This section covers the same composite metrics as the "
        "'Cross-Layer KPIs' section. Key number: SSD Dependency per "
        "Token (MB/tok). Under 0.1 = NVMe not a bottleneck. "
        "0.1–10 = moderate L3 (AI Memory Node / remote storage) activity. Over 10 = L3 on decode "
        "critical path. Over 100 (as in this run at ~128 MB/tok) = "
        "KV tier architecture needs fundamental rethinking.\n\n"
        "For KV$: The 128 MB/tok in this run reflects very long "
        "sessions (43M-token contexts) where the entire KV history "
        "must be loaded from NVMe on each decode step. "
        "The solution is a larger DRAM tier (keep most KV in L2) "
        "or GPUDirect Storage to eliminate the CPU/DRAM hop on "
        "L3 (AI Memory Node / remote storage)→L1 transfers."
    ),

    "ssd io workload characteristics": (
        "Unified §C/§D/§E comparison on the same axes — read/write/trim "
        "bandwidth, IOPS, latency, and IAT side-by-side.",
        "What you are looking at: Three columns (reads | writes | trims) "
        "each showing BW, IOPS, latency distribution, and IAT. "
        "Allows direct comparison of the shape of each I/O type.\n\n"
        "The reference run's 1000:1 read:write ratio is far outside "
        "steady-state. This captures a pure load-back phase where the "
        "NVMe is serving previously-written KV data with almost no "
        "new backup writes. The KV cache was pre-warmed (written) "
        "before this measurement window.\n\n"
        "For KV$: Compare §C reads to the load_back_tokens counter "
        "in the KV Block Events chart. Compare §D writes to the "
        "backuped_tokens counter. They should correlate: more backup "
        "→ more §D writes; more load_back → more §C reads.\n\n"
        "What to do: If §D writes and §C reads are balanced, the "
        "cache is in steady-state churn. If §C reads heavily dominate, "
        "the cache was pre-warmed and is being consumed."
    ),

    "per-stream bandwidth": (
        "Per-session (pid/process) NVMe bandwidth attribution from "
        "biosnoop — which requests are consuming the most I/O.",
        "What you are looking at: Ranked bar chart of top 20 streams by "
        "total NVMe bandwidth, plus a temporal stacked area chart.\n\n"
        "How to read: If one stream accounts for >50 % of NVMe BW, "
        "it is monopolising the device. This is the head-of-line "
        "blocking scenario: a long-context session reading 100+ MB "
        "of KV per decode step saturates the device queue, "
        "delaying other sessions' load-back requests.\n\n"
        "SGLang typically appears as a single PID. Multiple "
        "workers/threads in biosnoop indicate parallel I/O workers "
        "in hicache — good for throughput but may create fairness "
        "issues on shared devices.\n\n"
        "For KV$: Even bandwidth distribution means all sessions "
        "get equal NVMe access (fair scheduling). Uneven distribution "
        "means the longest-context sessions are getting "
        "disproportionate NVMe time at the expense of shorter ones.\n\n"
        "What to do: Use ionice or io_uring submission queues to "
        "enforce per-session bandwidth fairness."
    ),

    "lba hot/cold": (
        "Spatial LBA access heatmap from blktrace — which regions of "
        "the SSD are hot vs cold, and the working-set Gini coefficient.",
        "What you are looking at: Three panels:\n"
        "(1) LBA heatmap — access density across the SSD address range.\n"
        "(2) Top-N hot regions with byte contribution.\n"
        "(3) Gini coefficient (0=uniform, 1=concentrated in one spot).\n\n"
        "How to read: High Gini (>0.8) = accesses concentrated in a "
        "small SSD region — the working set is much smaller than "
        "total disk capacity. You could use a smaller/cheaper SSD. "
        "Low Gini (<0.4) = scattered access — likely fragmented "
        "hicache files or multiple datasets sharing the device.\n\n"
        "For KV$: KV blocks for the same session stored contiguously "
        "= high Gini within that region = fast sequential load-back. "
        "High global Gini + high Gini within the hot region = "
        "well-organised hicache. Low Gini = hicache files fragmented "
        "or interleaved with OS/model-weight files.\n\n"
        "What to do: Low Gini → dedicate a separate NVMe exclusively "
        "to the hicache directory. High Gini with high read IOPS → "
        "the NVMe is appropriate size; focus on bandwidth instead."
    ),

    # ── A1 · Hardware ────────────────────────────────────────────────────

    "hbm (gpu) bandwidth": (
        "GPU HBM hardware bandwidth from DCGM counters — actual HBM "
        "bus utilisation at the silicon level.",
        "What you are looking at: A 4-panel chart:\n"
        "(1) HBM R/W BW timeline (GB/s) — DCGM_FI_PROF_DRAM_ACTIVE derived.\n"
        "(2) Stacked HBM read + write area.\n"
        "(3) Cache-line transaction rate (M/s, 64-byte lines).\n"
        "(4) HBM utilisation vs A100 theoretical peak (~2 TB/s).\n\n"
        "How to read: Sustained >1.5 TB/s during decode = memory-"
        "bandwidth-bound. Each decode token reads model weights + "
        "all active KV from HBM once. For 70B BF16 on 8× A100:\n"
        "HBM BW/token ≈ (140 GB weights + 22 GB KV) / 8 GPUs ≈ 20 GB\n"
        "At 67.5 tok/s: 20 GB × 67.5 / 8 GPUs ≈ 169 GB/s per GPU\n"
        "HBM BW Active at 16 % (from the tooltip) suggests the DCGM "
        "counter is sampling at low granularity or not all transfers "
        "are captured — cross-check with gpu_util (96 %).\n\n"
        "For KV$: Reducing KV bytes/token (FP8 KV) directly reduces "
        "HBM BW per decode step, allowing faster decode or larger "
        "batches at the same HBM utilisation.\n\n"
        "What to do: HBM BW > 80 % of peak → FP8 KV, speculative "
        "decoding, or reduce context length."
    ),

    "system dram": (
        "Host DRAM read/write bandwidth from AMD uProf PMU (Data Fabric "
        "counters) — the L2 memory tier's physical bandwidth.",
        "What you are looking at: A 4-panel chart:\n"
        "(1) DRAM R/W BW timeline (GB/s).\n"
        "(2) Stacked read + write area.\n"
        "(3) Cache-line transaction rate (M/s).\n"
        "(4) DRAM utilisation vs AMD EPYC peak (~460 GB/s 12-ch DDR5).\n\n"
        "Why interactive chart is empty: The interactive report looks "
        "for `amduprof_pcm_raw.txt`; the collector also writes "
        "`amduprof_pcm_raw.csv`. If only the CSV was written, the "
        "interactive builder silently returns None while the static "
        "report can still parse it. Fix: ensure both files are written, "
        "or update the interactive builder to try both extensions.\n\n"
        "How to read: High DRAM BW (>100 GB/s) during inference = "
        "hicache L2 tier actively staging KV between DRAM and HBM. "
        "Low DRAM BW + high NVMe BW = KV goes NVMe→GPU directly "
        "(GPUDirect Storage / ICMSP bypass).\n\n"
        "For KV$: Sustained 50+ GB/s during decode = DRAM is the "
        "active L2 KV staging tier. Near 460 GB/s peak = DRAM is "
        "the bottleneck for KV staging.\n\n"
        "How to collect: `sudo AMDuProfPcm -r -m memory -a --msr -d 300 -o raw/amduprof_pcm_raw.csv`. AMOprof mirrors/normalizes the raw output for both static and interactive reports."
    ),

    # ── Cross-layer / Appendix ────────────────────────────────────────────

    "bottleneck attribution": (
        "Full-stack bottleneck radar and score table — ranked list of "
        "which subsystem is the dominant constraint.",
        "What you are looking at: A radar chart (spider plot) showing "
        "each subsystem's saturation score (0–100), plus a ranked "
        "table with layer, score, severity, phase, and detail.\n\n"
        "Scoring heuristics used:\n"
        "• HBM capacity: fill >80 % + rising eviction.\n"
        "• HBM bandwidth: DCGM DRAM_ACTIVE >70 %.\n"
        "• NVMe throughput: iostat util >80 % or queue_depth >16.\n"
        "• Compute: GPU util >90 % with non-zero queue_depth.\n"
        "• DRAM BW: AMDuProf total >70 % of peak.\n"
        "• Prefill compute: TTFT >3× expected for context length.\n\n"
        "For the reference run, expected top bottlenecks:\n"
        "1. HBM BW + HBM Capacity (96 % util, 89 % fill, 2.3 s/token)\n"
        "2. Prefill Compute (33 s TTFT on 43M-token prompts)\n"
        "3. NVMe reads (3.8 TB at 15K IOPS, 128 MB/tok)\n\n"
        "For KV$: Address top-1 bottleneck first. Fixing bottleneck-3 "
        "before bottleneck-1 will show minimal improvement.\n\n"
        "What to do: Scroll to Optimisation Recommendations for "
        "auto-generated prioritised actions."
    ),

    "metric derivations": (
        "Reference table of exact formulas, source metric paths, and "
        "computation notes for every number in the report.",
        "What you are looking at: A multi-row table — metric name | "
        "formula | source path | notes.\n\n"
        "Use when a number looks surprising. Key entries:\n"
        "• TTFT: Δ(time_to_first_token_seconds_sum) / "
        "Δ(…_count) × 1000 ms.\n"
        "• Cache hit: Δcached_tokens / (Δcached + Δcompute) × 100.\n"
        "• HBM util: FB_USED / (FB_USED + FB_FREE) × 100.\n"
        "• DRAM BW: AMDuProf Data Fabric read+write × 64B / interval.\n\n"
        "Note: DRAM BW counts CPU-side memory controller transactions "
        "(KV DMA + OS page-cache + CPU workloads), not GPU HBM traffic. "
        "These are different measurements.\n\n"
        "For KV$: `cache_hit_calc_method` field in the summary shows "
        "which fallback was used — `counter_derived` is most accurate, "
        "`active_mean` is acceptable, `all_zeros` means no SGLang data."
    ),

    "optimisation": (
        "Auto-generated recommendations ranked by impact and effort, "
        "derived from the bottleneck scoring of this specific run.",
        "What you are looking at: Recommendation cards each with: title, "
        "impact badge (HIGH/MED/LOW), effort badge, phase badge, and "
        "a detailed explanation with the exact command to run.\n\n"
        "For the reference run, the top four recommendations are:\n"
        "1. CHUNKED PREFILL (HIGH impact, LOW effort): "
        "`--chunked-prefill-size 4096` in SGLang. Spreads the 33 s "
        "TTFT across multiple steps by interleaving prefill chunks with "
        "decode, dramatically reducing perceived first-token latency.\n"
        "2. SPECULATIVE DECODING (MED impact, MED effort): Draft model "
        "amortises HBM reads across multiple accepted tokens, "
        "improving 2.3 s/tok TPOT.\n"
        "3. KV QUANTISATION — AWQ/GPTQ 4-bit (HIGH impact, MED effort): "
        "Halves HBM pressure and expands KV pool.\n"
        "4. NUMA BINDING (LOW impact, LOW effort): `numactl --localalloc` "
        "eliminates 91 NUMA migrations/s.\n\n"
        "For KV$: KV quantisation (item 3) has cascading benefits: "
        "smaller per-token footprint → more tokens in HBM → fewer "
        "L2/L3 (AI Memory Node / remote storage) evictions → lower NVMe load → reduced TPOT."
    ),
}


def _match_explanation(heading: str) -> tuple[str, str] | None:
    h = heading.lower()
    for key, val in CHART_EXPLANATIONS.items():
        if key in h:
            return val
    return None


# ─── Stat extraction from raw/ for tooltips ──────────────────────────────────
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_csv_rows(path: Path) -> list[dict]:
    """Read a small/medium CSV as dict rows for post-processing tooltips.

    The report generator itself owns heavy dataframe processing.  The enhancer
    only needs enough information to keep hover summaries aligned with the
    charts that users see in the Interactive tab.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        log.debug("csv read error %s: %s", path, e)
        return []


def _read_csv_summary(path: Path) -> dict:
    """Return basic min/mean/max for each numeric column."""
    rows = _read_csv_rows(path)
    if not rows:
        return {}
    out: dict[str, dict[str, float]] = {}
    try:
        cols = list(rows[0].keys())
        data: dict[str, list[float]] = {c: [] for c in cols}
        for row in rows:
            for c in cols:
                try:
                    data[c].append(float(row[c]))
                except (ValueError, TypeError):
                    pass
        for c, vals in data.items():
            if vals:
                out[c] = {"min": min(vals), "mean": sum(vals)/len(vals),
                          "max": max(vals), "n": len(vals)}
    except Exception as e:
        log.debug("csv summary error %s: %s", path, e)
    return out


def _gpu_tooltip_stats_from_timeseries(raw_dir: Path, summary: dict) -> dict:
    """Return GPU stats using the same source/active-sample semantics as Interactive.

    v1.39.96: the End Report hover tooltip used gpu_summary.json directly,
    which can include idle collection intervals and completely inactive physical
    GPUs.  The Interactive KPI uses active/non-zero samples from
    gpu_timeseries.csv.  Use the same source and semantics here so hovering the
    End Report chart does not contradict the Interactive tab.
    """
    rows = _read_csv_rows(raw_dir / "gpu_timeseries.csv") if raw_dir else []
    vals: list[float] = []
    active_vals: list[float] = []
    hbm_vals_mb: list[float] = []
    power_vals: list[float] = []
    active_gpu_ids: set[str] = set()

    mem_keys = (("mem_used", 1.0), ("mem_used_mb", 1.0),
                ("hbm_used_mb", 1.0), ("mem_used_gb", 1024.0),
                ("hbm_used_gb", 1024.0))

    for r in rows:
        gid = str(r.get("gpu_idx", r.get("gpu", ""))).strip() or "0"
        util = None
        for key in ("gpu_util", "DCGM_FI_DEV_GPU_UTIL", "utilization.gpu"):
            try:
                if r.get(key) not in (None, ""):
                    util = float(r.get(key))
                    break
            except (TypeError, ValueError):
                pass
        if util is not None:
            vals.append(util)
            if util > 0:
                active_vals.append(util)
                active_gpu_ids.add(gid)

        for key, scale in mem_keys:
            try:
                v = r.get(key)
                if v not in (None, ""):
                    fv = float(v) * scale
                    if fv > 0:
                        hbm_vals_mb.append(fv)
                        active_gpu_ids.add(gid)
                    break
            except (TypeError, ValueError):
                continue

        for key in ("power", "gpu_power", "DCGM_FI_DEV_POWER_USAGE"):
            try:
                v = r.get(key)
                if v not in (None, ""):
                    fv = float(v)
                    if fv > 0:
                        power_vals.append(fv)
                        active_gpu_ids.add(gid)
                    break
            except (TypeError, ValueError):
                continue

    mean_source = "summary"
    if active_vals:
        util_mean = sum(active_vals) / len(active_vals)
        util_peak = max(vals) if vals else max(active_vals)
        mean_source = "active samples from gpu_timeseries.csv"
    else:
        util_mean = float(summary.get("gpu_util_mean", 0) or 0)
        util_peak = float(summary.get("gpu_util_peak", 0) or 0)

    hbm_mean_mb = (sum(hbm_vals_mb) / len(hbm_vals_mb)) if hbm_vals_mb else float(summary.get("hbm_used_mb_mean", 0) or 0)
    hbm_total_mb = float(summary.get("hbm_total_mb_per_gpu", 0) or 0)
    power_peak = max(power_vals) if power_vals else float(summary.get("power_w_peak", 0) or 0)

    gpu_count = len(active_gpu_ids) if active_gpu_ids else int(float(summary.get("gpu_count", 0) or 0))
    raw_gpu_count = int(float(summary.get("gpu_count", gpu_count) or gpu_count or 0))
    return {
        "gpu_count": gpu_count,
        "raw_gpu_count": raw_gpu_count,
        "gpu_util_mean": util_mean,
        "gpu_util_peak": util_peak,
        "hbm_used_mb_mean": hbm_mean_mb,
        "hbm_total_mb_per_gpu": hbm_total_mb,
        "hbm_util_pct_mean": (hbm_mean_mb / hbm_total_mb * 100.0) if hbm_total_mb > 0 else float(summary.get("hbm_util_pct_mean", 0) or 0),
        "power_w_peak": power_peak,
        "source": mean_source,
    }


def _build_tooltip_data(raw_dir: Path) -> dict[str, str]:
    """Build a map of heading-keyword → tooltip HTML."""
    tt: dict[str, str] = {}

    sg = _read_json(raw_dir / "sglang_summary.json")
    if sg:
        # Bug fix: ai_op_decode_tok_s is the active-window mean of the gen_throughput
        # gauge and can be 0 when the gauge decays after the benchmark finishes.
        # Fall back to gen_tp_peak (the max observed) so the tooltip always shows
        # a meaningful throughput number.
        _tp_display = sg.get('ai_op_decode_tok_s') or sg.get('gen_tp_peak', 0)
        tt["sglang inference"] = (
            f"<b>TTFT</b> {sg.get('server_ttft_ms', 0):.0f}ms · "
            f"<b>TPOT</b> {sg.get('server_itl_ms', 0):.1f}ms · "
            f"<b>E2E</b> {sg.get('server_e2e_ms', 0):.0f}ms<br>"
            f"<b>Throughput</b> {_tp_display:.1f} tok/s · "
            f"<b>Cache hit</b> {sg.get('cache_hit_rate_realtime_pct', 0):.1f}%<br>"
            f"<b>Samples</b>: {sg.get('collection_samples', 0)} scrapes"
        )
        tt["ai operation phase"] = (
            f"<b>TTFT</b> {sg.get('server_ttft_ms', 0):.0f}ms "
            f"(prefill) · <b>TPOT</b> {sg.get('server_itl_ms', 0):.1f}ms (decode)<br>"
            f"Prefill compute / cache tokens: "
            f"{sg.get('rt_prefill_compute_tokens', 0):,} / "
            f"{sg.get('rt_prefill_cache_tokens', 0):,}"
        )
        tt["latency breakdown"] = (
            f"<b>TTFT</b> {sg.get('server_ttft_ms', 0):.0f}ms · "
            f"<b>TPOT</b> {sg.get('server_itl_ms', 0):.1f}ms · "
            f"<b>E2E</b> {sg.get('server_e2e_ms', 0):.0f}ms<br>"
            f"Decode tokens: {sg.get('rt_decode_tokens', 0):,}"
        )

    gpu = _read_json(raw_dir / "gpu_summary.json")
    if gpu:
        gpu_tt = _gpu_tooltip_stats_from_timeseries(raw_dir, gpu)
        _gpu_label = (
            f"<b>{gpu_tt.get('gpu_count', 0)} active GPUs</b>"
            if gpu_tt.get('raw_gpu_count', 0) and gpu_tt.get('gpu_count', 0) != gpu_tt.get('raw_gpu_count', 0)
            else f"<b>{gpu_tt.get('gpu_count', 0)} GPUs</b>"
        )
        tt["gpu utilisation"] = (
            f"{_gpu_label}<br>"
            f"<b>Util</b> mean {gpu_tt.get('gpu_util_mean', 0):.1f}%, "
            f"peak {gpu_tt.get('gpu_util_peak', 0):.1f}%<br>"
            f"<b>HBM</b> {gpu_tt.get('hbm_used_mb_mean', 0)/1024:.1f} GB / "
            f"{gpu_tt.get('hbm_total_mb_per_gpu', 0)/1024:.0f} GB "
            f"({gpu_tt.get('hbm_util_pct_mean', 0):.0f}%)<br>"
            f"<b>Power</b> peak {gpu_tt.get('power_w_peak', 0):.0f} W/GPU<br>"
            f"<span style='color:#94a3b8'>Source: {gpu_tt.get('source', 'summary')}</span>"
        )
        tt["hbm (gpu) bandwidth"] = (
            f"<b>HBM util</b> {gpu.get('hbm_util_pct_mean', 0):.0f}% · "
            f"DCGM active {gpu.get('dcgm_hbm_bw_active_pct', 0):.0f}%<br>"
            f"<b>HBM used</b> {gpu.get('hbm_used_mb_mean', 0)/1024:.1f} GB mean"
        )

    smart = _read_json(raw_dir / "smart_summary.json")
    if smart:
        tt["nvme io"] = (
            f"<b>{smart.get('model', '?')}</b> {smart.get('capacity_gb', 0)} GB<br>"
            f"<b>Temp</b> {smart.get('temperature_c', 0)}°C · "
            f"<b>WAF</b> {smart.get('waf', 0):.2f}<br>"
            f"<b>Lifetime used</b> {smart.get('lifetime_pct_used', 0)}%"
        )

    bt = _read_json(raw_dir / "summary.json")
    if bt:
        tt["ssd read workload"] = (
            f"<b>{bt.get('read_events', 0):,}</b> read events<br>"
            f"<b>R BW</b> {bt.get('read_bw_mb_s_mean', 0):.1f} MB/s · "
            f"<b>R IOPS</b> {bt.get('read_iops_mean', 0):.0f}<br>"
            f"<b>Read bytes</b> {bt.get('read_bytes_total', 0)/1e9:.2f} GB"
        )
        tt["ssd write workload"] = (
            f"<b>{bt.get('write_events', 0):,}</b> write events<br>"
            f"<b>W BW</b> {bt.get('write_bw_mb_s_mean', 0):.1f} MB/s · "
            f"<b>W IOPS</b> {bt.get('write_iops_mean', 0):.0f}<br>"
            f"<b>Write bytes</b> {bt.get('write_bytes_total', 0)/1e9:.2f} GB"
        )
        tt["trim / discard"] = (
            f"<b>TRIM events</b>: {bt.get('trim_events', 0):,}<br>"
            f"<b>TRIM IOPS</b> {bt.get('trim_iops_mean', 0):.2f} mean<br>"
            f"<b>TRIM bytes</b> {bt.get('trim_bytes_total', 0)/1e9:.2f} GB"
        )
        tt["nvme io workload"] = (
            f"<b>{bt.get('total_events', 0):,}</b> total block IOs<br>"
            f"<b>R:W ratio</b> {bt.get('rw_ratio', 0):.2f}× · "
            f"R={bt.get('read_events', 0):,} W={bt.get('write_events', 0):,} "
            f"T={bt.get('trim_events', 0):,}<br>"
            f"<b>Duration</b> {bt.get('duration_sec', 0):.0f} sec"
        )

    nvm = _read_csv_summary(raw_dir / "nvme_driver_timeseries.csv")
    if nvm:
        rd = nvm.get("rd_iops", {})
        wr = nvm.get("wr_iops", {})
        util = nvm.get("io_util_pct", {})
        tt["nvme io deep profiling"] = (
            f"<b>R IOPS</b> mean {rd.get('mean', 0):.0f}, "
            f"peak {rd.get('max', 0):.0f}<br>"
            f"<b>W IOPS</b> mean {wr.get('mean', 0):.0f}, "
            f"peak {wr.get('max', 0):.0f}<br>"
            f"<b>Util</b> mean {util.get('mean', 0):.1f}%, "
            f"peak {util.get('max', 0):.1f}%"
        )

    vm = _read_csv_summary(raw_dir / "vmstat_timeseries.csv")
    if vm:
        pswpin = vm.get("pswpin", {})
        pswpout = vm.get("pswpout", {})
        pgmaj = vm.get("pgmajfault", {})
        tt["swap storm"] = (
            f"<b>Swap-in</b> peak {pswpin.get('max', 0):.0f} pages<br>"
            f"<b>Swap-out</b> peak {pswpout.get('max', 0):.0f} pages<br>"
            f"<b>Major faults</b> peak {pgmaj.get('max', 0):.0f}"
        )
        tt["storage i/o"] = tt.get("storage i/o", "") + (
            f"<b>Major faults</b> peak {pgmaj.get('max', 0):.0f}/sec"
        )

    return tt


def _match_tooltip(heading: str, tooltips: dict[str, str]) -> str:
    h = heading.lower()
    for key, val in tooltips.items():
        if key in h:
            return val
    return ""


# ─── HTML rewriting ──────────────────────────────────────────────────────────
DARK_THEME_CSS = """
<style id="amoprof-enhancer-theme">
  /* ── Dark theme overrides (post-processor injected) ───────────────────── */
  body {
    background: #0f172a !important;
    color: #e2e8f0 !important;
  }
  /* Card containers: dark gray panel with strong borders */
  .card, div[style*="background:#fff"], div[style*="background:white"],
  div[style*="background:#ffffff"] {
    background: #1e293b !important;
    border-color: #475569 !important;
    color: #e2e8f0 !important;
  }
  /* ── Metric Derivations table: readable dark-mode cells ─────────────────
     The div[background:#ffffff] rule above turns td cell backgrounds dark,
     but the td's own inline color (#64748b, #475569, #334155) remains set
     for a light background — producing dark-on-dark invisible text.
     Fix: override td/th backgrounds to a readable dark shade and force
     text to a light slate so every column is legible in dark mode.      */
  /* Dark table cells for all cards EXCEPT the setup-panel (which keeps a light bg) */
  .card:not(.setup-panel) table td,
  .card:not(.setup-panel) table th {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-color: #334155 !important;
  }
  .card:not(.setup-panel) table tr:nth-child(even) td {
    background: #1e3a5b !important;
  }
  .card:not(.setup-panel) table td[style*="font-weight:600"],
  .card:not(.setup-panel) table td[style*="font-weight: 600"] {
    color: #f1f5f9 !important;
  }
  .card:not(.setup-panel) table td code {
    color: #7dd3fc !important;
    background: rgba(15,23,42,0.55) !important;
  }
  .card:not(.setup-panel) table th {
    background: #0f172a !important;
    color: #94a3b8 !important;
  }
  /* Setup-panel keeps a bright, light background in both light and dark mode */
  .setup-panel,
  .setup-panel table,
  .setup-panel table td,
  .setup-panel table th,
  .setup-panel .setup-info-box,
  .setup-panel div {
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #c7d2fe !important;
  }
  .setup-panel table tr:nth-child(even) td {
    background: #f0f4ff !important;
  }
  .setup-panel table th {
    background: #eef2ff !important;
    color: #3730a3 !important;
    font-weight: 700 !important;
  }
  .setup-panel h2, .setup-panel .setup-info-title {
    color: #1e1b4b !important;
  }
  /* Headings */
  h1, h2, h3, h4 {
    color: #f1f5f9 !important;
  }
  /* Subheadings inside cards (originally dark slate) become light */
  div[style*="color:#0f172a"], div[style*="color: #0f172a"],
  div[style*="color:#334155"], div[style*="color: #334155"],
  div[style*="color:#1e293b"] {
    color: #cbd5e1 !important;
  }
  /* Inline code spans (data source citations) — dim but readable */
  code, span[style*="background:#eff6ff"], span[style*="background:#faf5ff"] {
    background: #334155 !important;
    color: #e0f2fe !important;
  }
  /* KPI cards — keep distinct from outer cards */
  div[style*="border:1px solid #e2e8f0"] {
    background: #0f172a !important;
    border-color: #475569 !important;
  }
  /* The labels (small uppercase) inside KPIs */
  div[style*="color:#64748b"] {
    color: #94a3b8 !important;
  }


  /* ── Dark theme contrast fixes for generated inline styles ───────────── */
  body > div:first-of-type span[style*="font-size:22px"][style*="font-weight:900"] {
    color: #f8fafc !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.35);
  }
  body > div:first-of-type span[style*="font-size:22px"][style*="color:#4f46e5"] {
    color: #a5b4fc !important;
  }
  .card:not(.setup-panel) span[style*="color:#0f172a"],
  .card:not(.setup-panel) span[style*="color: #0f172a"],
  .card:not(.setup-panel) span[style*="color:#334155"],
  .card:not(.setup-panel) span[style*="color: #334155"],
  .card:not(.setup-panel) span[style*="color:#1e293b"],
  .card:not(.setup-panel) td[style*="color:#0f172a"],
  .card:not(.setup-panel) td[style*="color: #0f172a"],
  .card:not(.setup-panel) td[style*="color:#334155"],
  .card:not(.setup-panel) td[style*="color: #334155"],
  .card:not(.setup-panel) td[style*="color:#1e293b"],
  .card:not(.setup-panel) td[style*="color:#475569"],
  .card:not(.setup-panel) td[style*="color: #475569"],
  .card:not(.setup-panel) td[style*="color:#64748b"],
  .card:not(.setup-panel) td[style*="color: #64748b"],
  .card:not(.setup-panel) b[style*="color:#0f172a"],
  .card:not(.setup-panel) b[style*="color: #0f172a"],
  .card:not(.setup-panel) div[style*="color:#0f172a"],
  .card:not(.setup-panel) div[style*="color: #0f172a"],
  .card:not(.setup-panel) div[style*="color:#334155"],
  .card:not(.setup-panel) div[style*="color: #334155"],
  .card:not(.setup-panel) div[style*="color:#1e293b"] {
    color: #cbd5e1 !important;
  }
  /* Formula table code blocks: light on dark */
  .card:not(.setup-panel) td code[style*="color:#1e3a5f"],
  .card:not(.setup-panel) td code[style*="color: #1e3a5f"] {
    color: #7dd3fc !important;
    background: rgba(15,23,42,0.4) !important;
  }

  /* Keep setup/config section intentionally light even in dark theme. */
  .setup-panel {
    background: linear-gradient(135deg, #eef2ff 0%, #ffffff 55%, #f8fafc 100%) !important;
    border: 1px solid #a5b4fc !important;
    border-left: 6px solid #818cf8 !important;
    color: #0f172a !important;
    box-shadow: 0 6px 18px rgba(15,23,42,0.35) !important;
  }
  .setup-panel h2,
  .setup-panel .setup-info-title {
    color: #1e1b4b !important;
    text-shadow: none !important;
  }
  .setup-panel .setup-table,
  .setup-panel .setup-info-box,
  .setup-panel table {
    background: #ffffff !important;
    color: #0f172a !important;
  }
  .setup-panel th,
  .setup-panel td,
  .setup-panel .setup-key,
  .setup-panel .setup-value,
  .setup-panel p,
  .setup-panel div,
  .setup-panel span {
    color: #0f172a !important;
  }
  .setup-panel .setup-row-defaulted .setup-key,
  .setup-panel .setup-row-defaulted .setup-value {
    background: #fff7ed !important;
    color: #7c2d12 !important;
  }
  .setup-panel code {
    background: #eef2ff !important;
    color: #3730a3 !important;
  }

  /* Wrapper around each chart image — adds tooltip parent */
  figure.amoprof-chart {
    position: relative;
    margin: 0 0 8px 0;
    padding: 0;
    border-radius: 10px;
    overflow: hidden;
    background: #0f172a;
    border: 1px solid #475569;
    cursor: help;
  }
  figure.amoprof-chart img {
    width: 100%;
    display: block;
    border-radius: 9px;
    transition: transform 0.18s ease;
  }
  figure.amoprof-chart:hover img {
    transform: scale(1.005);
  }
  /* Tooltip — small badge in the top-right that reveals on hover */
  figure.amoprof-chart .amoprof-tt-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(15,23,42,0.85);
    color: #cbd5e1;
    border: 1px solid #475569;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    backdrop-filter: blur(2px);
    pointer-events: none;
    opacity: 0.65;
    transition: opacity 0.15s;
  }
  figure.amoprof-chart:hover .amoprof-tt-badge {
    opacity: 1;
  }
  figure.amoprof-chart .amoprof-tt {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0.96);
    min-width: 280px;
    max-width: 480px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
    border: 1px solid #64748b;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.05) inset;
    opacity: 0;
    pointer-events: none;
    visibility: hidden;
    transition: opacity 0.18s, transform 0.18s, visibility 0.18s;
    z-index: 10;
  }
  figure.amoprof-chart:hover .amoprof-tt {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
    visibility: visible;
  }
  figure.amoprof-chart .amoprof-tt b {
    color: #fbbf24;
  }
  /* Per-chart explanation block — sits below the chart */
  .amoprof-explain {
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid #22d3ee;
    margin: 10px 0 8px 0;
    padding: 10px 14px 12px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 12.5px;
    line-height: 1.6;
    color: #cbd5e1;
  }
  .amoprof-explain .amoprof-explain-summary {
    display: block;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
    color: #22d3ee;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .amoprof-explain p {
    margin: 6px 0 0 0;
  }
  /* Existing <details> blocks (formula reveals) — readable on dark */
  details.fml {
    background: rgba(15, 23, 42, 0.7) !important;
    border-color: #475569 !important;
  }
  details.fml summary {
    color: #f1f5f9 !important;
    background: rgba(30, 41, 59, 0.6) !important;
  }
  details.fml .fml-body, details.fml code {
    color: #e2e8f0 !important;
    background: rgba(0, 0, 0, 0.25) !important;
  }
  details.fml .fml-note {
    color: #94a3b8 !important;
  }
  /* Scrollbars */
  ::-webkit-scrollbar { width: 12px; background: #0f172a; }
  ::-webkit-scrollbar-thumb { background: #475569; border-radius: 6px; }
  ::-webkit-scrollbar-thumb:hover { background: #64748b; }
</style>
"""


LIGHT_THEME_CSS = """
<style id="amoprof-enhancer-theme">
  /* ── Light theme: keep original amoprof colors, add tooltip+explanation styling ── */

  /* Wrapper around each chart image */
  figure.amoprof-chart {
    position: relative;
    margin: 0 0 8px 0;
    padding: 0;
    border-radius: 10px;
    overflow: hidden;
    cursor: help;
    border: 1px solid #cbd5e1;
  }
  figure.amoprof-chart img {
    width: 100%;
    display: block;
    border-radius: 9px;
    transition: transform 0.18s ease;
  }
  figure.amoprof-chart:hover img {
    transform: scale(1.003);
  }
  figure.amoprof-chart .amoprof-tt-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(255,255,255,0.92);
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    pointer-events: none;
    opacity: 0.7;
    transition: opacity 0.15s;
  }
  figure.amoprof-chart:hover .amoprof-tt-badge {
    opacity: 1;
  }
  figure.amoprof-chart .amoprof-tt {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0.96);
    min-width: 280px;
    max-width: 480px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.55;
    box-shadow: 0 8px 24px rgba(15,23,42,0.4);
    opacity: 0;
    pointer-events: none;
    visibility: hidden;
    transition: opacity 0.18s, transform 0.18s, visibility 0.18s;
    z-index: 10;
  }
  figure.amoprof-chart:hover .amoprof-tt {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
    visibility: visible;
  }
  figure.amoprof-chart .amoprof-tt b {
    color: #fbbf24;
  }
  /* Per-chart explanation — light variant uses indigo accent on white */
  .amoprof-explain {
    background: linear-gradient(180deg, #f1f5f9 0%, #ffffff 100%);
    border-left: 3px solid #4f46e5;
    margin: 10px 0 14px 0;
    padding: 10px 14px 12px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 12.5px;
    line-height: 1.6;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  }
  .amoprof-explain .amoprof-explain-summary {
    display: block;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
    color: #4f46e5;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .amoprof-explain p {
    margin: 6px 0 0 0;
  }
</style>
"""


def _wrap_chart_image(match: re.Match, heading: str,
                       tooltips: dict[str, str]) -> str:
    """Wrap an <img> tag in a <figure> with hover tooltip + badge."""
    img_tag = match.group(0)
    tt_html = _match_tooltip(heading, tooltips)
    badge = "📊 hover for stats" if tt_html else "📊 chart"
    tt_div = f'<div class="amoprof-tt">{tt_html}</div>' if tt_html else ""
    return (f'<figure class="amoprof-chart">'
            f'<span class="amoprof-tt-badge">{badge}</span>'
            f'{tt_div}'
            f'{img_tag}'
            f'</figure>')


def _inject_explanations_and_tooltips(html: str, raw_dir: Path | None) -> str:
    """Find each <h2> heading, then locate the chart-containing <img> after it
    and wrap it. Also inject explanation block after the heading.

    We process the HTML by splitting on <h2> tags to delimit sections.
    Charts inside a section get the section's tooltip + explanation.
    """
    tooltips: dict[str, str] = {}
    if raw_dir is not None:
        try:
            tooltips = _build_tooltip_data(raw_dir)
        except Exception as e:
            log.warning("tooltip build failed: %s", e)

    # Pattern: split HTML at <h2 ...>...</h2> boundaries while keeping the H2 text
    # We will walk through and rewrite each section in place.

    h2_re = re.compile(r'(<h2[^>]*>(.*?)</h2>)', re.IGNORECASE | re.DOTALL)
    img_re = re.compile(r'<img\s+src="data:image/png;base64,[^"]+"[^>]*/?>',
                         re.IGNORECASE | re.DOTALL)

    # Find all H2 positions
    sections = list(h2_re.finditer(html))
    if not sections:
        return html

    pieces: list[str] = []
    last_end = 0
    for idx, m in enumerate(sections):
        # Append everything up to and including the H2 tag
        pieces.append(html[last_end:m.end()])
        heading_text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        # Insert explanation block right after the H2
        expl = _match_explanation(heading_text)
        if expl:
            summary, body = expl
            # Convert plain-text \n\n into <p> tags
            body_html = "".join(f"<p>{p.strip()}</p>" for p in body.split("\n\n") if p.strip())
            pieces.append(
                f'<div class="amoprof-explain">'
                f'<span class="amoprof-explain-summary">📖 What this chart shows</span>'
                f'{summary}{body_html}'
                f'</div>'
            )
        # Determine the end of this section (start of next H2, or EOF)
        section_end = sections[idx + 1].start() if idx + 1 < len(sections) else len(html)
        section_body = html[m.end():section_end]
        # Wrap each <img> inside this section
        def _wrap(mi):
            return _wrap_chart_image(mi, heading_text, tooltips)
        section_body = img_re.sub(_wrap, section_body)
        pieces.append(section_body)
        last_end = section_end

    # Append any tail content (unlikely, but safe)
    pieces.append(html[last_end:])
    return "".join(pieces)


def _strip_enhancer_injections(html: str) -> str:
    """Remove all previously-injected enhancer elements from an HTML string.

    Removes:
      • <div class="amoprof-explain">…</div> blocks (explanation text)
      • <figure class="amoprof-chart">…</figure> wrappers (unwraps back to bare <img>)
      • <style id="amoprof-enhancer-theme">…</style> (theme CSS)

    Returns a clean HTML string that the enhancer will treat as un-enhanced,
    allowing fresh re-injection with updated explanation text.
    """
    # Remove explanation blocks
    html = re.sub(r'<div class="amoprof-explain"[^>]*>.*?</div>',
                  '', html, flags=re.DOTALL)
    # Unwrap figure wrappers: extract the <img> and discard the rest
    def _unwrap_figure(m: re.Match) -> str:
        img_m = re.search(r'<img\s[^>]*/?>|<img\s[^>]+>', m.group(0), re.DOTALL)
        return img_m.group(0) if img_m else ''
    html = re.sub(r'<figure class="amoprof-chart"[^>]*>.*?</figure>',
                  _unwrap_figure, html, flags=re.DOTALL)
    # Remove theme CSS block
    html = re.sub(r'<style id="amoprof-enhancer-theme">.*?</style>',
                  '', html, flags=re.DOTALL)
    return html


def enhance_report(html_path: Path,
                   raw_dir: Path | None = None,
                   theme: str = "dark") -> Path:
    """Post-process an amoprof HTML report file in place.

    Idempotent — running this multiple times on the same file is a no-op for
    explanations and tooltips (detected by checking for the marker class
    `amoprof-explain` already present in the file). Theme CSS injection is
    also guarded by a unique <style> id.

    To force a re-injection of updated explanation text (e.g. after upgrading
    the package), call :func:`re_enhance_report` instead.

    Returns the same path (file is overwritten).
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8", errors="replace")

    already_enhanced = ('class="amoprof-explain"' in html
                        or 'class="amoprof-chart"' in html
                        or 'id="amoprof-enhancer-theme"' in html)

    # Inject theme CSS into <head> (right before </head>) — guarded by style id
    if theme in ("dark", "light"):
        if 'id="amoprof-enhancer-theme"' not in html:
            inject = DARK_THEME_CSS if theme == "dark" else LIGHT_THEME_CSS
            if "</head>" in html:
                html = html.replace("</head>", inject + "\n</head>", 1)
            else:
                html = inject + html

    # Inject per-chart tooltips and explanations — skip if already done
    if not already_enhanced:
        html = _inject_explanations_and_tooltips(html, raw_dir)
    else:
        log.info("amoprof enhancer: file already enhanced — only theme refreshed")

    html_path.write_text(html, encoding="utf-8")
    log.info("amoprof enhancer: wrote %s (%d KB, theme=%s)",
             html_path, len(html) // 1024, theme)
    return html_path


def re_enhance_report(html_path: Path,
                      raw_dir: Path | None = None,
                      theme: str = "dark") -> Path:
    """Strip all previous enhancer injections and re-enhance from scratch.

    Use this when upgrading amoprof and you want the updated explanation text
    to be applied to an existing report file that was enhanced by an older
    version. Unlike :func:`enhance_report`, this function always re-injects.

    Returns the same path (file is overwritten).
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8", errors="replace")
    log.info("amoprof enhancer: stripping old injections from %s", html_path)
    html = _strip_enhancer_injections(html)
    html_path.write_text(html, encoding="utf-8")
    log.info("amoprof enhancer: re-enhancing %s", html_path)
    return enhance_report(html_path, raw_dir=raw_dir, theme=theme)
