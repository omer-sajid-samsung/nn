"""
AMOprof — AI Workload-Aware Storage Profiler
============================================
Maps AI operations (prefill, decode, checkpoint, weight-load, kv-evict, mixed)
to NVMe SSD I/O dimensions (BW, IOPS, latency, WAF, endurance).

Given: model name + HBM capacity + DRAM capacity + NVMe capacity + AI operation
Derives: which memory tier is exercised, which framework to use, which I/O
         pattern is generated, and collects SSD metrics into CSV.
"""

__version__ = '1.39.112'
__author__  = "AMOprof"
