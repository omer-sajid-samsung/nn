"""Tier terminology helpers for AI inference memory hierarchy.

AMOprof uses:
  L1   = HBM
  L2   = host DRAM
  L3   = node-local storage / L3 local SSD / NVMe
  L3 = AI Memory Node / shared context-memory flash tier / remote KV backing tier
  L4   = durable object/file storage

NVIDIA's 2026 CMX/context-memory material describes a G3.5 layer between
compute-node memory/local tiers and durable storage for shared KV/context memory.
AMOprof reports this as "L3 (AI Memory Node)" for consistency with the
user-facing AI memory hierarchy used in these reports.
"""

L3_LOCAL = "L3 (local storage)"
L35_AIMEM = "L3 (AI Memory Node)"
L35_REMOTE = "L3 (AI Memory Node / remote storage)"

def normalize_tier_label(text: str) -> str:
    """Replace old ambiguous labels with explicit L3 terminology."""
    if text is None:
        return text
    s = str(text)
    replacements = [
        ("L3 (AI Memory Node / remote storage)", L35_REMOTE),
        ("L3 (AI Memory Node / remote storage)", L35_REMOTE),
        ("L3 (AI Memory Node / remote storage)", L35_REMOTE),
        ("L3 storage / AI Memory Node", L35_REMOTE),
        ("L3", L35_REMOTE),
        ("L3", L35_REMOTE),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s

def tier_from_backend(resolver_class: str | None = None,
                      backend_kind: str | None = None,
                      physical_local: bool = False) -> str:
    rc = (resolver_class or "").lower()
    bk = (backend_kind or "").lower()
    if physical_local or rc in {"local_ssd", "local_nvme", "mapped_local_ssd"}:
        return L3_LOCAL
    if rc in {"remote_storage", "mooncake", "ai_memory_node", "remote"}:
        return L35_AIMEM
    if any(x in bk for x in ("remote", "mooncake", "ai_memory", "ai-memory", "context")):
        return L35_AIMEM
    return L35_REMOTE
