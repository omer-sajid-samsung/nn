"""Shared L3 / HiCache backend resolver for AMOprof reports.

The SGLang HiCache movement counters (backuped/prefetched/load_back) are
logical tier-movement counters.  They do not by themselves say whether the
backing tier is L3 local SSD, Mooncake, CXL, or another remote backend.  This
module centralises that decision so Executive, Interactive, and End Report use
one interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_MISSING = {"", "unknown", "none", "n/a", "na", "null", "default", "disabled", "false", "0", "-", "—"}

@dataclass(frozen=True)
class L3BackendResolution:
    backend_class: str          # local_ssd | remote_mooncake | remote_storage | host_dram_only | unknown
    display_name: str           # Canonical visible tier label, e.g. L3 or L3.5
    logical_write_label: str    # what backuped_tokens_total means
    logical_read_label: str     # what prefetched means; load_back is diagnostic-only
    has_local_block_mapping: bool
    local_device: str = ""
    local_mount_path: str = ""
    local_storage_path: str = ""
    evidence: str = ""


def _norm(v: Any) -> str:
    return str(v if v is not None else "").strip()


def _is_set(v: Any) -> bool:
    s = _norm(v)
    return bool(s) and s.lower() not in _MISSING


def _get(setup: Mapping[str, Any], *keys: str) -> str:
    # exact first, then case-insensitive
    for k in keys:
        if k in setup and _is_set(setup.get(k)):
            return _norm(setup.get(k))
    lower = {str(k).lower(): v for k, v in setup.items()}
    for k in keys:
        v = lower.get(k.lower())
        if _is_set(v):
            return _norm(v)
    return ""


def _defaulted_fields(setup: Mapping[str, Any]) -> set[str]:
    s = _norm(setup.get("Defaulted fields", setup.get("defaulted_fields", "")))
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def _get_explicit(setup: Mapping[str, Any], *keys: str) -> str:
    defaulted = _defaulted_fields(setup)
    for k in keys:
        if k.lower() in defaulted:
            continue
        if k in setup and _is_set(setup.get(k)):
            return _norm(setup.get(k))
    lower = {str(k).lower(): v for k, v in setup.items()}
    for k in keys:
        if k.lower() in defaulted:
            continue
        v = lower.get(k.lower())
        if _is_set(v):
            return _norm(v)
    return ""


def resolve_l3_backend(setup: Mapping[str, Any] | None, launch_command: str | None = None) -> L3BackendResolution:
    setup = setup or {}
    launch = _norm(launch_command or _get(setup, "Launch command", "launch_command", "SGLang launch command", "sglang_launch_command"))

    # Use _get_explicit (not _get) so a DEFAULTED storage backend/type from the
    # fabricated reference profile does not get treated as real evidence. On a
    # Prometheus-only run with no setup_details, "HiCache storage backend=file"
    # is an assumption, not an observation, and must not force a local_ssd
    # classification (which would mislabel L3.5/AI-Memory-Node runs as L3 SSD).
    storage_type = _get_explicit(setup, "L3 storage type", "l3_storage_type", "L3 type", "l3_type", "Cache L3 (local storage) backend", "cache_l3_backend")
    backend = _get_explicit(setup, "HiCache storage backend", "hicache_storage_backend", "L3 backend", "l3_backend", "storage_backend", "hicache_backend")
    # Optional explicit tier naming override.  This is the preferred way to
    # disambiguate NVIDIA-style L3 vs L3.5 naming in reports:
    #   - L3   = local SSD/NVMe/file-backed storage tier
    #   - L3.5 = AI Memory Node / remote/shared/disaggregated backing tier
    explicit_tier = _get_explicit(
        setup,
        "KV cache tier", "kv_cache_tier", "KV backing tier", "kv_backing_tier",
        "L3 cache tier", "l3_cache_tier", "L3/L3.5 tier", "l3_l35_tier",
        "AMOprof cache tier", "amoprof_cache_tier", "Memory tier", "memory_tier"
    )
    device = _get_explicit(setup, "L3 Device", "L3 device", "L3 block device", "l3_device", "L3 (local storage) device", "l3_storage_device", "NVMe device", "ssd_device")
    mount = _get_explicit(setup, "L3 Mount Path", "L3 mount path", "L3 mount", "l3_mount_path", "L3 (local storage) mount path", "l3_storage_mount_path")
    path = _get_explicit(setup, "HiCache storage path", "hicache_storage_path", "L3 filesystem path", "L3 storage path", "l3_storage_path", "storage_path")

    blob = " ".join([storage_type, backend, device, mount, path, launch]).lower()
    # Do not let report terminology such as "L3 (AI Memory Node / remote storage)"
    # influence the resolver.  Only explicit backend/path/device/launch evidence
    # should classify a run as L3.  For local HiCache file storage, --ssd-device,
    # --file-storage-path, NVMe, or any concrete local mount/path is L3.
    launch_l = launch.lower()
    backend_l = backend.lower().strip()
    storage_l = storage_type.lower().strip()
    path_l = path.lower().strip()
    mount_l = mount.lower().strip()
    device_l = device.lower().strip()
    has_mapping = bool(device or mount or path)

    tier_l = explicit_tier.lower().strip()
    tier_blob = " ".join([tier_l, storage_l, backend_l]).lower()
    if tier_l in {"l3.5", "l35", "l3_5", "l3-5"} or any(x in tier_blob for x in ("ai memory", "ai-memory", "aimemory", "remote", "shared", "disaggregated", "mooncake", "moon cake")):
        return L3BackendResolution(
            "remote_storage", "L3.5 (AI Memory Node / remote storage)",
            "logical L3.5 write/offload from sglang_backuped_tokens_total",
            "logical L3.5 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            False, evidence="explicit L3.5 / AI Memory Node tier in setup_details")
    if tier_l == "l3" or any(x in tier_blob for x in ("local ssd", "local_ssd", "nvme", "ssd", "local storage", "local-storage")):
        return L3BackendResolution(
            "local_ssd", "L3 (SSD/local storage)",
            "logical L3 write/offload from sglang_backuped_tokens_total",
            "logical L3 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            has_mapping, local_device=device, local_mount_path=mount, local_storage_path=path,
            evidence="explicit L3 local SSD tier in setup_details")

    if any(x in blob for x in ("mooncake", "moon cake")):
        return L3BackendResolution(
            "remote_mooncake", "L3.5 (AI Memory Node / remote storage)",
            "logical L3.5 write/offload from sglang_backuped_tokens_total",
            "logical L3.5 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            False, evidence="mooncake in setup/launch")

    remote_strong = any(x in " ".join([storage_l, backend_l, launch_l, path_l, mount_l])
                        for x in ("s3://", "object://", "rdma", "nfs", "lustre", "weka", "ceph", "disaggregated", "ai-memory-node", "ai_memory_node"))                     or backend_l in {"remote", "remote_storage", "object", "s3", "nfs", "ceph", "weka", "lustre"}

    local_strong = bool(
        device_l or mount_l or path_l
        or "--ssd-device" in launch_l
        or "--file-storage-path" in launch_l
        or "hicache-storage-backend file" in launch_l
        or backend_l in {"file", "filesystem", "fs", "block", "disk", "nvme", "ssd", "local", "local_ssd"}
        or any(x in blob for x in ("nvme", "ssd", "/dev/", "file-storage-path", "file storage", "filesystem"))
    )

    # Strong remote technologies win only when present; generic word "remote"
    # inside a label does not.  Otherwise any concrete local evidence means L3.
    if remote_strong and not local_strong:
        return L3BackendResolution(
            "remote_storage", "L3.5 (AI Memory Node / remote storage)",
            "logical L3.5 write/offload from sglang_backuped_tokens_total",
            "logical L3.5 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            False, evidence="explicit remote/object/disaggregated backend in setup/launch")

    if local_strong:
        return L3BackendResolution(
            "local_ssd", "L3 (SSD/local storage)",
            "logical L3 write/offload from sglang_backuped_tokens_total",
            "logical L3 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            has_mapping, local_device=device, local_mount_path=mount, local_storage_path=path,
            evidence="explicit L3 local SSD/file/path/device evidence in setup/launch")

    if remote_strong:
        return L3BackendResolution(
            "remote_storage", "L3.5 (AI Memory Node / remote storage)",
            "logical L3.5 write/offload from sglang_backuped_tokens_total",
            "logical L3.5 read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
            False, evidence="explicit remote/object/disaggregated backend in setup/launch")

    hicache = _get(setup, "HiCache", "Hierarchical cache", "enable_hierarchical_cache", "hicache", "HiCache enabled")
    if hicache and hicache.lower() in {"l2", "host", "host_dram", "dram", "enabled-no-storage", "memory"}:
        return L3BackendResolution(
            "host_dram_only", "Host DRAM only",
            "not an L3 write; no backing L3 backend resolved",
            "not an L3 read; no backing L3 backend resolved",
            False, evidence="hicache enabled without resolved storage backend")

    return L3BackendResolution(
        "unknown", "Unresolved backing tier",
        "generic logical L3.5/HiCache backup from sglang_backuped_tokens_total",
        "generic logical L3.5/HiCache read/onboard from sglang_prefetched_tokens_total; sglang_load_back_tokens_total is diagnostic-only",
        False, evidence="no explicit backend evidence")

@dataclass(frozen=True)
class L3IoReconciliation:
    status: str                 # match | partial | mismatch | no_block | not_comparable | no_l3
    display_status: str
    note: str
    sglang_write_gb: float
    sglang_read_gb: float
    block_write_gb: float
    block_read_gb: float
    write_ratio: float
    read_ratio: float


def reconcile_l3_io(backend: L3BackendResolution, *,
                     sglang_write_gb: float = 0.0, sglang_read_gb: float = 0.0,
                     block_write_gb: float = 0.0, block_read_gb: float = 0.0,
                     blktrace_available: bool = False, tolerance: float = 0.35) -> L3IoReconciliation:
    """Compare SGLang logical L3 movement with local block telemetry.

    SGLang reports logical KV movement (tokens × KV bytes/token).  blktrace/
    blkparse/iostat report physical local block-device bytes.  They are directly
    comparable only when the resolved backend is local_ssd and setup maps L3 to
    a concrete local block device/mount/path.  The report should never merge
    the two silently; this helper returns an explicit status and note used by
    Executive, Interactive, and End Report sections.
    """
    sw, sr = max(float(sglang_write_gb or 0.0), 0.0), max(float(sglang_read_gb or 0.0), 0.0)
    bw, br = max(float(block_write_gb or 0.0), 0.0), max(float(block_read_gb or 0.0), 0.0)
    logical = sw + sr
    physical = bw + br
    wr = (bw / sw) if sw > 0 else (1.0 if bw == 0 else float('inf'))
    rr = (br / sr) if sr > 0 else (1.0 if br == 0 else float('inf'))
    if logical <= 0 and physical <= 0:
        return L3IoReconciliation('no_l3','No L3 activity','Neither SGLang nor local block telemetry shows L3 I/O in the selected window.',sw,sr,bw,br,wr,rr)
    if backend.backend_class != 'local_ssd' or not backend.has_local_block_mapping:
        return L3IoReconciliation('not_comparable','Not directly comparable',f'SGLang counters are logical {backend.display_name} movement; local block-device bytes are not treated as L3 because no resolved L3 local SSD block mapping is available.',sw,sr,bw,br,wr,rr)
    if not blktrace_available and physical <= 0:
        return L3IoReconciliation('no_block','No physical block trace','SGLang shows logical local-SSD L3 movement, but blktrace/blkparse physical bytes are missing or zero for the window.',sw,sr,bw,br,wr,rr)
    # Directional tolerance with a tiny absolute floor to avoid noise-only mismatches.
    def ok(logical_gb, physical_gb):
        if logical_gb <= 0:
            return physical_gb <= 0.01
        return abs(physical_gb - logical_gb) <= max(0.05, tolerance * logical_gb)
    w_ok, r_ok = ok(sw,bw), ok(sr,br)
    if w_ok and r_ok:
        return L3IoReconciliation('match','Consistent','SGLang logical L3 bytes and L3 local SSD physical bytes agree within tolerance for read/write directions.',sw,sr,bw,br,wr,rr)
    if (sw > 0 and bw <= 0.01) or (sr > 0 and br <= 0.01):
        return L3IoReconciliation('partial','Partial capture','SGLang shows logical local-SSD L3 movement, but one or more physical blktrace directions are missing/zero; report shows provenance separately and does not treat missing physical bytes as zero logical activity.',sw,sr,bw,br,wr,rr)
    return L3IoReconciliation('mismatch','Mismatch','SGLang logical local-SSD L3 bytes and physical blktrace bytes differ beyond tolerance; check traced device, time-window alignment, dropped events, filesystem buffering, and blkparse completion-only filtering.',sw,sr,bw,br,wr,rr)

