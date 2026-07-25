"""
amoprof/preflight.py — Real sanity checks for every collector before the
main collection run starts.

Unlike a simple "does the binary exist" check, each function here actually
INVOKES the tool briefly (≤2 sec) to confirm it can produce output. This
catches failure modes that pre-installation checks miss:

  - blktrace: present but no CAP_SYS_ADMIN / no debugfs / no buffer space
  - biosnoop-bpfcc: present but BCC mismatch / no BTF / can't attach probes
  - AMDuProfPcm: present but amdProfilingDriver not loaded / no MSR access
  - DCGM/nvidia-smi: present but driver mismatch / no GPUs visible
  - SGLang scrape: port reachable but no /metrics endpoint / empty response

Run order is intentional: cheap structural checks first (file existence,
device nodes), then short tool invocations.

All check functions are independent and return a CheckResult with a
uniform shape so the orchestrator can format and aggregate them.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import json
import re
import re
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("amoprof.preflight")

# How long to allow each tool's smoke test before we give up
TOOL_SMOKE_TIMEOUT_S = 4.0
# blktrace needs a real (short) capture window; anything below ~0.6s tends to
# produce no events on idle systems and looks like a failure.
BLKTRACE_SMOKE_DURATION_S = 1


@dataclasses.dataclass
class CheckResult:
    name:    str
    status:  str         # 'pass' | 'fail' | 'warn' | 'skipped'
    detail:  str         # human-readable
    fix:     str = ""    # remediation hint, blank if status==pass
    impact:  list[str] = dataclasses.field(default_factory=list)


def _run(cmd: list[str], timeout: float = TOOL_SMOKE_TIMEOUT_S,
         use_sudo: bool = False) -> tuple[int, str, str]:
    """Run a subprocess with timeout. Returns (rc, stdout, stderr).

    Returns rc=-1 on timeout, -2 on file-not-found, -3 on other exceptions.
    """
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"timeout after {timeout}s"
    except FileNotFoundError:
        return -2, "", "binary not found"
    except Exception as e:
        return -3, "", repr(e)


def _block_device_base(device: str | None) -> str:
    """Return /dev/nvme7n1 -> nvme7n1; empty when no device."""
    if not device:
        return ""
    return os.path.basename(str(device).strip())


def _biosnoop_disk_matches(line: str, device: str | None) -> bool:
    """Return True when a biosnoop data line refers to the target disk/partition."""
    base = _block_device_base(device)
    if not base:
        return False
    parts = line.split()
    # BCC biosnoop with -t: TIME(s) COMM PID DISK T SECTOR BYTES LAT(ms)
    # Without -t:           COMM PID DISK T SECTOR BYTES LAT(ms)
    # Be permissive and look for any token equal to disk or partition name.
    for tok in parts:
        tok = tok.strip()
        if tok == base:
            return True
        if base.startswith("nvme") and tok.startswith(base + "p"):
            return True
        if re.match(r"^[a-z]+[a-z]$", base) and tok.startswith(base) and tok[len(base):].isdigit():
            return True
    return False


def _biosnoop_supports_t_option(binary: str) -> bool:
    """Detect whether this biosnoop binary accepts -t."""
    try:
        r = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=5)
        help_txt = (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        try:
            r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
            help_txt = (r.stdout or "") + "\n" + (r.stderr or "")
        except Exception:
            return False
    return bool(re.search(r"(^|[\s,\[])-t([\s,\],]|$)", help_txt))


def _build_biosnoop_smoke_cmd(binary: str) -> tuple[list[str], bool]:
    cmd = [binary]
    supports_t = _biosnoop_supports_t_option(binary)
    if supports_t:
        cmd.append("-t")
    return cmd, supports_t


def _trigger_readonly_block_io(device: str, use_sudo: bool = True) -> tuple[int, str, str]:
    """Trigger a tiny read-only direct read to make biosnoop emit an event.

    This reads 256 KiB from the target block device into /dev/null. It does not
    write to the device.
    """
    cmd = ["dd", f"if={device}", "of=/dev/null", "bs=4096", "count=64",
           "iflag=direct", "status=none"]
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout while triggering read-only direct I/O"
    except Exception as e:
        return -2, "", repr(e)



def _blktrace_is_ebusy(stderr_text: str) -> bool:
    """Detect kernel EBUSY from blktrace setup.

    Typical stderr:
      BLKTRACESETUP(2) /dev/nvme6n1 failed: 16/Device or resource busy
    """
    t = (stderr_text or "").lower()
    return ("blktracesetup" in t and ("16/device" in t or "resource busy" in t or "device or resource busy" in t))


def _blktrace_kill_trace(device: str, blktrace_bin: str = "blktrace",
                         use_sudo: bool = True) -> tuple[int, str, str, list[str]]:
    """Ask blktrace to stop any existing trace attached to this device.

    This is the standard recovery for BLKTRACESETUP EBUSY, usually caused by a
    stale or still-running blktrace from an interrupted earlier run.
    """
    cmd = [blktrace_bin, "-d", device, "-k"]
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode, r.stdout or "", r.stderr or "", cmd
    except subprocess.TimeoutExpired:
        return -1, "", "timeout while running blktrace -k", cmd
    except Exception as e:
        return -2, "", repr(e), cmd


def _blktrace_smoke_capture(device: str, td: str, blktrace_bin: str,
                            use_sudo: bool) -> tuple[int, str, str, list[str]]:
    cmd = [blktrace_bin, "-d", device, "-w", str(BLKTRACE_SMOKE_DURATION_S),
           "-o", "smoke", "-D", td]
    rc, out, err = _run(cmd, timeout=BLKTRACE_SMOKE_DURATION_S + 3.0, use_sudo=use_sudo)
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    return rc, out, err, cmd



# ─── Required-baseline checks ───────────────────────────────────────────────
def check_ssd_device(device: str) -> CheckResult:
    """The SSD device path must be a real block device that exists."""
    p = Path(device)
    if not p.exists():
        return CheckResult(
            "SSD device", "fail",
            detail=f"{device} does not exist",
            fix=f"Verify the device name. List with: lsblk -d -o NAME,SIZE,MODEL",
            impact=["all NVMe charts (no device to monitor)"],
        )
    try:
        # /dev/nvme0n1 should be a block special file
        mode = p.lstat().st_mode
        if not (mode & 0o170000) == 0o060000:  # S_IFBLK
            return CheckResult(
                "SSD device", "warn",
                detail=f"{device} exists but is not a block device",
                fix="Pass the block-device path (e.g. /dev/nvme0n1, not a mount point)",
            )
    except Exception as e:
        return CheckResult("SSD device", "warn",
                            detail=f"could not stat {device}: {e}")
    return CheckResult("SSD device", "pass", detail=f"{device} is a block device")


def check_iostat() -> CheckResult:
    """iostat from sysstat: needed for the iostat_timeseries.csv fallback."""
    if not shutil.which("iostat"):
        return CheckResult(
            "iostat (sysstat)", "warn",
            detail="iostat not in PATH",
            fix="apt install sysstat  (or yum install sysstat)",
            impact=["fallback iostat-based NVMe driver timeseries"],
        )
    rc, out, err = _run(["iostat", "-V"])
    if rc != 0:
        return CheckResult("iostat (sysstat)", "warn",
                            detail=f"iostat -V failed: {err.strip()[:80]}")
    return CheckResult("iostat (sysstat)", "pass",
                        detail=out.splitlines()[0] if out else "ok")


def check_nvidia_gpu() -> CheckResult:
    """nvidia-smi: confirms driver is loaded and at least one GPU is visible."""
    if not shutil.which("nvidia-smi"):
        return CheckResult(
            "nvidia-smi (NVIDIA driver)", "fail",
            detail="nvidia-smi not in PATH",
            fix="Install NVIDIA driver; verify with: nvidia-smi",
            impact=["GPU utilization, HBM occupancy, power, temperature timelines"],
        )
    rc, out, err = _run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total",
         "--format=csv,noheader"], timeout=5.0)
    if rc != 0:
        return CheckResult(
            "nvidia-smi (NVIDIA driver)", "fail",
            detail=f"nvidia-smi failed (rc={rc}): {err.strip()[:120]}",
            fix="Check 'nvidia-smi' manually; reload driver if it errors",
            impact=["GPU charts (driver/runtime mismatch)"],
        )
    gpus = [line for line in out.splitlines() if line.strip()]
    if not gpus:
        return CheckResult(
            "nvidia-smi (NVIDIA driver)", "fail",
            detail="nvidia-smi ran but reports 0 GPUs",
            fix="Container may not have GPU access. Check --gpus arg/runtime config",
            impact=["GPU charts (no GPUs visible)"],
        )
    return CheckResult("nvidia-smi (NVIDIA driver)", "pass",
                        detail=f"{len(gpus)} GPU(s): {gpus[0].split(',')[1].strip()}")


def check_dcgm() -> CheckResult:
    """DCGM via dcgmi — the deeper GPU metrics path. nvidia-smi alone covers
    basic util/HBM/power; DCGM unlocks DRAM_ACTIVE / TENSOR_ACTIVE etc."""
    if not shutil.which("dcgmi"):
        return CheckResult(
            "dcgmi (DCGM)", "warn",
            detail="dcgmi not in PATH — basic GPU charts will work via nvidia-smi",
            fix="Install datacenter-gpu-manager for DCGM_FI_PROF_* metrics",
            impact=["§F HBM-Active fraction (DCGM_FI_PROF_DRAM_ACTIVE)"],
        )
    rc, out, _ = _run(["dcgmi", "discovery", "-l"])
    if rc != 0:
        return CheckResult(
            "dcgmi (DCGM)", "warn",
            detail="dcgmi present but discovery failed (daemon not running?)",
            fix="sudo systemctl start nvidia-dcgm",
            impact=["§F HBM-Active fraction"],
        )
    return CheckResult("dcgmi (DCGM)", "pass", detail="daemon responding")


def check_vmstat() -> CheckResult:
    p = Path("/proc/vmstat")
    if not p.exists() or not os.access(p, os.R_OK):
        return CheckResult(
            "/proc/vmstat", "fail",
            detail="/proc/vmstat not readable",
            fix="Likely an unusual container — needs /proc mounted",
            impact=["swap storm analysis, page-fault rates"],
        )
    # Verify pswpin/pswpout/pgmajfault are present
    txt = p.read_text()
    have = sum(1 for k in ("pswpin", "pswpout", "pgmajfault") if k in txt)
    if have < 3:
        return CheckResult(
            "/proc/vmstat", "warn",
            detail=f"only {have}/3 expected keys present",
            impact=["swap storm chart may be incomplete"],
        )
    return CheckResult("/proc/vmstat", "pass", detail="all swap/fault keys present")


def check_sglang_endpoint(host: str, port: int) -> CheckResult:
    """Confirm SGLang's /metrics endpoint is reachable and returns Prom data."""
    import urllib.request as _ur
    import urllib.error as _ue
    url = f"http://{host}:{port}/metrics"
    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError as e:
        return CheckResult(
            "SGLang /metrics", "fail",
            detail=f"cannot connect to {host}:{port} ({e})",
            fix=f"Verify SGLang is running and listening on port {port}. "
                f"Try: curl http://{host}:{port}/metrics | head",
            impact=["all SGLang inference charts (TTFT, TPOT, KV pool, hicache, ...)"],
        )
    try:
        req = _ur.Request(url, headers={"User-Agent": "amoprof-preflight"})
        with _ur.urlopen(req, timeout=3.0) as resp:
            head = resp.read(4096).decode("utf-8", errors="replace")
    except (_ue.URLError, _ue.HTTPError, OSError) as e:
        return CheckResult(
            "SGLang /metrics", "fail",
            detail=f"GET {url} failed: {e}",
            fix=f"SGLang must expose --enable-metrics. Test: curl {url} | head",
            impact=["all SGLang inference charts"],
        )
    # Sniff for SGLang-specific lines (sglang_*, vllm:*, tgi_*) — accept any
    n_sg = sum(1 for ln in head.splitlines()
                if ln and not ln.startswith("#")
                and any(p in ln for p in ("sglang_", "sglang:", "vllm", "tgi_")))
    if n_sg == 0:
        return CheckResult(
            "SGLang /metrics", "warn",
            detail=f"endpoint responds but no sglang_/vllm/tgi metrics in first 4KB",
            fix=f"Verify SGLang launched with --enable-metrics. "
                f"Got header: {head[:120]!r}",
            impact=["SGLang charts may be empty"],
        )
    return CheckResult("SGLang /metrics", "pass",
                        detail=f"{n_sg} inference metrics visible in first response")


# ─── Optional collectors — smoke tests ──────────────────────────────────────
def check_blktrace(device: str, blktrace_bin: str = "blktrace",
                    blkparse_bin: str = "blkparse",
                    use_sudo: bool = True) -> CheckResult:
    """blktrace smoke test:
      1. binary in PATH
      2. debugfs mounted
      3. short capture produces parseable events
    """
    if not shutil.which(blktrace_bin):
        return CheckResult(
            "blktrace (--enable-blktrace)", "fail",
            detail=f"{blktrace_bin} not in PATH",
            fix="apt install blktrace",
            impact=["§C/§D/§H request-level NVMe charts"],
        )
    if not shutil.which(blkparse_bin):
        return CheckResult(
            "blktrace (--enable-blktrace)", "fail",
            detail=f"{blkparse_bin} not in PATH",
            fix="apt install blktrace",
            impact=["§C/§D/§H request-level NVMe charts"],
        )
    # debugfs is the kernel-side dependency
    debugfs_p = Path("/sys/kernel/debug")
    if not debugfs_p.is_dir():
        return CheckResult(
            "blktrace (--enable-blktrace)", "fail",
            detail="/sys/kernel/debug does not exist",
            fix="mount -t debugfs none /sys/kernel/debug",
            impact=["§C/§D/§H charts"],
        )
    # Real smoke test: 1-second capture. If the kernel reports EBUSY, a stale
    # or still-active blktrace is attached to this device. Recover once with
    # `blktrace -d <dev> -k`, then retry. This avoids false permission-style
    # failures when the user is root and debugfs is fine.
    with tempfile.TemporaryDirectory(prefix="amoprof_preflight_") as td:
        prefix = Path(td) / "smoke"
        rc, _, err, smoke_cmd = _blktrace_smoke_capture(device, td, blktrace_bin, use_sudo)
        ebusy_recovered = False
        kill_detail = ""
        if rc != 0 and _blktrace_is_ebusy(err):
            k_rc, k_out, k_err, k_cmd = _blktrace_kill_trace(device, blktrace_bin, use_sudo)
            kill_detail = (
                f"EBUSY detected; ran cleanup: {' '.join(k_cmd)} "
                f"(rc={k_rc}, stderr={(k_err or '').strip()[:160]})"
            )
            if k_rc == 0:
                time.sleep(0.2)
                rc, _, err, smoke_cmd = _blktrace_smoke_capture(device, td, blktrace_bin, use_sudo)
                ebusy_recovered = (rc == 0)
        if rc != 0:
            if _blktrace_is_ebusy(err):
                return CheckResult(
                    "blktrace (--enable-blktrace)", "fail",
                    detail=(f"1-sec capture failed with EBUSY even after cleanup attempt "
                            f"(rc={rc}): {err.strip()[:180]}. {kill_detail}"),
                    fix=("Another blktrace is still active on this device. Stop the previous "
                         "AMOprof/blktrace process or run: blktrace -d "
                         f"{device} -k ; then retry. If intentional, disable --enable-blktrace."),
                    impact=["§C/§D/§H request-level NVMe charts"],
                )
            return CheckResult(
                "blktrace (--enable-blktrace)", "fail",
                detail=f"1-sec capture failed (rc={rc}): {err.strip()[:140]}",
                fix=("Common causes: not running as root (try sudo), "
                     "debugfs not mounted (mount -t debugfs none /sys/kernel/debug), "
                     "or blktrace not authorized (capabilities/CAP_SYS_ADMIN)."),
                impact=["§C/§D/§H request-level NVMe charts"],
            )
        # Confirm trace files were produced and contain events
        traces = list(Path(td).glob("smoke.blktrace.*"))
        if not traces:
            return CheckResult(
                "blktrace (--enable-blktrace)", "warn",
                detail="capture exited 0 but no trace files produced",
                impact=["§C/§D/§H charts"],
            )
        total_size = sum(t.stat().st_size for t in traces)
        extra = " after clearing stale/active prior trace" if 'ebusy_recovered' in locals() and ebusy_recovered else ""
        return CheckResult(
            "blktrace (--enable-blktrace)", "pass",
            detail=f"{len(traces)} trace file(s) captured, {total_size} bytes{extra}")



def _biosnoop_preflight_diag_dir() -> Path:
    """Directory for strict-sanity biosnoop preflight diagnostics.

    Preflight can abort before a metrics_run/raw directory exists, so these
    files intentionally live in a deterministic absolute path under CWD unless
    AMOPROF_PREFLIGHT_DIAG_DIR overrides it.
    """
    base = os.environ.get("AMOPROF_PREFLIGHT_DIAG_DIR")
    if base:
        d = Path(base).expanduser().resolve()
    else:
        d = (Path.cwd() / ".amoprof_preflight").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_biosnoop_preflight_diag(cmd: list[str],
                                   stdout_text: str = "",
                                   stderr_text: str = "",
                                   summary: dict | None = None) -> dict:
    d = _biosnoop_preflight_diag_dir()
    paths = {
        "command": d / "biosnoop_preflight_command.txt",
        "stdout": d / "biosnoop_preflight_stdout.txt",
        "stderr": d / "biosnoop_preflight_stderr.txt",
        "summary": d / "biosnoop_preflight_summary.json",
    }
    try:
        paths["command"].write_text(" ".join(cmd) + "\n", encoding="utf-8")
    except Exception:
        pass
    try:
        paths["stdout"].write_text(stdout_text or "", encoding="utf-8")
    except Exception:
        pass
    try:
        paths["stderr"].write_text(stderr_text or "", encoding="utf-8")
    except Exception:
        pass
    summary_obj = dict(summary or {})
    summary_obj.update({k + "_path": str(v.resolve()) for k, v in paths.items()})
    try:
        paths["summary"].write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {k + "_path": str(v.resolve()) for k, v in paths.items()}


def check_biosnoop(biosnoop_bin: str = "biosnoop",
                    use_sudo: bool = True,
                    device: str | None = None,
                    verify_event: bool = False) -> CheckResult:
    """biosnoop smoke check.

    Header emission proves that the BCC tool started and attached enough to
    print its column header. It does NOT prove that I/O events are emitted for
    the target SSD. When verify_event=True, AMOprof performs a tiny read-only
    direct read from --ssd-device while biosnoop is running and requires a
    matching event for that disk or partition.
    """
    import select as _sel
    import fcntl as _fcntl

    if not shutil.which(biosnoop_bin):
        return CheckResult(
            "biosnoop (--enable-biosnoop)", "fail",
            detail=f"{biosnoop_bin} not in PATH",
            fix="apt install bpfcc-tools  (Ubuntu) or yum install bcc-tools  (RHEL)",
            impact=["§I per-stream bandwidth (PID → stream attribution)"],
        )

    btf_present = Path("/sys/kernel/btf/vmlinux").exists()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Mirror the real collector. Biosnoop variants differ: some support -t,
    # while Ubuntu/bpfcc-tools builds often print TIME(s) by default and reject
    # -t. Auto-detect the valid form.
    base_cmd, supports_t = _build_biosnoop_smoke_cmd(biosnoop_bin)
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL"] + base_cmd
    else:
        cmd = base_cmd
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n", "-E", "PYTHONUNBUFFERED=1"] + cmd

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 env=env)
    except FileNotFoundError:
        return CheckResult(
            "biosnoop (--enable-biosnoop)", "fail",
            detail=f"failed to spawn {biosnoop_bin}",
            fix="apt install bpfcc-tools", impact=["§I per-stream"],
        )

    fd = proc.stdout.fileno()
    flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
    _fcntl.fcntl(fd, _fcntl.F_SETFL, flags | os.O_NONBLOCK)

    poll = _sel.poll()
    poll.register(fd, _sel.POLLIN)

    accumulated = ""
    header_seen = False
    event_seen = False
    trigger_done = False
    trigger_rc = None
    trigger_err = ""
    diag_paths = {}
    t_start = time.monotonic()

    # Phase 1: wait for header.
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        events = poll.poll(200)
        if events:
            try:
                chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
            except BlockingIOError:
                chunk = ""
            if chunk:
                accumulated += chunk
                if "PID" in accumulated and ("COMM" in accumulated or "DISK" in accumulated):
                    header_seen = True
                    break

    # Phase 2: if requested, actively prove event emission on target disk.
    if header_seen and verify_event and device:
        trigger_rc, _out, trigger_err = _trigger_readonly_block_io(device, use_sudo=use_sudo)
        trigger_done = True
        event_deadline = time.monotonic() + 5.0
        while time.monotonic() < event_deadline:
            if proc.poll() is not None:
                break
            events = poll.poll(200)
            if not events:
                continue
            try:
                chunk = os.read(fd, 8192).decode("utf-8", errors="replace")
            except BlockingIOError:
                continue
            if not chunk:
                continue
            accumulated += chunk
            for line in chunk.splitlines():
                if _biosnoop_disk_matches(line, device):
                    event_seen = True
                    break
            if event_seen:
                break

    # Graceful teardown — SIGINT lets BCC flush.
    try:
        proc.send_signal(2)
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    try:
        stderr = proc.stderr.read() if proc.stderr else ""
    except Exception:
        stderr = ""

    elapsed = time.monotonic() - t_start
    try:
        diag_paths = _write_biosnoop_preflight_diag(
            cmd,
            stdout_text=accumulated,
            stderr_text=stderr,
            summary={
                "device": device,
                "disk_base": _block_device_base(device),
                "header_seen": bool(header_seen),
                "event_seen": bool(event_seen),
                "verify_event": bool(verify_event),
                "trigger_done": bool(trigger_done),
                "trigger_returncode": trigger_rc,
                "trigger_stderr": trigger_err,
                "biosnoop_t_supported": bool(supports_t),
                "elapsed_s": round(elapsed, 3),
            },
        )
    except Exception:
        diag_paths = {}


    if header_seen and (not verify_event):
        return CheckResult(
            "biosnoop (--enable-biosnoop)", "pass",
            detail=(f"attach/header OK within {elapsed:.1f}s; event capture not verified "
                    f"(run with --strict-sanity to verify target-device events; "
                    f"biosnoop -t supported={'yes' if supports_t else 'no'}; BTF={'yes' if btf_present else 'no'})"),
        )

    if header_seen and verify_event and event_seen:
        return CheckResult(
            "biosnoop (--enable-biosnoop)", "pass",
            detail=(f"attach/header OK and captured target-device event for "
                    f"{_block_device_base(device)} after read-only smoke I/O "
                    f"(biosnoop -t supported={'yes' if supports_t else 'no'}; BTF={'yes' if btf_present else 'no'})"),
        )

    if header_seen and verify_event and not event_seen:
        diagnostic = (accumulated + "\n" + stderr).strip()
        sample = " | ".join(diagnostic.splitlines()[-5:])[:220] if diagnostic else "(no output)"
        trig = ""
        if trigger_done:
            trig = f"; smoke-read rc={trigger_rc}, err={trigger_err[:120]!r}"
        _stderr_path = diag_paths.get("stderr_path", "(preflight stderr path unavailable)")
        _stdout_path = diag_paths.get("stdout_path", "(preflight stdout path unavailable)")
        _summary_path = diag_paths.get("summary_path", "(preflight summary path unavailable)")
        _cmd_path = diag_paths.get("command_path", "(preflight command path unavailable)")
        return CheckResult(
            "biosnoop (--enable-biosnoop)", "warn",
            detail=(f"attach/header OK, but no biosnoop event was observed for "
                    f"{_block_device_base(device)} during read-only smoke I/O{trig}. "
                    f"This does not mean biosnoop is broken; it means target-device "
                    f"attribution was not proven during the tiny smoke probe. "
                    f"Last output: {sample}. "
                    f"Diagnostics: stderr={_stderr_path}; stdout={_stdout_path}; "
                    f"summary={_summary_path}; command={_cmd_path}"),
            fix=("Proceeding is allowed even with --strict-sanity because biosnoop attached "
                 "and printed its header. Inspect the absolute preflight paths printed above. "
                 "During the real run, check raw/biosnoop_events_all.csv and raw/biosnoop_events.csv, "
                 "then compare DISK names against blktrace. Keep blktrace as the authoritative physical "
                 "L3 local-storage I/O source; biosnoop is best-effort per-PID attribution."),
            impact=["§I per-stream bandwidth", "biosnoop per-PID attribution"],
        )

    # No header.
    diagnostic = (accumulated + "\n" + stderr).strip()
    last_lines = diagnostic.splitlines()[-3:] if diagnostic else ["(no output)"]
    btf_hint = "" if btf_present else (
        " (no BTF — first BCC run may need linux-headers-$(uname -r); "
        "try preloading by running 'sudo biosnoop-bpfcc -t' once)")
    _stderr_path = diag_paths.get("stderr_path", "(preflight stderr path unavailable)")
    _stdout_path = diag_paths.get("stdout_path", "(preflight stdout path unavailable)")
    _summary_path = diag_paths.get("summary_path", "(preflight summary path unavailable)")
    _cmd_path = diag_paths.get("command_path", "(preflight command path unavailable)")
    return CheckResult(
        "biosnoop (--enable-biosnoop)", "fail",
        detail=(f"no header in 8s. Last output: {' | '.join(last_lines)[:160]}. "
                f"Diagnostics: stderr={_stderr_path}; stdout={_stdout_path}; "
                f"summary={_summary_path}; command={_cmd_path}"),
        fix=("Test manually: sudo " + biosnoop_bin + "  (or add -t only if its help lists -t). "
             "If passwordless sudo isn't configured for this user, the smoke "
             "test sees an immediate exit. Pass --no-sudo to skip the sudo prefix"
             " if AMOprof itself runs as root."
             + btf_hint),
        impact=["§I per-stream bandwidth"],
    )


def check_amduprof_pcm(binary: str, use_sudo: bool = True) -> CheckResult:
    """AMDuProf PCM smoke: short capture with the SAME flags the real collector
    uses (-m memory -a --msr -i <ms>), then verify the CSV contains the
    'Total Mem Bw' header line that the bundled report's DRAM parser needs.

    This binary needs MSR access, so it almost always must run as root.
    The user's working command `AMDuProfPcm -m ipc -a -A system,package,core
    -d 60 -o ...` proves the binary works; our smoke test now mirrors the
    actual memory-bandwidth collector flags rather than inventing new ones.
    """
    if not Path(binary).exists():
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "fail",
            detail=f"{binary} does not exist",
            fix="Install AMDuProf from https://www.amd.com/en/developer/uprof.html "
                "and pass --amduprof-pcm-bin /path/AMDuProfPcm",
            impact=["§F DRAM Bandwidth chart"],
        )
    if not os.access(binary, os.X_OK):
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "fail",
            detail=f"{binary} is not executable",
            fix=f"chmod +x {binary}", impact=["§F DRAM Bandwidth"],
        )
    # MSR access is the most common failure mode — surface it as a hint
    msr_p = Path("/dev/cpu/0/msr")
    msr_hint = ""
    if not msr_p.exists():
        msr_hint = "  (no /dev/cpu/0/msr — run: sudo modprobe msr)"

    # Match exactly what AMDuProfPcm 5.2-606 accepts. Empirically `-i <ms>`
    # is rejected ("Failed to process args.") in this version's memory mode;
    # `-d <seconds>` is the canonical duration flag and works reliably.
    # The user's known-good command is:
    #   AMDuProfPcm -r -m memory -a -d 1 -o <path> --msr
    # AMDuProf adjusts the sampling interval to its minimum (~1200ms) so a
    # 3-second duration is enough to get at least 2 samples + the header.
    with tempfile.NamedTemporaryFile(suffix="_pcm.csv", delete=False) as f:
        out_csv = f.name
    # Remove the empty file AMDuProf would otherwise complain about
    try: Path(out_csv).unlink()
    except OSError: pass
    cmd = [binary, "-r", "-m", "memory", "-a", "--msr", "-d", "3", "-o", out_csv]
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        # AMDuProf -d 3 exits cleanly on its own after 3 sec; allow up to 8 sec
        # total to absorb startup/teardown overhead.
        rc, out, err = _run(cmd, timeout=8.0)
    except Exception as e:
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "fail",
            detail=f"failed to spawn {binary}: {e}",
            impact=["§F DRAM Bandwidth"],
        )

    if rc not in (0,):
        combined = (err or "") + (out or "")
        tail = combined.strip().splitlines()
        msg = tail[-1] if tail else ""
        try: Path(out_csv).unlink()
        except OSError: pass
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "fail",
            detail=f"AMDuProfPcm exited rc={rc}: {msg[:140]}{msr_hint}",
            fix=("Run manually with the EXACT command we used: "
                 f"{' '.join(cmd)}. "
                 "Common causes: needs root (run as root or configure passwordless "
                 "sudo for this binary), MSR module not loaded (sudo modprobe msr), "
                 "or this AMDuProf version doesn't support '-m memory' "
                 "(try the user's known-good profile: -m ipc instead)."),
            impact=["§F DRAM Bandwidth"],
        )

    try:
        size = Path(out_csv).stat().st_size if Path(out_csv).exists() else 0
        body = Path(out_csv).read_text(errors="replace") if size > 0 else ""
    finally:
        try: Path(out_csv).unlink()
        except OSError: pass

    if size == 0:
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "fail",
            detail=f"AMDuProfPcm ran but produced no output file{msr_hint}",
            fix=f"Run manually to inspect: {' '.join(cmd)}",
            impact=["§F DRAM Bandwidth"],
        )
    if "Total Mem Bw" not in body or "Total Mem RdBw" not in body:
        return CheckResult(
            "AMDuProf PCM (--enable-amduprof-pcm)", "warn",
            detail=(f"output written ({size} bytes) but missing 'Total Mem Bw' "
                     "/ 'RdBw' header — DRAM parser may skip it"),
            fix=("This usually means -m memory ran in a different submode. "
                 f"Sample of output: {body[:160].strip()!r}"),
            impact=["§F DRAM Bandwidth (header format mismatch)"],
        )
    return CheckResult(
        "AMDuProf PCM (--enable-amduprof-pcm)", "pass",
        detail=f"3-sec capture produced parseable DRAM-BW CSV ({size} bytes)")



def check_intel_pcm(binary: str = "pcm-memory", use_sudo: bool = True,
                    force_perf_imc: bool = False) -> CheckResult:
    """Intel DRAM bandwidth smoke test.

    Prefer Intel PCM pcm-memory when requested; perf uncore_imc is a fallback
    path. This mirrors --enable-dram --dram-tool intel-pcm/perf-imc.
    """
    if not force_perf_imc:
        # pcm-memory often needs root for MSR/PCI config access; --help is enough
        # to verify the binary path without requiring counters to be readable.
        path = shutil.which(binary) or (binary if Path(binary).exists() else "")
        if not path:
            return CheckResult(
                "Intel PCM pcm-memory (--enable-dram)", "warn",
                detail=f"{binary!r} not found; AMOprof will try perf uncore_imc fallback if available",
                fix="Install Intel PCM from https://github.com/intel/pcm or pass --dram-tool perf-imc",
                impact=["§F DRAM Bandwidth chart on Intel"],
            )
        rc, out, err = _run([path, "--help"], timeout=4.0, use_sudo=False)
        if rc in (0, 1):
            return CheckResult(
                "Intel PCM pcm-memory (--enable-dram)", "pass",
                detail=f"found {path}; runtime collection will use pcm-memory CSV output",
            )
        return CheckResult(
            "Intel PCM pcm-memory (--enable-dram)", "warn",
            detail=f"{path} exists but --help returned rc={rc}: {(err or out).strip()[:120]}",
            fix="Try running pcm-memory manually as root; or use --dram-tool perf-imc",
            impact=["§F DRAM Bandwidth chart on Intel"],
        )

    # perf fallback check.
    if not shutil.which("perf"):
        return CheckResult(
            "perf uncore_imc (--enable-dram --dram-tool perf-imc)", "warn",
            detail="perf not in PATH",
            fix="Install linux-tools/perf for this kernel, or install Intel PCM",
            impact=["§F DRAM Bandwidth chart on Intel"],
        )
    rc, out, err = _run(["perf", "list", "uncore_imc"], timeout=5.0, use_sudo=False)
    txt = (out + err).lower()
    if "cas_count" not in txt:
        return CheckResult(
            "perf uncore_imc (--enable-dram --dram-tool perf-imc)", "warn",
            detail="perf is present but uncore_imc CAS counters were not listed",
            fix="Use Intel PCM pcm-memory, run on bare metal, or load/enable uncore PMU support",
            impact=["§F DRAM Bandwidth chart on Intel"],
        )
    return CheckResult(
        "perf uncore_imc (--enable-dram --dram-tool perf-imc)", "pass",
        detail="perf lists uncore_imc CAS read/write counters",
    )

# ─── Orchestrator ───────────────────────────────────────────────────────────
def run_sanity_checks(args, sglang_host: str, sglang_port: int) -> list[CheckResult]:
    """Run every applicable check. Returns a list of CheckResult.

    The list ordering is intentional — required baselines first, then
    optional collectors. The caller decides what to do with failures
    (abort vs warn vs proceed).
    """
    results: list[CheckResult] = []

    # ── Required baselines ─────────────────────────────────────────────────
    try:
        from .cli import _resolve_ssd_devices
        _ssd_list = _resolve_ssd_devices(getattr(args, "ssd_device", "")) or\
                    [getattr(args, "ssd_device", "/dev/nvme0n1")]
    except Exception:
        _ssd_list = [getattr(args, "ssd_device", "/dev/nvme0n1")]
    for _dev in _ssd_list:
        results.append(check_ssd_device(_dev))
    results.append(check_iostat())
    results.append(check_vmstat())
    results.append(check_nvidia_gpu())
    results.append(check_dcgm())
    if sglang_port:
        results.append(check_sglang_endpoint(sglang_host, sglang_port))

    use_sudo = not getattr(args, "no_sudo", False)

    # ── Optional collectors ────────────────────────────────────────────────
    if getattr(args, "enable_blktrace", False):
        for _dev in _ssd_list:
            results.append(check_blktrace(
                _dev,
                blktrace_bin=getattr(args, "blktrace_bin", "blktrace"),
                blkparse_bin=getattr(args, "blkparse_bin", "blkparse"),
                use_sudo=use_sudo,
            ))

    if getattr(args, "enable_biosnoop", False):
        results.append(check_biosnoop(
            biosnoop_bin=getattr(args, "biosnoop_bin", "biosnoop"),
            use_sudo=use_sudo,
            device=getattr(args, "ssd_device", None),
            verify_event=bool(getattr(args, "strict_sanity", False)),
        ))

    if getattr(args, "enable_amduprof_pcm", False):
        setattr(args, "enable_dram", True)
        if getattr(args, "dram_tool", "auto") == "auto":
            setattr(args, "dram_tool", "amduprof")

    if getattr(args, "enable_dram", False):
        tool = (getattr(args, "dram_tool", "auto") or "auto").lower().replace("_", "-")
        if tool == "auto":
            try:
                cpuinfo = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
            except Exception:
                cpuinfo = ""
            tool = "amduprof" if "authenticamd" in cpuinfo else "intel-pcm" if "genuineintel" in cpuinfo else "intel-pcm"
        if tool in {"amduprof", "amd", "amduprof-pcm"}:
            results.append(check_amduprof_pcm(
                binary=getattr(args, "amduprof_pcm_bin",
                                "/opt/AMDuProf_5.2-606/bin/AMDuProfPcm"),
                use_sudo=use_sudo,
            ))
        elif tool in {"intel-pcm", "intel", "pcm", "pcm-memory"}:
            results.append(check_intel_pcm(
                binary=getattr(args, "intel_pcm_memory_bin", "pcm-memory"),
                use_sudo=use_sudo,
                force_perf_imc=False,
            ))
        elif tool in {"perf-imc", "perf", "imc", "imc-pmu"}:
            results.append(check_intel_pcm(
                binary=getattr(args, "intel_pcm_memory_bin", "pcm-memory"),
                use_sudo=use_sudo,
                force_perf_imc=True,
            ))

    return results


def summarize(results: list[CheckResult]) -> dict:
    """Aggregate counts and decide whether to abort.

    Required-baseline failures always abort. Optional collector failures
    only abort if --strict was given (handled by caller).
    """
    by_status = {"pass": 0, "fail": 0, "warn": 0, "skipped": 0}
    failed_required: list[CheckResult] = []
    failed_optional: list[CheckResult] = []
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if r.status == "fail":
            # Optional collectors all have "(--enable-*)" in name
            if "--enable-" in r.name:
                failed_optional.append(r)
            else:
                failed_required.append(r)
    return {
        "counts": by_status,
        "failed_required": failed_required,
        "failed_optional": failed_optional,
    }


def print_results(results: list[CheckResult]) -> None:
    """Pretty-print the check results."""
    sym = {"pass": "✓", "fail": "✗", "warn": "⚠", "skipped": "·"}
    log.info("=" * 68)
    log.info("PRE-FLIGHT SANITY CHECKS — verifying every collector can collect")
    log.info("=" * 68)
    for r in results:
        line = f"  {sym.get(r.status, '?')} {r.name:<40s} {r.detail[:80]}"
        if r.status == "pass":
            log.info(line)
        elif r.status == "warn":
            log.warning(line)
        elif r.status == "skipped":
            log.info(line)
        else:
            log.error(line)
            if r.fix:
                log.error(f"      → fix: {r.fix}")
            if r.impact:
                for imp in r.impact:
                    log.error(f"      → impact: {imp}")
    log.info("=" * 68)
