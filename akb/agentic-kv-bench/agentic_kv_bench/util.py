"""Shared utilities: logging, processes, ports, files.

Everything here is stdlib-only on purpose: this harness ships to bare
benchmark boxes where installing anything beyond PyYAML (pulled in by
inference-perf anyway) is a negotiation.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("akb")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log = logging.getLogger("akb")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


_FS_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def fs_safe(s: str) -> str:
    """Make an arbitrary string safe for a directory name."""
    return _FS_UNSAFE.sub("_", s).strip("_")[:80]


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout_s: float, abort=None, poll_s: float = 2.0) -> bool:
    waited = 0.0
    while waited < timeout_s:
        if port_open(host, port):
            return True
        if abort is not None and abort.is_set():
            return False
        time.sleep(poll_s)
        waited += poll_s
    return False


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def check_disk_space(path: Path, min_free_gb: float) -> None:
    avail = free_gb(path)
    if avail < min_free_gb:
        raise RuntimeError(
            f"only {avail:.1f}G free under {path} (need {min_free_gb:.0f}G). Aborting."
        )


class FileLock:
    """One sweep at a time per base dir; a second instance would corrupt telemetry."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(f"another sweep holds {self.path} — exiting.") from e
            raise
        self._fh.write(str(os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def popen_process_group(cmd: list[str], log_fh, cwd: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    """Launch in its own session/process group so we can kill the whole tree.

    Same trick as the bash harness's `setsid`: pgid == pid, so signalling the
    negative pgid hits every child/grandchild.
    """
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=True,
        text=True,
    )


def kill_tree(pid: int, sig: int = signal.SIGTERM, sudo: bool = False) -> None:
    """Signal a process group; fall back to the lone pid."""
    def _send(target: str) -> None:
        if sudo:
            subprocess.run(["sudo", "kill", f"-{signal.Signals(sig).name}", "--", target],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                if target.startswith("-"):
                    os.killpg(abs(int(target)), sig)
                else:
                    os.kill(int(target), sig)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        pgid = os.getpgid(pid)
        _send(f"-{pgid}")
    except (ProcessLookupError, PermissionError):
        _send(str(pid))


def pid_alive(pid: int, sudo: bool = False) -> bool:
    if sudo:
        return subprocess.run(["sudo", "kill", "-0", str(pid)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_tree_gracefully(pid: int, grace_s: float, sudo: bool = False, poll_s: float = 2.0,
                         what: str = "process") -> None:
    """TERM the tree, wait up to grace_s, then KILL the tree."""
    if not pid_alive(pid, sudo=sudo):
        return
    kill_tree(pid, signal.SIGTERM, sudo=sudo)
    waited = 0.0
    while waited < grace_s:
        if not pid_alive(pid, sudo=sudo):
            return
        time.sleep(poll_s)
        waited += poll_s
    log.warning("%s (pid %d) still alive after %.0fs of TERM — SIGKILLing tree.", what, pid, grace_s)
    kill_tree(pid, signal.SIGKILL, sudo=sudo)
    time.sleep(2)  # zombie reaping / fd flush takes a moment even after SIGKILL


def pgrep_first(pattern: str) -> int | None:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return int(r.stdout.split()[0])


def which_all(bins: list[str]) -> list[str]:
    """Return the subset of binaries missing from PATH."""
    return [b for b in bins if shutil.which(b) is None]


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def read_text_quiet(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""
