"""AMOProf lifecycle: run as a subprocess before and after each run.

Faithful port of the bash harness's semantics, because they were earned the
hard way:

  * `amoprof service` DAEMONIZES — the launcher exits in ~1s even on success.
    Readiness is therefore judged by the metrics port answering, never by the
    launcher PID.
  * The real daemon PID is discovered via pgrep after the port answers.
  * Refuse to start on top of a stale instance: it would hold the port and
    silently mix old telemetry into this run.
  * Stop = TERM the process group, long grace, then KILL the group.
  * Optional post-run command (e.g. `amoprof report ...`) runs after stop so
    a finished report artifact lands in the run dir next to the raw data.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .util import (log, pid_alive, port_open, shell_join, stop_tree_gracefully,
                   wait_for_port)

DEFAULT_AMOPROF_ARGS = {
    "metrics_host": "0.0.0.0",
    "port": 9101,
    "vllm_port": 8000,
    "lmcache_port": 6999,
    "collectors": "gpu,dram,vmstat,iostat,smart",
    "scrape_duration_s": 10,
    "interval_s": 1,
    "ssd_device": "/dev/nvme1n1",
    "blkparse_interval_s": 10,
    "lmcache_bytes_per_token": 18432,
    "lmcache_max_disk_gb": 20.0,
    "enable_blktrace": True,
    "enable_biosnoop": True,
    "enable_dram": True,
    "dram_tool": "auto",
    "extra_args": [],
}


class SudoKeepalive:
    """sudo caches credentials ~15 min; refresh so the 3am stop doesn't block."""

    def __init__(self, interval_s: int = 60):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        r = subprocess.run(["sudo", "-v"])
        if r.returncode != 0:
            raise RuntimeError("sudo needs a password; run where you can type it once, or use --no-sudo")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            subprocess.run(["sudo", "-n", "true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        self._stop.set()


@dataclass
class AMOProfService:
    bin: str
    cfg: dict = field(default_factory=lambda: dict(DEFAULT_AMOPROF_ARGS))
    sudo: bool = True
    hicache_path: str = "/opt/ls/lmcache-disk-cache"
    pid: int | None = None

    def _args(self, out_dir: Path) -> list[str]:
        c = {**DEFAULT_AMOPROF_ARGS, **self.cfg}
        args = [
            "service",
            "--metrics-port", str(c["port"]),
            "--metrics-host", str(c["metrics_host"]),
            "--vllm-port", str(c["vllm_port"]),
            "--lmcache-port", str(c["lmcache_port"]),
            "--collectors", str(c["collectors"]),
            "--scrape-duration-s", str(c["scrape_duration_s"]),
            "--interval-s", str(c["interval_s"]),
            "--ssd-device", str(c["ssd_device"]),
            "--hicache-path", str(self.hicache_path),
            "--output-dir", str(out_dir),
            "--blkparse-interval-s", str(c["blkparse_interval_s"]),
            "--lmcache-bytes-per-token", str(c["lmcache_bytes_per_token"]),
            "--lmcache-max-disk-gb", str(c["lmcache_max_disk_gb"]),
        ]
        for flag, key in (("--enable-blktrace", "enable_blktrace"),
                          ("--enable-biosnoop", "enable_biosnoop"),
                          ("--enable-dram", "enable_dram")):
            if c.get(key):
                args.append(flag)
        if c.get("enable_dram"):
            args += ["--dram-tool", str(c["dram_tool"])]
        args += [str(a) for a in c.get("extra_args", [])]
        return args

    def start(self, out_dir: Path, svc_log: Path, ready_timeout_s: int = 90,
              post_start_settle_s: int = 10, abort=None) -> bool:
        port = int({**DEFAULT_AMOPROF_ARGS, **self.cfg}["port"])
        if port_open("127.0.0.1", port):
            log.error("port %d already in use — leftover amoprof?", port)
            log.error("run: sudo pkill -f 'amoprof.*service'   (then re-run)")
            return False
        cmd = (["sudo"] if self.sudo else []) + [self.bin] + self._args(out_dir)
        log.info("starting amoprof -> %s", out_dir)
        log.debug("amoprof cmd: %s", shell_join(cmd))
        with open(svc_log, "a") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START: {shell_join(cmd)}\n")
            # launcher daemonizes and exits immediately; do NOT hold its pid
            self._launcher = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                              start_new_session=True)
        if not wait_for_port("127.0.0.1", port, ready_timeout_s, abort=abort):
            log.error("amoprof never opened port %d. Last log lines:", port)
            try:
                log.error("%s", "\n".join(svc_log.read_text().splitlines()[-20:]))
            except OSError:
                pass
            return False
        time.sleep(3)  # let the daemon finish init before we trust it
        self.pid = self._find_daemon_pid()
        if self.pid is None:
            log.error("port is up but could not find the amoprof daemon PID "
                      "(pgrep -af amoprof to debug)")
            return False
        log.info("amoprof is up (daemon pid %d), metrics on :%d; settling %ds",
                 self.pid, port, post_start_settle_s)
        time.sleep(post_start_settle_s)
        return True

    @staticmethod
    def _find_daemon_pid() -> int | None:
        r = subprocess.run(["pgrep", "-f", "amoprof.*service"], capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return int(r.stdout.split()[0])

    def stop(self, grace_s: int = 45, post_stop_settle_s: int = 10) -> None:
        # reap the launcher if it hasn't exited (harmless if it has)
        if getattr(self, "_launcher", None) is not None:
            try:
                self._launcher.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            self._launcher = None
        if self.pid is None:
            return
        if not pid_alive(self.pid, sudo=self.sudo):
            self.pid = None
            return
        log.info("stopping amoprof (pid %d), grace %ds...", self.pid, grace_s)
        stop_tree_gracefully(self.pid, grace_s, sudo=self.sudo, what="amoprof")
        self.pid = None
        log.info("amoprof stopped; settling %ds for final telemetry flush", post_stop_settle_s)
        time.sleep(post_stop_settle_s)

    def run_post(self, run_dir: Path, post_run: dict | None, amoprof_out: Path) -> int | None:
        """Optional post-run subprocess (e.g. render an AMOProf report).

        `{run_dir}` / `{amoprof_out}` placeholders in args are substituted.
        """
        if not post_run or not post_run.get("cmd"):
            return None
        cmd = [str(a).format(run_dir=run_dir, amoprof_out=amoprof_out) for a in post_run["cmd"]]
        if self.sudo and post_run.get("sudo", False):
            cmd = ["sudo"] + cmd
        timeout = int(post_run.get("timeout_s", 300))
        log.info("amoprof post-run: %s", shell_join(cmd))
        log_path = run_dir / "logs" / "amoprof_post.log"
        with open(log_path, "a") as fh:
            try:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                log.error("amoprof post-run timed out after %ds", timeout)
                return None
        if r.returncode != 0:
            log.error("amoprof post-run exited %d (log: %s)", r.returncode, log_path)
        return r.returncode
