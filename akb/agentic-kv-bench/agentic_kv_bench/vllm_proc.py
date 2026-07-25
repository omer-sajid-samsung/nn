"""vLLM server lifecycle: launch as a managed subprocess (opt-in via --manage-vllm).

Mirrors AMOProfService's shape (see amoprof.py): build a `vllm serve ...`
command from a spec's `vllm_launch:` block, launch it in its own process
group via `popen_process_group`, wait for the health endpoint, and stop it
gracefully by killing the group.

Without `--manage-vllm` (or when a spec has no `vllm_launch:` block), nothing
in this module runs and the harness's long-standing "assume it's already up,
fail if not" behavior (runner.ensure_vllm) is unchanged — this is additive,
not a replacement.

If the server is already healthy when `start()` is called, we adopt it
read-only: we never kill a server we didn't launch ourselves. This matters
for iterative work at a terminal where you often want to reuse a
hand-started server across several `akb run` invocations.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .util import (log, pid_alive, popen_process_group, shell_join,
                   stop_tree_gracefully, wait_for_port)

DEFAULT_VLLM_LAUNCH = {
    "venv": "",                  # path to a venv's activate script; "" = use current PATH
    "dtype": "float16",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.3,
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "max_num_seqs": None,        # None = vLLM's own default (128 as of this vLLM version)
    "no_enable_prefix_caching": False,
    "kv_transfer_config": None,  # dict -> JSON-encoded onto --kv-transfer-config
    "host": "0.0.0.0",
    "env": {},                   # extra env vars, e.g. LMCACHE_CONFIG_FILE
    "extra_args": [],
}


def _restart_key(model: str, cfg: dict) -> tuple:
    """Everything that changes what the server actually IS, for restart-needed checks."""
    c = {**DEFAULT_VLLM_LAUNCH, **cfg}
    relevant = {k: v for k, v in c.items() if k not in ("venv", "extra_args")}
    return (model, json.dumps(relevant, sort_keys=True))


@dataclass
class VLLMService:
    model: str
    cfg: dict = field(default_factory=lambda: dict(DEFAULT_VLLM_LAUNCH))
    port: int = 8000
    pid: int | None = None       # only set if WE launched it
    launch_key: tuple | None = None

    def _args(self) -> list[str]:
        c = {**DEFAULT_VLLM_LAUNCH, **self.cfg}
        args = [
            "vllm", "serve", self.model,
            "--dtype", str(c["dtype"]),
            "--max-model-len", str(c["max_model_len"]),
            "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
            "--tensor-parallel-size", str(c["tensor_parallel_size"]),
            "--port", str(self.port),
            "--host", str(c["host"]),
        ]
        if c.get("enforce_eager"):
            args.append("--enforce-eager")
        if c.get("no_enable_prefix_caching"):
            args.append("--no-enable-prefix-caching")
        if c.get("max_num_seqs"):
            args += ["--max-num-seqs", str(c["max_num_seqs"])]
        if c.get("kv_transfer_config"):
            args += ["--kv-transfer-config", json.dumps(c["kv_transfer_config"])]
        args += [str(a) for a in c.get("extra_args", [])]
        return args

    def start(self, base_url: str, log_path: Path, ready_timeout_s: int = 900,
              already_healthy=None, abort=None) -> bool:
        """`already_healthy` is an injected `healthy() -> bool` check (runner.vllm_healthy)
        so this module doesn't need to import runner.py (which imports this one)."""
        self.launch_key = _restart_key(self.model, self.cfg)
        if already_healthy and already_healthy():
            log.info("vLLM already healthy at %s — reusing, will not manage its lifecycle", base_url)
            self.pid = None
            return True

        c = {**DEFAULT_VLLM_LAUNCH, **self.cfg}
        venv = c.get("venv") or ""
        argv = self._args()
        if venv:
            shell_cmd = f'source "{venv}"; exec {shell_join(argv)}'
            cmd = ["bash", "-c", shell_cmd]
        else:
            cmd = argv

        host, _, port_s = base_url.rpartition(":")
        host = host.replace("http://", "").replace("https://", "") or "127.0.0.1"
        port = int(port_s) if port_s.isdigit() else self.port

        env = {**os.environ, **{str(k): str(v) for k, v in (c.get("env") or {}).items()}}

        log.info("starting vLLM (model=%s) -> %s", self.model, log_path)
        log.debug("vllm cmd: %s env_extra=%s", shell_join(cmd), c.get("env"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START: {shell_join(cmd)}\n")
            fh.flush()
            proc = popen_process_group(cmd, fh, env=env)
        self.pid = proc.pid
        self._proc = proc

        log.info("waiting up to %ds for vLLM health at %s ...", ready_timeout_s, base_url)
        healthy_fn = already_healthy or (lambda: wait_for_port(host, port, 1, poll_s=1))
        waited = 0.0
        poll_s = 5.0
        while waited < ready_timeout_s:
            if not pid_alive(self.pid):
                log.error("vLLM process died during startup. Last log lines:")
                try:
                    log.error("%s", "\n".join(log_path.read_text().splitlines()[-40:]))
                except OSError:
                    pass
                self.pid = None
                return False
            if healthy_fn():
                log.info("vLLM is healthy (pid %d) at %s", self.pid, base_url)
                return True
            if abort is not None and abort.is_set():
                return False
            time.sleep(poll_s)
            waited += poll_s
        log.error("vLLM never became healthy within %ds", ready_timeout_s)
        return False

    def needs_restart(self, model: str, cfg: dict) -> bool:
        """True if this running instance's launch config differs from what's wanted now."""
        return self.launch_key is not None and self.launch_key != _restart_key(model, cfg)

    def stop(self, grace_s: int = 60) -> None:
        if self.pid is None:
            return  # we never launched it (adopted an already-healthy server) - leave it alone
        log.info("stopping vLLM (pid %d), grace %ds...", self.pid, grace_s)
        proc = getattr(self, "_proc", None)
        if proc is not None:
            # We hold the Popen handle: use terminate()/wait() so the child is
            # actually reaped (os.kill(pid, 0) in stop_tree_gracefully would
            # keep reporting a zombie as "alive" until something calls wait(),
            # so that path alone would just burn the whole grace window).
            # vLLM's own SIGTERM handler cascades a clean shutdown to its TP
            # worker subprocesses itself (observed in vllm_server.log), so
            # signalling just the master process is sufficient.
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                log.warning("vLLM (pid %d) still alive after %ds of TERM — SIGKILLing.", self.pid, grace_s)
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None
        else:
            stop_tree_gracefully(self.pid, grace_s, what="vLLM")
        self.pid = None
        log.info("vLLM stopped.")
