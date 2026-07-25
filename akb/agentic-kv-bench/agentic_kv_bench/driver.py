"""The load driver: render an inference-perf config and run it as a subprocess.

inference-perf (kubernetes-sigs) is the community-standard load generator and
is engine-agnostic — the same configs run against the T4 dev box, house A100s
and MI355 unchanged. We treat it as a black box: we hand it a rendered YAML
config and a report directory, and it gives us TTFT/TPOT/ITL per request.

The harness owns everything around it: timeouts, process-tree kills, log
capture, and the phase markers that line its report up with AMOProf telemetry.
"""

from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path

import yaml

from .naming import RunLayout, set_dotted
from .util import log, popen_process_group, shell_join, stop_tree_gracefully

EXIT_TIMEOUT = 124
EXIT_ABORT = 130


def base_driver_config(model: str, base_url: str, report_dir: Path) -> dict:
    """Harness-owned defaults; the spec's inference_perf section merges over this."""
    return {
        "api": {"type": "completion", "streaming": True},
        "server": {
            "type": "vllm",
            "base_url": base_url,
            "model_name": model,
            "ignore_eos": True,
        },
        "tokenizer": {"pretrained_model_name_or_path": model},
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
                "percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
            },
            "session_lifecycle": {"summary": True, "per_stage": True, "per_session": False},
        },
        "storage": {"local_storage": {"path": str(report_dir)}},
    }


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def render_driver_config(model: str, base_url: str, layout: RunLayout,
                         spec_inference_perf: dict, overrides: dict | None = None,
                         prometheus_url: str | None = None,
                         prometheus_scrape_interval: int = 15) -> dict:
    cfg = deep_merge(base_driver_config(model, base_url, layout.report), spec_inference_perf)
    # spec may override model/server itself; CLI --server overrides everything
    cfg["server"]["model_name"] = model
    cfg["server"]["base_url"] = base_url
    cfg.setdefault("tokenizer", {})["pretrained_model_name_or_path"] = (
        cfg["tokenizer"].get("pretrained_model_name_or_path") or model)
    cfg["storage"] = {"local_storage": {"path": str(layout.report)}}  # never user-controlled
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    # Only wire inference-perf's own Prometheus client if the user has an actual
    # Prometheus server — vLLM's /metrics is an exposition endpoint, not a query
    # API, and our own scraper (telemetry.py) already covers it.
    if prometheus_url:
        cfg["metrics"] = {
            "type": "prometheus",
            "prometheus": {"url": prometheus_url, "scrape_interval": prometheus_scrape_interval},
        }
    return cfg


def write_driver_config(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_driver(cfg_path: Path, layout: RunLayout, timeout_s: int = 0,
               kill_grace_s: int = 20, abort=None, extra_args: list[str] | None = None) -> int:
    """Run inference-perf; return its exit code (124=our timeout, 130=abort)."""
    exe = shutil.which("inference-perf")
    if exe is None:
        log.error("inference-perf not on PATH (pip install inference-perf)")
        return 127
    cmd = [exe, "--config_file", str(cfg_path), "--log-level", "INFO"] + list(extra_args or [])
    log.info("driver: %s", shell_join(cmd))
    with open(layout.driver_log, "a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START: {shell_join(cmd)}\n")
        proc = popen_process_group(cmd, fh, cwd=layout.path)
        log.info("driver started (pid %d), log: %s", proc.pid, layout.driver_log)
        elapsed, status = 0, None
        while proc.poll() is None:
            if abort is not None and abort.is_set():
                log.warning("abort requested — stopping driver tree")
                stop_tree_gracefully(proc.pid, kill_grace_s, what="driver")
                proc.wait()
                return EXIT_ABORT
            if timeout_s and elapsed >= timeout_s:
                log.error("driver TIMEOUT after %ds — stopping tree (grace %ds)", elapsed, kill_grace_s)
                stop_tree_gracefully(proc.pid, kill_grace_s, what="driver")
                proc.wait()
                return EXIT_TIMEOUT
            time.sleep(5)
            elapsed += 5
        status = proc.returncode
        if status != 0:
            log.error("driver exited %d — tail of %s:", status, layout.driver_log)
        return status
