"""Run/sweep naming and the on-disk layout of a run.

Naming scheme (deterministic, sortable, self-describing):

  sweep dir : {base}/runs/{YYYYMMDD}_{sweep_name}/
  run dir   : {seq:02d}-{HHMMSS}_{model_short}_{spec_name}_{knobs}

  e.g.      03-031500_qwen3-8b-awq_w2-thrash_c16_i30000_u12

Every run dir has the same internal layout so downstream tooling (AMOProf
ingestion, report generation, eyeballs at 3am) never has to guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .util import fs_safe


def model_short(model: str) -> str:
    """Qwen/Qwen3-8B-AWQ -> qwen3-8b-awq"""
    return fs_safe(model.split("/")[-1].lower())


def build_knob_label(resolved_config: dict, knob_map: dict[str, str]) -> str:
    """Build the compact knob string from {short_name: dotted.path} pairs.

    Missing paths are skipped silently — knobs are cosmetics, not control flow.
    """
    parts = []
    for short, dotted in knob_map.items():
        val = dig(resolved_config, dotted)
        if val is None:
            continue
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        parts.append(f"{short}{val}")
    return "_".join(parts)


def dig(d: dict, dotted: str):
    """Dotted-path lookup; numeric segments index into lists."""
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def set_dotted(d: dict, dotted: str, value) -> None:
    """Dotted-path assignment; creates dicts on the way down."""
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur.setdefault(part, {})
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


@dataclass
class RunLayout:
    """The canonical contents of one run directory."""
    path: Path

    @property
    def spec_file(self) -> Path: return self.path / "spec.yaml"
    @property
    def driver_config(self) -> Path: return self.path / "inference_perf.yaml"
    @property
    def manifest(self) -> Path: return self.path / "manifest.json"
    @property
    def status_file(self) -> Path: return self.path / "status"
    @property
    def markers(self) -> Path: return self.path / "markers.txt"
    @property
    def run_summary(self) -> Path: return self.path / "run_summary.json"
    @property
    def logs(self) -> Path: return self.path / "logs"
    @property
    def driver_log(self) -> Path: return self.logs / "driver.log"
    @property
    def amoprof_log(self) -> Path: return self.logs / "amoprof_service.log"
    @property
    def telemetry(self) -> Path: return self.path / "telemetry"
    @property
    def vllm_metrics(self) -> Path: return self.telemetry / "vllm_metrics.prom"
    @property
    def lmcache_before(self) -> Path: return self.telemetry / "lmcache_before.json"
    @property
    def lmcache_after(self) -> Path: return self.telemetry / "lmcache_after.json"
    @property
    def amoprof_out(self) -> Path: return self.path / "amoprof"
    @property
    def report(self) -> Path: return self.path / "report"

    def makedirs(self) -> None:
        for p in (self.logs, self.telemetry, self.amoprof_out, self.report):
            p.mkdir(parents=True, exist_ok=True)

    def status(self) -> str | None:
        try:
            return self.status_file.read_text().strip()
        except OSError:
            return None

    def write_status(self, status: str) -> None:
        self.status_file.write_text(status + "\n")

    def write_manifest(self, data: dict) -> None:
        self.manifest.write_text(json.dumps(data, indent=2, default=str) + "\n")


def make_sweep_dir(base_dir: Path, sweep_name: str, day: str | None = None) -> Path:
    day = day or datetime.now(timezone.utc).strftime("%Y%m%d")
    return base_dir / "runs" / f"{day}_{fs_safe(sweep_name)}"


def make_run_dir(sweep_dir: Path, seq: int, model: str, spec_name: str, knob_label: str) -> RunLayout:
    """Deterministic name: {seq:02d}-{model_short}_{spec}_{knobs}.

    No timestamp on purpose — re-running the same sweep must land on the same
    run dir so resume can skip finished-OK runs.
    """
    bits = [f"{seq:02d}-{model_short(model)}", fs_safe(spec_name)]
    if knob_label:
        bits.append(fs_safe(knob_label))
    layout = RunLayout(sweep_dir / "_".join(bits))
    layout.makedirs()
    return layout
