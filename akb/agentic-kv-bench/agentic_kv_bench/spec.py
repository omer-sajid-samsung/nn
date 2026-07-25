"""Workload spec and sweep spec loading.

A *workload spec* is the unit of work Mandar reviews and AMOProf consumes.
It is deliberately mostly a literal inference-perf config (under
`inference_perf:`) so the driver schema can evolve without touching this
harness — plus harness-owned sections for prediction, telemetry and amoprof.

A *sweep spec* expands one workload spec into N runs via per-run `--set`
overrides.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .naming import set_dotted


@dataclass
class WorkloadSpec:
    name: str
    description: str = ""
    model: str = ""                                # HF id, e.g. Qwen/Qwen3-8B-AWQ
    inference_perf: dict = field(default_factory=dict)   # literal driver config subtree
    run_knobs: dict[str, str] = field(default_factory=dict)  # short -> dotted path (naming)
    prediction: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)
    amoprof: dict = field(default_factory=dict)
    timeouts: dict = field(default_factory=dict)
    vllm_launch: dict = field(default_factory=dict)  # optional; see vllm_proc.py, --manage-vllm
    source_path: Path | None = None

    def merged_inference_perf(self, overrides: dict | None = None) -> dict:
        cfg = copy.deepcopy(self.inference_perf)
        for dotted, value in (overrides or {}).items():
            set_dotted(cfg, dotted, value)
        return cfg


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where}: missing required key '{key}'")
    return d[key]


def load_workload_spec(path: Path) -> WorkloadSpec:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a YAML mapping")
    spec = WorkloadSpec(
        name=str(_require(raw, "name", str(path))),
        description=str(raw.get("description", "")),
        model=str(raw.get("model", "")),
        inference_perf=dict(_require(raw, "inference_perf", str(path))),
        run_knobs=dict(raw.get("run_knobs", {})),
        prediction=dict(raw.get("prediction", {})),
        telemetry=dict(raw.get("telemetry", {})),
        amoprof=dict(raw.get("amoprof", {})),
        timeouts=dict(raw.get("timeouts", {})),
        vllm_launch=dict(raw.get("vllm_launch", {})),
        source_path=path,
    )
    if not spec.model:
        # allow model to live only inside the driver config
        spec.model = str(spec.inference_perf.get("server", {}).get("model_name", ""))
    if not spec.model:
        raise ValueError(f"{path}: set 'model:' (or inference_perf.server.model_name)")
    return spec


@dataclass
class SweepRun:
    overrides: dict
    label: str = ""


@dataclass
class SweepSpec:
    name: str
    base_spec_path: Path
    runs: list[SweepRun]
    settle_s: int = 180
    source_path: Path | None = None


def load_sweep_spec(path: Path) -> SweepSpec:
    raw = yaml.safe_load(path.read_text())
    base = Path(_require(raw, "base_spec", str(path)))
    if not base.is_absolute():
        base = path.parent / base
    runs = []
    for i, entry in enumerate(raw.get("runs", [])):
        if "set" not in entry:
            raise ValueError(f"{path}: run #{i} missing 'set:'")
        runs.append(SweepRun(overrides=dict(entry["set"]), label=str(entry.get("label", ""))))
    if not runs:
        raise ValueError(f"{path}: sweep has no runs")
    return SweepSpec(
        name=str(raw.get("name", "sweep")),
        base_spec_path=base,
        runs=runs,
        settle_s=int(raw.get("settle_s", 180)),
        source_path=path,
    )


def parse_set_expr(expr: str) -> tuple[str, object]:
    """`--set a.b.c=VALUE` -> ('a.b.c', yaml_parsed_value).

    YAML-parsing the right-hand side gives us ints/floats/bools/lists for free.
    """
    if "=" not in expr:
        raise ValueError(f"--set expects key=value, got: {expr!r}")
    key, _, raw = expr.partition("=")
    value = yaml.safe_load(raw)
    return key.strip(), value


def parse_scalar_list(s: str, cast=int) -> list:
    """`"1 8 16"` or `"1,8,16"` -> [1, 8, 16]"""
    return [cast(x) for x in s.replace(",", " ").split()]
