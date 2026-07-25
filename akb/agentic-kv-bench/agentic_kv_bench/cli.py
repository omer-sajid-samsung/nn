"""akb — agentic KV bench harness CLI.

Commands:
  preflight   environment checks, no side effects
  run         a single workload spec (optionally with --set / convenience overrides)
  sweep       N runs from a sweep spec, or --spec + --concurrency-list
  report      aggregate a finished sweep dir into a table + sweep_report.md
  name        print the run dir name a spec+overrides would generate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .naming import build_knob_label, make_run_dir, model_short
from .runner import SweepRunner
from .spec import (load_sweep_spec, load_workload_spec, parse_scalar_list,
                   parse_set_expr)
from .util import log, setup_logging

DIST_KEYS = ("min", "max", "mean")


def _convenience_overrides(args, spec) -> dict:
    """Translate friendly flags into dotted-path overrides, load-type aware."""
    ov = {}
    data = spec.inference_perf.get("data", {}) or {}
    load = spec.inference_perf.get("load", {}) or {}
    dtype, ltype = data.get("type"), load.get("type", "constant")

    if getattr(args, "concurrency", None) is not None:
        if ltype == "concurrent":
            ov["load.stages.0.concurrency_level"] = args.concurrency
        elif ltype == "trace_session_replay":
            ov["load.stages.0.concurrent_sessions"] = args.concurrency
        else:
            ov["load.stages.0.rate"] = args.concurrency
            log.warning("load.type=%s: mapping --concurrency to request RATE %s (proxy)",
                        ltype, args.concurrency)
    if getattr(args, "rate", None) is not None:
        ov["load.stages.0.rate"] = args.rate
    if getattr(args, "duration", None) is not None:
        ov["load.stages.0.duration"] = args.duration
    if getattr(args, "num_requests", None) is not None:
        ov["load.stages.0.num_requests"] = args.num_requests

    if getattr(args, "input_tokens", None) is not None:
        if dtype == "shared_prefix":
            ov["data.shared_prefix.system_prompt_len"] = args.input_tokens
        elif dtype == "conversation_replay":
            ov["data.conversation_replay.shared_system_prompt_len"] = args.input_tokens
        else:
            for k in DIST_KEYS:
                ov[f"data.input_distribution.{k}"] = args.input_tokens
    if getattr(args, "unique_prompts", None) is not None:
        if dtype in ("random", "synthetic"):
            ov["data.input_distribution.total_count"] = args.unique_prompts
        elif dtype == "shared_prefix":
            ov["data.shared_prefix.num_groups"] = args.unique_prompts
        else:
            log.warning("--unique-prompts has no meaning for data.type=%s; use --set", dtype)
    if getattr(args, "output_tokens", None) is not None:
        if dtype == "shared_prefix":
            ov["data.shared_prefix.output_len"] = args.output_tokens
        elif dtype == "conversation_replay":
            for k in DIST_KEYS:
                ov[f"data.conversation_replay.output_tokens_per_turn.{k}"] = args.output_tokens
        else:
            for k in DIST_KEYS:
                ov[f"data.output_distribution.{k}"] = args.output_tokens
    return ov


def _collect_overrides(args, spec) -> dict:
    ov = _convenience_overrides(args, spec)
    for expr in getattr(args, "set", []) or []:
        key, value = parse_set_expr(expr)
        ov[key] = value
    return ov


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-dir", default="/opt/ls/agentic_kv_bench",
                   help="root for all output (default: %(default)s)")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--vllm-log", default="",
                   help="path to vllm_server.log (lets prediction parse true KV pool size)")
    p.add_argument("--vllm-boot-wait-s", type=int, default=900)
    p.add_argument("--manage-vllm", action="store_true",
                   help="launch/stop vLLM ourselves per the spec's vllm_launch: block, "
                        "instead of assuming it's already running externally")
    p.add_argument("--vllm-venv", default="/opt/ls/vllm/.venv/bin/activate",
                   help="activate script for vLLM's venv (used with --manage-vllm)")
    p.add_argument("--vllm-stop-grace-s", type=int, default=60)
    p.add_argument("--min-free-gb", type=float, default=15)
    p.add_argument("--settle-s", type=int, default=180, help="settle between runs")
    p.add_argument("--lmcache-disk-path", default="/opt/ls/lmcache-disk-cache")
    p.add_argument("--amoprof-bin", default=str(Path.home() / "amoprof_venv/bin/amoprof"))
    p.add_argument("--amoprof-port", type=int, default=9101)
    p.add_argument("--no-amoprof", action="store_true", help="skip amoprof entirely")
    p.add_argument("--no-sudo", action="store_true", help="amoprof/kill without sudo")
    p.add_argument("--dry-run", action="store_true", help="plan only; execute nothing")
    p.add_argument("--force", action="store_true", help="re-run even if a run is already OK")
    p.add_argument("--sweep-dir", default="",
                   help="use this sweep dir verbatim (for cross-day resume)")
    p.add_argument("--log-level", default="INFO")


def _add_workload_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", action="append", metavar="DOTTED.PATH=VALUE",
                   help="override any driver config key; repeatable. "
                        "Example: --set load.stages.0.concurrency_level=16")
    p.add_argument("--concurrency", type=int, help="sessions/requests in flight")
    p.add_argument("--rate", type=float, help="request rate QPS (constant/poisson load)")
    p.add_argument("--duration", type=int, help="stage duration seconds (constant/poisson)")
    p.add_argument("--num-requests", type=int, help="requests per stage (concurrent load)")
    p.add_argument("--input-tokens", type=int, help="prompt length in tokens (fixed)")
    p.add_argument("--unique-prompts", type=int, help="unique prompt pool size")
    p.add_argument("--output-tokens", type=int, help="generation length in tokens (fixed)")
    p.add_argument("--model", default="", help="override spec model (HF id)")


def cmd_run(args) -> int:
    spec = load_workload_spec(Path(args.spec))
    if args.model:
        spec.model = args.model
    ov = _collect_overrides(args, spec)
    runner = SweepRunner(args)
    return runner.run_sweep(spec.source_path, args.sweep_name or spec.name, [ov], [""])


def cmd_sweep(args) -> int:
    if args.sweep:
        ss = load_sweep_spec(Path(args.sweep))
        runner = SweepRunner(args)
        return runner.run_sweep(ss.base_spec_path, ss.name,
                                [r.overrides for r in ss.runs], [r.label for r in ss.runs])
    # --spec + --concurrency-list sugar: one run per level
    spec = load_workload_spec(Path(args.spec))
    if args.model:
        spec.model = args.model
    base_ov = _collect_overrides(args, spec)
    levels = parse_scalar_list(args.concurrency_list, int)
    runs, labels = [], []
    for c in levels:
        ov = dict(base_ov)
        ov.update(_concurrency_override(spec, c))
        runs.append(ov)
        labels.append(f"c{c:03d}")
    runner = SweepRunner(args)
    return runner.run_sweep(spec.source_path, args.sweep_name or spec.name, runs, labels)


def _concurrency_override(spec, level: int) -> dict:
    load = spec.inference_perf.get("load", {}) or {}
    ltype = load.get("type", "constant")
    if ltype == "concurrent":
        return {"load.stages.0.concurrency_level": level}
    if ltype == "trace_session_replay":
        return {"load.stages.0.concurrent_sessions": level}
    return {"load.stages.0.rate": level}


def cmd_preflight(args) -> int:
    import shutil
    from .runner import vllm_healthy
    from .util import free_gb, port_open, which_all
    ok = True
    def check(good: bool, msg: str):
        nonlocal ok
        log.info("%s %s", "PASS" if good else "FAIL", msg)
        ok = ok and good

    check(shutil.which("inference-perf") is not None, "inference-perf on PATH")
    check(not which_all(["curl", "pgrep", "flock"]), "system tools present")
    check(vllm_healthy(args.vllm_url), f"vLLM healthy at {args.vllm_url}")
    if not args.no_amoprof:
        check(Path(args.amoprof_bin).exists(), f"amoprof at {args.amoprof_bin}")
        check(not port_open("127.0.0.1", args.amoprof_port),
              f"amoprof port {args.amoprof_port} free (no stale instance)")
    check(free_gb(Path(args.base_dir)) >= args.min_free_gb,
          f">={args.min_free_gb:.0f}G free under {args.base_dir}")
    return 0 if ok else 1


def cmd_report(args) -> int:
    import json
    sweep_dir = Path(args.sweep_dir)
    summaries = sorted(sweep_dir.glob("*/run_summary.json"))
    if not summaries:
        log.error("no run_summary.json files under %s", sweep_dir)
        return 1
    rows = []
    for s in summaries:
        d = json.loads(s.read_text())
        perf = d.get("perf") or {}
        rows.append({
            "run": s.parent.name,
            "verdict": d.get("verdict", ""),
            "pressure": d.get("max_kv_pressure"),
            "lmcache_moved": len((d.get("lmcache_activity") or {}).get("deltas") or {}),
            "amoprof_files": d.get("amoprof_files", 0),
            "ttft_mean": (perf.get("ttft_ms") or {}).get("mean"),
            "ttft_p90": (perf.get("ttft_ms") or {}).get("p90"),
            "tpot_mean": (perf.get("tpot_ms") or {}).get("mean"),
        })
    headers = ["run", "verdict", "pressure", "lmcache_moved", "amoprof_files",
               "ttft_mean", "ttft_p90", "tpot_mean"]
    widths = [max(len(h), *(len(str(r[h] if r[h] is not None else "")) for r in rows)) for h in headers]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[h] if r[h] is not None else "").ljust(w)
                        for h, w in zip(headers, widths)))
    out = sweep_dir / "sweep_report.md"
    with open(out, "w") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "|".join("---" for _ in headers) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[h] if r[h] is not None else "") for h in headers) + " |\n")
    log.info("wrote %s", out)
    return 0


def cmd_name(args) -> int:
    spec = load_workload_spec(Path(args.spec))
    if args.model:
        spec.model = args.model
    ov = _collect_overrides(args, spec)
    merged = spec.merged_inference_perf(ov)
    knobs = build_knob_label(merged, spec.run_knobs)
    layout_name = f"01-{model_short(spec.model)}_{spec.name}" + (f"_{knobs}" if knobs else "")
    print(layout_name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="environment checks only")
    _add_common(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("run", help="single workload spec")
    p.add_argument("--spec", required=True)
    p.add_argument("--sweep-name", default="")
    _add_common(p)
    _add_workload_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="N runs from a sweep spec or --concurrency-list")
    p.add_argument("--sweep", default="", help="sweep spec YAML")
    p.add_argument("--spec", default="", help="workload spec (with --concurrency-list)")
    p.add_argument("--concurrency-list", default="", help='e.g. "1 8 16 100 250"')
    p.add_argument("--sweep-name", default="")
    _add_common(p)
    _add_workload_flags(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("report", help="aggregate a sweep dir")
    p.add_argument("--sweep-dir", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("name", help="print the run dir name a spec would get")
    p.add_argument("--spec", required=True)
    _add_workload_flags(p)
    p.set_defaults(func=cmd_name)

    args = parser.parse_args(argv)
    if args.command == "sweep" and not args.sweep and not (args.spec and args.concurrency_list):
        parser.error("sweep needs --sweep FILE or (--spec FILE --concurrency-list \"1 8 16\")")
    if args.command in ("run", "sweep", "preflight"):
        setup_logging(getattr(args, "log_level", "INFO"))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
