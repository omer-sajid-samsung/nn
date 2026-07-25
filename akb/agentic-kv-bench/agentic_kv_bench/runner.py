"""Run orchestration: one run, and the sweep loop.

Phase order per run (each boundary is marked with epoch+UTC for telemetry sync):

    preflight-light -> amoprof start -> scraper start -> LMCache before
    -> DRIVER -> LMCache after -> scraper stop -> amoprof stop
    -> amoprof post-run cmd -> validate -> status + summary row

A failed or timed-out run is recorded and the sweep CONTINUES — one bad run
must not waste the whole night. Re-run the same sweep and finished-OK runs
are skipped (resume).
"""

from __future__ import annotations

import json
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import __version__
from .amoprof import AMOProfService, SudoKeepalive
from .driver import (EXIT_ABORT, EXIT_TIMEOUT, render_driver_config, run_driver,
                     write_driver_config)
from .naming import (RunLayout, build_knob_label, make_run_dir, make_sweep_dir)
from .prediction import build_prediction
from .spec import SweepSpec, WorkloadSpec, load_workload_spec
from .telemetry import LMCacheSnapshotter, MarkerLog, VLLMMetricsScraper
from .util import (FileLock, check_disk_space, free_gb, log, read_text_quiet,
                   setup_logging, utc_now_iso, utc_stamp, which_all)
from .vllm_proc import VLLMService

import yaml


class AbortFlag:
    def __init__(self):
        self._event = threading.Event()

    def set(self):
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def vllm_healthy(base_url: str, timeout: float = 5.0) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def ensure_vllm(base_url: str, wait_s: int, abort: AbortFlag) -> bool:
    if vllm_healthy(base_url):
        return True
    log.warning("vLLM not answering at %s — waiting up to %ds...", base_url, wait_s)
    waited = 0
    while waited < wait_s:
        if abort.is_set():
            return False
        time.sleep(10)
        waited += 10
        if vllm_healthy(base_url):
            return True
    return False


def collect_provenance(vllm_url: str, model: str) -> dict:
    prov = {
        "harness_version": __version__,
        "date_utc": utc_now_iso(),
        "host": _run(["hostname"]),
        "kernel": _run(["uname", "-r"]),
        "gpus": _run(["nvidia-smi", "-L"]) or "nvidia-smi unavailable",
        "vllm_url": vllm_url,
        "model": model,
    }
    for pkg in ("inference-perf", "vllm", "lmcache"):
        show = _run(["pip", "show", pkg])
        version = next((l.removeprefix("Version: ") for l in show.splitlines()
                        if l.startswith("Version: ")), "")
        prov[f"pip_{pkg}"] = version or "not installed"
    return prov


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


STATUS_OK = "OK"


class RunRunner:
    """Executes a single run. Owns nothing across runs."""

    def __init__(self, args, spec: WorkloadSpec, overrides: dict, abort: AbortFlag):
        self.args = args
        self.spec = spec
        self.overrides = overrides
        self.abort = abort

    def execute(self, layout: RunLayout, seq_label: str) -> dict:
        markers = MarkerLog(layout.markers)
        t0 = time.time()
        result = {"run_dir": str(layout.path), "status": "FAIL_UNKNOWN", "duration_s": 0}

        merged = self.spec.merged_inference_perf(self.overrides)
        knob_label = build_knob_label(merged, self.spec.run_knobs)

        # ---- resolved spec + driver config, frozen into the run dir --------
        layout.spec_file.write_text(yaml.safe_dump({
            "name": self.spec.name,
            "description": self.spec.description,
            "model": self.spec.model,
            "overrides": self.overrides,
            "prediction": self.spec.prediction,
            "telemetry": self.spec.telemetry,
            "amoprof": self.spec.amoprof,
            "timeouts": self.spec.timeouts,
        }, sort_keys=False))
        tel = self.spec.telemetry
        driver_cfg = render_driver_config(
            model=self.spec.model, base_url=self.args.vllm_url, layout=layout,
            spec_inference_perf=self.spec.inference_perf, overrides=self.overrides,
            prometheus_url=tel.get("prometheus_url"),
            prometheus_scrape_interval=int(tel.get("prometheus_scrape_interval", 15)))
        write_driver_config(driver_cfg, layout.driver_config)

        # ---- pre-registered prediction ------------------------------------
        prediction = build_prediction(
            self.spec.prediction, driver_cfg,
            vllm_log_path=Path(self.args.vllm_log) if self.args.vllm_log else None)
        if prediction.expect_eviction is not None:
            log.info("prediction: working_set=%s tokens vs pool=%s tokens -> expect_eviction=%s",
                     prediction.working_set_tokens, prediction.pool_tokens, prediction.expect_eviction)
        for note in prediction.notes:
            log.info("prediction note: %s", note)

        manifest = {
            "run_id": layout.path.name,
            "seq": seq_label,
            "knobs": knob_label,
            "spec_name": self.spec.name,
            "model": self.spec.model,
            "started_utc": utc_now_iso(),
            "provenance": collect_provenance(self.args.vllm_url, self.spec.model),
            "prediction": prediction.to_dict(),
            "overrides": self.overrides,
        }
        layout.write_manifest(manifest)

        # ---- light per-run preflight ---------------------------------------
        check_disk_space(layout.path, float(self.args.min_free_gb))
        if not ensure_vllm(self.args.vllm_url, int(self.args.vllm_boot_wait_s), self.abort):
            layout.write_status("VLLM_DOWN")
            result.update(status="VLLM_DOWN", duration_s=int(time.time() - t0))
            return result

        markers.mark("run_id", layout.path.name)
        markers.mark("knobs", knob_label)
        markers.mark("spec", self.spec.name)

        amoprof_cfg = {**self.spec.amoprof, "port": self.args.amoprof_port}  # CLI wins over spec
        amoprof = AMOProfService(
            bin=self.args.amoprof_bin, cfg=amoprof_cfg, sudo=not self.args.no_sudo,
            hicache_path=self.args.lmcache_disk_path) if not self.args.no_amoprof else None
        scraper = VLLMMetricsScraper(
            self.args.vllm_url, layout.vllm_metrics,
            interval_s=int(tel.get("scrape_interval_s", 15))) if tel.get("scrape_vllm_metrics", True) else None
        lmcache = LMCacheSnapshotter(
            port=int(tel.get("lmcache_port", 6999)),
            endpoints=tuple(tel.get("lmcache_endpoints", ["/metrics", "/stats", "/cache/stats", "/health"])))

        driver_status = 1
        try:
            # ---- amoprof BEFORE --------------------------------------------
            if amoprof:
                markers.phase("amoprof_start")
                ok = amoprof.start(
                    layout.amoprof_out, layout.amoprof_log,
                    ready_timeout_s=int(self.spec.timeouts.get("amoprof_ready_s", 90)),
                    post_start_settle_s=int(self.spec.timeouts.get("post_start_settle_s", 10)),
                    abort=self.abort)
                if not ok:
                    layout.write_status("AMOPROF_START_FAIL")
                    result.update(status="AMOPROF_START_FAIL", duration_s=int(time.time() - t0))
                    return result
                markers.phase("amoprof_ready")

            if scraper:
                markers.phase("telemetry_start")
                scraper.start()
            markers.phase("lmcache_before")
            lmcache.snapshot(layout.lmcache_before)

            # ---- the workload itself ---------------------------------------
            markers.phase("workload_start")
            driver_status = run_driver(
                layout.driver_config, layout,
                timeout_s=int(self.spec.timeouts.get("run_timeout_s", 0)),
                kill_grace_s=int(self.spec.timeouts.get("workload_kill_grace_s", 20)),
                abort=self.abort,
                extra_args=list(tel.get("driver_extra_args", [])))
            markers.phase("workload_end")
            markers.mark("driver_exit_status", driver_status)

            markers.phase("lmcache_after")
            lmcache.snapshot(layout.lmcache_after)
        finally:
            if scraper:
                scraper.stop()
            # ---- amoprof AFTER ---------------------------------------------
            if amoprof:
                amoprof.stop(grace_s=int(self.spec.timeouts.get("amoprof_stop_grace_s", 45)),
                             post_stop_settle_s=int(self.spec.timeouts.get("post_stop_settle_s", 10)))
                markers.phase("amoprof_stop")
                rc = amoprof.run_post(layout.path, self.spec.amoprof.get("post_run"), layout.amoprof_out)
                if rc is not None:
                    markers.mark("amoprof_post_exit_status", rc)

        # ---- validate + bookkeep -------------------------------------------
        markers.phase("validation")
        summary = None
        try:
            from .validate import validate_run
            summary = validate_run(layout, prediction.to_dict(), driver_status)
        except Exception as e:  # validation must never eat a run
            log.exception("validation raised: %s", e)

        status = "OK" if driver_status == 0 else (
            "TIMEOUT" if driver_status == EXIT_TIMEOUT else
            "ABORTED" if driver_status == EXIT_ABORT else f"FAIL_{driver_status}")
        layout.write_status(status)
        duration = int(time.time() - t0)

        manifest["finished_utc"] = utc_now_iso()
        manifest["status"] = status
        manifest["duration_s"] = duration
        if summary:
            manifest["verdict"] = summary["verdict"]
            manifest["verdict_reason"] = summary["verdict_reason"]
        layout.write_manifest(manifest)

        result.update(status=status, duration_s=duration, knobs=knob_label,
                      verdict=summary["verdict"] if summary else "",
                      perf=summary.get("perf", {}) if summary else {})
        log.info("=== %s: done in %ds, status=%s, verdict=%s ===",
                 layout.path.name, duration, status, result.get("verdict", "?"))
        return result


class SweepRunner:
    def __init__(self, args):
        self.args = args
        self.abort = AbortFlag()
        self.keepalive: SudoKeepalive | None = None
        self.vllm_service: VLLMService | None = None

    # -- plumbing ------------------------------------------------------------
    def _install_signal_handlers(self):
        def handler(signum, _frame):
            log.warning("caught signal %s — aborting after current run cleans up", signum)
            self.abort.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _preflight(self, specs_base_dir: Path, spec: WorkloadSpec) -> None:
        missing = which_all(["curl", "pgrep", "flock"])
        if missing:
            raise RuntimeError(f"missing tools: {', '.join(missing)}")
        if not self.args.no_amoprof:
            from .util import port_open
            if port_open("127.0.0.1", int(self.args.amoprof_port)):
                raise RuntimeError(
                    f"port {self.args.amoprof_port} already taken (stale amoprof?). "
                    "Run: sudo pkill -f 'amoprof.*service' (then re-run)")
            if not Path(self.args.amoprof_bin).exists() and not self.args.dry_run:
                raise RuntimeError(f"amoprof not found at {self.args.amoprof_bin}")
            if not self.args.no_sudo:
                self.keepalive = SudoKeepalive()
                if not self.args.dry_run:
                    self.keepalive.start()
        check_disk_space(specs_base_dir, float(self.args.min_free_gb))

        manage_vllm = getattr(self.args, "manage_vllm", False) and bool(spec.vllm_launch)
        if manage_vllm:
            cfg = {**spec.vllm_launch}
            cfg.setdefault("venv", getattr(self.args, "vllm_venv", "") or "")
            self.vllm_service = VLLMService(model=spec.model, cfg=cfg)
            vllm_log_path = specs_base_dir / "vllm_server.log"
            if self.args.dry_run:
                log.info("[dry-run] would launch vLLM: model=%s cfg=%s", spec.model, cfg)
                return
            ok = self.vllm_service.start(
                self.args.vllm_url, vllm_log_path,
                ready_timeout_s=int(self.args.vllm_boot_wait_s),
                already_healthy=lambda: vllm_healthy(self.args.vllm_url), abort=self.abort)
            if not ok:
                raise RuntimeError(f"vLLM failed to start (model={spec.model}); see {vllm_log_path}")
            if not self.args.vllm_log and self.vllm_service.pid is not None:
                self.args.vllm_log = str(vllm_log_path)  # let prediction parse the real pool size
            return

        if not ensure_vllm(self.args.vllm_url, int(self.args.vllm_boot_wait_s), self.abort):
            raise RuntimeError(f"vLLM is not reachable at {self.args.vllm_url}. Start it first "
                               "(or pass --manage-vllm with a vllm_launch: block in the spec).")

    # -- the sweep ------------------------------------------------------------
    def run_sweep(self, spec_path: Path, sweep_name: str,
                  run_overrides: list[dict], run_labels: list[str]) -> int:
        base_dir = Path(self.args.base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(base_dir / ".sweep.lock")
        try:
            lock.acquire()
        except RuntimeError as e:
            log.error("%s", e)
            return 1
        try:
            self._install_signal_handlers()
            sweep_dir = Path(self.args.sweep_dir) if self.args.sweep_dir \
                else make_sweep_dir(base_dir, sweep_name)
            sweep_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(self.args.log_level, sweep_dir / "sweep.log")

            spec = load_workload_spec(spec_path)
            self._preflight(sweep_dir, spec)

            summary_csv = sweep_dir / "sweep_summary.csv"
            if not summary_csv.exists():
                summary_csv.write_text(
                    "run_dir,spec,knobs,status,verdict,duration_s,"
                    "ttft_mean_ms,ttft_p90_ms,tpot_mean_ms,max_kv_pressure\n")

            (sweep_dir / "sweep_manifest.json").write_text(json.dumps({
                "sweep": sweep_name, "spec": str(spec_path), "base_dir": str(base_dir),
                "created_utc": utc_now_iso(), "n_runs": len(run_overrides),
                "overrides": run_overrides, "harness_version": __version__,
            }, indent=2) + "\n")

            log.info("sweep '%s': %d runs. results root: %s", sweep_name, len(run_overrides), sweep_dir)
            results = []
            for i, (ov, label) in enumerate(zip(run_overrides, run_labels), start=1):
                if self.abort.is_set():
                    break
                merged = spec.merged_inference_perf(ov)
                knob_label = build_knob_label(merged, spec.run_knobs)
                layout = make_run_dir(sweep_dir, i, spec.model, label or spec.name, knob_label)
                if not self.args.force and layout.status() == STATUS_OK:
                    log.info("=== %s: already OK, skipping (resume) ===", layout.path.name)
                    continue
                if self.args.dry_run:
                    log.info("[dry-run] would execute run %d/%d in %s with overrides %s",
                             i, len(run_overrides), layout.path, ov)
                    continue
                runner = RunRunner(self.args, spec, ov, self.abort)
                result = runner.execute(layout, f"{i:02d}")
                results.append(result)
                self._append_summary(summary_csv, layout, result)
                if not self.abort.is_set() and i < len(run_overrides):
                    settle = int(self.args.settle_s)
                    log.info("settling %ds before next run...", settle)
                    time.sleep(settle)

            log.info("sweep finished. summary: %s", summary_csv)
            return 130 if self.abort.is_set() else 0
        finally:
            if self.vllm_service:
                self.vllm_service.stop(grace_s=int(getattr(self.args, "vllm_stop_grace_s", 60)))
            if self.keepalive:
                self.keepalive.stop()
            lock.release()

    @staticmethod
    def _append_summary(csv_path: Path, layout: RunLayout, result: dict) -> None:
        perf = result.get("perf") or {}
        ttft, tpot = perf.get("ttft_ms") or {}, perf.get("tpot_ms") or {}
        pressure = ""
        try:
            summary = json.loads(read_text_quiet(layout.run_summary) or "{}")
            p = summary.get("max_kv_pressure")
            pressure = f"{p:.3f}" if isinstance(p, float) else ""
        except (json.JSONDecodeError, OSError):
            pass
        row = [layout.path.name, result.get("spec", layout.path.name), result.get("knobs", ""),
               result.get("status", ""), result.get("verdict", ""), str(result.get("duration_s", "")),
               str(ttft.get("mean", "")), str(ttft.get("p90", "")), str(tpot.get("mean", "")), pressure]
        with open(csv_path, "a") as fh:
            fh.write(",".join(row) + "\n")
