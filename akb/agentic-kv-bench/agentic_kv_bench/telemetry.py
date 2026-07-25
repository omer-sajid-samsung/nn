"""Telemetry: phase markers, vLLM /metrics scraper, LMCache snapshots.

Markers are the sync mechanism: AMOProf rows, vLLM scrapes and harness events
all share wall clock, so a marker file of epoch timestamps per phase boundary
is what lets you line everything up afterwards.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

from .util import log, utc_now_iso


class MarkerLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.touch()

    def mark(self, key: str, value) -> None:
        with open(self.path, "a") as fh:
            fh.write(f"{key}={value}\n")

    def phase(self, name: str) -> None:
        """One call writes both the epoch and the human-readable UTC."""
        self.mark(f"{name}_epoch", int(time.time()))
        self.mark(f"{name}_utc", utc_now_iso())


class VLLMMetricsScraper:
    """Appends `# scrape_epoch=...` + raw exposition text, like the bash harness."""

    def __init__(self, base_url: str, out_path: Path, interval_s: int = 15):
        self.url = base_url.rstrip("/") + "/metrics"
        self.out_path = out_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vllm-scraper")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._scrape_once()
            self._stop.wait(self.interval_s)

    def _scrape_once(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            body = f"# scrape_error={e}"
        with open(self.out_path, "a") as fh:
            fh.write(f"# scrape_epoch={int(time.time())}\n{body}\n\n")

    def stop(self, timeout_s: float = 10) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)


class LMCacheSnapshotter:
    """Best-effort snapshots of the LMCache internal API before/after a run.

    We probe a configurable endpoint list and record whatever answers 200.
    Never fails a run — LMCache telemetry also flows through amoprof; this is
    belt-and-suspenders, and the diff feeds the eviction verdict.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6999,
                 endpoints: tuple[str, ...] = ("/metrics", "/stats", "/cache/stats", "/health")):
        self.base = f"http://{host}:{port}"
        self.endpoints = endpoints

    def snapshot(self, out_path: Path) -> dict:
        snap = {"epoch": int(time.time()), "utc": utc_now_iso(), "endpoints": {}}
        for ep in self.endpoints:
            try:
                with urllib.request.urlopen(self.base + ep, timeout=5) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                snap["endpoints"][ep] = {"status": 200, "body": body}
            except Exception as e:
                snap["endpoints"][ep] = {"status": None, "error": str(e)}
        out_path.write_text(json.dumps(snap, indent=2) + "\n")
        got = [ep for ep, r in snap["endpoints"].items() if r.get("status") == 200]
        log.info("LMCache snapshot -> %s (answered: %s)", out_path.name, ", ".join(got) or "none")
        return snap
