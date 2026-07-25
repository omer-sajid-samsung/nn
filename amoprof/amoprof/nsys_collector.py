from __future__ import annotations
import csv, subprocess, threading, time
from pathlib import Path

class NsysGpuTraceCollector:
    def __init__(self, pid: int | None = None, duration_s: float = 60.0,
                 binary: str = 'nsys', stats_binary: str = 'nsys',
                 output_base: str | None = None, report: str = 'cuda_gpu_trace',
                 gpu_metrics_devices: str = 'all', extra_args: list[str] | None = None,
                 work_dir: str | None = None):
        self.pid = pid
        self.duration_s = max(float(duration_s), 1.0)
        self.binary = binary
        self.stats_binary = stats_binary
        self.output_base = output_base
        self.report = report
        self.gpu_metrics_devices = gpu_metrics_devices
        self.extra_args = extra_args or []
        self.work_dir = Path(work_dir or '.')
        self.samples: list[dict] = []
        self._result: dict = {}
        self._thread = None
        self._t0 = 0.0
        self.profile_command = ''
        self.stats_command = ''
        self.rep_path = ''
        self.stats_stdout_path = ''
        self.profile_stdout_path = ''

    def start(self):
        self.samples.clear(); self._result = {}; self._t0 = time.time(); self.work_dir.mkdir(parents=True, exist_ok=True)
        if not self.pid:
            self._result = {'nsys_available': False, 'nsys_reason': '--pid or --nsys-pid is required for nsys profiling'}; return
        try:
            subprocess.run([self.binary, '--version'], capture_output=True, text=True, timeout=5)
        except Exception as e:
            self._result = {'nsys_available': False, 'nsys_reason': f'nsys not found: {self.binary}: {e}'}; return
        base = self.output_base or str(self.work_dir / 'dram_record')
        cmd = [self.binary, 'profile', '-p', str(self.pid), '--duration', str(int(self.duration_s)), '--gpu-metrics-devices', str(self.gpu_metrics_devices), '--force-overwrite', 'true', '-o', base]
        cmd += self.extra_args
        self.profile_command = ' '.join(cmd)
        self.rep_path = base if base.endswith('.nsys-rep') else base + '.nsys-rep'
        self.profile_stdout_path = str(self.work_dir / 'nsys_profile_output.log')
        self.stats_stdout_path = str(self.work_dir / f'nsys_{self.report}.csv')
        self._thread = threading.Thread(target=self._run, args=(cmd,), daemon=True); self._thread.start()

    @staticmethod
    def _f(v):
        try:
            if v is None: return None
            t = str(v).strip().replace(',', '')
            if t in {'', 'nan', 'NaN', 'N/A', 'None'}: return None
            return float(t)
        except Exception:
            return None

    @staticmethod
    def _dur_sec(v, col=''):
        x = NsysGpuTraceCollector._f(v)
        if x is None: return None
        c = (col or '').lower()
        if 'ns' in c: return x / 1e9
        if 'us' in c or 'µs' in c: return x / 1e6
        if 'ms' in c: return x / 1e3
        return x

    @staticmethod
    def _bytes(v, col=''):
        x = NsysGpuTraceCollector._f(v)
        if x is None: return None
        c = (col or '').lower()
        if 'kib' in c: return x * 1024
        if 'mib' in c: return x * 1024**2
        if 'gib' in c: return x * 1024**3
        if 'kb' in c: return x * 1000
        if 'mb' in c: return x * 1000**2
        if 'gb' in c: return x * 1000**3
        return x

    def _run(self, cmd):
        try:
            with open(self.profile_stdout_path, 'w', encoding='utf-8') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=self.duration_s + 120)
        except Exception as e:
            self._result = {'nsys_available': False, 'nsys_reason': str(e), 'nsys_profile_command': self.profile_command}; return
        rep = Path(self.rep_path)
        if not rep.exists():
            cands = sorted(self.work_dir.glob('*.nsys-rep'), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands: rep = cands[0]; self.rep_path = str(rep)
        if not rep.exists():
            self._result = {'nsys_available': False, 'nsys_reason': 'nsys report was not generated', 'nsys_profile_command': self.profile_command, 'nsys_profile_log': self.profile_stdout_path}; return
        stats_cmd = [self.stats_binary, 'stats', f'--report={self.report}', '--format=csv', '--output=.,-', str(rep)]
        self.stats_command = ' '.join(stats_cmd)
        try:
            st = subprocess.run(stats_cmd, cwd=str(self.work_dir), capture_output=True, text=True, timeout=240)
            raw = st.stdout or ''; err = st.stderr or ''
            Path(self.stats_stdout_path).write_text(raw + (('\n# STDERR\n' + err) if err else ''), encoding='utf-8')
            if st.returncode != 0 and not raw.strip():
                self._result = {'nsys_available': False, 'nsys_reason': f'nsys stats failed rc={st.returncode}: {err[-1000:]}', 'nsys_profile_command': self.profile_command, 'nsys_stats_command': self.stats_command, 'nsys_rep_path': str(rep)}; return
            self._parse(raw); self._result = self._summarise()
        except Exception as e:
            self._result = {'nsys_available': False, 'nsys_reason': f'nsys stats/parse failed: {e}', 'nsys_profile_command': self.profile_command, 'nsys_stats_command': self.stats_command, 'nsys_rep_path': str(rep)}

    def _parse(self, raw: str):
        lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith(('Processing', 'Exporting', 'Generating'))]
        if not lines:
            for c in sorted(self.work_dir.glob(f'*{self.report}*.csv')) + sorted(self.work_dir.glob('*.csv')):
                if c.name == Path(self.stats_stdout_path).name: continue
                try:
                    txt = c.read_text(encoding='utf-8', errors='ignore')
                    if ',' in txt: lines = txt.splitlines(); break
                except Exception: pass
        try: rows = list(csv.DictReader('\n'.join(lines).splitlines())) if lines else []
        except Exception: rows = []
        parsed = []
        for i, r in enumerate(rows):
            out = {'ts': round((self._t0 or time.time()) + i, 6), 'nsys_row': i}
            for k, v in r.items():
                if v is None or str(v).strip() == '': continue
                lk = (k or '').lower(); sk = k.strip().replace(' ', '_').replace('/', '_')
                if any(t in lk for t in ['name', 'kernel', 'api', 'device', 'context', 'stream', 'pid', 'tid']): out[sk] = str(v).strip()
                if 'duration' in lk:
                    sec = self._dur_sec(v, k)
                    if sec is not None: out['duration_sec'] = sec
                elif lk.startswith('start') or 'start' in lk:
                    sec = self._dur_sec(v, k)
                    if sec is not None: out['trace_start_sec'] = sec
                if any(t in lk for t in ['dram', 'hbm', 'memory', 'mem']):
                    val = self._bytes(v, k)
                    if val is None: continue
                    if 'read' in lk or 'rd' in lk: out['hbm_read_bytes'] = out.get('hbm_read_bytes', 0.0) + val
                    elif 'write' in lk or 'wr' in lk: out['hbm_write_bytes'] = out.get('hbm_write_bytes', 0.0) + val
                    elif 'throughput' in lk or 'bandwidth' in lk or 'active' in lk: out[sk] = val
            rb = float(out.get('hbm_read_bytes', 0) or 0); wb = float(out.get('hbm_write_bytes', 0) or 0)
            out['hbm_total_bytes'] = rb + wb; out['hbm_read_gb'] = round(rb/(1024**3), 6); out['hbm_write_gb'] = round(wb/(1024**3), 6); out['hbm_total_gb'] = round((rb+wb)/(1024**3), 6)
            parsed.append(out)
        self.samples = parsed

    def stop(self):
        # Give nsys a bounded window to finalize/export; do not hang AMOprof for minutes on Ctrl-C.
        if self._thread:
            self._thread.join(timeout=min(self.duration_s + 60, 120))
        if self._thread and self._thread.is_alive():
            return {
                'nsys_available': False,
                'nsys_reason': 'nsys did not finish before shutdown timeout; report may still be finalizing',
                'nsys_profile_command': self.profile_command,
                'nsys_rep_path': self.rep_path,
                'nsys_profile_log': self.profile_stdout_path,
                'nsys_stats_csv': self.stats_stdout_path,
            }
        return self._result or self._summarise()

    def _summarise(self):
        elapsed = max(time.time() - (self._t0 or time.time()), 0.001)
        rb = sum(float(s.get('hbm_read_bytes', 0) or 0) for s in self.samples); wb = sum(float(s.get('hbm_write_bytes', 0) or 0) for s in self.samples)
        return {'nsys_available': True, 'nsys_profile_command': self.profile_command, 'nsys_stats_command': self.stats_command, 'nsys_rep_path': self.rep_path, 'nsys_profile_log': self.profile_stdout_path, 'nsys_stats_csv': self.stats_stdout_path, 'nsys_report': self.report, 'nsys_samples': len(self.samples), 'hbm_read_bytes_total': int(rb), 'hbm_write_bytes_total': int(wb), 'hbm_total_bytes_total': int(rb+wb), 'hbm_read_gb_total': round(rb/(1024**3), 6), 'hbm_write_gb_total': round(wb/(1024**3), 6), 'hbm_total_gb_total': round((rb+wb)/(1024**3), 6), 'hbm_read_gb_s_est': round(rb/(1024**3)/elapsed, 6), 'hbm_write_gb_s_est': round(wb/(1024**3)/elapsed, 6), 'nsys_duration_s': round(elapsed, 3)}
