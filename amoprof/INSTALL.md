# AMOprof Installation Guide

This guide covers installation for metrics collection, SGLang analysis, and optional low-level tracing dependencies.

---

## Requirements

- Linux host with Python 3.10 or newer.
- Root or passwordless sudo for blktrace, biosnoop, perf, and DRAM PMU collection.
- SGLang `/metrics` endpoint if collecting SGLang counters.
- NVIDIA tools (`nvidia-smi`, optionally DCGM/dcgmi) for GPU/HBM collection.
- NVMe/block-device tools when profiling local L3 storage.

---

## Install AMOprof

```bash
mkdir -p amoprof_1_39_94
unzip amoprof_1_39_94.zip -d amoprof_1_39_94
cd amoprof_1_39_94
chmod +x install.sh scripts/install_dram_deps.sh
./install.sh
source ~/amoprof_venv/bin/activate
amoprof --version
```
> **Install path note:** run `pip install -e .` only from the package root — the directory containing `pyproject.toml`, `setup.py`, `README.md`, and the inner `amoprof/` Python package. Do **not** run it from inside the inner `amoprof/` directory. If pip says neither `setup.py` nor `pyproject.toml` was found, run `cd ..` until `ls` shows one of those files.


The default install is metrics-only. It does not install PyTorch or SGLang server dependencies.

---

## Installation modes

```bash
./install.sh                  # AMOprof + core metrics/reporting dependencies
./install.sh --sglang         # add PyTorch/SGLang client/server dependencies
./install.sh --full           # add SGLang + SWE-bench/SWE-agent dependencies
./install.sh --skip-apt       # skip system package installation
./install.sh --skip-venv      # install into the current Python environment
./install.sh --venv DIR       # use a custom virtualenv location
```

For systems that require DRAM bandwidth collection:

```bash
sudo ./install.sh --with-dram-deps --dram-tool auto
```

---

## Architecture-aware dependency installation

`install.sh` detects:

- CPU architecture from `uname -m`.
- CPU vendor from `/proc/cpuinfo`.
- Linux distribution and package manager from `/etc/os-release` and available package manager commands.
- GPU tool availability from `nvidia-smi` and `dcgmi`.

It installs best-effort package names for these package managers:

| Package manager | Distros |
|---|---|
| `apt` | Ubuntu, Debian, DGX OS variants |
| `dnf` / `yum` | RHEL, Rocky, Alma, Fedora |
| `zypper` | SUSE / SLES / openSUSE |
| `pacman` | Arch-based systems |

The DRAM PMU helper, `scripts/install_dram_deps.sh`, additionally chooses the backend based on architecture and CPU vendor:

| Platform | Default `--dram-tool auto` backend |
|---|---|
| AMD x86_64 | AMD uProf / `AMDuProfPcm` prerequisites; installer path optional |
| Intel x86_64 | Intel PCM / `pcm-memory`; optional source build |
| Other architectures | perf IMC fallback where supported |
| Unknown CPU vendor | perf IMC fallback where supported |

AMD uProf is not silently downloaded unless you provide an installer path or URL.

---

## Optional DRAM PMU installation examples

Auto-detect CPU/vendor:

```bash
sudo scripts/install_dram_deps.sh --dram-tool auto
```

AMD host with a local AMD uProf installer:

```bash
sudo scripts/install_dram_deps.sh \
  --dram-tool amduprof \
  --amduprof-installer ./AMDuProf_Linux_x64.deb
```

Intel host with native package or source-build fallback:

```bash
sudo scripts/install_dram_deps.sh \
  --dram-tool intel-pcm \
  --build-intel-pcm
```

Perf IMC fallback only:

```bash
sudo scripts/install_dram_deps.sh --dram-tool perf-imc
```

Dry-run without installing:

```bash
scripts/install_dram_deps.sh --dram-tool auto --dry-run
```

---

## System tools by collector

| Collector / report area | Required tools |
|---|---|
| GPU/HBM utilization, power, memory | `nvidia-smi`, optionally `dcgmi` / DCGM |
| SGLang metrics | SGLang `/metrics`, `curl` for manual checks |
| Local L3 physical block I/O | `blktrace`, `blkparse`, `iostat`, `nvme-cli`, sysfs |
| Per-stream storage attribution | `biosnoop-bpfcc` or BCC biosnoop |
| DRAM bandwidth | AMD uProf, Intel PCM, or perf IMC fallback |
| OS memory pressure | `/proc/vmstat`, `/proc/meminfo`, `numactl` |
| Optional detailed tracing | `bpftrace`, Nsight Systems `nsys` |

---

## Collection examples

### Local SGLang + local NVMe L3

```bash
sudo amoprof collect \
  --output-dir ./amoprof_results \
  --label dgx_lc_l3_nvme \
  --duration-s 900 \
  --interval-s 1 \
  --sglang-host 127.0.0.1 \
  --sglang-port 30000 \
  --model openai/gpt-oss-120b \
  --ssd-device /dev/nvme2n1 \
  --hicache-path /mnt/sglang_hicache \
  --enable-dram \
  --dram-tool auto \
  --enable-blktrace \
  --enable-biosnoop \
  --setup-details setup_details_sample.json \
  --strict-sanity
```

### Remote SGLang metrics + local storage node tracing

```bash
sudo amoprof collect \
  --output-dir ./amoprof_results \
  --label storage_node_l3_trace \
  --duration-s 900 \
  --interval-s 1 \
  --sglang-host 10.0.1.5 \
  --sglang-port 30000 \
  --ssd-device /dev/nvme2n1 \
  --hicache-path /mnt/sglang_hicache \
  --enable-blktrace \
  --enable-biosnoop \
  --setup-details setup_details_sample.json
```

---

## Analysis examples

### Analyze a collected run

```bash
amoprof analyze \
  --run-dir ./amoprof_results/metrics_run_YYYYMMDD_HHMMSS \
  --combined-report \
  --interactive-report \
  --prom-rate-window 5m
```

### Prometheus-only report

```bash
amoprof analyze \
  --prometheus-url http://PROM_HOST:9090 \
  --prom-instance msl-ssg-dgx1.msl.lab:30000 \
  --start 2026-06-17T23:15:54Z \
  --end   2026-06-17T23:25:54Z \
  --prom-step 15s \
  --combined-report \
  --interactive-report \
  --setup-details setup_details_sample.json \
  --prefer prometheus
```

### Analyze with benchmark summary for validation

```bash
amoprof analyze \
  --run-dir ./amoprof_results/metrics_run_YYYYMMDD_HHMMSS \
  --bench-summary ./perf_metrics_summary.json \
  --combined-report \
  --interactive-report
```

---

## Privilege guidance

Most passive CSV/report analysis does not need root. These collection features usually do:

- `--enable-blktrace`
- `--enable-biosnoop`
- `--enable-dram` with PMU tools
- `perf`-based collectors
- some NVMe SMART queries

For unattended collection, configure passwordless sudo for the needed binaries or run `amoprof collect` under sudo.

---

## Verify installation

```bash
source ~/amoprof_venv/bin/activate
amoprof --version
python -c "import amoprof; print(amoprof.__version__)"
python -m compileall -q amoprof
pytest -q tests/test_executive_symbol_regression.py
```

---

## Troubleshooting

### `amoprof` command not found
Activate the virtualenv:

```bash
source ~/amoprof_venv/bin/activate
```

### DRAM charts are empty
Install/check backend dependencies:

```bash
sudo scripts/install_dram_deps.sh --dram-tool auto
```

Then collect with:

```bash
sudo amoprof collect --enable-dram --dram-tool auto ...
```

### L3 physical charts are empty
Confirm the run used the correct device and tracing was enabled:

```bash
sudo amoprof collect --ssd-device /dev/nvme2n1 --enable-blktrace ...
```

### SMART/endurance fields are empty
Check `nvme-cli` and permissions:

```bash
sudo nvme smart-log /dev/nvme2n1
```


## L3 vs L3.5 setup input

To make report labels deterministic, add one of the following to `setup_details.json`:

```json
{ "KV cache tier": "L3", "L3 storage type": "NVMe SSD" }
```

for local SSD/NVMe, or:

```json
{ "KV cache tier": "L3.5", "L3 storage type": "AI Memory Node / remote storage" }
```

for AI Memory Node, Mooncake, or any remote/shared/disaggregated backing tier. If the field is omitted, AMOprof derives the label from backend/device/path evidence.
