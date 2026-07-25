# agentic-kv-bench (`akb`)

Controllable agentic workload harness for **vLLM + LMCache KV-offload profiling with AMOProf**.

This is the deliverable, not a script that produces numbers once: a spec-driven
harness that forces KV cache eviction on demand, runs the workload through
**[inference-perf](https://github.com/kubernetes-sigs/inference-perf)** (the
kubernetes-sigs standard load generator — engine-agnostic, so the same specs
run on the T4 dev box, house A100s, and MI355 unchanged), brackets every run
with AMOProf telemetry, and **pre-registers a physics prediction per run and
then checks the run against it**.

```
prediction (manifest.json)  ->  workload (inference-perf)  ->  verdict (run_summary.json)
   "this run MUST evict"          TTFT/TPOT per request        CONFIRMED / NOT_OBSERVED
```

If the verdict says `NOT_OBSERVED`, the harness — not the physics — is wrong,
and the run should not be reported. That is the whole point.

---

## Install

```bash
# on the bench box
cd agentic-kv-bench
pip install -e .            # gives you the `akb` command
pip install inference-perf  # the load driver
```

Requirements: Python ≥3.10, PyYAML (pulled by inference-perf). Optional:
`huggingface-hub` for model-config KV math (already present if vLLM is).

## The 10-minute smoke test (T4 dev box)

Start vLLM exactly as you already do (LMCache disk backend,
`--no-enable-prefix-caching`), then:

```bash
akb preflight --vllm-url http://127.0.0.1:8000

# The W2 spec is pre-tuned for the T4: 12 unique 30k-token prompts ≈ 361k
# tokens of KV vs a ~260k-token GPU pool => eviction MUST happen.
akb run --spec specs/w2_revisit_thrash.yaml --vllm-log /path/to/vllm_server.log
```

Pass criteria: `verdict: CONFIRMED` in the output and in
`runs/<date>_w2-revisit-thrash/*/run_summary.json`. This is the test the old
AgentBench runs could never pass — AgentBench prefixes are ~2k tokens, the
aggregate working set never reaches the pool, and the disk tier sits idle.

## The actual workload matrix

| Spec | Regime | The knob |
|---|---|---|
| `specs/w1_concurrency_cell.yaml` | N concurrent long sessions | `--concurrency-list "1 8 16 100 250"` |
| `specs/w2_revisit_thrash.yaml` | **SSD leg**: bounded unique-prompt pool > KV pool, revisited | `data.input_distribution.total_count` |
| `specs/w3_single_longctx.yaml` | single-session worst case (128k/256k), chunked prefill | `data.input_distribution.mean` |
| `specs/w4_agentic_conversation.yaml` | agentic multi-turn: big shared env prompt + turns + tool latency | `data.conversation_replay.*` |

```bash
# concurrency sweep (one run per level, resume-safe)
akb sweep --spec specs/w1_concurrency_cell.yaml --concurrency-list "1 8 16"

# or a declarative sweep file
akb sweep --sweep sweeps/w2_concurrency_sweep.yaml

# override anything, schema-free
akb run --spec specs/w2_revisit_thrash.yaml \
    --input-tokens 32768 --unique-prompts 64 --concurrency 16 \
    --set load.stages.0.num_requests=640

# what would this run be called?
akb name --spec specs/w2_revisit_thrash.yaml --concurrency 16

# aggregate after a sweep
akb report --sweep-dir /opt/ls/agentic_kv_bench/runs/20260725_w2-thrash-c-sweep
```

Resume: re-run the same command (same day) and finished-`OK` runs are skipped.
For cross-day resume pass `--sweep-dir <existing dir>`. `--force` re-runs anyway.

## Where runs live

```
{--base-dir}/runs/{YYYYMMDD}_{sweep-name}/
├── sweep_manifest.json          # what was planned
├── sweep_summary.csv            # one row per run, appended as runs finish
├── sweep.log
└── 01-qwen3-8b-awq_w2-revisit-thrash_c16_i30000_u12/
    ├── spec.yaml                # resolved workload spec (frozen)
    ├── inference_perf.yaml      # rendered driver config (frozen)
    ├── manifest.json            # provenance + PRE-REGISTERED PREDICTION
    ├── markers.txt              # epoch+UTC per phase boundary (telemetry sync)
    ├── status                   # OK / TIMEOUT / FAIL_n / AMOPROF_START_FAIL
    ├── run_summary.json         # verdict + perf + telemetry digest
    ├── logs/                    # driver.log, amoprof_service.log
    ├── telemetry/               # vllm_metrics.prom, lmcache_before/after.json
    ├── amoprof/                 # AMOProf's own output (--output-dir)
    └── report/                  # inference-perf reports (per-request lifecycle!)
```

## The prediction contract (why Mandar will trust this)

Before every run, the harness computes and freezes into `manifest.json`:

- `kv_bytes_per_token` — from the model's `config.json`
  (layers × 2 × kv_heads × head_dim × dtype_bytes; override via
  `prediction.kv_bytes_per_token` or `prediction.model_config_path`)
- `pool_tokens` — from `--vllm-log` (vLLM logs `GPU KV cache size: N tokens`),
  else `prediction.pool_tokens`, else an nvidia-smi estimate (marked as such)
- `working_set_tokens` — from the data generator's uniqueness knobs
- `expect_eviction` — `working_set > pool`

After the run, `run_summary.json` carries the verdict:

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | predicted eviction; peak KV pressure ≥95%; LMCache counters moved |
| `LIKELY` | predicted; pressure high; LMCache telemetry silent |
| `NOT_OBSERVED` | predicted but pressure stayed low — **harness bug or bad pool estimate; do not report this run** |
| `UNEXPECTED_PRESSURE` | no eviction predicted but the pool filled anyway |
| `IN_REGIME` | control run: no eviction predicted, none observed |
| `INDETERMINATE` | telemetry missing; the harness says so instead of guessing |

## AMOProf integration

AMOProf runs as a subprocess **before and after** each run:

1. **before**: `amoprof service ... --output-dir <run>/amoprof` — readiness is
   judged by the metrics port answering (the launcher daemonizes and exits
   immediately), stale instances on the port are refused, and the daemon's
   real PID is found via pgrep. Sudo credentials are kept alive for the whole
   sweep.
2. **after**: process-group TERM with a long grace, then KILL; settle window
   for the final scrape; then an optional `post_run` command (see below).

Configure per spec (defaults match the original bash harness):

```yaml
amoprof:
  collectors: gpu,dram,vmstat,iostat,smart
  ssd_device: /dev/nvme1n1
  lmcache_bytes_per_token: 18432
  post_run:                       # runs after stop, per run
    cmd: [/path/to/amoprof, report, --input, "{amoprof_out}", --output, "{run_dir}/amoprof/report.html"]
    timeout_s: 300
```

`markers.txt` is the sync mechanism: AMOProf rows, vLLM scrapes and driver
windows share wall clock — keep rows where
`workload_start_epoch <= ts <= workload_end_epoch`.

## Operational semantics (inherited from the overnight bash harness)

- one sweep at a time per base dir (`flock`)
- a failed/timed-out run is recorded and the sweep **continues**
- workload and amoprof are killed as **process trees** (own pgid via `start_new_session`)
- SIGINT/SIGTERM: finish cleanup of the current run, then stop
- `--min-free-gb` disk guard before every run
- run inside `tmux` for overnight sweeps

## Notes for the report

- `--no-enable-prefix-caching` on the server is **required**: vLLM APC would
  shadow LMCache hits and contaminate the revisit signal.
- `--enforce-eager` keeps attribution clean; TPOT under CUDA graphs will
  differ — say so in the report.
- W3 at 256k **exceeds Qwen3-32B's official envelope** (32k native / 131k
  YaRN). It is a KV-volume/chunked-prefill cell, not a quality cell.
  Qwen3-VL-32B does 256k natively; the multimodal leg is clean.
- LMCache's benefit shows up in **TTFT** under memory pressure, sometimes at
  the cost of ITL. Pre-register that expectation so worse decode numbers are
  not a surprise finding.
