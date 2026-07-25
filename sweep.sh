#!/usr/bin/env bash
#===============================================================================
# overnight_agent_sweep.sh
#
# Overnight AgentBench x vLLM(LMCache) x AMOProf sweep.
#
# For each concurrency level C in CONCURRENCY_LIST it:
#   1. Renders an AgentBench assignment config with agent concurrency = C.
#   2. Starts `amoprof service` writing straight into the run's own folder.
#   3. (Optional) Starts a tiny background scraper that snapshots vLLM's
#      Prometheus /metrics endpoint every few seconds into the run folder.
#   4. Writes epoch-timestamp markers (THIS is your sync mechanism: AMOProf
#      rows, vLLM scrapes, and markers all share the same wall clock).
#   5. Runs AgentBench with an optional hard timeout; on timeout/interrupt
#      it kills the whole workload process tree, not just the parent.
#   6. Stops the scraper + amoprof gracefully (long grace windows), records
#      status, appends one line to sweep_summary.csv.
#   7. Sleeps so disk/DRAM settle, then moves to the next C.
#
# A failed or timed-out run is recorded and the sweep CONTINUES — one bad
# run must not waste the whole night. Re-run with the same RUN_NAME and it
# skips runs that already finished OK (resume).
#
# Timing philosophy: everything waits GENEROUSLY. amoprof's `service`
# command daemonizes (its launcher exits immediately even on success), so
# readiness is judged by the metrics port answering, never by the launcher
# PID — and every start/stop gets a long grace window before we give up.
#===============================================================================

set -uo pipefail   # no `-e`: we handle errors per-run so the sweep survives

#===============================================================================
# EDIT-ME: paths and sweep definition
#===============================================================================

# Root for everything this script produces. One subfolder per RUN_NAME,
# one sub-subfolder per concurrency level.
BASE_DIR="${BASE_DIR:-/opt/ls/agentic_kv_bench}"

# Name of this sweep. Becomes the telemetry folder name.
# To RESUME a crashed sweep, re-run with the same RUN_NAME explicitly.
RUN_NAME="${RUN_NAME:-qwen3-8b-awq_lmcache_sweep_$(date +%Y%m%d)}"

# The sweep knob: N concurrent agent sessions, one run per value.
# Override from the shell:  CONCURRENCY_LIST="1 4 8 16 32" ./overnight_agent_sweep.sh
read -r -a CONCURRENCY_LIST <<< "${CONCURRENCY_LIST:-1 8 16}"

# --- AgentBench ---------------------------------------------------------------
AGENTBENCH_DIR="${AGENTBENCH_DIR:-/opt/ls/AgentBench}"   # repo root (has src/assigner)
AGENTBENCH_VENV="${AGENTBENCH_VENV:-/opt/ls/vllm/.venv/bin/activate}"
# Template config. If missing, the script creates it from your example
# with __AGENT_CONCURRENCY__ / __OUTPUT_DIR__ placeholders.
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${AGENTBENCH_DIR}/configs/assignments/sweep_template.yaml}"

# --- vLLM server (started by YOU, externally) ---------------------------------
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
VLLM_BOOT_WAIT_S="${VLLM_BOOT_WAIT_S:-900}"     # patience for model load / recovery
LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-/opt/ls/lmcache-disk-cache}"

# If the server dies mid-night, optionally restart it with your own command
# (0 = just abort the sweep so you can fix it in the morning).
RESTART_VLLM_ON_FAILURE="${RESTART_VLLM_ON_FAILURE:-0}"
VLLM_VENV="${VLLM_VENV:-/opt/ls/vllm/.venv/bin/activate}"
read -r -d '' VLLM_START_CMD <<'EOF' || true
vllm serve Qwen/Qwen3-8B-AWQ \
  --dtype float16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}' \
  --port 8000 \
  --uvicorn-log-level info
EOF
# NOTE: request logging is ON by default in vLLM — `--disable-log-requests=false`
# only parses on newer vLLM versions; drop the `=false` if the server errors on it.

# --- AMOProf ------------------------------------------------------------------
AMOPROF_BIN="${AMOPROF_BIN:-$HOME/amoprof_venv/bin/amoprof}"
AMOPROF_PORT="${AMOPROF_PORT:-9101}"
SUDO="${SUDO:-sudo}"     # amoprof needs root (smart/blktrace); set SUDO="" if it doesn't

#===============================================================================
# Timing knobs — deliberately generous. Overnight runs die from impatience,
# not from waiting an extra minute.
#===============================================================================
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-7200}"            # hard kill per run; 0 = no limit
SLEEP_BETWEEN_RUNS_S="${SLEEP_BETWEEN_RUNS_S:-180}"   # settle between concurrency levels
MIN_FREE_GB="${MIN_FREE_GB:-15}"                  # abort sweep below this free space
RESUME="${RESUME:-1}"                             # 1 = skip runs already OK
SCRAPE_VLLM_METRICS="${SCRAPE_VLLM_METRICS:-1}"   # 1 = snapshot vLLM /metrics per run
VLLM_SCRAPE_INTERVAL_S="${VLLM_SCRAPE_INTERVAL_S:-15}"

AMOPROF_READY_TIMEOUT_S="${AMOPROF_READY_TIMEOUT_S:-90}"  # wait for :9101 to answer
AMOPROF_STOP_GRACE_S="${AMOPROF_STOP_GRACE_S:-45}"        # TERM grace before SIGKILL
POST_START_SETTLE_S="${POST_START_SETTLE_S:-10}"  # let amoprof's first cycle stabilize
                                                  # before the workload starts hammering
POST_STOP_SETTLE_S="${POST_STOP_SETTLE_S:-10}"    # let amoprof flush its final cycle
WORKLOAD_KILL_GRACE_S="${WORKLOAD_KILL_GRACE_S:-20}"  # TERM grace for AgentBench tree
POLL_STEP_S=2                                     # sleep quantum in wait loops

#===============================================================================
# Derived paths + globals (don't edit)
#===============================================================================
RUNS_DIR="${BASE_DIR}/runs"
SWEEP_DIR="${RUNS_DIR}/${RUN_NAME}"
SWEEP_LOG="${SWEEP_DIR}/sweep.log"
SUMMARY_CSV="${SWEEP_DIR}/sweep_summary.csv"

ABORT=0                 # set by signal handler to break the sweep loop
AMOPROF_PID=""          # REAL pid of the amoprof daemon for the CURRENT run
SCRAPER_PID=""          # pid of the vLLM metrics scraper for the CURRENT run
WPID=""                 # pid of the AgentBench workload for the CURRENT run
SUDO_KEEPALIVE_PID=""

#===============================================================================
# Small helpers
#===============================================================================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Append a key=value marker to the current run's markers.txt.
# These epoch timestamps are how you line up AMOProf and vLLM data afterwards:
# keep rows where agentbench_start_epoch <= ts <= agentbench_end_epoch.
mark() { echo "$1=$2" >> "${RUN_DIR}/markers.txt"; }

# Kill a process AND its whole tree. We launch the workload and the scraper
# via `setsid`, which puts each in its own process group whose ID == its PID,
# so signalling the negative pgid hits every child/grandchild too.
kill_tree() {
  local pid="$1" sig="${2:-TERM}"
  local pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
  if [[ -n "${pgid}" ]]; then
    kill -"${sig}" -- "-${pgid}" 2>/dev/null || true
  else
    kill -"${sig}" "$pid" 2>/dev/null || true
  fi
}

# Graceful tree stop with a long grace window: TERM, wait up to $2 seconds,
# then KILL. Used for the workload and the scraper.
stop_tree_gracefully() {
  local pid="$1" grace_s="$2" waited=0
  [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  kill_tree "${pid}" TERM
  while (( waited < grace_s )); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep "${POLL_STEP_S}"; (( waited += POLL_STEP_S ))
  done
  log "process ${pid} still alive after ${grace_s}s of TERM — SIGKILLing tree."
  kill_tree "${pid}" KILL
}

vllm_healthy() {
  curl -sf --max-time 5 "${VLLM_BASE_URL}/health"    >/dev/null 2>&1 && return 0
  curl -sf --max-time 5 "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1 && return 0
  return 1
}

# Is anything accepting TCP connections on host:port right now?
port_open() {
  local host="$1" port="$2"
  (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null || return 1
  exec 3>&- 3<&-
  return 0
}

# Wait up to $3 seconds for host:port to answer, polling gently.
wait_for_port() {  # host port timeout_s
  local host="$1" port="$2" timeout="$3" waited=0
  while (( waited < timeout )); do
    port_open "${host}" "${port}" && return 0
    sleep "${POLL_STEP_S}"; (( waited += POLL_STEP_S ))
    (( ABORT )) && return 1
  done
  return 1
}

check_disk_space() {
  local avail_kb avail_gb
  avail_kb=$(df -k --output=avail "${RUNS_DIR}" | tail -1 | tr -d ' ')
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  if (( avail_gb < MIN_FREE_GB )); then
    log "FATAL: only ${avail_gb}G free under ${RUNS_DIR} (need ${MIN_FREE_GB}G). Aborting."
    exit 1
  fi
}

#===============================================================================
# Signal handling / cleanup — always leaves the box clean
#===============================================================================
on_signal() {
  ABORT=1
  log "Caught signal — aborting sweep, cleaning up current run (gracefully)..."
  [[ -n "${WPID}" ]] && stop_tree_gracefully "${WPID}" "${WORKLOAD_KILL_GRACE_S}"
}

cleanup() {
  [[ -n "${SCRAPER_PID}" ]] && { stop_tree_gracefully "${SCRAPER_PID}" 10; SCRAPER_PID=""; }
  [[ -n "${WPID}" ]]        && { stop_tree_gracefully "${WPID}" "${WORKLOAD_KILL_GRACE_S}"; WPID=""; }
  amoprof_stop
  [[ -n "${SUDO_KEEPALIVE_PID}" ]] && kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true
  sleep 2   # let tee flush the last log lines
}
trap on_signal INT TERM
trap cleanup EXIT

#===============================================================================
# AMOProf start/stop
#
# IMPORTANT: `amoprof service` DAEMONIZES. The `sudo setsid amoprof ...`
# launcher we spawn exits within a second even on success, while the real
# collector process lives on, reparented. So:
#   - readiness  = "the metrics port answers", never "launcher PID alive"
#   - the PID we track = the daemon's real PID, discovered via pgrep
#===============================================================================
amoprof_start() {  # $1 = per-run output dir, $2 = per-run service log
  local out_dir="$1" svc_log="$2"
  local args=(
    service
    --metrics-port "${AMOPROF_PORT}"
    --metrics-host 0.0.0.0
    --vllm-port 8000
    --lmcache-port 6999
    --collectors gpu,dram,vmstat,iostat,smart
    --scrape-duration-s 10
    --interval-s 1
    --ssd-device /dev/nvme1n1
    --hicache-path "${LMCACHE_DISK_PATH}"
    --output-dir "${out_dir}"
    --enable-blktrace
    --enable-biosnoop
    --enable-dram --dram-tool auto
    --blkparse-interval-s 10
    --lmcache-bytes-per-token 18432
    --lmcache-max-disk-gb 20.0
  )

  # Refuse to start on top of a stale instance — it would hold the port and
  # silently mix old telemetry into this run.
  if port_open 127.0.0.1 "${AMOPROF_PORT}"; then
    log "ERROR: port ${AMOPROF_PORT} already in use — leftover amoprof?"
    log "       Run: sudo pkill -f 'amoprof.*service'   (then re-run the sweep)"
    return 1
  fi

  log "Starting amoprof -> ${out_dir} (log: ${svc_log})"
  ${SUDO} setsid "${AMOPROF_BIN}" "${args[@]}" >> "${svc_log}" 2>&1 &

  log "Waiting up to ${AMOPROF_READY_TIMEOUT_S}s for amoprof metrics port ${AMOPROF_PORT}..."
  if ! wait_for_port 127.0.0.1 "${AMOPROF_PORT}" "${AMOPROF_READY_TIMEOUT_S}"; then
    log "ERROR: amoprof never opened port ${AMOPROF_PORT}. Last log lines:"
    tail -n 20 "${svc_log}" >&2 || true
    return 1
  fi

  # The port answers, but give the daemon a beat to fully initialize its
  # collectors before we trust it — then capture its real PID.
  sleep 3
  AMOPROF_PID=$(pgrep -f "amoprof.*service" | head -n1 || true)
  if [[ -z "${AMOPROF_PID}" ]]; then
    log "ERROR: port is up but could not find the amoprof daemon PID."
    log "       Run 'pgrep -af amoprof' and tell me what it shows."
    return 1
  fi
  log "amoprof is up (daemon pid ${AMOPROF_PID}), metrics on :${AMOPROF_PORT}."

  # Let the first scrape cycle complete cleanly before the workload starts.
  log "Settling ${POST_START_SETTLE_S}s before starting the workload..."
  sleep "${POST_START_SETTLE_S}"
  return 0
}

amoprof_stop() {
  [[ -z "${AMOPROF_PID}" ]] && return 0
  if ! ${SUDO} kill -0 "${AMOPROF_PID}" 2>/dev/null; then
    AMOPROF_PID=""
    return 0
  fi
  log "Stopping amoprof (pid ${AMOPROF_PID}), grace ${AMOPROF_STOP_GRACE_S}s..."

  # The daemon was setsid'd: its PID is also its process-group ID, so a
  # group-kill takes any blktrace/biosnoop children down with it.
  ${SUDO} kill -TERM -- "-${AMOPROF_PID}" 2>/dev/null \
    || ${SUDO} kill -TERM "${AMOPROF_PID}" 2>/dev/null || true

  local waited=0
  while (( waited < AMOPROF_STOP_GRACE_S )); do
    ${SUDO} kill -0 "${AMOPROF_PID}" 2>/dev/null || break
    sleep "${POLL_STEP_S}"; (( waited += POLL_STEP_S ))
  done

  if ${SUDO} kill -0 "${AMOPROF_PID}" 2>/dev/null; then
    log "amoprof still alive after ${AMOPROF_STOP_GRACE_S}s — SIGKILLing the group."
    ${SUDO} kill -KILL -- "-${AMOPROF_PID}" 2>/dev/null \
      || ${SUDO} kill -KILL "${AMOPROF_PID}" 2>/dev/null || true
    sleep 3   # even SIGKILL cleanup (zombie reaping, fd flush) takes a moment
  fi

  AMOPROF_PID=""
  log "amoprof stopped."
}

#===============================================================================
# vLLM metrics scraper (optional) — per-run Prometheus snapshots
#===============================================================================
scraper_start() {  # $1 = output file
  local out="$1"
  setsid bash -c '
    while :; do
      {
        echo "# scrape_epoch=$(date +%s)"
        curl -sf --max-time 5 "'"${VLLM_BASE_URL}"'/metrics"
        echo
      } >> "'"$1"'" 2>/dev/null
      sleep '"${VLLM_SCRAPE_INTERVAL_S}"'
    done' &
  SCRAPER_PID=$!
}

#===============================================================================
# AgentBench workload, with hard timeout + graceful tree kill
#===============================================================================
run_workload() {  # $1 = rendered config path, $2 = workload log
  local cfg="$1" logf="$2"
  # setsid: workload gets its own process group (pgid == pid) so timeout or
  # Ctrl-C kills the whole tree — AgentBench spawns task workers as children.
  setsid bash -c '
    source "$1"
    cd "$2"
    export PYTHONUNBUFFERED=1
    exec python -m src.assigner --config "$3"
  ' _ "${AGENTBENCH_VENV}" "${AGENTBENCH_DIR}" "${cfg}" >> "${logf}" 2>&1 &
  WPID=$!
  log "AgentBench started (pid ${WPID}), log: ${logf}"

  local elapsed=0 status=0
  while kill -0 "${WPID}" 2>/dev/null; do
    if (( ABORT )); then
      stop_tree_gracefully "${WPID}" "${WORKLOAD_KILL_GRACE_S}"
      wait "${WPID}" 2>/dev/null; WPID=""
      return 130
    fi
    if (( RUN_TIMEOUT_S > 0 && elapsed >= RUN_TIMEOUT_S )); then
      log "TIMEOUT after ${elapsed}s — stopping workload tree (grace ${WORKLOAD_KILL_GRACE_S}s)."
      stop_tree_gracefully "${WPID}" "${WORKLOAD_KILL_GRACE_S}"
      wait "${WPID}" 2>/dev/null; WPID=""
      return 124
    fi
    sleep 5; (( elapsed += 5 ))
  done
  wait "${WPID}"; status=$?
  WPID=""
  return "${status}"
}

#===============================================================================
# vLLM availability (wait patiently, optionally self-restart)
#===============================================================================
ensure_vllm() {
  local waited=0
  while ! vllm_healthy; do
    if (( waited == 0 )); then
      log "vLLM not answering at ${VLLM_BASE_URL} — waiting up to ${VLLM_BOOT_WAIT_S}s..."
      if (( RESTART_VLLM_ON_FAILURE )) && [[ -n "${VLLM_START_CMD}" ]]; then
        log "Attempting vLLM restart (log: ${SWEEP_DIR}/vllm_server.log)"
        setsid bash -c "source \"${VLLM_VENV}\"; exec ${VLLM_START_CMD}" \
          >> "${SWEEP_DIR}/vllm_server.log" 2>&1 &
      fi
    fi
    (( waited >= VLLM_BOOT_WAIT_S )) && return 1
    sleep 10; (( waited += 10 ))
    (( ABORT )) && return 1
  done
  return 0
}

#===============================================================================
# Preflight
#===============================================================================
preflight() {
  mkdir -p "${SWEEP_DIR}"

  # One sweep at a time — a second instance would corrupt telemetry folders.
  exec 9>"${BASE_DIR}/.sweep.lock"
  flock -n 9 || { echo "Another sweep holds ${BASE_DIR}/.sweep.lock — exiting."; exit 1; }

  # All script output goes to both the terminal and sweep.log.
  exec > >(tee -a "${SWEEP_LOG}") 2>&1

  # An SSH drop at 2am kills a foreground job; tmux/screen survives it.
  if [[ -z "${TMUX:-}" && -z "${STY:-}" && -t 1 ]]; then
    log "WARNING: not inside tmux/screen. Strongly recommend: tmux new -s sweep"
  fi

  # sudo credentials are cached ~15 min by default — the keepalive refreshes
  # them so the 3am amoprof stop doesn't die waiting for a password prompt.
  if [[ "${SUDO}" == "sudo" ]]; then
    sudo -v || { log "FATAL: sudo needs a password; run this where you can type it once."; exit 1; }
    ( while :; do sudo -n true; sleep 60; done ) &
    SUDO_KEEPALIVE_PID=$!
  fi

  local missing=0
  for f in "${AGENTBENCH_VENV}" "${VLLM_VENV}"; do
    [[ -f "$f" ]] || { log "ERROR: venv activate not found: $f"; missing=1; }
  done
  [[ -d "${AGENTBENCH_DIR}/src" ]] || { log "ERROR: AgentBench repo not at ${AGENTBENCH_DIR}"; missing=1; }
  [[ -f "${AMOPROF_BIN}" ]] || { log "ERROR: amoprof not found at ${AMOPROF_BIN}"; missing=1; }
  for b in curl setsid flock pgrep ps df sed tail; do
    command -v "$b" >/dev/null || { log "ERROR: missing tool: $b"; missing=1; }
  done
  (( missing )) && exit 1

  # No stale amoprof allowed — it would hold the port and contaminate runs.
  if port_open 127.0.0.1 "${AMOPROF_PORT}"; then
    log "FATAL: port ${AMOPROF_PORT} already taken (stale amoprof?)."
    log "       Run: sudo pkill -f 'amoprof.*service'   (then re-run the sweep)"
    exit 1
  fi

  # Create the config template on first run so the script works out of the box.
  if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
    log "Creating config template at ${CONFIG_TEMPLATE}"
    cat > "${CONFIG_TEMPLATE}" <<'EOF'
import: definition.yaml

concurrency:
  task:
    dbbench-std: 1
    os-std: 1
  agent:
    qwen3-8b-awq: __AGENT_CONCURRENCY__   # <- the sweep knob

assignments:
  - agent: [qwen3-8b-awq]
    task: [dbbench-std, os-std]

output: "__OUTPUT_DIR__/{TIMESTAMP}"
EOF
  fi

  # vLLM must be up before we burn a run; wait through model load if needed.
  if ! ensure_vllm; then
    log "FATAL: vLLM is not reachable at ${VLLM_BASE_URL}. Start it first, then re-run."
    exit 1
  fi
  log "vLLM is healthy at ${VLLM_BASE_URL}."

  check_disk_space

  # Sweep-level provenance — handy when you write the AMOProf report.
  {
    echo "run_name=${RUN_NAME}"
    echo "concurrency_list=${CONCURRENCY_LIST[*]}"
    echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host=$(hostname)"
    echo "kernel=$(uname -r)"
    command -v nvidia-smi >/dev/null && nvidia-smi -L
    git -C "${AGENTBENCH_DIR}" rev-parse HEAD 2>/dev/null | sed 's/^/agentbench_commit=/'
    df -h "${RUNS_DIR}" "${LMCACHE_DISK_PATH}" 2>/dev/null
  } > "${SWEEP_DIR}/sweep_info.txt"

  [[ -f "${SUMMARY_CSV}" ]] || \
    echo "run_id,concurrency,start_utc,duration_s,status,run_dir" > "${SUMMARY_CSV}"
}

#===============================================================================
# One run = one concurrency level
#===============================================================================
run_one() {
  local c="$1"
  local cid; cid=$(printf 'c%03d' "${c}")
  RUN_DIR="${SWEEP_DIR}/${cid}"
  mkdir -p "${RUN_DIR}"

  # Resume support: a finished-OK run is never repeated.
  if (( RESUME )) && [[ -f "${RUN_DIR}/status" ]] && grep -q '^OK' "${RUN_DIR}/status"; then
    log "=== ${cid}: already OK, skipping (RESUME=1) ==="
    return 0
  fi

  log "=== ${cid}: concurrency=${c} ==="
  check_disk_space

  # Render config NEXT TO the template so `import: definition.yaml` keeps
  # resolving relatively; a copy goes into the run folder for the record.
  local rendered="${CONFIG_TEMPLATE%/*}/.sweep_rendered_${RUN_NAME}_${cid}.yaml"
  sed -e "s/__AGENT_CONCURRENCY__/${c}/g" \
      -e "s|__OUTPUT_DIR__|${RUN_DIR}/agentbench_outputs|g" \
      "${CONFIG_TEMPLATE}" > "${rendered}"
  cp "${rendered}" "${RUN_DIR}/config.yaml"

  # Per-run provenance.
  {
    echo "run_id=${cid}"
    echo "concurrency=${c}"
    echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
    free -g | sed 's/^/mem: /'
  } > "${RUN_DIR}/metadata.txt"

  : > "${RUN_DIR}/markers.txt"
  mark run_id "${cid}"
  mark concurrency "${c}"

  # vLLM died between runs? Wait once; if it's really gone, stop the sweep —
  # remaining levels would all fail anyway. Fix it and re-run same RUN_NAME.
  if ! ensure_vllm; then
    log "ERROR: vLLM is down and did not recover — stopping sweep at ${cid}."
    echo "VLLM_DOWN" > "${RUN_DIR}/status"
    return 1
  fi

  local t0 t1 status
  t0=$(date +%s)
  mark amoprof_start_epoch "${t0}"

  if ! amoprof_start "${RUN_DIR}/amoprof" "${RUN_DIR}/amoprof_service.log"; then
    echo "AMOPROF_START_FAIL" > "${RUN_DIR}/status"
    return 1   # next level might still work, so the sweep continues
  fi
  mark amoprof_ready_epoch "$(date +%s)"

  if (( SCRAPE_VLLM_METRICS )); then
    scraper_start "${RUN_DIR}/vllm_metrics.prom"
  fi

  mark agentbench_start_epoch "$(date +%s)"
  mark agentbench_start_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  run_workload "${rendered}" "${RUN_DIR}/agentbench.log"
  status=$?

  t1=$(date +%s)
  mark agentbench_end_epoch "${t1}"
  mark agentbench_end_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  mark agentbench_exit_status "${status}"

  [[ -n "${SCRAPER_PID}" ]] && { stop_tree_gracefully "${SCRAPER_PID}" 10; SCRAPER_PID=""; }

  amoprof_stop
  mark amoprof_stop_epoch "$(date +%s)"

  # Let amoprof's final scrape cycle flush to disk before we inspect it.
  log "Settling ${POST_STOP_SETTLE_S}s for final telemetry flush..."
  sleep "${POST_STOP_SETTLE_S}"

  # Map exit code -> readable status. 124 = our timeout, 130 = Ctrl-C.
  local status_str="OK"
  (( status == 0 ))   || status_str="FAIL_${status}"
  (( status == 124 )) && status_str="TIMEOUT"
  (( status == 130 )) && status_str="ABORTED"
  echo "${status_str}" > "${RUN_DIR}/status"

  # Sanity: did amoprof actually write anything?
  if [[ -z "$(ls -A "${RUN_DIR}/amoprof" 2>/dev/null)" ]]; then
    log "WARNING: ${RUN_DIR}/amoprof is empty — check amoprof_service.log."
  fi

  echo "${cid},${c},$(date -u -d "@${t0}" '+%Y-%m-%dT%H:%M:%SZ'),$(( t1 - t0 )),${status_str},${RUN_DIR}" \
    >> "${SUMMARY_CSV}"
  log "=== ${cid}: done in $(( t1 - t0 ))s, status=${status_str} ==="
  return 0
}

#===============================================================================
# Main
#===============================================================================
preflight

log "Sweep '${RUN_NAME}' starting. Levels: ${CONCURRENCY_LIST[*]}"
log "Results root: ${SWEEP_DIR}"

n="${#CONCURRENCY_LIST[@]}"
i=0
for C in "${CONCURRENCY_LIST[@]}"; do
  (( ABORT )) && break
  i=$(( i + 1 ))
  run_one "${C}" || true   # a failed run never kills the sweep
  (( ABORT )) && break
  if (( i < n )); then
    log "Settling for ${SLEEP_BETWEEN_RUNS_S}s before next level..."
    sleep "${SLEEP_BETWEEN_RUNS_S}"
  fi
done

log "Sweep finished. Summary:"
cat "${SUMMARY_CSV}"