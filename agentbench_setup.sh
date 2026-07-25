#!/usr/bin/env bash
# agentbench_setup.sh — AgentBench on the $LS box.
# Flow: clone -> docker pulls/builds/compose on MAIN -> checkout v0.2 ->
#       conda env + requirements + configs -> print run instructions.
set -euo pipefail

LS=/opt/ls
AB_DIR="$LS/AgentBench"
CONDA="$LS/miniforge3/bin/conda"
ENV_NAME=agent-bench

# --- must match whatever vLLM is serving on this box ---
AGENT_NAME="qwen3-8b-awq"
MODEL_ID="Qwen/Qwen3-8B-AWQ"     # body.model in the API call — must equal the served model id
VLLM_URL="http://localhost:8000/v1/chat/completions"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }

# docker without sudo if the group is active, else fall back
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# ---------------------------------------------------------------
log "1/5  Clone AgentBench (main branch)"
# ---------------------------------------------------------------
cd "$LS"
# [ -d "$AB_DIR/.git" ] || git clone https://github.com/THUDM/AgentBench.git
cd "$AB_DIR"
git checkout main
git pull --ff-only || true

# ---------------------------------------------------------------
log "2/5  Docker: pulls, local-os image builds, compose up  (ALL on main)"
# ---------------------------------------------------------------
$DOCKER pull mysql
$DOCKER pull ubuntu

$DOCKER build -f data/os_interaction/res/dockerfiles/default  data/os_interaction/res/dockerfiles --tag local-os/default
$DOCKER build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
$DOCKER build -f data/os_interaction/res/dockerfiles/ubuntu   data/os_interaction/res/dockerfiles --tag local-os/ubuntu

# bring the stack up detached, on main — this is the step that breaks on v0.2
$DOCKER compose -f extra/docker-compose.yml config --services
$DOCKER compose -f extra/docker-compose.yml up -d --build \
  controller redis dbbench-std os_interaction-std

echo "Waiting for the compose stack to settle..."
sleep 20
$DOCKER compose -f extra/docker-compose.yml ps

# ---------------------------------------------------------------
log "3/5  Switch to v0.2 (configs/deps below are all v0.2-side)"
# ---------------------------------------------------------------
git checkout v0.2
git pull --ff-only || true

# ---------------------------------------------------------------
log "4/5  Conda env ($ENV_NAME, python 3.9) + v0.2 requirements"
# ---------------------------------------------------------------
"$CONDA" create -n "$ENV_NAME" python=3.9 -y
"$CONDA" run -n "$ENV_NAME" pip install --upgrade pip
"$CONDA" run -n "$ENV_NAME" pip install -r requirements.txt

# ---------------------------------------------------------------
log "5/5  Agent + assignment configs"
# ---------------------------------------------------------------
mkdir -p configs/agents configs/assignments

cat >> configs/agents/api_agents.yaml <<EOF
${AGENT_NAME}:
    import: "./openai-chat.yaml"
    parameters:
        name: "${AGENT_NAME}"
        url: ${VLLM_URL}
        headers:
            Content-Type: application/json
            Authorization: Bearer EMPTY
        body:
            model: "${MODEL_ID}"
            max_tokens: 512
            temperature: 0
EOF

cat > configs/assignments/vllm_lite.yaml <<EOF
import: definition.yaml

concurrency:
  task:
    dbbench-std: 1
    os-std: 1
  agent:
    ${AGENT_NAME}: 1      # <- your "N agent sessions" knob

assignments:
  - agent: [${AGENT_NAME}]
    task: [dbbench-std, os-std]

output: "outputs/{TIMESTAMP}"
EOF

cat <<EOF

============================================================
 DONE. To run:

 1. Make sure your vLLM server is up on :8000 serving ${MODEL_ID}
    (tmux new -s vllm -> bash /opt/ls/vllm/serve.sh, or your 32B setup)

 2. tmux new -s agentbench
    source $LS/miniforge3/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    cd $AB_DIR
    python -m src.start_task -a --config configs/start_task_lite.yaml

 3. Wait ~1 min until you see "200 OK", then detach: Ctrl-b d

 Notes:
 - Compose stack is already running detached from step 2.
   Check anytime: $DOCKER compose -f $AB_DIR/extra/docker-compose.yml ps
 - '$DOCKER ps' permission denied? Run: newgrp docker  (or re-login)
 - Model mismatch = instant 404s: the body.model above must equal
   the id your vLLM server reports at :8000/v1/models.
============================================================
EOF