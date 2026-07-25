#!/usr/bin/env bash
# setup_server.sh — g4dn (T4) one-shot server setup. Everything big lands on /opt/ls.
# Usage: bash setup_server.sh    (run as your normal user; it sudo's where needed)
set -euo pipefail

LS=/opt/ls
USER_NAME="${SUDO_USER:-$USER}"
log() { echo -e "\n\033[1;32m==> $*\033[0m"; }

[ -d "$LS" ] || { echo "ERROR: $LS is not mounted. Fix the mount first."; exit 1; }

# ---------------------------------------------------------------
log "1/6  Directory skeleton on \$LS + /etc/environment cache redirects"
# ---------------------------------------------------------------
sudo mkdir -p "$LS"/{cache/{huggingface,pip,uv,torch-extensions,triton,torchinductor,numba,flashinfer},cargo,go,npm,tmp,bin,docker,containerd,miniforge3}
sudo chown -R "$USER_NAME":"$USER_NAME" "$LS"
chmod 1777 "$LS/tmp"

add_env() { grep -q "^$1=" /etc/environment 2>/dev/null || echo "$1=$2" | sudo tee -a /etc/environment >/dev/null; }
add_env XDG_CACHE_HOME          "$LS/cache"
add_env HF_HOME                 "$LS/cache/huggingface"
add_env PIP_CACHE_DIR           "$LS/cache/pip"
add_env UV_CACHE_DIR            "$LS/cache/uv"
add_env TORCH_EXTENSIONS_DIR    "$LS/cache/torch-extensions"
add_env TRITON_CACHE_DIR        "$LS/cache/triton"
add_env TORCHINDUCTOR_CACHE_DIR "$LS/cache/torchinductor"
add_env NUMBA_CACHE_DIR         "$LS/cache/numba"
add_env FLASHINFER_WORKSPACE_DIR "$LS/cache/flashinfer"
add_env CARGO_HOME              "$LS/cargo"
add_env GOPATH                  "$LS/go"
add_env NPM_CONFIG_CACHE        "$LS/npm"
add_env TMPDIR                  "$LS/tmp"
# apply to this session too
set -a; . /etc/environment; set +a

# ---------------------------------------------------------------
log "2/6  Build tools & utilities (gcc, g++, make, cmake, git, ...)"
# ---------------------------------------------------------------
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential gcc g++ make cmake ninja-build pkg-config \
  git git-lfs curl wget ca-certificates gnupg lsb-release \
  unzip zip jq tmux htop nvtop \
  libssl-dev zlib1g-dev libffi-dev
git lfs install --system || true

# ---------------------------------------------------------------
log "3/6  NVIDIA driver check"
# ---------------------------------------------------------------
NEED_REBOOT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "Driver already present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) — leaving it alone."
  echo "(Do NOT run ubuntu-drivers autoinstall; it would replace a working driver.)"
else
  echo "No working driver found — installing via ubuntu-drivers."
  sudo apt install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  NEED_REBOOT=1
fi

# ---------------------------------------------------------------
log "4/6  Docker + NVIDIA container toolkit, data rooted on \$LS"
# ---------------------------------------------------------------
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

# docker images/containers/volumes -> $LS
[ -f /etc/docker/daemon.json ] && sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%s)
echo "{ \"data-root\": \"$LS/docker\" }" | sudo tee /etc/docker/daemon.json >/dev/null

# containerd root -> $LS (prepend top-level key before any [table])
sudo mkdir -p /etc/containerd
[ -f /etc/containerd/config.toml ] && sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak.$(date +%s)
{ echo "root = \"$LS/containerd\""; [ -f /etc/containerd/config.toml ] && cat /etc/containerd/config.toml || true; } | sudo tee /etc/containerd/config.toml >/dev/null

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart containerd docker
sudo systemctl enable containerd docker
sudo usermod -aG docker "$USER_NAME"

# ---------------------------------------------------------------
log "5/6  Miniforge (conda) + uv on \$LS"
# ---------------------------------------------------------------
if [ ! -x "$LS/miniforge3/bin/conda" ]; then
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o "$LS/tmp/miniforge.sh"
  bash "$LS/tmp/miniforge.sh" -b -p "$LS/miniforge3"
  rm -f "$LS/tmp/miniforge.sh"
fi
"$LS/miniforge3/bin/conda" init bash >/dev/null
# pkgs/envs already live under $LS/miniforge3 — no extra condarc needed.

if [ ! -x "$LS/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LS/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
grep -q "$LS/bin" ~/.bashrc || echo "export PATH=\"$LS/bin:\$PATH\"" >> ~/.bashrc

# ---------------------------------------------------------------
log "6/6  Verification"
# ---------------------------------------------------------------
gcc --version | head -1
docker --version
sudo docker run --rm hello-world >/dev/null && echo "docker: OK"
sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1 \
  && echo "docker GPU passthrough: OK" \
  || echo "docker GPU passthrough: CHECK (only matters if you run GPU containers)"
"$LS/miniforge3/bin/conda" --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cat <<EOF

============================================================
 DONE.
  - Log out and back in (or run: newgrp docker) so the docker
    group and /etc/environment vars take effect.
  - Conda: source ~/.bashrc, then 'conda activate'.
  - No system CUDA was installed — pip wheels carry their own.
    If a build ever demands nvcc:
      conda install -c nvidia cuda-nvcc
EOF
[ "$NEED_REBOOT" = "1" ] && echo "  - REBOOT REQUIRED (fresh driver install): sudo reboot"
echo "============================================================"