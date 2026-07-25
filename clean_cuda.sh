#!/bin/bash
# =============================================================================
# Undo apt-installed CUDA toolkit and clean up
# Prepares for reinstall to $LS
# =============================================================================

set -e

LS="/opt/large-storage"

echo "=========================================="
echo "Removing apt-installed nvidia-cuda-toolkit"
echo "=========================================="

# =============================================================================
# STEP 1: Remove the apt package
# =============================================================================
echo ""
echo "[1/5] Removing nvidia-cuda-toolkit..."
sudo apt remove --purge -y nvidia-cuda-toolkit

echo "✓ Package removed"

# =============================================================================
# STEP 2: Remove leftover CUDA files from system paths
# =============================================================================
echo ""
echo "[2/5] Cleaning up leftover CUDA files..."

# Common apt CUDA installation paths
SYSTEM_CUDA_PATHS=(
    "/usr/lib/cuda"
    "/usr/local/cuda"
    "/usr/local/cuda-11"
    "/usr/local/cuda-12"
    "/usr/local/cuda-11.*"
    "/usr/local/cuda-12.*"
    "/usr/share/cuda"
    "/usr/include/cuda"
    "/usr/bin/nvcc"
    "/usr/lib/x86_64-linux-gnu/libcuda*"
    "/usr/lib/x86_64-linux-gnu/libcudart*"
    "/usr/lib/x86_64-linux-gnu/libcublas*"
    "/usr/lib/x86_64-linux-gnu/libcufft*"
    "/usr/lib/x86_64-linux-gnu/libcurand*"
    "/usr/lib/x86_64-linux-gnu/libcusolver*"
    "/usr/lib/x86_64-linux-gnu/libcusparse*"
    "/usr/lib/x86_64-linux-gnu/libnpp*"
)

for path in "${SYSTEM_CUDA_PATHS[@]}"; do
    if ls $path 1> /dev/null 2>&1; then
        echo "  Removing: $path"
        sudo rm -rf $path 2>/dev/null || true
    fi
done

echo "✓ System CUDA paths cleaned"

# =============================================================================
# STEP 3: Clean apt cache and orphaned packages
# =============================================================================
echo ""
echo "[3/5] Cleaning apt cache and orphaned packages..."
sudo apt autoremove -y
sudo apt autoclean
sudo apt clean

echo "✓ Apt cleaned"

# =============================================================================
# STEP 4: Remove old environment entries from .bashrc
# =============================================================================
echo ""
echo "[4/5] Cleaning old CUDA entries from ~/.bashrc..."

# Remove old CUDA-related lines (the ones that point to /usr/local/cuda)
# Be careful not to remove the $LS-based ones if they exist

# Remove lines containing /usr/local/cuda but NOT $LS/cuda
if [ -f "$HOME/.bashrc" ]; then
    # Create backup
    cp "$HOME/.bashrc" "$HOME/.bashrc.backup.$(date +%s)"

    # Remove lines that reference system CUDA paths
    sed -i '/\/usr\/local\/cuda/d' "$HOME/.bashrc" 2>/dev/null || true
    sed -i '/CUDA_HOME=\/usr/d' "$HOME/.bashrc" 2>/dev/null || true

    echo "✓ Old CUDA entries removed from ~/.bashrc"
    echo "  (backup saved as ~/.bashrc.backup.*)"
fi

# =============================================================================
# STEP 5: Verify cleanup
# =============================================================================
echo ""
echo "[5/5] Verifying cleanup..."

echo ""
echo "--- Checking for remaining nvcc ---"
which nvcc 2>/dev/null && nvcc --version || echo "  ✓ nvcc not found in PATH"

echo ""
echo "--- Checking /usr/local/cuda ---"
ls -la /usr/local/cuda 2>/dev/null || echo "  ✓ /usr/local/cuda does not exist"

echo ""
echo "--- Checking /usr/lib/cuda ---"
ls -la /usr/lib/cuda 2>/dev/null || echo "  ✓ /usr/lib/cuda does not exist"

echo ""
echo "--- Remaining CUDA packages ---"
dpkg -l | grep -i cuda | grep -v "nvidia-cuda-toolkit" || echo "  ✓ No CUDA packages remaining"

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the setup script to install CUDA to $LS:"
echo "     ./setup_t4_server_ls.sh"
echo ""
echo "  2. Or manually install CUDA to $LS:"
echo "     wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
echo "     sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --toolkitpath=$LS/cuda --no-drm --no-man-page --no-opengl-libs --override"
echo ""
echo "  3. Then source ~/.bashrc and verify:"
echo "     source ~/.bashrc"
echo "     nvcc --version"
echo ""