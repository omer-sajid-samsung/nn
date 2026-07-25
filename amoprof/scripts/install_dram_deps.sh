#!/usr/bin/env bash
# =============================================================================
# AMOprof DRAM PMU dependency installer
#
# Installs/checks dependencies needed by:
#   amoprof collect --enable-dram --dram-tool auto
#
# Backend mapping:
#   AMD   -> AMDuProfPcm collector
#   Intel -> Intel PCM pcm-memory collector
#   Any   -> perf uncore/IMC fallback where supported
#
# The script is intentionally conservative:
#   * It detects existing tools first.
#   * It uses the native OS package manager where possible.
#   * It does not silently download proprietary AMD uProf unless you provide
#     a local installer path or URL.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${BOLD}── $* ──${NC}"; }

DRAM_TOOL="${AMOPROF_DRAM_TOOL:-auto}"       # auto | all | amduprof | intel-pcm | perf-imc | none
DRY_RUN=0
ASSUME_YES=1
NO_SUDO=0
BUILD_INTEL_PCM=0
AMDUPROF_INSTALLER="${AMOPROF_AMDUPROF_INSTALLER:-}"
AMDUPROF_URL="${AMOPROF_AMDUPROF_URL:-}"
PREFIX="${AMOPROF_PREFIX:-/usr/local}"

usage() {
cat <<'EOF'
Usage:
  scripts/install_dram_deps.sh [OPTIONS]

Options:
  --dram-tool TOOL          auto | all | amduprof | intel-pcm | perf-imc | none
                            default: auto, based on CPU vendor
  --build-intel-pcm         If distro package is missing, build Intel PCM from GitHub
  --amduprof-installer PATH Install AMD uProf from a local .deb/.rpm/.tar.* installer
  --amduprof-url URL        Download AMD uProf installer URL, then install
  --prefix DIR              Prefix for source-built Intel PCM, default /usr/local
  --dry-run                 Print actions without installing
  --no-sudo                 Do not use sudo; assume current user is root
  --no                      Do not pass -y/--noconfirm to package manager
  --help                    Show this help

Examples:
  # Auto-detect CPU and install only needed DRAM backend dependencies
  sudo scripts/install_dram_deps.sh --dram-tool auto

  # Install both AMD and Intel DRAM backends where possible
  sudo scripts/install_dram_deps.sh --dram-tool all

  # Intel host: install Intel PCM package; if unavailable, build from source
  sudo scripts/install_dram_deps.sh --dram-tool intel-pcm --build-intel-pcm

  # AMD host: install prerequisites and install AMD uProf from local installer
  sudo scripts/install_dram_deps.sh --dram-tool amduprof \
       --amduprof-installer ./AMDuProf_Linux_x64.deb

Notes:
  * --dram-tool auto is architecture/vendor aware.
  * AMD x86_64 hosts select AMD uProf prerequisites by default.
  * Intel x86_64 hosts select Intel PCM by default.
  * Non-x86 or unknown platforms use perf IMC/core PMU dependencies where available.
  * Intel PCM package name on Debian/Ubuntu is usually "pcm"; it provides
    pcm-memory.
  * AMD uProf/AMDuProfPcm is not silently downloaded unless a local installer
    or explicit URL is provided.
EOF
exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dram-tool) DRAM_TOOL="$2"; shift 2 ;;
        --build-intel-pcm) BUILD_INTEL_PCM=1; shift ;;
        --amduprof-installer) AMDUPROF_INSTALLER="$2"; shift 2 ;;
        --amduprof-url) AMDUPROF_URL="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-sudo) NO_SUDO=1; shift ;;
        --no) ASSUME_YES=0; shift ;;
        --help|-h) usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

SUDO=""
if [[ "$NO_SUDO" -eq 0 && "${EUID:-$(id -u)}" -ne 0 ]]; then
    SUDO="sudo"
fi

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ $*"
    else
        "$@"
    fi
}

sh_run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ $*"
    else
        bash -lc "$*"
    fi
}

detect_cpu_vendor() {
    local cpu
    cpu="$(tr '[:upper:]' '[:lower:]' < /proc/cpuinfo 2>/dev/null || true)"
    if grep -qi "authenticamd" <<<"$cpu"; then echo "amd"; return; fi
    if grep -qi "genuineintel" <<<"$cpu"; then echo "intel"; return; fi
    echo "unknown"
}

detect_arch() {
    local a
    a="$(uname -m 2>/dev/null || echo unknown)"
    case "$a" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        ppc64le) echo "ppc64le" ;;
        *) echo "$a" ;;
    esac
}

load_os_release() {
    OS_ID="unknown"; OS_LIKE=""; OS_VERSION_ID=""
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_LIKE="${ID_LIKE:-}"
        OS_VERSION_ID="${VERSION_ID:-}"
    fi
}

detect_pm() {
    if command -v apt-get >/dev/null 2>&1; then echo "apt"; return; fi
    if command -v dnf >/dev/null 2>&1; then echo "dnf"; return; fi
    if command -v yum >/dev/null 2>&1; then echo "yum"; return; fi
    if command -v zypper >/dev/null 2>&1; then echo "zypper"; return; fi
    if command -v pacman >/dev/null 2>&1; then echo "pacman"; return; fi
    echo "unknown"
}

pm_install() {
    local packages=("$@")
    [[ "${#packages[@]}" -eq 0 ]] && return 0

    case "$PM" in
        apt)
            run $SUDO apt-get update -qq
            if [[ "$ASSUME_YES" -eq 1 ]]; then
                run $SUDO apt-get install -y --no-install-recommends "${packages[@]}"
            else
                run $SUDO apt-get install --no-install-recommends "${packages[@]}"
            fi
            ;;
        dnf)
            if [[ "$ASSUME_YES" -eq 1 ]]; then
                run $SUDO dnf install -y "${packages[@]}"
            else
                run $SUDO dnf install "${packages[@]}"
            fi
            ;;
        yum)
            if [[ "$ASSUME_YES" -eq 1 ]]; then
                run $SUDO yum install -y "${packages[@]}"
            else
                run $SUDO yum install "${packages[@]}"
            fi
            ;;
        zypper)
            if [[ "$ASSUME_YES" -eq 1 ]]; then
                run $SUDO zypper --non-interactive install "${packages[@]}"
            else
                run $SUDO zypper install "${packages[@]}"
            fi
            ;;
        pacman)
            if [[ "$ASSUME_YES" -eq 1 ]]; then
                run $SUDO pacman -Sy --needed --noconfirm "${packages[@]}"
            else
                run $SUDO pacman -Sy --needed "${packages[@]}"
            fi
            ;;
        *)
            warn "No supported package manager detected; install manually: ${packages[*]}"
            return 1
            ;;
    esac
}

pkg_present_or_cmd_present() {
    local cmd="$1"
    shift || true
    if command -v "$cmd" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

apt_install_best_effort() {
    # Install one package at a time so an unavailable optional package does not
    # abort the whole dependency setup.
    local p
    for p in "$@"; do
        if dpkg -s "$p" >/dev/null 2>&1; then
            ok "$p already installed"
        else
            info "Installing $p"
            if ! pm_install "$p"; then
                warn "Could not install $p from package manager; continuing"
            fi
        fi
    done
}

install_core_kernel_perf_deps() {
    section "Core DRAM PMU / kernel-counter dependencies"
    case "$PM" in
        apt)
            # linux-tools-$(uname -r) is best on Ubuntu, linux-perf on Debian,
            # linux-tools-generic is useful fallback on Ubuntu.
            local core=(msr-tools numactl sysstat pciutils procps kmod)
            apt_install_best_effort "${core[@]}"

            if ! command -v perf >/dev/null 2>&1; then
                apt_install_best_effort "linux-tools-$(uname -r)" linux-tools-generic linux-perf
            else
                ok "perf already available: $(command -v perf)"
            fi
            ;;
        dnf|yum)
            # perf package names vary across RHEL/Fedora clones; try common ones.
            pm_install msr-tools numactl sysstat pciutils procps-ng kmod || true
            if ! command -v perf >/dev/null 2>&1; then
                pm_install perf || pm_install linux-tools || warn "Could not install perf; install kernel-tools/perf manually"
            else
                ok "perf already available: $(command -v perf)"
            fi
            ;;
        zypper)
            pm_install msr-tools numactl sysstat pciutils procps kmod || true
            if ! command -v perf >/dev/null 2>&1; then
                pm_install perf || warn "Could not install perf; install manually"
            fi
            ;;
        pacman)
            pm_install msr-tools numactl sysstat pciutils procps-ng kmod linux-tools || true
            ;;
        *)
            warn "Install manually: msr-tools numactl sysstat pciutils procps kmod perf"
            ;;
    esac

    if command -v modprobe >/dev/null 2>&1; then
        if lsmod 2>/dev/null | grep -q '^msr'; then
            ok "msr kernel module already loaded"
        else
            info "Loading msr kernel module"
            run $SUDO modprobe msr || warn "Could not load msr module; pcm/perf may need root or kernel support"
        fi
    fi
}

install_intel_pcm_pkg() {
    section "Intel PCM / pcm-memory"

    if [[ "${ARCH:-$(detect_arch)}" != "x86_64" ]]; then
        warn "Intel PCM is only installed automatically on x86_64. Current architecture: ${ARCH:-$(detect_arch)}"
        warn "Using perf-imc/core PMU fallback where supported."
        return 0
    fi

    if command -v pcm-memory >/dev/null 2>&1; then
        ok "pcm-memory already available: $(command -v pcm-memory)"
        return 0
    fi

    case "$PM" in
        apt)
            # Debian/Ubuntu package is "pcm"; it provides pcm-memory.
            if pm_install pcm; then
                if command -v pcm-memory >/dev/null 2>&1; then
                    ok "pcm-memory installed: $(command -v pcm-memory)"
                    return 0
                fi
            fi
            warn "Package 'pcm' did not provide pcm-memory on this distro/release"
            ;;
        dnf|yum)
            # EPEL/Fedora may provide pcm on some releases. Best effort.
            if pm_install pcm; then
                if command -v pcm-memory >/dev/null 2>&1; then
                    ok "pcm-memory installed: $(command -v pcm-memory)"
                    return 0
                fi
            fi
            warn "Could not install pcm-memory from native repos"
            ;;
        zypper|pacman)
            if pm_install pcm; then
                if command -v pcm-memory >/dev/null 2>&1; then
                    ok "pcm-memory installed: $(command -v pcm-memory)"
                    return 0
                fi
            fi
            warn "Could not install pcm-memory from native repos"
            ;;
        *)
            warn "No package-manager path for Intel PCM"
            ;;
    esac

    if [[ "$BUILD_INTEL_PCM" -eq 1 ]]; then
        build_intel_pcm_from_source
    else
        warn "Intel PCM missing. Re-run with --build-intel-pcm to build from source, or install manually:"
        warn "  git clone https://github.com/intel/pcm && cd pcm && mkdir build && cd build && cmake .. && make -j && sudo make install"
    fi
}

install_build_deps() {
    section "Build dependencies"
    case "$PM" in
        apt)
            pm_install git ca-certificates cmake make g++ pkg-config libpci-dev || true
            ;;
        dnf|yum)
            pm_install git ca-certificates cmake make gcc-c++ pkgconf-pkg-config pciutils-devel || true
            ;;
        zypper)
            pm_install git ca-certificates cmake make gcc-c++ pkg-config pciutils-devel || true
            ;;
        pacman)
            pm_install git ca-certificates cmake make gcc pkgconf pciutils || true
            ;;
        *)
            warn "Install manually: git cmake make C++ compiler pkg-config PCI headers"
            ;;
    esac
}

build_intel_pcm_from_source() {
    install_build_deps
    section "Building Intel PCM from source"

    local build_root="${TMPDIR:-/tmp}/amoprof_intel_pcm_build"
    run rm -rf "$build_root"
    run mkdir -p "$build_root"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ git clone --depth=1 https://github.com/intel/pcm $build_root/pcm"
        echo "+ cmake -S $build_root/pcm -B $build_root/pcm/build -DCMAKE_INSTALL_PREFIX=$PREFIX"
        echo "+ cmake --build $build_root/pcm/build -j$(nproc)"
        echo "+ $SUDO cmake --install $build_root/pcm/build"
        return 0
    fi

    git clone --depth=1 https://github.com/intel/pcm "$build_root/pcm"
    cmake -S "$build_root/pcm" -B "$build_root/pcm/build" -DCMAKE_INSTALL_PREFIX="$PREFIX"
    cmake --build "$build_root/pcm/build" -j"$(nproc)"
    $SUDO cmake --install "$build_root/pcm/build"

    if command -v pcm-memory >/dev/null 2>&1; then
        ok "pcm-memory installed: $(command -v pcm-memory)"
    elif [[ -x "$PREFIX/bin/pcm-memory" ]]; then
        ok "pcm-memory installed: $PREFIX/bin/pcm-memory"
    else
        warn "Intel PCM build completed, but pcm-memory was not found on PATH"
    fi
}

install_amduprof() {
    section "AMD uProf / AMDuProfPcm"

    if [[ "${ARCH:-$(detect_arch)}" != "x86_64" ]]; then
        warn "AMD uProf / AMDuProfPcm automatic setup is supported only on x86_64. Current architecture: ${ARCH:-$(detect_arch)}"
        warn "Using perf-imc/core PMU fallback where supported."
        return 0
    fi

    if command -v AMDuProfPcm >/dev/null 2>&1; then
        ok "AMDuProfPcm already available: $(command -v AMDuProfPcm)"
        return 0
    fi

    for p in /opt/AMDuProf_*/bin/AMDuProfPcm /opt/AMDuProf*/bin/AMDuProfPcm; do
        if [[ -x "$p" ]]; then
            ok "AMDuProfPcm already available: $p"
            warn "Set AMOPROF_AMDUPROF_PCM_BIN=$p or pass --amduprof-pcm-bin $p if it is not on PATH."
            return 0
        fi
    done

    # Install common compatibility/runtime deps only; AMD uProf installer itself
    # must come from local path/URL due packaging/license/version differences.
    case "$PM" in
        apt)
            apt_install_best_effort libnuma1 libstdc++6 libgcc-s1 pciutils tar gzip xz-utils
            ;;
        dnf|yum)
            pm_install numactl-libs libstdc++ libgcc pciutils tar gzip xz || true
            ;;
        zypper)
            pm_install libnuma1 libstdc++6 libgcc_s1 pciutils tar gzip xz || true
            ;;
        pacman)
            pm_install numactl gcc-libs pciutils tar gzip xz || true
            ;;
        *)
            warn "Install AMD uProf runtime deps manually: libnuma, libstdc++, libgcc, pciutils"
            ;;
    esac

    local installer="$AMDUPROF_INSTALLER"
    if [[ -z "$installer" && -n "$AMDUPROF_URL" ]]; then
        section "Downloading AMD uProf installer"
        local fname="/tmp/$(basename "$AMDUPROF_URL")"
        if command -v curl >/dev/null 2>&1; then
            run curl -L "$AMDUPROF_URL" -o "$fname"
        elif command -v wget >/dev/null 2>&1; then
            run wget -O "$fname" "$AMDUPROF_URL"
        else
            error "curl/wget not found; cannot download AMD uProf URL"
            return 1
        fi
        installer="$fname"
    fi

    if [[ -z "$installer" ]]; then
        warn "AMDuProfPcm not found. AMD uProf is not installed automatically without an installer path/URL."
        warn "Download AMD uProf from AMD, then run one of:"
        warn "  sudo scripts/install_dram_deps.sh --dram-tool amduprof --amduprof-installer ./AMDuProf_Linux_x64.deb"
        warn "  sudo scripts/install_dram_deps.sh --dram-tool amduprof --amduprof-installer ./AMDuProf_Linux_x64.rpm"
        warn "  sudo scripts/install_dram_deps.sh --dram-tool amduprof --amduprof-installer ./AMDuProf_Linux_x64.tar.bz2"
        return 0
    fi

    if [[ ! -f "$installer" ]]; then
        error "AMD uProf installer not found: $installer"
        return 1
    fi

    case "$installer" in
        *.deb)
            if [[ "$PM" == "apt" ]]; then
                run $SUDO apt-get install -y "$installer"
            else
                run $SUDO dpkg -i "$installer"
            fi
            ;;
        *.rpm)
            if [[ "$PM" == "dnf" ]]; then
                run $SUDO dnf install -y "$installer"
            elif [[ "$PM" == "yum" ]]; then
                run $SUDO yum install -y "$installer"
            else
                run $SUDO rpm -Uvh "$installer"
            fi
            ;;
        *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tbz2|*.tar.xz|*.txz)
            local dest="/opt"
            run $SUDO mkdir -p "$dest"
            run $SUDO tar -xf "$installer" -C "$dest"
            ;;
        *)
            error "Unsupported AMD uProf installer type: $installer"
            return 1
            ;;
    esac

    if command -v AMDuProfPcm >/dev/null 2>&1; then
        ok "AMDuProfPcm installed: $(command -v AMDuProfPcm)"
    else
        for p in /opt/AMDuProf_*/bin/AMDuProfPcm /opt/AMDuProf*/bin/AMDuProfPcm; do
            if [[ -x "$p" ]]; then
                ok "AMDuProfPcm installed: $p"
                warn "Add to PATH or pass --amduprof-pcm-bin $p"
                return 0
            fi
        done
        warn "AMD uProf install finished but AMDuProfPcm was not found. Check installer layout."
    fi
}

verify_backend() {
    section "Verification"

    local vendor
    vendor="$(detect_cpu_vendor)"

    echo "CPU vendor     : $vendor"
    echo "Architecture   : ${ARCH:-$(detect_arch)}"
    echo "OS             : ${OS_ID:-unknown} ${OS_VERSION_ID:-} (like: ${OS_LIKE:-})"
    echo "Package manager: $PM"
    echo "Requested tool : $DRAM_TOOL"

    if command -v perf >/dev/null 2>&1; then
        ok "perf: $(command -v perf)"
    else
        warn "perf not found; --dram-tool perf-imc fallback may be unavailable"
    fi

    if command -v pcm-memory >/dev/null 2>&1; then
        ok "Intel pcm-memory: $(command -v pcm-memory)"
    else
        warn "Intel pcm-memory missing"
    fi

    if command -v AMDuProfPcm >/dev/null 2>&1; then
        ok "AMDuProfPcm: $(command -v AMDuProfPcm)"
    else
        local found=""
        for p in /opt/AMDuProf_*/bin/AMDuProfPcm /opt/AMDuProf*/bin/AMDuProfPcm; do
            if [[ -x "$p" ]]; then found="$p"; break; fi
        done
        if [[ -n "$found" ]]; then
            ok "AMDuProfPcm: $found"
        else
            warn "AMDuProfPcm missing"
        fi
    fi

    cat <<EOF

Suggested AMOprof command:
  amoprof collect --enable-dram --dram-tool auto ...

Explicit AMD:
  amoprof collect --enable-dram --dram-tool amduprof --amduprof-pcm-bin /opt/AMDuProf_*/bin/AMDuProfPcm ...

Explicit Intel:
  amoprof collect --enable-dram --dram-tool intel-pcm --intel-pcm-memory-bin pcm-memory ...

Perf fallback:
  amoprof collect --enable-dram --dram-tool perf-imc ...
EOF
}

main() {
    load_os_release
    PM="$(detect_pm)"
    CPU_VENDOR="$(detect_cpu_vendor)"
    ARCH="$(detect_arch)"

    section "Detected platform"
    echo "CPU vendor     : $CPU_VENDOR"
    echo "Architecture   : $ARCH"
    echo "OS             : $OS_ID $OS_VERSION_ID (like: $OS_LIKE)"
    echo "Package manager: $PM"

    case "$DRAM_TOOL" in
        auto)
            if [[ "$ARCH" == "x86_64" && "$CPU_VENDOR" == "amd" ]]; then
                SELECTED=("amduprof")
            elif [[ "$ARCH" == "x86_64" && "$CPU_VENDOR" == "intel" ]]; then
                SELECTED=("intel-pcm")
            else
                SELECTED=("perf-imc")
            fi
            ;;
        all)
            SELECTED=("amduprof" "intel-pcm" "perf-imc")
            ;;
        amduprof|intel-pcm|perf-imc|none)
            SELECTED=("$DRAM_TOOL")
            ;;
        *)
            error "Invalid --dram-tool: $DRAM_TOOL"
            exit 2
            ;;
    esac

    if [[ "${SELECTED[*]}" == "none" ]]; then
        warn "--dram-tool none selected; nothing to install"
        exit 0
    fi

    echo "Selected backend(s): ${SELECTED[*]}"

    install_core_kernel_perf_deps

    local tool
    for tool in "${SELECTED[@]}"; do
        case "$tool" in
            amduprof) install_amduprof ;;
            intel-pcm) install_intel_pcm_pkg ;;
            perf-imc) ok "perf-imc uses core perf/MSR dependencies installed above" ;;
        esac
    done

    verify_backend
}

main "$@"
