#!/usr/bin/env bash

set -Eeuo pipefail

readonly REQUIRED_OS_ID="ubuntu"
readonly REQUIRED_OS_VERSION="26.04"
readonly REQUIRED_ARCH="amd64"
readonly CUDA_VERSION="13.2"
readonly CUDA_VERSION_FILE="$(sed 's/\./-/g' <<< $CUDA_VERSION)"
readonly CUDA_INSTALLER_VERSION="13.2.0"
readonly CUDA_DRIVER_BUILD="595.45.04"
readonly MINIMUM_DRIVER_VERSION="595.45.04"
readonly INSTALLER_NAME="cuda_${CUDA_INSTALLER_VERSION}_${CUDA_DRIVER_BUILD}_linux.run"
readonly INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_INSTALLER_VERSION}/local_installers/${INSTALLER_NAME}"
readonly INSTALLER_MD5="656f4a652313abd118fb0ae1a8b902d3"
readonly CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}"

log() {
    printf '[cuda-installer] %s\n' "$*"
}

die() {
    printf '[cuda-installer] ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ ${EUID} -eq 0 ]]; then
    SUDO=()
else
    command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root."
    SUDO=(sudo)
fi

[[ -r /etc/os-release ]] || die "Cannot read /etc/os-release."
# shellcheck disable=SC1091
source /etc/os-release

[[ ${ID:-} == "${REQUIRED_OS_ID}" ]] || die "This script requires Ubuntu. Detected: ${ID:-unknown}."
[[ ${VERSION_ID:-} == "${REQUIRED_OS_VERSION}" ]] || die "This script requires Ubuntu ${REQUIRED_OS_VERSION}. Detected: ${VERSION_ID:-unknown}."
[[ $(dpkg --print-architecture) == "${REQUIRED_ARCH}" ]] || die "This installer supports x86_64/amd64 only."

log "Installing prerequisites."
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    gcc-15 \
    g++-15 \
    wget

command -v nvidia-smi >/dev/null 2>&1 || die \
    "Install an Ubuntu-compatible NVIDIA driver first; CUDA ${CUDA_VERSION} requires driver ${MINIMUM_DRIVER_VERSION} or newer."

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')
[[ -n ${driver_version} ]] || die "Could not determine the NVIDIA driver version."
dpkg --compare-versions "${driver_version}" ge "${MINIMUM_DRIVER_VERSION}" || die \
    "NVIDIA driver ${driver_version} is too old; CUDA ${CUDA_VERSION} requires ${MINIMUM_DRIVER_VERSION} or newer."

if [[ -x "${CUDA_ROOT}/bin/nvcc" ]] && "${CUDA_ROOT}/bin/nvcc" --version | grep -q 'release 13\.0'; then
    log "CUDA Toolkit ${CUDA_VERSION} is already installed; skipping the installer."
else
    if [[ -e ${CUDA_ROOT} ]]; then
        die "${CUDA_ROOT} exists but is not a valid CUDA ${CUDA_VERSION} installation. Remove or repair it before retrying."
    fi

    if dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' 'cuda-toolkit-*' 2>/dev/null | grep -q '^ii '; then
        die "An APT-managed CUDA Toolkit is installed. Do not mix APT and runfile Toolkit installations."
    fi

    work_dir=$(mktemp -d -t cuda-${CUDA_VERSION}-installer.XXXXXXXX)
    trap 'rm -rf -- "${work_dir}"' EXIT
    installer_path="${work_dir}/${INSTALLER_NAME}"

    log "Downloading CUDA Toolkit ${CUDA_VERSION}."
    wget --https-only --secure-protocol=TLSv1_2 --show-progress \
        --output-document="${installer_path}" \
        "${INSTALLER_URL}"

    log "Verifying NVIDIA's published checksum."
    printf '%s  %s\n' "${INSTALLER_MD5}" "${installer_path}" | md5sum --check --status || \
        die "Installer checksum validation failed."

    log "Installing the CUDA Toolkit without replacing the NVIDIA driver."
    "${SUDO[@]}" sh "${installer_path}" \
        --silent \
        --toolkit \
        --toolkitpath="${CUDA_ROOT}" \
        --override
fi

if [[ -e /usr/local/cuda && ! -L /usr/local/cuda ]]; then
    die "/usr/local/cuda exists and is not a symbolic link; refusing to overwrite it."
fi

"${SUDO[@]}" ln -sfnT "${CUDA_ROOT}" /usr/local/cuda

"${SUDO[@]}" tee /etc/profile.d/cuda-${CUDA_VERSION}.sh >/dev/null <<'EOF'
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export CUDA_PATH=/usr/local/cuda-${CUDA_VERSION}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}
EOF

printf '%s\n' "${CUDA_ROOT}/lib64" | "${SUDO[@]}" tee /etc/ld.so.conf.d/cuda-${CUDA_VERSION_FILE}.conf >/dev/null
"${SUDO[@]}" ldconfig

log "Installation complete."
"${CUDA_ROOT}/bin/nvcc" --version
nvidia-smi
log "Open a new shell, or run: source /etc/profile.d/cuda-${CUDA_VERSION}.sh"
