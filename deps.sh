#!/usr/bin/env bash

set -Eeuo pipefail

readonly REQUIRED_OS_ID="ubuntu"
readonly REQUIRED_OS_VERSION="26.04"
readonly REQUIRED_ARCH="amd64"
readonly CUDA_VERSION="13.2"
readonly CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}"
readonly CUDA_SYMLINK="/usr/local/cuda"
readonly MINIMUM_DRIVER_VERSION="595.45.04"
readonly REQUIRED_GCC_MAJOR="15"
readonly CUDA_PROFILE="/etc/profile.d/cuda-${CUDA_VERSION}.sh"
readonly CUDA_LD_CONFIG="/etc/ld.so.conf.d/cuda-13-2.conf"

info() {
    printf '\033[1;36m[INFO]\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2
}

die() {
    printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2
    exit 1
}

if [[ ${EUID} -eq 0 ]]; then
    SUDO=()
else
    command -v sudo >/dev/null 2>&1 || die "sudo is required when not running as root."
    SUDO=(sudo)
fi

check_platform() {
    [[ -r /etc/os-release ]] || die "Cannot read /etc/os-release."

    # shellcheck disable=SC1091
    source /etc/os-release

    [[ ${ID:-} == "${REQUIRED_OS_ID}" ]] || \
        die "Ubuntu is required. Detected: ${ID:-unknown}."
    [[ ${VERSION_ID:-} == "${REQUIRED_OS_VERSION}" ]] || \
        die "Ubuntu ${REQUIRED_OS_VERSION} is required. Detected: ${VERSION_ID:-unknown}."
    [[ $(dpkg --print-architecture) == "${REQUIRED_ARCH}" ]] || \
        die "This script supports x86_64/amd64 only."
}

install_dependencies() {
    info "Installing build dependencies for CUDA ${CUDA_VERSION}."

    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        gcc-15 \
        g++-15 \
        ninja-build \
        pciutils \
        pkg-config \
        wget
}

check_driver() {
    command -v nvidia-smi >/dev/null 2>&1 || die \
        "nvidia-smi was not found. Install a compatible NVIDIA driver before running this script."

    local driver_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
        | head -n1 \
        | tr -d '[:space:]')

    [[ -n ${driver_version} ]] || die "Could not determine the NVIDIA driver version."

    dpkg --compare-versions "${driver_version}" ge "${MINIMUM_DRIVER_VERSION}" || die \
        "Driver ${driver_version} is too old; CUDA ${CUDA_VERSION} requires ${MINIMUM_DRIVER_VERSION} or newer."

    info "NVIDIA driver ${driver_version} is compatible with CUDA ${CUDA_VERSION}."
}

check_compiler() {
    [[ -x /usr/bin/gcc-15 ]] || die "/usr/bin/gcc-15 is missing."
    [[ -x /usr/bin/g++-15 ]] || die "/usr/bin/g++-15 is missing."

    local gcc_major
    gcc_major=$(/usr/bin/gcc-15 -dumpfullversion -dumpversion | cut -d. -f1)
    [[ ${gcc_major} == "${REQUIRED_GCC_MAJOR}" ]] || die \
        "Expected GCC ${REQUIRED_GCC_MAJOR}, but found GCC ${gcc_major}."

    info "GCC $(/usr/bin/gcc-15 -dumpfullversion -dumpversion) is available."
}

check_toolkit() {
    local nvcc="${CUDA_ROOT}/bin/nvcc"
    pushd .
    cd /tmp
    wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run
    echo "656f4a652313abd118fb0ae1a8b902d3  cuda_13.2.0_595.45.04_linux.run" | md5sum --check
    sudo sh cuda_13.2.0_595.45.04_linux.run \
    --silent \
    --toolkit \
    --toolkitpath=/usr/local/cuda-13.2 \
    --override
    popd
    [[ -x ${nvcc} ]] || die \
        "CUDA ${CUDA_VERSION} was not found at ${CUDA_ROOT}. Run the CUDA 13.2 installer first."

    "${nvcc}" --version | grep -q "release ${CUDA_VERSION}" || die \
        "${nvcc} is not CUDA ${CUDA_VERSION}."

    info "CUDA Toolkit ${CUDA_VERSION} is installed at ${CUDA_ROOT}."
}

configure_cuda() {
    check_toolkit

    if [[ -e ${CUDA_SYMLINK} && ! -L ${CUDA_SYMLINK} ]]; then
        die "${CUDA_SYMLINK} exists and is not a symbolic link; refusing to overwrite it."
    fi

    "${SUDO[@]}" ln -sfnT "${CUDA_ROOT}" "${CUDA_SYMLINK}"

    "${SUDO[@]}" tee "${CUDA_PROFILE}" >/dev/null <<EOF
export CUDA_HOME=${CUDA_ROOT}
export CUDA_PATH=${CUDA_ROOT}
export PATH=${CUDA_ROOT}/bin\${PATH:+:\${PATH}}
export NVCC_CCBIN=/usr/bin/g++-15
EOF

    printf '%s\n' "${CUDA_ROOT}/lib64" \
        | "${SUDO[@]}" tee "${CUDA_LD_CONFIG}" >/dev/null
    "${SUDO[@]}" ldconfig

    info "Configured CUDA ${CUDA_VERSION} as the active toolkit."
}

run_smoke_test() {
    local test_dir source_file binary_file
    test_dir=$(mktemp -d -t cuda-13.2-test.XXXXXXXX)
    source_file="${test_dir}/cuda_check.cu"
    binary_file="${test_dir}/cuda_check"

    cat >"${source_file}" <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    std::printf("CUDA devices detected: %d\n", device_count);
    return device_count > 0 ? 0 : 1;
}
EOF

    if ! "${CUDA_ROOT}/bin/nvcc" \
        -ccbin /usr/bin/g++-15 \
        "${source_file}" \
        -o "${binary_file}"; then
        rm -rf -- "${test_dir}"
        die "CUDA compilation test failed."
    fi

    if ! "${binary_file}"; then
        rm -rf -- "${test_dir}"
        die "CUDA runtime test failed."
    fi

    rm -rf -- "${test_dir}"
    info "CUDA compilation and runtime tests passed."
}

verify() {
    check_platform
    check_driver
    check_compiler
    check_toolkit
    run_smoke_test

    info "Dependency verification completed successfully."
}

install() {
    check_platform
    install_dependencies
    check_driver
    check_compiler
    configure_cuda
    run_smoke_test

    info "Dependencies for CUDA ${CUDA_VERSION} are ready."
    info "Open a new shell, or run: source ${CUDA_PROFILE}"
}

uninstall_configuration() {
    warn "Removing only configuration created by this wrapper."
    warn "The NVIDIA driver, CUDA Toolkit and shared build packages will remain installed."

    if [[ -L ${CUDA_SYMLINK} ]] \
        && [[ $(readlink -f "${CUDA_SYMLINK}") == "${CUDA_ROOT}" ]]; then
        "${SUDO[@]}" rm -- "${CUDA_SYMLINK}"
    fi

    "${SUDO[@]}" rm -f -- "${CUDA_PROFILE}" "${CUDA_LD_CONFIG}"
    "${SUDO[@]}" ldconfig

    info "CUDA ${CUDA_VERSION} wrapper configuration removed."
    info "To remove the runfile Toolkit itself, run: sudo ${CUDA_ROOT}/bin/cuda-uninstaller"
}

compiler_info() {
    local version=${1:-15}

    [[ ${version} == "15" ]] || die \
        "CUDA ${CUDA_VERSION} on Ubuntu ${REQUIRED_OS_VERSION} is configured for GCC 15."

    check_compiler
    info "Global /usr/bin/gcc and /usr/bin/g++ links were not modified."
    printf 'For a single build, use:\n'
    printf '  CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 <command>\n'
}

help_menu() {
    cat <<EOF
CUDA ${CUDA_VERSION} Dependency Wrapper for Ubuntu ${REQUIRED_OS_VERSION}

Usage:
  $0 --action=install
  $0 --action=verify
  $0 --action=compiler --version=15
  $0 --action=uninstall

Actions:
  install      Install build dependencies, configure CUDA and run a smoke test.
  verify       Verify the driver, compiler, Toolkit, compilation and GPU runtime.
  compiler     Validate GCC 15 without changing global compiler symlinks.
  uninstall    Remove only this wrapper's CUDA profile, linker configuration
               and active CUDA symlink. It does not remove the driver or Toolkit.

This wrapper intentionally does not install or remove NVIDIA drivers or CUDA
APT packages. CUDA ${CUDA_VERSION} is managed by the separate NVIDIA runfile installer.
EOF
}

action=""
version="15"

for argument in "$@"; do
    case ${argument} in
        --action=*) action=${argument#*=} ;;
        --version=*) version=${argument#*=} ;;
        -h|--help) help_menu; exit 0 ;;
        *) die "Unknown argument: ${argument}" ;;
    esac
done

case ${action} in
    i|install) install ;;
    v|verify) verify ;;
    c|cc|compiler|change-compiler) compiler_info "${version}" ;;
    u|uninstall) uninstall_configuration ;;
    "") help_menu; exit 1 ;;
    *) die "Unknown action: ${action}" ;;
esac

