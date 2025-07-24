#!/bin/bash

# YICA-Mirage Cross-Platform Installation Script
# Supports pip, brew, apt, yum, and manual installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# Configuration
PACKAGE_NAME="yica-mirage"
VERSION="1.0.0"
GITHUB_REPO="yica-ai/yica-mirage"
INSTALL_PREFIX="/usr/local"

# Platform detection
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case "$os" in
        linux*)
            if command -v apt-get >/dev/null 2>&1; then
                PLATFORM="debian"
                PKG_MANAGER="apt"
            elif command -v yum >/dev/null 2>&1; then
                PLATFORM="rhel"
                PKG_MANAGER="yum"
            elif command -v dnf >/dev/null 2>&1; then
                PLATFORM="rhel"
                PKG_MANAGER="dnf"
            elif command -v pacman >/dev/null 2>&1; then
                PLATFORM="arch"
                PKG_MANAGER="pacman"
            else
                PLATFORM="linux"
                PKG_MANAGER="manual"
            fi
            ;;
        darwin*)
            PLATFORM="macos"
            if command -v brew >/dev/null 2>&1; then
                PKG_MANAGER="brew"
            else
                PKG_MANAGER="manual"
            fi
            ;;
        *)
            PLATFORM="unknown"
            PKG_MANAGER="manual"
            ;;
    esac
    
    log_info "Detected platform: $PLATFORM ($arch)"
    log_info "Package manager: $PKG_MANAGER"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        missing_deps+=("python3")
    else
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Python version: $python_version"
        if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
            log_error "Python 3.8+ required, found $python_version"
            missing_deps+=("python3>=3.8")
        fi
    fi
    
    # Check pip
    if ! command -v pip3 >/dev/null 2>&1; then
        missing_deps+=("python3-pip")
    fi
    
    # Check cmake (for source builds)
    if ! command -v cmake >/dev/null 2>&1; then
        log_warn "CMake not found - required for source builds"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
    
    log_info "All dependencies satisfied"
    return 0
}

# Install via pip
install_pip() {
    log_info "Installing YICA-Mirage via pip..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install with optional dependencies
    if [[ "$1" == "--cuda" ]]; then
        python3 -m pip install "${PACKAGE_NAME}[cuda]"
    elif [[ "$1" == "--dev" ]]; then
        python3 -m pip install "${PACKAGE_NAME}[dev]"
    elif [[ "$1" == "--all" ]]; then
        python3 -m pip install "${PACKAGE_NAME}[all]"
    else
        python3 -m pip install "${PACKAGE_NAME}"
    fi
    
    log_info "YICA-Mirage installed successfully via pip"
}

# Install via Homebrew (macOS)
install_brew() {
    log_info "Installing YICA-Mirage via Homebrew..."
    
    # Add tap if not exists
    brew tap yica-ai/tap || true
    
    # Install package
    brew install yica-mirage
    
    log_info "YICA-Mirage installed successfully via Homebrew"
}

# Install via APT (Debian/Ubuntu)
install_apt() {
    log_info "Installing YICA-Mirage via APT..."
    
    # Update package list
    sudo apt-get update
    
    # Install dependencies
    sudo apt-get install -y \
        python3 python3-pip python3-dev \
        cmake ninja-build \
        libz3-4 libz3-dev \
        python3-numpy python3-torch
    
    # Add repository
    wget -qO - https://packages.yica.ai/gpg.key | sudo apt-key add -
    echo "deb https://packages.yica.ai/debian stable main" | sudo tee /etc/apt/sources.list.d/yica.list
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y yica-mirage python3-yica-mirage
    
    log_info "YICA-Mirage installed successfully via APT"
}

# Install via YUM/DNF (RHEL/CentOS/Fedora)
install_yum() {
    log_info "Installing YICA-Mirage via $PKG_MANAGER..."
    
    # Install dependencies
    sudo $PKG_MANAGER install -y \
        python3 python3-pip python3-devel \
        cmake ninja-build \
        z3 z3-devel \
        python3-numpy python3-torch
    
    # Add repository
    sudo tee /etc/yum.repos.d/yica.repo > /dev/null <<EOF
[yica]
name=YICA Repository
baseurl=https://packages.yica.ai/rpm/\$basearch
enabled=1
gpgcheck=1
gpgkey=https://packages.yica.ai/gpg.key
EOF
    
    # Install package
    sudo $PKG_MANAGER install -y yica-mirage python3-yica-mirage
    
    log_info "YICA-Mirage installed successfully via $PKG_MANAGER"
}

# Manual installation from source
install_manual() {
    log_info "Installing YICA-Mirage manually from source..."
    
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Download source
    log_info "Downloading source code..."
    curl -L "https://github.com/$GITHUB_REPO/archive/v$VERSION.tar.gz" -o "$PACKAGE_NAME-$VERSION.tar.gz"
    tar -xzf "$PACKAGE_NAME-$VERSION.tar.gz"
    cd "$PACKAGE_NAME-$VERSION"
    
    # Build C++ components
    log_info "Building C++ components..."
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_TESTS=OFF
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # Install C++ components
    sudo make install
    
    # Install Python package
    log_info "Installing Python package..."
    cd ../mirage/python
    python3 -m pip install .
    
    # Cleanup
    cd /
    rm -rf "$temp_dir"
    
    log_info "YICA-Mirage installed successfully from source"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Python package
    if python3 -c "import yica_mirage; print(f'YICA-Mirage version: {yica_mirage.__version__}')" 2>/dev/null; then
        log_info "Python package verified"
    else
        log_error "Python package verification failed"
        return 1
    fi
    
    # Check command-line tools
    if command -v yica-optimizer >/dev/null 2>&1; then
        log_info "Command-line tools verified"
        yica-optimizer --version
    else
        log_warn "Command-line tools not found in PATH"
    fi
    
    log_info "Installation verification completed"
}

# Show usage
show_usage() {
    cat << EOF
YICA-Mirage Installation Script

Usage: $0 [OPTIONS]

Options:
    --method METHOD     Installation method (pip, brew, apt, yum, manual)
    --cuda             Install with CUDA support (pip only)
    --dev              Install development dependencies (pip only)
    --all              Install all optional dependencies (pip only)
    --prefix PREFIX    Installation prefix for manual builds (default: /usr/local)
    --help             Show this help message

Examples:
    $0                          # Auto-detect and install
    $0 --method pip --cuda      # Install via pip with CUDA support
    $0 --method brew            # Install via Homebrew (macOS)
    $0 --method manual          # Manual installation from source

Supported Platforms:
    - Linux (Debian/Ubuntu, RHEL/CentOS/Fedora, Arch)
    - macOS (Intel/Apple Silicon)
    - Windows (via WSL or manual)

For more information, visit: https://github.com/$GITHUB_REPO
EOF
}

# Main installation function
main() {
    local method=""
    local pip_extras=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --method)
                method="$2"
                shift 2
                ;;
            --cuda)
                pip_extras="--cuda"
                shift
                ;;
            --dev)
                pip_extras="--dev"
                shift
                ;;
            --all)
                pip_extras="--all"
                shift
                ;;
            --prefix)
                INSTALL_PREFIX="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "YICA-Mirage Installation Script v$VERSION"
    log_info "=========================================="
    
    # Detect platform if method not specified
    if [[ -z "$method" ]]; then
        detect_platform
        method="$PKG_MANAGER"
    fi
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi
    
    # Install based on method
    case "$method" in
        pip)
            install_pip "$pip_extras"
            ;;
        brew)
            install_brew
            ;;
        apt)
            install_apt
            ;;
        yum|dnf)
            install_yum
            ;;
        manual)
            install_manual
            ;;
        *)
            log_error "Unsupported installation method: $method"
            log_info "Falling back to pip installation..."
            install_pip "$pip_extras"
            ;;
    esac
    
    # Verify installation
    verify_installation
    
    log_info "=========================================="
    log_info "YICA-Mirage installation completed successfully!"
    log_info "Run 'yica-optimizer --help' to get started."
}

# Run main function
main "$@" 