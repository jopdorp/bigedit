#!/bin/bash
#
# BigEdit Installation Script
#
# Usage:
#   ./install.sh                    Install with systemd service (default)
#   ./install.sh --no-systemd       Install without systemd service
#   ./install.sh --uninstall        Uninstall bigedit
#
# Environment variables:
#   BIGEDIT_PREFIX=/usr/local       Installation prefix (default: /usr/local)
#   BIGEDIT_ENABLE_SYSTEMD=yes      Enable systemd service (yes/no)
#

set -e

# Configuration
PREFIX="${BIGEDIT_PREFIX:-/usr/local}"
ENABLE_SYSTEMD="${BIGEDIT_ENABLE_SYSTEMD:-yes}"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

# Parse arguments
UNINSTALL=false
for arg in "$@"; do
    case $arg in
        --no-systemd)
            ENABLE_SYSTEMD=no
            ;;
        --uninstall)
            UNINSTALL=true
            ;;
        --help|-h)
            echo "BigEdit Installation Script"
            echo ""
            echo "Usage:"
            echo "  ./install.sh                    Install with systemd service (default)"
            echo "  ./install.sh --no-systemd       Install without systemd service"
            echo "  ./install.sh --uninstall        Uninstall bigedit"
            echo ""
            echo "Environment variables:"
            echo "  BIGEDIT_PREFIX=/usr/local       Installation prefix"
            echo "  BIGEDIT_ENABLE_SYSTEMD=yes      Enable systemd service (yes/no)"
            exit 0
            ;;
        *)
            warn "Unknown argument: $arg"
            ;;
    esac
done

# Uninstall
if [ "$UNINSTALL" = true ]; then
    info "Uninstalling BigEdit..."
    
    # Stop systemd service
    if systemctl --user is-active bigedit-watcher.service >/dev/null 2>&1; then
        info "Stopping bigedit-watcher service..."
        systemctl --user stop bigedit-watcher.service || true
        systemctl --user disable bigedit-watcher.service || true
    fi
    
    # Remove systemd files
    rm -f "$SYSTEMD_USER_DIR/bigedit-watcher.service"
    rm -f "$SYSTEMD_USER_DIR/bigedit-fuse@.service"
    systemctl --user daemon-reload 2>/dev/null || true
    
    # Remove binaries
    sudo rm -f "$PREFIX/bin/bigedit"
    sudo rm -f "$PREFIX/bin/bigedit-fuse"
    sudo rm -f "$PREFIX/bin/bigedit-watcher"
    
    success "BigEdit uninstalled successfully!"
    exit 0
fi

# Check dependencies
info "Checking dependencies..."

if ! command -v cargo &>/dev/null; then
    error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
fi

if ! command -v fusermount3 &>/dev/null; then
    warn "fuse3 not found. Installing..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y fuse3
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y fuse3
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm fuse3
    else
        error "Could not install fuse3. Please install it manually."
    fi
fi

success "Dependencies OK"

# Build
info "Building BigEdit..."
cargo build --release
success "Build complete"

# Install binaries
info "Installing binaries to $PREFIX/bin..."
sudo install -D -m 755 target/release/bigedit "$PREFIX/bin/bigedit"
sudo install -D -m 755 target/release/bigedit-fuse "$PREFIX/bin/bigedit-fuse"
sudo install -D -m 755 systemd/bigedit-watcher "$PREFIX/bin/bigedit-watcher"
success "Binaries installed"

# Install systemd services
if [ "$ENABLE_SYSTEMD" = "yes" ]; then
    info "Installing systemd user services..."
    mkdir -p "$SYSTEMD_USER_DIR"
    install -m 644 systemd/bigedit-watcher.service "$SYSTEMD_USER_DIR/"
    install -m 644 systemd/bigedit-fuse@.service "$SYSTEMD_USER_DIR/"
    
    # Update ExecStart paths in service files
    sed -i "s|/usr/bin/bigedit|$PREFIX/bin/bigedit|g" "$SYSTEMD_USER_DIR/bigedit-watcher.service"
    sed -i "s|/usr/bin/bigedit|$PREFIX/bin/bigedit|g" "$SYSTEMD_USER_DIR/bigedit-fuse@.service"
    
    systemctl --user daemon-reload
    success "Systemd services installed"
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    BigEdit Installation Complete                  ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                                                                   ║"
    echo "║  To enable the auto-mount service, run:                          ║"
    echo "║                                                                   ║"
    echo "║    systemctl --user enable --now bigedit-watcher.service         ║"
    echo "║                                                                   ║"
    echo "║  This will automatically mount FUSE views for files you edit     ║"
    echo "║  so other programs can see your changes before compacting.       ║"
    echo "║                                                                   ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    BigEdit Installation Complete                  ║"
    echo "║                                                                   ║"
    echo "║  Systemd service was not installed (--no-systemd flag).          ║"
    echo "║  You can install it later by running:                            ║"
    echo "║                                                                   ║"
    echo "║    ./install.sh                                                   ║"
    echo "║                                                                   ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
fi

echo ""
echo "Usage:"
echo "    bigedit <filename>     Edit a file"
echo ""
echo "Keybindings:"
echo "    Ctrl+O     Save (journal mode)"
echo "    Ctrl+J     Compact (full file rewrite)"
echo "    Ctrl+T     Toggle FUSE mode"
echo "    Ctrl+X     Exit"
echo "    Ctrl+G     Help"
echo ""
success "Installation complete!"
