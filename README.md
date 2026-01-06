# BigEdit

🖥️ A streaming TUI editor for very large files. Edit multi-gigabyte files without loading them into memory!

![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux-green.svg)

## Features

- **Edit files larger than RAM** - Only loads what's visible on screen
- **Nano-like keybindings** - Familiar interface for terminal users
- **FUSE virtual filesystem** - Other programs can see your changes before you compact
- **Journal-based saves** - Instant writes without rewriting the entire file
- **Systemd service** - Auto-mount patched file views on boot

## Installation

### From APT Repository (Ubuntu/Debian)

```bash
# Add the repository
echo "deb [trusted=yes arch=amd64] https://jopdorp.github.io/bigedit stable main" | sudo tee /etc/apt/sources.list.d/bigedit.list

# Update and install
sudo apt update
sudo apt install bigedit

# Enable the auto-mount service (optional but recommended)
systemctl --user enable --now bigedit-watcher.service
```

### From Source

```bash
# Clone the repository
git clone https://github.com/jopdorp/bigedit.git
cd bigedit

# Install with systemd service (default)
./install.sh

# Or install without systemd service
./install.sh --no-systemd

# To uninstall
./install.sh --uninstall
```

### Build Dependencies

- Rust 1.70+
- libfuse3-dev
- pkg-config

```bash
# Ubuntu/Debian
sudo apt install libfuse3-dev pkg-config fuse3

# Fedora
sudo dnf install fuse3-devel pkg-config fuse3

# Arch
sudo pacman -S fuse3 pkg-config
```

## Usage

```bash
bigedit <filename>
```

### Keybindings

| Key | Action |
|-----|--------|
| **Navigation** | |
| Arrow Keys | Move cursor |
| Home/End | Start/end of line |
| PgUp/PgDn | Page up/down |
| Ctrl+Home | Go to start of file |
| Ctrl+End | Go to end of file |
| **Editing** | |
| Ctrl+K | Cut current line |
| Ctrl+U | Paste (uncut) |
| Backspace | Delete char before cursor |
| Delete | Delete char at cursor |
| **File Operations** | |
| Ctrl+O | Save (journal mode - instant) |
| Ctrl+S | Save (alternative) |
| Ctrl+J | Compact (full file rewrite) |
| Ctrl+T | Toggle FUSE mode |
| Ctrl+X | Exit |
| **Search** | |
| Ctrl+W | Search forward |
| F3 | Find next |
| **Help** | |
| Ctrl+G | Show help |

## How It Works

### Journal-Based Saving

When you save with `Ctrl+O`, BigEdit doesn't rewrite the entire file. Instead, it saves your changes to a journal file (`.filename.bigedit-journal`). This makes saves instant, even for huge files.

When you're ready to finalize your changes, use `Ctrl+J` to "compact" - this rewrites the original file with all patches applied.

### FUSE Virtual Filesystem

When FUSE mode is enabled (`Ctrl+T`), BigEdit creates a virtual file at `.filename.view/filename` that shows your patched content. Other programs can read this file to see your changes:

```bash
# In one terminal: edit a file
bigedit huge_file.sql

# In another terminal: view the patched content
cat .huge_file.sql.view/huge_file.sql
grep "SELECT" .huge_file.sql.view/huge_file.sql
```

### Auto-Mount Service

If you enable the systemd service, BigEdit will automatically mount FUSE views for any files with active journal files when your system starts:

```bash
# Enable auto-mount
systemctl --user enable --now bigedit-watcher.service

# Check status
systemctl --user status bigedit-watcher.service

# View logs
journalctl --user -u bigedit-watcher.service
```

## Configuration

### Watcher Directories

The watcher service monitors `$HOME` by default. To customize:

```bash
# Edit the watch directories file
mkdir -p ~/.config/bigedit
echo "/home/user" > ~/.config/bigedit/watch-dirs
echo "/data/projects" >> ~/.config/bigedit/watch-dirs

# Restart the service
systemctl --user restart bigedit-watcher.service
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   bigedit       │────▶│  journal file    │────▶│  bigedit-fuse   │
│  (TUI editor)   │     │  (.bigedit-      │     │  (FUSE daemon)  │
│                 │     │   journal)       │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  original file  │     │  patches stored  │     │  .file.view/    │
│                 │     │  as binary data  │     │  (virtual file) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
