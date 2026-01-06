# BigEdit

🖥️ A streaming TUI editor for very large files. Edit multi-gigabyte files without loading them into memory!

![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux-green.svg)

## Features

- **Edit files larger than RAM** - Only loads what's visible on screen
- **Nano-like keybindings** - Familiar interface for terminal users
- **Vi mode** - Toggle with F2 for vi-style editing (h/j/k/l, i/a/o, :w/:q)
- **Undo/Redo** - Full undo support (Ctrl+Z/Ctrl+Y in Nano, u/Ctrl+R in Vi)
- **FUSE virtual filesystem** - Other programs can see your changes before you compact
- **Journal-based saves** - Instant writes without rewriting the entire file
- **Systemd service** - Auto-mount patched file views on boot

## Installation

### Ubuntu/Debian (APT)

```bash
# Add the repository
echo "deb [trusted=yes arch=amd64] https://jopdorp.github.io/bigedit stable main" | sudo tee /etc/apt/sources.list.d/bigedit.list

# Update and install
sudo apt update
sudo apt install bigedit
```

The bigedit-watcher service is enabled by default. To start it immediately:
```bash
systemctl --user daemon-reload && systemctl --user start bigedit-watcher.service
```

### Arch Linux (AUR)

```bash
# Using yay (install yay first if needed: https://github.com/Jguer/yay)
yay -S bigedit

# Or manually with makepkg
git clone https://aur.archlinux.org/bigedit.git
cd bigedit
makepkg -si
```

The bigedit-watcher service is enabled by default. To start it immediately:
```bash
systemctl --user daemon-reload && systemctl --user start bigedit-watcher.service
```

### macOS / Linux (Homebrew)

```bash
brew tap jopdorp/bigedit
brew install bigedit
```

> **Note:** On macOS, bigedit works without FUSE for basic editing. For FUSE features (virtual file view for other programs), install [macFUSE](https://osxfuse.github.io/) first:
> ```bash
> brew install --cask macfuse
> ```
> After installation:
> 1. Go to **System Settings → Privacy & Security**
> 2. Allow the system extension from developer **Benjamin Fleischer**
> 3. Reboot your Mac
> 4. Reinstall bigedit with FUSE support:
> ```bash
> brew reinstall bigedit
> ```

### From Source

```bash
# Clone the repository
git clone https://github.com/jopdorp/bigedit.git
cd bigedit

# Install with systemd service (default)
./install.sh

# Or install without systemd service
./install.sh --no-systemd

# Build without FUSE (if libfuse/macFUSE not available)
cargo install --path . --no-default-features

# To uninstall
./install.sh --uninstall
```

### Build Dependencies

- Rust 1.70+
- libfuse3-dev (Linux) or macFUSE (macOS) - *optional for FUSE features*
- pkg-config

```bash
# Ubuntu/Debian
sudo apt install libfuse3-dev pkg-config fuse3 inotify-tools

# Fedora
sudo dnf install fuse3-devel pkg-config fuse3 inotify-tools

# Arch
sudo pacman -S fuse3 pkg-config inotify-tools

# macOS (optional, for FUSE features)
brew install --cask macfuse
```

## Usage

```bash
bigedit <filename>
```

### Keybindings (Nano Mode - Default)

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
| F1 | Show help |
| **Mode Toggle** | |
| Ctrl+V | Switch to Vi mode |

### Vi Mode

Press `Ctrl+V` to switch to Vi mode. The bottom bar will show the current mode (`-- NORMAL --`, `-- INSERT --`, etc.).

#### Normal Mode
| Key | Action |
|-----|--------|
| h/j/k/l | Move cursor left/down/up/right |
| w/b | Next/previous word |
| 0/$ | Start/end of line |
| gg/G | Start/end of file |
| x | Delete character |
| dd | Delete line |
| yy | Yank (copy) line |
| p | Paste |
| i | Insert before cursor |
| a | Insert after cursor |
| o/O | Open line below/above |
| : | Enter command mode |
| / | Search |
| n | Find next |

#### Insert Mode
| Key | Action |
|-----|--------|
| Esc | Return to normal mode |
| Arrow keys | Navigate |
| All text input | Insert characters |

#### Command Mode (after pressing `:`)
| Command | Action |
|---------|--------|
| :w | Save (journal mode) |
| :q | Quit |
| :q! | Quit without saving |
| :wq | Save and quit |
| :help | Show help |

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
│  (unchanged)    │     │  as binary data  │     │  (FUSE mount)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ file.edited     │
                                                 │ (symlink)       │
                                                 └─────────────────┘
```

**File layout when editing `data.sql`:**
```
data.sql                    # Original file (unchanged until compact)
.data.sql.bigedit-journal   # Patches stored here (instant save)
.data.sql.view/             # FUSE mount directory
  └── data.sql              # Virtual file with patches applied
data.sql.edited             # Symlink → .data.sql.view/data.sql
```

Other programs can read `data.sql.edited` to see your changes in real-time, before you write them to the original file with Ctrl+J.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
