//! bigedit - A streaming TUI editor for very large files
//!
//! This editor allows editing multi-GB text files without loading them entirely
//! into memory. It uses a viewport + patch list + streaming save approach.

mod editor;
mod journal;
mod overlay;
mod patches;
mod save;
mod search;
mod types;
mod ui;
mod viewport;

use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;

use ui::App;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return Ok(());
    }

    // Handle help flag
    if args[1] == "-h" || args[1] == "--help" {
        print_usage(&args[0]);
        return Ok(());
    }

    // Handle version flag
    if args[1] == "-v" || args[1] == "--version" {
        println!("bigedit {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let path = PathBuf::from(&args[1]);

    // Create and run the application
    let mut app = App::new(&path)
        .context("Failed to initialize editor")?;

    app.run()
        .context("Editor error")?;

    Ok(())
}

fn print_usage(program: &str) {
    println!(
        r#"bigedit - Streaming TUI editor for very large files

USAGE:
    {} <file>

DESCRIPTION:
    Edit multi-GB text files without loading them into memory.
    Uses a nano-like interface with streaming read/write.

KEYBINDINGS:
    Navigation:
        Arrow Keys      Move cursor
        Home/End        Start/end of line
        PgUp/PgDn       Page up/down
        Ctrl+Home       Go to start of file
        Ctrl+End        Go to end of file

    Editing:
        Ctrl+K          Cut current line
        Ctrl+U          Paste (uncut)
        Backspace       Delete char before cursor
        Delete          Delete char at cursor

    File Operations:
        Ctrl+O          Write Out (save)
        Ctrl+S          Save (alternative)
        Ctrl+X          Exit

    Search:
        Ctrl+W          Search forward
        F3              Find next

    Help:
        Ctrl+G          Show help

OPTIONS:
    -h, --help      Show this help message
    -v, --version   Show version

FEATURES:
    • Fast open - only loads a small viewport of the file
    • Streaming search - search through entire file without loading it all
    • Safe save - writes to temp file first, then atomic rename
    • Patch-based editing - edits are recorded as patches, applied on save

EXAMPLES:
    {} large_file.sql
    {} /var/log/huge.log
"#,
        program, program, program
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_help() {
        // Just verify the usage function doesn't panic
        print_usage("bigedit");
    }
}

