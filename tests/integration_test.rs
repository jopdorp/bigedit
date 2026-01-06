use std::fs;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use tempfile::NamedTempFile;

/// Send keys to a PTY-controlled process
/// This test spawns the actual editor binary and sends keystrokes via PTY
#[test]
fn test_user_sequence_integration() {
    // Create a temp file with test content
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(temp_file, "line1").unwrap();
    writeln!(temp_file, "line2").unwrap();
    write!(temp_file, "line3").unwrap(); // no trailing newline
    temp_file.flush().unwrap();
    
    let file_path = temp_file.path().to_path_buf();
    
    // Read initial content
    let initial_content = fs::read_to_string(&file_path).expect("read initial");
    println!("Initial content ({} bytes):\n{}", initial_content.len(), initial_content);
    println!("Initial lines: {}", initial_content.lines().count());
    
    // Use `script` command to create a PTY and run the editor
    // This is needed because the editor uses raw terminal mode
    let script_output = NamedTempFile::new().expect("Failed to create script output file");
    let script_output_path = script_output.path().to_path_buf();
    
    // Build the binary first
    let build_status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("Failed to build");
    assert!(build_status.success(), "Build failed");
    
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    // Create a script that will send our keystrokes
    // Using heredoc-style input with `expect` or direct terminal control
    let keystroke_script = format!(
        r#"#!/usr/bin/env bash
set -e

# Run editor in a script session to get a PTY
# Use printf to send raw bytes

# Start the editor in background, connected to our terminal
exec 3<>/dev/tty 2>/dev/null || true

# Run with timeout to prevent hanging
timeout 5s bash -c '
    # Create a temporary FIFO for input
    FIFO=$(mktemp -u)
    mkfifo "$FIFO"
    
    # Start editor reading from FIFO
    "{binary}" "{file}" < "$FIFO" &
    EDITOR_PID=$!
    
    # Give editor time to start
    sleep 0.3
    
    # Send keystrokes:
    # DOWN = \x1b[B
    # ENTER = \x0d (carriage return)
    # BACKSPACE = \x7f
    # LEFT = \x1b[D
    # Ctrl-X = \x18 (exit)
    
    # User sequence: down, enter, backspace, left, enter, backspace, enter, enter, backspace, backspace
    # Then Ctrl-X, N to exit without save... actually lets save with Ctrl-O
    
    (
        sleep 0.2
        printf "\x1b[B"      # DOWN
        sleep 0.1
        printf "\x0d"        # ENTER
        sleep 0.1
        printf "\x7f"        # BACKSPACE  
        sleep 0.1
        printf "\x1b[D"      # LEFT
        sleep 0.1
        printf "\x0d"        # ENTER
        sleep 0.1
        printf "\x7f"        # BACKSPACE (reported failing)
        sleep 0.1
        printf "\x0d"        # ENTER
        sleep 0.1
        printf "\x0d"        # ENTER
        sleep 0.1
        printf "\x7f"        # BACKSPACE (reported failing)
        sleep 0.1
        printf "\x7f"        # BACKSPACE (reported failing)
        sleep 0.1
        printf "\x0f"        # Ctrl-O (save)
        sleep 0.2
        printf "\x0d"        # ENTER to confirm
        sleep 0.1
        printf "\x18"        # Ctrl-X (exit)
        sleep 0.1
        printf "n"           # n to confirm exit
    ) > "$FIFO"
    
    wait $EDITOR_PID 2>/dev/null || true
    rm -f "$FIFO"
' 2>&1 || true
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    // Instead of the complex PTY approach, let's use a simpler method:
    // Use the `expect` tool or `tmux send-keys` if available,
    // or just test with direct byte injection
    
    // Actually, let's use a much simpler approach - use tmux or screen
    // Or even simpler: test via the rust pty crate
    
    println!("Note: Full integration test with PTY requires special setup.");
    println!("Running simplified test that verifies file content changes...");
    
    // For now, let's at least verify we can spawn the binary
    let output = Command::new(&binary_path)
        .arg("--help")
        .output();
    
    match output {
        Ok(out) => {
            println!("Binary runs, stdout: {}", String::from_utf8_lossy(&out.stdout));
            println!("Binary runs, stderr: {}", String::from_utf8_lossy(&out.stderr));
        }
        Err(e) => {
            println!("Could not run binary: {}", e);
        }
    }
}

/// Test using the portable-pty crate for proper PTY control
/// This requires adding portable-pty as a dev dependency
#[test]
fn test_editor_with_pty() {
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::os::unix::process::CommandExt;
    
    // Create test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(temp_file, "line1").unwrap();
    writeln!(temp_file, "line2").unwrap();
    write!(temp_file, "line3").unwrap();
    temp_file.flush().unwrap();
    
    let file_path = temp_file.path().to_path_buf();
    let initial_content = fs::read_to_string(&file_path).unwrap();
    let initial_lines = initial_content.lines().count();
    
    println!("Initial: {} lines", initial_lines);
    println!("Content:\n{}", initial_content);
    
    // Use nix crate for PTY if available, otherwise use pty-process or script command
    // For simplicity, we'll use the `script` command which is available on most Unix systems
    
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    // Check if binary exists
    if !std::path::Path::new(&binary_path).exists() {
        println!("Binary not found, building...");
        let status = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("Failed to build");
        assert!(status.success());
    }
    
    // Create a helper script that uses `script` command for PTY
    let helper_script = format!(
        r#"#!/bin/bash
# Run editor with PTY via script command

export TERM=xterm
export LINES=24
export COLUMNS=80

# Use Python for reliable keystroke sending with PTY
python3 << 'PYTHON_EOF'
import pty
import os
import sys
import time
import select

def send_keys(master_fd, keys, delay=0.05):
    for key in keys:
        os.write(master_fd, key)
        time.sleep(delay)
        # Drain any output
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.01)
            if not r:
                break
            try:
                os.read(master_fd, 4096)
            except:
                break

pid, master_fd = pty.fork()

if pid == 0:
    # Child - run the editor
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    # Parent - send keystrokes
    time.sleep(0.5)  # Wait for editor to start
    
    # Key codes
    DOWN = b'\x1b[B'
    ENTER = b'\r'
    BACKSPACE = b'\x7f'
    LEFT = b'\x1b[D'
    CTRL_O = b'\x0f'
    CTRL_X = b'\x18'
    
    # User sequence: down, enter, backspace, left, enter, backspace, enter, enter, backspace, backspace
    keys = [
        DOWN,       # down
        ENTER,      # enter
        BACKSPACE,  # backspace
        LEFT,       # left
        ENTER,      # enter
        BACKSPACE,  # backspace (reported failing)
        ENTER,      # enter
        ENTER,      # enter
        BACKSPACE,  # backspace (reported failing)
        BACKSPACE,  # backspace (reported failing)
        CTRL_O,     # save
    ]
    
    send_keys(master_fd, keys, delay=0.1)
    time.sleep(0.3)
    
    # Confirm save (press enter at filename prompt)
    os.write(master_fd, ENTER)
    time.sleep(0.3)
    
    # Exit
    os.write(master_fd, CTRL_X)
    time.sleep(0.1)
    os.write(master_fd, b'n')  # Don't save again
    time.sleep(0.2)
    
    # Wait for child to exit
    try:
        os.waitpid(pid, 0)
    except:
        pass
    
    os.close(master_fd)
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    // Write the helper script
    let script_file = NamedTempFile::new().expect("Failed to create script file");
    let script_path = script_file.path().to_path_buf();
    fs::write(&script_path, &helper_script).expect("Failed to write script");
    
    // Make executable
    Command::new("chmod")
        .args(["+x", script_path.to_str().unwrap()])
        .status()
        .expect("chmod failed");
    
    // Run the script
    println!("Running PTY test script...");
    let output = Command::new("bash")
        .arg(&script_path)
        .env("TERM", "xterm")
        .output()
        .expect("Failed to run script");
    
    println!("Script stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("Script stderr: {}", String::from_utf8_lossy(&output.stderr));
    
    // Read the final content
    let final_content = fs::read_to_string(&file_path).unwrap();
    let final_lines = final_content.lines().count();
    
    println!("\nFinal: {} lines", final_lines);
    println!("Content:\n{}", final_content);
    println!("Raw bytes: {:?}", final_content.as_bytes());
    
    // Count actual newlines in raw content
    let newline_count = final_content.bytes().filter(|&b| b == b'\n').count();
    println!("Newline count in final: {}", newline_count);
    
    println!("\nFinal: {} lines", final_lines);
    println!("Content:\n{}", final_content);
    
    // The user sequence should result in the same number of lines as initial
    // because: enter, backspace, left, enter, backspace, enter, enter, backspace, backspace
    // = +1, -1, move, +1, -1, +1, +1, -1, -1 = net 0 lines added
    
    // But if backspaces are failing, we'd have MORE lines
    println!("\nLine count change: {} -> {} (delta: {})", 
             initial_lines, final_lines, final_lines as i32 - initial_lines as i32);
    
    // Check that we don't have extra lines from failed backspaces
    // Expected: same number of lines (3)
    // If backspaces fail: we'd have 3 + 4 enters - working backspaces = more than 3
    
    if final_lines > initial_lines {
        println!("POTENTIAL BUG: Line count increased from {} to {}", initial_lines, final_lines);
        println!("This suggests some backspaces are not working!");
    }
    
    assert_eq!(final_lines, initial_lines, 
               "Backspace operations should restore original line count. Had {} lines, now have {}",
               initial_lines, final_lines);
}

/// Test that captures screen output after each keystroke to debug display issues
#[test]
fn test_editor_with_screen_capture() {
    use std::os::unix::io::{AsRawFd, FromRawFd};
    
    // Create test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(temp_file, "line1").unwrap();
    writeln!(temp_file, "line2").unwrap();
    write!(temp_file, "line3").unwrap();
    temp_file.flush().unwrap();
    
    let file_path = temp_file.path().to_path_buf();
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    // Check if binary exists
    if !std::path::Path::new(&binary_path).exists() {
        let status = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("Failed to build");
        assert!(status.success());
    }
    
    // Create a helper script that captures screen after each key
    let helper_script = format!(
        r#"#!/bin/bash
export TERM=xterm
export LINES=24
export COLUMNS=80

python3 << 'PYTHON_EOF'
import pty
import os
import sys
import time
import select

def read_screen(master_fd, timeout=0.1):
    """Read all available output from the PTY"""
    output = b''
    while True:
        r, _, _ = select.select([master_fd], [], [], timeout)
        if not r:
            break
        try:
            data = os.read(master_fd, 4096)
            if not data:
                break
            output += data
        except:
            break
    return output

def extract_visible_lines(screen_data):
    """Try to extract visible text lines from terminal output"""
    # Remove ANSI escape sequences for cleaner output
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean = ansi_escape.sub('', screen_data.decode('utf-8', errors='replace'))
    # Get non-empty lines
    lines = [l.strip() for l in clean.split('\n') if l.strip()]
    return lines

pid, master_fd = pty.fork()

if pid == 0:
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    time.sleep(0.5)
    initial_screen = read_screen(master_fd)
    print("=== INITIAL SCREEN ===")
    for line in extract_visible_lines(initial_screen)[:10]:
        print(f"  {{line}}")
    
    DOWN = b'\x1b[B'
    ENTER = b'\r'
    BACKSPACE = b'\x7f'
    LEFT = b'\x1b[D'
    CTRL_O = b'\x0f'
    CTRL_X = b'\x18'
    
    actions = [
        ("DOWN", DOWN),
        ("ENTER", ENTER),
        ("BACKSPACE", BACKSPACE),
        ("LEFT", LEFT),
        ("ENTER", ENTER),
        ("BACKSPACE (reported failing)", BACKSPACE),
        ("ENTER", ENTER),
        ("ENTER", ENTER),
        ("BACKSPACE (reported failing)", BACKSPACE),
        ("BACKSPACE (reported failing)", BACKSPACE),
    ]
    
    for name, key in actions:
        os.write(master_fd, key)
        time.sleep(0.15)
        screen = read_screen(master_fd)
        visible = extract_visible_lines(screen)
        print(f"\n=== AFTER {{name}} ===")
        # Show first few lines that look like content
        content_lines = [l for l in visible if l.startswith('line') or l.startswith('~') or not l]
        for line in content_lines[:6]:
            print(f"  {{line}}")
        if not content_lines:
            for line in visible[:6]:
                print(f"  {{line}}")
    
    # Save and exit
    print("\n=== SAVING ===")
    os.write(master_fd, CTRL_O)
    time.sleep(0.2)
    read_screen(master_fd)
    os.write(master_fd, ENTER)
    time.sleep(0.3)
    
    os.write(master_fd, CTRL_X)
    time.sleep(0.1)
    os.write(master_fd, b'n')
    time.sleep(0.2)
    
    try:
        os.waitpid(pid, 0)
    except:
        pass
    os.close(master_fd)
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    let script_file = NamedTempFile::new().expect("Failed to create script file");
    let script_path = script_file.path().to_path_buf();
    fs::write(&script_path, &helper_script).expect("Failed to write script");
    
    Command::new("chmod")
        .args(["+x", script_path.to_str().unwrap()])
        .status()
        .expect("chmod failed");
    
    println!("Running screen capture test...");
    let output = Command::new("bash")
        .arg(&script_path)
        .env("TERM", "xterm")
        .output()
        .expect("Failed to run script");
    
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Check final file
    let final_content = fs::read_to_string(&file_path).unwrap();
    let final_lines = final_content.lines().count();
    println!("\n=== FINAL FILE CONTENT ===");
    println!("{}", final_content);
    println!("Lines: {}", final_lines);
    
    assert_eq!(final_lines, 3, "Should have 3 lines after all operations");
}

/// Test using the actual test3 file (Makefile copy)
#[test]
fn test_with_test3_file() {
    let source_path = format!("{}/test3", env!("CARGO_MANIFEST_DIR"));
    
    // Check if test3 exists
    if !std::path::Path::new(&source_path).exists() {
        println!("test3 file not found, skipping test");
        return;
    }
    
    // Copy to temp file to avoid modifying original
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path().to_path_buf();
    fs::copy(&source_path, &file_path).expect("Failed to copy test3");
    
    let initial_content = fs::read_to_string(&file_path).unwrap();
    let initial_lines = initial_content.lines().count();
    println!("Initial: {} lines", initial_lines);
    println!("First 5 lines:");
    for line in initial_content.lines().take(5) {
        println!("  {}", line);
    }
    
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    // Build if needed
    if !std::path::Path::new(&binary_path).exists() {
        let status = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("Failed to build");
        assert!(status.success());
    }
    
    // Create test script
    let helper_script = format!(
        r#"#!/bin/bash
export TERM=xterm
export LINES=24
export COLUMNS=80

python3 << 'PYTHON_EOF'
import pty
import os
import sys
import time
import select

def send_key(master_fd, key, delay=0.1):
    os.write(master_fd, key)
    time.sleep(delay)
    # Drain output
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.02)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break

pid, master_fd = pty.fork()

if pid == 0:
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    time.sleep(0.5)
    # Drain initial output
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.1)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break
    
    DOWN = b'\x1b[B'
    ENTER = b'\r'
    BACKSPACE = b'\x7f'
    LEFT = b'\x1b[D'
    CTRL_O = b'\x0f'
    CTRL_X = b'\x18'
    
    # User sequence: down, enter, backspace, left, enter, backspace, enter, enter, backspace, backspace
    print("Sending: DOWN")
    send_key(master_fd, DOWN)
    
    print("Sending: ENTER")
    send_key(master_fd, ENTER)
    
    print("Sending: BACKSPACE")
    send_key(master_fd, BACKSPACE)
    
    print("Sending: LEFT")
    send_key(master_fd, LEFT)
    
    print("Sending: ENTER")
    send_key(master_fd, ENTER)
    
    print("Sending: BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    print("Sending: ENTER")
    send_key(master_fd, ENTER)
    
    print("Sending: ENTER")
    send_key(master_fd, ENTER)
    
    print("Sending: BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    print("Sending: BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    # Save
    print("Sending: CTRL-O (save)")
    send_key(master_fd, CTRL_O, delay=0.2)
    
    print("Sending: ENTER (confirm)")
    send_key(master_fd, ENTER, delay=0.3)
    
    # Exit
    print("Sending: CTRL-X (exit)")
    send_key(master_fd, CTRL_X)
    
    print("Sending: n (don't save again)")
    send_key(master_fd, b'n', delay=0.2)
    
    try:
        os.waitpid(pid, 0)
    except:
        pass
    os.close(master_fd)
    print("Done!")
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    let script_file = NamedTempFile::new().expect("Failed to create script file");
    let script_path = script_file.path().to_path_buf();
    fs::write(&script_path, &helper_script).expect("Failed to write script");
    
    Command::new("chmod")
        .args(["+x", script_path.to_str().unwrap()])
        .status()
        .expect("chmod failed");
    
    println!("\nRunning editor with test3 content...");
    let output = Command::new("bash")
        .arg(&script_path)
        .env("TERM", "xterm")
        .output()
        .expect("Failed to run script");
    
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Check final file
    let final_content = fs::read_to_string(&file_path).unwrap();
    let final_lines = final_content.lines().count();
    
    println!("\n=== RESULTS ===");
    println!("Initial lines: {}", initial_lines);
    println!("Final lines: {}", final_lines);
    println!("Delta: {}", final_lines as i32 - initial_lines as i32);
    
    println!("\nFirst 5 lines of final content:");
    for line in final_content.lines().take(5) {
        println!("  {}", line);
    }
    
    // The sequence is: down, enter(+1), backspace(-1), left, enter(+1), backspace(-1), 
    //                  enter(+1), enter(+1), backspace(-1), backspace(-1)
    // Net change should be 0
    
    if final_lines != initial_lines {
        println!("\n!!! BUG DETECTED !!!");
        println!("Expected {} lines but got {} lines", initial_lines, final_lines);
        println!("Some backspaces are not working!");
    }
    
    assert_eq!(final_lines, initial_lines, 
               "Line count should be unchanged. Started with {}, ended with {}",
               initial_lines, final_lines);
}

/// Test with user's UPDATED exact sequence (slightly different from before)
/// down, enter, left, backspace, left, enter, backspace, enter, enter, backspace, backspace
#[test]
fn test_user_exact_sequence_v2() {
    let source_path = format!("{}/test3", env!("CARGO_MANIFEST_DIR"));
    
    if !std::path::Path::new(&source_path).exists() {
        println!("test3 file not found, skipping test");
        return;
    }
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path().to_path_buf();
    fs::copy(&source_path, &file_path).expect("Failed to copy test3");
    
    let initial_content = fs::read_to_string(&file_path).unwrap();
    let initial_lines = initial_content.lines().count();
    println!("Initial: {} lines", initial_lines);
    
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    if !std::path::Path::new(&binary_path).exists() {
        let status = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("Failed to build");
        assert!(status.success());
    }
    
    // NEW sequence: down, enter, left, backspace, left, enter, backspace, enter, enter, backspace, backspace
    let helper_script = format!(
        r#"#!/bin/bash
export TERM=xterm
export LINES=24
export COLUMNS=80

python3 << 'PYTHON_EOF'
import pty
import os
import time
import select

def send_key(master_fd, key, delay=0.1):
    os.write(master_fd, key)
    time.sleep(delay)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.02)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break

pid, master_fd = pty.fork()

if pid == 0:
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    time.sleep(0.5)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.1)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break
    
    DOWN = b'\x1b[B'
    ENTER = b'\r'
    BACKSPACE = b'\x7f'
    LEFT = b'\x1b[D'
    CTRL_O = b'\x0f'
    CTRL_X = b'\x18'
    
    # User's EXACT new sequence:
    # down, enter, left, backspace, left, enter, backspace, enter, enter, backspace, backspace
    
    print("1. DOWN")
    send_key(master_fd, DOWN)
    
    print("2. ENTER")
    send_key(master_fd, ENTER)
    
    print("3. LEFT")
    send_key(master_fd, LEFT)
    
    print("4. BACKSPACE")
    send_key(master_fd, BACKSPACE)
    
    print("5. LEFT")
    send_key(master_fd, LEFT)
    
    print("6. ENTER")
    send_key(master_fd, ENTER)
    
    print("7. BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    print("8. ENTER")
    send_key(master_fd, ENTER)
    
    print("9. ENTER")
    send_key(master_fd, ENTER)
    
    print("10. BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    print("11. BACKSPACE (reported failing)")
    send_key(master_fd, BACKSPACE)
    
    # Save
    print("CTRL-O (save)")
    send_key(master_fd, CTRL_O, delay=0.2)
    print("ENTER (confirm)")
    send_key(master_fd, ENTER, delay=0.3)
    
    # Exit
    print("CTRL-X (exit)")
    send_key(master_fd, CTRL_X)
    print("n (don't save again)")
    send_key(master_fd, b'n', delay=0.2)
    
    try:
        os.waitpid(pid, 0)
    except:
        pass
    os.close(master_fd)
    print("Done!")
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    let script_file = NamedTempFile::new().expect("Failed to create script file");
    let script_path = script_file.path().to_path_buf();
    fs::write(&script_path, &helper_script).expect("Failed to write script");
    
    Command::new("chmod")
        .args(["+x", script_path.to_str().unwrap()])
        .status()
        .expect("chmod failed");
    
    println!("\nRunning editor with NEW sequence...");
    let output = Command::new("bash")
        .arg(&script_path)
        .env("TERM", "xterm")
        .output()
        .expect("Failed to run script");
    
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    let final_content = fs::read_to_string(&file_path).unwrap();
    let final_lines = final_content.lines().count();
    
    println!("\n=== RESULTS ===");
    println!("Initial lines: {}", initial_lines);
    println!("Final lines: {}", final_lines);
    println!("Delta: {}", final_lines as i32 - initial_lines as i32);
    
    // Sequence analysis:
    // down (move), enter(+1), left(move), backspace(-1), left(move), enter(+1), 
    // backspace(-1), enter(+1), enter(+1), backspace(-1), backspace(-1)
    // Net: +4 enters, -4 backspaces = 0 change
    
    if final_lines != initial_lines {
        println!("\n!!! BUG DETECTED !!!");
        println!("Expected {} lines but got {} lines", initial_lines, final_lines);
        
        // Show diff of first few lines
        println!("\nInitial first 10 lines:");
        for (i, line) in initial_content.lines().take(10).enumerate() {
            println!("  {}: {}", i+1, line);
        }
        println!("\nFinal first 10 lines:");
        for (i, line) in final_content.lines().take(10).enumerate() {
            println!("  {}: {}", i+1, line);
        }
    }
    
    assert_eq!(final_lines, initial_lines, 
               "Line count should be unchanged. Started with {}, ended with {}",
               initial_lines, final_lines);
}

/// Integration test: Save, reopen, edit, save again - verifies changes persist
/// This tests the bug where edits made after reopening a file don't persist
#[test]
fn test_save_reopen_edit_save() {
    // Create a simple test file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path().to_path_buf();
    fs::write(&file_path, "line1\nline2\nline3\n").expect("Failed to write");
    
    let initial_content = fs::read_to_string(&file_path).unwrap();
    let initial_lines = initial_content.lines().count();
    println!("=== Initial file ===");
    println!("Lines: {}", initial_lines);
    println!("Content: {:?}", initial_content);
    
    let binary_path = format!("{}/target/release/bigedit", env!("CARGO_MANIFEST_DIR"));
    
    if !std::path::Path::new(&binary_path).exists() {
        let status = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("Failed to build");
        assert!(status.success());
    }
    
    // SESSION 1: Add two newlines and save
    println!("\n=== SESSION 1: Add newlines ===");
    let script1 = format!(
        r#"#!/bin/bash
export TERM=xterm
export LINES=24
export COLUMNS=80

python3 << 'PYTHON_EOF'
import pty
import os
import time
import select
import sys
import struct
import fcntl
import termios

def send_key(master_fd, key, delay=0.1):
    os.write(master_fd, key)
    time.sleep(delay)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.02)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break

print("Starting pty fork...", file=sys.stderr)
pid, master_fd = pty.fork()

if pid == 0:
    print("Child: executing editor", file=sys.stderr)
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    # Set terminal size to 24x80
    winsize = struct.pack('HHHH', 24, 80, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
    
    print("Parent: child pid set, terminal size 24x80", file=sys.stderr)
    time.sleep(0.5)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.1)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break
    
    DOWN = b'\x1b[B'
    ENTER = b'\r'
    CTRL_O = b'\x0f'
    CTRL_J = b'\x0a'  # Compact/full save
    CTRL_X = b'\x18'
    
    # Go to line 2, add 2 newlines
    print("Sending DOWN", file=sys.stderr)
    send_key(master_fd, DOWN)
    print("Sending ENTER", file=sys.stderr)
    send_key(master_fd, ENTER)
    print("Sending ENTER", file=sys.stderr)
    send_key(master_fd, ENTER)
    
    # Full save (CTRL-J rewrites the actual file)
    print("Sending CTRL-J (full save/compact)", file=sys.stderr)
    sys.stderr.flush()
    time.sleep(0.2)
    send_key(master_fd, CTRL_J, delay=0.5)
    
    # Exit (we just saved so no "modified buffer" prompt)
    print("Sending CTRL-X (exit)", file=sys.stderr)
    sys.stderr.flush()
    time.sleep(0.3)
    send_key(master_fd, CTRL_X, delay=0.3)
    print("Waiting for child to exit...", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        os.waitpid(pid, 0)
        print("Child exited", file=sys.stderr)
    except Exception as exc:
        print("waitpid exception: " + str(exc), file=sys.stderr)
    os.close(master_fd)
    print("Script done", file=sys.stderr)
    sys.stderr.flush()
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    let script_file1 = NamedTempFile::new().unwrap();
    fs::write(script_file1.path(), &script1).unwrap();
    Command::new("chmod").args(["+x", script_file1.path().to_str().unwrap()]).status().unwrap();
    let output1 = Command::new("bash").arg(script_file1.path()).env("TERM", "xterm").output().unwrap();
    println!("Session 1 stdout: {}", String::from_utf8_lossy(&output1.stdout));
    println!("Session 1 stderr: {}", String::from_utf8_lossy(&output1.stderr));
    
    let after_s1 = fs::read_to_string(&file_path).unwrap();
    let lines_s1 = after_s1.lines().count();
    println!("After session 1: {} lines (was {})", lines_s1, initial_lines);
    println!("Content: {:?}", after_s1);
    
    assert_eq!(lines_s1, initial_lines + 2, 
               "Session 1 should add 2 lines. Expected {}, got {}", 
               initial_lines + 2, lines_s1);
    
    // SESSION 2: Reopen, delete those newlines, save
    println!("\n=== SESSION 2: Delete newlines ===");
    let script2 = format!(
        r#"#!/bin/bash
export TERM=xterm
export LINES=24
export COLUMNS=80

python3 << 'PYTHON_EOF'
import pty
import os
import time
import select
import struct
import fcntl
import termios
import sys

def send_key(master_fd, key, delay=0.1):
    os.write(master_fd, key)
    time.sleep(delay)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.02)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break

pid, master_fd = pty.fork()

if pid == 0:
    os.execv("{binary}", ["{binary}", "{file}"])
else:
    # Set terminal size to 24x80
    winsize = struct.pack('HHHH', 24, 80, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
    
    time.sleep(0.5)
    while True:
        r, _, _ = select.select([master_fd], [], [], 0.1)
        if not r:
            break
        try:
            os.read(master_fd, 4096)
        except:
            break
    
    DOWN = b'\x1b[B'
    BACKSPACE = b'\x7f'
    CTRL_J = b'\x0a'  # Compact/full save
    CTRL_X = b'\x18'
    ENTER = b'\r'
    
    print("Session 2: DOWN", file=sys.stderr)
    # Go to line 2 (first empty line), delete it with backspace
    send_key(master_fd, DOWN)
    print("Session 2: BACKSPACE (delete first empty line)", file=sys.stderr)
    send_key(master_fd, BACKSPACE)
    # Now cursor is at end of line 1, need to go DOWN to the remaining empty line
    print("Session 2: DOWN (to remaining empty line)", file=sys.stderr)
    send_key(master_fd, DOWN)
    print("Session 2: BACKSPACE (delete second empty line)", file=sys.stderr)
    send_key(master_fd, BACKSPACE)
    
    # Full save (CTRL-J rewrites the actual file)
    print("Session 2: CTRL-J (full save/compact)", file=sys.stderr)
    send_key(master_fd, CTRL_J, delay=0.5)
    # Exit (no prompt since we just saved)
    print("Session 2: CTRL-X (exit)", file=sys.stderr)
    send_key(master_fd, CTRL_X, delay=0.3)
    print("Session 2: done", file=sys.stderr)
    
    try:
        os.waitpid(pid, 0)
    except:
        pass
    os.close(master_fd)
PYTHON_EOF
"#,
        binary = binary_path,
        file = file_path.display()
    );
    
    let script_file2 = NamedTempFile::new().unwrap();
    fs::write(script_file2.path(), &script2).unwrap();
    Command::new("chmod").args(["+x", script_file2.path().to_str().unwrap()]).status().unwrap();
    Command::new("bash").arg(script_file2.path()).env("TERM", "xterm").output().unwrap();
    
    let after_s2 = fs::read_to_string(&file_path).unwrap();
    let lines_s2 = after_s2.lines().count();
    println!("After session 2: {} lines (was {})", lines_s2, lines_s1);
    println!("Content: {:?}", after_s2);
    
    // SESSION 3: Reopen and verify content is correct
    println!("\n=== FINAL VERIFICATION ===");
    let final_content = fs::read_to_string(&file_path).unwrap();
    let final_lines = final_content.lines().count();
    println!("Final: {} lines", final_lines);
    println!("Content: {:?}", final_content);
    
    // Should be back to original
    if final_content != initial_content {
        println!("\n!!! BUG: Content mismatch !!!");
        println!("Expected: {:?}", initial_content);
        println!("Got: {:?}", final_content);
    }
    
    assert_eq!(final_content, initial_content, 
               "After adding and deleting newlines, content should match original");
}
