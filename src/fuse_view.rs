//! FUSE-based virtual file view
//!
//! This module provides a FUSE filesystem that exposes a patched view of the file.
//! Other programs (like `less`, `cat`, etc.) can read the file with all patches applied
//! without actually rewriting the original file.
//!
//! The mount point is a directory containing a single file with the patched content.

use crate::patches::PatchList;
use crate::journal;

use anyhow::{Context, Result};
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry,
    ReplyOpen, Request,
};
use libc::ENOENT;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Inode numbers
const ROOT_INO: u64 = 1;
const FILE_INO: u64 = 2;

/// TTL for attributes
const TTL: Duration = Duration::from_secs(1);

/// State shared between FUSE filesystem and main editor
pub struct FuseState {
    /// Path to the original file
    pub original_path: PathBuf,
    /// Current patches (updated on each save)
    pub patches: PatchList,
    /// Computed file size with patches applied
    pub virtual_size: u64,
    /// Original file size
    pub original_size: u64,
    /// Filename to expose
    pub filename: String,
}

impl FuseState {
    pub fn new(original_path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(original_path)?;
        let original_size = metadata.len();
        let filename = original_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        // Load patches from journal if it exists
        let patches = journal::load_from_journal(original_path)?
            .unwrap_or_else(PatchList::new);
        
        let virtual_size = compute_virtual_size(original_size, &patches);

        Ok(Self {
            original_path: original_path.to_path_buf(),
            patches,
            virtual_size,
            original_size,
            filename,
        })
    }

    /// Update patches and recompute virtual size
    pub fn update_patches(&mut self, patches: &PatchList) {
        self.patches = patches.clone();
        self.virtual_size = compute_virtual_size(self.original_size, &self.patches);
    }

    /// Reload patches from journal
    pub fn reload_from_journal(&mut self) -> Result<()> {
        if let Some(patches) = journal::load_from_journal(&self.original_path)? {
            self.patches = patches;
        } else {
            self.patches = PatchList::new();
        }
        self.virtual_size = compute_virtual_size(self.original_size, &self.patches);
        Ok(())
    }
}

/// Compute the virtual file size after applying patches
fn compute_virtual_size(original_size: u64, patches: &PatchList) -> u64 {
    let mut size = original_size as i64;
    for patch in patches.patches() {
        let removed = (patch.end - patch.start) as i64;
        let added = patch.replacement.len() as i64;
        size += added - removed;
    }
    size.max(0) as u64
}

/// FUSE filesystem that exposes a single patched file
struct PatchedFileFS {
    state: Arc<Mutex<FuseState>>,
}

impl PatchedFileFS {
    fn new(state: Arc<Mutex<FuseState>>) -> Self {
        Self { state }
    }

    fn file_attr(&self) -> FileAttr {
        let state = self.state.lock().unwrap();
        FileAttr {
            ino: FILE_INO,
            size: state.virtual_size,
            blocks: (state.virtual_size + 511) / 512,
            atime: SystemTime::now(),
            mtime: SystemTime::now(),
            ctime: SystemTime::now(),
            crtime: UNIX_EPOCH,
            kind: FileType::RegularFile,
            perm: 0o444, // Read-only
            nlink: 1,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            blksize: 4096,
            flags: 0,
        }
    }

    fn root_attr(&self) -> FileAttr {
        FileAttr {
            ino: ROOT_INO,
            size: 0,
            blocks: 0,
            atime: SystemTime::now(),
            mtime: SystemTime::now(),
            ctime: SystemTime::now(),
            crtime: UNIX_EPOCH,
            kind: FileType::Directory,
            perm: 0o555,
            nlink: 2,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            blksize: 4096,
            flags: 0,
        }
    }
}

impl Filesystem for PatchedFileFS {
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        if parent != ROOT_INO {
            reply.error(ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        if name.to_str() == Some(&state.filename) {
            drop(state);
            reply.entry(&TTL, &self.file_attr(), 0);
        } else {
            reply.error(ENOENT);
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyAttr) {
        match ino {
            ROOT_INO => reply.attr(&TTL, &self.root_attr()),
            FILE_INO => reply.attr(&TTL, &self.file_attr()),
            _ => reply.error(ENOENT),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        if ino == FILE_INO {
            reply.opened(0, 0);
        } else {
            reply.error(ENOENT);
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        if ino != FILE_INO {
            reply.error(ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        let offset = offset as u64;
        let size = size as usize;

        // Read patched content
        match read_patched_range(&state.original_path, &state.patches, offset, size) {
            Ok(data) => reply.data(&data),
            Err(_) => reply.error(libc::EIO),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        if ino != ROOT_INO {
            reply.error(ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        let entries = vec![
            (ROOT_INO, FileType::Directory, "."),
            (ROOT_INO, FileType::Directory, ".."),
            (FILE_INO, FileType::RegularFile, &state.filename),
        ];

        for (i, (ino, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            if reply.add(ino, (i + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn opendir(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        if ino == ROOT_INO {
            reply.opened(0, 0);
        } else {
            reply.error(ENOENT);
        }
    }
}

/// Read a range from the patched file view
fn read_patched_range(
    original_path: &Path,
    patches: &PatchList,
    offset: u64,
    size: usize,
) -> Result<Vec<u8>> {
    let mut file = File::open(original_path)?;
    let file_size = file.metadata()?.len();

    // For simplicity, we'll read the entire relevant portion and apply patches
    // This could be optimized for very large files
    let mut result = Vec::with_capacity(size);
    let mut virtual_pos: u64 = 0;
    let mut file_pos: u64 = 0;

    // Sort patches by start position
    let mut sorted_patches: Vec<_> = patches.patches().to_vec();
    sorted_patches.sort_by_key(|p| p.start);

    for patch in &sorted_patches {
        // Read original bytes before this patch
        if file_pos < patch.start && virtual_pos < offset + size as u64 {
            let original_bytes_before = patch.start - file_pos;
            
            // Calculate how much of this we need
            let start_in_segment = if virtual_pos < offset {
                (offset - virtual_pos).min(original_bytes_before)
            } else {
                0
            };
            
            let end_in_segment = original_bytes_before.min(
                if virtual_pos + original_bytes_before > offset {
                    (offset + size as u64 - virtual_pos).min(original_bytes_before)
                } else {
                    0
                }
            );

            if start_in_segment < end_in_segment && virtual_pos + end_in_segment > offset {
                let read_start = file_pos + start_in_segment;
                let read_len = (end_in_segment - start_in_segment) as usize;
                
                file.seek(SeekFrom::Start(read_start))?;
                let mut buf = vec![0u8; read_len];
                file.read_exact(&mut buf)?;
                result.extend_from_slice(&buf);
            }

            virtual_pos += original_bytes_before;
            file_pos = patch.start;
        }

        // Handle the patch replacement
        let patch_len = patch.replacement.len() as u64;
        if patch_len > 0 && virtual_pos < offset + size as u64 && virtual_pos + patch_len > offset {
            let start_in_patch = if virtual_pos < offset {
                (offset - virtual_pos) as usize
            } else {
                0
            };
            let end_in_patch = patch_len.min(offset + size as u64 - virtual_pos) as usize;

            if start_in_patch < end_in_patch {
                result.extend_from_slice(&patch.replacement[start_in_patch..end_in_patch]);
            }
        }

        virtual_pos += patch_len;
        file_pos = patch.end;
    }

    // Read remaining original bytes after all patches
    if file_pos < file_size && virtual_pos < offset + size as u64 {
        let remaining = file_size - file_pos;
        
        let start_in_segment = if virtual_pos < offset {
            (offset - virtual_pos).min(remaining)
        } else {
            0
        };
        
        let bytes_needed = (size - result.len()) as u64;
        let end_in_segment = remaining.min(start_in_segment + bytes_needed);

        if start_in_segment < end_in_segment {
            let read_start = file_pos + start_in_segment;
            let read_len = (end_in_segment - start_in_segment) as usize;
            
            file.seek(SeekFrom::Start(read_start))?;
            let mut buf = vec![0u8; read_len];
            let bytes_read = file.read(&mut buf)?;
            buf.truncate(bytes_read);
            result.extend_from_slice(&buf);
        }
    }

    Ok(result)
}

/// Get the path to the PID file for a FUSE daemon
fn pid_file_path(file_path: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let filename = file_path.file_name().unwrap_or_default().to_string_lossy();
    parent.join(format!(".{}.bigedit-fuse.pid", filename))
}

/// Get the mount point path for a file
fn mount_point_path(file_path: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let base_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    parent.join(format!(".{}.view", base_name))
}

/// Get the symlink path for a file (the user-friendly .edited file)
fn symlink_path(file_path: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let base_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    parent.join(format!("{}.edited", base_name))
}

/// Handle to a running FUSE daemon
pub struct FuseMount {
    /// Original file path
    file_path: PathBuf,
    /// Mount point directory
    mount_point: PathBuf,
    /// Path to the virtual file (the .edited symlink)
    virtual_file_path: PathBuf,
    /// PID of the daemon process (if spawned by us)
    daemon_pid: Option<u32>,
}

impl FuseMount {
    /// Create and mount a FUSE filesystem for the given file using a daemon process
    pub fn new(original_path: &Path) -> Result<Self> {
        let file_path = original_path.canonicalize()
            .context("Failed to get absolute path")?;

        let mount_point = mount_point_path(&file_path);
        let virtual_file_path = symlink_path(&file_path);
        let pid_file = pid_file_path(&file_path);

        // Check if daemon is already running
        if pid_file.exists() {
            if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = pid_str.trim().parse::<i32>() {
                    // Check if process is still running
                    if unsafe { libc::kill(pid, 0) } == 0 {
                        // Daemon is already running, just return a handle
                        return Ok(Self {
                            file_path,
                            mount_point,
                            virtual_file_path,
                            daemon_pid: None, // We didn't spawn it
                        });
                    }
                }
            }
            // Stale PID file, remove it
            let _ = std::fs::remove_file(&pid_file);
        }

        // Find the bigedit-fuse binary
        let daemon_path = Self::find_daemon_binary()?;

        // Spawn the daemon process
        let child = std::process::Command::new(&daemon_path)
            .arg(&file_path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn bigedit-fuse daemon")?;

        let daemon_pid = child.id();

        // Wait a bit for the daemon to start and mount
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify the mount point exists
        if !mount_point.exists() {
            anyhow::bail!("FUSE daemon failed to create mount point");
        }

        Ok(Self {
            file_path,
            mount_point,
            virtual_file_path,
            daemon_pid: Some(daemon_pid),
        })
    }

    /// Find the bigedit-fuse daemon binary
    fn find_daemon_binary() -> Result<PathBuf> {
        // First, check next to the current executable
        if let Ok(current_exe) = std::env::current_exe() {
            let daemon_path = current_exe.parent()
                .map(|p| p.join("bigedit-fuse"));
            if let Some(path) = daemon_path {
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        // Check in PATH
        if let Ok(output) = std::process::Command::new("which")
            .arg("bigedit-fuse")
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout);
                return Ok(PathBuf::from(path.trim()));
            }
        }

        anyhow::bail!("bigedit-fuse daemon not found. Make sure it's installed or in the same directory as bigedit.")
    }

    /// Check if a FUSE mount is already running for a file
    pub fn is_running(file_path: &Path) -> bool {
        let pid_file = pid_file_path(file_path);
        if !pid_file.exists() {
            return false;
        }
        
        if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
            if let Ok(pid) = pid_str.trim().parse::<i32>() {
                return unsafe { libc::kill(pid, 0) } == 0;
            }
        }
        false
    }

    /// Get the path to the virtual patched file
    pub fn virtual_path(&self) -> &Path {
        &self.virtual_file_path
    }

    /// Get the mount point directory
    pub fn mount_point(&self) -> &Path {
        &self.mount_point
    }

    /// Kill the daemon and unmount the filesystem
    pub fn unmount(self) -> Result<()> {
        let pid_file = pid_file_path(&self.file_path);
        
        if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
            if let Ok(pid) = pid_str.trim().parse::<i32>() {
                // Send SIGTERM to the daemon
                unsafe { libc::kill(pid, libc::SIGTERM); }
                
                // Wait a bit for it to clean up
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        
        // Clean up PID file if it still exists
        let _ = std::fs::remove_file(&pid_file);
        
        // Clean up symlink
        let symlink = symlink_path(&self.file_path);
        let _ = std::fs::remove_file(&symlink);
        
        // Try to unmount and remove mount point
        let _ = std::process::Command::new("fusermount3")
            .args(["-u", &self.mount_point.display().to_string()])
            .output();
        let _ = std::fs::remove_dir(&self.mount_point);
        
        Ok(())
    }
}

// Note: We don't implement Drop to kill the daemon automatically
// because we want the mount to persist after the editor exits

/// Kill any running FUSE daemon for a file (used when compacting)
pub fn kill_fuse_daemon(file_path: &Path) -> Result<()> {
    let pid_file = pid_file_path(file_path);
    let mount_point = mount_point_path(file_path);
    let symlink = symlink_path(file_path);
    
    if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
        if let Ok(pid) = pid_str.trim().parse::<i32>() {
            unsafe { libc::kill(pid, libc::SIGTERM); }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&symlink);
    let _ = std::process::Command::new("fusermount3")
        .args(["-u", &mount_point.display().to_string()])
        .output();
    let _ = std::fs::remove_dir(&mount_point);
    
    Ok(())
}

/// Check if FUSE is available on this system
pub fn is_fuse_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // On Linux, check if /dev/fuse exists
        Path::new("/dev/fuse").exists()
    }
    
    #[cfg(target_os = "macos")]
    {
        // On macOS, check if macFUSE is installed
        Path::new("/Library/Filesystems/macfuse.fs").exists() ||
        Path::new("/usr/local/lib/libfuse.dylib").exists() ||
        Path::new("/opt/homebrew/lib/libfuse.dylib").exists()
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_virtual_size() {
        let mut patches = PatchList::new();
        
        // Original size 100
        assert_eq!(compute_virtual_size(100, &patches), 100);
        
        // Insert 10 bytes
        patches.insert(50, b"0123456789");
        assert_eq!(compute_virtual_size(100, &patches), 110);
    }

    #[test]
    fn test_read_patched_range_no_patches() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();
        file.flush().unwrap();

        let patches = PatchList::new();
        let data = read_patched_range(file.path(), &patches, 0, 13).unwrap();
        assert_eq!(data, b"Hello, World!");
    }

    #[test]
    fn test_read_patched_range_with_insertion() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello World!").unwrap();
        file.flush().unwrap();

        let mut patches = PatchList::new();
        patches.insert(5, b", Beautiful");

        let data = read_patched_range(file.path(), &patches, 0, 50).unwrap();
        assert_eq!(String::from_utf8_lossy(&data), "Hello, Beautiful World!");
    }

    #[test]
    fn test_fuse_mount_creation() {
        // Create a test file
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Original content here\n").unwrap();
        file.flush().unwrap();
        
        let file_path = file.path();
        println!("Test file: {:?}", file_path);
        
        // Try to create FuseState
        match FuseState::new(file_path) {
            Ok(state) => {
                println!("FuseState created successfully");
                println!("  Original size: {}", state.original_size);
                println!("  Virtual size: {}", state.virtual_size);
                println!("  Filename: {}", state.filename);
            }
            Err(e) => {
                println!("FuseState creation failed: {}", e);
            }
        }
        
        // Try to create FuseMount
        println!("\nAttempting to create FuseMount...");
        match FuseMount::new(file_path) {
            Ok(mount) => {
                println!("FuseMount created successfully!");
                println!("  Mount point: {:?}", mount.mount_point);
                println!("  Virtual file: {:?}", mount.virtual_file_path);
                
                // Check if mount point exists
                println!("  Mount point exists: {}", mount.mount_point.exists());
                
                // Try to list the mount point
                match std::fs::read_dir(&mount.mount_point) {
                    Ok(entries) => {
                        println!("  Contents:");
                        for entry in entries {
                            if let Ok(e) = entry {
                                println!("    - {:?}", e.file_name());
                            }
                        }
                    }
                    Err(e) => println!("  Failed to read mount point: {}", e),
                }
                
                // Try to read the virtual file
                match std::fs::read(&mount.virtual_file_path) {
                    Ok(content) => {
                        println!("  Virtual file content: {:?}", String::from_utf8_lossy(&content));
                    }
                    Err(e) => println!("  Failed to read virtual file: {}", e),
                }
                
                // Cleanup
                drop(mount);
                println!("Mount dropped (should unmount)");
            }
            Err(e) => {
                println!("FuseMount creation FAILED: {}", e);
                println!("Error details: {:?}", e);
            }
        }
    }
}
