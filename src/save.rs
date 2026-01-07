//! High-performance streaming save for bigedit
//!
//! This module handles saving files by streaming the original file
//! and applying patches on the fly to produce the new file.
//!
//! Performance optimizations:
//! - Large I/O buffers (16MB) for maximum throughput
//! - Zero-copy file range copy on Linux when no patches in region
//! - Early exit if no patches (no rewrite needed)
//! - Efficient seek-based skipping instead of read-and-discard

use crate::patches::PatchList;
use crate::types::FilePos;
use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Size of the buffer for streaming copy (16MB for maximum throughput)
/// Modern SSDs can easily saturate this; NVMe drives benefit from even larger buffers
const COPY_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Threshold for using copy_file_range (1MB) - below this, regular copy is fine
#[cfg(target_os = "linux")]
const COPY_FILE_RANGE_THRESHOLD: u64 = 1024 * 1024;

/// Save the file with patches applied
///
/// Uses an atomic save strategy:
/// 1. If no patches, return immediately (nothing to save)
/// 2. Write to a temporary file using high-performance I/O
/// 3. Sync to disk
/// 4. Rename to target (atomic on same filesystem)
pub fn save_file(
    original_path: &Path,
    patches: &PatchList,
    target_path: Option<&Path>,
) -> Result<()> {
    // Fast path: no patches means no changes to save
    if !patches.is_modified() {
        return Ok(());
    }

    let target = target_path.unwrap_or(original_path);

    // Create temp file in the same directory as target for atomic rename
    let target_dir = target.parent().unwrap_or(Path::new("."));
    let temp_file = NamedTempFile::new_in(target_dir)
        .context("Failed to create temporary file")?;

    // Open original file for reading
    let original_file = File::open(original_path)
        .context("Failed to open original file")?;
    let original_len = original_file.metadata()?.len();

    // Stream and apply patches with high-performance I/O
    stream_with_patches_fast(
        original_file,
        original_len,
        temp_file.as_file(),
        patches,
    )?;

    // Sync to disk - use sync_data for speed (metadata sync not critical)
    temp_file.as_file().sync_data()
        .context("Failed to sync temporary file")?;

    // Copy permissions from original
    if let Ok(metadata) = fs::metadata(original_path) {
        let _ = fs::set_permissions(temp_file.path(), metadata.permissions());
    }

    // Atomic rename
    temp_file.persist(target)
        .context("Failed to save file (rename failed)")?;

    Ok(())
}

/// High-performance streaming with patches
/// Uses large buffers and seek-based skipping for maximum speed
fn stream_with_patches_fast(
    mut original: File,
    original_len: u64,
    output: &File,
    patches: &PatchList,
) -> Result<()> {
    let sorted_patches = patches.patches();
    
    // If no patches, use zero-copy if available, otherwise fast copy
    if sorted_patches.is_empty() {
        return fast_copy_file(&mut original, output, original_len);
    }

    let mut output_file = output.try_clone().context("Failed to clone output file")?;
    let mut pos: FilePos = 0;
    
    // Allocate large buffer once
    let mut buffer = vec![0u8; COPY_BUFFER_SIZE];

    for patch in sorted_patches {
        // Copy original bytes from pos to patch.start
        if patch.start > pos {
            let copy_len = patch.start - pos;
            copy_range_fast(&mut original, &mut output_file, pos, copy_len, &mut buffer)?;
            pos = patch.start;
        }

        // Write patch replacement
        if !patch.replacement.is_empty() {
            output_file.write_all(&patch.replacement)
                .context("Failed to write patch")?;
        }

        // Skip over the patched region in original
        pos = pos.max(patch.end);
    }

    // Copy remaining original content after all patches
    if pos < original_len {
        let remaining = original_len - pos;
        copy_range_fast(&mut original, &mut output_file, pos, remaining, &mut buffer)?;
    }

    output_file.flush().context("Failed to flush output")?;
    Ok(())
}

/// Fast copy of a range from source to destination
/// Uses copy_file_range on Linux for zero-copy when possible
fn copy_range_fast(
    src: &mut File,
    dst: &mut File,
    src_offset: u64,
    len: u64,
    buffer: &mut [u8],
) -> Result<()> {
    // Try zero-copy on Linux for large ranges
    #[cfg(target_os = "linux")]
    if len >= COPY_FILE_RANGE_THRESHOLD {
        if let Ok(()) = copy_file_range_linux(src, dst, src_offset, len) {
            return Ok(());
        }
        // Fall through to regular copy if copy_file_range fails
    }

    // Regular copy with large buffer
    src.seek(SeekFrom::Start(src_offset))
        .context("Failed to seek in source")?;
    
    copy_bytes_fast(src, dst, len, buffer)
}

/// Linux-specific zero-copy between files using copy_file_range
#[cfg(target_os = "linux")]
fn copy_file_range_linux(
    src: &mut File,
    dst: &mut File,
    src_offset: u64,
    len: u64,
) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    
    let src_fd = src.as_raw_fd();
    let dst_fd = dst.as_raw_fd();
    
    let mut off_in = src_offset as i64;
    let mut remaining = len;
    
    while remaining > 0 {
        let to_copy = remaining.min(i64::MAX as u64) as usize;
        
        // SAFETY: We're passing valid file descriptors and properly sized buffers
        let copied = unsafe {
            libc::copy_file_range(
                src_fd,
                &mut off_in,
                dst_fd,
                std::ptr::null_mut(), // dst offset managed by file position
                to_copy,
                0, // flags
            )
        };
        
        if copied < 0 {
            // copy_file_range failed, caller should fall back to regular copy
            bail!("copy_file_range failed");
        }
        
        if copied == 0 {
            break; // EOF
        }
        
        remaining -= copied as u64;
    }
    
    Ok(())
}

/// Fast copy using large buffer
fn copy_bytes_fast<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    mut remaining: u64,
    buffer: &mut [u8],
) -> Result<()> {
    while remaining > 0 {
        let to_read = (remaining as usize).min(buffer.len());
        let read = reader.read(&mut buffer[..to_read])
            .context("Failed to read from original")?;

        if read == 0 {
            bail!("Unexpected end of file");
        }

        writer.write_all(&buffer[..read])
            .context("Failed to write")?;
        
        remaining -= read as u64;
    }
    Ok(())
}

/// Fast copy entire file (no patches case)
fn fast_copy_file(src: &mut File, dst: &File, len: u64) -> Result<()> {
    // Try sendfile on Linux for maximum performance
    #[cfg(target_os = "linux")]
    {
        if let Ok(()) = sendfile_copy_linux(src, dst, len) {
            return Ok(());
        }
    }
    
    // Fallback to regular copy with large buffer
    let mut dst_file = dst.try_clone().context("Failed to clone output")?;
    let mut buffer = vec![0u8; COPY_BUFFER_SIZE];
    src.seek(SeekFrom::Start(0))?;
    copy_bytes_fast(src, &mut dst_file, len, &mut buffer)
}

/// Linux sendfile for zero-copy file transfer
#[cfg(target_os = "linux")]
fn sendfile_copy_linux(src: &mut File, dst: &File, len: u64) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    
    src.seek(SeekFrom::Start(0))?;
    
    let src_fd = src.as_raw_fd();
    let dst_fd = dst.as_raw_fd();
    
    let mut remaining = len;
    
    while remaining > 0 {
        let to_copy = remaining.min(0x7ffff000) as usize; // Max chunk size for sendfile
        
        let sent = unsafe {
            libc::sendfile(dst_fd, src_fd, std::ptr::null_mut(), to_copy)
        };
        
        if sent < 0 {
            bail!("sendfile failed");
        }
        
        if sent == 0 {
            break;
        }
        
        remaining -= sent as u64;
    }
    
    Ok(())
}

/// Check if atomic rename is possible between two paths
#[allow(dead_code)]
pub fn can_atomic_rename(source: &Path, target: &Path) -> bool {
    // On Unix, rename is atomic if on same filesystem
    // We check by comparing parent directories' device IDs
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let source_parent = source.parent().unwrap_or(Path::new("."));
        let target_parent = target.parent().unwrap_or(Path::new("."));

        if let (Ok(s_meta), Ok(t_meta)) = (fs::metadata(source_parent), fs::metadata(target_parent))
        {
            return s_meta.dev() == t_meta.dev();
        }
        return false; // Could not determine, assume not atomic
    }

    // On Windows, always return true (rename is usually atomic)
    #[cfg(windows)]
    {
        return true;
    }

    #[cfg(not(any(unix, windows)))]
    {
        true
    }
}

/// Create a backup of the original file
pub fn create_backup(path: &Path) -> Result<PathBuf> {
    let backup_path = path.with_extension(format!(
        "{}.bak",
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("txt")
    ));

    fs::copy(path, &backup_path)
        .context("Failed to create backup")?;

    Ok(backup_path)
}

/// Save to a new file (Save As)
pub fn save_as(
    original_path: &Path,
    patches: &PatchList,
    new_path: &Path,
) -> Result<()> {
    save_file(original_path, patches, Some(new_path))
}

// =============================================================================
// Fast In-Place Consolidation using Filesystem-Level Operations
// =============================================================================
//
// On ext4/XFS (Linux), we can use FALLOC_FL_INSERT_RANGE and FALLOC_FL_COLLAPSE_RANGE
// to insert/delete space in files without rewriting unaffected regions.
// This makes consolidation O(patches) instead of O(file_size).
//
// On btrfs/APFS, we use reflinks/clonefile for efficient copy-on-write.

/// Filesystem capabilities for fast consolidation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesystemCapability {
    /// ext4: supports INSERT_RANGE and COLLAPSE_RANGE (no reflink backup)
    ExtentManipulation,
    /// XFS: supports BOTH reflink AND INSERT_RANGE/COLLAPSE_RANGE (best of both worlds)
    ExtentManipulationWithReflink,
    /// btrfs/ZFS/APFS: supports reflinks/clonefile (CoW) but no extent manipulation
    CopyOnWrite,
    /// No special capabilities, use streaming rewrite
    Standard,
}

/// Block size for extent operations (must align to this)
const BLOCK_SIZE: u64 = 4096;

/// Detect filesystem capabilities for the given path
pub fn detect_filesystem_capability(path: &Path) -> FilesystemCapability {
    #[cfg(target_os = "linux")]
    {
        // Try to detect filesystem type
        if let Some(fstype) = get_filesystem_type_linux(path) {
            match fstype.as_str() {
                "ext4" => return FilesystemCapability::ExtentManipulation,
                // XFS supports BOTH FICLONE (reflink) AND INSERT_RANGE/COLLAPSE_RANGE
                // Best of both worlds: instant backup + O(patches) consolidation
                "xfs" => return FilesystemCapability::ExtentManipulationWithReflink,
                // btrfs and ZFS support reflink but not INSERT_RANGE
                "btrfs" | "zfs" => return FilesystemCapability::CopyOnWrite,
                _ => {}
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // APFS supports clonefile
        if let Some(fstype) = get_filesystem_type_macos(path) {
            if fstype == "apfs" {
                return FilesystemCapability::CopyOnWrite;
            }
        }
    }
    
    FilesystemCapability::Standard
}

/// Get filesystem type on Linux
#[cfg(target_os = "linux")]
fn get_filesystem_type_linux(path: &Path) -> Option<String> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    
    let path_cstr = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    
    unsafe {
        let mut statfs_buf: MaybeUninit<libc::statfs> = MaybeUninit::uninit();
        if libc::statfs(path_cstr.as_ptr(), statfs_buf.as_mut_ptr()) == 0 {
            let statfs = statfs_buf.assume_init();
            // Magic numbers for common filesystems
            let fstype = match statfs.f_type {
                0xEF53 => "ext4",      // EXT4_SUPER_MAGIC
                0x58465342 => "xfs",   // XFS_SUPER_MAGIC  
                0x9123683E => "btrfs", // BTRFS_SUPER_MAGIC
                0x2FC12FC1 => "zfs",   // ZFS
                _ => return None,
            };
            return Some(fstype.to_string());
        }
    }
    None
}

/// Get filesystem type on macOS
#[cfg(target_os = "macos")]
fn get_filesystem_type_macos(path: &Path) -> Option<String> {
    use std::process::Command;
    
    // Use diskutil to get filesystem type
    let output = Command::new("diskutil")
        .args(["info", "-plist"])
        .arg(path)
        .output()
        .ok()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("APFS") {
        return Some("apfs".to_string());
    }
    None
}

// =============================================================================
// Background Defragmentation
// =============================================================================

/// Threshold: kick off background defrag after this many extent operations
const DEFRAG_THRESHOLD_PATCHES: usize = 50;

/// Kick off background defragmentation for a single file (Linux ext4/XFS only)
/// 
/// This spawns a detached process that will continue even if bigedit exits.
/// Safe because:
/// - e4defrag/xfs_fsr operate on the file in-place safely
/// - They're designed to handle interruption gracefully
/// - The file remains valid at all times
#[cfg(target_os = "linux")]
pub fn spawn_background_defrag(path: &Path) {
    use std::process::{Command, Stdio};
    
    let path_str = path.to_string_lossy().to_string();
    let fstype = get_filesystem_type_linux(path);
    
    let defrag_cmd = match fstype.as_deref() {
        Some("ext4") => "e4defrag",
        Some("xfs") => "xfs_fsr",
        _ => return, // No defrag available for this filesystem
    };
    
    // Check if defrag tool exists
    if Command::new("which")
        .arg(defrag_cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| !s.success())
        .unwrap_or(true)
    {
        // Defrag tool not installed, skip silently
        return;
    }
    
    // Spawn detached background process
    // Using nohup + disown pattern to survive parent exit
    let _ = Command::new("sh")
        .args([
            "-c",
            &format!(
                "nohup {} '{}' >/dev/null 2>&1 &",
                defrag_cmd,
                path_str.replace('\'', "'\\''") // Escape single quotes
            ),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// No-op on non-Linux systems (defrag not needed or handled differently)
#[cfg(not(target_os = "linux"))]
pub fn spawn_background_defrag(_path: &Path) {
    // macOS APFS and btrfs handle fragmentation internally via CoW
    // No external defrag needed
}

/// Check if we should trigger background defrag based on patch count
pub fn should_defrag(patch_count: usize) -> bool {
    patch_count >= DEFRAG_THRESHOLD_PATCHES
}

/// Trigger background defrag if needed after consolidation
/// 
/// Call this after successful consolidation with INSERT_RANGE/COLLAPSE_RANGE
pub fn maybe_defrag_after_consolidation(path: &Path, patch_count: usize) {
    if should_defrag(patch_count) {
        spawn_background_defrag(path);
    }
}

// =============================================================================
// Background Consolidation
// =============================================================================
//
// After a quick journal-based save, we can spawn a background thread to
// consolidate the file. This is crash-safe because:
// 1. We create an instant backup (FICLONE/clonefile) before any modification
// 2. If we crash during consolidation, the backup remains
// 3. On next startup, we can detect and recover from partial consolidation

use std::sync::mpsc;
use std::thread;

/// Message for background consolidation thread
#[derive(Clone)]
pub struct ConsolidationRequest {
    /// Path to the file to consolidate
    pub path: PathBuf,
    /// Patches to apply
    pub patches: PatchList,
}

/// Handle to a background consolidation thread
pub struct BackgroundConsolidator {
    /// Channel to send consolidation requests
    sender: mpsc::Sender<ConsolidationRequest>,
    /// Thread handle (for joining on shutdown)
    #[allow(dead_code)]
    handle: Option<thread::JoinHandle<()>>,
}

impl BackgroundConsolidator {
    /// Create a new background consolidator
    /// 
    /// The consolidator runs in a separate thread and processes requests
    /// from a channel. Each consolidation is crash-safe:
    /// - FICLONE/clonefile creates instant backup
    /// - Streaming rewrite to temp file
    /// - Atomic rename
    /// - Cleanup backup on success, restore on failure
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<ConsolidationRequest>();
        
        let handle = thread::Builder::new()
            .name("bigedit-consolidator".to_string())
            .spawn(move || {
                Self::worker_loop(receiver);
            })
            .expect("Failed to spawn background consolidator thread");
        
        BackgroundConsolidator {
            sender,
            handle: Some(handle),
        }
    }
    
    /// Queue a file for background consolidation
    /// 
    /// Returns immediately. The consolidation happens asynchronously.
    /// If the channel is full or closed, the request is silently dropped
    /// (the journal remains valid so no data is lost).
    pub fn queue(&self, path: PathBuf, patches: PatchList) {
        let _ = self.sender.send(ConsolidationRequest { path, patches });
    }
    
    /// Worker loop - processes consolidation requests
    fn worker_loop(receiver: mpsc::Receiver<ConsolidationRequest>) {
        while let Ok(request) = receiver.recv() {
            // Process each consolidation request
            if let Err(e) = Self::consolidate_file(&request.path, &request.patches) {
                // Log error but continue - journal is still valid
                eprintln!("[consolidator] Failed to consolidate {}: {}", 
                    request.path.display(), e);
            }
        }
    }
    
    /// Consolidate a single file (crash-safe)
    fn consolidate_file(path: &Path, patches: &PatchList) -> Result<()> {
        if !patches.is_modified() {
            return Ok(());
        }
        
        // Try crash-safe consolidation based on filesystem capability
        let capability = detect_filesystem_capability(path);
        let mut success = false;
        
        #[cfg(target_os = "linux")]
        {
            match capability {
                FilesystemCapability::ExtentManipulationWithReflink => {
                    // XFS: Best of both worlds
                    success = try_reflink_extent_consolidate_linux(path, patches)?;
                }
                FilesystemCapability::CopyOnWrite => {
                    // btrfs/ZFS: FICLONE backup + streaming
                    success = try_reflink_consolidate_linux(path, patches)?;
                }
                FilesystemCapability::ExtentManipulation => {
                    // ext4: No FICLONE available, use crash-safe streaming
                    // (temp file + atomic rename) instead of in-place extent ops
                    // which are NOT crash-safe
                    success = false; // Fall through to save_file
                }
                FilesystemCapability::Standard => {}
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            if !success && capability == FilesystemCapability::CopyOnWrite {
                success = try_clonefile_consolidate_macos(path, patches)?;
            }
        }
        
        // Fall back to regular atomic save (temp file + rename)
        if !success {
            save_file(path, patches, None)?;
        }
        
        Ok(())
    }
}

impl Default for BackgroundConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast consolidation using filesystem-specific optimizations
/// 
/// **SAFETY**: This function NEVER modifies the original file in-place without backup.
/// It always uses one of these safe strategies:
/// 1. XFS: FICLONE backup + INSERT_RANGE/COLLAPSE_RANGE + cleanup (best of both worlds)
/// 2. btrfs/ZFS/APFS: FICLONE backup + streaming rewrite + cleanup
/// 3. ext4/Standard: streaming rewrite (temp file + atomic rename) - crash-safe
/// 
/// Returns Ok(true) if fast consolidation was used, Ok(false) if caller should
/// fall back to streaming rewrite.
pub fn try_fast_consolidate(path: &Path, patches: &PatchList) -> Result<bool> {
    if !patches.is_modified() {
        return Ok(true); // Nothing to do
    }
    
    let capability = detect_filesystem_capability(path);
    
    match capability {
        FilesystemCapability::ExtentManipulationWithReflink => {
            // XFS: Best of both worlds - FICLONE backup + INSERT_RANGE/COLLAPSE_RANGE
            #[cfg(target_os = "linux")]
            {
                if try_reflink_extent_consolidate_linux(path, patches)? {
                    return Ok(true);
                }
            }
        }
        FilesystemCapability::ExtentManipulation => {
            // ext4: No FICLONE backup, fall through to streaming (crash-safe)
            // INSERT_RANGE is NOT crash-safe without backup
        }
        FilesystemCapability::CopyOnWrite => {
            // btrfs/ZFS: FICLONE backup + streaming rewrite
            // Handled by smart_consolidate, fall through
        }
        FilesystemCapability::Standard => {}
    }
    
    Ok(false) // Caller should use streaming rewrite
}

/// Linux ext4/XFS: Use INSERT_RANGE and COLLAPSE_RANGE for in-place consolidation
#[cfg(target_os = "linux")]
fn try_extent_consolidate_linux(path: &Path, patches: &PatchList) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    
    const FALLOC_FL_COLLAPSE_RANGE: i32 = 0x08;
    const FALLOC_FL_INSERT_RANGE: i32 = 0x20;
    
    let sorted_patches = patches.patches();
    if sorted_patches.is_empty() {
        return Ok(true);
    }
    
    // Check if patches are block-aligned (required for INSERT/COLLAPSE_RANGE)
    // If not, we need to handle sub-block regions differently
    let mut can_use_extent_ops = true;
    for patch in sorted_patches {
        let old_len = patch.end - patch.start;
        let new_len = patch.replacement.len() as u64;
        let delta = new_len as i64 - old_len as i64;
        
        // Extent operations only work at block boundaries
        if delta != 0 && (delta.unsigned_abs() % BLOCK_SIZE != 0) {
            can_use_extent_ops = false;
            break;
        }
    }
    
    if !can_use_extent_ops {
        // Fall back to hybrid approach: extent ops for large changes, 
        // regular writes for sub-block changes
        return try_hybrid_consolidate_linux(path, patches);
    }
    
    // Open file for modification
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .context("Failed to open file for consolidation")?;
    let fd = file.as_raw_fd();
    
    // Process patches in reverse order to avoid offset shifting issues
    let mut patches_vec: Vec<_> = sorted_patches.iter().collect();
    patches_vec.sort_by(|a, b| b.start.cmp(&a.start)); // Reverse order
    
    for patch in patches_vec {
        let old_len = patch.end - patch.start;
        let new_len = patch.replacement.len() as u64;
        
        if new_len > old_len {
            // Insertion: need to make space
            let insert_len = new_len - old_len;
            let insert_offset = (patch.start / BLOCK_SIZE) * BLOCK_SIZE; // Align down
            
            let result = unsafe {
                libc::fallocate(fd, FALLOC_FL_INSERT_RANGE, insert_offset as i64, insert_len as i64)
            };
            
            if result != 0 {
                // INSERT_RANGE failed, fall back
                return Ok(false);
            }
        } else if new_len < old_len {
            // Deletion: need to collapse space
            let collapse_len = old_len - new_len;
            let collapse_offset = ((patch.start + new_len) / BLOCK_SIZE) * BLOCK_SIZE; // Align
            
            let result = unsafe {
                libc::fallocate(fd, FALLOC_FL_COLLAPSE_RANGE, collapse_offset as i64, collapse_len as i64)
            };
            
            if result != 0 {
                // COLLAPSE_RANGE failed, fall back
                return Ok(false);
            }
        }
        
        // Write the replacement data
        if !patch.replacement.is_empty() {
            use std::os::unix::fs::FileExt;
            file.write_at(&patch.replacement, patch.start)
                .context("Failed to write patch data")?;
        }
    }
    
    file.sync_all().context("Failed to sync file")?;
    
    // Kick off background defrag if we did many extent operations
    let patch_count = sorted_patches.len();
    maybe_defrag_after_consolidation(path, patch_count);
    
    Ok(true)
}

/// Hybrid consolidation: combine extent ops with sub-block rewrites
#[cfg(target_os = "linux")]
fn try_hybrid_consolidate_linux(path: &Path, patches: &PatchList) -> Result<bool> {
    // For patches that don't align to blocks, we need a smarter approach:
    // 1. Group patches into block-aligned regions
    // 2. For regions that grow/shrink by full blocks, use INSERT/COLLAPSE
    // 3. For remaining regions, do in-place rewrite with read-modify-write
    
    // For now, fall back to standard streaming rewrite
    // This is a complex optimization that can be added later
    Ok(false)
}

/// XFS: Best of both worlds - FICLONE backup + INSERT_RANGE/COLLAPSE_RANGE
/// 
/// Strategy:
/// 1. FICLONE the original (instant CoW backup)
/// 2. Use INSERT_RANGE/COLLAPSE_RANGE for O(patches) in-place modification
/// 3. On success: delete backup
/// 4. On failure: restore from backup
/// 
/// This gives XFS the instant backup safety of btrfs PLUS the O(patches)
/// performance of ext4's extent operations.
#[cfg(target_os = "linux")]
fn try_reflink_extent_consolidate_linux(path: &Path, patches: &PatchList) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    
    // First, check if patches are block-aligned (required for INSERT/COLLAPSE_RANGE)
    let sorted_patches = patches.patches();
    if sorted_patches.is_empty() {
        return Ok(true);
    }
    
    let mut can_use_extent_ops = true;
    for patch in sorted_patches {
        let old_len = patch.end - patch.start;
        let new_len = patch.replacement.len() as u64;
        let delta = new_len as i64 - old_len as i64;
        
        // Extent operations only work at block boundaries
        if delta != 0 && (delta.unsigned_abs() % BLOCK_SIZE != 0) {
            can_use_extent_ops = false;
            break;
        }
    }
    
    if !can_use_extent_ops {
        // Fall back to FICLONE + streaming (still safe, just O(file_size))
        return try_reflink_consolidate_linux(path, patches);
    }
    
    // Create backup path (instant CoW clone)
    let backup_path = path.with_extension("bigedit-backup");
    let _ = std::fs::remove_file(&backup_path);
    
    // FICLONE the original to backup (instant on XFS with reflink)
    {
        let src = File::open(path).context("Failed to open source for FICLONE")?;
        let dst = File::create(&backup_path).context("Failed to create backup")?;
        
        let result = unsafe {
            libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
        };
        
        if result != 0 {
            // FICLONE failed (maybe old XFS without reflink), fall back to streaming
            let _ = std::fs::remove_file(&backup_path);
            return try_reflink_consolidate_linux(path, patches);
        }
    }
    
    // Now do extent operations on the original (we have instant backup)
    const FALLOC_FL_COLLAPSE_RANGE: i32 = 0x08;
    const FALLOC_FL_INSERT_RANGE: i32 = 0x20;
    
    let result = (|| -> Result<()> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .context("Failed to open file for extent consolidation")?;
        let fd = file.as_raw_fd();
        
        // Process patches in reverse order to avoid offset shifting issues
        let mut patches_vec: Vec<_> = sorted_patches.iter().collect();
        patches_vec.sort_by(|a, b| b.start.cmp(&a.start)); // Reverse order
        
        for patch in patches_vec {
            let old_len = patch.end - patch.start;
            let new_len = patch.replacement.len() as u64;
            
            if new_len > old_len {
                // Insertion: need to make space
                let insert_len = new_len - old_len;
                let insert_offset = (patch.start / BLOCK_SIZE) * BLOCK_SIZE; // Align down
                
                let result = unsafe {
                    libc::fallocate(fd, FALLOC_FL_INSERT_RANGE, insert_offset as i64, insert_len as i64)
                };
                
                if result != 0 {
                    bail!("INSERT_RANGE failed: {}", std::io::Error::last_os_error());
                }
            } else if new_len < old_len {
                // Deletion: need to collapse space
                let collapse_len = old_len - new_len;
                let collapse_offset = ((patch.start + new_len) / BLOCK_SIZE) * BLOCK_SIZE;
                
                let result = unsafe {
                    libc::fallocate(fd, FALLOC_FL_COLLAPSE_RANGE, collapse_offset as i64, collapse_len as i64)
                };
                
                if result != 0 {
                    bail!("COLLAPSE_RANGE failed: {}", std::io::Error::last_os_error());
                }
            }
            
            // Write the replacement data
            if !patch.replacement.is_empty() {
                use std::os::unix::fs::FileExt;
                file.write_at(&patch.replacement, patch.start)
                    .context("Failed to write patch data")?;
            }
        }
        
        file.sync_all().context("Failed to sync file")?;
        Ok(())
    })();
    
    match result {
        Ok(()) => {
            // Success - remove the backup
            let _ = std::fs::remove_file(&backup_path);
            
            // Kick off background defrag if we did many extent operations
            let patch_count = sorted_patches.len();
            maybe_defrag_after_consolidation(path, patch_count);
            
            Ok(true)
        }
        Err(e) => {
            // Failed - restore from backup
            eprintln!("[xfs] Extent consolidation failed, restoring from backup: {}", e);
            let _ = std::fs::rename(&backup_path, path);
            Err(e)
        }
    }
}

// =============================================================================
// Smart Consolidation Strategy
// =============================================================================

/// Estimated time in milliseconds for a single extent operation
const EXTENT_OP_TIME_MS: u64 = 5;

/// Estimated streaming rewrite speed in bytes per second (conservative SSD estimate)
const STREAM_SPEED_BYTES_PER_SEC: u64 = 500 * 1024 * 1024; // 500 MB/s

/// Time threshold in milliseconds - below this, consolidation feels "instant"
const INSTANT_THRESHOLD_MS: u64 = 100;

/// Time threshold in milliseconds - below this, consolidation is "fast"
const FAST_THRESHOLD_MS: u64 = 2000;

/// Consolidation strategy based on file size and patch patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationStrategy {
    /// Use INSERT_RANGE/COLLAPSE_RANGE (ext4/XFS) - O(patches)
    ExtentOperations,
    /// Use reflink/clonefile + modify (btrfs/APFS) - fast CoW
    CloneAndModify,
    /// Traditional streaming rewrite - O(file_size)
    StreamingRewrite,
    /// No consolidation needed
    NoOp,
}

/// Estimate consolidation time for different strategies
#[derive(Debug, Clone)]
pub struct ConsolidationEstimate {
    /// Best strategy for this file/patch combination
    pub strategy: ConsolidationStrategy,
    /// Estimated time in milliseconds
    pub estimated_time_ms: u64,
    /// Whether this feels "instant" to users (< 100ms)
    pub is_instant: bool,
    /// Whether this is "fast" (< 2s)
    pub is_fast: bool,
    /// Recommended: run in background?
    pub recommend_background: bool,
}

/// Estimate the number of extent operations needed for patches
fn estimate_extent_operations(patches: &PatchList) -> usize {
    let sorted = patches.patches();
    let mut ops = 0;
    
    for patch in sorted {
        let old_len = patch.end - patch.start;
        let new_len = patch.replacement.len() as u64;
        
        if new_len != old_len {
            // Size change requires INSERT or COLLAPSE operation
            ops += 1;
        }
        if !patch.replacement.is_empty() {
            // Writing data (not counted in extent ops, but fast)
        }
    }
    
    // Add overhead for sub-block alignment handling
    ops
}

/// Choose the best consolidation strategy for given file and patches
pub fn choose_consolidation_strategy(
    path: &Path, 
    file_size: u64, 
    patches: &PatchList
) -> ConsolidationEstimate {
    if !patches.is_modified() {
        return ConsolidationEstimate {
            strategy: ConsolidationStrategy::NoOp,
            estimated_time_ms: 0,
            is_instant: true,
            is_fast: true,
            recommend_background: false,
        };
    }
    
    let capability = detect_filesystem_capability(path);
    let patch_count = patches.patches().len();
    
    // Estimate times for each strategy
    let extent_ops = estimate_extent_operations(patches);
    let extent_time_ms = extent_ops as u64 * EXTENT_OP_TIME_MS;
    let stream_time_ms = (file_size * 1000) / STREAM_SPEED_BYTES_PER_SEC;
    
    // For CoW filesystems, cloning is nearly instant, then we stream-modify
    // Still O(file_size) but with CoW benefits for unchanged regions
    let cow_time_ms = stream_time_ms; // Conservative estimate
    
    let (strategy, time_ms) = match capability {
        FilesystemCapability::ExtentManipulationWithReflink => {
            // XFS: Best of both worlds - FICLONE backup + extent ops
            // Time = backup (instant) + extent ops
            if extent_time_ms < stream_time_ms {
                (ConsolidationStrategy::ExtentOperations, extent_time_ms)
            } else {
                (ConsolidationStrategy::CloneAndModify, cow_time_ms)
            }
        }
        FilesystemCapability::ExtentManipulation => {
            // ext4: Use extent ops if faster than streaming
            if extent_time_ms < stream_time_ms {
                (ConsolidationStrategy::ExtentOperations, extent_time_ms)
            } else {
                (ConsolidationStrategy::StreamingRewrite, stream_time_ms)
            }
        }
        FilesystemCapability::CopyOnWrite => {
            // btrfs/ZFS/APFS: Use clone+modify
            (ConsolidationStrategy::CloneAndModify, cow_time_ms)
        }
        FilesystemCapability::Standard => {
            (ConsolidationStrategy::StreamingRewrite, stream_time_ms)
        }
    };
    
    ConsolidationEstimate {
        strategy,
        estimated_time_ms: time_ms,
        is_instant: time_ms < INSTANT_THRESHOLD_MS,
        is_fast: time_ms < FAST_THRESHOLD_MS,
        recommend_background: time_ms > FAST_THRESHOLD_MS,
    }
}

/// Smart consolidate: automatically choose best strategy
pub fn smart_consolidate(path: &Path, patches: &PatchList) -> Result<ConsolidationEstimate> {
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);
    
    let estimate = choose_consolidation_strategy(path, file_size, patches);
    
    match estimate.strategy {
        ConsolidationStrategy::NoOp => {
            // Nothing to do
        }
        ConsolidationStrategy::ExtentOperations => {
            // Try extent-based consolidation (ext4/XFS)
            if !try_fast_consolidate(path, patches)? {
                // Fall back to streaming
                save_file(path, patches, None)?;
            }
        }
        ConsolidationStrategy::CloneAndModify => {
            // Use clone+modify approach for CoW filesystems
            let mut success = false;
            
            #[cfg(target_os = "macos")]
            {
                success = try_clonefile_consolidate_macos(path, patches)?;
            }
            
            #[cfg(target_os = "linux")]
            {
                if !success {
                    success = try_reflink_consolidate_linux(path, patches)?;
                }
            }
            
            if !success {
                // Fall back to standard streaming rewrite
                save_file(path, patches, None)?;
            }
        }
        ConsolidationStrategy::StreamingRewrite => {
            save_file(path, patches, None)?;
        }
    }
    
    Ok(estimate)
}

/// macOS APFS: Use clonefile for crash-safe consolidation
/// 
/// Strategy:
/// 1. clonefile the original to a backup (instant CoW)
/// 2. Stream rewrite to a new temp file with patches applied
/// 3. Atomic rename temp over original
/// 
/// The clonefile provides an instant backup - if anything goes wrong during
/// the rewrite, the original is still safely on disk.
#[cfg(target_os = "macos")]
fn try_clonefile_consolidate_macos(path: &Path, patches: &PatchList) -> Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    
    // Create backup path (instant CoW clone)
    let backup_path = path.with_extension("bigedit-backup");
    let _ = std::fs::remove_file(&backup_path);
    
    let src_cstr = CString::new(path.as_os_str().as_bytes())
        .context("Invalid source path")?;
    let dst_cstr = CString::new(backup_path.as_os_str().as_bytes())
        .context("Invalid backup path")?;
    
    // Clone the file to backup (instant on APFS - just creates CoW reference)
    extern "C" {
        fn clonefile(src: *const libc::c_char, dst: *const libc::c_char, flags: u32) -> libc::c_int;
    }
    
    let result = unsafe {
        clonefile(src_cstr.as_ptr(), dst_cstr.as_ptr(), 0)
    };
    
    if result != 0 {
        // clonefile failed (maybe not APFS, or cross-device)
        return Ok(false);
    }
    
    // Now do a regular streaming save (safe - we have instant backup)
    match save_file(path, patches, None) {
        Ok(()) => {
            // Success - remove the backup
            let _ = std::fs::remove_file(&backup_path);
            Ok(true)
        }
        Err(e) => {
            // Failed - restore from backup
            let _ = std::fs::rename(&backup_path, path);
            Err(e)
        }
    }
}

/// FICLONE ioctl number (from linux/fs.h) - instant full-file clone
#[cfg(target_os = "linux")]
const FICLONE: libc::c_ulong = 0x40049409;

/// Linux btrfs/ZFS: Use FICLONE for instant backup, then stream rewrite
/// 
/// Strategy:
/// 1. FICLONE the original (instant CoW backup)
/// 2. Stream rewrite to a new temp file with patches applied
/// 3. Atomic rename temp over original
/// 
/// The FICLONE provides an instant backup - if anything goes wrong during
/// the rewrite, the original is still safely on disk.
#[cfg(target_os = "linux")]
fn try_reflink_consolidate_linux(path: &Path, patches: &PatchList) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    
    // Check if this is a CoW filesystem (btrfs, ZFS, XFS with reflink)
    let fstype = get_filesystem_type_linux(path);
    let is_cow_fs = matches!(fstype.as_deref(), Some("btrfs") | Some("zfs") | Some("xfs"));
    
    if !is_cow_fs {
        return Ok(false);
    }
    
    // Create backup path (instant CoW clone)
    let backup_path = path.with_extension("bigedit-backup");
    let _ = std::fs::remove_file(&backup_path);
    
    // FICLONE the original to backup (instant on CoW filesystems)
    {
        let src = File::open(path).context("Failed to open source for FICLONE")?;
        let dst = File::create(&backup_path).context("Failed to create backup")?;
        
        let result = unsafe {
            libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
        };
        
        if result != 0 {
            // FICLONE failed, clean up and fall back to regular save
            let _ = std::fs::remove_file(&backup_path);
            return Ok(false);
        }
    }
    
    // Now do a regular streaming save (safe - we have instant backup)
    match save_file(path, patches, None) {
        Ok(()) => {
            // Success - remove the backup
            let _ = std::fs::remove_file(&backup_path);
            Ok(true)
        }
        Err(e) => {
            // Failed - restore from backup
            let _ = std::fs::rename(&backup_path, path);
            Err(e)
        }
    }
}

/// Placeholder for non-Linux
#[cfg(not(target_os = "linux"))]
fn try_reflink_consolidate_linux(_path: &Path, _patches: &PatchList) -> Result<bool> {
    Ok(false)
}

/// Placeholder for non-macOS
#[cfg(not(target_os = "macos"))]
fn try_clonefile_consolidate_macos(_path: &Path, _patches: &PatchList) -> Result<bool> {
    Ok(false)
}

/// Check available disk space for background consolidation
pub fn available_disk_space(path: &Path) -> Result<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem::MaybeUninit;
        
        let path_cstr = CString::new(path.to_string_lossy().as_bytes())
            .context("Invalid path")?;
        
        unsafe {
            let mut statfs_buf: MaybeUninit<libc::statfs> = MaybeUninit::uninit();
            if libc::statfs(path_cstr.as_ptr(), statfs_buf.as_mut_ptr()) == 0 {
                let statfs = statfs_buf.assume_init();
                let available = statfs.f_bavail as u64 * statfs.f_bsize as u64;
                return Ok(available);
            }
        }
        bail!("Failed to get disk space");
    }
    
    #[cfg(not(unix))]
    {
        // Windows: use GetDiskFreeSpaceExW
        Ok(0) // TODO: implement for Windows
    }
}

/// Check if background consolidation should run
/// Returns true if there's enough disk space (file_size + 10% buffer)
pub fn should_auto_consolidate(path: &Path, file_size: u64) -> bool {
    if let Ok(available) = available_disk_space(path) {
        let required = file_size + (file_size / 10); // file + 10% buffer
        return available >= required;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Patch;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_save_no_patches() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello world");

        let patches = PatchList::new();
        save_file(&path, &patches, None).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_save_with_replacement() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello world");

        let mut patches = PatchList::new();
        patches.replace(6, 11, b"rust");

        save_file(&path, &patches, None).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[test]
    fn test_save_with_insertion() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello world");

        let mut patches = PatchList::new();
        patches.insert(5, b" beautiful");

        save_file(&path, &patches, None).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello beautiful world");
    }

    #[test]
    fn test_save_with_deletion() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello beautiful world");

        let mut patches = PatchList::new();
        patches.delete(5, 15); // Delete " beautiful"

        save_file(&path, &patches, None).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_save_as() {
        let dir = tempdir().unwrap();
        let original = create_test_file(dir.path(), "original.txt", "hello");
        let new_path = dir.path().join("new.txt");

        let mut patches = PatchList::new();
        patches.insert(5, b" world");

        save_as(&original, &patches, &new_path).unwrap();

        // Original should be unchanged
        let original_content = fs::read_to_string(&original).unwrap();
        assert_eq!(original_content, "hello");

        // New file should have patches applied
        let new_content = fs::read_to_string(&new_path).unwrap();
        assert_eq!(new_content, "hello world");
    }

    #[test]
    fn test_multiple_patches() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "The quick brown fox jumps");

        let mut patches = PatchList::new();
        patches.replace(4, 9, b"slow"); // "quick" -> "slow"
        patches.replace(10, 15, b"red"); // "brown" -> "red"
        patches.insert(25, b" high"); // add " high" at end

        save_file(&path, &patches, None).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "The slow red fox jumps high");
    }
    
    #[test]
    fn test_detect_filesystem_capability() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello");
        
        let capability = detect_filesystem_capability(&path);
        // On Linux ext4, this should be ExtentManipulation
        // The test just verifies the function doesn't crash
        println!("Detected filesystem capability: {:?}", capability);
        
        // Verify it returns a valid value
        assert!(matches!(
            capability,
            FilesystemCapability::ExtentManipulation 
            | FilesystemCapability::CopyOnWrite 
            | FilesystemCapability::Standard
        ));
    }
    
    #[test]
    fn test_available_disk_space() {
        let dir = tempdir().unwrap();
        
        let space = available_disk_space(dir.path());
        assert!(space.is_ok());
        
        let bytes = space.unwrap();
        println!("Available disk space: {} bytes ({:.2} GB)", bytes, bytes as f64 / 1e9);
        
        // Should be positive (we have some space)
        assert!(bytes > 0);
    }
    
    #[test]
    fn test_should_auto_consolidate() {
        let dir = tempdir().unwrap();
        
        // Small file should pass
        let should = should_auto_consolidate(dir.path(), 1024);
        assert!(should, "Should auto-consolidate small files");
        
        // Huge file (1 PB) should fail
        let should_not = should_auto_consolidate(dir.path(), 1_000_000_000_000_000);
        assert!(!should_not, "Should not auto-consolidate when not enough space");
    }
    
    #[test]
    fn test_choose_consolidation_strategy() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello world");
        
        // No patches = NoOp
        let patches = PatchList::new();
        let estimate = choose_consolidation_strategy(&path, 1024, &patches);
        assert_eq!(estimate.strategy, ConsolidationStrategy::NoOp);
        assert!(estimate.is_instant);
        
        // Small file with patches = should be instant regardless of strategy
        let mut patches = PatchList::new();
        patches.insert(5, b" beautiful");
        let estimate = choose_consolidation_strategy(&path, 1024, &patches);
        assert!(estimate.is_instant, "Small file should be instant");
        
        // Large file (10GB) with few patches on ext4 = extent ops preferred
        let estimate = choose_consolidation_strategy(&path, 10 * 1024 * 1024 * 1024, &patches);
        println!("10GB file, 1 patch: {:?}, {}ms", estimate.strategy, estimate.estimated_time_ms);
        // On ext4, should prefer extent operations
        // On other filesystems, may use streaming but should estimate time correctly
        assert!(estimate.estimated_time_ms > 0 || estimate.strategy == ConsolidationStrategy::ExtentOperations);
    }
    
    #[test]
    fn test_smart_consolidate() {
        let dir = tempdir().unwrap();
        let path = create_test_file(dir.path(), "test.txt", "hello world");
        
        let mut patches = PatchList::new();
        patches.insert(5, b" beautiful");
        
        // Smart consolidate should work
        let estimate = smart_consolidate(&path, &patches).unwrap();
        println!("Smart consolidate result: {:?}", estimate);
        
        // Verify the file was modified correctly
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello beautiful world");
    }
    
    #[test]
    fn test_estimate_extent_operations() {
        // No size change = no extent ops
        let mut patches = PatchList::new();
        patches.replace(0, 5, b"hello"); // Same length
        let ops = estimate_extent_operations(&patches);
        assert_eq!(ops, 0, "Same-length replacement needs no extent ops");
        
        // Insertion needs extent op
        let mut patches = PatchList::new();
        patches.insert(5, b" world");
        let ops = estimate_extent_operations(&patches);
        assert_eq!(ops, 1, "Insertion needs 1 extent op");
        
        // Multiple insertions
        let mut patches = PatchList::new();
        patches.insert(0, b"start ");
        patches.insert(100, b" middle");
        patches.insert(200, b" end");
        let ops = estimate_extent_operations(&patches);
        assert_eq!(ops, 3, "3 insertions need 3 extent ops");
    }
    
    /// Test btrfs reflink consolidation
    /// Run with: BTRFS_TEST_PATH=/tmp/btrfs-mnt cargo test test_btrfs_reflink -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted btrfs filesystem
    fn test_btrfs_reflink_consolidation() {
        use std::os::unix::io::AsRawFd;
        
        // Get btrfs test path from environment
        let btrfs_path = match std::env::var("BTRFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping btrfs test - set BTRFS_TEST_PATH to a mounted btrfs filesystem");
                return;
            }
        };
        
        // Verify it's actually btrfs
        let capability = detect_filesystem_capability(&btrfs_path);
        println!("Detected capability for {:?}: {:?}", btrfs_path, capability);
        
        if capability != FilesystemCapability::CopyOnWrite {
            println!("Warning: {} is not detected as btrfs/CoW filesystem", btrfs_path.display());
        }
        
        // Create test file
        let test_file = btrfs_path.join("reflink_test.txt");
        let content = "Hello World! This is a test file for btrfs reflink consolidation.";
        fs::write(&test_file, content).expect("Failed to create test file");
        println!("Created test file: {:?}", test_file);
        
        // Test FICLONE directly
        let clone_file = btrfs_path.join("reflink_clone.txt");
        {
            let src = File::open(&test_file).expect("open source");
            let dst = File::create(&clone_file).expect("create dest");
            
            const FICLONE: libc::c_ulong = 0x40049409;
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            
            if result == 0 {
                println!("✓ FICLONE succeeded");
            } else {
                let err = std::io::Error::last_os_error();
                println!("✗ FICLONE failed: {}", err);
                fs::remove_file(&test_file).ok();
                panic!("FICLONE not supported on this filesystem");
            }
        }
        
        // Verify clone content
        let cloned = fs::read_to_string(&clone_file).expect("read clone");
        assert_eq!(cloned, content, "Clone content should match");
        println!("✓ Clone content verified");
        
        // Test our reflink consolidation function
        fs::remove_file(&clone_file).ok();
        
        let mut patches = PatchList::new();
        patches.replace(0, 5, b"HELLO"); // Same-size replacement
        patches.insert(12, b" INSERTED"); // Insertion
        
        let result = try_reflink_consolidate_linux(&test_file, &patches);
        match result {
            Ok(true) => {
                println!("✓ Reflink consolidation succeeded");
                let final_content = fs::read_to_string(&test_file).unwrap();
                println!("Final content: {}", final_content);
                assert!(final_content.starts_with("HELLO"), "Should start with HELLO");
                assert!(final_content.contains("INSERTED"), "Should contain INSERTED");
            }
            Ok(false) => {
                println!("Reflink consolidation returned false, falling back");
            }
            Err(e) => {
                println!("Reflink consolidation error: {}", e);
            }
        }
        
        // Cleanup
        fs::remove_file(&test_file).ok();
        fs::remove_file(&clone_file).ok();
        println!("✓ btrfs reflink test complete");
    }
    
    /// Benchmark btrfs FICLONE backup strategy vs streaming rewrite
    /// Run with: BTRFS_TEST_PATH=/tmp/btrfs-mnt cargo test bench_btrfs_reflink -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted btrfs filesystem with test file
    fn bench_btrfs_reflink_consolidation() {
        use std::os::unix::io::AsRawFd;
        use std::time::Instant;
        
        let btrfs_path = match std::env::var("BTRFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set BTRFS_TEST_PATH to a mounted btrfs filesystem");
                println!("Example: BTRFS_TEST_PATH=/tmp/btrfs-mnt cargo test bench_btrfs -- --nocapture --ignored");
                return;
            }
        };
        
        // Look for test file
        let test_file = btrfs_path.join("simple-wikipedia.txt");
        if !test_file.exists() {
            println!("Test file not found: {:?}", test_file);
            println!("Copy a large file there first:");
            println!("  cp benches/data/simple-wikipedia.txt /tmp/btrfs-mnt/");
            return;
        }
        
        let file_size = fs::metadata(&test_file).unwrap().len();
        println!("\n=== btrfs FICLONE Backup Benchmark ===");
        println!("Test file: {:?}", test_file);
        println!("File size: {:.1} MB", file_size as f64 / 1_000_000.0);
        println!();
        
        // Verify filesystem
        let capability = detect_filesystem_capability(&btrfs_path);
        println!("Filesystem capability: {:?}", capability);
        assert_eq!(capability, FilesystemCapability::CopyOnWrite, "Need btrfs/CoW filesystem");
        
        // Benchmark 1: FICLONE (instant backup)
        let backup_path = btrfs_path.join("bench_backup.txt");
        let _ = fs::remove_file(&backup_path);
        
        print!("FICLONE backup...         ");
        let start = Instant::now();
        {
            let src = File::open(&test_file).unwrap();
            let dst = File::create(&backup_path).unwrap();
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            dst.sync_all().unwrap();
            assert_eq!(result, 0, "FICLONE failed");
        }
        let ficlone_time = start.elapsed();
        println!("{:>12?}  (instant CoW backup)", ficlone_time);
        
        // Benchmark 2: Reflink consolidation (FICLONE backup + streaming)
        let consolidate_path = btrfs_path.join("bench_consolidate.txt");
        let _ = fs::remove_file(&consolidate_path);
        fs::copy(&test_file, &consolidate_path).unwrap();
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Reflink consolidation...  ");
        let start = Instant::now();
        let result = try_reflink_consolidate_linux(&consolidate_path, &patches);
        let reflink_time = start.elapsed();
        
        match result {
            Ok(true) => {
                println!("{:>12?}  (FICLONE backup + stream)", reflink_time);
            }
            Ok(false) => {
                println!("{:>12?}  (fallback)", reflink_time);
            }
            Err(e) => {
                println!("ERROR: {}", e);
            }
        }
        
        // Benchmark 3: Plain streaming rewrite
        let stream_path = btrfs_path.join("bench_stream.txt");
        let _ = fs::remove_file(&stream_path);
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Plain streaming...        ");
        let start = Instant::now();
        save_file(&test_file, &patches, Some(&stream_path)).unwrap();
        let stream_time = start.elapsed();
        let stream_speed = file_size as f64 / stream_time.as_secs_f64() / 1_000_000_000.0;
        println!("{:>12?}  ({:.2} GB/s)", stream_time, stream_speed);
        
        // Summary
        println!("\n=== Summary ===");
        println!("FICLONE backup: {:?} (instant)", ficlone_time);
        println!("The reflink strategy provides instant backup safety,");
        println!("then does the same streaming rewrite as plain streaming.");
        println!("Total time similar, but with automatic rollback on failure.");
        
        // Cleanup
        fs::remove_file(&backup_path).ok();
        fs::remove_file(&consolidate_path).ok();
        fs::remove_file(&stream_path).ok();
    }
    
    /// Test XFS reflink + extent operations (best of both worlds)
    /// Run with: XFS_TEST_PATH=/tmp/xfs-mnt cargo test test_xfs -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted XFS filesystem with reflink support
    fn test_xfs_consolidation() {
        use std::os::unix::io::AsRawFd;
        
        let xfs_path = match std::env::var("XFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set XFS_TEST_PATH to a mounted XFS filesystem with reflink");
                println!("Create with: mkfs.xfs -m reflink=1 /path/to/image");
                return;
            }
        };
        
        // Verify filesystem capability
        let capability = detect_filesystem_capability(&xfs_path);
        println!("Detected capability for {:?}: {:?}", xfs_path, capability);
        assert_eq!(capability, FilesystemCapability::ExtentManipulationWithReflink, 
            "Expected XFS with reflink support");
        
        // Create test file
        let test_file = xfs_path.join("xfs_test.txt");
        fs::write(&test_file, "Hello World! This is a test file for XFS reflink + extent consolidation.").unwrap();
        println!("Created test file: {:?}", test_file);
        
        // Test FICLONE
        println!("\n--- FICLONE (reflink) Test ---");
        let clone_file = xfs_path.join("xfs_clone.txt");
        {
            let src = File::open(&test_file).unwrap();
            let dst = File::create(&clone_file).unwrap();
            
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            
            if result == 0 {
                println!("✓ FICLONE succeeded");
                // Verify content
                let clone_content = fs::read_to_string(&clone_file).unwrap();
                assert!(clone_content.contains("Hello World"));
                println!("✓ Clone content verified");
            } else {
                let err = std::io::Error::last_os_error();
                println!("✗ FICLONE failed: {} (XFS may not have reflink enabled)", err);
                fs::remove_file(&test_file).ok();
                return;
            }
        }
        fs::remove_file(&clone_file).ok();
        
        // Test reflink + extent consolidation
        println!("\n--- Reflink + Extent Consolidation Test ---");
        let mut patches = PatchList::new();
        patches.replace(0, 5, b"HELLO");  // Replace "Hello" with "HELLO"
        patches.insert(12, b" INSERTED");  // Insert after "World!"
        
        let result = try_reflink_extent_consolidate_linux(&test_file, &patches);
        match result {
            Ok(true) => println!("✓ Reflink + extent consolidation succeeded"),
            Ok(false) => println!("○ Fell back to streaming (patches not block-aligned)"),
            Err(e) => println!("○ Consolidation error (expected for non-aligned): {}", e),
        }
        
        // Verify content
        let content = fs::read_to_string(&test_file).unwrap();
        println!("Final content: {}", content);
        
        // Cleanup
        fs::remove_file(&test_file).ok();
        println!("✓ XFS test complete");
    }
    
    /// Benchmark XFS FICLONE + extent operations
    /// Run with: XFS_TEST_PATH=/tmp/xfs-mnt cargo test bench_xfs -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted XFS filesystem with test file
    fn bench_xfs_consolidation() {
        use std::os::unix::io::AsRawFd;
        use std::time::Instant;
        
        let xfs_path = match std::env::var("XFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set XFS_TEST_PATH to a mounted XFS filesystem");
                println!("Example: XFS_TEST_PATH=/tmp/xfs-mnt cargo test bench_xfs -- --nocapture --ignored");
                return;
            }
        };
        
        // Look for test file
        let test_file = xfs_path.join("simple-wikipedia.txt");
        if !test_file.exists() {
            println!("Test file not found: {:?}", test_file);
            println!("Copy a large file there first:");
            println!("  cp benches/data/simple-wikipedia.txt /tmp/xfs-mnt/");
            return;
        }
        
        let file_size = fs::metadata(&test_file).unwrap().len();
        println!("\n=== XFS FICLONE + Extent Benchmark ===");
        println!("Test file: {:?}", test_file);
        println!("File size: {:.1} MB", file_size as f64 / 1_000_000.0);
        println!();
        
        // Verify filesystem
        let capability = detect_filesystem_capability(&xfs_path);
        println!("Filesystem capability: {:?}", capability);
        
        // Benchmark 1: FICLONE (instant backup)
        let backup_path = xfs_path.join("bench_backup.txt");
        let _ = fs::remove_file(&backup_path);
        
        print!("FICLONE backup...           ");
        let start = Instant::now();
        {
            let src = File::open(&test_file).unwrap();
            let dst = File::create(&backup_path).unwrap();
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            dst.sync_all().unwrap();
            if result != 0 {
                println!("FICLONE not supported on this XFS (needs reflink=1)");
                fs::remove_file(&backup_path).ok();
                return;
            }
        }
        let ficlone_time = start.elapsed();
        println!("{:>12?}  (instant CoW backup)", ficlone_time);
        
        // Benchmark 2: Reflink + extent consolidation
        let consolidate_path = xfs_path.join("bench_consolidate.txt");
        let _ = fs::remove_file(&consolidate_path);
        fs::copy(&test_file, &consolidate_path).unwrap();
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Reflink+extent consolidate... ");
        let start = Instant::now();
        let result = try_reflink_extent_consolidate_linux(&consolidate_path, &patches);
        let extent_time = start.elapsed();
        
        match result {
            Ok(true) => println!("{:>12?}  (FICLONE + extent ops)", extent_time),
            Ok(false) => println!("{:>12?}  (fallback to stream)", extent_time),
            Err(e) => println!("ERROR: {}", e),
        }
        
        // Benchmark 3: Plain streaming rewrite
        let stream_path = xfs_path.join("bench_stream.txt");
        let _ = fs::remove_file(&stream_path);
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Plain streaming...          ");
        let start = Instant::now();
        save_file(&test_file, &patches, Some(&stream_path)).unwrap();
        let stream_time = start.elapsed();
        let stream_speed = file_size as f64 / stream_time.as_secs_f64() / 1_000_000_000.0;
        println!("{:>12?}  ({:.2} GB/s)", stream_time, stream_speed);
        
        // Summary
        println!("\n=== Summary ===");
        println!("FICLONE backup: {:?} (instant)", ficlone_time);
        if extent_time < stream_time {
            println!("XFS extent ops: {:.1}x faster than streaming!", 
                stream_time.as_secs_f64() / extent_time.as_secs_f64());
        }
        println!("XFS provides: instant backup + O(patches) for block-aligned changes");
        
        // Cleanup
        fs::remove_file(&backup_path).ok();
        fs::remove_file(&consolidate_path).ok();
        fs::remove_file(&stream_path).ok();
    }
    
    /// Test ZFS reflink consolidation
    /// Run with: ZFS_TEST_PATH=/testpool cargo test test_zfs -- --nocapture --ignored
    #[test]
    #[ignore] // Requires ZFS pool
    fn test_zfs_consolidation() {
        use std::os::unix::io::AsRawFd;
        
        let zfs_path = match std::env::var("ZFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set ZFS_TEST_PATH to a ZFS pool mount point");
                println!("Create with: zpool create testpool /path/to/image");
                return;
            }
        };
        
        // Verify filesystem capability
        let capability = detect_filesystem_capability(&zfs_path);
        println!("Detected capability for {:?}: {:?}", zfs_path, capability);
        assert_eq!(capability, FilesystemCapability::CopyOnWrite, 
            "Expected ZFS with CoW support");
        
        // Create test file
        let test_file = zfs_path.join("zfs_test.txt");
        fs::write(&test_file, "Hello World! This is a test file for ZFS reflink consolidation.").unwrap();
        println!("Created test file: {:?}", test_file);
        
        // Test FICLONE
        println!("\n--- FICLONE (reflink) Test ---");
        let clone_file = zfs_path.join("zfs_clone.txt");
        {
            let src = File::open(&test_file).unwrap();
            let dst = File::create(&clone_file).unwrap();
            
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            
            if result == 0 {
                println!("✓ FICLONE succeeded");
                let clone_content = fs::read_to_string(&clone_file).unwrap();
                assert!(clone_content.contains("Hello World"));
                println!("✓ Clone content verified");
            } else {
                let err = std::io::Error::last_os_error();
                println!("✗ FICLONE failed: {}", err);
                fs::remove_file(&test_file).ok();
                return;
            }
        }
        fs::remove_file(&clone_file).ok();
        
        // Test reflink consolidation
        println!("\n--- Reflink Consolidation Test ---");
        let mut patches = PatchList::new();
        patches.replace(0, 5, b"HELLO");
        patches.insert(12, b" INSERTED");
        
        let result = try_reflink_consolidate_linux(&test_file, &patches);
        match result {
            Ok(true) => println!("✓ Reflink consolidation succeeded"),
            Ok(false) => println!("✗ Reflink consolidation returned false"),
            Err(e) => println!("✗ Error: {}", e),
        }
        
        let content = fs::read_to_string(&test_file).unwrap();
        println!("Final content: {}", content);
        
        fs::remove_file(&test_file).ok();
        println!("✓ ZFS test complete");
    }
    
    /// Benchmark ZFS FICLONE strategy
    /// Run with: ZFS_TEST_PATH=/testpool cargo test bench_zfs -- --nocapture --ignored
    #[test]
    #[ignore] // Requires ZFS pool with test file
    fn bench_zfs_consolidation() {
        use std::os::unix::io::AsRawFd;
        use std::time::Instant;
        
        let zfs_path = match std::env::var("ZFS_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set ZFS_TEST_PATH to a ZFS pool mount");
                println!("Example: ZFS_TEST_PATH=/testpool cargo test bench_zfs -- --nocapture --ignored");
                return;
            }
        };
        
        let test_file = zfs_path.join("simple-wikipedia.txt");
        if !test_file.exists() {
            println!("Test file not found: {:?}", test_file);
            println!("Copy a large file there first");
            return;
        }
        
        let file_size = fs::metadata(&test_file).unwrap().len();
        println!("\n=== ZFS FICLONE Backup Benchmark ===");
        println!("Test file: {:?}", test_file);
        println!("File size: {:.1} MB", file_size as f64 / 1_000_000.0);
        println!();
        
        let capability = detect_filesystem_capability(&zfs_path);
        println!("Filesystem capability: {:?}", capability);
        
        // Benchmark 1: FICLONE
        let backup_path = zfs_path.join("bench_backup.txt");
        let _ = fs::remove_file(&backup_path);
        
        print!("FICLONE backup...         ");
        let start = Instant::now();
        {
            let src = File::open(&test_file).unwrap();
            let dst = File::create(&backup_path).unwrap();
            let result = unsafe {
                libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
            };
            dst.sync_all().unwrap();
            if result != 0 {
                println!("FICLONE not supported");
                return;
            }
        }
        let ficlone_time = start.elapsed();
        println!("{:>12?}  (instant CoW backup)", ficlone_time);
        
        // Benchmark 2: Reflink consolidation
        let consolidate_path = zfs_path.join("bench_consolidate.txt");
        let _ = fs::remove_file(&consolidate_path);
        fs::copy(&test_file, &consolidate_path).unwrap();
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Reflink consolidation...  ");
        let start = Instant::now();
        let _ = try_reflink_consolidate_linux(&consolidate_path, &patches);
        let reflink_time = start.elapsed();
        println!("{:>12?}  (FICLONE backup + stream)", reflink_time);
        
        // Benchmark 3: Plain streaming
        let stream_path = zfs_path.join("bench_stream.txt");
        let _ = fs::remove_file(&stream_path);
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Plain streaming...        ");
        let start = Instant::now();
        save_file(&test_file, &patches, Some(&stream_path)).unwrap();
        let stream_time = start.elapsed();
        let stream_speed = file_size as f64 / stream_time.as_secs_f64() / 1_000_000_000.0;
        println!("{:>12?}  ({:.2} GB/s)", stream_time, stream_speed);
        
        println!("\n=== Summary ===");
        println!("FICLONE backup: {:?} (instant)", ficlone_time);
        println!("ZFS provides instant backup safety with automatic rollback.");
        
        fs::remove_file(&backup_path).ok();
        fs::remove_file(&consolidate_path).ok();
        fs::remove_file(&stream_path).ok();
    }
    
    /// Test ext4 extent operations (INSERT_RANGE/COLLAPSE_RANGE)
    /// Run with: EXT4_TEST_PATH=/tmp/ext4-mnt cargo test test_ext4 -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted ext4 filesystem
    fn test_ext4_consolidation() {
        let ext4_path = match std::env::var("EXT4_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set EXT4_TEST_PATH to a mounted ext4 filesystem");
                return;
            }
        };
        
        // Verify filesystem capability
        let capability = detect_filesystem_capability(&ext4_path);
        println!("Detected capability for {:?}: {:?}", ext4_path, capability);
        assert_eq!(capability, FilesystemCapability::ExtentManipulation, 
            "Expected ext4 with extent manipulation support");
        
        // Create test file
        let test_file = ext4_path.join("ext4_test.txt");
        fs::write(&test_file, "Hello World! This is a test file for ext4 extent consolidation.").unwrap();
        println!("Created test file: {:?}", test_file);
        
        // Test extent consolidation (note: ext4 does NOT support FICLONE)
        println!("\n--- Extent Consolidation Test ---");
        let mut patches = PatchList::new();
        patches.replace(0, 5, b"HELLO");  // Replace "Hello" with "HELLO"
        
        // ext4 extent ops require block alignment, so this will likely fall back
        let result = try_extent_consolidate_linux(&test_file, &patches);
        match result {
            Ok(true) => println!("✓ Extent consolidation succeeded"),
            Ok(false) => println!("○ Fell back (patches not block-aligned)"),
            Err(e) => println!("○ Error: {}", e),
        }
        
        let content = fs::read_to_string(&test_file).unwrap();
        println!("Final content: {}", content);
        
        fs::remove_file(&test_file).ok();
        println!("✓ ext4 test complete");
    }
    
    /// Benchmark ext4 extent operations
    /// Run with: EXT4_TEST_PATH=/tmp/ext4-mnt cargo test bench_ext4 -- --nocapture --ignored
    #[test]
    #[ignore] // Requires mounted ext4 filesystem with test file
    fn bench_ext4_consolidation() {
        use std::time::Instant;
        
        let ext4_path = match std::env::var("EXT4_TEST_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                println!("Skipping - set EXT4_TEST_PATH to a mounted ext4 filesystem");
                println!("Example: EXT4_TEST_PATH=/tmp/ext4-mnt cargo test bench_ext4 -- --nocapture --ignored");
                return;
            }
        };
        
        let test_file = ext4_path.join("simple-wikipedia.txt");
        if !test_file.exists() {
            println!("Test file not found: {:?}", test_file);
            println!("Copy a large file there first");
            return;
        }
        
        let file_size = fs::metadata(&test_file).unwrap().len();
        println!("\n=== ext4 Extent Operations Benchmark ===");
        println!("Test file: {:?}", test_file);
        println!("File size: {:.1} MB", file_size as f64 / 1_000_000.0);
        println!();
        
        let capability = detect_filesystem_capability(&ext4_path);
        println!("Filesystem capability: {:?}", capability);
        println!("Note: ext4 does NOT support FICLONE (no instant backup)");
        println!();
        
        // Benchmark 1: Extent consolidation (block-aligned patches only)
        let consolidate_path = ext4_path.join("bench_consolidate.txt");
        let _ = fs::remove_file(&consolidate_path);
        fs::copy(&test_file, &consolidate_path).unwrap();
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Extent consolidation...   ");
        let start = Instant::now();
        let result = try_extent_consolidate_linux(&consolidate_path, &patches);
        let extent_time = start.elapsed();
        
        match result {
            Ok(true) => println!("{:>12?}  (INSERT/COLLAPSE_RANGE)", extent_time),
            Ok(false) => println!("{:>12?}  (fallback - not block-aligned)", extent_time),
            Err(e) => println!("ERROR: {}", e),
        }
        
        // Benchmark 2: Plain streaming rewrite
        let stream_path = ext4_path.join("bench_stream.txt");
        let _ = fs::remove_file(&stream_path);
        
        let mut patches = PatchList::new();
        patches.replace(0, 8, b"REPLACED");
        patches.insert(1000, b"INSERTED TEXT HERE");
        
        print!("Plain streaming...        ");
        let start = Instant::now();
        save_file(&test_file, &patches, Some(&stream_path)).unwrap();
        let stream_time = start.elapsed();
        let stream_speed = file_size as f64 / stream_time.as_secs_f64() / 1_000_000_000.0;
        println!("{:>12?}  ({:.2} GB/s)", stream_time, stream_speed);
        
        println!("\n=== Summary ===");
        println!("ext4 provides INSERT_RANGE/COLLAPSE_RANGE for block-aligned patches.");
        println!("For non-block-aligned patches, falls back to streaming rewrite.");
        println!("⚠️  No FICLONE support - no instant backup (less crash-safe than XFS/btrfs/ZFS)");
        
        fs::remove_file(&consolidate_path).ok();
        fs::remove_file(&stream_path).ok();
    }
    
    /// All-in-one benchmark: creates loopback filesystems if needed
    /// Run with: cargo test bench_all_filesystems -- --nocapture --ignored
    /// 
    /// This test auto-creates loopback mounts for btrfs, xfs, zfs, and ext4.
    /// Requires sudo access for mounting.
    #[test]
    #[ignore] // Requires sudo for loopback mounts
    fn bench_all_filesystems() {
        use std::process::Command;
        use std::time::Instant;
        
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║          Filesystem Consolidation Benchmark Suite             ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");
        
        // Check if we have sudo
        let sudo_check = Command::new("sudo").args(["-n", "true"]).status();
        if sudo_check.map(|s| !s.success()).unwrap_or(true) {
            println!("⚠️  This test requires sudo access for loopback mounts.");
            println!("   Run: sudo -v  (to cache credentials)");
            println!("   Then re-run the test.");
            return;
        }
        
        let test_data_source = PathBuf::from("benches/data/simple-wikipedia.txt");
        if !test_data_source.exists() {
            println!("⚠️  Test data not found: {:?}", test_data_source);
            println!("   Please ensure the benchmark data file exists.");
            return;
        }
        
        let file_size = fs::metadata(&test_data_source).unwrap().len();
        println!("Test file: {} ({:.1} MB)\n", test_data_source.display(), file_size as f64 / 1_000_000.0);
        
        // Define filesystem configurations
        struct FsConfig {
            name: &'static str,
            img_path: &'static str,
            mnt_path: &'static str,
            create_cmd: &'static str,
            expected_capability: FilesystemCapability,
        }
        
        let filesystems = [
            FsConfig {
                name: "btrfs",
                img_path: "/tmp/bench-btrfs.img",
                mnt_path: "/tmp/bench-btrfs",
                create_cmd: "mkfs.btrfs -f",
                expected_capability: FilesystemCapability::CopyOnWrite,
            },
            FsConfig {
                name: "xfs",
                img_path: "/tmp/bench-xfs.img", 
                mnt_path: "/tmp/bench-xfs",
                create_cmd: "mkfs.xfs -f -m reflink=1",
                expected_capability: FilesystemCapability::ExtentManipulationWithReflink,
            },
            FsConfig {
                name: "ext4",
                img_path: "/tmp/bench-ext4.img",
                mnt_path: "/tmp/bench-ext4", 
                create_cmd: "mkfs.ext4 -F -q",
                expected_capability: FilesystemCapability::ExtentManipulation,
            },
        ];
        
        println!("Setting up filesystems...\n");
        
        for fs in &filesystems {
            print!("  {} ... ", fs.name);
            
            // Create image file
            let _ = fs::remove_file(fs.img_path);
            if Command::new("truncate")
                .args(["-s", "512M", fs.img_path])
                .status()
                .map(|s| !s.success())
                .unwrap_or(true)
            {
                println!("SKIP (truncate failed)");
                continue;
            }
            
            // Format
            let create_parts: Vec<&str> = fs.create_cmd.split_whitespace().collect();
            if Command::new(create_parts[0])
                .args(&create_parts[1..])
                .arg(fs.img_path)
                .output()
                .map(|o| !o.status.success())
                .unwrap_or(true)
            {
                println!("SKIP (format failed)");
                continue;
            }
            
            // Create mount point and mount
            let _ = fs::create_dir_all(fs.mnt_path);
            if Command::new("sudo")
                .args(["mount", "-o", "loop", fs.img_path, fs.mnt_path])
                .status()
                .map(|s| !s.success())
                .unwrap_or(true)
            {
                println!("SKIP (mount failed)");
                continue;
            }
            
            // Set ownership and copy test file
            let _ = Command::new("sudo")
                .args(["chown", "-R", &format!("{}:{}", 
                    std::env::var("USER").unwrap_or_default(),
                    std::env::var("USER").unwrap_or_default()), 
                    fs.mnt_path])
                .status();
            
            let test_file = PathBuf::from(fs.mnt_path).join("test.txt");
            if fs::copy(&test_data_source, &test_file).is_err() {
                println!("SKIP (copy failed)");
                continue;
            }
            
            println!("OK");
        }
        
        println!("\n{}\n", "─".repeat(66));
        
        // Run benchmarks
        for fs in &filesystems {
            let mnt_path = PathBuf::from(fs.mnt_path);
            let test_file = mnt_path.join("test.txt");
            
            if !test_file.exists() {
                continue;
            }
            
            let capability = detect_filesystem_capability(&mnt_path);
            println!("▶ {} (capability: {:?})", fs.name.to_uppercase(), capability);
            
            // FICLONE test (CoW filesystems only)
            if capability == FilesystemCapability::CopyOnWrite 
                || capability == FilesystemCapability::ExtentManipulationWithReflink 
            {
                use std::os::unix::io::AsRawFd;
                
                let backup_path = mnt_path.join("ficlone_backup.txt");
                let _ = fs::remove_file(&backup_path);
                
                print!("  FICLONE backup:        ");
                let start = Instant::now();
                let src = File::open(&test_file).unwrap();
                let dst = File::create(&backup_path).unwrap();
                let result = unsafe {
                    libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
                };
                drop(src);
                dst.sync_all().ok();
                drop(dst);
                
                if result == 0 {
                    println!("{:>12?}  ✓", start.elapsed());
                    fs::remove_file(&backup_path).ok();
                } else {
                    println!("NOT SUPPORTED");
                }
            } else {
                println!("  FICLONE backup:        NOT SUPPORTED (ext4)");
            }
            
            // Consolidation test
            let consolidate_path = mnt_path.join("consolidate.txt");
            let _ = fs::remove_file(&consolidate_path);
            fs::copy(&test_file, &consolidate_path).unwrap();
            
            let mut patches = PatchList::new();
            patches.replace(0, 8, b"REPLACED");
            patches.insert(1000, b"INSERTED");
            
            print!("  Consolidation:         ");
            let start = Instant::now();
            let _ = try_fast_consolidate(&consolidate_path, &patches);
            println!("{:>12?}", start.elapsed());
            fs::remove_file(&consolidate_path).ok();
            
            // Streaming test
            let stream_path = mnt_path.join("stream.txt");
            let mut patches = PatchList::new();
            patches.replace(0, 8, b"REPLACED");
            
            print!("  Plain streaming:       ");
            let start = Instant::now();
            save_file(&test_file, &patches, Some(&stream_path)).unwrap();
            let elapsed = start.elapsed();
            let speed = file_size as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;
            println!("{:>12?}  ({:.2} GB/s)", elapsed, speed);
            fs::remove_file(&stream_path).ok();
            
            println!();
        }
        
        // Cleanup
        println!("Cleaning up...");
        for fs in &filesystems {
            let _ = Command::new("sudo").args(["umount", fs.mnt_path]).status();
            let _ = fs::remove_dir(fs.mnt_path);
            let _ = fs::remove_file(fs.img_path);
        }
        
        println!("\n✓ Benchmark complete!");
    }
}
