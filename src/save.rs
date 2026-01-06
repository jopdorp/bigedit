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
}
