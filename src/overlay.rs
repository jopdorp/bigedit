//! OverlayFS-based persistence for zero-overhead saves
//!
//! This module uses Linux's OverlayFS to provide instant saves. The original file
//! remains unchanged (lower layer), and all modifications are written to a small
//! overlay directory (upper layer). The merged view shows the edited file.
//!
//! Benefits:
//! - True zero-copy: kernel handles all the magic
//! - Instant saves: just write the changed bytes to upper layer
//! - Atomic: overlay mount/unmount is atomic
//! - No journal management needed
//!
//! Requirements:
//! - Linux with OverlayFS support (kernel 3.18+)
//! - fusermount3 or root privileges for mounting
//! - fuse-overlayfs package for unprivileged mounts

use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Directory structure for overlay
/// .bigedit-overlay/
///   lower/      - symlink to original file's directory (read-only)
///   upper/      - modified files stored here
///   work/       - overlayfs workdir
///   merged/     - the merged view (mount point)
const OVERLAY_DIR_NAME: &str = ".bigedit-overlay";

/// Check if fuse-overlayfs is available (for unprivileged mounts)
pub fn has_fuse_overlayfs() -> bool {
    Command::new("which")
        .arg("fuse-overlayfs")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if we have root privileges (for kernel overlayfs)
pub fn has_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

/// Check if overlay support is available
pub fn is_available() -> bool {
    has_fuse_overlayfs() || has_root()
}

/// Overlay session for a file
pub struct OverlaySession {
    /// Original file path
    original_path: PathBuf,
    /// Overlay directory (contains lower, upper, work, merged)
    overlay_dir: PathBuf,
    /// Path to the file in merged view
    merged_file_path: PathBuf,
    /// Whether overlay is currently mounted
    is_mounted: bool,
    /// Whether using fuse-overlayfs (vs kernel overlayfs)
    use_fuse: bool,
}

impl OverlaySession {
    /// Create a new overlay session for the given file
    pub fn new(file_path: &Path) -> Result<Self> {
        let original_path = file_path.canonicalize()
            .context("Failed to get absolute path")?;
        
        let parent = original_path.parent()
            .context("File has no parent directory")?;
        
        let filename = original_path.file_name()
            .context("File has no name")?;
        
        // Create overlay directory structure next to the file
        let overlay_dir = parent.join(format!("{}.{}", 
            filename.to_string_lossy(), OVERLAY_DIR_NAME));
        
        let merged_file_path = overlay_dir.join("merged").join(filename);
        
        let use_fuse = !has_root() && has_fuse_overlayfs();
        
        Ok(Self {
            original_path,
            overlay_dir,
            merged_file_path,
            is_mounted: false,
            use_fuse,
        })
    }

    /// Set up and mount the overlay filesystem
    pub fn mount(&mut self) -> Result<()> {
        if self.is_mounted {
            return Ok(());
        }

        // Create directory structure
        let lower = self.overlay_dir.join("lower");
        let upper = self.overlay_dir.join("upper");
        let work = self.overlay_dir.join("work");
        let merged = self.overlay_dir.join("merged");

        fs::create_dir_all(&lower)?;
        fs::create_dir_all(&upper)?;
        fs::create_dir_all(&work)?;
        fs::create_dir_all(&merged)?;

        // Symlink original file's directory as lower
        let original_dir = self.original_path.parent().unwrap();
        let lower_link = lower.join("source");
        if !lower_link.exists() {
            std::os::unix::fs::symlink(original_dir, &lower_link)?;
        }

        // For overlayfs, we need to set up differently
        // Copy original file to lower (overlayfs needs actual files, not symlinks for the lower)
        let original_filename = self.original_path.file_name().unwrap();
        let lower_file = lower.join(original_filename);
        
        if !lower_file.exists() {
            // Create a hard link if possible, otherwise we need the file to be accessible
            // For now, we'll work with the original directly by adjusting our approach
        }

        if self.use_fuse {
            self.mount_fuse_overlay(&lower, &upper, &work, &merged)?;
        } else {
            self.mount_kernel_overlay(&lower, &upper, &work, &merged)?;
        }

        self.is_mounted = true;
        Ok(())
    }

    /// Mount using fuse-overlayfs (unprivileged)
    fn mount_fuse_overlay(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        merged: &Path,
    ) -> Result<()> {
        let original_dir = self.original_path.parent().unwrap();
        
        let status = Command::new("fuse-overlayfs")
            .arg("-o")
            .arg(format!(
                "lowerdir={},upperdir={},workdir={}",
                original_dir.display(),
                upper.display(),
                work.display()
            ))
            .arg(merged)
            .status()
            .context("Failed to run fuse-overlayfs")?;

        if !status.success() {
            bail!("fuse-overlayfs mount failed");
        }

        Ok(())
    }

    /// Mount using kernel overlayfs (requires root)
    fn mount_kernel_overlay(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        merged: &Path,
    ) -> Result<()> {
        let original_dir = self.original_path.parent().unwrap();
        
        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            original_dir.display(),
            upper.display(),
            work.display()
        );

        let status = Command::new("mount")
            .arg("-t")
            .arg("overlay")
            .arg("overlay")
            .arg("-o")
            .arg(&options)
            .arg(merged)
            .status()
            .context("Failed to run mount")?;

        if !status.success() {
            bail!("Kernel overlayfs mount failed (requires root)");
        }

        Ok(())
    }

    /// Unmount the overlay
    pub fn unmount(&mut self) -> Result<()> {
        if !self.is_mounted {
            return Ok(());
        }

        let merged = self.overlay_dir.join("merged");

        let status = if self.use_fuse {
            Command::new("fusermount3")
                .arg("-u")
                .arg(&merged)
                .status()
                .or_else(|_| {
                    Command::new("fusermount")
                        .arg("-u")
                        .arg(&merged)
                        .status()
                })
                .context("Failed to unmount fuse-overlayfs")?
        } else {
            Command::new("umount")
                .arg(&merged)
                .status()
                .context("Failed to unmount overlayfs")?
        };

        if !status.success() {
            bail!("Unmount failed");
        }

        self.is_mounted = false;
        Ok(())
    }

    /// Get the path to the merged file view
    pub fn merged_path(&self) -> &Path {
        &self.merged_file_path
    }

    /// Get the path to the upper (modified) layer
    pub fn upper_dir(&self) -> PathBuf {
        self.overlay_dir.join("upper")
    }

    /// Check if there are modifications in the upper layer
    pub fn has_modifications(&self) -> bool {
        let upper = self.upper_dir();
        if let Ok(entries) = fs::read_dir(&upper) {
            entries.count() > 0
        } else {
            false
        }
    }

    /// Commit changes: replace original with merged content
    pub fn commit(&mut self) -> Result<()> {
        if !self.has_modifications() {
            return Ok(());
        }

        // Read from merged, write to temp, rename to original
        let temp_path = self.original_path.with_extension("bigedit-commit-tmp");
        
        {
            let mut merged_file = File::open(&self.merged_file_path)
                .context("Failed to open merged file")?;
            let mut temp_file = File::create(&temp_path)
                .context("Failed to create temp file")?;
            
            std::io::copy(&mut merged_file, &mut temp_file)?;
            temp_file.sync_all()?;
        }

        // Unmount first
        self.unmount()?;

        // Atomic rename
        fs::rename(&temp_path, &self.original_path)
            .context("Failed to replace original file")?;

        // Clean up overlay directory
        self.cleanup()?;

        Ok(())
    }

    /// Discard changes and clean up
    pub fn discard(&mut self) -> Result<()> {
        self.unmount()?;
        self.cleanup()?;
        Ok(())
    }

    /// Clean up overlay directory
    fn cleanup(&self) -> Result<()> {
        if self.overlay_dir.exists() {
            fs::remove_dir_all(&self.overlay_dir)
                .context("Failed to remove overlay directory")?;
        }
        Ok(())
    }

    /// Check if overlay is mounted
    pub fn is_mounted(&self) -> bool {
        self.is_mounted
    }
}

impl Drop for OverlaySession {
    fn drop(&mut self) {
        // Try to unmount on drop, but don't panic
        let _ = self.unmount();
    }
}

/// A simpler alternative: use a sparse copy-on-write file
/// This doesn't require overlayfs but achieves similar goals using
/// reflinks (on filesystems that support them like btrfs, xfs, APFS)
pub struct ReflinkSession {
    /// Original file path
    original_path: PathBuf,
    /// Working copy (reflinked)
    work_path: PathBuf,
    /// Whether we have a working copy
    has_copy: bool,
}

impl ReflinkSession {
    /// Create a new reflink session
    pub fn new(file_path: &Path) -> Result<Self> {
        let original_path = file_path.canonicalize()?;
        let work_path = original_path.with_extension("bigedit-work");
        
        Ok(Self {
            original_path,
            work_path,
            has_copy: false,
        })
    }

    /// Check if reflinks are supported
    pub fn is_available(path: &Path) -> bool {
        // Try to create a small test file and reflink it
        let test_src = path.with_extension("bigedit-reflink-test-src");
        let test_dst = path.with_extension("bigedit-reflink-test-dst");
        
        let result = (|| -> Result<bool> {
            fs::write(&test_src, b"test")?;
            
            let status = Command::new("cp")
                .arg("--reflink=always")
                .arg(&test_src)
                .arg(&test_dst)
                .status()?;
            
            Ok(status.success())
        })();
        
        let _ = fs::remove_file(&test_src);
        let _ = fs::remove_file(&test_dst);
        
        result.unwrap_or(false)
    }

    /// Create a reflinked working copy (instant, no data copying)
    pub fn create_work_copy(&mut self) -> Result<()> {
        if self.has_copy {
            return Ok(());
        }

        let status = Command::new("cp")
            .arg("--reflink=always")
            .arg(&self.original_path)
            .arg(&self.work_path)
            .status()
            .context("Failed to create reflink copy")?;

        if !status.success() {
            bail!("Reflink copy failed (filesystem may not support reflinks)");
        }

        self.has_copy = true;
        Ok(())
    }

    /// Get the working copy path
    pub fn work_path(&self) -> &Path {
        &self.work_path
    }

    /// Commit: rename work copy to original
    pub fn commit(&mut self) -> Result<()> {
        if !self.has_copy {
            return Ok(());
        }

        fs::rename(&self.work_path, &self.original_path)
            .context("Failed to replace original")?;
        
        self.has_copy = false;
        Ok(())
    }

    /// Discard changes
    pub fn discard(&mut self) -> Result<()> {
        if self.has_copy {
            let _ = fs::remove_file(&self.work_path);
            self.has_copy = false;
        }
        Ok(())
    }
}

impl Drop for ReflinkSession {
    fn drop(&mut self) {
        // Clean up work file on drop
        if self.has_copy {
            let _ = fs::remove_file(&self.work_path);
        }
    }
}

/// Determine the best available save strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SaveStrategy {
    /// Use OverlayFS (best: true copy-on-write at kernel level)
    Overlay,
    /// Use reflinks (good: instant copy, CoW at filesystem level)
    Reflink,
    /// Use journal (fallback: patch-based, works everywhere)
    Journal,
}

impl SaveStrategy {
    /// Detect the best available strategy for a given file
    pub fn detect(file_path: &Path) -> Self {
        if is_available() {
            SaveStrategy::Overlay
        } else if ReflinkSession::is_available(file_path) {
            SaveStrategy::Reflink
        } else {
            SaveStrategy::Journal
        }
    }

    /// Get a description of the strategy
    pub fn description(&self) -> &'static str {
        match self {
            SaveStrategy::Overlay => "OverlayFS (kernel CoW)",
            SaveStrategy::Reflink => "Reflink (filesystem CoW)", 
            SaveStrategy::Journal => "Journal (patch-based)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_detection() {
        // Just verify detection doesn't crash
        let strategy = SaveStrategy::detect(Path::new("/tmp/test.txt"));
        println!("Detected strategy: {:?}", strategy);
    }

    #[test]
    fn test_overlay_session_new() {
        let session = OverlaySession::new(Path::new("/tmp/test.txt"));
        // May fail if /tmp/test.txt doesn't exist, that's OK
        if let Ok(session) = session {
            assert!(!session.is_mounted());
        }
    }
}
