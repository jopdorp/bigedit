//! Journal-based persistence for near-instant saves
//!
//! Instead of rewriting the entire file on every save, we write patches to a small
//! journal file. This makes saves nearly instant regardless of file size.
//!
//! Journal format:
//! - Header: magic bytes + version + original file hash
//! - Entries: sequence of patches with checksums
//!
//! On open, if a journal exists, patches are loaded and applied to the view.
//! Full compaction (rewriting the file) happens only when:
//! - User explicitly requests it
//! - Journal grows too large (configurable threshold)
//! - File is being saved to a different location

use crate::patches::PatchList;
use crate::types::{FilePos, Patch};
use anyhow::{bail, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes identifying a bigedit journal file
const JOURNAL_MAGIC: &[u8; 8] = b"BIGEDITJ";

/// Current journal format version
const JOURNAL_VERSION: u32 = 1;

/// Maximum journal size before suggesting compaction (default: 10MB)
const DEFAULT_JOURNAL_COMPACT_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Get the journal path for a given file
pub fn journal_path(file_path: &Path) -> PathBuf {
    let mut journal = file_path.to_path_buf();
    let filename = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    journal.set_file_name(format!(".{}.bigedit-journal", filename));
    journal
}

/// Check if a journal exists for the given file
pub fn has_journal(file_path: &Path) -> bool {
    journal_path(file_path).exists()
}

/// Journal header stored at the start of the journal file
#[derive(Debug, Clone)]
struct JournalHeader {
    /// Original file size when journal was created
    original_size: u64,
    /// Simple checksum of first 4KB of original file for validation
    original_checksum: u32,
}

impl JournalHeader {
    fn new(original_size: u64, original_checksum: u32) -> Self {
        Self {
            original_size,
            original_checksum,
        }
    }

    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(JOURNAL_MAGIC)?;
        writer.write_all(&JOURNAL_VERSION.to_le_bytes())?;
        writer.write_all(&self.original_size.to_le_bytes())?;
        writer.write_all(&self.original_checksum.to_le_bytes())?;
        Ok(())
    }

    fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != JOURNAL_MAGIC {
            bail!("Invalid journal file (bad magic)");
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != JOURNAL_VERSION {
            bail!("Unsupported journal version: {}", version);
        }

        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let original_size = u64::from_le_bytes(size_bytes);

        let mut checksum_bytes = [0u8; 4];
        reader.read_exact(&mut checksum_bytes)?;
        let original_checksum = u32::from_le_bytes(checksum_bytes);

        Ok(Self {
            original_size,
            original_checksum,
        })
    }

    /// Size of header in bytes
    const SIZE: usize = 8 + 4 + 8 + 4; // magic + version + size + checksum
}

/// A single journal entry representing one save operation
#[derive(Debug, Clone)]
struct JournalEntry {
    /// Patches in this entry
    patches: Vec<Patch>,
}

impl JournalEntry {
    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write number of patches
        let count = self.patches.len() as u32;
        writer.write_all(&count.to_le_bytes())?;

        for patch in &self.patches {
            // Write patch: start (u64) + end (u64) + replacement_len (u32) + replacement bytes
            writer.write_all(&patch.start.to_le_bytes())?;
            writer.write_all(&patch.end.to_le_bytes())?;
            let repl_len = patch.replacement.len() as u32;
            writer.write_all(&repl_len.to_le_bytes())?;
            writer.write_all(&patch.replacement)?;
        }

        Ok(())
    }

    fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut patches = Vec::with_capacity(count);
        for _ in 0..count {
            let mut start_bytes = [0u8; 8];
            reader.read_exact(&mut start_bytes)?;
            let start = u64::from_le_bytes(start_bytes);

            let mut end_bytes = [0u8; 8];
            reader.read_exact(&mut end_bytes)?;
            let end = u64::from_le_bytes(end_bytes);

            let mut repl_len_bytes = [0u8; 4];
            reader.read_exact(&mut repl_len_bytes)?;
            let repl_len = u32::from_le_bytes(repl_len_bytes) as usize;

            let mut replacement = vec![0u8; repl_len];
            reader.read_exact(&mut replacement)?;

            patches.push(Patch::new(start, end, replacement));
        }

        Ok(Self { patches })
    }
}

/// Compute a simple checksum of the first N bytes of a file
fn compute_file_checksum(file: &mut File, max_bytes: usize) -> Result<u32> {
    file.seek(SeekFrom::Start(0))?;
    let mut buffer = vec![0u8; max_bytes.min(4096)];
    let read = file.read(&mut buffer)?;
    buffer.truncate(read);

    // Simple checksum (not cryptographic, just for detecting file changes)
    let mut sum: u32 = 0;
    for (i, &byte) in buffer.iter().enumerate() {
        sum = sum.wrapping_add((byte as u32).wrapping_mul((i as u32).wrapping_add(1)));
    }
    Ok(sum)
}

/// Save patches to journal file (near-instant operation)
pub fn save_to_journal(file_path: &Path, patches: &PatchList) -> Result<()> {
    if !patches.is_modified() {
        // If there are no patches, delete the journal if it exists
        let journal = journal_path(file_path);
        if journal.exists() {
            let _ = std::fs::remove_file(&journal);
        }
        return Ok(());
    }

    let journal = journal_path(file_path);

    // Always overwrite the journal with the complete current patch list.
    // This is simpler and more reliable than appending deltas, since
    // the PatchList already represents the complete state after reopening.
    create_journal(file_path, &journal, patches)?;

    Ok(())
}

/// Create a new journal file
fn create_journal(file_path: &Path, journal: &Path, patches: &PatchList) -> Result<()> {
    let mut original = File::open(file_path).context("Failed to open original file")?;
    let original_size = original.metadata()?.len();
    let checksum = compute_file_checksum(&mut original, 4096)?;

    let file = File::create(journal).context("Failed to create journal")?;
    let mut writer = BufWriter::new(file);

    // Write header
    let header = JournalHeader::new(original_size, checksum);
    header.write_to(&mut writer)?;

    // Write patches as first entry
    let entry = JournalEntry {
        patches: patches.patches().to_vec(),
    };
    entry.write_to(&mut writer)?;

    writer.flush()?;
    Ok(())
}

/// Append patches to existing journal
fn append_to_journal(journal: &Path, patches: &PatchList) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(journal)
        .context("Failed to open journal for append")?;

    let entry = JournalEntry {
        patches: patches.patches().to_vec(),
    };

    let mut writer = BufWriter::new(&mut file);
    entry.write_to(&mut writer)?;
    writer.flush()?;

    Ok(())
}

/// Load patches from journal file
pub fn load_from_journal(file_path: &Path) -> Result<Option<PatchList>> {
    let journal = journal_path(file_path);

    if !journal.exists() {
        return Ok(None);
    }

    // Validate the journal matches the current file
    let mut original = File::open(file_path).context("Failed to open original file")?;
    let current_size = original.metadata()?.len();
    let current_checksum = compute_file_checksum(&mut original, 4096)?;

    let file = File::open(&journal).context("Failed to open journal")?;
    let mut reader = BufReader::new(file);

    let header = JournalHeader::read_from(&mut reader)?;

    // Check if journal matches the file
    if header.original_size != current_size || header.original_checksum != current_checksum {
        // Journal doesn't match - file was modified externally
        // Delete the stale journal
        let _ = std::fs::remove_file(&journal);
        return Ok(None);
    }

    // Read all journal entries and merge patches
    let mut all_patches = PatchList::new();

    loop {
        match JournalEntry::read_from(&mut reader) {
            Ok(entry) => {
                for patch in entry.patches {
                    all_patches.add_patch(patch);
                }
            }
            Err(_) => break, // EOF or error - stop reading
        }
    }

    Ok(Some(all_patches))
}

/// Get the size of the journal file
pub fn journal_size(file_path: &Path) -> Option<u64> {
    let journal = journal_path(file_path);
    std::fs::metadata(&journal).ok().map(|m| m.len())
}

/// Check if journal should be compacted (has grown too large)
pub fn should_compact(file_path: &Path) -> bool {
    journal_size(file_path)
        .map(|size| size > DEFAULT_JOURNAL_COMPACT_THRESHOLD)
        .unwrap_or(false)
}

/// Delete the journal file (after successful compaction)
pub fn delete_journal(file_path: &Path) -> Result<()> {
    let journal = journal_path(file_path);
    if journal.exists() {
        std::fs::remove_file(&journal).context("Failed to delete journal")?;
    }
    Ok(())
}

/// Compact: rewrite the file with all patches applied and delete journal
/// 
/// Uses fast in-place consolidation on ext4/XFS when possible (O(patches)),
/// otherwise falls back to streaming rewrite (O(file_size)).
pub fn compact(file_path: &Path, patches: &PatchList) -> Result<()> {
    use crate::save::{save_file, try_fast_consolidate};

    // Try fast consolidation first (ext4/XFS extent operations)
    match try_fast_consolidate(file_path, patches) {
        Ok(true) => {
            // Fast consolidation succeeded
            delete_journal(file_path)?;
            return Ok(());
        }
        Ok(false) => {
            // Fast consolidation not available, fall through to streaming
        }
        Err(e) => {
            // Log error but continue with fallback
            eprintln!("[compact] Fast consolidation failed, using fallback: {}", e);
        }
    }

    // Fallback: Full streaming save to the file
    save_file(file_path, patches, None)?;

    // Delete the journal
    delete_journal(file_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_journal_path() {
        let path = Path::new("/tmp/myfile.sql");
        let journal = journal_path(path);
        assert_eq!(journal, Path::new("/tmp/.myfile.sql.bigedit-journal"));
    }

    #[test]
    fn test_journal_roundtrip() {
        let dir = tempdir().unwrap();
        let file_path = create_test_file(dir.path(), "test.txt", "hello world");

        // Create patches
        let mut patches = PatchList::new();
        patches.insert(5, b" beautiful");

        // Save to journal
        save_to_journal(&file_path, &patches).unwrap();

        // Verify journal exists
        assert!(has_journal(&file_path));

        // Load from journal
        let loaded = load_from_journal(&file_path).unwrap();
        assert!(loaded.is_some());

        let loaded_patches = loaded.unwrap();
        assert_eq!(loaded_patches.patches().len(), 1);
        assert_eq!(loaded_patches.patches()[0].start, 5);
        assert_eq!(loaded_patches.patches()[0].replacement, b" beautiful");
    }

    #[test]
    fn test_journal_detects_file_change() {
        let dir = tempdir().unwrap();
        let file_path = create_test_file(dir.path(), "test.txt", "hello world");

        // Create patches and journal
        let mut patches = PatchList::new();
        patches.insert(5, b" beautiful");
        save_to_journal(&file_path, &patches).unwrap();

        // Modify the original file externally
        std::fs::write(&file_path, "modified content!").unwrap();

        // Journal should be invalidated
        let loaded = load_from_journal(&file_path).unwrap();
        assert!(loaded.is_none());

        // Journal should be deleted
        assert!(!has_journal(&file_path));
    }

    #[test]
    fn test_journal_overwrite() {
        let dir = tempdir().unwrap();
        let file_path = create_test_file(dir.path(), "test.txt", "hello world");

        // First save with one patch
        let mut patches1 = PatchList::new();
        patches1.insert(5, b" beautiful");
        save_to_journal(&file_path, &patches1).unwrap();

        // Verify first save
        let loaded1 = load_from_journal(&file_path).unwrap().unwrap();
        assert_eq!(loaded1.patches().len(), 1);

        // Second save overwrites with new complete state
        // (simulating: user reopened file, patches loaded, made more edits)
        let mut patches2 = load_from_journal(&file_path).unwrap().unwrap();
        patches2.insert(0, b"Say: ");
        save_to_journal(&file_path, &patches2).unwrap();

        // Load should have both patches (complete state)
        let loaded = load_from_journal(&file_path).unwrap().unwrap();
        assert_eq!(loaded.patches().len(), 2);
    }
}
