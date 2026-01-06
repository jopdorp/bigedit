//! Search functionality for bigedit
//!
//! This module provides two search modes:
//! 1. Viewport search - instant search within loaded content
//! 2. Streaming search - forward search through the entire file

use crate::patches::PatchList;
use crate::types::FilePos;
use anyhow::{Context, Result};
use memchr::memmem;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Size of chunks for streaming search (8MB)
const SEARCH_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Result of a search operation
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Byte offset in the original file
    pub position: FilePos,
    /// Length of the match in bytes
    pub length: usize,
}

impl SearchResult {
    pub fn new(position: FilePos, length: usize) -> Self {
        Self { position, length }
    }
}

/// Search within a byte buffer
pub fn search_in_buffer(buffer: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return vec![];
    }

    let finder = memmem::Finder::new(pattern);
    finder.find_iter(buffer).collect()
}

/// Search within a byte buffer, case insensitive
pub fn search_in_buffer_case_insensitive(buffer: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return vec![];
    }

    // Convert both to lowercase for comparison
    let buffer_lower: Vec<u8> = buffer.iter().map(|b| b.to_ascii_lowercase()).collect();
    let pattern_lower: Vec<u8> = pattern.iter().map(|b| b.to_ascii_lowercase()).collect();

    let finder = memmem::Finder::new(&pattern_lower);
    finder.find_iter(&buffer_lower).collect()
}

/// Find the first match in a buffer
pub fn find_first_in_buffer(buffer: &[u8], pattern: &[u8]) -> Option<usize> {
    if pattern.is_empty() {
        return None;
    }

    let finder = memmem::Finder::new(pattern);
    finder.find(buffer)
}

/// Find the first match in a buffer, case insensitive
pub fn find_first_case_insensitive(buffer: &[u8], pattern: &[u8]) -> Option<usize> {
    if pattern.is_empty() {
        return None;
    }

    let buffer_lower: Vec<u8> = buffer.iter().map(|b| b.to_ascii_lowercase()).collect();
    let pattern_lower: Vec<u8> = pattern.iter().map(|b| b.to_ascii_lowercase()).collect();

    let finder = memmem::Finder::new(&pattern_lower);
    finder.find(&buffer_lower)
}

/// Streaming forward search through a file
///
/// Searches from `start_pos` forward, applying patches as needed.
/// Returns the first match found, or None if no match.
pub fn streaming_search_forward(
    file: &mut File,
    file_length: u64,
    start_pos: FilePos,
    pattern: &[u8],
    patches: &PatchList,
    case_insensitive: bool,
) -> Result<Option<SearchResult>> {
    if pattern.is_empty() {
        return Ok(None);
    }

    let pattern_len = pattern.len();
    let overlap = pattern_len.saturating_sub(1);

    let mut pos = start_pos;
    let mut carry_over = Vec::new();

    while pos < file_length {
        // Seek to position
        file.seek(SeekFrom::Start(pos))
            .context("Failed to seek during search")?;

        // Read a chunk
        let to_read = ((file_length - pos) as usize).min(SEARCH_CHUNK_SIZE);
        let mut buffer = vec![0u8; to_read];
        let bytes_read = file.read(&mut buffer)
            .context("Failed to read during search")?;
        buffer.truncate(bytes_read);

        if bytes_read == 0 {
            break;
        }

        // Apply patches to this chunk
        let chunk_end = pos + bytes_read as u64;
        let patched_buffer = apply_patches_for_search(&buffer, pos, patches);

        // Combine with carry over from previous chunk
        let search_buffer = if carry_over.is_empty() {
            patched_buffer
        } else {
            let mut combined = carry_over.clone();
            combined.extend_from_slice(&patched_buffer);
            combined
        };

        // Search in buffer
        let find_result = if case_insensitive {
            find_first_case_insensitive(&search_buffer, pattern)
        } else {
            find_first_in_buffer(&search_buffer, pattern)
        };

        if let Some(offset) = find_result {
            // Account for carry over offset
            let carry_len = carry_over.len();
            let actual_offset = if offset >= carry_len {
                offset - carry_len
            } else {
                // Match spans carry over boundary
                0
            };

            let match_pos = pos + actual_offset as u64 - if offset < carry_len {
                (carry_len - offset) as u64
            } else {
                0
            };

            return Ok(Some(SearchResult::new(match_pos, pattern_len)));
        }

        // Keep overlap for next chunk
        carry_over = if search_buffer.len() > overlap {
            search_buffer[search_buffer.len() - overlap..].to_vec()
        } else {
            search_buffer
        };

        pos = chunk_end;
    }

    Ok(None)
}

/// Streaming backward search through a file
pub fn streaming_search_backward(
    file: &mut File,
    file_length: u64,
    start_pos: FilePos,
    pattern: &[u8],
    patches: &PatchList,
    case_insensitive: bool,
) -> Result<Option<SearchResult>> {
    if pattern.is_empty() || start_pos == 0 {
        return Ok(None);
    }

    let pattern_len = pattern.len();
    let overlap = pattern_len.saturating_sub(1);

    let mut pos = start_pos;
    let mut carry_over = Vec::new();

    while pos > 0 {
        // Calculate chunk start
        let chunk_start = pos.saturating_sub(SEARCH_CHUNK_SIZE as u64);
        let chunk_size = (pos - chunk_start) as usize;

        // Seek and read
        file.seek(SeekFrom::Start(chunk_start))
            .context("Failed to seek during backward search")?;

        let mut buffer = vec![0u8; chunk_size];
        let bytes_read = file.read(&mut buffer)
            .context("Failed to read during backward search")?;
        buffer.truncate(bytes_read);

        // Apply patches
        let patched_buffer = apply_patches_for_search(&buffer, chunk_start, patches);

        // Combine with carry over (prepend for backward search)
        let search_buffer = if carry_over.is_empty() {
            patched_buffer
        } else {
            let mut combined = patched_buffer.clone();
            combined.extend_from_slice(&carry_over);
            combined
        };

        // Find last match in buffer
        let matches = if case_insensitive {
            search_in_buffer_case_insensitive(&search_buffer, pattern)
        } else {
            search_in_buffer(&search_buffer, pattern)
        };

        if let Some(&offset) = matches.last() {
            let match_pos = chunk_start + offset as u64;
            return Ok(Some(SearchResult::new(match_pos, pattern_len)));
        }

        // Keep overlap for next chunk
        carry_over = if search_buffer.len() > overlap {
            search_buffer[..overlap].to_vec()
        } else {
            search_buffer
        };

        pos = chunk_start;
    }

    Ok(None)
}

/// Apply patches to a buffer for search purposes
/// This is a simplified version that just returns patched content
fn apply_patches_for_search(buffer: &[u8], buffer_start: FilePos, patches: &PatchList) -> Vec<u8> {
    let buffer_end = buffer_start + buffer.len() as u64;

    // Find patches affecting this range
    let relevant_patches: Vec<_> = patches
        .patches()
        .iter()
        .filter(|p| p.start < buffer_end && p.end > buffer_start)
        .collect();

    if relevant_patches.is_empty() {
        return buffer.to_vec();
    }

    // Apply patches (simplified - just replace ranges)
    let mut result = Vec::new();
    let mut pos = buffer_start;

    for patch in relevant_patches {
        // Copy original content before patch
        if patch.start > pos {
            let copy_start = (pos - buffer_start) as usize;
            let copy_end = ((patch.start - buffer_start) as usize).min(buffer.len());
            if copy_start < copy_end {
                result.extend_from_slice(&buffer[copy_start..copy_end]);
            }
        }

        // Add patch replacement
        result.extend_from_slice(&patch.replacement);
        pos = patch.end.max(pos);
    }

    // Copy remaining
    if pos < buffer_end {
        let copy_start = (pos - buffer_start) as usize;
        if copy_start < buffer.len() {
            result.extend_from_slice(&buffer[copy_start..]);
        }
    }

    result
}

/// Search configuration
#[derive(Debug, Clone, Default)]
pub struct SearchConfig {
    pub pattern: Vec<u8>,
    pub case_insensitive: bool,
    pub wrap_around: bool,
}

impl SearchConfig {
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.as_bytes().to_vec(),
            case_insensitive: false,
            wrap_around: true,
        }
    }

    pub fn case_insensitive(mut self, value: bool) -> Self {
        self.case_insensitive = value;
        self
    }

    pub fn wrap_around(mut self, value: bool) -> Self {
        self.wrap_around = value;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_search_in_buffer() {
        let buffer = b"hello world hello";
        let pattern = b"hello";

        let results = search_in_buffer(buffer, pattern);
        assert_eq!(results, vec![0, 12]);
    }

    #[test]
    fn test_search_case_insensitive() {
        let buffer = b"Hello World HELLO";
        let pattern = b"hello";

        let results = search_in_buffer_case_insensitive(buffer, pattern);
        assert_eq!(results, vec![0, 12]);
    }

    #[test]
    fn test_find_first() {
        let buffer = b"the quick brown fox";
        let pattern = b"quick";

        assert_eq!(find_first_in_buffer(buffer, pattern), Some(4));
        assert_eq!(find_first_in_buffer(buffer, b"cat"), None);
    }

    #[test]
    fn test_streaming_search_forward() {
        let temp = create_test_file("hello world hello universe");
        let mut file = File::open(temp.path()).unwrap();
        let file_length = file.metadata().unwrap().len();
        let patches = PatchList::new();

        // Search from beginning
        let result = streaming_search_forward(
            &mut file,
            file_length,
            0,
            b"hello",
            &patches,
            false,
        )
        .unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().position, 0);

        // Search from middle
        let result = streaming_search_forward(
            &mut file,
            file_length,
            6,
            b"hello",
            &patches,
            false,
        )
        .unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().position, 12);
    }

    #[test]
    fn test_streaming_search_not_found() {
        let temp = create_test_file("hello world");
        let mut file = File::open(temp.path()).unwrap();
        let file_length = file.metadata().unwrap().len();
        let patches = PatchList::new();

        let result = streaming_search_forward(
            &mut file,
            file_length,
            0,
            b"goodbye",
            &patches,
            false,
        )
        .unwrap();

        assert!(result.is_none());
    }
}
