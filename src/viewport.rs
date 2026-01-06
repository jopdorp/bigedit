//! Viewport management for bigedit
//!
//! This module handles loading and managing a viewport (window) into a large file.
//! The viewport loads a small slice of the file around the cursor position.

use crate::patches::{apply_patches_to_slice, PatchList};
use crate::types::{FilePos, LineSpan, MappingSegment, ViewportMapping};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Default viewport size in bytes (4MB)
pub const DEFAULT_VIEWPORT_SIZE: usize = 4 * 1024 * 1024;

/// Margin in bytes to keep loaded before/after visible area
pub const VIEWPORT_MARGIN: usize = 64 * 1024;

/// Maximum line length to display (prevents UI lock on huge lines)
pub const MAX_LINE_DISPLAY_WIDTH: usize = 4096;

/// Minimum lines to load in viewport
pub const MIN_VIEWPORT_LINES: usize = 100;

/// A viewport into a large file
#[derive(Debug)]
pub struct Viewport {
    /// Byte offset in original file where viewport starts
    pub start: FilePos,
    /// Byte offset in original file where viewport ends (exclusive)
    pub end: FilePos,
    /// Raw bytes loaded from the original file
    pub bytes: Vec<u8>,
    /// Rendered bytes after applying patches
    pub render_bytes: Vec<u8>,
    /// Parsed line boundaries within render_bytes
    pub lines: Vec<LineSpan>,
    /// Mapping from render positions to original positions
    pub mapping: ViewportMapping,
    /// Total file length
    pub file_length: u64,
}

impl Viewport {
    /// Create a new empty viewport
    pub fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            bytes: Vec::new(),
            render_bytes: Vec::new(),
            lines: Vec::new(),
            mapping: ViewportMapping::new(),
            file_length: 0,
        }
    }

    /// Load viewport from a file starting at the given position
    pub fn load(
        file: &mut File,
        file_length: u64,
        start: FilePos,
        max_bytes: usize,
        patches: &PatchList,
    ) -> Result<Self> {
        // Clamp start to file bounds
        let start = start.min(file_length);

        // Calculate how many bytes to read
        let bytes_to_read = (file_length - start).min(max_bytes as u64) as usize;

        // Seek and read
        file.seek(SeekFrom::Start(start))
            .context("Failed to seek in file")?;

        let mut bytes = vec![0u8; bytes_to_read];
        let bytes_read = file.read(&mut bytes).context("Failed to read file")?;
        bytes.truncate(bytes_read);

        let end = start + bytes_read as u64;

        // Apply patches to get render bytes
        // Note: Use <= for start comparison to include insert patches at the end (where start == end)
        let relevant_patches: Vec<_> = patches
            .patches()
            .iter()
            .filter(|p| p.start <= end && p.end >= start)
            .cloned()
            .collect();

        let (render_bytes, mapping_entries) = apply_patches_to_slice(&bytes, start, &relevant_patches);

        // Build viewport mapping from mapping entries
        let mut mapping = ViewportMapping::new();
        for entry in mapping_entries {
            let segment = if let Some(orig) = entry.original_start {
                MappingSegment::from_original(
                    entry.render_start,
                    entry.render_start + entry.render_len,
                    orig,
                )
            } else {
                MappingSegment::inserted(entry.render_start, entry.render_start + entry.render_len)
            };
            mapping.push(segment);
        }

        // Parse line boundaries
        let lines = parse_lines(&render_bytes);

        Ok(Self {
            start,
            end,
            bytes,
            render_bytes,
            lines,
            mapping,
            file_length,
        })
    }

    /// Load viewport from a file path
    pub fn load_from_path(
        path: &Path,
        start: FilePos,
        max_bytes: usize,
        patches: &PatchList,
    ) -> Result<Self> {
        let mut file = File::open(path).context("Failed to open file")?;
        let metadata = file.metadata().context("Failed to get file metadata")?;
        let file_length = metadata.len();

        Self::load(&mut file, file_length, start, max_bytes, patches)
    }

    /// Reload the viewport with current patches
    pub fn reload(&mut self, file: &mut File, patches: &PatchList) -> Result<()> {
        let new_viewport = Self::load(
            file,
            self.file_length,
            self.start,
            self.bytes.capacity().max(DEFAULT_VIEWPORT_SIZE),
            patches,
        )?;
        *self = new_viewport;
        Ok(())
    }

    /// Check if a position is within the loaded viewport
    pub fn contains(&self, pos: FilePos) -> bool {
        pos >= self.start && pos < self.end
    }

    /// Check if viewport needs to be shifted (cursor near edge)
    pub fn needs_shift(&self, cursor_pos: FilePos, margin: usize) -> bool {
        if cursor_pos < self.start + margin as u64 && self.start > 0 {
            return true;
        }
        if cursor_pos + margin as u64 > self.end && self.end < self.file_length {
            return true;
        }
        false
    }

    /// Get the number of lines in the viewport
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get a line's content as bytes
    pub fn line_bytes(&self, line_idx: usize) -> Option<&[u8]> {
        self.lines.get(line_idx).map(|span| {
            &self.render_bytes[span.start_idx..span.end_idx]
        })
    }

    /// Get a line's content as a string (lossy UTF-8 conversion)
    pub fn line_str(&self, line_idx: usize) -> Option<String> {
        self.line_bytes(line_idx)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// Get visible lines for display (with max width limit)
    pub fn visible_lines(&self, start_line: usize, count: usize) -> Vec<String> {
        (start_line..start_line + count)
            .filter_map(|i| self.line_str(i))
            .map(|s| {
                if s.len() > MAX_LINE_DISPLAY_WIDTH {
                    format!("{}...", &s[..MAX_LINE_DISPLAY_WIDTH - 3])
                } else {
                    s
                }
            })
            .collect()
    }

    /// Convert a (row, col) position to a byte offset in render_bytes
    pub fn row_col_to_render_byte(&self, row: usize, col: usize) -> Option<usize> {
        let line = self.lines.get(row)?;
        let line_bytes = &self.render_bytes[line.start_idx..line.end_idx];

        // Convert column (grapheme index) to byte offset within line
        let byte_offset = grapheme_to_byte_offset(line_bytes, col);
        Some(line.start_idx + byte_offset)
    }

    /// Convert a render byte offset to (row, col) position
    pub fn render_byte_to_row_col(&self, byte_offset: usize) -> Option<(usize, usize)> {
        for (row, line) in self.lines.iter().enumerate() {
            let line_end = if line.has_newline {
                line.end_idx + 1
            } else {
                line.end_idx
            };

            if byte_offset >= line.start_idx && byte_offset <= line_end {
                let offset_in_line = byte_offset.saturating_sub(line.start_idx);
                let line_bytes = &self.render_bytes[line.start_idx..line.end_idx];
                let col = byte_to_grapheme_offset(line_bytes, offset_in_line);
                return Some((row, col));
            }
        }
        None
    }

    /// Get the byte offset in render_bytes for the start of a line
    pub fn line_start_byte(&self, line_idx: usize) -> Option<usize> {
        self.lines.get(line_idx).map(|span| span.start_idx)
    }

    /// Get the byte offset in render_bytes for the end of a line
    pub fn line_end_byte(&self, line_idx: usize) -> Option<usize> {
        self.lines.get(line_idx).map(|span| span.end_idx)
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse line boundaries from bytes
pub fn parse_lines(bytes: &[u8]) -> Vec<LineSpan> {
    let mut lines = Vec::new();
    let mut line_start = 0;

    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            lines.push(LineSpan::new(line_start, i, true));
            line_start = i + 1;
        }
    }

    // Handle last line without newline
    if line_start <= bytes.len() {
        let has_newline = false;
        lines.push(LineSpan::new(line_start, bytes.len(), has_newline));
    }

    lines
}

/// Convert a grapheme cluster index to a byte offset
fn grapheme_to_byte_offset(bytes: &[u8], grapheme_idx: usize) -> usize {
    use unicode_segmentation::UnicodeSegmentation;

    let s = String::from_utf8_lossy(bytes);
    let mut byte_offset = 0;

    for (idx, grapheme) in s.graphemes(true).enumerate() {
        if idx >= grapheme_idx {
            break;
        }
        byte_offset += grapheme.len();
    }

    byte_offset.min(bytes.len())
}

/// Convert a byte offset to a grapheme cluster index
fn byte_to_grapheme_offset(bytes: &[u8], byte_offset: usize) -> usize {
    use unicode_segmentation::UnicodeSegmentation;

    let s = String::from_utf8_lossy(bytes);
    let mut current_byte = 0;
    let mut grapheme_idx = 0;

    for grapheme in s.graphemes(true) {
        if current_byte >= byte_offset {
            break;
        }
        current_byte += grapheme.len();
        grapheme_idx += 1;
    }

    grapheme_idx
}

/// Count grapheme clusters in a byte slice
pub fn count_graphemes(bytes: &[u8]) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    let s = String::from_utf8_lossy(bytes);
    s.graphemes(true).count()
}

/// Get the display width of a string (accounting for wide characters)
pub fn display_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    s.width()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lines_simple() {
        let bytes = b"hello\nworld\n";
        let lines = parse_lines(bytes);

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], LineSpan::new(0, 5, true));
        assert_eq!(lines[1], LineSpan::new(6, 11, true));
        assert_eq!(lines[2], LineSpan::new(12, 12, false)); // empty last line
    }

    #[test]
    fn test_parse_lines_no_trailing_newline() {
        let bytes = b"hello\nworld";
        let lines = parse_lines(bytes);

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], LineSpan::new(0, 5, true));
        assert_eq!(lines[1], LineSpan::new(6, 11, false));
    }

    #[test]
    fn test_parse_lines_empty() {
        let bytes = b"";
        let lines = parse_lines(bytes);

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], LineSpan::new(0, 0, false));
    }

    #[test]
    fn test_grapheme_conversion() {
        // ASCII
        let bytes = b"hello";
        assert_eq!(grapheme_to_byte_offset(bytes, 2), 2);
        assert_eq!(byte_to_grapheme_offset(bytes, 2), 2);

        // UTF-8 with multi-byte chars
        let bytes = "héllo".as_bytes();
        assert_eq!(grapheme_to_byte_offset(bytes, 2), 3); // 'h' + 'é' (2 bytes)
        assert_eq!(byte_to_grapheme_offset(bytes, 3), 2);
    }

    #[test]
    fn test_count_graphemes() {
        assert_eq!(count_graphemes(b"hello"), 5);
        assert_eq!(count_graphemes("héllo".as_bytes()), 5);
        assert_eq!(count_graphemes("👨‍👩‍👧".as_bytes()), 1); // Family emoji is one grapheme
    }
}
