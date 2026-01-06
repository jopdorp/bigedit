//! Core data types for bigedit
//!
//! This module defines the fundamental types used throughout the editor,
//! including coordinates, patches, pieces, and viewport mappings.

use std::ops::Range;

/// Byte offset in the original file (canonical coordinate system)
pub type FilePos = u64;

/// A span representing a line within a byte buffer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSpan {
    /// Start index within the buffer (inclusive)
    pub start_idx: usize,
    /// End index within the buffer (exclusive, before newline)
    pub end_idx: usize,
    /// Whether this line ends with a newline character
    pub has_newline: bool,
}

impl LineSpan {
    pub fn new(start_idx: usize, end_idx: usize, has_newline: bool) -> Self {
        Self {
            start_idx,
            end_idx,
            has_newline,
        }
    }

    /// Returns the length of the line content (excluding newline)
    pub fn len(&self) -> usize {
        self.end_idx - self.start_idx
    }

    /// Returns true if the line is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the full length including newline if present
    pub fn full_len(&self) -> usize {
        self.len() + if self.has_newline { 1 } else { 0 }
    }
}

/// A patch representing a replacement of bytes in the original file
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    /// Inclusive byte offset in original file where patch starts
    pub start: FilePos,
    /// Exclusive byte offset in original file where patch ends
    pub end: FilePos,
    /// Bytes to insert instead of original[start..end]
    pub replacement: Vec<u8>,
}

impl Patch {
    pub fn new(start: FilePos, end: FilePos, replacement: Vec<u8>) -> Self {
        Self {
            start,
            end,
            replacement,
        }
    }

    /// Create an insertion patch (inserts bytes at position without removing anything)
    pub fn insert(pos: FilePos, bytes: Vec<u8>) -> Self {
        Self::new(pos, pos, bytes)
    }

    /// Create a deletion patch (removes bytes without inserting anything)
    pub fn delete(start: FilePos, end: FilePos) -> Self {
        Self::new(start, end, Vec::new())
    }

    /// Returns the original range this patch affects
    pub fn original_range(&self) -> Range<FilePos> {
        self.start..self.end
    }

    /// Returns the length of bytes removed from original
    pub fn deleted_len(&self) -> u64 {
        self.end - self.start
    }

    /// Returns the length of bytes inserted
    pub fn inserted_len(&self) -> usize {
        self.replacement.len()
    }

    /// Returns the net change in document length from this patch
    pub fn length_delta(&self) -> i64 {
        self.inserted_len() as i64 - self.deleted_len() as i64
    }

    /// Check if this patch overlaps or touches another patch
    pub fn overlaps_or_touches(&self, other: &Patch) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Check if this patch strictly overlaps another patch
    pub fn overlaps(&self, other: &Patch) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if a position falls within this patch's original range
    pub fn contains_pos(&self, pos: FilePos) -> bool {
        pos >= self.start && pos < self.end
    }
}

/// Source of bytes in a piece table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceSource {
    /// Bytes come from the original file
    Original,
    /// Bytes come from the add buffer (inserted text)
    Add,
}

/// A piece in the piece table, referencing a range of bytes from a source
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    /// Which buffer this piece references
    pub source: PieceSource,
    /// Byte offset within the source
    pub start: u64,
    /// Length in bytes
    pub len: u64,
}

impl Piece {
    pub fn new(source: PieceSource, start: u64, len: u64) -> Self {
        Self { source, start, len }
    }

    pub fn original(start: u64, len: u64) -> Self {
        Self::new(PieceSource::Original, start, len)
    }

    pub fn add(start: u64, len: u64) -> Self {
        Self::new(PieceSource::Add, start, len)
    }

    pub fn end(&self) -> u64 {
        self.start + self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A segment in the viewport mapping, tracking how rendered bytes map to original positions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingSegment {
    /// Start index in the rendered buffer
    pub render_start: usize,
    /// End index in the rendered buffer (exclusive)
    pub render_end: usize,
    /// If Some, the original file position this segment starts at
    /// If None, this is inserted text with no original position
    pub original_pos: Option<FilePos>,
}

impl MappingSegment {
    pub fn new(render_start: usize, render_end: usize, original_pos: Option<FilePos>) -> Self {
        Self {
            render_start,
            render_end,
            original_pos,
        }
    }

    /// Create a segment mapping to original file
    pub fn from_original(render_start: usize, render_end: usize, original_pos: FilePos) -> Self {
        Self::new(render_start, render_end, Some(original_pos))
    }

    /// Create a segment for inserted text (no original mapping)
    pub fn inserted(render_start: usize, render_end: usize) -> Self {
        Self::new(render_start, render_end, None)
    }

    pub fn len(&self) -> usize {
        self.render_end - self.render_start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a rendered byte index falls within this segment
    pub fn contains_render_pos(&self, pos: usize) -> bool {
        pos >= self.render_start && pos < self.render_end
    }

    /// Map a rendered byte position to an original file position
    /// Returns None if this is an inserted segment or position is out of range
    pub fn map_to_original(&self, render_pos: usize) -> Option<FilePos> {
        if !self.contains_render_pos(render_pos) {
            return None;
        }
        self.original_pos
            .map(|orig| orig + (render_pos - self.render_start) as u64)
    }
}

/// Mapping from rendered viewport positions back to original file positions
#[derive(Debug, Clone, Default)]
pub struct ViewportMapping {
    /// Segments in order of rendered position
    pub segments: Vec<MappingSegment>,
}

impl ViewportMapping {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn with_segments(segments: Vec<MappingSegment>) -> Self {
        Self { segments }
    }

    pub fn push(&mut self, segment: MappingSegment) {
        self.segments.push(segment);
    }

    /// Find the segment containing a rendered byte position
    pub fn find_segment(&self, render_pos: usize) -> Option<&MappingSegment> {
        self.segments
            .iter()
            .find(|seg| seg.contains_render_pos(render_pos))
    }

    /// Map a rendered byte position to an original file position
    /// Returns the original position and whether it's at an insertion point
    pub fn map_to_original(&self, render_pos: usize) -> MapResult {
        // Find the segment containing this position
        if let Some(segment) = self.find_segment(render_pos) {
            if let Some(orig) = segment.map_to_original(render_pos) {
                return MapResult::Original(orig);
            } else {
                // This is inserted text - calculate offset within the insertion
                let offset_in_insert = render_pos.saturating_sub(segment.render_start);
                return MapResult::Inserted {
                    before: self.find_original_before(render_pos),
                    after: self.find_original_after(render_pos),
                    offset_in_insert,
                };
            }
        }

        // Position is beyond all segments - check if we're right at the end of an insertion
        if let Some(last) = self.segments.last() {
            if render_pos == last.render_end {
                // We're exactly at the end of the last segment
                if last.original_pos.is_none() {
                    // Last segment is inserted text - we're at the end of it
                    return MapResult::Inserted {
                        before: self.find_original_before(render_pos),
                        after: self.find_original_after(render_pos),
                        offset_in_insert: last.len(), // We're at the end
                    };
                } else {
                    // Last segment is original - return position after it
                    return MapResult::Original(last.original_pos.unwrap() + last.len() as u64);
                }
            }
            if let Some(orig) = last.original_pos {
                return MapResult::Original(orig + last.len() as u64);
            }
        }

        MapResult::Unknown
    }

    /// Find the nearest original position before a rendered position
    fn find_original_before(&self, render_pos: usize) -> Option<FilePos> {
        for segment in self.segments.iter().rev() {
            if segment.render_end <= render_pos {
                if let Some(orig) = segment.original_pos {
                    return Some(orig + segment.len() as u64);
                }
            }
        }
        None
    }

    /// Find the nearest original position after a rendered position
    fn find_original_after(&self, render_pos: usize) -> Option<FilePos> {
        for segment in &self.segments {
            if segment.render_start >= render_pos {
                if let Some(orig) = segment.original_pos {
                    return Some(orig);
                }
            }
        }
        None
    }
}

/// Result of mapping a rendered position to original file coordinates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapResult {
    /// Position maps directly to an original file position
    Original(FilePos),
    /// Position is in inserted text, with optional original positions before/after
    /// and the byte offset within the inserted segment
    Inserted {
        before: Option<FilePos>,
        after: Option<FilePos>,
        /// Byte offset within the inserted text (0 = at start, len = at end)
        offset_in_insert: usize,
    },
    /// Could not determine mapping
    Unknown,
}

/// Cursor position in the viewport (visual coordinates)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CursorPos {
    /// Row in the viewport (0-indexed)
    pub row: usize,
    /// Column in the viewport (0-indexed, in terms of grapheme clusters)
    pub col: usize,
}

impl CursorPos {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub fn origin() -> Self {
        Self::new(0, 0)
    }
}

/// Input style for the editor (Nano-like vs Vi-like keybindings)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputStyle {
    #[default]
    Nano,
    Vi,
}

/// Vi mode states (only used when InputStyle is Vi)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViMode {
    #[default]
    Normal,
    Insert,
    Command,
    Visual,
}

/// Edit mode for the editor (dialogs, prompts, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditorMode {
    #[default]
    Normal,
    Search,
    Save,
    Help,
    Exit,
}

/// A cut buffer for cut/paste operations
#[derive(Debug, Clone, Default)]
pub struct CutBuffer {
    /// The cut content (bytes)
    pub content: Vec<u8>,
}

impl CutBuffer {
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
        }
    }

    pub fn set(&mut self, content: Vec<u8>) {
        self.content = content;
    }

    pub fn get(&self) -> &[u8] {
        &self.content
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn clear(&mut self) {
        self.content.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_span() {
        let span = LineSpan::new(0, 10, true);
        assert_eq!(span.len(), 10);
        assert_eq!(span.full_len(), 11);
        assert!(!span.is_empty());

        let empty = LineSpan::new(5, 5, false);
        assert!(empty.is_empty());
        assert_eq!(empty.full_len(), 0);
    }

    #[test]
    fn test_patch() {
        let insert = Patch::insert(100, b"hello".to_vec());
        assert_eq!(insert.start, 100);
        assert_eq!(insert.end, 100);
        assert_eq!(insert.deleted_len(), 0);
        assert_eq!(insert.inserted_len(), 5);
        assert_eq!(insert.length_delta(), 5);

        let delete = Patch::delete(50, 60);
        assert_eq!(delete.deleted_len(), 10);
        assert_eq!(delete.inserted_len(), 0);
        assert_eq!(delete.length_delta(), -10);

        let replace = Patch::new(20, 25, b"replacement".to_vec());
        assert_eq!(replace.length_delta(), 6); // +11 - 5 = +6
    }

    #[test]
    fn test_patch_overlap() {
        let p1 = Patch::new(10, 20, vec![]);
        let p2 = Patch::new(15, 25, vec![]);
        let p3 = Patch::new(20, 30, vec![]);
        let p4 = Patch::new(30, 40, vec![]);

        assert!(p1.overlaps(&p2));
        assert!(!p1.overlaps(&p3)); // touches but doesn't overlap
        assert!(p1.overlaps_or_touches(&p3));
        assert!(!p1.overlaps_or_touches(&p4));
    }

    #[test]
    fn test_mapping_segment() {
        let seg = MappingSegment::from_original(0, 100, 1000);
        assert!(seg.contains_render_pos(50));
        assert!(!seg.contains_render_pos(100));
        assert_eq!(seg.map_to_original(50), Some(1050));

        let inserted = MappingSegment::inserted(100, 110);
        assert_eq!(inserted.map_to_original(105), None);
    }

    #[test]
    fn test_viewport_mapping() {
        let mut mapping = ViewportMapping::new();
        mapping.push(MappingSegment::from_original(0, 100, 0));
        mapping.push(MappingSegment::inserted(100, 110));
        mapping.push(MappingSegment::from_original(110, 200, 100));

        assert_eq!(mapping.map_to_original(50), MapResult::Original(50));
        assert_eq!(mapping.map_to_original(150), MapResult::Original(140));

        match mapping.map_to_original(105) {
            MapResult::Inserted { before, after, .. } => {
                assert_eq!(before, Some(100));
                assert_eq!(after, Some(100));
            }
            _ => panic!("Expected Inserted result"),
        }
    }
}
