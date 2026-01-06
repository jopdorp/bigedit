//! Patch management for bigedit
//!
//! This module handles the patch list that records all edits made to the file.
//! Patches are stored sorted by position and are merged when they overlap or touch.

use crate::types::{FilePos, Patch};

/// A list of patches representing edits to the original file
#[derive(Debug, Clone, Default)]
pub struct PatchList {
    /// Patches sorted by start position, non-overlapping
    patches: Vec<Patch>,
    /// Add buffer containing all inserted text
    add_buffer: Vec<u8>,
}

impl PatchList {
    pub fn new() -> Self {
        Self {
            patches: Vec::new(),
            add_buffer: Vec::new(),
        }
    }

    /// Returns true if there are any patches (file has been modified)
    pub fn is_modified(&self) -> bool {
        !self.patches.is_empty()
    }

    /// Returns the number of patches
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Returns true if there are no patches
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Get all patches (sorted by start position)
    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }

    /// Clear all patches
    pub fn clear(&mut self) {
        self.patches.clear();
        self.add_buffer.clear();
    }

    /// Insert bytes at a position in the original file coordinates
    pub fn insert(&mut self, pos: FilePos, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        let patch = Patch::insert(pos, bytes.to_vec());
        self.add_patch(patch);
    }

    /// Insert bytes within an existing insertion at the given original position
    /// The offset specifies where within the existing insertion's replacement to insert
    pub fn insert_within(&mut self, pos: FilePos, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        // Find the patch at this position - match either:
        // 1. Pure insertions (start == end == pos)
        // 2. Patches that start at this position and have a replacement we can insert into
        if let Some(patch_idx) = self.patches.iter().position(|p| p.start == pos && !p.replacement.is_empty()) {
            // Splice into its replacement at the given offset
            let patch = &mut self.patches[patch_idx];
            let insert_at = offset.min(patch.replacement.len());
            patch.replacement.splice(insert_at..insert_at, bytes.iter().cloned());
        } else {
            // No existing patch with replacement at this position, just do a regular insert
            self.insert(pos, bytes);
        }
    }

    /// Delete bytes within an existing insertion at the given original position
    /// The offset specifies the start position within the insertion, len is how many bytes to delete
    pub fn delete_within(&mut self, pos: FilePos, offset: usize, len: usize) {
        if len == 0 {
            return;
        }

        // Find the patch at this position - match patches that start at this position and have a replacement
        if let Some(patch_idx) = self.patches.iter().position(|p| p.start == pos && !p.replacement.is_empty()) {
            // Remove bytes from its replacement
            let patch = &mut self.patches[patch_idx];
            let delete_start = offset.min(patch.replacement.len());
            let delete_end = (offset + len).min(patch.replacement.len());
            if delete_start < delete_end {
                patch.replacement.drain(delete_start..delete_end);
                // If the patch replacement is now empty and it was a pure insertion, remove it
                // If it's a replacement patch (start != end), keep it as a delete patch
                if patch.replacement.is_empty() && patch.start == patch.end {
                    self.patches.remove(patch_idx);
                }
            }
        }
    }

    /// Delete bytes in the original file coordinates
    pub fn delete(&mut self, start: FilePos, end: FilePos) {
        if start >= end {
            return;
        }

        let patch = Patch::delete(start, end);
        self.add_patch(patch);
    }

    /// Replace bytes in the original file coordinates
    pub fn replace(&mut self, start: FilePos, end: FilePos, replacement: &[u8]) {
        let patch = Patch::new(start, end, replacement.to_vec());
        self.add_patch(patch);
    }

    /// Add a patch, merging with existing patches as needed
    pub fn add_patch(&mut self, new_patch: Patch) {
        if self.patches.is_empty() {
            self.patches.push(new_patch);
            return;
        }

        // Find patches that overlap or touch the new patch
        let mut merge_start = None;
        let mut merge_end = None;

        for (i, existing) in self.patches.iter().enumerate() {
            if new_patch.overlaps_or_touches(existing) {
                if merge_start.is_none() {
                    merge_start = Some(i);
                }
                merge_end = Some(i);
            } else if existing.start > new_patch.end {
                // Past the new patch, no more overlaps possible
                break;
            }
        }

        match (merge_start, merge_end) {
            (Some(start_idx), Some(end_idx)) => {
                // Merge with overlapping patches
                let merged = self.merge_patches(start_idx, end_idx, new_patch);
                self.patches.drain(start_idx..=end_idx);
                self.patches.insert(start_idx, merged);
            }
            (None, None) => {
                // No overlaps, insert at the right position
                let insert_pos = self
                    .patches
                    .iter()
                    .position(|p| p.start > new_patch.start)
                    .unwrap_or(self.patches.len());
                self.patches.insert(insert_pos, new_patch);
            }
            _ => unreachable!(),
        }
    }

    /// Merge a new patch with existing patches at indices [start_idx..=end_idx]
    fn merge_patches(&self, start_idx: usize, end_idx: usize, new_patch: Patch) -> Patch {
        let first = &self.patches[start_idx];
        let last = &self.patches[end_idx];

        // Calculate the merged range in original file coordinates
        let merged_start = first.start.min(new_patch.start);
        let merged_end = last.end.max(new_patch.end);

        // Build the replacement by combining all the patches
        // This is the tricky part: we need to properly interleave the patches
        let mut replacement = Vec::new();

        // Collect all patches including the new one, sorted by start
        let mut all_patches: Vec<&Patch> = self.patches[start_idx..=end_idx].iter().collect();
        all_patches.push(&new_patch);
        all_patches.sort_by_key(|p| p.start);

        // We need to build the replacement content
        // Each patch transforms a portion of [merged_start..merged_end]
        // We track our position in the merged range
        let mut current_pos = merged_start;

        for patch in &all_patches {
            if patch.start > current_pos {
                // There's a gap - this represents original content
                // But since we're merging, gaps should be filled by other patches
                // If we have gaps between patches, we'd need the original content
                // For simplicity, assume patches are contiguous after merging
            }

            if patch.start <= current_pos && patch.end > current_pos {
                // This patch covers current_pos
                if patch.start < current_pos {
                    // Partial overlap - take portion of replacement
                    let offset = (current_pos - patch.start) as usize;
                    if offset < patch.replacement.len() {
                        replacement.extend_from_slice(&patch.replacement[offset..]);
                    }
                } else {
                    replacement.extend_from_slice(&patch.replacement);
                }
                current_pos = patch.end;
            } else if patch.start >= current_pos {
                // Patch starts at or after current pos
                replacement.extend_from_slice(&patch.replacement);
                current_pos = current_pos.max(patch.end);
            }
        }

        Patch::new(merged_start, merged_end, replacement)
    }

    /// Find patches that affect a given byte range
    pub fn patches_in_range(&self, start: FilePos, end: FilePos) -> Vec<&Patch> {
        self.patches
            .iter()
            .filter(|p| p.start < end && p.end > start)
            .collect()
    }

    /// Calculate the adjusted position after applying all patches before pos
    /// This maps from original file coordinates to patched document coordinates
    pub fn original_to_patched(&self, orig_pos: FilePos) -> u64 {
        let mut offset: i64 = 0;

        for patch in &self.patches {
            if patch.end <= orig_pos {
                // Patch is entirely before this position
                offset += patch.length_delta();
            } else if patch.start < orig_pos {
                // Position is inside a patch
                // Map to the end of the patch's replacement
                let into_patch = orig_pos - patch.start;
                let clamped = (into_patch as usize).min(patch.replacement.len());
                return (patch.start as i64 + offset + clamped as i64) as u64;
            } else {
                break;
            }
        }

        (orig_pos as i64 + offset) as u64
    }

    /// Calculate the original position from a patched document position
    /// This maps from patched document coordinates to original file coordinates
    pub fn patched_to_original(&self, patched_pos: u64) -> PatchedPosResult {
        let mut current_patched: u64 = 0;
        let mut current_original: u64 = 0;

        for patch in &self.patches {
            let gap_len = patch.start - current_original;

            // Check if position is in the gap before this patch
            if patched_pos < current_patched + gap_len {
                let offset = patched_pos - current_patched;
                return PatchedPosResult::Original(current_original + offset);
            }

            current_patched += gap_len;
            current_original = patch.start;

            // Check if position is inside this patch's replacement
            let repl_len = patch.replacement.len() as u64;
            if patched_pos < current_patched + repl_len {
                let offset = patched_pos - current_patched;
                return PatchedPosResult::InPatch {
                    patch_start: patch.start,
                    patch_end: patch.end,
                    offset_in_replacement: offset as usize,
                };
            }

            current_patched += repl_len;
            current_original = patch.end;
        }

        // Position is after all patches
        let offset = patched_pos - current_patched;
        PatchedPosResult::Original(current_original + offset)
    }

    /// Get the total document length after applying all patches
    pub fn patched_length(&self, original_length: u64) -> u64 {
        let delta: i64 = self.patches.iter().map(|p| p.length_delta()).sum();
        (original_length as i64 + delta) as u64
    }
}

/// Result of mapping a patched position to original coordinates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchedPosResult {
    /// Position maps to an original file position
    Original(FilePos),
    /// Position is inside a patch's replacement
    InPatch {
        patch_start: FilePos,
        patch_end: FilePos,
        offset_in_replacement: usize,
    },
}

/// Apply patches to a byte slice, producing the patched content
/// The slice represents bytes from [slice_start..slice_start+slice.len()) in original coordinates
pub fn apply_patches_to_slice(
    slice: &[u8],
    slice_start: FilePos,
    patches: &[Patch],
) -> (Vec<u8>, Vec<MappingEntry>) {
    let slice_end = slice_start + slice.len() as u64;
    let mut result = Vec::new();
    let mut mapping = Vec::new();
    let mut current_pos = slice_start;

    // Filter to patches that affect this range
    // Note: Use <= for start comparison to include insert patches at the end (where patch.start == slice_end)
    let relevant_patches: Vec<&Patch> = patches
        .iter()
        .filter(|p| p.start <= slice_end && p.end >= slice_start)
        .collect();

    for patch in relevant_patches {
        // Copy original content before this patch
        if patch.start > current_pos {
            let copy_start = (current_pos - slice_start) as usize;
            let copy_end = ((patch.start - slice_start) as usize).min(slice.len());
            if copy_start < copy_end {
                let orig_content = &slice[copy_start..copy_end];
                mapping.push(MappingEntry {
                    render_start: result.len(),
                    render_len: orig_content.len(),
                    original_start: Some(current_pos),
                });
                result.extend_from_slice(orig_content);
            }
        }

        // Apply the patch replacement
        if !patch.replacement.is_empty() {
            mapping.push(MappingEntry {
                render_start: result.len(),
                render_len: patch.replacement.len(),
                original_start: None, // This is inserted/replacement text
            });
            result.extend_from_slice(&patch.replacement);
        }

        current_pos = patch.end.max(current_pos);
    }

    // Copy remaining original content after all patches
    if current_pos < slice_end {
        let copy_start = (current_pos - slice_start) as usize;
        if copy_start < slice.len() {
            let orig_content = &slice[copy_start..];
            mapping.push(MappingEntry {
                render_start: result.len(),
                render_len: orig_content.len(),
                original_start: Some(current_pos),
            });
            result.extend_from_slice(orig_content);
        }
    }

    (result, mapping)
}

/// Entry in the mapping from rendered bytes to original positions
#[derive(Debug, Clone)]
pub struct MappingEntry {
    /// Start position in the rendered buffer
    pub render_start: usize,
    /// Length in the rendered buffer
    pub render_len: usize,
    /// Original file position (None if this is inserted text)
    pub original_start: Option<FilePos>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_list_insert() {
        let mut list = PatchList::new();
        list.insert(10, b"hello");
        assert_eq!(list.len(), 1);
        assert!(list.is_modified());

        let patches = list.patches();
        assert_eq!(patches[0].start, 10);
        assert_eq!(patches[0].end, 10);
        assert_eq!(patches[0].replacement, b"hello");
    }

    #[test]
    fn test_patch_list_delete() {
        let mut list = PatchList::new();
        list.delete(10, 20);
        assert_eq!(list.len(), 1);

        let patches = list.patches();
        assert_eq!(patches[0].start, 10);
        assert_eq!(patches[0].end, 20);
        assert!(patches[0].replacement.is_empty());
    }

    #[test]
    fn test_patch_list_no_overlap() {
        let mut list = PatchList::new();
        list.insert(10, b"a");
        list.insert(30, b"b");
        list.insert(20, b"c");

        assert_eq!(list.len(), 3);
        // Should be sorted
        let patches = list.patches();
        assert_eq!(patches[0].start, 10);
        assert_eq!(patches[1].start, 20);
        assert_eq!(patches[2].start, 30);
    }

    #[test]
    fn test_patched_length() {
        let mut list = PatchList::new();
        list.insert(10, b"hello"); // +5
        list.delete(20, 25); // -5
        list.replace(30, 32, b"longer"); // +4

        let original_len = 100;
        assert_eq!(list.patched_length(original_len), 104);
    }

    #[test]
    fn test_apply_patches_to_slice() {
        let original = b"hello world";
        let patches = vec![Patch::new(6, 11, b"rust".to_vec())];

        let (result, _mapping) = apply_patches_to_slice(original, 0, &patches);
        assert_eq!(result, b"hello rust");
    }

    #[test]
    fn test_apply_patches_insertion() {
        let original = b"hello world";
        let patches = vec![Patch::insert(5, b" beautiful".to_vec())];

        let (result, _mapping) = apply_patches_to_slice(original, 0, &patches);
        assert_eq!(result, b"hello beautiful world");
    }

    #[test]
    fn test_patched_to_original() {
        let mut list = PatchList::new();
        // Insert 5 bytes at position 10
        list.insert(10, b"hello");

        // Position before insert should map directly
        assert_eq!(list.patched_to_original(5), PatchedPosResult::Original(5));

        // Position after insert should be offset by 5
        assert_eq!(
            list.patched_to_original(20),
            PatchedPosResult::Original(15)
        );
    }
}
