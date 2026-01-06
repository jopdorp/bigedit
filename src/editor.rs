//! Editor state and operations for bigedit
//!
//! This module manages the editor state, cursor movement, and edit operations.

use crate::journal;
use crate::patches::PatchList;
use crate::types::{CursorPos, CutBuffer, EditorMode, FilePos, InputStyle, MapResult, ViMode};
use crate::viewport::{count_graphemes, Viewport, DEFAULT_VIEWPORT_SIZE, VIEWPORT_MARGIN};
use anyhow::{Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The main editor state
pub struct Editor {
    /// Path to the file being edited
    pub path: PathBuf,
    /// File handle (kept open for reading)
    pub file: File,
    /// Total file length
    pub file_length: u64,
    /// Current viewport
    pub viewport: Viewport,
    /// Base patches from journal (already persisted)
    pub base_patches: PatchList,
    /// Session patches (new edits, not yet persisted)
    pub patches: PatchList,
    /// Cursor position in viewport coordinates
    pub cursor: CursorPos,
    /// First visible line in viewport
    pub scroll_offset: usize,
    /// Current editor mode
    pub mode: EditorMode,
    /// Input style (Nano vs Vi keybindings)
    pub input_style: InputStyle,
    /// Vi mode state (only used when input_style is Vi)
    pub vi_mode: ViMode,
    /// Vi command buffer (for commands like "dd", "3w", etc.)
    pub vi_command_buffer: String,
    /// Cut buffer for cut/paste
    pub cut_buffer: CutBuffer,
    /// Status message to display
    pub status_message: Option<String>,
    /// Search query
    pub search_query: String,
    /// Input buffer for prompts
    pub input_buffer: String,
    /// Whether the file has been modified
    pub modified: bool,
}

impl Editor {
    /// Open a file for editing
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open file")?;
        let metadata = file.metadata().context("Failed to get file metadata")?;
        let file_length = metadata.len();

        // Load patches from journal if it exists (for instant save continuity)
        let journal_patches = journal::load_from_journal(path)?;
        let has_journal = journal_patches.is_some();
        let journal_patch_count = journal_patches.as_ref().map(|p| p.patches().len()).unwrap_or(0);
        
        // Store journal patches as base, start with empty session patches
        let base_patches = journal_patches.unwrap_or_else(PatchList::new);
        let patches = PatchList::new();
        
        let mut file_for_viewport = File::open(path)?;
        let viewport = Viewport::load(
            &mut file_for_viewport,
            file_length,
            0,
            DEFAULT_VIEWPORT_SIZE,
            &base_patches,
        )?;

        let status_message = if has_journal {
            Some(format!("Loaded with journal ({} patches applied)", journal_patch_count))
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            file,
            file_length,
            viewport,
            base_patches,
            patches,
            cursor: CursorPos::origin(),
            scroll_offset: 0,
            mode: EditorMode::Normal,
            input_style: InputStyle::Nano,
            vi_mode: ViMode::Normal,
            vi_command_buffer: String::new(),
            cut_buffer: CutBuffer::new(),
            status_message,
            search_query: String::new(),
            input_buffer: String::new(),
            modified: false,
        })
    }

    /// Toggle between Nano and Vi input styles
    pub fn toggle_input_style(&mut self) {
        match self.input_style {
            InputStyle::Nano => {
                self.input_style = InputStyle::Vi;
                self.vi_mode = ViMode::Normal;
            }
            InputStyle::Vi => {
                self.input_style = InputStyle::Nano;
            }
        }
    }

    /// Set vi mode and update status
    pub fn set_vi_mode(&mut self, mode: ViMode) {
        self.vi_mode = mode;
        self.vi_command_buffer.clear();
    }

    /// Create a new empty file for editing
    pub fn new_file(path: &Path) -> Result<Self> {
        // Create an empty file
        std::fs::write(path, "").context("Failed to create file")?;
        Self::open(path)
    }

    /// Get all patches (base + session) combined for viewport loading
    fn all_patches(&self) -> PatchList {
        let mut combined = self.base_patches.clone();
        for patch in self.patches.patches() {
            combined.add_patch(patch.clone());
        }
        combined
    }

    /// Reload the viewport at a new position
    pub fn reload_viewport(&mut self, start: FilePos) -> Result<()> {
        let mut file = File::open(&self.path)?;
        let all_patches = self.all_patches();
        self.viewport = Viewport::load(
            &mut file,
            self.file_length,
            start,
            DEFAULT_VIEWPORT_SIZE,
            &all_patches,
        )?;
        Ok(())
    }

    /// Reload the viewport at current position (after edits)
    pub fn refresh_viewport(&mut self) -> Result<()> {
        self.reload_viewport(self.viewport.start)
    }

    /// Move cursor up
    pub fn cursor_up(&mut self) {
        if self.cursor.row > 0 {
            self.cursor.row -= 1;
            self.clamp_cursor_col();
        } else if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
            self.clamp_cursor_col();
        } else if self.viewport.start > 0 {
            // Need to load earlier content
            let new_start = self.viewport.start.saturating_sub(VIEWPORT_MARGIN as u64);
            if self.reload_viewport(new_start).is_ok() {
                // Position cursor at the bottom of new content
                if !self.viewport.lines.is_empty() {
                    self.cursor.row = self.viewport.line_count().saturating_sub(1);
                }
            }
        }
    }

    /// Move cursor down
    pub fn cursor_down(&mut self) {
        let max_row = self.viewport.line_count().saturating_sub(1);

        if self.cursor.row < max_row {
            self.cursor.row += 1;
            self.clamp_cursor_col();
        } else if self.viewport.end < self.file_length {
            // Need to load later content
            let new_start = self.viewport.start + VIEWPORT_MARGIN as u64;
            if self.reload_viewport(new_start).is_ok() {
                self.cursor.row = 0;
                self.clamp_cursor_col();
            }
        }
    }

    /// Move cursor left
    pub fn cursor_left(&mut self) {
        if self.cursor.col > 0 {
            self.cursor.col -= 1;
        } else if self.cursor.row > 0 {
            // Move to end of previous line
            self.cursor.row -= 1;
            self.cursor.col = self.current_line_grapheme_count();
        }
    }

    /// Move cursor right
    pub fn cursor_right(&mut self) {
        let line_len = self.current_line_grapheme_count();

        if self.cursor.col < line_len {
            self.cursor.col += 1;
        } else if self.cursor.row < self.viewport.line_count().saturating_sub(1) {
            // Move to start of next line
            self.cursor.row += 1;
            self.cursor.col = 0;
        }
    }

    /// Move to start of line
    pub fn cursor_home(&mut self) {
        self.cursor.col = 0;
    }

    /// Move to end of line
    pub fn cursor_end(&mut self) {
        self.cursor.col = self.current_line_grapheme_count();
    }

    /// Move to next word (vi 'w' command)
    pub fn cursor_next_word(&mut self) {
        if let Some(line_bytes) = self.viewport.line_bytes(self.cursor.row) {
            let line = String::from_utf8_lossy(line_bytes);
            use unicode_segmentation::UnicodeSegmentation;
            let graphemes: Vec<&str> = line.graphemes(true).collect();
            
            let mut col = self.cursor.col;
            let len = graphemes.len();
            
            // Skip current word (non-whitespace)
            while col < len && !graphemes[col].chars().all(char::is_whitespace) {
                col += 1;
            }
            // Skip whitespace
            while col < len && graphemes[col].chars().all(char::is_whitespace) {
                col += 1;
            }
            
            if col >= len {
                // Move to next line
                if self.cursor.row + 1 < self.viewport.line_count() {
                    self.cursor.row += 1;
                    self.cursor.col = 0;
                    // Skip leading whitespace on new line
                    self.cursor_next_word_start();
                }
            } else {
                self.cursor.col = col;
            }
        }
    }

    /// Skip to start of next word on current line (helper)
    fn cursor_next_word_start(&mut self) {
        if let Some(line_bytes) = self.viewport.line_bytes(self.cursor.row) {
            let line = String::from_utf8_lossy(line_bytes);
            use unicode_segmentation::UnicodeSegmentation;
            let graphemes: Vec<&str> = line.graphemes(true).collect();
            
            let mut col = self.cursor.col;
            // Skip leading whitespace
            while col < graphemes.len() && graphemes[col].chars().all(char::is_whitespace) {
                col += 1;
            }
            self.cursor.col = col;
        }
    }

    /// Move to previous word (vi 'b' command)
    pub fn cursor_prev_word(&mut self) {
        if let Some(line_bytes) = self.viewport.line_bytes(self.cursor.row) {
            let line = String::from_utf8_lossy(line_bytes);
            use unicode_segmentation::UnicodeSegmentation;
            let graphemes: Vec<&str> = line.graphemes(true).collect();
            
            if self.cursor.col == 0 {
                // Move to previous line
                if self.cursor.row > 0 {
                    self.cursor.row -= 1;
                    self.cursor.col = self.current_line_grapheme_count();
                    self.cursor_prev_word();
                }
                return;
            }
            
            let mut col = self.cursor.col.saturating_sub(1);
            
            // Skip whitespace backwards
            while col > 0 && graphemes[col].chars().all(char::is_whitespace) {
                col -= 1;
            }
            // Skip word backwards
            while col > 0 && !graphemes[col - 1].chars().all(char::is_whitespace) {
                col -= 1;
            }
            
            self.cursor.col = col;
        }
    }

    /// Yank (copy) current line to cut buffer
    pub fn yank_line(&mut self) -> Result<()> {
        if let Some(line_bytes) = self.viewport.line_bytes(self.cursor.row) {
            let mut content = line_bytes.to_vec();
            content.push(b'\n');
            self.cut_buffer.set(content);
        }
        Ok(())
    }

    /// Page up
    pub fn page_up(&mut self, visible_lines: usize) {
        if self.scroll_offset >= visible_lines {
            self.scroll_offset -= visible_lines;
            // Move cursor up by same amount, but not below 0
            self.cursor.row = self.cursor.row.saturating_sub(visible_lines);
        } else if self.scroll_offset > 0 {
            // Move cursor up by the remaining scroll amount
            let moved = self.scroll_offset;
            self.scroll_offset = 0;
            self.cursor.row = self.cursor.row.saturating_sub(moved);
        } else if self.viewport.start > 0 {
            // Load earlier content
            let new_start = self.viewport.start.saturating_sub(DEFAULT_VIEWPORT_SIZE as u64 / 2);
            if self.reload_viewport(new_start).is_ok() {
                // Position at the end of new viewport
                self.scroll_offset = self.viewport.line_count().saturating_sub(visible_lines);
                self.cursor.row = self.viewport.line_count().saturating_sub(1);
                self.cursor.col = 0;
            }
        }
        self.clamp_cursor_col();
    }

    /// Page down
    pub fn page_down(&mut self, visible_lines: usize) {
        let max_scroll = self.viewport.line_count().saturating_sub(visible_lines);
        let max_row = self.viewport.line_count().saturating_sub(1);

        if self.scroll_offset + visible_lines < self.viewport.line_count() {
            // Scroll within current viewport
            self.scroll_offset = (self.scroll_offset + visible_lines).min(max_scroll);
            // Move cursor down by same amount, keeping it in visible area
            self.cursor.row = (self.cursor.row + visible_lines).min(max_row);
        } else if self.viewport.end < self.file_length {
            // Load later content
            let new_start = self.viewport.start + DEFAULT_VIEWPORT_SIZE as u64 / 2;
            if self.reload_viewport(new_start).is_ok() {
                self.scroll_offset = 0;
                self.cursor.row = 0;
                self.cursor.col = 0;
            }
        }
        self.clamp_cursor_col();
    }

    /// Clamp cursor column to current line length
    fn clamp_cursor_col(&mut self) {
        let max_col = self.current_line_grapheme_count();
        self.cursor.col = self.cursor.col.min(max_col);
    }

    /// Get the grapheme count of the current line
    fn current_line_grapheme_count(&self) -> usize {
        self.viewport
            .line_bytes(self.cursor.row)
            .map(count_graphemes)
            .unwrap_or(0)
    }

    /// Get the current cursor position in original file coordinates
    pub fn cursor_to_original_pos(&self) -> Option<FilePos> {
        // Get render byte position
        let render_byte = self.viewport.row_col_to_render_byte(self.cursor.row, self.cursor.col)?;

        // Map to original position using viewport mapping
        match self.viewport.mapping.map_to_original(render_byte) {
            MapResult::Original(pos) => Some(pos),
            MapResult::Inserted { before, after, .. } => {
                // Prefer the position before inserted text
                before.or(after)
            }
            MapResult::Unknown => {
                // Fallback: calculate based on viewport start
                // This handles the case when mapping is empty or position not found
                Some(self.viewport.start + render_byte as u64)
            }
        }
    }

    /// Get the original file position of the start of a given line
    fn line_start_original_pos(&self, row: usize) -> Option<FilePos> {
        // Get the render byte offset of line start
        let render_byte = self.viewport.line_start_byte(row)?;

        // Map to original position
        match self.viewport.mapping.map_to_original(render_byte) {
            MapResult::Original(pos) => Some(pos),
            MapResult::Inserted { before, after, .. } => before.or(after),
            MapResult::Unknown => Some(self.viewport.start + render_byte as u64),
        }
    }

    /// Insert a character at cursor position
    pub fn insert_char(&mut self, ch: char) -> Result<()> {
        let mut buf = [0u8; 4];
        let bytes = ch.encode_utf8(&mut buf).as_bytes();
        self.insert_bytes(bytes)
    }

    /// Insert bytes at cursor position
    pub fn insert_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        // Get render byte position
        let render_byte = match self.viewport.row_col_to_render_byte(self.cursor.row, self.cursor.col) {
            Some(b) => b,
            None => return Ok(()),
        };

        // Map to original position using viewport mapping
        match self.viewport.mapping.map_to_original(render_byte) {
            MapResult::Original(pos) => {
                // Cursor is on original text - insert at this position
                self.patches.insert(pos, bytes);
            }
            MapResult::Inserted { before, after, offset_in_insert } => {
                // Cursor is inside inserted text
                // We need to insert within the combined patches at the appropriate offset
                let pos = before.or(after).unwrap_or(self.viewport.start);
                
                // Try to insert within an existing patch in combined patches
                let mut combined = self.all_patches();
                combined.insert_within(pos, offset_in_insert, bytes);
                
                // Replace our patches with the combined result
                // This is a simplification - we merge base_patches into patches
                self.patches = combined;
                self.base_patches = PatchList::new();
            }
            MapResult::Unknown => {
                // Fallback: calculate based on viewport start
                let pos = self.viewport.start + render_byte as u64;
                self.patches.insert(pos, bytes);
            }
        }
        
        self.modified = true;
        self.refresh_viewport()?;

        // Move cursor forward
        for _ in 0..count_graphemes(bytes) {
            self.cursor_right();
        }
        Ok(())
    }

    /// Delete character before cursor (backspace)
    pub fn delete_backward(&mut self) -> Result<()> {
        if self.cursor.col == 0 && self.cursor.row == 0 {
            return Ok(()); // At start of viewport
        }

        // Move cursor back
        self.cursor_left();

        // Delete character at new position
        self.delete_forward()
    }

    /// Delete character at cursor (delete key)
    pub fn delete_forward(&mut self) -> Result<()> {
        // Get render byte position
        let render_byte = match self.viewport.row_col_to_render_byte(self.cursor.row, self.cursor.col) {
            Some(b) => b,
            None => return Ok(()),
        };

        // Get the byte length of the grapheme at cursor
        let grapheme_len = self.get_grapheme_len_at(render_byte);
        if grapheme_len == 0 {
            return Ok(());
        }

        // Map to original position using viewport mapping
        match self.viewport.mapping.map_to_original(render_byte) {
            MapResult::Original(pos) => {
                // Cursor is on original text - delete at this position
                self.patches.delete(pos, pos + grapheme_len as u64);
            }
            MapResult::Inserted { before, after, offset_in_insert } => {
                // Cursor is inside inserted text - delete from the patch
                let pos = before.or(after).unwrap_or(self.viewport.start);
                
                // Delete from the combined patches
                let mut combined = self.all_patches();
                combined.delete_within(pos, offset_in_insert, grapheme_len);
                
                // Replace our patches with the combined result
                self.patches = combined;
                self.base_patches = PatchList::new();
            }
            MapResult::Unknown => {
                // Fallback: calculate based on viewport start
                let pos = self.viewport.start + render_byte as u64;
                self.patches.delete(pos, pos + grapheme_len as u64);
            }
        }
        
        self.modified = true;
        self.refresh_viewport()?;
        Ok(())
    }

    /// Get the byte length of the grapheme at a render byte position
    fn get_grapheme_len_at(&self, render_byte: usize) -> usize {
        use unicode_segmentation::UnicodeSegmentation;

        if render_byte >= self.viewport.render_bytes.len() {
            return 0;
        }

        let remaining = &self.viewport.render_bytes[render_byte..];
        let s = String::from_utf8_lossy(remaining);

        s.graphemes(true).next().map(|g| g.len()).unwrap_or(0)
    }

    /// Insert a newline at cursor position
    pub fn insert_newline(&mut self) -> Result<()> {
        // Get render byte position
        let render_byte = match self.viewport.row_col_to_render_byte(self.cursor.row, self.cursor.col) {
            Some(b) => b,
            None => return Ok(()),
        };

        // Map to original position using viewport mapping
        match self.viewport.mapping.map_to_original(render_byte) {
            MapResult::Original(pos) => {
                // Cursor is on original text - insert at this position
                self.patches.insert(pos, b"\n");
            }
            MapResult::Inserted { before, after, offset_in_insert } => {
                // Cursor is inside inserted text
                // We need to insert within the combined patches at the appropriate offset
                let pos = before.or(after).unwrap_or(self.viewport.start);
                
                // Try to insert within an existing patch in combined patches
                let mut combined = self.all_patches();
                combined.insert_within(pos, offset_in_insert, b"\n");
                
                // Replace our patches with the combined result
                self.patches = combined;
                self.base_patches = PatchList::new();
            }
            MapResult::Unknown => {
                // Fallback: calculate based on viewport start
                let pos = self.viewport.start + render_byte as u64;
                self.patches.insert(pos, b"\n");
            }
        }
        
        self.modified = true;
        self.refresh_viewport()?;
        
        // Move cursor to start of new line
        self.cursor.row += 1;
        self.cursor.col = 0;
        Ok(())
    }

    /// Cut the current line
    pub fn cut_line(&mut self) -> Result<()> {
        if let Some(line_bytes) = self.viewport.line_bytes(self.cursor.row) {
            let line = self.viewport.lines.get(self.cursor.row).unwrap();

            // Include the newline if present
            let cut_content = if line.has_newline {
                let mut content = line_bytes.to_vec();
                content.push(b'\n');
                content
            } else {
                line_bytes.to_vec()
            };

            // Count graphemes in the line (including newline if present)
            let grapheme_count = count_graphemes(&cut_content);
            
            self.cut_buffer.set(cut_content);

            // Move cursor to start of line
            self.cursor.col = 0;
            
            // Delete each character in the line
            for _ in 0..grapheme_count {
                self.delete_forward()?;
            }
        }
        Ok(())
    }

    /// Paste (uncut) the cut buffer
    pub fn paste(&mut self) -> Result<()> {
        if !self.cut_buffer.is_empty() {
            let content = self.cut_buffer.content.clone();
            self.insert_bytes(&content)?;
        }
        Ok(())
    }

    /// Set a status message
    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
    }

    /// Clear status message
    pub fn clear_status(&mut self) {
        self.status_message = None;
    }

    /// Get the absolute cursor byte position for display
    pub fn cursor_byte_position(&self) -> u64 {
        self.cursor_to_original_pos().unwrap_or(self.viewport.start)
    }

    /// Check if file is modified
    pub fn is_modified(&self) -> bool {
        self.modified || self.patches.is_modified()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    /// Test bug: file on disk with empty lines, consecutive backspaces corrupt content
    /// This is the exact user-reported bug scenario
    #[test]
    fn test_consecutive_backspace_on_empty_lines_from_disk() {
        // Create file with empty lines already on disk (simulating reopened file)
        // Note: no trailing newline to avoid extra empty line
        let file = create_test_file("line1\n\n\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();
        
        println!("Initial content:");
        println!("{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Lines: {}", editor.viewport.line_count());
        
        assert_eq!(editor.viewport.line_count(), 5);
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("".to_string())); // empty line
        assert_eq!(editor.viewport.line_str(2), Some("".to_string())); // empty line
        assert_eq!(editor.viewport.line_str(3), Some("line2".to_string()));
        
        // Go to line 2 (first empty line)
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 1);
        assert_eq!(editor.cursor.col, 0);
        
        // First backspace - should delete the newline at end of line1, joining with empty line
        println!("\nBefore backspace 1: cursor ({}, {})", editor.cursor.row, editor.cursor.col);
        editor.delete_backward().unwrap();
        println!("After backspace 1: cursor ({}, {}), lines: {}", 
                 editor.cursor.row, editor.cursor.col, editor.viewport.line_count());
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        
        assert_eq!(editor.viewport.line_count(), 4, "Should have 4 lines after first backspace");
        // Cursor should be at end of line1 (row 0, col 5)
        assert_eq!(editor.cursor.row, 0);
        assert_eq!(editor.cursor.col, 5);
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()), 
                   "line1 should be unchanged after first backspace");
        
        // Second consecutive backspace - user expects to delete another empty line
        // But cursor is now at end of line1, so backspace will delete the '1'
        // This is EXPECTED BEHAVIOR - not a bug!
        // To delete the next empty line, user needs to press DOWN first
        println!("\nBefore backspace 2: cursor ({}, {})", editor.cursor.row, editor.cursor.col);
        editor.delete_backward().unwrap();
        println!("After backspace 2: cursor ({}, {}), lines: {}", 
                 editor.cursor.row, editor.cursor.col, editor.viewport.line_count());
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        
        // After second backspace at end of "line1", we delete the '1'
        assert_eq!(editor.viewport.line_str(0), Some("line".to_string()),
                   "After backspace at end of line1, the '1' is deleted (expected behavior)");
    }

    /// Test the CORRECT way to delete multiple empty lines: DOWN between backspaces
    #[test]
    fn test_delete_multiple_empty_lines_correctly() {
        // Note: no trailing newline to avoid extra empty line
        let file = create_test_file("line1\n\n\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();
        
        assert_eq!(editor.viewport.line_count(), 5);
        
        // Go to first empty line and delete it
        editor.cursor_down();
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 4);
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
        
        // Cursor is now at end of line1 - need to go DOWN to the remaining empty line
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 1); // Now on the remaining empty line
        
        // Delete the second empty line
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 3);
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("line2".to_string()));
        assert_eq!(editor.viewport.line_str(2), Some("line3".to_string()));
    }

    #[test]
    fn test_editor_open() {
        let file = create_test_file("hello\nworld\n");
        let editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_count(), 3);
        assert_eq!(editor.cursor, CursorPos::origin());
        assert!(!editor.is_modified());
    }

    #[test]
    fn test_cursor_movement() {
        let file = create_test_file("hello\nworld\n");
        let mut editor = Editor::open(file.path()).unwrap();

        editor.cursor_right();
        assert_eq!(editor.cursor.col, 1);

        editor.cursor_down();
        assert_eq!(editor.cursor.row, 1);

        editor.cursor_left();
        assert_eq!(editor.cursor.col, 0);

        editor.cursor_up();
        assert_eq!(editor.cursor.row, 0);
    }

    #[test]
    fn test_insert_char() {
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        editor.insert_char('X').unwrap();
        assert!(editor.is_modified());
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()));
    }

    #[test]
    fn test_delete_backward() {
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        editor.cursor.col = 3; // Position after 'l'
        editor.delete_backward().unwrap();

        assert!(editor.is_modified());
        assert_eq!(editor.viewport.line_str(0), Some("helo".to_string()));
    }

    #[test]
    fn test_cursor_movement_after_reload() {
        // Create a file with multiple lines
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let file = create_test_file(content);
        let mut editor = Editor::open(file.path()).unwrap();

        // Verify initial state
        assert_eq!(editor.viewport.start, 0);
        assert_eq!(editor.viewport.line_count(), 6); // 5 lines + empty line after last \n
        assert_eq!(editor.cursor.row, 0);
        assert_eq!(editor.cursor.col, 0);

        // Reload viewport at a different position (middle of line2)
        editor.reload_viewport(7).unwrap(); // After "line1\n"
        
        // Cursor should be reset, viewport should have content
        assert!(editor.viewport.line_count() > 0, "viewport should have lines");
        
        // Move cursor - this should work
        let initial_row = editor.cursor.row;
        editor.cursor_down();
        assert_eq!(editor.cursor.row, initial_row + 1, "cursor should move down");
        
        editor.cursor_up();
        assert_eq!(editor.cursor.row, initial_row, "cursor should move back up");
        
        editor.cursor_right();
        assert_eq!(editor.cursor.col, 1, "cursor should move right");
    }

    #[test]
    fn test_cut_line_after_reload() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let file = create_test_file(content);
        let mut editor = Editor::open(file.path()).unwrap();

        // Reload at offset (simulating page down)
        editor.reload_viewport(6).unwrap(); // Start at "line2\n..."
        editor.cursor.row = 0;
        editor.cursor.col = 0;
        
        // Cut the line
        let _line_before = editor.viewport.line_str(0).unwrap();
        editor.cut_line().unwrap();
        
        // Should have cut something
        assert!(!editor.cut_buffer.is_empty(), "cut buffer should have content");
        assert!(editor.is_modified(), "editor should be modified");
    }

    #[test]
    fn test_insert_with_base_patches() {
        // Simulate having a journal with existing patches
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Simulate a base patch (as if loaded from journal)
        // Insert "X" at position 0, making "Xhello"
        editor.base_patches.insert(0, b"X");
        
        // Refresh viewport to show the patched content
        editor.refresh_viewport().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()));
        
        // Now try to insert at the current cursor position (0)
        // This should insert at the START of the rendered content
        editor.insert_char('Y').unwrap();
        
        // Should now be "YXhello"
        assert_eq!(editor.viewport.line_str(0), Some("YXhello".to_string()), 
                   "Insert should work with base patches present");
    }

    #[test]
    fn test_insert_middle_with_base_patches() {
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Base patch: insert "X" at position 2, making "heXllo"
        editor.base_patches.insert(2, b"X");
        editor.refresh_viewport().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("heXllo".to_string()));
        
        // Move cursor to position 3 (after 'X')
        editor.cursor.col = 3;
        
        // Insert 'Y'
        editor.insert_char('Y').unwrap();
        
        // Should be "heXYllo"
        assert_eq!(editor.viewport.line_str(0), Some("heXYllo".to_string()),
                   "Insert after base patch insertion point should work");
    }

    #[test]
    fn test_backspace_on_inserted_text() {
        // Test backspace on text that was just inserted
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert 'X' at position 0
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()));
        
        // Cursor should now be at position 1 (after 'X')
        assert_eq!(editor.cursor.col, 1);
        
        // Backspace should delete the 'X'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()),
                   "Backspace should delete inserted character");
    }

    #[test]
    fn test_backspace_with_base_patches() {
        // Test backspace on text from base patches (journal)
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Simulate base patch inserting "XY" at position 0
        editor.base_patches.insert(0, b"XY");
        editor.refresh_viewport().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("XYhello".to_string()));
        
        // Move cursor to position 2 (after "XY")
        editor.cursor.col = 2;
        
        // Backspace should delete 'Y'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()),
                   "Backspace should delete from base patch insertion");
    }

    #[test]
    fn test_delete_forward_on_inserted_text() {
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert 'X' at position 0
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()));
        
        // Move cursor back to position 0 (on 'X')
        editor.cursor.col = 0;
        
        // Delete forward should delete 'X'
        editor.delete_forward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()),
                   "Delete forward should delete inserted character");
    }

    #[test]
    fn test_cut_line_with_patches() {
        let file = create_test_file("line1\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert text at start of line 1
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xline1".to_string()));
        
        // Cut line should work
        editor.cursor.col = 0;
        editor.cut_line().unwrap();
        
        // Line 1 should now be gone (line2 becomes first line)
        assert_eq!(editor.viewport.line_str(0), Some("line2".to_string()),
                   "Cut line should work with inserted text");
    }

    #[test]
    fn test_multiple_backspaces_on_inserted_text() {
        // Test multiple backspaces on text that was just inserted
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert "ABC" at position 0
        editor.insert_char('A').unwrap();
        editor.insert_char('B').unwrap();
        editor.insert_char('C').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABChello".to_string()),
                   "After inserting ABC");
        assert_eq!(editor.cursor.col, 3, "Cursor should be at position 3");
        
        // Backspace once - should delete 'C'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABhello".to_string()),
                   "After first backspace");
        assert_eq!(editor.cursor.col, 2, "Cursor should be at position 2");
        
        // Backspace again - should delete 'B'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Ahello".to_string()),
                   "After second backspace");
        assert_eq!(editor.cursor.col, 1, "Cursor should be at position 1");
        
        // Backspace again - should delete 'A'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()),
                   "After third backspace, should be back to original");
        assert_eq!(editor.cursor.col, 0, "Cursor should be at position 0");
    }

    #[test]
    fn test_insert_newline_then_backspace() {
        // Test inserting a newline and then backspacing it
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move cursor to middle
        editor.cursor.col = 2;
        
        // Insert newline - should split "hello" into "he" and "llo"
        editor.insert_newline().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("he".to_string()),
                   "First line after newline insert");
        assert_eq!(editor.viewport.line_str(1), Some("llo".to_string()),
                   "Second line after newline insert");
        
        // Cursor should be at start of second line
        assert_eq!(editor.cursor.row, 1, "Cursor should be on row 1");
        assert_eq!(editor.cursor.col, 0, "Cursor should be at col 0");
        
        // Backspace should join the lines back together
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()),
                   "After backspace, lines should be joined");
        assert_eq!(editor.cursor.row, 0, "Cursor should be back on row 0");
        assert_eq!(editor.cursor.col, 2, "Cursor should be at col 2");
    }

    #[test]
    fn test_backspace_across_patch_boundary() {
        // Test backspace when cursor is right after a patch insertion
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert "X" in the middle (at position 2)
        editor.cursor.col = 2;
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("heXllo".to_string()),
                   "After inserting X");
        
        // Cursor should be at position 3
        assert_eq!(editor.cursor.col, 3);
        
        // Backspace should delete the 'X'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()),
                   "After backspace, X should be deleted");
        
        // Now backspace should delete 'e' from original
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("hllo".to_string()),
                   "After second backspace, 'e' should be deleted");
    }

    #[test]
    fn test_insert_at_end_of_insertion() {
        // Test inserting at the end of previously inserted text
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert "AB" at position 0
        editor.insert_char('A').unwrap();
        editor.insert_char('B').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABhello".to_string()));
        assert_eq!(editor.cursor.col, 2);
        
        // Insert 'C' - should go after 'B'
        editor.insert_char('C').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABChello".to_string()),
                   "C should be inserted after B");
    }

    #[test]
    fn test_backspace_then_insert() {
        // Test backspace followed by insert
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert "AB" at position 0
        editor.insert_char('A').unwrap();
        editor.insert_char('B').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABhello".to_string()));
        
        // Backspace to delete 'B'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Ahello".to_string()));
        
        // Insert 'X' - should go where 'B' was
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("AXhello".to_string()),
                   "X should replace where B was");
    }

    #[test]
    fn test_multiline_with_patches() {
        // Test editing multiline content with existing patches
        let file = create_test_file("line1\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert on first line
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xline1".to_string()));
        
        // Move to second line
        editor.cursor.row = 1;
        editor.cursor.col = 0;
        
        // Insert on second line
        editor.insert_char('Y').unwrap();
        assert_eq!(editor.viewport.line_str(1), Some("Yline2".to_string()));
        
        // Check first line is still correct
        assert_eq!(editor.viewport.line_str(0), Some("Xline1".to_string()),
                   "First line should still have X");
    }

    #[test]
    fn test_edit_after_journal_load() {
        // Simulate what happens when editing a file with journal patches
        let file = create_test_file("original text");
        let mut editor = Editor::open(file.path()).unwrap();

        // Simulate journal patches (as if loaded from journal)
        editor.base_patches.insert(0, b"PREFIX ");
        editor.refresh_viewport().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("PREFIX original text".to_string()));
        
        // Now edit at various positions
        // 1. Edit at the very start (before PREFIX)
        editor.cursor.col = 0;
        editor.insert_char('A').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("APREFIX original text".to_string()),
                   "Insert at start should work");
        
        // 2. Edit in the middle of PREFIX
        editor.cursor.col = 4;  // After "APRE"
        editor.insert_char('B').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("APREBFIX original text".to_string()),
                   "Insert in middle of patch should work");
        
        // 3. Backspace in the patched area
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("APREFIX original text".to_string()),
                   "Backspace should remove B");
    }

    #[test]
    fn test_backspace_on_newline_in_patch() {
        // Test backspace when the newline itself was inserted
        let file = create_test_file("helloworld");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move cursor to middle and insert newline
        editor.cursor.col = 5;
        editor.insert_newline().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 2, "Should have 2 lines now");
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("world".to_string()));
        assert_eq!(editor.cursor.row, 1);
        assert_eq!(editor.cursor.col, 0);
        
        // Now backspace should delete the newline
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 1, "Should be back to 1 line");
        assert_eq!(editor.viewport.line_str(0), Some("helloworld".to_string()),
                   "Content should be rejoined");
        assert_eq!(editor.cursor.row, 0);
        assert_eq!(editor.cursor.col, 5);
    }

    #[test]
    fn test_delete_original_after_insert() {
        // Insert something, then try to delete original content after it
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert 'X' at start
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xhello".to_string()));
        assert_eq!(editor.cursor.col, 1);
        
        // Delete the 'h' (original content after insertion)
        editor.delete_forward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xello".to_string()),
                   "Should delete 'h' from original");
    }

    #[test]
    fn test_alternating_insert_delete() {
        // Test a realistic editing pattern: insert, move, delete, insert
        let file = create_test_file("abcdef");
        let mut editor = Editor::open(file.path()).unwrap();

        // Insert 'X' at start
        editor.insert_char('X').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xabcdef".to_string()));
        
        // Move right past 'a' and 'b'
        editor.cursor_right();
        editor.cursor_right();
        assert_eq!(editor.cursor.col, 3);
        
        // Insert 'Y'
        editor.insert_char('Y').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("XabYcdef".to_string()));
        
        // Backspace to delete 'Y'
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xabcdef".to_string()));
        
        // Delete forward to delete 'c' (original content)
        editor.delete_forward().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("Xabdef".to_string()));
        
        // Insert 'Z'
        editor.insert_char('Z').unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("XabZdef".to_string()));
    }

    #[test]
    fn test_cut_line_basic() {
        // Basic cut line test without any patches
        let file = create_test_file("line1\nline2\nline3\n");
        let mut editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("line2".to_string()));
        assert_eq!(editor.viewport.line_str(2), Some("line3".to_string()));
        
        // Cut line 1
        editor.cursor.row = 0;
        editor.cursor.col = 0;
        editor.cut_line().unwrap();
        
        // line2 should now be first
        assert_eq!(editor.viewport.line_str(0), Some("line2".to_string()),
                   "After cutting line1, line2 should be first");
        assert_eq!(editor.viewport.line_str(1), Some("line3".to_string()));
    }

    #[test]
    fn test_backspace_removes_empty_line() {
        // Test backspace at start of line removes the newline before it
        let file = create_test_file("line1\n\nline3\n");
        let mut editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_count(), 4); // line1, empty, line3, empty after \n
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("".to_string())); // empty line
        assert_eq!(editor.viewport.line_str(2), Some("line3".to_string()));
        
        // Move to empty line
        editor.cursor.row = 1;
        editor.cursor.col = 0;
        
        // Backspace should remove the newline from line1, joining with empty line
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()),
                   "After backspace on empty line");
        assert_eq!(editor.viewport.line_str(1), Some("line3".to_string()),
                   "line3 should now be second line");
    }

    #[test]
    fn test_delete_forward_on_empty_line() {
        // Test delete key on empty line removes it
        let file = create_test_file("line1\n\nline3\n");
        let mut editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_count(), 4);
        
        // Move to end of line1
        editor.cursor.row = 0;
        editor.cursor.col = 5; // at end of "line1"
        
        // Delete forward should remove the newline
        editor.delete_forward().unwrap();
        
        // Now line1 and empty line should be joined (line1 + "")
        assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()),
                   "line1 should be joined with empty line");
    }

    #[test]
    fn test_delete_newline_at_end_of_line() {
        // Test deleting newline at end of a line
        let file = create_test_file("abc\ndef\n");
        let mut editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_str(0), Some("abc".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("def".to_string()));
        
        // Move cursor to end of first line
        editor.cursor.row = 0;
        editor.cursor.col = 3;
        
        // Delete forward should remove newline, joining lines
        editor.delete_forward().unwrap();
        
        assert_eq!(editor.viewport.line_str(0), Some("abcdef".to_string()),
                   "Lines should be joined after deleting newline");
        assert_eq!(editor.viewport.line_count(), 2); // abcdef and empty
    }

    #[test]
    fn test_backspace_at_line_start_joins_lines() {
        // Test that backspace at start of line joins with previous
        let file = create_test_file("abc\ndef\n");
        let mut editor = Editor::open(file.path()).unwrap();

        assert_eq!(editor.viewport.line_str(0), Some("abc".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("def".to_string()));
        
        // Move to start of second line
        editor.cursor.row = 1;
        editor.cursor.col = 0;
        
        // Backspace should delete newline from previous line
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_str(0), Some("abcdef".to_string()),
                   "Lines should be joined after backspace at line start");
        assert_eq!(editor.cursor.row, 0, "Cursor should move to row 0");
        assert_eq!(editor.cursor.col, 3, "Cursor should be at position 3");
    }

    #[test]
    fn test_insert_newline_at_end_then_delete() {
        // Test inserting a newline at end of file and deleting it
        let file = create_test_file("hello");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move cursor to end of line
        editor.cursor.col = 5;
        assert_eq!(editor.cursor.col, 5);
        
        // Insert newline at end
        editor.insert_newline().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 2, "Should have 2 lines after newline");
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("".to_string()));
        
        // Cursor should be at start of new empty line
        assert_eq!(editor.cursor.row, 1, "Cursor row after newline");
        assert_eq!(editor.cursor.col, 0, "Cursor col after newline");
        
        // Backspace should delete the newline
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 1, "Should have 1 line after backspace");
        assert_eq!(editor.viewport.line_str(0), Some("hello".to_string()));
        assert_eq!(editor.cursor.row, 0);
        assert_eq!(editor.cursor.col, 5);
    }

    #[test]
    fn test_multiple_newlines_then_delete() {
        // Test inserting multiple newlines and deleting them one by one
        let file = create_test_file("text");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move to end
        editor.cursor.col = 4;
        
        // Insert 3 newlines
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 4, "Should have 4 lines");
        assert_eq!(editor.cursor.row, 3);
        
        // Delete all 3 newlines
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 3, "Should have 3 lines after first backspace");
        
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 2, "Should have 2 lines after second backspace");
        
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 1, "Should have 1 line after third backspace");
        
        assert_eq!(editor.viewport.line_str(0), Some("text".to_string()));
    }

    #[test]
    fn test_newlines_navigate_down_then_delete() {
        // User scenario: insert newlines, press down, delete middle one
        let file = create_test_file("text");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move to end
        editor.cursor.col = 4;
        
        // Insert 3 newlines - cursor ends up on row 3
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 4);
        assert_eq!(editor.cursor.row, 3);
        
        // Now go UP to row 2 (third line, which is empty)
        editor.cursor_up();
        assert_eq!(editor.cursor.row, 2);
        
        // Try to delete with backspace - should delete the newline ending row 1
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 3, "Should have 3 lines after deleting second newline");
        
        // Now cursor should be at end of row 1 (which is still empty)
        assert_eq!(editor.cursor.row, 1);
        
        // Delete again - should remove another newline
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 2, "Should have 2 lines");
    }

    #[test]
    fn test_newlines_from_start_navigate_down_delete() {
        // User scenario: start at row 0, insert newlines, navigate down, delete
        let file = create_test_file("text");
        let mut editor = Editor::open(file.path()).unwrap();

        // Start at beginning, move to end of "text"
        editor.cursor.col = 4;
        
        // Insert first newline - now on row 1
        editor.insert_newline().unwrap();
        
        // Insert second newline - now on row 2
        editor.insert_newline().unwrap();
        
        // Go back to row 0 
        editor.cursor.row = 0;
        editor.cursor.col = 4;
        
        // Go DOWN to row 1
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 1);
        
        // Now try backspace on row 1 (should delete the newline ending row 0)
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 2, "Should have 2 lines");
        assert_eq!(editor.cursor.row, 0);
        assert_eq!(editor.cursor.col, 4);
    }

    #[test]
    fn test_newlines_detailed_navigation() {
        // Detailed trace of the user scenario
        let file = create_test_file("abc");
        let mut editor = Editor::open(file.path()).unwrap();

        // Move to end of "abc"
        editor.cursor.col = 3;
        
        // Insert 3 newlines
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        
        // We're at row 3, col 0 - the last empty line
        assert_eq!(editor.cursor.row, 3);
        assert_eq!(editor.viewport.line_count(), 4);
        
        // Press Down - should NOT move (we're at last line)
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 3); // Still at row 3
        
        // Press Up - should move to row 2
        editor.cursor_up();
        assert_eq!(editor.cursor.row, 2);
        
        // Press Down - back to row 3
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 3);
        
        // Now backspace from row 3
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 3);
        assert_eq!(editor.cursor.row, 2);
        
        // Now we're on row 2 (empty), try to go down then delete
        editor.cursor_down();
        // Can't go down, we're at last line now
        assert_eq!(editor.cursor.row, 2);
        
        // Backspace again
        editor.delete_backward().unwrap();
        
        assert_eq!(editor.viewport.line_count(), 2);
    }

    #[test]
    fn test_user_scenario_down_enter_up_enter_left_enter_down_backspace() {
        // Exact user scenario:
        // 1. Open file with content
        // 2. Press down, create newline
        // 3. Press up, create newline  
        // 4. Press left multiple times until at end of text line
        // 5. Press enter to create newline
        // 6. Press down
        // 7. Backspace - fails
        
        let file = create_test_file("SHELL := bash\n.ONESHELL:");
        let mut editor = Editor::open(file.path()).unwrap();
        
        println!("Initial state:");
        println!("  lines={}, cursor=({},{})", editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        for i in 0..editor.viewport.line_count() {
            println!("  line {}: {:?}", i, editor.viewport.line_str(i));
        }
        
        // 1. Press DOWN - move to line 1
        editor.cursor_down();
        println!("\nAfter DOWN: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // 2. Create newline (cursor is at start of line 1)
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={}, cursor=({},{})", editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        
        // 3. Press UP
        editor.cursor_up();
        println!("After UP: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // 4. Create newline
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={}, cursor=({},{})", editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        
        // 5. Press LEFT multiple times until at end of first line with text
        // Currently we're on an empty line, left will go to end of previous line
        for i in 0..20 {
            let prev_row = editor.cursor.row;
            let prev_col = editor.cursor.col;
            editor.cursor_left();
            println!("After LEFT {}: cursor=({},{}) -> ({},{})", i+1, prev_row, prev_col, editor.cursor.row, editor.cursor.col);
            if editor.cursor.row == 0 && editor.cursor.col == 13 {
                // At end of "SHELL := bash"
                break;
            }
        }
        
        println!("\nAfter LEFT's: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        println!("Current line content: {:?}", editor.viewport.line_str(editor.cursor.row));
        
        // 6. Press ENTER to create newline
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={}, cursor=({},{})", editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        for i in 0..editor.viewport.line_count().min(5) {
            println!("  line {}: {:?}", i, editor.viewport.line_str(i));
        }
        
        // 7. Press DOWN
        editor.cursor_down();
        println!("After DOWN: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // 8. Backspace - this is where it fails
        let before_lines = editor.viewport.line_count();
        println!("\nAbout to BACKSPACE from cursor=({},{})", editor.cursor.row, editor.cursor.col);
        println!("render_byte at cursor: {:?}", editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col));
        
        editor.delete_backward().unwrap();
        
        println!("After BACKSPACE: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), before_lines, editor.cursor.row, editor.cursor.col);
        
        assert!(editor.viewport.line_count() < before_lines, 
                "Backspace should have removed a line! Was {}, now {}", 
                before_lines, editor.viewport.line_count());
    }

    #[test]
    fn test_random_editing_scenario_1() {
        // Random editing: insert newlines at various positions, navigate, delete
        let file = create_test_file("abc\ndef\nghi");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Navigate to middle of first line
        editor.cursor.col = 1;
        editor.insert_newline().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("a".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("bc".to_string()));
        
        // Go down, right, insert newline
        editor.cursor_down();
        editor.cursor_right();
        editor.cursor_right();
        editor.insert_newline().unwrap();
        
        // Go up, left, backspace
        editor.cursor_up();
        editor.cursor_left();
        let lines_before = editor.viewport.line_count();
        editor.delete_backward().unwrap();
        assert!(editor.viewport.line_count() <= lines_before);
        
        // Multiple navigations and edits
        for _ in 0..3 {
            editor.cursor_down();
        }
        editor.insert_newline().unwrap();
        editor.cursor_up();
        editor.delete_backward().unwrap();
    }

    #[test]
    fn test_random_editing_scenario_2() {
        // Heavy newline insertion and deletion
        let file = create_test_file("line1\nline2\nline3\nline4");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Insert newlines at end of each line
        for row in 0..4 {
            editor.cursor.row = row;
            editor.cursor_end();
            editor.insert_newline().unwrap();
        }
        
        let lines_after_inserts = editor.viewport.line_count();
        assert!(lines_after_inserts >= 8, "Should have at least 8 lines, got {}", lines_after_inserts);
        
        // Now delete all the inserted newlines by navigating
        editor.cursor.row = 0;
        editor.cursor.col = 0;
        
        // Go to row 1 (empty line after line1) and delete
        editor.cursor_down();
        editor.delete_backward().unwrap();
        
        // Continue deleting
        editor.cursor_down();
        editor.cursor_down();
        editor.delete_backward().unwrap();
    }

    #[test]
    fn test_random_editing_scenario_3() {
        // Navigate with arrows, insert, delete randomly
        let file = create_test_file("ABCDEFGH\nIJKLMNOP\nQRSTUVWX");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Right right right, enter
        editor.cursor_right();
        editor.cursor_right();
        editor.cursor_right();
        editor.insert_newline().unwrap();
        assert_eq!(editor.viewport.line_str(0), Some("ABC".to_string()));
        
        // Down down, left left, enter
        editor.cursor_down();
        editor.cursor_down();
        editor.cursor_left();
        editor.cursor_left();
        editor.insert_newline().unwrap();
        
        // Up, backspace
        editor.cursor_up();
        let before = editor.viewport.line_count();
        editor.delete_backward().unwrap();
        
        // Right right right right, down, enter
        editor.cursor_right();
        editor.cursor_right();
        editor.cursor_right();
        editor.cursor_right();
        editor.cursor_down();
        editor.insert_newline().unwrap();
        
        // Left, backspace, backspace
        editor.cursor_left();
        editor.delete_backward().unwrap();
        editor.delete_backward().unwrap();
    }

    #[test]
    fn test_random_editing_scenario_4() {
        // Create many empty lines then delete them in random order
        let file = create_test_file("start\nend");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Go to end of first line, insert 5 newlines
        editor.cursor_end();
        for _ in 0..5 {
            editor.insert_newline().unwrap();
        }
        assert_eq!(editor.viewport.line_count(), 7); // start + 5 empty + end
        
        // Go to middle empty line
        editor.cursor.row = 3;
        editor.cursor.col = 0;
        
        // Delete from middle
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 6);
        
        // Go down, delete
        editor.cursor_down();
        editor.cursor_down();
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 5);
        
        // Go to first empty line, delete
        editor.cursor.row = 1;
        editor.cursor.col = 0;
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 4);
    }

    #[test]
    fn test_random_editing_scenario_5() {
        // Interleave text insertion with newlines
        let file = create_test_file("X");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // At start: "X"
        editor.cursor.col = 1; // After X
        
        // Insert newline, type, newline, type
        editor.insert_newline().unwrap();
        editor.insert_char('A').unwrap();
        editor.insert_newline().unwrap();
        editor.insert_char('B').unwrap();
        editor.insert_newline().unwrap();
        editor.insert_char('C').unwrap();
        
        // Now we have: X / A / B / C
        assert_eq!(editor.viewport.line_str(0), Some("X".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("A".to_string()));
        assert_eq!(editor.viewport.line_str(2), Some("B".to_string()));
        assert_eq!(editor.viewport.line_str(3), Some("C".to_string()));
        
        // Now delete backwards through it all
        editor.delete_backward().unwrap(); // Delete C
        editor.delete_backward().unwrap(); // Delete newline before C
        assert_eq!(editor.viewport.line_str(2), Some("B".to_string()));
        
        editor.delete_backward().unwrap(); // Delete B
        editor.delete_backward().unwrap(); // Delete newline before B
        
        editor.delete_backward().unwrap(); // Delete A
        editor.delete_backward().unwrap(); // Delete newline before A
        
        assert_eq!(editor.viewport.line_count(), 1);
        assert_eq!(editor.viewport.line_str(0), Some("X".to_string()));
    }

    #[test]
    fn test_random_editing_scenario_6() {
        // Navigate down into empty lines, up, down, delete
        let file = create_test_file("top");
        let mut editor = Editor::open(file.path()).unwrap();
        
        editor.cursor_end();
        
        // Create 3 empty lines
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        editor.insert_newline().unwrap();
        
        // Cursor is at row 3
        assert_eq!(editor.cursor.row, 3);
        
        // Up up
        editor.cursor_up();
        editor.cursor_up();
        assert_eq!(editor.cursor.row, 1);
        
        // Down
        editor.cursor_down();
        assert_eq!(editor.cursor.row, 2);
        
        // Backspace - should delete newline ending row 1
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 3);
        assert_eq!(editor.cursor.row, 1);
        
        // Down, backspace
        editor.cursor_down();
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 2);
        
        // Down (can't go past last line), backspace
        editor.cursor_down();
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 1);
    }

    #[test]
    fn test_random_editing_scenario_7() {
        // Rapid left/right navigation with newlines
        let file = create_test_file("0123456789");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Go to position 5
        for _ in 0..5 {
            editor.cursor_right();
        }
        editor.insert_newline().unwrap();
        
        // Now: "01234" / "56789"
        assert_eq!(editor.viewport.line_str(0), Some("01234".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("56789".to_string()));
        
        // Left left left (wraps to previous line)
        editor.cursor_left();
        editor.cursor_left();
        editor.cursor_left();
        
        editor.insert_newline().unwrap();
        // Now: "01" / "234" / "56789"
        
        // Right right right right right (wraps to next line)
        for _ in 0..8 {
            editor.cursor_right();
        }
        
        editor.insert_newline().unwrap();
        
        // Backspace through everything
        for _ in 0..3 {
            // Go to an empty-ish spot and delete
            editor.cursor.row = 1;
            editor.cursor.col = 0;
            if editor.cursor.row > 0 || editor.cursor.col > 0 {
                editor.delete_backward().unwrap();
            }
        }
    }

    #[test] 
    fn test_random_editing_scenario_8() {
        // Edge case: insert at very beginning, navigate, delete
        let file = create_test_file("content");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Insert newline at very start
        editor.cursor.col = 0;
        editor.insert_newline().unwrap();
        
        assert_eq!(editor.viewport.line_str(0), Some("".to_string()));
        assert_eq!(editor.viewport.line_str(1), Some("content".to_string()));
        
        // We're on row 1, go back up and right
        editor.cursor_up();
        assert_eq!(editor.cursor.row, 0);
        
        // Go down, left to end of empty line
        editor.cursor_down();
        editor.cursor_left();
        // Now at row 0, end (which is position 0 for empty line)
        
        // Down again, we're on "content"
        editor.cursor_down();
        editor.cursor.col = 0;
        
        // Backspace - should delete the newline
        editor.delete_backward().unwrap();
        assert_eq!(editor.viewport.line_count(), 1);
        assert_eq!(editor.viewport.line_str(0), Some("content".to_string()));
    }

    #[test]
    fn test_stress_random_operations() {
        // Stress test with many random-ish operations
        let file = create_test_file("The quick brown fox\njumps over\nthe lazy dog");
        let mut editor = Editor::open(file.path()).unwrap();
        
        let operations = [
            "right", "right", "right", "enter",
            "down", "right", "right", "enter", 
            "up", "left", "backspace",
            "down", "down", "enter",
            "left", "left", "left", "enter",
            "up", "backspace",
            "right", "right", "enter",
            "down", "backspace",
            "up", "up", "up",
            "right", "right", "right", "right",
            "enter", "enter", "enter",
            "down", "backspace",
            "down", "backspace", 
            "down", "backspace",
        ];
        
        for op in operations {
            match op {
                "up" => editor.cursor_up(),
                "down" => editor.cursor_down(),
                "left" => editor.cursor_left(),
                "right" => editor.cursor_right(),
                "enter" => { editor.insert_newline().unwrap(); },
                "backspace" => { editor.delete_backward().unwrap(); },
                _ => {}
            }
        }
        
        // Just verify we didn't crash and have some content
        assert!(editor.viewport.line_count() >= 1);
    }

    #[test]
    fn test_exact_user_failure_sequence() {
        // Exact sequence that user reports failing:
        // down, enter, left, up, backspace, enter, backspace (FAIL), enter, enter, backspace (FAIL), backspace (FAIL)
        
        let file = create_test_file("line1\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();
        
        println!("Initial: lines={}, cursor=({},{})", 
                 editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        for i in 0..editor.viewport.line_count() {
            println!("  line {}: {:?}", i, editor.viewport.line_str(i));
        }
        
        // DOWN
        editor.cursor_down();
        println!("\nAfter DOWN: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // ENTER
        let lines_before = editor.viewport.line_count();
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        assert!(editor.viewport.line_count() > lines_before, "ENTER should add a line");
        
        // LEFT
        editor.cursor_left();
        println!("After LEFT: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // UP
        editor.cursor_up();
        println!("After UP: cursor=({},{})", editor.cursor.row, editor.cursor.col);
        
        // BACKSPACE
        let lines_before = editor.viewport.line_count();
        println!("\nBefore 1st BACKSPACE: lines={}, cursor=({},{})", 
                 lines_before, editor.cursor.row, editor.cursor.col);
        editor.delete_backward().unwrap();
        println!("After 1st BACKSPACE: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        
        // ENTER
        let lines_before = editor.viewport.line_count();
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        assert!(editor.viewport.line_count() > lines_before, "ENTER should add a line");
        
        // BACKSPACE - user reports this fails
        let lines_before = editor.viewport.line_count();
        println!("\nBefore 2nd BACKSPACE (reported failing): lines={}, cursor=({},{})", 
                 lines_before, editor.cursor.row, editor.cursor.col);
        println!("  render_byte: {:?}", editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col));
        for i in 0..editor.viewport.line_count().min(5) {
            println!("  line {}: {:?}", i, editor.viewport.line_str(i));
        }
        
        editor.delete_backward().unwrap();
        println!("After 2nd BACKSPACE: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        
        // Check if backspace worked
        let backspace_worked_1 = editor.viewport.line_count() < lines_before || 
                                  editor.cursor.col > 0 || editor.cursor.row < lines_before;
        
        // ENTER
        let lines_before = editor.viewport.line_count();
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={}, cursor=({},{})", 
                 editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        
        // ENTER
        let lines_before = editor.viewport.line_count();
        editor.insert_newline().unwrap();
        println!("After ENTER: lines={}, cursor=({},{})", 
                 editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
        
        // BACKSPACE - user reports this fails
        let lines_before = editor.viewport.line_count();
        println!("\nBefore 3rd BACKSPACE (reported failing): lines={}, cursor=({},{})", 
                 lines_before, editor.cursor.row, editor.cursor.col);
        println!("  render_byte: {:?}", editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col));
        
        editor.delete_backward().unwrap();
        println!("After 3rd BACKSPACE: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        
        assert!(editor.viewport.line_count() < lines_before, 
                "3rd BACKSPACE should remove a line! Was {}, now {}", 
                lines_before, editor.viewport.line_count());
        
        // BACKSPACE - user reports this fails
        let lines_before = editor.viewport.line_count();
        println!("\nBefore 4th BACKSPACE (reported failing): lines={}, cursor=({},{})", 
                 lines_before, editor.cursor.row, editor.cursor.col);
        
        editor.delete_backward().unwrap();
        println!("After 4th BACKSPACE: lines={} (was {}), cursor=({},{})", 
                 editor.viewport.line_count(), lines_before, editor.cursor.row, editor.cursor.col);
        
        assert!(editor.viewport.line_count() < lines_before, 
                "4th BACKSPACE should remove a line! Was {}, now {}", 
                lines_before, editor.viewport.line_count());
    }

    #[test]
    fn test_down_enter_left_backspace_enter_enter_backspace_backspace() {
        // User's EXACT bug sequence:
        // down, enter, left, backspace, enter, enter, backspace(fails), backspace(fails)
        let file = create_test_file("line1\nline2\nline3");
        let mut editor = Editor::open(file.path()).unwrap();
        
        println!("=== Initial state ===");
        println!("Lines: {}", editor.viewport.line_count());
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // 1. DOWN
        editor.cursor_down();
        println!("\n=== After DOWN ===");
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // 2. ENTER
        editor.insert_newline().unwrap();
        println!("\n=== After ENTER ===");
        println!("Lines: {}", editor.viewport.line_count());
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // 3. LEFT
        editor.cursor_left();
        println!("\n=== After LEFT ===");
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // 4. BACKSPACE
        let lines_before = editor.viewport.line_count();
        editor.delete_backward().unwrap();
        println!("\n=== After BACKSPACE ===");
        println!("Lines: {} (was {})", editor.viewport.line_count(), lines_before);
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // 5. ENTER
        let lines_before = editor.viewport.line_count();
        println!("\n=== Before ENTER (5) ===");
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        let render_byte = editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col);
        println!("render_byte: {:?}", render_byte);
        if let Some(rb) = render_byte {
            let map_result = editor.viewport.mapping.map_to_original(rb);
            println!("map_to_original: {:?}", map_result);
        }
        println!("Patches before: {:?}", editor.patches);
        
        editor.insert_newline().unwrap();
        println!("\n=== After ENTER (5) ===");
        println!("Lines: {} (was {})", editor.viewport.line_count(), lines_before);
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        println!("Patches after: {:?}", editor.patches);
        
        // 6. ENTER
        let lines_before = editor.viewport.line_count();
        editor.insert_newline().unwrap();
        println!("\n=== After ENTER (6) ===");
        println!("Lines: {} (was {})", editor.viewport.line_count(), lines_before);
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        // Check render_byte mapping before backspace
        let render_byte = editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col);
        println!("\nBefore 7. BACKSPACE:");
        println!("  render_byte: {:?}", render_byte);
        if let Some(rb) = render_byte {
            let map_result = editor.viewport.mapping.map_to_original(rb);
            println!("  map_to_original: {:?}", map_result);
            let grapheme_len = editor.get_grapheme_len_at(rb);
            println!("  grapheme_len: {}", grapheme_len);
            if rb > 0 {
                println!("  byte at cursor: {:?}", editor.viewport.render_bytes.get(rb));
                println!("  byte before cursor: {:?}", editor.viewport.render_bytes.get(rb-1));
            }
        }
        
        // 7. BACKSPACE - user reports this FAILS
        let lines_before = editor.viewport.line_count();
        println!("\n=== BACKSPACE (7) - reported failing ===");
        editor.delete_backward().unwrap();
        println!("Lines: {} (was {})", editor.viewport.line_count(), lines_before);
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        let backspace7_worked = editor.viewport.line_count() < lines_before;
        println!("Backspace 7 worked: {}", backspace7_worked);
        
        // 8. BACKSPACE - user reports this FAILS
        let lines_before = editor.viewport.line_count();
        println!("\n=== BACKSPACE (8) - reported failing ===");
        
        // Check state before
        let render_byte = editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col);
        println!("  render_byte: {:?}", render_byte);
        if let Some(rb) = render_byte {
            let map_result = editor.viewport.mapping.map_to_original(rb);
            println!("  map_to_original: {:?}", map_result);
        }
        
        editor.delete_backward().unwrap();
        println!("Lines: {} (was {})", editor.viewport.line_count(), lines_before);
        println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        println!("Cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
        
        let backspace8_worked = editor.viewport.line_count() < lines_before;
        println!("Backspace 8 worked: {}", backspace8_worked);
        
        // Final check
        println!("\n=== FINAL ===");
        println!("Initial lines: 3");
        println!("Final lines: {}", editor.viewport.line_count());
        
        // Net effect should be: +1 (enter) -1 (backspace after left) +1 (enter) +1 (enter) -1 (backspace) -1 (backspace) = 0
        assert_eq!(editor.viewport.line_count(), 3, 
                   "Should end with 3 lines but got {}", editor.viewport.line_count());
    }

    #[test]
    fn test_save_reopen_edit() {
        // Test that changes persist correctly after save and reopen
        let file = create_test_file("line1\nline2\nline3");
        let file_path = file.path().to_path_buf();
        
        println!("=== Session 1: Add newlines and save ===");
        {
            let mut editor = Editor::open(&file_path).unwrap();
            println!("Initial lines: {}", editor.viewport.line_count());
            
            // Go down and insert 2 newlines
            editor.cursor_down();
            editor.insert_newline().unwrap();
            editor.insert_newline().unwrap();
            
            println!("After 2 enters: {} lines", editor.viewport.line_count());
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            
            // Save the file
            crate::save::save_file(&file_path, &editor.all_patches(), None).unwrap();
            println!("Saved!");
        }
        
        // Check file on disk
        let content_after_s1 = std::fs::read_to_string(&file_path).unwrap();
        println!("\nFile on disk after session 1:");
        println!("{}", content_after_s1);
        println!("Lines: {}", content_after_s1.lines().count());
        assert_eq!(content_after_s1.lines().count(), 5, "Should have 5 lines after adding 2 newlines");
        
        println!("\n=== Session 2: Remove newlines and save ===");
        {
            let mut editor = Editor::open(&file_path).unwrap();
            println!("Reopened, lines: {}", editor.viewport.line_count());
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            println!("Mapping segments: {:?}", editor.viewport.mapping.segments);
            println!("Patches: {:?}", editor.patches);
            println!("Base patches: {:?}", editor.base_patches);
            
            // Go to line 2 (first empty line) and backspace twice
            editor.cursor_down(); // line 2 (empty)
            println!("After down, cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
            
            let rb = editor.viewport.row_col_to_render_byte(editor.cursor.row, editor.cursor.col);
            println!("render_byte: {:?}", rb);
            if let Some(rb) = rb {
                println!("bytes around cursor: {:?}", &editor.viewport.render_bytes[rb.saturating_sub(3)..rb.min(editor.viewport.render_bytes.len())+3.min(editor.viewport.render_bytes.len() - rb)]);
            }
            
            editor.delete_backward().unwrap(); // delete first empty line - joins with line1
            println!("After backspace 1: {} lines, cursor: ({}, {})", 
                     editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            println!("Patches after bs1: {:?}", editor.patches);
            println!("Mapping after bs1: {:?}", editor.viewport.mapping.segments);
            
            // After first backspace, cursor is at end of line 0 (row 0, col 5)
            // To delete the second empty line, we need to go DOWN first to the new empty line
            editor.cursor_down(); // move to the remaining empty line (now row 1)
            println!("After down, cursor: ({}, {})", editor.cursor.row, editor.cursor.col);
            
            editor.delete_backward().unwrap(); // delete second empty line
            println!("After backspace 2: {} lines, cursor: ({}, {})", 
                     editor.viewport.line_count(), editor.cursor.row, editor.cursor.col);
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            
            // Save the file
            println!("Patches before save: {:?}", editor.all_patches());
            crate::save::save_file(&file_path, &editor.all_patches(), None).unwrap();
            println!("Saved!");
        }
        
        // Check file on disk
        let content_after_s2 = std::fs::read_to_string(&file_path).unwrap();
        println!("\nFile on disk after session 2:");
        println!("{}", content_after_s2);
        println!("Lines: {}", content_after_s2.lines().count());
        
        println!("\n=== Session 3: Verify the file ===");
        {
            let editor = Editor::open(&file_path).unwrap();
            println!("Reopened, lines: {}", editor.viewport.line_count());
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
        }
        
        assert_eq!(content_after_s2.lines().count(), 3, 
                   "Should have 3 lines after removing 2 newlines, but got {}", 
                   content_after_s2.lines().count());
    }

    /// Test the exact user bug: create newlines, save to JOURNAL, reopen, delete, save, reopen
    /// The journal-based save should persist changes across sessions
    #[test]
    fn test_journal_save_reopen_delete_reopen() {
        let file = create_test_file("line1\nline2\nline3");
        let file_path = file.path().to_path_buf();
        
        println!("=== Initial file ===");
        println!("Content: {:?}", std::fs::read_to_string(&file_path).unwrap());
        
        // Session 1: Add newlines and save to journal
        println!("\n=== Session 1: Add newlines, save to journal ===");
        {
            let mut editor = Editor::open(&file_path).unwrap();
            println!("Initial lines: {}", editor.viewport.line_count());
            assert_eq!(editor.viewport.line_count(), 3);
            
            // DOWN, ENTER, ENTER (add 2 empty lines)
            editor.cursor_down();
            editor.insert_newline().unwrap();
            editor.insert_newline().unwrap();
            
            println!("After edits: {} lines", editor.viewport.line_count());
            assert_eq!(editor.viewport.line_count(), 5);
            
            // Save to journal (like CTRL-O)
            journal::save_to_journal(&file_path, &editor.patches).unwrap();
            println!("Saved to journal!");
            
            // Verify journal was created
            let journal = journal::journal_path(&file_path);
            assert!(journal.exists(), "Journal should exist after save");
            println!("Journal created: {:?}", journal);
        }
        
        // Session 2: Reopen (should load journal), verify we see 5 lines
        println!("\n=== Session 2: Reopen, verify edits persist ===");
        {
            let editor = Editor::open(&file_path).unwrap();
            println!("Reopened, lines: {}", editor.viewport.line_count());
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            
            assert_eq!(editor.viewport.line_count(), 5, 
                       "Should see 5 lines after reopening (2 newlines were added)");
        }
        
        // Session 3: Delete the newlines and save to journal
        println!("\n=== Session 3: Delete newlines, save to journal ===");
        {
            let mut editor = Editor::open(&file_path).unwrap();
            println!("Lines before delete: {}", editor.viewport.line_count());
            
            // Go to first empty line and delete
            editor.cursor_down(); // row 1 (first empty line)
            editor.delete_backward().unwrap();
            println!("After first backspace: {} lines", editor.viewport.line_count());
            
            // Go to remaining empty line and delete
            editor.cursor_down(); // row 1 (remaining empty line)
            editor.delete_backward().unwrap();
            println!("After second backspace: {} lines", editor.viewport.line_count());
            
            assert_eq!(editor.viewport.line_count(), 3);
            
            // Save to journal
            journal::save_to_journal(&file_path, &editor.all_patches()).unwrap();
            println!("Saved to journal!");
        }
        
        // Session 4: Reopen and verify
        println!("\n=== Session 4: Final verification ===");
        {
            let editor = Editor::open(&file_path).unwrap();
            println!("Final reopened, lines: {}", editor.viewport.line_count());
            println!("Content:\n{}", String::from_utf8_lossy(&editor.viewport.render_bytes));
            
            assert_eq!(editor.viewport.line_count(), 3,
                       "Should have 3 lines after deleting the newlines and reopening");
            assert_eq!(editor.viewport.line_str(0), Some("line1".to_string()));
            assert_eq!(editor.viewport.line_str(1), Some("line2".to_string()));
            assert_eq!(editor.viewport.line_str(2), Some("line3".to_string()));
        }
        
        // Cleanup journal
        let _ = journal::delete_journal(&file_path);
    }
}