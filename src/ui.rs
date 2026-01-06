//! TUI (Terminal User Interface) for bigedit
//!
//! This module provides a nano-like interface using ratatui and crossterm.

use crate::editor::Editor;
#[cfg(feature = "fuse")]
use crate::fuse_view::FuseMount;
use crate::journal;
use crate::overlay::{SaveStrategy, OverlaySession, ReflinkSession};
use crate::save::save_file;
use crate::search::{find_first_in_buffer, streaming_search_forward};
use crate::types::{EditorMode, InputStyle, ViMode};
use crate::viewport::display_width;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Terminal;
use std::fs::File;
use std::io::{self, Stdout};
use std::path::Path;

/// Help text content (Nano mode)
const HELP_TEXT_NANO: &str = r#"
bigedit - Streaming Editor for Large Files (Nano Mode)

Navigation:
  Arrow Keys    Move cursor
  Home/End      Start/end of line
  PgUp/PgDn     Page up/down
  Ctrl+Home     Go to start of file
  Ctrl+End      Go to end of file

Editing:
  Ctrl+K        Cut current line
  Ctrl+U        Paste (uncut)
  Ctrl+Z        Undo
  Ctrl+Y        Redo
  Backspace     Delete char before cursor
  Delete        Delete char at cursor

File Operations:
  Ctrl+O        Write Out (instant save to journal)
  Ctrl+S        Save (same as Ctrl+O)
  Ctrl+J        Compact (full file rewrite, slow)
  Ctrl+T        Toggle save mode (Journal / FUSE)
  Ctrl+X        Exit

Search:
  Ctrl+W        Search forward
  Ctrl+Q        Search backward
  F3            Find next

Help:
  Ctrl+G        Show this help
  F1            Show this help
  Esc           Cancel/close prompt

Toggle Mode:
  F2            Switch to Vi mode

Press any key to close this help...
"#;

/// Help text content (Vi mode)
const HELP_TEXT_VI: &str = r#"
bigedit - Streaming Editor for Large Files (Vi Mode)

NORMAL MODE:
  h/j/k/l       Move cursor left/down/up/right
  w/b           Next/previous word
  0/$           Start/end of line
  gg/G          Start/end of file
  x             Delete character
  dd            Delete line
  yy            Yank (copy) line
  p             Paste after cursor
  u             Undo
  Ctrl+R        Redo

INSERT MODE:
  i             Insert before cursor
  a             Insert after cursor
  o             Open line below
  O             Open line above
  Esc           Return to normal mode

COMMAND MODE (press : in normal mode):
  :w            Save (journal mode)
  :q            Quit
  :wq           Save and quit
  :q!           Quit without saving
  :help         Show this help

SEARCH:
  /pattern      Search forward
  n             Find next
  N             Find previous

Toggle Mode:
  F2            Switch to Nano mode

Press any key to close this help...
"#;

/// The TUI application
pub struct App {
    pub editor: Editor,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    should_quit: bool,
    /// The current save strategy
    save_strategy: SaveStrategy,
    /// Active overlay session (if using overlay strategy)
    overlay_session: Option<OverlaySession>,
    /// Active reflink session (if using reflink strategy)
    reflink_session: Option<ReflinkSession>,
    /// Active FUSE mount (if using FuseView strategy)
    #[cfg(feature = "fuse")]
    fuse_mount: Option<FuseMount>,
}

impl App {
    /// Create a new app with the given file
    pub fn new(path: &Path) -> Result<Self> {
        // Set up terminal
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        // Detect best save strategy
        let mut save_strategy = SaveStrategy::detect(path);
        
        // If FUSE is not compiled in, fall back to Journal mode
        #[cfg(not(feature = "fuse"))]
        {
            if save_strategy == SaveStrategy::FuseView {
                save_strategy = SaveStrategy::Journal;
            }
        }

        // Open editor
        let mut editor = if path.exists() {
            Editor::open(path)?
        } else {
            Editor::new_file(path)?
        };

        // Show detected strategy
        #[cfg(feature = "fuse")]
        editor.set_status(format!("Mode: {} | ^T=toggle mode, ^J=write to file", save_strategy.description()));
        #[cfg(not(feature = "fuse"))]
        editor.set_status("Mode: Journal | ^J=write to file (FUSE not available)");

        Ok(Self {
            editor,
            terminal,
            should_quit: false,
            save_strategy,
            overlay_session: None,
            reflink_session: None,
            #[cfg(feature = "fuse")]
            fuse_mount: None,
        })
    }

    /// Enable vi mode
    pub fn set_vi_mode(&mut self) {
        self.editor.input_style = InputStyle::Vi;
        self.editor.vi_mode = ViMode::Normal;
        self.editor.set_status("Vi mode | F1=help, F2=toggle to nano");
    }

    /// Run the main event loop
    pub fn run(&mut self) -> Result<()> {
        while !self.should_quit {
            self.draw()?;
            self.handle_events()?;
        }
        Ok(())
    }

    /// Draw the UI
    fn draw(&mut self) -> Result<()> {
        let editor = &self.editor;
        let save_strategy = self.save_strategy;

        self.terminal.draw(|frame| {
            let area = frame.area();

            // Create layout: main area, status bar, help bar
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),    // Main text area
                    Constraint::Length(1), // Status bar
                    Constraint::Length(1), // Help bar
                ])
                .split(area);

            // Draw main text area
            draw_text_area(frame, editor, chunks[0]);

            // Draw status bar
            draw_status_bar(frame, editor, chunks[1]);

            // Draw help/mode bar
            draw_help_bar(frame, editor, save_strategy, chunks[2]);

            // Draw cursor
            let visible_row = editor.cursor.row.saturating_sub(editor.scroll_offset);
            let cursor_x = calculate_cursor_x(editor, chunks[0]);
            let cursor_y = chunks[0].y + visible_row as u16;

            if cursor_y < chunks[0].y + chunks[0].height {
                frame.set_cursor_position((cursor_x, cursor_y));
            }
        })?;

        Ok(())
    }

    /// Handle input events
    fn handle_events(&mut self) -> Result<()> {
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // F2 toggles input style (works in both nano and vi)
                if key.code == KeyCode::F(2) {
                    self.editor.toggle_input_style();
                    let mode_name = match self.editor.input_style {
                        InputStyle::Nano => "Nano",
                        InputStyle::Vi => "Vi",
                    };
                    self.editor.set_status(format!("Switched to {} mode (F2 to toggle)", mode_name));
                    return Ok(());
                }
                
                // F1 shows help in both modes
                if key.code == KeyCode::F(1) {
                    self.editor.mode = EditorMode::Help;
                    return Ok(());
                }

                // Dispatch based on editor mode (dialogs take precedence)
                match self.editor.mode {
                    EditorMode::Normal => {
                        // In normal editor mode, dispatch based on input style
                        match self.editor.input_style {
                            InputStyle::Nano => self.handle_nano_mode(key)?,
                            InputStyle::Vi => self.handle_vi_mode(key)?,
                        }
                    }
                    EditorMode::Search => self.handle_search_mode(key)?,
                    EditorMode::Save => self.handle_save_mode(key)?,
                    EditorMode::Help => self.handle_help_mode(key)?,
                    EditorMode::Exit => self.handle_exit_mode(key)?,
                }
            }
        }
        Ok(())
    }

    /// Handle keypress in nano mode (original behavior)
    fn handle_nano_mode(&mut self, key: KeyEvent) -> Result<()> {
        match (key.modifiers, key.code) {
            // Exit
            (KeyModifiers::CONTROL, KeyCode::Char('x')) => {
                if self.editor.is_modified() {
                    self.editor.mode = EditorMode::Exit;
                    self.editor.set_status("Save modified buffer? (y/n/c)");
                } else {
                    self.should_quit = true;
                }
            }

            // Save
            (KeyModifiers::CONTROL, KeyCode::Char('o'))
            | (KeyModifiers::CONTROL, KeyCode::Char('s')) => {
                self.save_file()?;
            }

            // Search
            (KeyModifiers::CONTROL, KeyCode::Char('w')) => {
                self.editor.mode = EditorMode::Search;
                self.editor.input_buffer.clear();
                // Show previous search query as hint
                if self.editor.search_query.is_empty() {
                    self.editor.set_status("Search: ");
                } else {
                    self.editor.set_status(format!("Search (Enter for '{}'): ", self.editor.search_query));
                }
            }

            // Help
            (KeyModifiers::CONTROL, KeyCode::Char('g')) => {
                self.editor.mode = EditorMode::Help;
            }

            // Debug info
            (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
                let info = format!(
                    "vp.start={} vp.end={} vp.lines={} cursor=({},{}) scroll={} mapping_segs={}",
                    self.editor.viewport.start,
                    self.editor.viewport.end,
                    self.editor.viewport.line_count(),
                    self.editor.cursor.row,
                    self.editor.cursor.col,
                    self.editor.scroll_offset,
                    self.editor.viewport.mapping.segments.len()
                );
                self.editor.set_status(info);
            }

            // Compact (full file rewrite with all patches applied)
            (KeyModifiers::CONTROL, KeyCode::Char('j')) => {
                self.editor.set_status("Compacting (full file rewrite)...");
                self.compact_file()?;
            }

            // Toggle save mode
            (KeyModifiers::CONTROL, KeyCode::Char('t')) => {
                self.toggle_save_mode()?;
            }

            // Cut line
            (KeyModifiers::CONTROL, KeyCode::Char('k')) => {
                self.editor.cut_line()?;
                self.terminal.clear()?;
                self.editor.set_status("Line cut");
            }

            // Paste
            (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
                self.editor.paste()?;
                self.editor.set_status("Pasted");
            }

            // Undo
            (KeyModifiers::CONTROL, KeyCode::Char('z')) => {
                self.editor.undo()?;
                self.terminal.clear()?;
            }

            // Redo
            (KeyModifiers::CONTROL, KeyCode::Char('y')) => {
                self.editor.redo()?;
                self.terminal.clear()?;
            }

            // Navigation
            (KeyModifiers::NONE, KeyCode::Up) => self.editor.cursor_up(),
            (KeyModifiers::NONE, KeyCode::Down) => self.editor.cursor_down(),
            (KeyModifiers::NONE, KeyCode::Left) => self.editor.cursor_left(),
            (KeyModifiers::NONE, KeyCode::Right) => self.editor.cursor_right(),
            (KeyModifiers::NONE, KeyCode::Home) => self.editor.cursor_home(),
            (KeyModifiers::NONE, KeyCode::End) => self.editor.cursor_end(),
            (KeyModifiers::NONE, KeyCode::PageUp) => {
                let height = self.terminal.size()?.height as usize - 3;
                self.editor.page_up(height);
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::PageDown) => {
                let height = self.terminal.size()?.height as usize - 3;
                self.editor.page_down(height);
                self.terminal.clear()?;
            }

            // Go to start/end of file
            (KeyModifiers::CONTROL, KeyCode::Home) => {
                self.editor.reload_viewport(0)?;
                self.editor.cursor.row = 0;
                self.editor.cursor.col = 0;
                self.editor.scroll_offset = 0;
            }
            (KeyModifiers::CONTROL, KeyCode::End) => {
                let file_len = self.editor.file_length;
                let start = file_len.saturating_sub(1024 * 1024); // Load last 1MB
                self.editor.reload_viewport(start)?;
                self.editor.cursor.row = self.editor.viewport.line_count().saturating_sub(1);
                self.editor.cursor_end();
            }

            // Editing
            (KeyModifiers::NONE, KeyCode::Backspace) => {
                self.editor.delete_backward()?;
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::Delete) => {
                self.editor.delete_forward()?;
            }
            (KeyModifiers::NONE, KeyCode::Enter) => {
                self.editor.insert_newline()?;
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::Tab) => {
                self.editor.insert_bytes(b"    ")?; // 4 spaces
            }
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
                self.editor.insert_char(c)?;
            }

            // Find next (F3)
            (KeyModifiers::NONE, KeyCode::F(3)) => {
                self.find_next()?;
            }

            _ => {}
        }
        Ok(())
    }

    /// Handle keypress in vi mode
    fn handle_vi_mode(&mut self, key: KeyEvent) -> Result<()> {
        match self.editor.vi_mode {
            ViMode::Normal => self.handle_vi_normal(key)?,
            ViMode::Insert => self.handle_vi_insert(key)?,
            ViMode::Command => self.handle_vi_command(key)?,
            ViMode::Visual => {} // TODO: implement visual mode
        }
        Ok(())
    }

    /// Handle vi normal mode
    fn handle_vi_normal(&mut self, key: KeyEvent) -> Result<()> {
        match (key.modifiers, key.code) {
            // Mode switching
            (KeyModifiers::NONE, KeyCode::Char('i')) => {
                self.editor.set_vi_mode(ViMode::Insert);
            }
            (KeyModifiers::NONE, KeyCode::Char('a')) => {
                self.editor.cursor_right();
                self.editor.set_vi_mode(ViMode::Insert);
            }
            (KeyModifiers::NONE, KeyCode::Char('o')) => {
                self.editor.cursor_end();
                self.editor.insert_newline()?;
                self.editor.set_vi_mode(ViMode::Insert);
                self.terminal.clear()?;
            }
            (KeyModifiers::SHIFT, KeyCode::Char('O')) => {
                self.editor.cursor_home();
                self.editor.insert_newline()?;
                self.editor.cursor_up();
                self.editor.set_vi_mode(ViMode::Insert);
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::Char(':')) => {
                self.editor.set_vi_mode(ViMode::Command);
                self.editor.input_buffer.clear();
                self.editor.set_status(":");
            }
            (KeyModifiers::NONE, KeyCode::Char('/')) => {
                self.editor.mode = EditorMode::Search;
                self.editor.input_buffer.clear();
                self.editor.set_status("/");
            }

            // Navigation
            (KeyModifiers::NONE, KeyCode::Char('h')) | (KeyModifiers::NONE, KeyCode::Left) => {
                self.editor.cursor_left();
            }
            (KeyModifiers::NONE, KeyCode::Char('j')) | (KeyModifiers::NONE, KeyCode::Down) => {
                self.editor.cursor_down();
            }
            (KeyModifiers::NONE, KeyCode::Char('k')) | (KeyModifiers::NONE, KeyCode::Up) => {
                self.editor.cursor_up();
            }
            (KeyModifiers::NONE, KeyCode::Char('l')) | (KeyModifiers::NONE, KeyCode::Right) => {
                self.editor.cursor_right();
            }
            (KeyModifiers::NONE, KeyCode::Char('0')) | (KeyModifiers::NONE, KeyCode::Home) => {
                self.editor.cursor_home();
            }
            (KeyModifiers::NONE, KeyCode::Char('$')) | (KeyModifiers::NONE, KeyCode::End) => {
                self.editor.cursor_end();
            }
            (KeyModifiers::NONE, KeyCode::Char('w')) => {
                self.editor.cursor_next_word();
            }
            (KeyModifiers::NONE, KeyCode::Char('b')) => {
                self.editor.cursor_prev_word();
            }
            (KeyModifiers::SHIFT, KeyCode::Char('G')) => {
                // Go to end of file
                let file_len = self.editor.file_length;
                let start = file_len.saturating_sub(1024 * 1024);
                self.editor.reload_viewport(start)?;
                self.editor.cursor.row = self.editor.viewport.line_count().saturating_sub(1);
                self.editor.cursor_end();
            }
            (KeyModifiers::NONE, KeyCode::Char('g')) => {
                // Store 'g' in command buffer, wait for second 'g'
                if self.editor.vi_command_buffer == "g" {
                    // gg - go to start of file
                    self.editor.reload_viewport(0)?;
                    self.editor.cursor.row = 0;
                    self.editor.cursor.col = 0;
                    self.editor.scroll_offset = 0;
                    self.editor.vi_command_buffer.clear();
                } else {
                    self.editor.vi_command_buffer = "g".to_string();
                }
            }
            (KeyModifiers::NONE, KeyCode::PageUp) => {
                let height = self.terminal.size()?.height as usize - 3;
                self.editor.page_up(height);
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::PageDown) => {
                let height = self.terminal.size()?.height as usize - 3;
                self.editor.page_down(height);
                self.terminal.clear()?;
            }

            // Editing
            (KeyModifiers::NONE, KeyCode::Char('x')) => {
                self.editor.delete_forward()?;
            }
            (KeyModifiers::NONE, KeyCode::Char('d')) => {
                if self.editor.vi_command_buffer == "d" {
                    // dd - delete line
                    self.editor.cut_line()?;
                    self.terminal.clear()?;
                    self.editor.vi_command_buffer.clear();
                } else {
                    self.editor.vi_command_buffer = "d".to_string();
                }
            }
            (KeyModifiers::NONE, KeyCode::Char('y')) => {
                if self.editor.vi_command_buffer == "y" {
                    // yy - yank line
                    self.editor.yank_line()?;
                    self.editor.set_status("Line yanked");
                    self.editor.vi_command_buffer.clear();
                } else {
                    self.editor.vi_command_buffer = "y".to_string();
                }
            }
            (KeyModifiers::NONE, KeyCode::Char('p')) => {
                self.editor.paste()?;
                self.editor.set_status("Pasted");
            }

            // Undo/Redo
            (KeyModifiers::NONE, KeyCode::Char('u')) => {
                self.editor.undo()?;
                self.terminal.clear()?;
            }
            (KeyModifiers::CONTROL, KeyCode::Char('r')) => {
                self.editor.redo()?;
                self.terminal.clear()?;
            }

            // Search
            (KeyModifiers::NONE, KeyCode::Char('n')) => {
                self.find_next()?;
            }

            _ => {
                // Clear command buffer on unrecognized key
                self.editor.vi_command_buffer.clear();
            }
        }
        Ok(())
    }

    /// Handle vi insert mode
    fn handle_vi_insert(&mut self, key: KeyEvent) -> Result<()> {
        match (key.modifiers, key.code) {
            (KeyModifiers::NONE, KeyCode::Esc) => {
                self.editor.set_vi_mode(ViMode::Normal);
            }
            (KeyModifiers::NONE, KeyCode::Backspace) => {
                self.editor.delete_backward()?;
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::Delete) => {
                self.editor.delete_forward()?;
            }
            (KeyModifiers::NONE, KeyCode::Enter) => {
                self.editor.insert_newline()?;
                self.terminal.clear()?;
            }
            (KeyModifiers::NONE, KeyCode::Tab) => {
                self.editor.insert_bytes(b"    ")?;
            }
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
                self.editor.insert_char(c)?;
            }
            // Arrow keys still work in insert mode
            (KeyModifiers::NONE, KeyCode::Up) => self.editor.cursor_up(),
            (KeyModifiers::NONE, KeyCode::Down) => self.editor.cursor_down(),
            (KeyModifiers::NONE, KeyCode::Left) => self.editor.cursor_left(),
            (KeyModifiers::NONE, KeyCode::Right) => self.editor.cursor_right(),
            (KeyModifiers::NONE, KeyCode::Home) => self.editor.cursor_home(),
            (KeyModifiers::NONE, KeyCode::End) => self.editor.cursor_end(),
            _ => {}
        }
        Ok(())
    }

    /// Handle vi command mode (after pressing :)
    fn handle_vi_command(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.editor.set_vi_mode(ViMode::Normal);
                self.editor.clear_status();
            }
            KeyCode::Enter => {
                let cmd = self.editor.input_buffer.clone();
                self.editor.set_vi_mode(ViMode::Normal);
                self.execute_vi_command(&cmd)?;
            }
            KeyCode::Backspace => {
                if self.editor.input_buffer.is_empty() {
                    self.editor.set_vi_mode(ViMode::Normal);
                    self.editor.clear_status();
                } else {
                    self.editor.input_buffer.pop();
                    self.editor.set_status(format!(":{}", self.editor.input_buffer));
                }
            }
            KeyCode::Char(c) => {
                self.editor.input_buffer.push(c);
                self.editor.set_status(format!(":{}", self.editor.input_buffer));
            }
            _ => {}
        }
        Ok(())
    }

    /// Execute a vi command (from command mode)
    fn execute_vi_command(&mut self, cmd: &str) -> Result<()> {
        match cmd.trim() {
            "w" => {
                self.save_file()?;
            }
            "q" => {
                if self.editor.is_modified() {
                    self.editor.set_status("No write since last change (use :q! to override)");
                } else {
                    self.should_quit = true;
                }
            }
            "q!" => {
                self.should_quit = true;
            }
            "wq" | "x" => {
                self.save_file()?;
                self.should_quit = true;
            }
            "help" | "h" => {
                self.editor.mode = EditorMode::Help;
            }
            _ => {
                self.editor.set_status(format!("Unknown command: {}", cmd));
            }
        }
        Ok(())
    }

    /// Handle keypress in search mode
    fn handle_search_mode(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.editor.mode = EditorMode::Normal;
                self.editor.clear_status();
            }
            KeyCode::Enter => {
                // If input is empty, reuse the last search query
                if !self.editor.input_buffer.is_empty() {
                    self.editor.search_query = self.editor.input_buffer.clone();
                }
                // Only search if we have a query (either new or previous)
                self.editor.mode = EditorMode::Normal;
                if !self.editor.search_query.is_empty() {
                    self.find_next()?;
                } else {
                    self.editor.set_status("No search pattern");
                }
            }
            KeyCode::Backspace => {
                self.editor.input_buffer.pop();
                self.editor.set_status(format!("Search: {}", self.editor.input_buffer));
            }
            KeyCode::Char(c) => {
                self.editor.input_buffer.push(c);
                self.editor.set_status(format!("Search: {}", self.editor.input_buffer));
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle keypress in save mode
    fn handle_save_mode(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.editor.mode = EditorMode::Normal;
                self.editor.clear_status();
            }
            KeyCode::Enter => {
                // Save with current filename or input buffer
                self.save_file()?;
                self.editor.mode = EditorMode::Normal;
            }
            KeyCode::Char(c) => {
                self.editor.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.editor.input_buffer.pop();
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle keypress in help mode
    fn handle_help_mode(&mut self, key: KeyEvent) -> Result<()> {
        // Any key closes help
        match key.code {
            _ => {
                self.editor.mode = EditorMode::Normal;
                self.editor.clear_status();
            }
        }
        Ok(())
    }

    /// Handle keypress in exit confirmation mode
    fn handle_exit_mode(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.save_file()?;
                self.should_quit = true;
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                self.should_quit = true;
            }
            KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Esc => {
                self.editor.mode = EditorMode::Normal;
                self.editor.clear_status();
            }
            _ => {}
        }
        Ok(())
    }

    /// Save the current file (uses best available strategy for near-instant save)
    fn save_file(&mut self) -> Result<()> {
        let start = std::time::Instant::now();
        
        match self.save_strategy {
            #[cfg(feature = "fuse")]
            SaveStrategy::FuseView => {
                // Save to journal first
                journal::save_to_journal(&self.editor.path, &self.editor.patches)?;
                
                // Create FUSE mount if it doesn't exist yet
                if self.fuse_mount.is_none() {
                    match FuseMount::new(&self.editor.path) {
                        Ok(mount) => {
                            self.fuse_mount = Some(mount);
                        }
                        Err(e) => {
                            // Report error but continue with journal save
                            self.editor.set_status(format!("FUSE mount failed: {}", e));
                        }
                    }
                }
                
                // The FUSE daemon reads patches directly from the journal file,
                // so no need to update it here - it will pick up changes on next read
            }
            #[cfg(not(feature = "fuse"))]
            SaveStrategy::FuseView => {
                // FUSE not available, fall back to journal
                journal::save_to_journal(&self.editor.path, &self.editor.patches)?;
            }
            SaveStrategy::Overlay => {
                // Initialize overlay session if needed
                if self.overlay_session.is_none() {
                    match OverlaySession::new(&self.editor.path) {
                        Ok(session) => self.overlay_session = Some(session),
                        Err(_) => {
                            // Fall back to journal
                            self.save_strategy = SaveStrategy::Journal;
                            return self.save_file();
                        }
                    }
                }
                
                // For overlay, we still use journal as the write mechanism
                // The overlay handles the CoW at mount time
                journal::save_to_journal(&self.editor.path, &self.editor.patches)?;
            }
            SaveStrategy::Reflink => {
                // Initialize reflink session if needed  
                if self.reflink_session.is_none() {
                    match ReflinkSession::new(&self.editor.path) {
                        Ok(mut session) => {
                            if let Err(_) = session.create_work_copy() {
                                // Fall back to journal
                                self.save_strategy = SaveStrategy::Journal;
                                return self.save_file();
                            }
                            self.reflink_session = Some(session);
                        }
                        Err(_) => {
                            self.save_strategy = SaveStrategy::Journal;
                            return self.save_file();
                        }
                    }
                }
                
                // Use journal on the work copy
                journal::save_to_journal(&self.editor.path, &self.editor.patches)?;
            }
            SaveStrategy::Journal => {
                // Standard journal-based save
                journal::save_to_journal(&self.editor.path, &self.editor.patches)?;
            }
        }
        
        let elapsed = start.elapsed();
        
        // Check if compaction is recommended
        let compact_msg = if journal::should_compact(&self.editor.path) {
            " (Ctrl+J to compact)"
        } else {
            ""
        };
        
        // Move session patches to base (they're now persisted in journal)
        for patch in self.editor.patches.patches().to_vec() {
            self.editor.base_patches.add_patch(patch);
        }
        self.editor.patches.clear();
        self.editor.modified = false;
        self.editor.set_status(format!(
            "Saved in {:.1}ms [{}]{}",
            elapsed.as_secs_f64() * 1000.0,
            self.save_strategy.description(),
            compact_msg
        ));
        Ok(())
    }

    /// Compact: rewrite file with all patches (slow, full save)
    fn compact_file(&mut self) -> Result<()> {
        let start = std::time::Instant::now();
        
        // Combine base patches and session patches for full save
        let mut all_patches = self.editor.base_patches.clone();
        for patch in self.editor.patches.patches() {
            all_patches.add_patch(patch.clone());
        }
        
        // Full save with all patches
        save_file(&self.editor.path, &all_patches, None)?;
        
        // Delete journal
        journal::delete_journal(&self.editor.path)?;
        
        // Clean up overlay/reflink sessions
        if let Some(ref mut session) = self.overlay_session {
            let _ = session.discard();
        }
        self.overlay_session = None;
        
        if let Some(ref mut session) = self.reflink_session {
            let _ = session.discard();
        }
        self.reflink_session = None;

        // Clean up FUSE mount (will be recreated on next save in FUSE mode)
        #[cfg(feature = "fuse")]
        if let Some(mount) = self.fuse_mount.take() {
            let _ = mount.unmount();
        }
        
        let elapsed = start.elapsed();
        let file_size = std::fs::metadata(&self.editor.path)?.len();
        
        // Clear both patch lists (all changes now in file)
        self.editor.base_patches.clear();
        self.editor.patches.clear();
        self.editor.modified = false;
        
        // Reload viewport to reflect new file state
        self.editor.file_length = file_size;
        self.editor.refresh_viewport()?;
        
        self.editor.set_status(format!(
            "Compacted {} in {:.1}s",
            format_size(file_size),
            elapsed.as_secs_f64()
        ));
        Ok(())
    }

    /// Toggle between save modes (Journal <-> FuseView)
    fn toggle_save_mode(&mut self) -> Result<()> {
        #[cfg(not(feature = "fuse"))]
        {
            // FUSE not available, show message
            self.editor.set_status("FUSE mode not available. Install macFUSE (macOS) or libfuse (Linux) and rebuild with 'fuse' feature.");
            return Ok(());
        }
        
        #[cfg(feature = "fuse")]
        {
            let old_strategy = self.save_strategy;
            let new_strategy = self.save_strategy.next();

            // If switching away from FuseView, unmount
            if old_strategy == SaveStrategy::FuseView && new_strategy != SaveStrategy::FuseView {
                if let Some(mount) = self.fuse_mount.take() {
                    let _ = mount.unmount();
                }
            }

            // If switching to FuseView, mount
            if new_strategy == SaveStrategy::FuseView && old_strategy != SaveStrategy::FuseView {
                match FuseMount::new(&self.editor.path) {
                    Ok(mount) => {
                        let view_path = mount.virtual_path().display().to_string();
                        self.fuse_mount = Some(mount);
                        self.save_strategy = new_strategy;
                        self.editor.set_status(format!(
                            "Mode: {} | View at: {}",
                            new_strategy.description(),
                            view_path
                        ));
                        return Ok(());
                    }
                    Err(e) => {
                        self.editor.set_status(format!("FUSE mount failed: {} - staying in Journal mode", e));
                        return Ok(());
                    }
                }
            }

            self.save_strategy = new_strategy;
            self.editor.set_status(format!(
                "Mode: {} | ^T=toggle, ^J=write to file",
                new_strategy.description()
            ));
            Ok(())
        }
    }

    /// Find next occurrence of search pattern
    fn find_next(&mut self) -> Result<()> {
        if self.editor.search_query.is_empty() {
            self.editor.set_status("No search pattern");
            return Ok(());
        }

        let pattern = self.editor.search_query.as_bytes();

        // First try to find in current viewport
        let render_pos = self
            .editor
            .viewport
            .row_col_to_render_byte(self.editor.cursor.row, self.editor.cursor.col)
            .unwrap_or(0);

        let search_from = render_pos + 1; // Start after cursor
        let remaining = &self.editor.viewport.render_bytes[search_from.min(self.editor.viewport.render_bytes.len())..];

        if let Some(offset) = find_first_in_buffer(remaining, pattern) {
            // Found in viewport
            let byte_pos = search_from + offset;
            if let Some((row, col)) = self.editor.viewport.render_byte_to_row_col(byte_pos) {
                self.editor.cursor.row = row;
                self.editor.cursor.col = col;
                self.editor.set_status(format!("Found at offset {}", byte_pos));
                return Ok(());
            }
        }

        // Not found in viewport, do streaming search
        let current_pos = self.editor.cursor_to_original_pos().unwrap_or(0);
        let mut file = File::open(&self.editor.path)?;

        if let Some(result) = streaming_search_forward(
            &mut file,
            self.editor.file_length,
            current_pos + 1,
            pattern,
            &self.editor.patches,
            false,
        )? {
            // Found - reload viewport at that position
            let new_start = result.position.saturating_sub(1024); // Some margin before match
            self.editor.reload_viewport(new_start)?;

            // Position cursor at match
            let offset_in_viewport = (result.position - new_start) as usize;
            if let Some((row, col)) = self.editor.viewport.render_byte_to_row_col(offset_in_viewport)
            {
                self.editor.cursor.row = row;
                self.editor.cursor.col = col;
            }

            self.editor.set_status(format!(
                "Found at byte {}",
                result.position
            ));
        } else {
            self.editor.set_status("Pattern not found");
        }

        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Restore terminal
        let _ = terminal::disable_raw_mode();
        let _ = self.terminal.backend_mut().execute(LeaveAlternateScreen);
    }
}

/// Draw the main text area
fn draw_text_area(frame: &mut ratatui::Frame, editor: &Editor, area: Rect) {
    // Clear the area first to prevent rendering artifacts
    frame.render_widget(Clear, area);

    if editor.mode == EditorMode::Help {
        // Draw help screen (different text based on input style)
        let help_text = match editor.input_style {
            InputStyle::Nano => HELP_TEXT_NANO,
            InputStyle::Vi => HELP_TEXT_VI,
        };
        let title = match editor.input_style {
            InputStyle::Nano => "Help (Nano Mode)",
            InputStyle::Vi => "Help (Vi Mode)",
        };
        let help = Paragraph::new(help_text)
            .style(Style::default().fg(Color::White))
            .block(Block::default().borders(Borders::ALL).title(title));
        frame.render_widget(help, area);
        return;
    }

    let height = area.height as usize;
    let visible_lines = editor.viewport.visible_lines(editor.scroll_offset, height);

    let width = area.width as usize;
    
    let mut lines: Vec<Line> = visible_lines
        .into_iter()
        .map(|s| {
            // Pad line to full width to clear any old characters
            let display_len = unicode_width::UnicodeWidthStr::width(s.as_str());
            if display_len < width {
                let padding = " ".repeat(width - display_len);
                Line::from(format!("{}{}", s, padding))
            } else {
                Line::from(s)
            }
        })
        .collect();

    // Pad with empty lines if needed
    while lines.len() < height {
        lines.push(Line::from(format!("{:width$}", "~", width = width)));
    }

    let text_widget = Paragraph::new(lines).style(Style::default().fg(Color::White));

    frame.render_widget(text_widget, area);
}

/// Draw the status bar
fn draw_status_bar(frame: &mut ratatui::Frame, editor: &Editor, area: Rect) {
    let filename = editor
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("untitled");

    let modified_marker = if editor.is_modified() { " [Modified]" } else { "" };

    let cursor_info = format!(
        "Ln {}, Col {} | Byte {}",
        editor.cursor.row + 1,
        editor.cursor.col + 1,
        editor.cursor_byte_position()
    );

    let file_info = format!(
        "{}{} | {} bytes",
        filename,
        modified_marker,
        editor.file_length
    );

    let status = if let Some(ref msg) = editor.status_message {
        msg.clone()
    } else {
        format!("{} | {}", file_info, cursor_info)
    };

    let status_bar = Paragraph::new(Line::from(vec![Span::styled(
        format!(" {:<width$}", status, width = area.width as usize - 1),
        Style::default()
            .fg(Color::Black)
            .bg(Color::White),
    )]));

    frame.render_widget(status_bar, area);
}

/// Draw the help/shortcut bar
fn draw_help_bar(frame: &mut ratatui::Frame, editor: &Editor, save_strategy: SaveStrategy, area: Rect) {
    // Vi mode: show mode indicator instead of shortcuts
    if editor.input_style == InputStyle::Vi {
        let mode_text = match editor.vi_mode {
            ViMode::Normal => "-- NORMAL --",
            ViMode::Insert => "-- INSERT --",
            ViMode::Command => "-- COMMAND --",
            ViMode::Visual => "-- VISUAL --",
        };
        
        let mode_color = match editor.vi_mode {
            ViMode::Normal => Color::Cyan,
            ViMode::Insert => Color::Green,
            ViMode::Command => Color::Yellow,
            ViMode::Visual => Color::Magenta,
        };

        let spans = vec![
            Span::styled(
                format!(" {} ", mode_text),
                Style::default()
                    .fg(Color::Black)
                    .bg(mode_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" [{}] ", save_strategy.description()),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " F1=Help ^V=Nano ",
                Style::default().fg(Color::DarkGray),
            ),
        ];

        let help_bar = Paragraph::new(Line::from(spans))
            .style(Style::default().bg(Color::DarkGray));

        frame.render_widget(help_bar, area);
        return;
    }

    // Nano mode: show shortcuts
    let shortcuts = match editor.mode {
        EditorMode::Search => vec![
            ("Enter", "Search"),
            ("Esc", "Cancel"),
        ],
        EditorMode::Exit => vec![
            ("Y", "Yes"),
            ("N", "No"),
            ("C", "Cancel"),
        ],
        _ => vec![
            ("^O", "Save"),
            ("^Z", "Undo"),
            ("^Y", "Redo"),
            ("^W", "Search"),
            ("^G", "Help"),
            ("^X", "Exit"),
        ],
    };

    let spans: Vec<Span> = shortcuts
        .iter()
        .flat_map(|(key, desc)| {
            vec![
                Span::styled(
                    format!(" {} ", key),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("{} ", desc), Style::default().fg(Color::White)),
            ]
        })
        .collect();

    // Add mode indicator at the end
    let mode_indicator = Span::styled(
        format!(" [{}] ", save_strategy.description()),
        Style::default()
            .fg(Color::Yellow)
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    let mut all_spans = spans;
    all_spans.push(mode_indicator);
    
    // Add vi mode toggle hint
    all_spans.push(Span::styled(
        " ^V=Vi ",
        Style::default().fg(Color::DarkGray),
    ));

    let help_bar = Paragraph::new(Line::from(all_spans))
        .style(Style::default().bg(Color::DarkGray));

    frame.render_widget(help_bar, area);
}

/// Calculate the cursor X position accounting for unicode width
fn calculate_cursor_x(editor: &Editor, area: Rect) -> u16 {
    if let Some(line_bytes) = editor.viewport.line_bytes(editor.cursor.row) {
        let line_str = String::from_utf8_lossy(line_bytes);
        use unicode_segmentation::UnicodeSegmentation;

        let mut width = 0;
        for (i, grapheme) in line_str.graphemes(true).enumerate() {
            if i >= editor.cursor.col {
                break;
            }
            width += display_width(grapheme);
        }

        area.x + width as u16
    } else {
        area.x
    }
}

/// Format a byte size as human-readable
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EditorMode, InputStyle, ViMode};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    /// Test format_size function
    #[test]
    fn test_format_size() {
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(1536), "1.50 KB");
    }

    /// Test cursor X calculation with ASCII
    #[test]
    fn test_cursor_x_ascii() {
        let file = create_test_file("hello world");
        let editor = Editor::open(file.path()).unwrap();
        
        // Create a mock area
        let area = Rect::new(0, 0, 80, 24);
        
        // Cursor at start should be at area.x
        let x = calculate_cursor_x(&editor, area);
        assert_eq!(x, 0);
    }

    /// Test help text constants are not empty
    #[test]
    fn test_help_text_nano_exists() {
        assert!(!HELP_TEXT_NANO.is_empty());
        assert!(HELP_TEXT_NANO.contains("Nano Mode"));
        assert!(HELP_TEXT_NANO.contains("Ctrl+Z"));  // Undo
        assert!(HELP_TEXT_NANO.contains("Ctrl+Y"));  // Redo
        assert!(HELP_TEXT_NANO.contains("Ctrl+X"));  // Exit
        assert!(HELP_TEXT_NANO.contains("Ctrl+O"));  // Save
        assert!(HELP_TEXT_NANO.contains("Ctrl+W"));  // Search
    }

    /// Test vi help text
    #[test]
    fn test_help_text_vi_exists() {
        assert!(!HELP_TEXT_VI.is_empty());
        assert!(HELP_TEXT_VI.contains("Vi Mode"));
        assert!(HELP_TEXT_VI.contains("h/j/k/l"));  // Navigation
        assert!(HELP_TEXT_VI.contains("INSERT MODE"));
        assert!(HELP_TEXT_VI.contains("NORMAL MODE"));
        assert!(HELP_TEXT_VI.contains(":w"));  // Save command
        assert!(HELP_TEXT_VI.contains(":q"));  // Quit command
        assert!(HELP_TEXT_VI.contains("u"));  // Undo
        assert!(HELP_TEXT_VI.contains("Ctrl+R"));  // Redo
    }

    /// Test that EditorMode transitions work
    #[test]
    fn test_editor_modes() {
        let file = create_test_file("test");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Default is Normal mode
        assert_eq!(editor.mode, EditorMode::Normal);
        
        // Can switch to Help
        editor.mode = EditorMode::Help;
        assert_eq!(editor.mode, EditorMode::Help);
        
        // Can switch to Search
        editor.mode = EditorMode::Search;
        assert_eq!(editor.mode, EditorMode::Search);
        
        // Can switch to Exit
        editor.mode = EditorMode::Exit;
        assert_eq!(editor.mode, EditorMode::Exit);
        
        // Back to Normal
        editor.mode = EditorMode::Normal;
        assert_eq!(editor.mode, EditorMode::Normal);
    }

    /// Test InputStyle and ViMode interactions
    #[test]
    fn test_input_style_vi_mode() {
        let file = create_test_file("test");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Switch to vi
        editor.input_style = InputStyle::Vi;
        editor.vi_mode = ViMode::Normal;
        
        // Can transition between vi modes
        editor.set_vi_mode(ViMode::Insert);
        assert_eq!(editor.vi_mode, ViMode::Insert);
        
        editor.set_vi_mode(ViMode::Command);
        assert_eq!(editor.vi_mode, ViMode::Command);
        
        editor.set_vi_mode(ViMode::Normal);
        assert_eq!(editor.vi_mode, ViMode::Normal);
    }

    /// Test toggle_input_style
    #[test]
    fn test_toggle_input_style() {
        let file = create_test_file("test");
        let mut editor = Editor::open(file.path()).unwrap();
        
        assert_eq!(editor.input_style, InputStyle::Nano);
        
        editor.toggle_input_style();
        assert_eq!(editor.input_style, InputStyle::Vi);
        
        editor.toggle_input_style();
        assert_eq!(editor.input_style, InputStyle::Nano);
    }

    /// Test search query storage
    #[test]
    fn test_search_query_storage() {
        let file = create_test_file("hello world hello");
        let mut editor = Editor::open(file.path()).unwrap();
        
        // Search query starts empty
        assert!(editor.search_query.is_empty());
        
        // Set a search query
        editor.search_query = "hello".to_string();
        assert_eq!(editor.search_query, "hello");
    }
}
