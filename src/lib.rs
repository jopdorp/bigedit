//! bigedit library - Core functionality for the streaming TUI editor
//!
//! This library exposes the core modules for use in benchmarks and tests.

pub mod config;
pub mod editor;
#[cfg(feature = "fuse")]
pub mod fuse_view;
pub mod journal;
pub mod language;
pub mod overlay;
pub mod patches;
pub mod save;
pub mod search;
pub mod search_service;
pub mod semantic;
pub mod types;
pub mod ui;
pub mod viewport;
