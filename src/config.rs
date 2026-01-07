//! Configuration management for bigedit
//!
//! Loads user preferences from ~/.config/bigedit/config.toml

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Default config file contents
const DEFAULT_CONFIG: &str = r#"# bigedit configuration file
# Location: ~/.config/bigedit/config.toml

[editor]
# Default input style: "nano" or "vi"
input_style = "nano"

# Show line numbers
line_numbers = false

# Tab width (spaces)
tab_width = 4

[search]
# Case sensitive search by default
case_sensitive = false

# Wrap around when reaching end of file
wrap_around = true

[fts_search]
# Full-text search indexing mode:
# - "auto": enable for files 10MB-200GB automatically
# - "always": always build FTS index
# - "never": disable FTS indexing (use regex only)
mode = "auto"

# Minimum file size to auto-enable FTS (bytes)
auto_min_size = 10485760  # 10 MB

# Maximum file size for auto-enable (bytes) - above this, user must confirm
auto_max_size = 214748364800  # 200 GB

# Maximum index size on disk (bytes) - 0 for unlimited
max_index_size = 5368709120  # 5 GB

# Use persistent cache (saves index between sessions)
persistent_cache = true

# Throttle for background indexing (MB/s, 0 for unlimited)
throttle_mb_per_sec = 0

[semantic_search]
# Enable semantic/fuzzy search indexing
enabled = false

# Embedding mode: "hybrid", "keywords", "hierarchical", "off"
# - hybrid: combines keyword extraction + hierarchical (best quality)
# - keywords: extract keywords, embed them (fast, good for exact terms)
# - hierarchical: embed text windows, mean pool (good for semantic meaning)
# - off: disable semantic search entirely
mode = "hybrid"

# Throttle for background indexing
# - "auto": benchmark disk speed, use 1/4 of it (recommended)
# - number: explicit MB/s limit (e.g., 100)
throttle = "auto"

# Keyword weight for hybrid mode (0.0 to 1.0)
# Higher = more weight on keyword matches
keyword_weight = 0.6

[save]
# Create backup before saving
create_backup = false

# Default save mode: "journal" or "full"
save_mode = "journal"
"#;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub editor: EditorConfig,
    pub search: SearchConfig,
    pub fts_search: FtsSearchConfig,
    pub semantic_search: SemanticSearchConfig,
    pub save: SaveConfig,
}

/// Editor preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EditorConfig {
    /// Default input style: "nano" or "vi"
    pub input_style: String,
    /// Show line numbers
    pub line_numbers: bool,
    /// Tab width in spaces
    pub tab_width: usize,
}

/// Search preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Case sensitive search
    pub case_sensitive: bool,
    /// Wrap around at EOF
    pub wrap_around: bool,
}

/// FTS (Full-Text Search) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FtsSearchConfig {
    /// Indexing mode: "auto", "always", "never"
    pub mode: FtsSearchMode,
    /// Minimum file size for auto-enable (bytes)
    pub auto_min_size: u64,
    /// Maximum file size for auto-enable (bytes)
    pub auto_max_size: u64,
    /// Maximum index size on disk (bytes), 0 for unlimited
    pub max_index_size: u64,
    /// Use persistent cache between sessions
    pub persistent_cache: bool,
    /// Throttle for indexing (MB/s), 0 for unlimited
    pub throttle_mb_per_sec: u64,
}

/// FTS indexing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FtsSearchMode {
    /// Auto-enable for files between min and max size
    #[default]
    Auto,
    /// Always build FTS index
    Always,
    /// Never use FTS (regex only)
    Never,
}

impl Default for FtsSearchConfig {
    fn default() -> Self {
        Self {
            mode: FtsSearchMode::Auto,
            auto_min_size: 10 * 1024 * 1024,           // 10 MB
            auto_max_size: 200 * 1024 * 1024 * 1024,   // 200 GB
            max_index_size: 5 * 1024 * 1024 * 1024,    // 5 GB
            persistent_cache: true,
            throttle_mb_per_sec: 0,                     // Unlimited
        }
    }
}

impl FtsSearchConfig {
    /// Check if FTS should be enabled for a given file size
    pub fn should_enable(&self, file_size: u64) -> bool {
        match self.mode {
            FtsSearchMode::Always => true,
            FtsSearchMode::Never => false,
            FtsSearchMode::Auto => {
                file_size >= self.auto_min_size && file_size <= self.auto_max_size
            }
        }
    }
    
    /// Get throttle in bytes per second
    pub fn throttle_bytes_per_sec(&self) -> u64 {
        if self.throttle_mb_per_sec == 0 {
            0 // Unlimited
        } else {
            self.throttle_mb_per_sec * 1024 * 1024
        }
    }
}

/// Semantic search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticSearchConfig {
    /// Enable semantic search
    pub enabled: bool,
    /// Embedding mode
    pub mode: SemanticSearchMode,
    /// Throttle setting
    pub throttle: ThrottleSetting,
    /// Keyword weight for hybrid mode (0.0-1.0)
    pub keyword_weight: f32,
}

/// Embedding mode for semantic search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SemanticSearchMode {
    /// Hybrid: keywords + hierarchical (best quality)
    #[default]
    Hybrid,
    /// Keywords only (fast, good for exact terms)
    Keywords,
    /// Hierarchical embedding (good for semantic meaning)
    Hierarchical,
    /// Disabled
    Off,
}

/// Throttle setting for background indexing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ThrottleSetting {
    /// Auto-benchmark disk speed, use 1/4
    #[serde(rename = "auto")]
    Auto,
    /// Manual MB/s limit
    Manual(u64),
}

impl Default for ThrottleSetting {
    fn default() -> Self {
        ThrottleSetting::Auto
    }
}

// Custom deserialize to handle "auto" string
impl ThrottleSetting {
    pub fn get_bytes_per_sec(&self, benchmarked_speed: Option<u64>) -> u64 {
        const MB: u64 = 1024 * 1024;
        const DEFAULT_SPEED: u64 = 100 * MB; // 100 MB/s fallback
        
        match self {
            ThrottleSetting::Auto => {
                benchmarked_speed.unwrap_or(DEFAULT_SPEED) / 4
            }
            ThrottleSetting::Manual(mb_s) => mb_s * MB,
        }
    }
}

/// Save preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SaveConfig {
    /// Create backup before saving
    pub create_backup: bool,
    /// Default save mode
    pub save_mode: String,
}

// Default implementations
impl Default for Config {
    fn default() -> Self {
        Self {
            editor: EditorConfig::default(),
            search: SearchConfig::default(),
            semantic_search: SemanticSearchConfig::default(),
            fts_search: FtsSearchConfig::default(),
            save: SaveConfig::default(),
        }
    }
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            input_style: "nano".to_string(),
            line_numbers: false,
            tab_width: 4,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            wrap_around: true,
        }
    }
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SemanticSearchMode::Hybrid,
            throttle: ThrottleSetting::Auto,
            keyword_weight: 0.6,
        }
    }
}

impl Default for SaveConfig {
    fn default() -> Self {
        Self {
            create_backup: false,
            save_mode: "journal".to_string(),
        }
    }
}

impl Config {
    /// Get the config file path
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("bigedit").join("config.toml"))
    }

    /// Load config from file, or return defaults
    pub fn load() -> Self {
        Self::try_load().unwrap_or_default()
    }

    /// Try to load config from file
    pub fn try_load() -> Result<Self> {
        let path = Self::config_path()
            .context("Could not determine config directory")?;
        
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        
        Ok(config)
    }

    /// Save current config to file
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()
            .context("Could not determine config directory")?;
        
        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }

        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;
        
        fs::write(&path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        
        Ok(())
    }

    /// Create default config file if it doesn't exist
    pub fn create_default_if_missing() -> Result<bool> {
        let path = Self::config_path()
            .context("Could not determine config directory")?;
        
        if path.exists() {
            return Ok(false);
        }

        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }

        fs::write(&path, DEFAULT_CONFIG)
            .with_context(|| format!("Failed to write default config: {}", path.display()))?;
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.editor.input_style, "nano");
        assert!(!config.semantic_search.enabled);
        assert_eq!(config.semantic_search.mode, SemanticSearchMode::Hybrid);
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
[editor]
input_style = "vi"
line_numbers = true

[semantic_search]
enabled = true
mode = "keywords"
throttle = 150
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.editor.input_style, "vi");
        assert!(config.editor.line_numbers);
        assert!(config.semantic_search.enabled);
        assert_eq!(config.semantic_search.mode, SemanticSearchMode::Keywords);
        assert_eq!(config.semantic_search.throttle, ThrottleSetting::Manual(150));
    }

    #[test]
    fn test_throttle_auto() {
        let throttle = ThrottleSetting::Auto;
        // With 400 MB/s benchmark, should use 100 MB/s
        let speed = throttle.get_bytes_per_sec(Some(400 * 1024 * 1024));
        assert_eq!(speed, 100 * 1024 * 1024);
    }

    #[test]
    fn test_throttle_manual() {
        let throttle = ThrottleSetting::Manual(50);
        let speed = throttle.get_bytes_per_sec(None);
        assert_eq!(speed, 50 * 1024 * 1024);
    }

    #[test]
    fn test_semantic_mode_parse() {
        // Test parsing modes within a full config structure
        let test_cases = [
            (r#"mode = "hybrid""#, SemanticSearchMode::Hybrid),
            (r#"mode = "keywords""#, SemanticSearchMode::Keywords),
            (r#"mode = "hierarchical""#, SemanticSearchMode::Hierarchical),
            (r#"mode = "off""#, SemanticSearchMode::Off),
        ];
        
        #[derive(Deserialize)]
        struct TestMode {
            mode: SemanticSearchMode,
        }
        
        for (toml_str, expected) in test_cases {
            let parsed: TestMode = toml::from_str(toml_str).unwrap();
            assert_eq!(parsed.mode, expected);
        }
    }
}
