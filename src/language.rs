//! Language detection for bigedit semantic search
//!
//! Detects the primary language of a file by sampling random chunks
//! and selecting the appropriate embedding model.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(feature = "semantic-search")]
use rand::Rng;

#[cfg(feature = "semantic-search")]
use whatlang::{detect, Lang};

/// Language families for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageFamily {
    /// English - use English-specific model
    English,
    /// Chinese (Simplified/Traditional) - use Chinese model
    Chinese,
    /// Other languages - use multilingual model
    Multilingual,
    /// Primarily code/data - use keyword-only or code model
    Code,
    /// Unknown or mixed - use multilingual model
    Unknown,
}

impl std::fmt::Display for LanguageFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LanguageFamily::English => write!(f, "English"),
            LanguageFamily::Chinese => write!(f, "Chinese"),
            LanguageFamily::Multilingual => write!(f, "Multilingual"),
            LanguageFamily::Code => write!(f, "Code"),
            LanguageFamily::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detected language information
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    /// Primary language family
    pub family: LanguageFamily,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Specific language if detected (ISO 639-3 code)
    pub specific_language: Option<String>,
    /// Distribution of detected languages
    pub distribution: HashMap<String, usize>,
}

/// Recommended embedding model based on detected language
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    /// Model identifier
    pub model_id: &'static str,
    /// Model file name (ONNX)
    pub model_file: &'static str,
    /// Tokenizer file name
    pub tokenizer_file: &'static str,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Human readable description
    pub description: &'static str,
}

/// Model configurations for different language families
impl EmbeddingModelConfig {
    pub fn for_english() -> Self {
        Self {
            model_id: "bge-small-en-v1.5",
            model_file: "bge-small-en-v1.5.onnx",
            tokenizer_file: "bge-small-en-v1.5-tokenizer.json",
            dimension: 384,
            max_seq_length: 512,
            description: "BAAI BGE Small English - optimized for English text",
        }
    }

    pub fn for_chinese() -> Self {
        Self {
            model_id: "bge-small-zh-v1.5",
            model_file: "bge-small-zh-v1.5.onnx",
            tokenizer_file: "bge-small-zh-v1.5-tokenizer.json",
            dimension: 512,
            max_seq_length: 512,
            description: "BAAI BGE Small Chinese - optimized for Chinese text",
        }
    }

    pub fn for_multilingual() -> Self {
        Self {
            model_id: "paraphrase-multilingual-MiniLM-L12-v2",
            model_file: "paraphrase-multilingual-MiniLM-L12-v2.onnx",
            tokenizer_file: "paraphrase-multilingual-MiniLM-L12-v2-tokenizer.json",
            dimension: 384,
            max_seq_length: 512,
            description: "Multilingual MiniLM - supports 50+ languages",
        }
    }

    pub fn for_code() -> Self {
        // For code, we use English model (most code comments/identifiers are English)
        Self {
            model_id: "bge-small-en-v1.5",
            model_file: "bge-small-en-v1.5.onnx",
            tokenizer_file: "bge-small-en-v1.5-tokenizer.json",
            dimension: 384,
            max_seq_length: 512,
            description: "Code mode - keyword extraction prioritized",
        }
    }

    pub fn for_family(family: LanguageFamily) -> Self {
        match family {
            LanguageFamily::English => Self::for_english(),
            LanguageFamily::Chinese => Self::for_chinese(),
            LanguageFamily::Multilingual => Self::for_multilingual(),
            LanguageFamily::Code => Self::for_code(),
            LanguageFamily::Unknown => Self::for_multilingual(),
        }
    }
}

/// Map whatlang Lang to our LanguageFamily
/// whatlang supports 69 languages - we map them to model families
#[cfg(feature = "semantic-search")]
fn lang_to_family(lang: Lang) -> LanguageFamily {
    match lang {
        Lang::Eng => LanguageFamily::English,
        Lang::Cmn => LanguageFamily::Chinese,
        // All other 67 languages use multilingual model
        _ => LanguageFamily::Multilingual,
    }
}

/// Get human-readable name for a language
#[cfg(feature = "semantic-search")]
fn lang_to_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Epo => "Esperanto",
        Lang::Eng => "English",
        Lang::Rus => "Russian",
        Lang::Cmn => "Chinese (Mandarin)",
        Lang::Spa => "Spanish",
        Lang::Por => "Portuguese",
        Lang::Ita => "Italian",
        Lang::Ben => "Bengali",
        Lang::Fra => "French",
        Lang::Deu => "German",
        Lang::Ukr => "Ukrainian",
        Lang::Kat => "Georgian",
        Lang::Ara => "Arabic",
        Lang::Hin => "Hindi",
        Lang::Jpn => "Japanese",
        Lang::Heb => "Hebrew",
        Lang::Yid => "Yiddish",
        Lang::Pol => "Polish",
        Lang::Amh => "Amharic",
        Lang::Jav => "Javanese",
        Lang::Kor => "Korean",
        Lang::Nob => "Norwegian (Bokmål)",
        Lang::Dan => "Danish",
        Lang::Swe => "Swedish",
        Lang::Fin => "Finnish",
        Lang::Tur => "Turkish",
        Lang::Nld => "Dutch",
        Lang::Hun => "Hungarian",
        Lang::Ces => "Czech",
        Lang::Ell => "Greek",
        Lang::Bul => "Bulgarian",
        Lang::Bel => "Belarusian",
        Lang::Mar => "Marathi",
        Lang::Kan => "Kannada",
        Lang::Ron => "Romanian",
        Lang::Slv => "Slovenian",
        Lang::Hrv => "Croatian",
        Lang::Srp => "Serbian",
        Lang::Mkd => "Macedonian",
        Lang::Lit => "Lithuanian",
        Lang::Lav => "Latvian",
        Lang::Est => "Estonian",
        Lang::Tam => "Tamil",
        Lang::Vie => "Vietnamese",
        Lang::Urd => "Urdu",
        Lang::Tha => "Thai",
        Lang::Guj => "Gujarati",
        Lang::Uzb => "Uzbek",
        Lang::Pan => "Punjabi",
        Lang::Aze => "Azerbaijani",
        Lang::Ind => "Indonesian",
        Lang::Tel => "Telugu",
        Lang::Pes => "Persian",
        Lang::Mal => "Malayalam",
        Lang::Ori => "Oriya",
        Lang::Mya => "Burmese",
        Lang::Nep => "Nepali",
        Lang::Sin => "Sinhalese",
        Lang::Khm => "Khmer",
        Lang::Tuk => "Turkmen",
        Lang::Aka => "Akan",
        Lang::Zul => "Zulu",
        Lang::Sna => "Shona",
        Lang::Afr => "Afrikaans",
        Lang::Lat => "Latin",
        Lang::Slk => "Slovak",
        Lang::Cat => "Catalan",
        Lang::Tgl => "Tagalog",
        Lang::Hye => "Armenian",
    }
}

/// Detect if content looks like source code
fn is_likely_code(text: &str) -> bool {
    // Code indicators
    let code_patterns = [
        "fn ", "pub ", "impl ", "struct ", "enum ",  // Rust
        "def ", "class ", "import ", "from ",        // Python
        "function ", "const ", "let ", "var ",       // JavaScript
        "public ", "private ", "void ", "int ",      // Java/C++
        "SELECT ", "INSERT ", "UPDATE ", "FROM ",    // SQL
        "<?php", "<?=",                               // PHP
        "<html", "<div", "<script",                  // HTML
        "package ", "func ",                         // Go
    ];
    
    let text_lower = text.to_lowercase();
    let pattern_matches = code_patterns.iter()
        .filter(|p| text_lower.contains(&p.to_lowercase()))
        .count();
    
    // Check for high symbol density (code tends to have lots of brackets, semicolons)
    let symbol_count = text.chars()
        .filter(|c| matches!(c, '{' | '}' | '[' | ']' | '(' | ')' | ';' | '=' | '<' | '>'))
        .count();
    let symbol_ratio = symbol_count as f32 / text.len().max(1) as f32;
    
    // Check for lack of natural language features
    let has_sentences = text.contains(". ") || text.contains("? ") || text.contains("! ");
    
    // Consider it code if:
    // - Multiple code patterns found, OR
    // - High symbol density AND few sentences
    pattern_matches >= 2 || (symbol_ratio > 0.05 && !has_sentences)
}

/// Detect language from file by sampling random chunks
#[cfg(feature = "semantic-search")]
pub fn detect_file_language(path: &Path) -> Result<LanguageDetectionResult> {
    let mut file = File::open(path).context("Failed to open file for language detection")?;
    let file_size = file.metadata()?.len();
    
    if file_size == 0 {
        return Ok(LanguageDetectionResult {
            family: LanguageFamily::Unknown,
            confidence: 0.0,
            specific_language: None,
            distribution: HashMap::new(),
        });
    }
    
    // Sample configuration
    const SAMPLE_SIZE: usize = 4096;
    const MAX_SAMPLES: usize = 24;
    
    let num_samples = if file_size < SAMPLE_SIZE as u64 {
        1
    } else {
        MAX_SAMPLES.min((file_size / SAMPLE_SIZE as u64) as usize)
    };
    
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);
    let mut combined_text = String::new();
    
    // Collect random samples
    for _ in 0..num_samples {
        let max_offset = file_size.saturating_sub(SAMPLE_SIZE as u64);
        let offset = if max_offset > 0 {
            rng.gen_range(0..max_offset)
        } else {
            0
        };
        
        file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; SAMPLE_SIZE];
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);
        
        // Convert to string, skipping invalid UTF-8
        if let Ok(text) = String::from_utf8(buffer.clone()) {
            samples.push(text.clone());
            combined_text.push_str(&text);
            combined_text.push('\n');
        } else {
            // Try lossy conversion
            let text = String::from_utf8_lossy(&buffer).into_owned();
            samples.push(text.clone());
            combined_text.push_str(&text);
            combined_text.push('\n');
        }
    }
    
    // Check if it's primarily code
    let code_sample_count = samples.iter()
        .filter(|s| is_likely_code(s))
        .count();
    
    if code_sample_count as f32 / samples.len() as f32 > 0.6 {
        return Ok(LanguageDetectionResult {
            family: LanguageFamily::Code,
            confidence: code_sample_count as f32 / samples.len() as f32,
            specific_language: None,
            distribution: HashMap::new(),
        });
    }
    
    // Detect language in each sample
    let mut lang_counts: HashMap<String, usize> = HashMap::new();
    let mut family_counts: HashMap<LanguageFamily, usize> = HashMap::new();
    
    for sample in &samples {
        if let Some(info) = detect(sample) {
            let lang = info.lang();
            let name = lang_to_name(lang).to_string();
            let family = lang_to_family(lang);
            
            *lang_counts.entry(name).or_insert(0) += 1;
            *family_counts.entry(family).or_insert(0) += 1;
        }
    }
    
    // Find the dominant language family
    let (dominant_family, count) = family_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(f, c)| (*f, *c))
        .unwrap_or((LanguageFamily::Unknown, 0));
    
    // Find the most common specific language
    let specific_language = lang_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(name, _)| name.clone());
    
    let confidence = if samples.is_empty() {
        0.0
    } else {
        count as f32 / samples.len() as f32
    };
    
    Ok(LanguageDetectionResult {
        family: dominant_family,
        confidence,
        specific_language,
        distribution: lang_counts,
    })
}

/// Detect language from a string slice (for testing or small content)
#[cfg(feature = "semantic-search")]
pub fn detect_text_language(text: &str) -> LanguageDetectionResult {
    if text.is_empty() {
        return LanguageDetectionResult {
            family: LanguageFamily::Unknown,
            confidence: 0.0,
            specific_language: None,
            distribution: HashMap::new(),
        };
    }
    
    // Check for code first
    if is_likely_code(text) {
        return LanguageDetectionResult {
            family: LanguageFamily::Code,
            confidence: 0.8,
            specific_language: None,
            distribution: HashMap::new(),
        };
    }
    
    // Detect language
    if let Some(info) = detect(text) {
        let lang = info.lang();
        let name = lang_to_name(lang).to_string();
        let family = lang_to_family(lang);
        let confidence = info.confidence() as f32;
        
        let mut distribution = HashMap::new();
        distribution.insert(name.clone(), 1);
        
        LanguageDetectionResult {
            family,
            confidence,
            specific_language: Some(name),
            distribution,
        }
    } else {
        LanguageDetectionResult {
            family: LanguageFamily::Unknown,
            confidence: 0.0,
            specific_language: None,
            distribution: HashMap::new(),
        }
    }
}

/// Stub for non-semantic-search builds
#[cfg(not(feature = "semantic-search"))]
pub fn detect_file_language(_path: &Path) -> Result<LanguageDetectionResult> {
    Ok(LanguageDetectionResult {
        family: LanguageFamily::Unknown,
        confidence: 0.0,
        specific_language: None,
        distribution: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_detection() {
        let rust_code = r#"
            fn main() {
                let x = 42;
                println!("Hello, world!");
            }
        "#;
        assert!(is_likely_code(rust_code));
        
        let english_text = "This is a normal English sentence. It has punctuation and reads naturally.";
        assert!(!is_likely_code(english_text));
        
        let python_code = r#"
            def hello():
                import os
                from sys import path
                return "Hello"
        "#;
        assert!(is_likely_code(python_code));
    }

    #[test]
    fn test_model_config() {
        let english = EmbeddingModelConfig::for_english();
        assert_eq!(english.dimension, 384);
        
        let chinese = EmbeddingModelConfig::for_chinese();
        assert_eq!(chinese.dimension, 512);
        
        let multi = EmbeddingModelConfig::for_multilingual();
        assert_eq!(multi.dimension, 384);
    }

    #[cfg(feature = "semantic-search")]
    #[test]
    fn test_text_language_detection() {
        let english = "The quick brown fox jumps over the lazy dog. This is clearly English text.";
        let result = detect_text_language(english);
        assert_eq!(result.family, LanguageFamily::English);
        assert!(result.confidence > 0.5);
        
        let spanish = "Hola, ¿cómo estás? Esta es una oración en español muy clara.";
        let result = detect_text_language(spanish);
        assert_eq!(result.family, LanguageFamily::Multilingual);
        
        let code = "fn main() { let x = vec![1,2,3]; for i in x { println!(\"{}\", i); } }";
        let result = detect_text_language(code);
        assert_eq!(result.family, LanguageFamily::Code);
    }

    #[cfg(feature = "semantic-search")]
    #[test]
    fn test_lang_mapping() {
        assert_eq!(lang_to_family(Lang::Eng), LanguageFamily::English);
        assert_eq!(lang_to_family(Lang::Cmn), LanguageFamily::Chinese);
        assert_eq!(lang_to_family(Lang::Spa), LanguageFamily::Multilingual);
        assert_eq!(lang_to_family(Lang::Jpn), LanguageFamily::Multilingual);
        assert_eq!(lang_to_family(Lang::Ara), LanguageFamily::Multilingual);
        assert_eq!(lang_to_family(Lang::Zul), LanguageFamily::Multilingual);
    }

    #[cfg(feature = "semantic-search")]
    #[test]
    fn test_all_langs_have_names() {
        // Verify all 69 whatlang languages have names
        let langs = [
            Lang::Epo, Lang::Eng, Lang::Rus, Lang::Cmn, Lang::Spa,
            Lang::Por, Lang::Ita, Lang::Ben, Lang::Fra, Lang::Deu,
            Lang::Ukr, Lang::Kat, Lang::Ara, Lang::Hin, Lang::Jpn,
            Lang::Heb, Lang::Yid, Lang::Pol, Lang::Amh, Lang::Jav,
            Lang::Kor, Lang::Nob, Lang::Dan, Lang::Swe, Lang::Fin,
            Lang::Tur, Lang::Nld, Lang::Hun, Lang::Ces, Lang::Ell,
            Lang::Bul, Lang::Bel, Lang::Mar, Lang::Kan, Lang::Ron,
            Lang::Slv, Lang::Hrv, Lang::Srp, Lang::Mkd, Lang::Lit,
            Lang::Lav, Lang::Est, Lang::Tam, Lang::Vie, Lang::Urd,
            Lang::Tha, Lang::Guj, Lang::Uzb, Lang::Pan, Lang::Aze,
            Lang::Ind, Lang::Tel, Lang::Pes, Lang::Mal, Lang::Ori,
            Lang::Mya, Lang::Nep, Lang::Sin, Lang::Khm, Lang::Tuk,
            Lang::Aka, Lang::Zul, Lang::Sna, Lang::Afr, Lang::Lat,
            Lang::Slk, Lang::Cat, Lang::Tgl, Lang::Hye,
        ];
        
        for lang in langs {
            let name = lang_to_name(lang);
            assert!(!name.is_empty(), "Language {:?} has no name", lang);
        }
    }
}
