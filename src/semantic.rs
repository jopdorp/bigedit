//! Semantic search for bigedit
//!
//! This module provides fuzzy/semantic search capabilities using embeddings.
//! It builds a background index of file chunks and allows fast similarity search.
//!
//! Architecture (hybrid approach for speed + quality):
//! 1. **Model2Vec** (fast static embeddings) - indexes all chunks at ~10 MB/s
//! 2. **IDF pre-filtering** - finds top-k candidates with query term overlap  
//! 3. **ONNX re-ranking** - neural re-rank of top candidates for quality
//!
//! Flow:
//! 1. Index file once with Model2Vec (fast: 136MB Wikipedia in ~6s)
//! 2. Search: IDF filter → Model2Vec similarity → optional ONNX re-rank

use anyhow::{Context, Result};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "semantic-search")]
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "semantic-search")]
use tokenizers::Tokenizer;

use crate::config::{SemanticSearchConfig, SemanticSearchMode, ThrottleSetting};

/// Size constants
const KB: u64 = 1024;
const MB: u64 = 1024 * KB;
const GB: u64 = 1024 * MB;

/// Embedding vector dimension (BGE-small uses 384, Model2Vec multilingual uses 256)
pub const EMBEDDING_DIM: usize = 384;

/// Model2Vec embedding dimension (potion models use 256)
pub const MODEL2VEC_DIM: usize = 256;

/// Calculate optimal chunk size based on file size
pub fn chunk_size_for_file(file_size: u64) -> usize {
    match file_size {
        0..=10_000_000 =>           4 * 1024,    // ≤10MB: 4KB chunks
        0..=100_000_000 =>          8 * 1024,    // ≤100MB: 8KB chunks  
        0..=500_000_000 =>         16 * 1024,    // ≤500MB: 16KB chunks
        0..=1_000_000_000 =>       32 * 1024,    // ≤1GB: 32KB chunks
        0..=5_000_000_000 =>       64 * 1024,    // ≤5GB: 64KB chunks
        0..=20_000_000_000 =>     128 * 1024,    // ≤20GB: 128KB chunks
        0..=50_000_000_000 =>     256 * 1024,    // ≤50GB: 256KB chunks
        0..=100_000_000_000 =>    512 * 1024,    // ≤100GB: 512KB chunks
        0..=200_000_000_000 =>   1024 * 1024,    // ≤200GB: 1MB chunks
        _ =>                     4096 * 1024,    // >200GB: 4MB chunks
    }
}

/// Benchmark disk read speed and return bytes/sec
pub fn benchmark_disk_speed(path: &Path) -> Result<u64> {
    const BENCHMARK_SIZE: usize = 16 * 1024 * 1024; // Read 16MB
    const MIN_SPEED: u64 = 50 * MB;  // Floor at 50MB/s
    const MAX_SPEED: u64 = 2000 * MB; // Cap at 2GB/s
    
    let mut file = File::open(path).context("Failed to open file for benchmark")?;
    let file_size = file.metadata()?.len();
    
    if file_size < 1024 {
        // File too small to benchmark meaningfully
        return Ok(100 * MB);
    }
    
    // Read from middle of file (avoid cached start)
    let read_size = BENCHMARK_SIZE.min((file_size / 2) as usize);
    let offset = file_size / 4;
    
    let mut buffer = vec![0u8; read_size];
    file.seek(SeekFrom::Start(offset))?;
    
    let start = Instant::now();
    file.read_exact(&mut buffer)?;
    let elapsed = start.elapsed();
    
    // Avoid division by zero
    if elapsed.as_nanos() == 0 {
        return Ok(MAX_SPEED);
    }
    
    let speed = (read_size as f64 / elapsed.as_secs_f64()) as u64;
    Ok(speed.clamp(MIN_SPEED, MAX_SPEED))
}

/// A single chunk in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedChunk {
    /// Byte offset in the file where this chunk starts
    pub offset: u64,
    /// Length of the chunk in bytes
    pub length: usize,
    /// Embedding vector (quantized to i8 for memory efficiency)
    pub embedding: Vec<i8>,
    /// Token frequencies for IDF-based pre-filtering (top tokens only)
    /// Maps token hash → count for fast lookup
    pub token_freqs: Vec<(u32, u16)>,
}

/// Search result from semantic search
#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    /// Byte offset in file
    pub offset: u64,
    /// Chunk length
    pub length: usize,
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
    /// Preview text (first ~100 chars of chunk)
    pub preview: String,
}

/// Serializable index data for persistence
#[derive(Serialize, Deserialize)]
struct PersistedIndex {
    /// Version for forward compatibility
    version: u32,
    /// File path (for verification)
    file_path: String,
    /// File size when indexed (for staleness check)
    file_size: u64,
    /// File modification time (for staleness check)
    mtime_secs: u64,
    /// Chunk size used
    chunk_size: usize,
    /// All indexed chunks
    chunks: Vec<IndexedChunk>,
    /// IDF scores
    idf_scores: HashMap<u32, f32>,
    /// Document count
    doc_count: usize,
}

/// The semantic search index
pub struct SemanticIndex {
    /// Path to the indexed file
    file_path: PathBuf,
    /// File size when indexed
    file_size: u64,
    /// Chunk size used
    chunk_size: usize,
    /// Indexed chunks
    chunks: Vec<IndexedChunk>,
    /// Configuration
    config: SemanticSearchConfig,
    /// Whether indexing is complete
    indexing_complete: AtomicBool,
    /// Current indexing progress (bytes processed)
    bytes_indexed: AtomicU64,
    /// Throttle in bytes/sec
    throttle_bytes_per_sec: u64,
    /// Global IDF scores: token_hash → log(N/df) for pre-filtering
    idf_scores: HashMap<u32, f32>,
    /// Total document (chunk) count for IDF
    doc_count: usize,
}

impl SemanticIndex {
    /// Create a new semantic index for a file
    pub fn new(file_path: &Path, config: SemanticSearchConfig) -> Result<Self> {
        let file = File::open(file_path)?;
        let file_size = file.metadata()?.len();
        let chunk_size = chunk_size_for_file(file_size);
        
        // Benchmark disk speed if throttle is auto
        let throttle_bytes_per_sec = match &config.throttle {
            ThrottleSetting::Auto => {
                let disk_speed = benchmark_disk_speed(file_path)?;
                disk_speed / 4
            }
            ThrottleSetting::Manual(mb_s) => mb_s * MB,
        };
        
        Ok(Self {
            file_path: file_path.to_path_buf(),
            file_size,
            chunk_size,
            chunks: Vec::new(),
            config,
            indexing_complete: AtomicBool::new(false),
            bytes_indexed: AtomicU64::new(0),
            throttle_bytes_per_sec,
            idf_scores: HashMap::new(),
            doc_count: 0,
        })
    }
    
    /// Get indexing progress as a percentage
    pub fn progress(&self) -> f32 {
        if self.file_size == 0 {
            return 100.0;
        }
        let indexed = self.bytes_indexed.load(Ordering::Relaxed);
        (indexed as f32 / self.file_size as f32) * 100.0
    }
    
    /// Check if indexing is complete
    pub fn is_complete(&self) -> bool {
        self.indexing_complete.load(Ordering::Relaxed)
    }
    
    /// Get estimated chunks count
    pub fn estimated_chunks(&self) -> usize {
        ((self.file_size as usize) / self.chunk_size) + 1
    }
    
    /// Get current chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
    
    /// Get chunk size being used
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    
    /// Get throttle speed
    pub fn throttle_mb_s(&self) -> u64 {
        self.throttle_bytes_per_sec / MB
    }
    
    /// Search the index for similar chunks using hybrid IDF + embedding approach
    /// 
    /// 1. If query has tokens, use IDF to find top-k candidates with term overlap
    /// 2. Score candidates with embedding similarity
    /// 3. Return top results
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SemanticSearchResult> {
        self.search_with_idf(query_embedding, None, top_k)
    }
    
    /// Search with optional IDF pre-filtering using query tokens
    /// 
    /// `query_tokens`: Optional token hashes from the query for IDF pre-filtering
    /// If provided, will first narrow down to chunks with term overlap before
    /// computing expensive embedding similarity.
    pub fn search_with_idf(
        &self, 
        query_embedding: &[f32], 
        query_tokens: Option<&[(u32, u16)]>,
        top_k: usize
    ) -> Vec<SemanticSearchResult> {
        if self.chunks.is_empty() {
            return Vec::new();
        }
        
        // Determine which chunks to score
        let candidates: Vec<usize> = if let Some(tokens) = query_tokens {
            // IDF pre-filtering: score chunks by BM25-like term overlap
            let prefilter_k = (top_k * 100).min(self.chunks.len()); // Get 100x candidates for re-ranking
            
            let mut chunk_scores: Vec<(usize, f32)> = self.chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let score = self.idf_score(tokens, &chunk.token_freqs);
                    (i, score)
                })
                .filter(|(_, score)| *score > 0.0) // Only chunks with term overlap
                .collect();
            
            chunk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            chunk_scores.truncate(prefilter_k);
            
            if chunk_scores.is_empty() {
                // No term overlap, fall back to all chunks
                (0..self.chunks.len()).collect()
            } else {
                chunk_scores.into_iter().map(|(i, _)| i).collect()
            }
        } else {
            // No IDF filtering, score all chunks
            (0..self.chunks.len()).collect()
        };
        
        // Calculate embedding similarity for candidates
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&i| {
                let chunk = &self.chunks[i];
                let score = cosine_similarity_quantized(query_embedding, &chunk.embedding);
                (i, score)
            })
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top_k results
        results.truncate(top_k);
        
        // Convert to SemanticSearchResult
        results
            .into_iter()
            .map(|(i, score)| {
                let chunk = &self.chunks[i];
                SemanticSearchResult {
                    offset: chunk.offset,
                    length: chunk.length,
                    score,
                    preview: String::new(), // Will be filled by caller
                }
            })
            .collect()
    }
    
    /// Calculate IDF-weighted term overlap score between query and chunk
    fn idf_score(&self, query_tokens: &[(u32, u16)], chunk_tokens: &[(u32, u16)]) -> f32 {
        let mut score = 0.0f32;
        
        for (query_hash, query_count) in query_tokens {
            // Check if chunk contains this token
            for (chunk_hash, chunk_count) in chunk_tokens {
                if query_hash == chunk_hash {
                    // Get IDF weight (default to 1.0 if not found)
                    let idf = self.idf_scores.get(query_hash).copied().unwrap_or(1.0);
                    // TF-IDF style score: min of query/chunk count * IDF
                    score += (*query_count.min(chunk_count) as f32) * idf;
                    break;
                }
            }
        }
        
        score
    }
    
    /// Add a chunk to the index with embedding only (legacy)
    pub fn add_chunk(&mut self, offset: u64, length: usize, embedding: Vec<f32>) {
        self.add_chunk_with_tokens(offset, length, embedding, Vec::new());
    }
    
    /// Add a chunk to the index with embedding and token frequencies
    pub fn add_chunk_with_tokens(
        &mut self, 
        offset: u64, 
        length: usize, 
        embedding: Vec<f32>,
        token_freqs: Vec<(u32, u16)>
    ) {
        // Quantize to i8 for memory efficiency
        let quantized = quantize_embedding(&embedding);
        self.chunks.push(IndexedChunk {
            offset,
            length,
            embedding: quantized,
            token_freqs,
        });
        self.bytes_indexed.fetch_add(length as u64, Ordering::Relaxed);
    }
    
    /// Finalize the index: compute IDF scores from collected token frequencies
    pub fn finalize_idf(&mut self) {
        // Count document frequency for each token
        let mut doc_freq: HashMap<u32, usize> = HashMap::new();
        
        for chunk in &self.chunks {
            for (token_hash, _) in &chunk.token_freqs {
                *doc_freq.entry(*token_hash).or_insert(0) += 1;
            }
        }
        
        // Compute IDF: log(N / df) with smoothing
        let n = self.chunks.len() as f32;
        self.doc_count = self.chunks.len();
        
        for (token_hash, df) in doc_freq {
            let idf = (n / (df as f32 + 1.0)).ln() + 1.0; // +1 smoothing
            self.idf_scores.insert(token_hash, idf);
        }
    }
    
    /// Mark indexing as complete
    pub fn mark_complete(&self) {
        self.indexing_complete.store(true, Ordering::Relaxed);
    }
    
    /// Get the cache path for this file's index
    fn cache_path(file_path: &Path) -> PathBuf {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let cache_base = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("bigedit")
            .join("index");
        
        // Hash the canonical path for uniqueness
        let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.to_path_buf());
        let mut hasher = DefaultHasher::new();
        canonical.hash(&mut hasher);
        let hash = hasher.finish();
        
        cache_base.join(format!("{:016x}.idx", hash))
    }
    
    /// Try to load a cached index for this file
    /// 
    /// Returns None if:
    /// - No cache exists
    /// - Cache is stale (file modified since indexing)
    /// - Cache is corrupted or incompatible version
    pub fn load_cached(file_path: &Path, config: SemanticSearchConfig) -> Option<Self> {
        let cache_path = Self::cache_path(file_path);
        
        // Check if cache file exists
        if !cache_path.exists() {
            return None;
        }
        
        // Get current file metadata
        let file_meta = std::fs::metadata(file_path).ok()?;
        let file_size = file_meta.len();
        let mtime = file_meta.modified().ok()?
            .duration_since(std::time::UNIX_EPOCH).ok()?
            .as_secs();
        
        // Load and deserialize
        let cache_file = File::open(&cache_path).ok()?;
        let reader = BufReader::new(cache_file);
        let persisted: PersistedIndex = bincode::deserialize_from(reader).ok()?;
        
        // Version check
        if persisted.version != 1 {
            eprintln!("[cache] Index version mismatch, will rebuild");
            return None;
        }
        
        // Staleness check
        if persisted.file_size != file_size || persisted.mtime_secs != mtime {
            eprintln!("[cache] Index stale (file modified), will rebuild");
            return None;
        }
        
        eprintln!("[cache] Loaded {} chunks from disk cache", persisted.chunks.len());
        
        // Reconstruct the index
        let throttle_bytes_per_sec = match &config.throttle {
            ThrottleSetting::Auto => 100 * MB, // Default since we're loading
            ThrottleSetting::Manual(mb_s) => mb_s * MB,
        };
        
        Some(Self {
            file_path: file_path.to_path_buf(),
            file_size,
            chunk_size: persisted.chunk_size,
            chunks: persisted.chunks,
            config,
            indexing_complete: AtomicBool::new(true),
            bytes_indexed: AtomicU64::new(file_size),
            throttle_bytes_per_sec,
            idf_scores: persisted.idf_scores,
            doc_count: persisted.doc_count,
        })
    }
    
    /// Save the index to disk cache
    /// 
    /// Should be called after indexing is complete
    pub fn save_to_cache(&self) -> Result<()> {
        if !self.is_complete() {
            anyhow::bail!("Cannot save incomplete index");
        }
        
        let cache_path = Self::cache_path(&self.file_path);
        
        // Create cache directory
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Get file modification time
        let file_meta = std::fs::metadata(&self.file_path)?;
        let mtime_secs = file_meta.modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        let persisted = PersistedIndex {
            version: 1,
            file_path: self.file_path.to_string_lossy().to_string(),
            file_size: self.file_size,
            mtime_secs,
            chunk_size: self.chunk_size,
            chunks: self.chunks.clone(),
            idf_scores: self.idf_scores.clone(),
            doc_count: self.doc_count,
        };
        
        // Write to temp file then rename for atomicity
        let temp_path = cache_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &persisted)
            .map_err(|e| anyhow::anyhow!("Serialize error: {}", e))?;
        
        std::fs::rename(&temp_path, &cache_path)?;
        
        eprintln!("[cache] Saved {} chunks to {}", self.chunks.len(), cache_path.display());
        
        Ok(())
    }
    
    /// Clear the cache for this file
    pub fn clear_cache(file_path: &Path) -> Result<()> {
        let cache_path = Self::cache_path(file_path);
        if cache_path.exists() {
            std::fs::remove_file(&cache_path)?;
        }
        Ok(())
    }
    
    /// Build the index from file using FastEmbeddingModel
    /// 
    /// This is a standalone method that can be called from a background thread.
    /// Uses parallel processing for speed.
    pub fn build_from_file<F>(&mut self, model: &FastEmbeddingModel, mut progress_callback: F) -> Result<()>
    where F: FnMut(usize, usize, u64, u64)
    {
        use std::io::BufRead;
        use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
        use std::sync::Arc;
        use rayon::prelude::*;
        
        // Use physical cores for RAM-bound workload
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let physical_cores = std::cmp::max(1, num_cpus / 2);
        
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(physical_cores)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;
        
        let chunk_size = self.chunk_size;
        
        // Streaming: process file in 500MB windows
        const WINDOW_SIZE: u64 = 500 * 1024 * 1024;
        let batch_size = 200;
        
        let total_chunks = (self.file_size as usize + chunk_size - 1) / chunk_size;
        let chunks_done = Arc::new(AtomicUsize::new(0));
        let bytes_done = Arc::new(AtomicU64::new(0));
        
        let mut window_start = 0u64;
        
        while window_start < self.file_size {
            let window_end = std::cmp::min(window_start + WINDOW_SIZE, self.file_size);
            
            // Read this window's chunks
            let mut file = File::open(&self.file_path)?;
            file.seek(SeekFrom::Start(window_start))?;
            
            let mut line_reader = std::io::BufReader::new(file);
            let mut current_offset = window_start;
            let mut window_chunks: Vec<(u64, String)> = Vec::new();
            
            loop {
                let mut chunk = String::new();
                let mut bytes_read = 0;
                
                loop {
                    let mut line = String::new();
                    match line_reader.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(n) => {
                            bytes_read += n;
                            chunk.push_str(&line);
                            if bytes_read >= chunk_size {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                
                if chunk.is_empty() {
                    break;
                }
                
                let chunk_end = current_offset + chunk.len() as u64;
                window_chunks.push((current_offset, chunk));
                current_offset = chunk_end;
                
                if current_offset >= window_end {
                    break;
                }
            }
            
            // Process in parallel batches
            for batch in window_chunks.chunks(batch_size) {
                let batch_results: Vec<_> = pool.install(|| {
                    batch.par_iter()
                        .map(|(offset, text)| {
                            let embedding = model.embed(text).unwrap_or_else(|_| vec![0.0; MODEL2VEC_DIM]);
                            let tokens = tokenize_for_idf(text);
                            (*offset, text.len(), embedding, tokens)
                        })
                        .collect()
                });
                
                for (offset, length, embedding, tokens) in batch_results {
                    self.add_chunk_with_tokens(offset, length, embedding, tokens);
                    chunks_done.fetch_add(1, Ordering::Relaxed);
                    bytes_done.fetch_add(length as u64, Ordering::Relaxed);
                }
                
                progress_callback(
                    chunks_done.load(Ordering::Relaxed),
                    total_chunks,
                    bytes_done.load(Ordering::Relaxed),
                    self.file_size
                );
            }
            
            window_start = current_offset;
        }
        
        self.finalize_idf();
        self.mark_complete();
        Ok(())
    }
}

/// Embedding model wrapper for ONNX inference
#[cfg(feature = "semantic-search")]
pub struct EmbeddingModel {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
    max_seq_length: usize,
}

#[cfg(feature = "semantic-search")]
impl EmbeddingModel {
    /// Load model from ONNX file and tokenizer JSON
    /// 
    /// Automatically tries GPU acceleration if available:
    /// - Intel GPU via OpenVINO
    /// - Falls back to CPU if no GPU available
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        use ort::execution_providers::ExecutionProvider;
        
        let mut builder = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?;
        
        // Try to use GPU execution providers if available
        // OpenVINO for Intel GPUs (including Arc)
        #[cfg(feature = "openvino")]
        {
            use ort::execution_providers::OpenVINOExecutionProvider;
            let openvino = OpenVINOExecutionProvider::default();
            if openvino.is_available() {
                eprintln!("Using OpenVINO (Intel GPU) for embeddings");
                builder = builder.with_execution_providers([openvino.build()])
                    .context("Failed to configure OpenVINO")?;
            }
        }
        
        // CUDA for NVIDIA GPUs
        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::CUDAExecutionProvider;
            let cuda = CUDAExecutionProvider::default();
            if cuda.is_available() {
                eprintln!("Using CUDA (NVIDIA GPU) for embeddings");
                builder = builder.with_execution_providers([cuda.build()])
                    .context("Failed to configure CUDA")?;
            }
        }
        
        let session = builder.commit_from_file(model_path)
            .with_context(|| format!("Failed to load ONNX model: {}", model_path.display()))?;
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // BGE-small uses 384 dimensions
        Ok(Self {
            session,
            tokenizer,
            dimension: 384,
            max_seq_length: 512,
        })
    }
    
    /// Load the default bundled model (downloads if needed)
    pub fn load_default() -> Result<Self> {
        let model_dir = Self::model_dir()?;
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        if !model_path.exists() || !tokenizer_path.exists() {
            Self::download_default_model(&model_dir)?;
        }
        
        Self::load(&model_path, &tokenizer_path)
    }
    
    /// Get model directory
    fn model_dir() -> Result<PathBuf> {
        let dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bigedit")
            .join("models")
            .join("bge-small-en");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }
    
    /// Download the default model
    fn download_default_model(model_dir: &Path) -> Result<()> {
        use std::process::Command;
        
        // BGE-small-en-v1.5 ONNX from HuggingFace
        let model_url = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
        let tokenizer_url = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json";
        
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        eprintln!("Downloading embedding model (~33MB)...");
        
        // Download model
        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&model_path)
            .arg(model_url)
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to download model");
        }
        
        // Download tokenizer
        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&tokenizer_path)
            .arg(tokenizer_url)
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to download tokenizer");
        }
        
        eprintln!("Model downloaded to {}", model_dir.display());
        Ok(())
    }
    
    /// Embed a single text, returns normalized embedding
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let input_ids: Vec<i64> = encoding.get_ids()
            .iter()
            .take(self.max_seq_length)
            .map(|&id| id as i64)
            .collect();
        
        let attention_mask: Vec<i64> = encoding.get_attention_mask()
            .iter()
            .take(self.max_seq_length)
            .map(|&m| m as i64)
            .collect();
        
        let token_type_ids: Vec<i64> = encoding.get_type_ids()
            .iter()
            .take(self.max_seq_length)
            .map(|&t| t as i64)
            .collect();
        
        let seq_len = input_ids.len();
        
        // Create input tensors using ort's Tensor::from_array with (shape, data) tuple
        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))?;
        let token_type_ids_tensor = Tensor::from_array(([1, seq_len], token_type_ids))?;
        
        // Run inference - ort::inputs! returns a Vec, not Result
        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];
        let outputs = self.session.run(inputs)?;
        
        // Extract embedding from output (mean pooling over sequence)
        // Try different output names as models vary
        let output = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("sentence_embedding"))
            .or_else(|| outputs.get("output"))
            .context("Model output not found")?;
        
        // try_extract_array returns ndarray::ArrayViewD<f32>
        let array: ndarray::ArrayViewD<f32> = output.try_extract_array()?;
        let shape = array.shape();
        
        let embedding: Vec<f32> = if shape.len() == 3 {
            // Shape: [batch, seq_len, hidden_dim] - need mean pooling
            let hidden_dim = shape[2];
            let seq_len = shape[1];
            // Mean pool over sequence dimension
            (0..hidden_dim)
                .map(|d| {
                    (0..seq_len).map(|s| array[[0, s, d]]).sum::<f32>() / seq_len as f32
                })
                .collect()
        } else if shape.len() == 2 {
            // Shape: [batch, hidden_dim] - already pooled
            array.iter().cloned().collect()
        } else {
            anyhow::bail!("Unexpected output shape: {:?}", shape);
        };
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };
        
        Ok(normalized)
    }
    
    /// Embed multiple texts in batch (true batched ONNX inference for efficiency)
    /// 
    /// This runs all texts through the model in a single forward pass, which is
    /// much faster than calling embed() in a loop due to reduced ONNX overhead.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use ort::value::Tensor;
        
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // For very small batches, single inference may be faster
        if texts.len() == 1 {
            return Ok(vec![self.embed(texts[0])?]);
        }
        
        // Tokenize all texts and find max sequence length
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| self.tokenizer.encode(*text, true))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_length))
            .max()
            .unwrap_or(1);
        
        let batch_size = encodings.len();
        
        // Prepare padded batch tensors
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        
        for encoding in &encodings {
            let ids = encoding.get_ids();
            let attn = encoding.get_attention_mask();
            let types = encoding.get_type_ids();
            
            let seq_len = ids.len().min(self.max_seq_length);
            
            // Add actual tokens
            for i in 0..seq_len {
                input_ids.push(ids[i] as i64);
                attention_mask.push(attn[i] as i64);
                token_type_ids.push(types[i] as i64);
            }
            
            // Pad to max_len (pad_token_id is typically 0)
            for _ in seq_len..max_len {
                input_ids.push(0);      // padding token
                attention_mask.push(0); // mask padding
                token_type_ids.push(0);
            }
        }
        
        // Create batch tensors
        let input_ids_tensor = Tensor::from_array(([batch_size, max_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([batch_size, max_len], attention_mask.clone()))?;
        let token_type_ids_tensor = Tensor::from_array(([batch_size, max_len], token_type_ids))?;
        
        // Run batch inference
        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];
        let outputs = self.session.run(inputs)?;
        
        // Extract embeddings
        let output = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("sentence_embedding"))
            .or_else(|| outputs.get("output"))
            .context("Model output not found")?;
        
        let array: ndarray::ArrayViewD<f32> = output.try_extract_array()?;
        let shape = array.shape();
        
        let mut results = Vec::with_capacity(batch_size);
        
        if shape.len() == 3 {
            // Shape: [batch, seq_len, hidden_dim] - need masked mean pooling
            let hidden_dim = shape[2];
            let seq_len_out = shape[1];
            
            // Re-create attention mask as f32 for proper masking in mean pool
            let mut attn_offset = 0;
            for batch_idx in 0..batch_size {
                // Masked mean pooling: only average over non-padding tokens
                let mut embedding = vec![0.0f32; hidden_dim];
                let mut valid_tokens = 0.0f32;
                
                for s in 0..seq_len_out {
                    let mask_val = attention_mask[attn_offset + s];
                    if mask_val > 0 {
                        valid_tokens += 1.0;
                        for d in 0..hidden_dim {
                            embedding[d] += array[[batch_idx, s, d]];
                        }
                    }
                }
                
                // Divide by number of valid tokens
                if valid_tokens > 0.0 {
                    for d in 0..hidden_dim {
                        embedding[d] /= valid_tokens;
                    }
                }
                
                // L2 normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                let normalized: Vec<f32> = if norm > 0.0 {
                    embedding.iter().map(|x| x / norm).collect()
                } else {
                    embedding
                };
                
                results.push(normalized);
                attn_offset += max_len;
            }
        } else if shape.len() == 2 {
            // Shape: [batch, hidden_dim] - already pooled
            let hidden_dim = shape[1];
            for batch_idx in 0..batch_size {
                let embedding: Vec<f32> = (0..hidden_dim)
                    .map(|d| array[[batch_idx, d]])
                    .collect();
                
                // L2 normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                let normalized: Vec<f32> = if norm > 0.0 {
                    embedding.iter().map(|x| x / norm).collect()
                } else {
                    embedding
                };
                
                results.push(normalized);
            }
        } else {
            anyhow::bail!("Unexpected output shape: {:?}", shape);
        }
        
        Ok(results)
    }
    
    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Fast embedding model using Model2Vec static embeddings
/// 
/// This is ~825x faster than ONNX (0.2ms vs 165ms per chunk) at the cost of
/// some accuracy. Ideal for indexing large files quickly.
pub struct FastEmbeddingModel {
    model: model2vec_rs::model::StaticModel,
    dimension: usize,
}

impl FastEmbeddingModel {
    /// Load the default multilingual Model2Vec model
    /// 
    /// Uses potion-multilingual-128M which supports 101 languages.
    /// Downloads on first use (~50MB).
    pub fn load_default() -> Result<Self> {
        Self::load("minishlab/potion-multilingual-128M")
    }
    
    /// Load a specific Model2Vec model from HuggingFace
    /// 
    /// Common models:
    /// - "minishlab/potion-base-8M" (fastest, English)
    /// - "minishlab/potion-base-32M" (balanced, English)
    /// - "minishlab/potion-multilingual-128M" (101 languages)
    pub fn load(model_id: &str) -> Result<Self> {
        let model = model2vec_rs::model::StaticModel::from_pretrained(model_id, None, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to load Model2Vec model: {}", e))?;
        
        Ok(Self {
            model,
            dimension: MODEL2VEC_DIM,
        })
    }
    
    /// Embed a single text, returns normalized embedding
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let embeddings = self.model.encode(&texts);
        
        if embeddings.is_empty() {
            anyhow::bail!("Model2Vec returned no embeddings");
        }
        
        // Already normalized by the model
        Ok(embeddings[0].clone())
    }
    
    /// Embed multiple texts in batch (very efficient for Model2Vec)
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        let embeddings = self.model.encode(&texts.to_vec());
        Ok(embeddings)
    }
    
    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Simple tokenizer for IDF pre-filtering
/// 
/// Extracts word tokens and hashes them for fast lookup.
/// This is used for BM25-style term matching before embedding similarity.
pub fn tokenize_for_idf(text: &str) -> Vec<(u32, u16)> {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    
    let mut token_counts: HashMap<u32, u16> = HashMap::new();
    
    // Simple word tokenization: split on whitespace and punctuation
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '\'') {
        let word = word.trim().to_lowercase();
        if word.len() < 2 || word.len() > 50 {
            continue; // Skip very short or very long tokens
        }
        
        // Hash the token
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.hash(&mut hasher);
        let hash = hasher.finish() as u32;
        
        // Increment count (saturating at u16::MAX)
        let count = token_counts.entry(hash).or_insert(0);
        *count = count.saturating_add(1);
    }
    
    // Convert to sorted Vec for deterministic ordering and efficient lookup
    let mut tokens: Vec<(u32, u16)> = token_counts.into_iter().collect();
    tokens.sort_by_key(|(hash, _)| *hash);
    
    // Keep only top tokens by frequency to save memory
    if tokens.len() > 100 {
        tokens.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        tokens.truncate(100);
        tokens.sort_by_key(|(hash, _)| *hash);
    }
    
    tokens
}

/// Quantize f32 embedding to i8 for memory efficiency
fn quantize_embedding(embedding: &[f32]) -> Vec<i8> {
    // Find max absolute value for scaling
    let max_abs = embedding.iter()
        .map(|x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));
    
    if max_abs == 0.0 {
        return vec![0i8; embedding.len()];
    }
    
    // Scale to [-127, 127]
    let scale = 127.0 / max_abs;
    embedding.iter()
        .map(|x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Calculate cosine similarity between f32 query and quantized (i8) stored embedding
fn cosine_similarity_quantized(query: &[f32], stored: &[i8]) -> f32 {
    if query.len() != stored.len() {
        return 0.0;
    }
    
    let mut dot_product = 0.0f32;
    let mut query_norm = 0.0f32;
    let mut stored_norm = 0.0f32;
    
    for (q, s) in query.iter().zip(stored.iter()) {
        let s_f32 = *s as f32;
        dot_product += q * s_f32;
        query_norm += q * q;
        stored_norm += s_f32 * s_f32;
    }
    
    let norm_product = (query_norm * stored_norm).sqrt();
    if norm_product == 0.0 {
        return 0.0;
    }
    
    dot_product / norm_product
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_size_scaling() {
        // Test values that are clearly within each tier
        assert_eq!(chunk_size_for_file(5 * MB), 4 * 1024);        // 5MB -> 4KB (<=10MB)
        assert_eq!(chunk_size_for_file(50 * MB), 8 * 1024);       // 50MB -> 8KB (<=100MB)
        assert_eq!(chunk_size_for_file(300 * MB), 16 * 1024);     // 300MB -> 16KB (<=500MB)
        assert_eq!(chunk_size_for_file(800 * MB), 32 * 1024);     // 800MB -> 32KB (<=1GB)
        assert_eq!(chunk_size_for_file(3 * GB), 64 * 1024);       // 3GB -> 64KB (<=5GB)
        assert_eq!(chunk_size_for_file(15 * GB), 128 * 1024);     // 15GB -> 128KB (<=20GB)
        assert_eq!(chunk_size_for_file(40 * GB), 256 * 1024);     // 40GB -> 256KB (<=50GB)
        assert_eq!(chunk_size_for_file(80 * GB), 512 * 1024);     // 80GB -> 512KB (<=100GB)
        assert_eq!(chunk_size_for_file(150 * GB), 1024 * 1024);   // 150GB -> 1MB (<=200GB)
        assert_eq!(chunk_size_for_file(500 * GB), 4096 * 1024);   // 500GB -> 4MB (>200GB)
    }
    
    #[test]
    fn test_quantize_embedding() {
        let embedding = vec![0.5, -0.5, 1.0, -1.0, 0.0];
        let quantized = quantize_embedding(&embedding);
        
        assert_eq!(quantized.len(), 5);
        assert_eq!(quantized[2], 127);   // max positive
        assert_eq!(quantized[3], -127);  // max negative
        assert_eq!(quantized[4], 0);     // zero
    }
    
    #[test]
    fn test_cosine_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let stored = vec![127i8, 0, 0];
        
        let similarity = cosine_similarity_quantized(&query, &stored);
        assert!((similarity - 1.0).abs() < 0.01); // Should be ~1.0
        
        let orthogonal = vec![0i8, 127, 0];
        let similarity2 = cosine_similarity_quantized(&query, &orthogonal);
        assert!(similarity2.abs() < 0.01); // Should be ~0.0
    }
    
    #[test]
    fn test_tokenize_for_idf() {
        let tokens = tokenize_for_idf("Hello world! Hello again.");
        
        // Should have tokens for "hello", "world", "again"
        assert!(!tokens.is_empty());
        
        // "hello" appears twice, should have count 2
        let hello_tokens: Vec<_> = tokenize_for_idf("hello");
        assert_eq!(hello_tokens.len(), 1);
        
        // Check that the same word produces the same hash
        let tokens1 = tokenize_for_idf("test");
        let tokens2 = tokenize_for_idf("test");
        assert_eq!(tokens1[0].0, tokens2[0].0);
    }
    
    #[test]
    fn test_idf_scoring() {
        let config = crate::config::SemanticSearchConfig::default();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_idf.txt");
        std::fs::write(&temp_file, "test content").unwrap();
        
        let mut index = SemanticIndex::new(&temp_file, config).unwrap();
        
        // Add some chunks with tokens
        let embedding = vec![0.0f32; 256];
        let tokens1 = tokenize_for_idf("the quick brown fox");
        let tokens2 = tokenize_for_idf("the lazy brown dog");
        let tokens3 = tokenize_for_idf("hello world");
        
        index.add_chunk_with_tokens(0, 100, embedding.clone(), tokens1);
        index.add_chunk_with_tokens(100, 100, embedding.clone(), tokens2);
        index.add_chunk_with_tokens(200, 100, embedding.clone(), tokens3);
        
        index.finalize_idf();
        
        // "the" and "brown" appear in 2 chunks, should have lower IDF
        // "quick" appears in 1 chunk, should have higher IDF
        assert!(index.doc_count == 3);
        assert!(!index.idf_scores.is_empty());
        
        std::fs::remove_file(temp_file).ok();
    }
}
