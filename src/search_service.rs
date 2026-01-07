//! Unified search service supporting multiple strategies
//!
//! Search strategies:
//! - Memmem: Fast streaming search using memchr (~5 GB/s)
//! - Regex: Pattern matching with regex crate (~50-500 MB/s)
//! - Semantic: Meaning-based search with embeddings (optional feature)

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use memchr::memmem;
use regex::bytes::Regex;

/// Search direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDirection {
    Forward,
    Backward,
}

/// Search strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SearchStrategy {
    /// Fast byte-level search using memchr (default, ~5 GB/s)
    #[default]
    Memmem,
    /// Regex pattern matching (~50-500 MB/s)
    Regex,
    /// Semantic/embedding-based search (requires semantic-search feature)
    #[cfg(feature = "semantic-search")]
    Semantic,
    /// Hybrid: semantic pre-filter + exact match for instant results
    /// Background scan continues for complete results
    #[cfg(feature = "semantic-search")]
    Hybrid,
}

/// Search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// The search query
    pub query: String,
    /// Case-sensitive search
    pub case_sensitive: bool,
    /// Direction of search
    pub direction: SearchDirection,
    /// Wrap around at file boundaries
    pub wrap_around: bool,
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Use regex pattern (for Memmem strategy, treat query as literal)
    pub use_regex: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: true,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        }
    }
}

/// A search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Byte offset in file where match starts
    pub offset: u64,
    /// Length of match in bytes
    pub length: usize,
    /// Optional snippet of matched text
    pub snippet: Option<String>,
}

/// Disk-based search result cache using disklru
/// 
/// Stores search results on disk with LRU eviction.
/// Limits: 100 queries max, 500MB total size.
mod search_cache {
    use super::*;
    use serde::{Serialize, Deserialize};
    use std::path::PathBuf;
    
    /// Maximum number of cached search queries
    const MAX_QUERIES: usize = 100;
    
    /// Maximum total cache size in bytes (500 MB)
    const MAX_CACHE_SIZE_BYTES: u64 = 500 * 1024 * 1024;
    
    /// Cache key for search results
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CacheKey {
        pub query: String,
        pub case_sensitive: bool,
        pub strategy_id: u8, // 0=Memmem, 1=Regex, 2=Semantic, 3=Hybrid
    }
    
    impl CacheKey {
        pub fn new(query: &str, case_sensitive: bool, strategy: SearchStrategy) -> Self {
            let strategy_id = match strategy {
                SearchStrategy::Memmem => 0,
                SearchStrategy::Regex => 1,
                #[cfg(feature = "semantic-search")]
                SearchStrategy::Semantic => 2,
                #[cfg(feature = "semantic-search")]
                SearchStrategy::Hybrid => 3,
            };
            Self {
                query: query.to_string(),
                case_sensitive,
                strategy_id,
            }
        }
        
        /// Convert to cache key string
        pub fn to_key_string(&self) -> String {
            format!("{}:{}:{}", self.strategy_id, self.case_sensitive as u8, self.query)
        }
    }
    
    /// Cached search results (serializable)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CachedResults {
        /// Match offsets (we only store offsets to save space)
        pub offsets: Vec<u64>,
        /// Match length (same for all matches in a query)
        pub match_length: usize,
        /// Whether search is complete
        pub complete: bool,
    }
    
    impl CachedResults {
        pub fn new(matches: &[SearchResult], complete: bool) -> Self {
            let match_length = matches.first().map(|m| m.length).unwrap_or(0);
            Self {
                offsets: matches.iter().map(|m| m.offset).collect(),
                match_length,
                complete,
            }
        }
        
        pub fn to_matches(&self) -> Vec<SearchResult> {
            self.offsets.iter().map(|&offset| SearchResult {
                offset,
                length: self.match_length,
                snippet: None,
            }).collect()
        }
    }
    
    /// Disk-based LRU cache for search results (thread-safe with interior mutability)
    pub struct DiskCache {
        store: std::sync::Mutex<disklru::Store<String, Vec<u8>>>,
        cache_dir: PathBuf,
    }
    
    impl DiskCache {
        /// Create or open cache in the given directory
        pub fn new(file_path: &Path) -> Result<Self> {
            // Cache dir is ~/.cache/bigedit/search/<file_hash>/
            let cache_base = dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("bigedit")
                .join("search");
            
            // Use file path hash to separate caches per file
            let file_hash = Self::hash_path(file_path);
            let cache_dir = cache_base.join(file_hash);
            
            // Create directory if needed
            std::fs::create_dir_all(&cache_dir)?;
            
            // Open LRU store with max 100 entries
            let store = disklru::Store::open_with_path(&cache_dir, MAX_QUERIES)
                .map_err(|e| anyhow::anyhow!("Failed to open cache: {:?}", e))?;
            
            let cache = Self { 
                store: std::sync::Mutex::new(store),
                cache_dir,
            };
            
            // Check and enforce size limit on startup
            cache.enforce_size_limit()?;
            
            Ok(cache)
        }
        
        fn hash_path(path: &Path) -> String {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        }
        
        /// Calculate total size of cache directory (recursive)
        fn get_cache_size(&self) -> u64 {
            fn dir_size(path: &Path) -> u64 {
                let mut total = 0;
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if path.is_file() {
                            if let Ok(meta) = path.metadata() {
                                total += meta.len();
                            }
                        } else if path.is_dir() {
                            total += dir_size(&path);
                        }
                    }
                }
                total
            }
            dir_size(&self.cache_dir)
        }
        
        /// Evict oldest entries until cache is under size limit
        fn enforce_size_limit(&self) -> Result<()> {
            let mut current_size = self.get_cache_size();
            
            if current_size <= MAX_CACHE_SIZE_BYTES {
                return Ok(());
            }
            
            let mut store = self.store.lock()
                .map_err(|_| anyhow::anyhow!("Cache lock poisoned"))?;
            
            // Evict LRU entries until under limit
            while current_size > MAX_CACHE_SIZE_BYTES {
                match store.lru() {
                    Ok(Some(key)) => {
                        let _ = store.remove(&key);
                    }
                    _ => break,
                }
                // Recheck size (could optimize by estimating, but this is safer)
                drop(store);
                current_size = self.get_cache_size();
                store = self.store.lock()
                    .map_err(|_| anyhow::anyhow!("Cache lock poisoned"))?;
            }
            
            let _ = store.flush();
            Ok(())
        }
        
        /// Get cached results for a query
        pub fn get(&self, key: &CacheKey) -> Option<CachedResults> {
            let key_str = key.to_key_string();
            let mut store = self.store.lock().ok()?;
            match store.get(&key_str) {
                Ok(Some(data)) => bincode::deserialize(&data).ok(),
                _ => None,
            }
        }
        
        /// Store results in cache
        pub fn put(&self, key: &CacheKey, results: &CachedResults) -> Result<()> {
            let key_str = key.to_key_string();
            let data = bincode::serialize(results)
                .map_err(|e| anyhow::anyhow!("Serialize error: {}", e))?;
            
            {
                let mut store = self.store.lock()
                    .map_err(|_| anyhow::anyhow!("Cache lock poisoned"))?;
                
                store.insert(&key_str, &data)
                    .map_err(|e| anyhow::anyhow!("Cache write error: {:?}", e))?;
                
                // Commit to disk
                let _ = store.flush();
            }
            
            // Enforce size limit after insert
            self.enforce_size_limit()?;
            
            Ok(())
        }
        
        /// Clear all cached results for this file
        pub fn clear(&self) -> Result<()> {
            let mut store = self.store.lock()
                .map_err(|_| anyhow::anyhow!("Cache lock poisoned"))?;
            
            // disklru doesn't have a clear method, so we remove entries one by one
            while let Ok(Some(key)) = store.lru() {
                let _ = store.remove(&key);
            }
            let _ = store.flush();
            Ok(())
        }
    }
}

/// Progress information for hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchProgress {
    /// Number of semantic candidate chunks searched
    pub semantic_chunks_searched: usize,
    /// Total semantic candidate chunks
    pub semantic_chunks_total: usize,
    /// Bytes scanned in full search
    pub bytes_scanned: u64,
    /// Total file bytes
    pub bytes_total: u64,
    /// Whether semantic phase is complete
    pub semantic_complete: bool,
    /// Whether full scan is complete
    pub full_scan_complete: bool,
}

impl HybridSearchProgress {
    /// Create a new progress tracker
    pub fn new(semantic_total: usize, bytes_total: u64) -> Self {
        Self {
            semantic_chunks_searched: 0,
            semantic_chunks_total: semantic_total,
            bytes_scanned: 0,
            bytes_total,
            semantic_complete: false,
            full_scan_complete: false,
        }
    }
    
    /// Check if search is complete
    pub fn is_complete(&self) -> bool {
        self.full_scan_complete
    }
    
    /// Get a human-readable status message
    pub fn status_message(&self) -> String {
        if self.full_scan_complete {
            "Search complete".to_string()
        } else if self.semantic_complete {
            let pct = (self.bytes_scanned as f64 / self.bytes_total as f64 * 100.0) as u32;
            format!("Scanning full file... {}%", pct)
        } else {
            format!("Semantic search: {}/{} chunks", 
                    self.semantic_chunks_searched, self.semantic_chunks_total)
        }
    }
}

/// Result from hybrid search (instant + background)
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Matches found so far (instant results from semantic chunks)
    pub matches: Vec<SearchResult>,
    /// Number of matches from semantic pre-filtering (instant)
    pub instant_match_count: usize,
    /// Progress information
    pub progress: HybridSearchProgress,
}

/// Unified search service
pub struct SearchService {
    /// Path to the file being searched
    path: PathBuf,
    /// File size in bytes
    file_size: u64,
    /// Compiled regex (cached)
    compiled_regex: Option<Regex>,
    /// Last regex pattern used
    last_regex_pattern: Option<String>,
    /// Disk-based LRU cache for search results (100 queries, 500MB max)
    disk_cache: Option<search_cache::DiskCache>,
    /// Semantic index (optional, built on demand)
    #[cfg(feature = "semantic-search")]
    semantic_index: Option<crate::semantic::SemanticIndex>,
    /// Stale index kept for approximate search during rebuild
    /// Contains the old index after file was modified (offsets are approximate)
    #[cfg(feature = "semantic-search")]
    stale_index: Option<crate::semantic::SemanticIndex>,
    /// Fast embedding model for indexing (Model2Vec, optional)
    #[cfg(feature = "semantic-search")]
    fast_embedding_model: Option<crate::semantic::FastEmbeddingModel>,
    /// ONNX embedding model for re-ranking (optional, loaded on demand)
    #[cfg(feature = "semantic-search")]
    embedding_model: Option<crate::semantic::EmbeddingModel>,
    /// Hybrid search state: tracks progress for background continuation
    #[cfg(feature = "semantic-search")]
    hybrid_state: Option<HybridSearchState>,
}

/// State for continuing hybrid search in background
#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone)]
struct HybridSearchState {
    /// The query being searched
    query: String,
    /// Case sensitivity
    case_sensitive: bool,
    /// Semantic chunks already searched (sorted by offset)
    searched_ranges: Vec<(u64, u64)>,  // (start, end) byte ranges
    /// Current position in full file scan
    full_scan_offset: u64,
    /// All matches found so far
    all_matches: Vec<SearchResult>,
    /// Matches from instant semantic phase
    instant_matches: usize,
    /// Current semantic expansion level (for exponential search)
    /// Levels: 100, 300, 1000, 3000, 10000, 30000, 100000, ...
    semantic_level: usize,
    /// All semantic results (sorted by relevance), for exponential expansion
    all_semantic_results: Vec<crate::semantic::SemanticSearchResult>,
    /// Query embedding (cached for expansion)
    query_embedding: Vec<f32>,
}

impl SearchService {
    /// Create a new search service for the given file
    /// 
    /// The second parameter is kept for API compatibility but is no longer used
    /// since FTS5 has been removed.
    pub fn new<P: AsRef<Path>>(path: P, _config: crate::config::FtsSearchConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file_size = if path.exists() {
            std::fs::metadata(&path)
                .with_context(|| format!("Failed to get file metadata: {}", path.display()))?
                .len()
        } else {
            0
        };

        // Initialize disk cache (ignore errors - cache is optional)
        let disk_cache = search_cache::DiskCache::new(&path).ok();

        Ok(Self {
            path,
            file_size,
            compiled_regex: None,
            last_regex_pattern: None,
            disk_cache,
            #[cfg(feature = "semantic-search")]
            semantic_index: None,
            #[cfg(feature = "semantic-search")]
            stale_index: None,
            #[cfg(feature = "semantic-search")]
            fast_embedding_model: None,
            #[cfg(feature = "semantic-search")]
            embedding_model: None,
            #[cfg(feature = "semantic-search")]
            hybrid_state: None,
        })
    }
    
    /// Get cached results for a query from disk
    pub fn get_cached_results(&self, options: &SearchOptions) -> Option<Vec<SearchResult>> {
        let cache = self.disk_cache.as_ref()?;
        let key = search_cache::CacheKey::new(&options.query, options.case_sensitive, options.strategy);
        cache.get(&key).map(|r| r.to_matches())
    }
    
    /// Cache search results to disk
    pub fn cache_results(&self, options: &SearchOptions, matches: &[SearchResult], complete: bool) {
        if let Some(cache) = &self.disk_cache {
            let key = search_cache::CacheKey::new(&options.query, options.case_sensitive, options.strategy);
            let results = search_cache::CachedResults::new(matches, complete);
            let _ = cache.put(&key, &results); // Ignore cache write errors
        }
    }
    
    /// Clear the search cache
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.disk_cache {
            let _ = cache.clear();
        }
    }
    
    /// Invalidate the semantic index (call this when file is edited)
    /// 
    /// The index is moved to stale_index for approximate search hints
    /// while rebuilding. The stale index has wrong offsets but can still
    /// help find approximate locations.
    #[cfg(feature = "semantic-search")]
    pub fn invalidate_semantic_index(&mut self) {
        // Keep the old index as stale for approximate hints
        if self.semantic_index.is_some() {
            self.stale_index = self.semantic_index.take();
        }
        self.hybrid_state = None;
        // Also clear search result cache since offsets are now wrong
        self.clear_cache();
    }
    
    /// Check if semantic index exists (fresh or stale)
    #[cfg(feature = "semantic-search")]
    pub fn has_semantic_index(&self) -> bool {
        self.semantic_index.is_some()
    }
    
    /// Check if there's a stale index for approximate search
    #[cfg(feature = "semantic-search")]
    pub fn has_stale_index(&self) -> bool {
        self.stale_index.is_some()
    }
    
    /// Set a pre-built semantic index (from background thread)
    #[cfg(feature = "semantic-search")]
    pub fn set_semantic_index(&mut self, index: crate::semantic::SemanticIndex) {
        self.semantic_index = Some(index);
        // Clear stale index now that we have a fresh one
        self.stale_index = None;
        // Also ensure we have the embedding model
        if self.fast_embedding_model.is_none() {
            if let Ok(model) = crate::semantic::FastEmbeddingModel::load_default() {
                self.fast_embedding_model = Some(model);
            }
        }
    }
    
    /// Try to pre-load the semantic index from disk cache
    /// 
    /// This is non-blocking - only loads if cache exists and is valid.
    /// If no cache, returns immediately (index will be built on first search).
    #[cfg(feature = "semantic-search")]
    pub fn try_load_cached_index(&mut self) {
        use crate::semantic::{SemanticIndex, FastEmbeddingModel};
        use crate::config::SemanticSearchConfig;
        
        // Only try if we don't already have an index
        if self.semantic_index.is_some() {
            return;
        }
        
        let config = SemanticSearchConfig::default();
        
        // Try to load from disk cache (fast if exists, returns None if not)
        if let Some(cached_index) = SemanticIndex::load_cached(&self.path, config) {
            // Also pre-load the embedding model for instant search
            if self.fast_embedding_model.is_none() {
                if let Ok(model) = FastEmbeddingModel::load_default() {
                    self.fast_embedding_model = Some(model);
                }
            }
            self.semantic_index = Some(cached_index);
        }
    }

    /// Get file size
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Search for the next match starting from offset
    pub fn search(&mut self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        if options.query.is_empty() {
            return Ok(None);
        }

        match options.direction {
            SearchDirection::Forward => self.search_forward(options, start_offset),
            SearchDirection::Backward => self.search_backward(options, start_offset),
        }
    }

    /// Forward search
    fn search_forward(&mut self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        #[cfg(feature = "semantic-search")]
        if options.strategy == SearchStrategy::Semantic {
            return self.search_semantic(options, start_offset);
        }
        
        #[cfg(feature = "semantic-search")]
        if options.strategy == SearchStrategy::Hybrid {
            // Hybrid: try semantic chunks first for instant result
            let result = self.search_hybrid_instant(options)?;
            if let Some(first_match) = result.matches.into_iter().next() {
                return Ok(Some(first_match));
            }
            // Semantic didn't find it - fall through to memmem for guaranteed result
            return self.search_forward_memmem(options, start_offset);
        }
        
        if options.use_regex || options.strategy == SearchStrategy::Regex {
            self.search_forward_regex(options, start_offset)
        } else {
            self.search_forward_memmem(options, start_offset)
        }
    }

    /// Backward search
    fn search_backward(&mut self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        #[cfg(feature = "semantic-search")]
        if options.strategy == SearchStrategy::Semantic {
            // Semantic search doesn't have a direction, use same logic
            return self.search_semantic(options, start_offset);
        }
        
        #[cfg(feature = "semantic-search")]
        if options.strategy == SearchStrategy::Hybrid {
            // Hybrid: try semantic chunks first for instant result
            let result = self.search_hybrid_instant(options)?;
            if let Some(first_match) = result.matches.into_iter().next() {
                return Ok(Some(first_match));
            }
            // Semantic didn't find it - fall through to memmem for guaranteed result
            return self.search_backward_memmem(options, start_offset);
        }
        
        if options.use_regex || options.strategy == SearchStrategy::Regex {
            self.search_backward_regex(options, start_offset)
        } else {
            self.search_backward_memmem(options, start_offset)
        }
    }
    
    /// Semantic search using embeddings (hybrid: Model2Vec + IDF pre-filtering)
    /// 
    /// Uses fast Model2Vec embeddings for indexing (~10 MB/s) and IDF-based
    /// pre-filtering to narrow down candidates before embedding similarity.
    /// 
    /// Builds an embedding index on first search, then searches are instant (<1ms).
    #[cfg(feature = "semantic-search")]
    fn search_semantic(&mut self, options: &SearchOptions, _start_offset: u64) -> Result<Option<SearchResult>> {
        use crate::semantic::{SemanticIndex, FastEmbeddingModel, tokenize_for_idf};
        use crate::config::SemanticSearchConfig;
        
        // Ensure we have a fast embedding model (Model2Vec)
        if self.fast_embedding_model.is_none() {
            self.fast_embedding_model = Some(FastEmbeddingModel::load_default()
                .context("Failed to load Model2Vec model")?);
        }
        
        // Ensure we have an index (try cache first)
        if self.semantic_index.is_none() && self.stale_index.is_none() {
            let config = SemanticSearchConfig::default();
            
            // Try to load from disk cache first
            if let Some(cached_index) = SemanticIndex::load_cached(&self.path, config.clone()) {
                self.semantic_index = Some(cached_index);
            }
            // If no cached index, return None so caller falls back to memmem
            // This allows searching while background indexing is in progress
        }
        
        // If we have neither fresh nor stale index, return None
        if self.semantic_index.is_none() && self.stale_index.is_none() {
            return Ok(None);
        }
        
        // Search using embeddings with IDF pre-filtering
        let model = self.fast_embedding_model.as_ref().unwrap();
        
        // Use fresh index if available, otherwise fall back to stale index
        let index = self.semantic_index.as_ref()
            .or(self.stale_index.as_ref())
            .ok_or_else(|| anyhow::anyhow!("No semantic index available"))?;
        
        // Embed the query
        let query_embedding = model.embed(&options.query)?;
        
        // Tokenize query for IDF pre-filtering
        let query_tokens = tokenize_for_idf(&options.query);
        
        // Search the index with IDF filtering
        let results = index.search_with_idf(&query_embedding, Some(&query_tokens), 1);
        
        if let Some(result) = results.first() {
            Ok(Some(SearchResult {
                offset: result.offset,
                length: result.length,
                snippet: None,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Build semantic index from file content
    #[cfg(feature = "semantic-search")]
    fn build_semantic_index(&mut self, index: &mut crate::semantic::SemanticIndex) -> Result<()> {
        use std::io::BufRead;
        
        let model = self.embedding_model.as_mut()
            .context("Embedding model not loaded")?;
        
        let file = File::open(&self.path)?;
        let reader = std::io::BufReader::new(file);
        
        let chunk_size = index.chunk_size();
        let mut current_offset = 0u64;
        let mut chunk = String::new();
        let mut line_reader = reader;
        
        loop {
            chunk.clear();
            let mut bytes_read = 0;
            
            // Read lines until we have a chunk
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
            
            // Embed the chunk
            let embedding = model.embed(&chunk)?;
            index.add_chunk(current_offset, chunk.len(), embedding);
            
            current_offset += chunk.len() as u64;
        }
        
        index.mark_complete();
        Ok(())
    }
    
    /// Build semantic index using fast Model2Vec embeddings
    /// 
    /// This is ~825x faster than ONNX (~10 MB/s vs 0.02 MB/s on CPU).
    /// Also extracts token frequencies for IDF-based pre-filtering.
    #[cfg(feature = "semantic-search")]
    fn build_semantic_index_fast(&mut self, index: &mut crate::semantic::SemanticIndex) -> Result<()> {
        self.build_semantic_index_fast_with_progress(index, |_, _, _, _| {})
    }
    
    /// Build semantic index with progress callback
    /// 
    /// Callback receives: (chunks_done, total_chunks, bytes_done, total_bytes)
    /// 
    /// Streaming approach: processes file in 500MB windows to avoid loading
    /// entire file into RAM. Uses parallel processing within each window.
    /// 
    /// Parallelism: Uses physical cores only for this RAM-bound workload.
    /// Model2Vec is essentially a lookup table (token → embedding), so memory
    /// bandwidth is the bottleneck, not CPU compute.
    #[cfg(feature = "semantic-search")]
    pub fn build_semantic_index_fast_with_progress<F>(
        &mut self, 
        index: &mut crate::semantic::SemanticIndex,
        mut progress_callback: F
    ) -> Result<()> 
    where F: FnMut(usize, usize, u64, u64)
    {
        use std::io::BufRead;
        use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
        use std::sync::Arc;
        use rayon::prelude::*;
        use crate::semantic::tokenize_for_idf;
        
        // Use physical cores for RAM-bound workload (Model2Vec is a lookup table)
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Assume ~half are physical cores (hyperthreading)
        let physical_cores = std::cmp::max(1, num_cpus / 2);
        eprintln!("   [Parallel indexing: {} physical cores (of {} logical)]", physical_cores, num_cpus);
        
        // Configure rayon to use physical cores only
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(physical_cores)
            .build()
            .context("Failed to create thread pool")?;
        
        let model = self.fast_embedding_model.as_ref()
            .context("Fast embedding model not loaded")?;
        
        let file_size = self.file_size;
        let chunk_size = index.chunk_size();
        
        // Streaming: process file in 500MB windows to limit RAM usage
        const WINDOW_SIZE: u64 = 500 * 1024 * 1024; // 500MB
        let batch_size = 200; // Chunks per parallel batch
        
        // Estimate total chunks for progress
        let total_chunks = (file_size as usize + chunk_size - 1) / chunk_size;
        let chunks_done = Arc::new(AtomicUsize::new(0));
        let bytes_done = Arc::new(AtomicU64::new(0));
        
        let mut window_start = 0u64;
        
        while window_start < file_size {
            let window_end = std::cmp::min(window_start + WINDOW_SIZE, file_size);
            
            // Read this window's chunks
            let mut file = File::open(&self.path)?;
            file.seek(SeekFrom::Start(window_start))?;
            
            let reader = std::io::BufReader::new(file);
            let mut line_reader = reader;
            let mut current_offset = window_start;
            let mut window_chunks: Vec<(u64, String)> = Vec::new();
            
            loop {
                let mut chunk = String::new();
                let mut bytes_read = 0;
                
                // Read lines until we have a chunk
                loop {
                    let mut line = String::new();
                    match line_reader.read_line(&mut line) {
                        Ok(0) => break, // EOF
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
                
                // Stop at window boundary (will continue in next window)
                if current_offset >= window_end {
                    break;
                }
            }
            
            // Process this window's chunks in parallel
            for batch in window_chunks.chunks(batch_size) {
                let batch_results: Vec<_> = pool.install(|| {
                    batch.par_iter()
                        .map(|(offset, text)| {
                            let embedding = model.embed(text).unwrap_or_default();
                            let token_freqs = tokenize_for_idf(text);
                            (*offset, text.len(), embedding, token_freqs)
                        })
                        .collect()
                });
                
                // Add to index (sequential to maintain order)
                for (offset, len, embedding, token_freqs) in batch_results {
                    index.add_chunk_with_tokens(offset, len, embedding, token_freqs);
                    bytes_done.fetch_add(len as u64, Ordering::Relaxed);
                }
                
                chunks_done.fetch_add(batch.len(), Ordering::Relaxed);
                progress_callback(
                    chunks_done.load(Ordering::Relaxed), 
                    total_chunks, 
                    bytes_done.load(Ordering::Relaxed), 
                    file_size
                );
            }
            
            // Move to next window
            window_start = current_offset;
        }
        
        // Finalize IDF scores after all chunks are added
        index.finalize_idf();
        index.mark_complete();
        Ok(())
    }
    
    /// Ensure semantic index is built, with progress callback
    #[cfg(feature = "semantic-search")]
    pub fn ensure_semantic_index_with_progress<F>(&mut self, mut progress_callback: F) -> Result<()>
    where F: FnMut(usize, usize, u64, u64)
    {
        use crate::semantic::FastEmbeddingModel;
        use crate::config::SemanticSearchConfig;
        
        if self.fast_embedding_model.is_none() {
            self.fast_embedding_model = Some(FastEmbeddingModel::load_default()
                .context("Failed to load Model2Vec model")?);
        }
        
        if self.semantic_index.is_none() {
            let config = SemanticSearchConfig::default();
            
            // Try to load from disk cache first
            if let Some(cached_index) = crate::semantic::SemanticIndex::load_cached(&self.path, config.clone()) {
                self.semantic_index = Some(cached_index);
            } else {
                // Build new index
                let mut index = crate::semantic::SemanticIndex::new(&self.path, config)?;
                self.build_semantic_index_fast_with_progress(&mut index, &mut progress_callback)?;
                
                // Save to disk cache for next time
                if let Err(e) = index.save_to_cache() {
                    eprintln!("[cache] Warning: Failed to save index: {}", e);
                }
                
                self.semantic_index = Some(index);
            }
        }
        
        Ok(())
    }
    
    /// Hybrid search: instant results from semantic chunks + exact match
    /// 
    /// This provides INSTANT results by:
    /// 1. Using semantic index to find the most relevant chunks
    /// 2. Performing exact string match within those chunks only
    /// 
    /// Returns immediately with matches found in semantic candidates.
    /// Call `search_hybrid_continue` to continue scanning the rest of the file.
    #[cfg(feature = "semantic-search")]
    /// Hybrid search: race memmem vs semantic for fastest first result
    /// 
    /// This provides the BEST of both worlds:
    /// 1. Memmem scan starts immediately from byte 0 (finds common words fast)
    /// 2. Semantic search finds top-N most relevant chunks (finds rare words fast)
    /// 3. Exponential expansion: 100 → 300 → 1000 → 3000 → 10000 → ALL chunks
    /// 4. Memmem scan runs to fill in any gaps
    /// 
    /// Returns immediately with first batch of results (~10ms for 100 chunks).
    /// Call `search_hybrid_expand` to progressively search more chunks.
    /// Call `search_hybrid_continue` to do full memmem scan for complete results.
    #[cfg(feature = "semantic-search")]
    pub fn search_hybrid_instant(&mut self, options: &SearchOptions) -> Result<HybridSearchResult> {
        use crate::semantic::{FastEmbeddingModel, SemanticIndex, tokenize_for_idf};
        use crate::config::SemanticSearchConfig;
        
        // Ensure semantic index is built
        if self.fast_embedding_model.is_none() {
            self.fast_embedding_model = Some(FastEmbeddingModel::load_default()
                .context("Failed to load Model2Vec model")?);
        }
        
        if self.semantic_index.is_none() && self.stale_index.is_none() {
            let config = SemanticSearchConfig::default();
            
            // Try to load from disk cache first
            if let Some(cached_index) = SemanticIndex::load_cached(&self.path, config.clone()) {
                self.semantic_index = Some(cached_index);
            }
            // If no cached index and no stale index, skip semantic phase
            // The caller will fall back to memmem search
            // This allows searching while background indexing is in progress
        }
        
        // If we have neither fresh nor stale index, return empty result
        // so the caller falls back to memmem
        if self.semantic_index.is_none() && self.stale_index.is_none() {
            return Ok(HybridSearchResult {
                matches: Vec::new(),
                instant_match_count: 0,
                progress: HybridSearchProgress::new(0, self.file_size),
            });
        }
        
        let model = self.fast_embedding_model.as_ref().unwrap();
        
        // Use fresh index if available, otherwise fall back to stale index
        // Stale index has approximate offsets but still helps find relevant chunks
        let index = self.semantic_index.as_ref()
            .or(self.stale_index.as_ref())
            .ok_or_else(|| anyhow::anyhow!("No semantic index available"))?;
        
        // Get ALL semantic results sorted by relevance (this is fast: ~20ms for 100k chunks)
        let total_chunks = index.chunk_count();
        let query_embedding = model.embed(&options.query)?;
        let query_tokens = tokenize_for_idf(&options.query);
        let all_semantic_results = index.search_with_idf(&query_embedding, Some(&query_tokens), total_chunks);
        
        // Exponential expansion: start with 100 chunks (~10ms)
        // This finds rare words quickly while being very fast
        let initial_level = 100.min(total_chunks);
        let instant_results: Vec<_> = all_semantic_results.iter().take(initial_level).cloned().collect();
        
        // Prepare search state
        let query_bytes = if options.case_sensitive {
            options.query.as_bytes().to_vec()
        } else {
            options.query.to_lowercase().as_bytes().to_vec()
        };
        let finder = memmem::Finder::new(&query_bytes);
        
        // Search initial semantic chunks for exact matches
        let mut semantic_matches = Vec::new();
        let mut searched_ranges = Vec::new();
        let mut file = File::open(&self.path)?;
        
        for result in &instant_results {
            let start = result.offset;
            let end = start + result.length as u64;
            searched_ranges.push((start, end));
            
            // Read chunk content
            file.seek(SeekFrom::Start(start))?;
            let mut buffer = vec![0u8; result.length];
            file.read_exact(&mut buffer)?;
            
            let search_buffer = if options.case_sensitive {
                buffer.clone()
            } else {
                buffer.iter().map(|b| b.to_ascii_lowercase()).collect()
            };
            
            // Find all matches in this chunk
            let mut offset = 0;
            while let Some(pos) = finder.find(&search_buffer[offset..]) {
                let match_offset = start + offset as u64 + pos as u64;
                semantic_matches.push(SearchResult {
                    offset: match_offset,
                    length: options.query.len(),
                    snippet: None,
                });
                offset += pos + 1;
            }
        }
        
        // Sort matches by offset
        semantic_matches.sort_by_key(|m| m.offset);
        
        let instant_count = semantic_matches.len();
        let searched_count = instant_results.len();
        
        // Initialize hybrid state for exponential expansion and background scan
        self.hybrid_state = Some(HybridSearchState {
            query: options.query.clone(),
            case_sensitive: options.case_sensitive,
            searched_ranges,
            full_scan_offset: 0,
            all_matches: semantic_matches.clone(),
            instant_matches: instant_count,
            semantic_level: initial_level,
            all_semantic_results,
            query_embedding: query_embedding.clone(),
        });
        
        let progress = HybridSearchProgress::new(total_chunks, self.file_size);
        
        Ok(HybridSearchResult {
            matches: semantic_matches,
            instant_match_count: instant_count,
            progress: HybridSearchProgress {
                semantic_chunks_searched: searched_count,
                semantic_chunks_total: total_chunks,
                semantic_complete: searched_count >= total_chunks,
                full_scan_complete: false,
                ..progress
            },
        })
    }
    
    /// Expand semantic search exponentially: 100 → 300 → 1000 → 3000 → 10000 → ...
    /// 
    /// Call this repeatedly to progressively search more semantic chunks.
    /// Each call roughly triples the coverage, taking ~3x longer.
    /// Returns when all semantic chunks are searched.
    #[cfg(feature = "semantic-search")]
    pub fn search_hybrid_expand(&mut self) -> Result<HybridSearchResult> {
        let state = self.hybrid_state.as_mut()
            .context("No hybrid search in progress")?;
        
        let total_chunks = state.all_semantic_results.len();
        
        // Already searched all semantic chunks?
        if state.semantic_level >= total_chunks {
            return Ok(HybridSearchResult {
                matches: state.all_matches.clone(),
                instant_match_count: state.instant_matches,
                progress: HybridSearchProgress {
                    semantic_chunks_searched: total_chunks,
                    semantic_chunks_total: total_chunks,
                    bytes_scanned: state.full_scan_offset,
                    bytes_total: self.file_size,
                    semantic_complete: true,
                    full_scan_complete: state.full_scan_offset >= self.file_size,
                },
            });
        }
        
        // Exponential expansion: 100 → 300 → 1000 → 3000 → 10000 → 30000 → 100000 → ALL
        let new_level = (state.semantic_level * 3).min(total_chunks);
        
        // Search chunks from current level to new level
        let query_bytes = if state.case_sensitive {
            state.query.as_bytes().to_vec()
        } else {
            state.query.to_lowercase().as_bytes().to_vec()
        };
        let finder = memmem::Finder::new(&query_bytes);
        
        let mut file = File::open(&self.path)?;
        
        for result in state.all_semantic_results[state.semantic_level..new_level].iter() {
            let start = result.offset;
            let end = start + result.length as u64;
            state.searched_ranges.push((start, end));
            
            // Read chunk content
            file.seek(SeekFrom::Start(start))?;
            let mut buffer = vec![0u8; result.length];
            file.read_exact(&mut buffer)?;
            
            let search_buffer = if state.case_sensitive {
                buffer.clone()
            } else {
                buffer.iter().map(|b| b.to_ascii_lowercase()).collect()
            };
            
            // Find all matches in this chunk
            let mut offset = 0;
            while let Some(pos) = finder.find(&search_buffer[offset..]) {
                let match_offset = start + offset as u64 + pos as u64;
                state.all_matches.push(SearchResult {
                    offset: match_offset,
                    length: state.query.len(),
                    snippet: None,
                });
                offset += pos + 1;
            }
        }
        
        state.semantic_level = new_level;
        
        // Sort all matches
        state.all_matches.sort_by_key(|m| m.offset);
        
        Ok(HybridSearchResult {
            matches: state.all_matches.clone(),
            instant_match_count: state.instant_matches,
            progress: HybridSearchProgress {
                semantic_chunks_searched: new_level,
                semantic_chunks_total: total_chunks,
                bytes_scanned: state.full_scan_offset,
                bytes_total: self.file_size,
                semantic_complete: new_level >= total_chunks,
                full_scan_complete: false,
            },
        })
    }
    
    /// Continue hybrid search: scan remaining file regions with memmem
    /// 
    /// Call this repeatedly to continue the background scan.
    /// Returns when `progress.is_complete()` is true.
    /// 
    /// `max_bytes`: Maximum bytes to scan in this call (for incremental progress)
    #[cfg(feature = "semantic-search")]
    pub fn search_hybrid_continue(&mut self, max_bytes: u64) -> Result<HybridSearchResult> {
        let total_chunks = self.semantic_index.as_ref().map(|i| i.chunk_count()).unwrap_or(0);
        let state = self.hybrid_state.as_mut()
            .context("No hybrid search in progress")?;
        
        if state.full_scan_offset >= self.file_size {
            // Already complete
            return Ok(HybridSearchResult {
                matches: state.all_matches.clone(),
                instant_match_count: state.instant_matches,
                progress: HybridSearchProgress {
                    semantic_chunks_searched: state.searched_ranges.len(),
                    semantic_chunks_total: total_chunks,
                    bytes_scanned: self.file_size,
                    bytes_total: self.file_size,
                    semantic_complete: true,
                    full_scan_complete: true,
                },
            });
        }
        
        let query_bytes = if state.case_sensitive {
            state.query.as_bytes().to_vec()
        } else {
            state.query.to_lowercase().as_bytes().to_vec()
        };
        let finder = memmem::Finder::new(&query_bytes);
        
        let mut file = File::open(&self.path)?;
        let mut bytes_scanned = 0u64;
        let buffer_size = 64 * 1024; // 64KB buffer
        let mut buffer = vec![0u8; buffer_size];
        let file_size = self.file_size;
        
        // Sort searched ranges for efficient skip
        state.searched_ranges.sort_by_key(|(start, _)| *start);
        
        while bytes_scanned < max_bytes && state.full_scan_offset < file_size {
            // Check if current position is in an already-searched range
            let skip_to = Self::find_next_unsearched(
                state.full_scan_offset, 
                &state.searched_ranges
            );
            
            if skip_to > state.full_scan_offset {
                state.full_scan_offset = skip_to;
                continue;
            }
            
            // Find end of unsearched region
            let region_end = Self::find_searched_region_start(
                state.full_scan_offset, 
                &state.searched_ranges,
                file_size
            );
            
            // Read and search
            file.seek(SeekFrom::Start(state.full_scan_offset))?;
            let to_read = ((region_end - state.full_scan_offset) as usize).min(buffer_size);
            let bytes_read = file.read(&mut buffer[..to_read])?;
            
            if bytes_read == 0 {
                state.full_scan_offset = file_size;
                break;
            }
            
            let search_buffer = if state.case_sensitive {
                buffer[..bytes_read].to_vec()
            } else {
                buffer[..bytes_read].iter().map(|b| b.to_ascii_lowercase()).collect()
            };
            
            // Find matches
            let mut offset = 0;
            while let Some(pos) = finder.find(&search_buffer[offset..]) {
                let match_offset = state.full_scan_offset + offset as u64 + pos as u64;
                
                // Check if this match is already found (in semantic chunks)
                if !state.all_matches.iter().any(|m| m.offset == match_offset) {
                    state.all_matches.push(SearchResult {
                        offset: match_offset,
                        length: state.query.len(),
                        snippet: None,
                    });
                }
                offset += pos + 1;
            }
            
            state.full_scan_offset += bytes_read as u64;
            bytes_scanned += bytes_read as u64;
        }
        
        // Sort all matches
        state.all_matches.sort_by_key(|m| m.offset);
        
        let is_complete = state.full_scan_offset >= self.file_size;
        
        Ok(HybridSearchResult {
            matches: state.all_matches.clone(),
            instant_match_count: state.instant_matches,
            progress: HybridSearchProgress {
                semantic_chunks_searched: state.semantic_level,
                semantic_chunks_total: total_chunks,
                bytes_scanned: state.full_scan_offset,
                bytes_total: self.file_size,
                semantic_complete: state.semantic_level >= total_chunks,
                full_scan_complete: is_complete,
            },
        })
    }
    
    /// Find the next position not covered by searched ranges
    #[cfg(feature = "semantic-search")]
    fn find_next_unsearched(pos: u64, ranges: &[(u64, u64)]) -> u64 {
        for (start, end) in ranges {
            if pos >= *start && pos < *end {
                return *end;
            }
        }
        pos
    }
    
    /// Find the start of the next searched range after pos
    #[cfg(feature = "semantic-search")]
    fn find_searched_region_start(pos: u64, ranges: &[(u64, u64)], file_size: u64) -> u64 {
        for (start, _) in ranges {
            if *start > pos {
                return *start;
            }
        }
        file_size
    }
    
    /// Get current hybrid search progress (if any)
    #[cfg(feature = "semantic-search")]
    pub fn hybrid_progress(&self) -> Option<HybridSearchProgress> {
        self.hybrid_state.as_ref().map(|state| {
            HybridSearchProgress {
                semantic_chunks_searched: state.searched_ranges.len(),
                semantic_chunks_total: state.searched_ranges.len(),
                bytes_scanned: state.full_scan_offset,
                bytes_total: self.file_size,
                semantic_complete: true,
                full_scan_complete: state.full_scan_offset >= self.file_size,
            }
        })
    }
    
    /// Clear hybrid search state
    #[cfg(feature = "semantic-search")]
    pub fn clear_hybrid_state(&mut self) {
        self.hybrid_state = None;
    }

    /// Forward search using memmem (memchr crate)
    fn search_forward_memmem(&self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        let query = if options.case_sensitive {
            options.query.as_bytes().to_vec()
        } else {
            options.query.to_lowercase().as_bytes().to_vec()
        };

        let finder = memmem::Finder::new(&query);
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(start_offset))?;

        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut current_offset = start_offset;
        let mut overlap = vec![0u8; query.len().saturating_sub(1)];
        let mut has_overlap = false;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            let search_buffer = if options.case_sensitive {
                buffer[..bytes_read].to_vec()
            } else {
                buffer[..bytes_read].iter().map(|b| b.to_ascii_lowercase()).collect()
            };

            // Check for match spanning chunks
            if has_overlap {
                let mut combined = overlap.clone();
                let take = query.len().saturating_sub(1).min(bytes_read);
                combined.extend_from_slice(&search_buffer[..take]);
                
                if let Some(pos) = finder.find(&combined) {
                    let match_offset = current_offset - overlap.len() as u64 + pos as u64;
                    return Ok(Some(SearchResult {
                        offset: match_offset,
                        length: query.len(),
                        snippet: None,
                    }));
                }
            }

            // Search in current chunk
            if let Some(pos) = finder.find(&search_buffer) {
                return Ok(Some(SearchResult {
                    offset: current_offset + pos as u64,
                    length: query.len(),
                    snippet: None,
                }));
            }

            // Save overlap for next iteration
            if bytes_read >= overlap.len() {
                let start = bytes_read - overlap.len();
                overlap.copy_from_slice(&search_buffer[start..bytes_read]);
                has_overlap = true;
            }

            current_offset += bytes_read as u64;
        }

        // Wrap around if enabled
        if options.wrap_around && start_offset > 0 {
            let mut wrapped_options = options.clone();
            wrapped_options.wrap_around = false;
            if let Some(result) = self.search_forward_memmem(&wrapped_options, 0)? {
                if result.offset < start_offset {
                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }

    /// Backward search using memmem
    fn search_backward_memmem(&self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        let query = if options.case_sensitive {
            options.query.as_bytes().to_vec()
        } else {
            options.query.to_lowercase().as_bytes().to_vec()
        };

        let finder = memmem::Finder::new(&query);
        let chunk_size = 64 * 1024u64;
        let mut file = File::open(&self.path)?;
        
        // Start from before start_offset
        let search_end = start_offset;
        let mut current_end = search_end;

        while current_end > 0 {
            let chunk_start = current_end.saturating_sub(chunk_size);
            let read_size = (current_end - chunk_start) as usize;
            
            file.seek(SeekFrom::Start(chunk_start))?;
            let mut buffer = vec![0u8; read_size];
            file.read_exact(&mut buffer)?;

            let search_buffer: Vec<u8> = if options.case_sensitive {
                buffer
            } else {
                buffer.iter().map(|b| b.to_ascii_lowercase()).collect()
            };

            // Find last match in buffer
            let mut last_match = None;
            let mut search_start = 0;
            while let Some(pos) = finder.find(&search_buffer[search_start..]) {
                let absolute_pos = chunk_start + search_start as u64 + pos as u64;
                if absolute_pos < search_end {
                    last_match = Some(SearchResult {
                        offset: absolute_pos,
                        length: query.len(),
                        snippet: None,
                    });
                    search_start += pos + 1;
                } else {
                    break;
                }
            }

            if let Some(result) = last_match {
                return Ok(Some(result));
            }

            current_end = chunk_start;
        }

        // Wrap around if enabled
        if options.wrap_around && start_offset < self.file_size {
            let mut wrapped_options = options.clone();
            wrapped_options.wrap_around = false;
            if let Some(result) = self.search_backward_memmem(&wrapped_options, self.file_size)? {
                if result.offset > start_offset {
                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }

    /// Compile or get cached regex
    fn get_regex(&mut self, pattern: &str, case_sensitive: bool) -> Result<&Regex> {
        let pattern_key = format!("{}:{}", case_sensitive, pattern);
        
        if self.last_regex_pattern.as_ref() != Some(&pattern_key) {
            let regex_pattern = if case_sensitive {
                pattern.to_string()
            } else {
                format!("(?i){}", pattern)
            };
            
            self.compiled_regex = Some(
                Regex::new(&regex_pattern)
                    .with_context(|| format!("Invalid regex pattern: {}", pattern))?
            );
            self.last_regex_pattern = Some(pattern_key);
        }

        Ok(self.compiled_regex.as_ref().unwrap())
    }

    /// Forward search using regex
    fn search_forward_regex(&mut self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        // Compile regex
        let regex_pattern = if options.case_sensitive {
            options.query.clone()
        } else {
            format!("(?i){}", options.query)
        };
        let regex = Regex::new(&regex_pattern)
            .with_context(|| format!("Invalid regex pattern: {}", options.query))?;

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(start_offset))?;

        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut current_offset = start_offset;
        
        // For regex, we need to handle potential matches at chunk boundaries
        // Keep some overlap to handle this
        let max_match_size = 1024; // Assume max match size
        let mut overlap_buffer = Vec::new();

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            // Combine overlap with current buffer
            let search_buffer = if overlap_buffer.is_empty() {
                &buffer[..bytes_read]
            } else {
                overlap_buffer.extend_from_slice(&buffer[..bytes_read]);
                &overlap_buffer[..]
            };

            // Search for match
            if let Some(m) = regex.find(search_buffer) {
                let offset_adjustment = if overlap_buffer.is_empty() {
                    0
                } else {
                    overlap_buffer.len() - bytes_read
                };
                
                let match_offset = current_offset - offset_adjustment as u64 + m.start() as u64;
                return Ok(Some(SearchResult {
                    offset: match_offset,
                    length: m.len(),
                    snippet: None,
                }));
            }

            // Keep overlap for next iteration
            current_offset += bytes_read as u64;
            let overlap_size = max_match_size.min(bytes_read);
            overlap_buffer = buffer[bytes_read - overlap_size..bytes_read].to_vec();
        }

        // Wrap around if enabled
        if options.wrap_around && start_offset > 0 {
            let mut wrapped_options = options.clone();
            wrapped_options.wrap_around = false;
            return self.search_forward_regex(&wrapped_options, 0);
        }

        Ok(None)
    }

    /// Backward search using regex
    fn search_backward_regex(&mut self, options: &SearchOptions, start_offset: u64) -> Result<Option<SearchResult>> {
        // For backward regex search, we need to find the last match before start_offset
        // This is less efficient as we need to scan from the beginning
        let regex_pattern = if options.case_sensitive {
            options.query.clone()
        } else {
            format!("(?i){}", options.query)
        };
        let regex = Regex::new(&regex_pattern)
            .with_context(|| format!("Invalid regex pattern: {}", options.query))?;

        let mut file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut current_offset = 0u64;
        let mut last_match: Option<SearchResult> = None;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            // Find all matches in this chunk
            for m in regex.find_iter(&buffer[..bytes_read]) {
                let match_offset = current_offset + m.start() as u64;
                if match_offset < start_offset {
                    last_match = Some(SearchResult {
                        offset: match_offset,
                        length: m.len(),
                        snippet: None,
                    });
                }
            }

            current_offset += bytes_read as u64;
            if current_offset >= start_offset {
                break;
            }
        }

        if last_match.is_some() {
            return Ok(last_match);
        }

        // Wrap around if enabled
        if options.wrap_around && start_offset < self.file_size {
            let mut wrapped_options = options.clone();
            wrapped_options.wrap_around = false;
            wrapped_options.direction = SearchDirection::Backward;
            return self.search_backward_regex(&wrapped_options, self.file_size);
        }

        Ok(None)
    }

    /// Count all matches in file
    pub fn count_matches(&self, query: &str, case_sensitive: bool, use_regex: bool) -> Result<usize> {
        if use_regex {
            self.count_matches_regex(query, case_sensitive)
        } else {
            self.count_matches_memmem(query, case_sensitive)
        }
    }

    /// Count matches using memmem
    fn count_matches_memmem(&self, query: &str, case_sensitive: bool) -> Result<usize> {
        let query_bytes = if case_sensitive {
            query.as_bytes().to_vec()
        } else {
            query.to_lowercase().as_bytes().to_vec()
        };

        let finder = memmem::Finder::new(&query_bytes);
        let mut file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut count = 0;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            let search_buffer: Vec<u8> = if case_sensitive {
                buffer[..bytes_read].to_vec()
            } else {
                buffer[..bytes_read].iter().map(|b| b.to_ascii_lowercase()).collect()
            };

            let mut pos = 0;
            while let Some(found) = finder.find(&search_buffer[pos..]) {
                count += 1;
                pos += found + 1;
            }
        }

        Ok(count)
    }

    /// Count matches using regex
    fn count_matches_regex(&self, pattern: &str, case_sensitive: bool) -> Result<usize> {
        let regex_pattern = if case_sensitive {
            pattern.to_string()
        } else {
            format!("(?i){}", pattern)
        };
        let regex = Regex::new(&regex_pattern)
            .with_context(|| format!("Invalid regex pattern: {}", pattern))?;

        let mut file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut count = 0;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            count += regex.find_iter(&buffer[..bytes_read]).count();
        }

        Ok(count)
    }

    /// Find all matches (for replace all)
    pub fn find_all(&self, query: &str, case_sensitive: bool, use_regex: bool) -> Result<Vec<SearchResult>> {
        if use_regex {
            self.find_all_regex(query, case_sensitive)
        } else {
            self.find_all_memmem(query, case_sensitive)
        }
    }

    /// Find all matches using memmem
    fn find_all_memmem(&self, query: &str, case_sensitive: bool) -> Result<Vec<SearchResult>> {
        let query_bytes = if case_sensitive {
            query.as_bytes().to_vec()
        } else {
            query.to_lowercase().as_bytes().to_vec()
        };

        let finder = memmem::Finder::new(&query_bytes);
        let mut file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut results = Vec::new();
        let mut current_offset = 0u64;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            let search_buffer: Vec<u8> = if case_sensitive {
                buffer[..bytes_read].to_vec()
            } else {
                buffer[..bytes_read].iter().map(|b| b.to_ascii_lowercase()).collect()
            };

            let mut pos = 0;
            while let Some(found) = finder.find(&search_buffer[pos..]) {
                results.push(SearchResult {
                    offset: current_offset + pos as u64 + found as u64,
                    length: query_bytes.len(),
                    snippet: None,
                });
                pos += found + 1;
            }

            current_offset += bytes_read as u64;
        }

        Ok(results)
    }

    /// Find all matches using regex
    fn find_all_regex(&self, pattern: &str, case_sensitive: bool) -> Result<Vec<SearchResult>> {
        let regex_pattern = if case_sensitive {
            pattern.to_string()
        } else {
            format!("(?i){}", pattern)
        };
        let regex = Regex::new(&regex_pattern)
            .with_context(|| format!("Invalid regex pattern: {}", pattern))?;

        let mut file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = vec![0u8; 64 * 1024];
        let mut results = Vec::new();
        let mut current_offset = 0u64;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            for m in regex.find_iter(&buffer[..bytes_read]) {
                results.push(SearchResult {
                    offset: current_offset + m.start() as u64,
                    length: m.len(),
                    snippet: None,
                });
            }

            current_offset += bytes_read as u64;
        }

        Ok(results)
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
    fn test_forward_search_memmem() {
        let file = create_test_file("hello world hello universe");
        let mut service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let options = SearchOptions {
            query: "hello".to_string(),
            case_sensitive: true,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        
        let result = service.search(&options, 0).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 0);
        
        let result = service.search(&options, 1).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 12);
    }

    #[test]
    fn test_backward_search_memmem() {
        let file = create_test_file("hello world hello universe");
        let mut service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let options = SearchOptions {
            query: "hello".to_string(),
            case_sensitive: true,
            direction: SearchDirection::Backward,
            wrap_around: false,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        
        let result = service.search(&options, 26).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 12);
    }

    #[test]
    fn test_case_insensitive_search() {
        let file = create_test_file("Hello World HELLO");
        let mut service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let options = SearchOptions {
            query: "hello".to_string(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        
        let result = service.search(&options, 0).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 0);
        
        let result = service.search(&options, 1).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 12);
    }

    #[test]
    fn test_regex_search() {
        let file = create_test_file("hello123 world456 hello789");
        let mut service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let options = SearchOptions {
            query: r"hello\d+".to_string(),
            case_sensitive: true,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Regex,
            use_regex: true,
        };
        
        let result = service.search(&options, 0).unwrap();
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.offset, 0);
        assert_eq!(r.length, 8); // "hello123"
    }

    #[test]
    fn test_count_matches() {
        let file = create_test_file("the quick brown the fox jumps the");
        let service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let count = service.count_matches("the", true, false).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_find_all() {
        let file = create_test_file("aaa bbb aaa ccc aaa");
        let service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let results = service.find_all("aaa", true, false).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].offset, 0);
        assert_eq!(results[1].offset, 8);
        assert_eq!(results[2].offset, 16);
    }

    #[test]
    fn test_wrap_around_forward() {
        let file = create_test_file("hello world");
        let mut service = SearchService::new(file.path(), crate::config::FtsSearchConfig::default()).unwrap();
        
        let options = SearchOptions {
            query: "hello".to_string(),
            case_sensitive: true,
            direction: SearchDirection::Forward,
            wrap_around: true,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        
        // Search from after "hello", should wrap and find it at 0
        let result = service.search(&options, 6).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().offset, 0);
    }
}
