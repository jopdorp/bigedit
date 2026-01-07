//! Wikipedia benchmark tests
//!
//! Benchmarks three search strategies:
//! - memmem: Fast byte-level search using memchr (~5 GB/s)
//! - regex: Pattern matching with regex crate (~50-500 MB/s)
//! - semantic: Embedding-based search (requires semantic-search feature)
//!
//! Run with: cargo test --test wikipedia_bench -- --ignored --nocapture
//! With semantic: cargo test --features semantic-search --test wikipedia_bench -- --ignored --nocapture

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use regex::bytes::Regex;

/// Get the data directory for Wikipedia dumps
fn data_dir() -> PathBuf {
    std::env::var("WIKI_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("benches/data"))
}

/// Download Wikipedia from HuggingFace
fn download_wiki_direct(wiki: &str) -> Result<PathBuf, String> {
    let dir = data_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create data dir: {}", e))?;

    let txt_path = dir.join(format!("{}.txt", wiki));

    if txt_path.exists() {
        let size = fs::metadata(&txt_path).map(|m| m.len()).unwrap_or(0);
        println!("  {} already exists ({} MB)", txt_path.display(), size / 1024 / 1024);
        return Ok(txt_path);
    }

    // Check for parquet file that needs extraction
    let parquet_path = dir.join(format!("{}.parquet", wiki));
    if parquet_path.exists() {
        println!("  Found parquet file, but text file missing.");
        println!("  Run: cd benches && source .venv/bin/activate && python3 -c \"");
        println!("    import pyarrow.parquet as pq");
        println!("    table = pq.read_table('data/{}.parquet')", wiki);
        println!("    with open('data/{}.txt', 'w') as f:", wiki);
        println!("      for t in table['text'].to_pylist(): f.write(t + '\\n\\n')");
        println!("  \"");
        return Err("Parquet exists but text not extracted".to_string());
    }

    if std::env::var("SKIP_DOWNLOAD").is_ok() {
        return Err(format!("SKIP_DOWNLOAD set but {} not found", txt_path.display()));
    }

    // Try to download from HuggingFace
    println!("  Downloading {} from HuggingFace...", wiki);
    
    let hf_url = match wiki {
        "simple-wikipedia" => "https://huggingface.co/datasets/rahular/simple-wikipedia/resolve/main/data/train-00000-of-00001-090b52ccb189d47a.parquet",
        _ => return Err(format!("Unknown wiki: {}. Manually download to {}", wiki, txt_path.display())),
    };

    let status = Command::new("curl")
        .args(["-L", "-o"])
        .arg(&parquet_path)
        .arg(hf_url)
        .status()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !status.success() {
        return Err("Download failed".to_string());
    }

    println!("  Downloaded parquet. Extract with Python:");
    println!("  cd benches && source .venv/bin/activate && pip install pyarrow");
    println!("  Then run this test again after extraction.");
    
    Err("Parquet downloaded, needs text extraction".to_string())
}

/// Search result for benchmarking
#[derive(Debug, Clone)]
struct BenchResult {
    strategy: String,
    query: String,
    matches_found: usize,
    time_ms: f64,
    first_match_pos: Option<u64>,
}

/// Run memmem streaming search (literal byte matching)
fn bench_memmem(file_path: &Path, query: &str) -> Result<BenchResult, String> {
    let start = Instant::now();

    let file = File::open(file_path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let pattern = query.as_bytes();
    let finder = memchr::memmem::Finder::new(pattern);

    let mut matches = 0;
    let mut first_pos: Option<u64> = None;
    let mut buffer = vec![0u8; 16 * 1024 * 1024]; // 16MB chunks
    let mut file_offset = 0u64;
    let overlap = pattern.len().saturating_sub(1);
    let mut carry_over = 0usize;

    loop {
        // Keep overlap from previous chunk
        if carry_over > 0 {
            let buf_len = buffer.len();
            buffer.copy_within((buf_len - carry_over).., 0);
        }

        let bytes_read = reader.read(&mut buffer[carry_over..]).map_err(|e| e.to_string())?;

        if bytes_read == 0 {
            break;
        }

        let chunk_len = carry_over + bytes_read;
        let search_area = &buffer[..chunk_len];

        for offset in finder.find_iter(search_area) {
            let absolute_pos = file_offset + offset as u64;
            if first_pos.is_none() {
                first_pos = Some(absolute_pos);
            }
            matches += 1;
        }

        file_offset += (chunk_len - overlap) as u64;
        carry_over = overlap.min(chunk_len);
    }

    let elapsed = start.elapsed();

    Ok(BenchResult {
        strategy: "memmem".to_string(),
        query: query.to_string(),
        matches_found: matches,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        first_match_pos: first_pos,
    })
}

/// Run regex streaming search
fn bench_regex(file_path: &Path, pattern: &str) -> Result<BenchResult, String> {
    let start = Instant::now();

    // Compile regex (case-insensitive by default for fair comparison)
    let regex = Regex::new(&format!("(?i){}", regex::escape(pattern)))
        .map_err(|e| format!("Invalid regex: {}", e))?;

    let file = File::open(file_path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let mut matches = 0;
    let mut first_pos: Option<u64> = None;
    let mut buffer = vec![0u8; 16 * 1024 * 1024]; // 16MB chunks
    let mut file_offset = 0u64;
    // For regex, keep larger overlap to handle multi-byte matches
    let overlap = 1024.min(pattern.len() * 10);
    let mut carry_over = 0usize;

    loop {
        if carry_over > 0 {
            let buf_len = buffer.len();
            buffer.copy_within((buf_len - carry_over).., 0);
        }

        let bytes_read = reader.read(&mut buffer[carry_over..]).map_err(|e| e.to_string())?;

        if bytes_read == 0 {
            break;
        }

        let chunk_len = carry_over + bytes_read;
        let search_area = &buffer[..chunk_len];

        for m in regex.find_iter(search_area) {
            let absolute_pos = file_offset + m.start() as u64;
            if first_pos.is_none() {
                first_pos = Some(absolute_pos);
            }
            matches += 1;
        }

        file_offset += (chunk_len - overlap) as u64;
        carry_over = overlap.min(chunk_len);
    }

    let elapsed = start.elapsed();

    Ok(BenchResult {
        strategy: "regex".to_string(),
        query: pattern.to_string(),
        matches_found: matches,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        first_match_pos: first_pos,
    })
}

/// Run regex with actual pattern (not escaped literal)
fn bench_regex_pattern(file_path: &Path, pattern: &str) -> Result<BenchResult, String> {
    let start = Instant::now();

    let regex = Regex::new(pattern).map_err(|e| format!("Invalid regex: {}", e))?;

    let file = File::open(file_path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let mut matches = 0;
    let mut first_pos: Option<u64> = None;
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    let mut file_offset = 0u64;
    let overlap = 1024;
    let mut carry_over = 0usize;

    loop {
        if carry_over > 0 {
            let buf_len = buffer.len();
            buffer.copy_within((buf_len - carry_over).., 0);
        }

        let bytes_read = reader.read(&mut buffer[carry_over..]).map_err(|e| e.to_string())?;

        if bytes_read == 0 {
            break;
        }

        let chunk_len = carry_over + bytes_read;
        let search_area = &buffer[..chunk_len];

        for m in regex.find_iter(search_area) {
            let absolute_pos = file_offset + m.start() as u64;
            if first_pos.is_none() {
                first_pos = Some(absolute_pos);
            }
            matches += 1;
        }

        file_offset += (chunk_len - overlap) as u64;
        carry_over = overlap.min(chunk_len);
    }

    let elapsed = start.elapsed();

    Ok(BenchResult {
        strategy: "regex-pattern".to_string(),
        query: pattern.to_string(),
        matches_found: matches,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        first_match_pos: first_pos,
    })
}

/// Semantic search benchmark (requires semantic-search feature)
/// 
/// Uses embedding index (index once, search instant)
#[cfg(feature = "semantic-search")]
fn bench_semantic(file_path: &Path, query: &str) -> Result<BenchResult, String> {
    // Use embedding-based search for all files
    bench_semantic_embeddings(file_path, query)
}

/// Embedding-based semantic search
/// Index once (slow), then search many times (instant)
#[cfg(feature = "semantic-search")]
fn bench_semantic_embeddings(file_path: &Path, query: &str) -> Result<BenchResult, String> {
    use bigedit::semantic::{EmbeddingModel, SemanticIndex};
    use bigedit::config::SemanticSearchConfig;
    use std::io::BufRead;
    use std::sync::Mutex;
    
    // Use static to cache index across queries (model is small to reload)
    static SEMANTIC_INDEX: std::sync::OnceLock<Mutex<Option<SemanticIndex>>> = std::sync::OnceLock::new();
    static INDEX_BUILT_FOR: std::sync::OnceLock<Mutex<Option<std::path::PathBuf>>> = std::sync::OnceLock::new();
    
    let file_size = fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);
    
    println!("    (Embedding-based semantic search)");
    
    // Check if we need to build the index
    let index_path = INDEX_BUILT_FOR.get_or_init(|| Mutex::new(None));
    let needs_build = {
        let guard = index_path.lock().unwrap();
        guard.as_ref() != Some(&file_path.to_path_buf())
    };
    
    if needs_build {
        println!("    Loading embedding model...");
        let model_start = Instant::now();
        let mut model = EmbeddingModel::load_default().map_err(|e| e.to_string())?;
        println!("    Model loaded in {:.2}s", model_start.elapsed().as_secs_f64());
        
        println!("    Building semantic index...");
        let index_start = Instant::now();
        
        let config = SemanticSearchConfig::default();
        let mut index = SemanticIndex::new(file_path, config).map_err(|e| e.to_string())?;
        
        let chunk_size = index.chunk_size();
        let file = File::open(file_path).map_err(|e| e.to_string())?;
        let reader = std::io::BufReader::new(file);
        let mut current_offset = 0u64;
        let mut chunks = 0;
        
        let mut chunk_text = String::new();
        let mut line_reader = reader;
        
        loop {
            chunk_text.clear();
            let mut bytes_read = 0;
            
            loop {
                let mut line = String::new();
                match line_reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(n) => {
                        bytes_read += n;
                        chunk_text.push_str(&line);
                        if bytes_read >= chunk_size {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            
            if chunk_text.is_empty() {
                break;
            }
            
            let embedding = model.embed(&chunk_text).map_err(|e| e.to_string())?;
            index.add_chunk(current_offset, chunk_text.len(), embedding);
            
            current_offset += chunk_text.len() as u64;
            chunks += 1;
            
            if chunks % 100 == 0 {
                let progress = (current_offset as f64 / file_size as f64) * 100.0;
                eprint!("\r    Indexed {} chunks ({:.1}%)...", chunks, progress);
            }
        }
        
        index.mark_complete();
        let elapsed = index_start.elapsed();
        let mb_per_sec = (file_size as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
        eprintln!("\r    Indexed {} chunks in {:.2}s ({:.1} MB/s)                    ", chunks, elapsed.as_secs_f64(), mb_per_sec);
        
        // Cache the index
        let index_storage = SEMANTIC_INDEX.get_or_init(|| Mutex::new(None));
        *index_storage.lock().unwrap() = Some(index);
        *index_path.lock().unwrap() = Some(file_path.to_path_buf());
    }
    
    // Get the cached index
    let index_storage = SEMANTIC_INDEX.get_or_init(|| Mutex::new(None));
    let index_guard = index_storage.lock().unwrap();
    let index = index_guard.as_ref().ok_or("Index not built")?;
    
    // Load model for query embedding (fast, model is cached by ort)
    let mut model = EmbeddingModel::load_default().map_err(|e| e.to_string())?;
    
    // Now search (this should be instant!)
    let search_start = Instant::now();
    
    let query_embedding = model.embed(query).map_err(|e| e.to_string())?;
    let results = index.search(&query_embedding, 10);
    
    let search_elapsed = search_start.elapsed();
    
    println!("    Search time: {:.3}ms (found {} results)", 
             search_elapsed.as_secs_f64() * 1000.0, results.len());
    
    Ok(BenchResult {
        strategy: "semantic-embed".to_string(),
        query: query.to_string(),
        matches_found: results.len(),
        time_ms: search_elapsed.as_secs_f64() * 1000.0,
        first_match_pos: results.first().map(|r| r.offset),
    })
}

#[cfg(not(feature = "semantic-search"))]
fn bench_semantic(_file_path: &Path, query: &str) -> Result<BenchResult, String> {
    Err(format!("semantic-search feature not enabled (query: {})", query))
}

/// Test queries - common words and phrases
fn test_queries() -> Vec<&'static str> {
    vec![
        "the",
        "Wikipedia",
        "Africa",
        "language",
        "history",
        "encyclopedia",
        "United States",
        "New York",
        "Simple English",
    ]
}

/// Regex-specific patterns to benchmark
fn regex_patterns() -> Vec<(&'static str, &'static str)> {
    vec![
        ("word boundary", r"\bthe\b"),
        ("digits", r"\d{4}"),           // 4-digit numbers (years)
        ("email-like", r"[a-z]+@[a-z]+\.[a-z]+"),
        ("capitalized", r"[A-Z][a-z]+"),
        ("url-like", r"https?://[^\s]+"),
    ]
}

/// Print results table
fn print_results(results: &[BenchResult], file_size_mb: f64) {
    println!();
    println!("| Strategy | Query | Matches | Time (ms) | MB/s |");
    println!("|----------|-------|---------|-----------|------|");
    for r in results {
        let mb_per_sec = if r.time_ms > 0.0 {
            file_size_mb / (r.time_ms / 1000.0)
        } else {
            0.0
        };
        let query_display = if r.query.len() > 20 {
            format!("{}...", &r.query[..17])
        } else {
            r.query.clone()
        };
        println!(
            "| {:12} | {:20} | {:7} | {:9.2} | {:7.1} |",
            r.strategy,
            query_display,
            r.matches_found,
            r.time_ms,
            mb_per_sec
        );
    }
}

/// Compare memmem vs regex accuracy
fn compare_memmem_regex(memmem_results: &[BenchResult], regex_results: &[BenchResult]) {
    println!();
    println!("=== Memmem vs Regex Comparison ===");
    println!("| Query | memmem | regex | Match? | memmem MB/s | regex MB/s |");
    println!("|-------|--------|-------|--------|-------------|------------|");

    for (m, r) in memmem_results.iter().zip(regex_results.iter()) {
        let matches = m.matches_found == r.matches_found;
        let icon = if matches { "✓" } else { "✗" };
        // Note: We don't have file_size here, so just show times
        println!(
            "| {:20} | {:6} | {:6} | {} | {:9.2}ms | {:9.2}ms |",
            if m.query.len() > 20 { &m.query[..17] } else { &m.query },
            m.matches_found,
            r.matches_found,
            icon,
            m.time_ms,
            r.time_ms,
        );
    }
}

#[test]
#[ignore] // Run with --ignored flag
fn bench_simple_english_wikipedia() {
    println!("\n=== Simple English Wikipedia Benchmark ===\n");
    println!("Comparing: memmem (literal) vs regex vs semantic (embeddings)\n");

    // Download/find the file
    let file_path = match download_wiki_direct("simple-wikipedia") {
        Ok(p) => p,
        Err(e) => {
            println!("Skipping: {}", e);
            return;
        }
    };

    let file_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
    let file_size_mb = file_size as f64 / 1024.0 / 1024.0;
    println!("\nFile: {} ({:.1} MB)\n", file_path.display(), file_size_mb);

    // ============ MEMMEM BENCHMARKS ============
    println!("--- Memmem (literal byte search, ~5 GB/s expected) ---");
    let mut memmem_results = Vec::new();
    for query in test_queries() {
        match bench_memmem(&file_path, query) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("  '{}': {} matches in {:.2}ms ({:.0} MB/s)", query, r.matches_found, r.time_ms, mb_s);
                memmem_results.push(r);
            }
            Err(e) => println!("  '{}': ERROR: {}", query, e),
        }
    }

    // ============ REGEX BENCHMARKS (literal, escaped) ============
    println!("\n--- Regex (escaped literal, case-insensitive) ---");
    let mut regex_results = Vec::new();
    for query in test_queries() {
        match bench_regex(&file_path, query) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("  '{}': {} matches in {:.2}ms ({:.0} MB/s)", query, r.matches_found, r.time_ms, mb_s);
                regex_results.push(r);
            }
            Err(e) => println!("  '{}': ERROR: {}", query, e),
        }
    }

    // ============ REGEX PATTERN BENCHMARKS ============
    println!("\n--- Regex Patterns (actual regex features) ---");
    let mut pattern_results = Vec::new();
    for (name, pattern) in regex_patterns() {
        match bench_regex_pattern(&file_path, pattern) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("  {} '{}': {} matches in {:.2}ms ({:.0} MB/s)", name, pattern, r.matches_found, r.time_ms, mb_s);
                pattern_results.push(r);
            }
            Err(e) => println!("  {} '{}': ERROR: {}", name, pattern, e),
        }
    }

    // ============ SEMANTIC BENCHMARKS ============
    println!("\n--- Semantic Search (embedding-based) ---");
    #[cfg(feature = "semantic-search")]
    {
        let semantic_queries = vec!["history of Africa", "programming language", "famous person"];
        for query in semantic_queries {
            match bench_semantic(&file_path, query) {
                Ok(r) => {
                    println!("  '{}': {} results in {:.2}ms", query, r.matches_found, r.time_ms);
                }
                Err(e) => println!("  '{}': ERROR: {}", query, e),
            }
        }
    }
    #[cfg(not(feature = "semantic-search"))]
    {
        println!("  (semantic-search feature not enabled)");
        println!("  Run with: cargo test --features semantic-search --test wikipedia_bench -- --ignored --nocapture");
    }

    // ============ COMPARISON ============
    println!("\n--- Results Summary ---");
    print_results(&memmem_results, file_size_mb);
    println!();
    print_results(&regex_results, file_size_mb);
    println!();
    print_results(&pattern_results, file_size_mb);
    
    compare_memmem_regex(&memmem_results, &regex_results);

    // ============ SUMMARY ============
    println!("\n=== Performance Summary ===");
    
    let avg_memmem_ms: f64 = memmem_results.iter().map(|r| r.time_ms).sum::<f64>() / memmem_results.len() as f64;
    let avg_regex_ms: f64 = regex_results.iter().map(|r| r.time_ms).sum::<f64>() / regex_results.len() as f64;
    
    println!("Average memmem: {:.2}ms ({:.0} MB/s)", avg_memmem_ms, file_size_mb / (avg_memmem_ms / 1000.0));
    println!("Average regex:  {:.2}ms ({:.0} MB/s)", avg_regex_ms, file_size_mb / (avg_regex_ms / 1000.0));
    println!("Regex slowdown: {:.1}x", avg_regex_ms / avg_memmem_ms);
}

#[test]
#[ignore]
fn bench_memmem_only() {
    println!("\n=== Memmem-only Benchmark ===\n");

    let file_path = match download_wiki_direct("simple-wikipedia") {
        Ok(p) => p,
        Err(e) => {
            println!("Skipping: {}", e);
            return;
        }
    };

    let file_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
    let file_size_mb = file_size as f64 / 1024.0 / 1024.0;
    println!("File: {} ({:.1} MB)\n", file_path.display(), file_size_mb);

    for query in test_queries() {
        match bench_memmem(&file_path, query) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("'{}': {} matches in {:.2}ms ({:.0} MB/s)", query, r.matches_found, r.time_ms, mb_s);
            }
            Err(e) => println!("'{}': ERROR: {}", query, e),
        }
    }
}

#[test]
#[ignore]
fn bench_regex_only() {
    println!("\n=== Regex-only Benchmark ===\n");

    let file_path = match download_wiki_direct("simple-wikipedia") {
        Ok(p) => p,
        Err(e) => {
            println!("Skipping: {}", e);
            return;
        }
    };

    let file_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
    let file_size_mb = file_size as f64 / 1024.0 / 1024.0;
    println!("File: {} ({:.1} MB)\n", file_path.display(), file_size_mb);

    println!("--- Literal (escaped) ---");
    for query in test_queries() {
        match bench_regex(&file_path, query) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("'{}': {} matches in {:.2}ms ({:.0} MB/s)", query, r.matches_found, r.time_ms, mb_s);
            }
            Err(e) => println!("'{}': ERROR: {}", query, e),
        }
    }

    println!("\n--- Patterns ---");
    for (name, pattern) in regex_patterns() {
        match bench_regex_pattern(&file_path, pattern) {
            Ok(r) => {
                let mb_s = file_size_mb / (r.time_ms / 1000.0);
                println!("{} '{}': {} matches in {:.2}ms ({:.0} MB/s)", name, pattern, r.matches_found, r.time_ms, mb_s);
            }
            Err(e) => println!("{} '{}': ERROR: {}", name, pattern, e),
        }
    }
}
