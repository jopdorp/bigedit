//! Search benchmarks comparing regex, FTS5, and semantic search
//!
//! Run with: cargo run --release --bin bench_search -- [file] [queries...]

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Benchmark result for a single search strategy
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub strategy: String,
    pub query: String,
    pub matches: usize,
    pub search_time: Duration,
    pub index_time: Option<Duration>,
    pub index_size: Option<u64>,
}

impl BenchResult {
    pub fn print(&self) {
        let search_ms = self.search_time.as_secs_f64() * 1000.0;
        print!(
            "  {:<12} | {:>6} matches | {:>10.2}ms search",
            self.strategy, self.matches, search_ms
        );
        if let Some(idx_time) = self.index_time {
            print!(" | {:>8.2}s index", idx_time.as_secs_f64());
        }
        if let Some(idx_size) = self.index_size {
            print!(" | {:>8} index", format_bytes(idx_size));
        }
        println!();
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    
    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

/// Benchmark regex search (streaming)
pub fn bench_regex_search(file_path: &Path, pattern: &str) -> Result<BenchResult> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    
    let regex = regex::RegexBuilder::new(&regex::escape(pattern))
        .case_insensitive(true)
        .build()
        .context("Invalid regex pattern")?;
    
    let start = Instant::now();
    let mut matches = 0;
    let mut line_buf = String::new();
    
    while reader.read_line(&mut line_buf)? > 0 {
        matches += regex.find_iter(&line_buf).count();
        line_buf.clear();
    }
    
    let search_time = start.elapsed();
    
    Ok(BenchResult {
        strategy: "regex".to_string(),
        query: pattern.to_string(),
        matches,
        search_time,
        index_time: None,
        index_size: None,
    })
}

/// Benchmark memmem/memchr search (streaming, fastest for exact match)
pub fn bench_memmem_search(file_path: &Path, pattern: &str) -> Result<BenchResult> {
    use memchr::memmem;
    
    let mut file = File::open(file_path)?;
    
    let pattern_bytes = pattern.as_bytes();
    let finder = memmem::Finder::new(pattern_bytes);
    
    let start = Instant::now();
    let mut matches = 0;
    
    // Stream through file in chunks with overlap
    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4MB chunks
    let overlap = pattern_bytes.len().saturating_sub(1);
    let mut buffer = vec![0u8; CHUNK_SIZE + overlap];
    let mut carry_over = 0usize;
    
    loop {
        // Read into buffer after carry-over bytes
        let bytes_read = file.read(&mut buffer[carry_over..])?;
        if bytes_read == 0 {
            break;
        }
        
        let total_bytes = carry_over + bytes_read;
        let search_end = if bytes_read < CHUNK_SIZE { total_bytes } else { CHUNK_SIZE };
        
        // Count matches in this chunk
        let mut pos = 0;
        while let Some(offset) = finder.find(&buffer[pos..search_end]) {
            matches += 1;
            pos += offset + 1; // Move past this match
        }
        
        // Carry over the last (pattern_len - 1) bytes to catch matches across chunk boundaries
        if bytes_read == CHUNK_SIZE && overlap > 0 {
            buffer.copy_within((CHUNK_SIZE)..(CHUNK_SIZE + overlap), 0);
            carry_over = overlap;
        } else {
            carry_over = 0;
        }
    }
    
    let search_time = start.elapsed();
    
    Ok(BenchResult {
        strategy: "memmem".to_string(),
        query: pattern.to_string(),
        matches,
        search_time,
        index_time: None,
        index_size: None,
    })
}

/// Benchmark case-insensitive memmem search
pub fn bench_memmem_case_insensitive(file_path: &Path, pattern: &str) -> Result<BenchResult> {
    let mut file = File::open(file_path)?;
    
    let pattern_lower = pattern.to_lowercase();
    let pattern_bytes = pattern_lower.as_bytes();
    let finder = memchr::memmem::Finder::new(pattern_bytes);
    
    let start = Instant::now();
    let mut matches = 0;
    
    const CHUNK_SIZE: usize = 4 * 1024 * 1024;
    let overlap = pattern_bytes.len().saturating_sub(1);
    let mut buffer = vec![0u8; CHUNK_SIZE + overlap];
    let mut lower_buffer = vec![0u8; CHUNK_SIZE + overlap];
    let mut carry_over = 0usize;
    
    loop {
        let bytes_read = file.read(&mut buffer[carry_over..])?;
        if bytes_read == 0 {
            break;
        }
        
        let total_bytes = carry_over + bytes_read;
        let search_end = if bytes_read < CHUNK_SIZE { total_bytes } else { CHUNK_SIZE };
        
        // Convert to lowercase for case-insensitive search
        for i in 0..search_end {
            lower_buffer[i] = buffer[i].to_ascii_lowercase();
        }
        
        let mut pos = 0;
        while let Some(offset) = finder.find(&lower_buffer[pos..search_end]) {
            matches += 1;
            pos += offset + 1;
        }
        
        if bytes_read == CHUNK_SIZE && overlap > 0 {
            buffer.copy_within((CHUNK_SIZE)..(CHUNK_SIZE + overlap), 0);
            carry_over = overlap;
        } else {
            carry_over = 0;
        }
    }
    
    let search_time = start.elapsed();
    
    Ok(BenchResult {
        strategy: "memmem-ci".to_string(),
        query: pattern.to_string(),
        matches,
        search_time,
        index_time: None,
        index_size: None,
    })
}

/// Run all benchmarks for a file and set of queries
pub fn run_benchmarks(file_path: &Path, queries: &[String]) -> Result<Vec<BenchResult>> {
    let file_size = std::fs::metadata(file_path)?.len();
    println!("=== Search Benchmarks ===");
    println!("File: {} ({})", file_path.display(), format_bytes(file_size));
    println!();
    
    let mut results = Vec::new();
    
    for query in queries {
        println!("Query: \"{}\"", query);
        println!("{:-<60}", "");
        
        // memmem (exact, case-sensitive) - fastest baseline
        match bench_memmem_search(file_path, query) {
            Ok(r) => {
                r.print();
                results.push(r);
            }
            Err(e) => eprintln!("  memmem failed: {}", e),
        }
        
        // memmem case-insensitive
        match bench_memmem_case_insensitive(file_path, query) {
            Ok(r) => {
                r.print();
                results.push(r);
            }
            Err(e) => eprintln!("  memmem-ci failed: {}", e),
        }
        
        // regex
        match bench_regex_search(file_path, query) {
            Ok(r) => {
                r.print();
                results.push(r);
            }
            Err(e) => eprintln!("  regex failed: {}", e),
        }
        
        println!();
    }
    
    Ok(results)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <file> [query1] [query2] ...", args[0]);
        eprintln!();
        eprintln!("If no queries provided, uses default test queries.");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} benches/data/simplewiki-content.txt \"Albert Einstein\" \"1879\"", args[0]);
        std::process::exit(1);
    }
    
    let file_path = PathBuf::from(&args[1]);
    if !file_path.exists() {
        eprintln!("File not found: {}", file_path.display());
        eprintln!();
        eprintln!("Download Wikipedia test data with:");
        eprintln!("  ./benches/download_wikipedia.sh benches/data simple");
        std::process::exit(1);
    }
    
    let queries: Vec<String> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        // Default queries for Wikipedia benchmarking
        vec![
            "the".to_string(),           // Very common word
            "Einstein".to_string(),      // Proper noun
            "1879".to_string(),          // Number (year)
            "quantum mechanics".to_string(), // Multi-word phrase
            "xyzzy".to_string(),         // Rare/non-existent (worst case)
        ]
    };
    
    run_benchmarks(&file_path, &queries)?;
    
    Ok(())
}
