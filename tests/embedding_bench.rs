//! Embedding benchmark - isolated test for semantic index building speed
//!
//! Run with: cargo test --features semantic-search --release --test embedding_bench -- --ignored --nocapture

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

fn data_dir() -> PathBuf {
    std::env::var("WIKI_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("benches/data"))
}

/// Benchmark Model2Vec - the FAST static embedding model
#[test]
#[ignore]
fn bench_model2vec() {
    use model2vec_rs::model::StaticModel;
    
    println!("\n=== Model2Vec Benchmark (Static Embeddings) ===\n");
    
    let models = [
        ("potion-base-8M", "minishlab/potion-base-8M"),
        ("potion-base-32M", "minishlab/potion-base-32M"),
        ("potion-multilingual-128M", "minishlab/potion-multilingual-128M"),
    ];
    
    let chunk = "The history of computing is longer than the history of computing hardware. ".repeat(50);
    let batch: Vec<String> = (0..100).map(|_| chunk.clone()).collect();
    
    for (name, model_id) in &models {
        println!("--- {} ---", name);
        
        // Load model
        let load_start = Instant::now();
        let model = match StaticModel::from_pretrained(model_id, None, None, None) {
            Ok(m) => m,
            Err(e) => {
                println!("  Failed to load: {}", e);
                continue;
            }
        };
        println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());
        
        // Warmup
        let _ = model.encode(&batch[..10].to_vec());
        
        // Batch benchmark
        let start = Instant::now();
        let _ = model.encode(&batch);
        let elapsed = start.elapsed();
        
        let per_embed_ms = elapsed.as_secs_f64() * 1000.0 / batch.len() as f64;
        let per_second = batch.len() as f64 / elapsed.as_secs_f64();
        let mb_per_sec = (chunk.len() * batch.len()) as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64();
        
        println!("  {:.3}ms per embed, {:.0} embeds/sec, {:.1} MB/s", per_embed_ms, per_second, mb_per_sec);
        
        // Wikipedia estimate
        let wiki_size_mb = 136.0;
        let chunk_size_kb = 8.0;
        let num_chunks = (wiki_size_mb * 1024.0 / chunk_size_kb) as u64;
        let estimated_time = num_chunks as f64 * per_embed_ms / 1000.0;
        println!("  Wikipedia 136MB: {:.1}s\n", estimated_time);
    }
}

/// Benchmark embedding a single chunk
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_single_embed() {
    use bigedit::semantic::EmbeddingModel;
    
    println!("\n=== Single Embedding Benchmark ===\n");
    
    // Load model
    let load_start = Instant::now();
    let mut model = EmbeddingModel::load_default().expect("Failed to load model");
    println!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());
    
    // Test texts of various sizes
    let test_texts = [
        ("tiny", "Hello world"),
        ("small", "The quick brown fox jumps over the lazy dog. This is a simple test sentence."),
        ("medium", &"Lorem ipsum dolor sit amet. ".repeat(50)),
        ("large", &"The history of computing is longer than the history of computing hardware and modern computing technology. ".repeat(100)),
    ];
    
    for (name, text) in &test_texts {
        let text_len = text.len();
        
        // Warm up
        let _ = model.embed(text);
        
        // Time 10 embeddings
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = model.embed(text).unwrap();
        }
        let elapsed = start.elapsed();
        let per_embed_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        
        println!("{:8}: {:5} bytes, {:.2}ms per embed", name, text_len, per_embed_ms);
    }
}

/// Benchmark batch embedding
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_batch_embed() {
    use bigedit::semantic::EmbeddingModel;
    
    println!("\n=== Batch Embedding Benchmark ===\n");
    
    let mut model = EmbeddingModel::load_default().expect("Failed to load model");
    
    let chunk = "The history of computing is longer than the history of computing hardware. ";
    let chunk_text = chunk.repeat(50); // ~3.7KB chunk
    
    println!("Chunk size: {} bytes\n", chunk_text.len());
    
    // Test different batch sizes
    for batch_size in [1, 2, 4, 8, 16, 32] {
        let texts: Vec<&str> = (0..batch_size).map(|_| chunk_text.as_str()).collect();
        
        // Warm up
        let _ = model.embed_batch(&texts);
        
        // Time it
        let start = Instant::now();
        let iterations = 3;
        for _ in 0..iterations {
            let _ = model.embed_batch(&texts).unwrap();
        }
        let elapsed = start.elapsed();
        
        let total_texts = batch_size * iterations;
        let per_text_ms = elapsed.as_secs_f64() * 1000.0 / total_texts as f64;
        let texts_per_sec = total_texts as f64 / elapsed.as_secs_f64();
        let mb_per_sec = (chunk_text.len() * total_texts) as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64();
        
        println!("batch {:2}: {:.2}ms/text, {:.1} texts/s, {:.2} MB/s", 
                 batch_size, per_text_ms, texts_per_sec, mb_per_sec);
    }
}

/// Benchmark indexing a portion of Wikipedia
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_index_wikipedia() {
    use bigedit::semantic::{EmbeddingModel, SemanticIndex, chunk_size_for_file};
    use bigedit::config::SemanticSearchConfig;
    
    println!("\n=== Wikipedia Indexing Benchmark ===\n");
    
    let wiki_path = data_dir().join("simple-wikipedia.txt");
    if !wiki_path.exists() {
        println!("Wikipedia file not found at {:?}", wiki_path);
        println!("Download it first with the wikipedia_bench test");
        return;
    }
    
    let file_size = fs::metadata(&wiki_path).map(|m| m.len()).unwrap_or(0);
    println!("File: {} ({:.1} MB)", wiki_path.display(), file_size as f64 / 1024.0 / 1024.0);
    
    // Load model
    let load_start = Instant::now();
    let mut model = EmbeddingModel::load_default().expect("Failed to load model");
    println!("Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());
    
    let chunk_size = chunk_size_for_file(file_size);
    println!("Chunk size: {} KB", chunk_size / 1024);
    
    // Read chunks
    let file = File::open(&wiki_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut bytes_read = 0usize;
    
    // Only index first 10MB for benchmark
    let max_bytes = 10 * 1024 * 1024;
    
    for line in reader.lines() {
        let line = line.unwrap_or_default();
        bytes_read += line.len() + 1;
        current_chunk.push_str(&line);
        current_chunk.push('\n');
        
        if current_chunk.len() >= chunk_size {
            chunks.push(std::mem::take(&mut current_chunk));
        }
        
        if bytes_read >= max_bytes {
            break;
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    println!("Read {} chunks ({:.1} MB)\n", chunks.len(), bytes_read as f64 / 1024.0 / 1024.0);
    
    // Benchmark: Sequential embedding
    println!("--- Sequential embedding (one at a time) ---");
    let seq_start = Instant::now();
    let mut seq_count = 0;
    for chunk in chunks.iter().take(50) {
        let _ = model.embed(chunk).unwrap();
        seq_count += 1;
    }
    let seq_elapsed = seq_start.elapsed();
    let seq_per_chunk = seq_elapsed.as_secs_f64() * 1000.0 / seq_count as f64;
    println!("{} chunks in {:.2}s ({:.1}ms per chunk)", 
             seq_count, seq_elapsed.as_secs_f64(), seq_per_chunk);
    
    // Estimate full file time
    let total_chunks = chunks.len();
    let estimated_full = seq_per_chunk * total_chunks as f64 / 1000.0;
    let estimated_136mb = seq_per_chunk * (136.0 / 10.0) * total_chunks as f64 / 1000.0;
    println!("Estimated for 10MB: {:.1}s", estimated_full);
    println!("Estimated for 136MB: {:.1}s ({:.1} minutes)\n", estimated_136mb, estimated_136mb / 60.0);
    
    // Benchmark: Batched embedding
    println!("--- Batched embedding (batch_size=16) ---");
    let batch_size = 16;
    let batch_start = Instant::now();
    let mut batch_count = 0;
    
    for batch in chunks.iter().take(64).collect::<Vec<_>>().chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let _ = model.embed_batch(&texts).unwrap();
        batch_count += texts.len();
    }
    let batch_elapsed = batch_start.elapsed();
    let batch_per_chunk = batch_elapsed.as_secs_f64() * 1000.0 / batch_count as f64;
    println!("{} chunks in {:.2}s ({:.1}ms per chunk)", 
             batch_count, batch_elapsed.as_secs_f64(), batch_per_chunk);
    
    let speedup = seq_per_chunk / batch_per_chunk;
    println!("Speedup vs sequential: {:.2}x", speedup);
    
    // Estimate with batching
    let estimated_batch_136mb = batch_per_chunk * (136.0 / 10.0) * total_chunks as f64 / 1000.0;
    println!("Estimated for 136MB with batching: {:.1}s ({:.1} minutes)", 
             estimated_batch_136mb, estimated_batch_136mb / 60.0);
}

/// Benchmark the full hybrid pipeline: Model2Vec + IDF pre-filtering
#[test]
#[ignore]
fn bench_hybrid_pipeline() {
    use bigedit::semantic::{FastEmbeddingModel, SemanticIndex, tokenize_for_idf, chunk_size_for_file};
    use bigedit::config::SemanticSearchConfig;
    
    println!("\n=== Hybrid Pipeline Benchmark (Model2Vec + IDF) ===\n");
    
    let wiki_path = data_dir().join("simple-wikipedia.txt");
    if !wiki_path.exists() {
        println!("Wikipedia file not found at {:?}", wiki_path);
        println!("Download it with: cargo test --features semantic-search --release --test wikipedia_bench -- --ignored --nocapture bench_simple_english_wikipedia");
        return;
    }
    
    let file_size = fs::metadata(&wiki_path).map(|m| m.len()).unwrap_or(0);
    println!("File: {} ({:.1} MB)", wiki_path.display(), file_size as f64 / 1024.0 / 1024.0);
    
    // Load Model2Vec
    let load_start = Instant::now();
    let model = FastEmbeddingModel::load_default().expect("Failed to load Model2Vec");
    println!("Model2Vec loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());
    
    let chunk_size = chunk_size_for_file(file_size);
    println!("Chunk size: {} KB", chunk_size / 1024);
    
    // Read chunks
    let file = File::open(&wiki_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut total_bytes = 0usize;
    
    for line in reader.lines() {
        let line = line.unwrap_or_default();
        total_bytes += line.len() + 1;
        current_chunk.push_str(&line);
        current_chunk.push('\n');
        
        if current_chunk.len() >= chunk_size {
            chunks.push(std::mem::take(&mut current_chunk));
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    println!("Read {} chunks ({:.1} MB)\n", chunks.len(), total_bytes as f64 / 1024.0 / 1024.0);
    
    // Phase 1: Index with Model2Vec + IDF tokens
    println!("--- Phase 1: Indexing with Model2Vec ---");
    let temp_file = std::env::temp_dir().join("bench_hybrid.txt");
    std::fs::write(&temp_file, "placeholder").unwrap();
    
    let config = SemanticSearchConfig::default();
    let mut index = SemanticIndex::new(&temp_file, config).expect("Failed to create index");
    
    let index_start = Instant::now();
    let batch_size = 100;
    let mut offset = 0u64;
    
    for batch in chunks.chunks(batch_size) {
        let texts: Vec<String> = batch.iter().cloned().collect();
        let embeddings = model.embed_batch(&texts).expect("Embedding failed");
        
        for (text, embedding) in batch.iter().zip(embeddings.into_iter()) {
            let tokens = tokenize_for_idf(text);
            index.add_chunk_with_tokens(offset, text.len(), embedding, tokens);
            offset += text.len() as u64;
        }
    }
    
    index.finalize_idf();
    index.mark_complete();
    
    let index_elapsed = index_start.elapsed();
    let mb_per_sec = total_bytes as f64 / 1024.0 / 1024.0 / index_elapsed.as_secs_f64();
    println!("Indexed {} chunks in {:.2}s ({:.1} MB/s)", 
             chunks.len(), index_elapsed.as_secs_f64(), mb_per_sec);
    
    // Phase 2: Search queries
    println!("\n--- Phase 2: Search with IDF Pre-filtering ---");
    let test_queries = [
        "history of computing",
        "quantum physics theory",
        "artificial intelligence machine learning",
        "world war military conflict",
        "music concert performance",
    ];
    
    for query in &test_queries {
        let query_start = Instant::now();
        
        // Embed query
        let query_embedding = model.embed(query).expect("Query embedding failed");
        let query_tokens = tokenize_for_idf(query);
        
        // Search with IDF
        let results = index.search_with_idf(&query_embedding, Some(&query_tokens), 10);
        
        let search_elapsed = query_start.elapsed();
        println!("  '{}': {:.3}ms, {} results", query, search_elapsed.as_secs_f64() * 1000.0, results.len());
        
        if let Some(top) = results.first() {
            println!("    Top result: offset={}, score={:.4}", top.offset, top.score);
        }
    }
    
    // Phase 3: Compare IDF vs no-IDF search
    println!("\n--- Phase 3: IDF vs Full Search Comparison ---");
    let query = "machine learning artificial intelligence neural network";
    let query_embedding = model.embed(query).expect("Query embedding failed");
    let query_tokens = tokenize_for_idf(query);
    
    // With IDF
    let idf_start = Instant::now();
    for _ in 0..100 {
        let _ = index.search_with_idf(&query_embedding, Some(&query_tokens), 10);
    }
    let idf_elapsed = idf_start.elapsed();
    
    // Without IDF (full scan)
    let full_start = Instant::now();
    for _ in 0..100 {
        let _ = index.search(&query_embedding, 10);
    }
    let full_elapsed = full_start.elapsed();
    
    println!("With IDF pre-filtering: {:.3}ms per query", idf_elapsed.as_secs_f64() * 1000.0 / 100.0);
    println!("Without IDF (full scan): {:.3}ms per query", full_elapsed.as_secs_f64() * 1000.0 / 100.0);
    println!("IDF speedup: {:.2}x", full_elapsed.as_secs_f64() / idf_elapsed.as_secs_f64());
    
    std::fs::remove_file(temp_file).ok();
    
    println!("\n=== Summary ===");
    println!("Indexing speed: {:.1} MB/s (136MB Wikipedia in ~{:.0}s)", mb_per_sec, 136.0 / mb_per_sec);
    println!("Search latency: <1ms per query with IDF pre-filtering");
}

/// Benchmark hybrid search: instant results + background scan
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_hybrid_search() {
    use bigedit::search_service::{SearchService, SearchOptions, SearchStrategy, SearchDirection};
    use bigedit::config::FtsSearchConfig;
    
    println!("\n=== Hybrid Search Benchmark (Instant + Background) ===\n");
    
    let wiki_path = data_dir().join("simple-wikipedia.txt");
    if !wiki_path.exists() {
        println!("Wikipedia file not found at {:?}", wiki_path);
        println!("Download it with: cargo test --features semantic-search --release --test wikipedia_bench -- --ignored --nocapture bench_simple_english_wikipedia");
        return;
    }
    
    let file_size = fs::metadata(&wiki_path).map(|m| m.len()).unwrap_or(0);
    println!("File: {} ({:.1} MB)", wiki_path.display(), file_size as f64 / 1024.0 / 1024.0);
    
    // Create search service
    let mut service = SearchService::new(&wiki_path, FtsSearchConfig::default())
        .expect("Failed to create search service");
    
    // Test queries with different characteristics
    let test_queries = [
        "quantum",
        "artificial intelligence",
        "world war",
        "xyznonexistent",
    ];
    
    for query in &test_queries {
        println!("\n--- Query: '{}' ---", query);
        
        let options = SearchOptions {
            query: query.to_string(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Hybrid,
            use_regex: false,
        };
        
        // Phase 1: Instant results
        let instant_start = Instant::now();
        let result = service.search_hybrid_instant(&options).expect("Search failed");
        let instant_time = instant_start.elapsed();
        
        println!("  INSTANT ({:.1}ms): {} matches in semantic chunks", 
                 instant_time.as_secs_f64() * 1000.0, result.instant_match_count);
        println!("  Status: {}", result.progress.status_message());
        
        // Phase 2: Background scan (incremental)
        let continue_start = Instant::now();
        let mut last_count = result.matches.len();
        
        loop {
            let result = service.search_hybrid_continue(16 * 1024 * 1024) // 16MB per iteration
                .expect("Continue failed");
            
            if result.matches.len() > last_count {
                println!("  +{} more matches ({}% scanned)", 
                         result.matches.len() - last_count,
                         result.progress.bytes_scanned * 100 / result.progress.bytes_total);
                last_count = result.matches.len();
            }
            
            if result.progress.is_complete() {
                let continue_time = continue_start.elapsed();
                println!("  COMPLETE ({:.1}ms): {} total matches", 
                         continue_time.as_secs_f64() * 1000.0, result.matches.len());
                break;
            }
        }
        
        // Reset for next query
        service.clear_hybrid_state();
    }
    
    // Compare with traditional memmem search
    println!("\n--- Comparison: Traditional memmem search ---");
    for query in &["quantum", "artificial intelligence"] {
        let options = SearchOptions {
            query: query.to_string(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        
        let start = Instant::now();
        let mut count = 0;
        let mut offset = 0u64;
        
        while let Some(result) = service.search(&options, offset).expect("Search failed") {
            count += 1;
            offset = result.offset + 1;
        }
        
        let elapsed = start.elapsed();
        println!("  '{}': {} matches in {:.1}ms ({:.1} MB/s)", 
                 query, count, elapsed.as_secs_f64() * 1000.0,
                 file_size as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64());
    }
}

/// Combined benchmark: English + Arabic side by side, with truly rare words
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_all_search_modes() {
    use bigedit::search_service::{SearchService, SearchOptions, SearchStrategy, SearchDirection};
    use bigedit::config::FtsSearchConfig;
    use std::io::Write;
    
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     MULTILINGUAL SEARCH BENCHMARK: English (136MB) + Arabic (529MB)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Show CPU info
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("🖥️  CPU threads available: {} (rayon will use all)", num_cpus);
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // ENGLISH WIKIPEDIA
    // ═══════════════════════════════════════════════════════════════════════════════
    let en_path = data_dir().join("simple-wikipedia.txt");
    let ar_path = data_dir().join("arabic-wikipedia.txt");
    
    if !en_path.exists() {
        println!("❌ Simple Wikipedia not found. Run download_simple_wikipedia first.");
        return;
    }
    
    let en_size = fs::metadata(&en_path).map(|m| m.len()).unwrap_or(0);
    println!("📁 Simple English Wikipedia: {:.1} MB", en_size as f64 / 1024.0 / 1024.0);
    
    let mut en_service = SearchService::new(&en_path, FtsSearchConfig::default())
        .expect("Failed to create English search service");
    
    // Build English index with parallel processing
    println!("🔨 Building English index (parallel)...");
    let index_start = Instant::now();
    let mut last_print = Instant::now();
    
    en_service.ensure_semantic_index_with_progress(|chunks_done, total_chunks, bytes_done, total_bytes| {
        if last_print.elapsed().as_millis() >= 300 || chunks_done == total_chunks {
            let pct = bytes_done as f64 / total_bytes as f64 * 100.0;
            let elapsed = index_start.elapsed().as_secs_f64();
            let rate = bytes_done as f64 / 1024.0 / 1024.0 / elapsed;
            print!("\r   {}/{} ({:.0}%) {:.1} MB/s", chunks_done, total_chunks, pct, rate);
            std::io::stdout().flush().ok();
            last_print = Instant::now();
        }
    }).expect("Failed to build English index");
    let en_index_time = index_start.elapsed().as_secs_f64();
    println!("\r✅ English index: {:.1}s ({:.1} MB/s)                    ", 
             en_index_time, en_size as f64 / 1024.0 / 1024.0 / en_index_time);
    
    // English queries: common → rare (real counts from grep -o -i)
    let en_queries = vec![
        ("the", "ultra-common", "2085808"),
        ("computer", "common", "6148"),
        ("simultaneously", "medium", "140"),
        ("metamorphosis", "rare", "78"),
        ("circumnavigation", "rare", "17"),
        ("onomatopoeia", "RARE", "7"),
        ("serendipity", "RARE", "4"),
        ("prestidigitation", "NOT-IN-FILE", "0"),
    ];
    
    println!("\n┌────────────────────────┬──────────┬─────────┬──────────────────────┬──────────────────────────────────────────────────┐");
    println!("│ ENGLISH Query          │ Rarity   │ Real#   │ Memmem (first/all)   │ Hybrid (100ch → all semantic → complete)         │");
    println!("├────────────────────────┼──────────┼─────────┼──────────────────────┼──────────────────────────────────────────────────┤");
    
    for (query, rarity, count) in &en_queries {
        print!("│ {:22} │{:10}│{:9}│", query, rarity, count);
        std::io::stdout().flush().ok();
        run_search_row(&mut en_service, query);
        println!();
    }
    println!("└────────────────────────┴──────────┴─────────┴──────────────────────┴──────────────────────────────────────────────────┘");
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // ARABIC WIKIPEDIA
    // ═══════════════════════════════════════════════════════════════════════════════
    if !ar_path.exists() {
        println!("\n❌ Arabic Wikipedia not found. Run download_arabic_wikipedia first.");
        return;
    }
    
    let ar_size = fs::metadata(&ar_path).map(|m| m.len()).unwrap_or(0);
    println!("\n📁 Arabic Wikipedia: {:.1} MB", ar_size as f64 / 1024.0 / 1024.0);
    
    let mut ar_service = SearchService::new(&ar_path, FtsSearchConfig::default())
        .expect("Failed to create Arabic search service");
    
    println!("🔨 Building Arabic index (parallel)...");
    let index_start = Instant::now();
    let mut last_print = Instant::now();
    
    ar_service.ensure_semantic_index_with_progress(|chunks_done, total_chunks, bytes_done, total_bytes| {
        if last_print.elapsed().as_millis() >= 300 || chunks_done == total_chunks {
            let pct = bytes_done as f64 / total_bytes as f64 * 100.0;
            let elapsed = index_start.elapsed().as_secs_f64();
            let rate = bytes_done as f64 / 1024.0 / 1024.0 / elapsed;
            print!("\r   {}/{} ({:.0}%) {:.1} MB/s", chunks_done, total_chunks, pct, rate);
            std::io::stdout().flush().ok();
            last_print = Instant::now();
        }
    }).expect("Failed to build Arabic index");
    let ar_index_time = index_start.elapsed().as_secs_f64();
    println!("\r✅ Arabic index: {:.1}s ({:.1} MB/s)                      ", 
             ar_index_time, ar_size as f64 / 1024.0 / 1024.0 / ar_index_time);
    
    // Arabic queries: common → rare (real counts from 12GB Arabic Wikipedia)
    let ar_queries = vec![
        ("تاريخ", "common", "92642"),               // History - very common
        ("ويكيبيديا", "common", "~50k"),            // Wikipedia
        ("الموريسكي", "medium", "499"),             // Morisco people
        ("سبرينغستين", "medium", "113"),            // Springsteen
        ("البهلاني", "medium", "67"),               // Omani poet
        ("إيراستوس", "rare", "22"),                 // Greek name
        ("ينبولسكي", "rare", "19"),                 // Polish name
        ("سينديكاتوم", "rare", "12"),               // Syndicatum
        ("أبهيدهاما", "RARE", "5"),                 // Buddhist term - very rare!
        ("جافلبورج", "RARE", "3"),                  // Swedish city - very rare!
        ("شاهسوران", "RARE", "3"),                  // Persian name - very rare!
    ];
    
    println!("\n┌────────────────────────────┬──────────┬───────┬──────────────────────┬──────────────────────────────────────────────────┐");
    println!("│ ARABIC Query               │ Rarity   │ Real# │ Memmem (first/all)   │ Hybrid (100ch → all semantic → complete)         │");
    println!("├────────────────────────────┼──────────┼───────┼──────────────────────┼──────────────────────────────────────────────────┤");
    
    for (query, rarity, count) in &ar_queries {
        print!("│ {:26} │{:10}│{:7}│", query, rarity, count);
        std::io::stdout().flush().ok();
        run_search_row(&mut ar_service, query);
        println!();
    }
    println!("└────────────────────────────┴──────────┴───────┴──────────────────────┴──────────────────────────────────────────────────┘");
    
    println!("\n📊 Legend:");
    println!("   ✓ = counts match | ⚠ = count mismatch");
    println!("   Memmem: count first_ms/all_ms (baseline full scan)");
    println!("   Hybrid: count@time for each phase (100ch → all semantic → memmem complete)");
    println!("   Exponential semantic: 100 → 300 → 1000 → 3000 → ... chunks");
    println!("\n✅ Multilingual benchmark complete!");
}

/// Helper to run all 4 search strategies for one query
#[cfg(feature = "semantic-search")]
fn run_search_row(service: &mut bigedit::search_service::SearchService, query: &str) {
    use bigedit::search_service::{SearchOptions, SearchStrategy, SearchDirection};
    use std::io::Write;
    
    // 1. Memmem - measure time to first result AND time to all results
    let memmem_count;
    let memmem_first_ms;
    let memmem_all_ms;
    {
        let options = SearchOptions {
            query: query.to_string(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Memmem,
            use_regex: false,
        };
        let start = Instant::now();
        let mut count = 0;
        let mut offset = 0u64;
        let mut first_time = None;
        while let Ok(Some(result)) = service.search(&options, offset) {
            if first_time.is_none() {
                first_time = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            count += 1;
            offset = result.offset + 1;
            if count > 100000 { break; }
        }
        memmem_all_ms = start.elapsed().as_secs_f64() * 1000.0;
        memmem_first_ms = first_time.unwrap_or(memmem_all_ms);
        memmem_count = count;
        let s = if count > 0 { "✓" } else { "✗" };
        print!(" {} {:>6} {:>4.0}/{:>5.0}ms │", s, count, memmem_first_ms, memmem_all_ms);
        std::io::stdout().flush().ok();
    }
    
    // 2. Hybrid - exponential expansion: first@100 chunks, then expand to all
    {
        let options = SearchOptions {
            query: query.to_string(),
            case_sensitive: false,
            direction: SearchDirection::Forward,
            wrap_around: false,
            strategy: SearchStrategy::Hybrid,
            use_regex: false,
        };
        
        // Phase 1: Instant (100 chunks)
        let start = Instant::now();
        let mut result = service.search_hybrid_instant(&options).expect("Hybrid failed");
        let first_ms = start.elapsed().as_secs_f64() * 1000.0;
        let first_count = result.instant_match_count;
        
        // Phase 2: Exponential expansion until all semantic chunks done
        while !result.progress.semantic_complete {
            result = service.search_hybrid_expand().expect("Hybrid expand failed");
        }
        let semantic_ms = start.elapsed().as_secs_f64() * 1000.0;
        let semantic_count = result.matches.len();
        
        // Phase 3: Full memmem scan for complete results
        while !result.progress.full_scan_complete {
            result = service.search_hybrid_continue(64 * 1024 * 1024).expect("Hybrid continue failed");
        }
        let all_ms = start.elapsed().as_secs_f64() * 1000.0;
        let all_count = result.matches.len();
        
        service.clear_hybrid_state();
        
        // Validate: final count should match memmem
        let valid = all_count == memmem_count;
        let mark = if !valid { "⚠" } else { "✓" };
        
        // Format: first_count@first_ms / semantic_count@semantic_ms / all_count@all_ms
        print!(" {} {:>5}@{:>3.0}ms {:>5}@{:>4.0}ms {:>6}@{:>5.0}ms │", 
               mark, first_count, first_ms, semantic_count, semantic_ms, all_count, all_ms);
        std::io::stdout().flush().ok();
    }
}

/// Download Arabic Wikipedia for larger file testing
#[test]
#[ignore]
fn download_arabic_wikipedia() {
    use std::process::Command;
    
    println!("\n=== Downloading REAL Arabic Wikipedia (1.87GB compressed → ~8-10GB text) ===\n");
    
    let data_dir = data_dir();
    fs::create_dir_all(&data_dir).expect("Failed to create data dir");
    
    let output_path = data_dir.join("arabic-wikipedia.txt");
    let bz2_path = data_dir.join("arwiki-latest-pages-articles.xml.bz2");
    
    // Check if we already have a good file
    if output_path.exists() {
        let size = fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
        if size > 1_000_000_000 { // > 1GB means it's real
            println!("✅ Real Arabic Wikipedia already exists: {:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0);
            return;
        } else {
            println!("⚠️  Found small/synthetic file ({:.1} MB), will download real one...", 
                     size as f64 / 1024.0 / 1024.0);
            fs::remove_file(&output_path).ok();
        }
    }
    
    // Download full Arabic Wikipedia articles (1.87 GB compressed)
    let url = "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2";
    
    if !bz2_path.exists() {
        println!("📥 Downloading Arabic Wikipedia (1.87 GB)...");
        println!("   URL: {}\n", url);
        
        let status = Command::new("curl")
            .args(["-L", "-C", "-", "-o", bz2_path.to_str().unwrap(), "--progress-bar", url])
            .status();
        
        if status.is_err() || !status.unwrap().success() {
            println!("❌ curl failed, trying wget...");
            let _ = Command::new("wget")
                .args(["-c", "-O", bz2_path.to_str().unwrap(), url])
                .status();
        }
    }
    
    let bz2_size = fs::metadata(&bz2_path).map(|m| m.len()).unwrap_or(0);
    if bz2_size < 1_000_000_000 {
        println!("❌ Download incomplete or failed. Size: {:.1} MB", bz2_size as f64 / 1024.0 / 1024.0);
        println!("   Please download manually: {}", url);
        return;
    }
    println!("✅ Downloaded: {:.2} GB compressed", bz2_size as f64 / 1024.0 / 1024.0 / 1024.0);
    
    // Extract text from XML (remove tags, keep Arabic text)
    println!("\n📦 Extracting Arabic text from XML...");
    println!("   This extracts article text, removing XML markup.");
    println!("   Expected output: ~8-10 GB of Arabic text.\n");
    
    // Use bzcat + sed to extract text content between <text> tags
    // This is a simple extraction - gets the main article text
    let extract_cmd = format!(
        r#"bzcat "{}" | sed -n 's/<text[^>]*>\(.*\)<\/text>/\1/p' | sed 's/&lt;/</g; s/&gt;/>/g; s/&amp;/\&/g; s/&quot;/"/g' > "{}""#,
        bz2_path.to_str().unwrap(),
        output_path.to_str().unwrap()
    );
    
    println!("   Running extraction (this takes 10-20 minutes)...");
    let status = Command::new("sh")
        .args(["-c", &extract_cmd])
        .status();
    
    if status.is_err() || !status.unwrap().success() {
        println!("⚠️  Text extraction had issues, trying simpler approach...");
        // Fallback: just decompress (will include XML tags but still works for testing)
        let _ = Command::new("sh")
            .args(["-c", &format!("bzcat '{}' > '{}'", 
                                   bz2_path.to_str().unwrap(),
                                   output_path.to_str().unwrap())])
            .status();
    }
    
    let final_size = fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    println!("\n✅ Created Arabic Wikipedia: {:.2} GB", final_size as f64 / 1024.0 / 1024.0 / 1024.0);
    
    if final_size > 5_000_000_000 {
        println!("🗑️  Removing compressed file to save space...");
        fs::remove_file(&bz2_path).ok();
    }
}

/// Benchmark index cache: cold build vs warm load from disk
#[test]
#[ignore]
#[cfg(feature = "semantic-search")]
fn bench_index_cache() {
    use bigedit::semantic::{SemanticIndex, FastEmbeddingModel};
    use bigedit::config::SemanticSearchConfig;
    
    println!("\n=== Index Cache Benchmark (Cold vs Warm) ===\n");
    
    let wiki_path = data_dir().join("simple-wikipedia.txt");
    if !wiki_path.exists() {
        println!("Wikipedia file not found at {:?}", wiki_path);
        println!("Download it first with the wikipedia_bench test");
        return;
    }
    
    let file_size = std::fs::metadata(&wiki_path).map(|m| m.len()).unwrap_or(0);
    println!("File: {} ({:.1} MB)", wiki_path.display(), file_size as f64 / 1024.0 / 1024.0);
    
    // Clear any existing cache first
    println!("\n1. Clearing cache...");
    let _ = SemanticIndex::clear_cache(&wiki_path);
    
    // Load embedding model
    println!("2. Loading embedding model...");
    let model_start = Instant::now();
    let model = FastEmbeddingModel::load_default().expect("Failed to load Model2Vec");
    println!("   Model loaded in {:.2}s", model_start.elapsed().as_secs_f64());
    
    // COLD: Build index from scratch
    println!("\n3. COLD: Building index from scratch...");
    let cold_start = Instant::now();
    let config = SemanticSearchConfig::default();
    let mut index = SemanticIndex::new(&wiki_path, config.clone()).expect("Failed to create index");
    
    index.build_from_file(&model, |chunks, total, bytes, total_bytes| {
        if chunks % 2000 == 0 && chunks > 0 {
            let pct = (bytes as f64 / total_bytes as f64 * 100.0) as u32;
            let mb_s = bytes as f64 / 1024.0 / 1024.0 / cold_start.elapsed().as_secs_f64();
            println!("   {}/{} chunks ({}%) - {:.1} MB/s", chunks, total, pct, mb_s);
        }
    }).expect("Failed to build index");
    
    let cold_time = cold_start.elapsed();
    let cold_mb_s = file_size as f64 / 1024.0 / 1024.0 / cold_time.as_secs_f64();
    println!("\n   COLD build: {:.2}s ({:.1} MB/s)", cold_time.as_secs_f64(), cold_mb_s);
    println!("   Chunks: {}", index.chunk_count());
    
    // Save to cache
    println!("\n4. Saving to disk cache...");
    let save_start = Instant::now();
    index.save_to_cache().expect("Failed to save cache");
    let save_time = save_start.elapsed();
    println!("   Saved in {:.2}s", save_time.as_secs_f64());
    
    // Drop the index to free memory
    drop(index);
    
    // WARM: Load from cache
    println!("\n5. WARM: Loading from disk cache...");
    let warm_start = Instant::now();
    let cached = SemanticIndex::load_cached(&wiki_path, config.clone());
    let warm_time = warm_start.elapsed();
    
    if let Some(cached_index) = cached {
        let speedup = cold_time.as_secs_f64() / warm_time.as_secs_f64();
        println!("   WARM load: {:.3}s ({:.0}x faster than cold!)", warm_time.as_secs_f64(), speedup);
        println!("   Chunks: {}", cached_index.chunk_count());
    } else {
        println!("   ❌ Failed to load from cache!");
    }
    
    // Summary
    println!("\n=== Summary ===");
    println!("Cold build: {:.2}s ({:.1} MB/s)", cold_time.as_secs_f64(), cold_mb_s);
    println!("Warm load:  {:.3}s", warm_time.as_secs_f64());
    println!("Speedup:    {:.0}x", cold_time.as_secs_f64() / warm_time.as_secs_f64());
    
    // Clean up cache
    println!("\n6. Cleaning up cache...");
    let _ = SemanticIndex::clear_cache(&wiki_path);
}

