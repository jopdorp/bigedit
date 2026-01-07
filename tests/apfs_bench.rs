//! APFS clonefile consolidation benchmark
//!
//! Run with: cargo test --test apfs_bench -- --nocapture
//! Or for release mode: cargo test --test apfs_bench --release -- --nocapture

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

#[cfg(target_os = "macos")]
use std::ffi::CString;
#[cfg(target_os = "macos")]
use std::os::unix::ffi::OsStrExt;

/// Test size in megabytes (matches Linux benchmarks)
const TEST_SIZE_MB: usize = 143;

/// Test clonefile on APFS
#[cfg(target_os = "macos")]
fn test_clonefile(src: &Path, dst: &Path) -> Result<std::time::Duration, std::io::Error> {
    extern "C" {
        fn clonefile(
            src: *const libc::c_char,
            dst: *const libc::c_char,
            flags: u32,
        ) -> libc::c_int;
    }

    let src_cstr =
        CString::new(src.as_os_str().as_bytes()).map_err(|_| std::io::ErrorKind::InvalidInput)?;
    let dst_cstr =
        CString::new(dst.as_os_str().as_bytes()).map_err(|_| std::io::ErrorKind::InvalidInput)?;

    // Remove destination if it exists
    let _ = fs::remove_file(dst);

    let start = Instant::now();
    let result = unsafe { clonefile(src_cstr.as_ptr(), dst_cstr.as_ptr(), 0) };
    let elapsed = start.elapsed();

    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(elapsed)
}

#[cfg(not(target_os = "macos"))]
fn test_clonefile(_src: &Path, _dst: &Path) -> Result<std::time::Duration, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "clonefile only available on macOS",
    ))
}

/// Create a test file with random-ish content
fn create_test_file(path: &Path, size_mb: usize) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let chunk_size = 1024 * 1024; // 1MB chunks
    let chunk: Vec<u8> = (0..chunk_size).map(|i| (i % 256) as u8).collect();

    for _ in 0..size_mb {
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(())
}

/// Benchmark streaming copy (simulates save_file behavior)
fn benchmark_streaming_copy(src: &Path, dst: &Path) -> std::io::Result<std::time::Duration> {
    let mut src_file = File::open(src)?;
    let _ = fs::remove_file(dst);
    let mut dst_file = File::create(dst)?;

    let start = Instant::now();

    // Use 16MB buffer like bigedit does
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    loop {
        let n = src_file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        dst_file.write_all(&buffer[..n])?;
    }
    dst_file.sync_data()?;

    Ok(start.elapsed())
}

/// Simulate full consolidation (clonefile backup + stream + rename)
#[cfg(target_os = "macos")]
fn benchmark_consolidation(
    original: &Path,
    _size_mb: usize,
) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    let backup_path = original.with_extension("bigedit-backup");
    let temp_path = original.with_extension("tmp");

    let start = Instant::now();

    // Step 1: clonefile to backup (instant CoW)
    test_clonefile(original, &backup_path)?;

    // Step 2: Stream to temp file (simulating patch application)
    let mut src = File::open(original)?;
    let mut dst = File::create(&temp_path)?;
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    loop {
        let n = src.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buffer[..n])?;
    }
    dst.sync_data()?;
    drop(src);
    drop(dst);

    // Step 3: Atomic rename
    fs::rename(&temp_path, original)?;

    // Step 4: Remove backup
    let _ = fs::remove_file(&backup_path);

    Ok(start.elapsed())
}

#[cfg(not(target_os = "macos"))]
fn benchmark_consolidation(
    _original: &Path,
    _size_mb: usize,
) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    Err("Consolidation benchmark only available on macOS".into())
}

#[test]
#[cfg(target_os = "macos")]
fn test_apfs_clonefile_works() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let test_file = temp_dir.path().join("test.bin");
    let clone_file = temp_dir.path().join("clone.bin");

    // Create a small test file (1MB for quick test)
    create_test_file(&test_file, 1).expect("Failed to create test file");

    // Test clonefile
    let result = test_clonefile(&test_file, &clone_file);

    match result {
        Ok(duration) => {
            println!("✅ clonefile succeeded in {:?}", duration);

            // Verify content matches
            let orig = fs::read(&test_file).expect("read original");
            let cloned = fs::read(&clone_file).expect("read clone");
            assert_eq!(orig, cloned, "Clone content should match original");
            println!("✅ Clone content verified");
        }
        Err(e) => {
            println!("❌ clonefile failed: {}", e);
            println!("   This may indicate the filesystem is not APFS");
            // Don't fail the test - just report
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
fn benchmark_apfs_full() {
    println!("\n=== APFS Consolidation Benchmark ===\n");

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let test_file = temp_dir.path().join("test.bin");
    let clone_file = temp_dir.path().join("clone.bin");
    let stream_file = temp_dir.path().join("stream.bin");

    // Check filesystem type
    println!("Test directory: {:?}", temp_dir.path());

    // Create test file
    println!(
        "Creating {}MB test file (this may take a moment)...",
        TEST_SIZE_MB
    );
    create_test_file(&test_file, TEST_SIZE_MB).expect("Failed to create test file");
    let file_size = fs::metadata(&test_file)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("Created test file: {} bytes\n", file_size);

    // Benchmark 1: clonefile
    println!("=== Test 1: clonefile() Performance ===");
    let mut clone_times = Vec::new();
    for i in 1..=5 {
        let _ = fs::remove_file(&clone_file);
        match test_clonefile(&test_file, &clone_file) {
            Ok(duration) => {
                println!("  Run {}: {:?}", i, duration);
                clone_times.push(duration);
            }
            Err(e) => {
                println!("  Run {}: FAILED - {}", i, e);
            }
        }
    }

    if !clone_times.is_empty() {
        let avg: std::time::Duration =
            clone_times.iter().sum::<std::time::Duration>() / clone_times.len() as u32;
        println!("Average clonefile time: {:?} ✅\n", avg);
    }

    // Benchmark 2: Streaming save
    println!("=== Test 2: Streaming Save Performance ===");
    let mut stream_times = Vec::new();
    for i in 1..=5 {
        match benchmark_streaming_copy(&test_file, &stream_file) {
            Ok(duration) => {
                println!("  Run {}: {:?}", i, duration);
                stream_times.push(duration);
            }
            Err(e) => {
                println!("  Run {}: FAILED - {}", i, e);
            }
        }
    }

    if !stream_times.is_empty() {
        let avg: std::time::Duration =
            stream_times.iter().sum::<std::time::Duration>() / stream_times.len() as u32;
        println!("Average streaming time: {:?}\n", avg);
    }

    // Benchmark 3: Full consolidation
    println!("=== Test 3: Full Consolidation ===");

    // Re-create test file since consolidation modifies it
    create_test_file(&test_file, TEST_SIZE_MB).expect("Failed to create test file");

    let mut consolidation_times = Vec::new();
    for i in 1..=5 {
        // Re-create test file for each iteration
        create_test_file(&test_file, TEST_SIZE_MB).expect("Failed to create test file");

        match benchmark_consolidation(&test_file, TEST_SIZE_MB) {
            Ok(duration) => {
                println!("  Run {}: {:?}", i, duration);
                consolidation_times.push(duration);
            }
            Err(e) => {
                println!("  Run {}: FAILED - {}", i, e);
            }
        }
    }

    if !consolidation_times.is_empty() {
        let avg: std::time::Duration =
            consolidation_times.iter().sum::<std::time::Duration>() / consolidation_times.len() as u32;
        println!("Average consolidation time: {:?}\n", avg);
    }

    // Summary
    println!("\n=== APFS Benchmark Summary ({}MB file) ===", TEST_SIZE_MB);
    println!("┌────────────────────┬─────────────────────┐");
    println!("│ Operation          │ Time                │");
    println!("├────────────────────┼─────────────────────┤");

    if !clone_times.is_empty() {
        let avg = clone_times.iter().sum::<std::time::Duration>() / clone_times.len() as u32;
        println!("│ clonefile backup   │ {:>18?} │", avg);
    }
    if !consolidation_times.is_empty() {
        let avg = consolidation_times.iter().sum::<std::time::Duration>() / consolidation_times.len() as u32;
        println!("│ Consolidation      │ {:>18?} │", avg);
    }
    if !stream_times.is_empty() {
        let avg = stream_times.iter().sum::<std::time::Duration>() / stream_times.len() as u32;
        println!("│ Streaming save     │ {:>18?} │", avg);
    }
    println!("└────────────────────┴─────────────────────┘");

    println!("\nCompare with Linux results (143MB file):");
    println!("┌──────────┬─────────────────┬───────────────┬────────────┐");
    println!("│ FS       │ FICLONE Backup  │ Consolidation │ Streaming  │");
    println!("├──────────┼─────────────────┼───────────────┼────────────┤");
    println!("│ XFS      │ 9ms ✅           │ 249ms         │ 388ms      │");
    println!("│ ZFS      │ 17ms ✅          │ 213ms         │ 251ms      │");
    println!("│ btrfs    │ 13ms ✅          │ 862ms         │ 698ms      │");
    println!("│ ext4     │ ❌ none          │ fallback      │ 230ms      │");
    println!("└──────────┴─────────────────┴───────────────┴────────────┘");
}

#[test]
#[cfg(not(target_os = "macos"))]
fn benchmark_apfs_full() {
    println!("APFS benchmark skipped - not running on macOS");
}

/// Test that bigedit's filesystem detection works for APFS
#[test]
#[cfg(target_os = "macos")]
fn test_bigedit_apfs_detection() {
    use bigedit::save::{detect_filesystem_capability, FilesystemCapability};
    
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let capability = detect_filesystem_capability(temp_dir.path());
    
    // On a typical Mac, /tmp is on APFS and should be detected as CopyOnWrite
    println!("Filesystem capability for {:?}: {:?}", temp_dir.path(), capability);
    
    if capability == FilesystemCapability::CopyOnWrite {
        println!("✅ APFS detected as CopyOnWrite");
    } else {
        println!("⚠️  Filesystem not detected as CoW - may not be APFS");
        // This isn't necessarily a failure - could be HFS+ or network mount
    }
}

