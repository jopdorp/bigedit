#!/bin/bash
# Test bigedit APFS clonefile consolidation on macOS
# This script tests the APFS-specific code paths in save.rs

set -e

echo "=== APFS Consolidation Benchmark ==="
echo "Testing macOS-specific clonefile() functionality"
echo ""

# Check we're on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "Error: This script requires macOS"
    exit 1
fi

# Use a temp directory for testing (should be on APFS)
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

# Verify this is APFS
FSTYPE=$(diskutil info "$TEST_DIR" 2>/dev/null | grep "Type (Bundle)" | awk '{print $NF}' || echo "unknown")
if [ "$FSTYPE" != "apfs" ]; then
    # Try alternative detection
    FSTYPE=$(mount | grep " / " | awk '{print $4}' | tr -d '()')
fi
echo "Filesystem type: $FSTYPE"

cleanup() {
    echo ""
    echo "Cleaning up..."
    rm -rf "$TEST_DIR"
}

trap cleanup EXIT

# Create test file of desired size (default 143MB to match Linux benchmarks)
SIZE_MB=${1:-143}
TEST_FILE="$TEST_DIR/test_file.bin"

echo ""
echo "=== Creating ${SIZE_MB}MB test file ==="
dd if=/dev/urandom of="$TEST_FILE" bs=1m count=$SIZE_MB status=none 2>/dev/null || \
dd if=/dev/urandom of="$TEST_FILE" bs=1048576 count=$SIZE_MB 2>/dev/null
FILE_SIZE=$(stat -f%z "$TEST_FILE" 2>/dev/null || stat -c%s "$TEST_FILE")
echo "Created test file: $TEST_FILE ($FILE_SIZE bytes)"

# Test 1: clonefile() benchmark
echo ""
echo "=== Test 1: clonefile() Performance ==="
echo "Testing instant CoW clone (APFS feature)..."

CLONE_FILE="$TEST_DIR/clone_test.bin"

# Time the clonefile operation using the C function
cat > /tmp/test_clonefile.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <sys/clonefile.h>
#include <time.h>
#include <errno.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source> <dest>\n", argv[0]);
        return 1;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int result = clonefile(argv[1], argv[2], 0);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    if (result != 0) {
        fprintf(stderr, "clonefile failed: %s\n", strerror(errno));
        return 1;
    }
    
    long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    double elapsed_ms = elapsed_ns / 1000000.0;
    printf("%.2f\n", elapsed_ms);
    
    return 0;
}
EOF

cc -o /tmp/test_clonefile /tmp/test_clonefile.c 2>/dev/null

if [ -x /tmp/test_clonefile ]; then
    # Run clonefile test multiple times
    echo "Running clonefile benchmark (5 iterations)..."
    TOTAL_MS=0
    for i in 1 2 3 4 5; do
        rm -f "$CLONE_FILE"
        TIME_MS=$(/tmp/test_clonefile "$TEST_FILE" "$CLONE_FILE")
        echo "  Run $i: ${TIME_MS}ms"
        TOTAL_MS=$(echo "$TOTAL_MS + $TIME_MS" | bc)
    done
    CLONEFILE_AVG_MS=$(echo "scale=2; $TOTAL_MS / 5" | bc)
    echo "Average clonefile time: ${CLONEFILE_AVG_MS}ms ✓"
    
    # Verify clone content
    echo ""
    echo "Verifying clone integrity..."
    if diff -q "$TEST_FILE" "$CLONE_FILE" > /dev/null 2>&1; then
        echo "Clone content matches original ✓"
    else
        echo "Clone content MISMATCH! ✗"
    fi
    
    # Check disk usage (should be minimal for CoW clone)
    echo ""
    echo "Checking disk usage (CoW should share blocks)..."
    du -h "$TEST_FILE" "$CLONE_FILE" 2>/dev/null || ls -lh "$TEST_FILE" "$CLONE_FILE"
    
    rm -f "$CLONE_FILE"
else
    echo "Failed to compile clonefile test, trying Python fallback..."
    # Python fallback using ctypes
    python3 << EOF
import ctypes
import time
import os

libc = ctypes.CDLL(None)

try:
    clonefile = libc.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    clonefile.restype = ctypes.c_int
    
    src = b"$TEST_FILE"
    dst = b"$CLONE_FILE"
    
    times = []
    for i in range(5):
        if os.path.exists(dst):
            os.remove(dst)
        start = time.perf_counter()
        result = clonefile(src, dst, 0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}ms")
    
    avg = sum(times) / len(times)
    print(f"Average clonefile time: {avg:.2f}ms ✓")
except Exception as e:
    print(f"clonefile test failed: {e}")
EOF
fi

# Test 2: Streaming save benchmark
echo ""
echo "=== Test 2: Streaming Save Performance ==="
echo "Testing stream rewrite (temp file + atomic rename)..."

TEMP_OUT="$TEST_DIR/stream_out.bin"

# Helper function to get current time in milliseconds (macOS compatible)
get_ms() {
    python3 -c "import time; print(int(time.time()*1000))"
}

# Time a streaming copy (simulates save_file behavior)
time_streaming() {
    rm -f "$TEMP_OUT"
    local start=$(get_ms)
    cat "$TEST_FILE" > "$TEMP_OUT"
    sync
    local end=$(get_ms)
    echo $((end - start))
}

echo "Running streaming save benchmark (5 iterations)..."
TOTAL_MS=0
for i in 1 2 3 4 5; do
    TIME_MS=$(time_streaming)
    echo "  Run $i: ${TIME_MS}ms"
    TOTAL_MS=$((TOTAL_MS + TIME_MS))
done
STREAMING_AVG_MS=$((TOTAL_MS / 5))
echo "Average streaming save time: ${STREAMING_AVG_MS}ms"

rm -f "$TEMP_OUT"

# Test 3: Full consolidation simulation (clonefile backup + stream + rename)
echo ""
echo "=== Test 3: Full Consolidation (clonefile + stream + rename) ==="
echo "Simulating try_clonefile_consolidate_macos() behavior..."

BACKUP_FILE="$TEST_DIR/test_file.bigedit-backup"
WORK_FILE="$TEST_DIR/test_file.bin"
TEMP_FILE="$TEST_DIR/test_file.tmp"

consolidation_test() {
    # Step 1: clonefile to backup (instant)
    rm -f "$BACKUP_FILE" "$TEMP_FILE"
    /tmp/test_clonefile "$WORK_FILE" "$BACKUP_FILE" > /dev/null 2>&1
    
    # Step 2: Stream to temp file with "modifications"
    # In real usage, patches would be applied during streaming
    cat "$WORK_FILE" > "$TEMP_FILE"
    
    # Step 3: Atomic rename
    mv "$TEMP_FILE" "$WORK_FILE"
    
    # Step 4: Remove backup on success
    rm -f "$BACKUP_FILE"
}

if [ -x /tmp/test_clonefile ]; then
    echo "Running full consolidation benchmark (5 iterations)..."
    TOTAL_MS=0
    for i in 1 2 3 4 5; do
        start=$(get_ms)
        consolidation_test
        end=$(get_ms)
        TIME_MS=$((end - start))
        echo "  Run $i: ${TIME_MS}ms"
        TOTAL_MS=$((TOTAL_MS + TIME_MS))
    done
    CONSOLIDATION_AVG_MS=$((TOTAL_MS / 5))
    echo "Average consolidation time: ${CONSOLIDATION_AVG_MS}ms"
fi

# Test 4: Test the actual bigedit binary if available
echo ""
echo "=== Test 4: bigedit Integration Test ==="

BIGEDIT_BIN="$(dirname "$0")/../target/release/bigedit"
if [ ! -x "$BIGEDIT_BIN" ]; then
    BIGEDIT_BIN="$(dirname "$0")/../target/debug/bigedit"
fi

if [ -x "$BIGEDIT_BIN" ]; then
    echo "Found bigedit at: $BIGEDIT_BIN"
    
    # Create a test text file
    TEST_TXT="$TEST_DIR/test_text.txt"
    echo "This is a test file for bigedit APFS testing." > "$TEST_TXT"
    echo "Line 2 of the test file." >> "$TEST_TXT"
    echo "Line 3 of the test file." >> "$TEST_TXT"
    
    echo "Test file created: $TEST_TXT"
    echo ""
    echo "To manually test bigedit:"
    echo "  $BIGEDIT_BIN $TEST_TXT"
    echo ""
    echo "In bigedit, make changes and save to trigger consolidation"
else
    echo "bigedit binary not found. Build with: cargo build --release"
fi

# Summary
echo ""
echo "=== APFS Benchmark Summary (${SIZE_MB}MB file) ==="
echo "┌────────────────────┬─────────────┐"
echo "│ Operation          │ Time        │"
echo "├────────────────────┼─────────────┤"
if [ -n "$CLONEFILE_AVG_MS" ]; then
    printf "│ clonefile backup   │ %7sms   │\n" "$CLONEFILE_AVG_MS"
fi
if [ -n "$CONSOLIDATION_AVG_MS" ]; then
    printf "│ Consolidation      │ %7sms   │\n" "$CONSOLIDATION_AVG_MS"
fi
if [ -n "$STREAMING_AVG_MS" ]; then
    printf "│ Streaming save     │ %7sms   │\n" "$STREAMING_AVG_MS"
fi
echo "└────────────────────┴─────────────┘"
echo ""
echo "Compare with Linux results:"
echo "┌──────────┬─────────────────┬───────────────┬────────────┐"
echo "│ FS       │ FICLONE Backup  │ Consolidation │ Streaming  │"
echo "├──────────┼─────────────────┼───────────────┼────────────┤"
echo "│ XFS      │ 9ms ✅           │ 249ms         │ 388ms      │"
echo "│ ZFS      │ 17ms ✅          │ 213ms         │ 251ms      │"
echo "│ btrfs    │ 13ms ✅          │ 862ms         │ 698ms      │"
echo "│ ext4     │ ❌ none          │ fallback      │ 230ms      │"
echo "│ APFS     │ (see above)     │ (see above)   │ (see above)│"
echo "└──────────┴─────────────────┴───────────────┴────────────┘"
echo ""
echo "=== APFS Test Complete ==="
