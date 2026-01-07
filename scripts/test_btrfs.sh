#!/bin/bash
# Test bigedit consolidation on btrfs filesystem
# Requires: btrfs-progs, sudo access

set -e

IMG="/tmp/bigedit-btrfs-test.img"
MNT="/tmp/bigedit-btrfs-mnt"
SIZE_MB=100

cleanup() {
    echo "Cleaning up..."
    sudo umount "$MNT" 2>/dev/null || true
    rm -f "$IMG"
    rmdir "$MNT" 2>/dev/null || true
}

trap cleanup EXIT

echo "=== btrfs Consolidation Test ==="

# Check for btrfs-progs
if ! which mkfs.btrfs >/dev/null 2>&1; then
    echo "Error: mkfs.btrfs not found. Install with: sudo apt install btrfs-progs"
    exit 1
fi

# Create loopback image
echo "Creating ${SIZE_MB}MB btrfs image..."
dd if=/dev/zero of="$IMG" bs=1M count=$SIZE_MB status=none

# Format as btrfs
echo "Formatting as btrfs..."
mkfs.btrfs -q "$IMG"

# Mount
echo "Mounting at $MNT..."
mkdir -p "$MNT"
sudo mount -o loop "$IMG" "$MNT"
sudo chown "$USER:$USER" "$MNT"

# Verify filesystem type
FSTYPE=$(df -T "$MNT" | tail -1 | awk '{print $2}')
echo "Filesystem type: $FSTYPE"

if [ "$FSTYPE" != "btrfs" ]; then
    echo "Error: Expected btrfs, got $FSTYPE"
    exit 1
fi

# Create test file
TEST_FILE="$MNT/test.txt"
echo "Hello World from btrfs test!" > "$TEST_FILE"
echo "Created test file: $TEST_FILE"

# Run Rust test that uses the btrfs mount
echo ""
echo "Running btrfs-specific tests..."
cd "$(dirname "$0")/.."

# Create a simple test binary
cat > /tmp/btrfs_test.rs << 'EOF'
use std::path::Path;
use std::fs;

fn main() {
    let test_dir = std::env::args().nth(1).expect("Need test dir argument");
    let test_file = Path::new(&test_dir).join("consolidation_test.txt");
    
    // Create test content (10KB)
    let content: String = (0..10000).map(|i| ((i % 26) as u8 + b'a') as char).collect();
    fs::write(&test_file, &content).expect("Failed to write test file");
    
    println!("Created test file: {:?} ({} bytes)", test_file, content.len());
    
    // Check filesystem detection
    println!("\n--- Filesystem Detection ---");
    
    // Try to detect using statfs
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    
    let path_cstr = CString::new(test_file.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut statfs_buf: MaybeUninit<libc::statfs> = MaybeUninit::uninit();
        if libc::statfs(path_cstr.as_ptr(), statfs_buf.as_mut_ptr()) == 0 {
            let statfs = statfs_buf.assume_init();
            println!("statfs.f_type = 0x{:X}", statfs.f_type);
            let fstype = match statfs.f_type {
                0xEF53 => "ext4",
                0x58465342 => "xfs",
                0x9123683E => "btrfs",
                0x2FC12FC1 => "zfs",
                _ => "unknown",
            };
            println!("Detected filesystem: {}", fstype);
        }
    }
    
    // Test FICLONE
    println!("\n--- FICLONE (reflink) Test ---");
    let clone_file = Path::new(&test_dir).join("clone_test.txt");
    
    use std::os::unix::io::AsRawFd;
    let src = fs::File::open(&test_file).expect("Failed to open source");
    let dst = fs::File::create(&clone_file).expect("Failed to create dest");
    
    const FICLONE: libc::c_ulong = 0x40049409;
    
    let result = unsafe {
        libc::ioctl(dst.as_raw_fd(), FICLONE, src.as_raw_fd())
    };
    
    if result == 0 {
        println!("✓ FICLONE succeeded! Reflink created.");
        
        // Verify the clone has same content
        let cloned_content = fs::read_to_string(&clone_file).expect("Failed to read clone");
        if cloned_content == content {
            println!("✓ Clone content matches original");
        } else {
            println!("✗ Clone content mismatch!");
        }
        
        // Check disk usage (reflink should share blocks)
        let src_meta = fs::metadata(&test_file).unwrap();
        let dst_meta = fs::metadata(&clone_file).unwrap();
        println!("Source size: {} bytes", src_meta.len());
        println!("Clone size: {} bytes", dst_meta.len());
        
        // Modify the clone
        println!("\n--- Modify Clone (CoW) Test ---");
        use std::os::unix::fs::FileExt;
        let dst_write = fs::OpenOptions::new().write(true).open(&clone_file).unwrap();
        dst_write.write_at(b"MODIFIED", 0).expect("Failed to write");
        drop(dst_write);
        
        let modified = fs::read_to_string(&clone_file).unwrap();
        if modified.starts_with("MODIFIED") {
            println!("✓ Clone was modified (CoW worked)");
        }
        
        let original = fs::read_to_string(&test_file).unwrap();
        if original == content {
            println!("✓ Original unchanged (CoW isolation works)");
        }
        
    } else {
        let err = std::io::Error::last_os_error();
        println!("✗ FICLONE failed: {} (this is expected on non-CoW filesystems)", err);
    }
    
    // Cleanup
    fs::remove_file(&test_file).ok();
    fs::remove_file(&clone_file).ok();
    
    println!("\n=== btrfs test complete ===");
}
EOF

# Compile and run
echo "Compiling test..."
rustc /tmp/btrfs_test.rs -o /tmp/btrfs_test 2>&1 || {
    echo "Compilation failed, trying with explicit libc..."
    # Need to link libc properly for statfs
    rustc /tmp/btrfs_test.rs -o /tmp/btrfs_test --extern libc=$(find ~/.cargo -name 'liblibc*.rlib' | head -1) 2>&1 || {
        echo "Still failed - running cargo test instead"
        BTRFS_TEST_DIR="$MNT" cargo test test_btrfs --features btrfs-test 2>&1 || echo "No btrfs-test feature"
    }
}

if [ -x /tmp/btrfs_test ]; then
    echo "Running test..."
    /tmp/btrfs_test "$MNT"
fi

echo ""
echo "=== Manual verification ==="
echo "You can also manually test with:"
echo "  cd $MNT"
echo "  cargo run --manifest-path $(pwd)/Cargo.toml -- testfile.txt"
echo ""
echo "The mount will be cleaned up when this script exits."
echo "Press Enter to cleanup and exit..."
read
