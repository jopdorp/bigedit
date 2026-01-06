//! FUSE daemon for bigedit
//!
//! This daemon runs as a separate process to provide a FUSE mount that exposes
//! the patched view of a file. It reads patches from the journal file and
//! serves the patched content through FUSE.
//!
//! Usage: bigedit-fuse <original-file-path>
//!
//! The daemon will:
//! - Create a mount point at .<filename>.view/
//! - Expose the patched file content through FUSE
//! - Watch the journal file for updates
//! - Exit when the journal file is deleted or SIGTERM is received

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry,
    Request,
};

// Re-use types from the main crate
// Note: We need to duplicate some code here since this is a separate binary

/// File position type
type FilePos = u64;

/// A single patch (edit) to the file
#[derive(Debug, Clone)]
struct Patch {
    start: FilePos,
    end: FilePos,
    replacement: Vec<u8>,
}

/// List of patches
#[derive(Debug, Clone, Default)]
struct PatchList {
    patches: Vec<Patch>,
}

impl PatchList {
    fn new() -> Self {
        Self { patches: Vec::new() }
    }

    fn iter(&self) -> impl Iterator<Item = &Patch> {
        self.patches.iter()
    }
}

/// Journal file magic number
const JOURNAL_MAGIC: &[u8; 8] = b"BIGEDIT\0";
const JOURNAL_VERSION: u32 = 1;

/// Get journal path for a file
fn journal_path(file_path: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let filename = file_path.file_name().unwrap_or_default().to_string_lossy();
    parent.join(format!(".{}.bigedit-journal", filename))
}

/// Load patches from journal file
fn load_from_journal(file_path: &Path) -> Result<PatchList> {
    let journal = journal_path(file_path);
    
    if !journal.exists() {
        return Ok(PatchList::new());
    }

    let mut file = File::open(&journal).context("Failed to open journal")?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    
    if &magic != JOURNAL_MAGIC {
        anyhow::bail!("Invalid journal magic");
    }

    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    
    if version != JOURNAL_VERSION {
        anyhow::bail!("Unsupported journal version");
    }

    // Skip original file size
    let mut _size_bytes = [0u8; 8];
    file.read_exact(&mut _size_bytes)?;

    // Skip checksum
    let mut _checksum = [0u8; 4];
    file.read_exact(&mut _checksum)?;

    // Read patch count
    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes);

    let mut patches = PatchList::new();
    
    for _ in 0..count {
        let mut start_bytes = [0u8; 8];
        let mut end_bytes = [0u8; 8];
        let mut len_bytes = [0u8; 4];
        
        file.read_exact(&mut start_bytes)?;
        file.read_exact(&mut end_bytes)?;
        file.read_exact(&mut len_bytes)?;
        
        let start = u64::from_le_bytes(start_bytes);
        let end = u64::from_le_bytes(end_bytes);
        let len = u32::from_le_bytes(len_bytes) as usize;
        
        let mut replacement = vec![0u8; len];
        file.read_exact(&mut replacement)?;
        
        patches.patches.push(Patch { start, end, replacement });
    }

    Ok(patches)
}

/// Compute virtual file size after patches
fn compute_virtual_size(original_size: u64, patches: &PatchList) -> u64 {
    let mut size = original_size as i64;
    for patch in patches.iter() {
        let deleted = (patch.end - patch.start) as i64;
        let inserted = patch.replacement.len() as i64;
        size = size - deleted + inserted;
    }
    size.max(0) as u64
}

/// Read a range of the patched file
fn read_patched_range(
    file_path: &Path,
    file_size: u64,
    patches: &PatchList,
    offset: u64,
    size: usize,
) -> Result<Vec<u8>> {
    let mut file = File::open(file_path)?;
    let mut result = Vec::with_capacity(size);

    // Track position in virtual file and original file
    let mut virtual_pos: u64 = 0;
    let mut file_pos: u64 = 0;

    for patch in patches.iter() {
        // Read original bytes before this patch
        if patch.start > file_pos {
            let orig_len = patch.start - file_pos;

            if virtual_pos + orig_len > offset && virtual_pos < offset + size as u64 {
                let start_in_segment = if virtual_pos < offset {
                    (offset - virtual_pos).min(orig_len)
                } else {
                    0
                };

                let bytes_needed = (size - result.len()) as u64;
                let end_in_segment = orig_len.min(start_in_segment + bytes_needed);

                if start_in_segment < end_in_segment {
                    let read_start = file_pos + start_in_segment;
                    let read_len = (end_in_segment - start_in_segment) as usize;

                    file.seek(SeekFrom::Start(read_start))?;
                    let mut buf = vec![0u8; read_len];
                    let bytes_read = file.read(&mut buf)?;
                    buf.truncate(bytes_read);
                    result.extend_from_slice(&buf);

                    if result.len() >= size {
                        return Ok(result);
                    }
                }
            }

            virtual_pos += orig_len;
        }

        // Handle the patch replacement
        let patch_len = patch.replacement.len() as u64;
        if patch_len > 0 && virtual_pos < offset + size as u64 && virtual_pos + patch_len > offset {
            let start_in_patch = if virtual_pos < offset {
                (offset - virtual_pos) as usize
            } else {
                0
            };
            let end_in_patch = patch_len.min(offset + size as u64 - virtual_pos) as usize;

            if start_in_patch < end_in_patch {
                result.extend_from_slice(&patch.replacement[start_in_patch..end_in_patch]);
            }
        }

        virtual_pos += patch_len;
        file_pos = patch.end;
    }

    // Read remaining original bytes after all patches
    if file_pos < file_size && virtual_pos < offset + size as u64 {
        let remaining = file_size - file_pos;

        let start_in_segment = if virtual_pos < offset {
            (offset - virtual_pos).min(remaining)
        } else {
            0
        };

        let bytes_needed = (size - result.len()) as u64;
        let end_in_segment = remaining.min(start_in_segment + bytes_needed);

        if start_in_segment < end_in_segment {
            let read_start = file_pos + start_in_segment;
            let read_len = (end_in_segment - start_in_segment) as usize;

            file.seek(SeekFrom::Start(read_start))?;
            let mut buf = vec![0u8; read_len];
            let bytes_read = file.read(&mut buf)?;
            buf.truncate(bytes_read);
            result.extend_from_slice(&buf);
        }
    }

    Ok(result)
}

// FUSE constants
const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;
const FILE_INO: u64 = 2;

/// Shared state for the FUSE filesystem
struct FuseState {
    file_path: PathBuf,
    filename: String,
    file_size: u64,
    patches: PatchList,
    virtual_size: u64,
}

impl FuseState {
    fn new(file_path: &Path) -> Result<Self> {
        let file_path = file_path.canonicalize()?;
        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        let file_size = std::fs::metadata(&file_path)?.len();
        let patches = load_from_journal(&file_path).unwrap_or_default();
        let virtual_size = compute_virtual_size(file_size, &patches);

        Ok(Self {
            file_path,
            filename,
            file_size,
            patches,
            virtual_size,
        })
    }

    fn reload_patches(&mut self) -> Result<()> {
        self.patches = load_from_journal(&self.file_path)?;
        self.virtual_size = compute_virtual_size(self.file_size, &self.patches);
        Ok(())
    }
}

/// FUSE filesystem implementation
struct PatchedFileFS {
    state: Arc<Mutex<FuseState>>,
}

impl PatchedFileFS {
    fn new(state: Arc<Mutex<FuseState>>) -> Self {
        Self { state }
    }

    fn file_attr(&self) -> FileAttr {
        let state = self.state.lock().unwrap();
        FileAttr {
            ino: FILE_INO,
            size: state.virtual_size,
            blocks: (state.virtual_size + 511) / 512,
            atime: SystemTime::now(),
            mtime: SystemTime::now(),
            ctime: SystemTime::now(),
            crtime: UNIX_EPOCH,
            kind: FileType::RegularFile,
            perm: 0o444,
            nlink: 1,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            blksize: 4096,
            flags: 0,
        }
    }

    fn root_attr(&self) -> FileAttr {
        FileAttr {
            ino: ROOT_INO,
            size: 0,
            blocks: 0,
            atime: SystemTime::now(),
            mtime: SystemTime::now(),
            ctime: SystemTime::now(),
            crtime: UNIX_EPOCH,
            kind: FileType::Directory,
            perm: 0o555,
            nlink: 2,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            blksize: 4096,
            flags: 0,
        }
    }
}

impl Filesystem for PatchedFileFS {
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &std::ffi::OsStr, reply: ReplyEntry) {
        if parent != ROOT_INO {
            reply.error(libc::ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        if name.to_str() == Some(&state.filename) {
            drop(state);
            reply.entry(&TTL, &self.file_attr(), 0);
        } else {
            reply.error(libc::ENOENT);
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyAttr) {
        match ino {
            ROOT_INO => reply.attr(&TTL, &self.root_attr()),
            FILE_INO => {
                // Reload patches on each getattr to pick up changes
                if let Ok(mut state) = self.state.lock() {
                    let _ = state.reload_patches();
                }
                reply.attr(&TTL, &self.file_attr());
            }
            _ => reply.error(libc::ENOENT),
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        if ino != FILE_INO {
            reply.error(libc::ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        match read_patched_range(
            &state.file_path,
            state.file_size,
            &state.patches,
            offset as u64,
            size as usize,
        ) {
            Ok(data) => reply.data(&data),
            Err(_) => reply.error(libc::EIO),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        if ino != ROOT_INO {
            reply.error(libc::ENOENT);
            return;
        }

        let state = self.state.lock().unwrap();
        let entries = vec![
            (ROOT_INO, FileType::Directory, "."),
            (ROOT_INO, FileType::Directory, ".."),
            (FILE_INO, FileType::RegularFile, &state.filename),
        ];

        for (i, (ino, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            if reply.add(ino, (i + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: fuser::ReplyOpen) {
        if ino == FILE_INO {
            // Reload patches when file is opened
            if let Ok(mut state) = self.state.lock() {
                let _ = state.reload_patches();
            }
            reply.opened(0, 0);
        } else {
            reply.error(libc::ENOENT);
        }
    }
}

/// Store PID file for the daemon
fn write_pid_file(file_path: &Path) -> Result<PathBuf> {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let filename = file_path.file_name().unwrap_or_default().to_string_lossy();
    let pid_path = parent.join(format!(".{}.bigedit-fuse.pid", filename));
    
    std::fs::write(&pid_path, format!("{}", std::process::id()))?;
    Ok(pid_path)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: bigedit-fuse <file-path>");
        eprintln!();
        eprintln!("Starts a FUSE daemon that exposes the patched view of a file.");
        eprintln!("The mount point will be at .<filename>.view/");
        std::process::exit(1);
    }

    let file_path = PathBuf::from(&args[1]);
    
    if !file_path.exists() {
        eprintln!("Error: File does not exist: {}", file_path.display());
        std::process::exit(1);
    }

    let file_path = file_path.canonicalize()?;
    
    // Create state
    let state = FuseState::new(&file_path)?;
    let filename = state.filename.clone();
    let state = Arc::new(Mutex::new(state));

    // Create mount point
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let base_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    let mount_point = parent.join(format!(".{}.view", base_name));

    std::fs::create_dir_all(&mount_point)?;

    // Write PID file
    let pid_path = write_pid_file(&file_path)?;

    // Set up signal handler for cleanup
    let mount_point_clone = mount_point.clone();
    let pid_path_clone = pid_path.clone();
    ctrlc::set_handler(move || {
        let _ = std::fs::remove_file(&pid_path_clone);
        let _ = std::fs::remove_dir(&mount_point_clone);
        std::process::exit(0);
    }).ok();

    eprintln!("bigedit-fuse: Mounting at {}", mount_point.display());
    eprintln!("bigedit-fuse: Virtual file at {}/{}", mount_point.display(), filename);
    eprintln!("bigedit-fuse: PID file at {}", pid_path.display());

    // Mount the filesystem (this blocks until unmounted)
    let fs = PatchedFileFS::new(state);
    let options = vec![
        MountOption::RO,
        MountOption::FSName("bigedit".to_string()),
    ];

    fuser::mount2(fs, &mount_point, &options)?;

    // Cleanup on normal exit
    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_dir(&mount_point);

    Ok(())
}
