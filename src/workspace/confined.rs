//! File access anchored beneath the workspace root.
//!
//! Path checks (`Workspace::resolve`) happen before an operation; if another local
//! process swapped a checked directory for a symlink in between, a later path-based
//! open or rename could land outside the workspace. On Linux, the functions here
//! open every directory and file with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS
//! | RESOLVE_NO_MAGICLINKS)` relative to a descriptor for the root, and then work
//! only through descriptors, so the kernel itself refuses any escape.
//!
//! Where that is unavailable (other systems, or a kernel older than 5.6) the
//! functions return `Ok(None)` and callers keep their existing path-checked code.

use std::fs::File;
use std::io;
use std::path::Path;

/// Open a regular file beneath `root` for reading. `relative` is workspace-relative.
pub fn open_read(root: &Path, relative: &Path) -> io::Result<Option<File>> {
    imp::open_read(root, relative)
}

/// Atomically replace (or create) `relative` beneath `root` with `bytes`, keeping
/// `mode` when given. The parent directory must already exist.
pub fn write_atomic(
    root: &Path,
    relative: &Path,
    bytes: &[u8],
    mode: Option<u32>,
) -> io::Result<Option<()>> {
    imp::write_atomic(root, relative, bytes, mode)
}

/// Create `relative` and any missing parents beneath `root`, one level at a time
/// through descriptors, so no level can be redirected through a symlink.
pub fn create_dir_all(root: &Path, relative: &Path) -> io::Result<Option<()>> {
    imp::create_dir_all(root, relative)
}

/// Remove the file `relative` beneath `root` (never a directory).
pub fn remove_file(root: &Path, relative: &Path) -> io::Result<Option<()>> {
    imp::remove_file(root, relative)
}

#[cfg(target_os = "linux")]
mod imp {
    use std::ffi::CString;
    use std::fs::File;
    use std::io::{self, Write};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    fn cstr(path: &Path) -> io::Result<CString> {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))
    }

    /// openat2 relative to `dir`, confined beneath it. None when the kernel lacks openat2.
    fn openat2(
        dir: &OwnedFd,
        relative: &Path,
        flags: i32,
        mode: u32,
    ) -> io::Result<Option<OwnedFd>> {
        let path = cstr(relative)?;
        let mut how: libc::open_how = unsafe { std::mem::zeroed() };
        how.flags = (flags | libc::O_CLOEXEC) as u64;
        how.mode = if flags & libc::O_CREAT != 0 {
            mode as u64
        } else {
            0
        };
        how.resolve =
            libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS;
        let fd = unsafe {
            libc::syscall(
                libc::SYS_openat2,
                dir.as_raw_fd(),
                path.as_ptr(),
                &how as *const libc::open_how,
                std::mem::size_of::<libc::open_how>(),
            )
        };
        if fd < 0 {
            let error = io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::ENOSYS) => Ok(None),
                Some(libc::EXDEV) | Some(libc::ELOOP) => Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "{} leaves the workspace or passes through a symlink; refused",
                        relative.display()
                    ),
                )),
                _ => Err(error),
            };
        }
        Ok(Some(unsafe { OwnedFd::from_raw_fd(fd as i32) }))
    }

    fn open_root(root: &Path) -> io::Result<OwnedFd> {
        let path = cstr(root)?;
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// The parent directory of `relative`, opened beneath the root, and the file name.
    fn parent(root: &Path, relative: &Path) -> io::Result<Option<(OwnedFd, CString)>> {
        let name = relative
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
        let name = CString::new(name.as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
        let root_fd = open_root(root)?;
        let directory = match relative.parent().filter(|p| !p.as_os_str().is_empty()) {
            Some(dir) => match openat2(&root_fd, dir, libc::O_RDONLY | libc::O_DIRECTORY, 0)? {
                Some(fd) => fd,
                None => return Ok(None),
            },
            None => root_fd,
        };
        Ok(Some((directory, name)))
    }

    pub fn open_read(root: &Path, relative: &Path) -> io::Result<Option<File>> {
        let root_fd = open_root(root)?;
        let Some(fd) = openat2(&root_fd, relative, libc::O_RDONLY | libc::O_NOFOLLOW, 0)? else {
            return Ok(None);
        };
        let file = File::from(fd);
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        Ok(Some(file))
    }

    pub fn write_atomic(
        root: &Path,
        relative: &Path,
        bytes: &[u8],
        mode: Option<u32>,
    ) -> io::Result<Option<()>> {
        let Some((directory, name)) = parent(root, relative)? else {
            return Ok(None);
        };
        let temp = CString::new(format!(".workbench-{}.tmp", uuid::Uuid::new_v4())).unwrap();
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                temp.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600 as libc::c_uint,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let written = (|| -> io::Result<()> {
            file.write_all(bytes)?;
            if let Some(mode) = mode {
                if unsafe { libc::fchmod(file.as_raw_fd(), mode & 0o7777) } != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            file.sync_all()?;
            // renameat works on the directory descriptor: a directory swapped after
            // it was opened cannot redirect the final name.
            if unsafe {
                libc::renameat(
                    directory.as_raw_fd(),
                    temp.as_ptr(),
                    directory.as_raw_fd(),
                    name.as_ptr(),
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
            unsafe { libc::fsync(directory.as_raw_fd()) };
            Ok(())
        })();
        if written.is_err() {
            unsafe { libc::unlinkat(directory.as_raw_fd(), temp.as_ptr(), 0) };
        }
        written.map(Some)
    }

    pub fn create_dir_all(root: &Path, relative: &Path) -> io::Result<Option<()>> {
        let mut directory = open_root(root)?;
        for component in relative.components() {
            let std::path::Component::Normal(name) = component else {
                if matches!(component, std::path::Component::CurDir) {
                    continue;
                }
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "only plain workspace-relative directory names are allowed",
                ));
            };
            let level = Path::new(name);
            let opened = match openat2(&directory, level, libc::O_RDONLY | libc::O_DIRECTORY, 0) {
                Ok(fd) => fd,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    let c_name = cstr(level)?;
                    if unsafe { libc::mkdirat(directory.as_raw_fd(), c_name.as_ptr(), 0o755) } != 0
                    {
                        let error = io::Error::last_os_error();
                        if error.kind() != io::ErrorKind::AlreadyExists {
                            return Err(error);
                        }
                    }
                    openat2(&directory, level, libc::O_RDONLY | libc::O_DIRECTORY, 0)?
                }
                Err(error) => return Err(error),
            };
            let Some(next) = opened else {
                return Ok(None);
            };
            directory = next;
        }
        Ok(Some(()))
    }

    pub fn remove_file(root: &Path, relative: &Path) -> io::Result<Option<()>> {
        let Some((directory, name)) = parent(root, relative)? else {
            return Ok(None);
        };
        // Flag 0: unlinkat refuses directories and removes a symlink itself, never
        // what it points to.
        if unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Some(()))
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::fs::File;
    use std::io;
    use std::path::Path;
    pub fn open_read(_: &Path, _: &Path) -> io::Result<Option<File>> {
        Ok(None)
    }
    pub fn write_atomic(_: &Path, _: &Path, _: &[u8], _: Option<u32>) -> io::Result<Option<()>> {
        Ok(None)
    }
    pub fn remove_file(_: &Path, _: &Path) -> io::Result<Option<()>> {
        Ok(None)
    }
    pub fn create_dir_all(_: &Path, _: &Path) -> io::Result<Option<()>> {
        Ok(None)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn reads_and_writes_inside_the_root_work() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        write_atomic(dir.path(), Path::new("sub/a.txt"), b"hello", Some(0o644))
            .unwrap()
            .unwrap();
        let mut text = String::new();
        open_read(dir.path(), Path::new("sub/a.txt"))
            .unwrap()
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(text, "hello");
        remove_file(dir.path(), Path::new("sub/a.txt"))
            .unwrap()
            .unwrap();
        assert!(!dir.path().join("sub/a.txt").exists());
    }

    #[test]
    fn a_directory_swapped_for_a_symlink_cannot_redirect_writes_or_reads_outside() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        // The race: "sub" was a real directory when checked, then replaced by a link.
        std::os::unix::fs::symlink(&outside, root.join("sub")).unwrap();
        let write = write_atomic(&root, Path::new("sub/new.txt"), b"x", None);
        assert!(write.is_err(), "{write:?}");
        assert!(!outside.join("new.txt").exists());
        assert!(open_read(&root, Path::new("sub/secret.txt")).is_err());
        assert!(remove_file(&root, Path::new("sub/secret.txt")).is_err());
        assert!(outside.join("secret.txt").exists());
        assert!(open_read(&root, Path::new("../outside/secret.txt")).is_err());
        assert!(create_dir_all(&root, Path::new("sub/made")).is_err());
        assert!(!outside.join("made").exists());
        create_dir_all(&root, Path::new("a/b/c")).unwrap().unwrap();
        assert!(root.join("a/b/c").is_dir());
    }
}
