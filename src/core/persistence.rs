//! Bounded, private configuration writes committed by atomic rename.
//! Failures before rename preserve the destination; directory-sync failures after
//! rename report durability uncertainty rather than pretending to roll back.
use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};

pub fn read_json<T: DeserializeOwned>(path: &Path, max_bytes: u64) -> Result<T> {
    let file = fs::File::open(path).with_context(|| format!("Read {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "{} exceeds the {} byte configuration limit; original preserved",
            path.display(),
            max_bytes
        );
    }
    serde_json::from_slice(&bytes)
        .with_context(|| format!("Invalid JSON in {}; original preserved", path.display()))
}

pub fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    write_atomic(path, &serde_json::to_vec_pretty(value)?)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .context("Destination requires a file name")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Preserve an unreadable configuration before an explicitly requested replacement.
pub fn backup_existing(path: &Path) -> Result<Option<std::path::PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut source = fs::File::open(path)?;
    let name = path
        .file_name()
        .context("Backup requires a file name")?
        .to_string_lossy();
    let backup = path.with_file_name(format!("{name}.recovery-{}", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut destination = options.open(&backup)?;
    std::io::copy(&mut source, &mut destination)?;
    destination.sync_all()?;
    Ok(Some(backup))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_bounded_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        write_json(&path, &serde_json::json!({"theme":"light"})).unwrap();
        let value: serde_json::Value = read_json(&path, 1024).unwrap();
        assert_eq!(value["theme"], "light");
        assert!(read_json::<serde_json::Value>(&path, 4).is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
    #[test]
    fn failed_replacement_leaves_existing_directory_and_no_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("occupied");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("keep"), "preserved").unwrap();
        assert!(write_atomic(&target, b"replacement").is_err());
        assert_eq!(
            fs::read_to_string(target.join("keep")).unwrap(),
            "preserved"
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
