//! ToolEngine filesystem responsibilities.
use super::*;

impl ToolEngine {
    pub(super) fn state(&self, path: &str) -> Result<PathState> {
        let resolved = self.workspace.resolve(path, true)?;
        match fs::symlink_metadata(&resolved) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    bail!("Symlink targets cannot be used for workspace actions");
                }
                if meta.is_file() {
                    let bytes = read_limited_beneath(self.root(), &resolved, MAX_FILE_BYTES)?;
                    Ok(PathState {
                        path: path.into(),
                        resolved,
                        kind: "file".into(),
                        sha256: Some(hash(&bytes)),
                        size: bytes.len() as u64,
                    })
                } else if meta.is_dir() {
                    Ok(PathState {
                        path: path.into(),
                        resolved,
                        kind: "directory".into(),
                        sha256: None,
                        size: 0,
                    })
                } else {
                    bail!("Only ordinary files and directories are supported");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PathState {
                path: path.into(),
                resolved,
                kind: "missing".into(),
                sha256: None,
                size: 0,
            }),
            Err(e) => Err(e.into()),
        }
    }

    pub(super) fn backup(&self, action: &PendingAction, path: &str) -> Result<Option<PathBuf>> {
        let state = self.state(path)?;
        if state.kind == "missing" {
            return Ok(None);
        }
        ensure_file(&state)?;
        let expected = action
            .expected
            .iter()
            .find(|s| s.path == path)
            .context("Path was not part of the approved action")?;
        if state != *expected {
            bail!("File changed before backup; no write was performed");
        }
        let bytes = read_limited_beneath(self.root(), &state.resolved, MAX_FILE_BYTES)?;
        if Some(hash(&bytes)) != state.sha256 {
            bail!("File changed while reading its backup");
        }
        let directory = self.backup_root.join(&action.id);
        fs::create_dir_all(&directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
        }
        let key = hash(path.as_bytes());
        let destination = directory.join(format!("{}.bak", &key[..20]));
        let mut file = private_create(&destination)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        let metadata = directory.join(format!("{}.json", &key[..20]));
        let mut meta = private_create(&metadata)?;
        meta.write_all(&serde_json::to_vec_pretty(&json!({"action_id":action.id,"original_path":state.resolved,"workspace":action.workspace,"relative_path":path,"sha256":state.sha256,"backup_file":destination}))?)?;
        meta.sync_all()?;
        Ok(Some(destination))
    }

    /// Remove a workspace file, beneath the root through descriptors where possible.
    pub(super) fn remove_confined(&self, path: &str) -> Result<()> {
        let target = self.workspace.resolve(path, false)?;
        if let Ok(relative) = target.strip_prefix(self.root()) {
            if crate::confined::remove_file(self.root(), relative)?.is_some() {
                return Ok(());
            }
        }
        fs::remove_file(target)?;
        Ok(())
    }

    /// Create a workspace directory and its parents, beneath the root through
    /// descriptors where possible.
    pub(super) fn create_dir_confined(&self, target: &Path) -> Result<()> {
        if let Ok(relative) = target.strip_prefix(self.root()) {
            if crate::confined::create_dir_all(self.root(), relative)?.is_some() {
                return Ok(());
            }
        }
        fs::create_dir_all(target)?;
        Ok(())
    }

    pub(super) fn write_atomic(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let target = self.workspace.resolve(path, true)?;
        let parent = target.parent().context("File has no parent")?;
        self.create_dir_confined(parent)?;
        // Re-resolve after creating parents; directory aliases must still be contained.
        let target = self.workspace.resolve(path, true)?;
        let parent = target.parent().context("File has no parent")?;
        // Write and rename through a descriptor for the parent opened beneath the
        // root, so a directory swapped after the checks cannot redirect the file.
        if let Ok(relative) = target.strip_prefix(self.root()) {
            #[cfg(unix)]
            let mode = {
                use std::os::unix::fs::PermissionsExt;
                fs::symlink_metadata(&target)
                    .ok()
                    .filter(|m| m.is_file())
                    .map(|m| m.permissions().mode())
            };
            #[cfg(not(unix))]
            let mode = None;
            if crate::confined::write_atomic(self.root(), relative, bytes, mode)?.is_some() {
                return Ok(());
            }
        }
        let temp = parent.join(format!(".workbench-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = private_create(&temp)?;
            if let Ok(metadata) = fs::metadata(&target) {
                file.set_permissions(metadata.permissions())?;
            }
            file.write_all(bytes)?;
            file.sync_all()?;
            let checked = self.workspace.resolve(path, true)?;
            if checked != target {
                bail!("Target resolution changed while writing");
            }
            fs::rename(&temp, &target)?;
            if let Ok(directory) = fs::File::open(parent) {
                let _ = directory.sync_all();
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}
