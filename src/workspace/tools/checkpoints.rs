//! Restoring files to the version saved before an approved change.
//!
//! Every approved write, replacement, move or delete first saves the file it is
//! about to change under `backups/<action id>/`. This lists those saved
//! versions for the connected project and puts one back. A restore keeps a copy
//! of what it replaces, so it can be undone in turn.
//!
//! (Idea: Gemini CLI's checkpointing and `/restore`; independently written.)
use super::*;
use std::time::UNIX_EPOCH;

/// How the file on disk now compares with a saved version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointState {
    /// The file is the same as the saved version: nothing to restore.
    SameAsSaved,
    /// The file was changed after this version was saved.
    ChangedSince,
    /// The file no longer exists (deleted or moved).
    Missing,
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// `<action id>/<file key>`; pass to `restore_checkpoint`.
    pub id: String,
    pub relative_path: String,
    /// Local time the version was saved, `YYYY-MM-DD HH:MM:SS`.
    pub saved_at: String,
    pub bytes: u64,
    pub state: CheckpointState,
}

/// Listing of saved versions, newest first, and how many older ones exist.
#[derive(Clone, Debug, Default)]
pub struct CheckpointList {
    pub checkpoints: Vec<Checkpoint>,
    pub older_not_listed: usize,
}

const MAX_LISTED: usize = 100;

fn safe_component(part: &str) -> bool {
    !part.is_empty()
        && part.len() <= 80
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl ToolEngine {
    /// Saved versions of files in this project, newest first.
    pub fn checkpoints(&self) -> Result<CheckpointList> {
        let mut found: Vec<(std::time::SystemTime, Checkpoint)> = Vec::new();
        let Ok(actions) = fs::read_dir(&self.backup_root) else {
            return Ok(CheckpointList::default());
        };
        for action in actions.flatten() {
            let action_id = action.file_name().to_string_lossy().into_owned();
            if !safe_component(&action_id) || !action.path().is_dir() {
                continue;
            }
            let Ok(files) = fs::read_dir(action.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().is_none_or(|e| e != "json") {
                    continue;
                }
                let Some(key) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                    continue;
                };
                let Ok(meta) = fs::read(&path)
                    .map_err(anyhow::Error::from)
                    .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(Into::into))
                else {
                    continue;
                };
                if meta["workspace"].as_str().map(Path::new) != Some(self.root()) {
                    continue;
                }
                let Some(relative) = meta["relative_path"].as_str() else {
                    continue;
                };
                let saved = action.path().join(format!("{key}.bak"));
                let Ok(saved_meta) = fs::metadata(&saved) else {
                    continue;
                };
                let when = saved_meta.modified().unwrap_or(UNIX_EPOCH);
                let state = match self.state(relative) {
                    Ok(now) if now.kind == "missing" => CheckpointState::Missing,
                    Ok(now) if now.sha256 == meta["sha256"].as_str().map(str::to_owned) => {
                        CheckpointState::SameAsSaved
                    }
                    Ok(_) => CheckpointState::ChangedSince,
                    Err(_) => continue,
                };
                found.push((
                    when,
                    Checkpoint {
                        id: format!("{action_id}/{key}"),
                        relative_path: relative.to_string(),
                        saved_at: chrono::DateTime::<chrono::Local>::from(when)
                            .format("%Y-%m-%d %H:%M:%S")
                            .to_string(),
                        bytes: saved_meta.len(),
                        state,
                    },
                ));
            }
        }
        found.sort_by_key(|(when, _)| std::cmp::Reverse(*when));
        let older_not_listed = found.len().saturating_sub(MAX_LISTED);
        found.truncate(MAX_LISTED);
        Ok(CheckpointList {
            checkpoints: found.into_iter().map(|(_, c)| c).collect(),
            older_not_listed,
        })
    }

    /// Puts the saved version back. What it replaces is saved first.
    pub fn restore_checkpoint(&self, id: &str) -> Result<Value> {
        let (action_id, key) = id.split_once('/').context("Not a saved-version id")?;
        if !safe_component(action_id) || !safe_component(key) {
            bail!("Not a saved-version id");
        }
        let directory = self.backup_root.join(action_id);
        let meta: Value = serde_json::from_slice(&fs::read(directory.join(format!("{key}.json")))?)
            .context("The saved-version record is unreadable")?;
        if meta["workspace"].as_str().map(Path::new) != Some(self.root()) {
            bail!("That saved version belongs to a different project");
        }
        let relative = meta["relative_path"]
            .as_str()
            .context("The saved-version record has no path")?
            .to_string();
        // The file path comes from the id, never from a path stored in the record.
        let bytes = read_limited(&directory.join(format!("{key}.bak")), MAX_FILE_BYTES)?;
        if Some(hash(&bytes)) != meta["sha256"].as_str().map(str::to_owned) {
            bail!(
                "The saved copy no longer matches its record (it was altered); it was not restored"
            );
        }
        // Resolving through the workspace re-applies every confinement rule.
        let now = self.state(&relative)?;
        let mut safety_copy = Value::Null;
        if now.kind != "missing" {
            ensure_file(&now)?;
            let current = read_limited_beneath(self.root(), &now.resolved, MAX_FILE_BYTES)?;
            let new_action = format!("restore-{}", uuid::Uuid::new_v4().simple());
            let new_dir = self.backup_root.join(&new_action);
            fs::create_dir_all(&new_dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&new_dir, fs::Permissions::from_mode(0o700))?;
            }
            let new_key = hash(relative.as_bytes());
            let new_key = &new_key[..20];
            let mut copy = private_create(&new_dir.join(format!("{new_key}.bak")))?;
            copy.write_all(&current)?;
            copy.sync_all()?;
            let mut record = private_create(&new_dir.join(format!("{new_key}.json")))?;
            record.write_all(&serde_json::to_vec_pretty(&json!({
                "action_id": new_action,
                "original_path": now.resolved,
                "workspace": self.root(),
                "relative_path": relative,
                "sha256": hash(&current),
                "backup_file": new_dir.join(format!("{new_key}.bak")),
            }))?)?;
            record.sync_all()?;
            safety_copy = json!(format!("{new_action}/{new_key}"));
        }
        self.write_atomic(&relative, &bytes)?;
        Ok(json!({
            "ok": true,
            "path": relative,
            "restored_bytes": bytes.len(),
            "undo_with": safety_copy,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::Workspace;

    fn engine(dir: &Path, backups: &Path) -> ToolEngine {
        ToolEngine::with_backup_dir(
            Workspace::new(dir.to_path_buf()).unwrap(),
            backups.to_path_buf(),
        )
    }

    fn approve(e: &ToolEngine, tool: &str, args: Value) -> Value {
        let PreparedTool::Approval(action) = e.prepare(tool, &args).unwrap() else {
            panic!("expected an approval")
        };
        e.approve(&action).unwrap()
    }

    #[test]
    fn an_edit_can_be_undone_and_the_undo_can_be_undone() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let file = dir.path().join("a.txt");
        fs::write(&file, "version one\n").unwrap();
        approve(
            &e,
            "write_file",
            json!({"path":"a.txt","content":"version two\n"}),
        );
        assert_eq!(fs::read_to_string(&file).unwrap(), "version two\n");

        let list = e.checkpoints().unwrap();
        assert_eq!(list.checkpoints.len(), 1);
        assert_eq!(list.checkpoints[0].relative_path, "a.txt");
        assert_eq!(list.checkpoints[0].state, CheckpointState::ChangedSince);

        let restored = e.restore_checkpoint(&list.checkpoints[0].id).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "version one\n");
        // What the restore replaced is itself restorable.
        let undo = restored["undo_with"].as_str().unwrap().to_string();
        e.restore_checkpoint(&undo).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "version two\n");
    }

    #[test]
    fn a_deleted_file_is_listed_as_missing_and_comes_back() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("gone.txt"), "precious\n").unwrap();
        approve(&e, "delete_file", json!({"path":"gone.txt"}));
        assert!(!dir.path().join("gone.txt").exists());
        let list = e.checkpoints().unwrap();
        assert_eq!(list.checkpoints[0].state, CheckpointState::Missing);
        let result = e.restore_checkpoint(&list.checkpoints[0].id).unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("gone.txt")).unwrap(),
            "precious\n"
        );
        assert!(result["undo_with"].is_null(), "nothing was there to save");
        assert_eq!(
            e.checkpoints().unwrap().checkpoints[0].state,
            CheckpointState::SameAsSaved
        );
    }

    #[test]
    fn only_this_projects_versions_are_listed_and_ids_cannot_escape() {
        let (one, two, backups) = (
            tempfile::tempdir().unwrap(),
            tempfile::tempdir().unwrap(),
            tempfile::tempdir().unwrap(),
        );
        let (a, b) = (
            engine(one.path(), backups.path()),
            engine(two.path(), backups.path()),
        );
        fs::write(one.path().join("x.txt"), "1").unwrap();
        approve(&a, "write_file", json!({"path":"x.txt","content":"2"}));
        assert_eq!(a.checkpoints().unwrap().checkpoints.len(), 1);
        assert!(b.checkpoints().unwrap().checkpoints.is_empty());
        let id = a.checkpoints().unwrap().checkpoints[0].id.clone();
        assert!(b
            .restore_checkpoint(&id)
            .unwrap_err()
            .to_string()
            .contains("different project"));
        for bad in ["../x", "a/../b", "/etc/passwd", "no-slash", "a/b/c", ""] {
            assert!(a.restore_checkpoint(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_tampered_saved_copy_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("t.txt"), "original").unwrap();
        approve(
            &e,
            "write_file",
            json!({"path":"t.txt","content":"changed"}),
        );
        let id = e.checkpoints().unwrap().checkpoints[0].id.clone();
        let (action, key) = id.split_once('/').unwrap();
        fs::write(
            backups.path().join(action).join(format!("{key}.bak")),
            "evil",
        )
        .unwrap();
        let err = e.restore_checkpoint(&id).unwrap_err().to_string();
        assert!(err.contains("altered"), "{err}");
        assert_eq!(
            fs::read_to_string(dir.path().join("t.txt")).unwrap(),
            "changed"
        );
    }
}
