//! Local schedules and explicitly selected instruction packages.
//! Scheduling is cooperative: the GUI claims work only while idle and running.
//! No OS daemon, shell script, or external message is created by this module.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

const MAX_SCHEDULES: usize = 500;
const MAX_SKILL_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub model: String,
    pub workspace: PathBuf,
    /// Serialized as an RFC3339 timestamp, including timezone, by chrono/serde.
    pub next_run: DateTime<Utc>,
    pub interval_minutes: Option<u64>,
    pub enabled: bool,
    #[serde(default)]
    pub last_run: Option<DateTime<Utc>>,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: String::new(),
            prompt: String::new(),
            model: String::new(),
            workspace: PathBuf::new(),
            next_run: Utc::now() + Duration::hours(1),
            interval_minutes: None,
            enabled: false,
            last_run: None,
        }
    }
}

impl Schedule {
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() || self.id.len() > 128 {
            bail!("Schedule requires a short ID");
        }
        if self.name.trim().is_empty() || self.name.len() > 256 {
            bail!("Schedule requires a name (up to 256 bytes)");
        }
        if self.prompt.trim().is_empty() || self.prompt.len() > 64_000 {
            bail!("Schedule prompt must contain 1–64,000 bytes");
        }
        if self.model.trim().is_empty() || self.model.len() > 256 {
            bail!("Choose a model for the schedule");
        }
        if !self.workspace.is_absolute() {
            bail!("Schedule workspace must be an absolute local directory path");
        }
        if self
            .interval_minutes
            .is_some_and(|n| n == 0 || n > 5_256_000)
        {
            bail!("Repeat interval must be between 1 minute and 10 years");
        }
        Ok(())
    }
}

pub fn load_schedules(directory: &Path) -> Result<Vec<Schedule>> {
    let path = directory.join("schedules.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    if fs::symlink_metadata(&path)?.file_type().is_symlink() {
        bail!("Schedule file must not be a symlink");
    }
    let bytes = read_bounded(&path, 4 * 1024 * 1024)?;
    let schedules: Vec<Schedule> = serde_json::from_slice(&bytes)
        .context("Invalid schedules.json; the existing file was preserved")?;
    validate_schedules(&schedules)?;
    Ok(schedules)
}

pub fn save_schedules(directory: &Path, schedules: &[Schedule]) -> Result<()> {
    validate_schedules(schedules)?;
    let bytes = serde_json::to_vec_pretty(schedules)?;
    if bytes.len() > 4 * 1024 * 1024 {
        bail!("Schedule configuration exceeds 4 MiB");
    }
    fs::create_dir_all(directory)?;
    let destination = directory.join("schedules.json");
    if fs::symlink_metadata(&destination).is_ok_and(|m| m.file_type().is_symlink()) {
        bail!("Refusing to replace a symlinked schedule file");
    }
    let temporary = directory.join(format!("schedules-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, &destination)?;
        if let Ok(parent) = fs::File::open(directory) {
            let _ = parent.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn validate_schedules(schedules: &[Schedule]) -> Result<()> {
    if schedules.len() > MAX_SCHEDULES {
        bail!("At most {MAX_SCHEDULES} schedules are supported");
    }
    let mut ids = HashSet::new();
    for schedule in schedules {
        schedule.validate()?;
        if !ids.insert(&schedule.id) {
            bail!("Duplicate schedule ID: {}", schedule.id);
        }
    }
    Ok(())
}

/// Returns at most `max` due jobs, oldest first, and advances the originals.
/// Persist the updated array BEFORE launching returned jobs. A repeat missed
/// during app downtime is run once; its next time is advanced past `now`.
pub fn take_due(schedules: &mut [Schedule], now: DateTime<Utc>, max: usize) -> Vec<Schedule> {
    let mut indices: Vec<usize> = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| s.enabled && s.next_run <= now && s.validate().is_ok())
        .map(|(index, _)| index)
        .collect();
    indices.sort_by_key(|&index| (schedules[index].next_run, schedules[index].id.clone()));
    let mut due = Vec::new();
    for index in indices.into_iter().take(max.min(MAX_SCHEDULES)) {
        let schedule = &mut schedules[index];
        due.push(schedule.clone());
        schedule.last_run = Some(now);
        if let Some(minutes) = schedule.interval_minutes.filter(|n| *n > 0) {
            let period = (minutes * 60) as i64;
            let late = now
                .signed_duration_since(schedule.next_run)
                .num_seconds()
                .max(0);
            let steps = late / period + 1;
            if let Some(next) = steps.checked_mul(period).and_then(|seconds| {
                schedule
                    .next_run
                    .checked_add_signed(Duration::seconds(seconds))
            }) {
                schedule.next_run = next;
            } else {
                // Extreme out-of-range timestamps are never retried in a hot loop.
                schedule.enabled = false;
            }
        } else {
            schedule.enabled = false;
        }
    }
    due
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSkill {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub instructions: String,
    pub resources: Vec<String>,
}

impl LocalSkill {
    pub fn prompt_context(&self) -> String {
        format!(
            "User-selected local skill: {}\nSource: {}\n\n{}",
            self.name,
            self.path.display(),
            self.instructions
        )
    }
}

/// Read SKILL.md from a directory selected by the user. Loading never executes
/// programs, installs packages, grants tools permissions, or follows symlinks.
pub fn load_skill(directory: &Path) -> Result<LocalSkill> {
    let root = directory
        .canonicalize()
        .context("Skill directory is unavailable")?;
    if !root.is_dir() {
        bail!("Select a directory containing SKILL.md");
    }
    let path = resolve_resource(&root, "SKILL.md")?;
    let source = String::from_utf8(read_bounded(&path, MAX_SKILL_BYTES)?)
        .context("SKILL.md must be UTF-8 text")?;
    let (metadata, instructions) = split_frontmatter(&source);
    let fallback = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Local skill")
        .to_owned();
    let name = metadata_value(metadata, "name")
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(fallback);
    let description = metadata_value(metadata, "description").unwrap_or_default();
    if instructions.trim().is_empty() {
        bail!("SKILL.md has no instructions");
    }
    let mut resources = Vec::new();
    let mut walker = ignore::WalkBuilder::new(&root);
    walker
        .hidden(true)
        .follow_links(false)
        .max_depth(Some(5))
        .filter_entry(|entry| {
            !matches!(
                entry.file_name().to_str(),
                Some("node_modules" | "target" | ".git" | ".venv")
            )
        });
    for item in walker.build() {
        if resources.len() >= 128 {
            break;
        }
        let entry = item?;
        if !entry.file_type().is_some_and(|f| f.is_file()) || entry.path() == path {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(&root)?
            .to_string_lossy()
            .replace('\\', "/");
        if resolve_resource(&root, &relative).is_ok() {
            resources.push(relative);
        }
    }
    resources.sort();
    Ok(LocalSkill {
        name,
        description,
        path: root,
        instructions: instructions.to_owned(),
        resources,
    })
}

pub fn read_skill_resource(skill: &LocalSkill, relative: &str, max_chars: usize) -> Result<String> {
    let path = resolve_resource(&skill.path, relative)?;
    let bytes = read_bounded(&path, MAX_SKILL_BYTES)?;
    if bytes.contains(&0) {
        bail!("Skill resource is binary; only text can be added to the prompt");
    }
    let text = String::from_utf8(bytes).context("Skill resource must be UTF-8")?;
    let cap = max_chars.clamp(1, MAX_SKILL_BYTES);
    let truncated = text.chars().count() > cap;
    let mut result: String = text.chars().take(cap).collect();
    if truncated {
        result.push_str("\n[Resource truncated]");
    }
    Ok(result)
}

fn resolve_resource(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if relative.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!("Skill resource must be a relative path without parent traversal");
    }
    let root = root.canonicalize()?;
    let mut current = root.clone();
    for component in path.components() {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)?.file_type().is_symlink() {
            bail!("Skill resources cannot use symlinks");
        }
    }
    let resolved = current.canonicalize()?;
    if !resolved.starts_with(&root) || !resolved.is_file() {
        bail!("Skill resource is not a file inside its directory");
    }
    Ok(resolved)
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > limit as u64 {
        bail!("File is not regular or exceeds {limit} bytes");
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut bytes = Vec::new();
    options
        .open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("File exceeded {limit} bytes while reading");
    }
    Ok(bytes)
}

fn split_frontmatter(source: &str) -> (&str, &str) {
    if let Some(rest) = source.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---\n") {
            return (&rest[..end], &rest[end + 5..]);
        }
    }
    ("", source)
}

// Name/description are optional display metadata. This intentionally does not
// interpret arbitrary YAML tags or construct executable objects.
fn metadata_value(source: &str, key: &str) -> Option<String> {
    for line in source.lines() {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim() == key {
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if matches!(value, "|" | ">" | "|-") {
                    return None;
                }
                return Some(value.to_owned());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn schedule(next: &str) -> Schedule {
        Schedule {
            name: "Review".into(),
            prompt: "Review the selected source".into(),
            model: "local:test".into(),
            workspace: PathBuf::from("/tmp"),
            next_run: DateTime::parse_from_rfc3339(next)
                .unwrap()
                .with_timezone(&Utc),
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn missed_intervals_are_collapsed_and_single_shots_disabled() {
        let now = DateTime::parse_from_rfc3339("2026-11-07T17:03:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let mut repeat = schedule("2026-11-01T10:00:00Z");
        repeat.interval_minutes = Some(5);
        let mut schedules = vec![repeat, schedule("2026-11-07T17:00:00Z")];
        let due = take_due(&mut schedules, now, 5);
        assert_eq!(due.len(), 2);
        assert!(schedules[0].next_run > now);
        assert!(schedules[0].next_run - now <= Duration::minutes(5));
        assert!(!schedules[1].enabled);
        assert!(take_due(&mut schedules, now, 5).is_empty());
    }

    #[test]
    fn schedule_roundtrip_and_invalid_save_preserves_previous_data() {
        let directory = tempfile::tempdir().unwrap();
        let saved = schedule("2026-11-07T17:00:00-05:00");
        save_schedules(directory.path(), std::slice::from_ref(&saved)).unwrap();
        let mut invalid = saved.clone();
        invalid.interval_minutes = Some(0);
        assert!(save_schedules(directory.path(), &[invalid]).is_err());
        let loaded = load_schedules(directory.path()).unwrap();
        assert_eq!(loaded[0].id, saved.id);
        assert_eq!(loaded[0].next_run, saved.next_run);
    }

    #[test]
    fn skill_loading_is_bounded_and_does_not_execute_resources() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("SKILL.md"),
            "---\nname: careful-review\ndescription: Source review\n---\nRead evidence first.",
        )
        .unwrap();
        fs::write(directory.path().join("script.sh"), "touch must-not-exist\n").unwrap();
        let skill = load_skill(directory.path()).unwrap();
        assert_eq!(skill.name, "careful-review");
        assert_eq!(skill.instructions, "Read evidence first.");
        assert!(skill.resources.contains(&"script.sh".to_owned()));
        assert!(!directory.path().join("must-not-exist").exists());
        assert!(read_skill_resource(&skill, "../outside", 1000).is_err());
        assert!(read_skill_resource(&skill, "script.sh", 5)
            .unwrap()
            .contains("truncated"));
    }

    #[test]
    #[cfg(unix)]
    fn skill_symlink_escape_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("SKILL.md"), "Instructions").unwrap();
        fs::write(outside.path().join("secret"), "outside").unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret"), directory.path().join("link"))
            .unwrap();
        let skill = load_skill(directory.path()).unwrap();
        assert!(read_skill_resource(&skill, "link", 1000).is_err());
    }
}
