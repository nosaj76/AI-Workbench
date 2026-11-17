//! Local crash evidence, kept in `<data dir>/crash-reports/`.
//!
//! Two independent signals, because a panic hook alone misses most real crashes:
//! - a panic hook writes one report per panic (message, location, thread,
//!   backtrace), then defers to the default hook so stderr output is unchanged;
//! - a `running.pid` marker exists while the desktop runs and is removed on a
//!   clean exit. Finding it at the next start, with that process gone, means the
//!   previous run ended without closing: a segfault, an out-of-memory kill, a GPU
//!   driver fault, a forced quit or a power loss.
//!
//! Nothing here is sent anywhere.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DIR: &str = "crash-reports";
const MARKER: &str = "running.pid";
const KEEP_REPORTS: usize = 20;

pub fn directory(data_dir: &Path) -> PathBuf {
    data_dir.join(DIR)
}

/// Write a report for every panic in this process, on any thread. Panics that a
/// background job catches are recorded too; the report says the process may
/// have continued.
pub fn install_panic_hook(data_dir: &Path) {
    let dir = directory(data_dir);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-text panic payload".into());
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".into());
        let thread = std::thread::current();
        let report = format!(
            "AI Workbench {version} panic report\n\
             Time: {time}\n\
             Process: {pid}\n\
             Thread: {thread}\n\
             Location: {location}\n\
             Message: {message}\n\n\
             A panic on the main thread closes the app; a panic in a background job \
             may have been caught and the app may have continued.\n\n\
             Backtrace:\n{backtrace}\n",
            version = env!("CARGO_PKG_VERSION"),
            time = chrono::Utc::now().to_rfc3339(),
            pid = std::process::id(),
            thread = thread.name().unwrap_or("unnamed"),
            backtrace = std::backtrace::Backtrace::force_capture(),
        );
        // Never panic inside the panic hook; a failed write only loses the report.
        let _ = write_report(&dir, &report);
        previous(info);
    }));
}

fn write_report(dir: &Path, report: &str) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!(
        "panic-{}-{}.txt",
        chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ"),
        std::process::id()
    ));
    fs::File::create(&path)?.write_all(report.as_bytes())?;
    prune(dir);
    Ok(path)
}

/// Keep the newest reports only, so a panic loop cannot fill the disk.
fn prune(dir: &Path) {
    let mut reports = list_reports(dir);
    if reports.len() > KEEP_REPORTS {
        reports.sort_by_key(|(_, modified)| *modified);
        for (path, _) in &reports[..reports.len() - KEEP_REPORTS] {
            let _ = fs::remove_file(path);
        }
    }
}

fn list_reports(dir: &Path) -> Vec<(PathBuf, SystemTime)> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("panic-") && name.ends_with(".txt")
        })
        .filter_map(|e| Some((e.path(), e.metadata().ok()?.modified().ok()?)))
        .collect()
}

/// The newest panic reports, newest first.
pub fn recent_reports(data_dir: &Path, limit: usize) -> Vec<PathBuf> {
    let mut reports = list_reports(&directory(data_dir));
    reports.sort_by_key(|(_, modified)| std::cmp::Reverse(*modified));
    reports.into_iter().take(limit).map(|(p, _)| p).collect()
}

/// A tracked desktop run. Dropping it without `finish` leaves the marker, which
/// is the point: only a clean exit removes it.
pub struct Session {
    marker: Option<PathBuf>,
}

impl Session {
    /// Remove the marker after a clean exit.
    pub fn finish(self) {
        if let Some(marker) = &self.marker {
            if read_pid(marker) == Some(std::process::id()) {
                let _ = fs::remove_file(marker);
            }
        }
    }
}

/// Start tracking this run and describe how the previous one ended, if it ended
/// badly. Another live instance keeps its own marker and this run is untracked.
pub fn begin_session(data_dir: &Path) -> (Session, Option<String>) {
    let dir = directory(data_dir);
    let marker = dir.join(MARKER);
    let previous_start = fs::metadata(&marker).and_then(|m| m.modified()).ok();
    let mut notice = None;
    if let Some(pid) = read_pid(&marker) {
        if pid != std::process::id() && process_alive(pid) {
            return (Session { marker: None }, None);
        }
        let panics: Vec<PathBuf> = list_reports(&dir)
            .into_iter()
            .filter(|(_, modified)| previous_start.is_none_or(|start| *modified >= start))
            .map(|(path, _)| path)
            .collect();
        notice = Some(match panics.iter().max() {
            Some(newest) => format!(
                "AI Workbench closed unexpectedly last time after an internal error. Report: {}",
                newest.display()
            ),
            None => format!(
                "AI Workbench did not shut down cleanly last time (crash, forced quit or power loss); no internal error was recorded. Reports are kept in {}",
                dir.display()
            ),
        });
    }
    let tracked = fs::create_dir_all(&dir)
        .and_then(|_| fs::write(&marker, std::process::id().to_string()))
        .is_ok();
    (
        Session {
            marker: tracked.then_some(marker),
        },
        notice,
    )
}

fn read_pid(marker: &Path) -> Option<u32> {
    fs::read_to_string(marker).ok()?.trim().parse().ok()
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    // Signal 0 checks existence only. EPERM means it exists under another user.
    let exists = unsafe { libc::kill(pid, 0) } == 0;
    exists || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_exit_leaves_no_notice() {
        let dir = tempfile::tempdir().unwrap();
        let (session, notice) = begin_session(dir.path());
        assert!(notice.is_none());
        session.finish();
        let (_session, notice) = begin_session(dir.path());
        assert!(notice.is_none());
    }

    #[test]
    fn unclean_exit_is_reported_with_its_panic_report() {
        let dir = tempfile::tempdir().unwrap();
        let reports = directory(dir.path());
        fs::create_dir_all(&reports).unwrap();
        // A previous run whose process no longer exists.
        fs::write(reports.join(MARKER), u32::MAX.to_string()).unwrap();
        let (_s, notice) = begin_session(dir.path());
        assert!(notice.unwrap().contains("did not shut down cleanly"));

        fs::write(reports.join(MARKER), u32::MAX.to_string()).unwrap();
        let report = write_report(&reports, "boom").unwrap();
        let (_s, notice) = begin_session(dir.path());
        let notice = notice.unwrap();
        assert!(notice.contains("internal error"), "{notice}");
        assert!(notice.contains(&report.display().to_string()));
    }

    #[test]
    fn another_live_instance_keeps_its_marker() {
        let dir = tempfile::tempdir().unwrap();
        let reports = directory(dir.path());
        fs::create_dir_all(&reports).unwrap();
        // The test runner's parent is alive and is not this process.
        let other = std::os::unix::process::parent_id();
        fs::write(reports.join(MARKER), other.to_string()).unwrap();
        let (session, notice) = begin_session(dir.path());
        assert!(notice.is_none());
        session.finish();
        assert_eq!(read_pid(&reports.join(MARKER)), Some(other));
    }

    #[test]
    fn old_reports_are_pruned() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..KEEP_REPORTS + 5 {
            fs::write(dir.path().join(format!("panic-{i:03}.txt")), "x").unwrap();
        }
        prune(dir.path());
        assert_eq!(list_reports(dir.path()).len(), KEEP_REPORTS);
    }
}
