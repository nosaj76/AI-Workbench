//! Durable cron/interval/one-shot jobs, transactional claims and a headless
//! scheduler. A daemon survives the GUI closing; user-systemd integration is
//! generated only when explicitly requested through the application.

use crate::ollama::ChatMessage;
use crate::runtime::{self, ChatEvent, ChatJob};
use crate::settings::Settings;
use crate::storage::{Entry, Session, Store};
use anyhow::{bail, Context, Result};
use chrono::{
    DateTime, Datelike, Duration as ChronoDuration, LocalResult, NaiveDate, TimeZone, Timelike, Utc,
};
use chrono_tz::Tz;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const MAX_JOBS: usize = 500;
const MAX_PROMPT: usize = 64_000;
const LEASE_SECONDS: i64 = 90;
pub const SERVICE_NAME: &str = "ollama-workbench-scheduler.service";

#[derive(Debug, Clone)]
pub struct CronSchedule {
    minute: Vec<bool>,
    hour: Vec<bool>,
    day: Vec<bool>,
    month: Vec<bool>,
    weekday: Vec<bool>,
    day_star: bool,
    weekday_star: bool,
    reboot: bool,
}

impl CronSchedule {
    /// Traditional five-field cron, including lists, inclusive ranges, steps,
    /// month/day names, Sunday 0/7, and the standard @... aliases.
    pub fn parse(expression: &str) -> Result<Self> {
        let expression = expression.trim();
        if expression.len() > 256 {
            bail!("Cron expression exceeds 256 bytes");
        }
        let normalized = match expression.to_ascii_lowercase().as_str() {
            "@yearly" | "@annually" => "0 0 1 1 *",
            "@monthly" => "0 0 1 * *",
            "@weekly" => "0 0 * * 0",
            "@daily" | "@midnight" => "0 0 * * *",
            "@hourly" => "0 * * * *",
            "@reboot" => "@reboot",
            _ => expression,
        };
        if normalized == "@reboot" {
            return Ok(Self {
                minute: vec![],
                hour: vec![],
                day: vec![],
                month: vec![],
                weekday: vec![],
                day_star: true,
                weekday_star: true,
                reboot: true,
            });
        }
        let fields = normalized.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 5 {
            bail!("Cron requires five fields: minute hour day-of-month month day-of-week");
        }
        Ok(Self {
            minute: parse_field(fields[0], 0, 59, &[])?,
            hour: parse_field(fields[1], 0, 23, &[])?,
            day: parse_field(fields[2], 1, 31, &[])?,
            month: parse_field(
                fields[3],
                1,
                12,
                &[
                    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                    "DEC",
                ],
            )?,
            weekday: parse_field(
                fields[4],
                0,
                7,
                &["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"],
            )?,
            day_star: fields[2].starts_with('*'),
            weekday_star: fields[4].starts_with('*'),
            reboot: false,
        })
    }
    pub fn is_reboot(&self) -> bool {
        self.reboot
    }
    pub fn matches(&self, at: DateTime<Utc>, timezone: &str) -> Result<bool> {
        if self.reboot {
            return Ok(false);
        }
        let tz: Tz = timezone
            .parse()
            .with_context(|| format!("Unknown IANA timezone: {timezone}"))?;
        let local = at.with_timezone(&tz);
        Ok(self.month[local.month() as usize]
            && self.date_matches(local.date_naive())
            && self.hour[local.hour() as usize]
            && self.minute[local.minute() as usize])
    }
    /// Searches local calendar candidates, then orders their UTC instants. A
    /// nonexistent DST wall time is skipped; a repeated wall time runs twice.
    pub fn next_after(&self, after: DateTime<Utc>, timezone: &str) -> Result<DateTime<Utc>> {
        if self.reboot {
            bail!("@reboot is activated by the daemon's boot observer, not a calendar time");
        }
        let tz: Tz = timezone
            .parse()
            .with_context(|| format!("Unknown IANA timezone: {timezone}"))?;
        let start = after.with_timezone(&tz).date_naive();
        for offset in 0..=(366 * 8) {
            let date = start
                .checked_add_signed(ChronoDuration::days(offset))
                .context("Cron date is out of range")?;
            if !self.month[date.month() as usize] || !self.date_matches(date) {
                continue;
            }
            let mut candidates = BTreeSet::new();
            for (hour, _) in self.hour.iter().enumerate().filter(|(_, on)| **on) {
                for (minute, _) in self.minute.iter().enumerate().filter(|(_, on)| **on) {
                    let wall = date
                        .and_hms_opt(hour as u32, minute as u32, 0)
                        .context("Invalid cron time")?;
                    match tz.from_local_datetime(&wall) {
                        LocalResult::Single(t) => {
                            candidates.insert(t.with_timezone(&Utc));
                        }
                        LocalResult::Ambiguous(a, b) => {
                            candidates.insert(a.with_timezone(&Utc));
                            candidates.insert(b.with_timezone(&Utc));
                        }
                        LocalResult::None => {}
                    }
                }
            }
            if let Some(next) = candidates.into_iter().find(|t| *t > after) {
                return Ok(next);
            }
        }
        bail!("Cron expression has no calendar occurrence in the next eight years")
    }
    fn date_matches(&self, date: NaiveDate) -> bool {
        let dom = self.day[date.day() as usize];
        let dow = date.weekday().num_days_from_sunday() as usize;
        let week = self.weekday[dow] || (dow == 0 && self.weekday[7]);
        if self.day_star || self.weekday_star {
            dom && week
        } else {
            dom || week
        }
    }
}

fn parse_field(raw: &str, min: usize, max: usize, names: &[&str]) -> Result<Vec<bool>> {
    let mut selected = vec![false; max + 1];
    let value = |text: &str| -> Result<usize> {
        if let Some(i) = names.iter().position(|n| n.eq_ignore_ascii_case(text)) {
            return Ok(if min == 1 { i + 1 } else { i });
        }
        let n: usize = text
            .parse()
            .with_context(|| format!("Invalid cron value: {text}"))?;
        if n < min || n > max {
            bail!("Cron value {n} is outside {min}–{max}");
        }
        Ok(n)
    };
    for term in raw.split(',') {
        if term.is_empty() {
            bail!("Empty cron list item");
        }
        let parts = term.split('/').collect::<Vec<_>>();
        if parts.len() > 2 {
            bail!("Invalid cron step: {term}");
        }
        let step = if parts.len() == 2 {
            parts[1]
                .parse::<usize>()
                .context("Cron step must be a positive integer")?
        } else {
            1
        };
        if step == 0 || step > max - min + 1 {
            bail!("Cron step is outside this field's range");
        }
        let (start, end) = if parts[0] == "*" {
            (min, max)
        } else if let Some((a, b)) = parts[0].split_once('-') {
            (value(a)?, value(b)?)
        } else {
            let n = value(parts[0])?;
            (n, if parts.len() == 2 { max } else { n })
        };
        if start > end {
            bail!("Cron ranges must be in ascending order");
        }
        for n in (start..=end).step_by(step) {
            selected[n] = true;
        }
    }
    if !selected.iter().any(|v| *v) {
        bail!("Cron field matches no values");
    }
    Ok(selected)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleTiming {
    Cron {
        expression: String,
        timezone: String,
    },
    Interval {
        minutes: u64,
    },
    Once {
        at: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub model: String,
    pub workspace: PathBuf,
    pub timing: ScheduleTiming,
    pub enabled: bool,
    pub next_run: DateTime<Utc>,
    pub timeout_seconds: u64,
    #[serde(default)]
    pub last_run: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_boot_id: Option<String>,
}

impl ScheduledJob {
    fn is_reboot(&self) -> bool {
        matches!(&self.timing, ScheduleTiming::Cron { expression, .. } if expression.trim().eq_ignore_ascii_case("@reboot"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cron(
        name: &str,
        prompt: &str,
        model: &str,
        workspace: PathBuf,
        expression: &str,
        timezone: &str,
        now: DateTime<Utc>,
    ) -> Result<Self> {
        let parsed = CronSchedule::parse(expression)?;
        let next = if parsed.is_reboot() {
            DateTime::<Utc>::MAX_UTC
        } else {
            parsed.next_after(now, timezone)?
        };
        let job = Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            prompt: prompt.into(),
            model: model.into(),
            workspace,
            timing: ScheduleTiming::Cron {
                expression: expression.into(),
                timezone: timezone.into(),
            },
            enabled: true,
            next_run: next,
            timeout_seconds: 1800,
            last_run: None,
            last_boot_id: None,
        };
        job.validate()?;
        Ok(job)
    }
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty()
            || self.id.len() > 128
            || self.name.trim().is_empty()
            || self.name.len() > 256
        {
            bail!("Scheduled job requires a short ID and name");
        }
        if self.prompt.trim().is_empty() || self.prompt.len() > MAX_PROMPT {
            bail!("Scheduled prompt must contain 1–64,000 bytes");
        }
        if self.model.trim().is_empty() || self.model.len() > 256 {
            bail!("Scheduled job requires a model");
        }
        if !self.workspace.is_absolute() {
            bail!("Scheduled workspace must be an absolute local directory");
        }
        if !(30..=86400).contains(&self.timeout_seconds) {
            bail!("Job timeout must be 30–86,400 seconds");
        }
        match &self.timing {
            ScheduleTiming::Cron {
                expression,
                timezone,
            } => {
                CronSchedule::parse(expression)?;
                timezone
                    .parse::<Tz>()
                    .with_context(|| format!("Unknown IANA timezone: {timezone}"))?;
            }
            ScheduleTiming::Interval { minutes } => {
                if !(1..=5_256_000).contains(minutes) {
                    bail!("Repeat interval must be 1 minute to 10 years");
                }
            }
            ScheduleTiming::Once { .. } => {}
        }
        Ok(())
    }
    pub fn timing_label(&self) -> String {
        match &self.timing {
            ScheduleTiming::Cron {
                expression,
                timezone,
            } => format!("{expression} · {timezone}"),
            ScheduleTiming::Interval { minutes } => format!("Every {minutes} minutes"),
            ScheduleTiming::Once { at } => format!("Once at {}", at.to_rfc3339()),
        }
    }
    pub fn recalculate(&mut self, now: DateTime<Utc>) -> Result<()> {
        self.next_run = match &self.timing {
            ScheduleTiming::Cron {
                expression,
                timezone,
            } => {
                let cron = CronSchedule::parse(expression)?;
                if cron.is_reboot() {
                    DateTime::<Utc>::MAX_UTC
                } else {
                    cron.next_after(now, timezone)?
                }
            }
            ScheduleTiming::Interval { minutes } => now
                .checked_add_signed(ChronoDuration::minutes(*minutes as i64))
                .context("Scheduled time is out of range")?,
            ScheduleTiming::Once { at } => *at,
        };
        self.validate()
    }
    fn advance(&mut self, now: DateTime<Utc>) -> Result<()> {
        self.last_run = Some(now);
        match &self.timing {
            ScheduleTiming::Cron {
                expression,
                timezone,
            } => {
                let cron = CronSchedule::parse(expression)?;
                if cron.is_reboot() {
                    self.next_run = DateTime::<Utc>::MAX_UTC;
                } else {
                    self.next_run = cron.next_after(now, timezone)?;
                }
            }
            ScheduleTiming::Interval { minutes } => {
                let period = *minutes as i64 * 60;
                let late = now
                    .signed_duration_since(self.next_run)
                    .num_seconds()
                    .max(0);
                self.next_run = self
                    .next_run
                    .checked_add_signed(ChronoDuration::seconds(
                        (late / period + 1)
                            .checked_mul(period)
                            .context("Repeat interval overflow")?,
                    ))
                    .context("Scheduled time is out of range")?;
            }
            ScheduleTiming::Once { .. } => self.enabled = false,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimedRun {
    pub run_id: String,
    pub owner: String,
    pub scheduled_for: DateTime<Utc>,
    pub job: ScheduledJob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: String,
    pub job_id: String,
    pub job_name: String,
    pub scheduled_for: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub state: String,
    pub session_id: Option<String>,
    pub output: String,
    pub error: String,
    pub blocked_actions: Vec<Value>,
}

pub struct SchedulerStore {
    conn: Connection,
}
impl SchedulerStore {
    pub fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join("scheduler.sqlite3");
        if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            bail!("Scheduler database must not be a symlink");
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; CREATE TABLE IF NOT EXISTS scheduler_jobs(id TEXT PRIMARY KEY,name TEXT NOT NULL,enabled INTEGER NOT NULL,next_run INTEGER NOT NULL,data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS scheduler_runs(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,job_name TEXT NOT NULL,scheduled_for INTEGER NOT NULL,started_at TEXT NOT NULL,finished_at TEXT,state TEXT NOT NULL,owner TEXT NOT NULL,lease_until INTEGER NOT NULL,session_id TEXT,output TEXT NOT NULL DEFAULT '',error TEXT NOT NULL DEFAULT '',blocked_actions TEXT NOT NULL DEFAULT '[]',UNIQUE(job_id,scheduled_for)); CREATE INDEX IF NOT EXISTS scheduler_due ON scheduler_jobs(enabled,next_run); CREATE TABLE IF NOT EXISTS scheduler_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);")?;
        Ok(Self { conn })
    }
    pub fn list(&self) -> Result<Vec<ScheduledJob>> {
        let mut statement = self
            .conn
            .prepare("SELECT data FROM scheduler_jobs ORDER BY name,id")?;
        let data = statement
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        data.into_iter()
            .map(|s| serde_json::from_str(&s).context("Invalid stored scheduled job"))
            .collect()
    }
    pub fn upsert(&mut self, job: &ScheduledJob) -> Result<()> {
        job.validate()?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM scheduler_jobs WHERE id<>?1",
            [&job.id],
            |r| r.get(0),
        )?;
        if count >= MAX_JOBS as i64 {
            bail!("At most {MAX_JOBS} scheduled jobs are supported");
        }
        let mut job = job.clone();
        if job.is_reboot() {
            // An edit can have been opened before the daemon claimed the job.
            // Preserve both already-consumed and still-pending boot occurrences.
            let saved: Option<String> = tx
                .query_row(
                    "SELECT data FROM scheduler_jobs WHERE id=?1",
                    [&job.id],
                    |r| r.get(0),
                )
                .optional()?;
            let previous: Option<ScheduledJob> =
                saved.as_deref().map(serde_json::from_str).transpose()?;
            if let Some(old) = previous.filter(ScheduledJob::is_reboot) {
                job.next_run = old.next_run;
                job.last_run = old.last_run;
                job.last_boot_id = old.last_boot_id;
            } else {
                job.next_run = DateTime::<Utc>::MAX_UTC;
                job.last_boot_id = None;
            }
        }
        Self::write_job(&tx, &job)?;
        tx.commit()?;
        Ok(())
    }
    fn write_job(conn: &Connection, job: &ScheduledJob) -> Result<()> {
        conn.execute("INSERT INTO scheduler_jobs(id,name,enabled,next_run,data) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET name=excluded.name,enabled=excluded.enabled,next_run=excluded.next_run,data=excluded.data",params![job.id,job.name,job.enabled,job.next_run.timestamp(),serde_json::to_string(job)?])?;
        Ok(())
    }

    pub fn queue_now(&mut self, id: &str, now: DateTime<Utc>) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let data: String = tx
            .query_row("SELECT data FROM scheduler_jobs WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .context("Scheduled job is no longer available")?;
        let mut job: ScheduledJob = serde_json::from_str(&data)?;
        job.enabled = true;
        job.next_run = now;
        Self::write_job(&tx, &job)?;
        tx.commit()?;
        Ok(())
    }
    /// Deleting a schedule preserves its historical runs and saved conversations.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM scheduler_jobs WHERE id=?1", [id])?;
        Ok(())
    }
    pub fn history(&self, limit: usize) -> Result<Vec<RunRecord>> {
        let mut stmt=self.conn.prepare("SELECT id,job_id,job_name,scheduled_for,started_at,finished_at,state,session_id,output,error,blocked_actions FROM scheduler_runs ORDER BY started_at DESC,id DESC LIMIT ?1")?;
        let rows = stmt
            .query_map([limit.min(1000) as i64], |r| {
                let stamp: i64 = r.get(3)?;
                let blocked: String = r.get(10)?;
                Ok(RunRecord {
                    id: r.get(0)?,
                    job_id: r.get(1)?,
                    job_name: r.get(2)?,
                    scheduled_for: DateTime::from_timestamp(stamp, 0)
                        .map(|t| t.to_rfc3339())
                        .unwrap_or_else(|| stamp.to_string()),
                    started_at: r.get(4)?,
                    finished_at: r.get(5)?,
                    state: r.get(6)?,
                    session_id: r.get(7)?,
                    output: r.get(8)?,
                    error: r.get(9)?,
                    blocked_actions: serde_json::from_str(&blocked).unwrap_or_default(),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
    pub fn claim_due(
        &mut self,
        now: DateTime<Utc>,
        owner: &str,
        lease_seconds: i64,
    ) -> Result<Option<ClaimedRun>> {
        if owner.is_empty() || owner.len() > 128 || !(5..=3600).contains(&lease_seconds) {
            bail!("Invalid scheduler owner or lease");
        }
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("UPDATE scheduler_runs SET state='interrupted',finished_at=?1,error='Worker lease expired; the previous run is not silently repeated' WHERE state='running' AND lease_until<?2",params![now.to_rfc3339(),now.timestamp()])?;
        let data:Option<String>=tx.query_row("SELECT data FROM scheduler_jobs j WHERE enabled=1 AND next_run<=?1 AND NOT EXISTS(SELECT 1 FROM scheduler_runs r WHERE r.job_id=j.id AND r.state='running') ORDER BY next_run,id LIMIT 1",[now.timestamp()],|r|r.get(0)).optional()?;
        let Some(data) = data else {
            tx.commit()?;
            return Ok(None);
        };
        let mut job: ScheduledJob = serde_json::from_str(&data)?;
        job.validate()?;
        let run = ClaimedRun {
            run_id: uuid::Uuid::new_v4().to_string(),
            owner: owner.into(),
            scheduled_for: job.next_run,
            job: job.clone(),
        };
        job.advance(now)?;
        tx.execute(
            "UPDATE scheduler_jobs SET enabled=?1,next_run=?2,data=?3 WHERE id=?4",
            params![
                job.enabled,
                job.next_run.timestamp(),
                serde_json::to_string(&job)?,
                job.id
            ],
        )?;
        let inserted=tx.execute("INSERT OR IGNORE INTO scheduler_runs(id,job_id,job_name,scheduled_for,started_at,state,owner,lease_until) VALUES(?1,?2,?3,?4,?5,'running',?6,?7)",params![run.run_id,run.job.id,run.job.name,run.scheduled_for.timestamp(),now.to_rfc3339(),owner,now.timestamp()+lease_seconds])?;
        tx.commit()?;
        Ok(if inserted == 1 { Some(run) } else { None })
    }
    pub fn heartbeat(
        &mut self,
        run: &ClaimedRun,
        now: DateTime<Utc>,
        lease_seconds: i64,
    ) -> Result<bool> {
        Ok(self.conn.execute("UPDATE scheduler_runs SET lease_until=?1 WHERE id=?2 AND owner=?3 AND state='running' AND lease_until>=?4",params![now.timestamp()+lease_seconds,run.run_id,run.owner,now.timestamp()])?==1)
    }
    pub fn finish(
        &mut self,
        run: &ClaimedRun,
        state: &str,
        session_id: Option<&str>,
        output: &str,
        error: &str,
        blocked: &[Value],
    ) -> Result<()> {
        if !["completed", "failed", "cancelled", "needs_approval"].contains(&state) {
            bail!("Invalid final scheduled-run state");
        }
        let output = truncate(output, 2 * 1024 * 1024);
        let error = truncate(error, 64_000);
        let changed=self.conn.execute("UPDATE scheduler_runs SET state=?1,finished_at=?2,session_id=COALESCE(?3,session_id),output=CASE WHEN length(?4)>0 THEN ?4 ELSE output END,error=?5,blocked_actions=CASE WHEN ?6='[]' THEN blocked_actions ELSE ?6 END WHERE id=?7 AND owner=?8 AND state='running'",params![state,Utc::now().to_rfc3339(),session_id,output,error,serde_json::to_string(blocked)?,run.run_id,run.owner])?;
        if changed != 1 {
            bail!("Scheduled run lost its lease; its stored result was not overwritten");
        }
        Ok(())
    }
    /// Keep the conversation link and denied-action previews durable even when
    /// the daemon is interrupted before it can write a final run result.
    pub fn save_progress(
        &mut self,
        run: &ClaimedRun,
        session_id: &str,
        output: Option<&str>,
        blocked: &[Value],
    ) -> Result<()> {
        let output = output.map(|value| truncate(value, 2 * 1024 * 1024));
        let changed = self.conn.execute(
            "UPDATE scheduler_runs SET session_id=?1,output=COALESCE(?2,output),blocked_actions=?3 WHERE id=?4 AND owner=?5 AND state='running'",
            params![session_id, output, serde_json::to_string(blocked)?, run.run_id, run.owner],
        )?;
        if changed != 1 {
            bail!("Scheduled run lost its lease; its saved progress was not overwritten");
        }
        Ok(())
    }
    pub fn observe_boot(&mut self, boot_id: &str, now: DateTime<Utc>) -> Result<usize> {
        if boot_id.is_empty() || boot_id.len() > 128 {
            bail!("Invalid boot identifier");
        }
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let rows = {
            let mut stmt = tx.prepare("SELECT data FROM scheduler_jobs WHERE enabled=1")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let mut count = 0;
        for data in rows {
            let mut job: ScheduledJob = serde_json::from_str(&data)?;
            if let ScheduleTiming::Cron { expression, .. } = &job.timing {
                if CronSchedule::parse(expression)?.is_reboot()
                    && job.last_boot_id.as_deref() != Some(boot_id)
                {
                    job.last_boot_id = Some(boot_id.into());
                    job.next_run = now;
                    tx.execute(
                        "UPDATE scheduler_jobs SET next_run=?1,data=?2 WHERE id=?3",
                        params![now.timestamp(), serde_json::to_string(&job)?, job.id],
                    )?;
                    count += 1;
                }
            }
        }
        tx.commit()?;
        Ok(count)
    }
    /// Migration is one transaction and idempotent. v0.2 no longer executes the
    /// old JSON scheduler; the JSON file is retained as a recovery copy.
    pub fn migrate_legacy(&mut self, dir: &Path) -> Result<usize> {
        if self
            .conn
            .query_row(
                "SELECT value FROM scheduler_meta WHERE key='legacy_json_imported'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?
            .is_some()
        {
            return Ok(0);
        }
        let legacy = crate::automation::load_schedules(dir)?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut count = 0;
        for old in legacy {
            let timing = if let Some(minutes) = old.interval_minutes {
                ScheduleTiming::Interval { minutes }
            } else {
                ScheduleTiming::Once { at: old.next_run }
            };
            let job = ScheduledJob {
                id: old.id,
                name: old.name,
                prompt: old.prompt,
                model: old.model,
                workspace: old.workspace,
                timing,
                enabled: old.enabled,
                next_run: old.next_run,
                timeout_seconds: 1800,
                last_run: old.last_run,
                last_boot_id: None,
            };
            job.validate()?;
            count+=tx.execute("INSERT OR IGNORE INTO scheduler_jobs(id,name,enabled,next_run,data) VALUES(?1,?2,?3,?4,?5)",params![job.id,job.name,job.enabled,job.next_run.timestamp(),serde_json::to_string(&job)?])?;
        }
        tx.execute(
            "INSERT OR IGNORE INTO scheduler_meta(key,value) VALUES('legacy_json_imported',?1)",
            [Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(count)
    }
}

pub fn migrate_legacy(dir: &Path) -> Result<usize> {
    SchedulerStore::open(dir)?.migrate_legacy(dir)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub pid: u32,
    #[serde(default)]
    pub start_ticks: u64,
    pub owner: String,
    pub updated_at: String,
    pub running: bool,
    pub active_run: Option<String>,
    pub message: String,
}
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    Started(String),
    Progress(String),
    Finished {
        run_id: String,
        state: String,
        session_id: String,
    },
    Stopped,
}

/// Run headlessly until cancelled. The advisory lock prevents two daemon
/// instances; SQL claims also protect against separate/manual queue consumers.
pub fn run_daemon<F>(dir: &Path, cancel: Arc<AtomicBool>, mut notify: F) -> Result<()>
where
    F: FnMut(SchedulerEvent),
{
    fs::create_dir_all(dir)?;
    let _lock = DaemonLock::acquire(dir)?;
    let mut store = SchedulerStore::open(dir)?;
    store.migrate_legacy(dir)?;
    let owner = uuid::Uuid::new_v4().to_string();
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap_or_else(|_| format!("daemon-{owner}"));
    let mut status = DaemonStatus {
        pid: std::process::id(),
        start_ticks: daemon_start_ticks(std::process::id()).unwrap_or(0),
        owner: owner.clone(),
        updated_at: Utc::now().to_rfc3339(),
        running: true,
        active_run: None,
        message: "Waiting for scheduled work".into(),
    };
    write_status(dir, &status)?;
    notify(SchedulerEvent::Started(
        "Background scheduler started".into(),
    ));
    let result = (|| -> Result<()> {
        let mut last_status = Instant::now();
        while !cancel.load(Ordering::Relaxed) {
            store.observe_boot(boot_id.trim(), Utc::now())?;
            if let Some(run) = store.claim_due(Utc::now(), &owner, LEASE_SECONDS)? {
                status.active_run = Some(run.run_id.clone());
                status.message = format!("Running {}", run.job.name);
                status.updated_at = Utc::now().to_rfc3339();
                write_status(dir, &status)?;
                notify(SchedulerEvent::Progress(status.message.clone()));
                let outcome = run_claimed(dir, &mut store, &run, cancel.clone(), |message| {
                    notify(SchedulerEvent::Progress(message))
                });
                match outcome {
                    Ok((state, session_id)) => notify(SchedulerEvent::Finished {
                        run_id: run.run_id,
                        state,
                        session_id,
                    }),
                    Err(error) => {
                        let message = format!("{error:#}");
                        let _ = store.finish(&run, "failed", None, "", &message, &[]);
                        notify(SchedulerEvent::Progress(format!(
                            "{} failed: {message}",
                            run.job.name
                        )));
                    }
                }
                status.active_run = None;
                status.message = "Waiting for scheduled work".into();
                status.updated_at = Utc::now().to_rfc3339();
                write_status(dir, &status)?;
            }
            if last_status.elapsed() > Duration::from_secs(10) {
                status.updated_at = Utc::now().to_rfc3339();
                write_status(dir, &status)?;
                last_status = Instant::now();
            }
            for _ in 0..10 {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        Ok(())
    })();
    status.running = false;
    status.active_run = None;
    status.updated_at = Utc::now().to_rfc3339();
    status.message = if result.is_ok() {
        "Background scheduler stopped".into()
    } else {
        "Background scheduler stopped after an error".into()
    };
    let stopped = write_status(dir, &status).context("Save the final scheduler status");
    notify(SchedulerEvent::Stopped);
    result.and(stopped)
}

pub fn run_claimed<F>(
    dir: &Path,
    scheduler: &mut SchedulerStore,
    run: &ClaimedRun,
    cancel: Arc<AtomicBool>,
    mut notify: F,
) -> Result<(String, String)>
where
    F: FnMut(String),
{
    let mut settings = Settings::load(dir)?;
    settings.automatic_memory = false;
    settings.agents_enabled = false;
    settings.model_routing = false;
    settings.network_tools = false;
    settings.mcp_enabled = false;
    settings.tools_enabled = true;
    settings.validate_endpoint()?;
    let workspace = crate::files::Workspace::new(run.job.workspace.clone())?;
    let engine = runtime::configured_engine(&settings, workspace)?;
    let mut session = Session {
        title: format!("Scheduled · {}", run.job.name),
        project: Some(run.job.workspace.display().to_string()),
        ..Default::default()
    };
    session.entries.push(Entry::new(ChatMessage {
        role: "user".into(),
        content: run.job.prompt.clone(),
        ..Default::default()
    }));
    let store = Store::open(dir)?;
    store.save(&mut session)?;
    scheduler.save_progress(run, &session.id, None, &[])?;
    let messages=vec![ChatMessage{role:"system".into(),content:format!("{}\nThis is background scheduled work. Read-only workspace tools are available. File changes, shell commands, network and external integrations require a person to review them and will be denied. Do not claim a denied action happened.",settings.system_prompt),..Default::default()},session.entries[0].message.clone()];
    let (tx, rx) = mpsc::channel();
    let worker_cancel = Arc::new(AtomicBool::new(false));
    let _worker_guard = CancelWorkerOnDrop(worker_cancel.clone());
    runtime::spawn_chat(
        ChatJob {
            settings,
            messages,
            workspace: Some(run.job.workspace.clone()),
            model: run.job.model.clone(),
            engine: Some(engine),
            retrieval_path: dir.join("knowledge.sqlite3"),
        },
        tx,
        worker_cancel.clone(),
    );
    let deadline = Instant::now() + Duration::from_secs(run.job.timeout_seconds);
    let mut heartbeat = Instant::now();
    let mut stop_started = None;
    let mut error = String::new();
    let mut blocked = Vec::new();
    let mut stats = Value::Null;
    let mut partial = String::new();
    let mut thinking = String::new();
    let mut finished = false;
    loop {
        if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
            worker_cancel.store(true, Ordering::Relaxed);
            stop_started.get_or_insert_with(Instant::now);
            if Instant::now() >= deadline && error.is_empty() {
                error = "Scheduled run exceeded its configured timeout".into();
            }
        }
        if stop_started.is_some_and(|t: Instant| t.elapsed() > Duration::from_secs(10)) {
            break;
        }
        if heartbeat.elapsed() > Duration::from_secs(10) {
            if !scheduler.heartbeat(run, Utc::now(), LEASE_SECONDS)? {
                worker_cancel.store(true, Ordering::Relaxed);
                bail!("Scheduled worker lost its lease");
            }
            heartbeat = Instant::now();
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(ChatEvent::Begin(_)) => {
                partial.clear();
                thinking.clear();
                stats = Value::Null;
            }
            Ok(ChatEvent::Content(s)) => {
                if partial.len() < 2 * 1024 * 1024 {
                    partial.push_str(&s);
                }
            }
            Ok(ChatEvent::Thinking(s)) => {
                if thinking.len() < 2 * 1024 * 1024 {
                    thinking.push_str(&s);
                }
            }
            Ok(ChatEvent::Stats(value)) => stats = value,
            Ok(ChatEvent::BackgroundFailure { task, reason }) => {
                if let Some(entry) = session
                    .entries
                    .iter_mut()
                    .rev()
                    .take_while(|e| e.message.role != "user")
                    .find(|e| e.message.role == "assistant")
                {
                    if !entry.stats.is_object() {
                        entry.stats = json!({});
                    }
                    let failures = entry
                        .stats
                        .as_object_mut()
                        .unwrap()
                        .entry("background_failures")
                        .or_insert_with(|| json!([]));
                    if let Some(failures) = failures.as_array_mut() {
                        if failures.len() < 8 {
                            failures.push(json!({"task": task, "reason": reason}));
                        }
                    }
                    store.save(&mut session)?;
                }
            }
            Ok(ChatEvent::Superseded(_)) => {
                partial.clear();
                thinking.clear();
                stats = Value::Null;
            }
            Ok(ChatEvent::Complete(message)) => {
                let output = message.content.clone();
                let mut entry = Entry::new(message);
                entry.model = run.job.model.clone();
                entry.stats = stats.clone();
                session.entries.push(entry);
                partial.clear();
                thinking.clear();
                store.save(&mut session)?;
                scheduler.save_progress(run, &session.id, Some(&output), &blocked)?;
            }
            Ok(ChatEvent::UnexecutedToolText {
                message,
                names,
                retrying,
            }) => {
                let mut entry = Entry::new(message);
                entry.model = run.job.model.clone();
                entry.stats = stats.clone();
                if !entry.stats.is_object() {
                    entry.stats = json!({});
                }
                entry.stats["unexecuted_tool_text"] = json!(true);
                entry.stats["unexecuted_tool_names"] = json!(&names);
                entry.display = Some(format!(
                    "{} model tool request(s) were printed as text and not executed{}.",
                    names.len(),
                    if retrying {
                        "; retrying native protocol"
                    } else {
                        ""
                    }
                ));
                session.entries.push(entry);
                partial.clear();
                thinking.clear();
                store.save(&mut session)?;
                notify(format!(
                    "{} unexecuted text-form tool request(s)",
                    names.len()
                ));
            }
            Ok(ChatEvent::Append(entry)) => {
                session.entries.push(entry);
                store.save(&mut session)?;
            }
            Ok(ChatEvent::Evidence { context, sources }) => {
                if let Some(entry) = session
                    .entries
                    .iter_mut()
                    .rev()
                    .find(|e| e.message.role == "user")
                {
                    entry.context = context;
                    entry.sources.extend(sources);
                }
            }
            Ok(ChatEvent::Approval { action, reply }) => {
                blocked.push(json!({"title":action.title,"description":action.description,"details":action.details}));
                let _ = reply.send(false);
                scheduler.save_progress(run, &session.id, None, &blocked)?;
                notify("Background action needs interactive approval; preview was saved".into());
            }
            Ok(ChatEvent::Audit { action, details }) => {
                store.audit(&session.id, &action, &details)?
            }
            Ok(ChatEvent::Status(message)) => notify(message),
            // Live activity is for the desktop console; scheduled runs keep the
            // complete tool results that arrive through Append.
            Ok(ChatEvent::ToolStarted { .. } | ChatEvent::ToolOutput { .. })
            | Ok(ChatEvent::ToolFinished { .. }) => {}
            Ok(ChatEvent::Error(message)) => error = message,
            Ok(ChatEvent::Finished) => {
                finished = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if error.is_empty() {
                    error = "Scheduled inference worker disconnected".into();
                }
                break;
            }
        }
    }
    worker_cancel.store(true, Ordering::Relaxed);
    if !partial.is_empty() || !thinking.is_empty() {
        let mut entry = Entry::new(ChatMessage {
            role: "assistant".into(),
            content: partial,
            thinking,
            ..Default::default()
        });
        entry.model = run.job.model.clone();
        entry.stats = json!({"incomplete":true});
        session.entries.push(entry);
    }
    store.save(&mut session)?;
    let final_answer = session
        .entries
        .iter()
        .rev()
        .find(|entry| entry.message.role == "assistant");
    if error.is_empty() {
        if let Some(entry) = final_answer.filter(|entry| entry.stats["incomplete"] == true) {
            error = entry.stats["failure_reason"]
                .as_str()
                .unwrap_or("The model stopped before completing its answer")
                .to_owned();
        }
    }
    let state = if cancel.load(Ordering::Relaxed) {
        "cancelled"
    } else if !blocked.is_empty() {
        "needs_approval"
    } else if !error.is_empty() || !finished {
        "failed"
    } else {
        "completed"
    };
    if !blocked.is_empty() {
        let explanation="One or more requested actions were not executed because they require interactive approval. Their previews are stored with this run.";
        if error.is_empty() {
            error = explanation.into();
        } else {
            error.push_str(&format!("\n{explanation}"));
        }
    }
    let output = final_answer
        .map(|entry| entry.message.content.as_str())
        .unwrap_or("");
    scheduler.finish(run, state, Some(&session.id), output, &error, &blocked)?;
    Ok((state.into(), session.id))
}

struct CancelWorkerOnDrop(Arc<AtomicBool>);
impl Drop for CancelWorkerOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

struct DaemonLock {
    file: std::fs::File,
}
impl DaemonLock {
    fn acquire(dir: &Path) -> Result<Self> {
        let path = dir.join("scheduler.lock");
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let file = options.open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                bail!("A background scheduler is already running for this data directory");
            }
        }
        Ok(Self { file })
    }
}
impl Drop for DaemonLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe {
                libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
            }
        }
    }
}

pub fn daemon_status(dir: &Path) -> Result<Option<DaemonStatus>> {
    let path = dir.join("scheduler-status.json");
    if !path.exists() {
        return Ok(None);
    }
    if fs::symlink_metadata(&path)?.file_type().is_symlink() {
        bail!("Scheduler status must not be a symlink");
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(64_001).read_to_end(&mut bytes)?;
    if bytes.len() > 64_000 {
        bail!("Scheduler status is oversized");
    }
    let mut status: DaemonStatus = serde_json::from_slice(&bytes)?;
    if !daemon_identity_matches(&status, dir) {
        status.running = false;
    }
    Ok(Some(status))
}

static CLI_STOP: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
extern "C" fn daemon_signal(_signal: libc::c_int) {
    CLI_STOP.store(true, Ordering::Relaxed);
}

/// CLI entry point with graceful SIGINT/SIGTERM handling. This is called only
/// for --daemon; it does not change signal handling in the graphical process.
pub fn run_daemon_cli(dir: &Path) -> Result<()> {
    CLI_STOP.store(false, Ordering::Relaxed);
    #[cfg(unix)]
    unsafe {
        libc::signal(
            libc::SIGINT,
            daemon_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            daemon_signal as *const () as libc::sighandler_t,
        );
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let watcher = cancel.clone();
    std::thread::spawn(move || {
        while !watcher.load(Ordering::Relaxed) {
            if CLI_STOP.load(Ordering::Relaxed) {
                watcher.store(true, Ordering::Relaxed);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    let result = run_daemon(dir, cancel.clone(), |event| {
        println!("{} {event:?}", Utc::now().to_rfc3339())
    });
    cancel.store(true, Ordering::Relaxed);
    result
}

/// Portable-to-Linux fallback when a user systemd session is unavailable. The
/// process runs the same durable scheduler and keeps running after GUI exit.
pub fn start_background(dir: &Path) -> Result<DaemonStatus> {
    if let Some(status) = daemon_status(dir)?.filter(|s| s.running) {
        return Ok(status);
    }
    fs::create_dir_all(dir)?;
    let dir = dir.canonicalize()?;
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let log = options.open(dir.join("scheduler.log"))?;
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .arg("--daemon")
        .arg("--data-dir")
        .arg(&dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log.try_clone()?))
        .stderr(std::process::Stdio::from(log));
    crate::integrations::configure_process_group(&mut command);
    let mut child = command
        .spawn()
        .context("Start background scheduler process")?;
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if let Some(exit) = child.try_wait()? {
            bail!("Background scheduler exited ({exit}); inspect scheduler.log");
        }
        if let Some(status) = daemon_status(&dir)?.filter(|s| s.running && s.pid == child.id()) {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return Ok(status);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    crate::integrations::kill_process_group(&mut child);
    let _ = child.wait();
    bail!("Background scheduler did not report ready within eight seconds")
}

pub fn stop_background(dir: &Path) -> Result<()> {
    let Some(status) = daemon_status(dir)?.filter(|s| s.running) else {
        return Ok(());
    };
    #[cfg(unix)]
    unsafe {
        libc::kill(status.pid as i32, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline && daemon_identity_matches(&status, dir) {
        std::thread::sleep(Duration::from_millis(100));
    }
    if daemon_identity_matches(&status, dir) {
        #[cfg(unix)]
        unsafe {
            if libc::getpgid(status.pid as i32) == status.pid as i32 {
                libc::kill(-(status.pid as i32), libc::SIGKILL);
            } else {
                libc::kill(status.pid as i32, libc::SIGKILL);
            }
        }
    }
    Ok(())
}

fn daemon_identity_matches(status: &DaemonStatus, dir: &Path) -> bool {
    if !status.running
        || status.pid < 2
        || status.pid > i32::MAX as u32
        || daemon_start_ticks(status.pid).ok() != Some(status.start_ticks)
    {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        let Ok(cmdline) = fs::read(format!("/proc/{}/cmdline", status.pid)) else {
            return false;
        };
        if !cmdline.split(|b| *b == 0).any(|arg| arg == b"--daemon") {
            return false;
        }
        // The daemon holds this lock for its whole lifetime, including when
        // it resolved the directory through the environment or XDG defaults.
        use std::os::unix::fs::MetadataExt;
        let Ok(lock) = fs::metadata(dir.join("scheduler.lock")) else {
            return false;
        };
        let Ok(descriptors) = fs::read_dir(format!("/proc/{}/fd", status.pid)) else {
            return false;
        };
        descriptors.flatten().any(|fd| {
            fs::metadata(fd.path())
                .is_ok_and(|open| open.dev() == lock.dev() && open.ino() == lock.ino())
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = dir;
        false
    }
}
fn daemon_start_ticks(pid: u32) -> Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    stat.rsplit_once(')')
        .context("Invalid daemon process status")?
        .1
        .split_whitespace()
        .nth(19)
        .context("Missing process start time")?
        .parse()
        .context("Invalid process start time")
}

fn write_status(dir: &Path, status: &DaemonStatus) -> Result<()> {
    atomic_write(
        &dir.join("scheduler-status.json"),
        &serde_json::to_vec_pretty(status)?,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceAction {
    Start,
    Stop,
    Restart,
    Enable,
    Disable,
    Status,
}
impl ServiceAction {
    pub fn args(self) -> Vec<String> {
        let action = match self {
            Self::Start => vec!["start"],
            Self::Stop => vec!["stop"],
            Self::Restart => vec!["restart"],
            Self::Enable => vec!["enable", "--now"],
            Self::Disable => vec!["disable", "--now"],
            Self::Status => vec!["status", "--no-pager"],
        };
        std::iter::once("--user".into())
            .chain(action.into_iter().map(String::from))
            .chain(std::iter::once(SERVICE_NAME.into()))
            .collect()
    }
    pub fn preview(self) -> String {
        format!("systemctl {}", self.args().join(" "))
    }
}

pub fn service_unit(executable: &Path, data_dir: &Path) -> Result<String> {
    if !executable.is_absolute() || !data_dir.is_absolute() {
        bail!("Service executable and data directory must be absolute paths");
    }
    let executable = systemd_argument(&executable.to_string_lossy())?;
    let data_dir = systemd_argument(&data_dir.to_string_lossy())?;
    Ok(format!("[Unit]\nDescription=AI Workbench background scheduler\nAfter=network-online.target\n\n[Service]\nType=simple\nExecStart={executable} --daemon --data-dir {data_dir}\nRestart=on-failure\nRestartSec=5\nKillMode=control-group\nTimeoutStopSec=20\nNoNewPrivileges=true\n\n[Install]\nWantedBy=default.target\n"))
}

pub fn install_user_service(executable: &Path, data_dir: &Path) -> Result<PathBuf> {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .context("Could not find the user configuration directory")?;
    let unit = service_unit(executable, data_dir)?;
    let directory = config.join("systemd/user");
    fs::create_dir_all(&directory)?;
    let path = directory.join(SERVICE_NAME);
    if path.exists() {
        fs::copy(&path, directory.join(format!("{SERVICE_NAME}.previous")))?;
    }
    atomic_write(&path, unit.as_bytes())?;
    let spec = crate::integrations::CommandSpec {
        program: "systemctl".into(),
        args: vec!["--user".into(), "daemon-reload".into()],
        timeout_secs: 30,
        ..Default::default()
    };
    let result = crate::integrations::run_command(&spec, data_dir)?;
    if result.exit_code != Some(0) || result.timed_out || result.cancelled {
        bail!(
            "Service unit saved to {}, but systemd reload failed: {}",
            path.display(),
            result.stderr
        );
    }
    Ok(path)
}

pub fn service_control(
    action: ServiceAction,
    data_dir: &Path,
) -> Result<crate::integrations::CommandResult> {
    let spec = crate::integrations::CommandSpec {
        program: "systemctl".into(),
        args: action.args(),
        timeout_secs: 30,
        ..Default::default()
    };
    crate::integrations::run_command(&spec, data_dir)
}

pub fn systemd_argument(text: &str) -> Result<String> {
    if text.contains(['\0', '\n', '\r']) {
        bail!("Service paths cannot contain NUL or line breaks");
    }
    Ok(format!(
        "\"{}\"",
        text.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
            .replace('$', "$$")
    ))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        bail!("Refusing to replace a symlinked state file");
    }
    crate::persistence::write_atomic(path, bytes)
}
fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.into()
    } else {
        let mut end = max;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}\n[Stored output truncated]", &text[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn utc(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .unwrap()
            .with_timezone(&Utc)
    }
    #[cfg(unix)]
    #[test]
    fn state_writes_keep_symlinks_and_report_post_rename_durability_errors() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("status");
        let link = dir.path().join("status-link");
        symlink(&file, &link).unwrap();
        assert!(atomic_write(&link, b"data").is_err());
        assert!(!file.exists());
        atomic_write(&file, b"old").unwrap();
        assert!(atomic_write(&link, b"replacement").is_err());
        assert_eq!(fs::read(&file).unwrap(), b"old");

        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o300)).unwrap();
        // Privileged test runners can open the directory despite its mode.
        let directory_readable = File::open(dir.path()).is_ok();
        let result = atomic_write(&file, b"new");
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        if directory_readable {
            result.unwrap();
        } else {
            assert!(result.unwrap_err().to_string().contains("was replaced"));
        }
        assert_eq!(fs::read(&file).unwrap(), b"new");
    }

    #[test]
    fn cron_lists_steps_names_and_posix_day_semantics() {
        let schedule = CronSchedule::parse("*/15 9-17/2 * JAN,MAR MON-FRI").unwrap();
        assert!(schedule
            .matches(utc("2026-03-02T14:15:00Z"), "America/New_York")
            .unwrap());
        assert!(!schedule
            .matches(utc("2026-03-02T15:15:00Z"), "America/New_York")
            .unwrap());
        let or = CronSchedule::parse("0 9 13 * MON").unwrap();
        assert!(or.matches(utc("2026-11-13T09:00:00Z"), "UTC").unwrap());
        assert!(or.matches(utc("2026-11-09T09:00:00Z"), "UTC").unwrap());
        assert!(CronSchedule::parse("0 0 * * 7")
            .unwrap()
            .matches(utc("2026-11-08T00:00:00Z"), "UTC")
            .unwrap());
        for invalid in [
            "* * * *",
            "*/0 * * * *",
            "60 * * * *",
            "0 0 0 * *",
            "0 0 * 13 *",
            "0 0 * * FUNDAY",
            "0 0 31 FEB *",
        ] {
            let parsed = CronSchedule::parse(invalid);
            assert!(
                parsed.is_err()
                    || parsed
                        .unwrap()
                        .next_after(utc("2026-01-01T00:00:00Z"), "UTC")
                        .is_err(),
                "{invalid}"
            );
        }
    }
    #[test]
    fn cron_skips_nonexistent_dst_and_keeps_both_repeated_times() {
        let spring = CronSchedule::parse("30 2 * * *").unwrap();
        assert_eq!(
            spring
                .next_after(utc("2026-03-08T05:00:00Z"), "America/New_York")
                .unwrap(),
            utc("2026-03-09T06:30:00Z")
        );
        let fall = CronSchedule::parse("30 1 * * *").unwrap();
        let a = fall
            .next_after(utc("2026-11-01T04:00:00Z"), "America/New_York")
            .unwrap();
        let b = fall.next_after(a, "America/New_York").unwrap();
        assert_eq!(a, utc("2026-11-01T05:30:00Z"));
        assert_eq!(b, utc("2026-11-01T06:30:00Z"));
        assert!(CronSchedule::parse("@reboot").unwrap().is_reboot());
    }
    #[test]
    fn transactional_claims_leases_and_history_are_durable() {
        let dir = tempfile::tempdir().unwrap();
        let now = utc("2026-11-07T10:00:00Z");
        let mut a = SchedulerStore::open(dir.path()).unwrap();
        let mut b = SchedulerStore::open(dir.path()).unwrap();
        let mut job = ScheduledJob::cron(
            "review",
            "Inspect files",
            "test",
            dir.path().into(),
            "* * * * *",
            "UTC",
            now,
        )
        .unwrap();
        job.next_run = now;
        a.upsert(&job).unwrap();
        let run = a.claim_due(now, "worker-a", 30).unwrap().unwrap();
        assert!(b.claim_due(now, "worker-b", 30).unwrap().is_none());
        assert!(!b
            .heartbeat(
                &ClaimedRun {
                    owner: "worker-b".into(),
                    ..run.clone()
                },
                now,
                30
            )
            .unwrap());
        a.finish(&run, "completed", Some("session"), "real result", "", &[])
            .unwrap();
        let rows = b.history(10).unwrap();
        assert_eq!(rows[0].state, "completed");
        assert_eq!(rows[0].output, "real result");
        let next = b
            .claim_due(now + ChronoDuration::minutes(1), "worker-b", 10)
            .unwrap()
            .unwrap();
        b.save_progress(
            &next,
            "interrupted-session",
            Some("Partial review"),
            &[json!({"title":"Requested file edit","details":{"path":"note.txt"}})],
        )
        .unwrap();
        assert!(a
            .claim_due(now + ChronoDuration::seconds(80), "worker-c", 10)
            .unwrap()
            .is_none());
        let recovered = a.history(1).unwrap().remove(0);
        assert_eq!(recovered.state, "interrupted");
        assert_eq!(recovered.session_id.as_deref(), Some("interrupted-session"));
        assert_eq!(recovered.output, "Partial review");
        assert_eq!(recovered.blocked_actions.len(), 1);
        assert!(b.finish(&next, "completed", None, "late", "", &[]).is_err());
        assert!(b.save_progress(&next, "late", None, &[]).is_err());
    }
    #[test]
    fn legacy_migration_and_boot_jobs_do_not_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let now = Utc::now();
        let old = crate::automation::Schedule {
            id: "legacy".into(),
            name: "old".into(),
            prompt: "review".into(),
            model: "test".into(),
            workspace: dir.path().into(),
            next_run: now,
            interval_minutes: Some(5),
            enabled: true,
            last_run: None,
        };
        crate::automation::save_schedules(dir.path(), &[old]).unwrap();
        let mut store = SchedulerStore::open(dir.path()).unwrap();
        assert_eq!(store.migrate_legacy(dir.path()).unwrap(), 1);
        assert_eq!(store.migrate_legacy(dir.path()).unwrap(), 0);
        assert!(matches!(
            store.list().unwrap()[0].timing,
            ScheduleTiming::Interval { minutes: 5 }
        ));
        let boot = ScheduledJob::cron(
            "boot",
            "review",
            "test",
            dir.path().into(),
            "@reboot",
            "UTC",
            now,
        )
        .unwrap();
        store.upsert(&boot).unwrap();
        assert_eq!(store.observe_boot("boot-1", now).unwrap(), 1);
        assert_eq!(store.observe_boot("boot-1", now).unwrap(), 0);
        assert_eq!(store.observe_boot("boot-2", now).unwrap(), 1);
    }
    #[test]
    fn editing_a_reboot_job_preserves_pending_and_consumed_occurrences() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SchedulerStore::open(dir.path()).unwrap();
        let now = Utc::now();
        let mut edit = ScheduledJob::cron(
            "job",
            "prompt",
            "model",
            dir.path().into(),
            "@reboot",
            "UTC",
            now,
        )
        .unwrap();
        store.upsert(&edit).unwrap();
        store.observe_boot("boot-1", now).unwrap();
        edit.name = "renamed".into();
        edit.recalculate(now).unwrap();
        store.upsert(&edit).unwrap();
        let run = store.claim_due(now, "worker", 90).unwrap().unwrap();
        assert_eq!(run.job.name, "renamed");
        store
            .finish(&run, "completed", None, "done", "", &[])
            .unwrap();
        let later = now + ChronoDuration::minutes(1);
        edit.prompt = "new prompt".into();
        edit.recalculate(later).unwrap();
        store.upsert(&edit).unwrap();
        assert_eq!(store.observe_boot("boot-1", later).unwrap(), 0);
        assert!(store.claim_due(later, "worker", 90).unwrap().is_none());
        assert_eq!(store.observe_boot("boot-2", later).unwrap(), 1);
        let run = store.claim_due(later, "worker", 90).unwrap().unwrap();
        assert_eq!(run.job.prompt, "new prompt");
        store
            .finish(&run, "completed", None, "done", "", &[])
            .unwrap();
        let manual = later + ChronoDuration::minutes(1);
        store.queue_now(&edit.id, manual).unwrap();
        assert!(store.claim_due(manual, "worker", 90).unwrap().is_some());
    }

    #[test]
    fn service_unit_quotes_paths_and_disallows_line_injection() {
        let unit = service_unit(
            Path::new("/tmp/my app/100% tool"),
            Path::new("/tmp/data $d"),
        )
        .unwrap();
        assert!(unit.contains("\"/tmp/my app/100%% tool\""));
        assert!(unit.contains("--daemon --data-dir \"/tmp/data $$d\""));
        assert!(systemd_argument("bad\nExecStart=bad").is_err());
        assert_eq!(
            ServiceAction::Enable.args(),
            ["--user", "enable", "--now", SERVICE_NAME]
        );
    }

    #[test]
    fn background_action_is_saved_as_needing_approval_not_success() {
        check_scheduled_tool_exchange(false);
    }

    #[test]
    fn scheduled_model_failure_after_read_keeps_partial_output_and_failure_reason() {
        check_scheduled_tool_exchange(true);
    }

    fn check_scheduled_tool_exchange(fail_after_read: bool) {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        // Observe the protocol independently of the assertion: if the mock
        // receives too few requests, show the scheduler's actual error rather
        // than only a fixture-thread timeout. The old ten-second fixed window
        // could expire during a heavily loaded, parallel test run.
        let request_log = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let server_log = request_log.clone();
        let stop_server = Arc::new(AtomicBool::new(false));
        let server_stop = stop_server.clone();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut chat_calls = 0;
            let mut requests = 0;
            while requests < 3 && !server_stop.load(Ordering::Relaxed) {
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(connection) => break connection,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            if server_stop.load(Ordering::Relaxed) {
                                return;
                            }
                            assert!(
                                Instant::now() < deadline,
                                "Scheduler protocol fixture timed out after {} request(s): {:?}",
                                requests,
                                server_log.lock().unwrap()
                            );
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("{error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0u8; 4096];
                let mut header_end = None;
                loop {
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    if header_end.is_none() {
                        header_end = bytes
                            .windows(4)
                            .position(|w| w == b"\r\n\r\n")
                            .map(|p| p + 4);
                    }
                    if let Some(end) = header_end {
                        let headers = String::from_utf8_lossy(&bytes[..end]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                line.split_once(':')
                                    .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                                    .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + length {
                            break;
                        }
                    }
                }
                let end = header_end.unwrap();
                let path = String::from_utf8_lossy(&bytes[..end])
                    .lines()
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .to_string();
                server_log.lock().unwrap().push(path.clone());
                let body = if path == "/api/show" {
                    json!({"capabilities":["completion","tools"],"details":{}})
                } else if path == "/api/chat" {
                    let payload: Value = serde_json::from_slice(&bytes[end..]).unwrap();
                    chat_calls += 1;
                    if fail_after_read && chat_calls == 1 {
                        json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"note.txt"}}}]},"done":true})
                    } else if fail_after_read {
                        assert!(payload["messages"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|m| m["role"] == "tool"
                                && m["content"].as_str().unwrap_or("").contains("original")));
                        json!({"error":"fixture inference failure after read"})
                    } else if chat_calls == 1 {
                        json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"write_file","arguments":{"path":"note.txt","content":"changed"}}}]},"done":true})
                    } else {
                        assert!(payload["messages"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|m| m["role"] == "tool"
                                && m["content"].as_str().unwrap_or("").contains("denied")));
                        json!({"message":{"role":"assistant","content":"The edit needs interactive approval."},"done":true})
                    }
                } else {
                    panic!("Unexpected API request {path}")
                };
                let body = format!("{body}\n");
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
                requests += 1;
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("note.txt"), "original").unwrap();
        let settings = Settings {
            base_url: endpoint,
            model: "test".into(),
            // The mock answers a fixed script; a tool-call probe would consume it.
            native_tool_probe: false,
            ..Default::default()
        };
        settings.save(dir.path()).unwrap();
        let now = Utc::now();
        let mut job = ScheduledJob::cron(
            "edit request",
            if fail_after_read {
                "Read note.txt and describe its contents"
            } else {
                "Change note.txt"
            },
            "test",
            workspace.path().into(),
            "* * * * *",
            "UTC",
            now,
        )
        .unwrap();
        job.next_run = now;
        let mut scheduler = SchedulerStore::open(dir.path()).unwrap();
        scheduler.upsert(&job).unwrap();
        let run = scheduler.claim_due(now, "worker", 90).unwrap().unwrap();
        let (state, session) = run_claimed(
            dir.path(),
            &mut scheduler,
            &run,
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        // Stop and join the fixture even when the assertion fails. This avoids
        // a detached mock thread masking the real scheduler failure with its
        // own timeout panic.
        stop_server.store(true, Ordering::Relaxed);
        server.join().unwrap();
        let requests = request_log.lock().unwrap().clone();
        let history = scheduler.history(1).unwrap();
        assert_eq!(
            state,
            if fail_after_read {
                "failed"
            } else {
                "needs_approval"
            },
            "scheduler error: {}; observed API paths: {requests:?}; saved blocked actions: {:?}",
            history[0].error, history[0].blocked_actions
        );
        assert_eq!(requests, vec!["/api/show", "/api/chat", "/api/chat"]);
        assert_eq!(
            fs::read_to_string(workspace.path().join("note.txt")).unwrap(),
            "original"
        );
        if fail_after_read {
            assert!(
                history[0]
                    .error
                    .contains("fixture inference failure after read"),
                "{}",
                history[0].error
            );
            assert!(!history[0].output.is_empty());
            assert!(history[0].blocked_actions.is_empty());
        } else {
            assert_eq!(history[0].blocked_actions.len(), 1);
        }
        assert_eq!(history[0].session_id.as_deref(), Some(session.as_str()));
        assert!(Store::open(dir.path())
            .unwrap()
            .load(&session)
            .unwrap()
            .entries
            .iter()
            .any(|e| e.message.role == "tool"));
    }
}
