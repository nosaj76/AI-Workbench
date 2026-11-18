//! Long-term Q&A library.
//!
//! Every answer is kept as an organised, human-readable Markdown file:
//! `qa-library/<topic>/<question>.md`, with a YAML-style header, the answer,
//! its sources and any previous answers, plus a generated `INDEX.md` grouped by
//! topic. The files are the durable record: they can be read, searched (grep,
//! an editor, Obsidian) and backed up without Workbench.
//!
//! SQLite (`qa_cache.sqlite3`) is only a fast index for matching reworded
//! questions and tracking refresh dates. Each Markdown file ends with its full
//! machine-readable record in an HTML comment, so a lost index is rebuilt from
//! the files.
//!
//! Answers are never deleted automatically. Factual answers are kept as they
//! are; answers that change over time are refreshed when asked after their
//! refresh date, and the replaced answer is kept under "Previous answers".
use anyhow::{bail, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

pub const LIBRARY_DIR: &str = "qa-library";
/// Default refresh period for answers that change over time.
pub const DEFAULT_REFRESH_DAYS: u32 = 30;
/// Refresh period for time-sensitive answers (capped by the configured period).
pub const TIME_SENSITIVE_DAYS: u32 = 3;
/// Reworded questions whose significant words overlap at least this much
/// reuse the saved answer.
const SIMILARITY_THRESHOLD: f64 = 0.85;
/// Previous answers kept in each file.
const MAX_HISTORY: usize = 5;
const RECORD_MARKER: &str = "<!-- qa-record";

/// How likely the correct answer is to change, which decides refreshing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerKind {
    /// Definitions, maths, history, fixed procedures: kept, never refreshed.
    Factual,
    /// Software usage, APIs, recommendations: refreshed after a period.
    Changing,
    /// Latest versions, prices, current events: refreshed after a few days.
    TimeSensitive,
}

impl AnswerKind {
    pub const ALL: [AnswerKind; 3] = [Self::Factual, Self::Changing, Self::TimeSensitive];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Factual => "factual",
            Self::Changing => "changing",
            Self::TimeSensitive => "time_sensitive",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "factual" | "stable" | "fact" => Some(Self::Factual),
            "changing" | "changes" | "evolving" => Some(Self::Changing),
            "time_sensitive" | "volatile" | "current" => Some(Self::TimeSensitive),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Factual => "Factual",
            Self::Changing => "Changes over time",
            Self::TimeSensitive => "Time-sensitive",
        }
    }

    /// Days until a refresh is due; None means never.
    pub fn refresh_days(self, changing_days: u32) -> Option<u32> {
        let changing_days = changing_days.clamp(1, 3650);
        match self {
            Self::Factual => None,
            Self::Changing => Some(changing_days),
            Self::TimeSensitive => Some(TIME_SENSITIVE_DAYS.min(changing_days)),
        }
    }
}

/// Questions whose answers obviously depend on when they are asked.
pub fn obviously_time_sensitive(question: &str) -> bool {
    let lower = question.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has = |w: &str| words.contains(&w);
    [
        "latest",
        "newest",
        "current",
        "currently",
        "today",
        "tonight",
        "yesterday",
        "tomorrow",
        "now",
        "price",
        "prices",
        "cost",
        "stock",
        "weather",
        "news",
        "upcoming",
        "recent",
        "recently",
        "score",
    ]
    .iter()
    .any(|w| has(w))
        || lower.contains("this week")
        || lower.contains("this month")
        || lower.contains("this year")
        || lower.contains("release date")
        || words
            .iter()
            .any(|w| w.len() == 4 && w.starts_with("20") && w.chars().all(|c| c.is_ascii_digit()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QaSource {
    pub id: usize,
    pub title: String,
    pub url: String,
    pub site: String,
    /// "stack_exchange_api", "text_browser:<name>" or "search_snippet".
    pub method: String,
    pub content: String,
    /// Topic tags from the source site (Stack Exchange), when available.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl QaSource {
    pub fn via(&self) -> String {
        match self.method.as_str() {
            "stack_exchange_api" => "read via the Stack Exchange API".to_owned(),
            "search_snippet" => "search snippet only (page not readable)".to_owned(),
            other => format!(
                "read with {}",
                other.strip_prefix("text_browser:").unwrap_or(other)
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreviousAnswer {
    pub saved_at: String,
    pub model: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedQa {
    pub id: String,
    pub question: String,
    /// One or two sentences: the compact form given to chat models, so a
    /// reused answer costs little model context.
    #[serde(default)]
    pub summary: String,
    /// For facts: the question in a few words.
    #[serde(default)]
    pub short_question: String,
    /// For facts: a short explanation of the one-sentence answer.
    #[serde(default)]
    pub explanation: String,
    pub answer: String,
    pub sources: Vec<QaSource>,
    pub model: String,
    pub kind: AnswerKind,
    pub topic: String,
    pub saved_at: String,
    /// When a refresh is due; None for factual answers (never).
    pub refresh_after: Option<String>,
    #[serde(default)]
    pub history: Vec<PreviousAnswer>,
    /// Markdown file, relative to the library directory.
    #[serde(default)]
    pub path: String,
    /// Times reused (index only; not written to the file).
    #[serde(skip)]
    pub hits: u64,
    /// 1.0 for the same question; lower for a reworded match (lookup only).
    #[serde(skip)]
    pub similarity: f64,
}

impl CachedQa {
    pub fn needs_refresh(&self) -> bool {
        let now = chrono::Utc::now().to_rfc3339();
        self.refresh_after
            .as_deref()
            .is_some_and(|due| due <= now.as_str())
    }

    /// e.g. "factual · kept, never refreshed" or "refresh due 2026-12-08".
    pub fn refresh_label(&self) -> String {
        match &self.refresh_after {
            None => "factual · kept, never refreshed".to_owned(),
            Some(_) if self.needs_refresh() => format!("{} · due for refresh", self.kind.label()),
            Some(due) => format!("{} · refresh after {}", self.kind.label(), short_date(due)),
        }
    }
}

pub fn short_date(rfc3339: &str) -> String {
    rfc3339.get(..10).unwrap_or(rfc3339).to_owned()
}

fn refresh_after(saved_at: &str, kind: AnswerKind, changing_days: u32) -> Option<String> {
    let saved = chrono::DateTime::parse_from_rfc3339(saved_at)
        .map(|t| t.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());
    kind.refresh_days(changing_days)
        .map(|days| (saved + chrono::Duration::days(i64::from(days))).to_rfc3339())
}

// ------------------------------------------------------------ matching ----

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did", "how", "what",
    "why", "when", "where", "which", "who", "can", "could", "should", "would", "will", "i", "my",
    "me", "we", "our", "you", "your", "to", "of", "in", "on", "for", "with", "and", "or", "it",
    "its", "this", "that", "from", "by", "as", "at", "into", "about", "there", "any", "way",
    "ways", "best", "possible", "please", "get", "some",
];

/// Significant words of a question, lowercased, deduplicated and sorted, so
/// punctuation, word order and filler words do not change the match key.
pub fn normalize_question(question: &str) -> String {
    let words: BTreeSet<String> = question
        .to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '+' | '#')))
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
        .map(str::to_owned)
        .collect();
    words.into_iter().collect::<Vec<_>>().join(" ")
}

/// Words that say how a question is asked rather than what it asks ("in the
/// command line", "using the terminal", "the easiest way"). Ignored when
/// comparing questions only; stored keys keep them.
const PHRASING_WORDS: &[&str] = &[
    // Left over from contractions ("what's", "don't", "I'd").
    "s",
    "t",
    "ll",
    "re",
    "ve",
    "d",
    "m",
    "command",
    "line",
    "commandline",
    "terminal",
    "cli",
    "shell",
    "console",
    "using",
    "use",
    "via",
    "easiest",
    "easy",
    "simplest",
    "simple",
    "quickly",
    "quick",
    "properly",
    "correctly",
    "exactly",
    "step",
    "steps",
    "help",
    "need",
    "want",
    "know",
    "tell",
    "explain",
];

fn similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let significant = |text: &str| -> BTreeSet<String> {
        let all: BTreeSet<&str> = text.split(' ').filter(|w| !w.is_empty()).collect();
        let kept: BTreeSet<String> = all
            .iter()
            .filter(|w| !PHRASING_WORDS.contains(w))
            .map(|w| w.to_string())
            .collect();
        // A question made only of such words keeps them all.
        if kept.len() >= 3 {
            kept
        } else {
            all.into_iter().map(str::to_owned).collect()
        }
    };
    let (a, b) = (significant(a), significant(b));
    if a == b && a.len() >= 3 {
        return 1.0;
    }
    // Short questions must match exactly; one word changes their meaning.
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    a.intersection(&b).count() as f64 / a.union(&b).count() as f64
}

/// Lowercase ASCII words joined by hyphens, for folder and file names.
pub fn slug(text: &str, max: usize) -> String {
    let mut out = String::new();
    for word in text
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '+' && c != '#')
        .filter(|w| !w.is_empty())
    {
        let word = word.replace('+', "plus").replace('#', "sharp");
        if !out.is_empty() && out.len() + 1 + word.len() > max {
            break;
        }
        if !out.is_empty() {
            out.push('-');
        }
        out.push_str(&word[..word.len().min(max)]);
    }
    out
}

// ------------------------------------------------------------ markdown ----

/// A JSON string, which is also a valid quoted YAML scalar.
fn quoted(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into())
}

/// The Markdown file for one answer.
pub fn render_markdown(entry: &CachedQa) -> String {
    let mut out = String::from("---\n");
    out.push_str(&format!("id: {}\n", quoted(&entry.id)));
    out.push_str(&format!("question: {}\n", quoted(&entry.question)));
    out.push_str(&format!("kind: {}\n", entry.kind.as_str()));
    out.push_str(&format!("topic: {}\n", quoted(&entry.topic)));
    if entry.kind == AnswerKind::Factual && !entry.short_question.is_empty() {
        out.push_str(&format!("fact: {}\n", quoted(&entry.short_question)));
    }
    out.push_str(&format!("saved: {}\n", short_date(&entry.saved_at)));
    out.push_str(&format!(
        "refresh: {}\n",
        entry
            .refresh_after
            .as_deref()
            .map(short_date)
            .unwrap_or_else(|| "never".into())
    ));
    out.push_str(&format!("model: {}\n", quoted(&entry.model)));
    out.push_str(&format!("sources: {}\n---\n\n", entry.sources.len()));
    out.push_str(&format!("# {}\n\n", entry.question.trim()));
    if !entry.summary.trim().is_empty() {
        out.push_str(&format!("> **Short answer:** {}\n", entry.summary.trim()));
        if !entry.explanation.trim().is_empty() {
            out.push_str(&format!(">\n> **Why:** {}\n", entry.explanation.trim()));
        }
        out.push('\n');
    }
    if entry.answer.trim().is_empty() {
        out.push_str("_Sources were saved for this question; no answer has been written yet._\n");
    } else {
        out.push_str(entry.answer.trim());
        out.push('\n');
    }
    if !entry.sources.is_empty() {
        out.push_str("\n## Sources\n\n");
        for source in &entry.sources {
            out.push_str(&format!(
                "{}. [{}]({}) — {} · {}\n",
                source.id,
                source.title.replace(['[', ']'], ""),
                source.url,
                source.site,
                source.via()
            ));
        }
    }
    if !entry.history.is_empty() {
        out.push_str("\n## Previous answers\n");
        for previous in &entry.history {
            out.push_str(&format!(
                "\n<details><summary>Saved {}{}</summary>\n\n{}\n\n</details>\n",
                short_date(&previous.saved_at),
                if previous.model.is_empty() {
                    String::new()
                } else {
                    format!(" by {}", previous.model)
                },
                previous.answer.trim()
            ));
        }
    }
    // The complete record (including source excerpts) for rebuilding the
    // index. "--" cannot appear inside an HTML comment, so it is escaped as a
    // JSON unicode escape, which decodes back to the same text.
    let record = serde_json::to_string(entry)
        .unwrap_or_default()
        .replace("--", "-\\u002d");
    out.push_str(&format!("\n{RECORD_MARKER}\n{record}\n-->\n"));
    out
}

/// Read the record back from a library file.
pub fn parse_markdown(text: &str) -> Option<CachedQa> {
    let start = text.rfind(RECORD_MARKER)? + RECORD_MARKER.len();
    let end = start + text[start..].find("-->")?;
    serde_json::from_str(text[start..end].trim()).ok()
}

fn render_index(entries: &[CachedQa]) -> String {
    let mut topics: BTreeMap<&str, Vec<&CachedQa>> = BTreeMap::new();
    for entry in entries {
        topics.entry(entry.topic.as_str()).or_default().push(entry);
    }
    let mut out = format!(
        "# Q&A library\n\n{} answers in {} topics. Factual answers are kept as they are; others are refreshed when asked after their refresh date, and replaced answers stay in each file under \"Previous answers\".\n",
        entries.len(),
        topics.len()
    );
    for (topic, mut items) in topics {
        items.sort_by_key(|item| item.question.to_lowercase());
        out.push_str(&format!("\n## {topic}\n\n"));
        for item in items {
            out.push_str(&format!(
                "- [{}]({}) — {} · saved {}\n",
                item.question.replace(['[', ']'], ""),
                item.path.replace(' ', "%20"),
                match &item.refresh_after {
                    None => "factual".to_owned(),
                    Some(due) => format!("refresh after {}", short_date(due)),
                },
                short_date(&item.saved_at)
            ));
        }
    }
    out
}

// --------------------------------------------------------------- facts ----

/// A settled fact: a short question, a one-sentence answer and a short
/// explanation, stored as a row of the `facts` table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fact {
    /// Same ID as the library entry it came from.
    pub id: String,
    pub question: String,
    pub answer: String,
    pub explanation: String,
    pub topic: String,
    pub sources: Vec<String>,
    pub saved_at: String,
    /// Markdown file, relative to the library directory.
    pub library_path: String,
}

/// Plain text of the first `max` sentences of Markdown text.
fn first_sentences(markdown: &str, max: usize, max_chars: usize) -> String {
    let plain: String = markdown
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && !l.starts_with("```") && !l.starts_with('#') && !l.starts_with('|')
        })
        .collect::<Vec<_>>()
        .join(" ")
        .replace(['*', '`', '>'], "");
    let mut out = String::new();
    let mut count = 0;
    for piece in plain.split_inclusive(['.', '!', '?']) {
        out.push_str(piece);
        count += 1;
        if count >= max || out.chars().count() >= max_chars {
            break;
        }
    }
    out.trim().chars().take(max_chars).collect()
}

/// Make sure a factual entry has its fact fields, deriving any the model did
/// not provide (e.g. when the user marks an answer factual by hand).
fn complete_fact_fields(entry: &mut CachedQa) {
    if entry.kind != AnswerKind::Factual {
        return;
    }
    if entry.short_question.trim().is_empty() {
        entry.short_question = entry.question.trim().chars().take(120).collect();
    }
    if entry.summary.trim().is_empty() {
        entry.summary = first_sentences(&entry.answer, 1, 300);
    }
    if entry.explanation.trim().is_empty() {
        entry.explanation = first_sentences(&entry.answer, 3, 500);
    }
}

/// An FTS5 query matching all the words of free text (each word quoted, so
/// punctuation cannot form query syntax).
fn fts_query(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{w}\""))
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_facts(facts: &[Fact]) -> String {
    let mut topics: BTreeMap<&str, Vec<&Fact>> = BTreeMap::new();
    for fact in facts {
        topics.entry(fact.topic.as_str()).or_default().push(fact);
    }
    let mut out = format!(
        "# Facts\n\n{} settled facts: a short question, a one-sentence answer and why. Full answers and sources are in each linked file.\n",
        facts.len()
    );
    for (topic, mut items) in topics {
        items.sort_by_key(|item| item.question.to_lowercase());
        out.push_str(&format!("\n## {topic}\n"));
        for fact in items {
            out.push_str(&format!(
                "\n**{}** {}  \n{} ([details]({}))\n",
                fact.question.trim(),
                fact.answer.trim(),
                fact.explanation.trim(),
                fact.library_path.replace(' ', "%20")
            ));
        }
    }
    out
}

// --------------------------------------------------------------- store ----

/// The content of a new or refreshed answer.
pub struct NewAnswer<'a> {
    pub question: &'a str,
    /// The answer in one or two sentences.
    pub summary: &'a str,
    /// For facts: the question in a few words.
    pub short_question: &'a str,
    /// For facts: a short explanation.
    pub explanation: &'a str,
    pub answer: &'a str,
    pub sources: &'a [QaSource],
    pub model: &'a str,
    pub kind: AnswerKind,
    pub topic: &'a str,
}

pub struct QaCache {
    conn: Connection,
    library: PathBuf,
}

impl QaCache {
    /// Open the index and library under `dir`, rebuilding the index from the
    /// Markdown files when it is empty.
    pub fn open(dir: &Path) -> Result<Self> {
        let library = dir.join(LIBRARY_DIR);
        fs::create_dir_all(&library)?;
        let conn = Connection::open(dir.join("qa_cache.sqlite3"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS qa_library(
                id TEXT PRIMARY KEY,
                normalized TEXT NOT NULL UNIQUE,
                record TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS facts(
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                explanation TEXT NOT NULL,
                topic TEXT NOT NULL,
                sources TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                library_path TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                id UNINDEXED, question, answer, explanation, topic, tokenize='unicode61'
             );",
        )?;
        let cache = Self { conn, library };
        let empty: i64 = cache
            .conn
            .query_row("SELECT COUNT(*) FROM qa_library", [], |r| r.get(0))?;
        if empty == 0 {
            cache.import_library()?;
        }
        Ok(cache)
    }

    pub fn library_dir(&self) -> &Path {
        &self.library
    }

    fn decode(record: &str, hits: i64) -> Option<CachedQa> {
        let mut entry: CachedQa = serde_json::from_str(record).ok()?;
        entry.hits = hits.max(0) as u64;
        entry.similarity = 1.0;
        Some(entry)
    }

    fn put(&self, entry: &CachedQa) -> Result<()> {
        let key = normalize_question(&entry.question);
        if key.is_empty() {
            bail!("The question has no searchable words");
        }
        self.conn.execute(
            "INSERT INTO qa_library(id, normalized, record, saved_at) VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET normalized = excluded.normalized,
                record = excluded.record, saved_at = excluded.saved_at",
            params![entry.id, key, serde_json::to_string(entry)?, entry.saved_at],
        )?;
        Ok(())
    }

    fn write_file(&self, entry: &CachedQa) -> Result<()> {
        crate::persistence::write_atomic(
            &self.library.join(&entry.path),
            render_markdown(entry).as_bytes(),
        )
    }

    fn write_index(&self) -> Result<()> {
        let entries = self.list(100_000)?;
        crate::persistence::write_atomic(
            &self.library.join("INDEX.md"),
            render_index(&entries).as_bytes(),
        )?;
        crate::persistence::write_atomic(
            &self.library.join("FACTS.md"),
            render_facts(&self.facts()?).as_bytes(),
        )
    }

    /// Save an entry everywhere: its Markdown file, the index row, the facts
    /// table (factual entries only), INDEX.md and FACTS.md.
    fn save(&self, entry: &mut CachedQa) -> Result<()> {
        complete_fact_fields(entry);
        self.write_file(entry)?;
        self.put(entry)?;
        self.sync_fact(entry)?;
        self.write_index()
    }

    /// Keep the `facts` table in step with the entry: factual entries have a
    /// row; any other kind has none.
    fn sync_fact(&self, entry: &CachedQa) -> Result<()> {
        self.conn
            .execute("DELETE FROM facts WHERE id = ?1", params![entry.id])?;
        self.conn
            .execute("DELETE FROM facts_fts WHERE id = ?1", params![entry.id])?;
        if entry.kind != AnswerKind::Factual || entry.summary.trim().is_empty() {
            return Ok(());
        }
        let sources: Vec<&str> = entry.sources.iter().map(|s| s.url.as_str()).collect();
        self.conn.execute(
            "INSERT INTO facts(id, question, answer, explanation, topic, sources, saved_at, library_path)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                entry.id,
                entry.short_question,
                entry.summary,
                entry.explanation,
                entry.topic,
                serde_json::to_string(&sources)?,
                entry.saved_at,
                entry.path
            ],
        )?;
        self.conn.execute(
            "INSERT INTO facts_fts(id, question, answer, explanation, topic) VALUES(?1, ?2, ?3, ?4, ?5)",
            params![entry.id, entry.short_question, entry.summary, entry.explanation, entry.topic],
        )?;
        Ok(())
    }

    fn fact_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Fact> {
        let sources: String = row.get(5)?;
        Ok(Fact {
            id: row.get(0)?,
            question: row.get(1)?,
            answer: row.get(2)?,
            explanation: row.get(3)?,
            topic: row.get(4)?,
            sources: serde_json::from_str(&sources).unwrap_or_default(),
            saved_at: row.get(6)?,
            library_path: row.get(7)?,
        })
    }

    /// Every fact, by topic and question.
    pub fn facts(&self) -> Result<Vec<Fact>> {
        let mut statement = self.conn.prepare(
            "SELECT id, question, answer, explanation, topic, sources, saved_at, library_path
             FROM facts ORDER BY topic, question COLLATE NOCASE",
        )?;
        let rows = statement
            .query_map([], Self::fact_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Facts matching all words of `text`, best match first (full-text search).
    pub fn search_facts(&self, text: &str, limit: usize) -> Result<Vec<Fact>> {
        let query = fts_query(text);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut statement = self.conn.prepare(
            "SELECT f.id, f.question, f.answer, f.explanation, f.topic, f.sources, f.saved_at, f.library_path
             FROM facts_fts JOIN facts f ON f.id = facts_fts.id
             WHERE facts_fts MATCH ?1 ORDER BY bm25(facts_fts) LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![query, limit.clamp(1, 1_000) as i64], Self::fact_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// A free relative path `<topic>/<question>.md` for a new entry.
    fn new_path(&self, topic: &str, question: &str, id: &str) -> String {
        let folder = match slug(topic, 30) {
            s if s.is_empty() => "general".to_owned(),
            s => s,
        };
        let name = match slug(question, 60) {
            s if s.is_empty() => "question".to_owned(),
            s => s,
        };
        let candidate = format!("{folder}/{name}.md");
        if self.library.join(&candidate).exists() {
            format!("{folder}/{name}-{}.md", &id[..8.min(id.len())])
        } else {
            candidate
        }
    }

    /// The best saved entry for this question (the same question, or a
    /// reworded one), whether or not it is due for a refresh.
    pub fn lookup(&self, question: &str) -> Result<Option<CachedQa>> {
        let key = normalize_question(question);
        if key.is_empty() {
            return Ok(None);
        }
        let mut statement = self
            .conn
            .prepare("SELECT normalized, record, hits FROM qa_library")?;
        let mut best: Option<(f64, String, i64)> = None;
        for row in statement.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (normalized, record, hits) = row?;
            let score = similarity(&key, &normalized);
            if score >= SIMILARITY_THRESHOLD && best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, record, hits));
            }
        }
        Ok(best.and_then(|(score, record, hits)| {
            Self::decode(&record, hits).map(|mut entry| {
                entry.similarity = score;
                entry
            })
        }))
    }

    /// Save a new answer for a question. An existing entry keeps its file and
    /// ID; its previous answer moves to the file's "Previous answers".
    pub fn store(&self, new: NewAnswer<'_>, changing_days: u32) -> Result<CachedQa> {
        let NewAnswer {
            question,
            summary,
            short_question,
            explanation,
            answer,
            sources,
            model,
            kind,
            topic,
        } = new;
        let key = normalize_question(question);
        if key.is_empty() {
            bail!("The question has no searchable words");
        }
        let existing = self
            .conn
            .query_row(
                "SELECT record, hits FROM qa_library WHERE normalized = ?1",
                params![key],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )
            .optional()?
            .and_then(|(record, hits)| Self::decode(&record, hits));
        let saved_at = chrono::Utc::now().to_rfc3339();
        let topic = match slug(topic, 30) {
            s if s.is_empty() => "general".to_owned(),
            s => s,
        };
        let mut entry = match existing {
            Some(mut old) => {
                if !old.answer.trim().is_empty() && old.answer.trim() != answer.trim() {
                    old.history.insert(
                        0,
                        PreviousAnswer {
                            saved_at: old.saved_at.clone(),
                            model: old.model.clone(),
                            answer: old.answer.clone(),
                        },
                    );
                    old.history.truncate(MAX_HISTORY);
                }
                CachedQa {
                    question: question.trim().to_owned(),
                    summary: summary.trim().to_owned(),
                    short_question: short_question.trim().to_owned(),
                    explanation: explanation.trim().to_owned(),
                    answer: answer.to_owned(),
                    sources: sources.to_vec(),
                    model: model.to_owned(),
                    kind,
                    refresh_after: refresh_after(&saved_at, kind, changing_days),
                    saved_at,
                    ..old
                }
            }
            None => {
                let id = uuid::Uuid::new_v4().to_string();
                CachedQa {
                    path: self.new_path(&topic, question, &id),
                    id,
                    question: question.trim().to_owned(),
                    summary: summary.trim().to_owned(),
                    short_question: short_question.trim().to_owned(),
                    explanation: explanation.trim().to_owned(),
                    answer: answer.to_owned(),
                    sources: sources.to_vec(),
                    model: model.to_owned(),
                    kind,
                    topic,
                    refresh_after: refresh_after(&saved_at, kind, changing_days),
                    saved_at,
                    history: Vec::new(),
                    hits: 0,
                    similarity: 1.0,
                }
            }
        };
        self.save(&mut entry)?;
        Ok(entry)
    }

    /// Fill in the answer for an entry that so far only held sources.
    pub fn set_answer(
        &self,
        id: &str,
        summary: &str,
        answer: &str,
        model: &str,
    ) -> Result<CachedQa> {
        let mut entry = self
            .get(id)?
            .ok_or_else(|| anyhow::anyhow!("No such Q&A entry"))?;
        entry.summary = summary.trim().to_owned();
        entry.answer = answer.to_owned();
        entry.model = model.to_owned();
        self.save(&mut entry)?;
        Ok(entry)
    }

    /// Reclassify an entry; its refresh date is recalculated from when it was saved.
    pub fn set_kind(&self, id: &str, kind: AnswerKind, changing_days: u32) -> Result<CachedQa> {
        let mut entry = self
            .get(id)?
            .ok_or_else(|| anyhow::anyhow!("No such Q&A entry"))?;
        entry.kind = kind;
        entry.refresh_after = refresh_after(&entry.saved_at, kind, changing_days);
        self.save(&mut entry)?;
        Ok(entry)
    }

    pub fn record_hit(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE qa_library SET hits = hits + 1 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<CachedQa>> {
        Ok(self
            .conn
            .query_row(
                "SELECT record, hits FROM qa_library WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )
            .optional()?
            .and_then(|(record, hits)| Self::decode(&record, hits)))
    }

    /// Newest first.
    pub fn list(&self, limit: usize) -> Result<Vec<CachedQa>> {
        self.query("", limit)
    }

    /// Entries whose question, answer, topic or sources contain `text`
    /// (case-insensitive), newest first.
    pub fn search(&self, text: &str, limit: usize) -> Result<Vec<CachedQa>> {
        self.query(text.trim(), limit)
    }

    fn query(&self, text: &str, limit: usize) -> Result<Vec<CachedQa>> {
        let pattern = format!(
            "%{}%",
            text.replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_")
        );
        let mut statement = self.conn.prepare(
            "SELECT record, hits FROM qa_library WHERE record LIKE ?1 ESCAPE '\\'
             ORDER BY saved_at DESC LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![pattern, limit.clamp(1, 100_000) as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows
            .into_iter()
            .filter_map(|(record, hits)| Self::decode(&record, hits))
            .collect())
    }

    /// Delete one entry and its Markdown file (an explicit user action).
    pub fn delete(&self, id: &str) -> Result<bool> {
        let Some(entry) = self.get(id)? else {
            return Ok(false);
        };
        let file = self.library.join(&entry.path);
        if !entry.path.is_empty() && file.starts_with(&self.library) && file.is_file() {
            fs::remove_file(&file)?;
        }
        self.conn
            .execute("DELETE FROM qa_library WHERE id = ?1", params![id])?;
        self.conn
            .execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        self.conn
            .execute("DELETE FROM facts_fts WHERE id = ?1", params![id])?;
        self.write_index()?;
        Ok(true)
    }

    /// Rebuild the index from the library's Markdown files. Returns the number
    /// of entries imported.
    pub fn import_library(&self) -> Result<usize> {
        let mut files = Vec::new();
        collect_markdown(&self.library, &mut files);
        let mut count = 0;
        for file in files {
            let Ok(text) = fs::read_to_string(&file) else {
                continue;
            };
            let Some(mut entry) = parse_markdown(&text) else {
                continue;
            };
            if let Ok(relative) = file.strip_prefix(&self.library) {
                entry.path = relative.to_string_lossy().replace('\\', "/");
            }
            if self.put(&entry).is_ok() && self.sync_fact(&entry).is_ok() {
                count += 1;
            }
        }
        if count > 0 {
            self.write_index()?;
        }
        Ok(count)
    }
}

fn collect_markdown(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect_markdown(&path, files);
        } else if kind.is_file()
            && path.extension().is_some_and(|e| e == "md")
            && path.file_name().is_some_and(|n| n != "INDEX.md")
        {
            files.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(tags: &[&str]) -> QaSource {
        QaSource {
            id: 1,
            title: "How do I undo [git]".into(),
            url: "https://stackoverflow.com/questions/927358".into(),
            site: "stackoverflow.com".into(),
            method: "stack_exchange_api".into(),
            content: "Use git reset --soft HEAD~1 -- keeps changes. <!-- not a marker -->".into(),
            tags: tags.iter().map(|t| (*t).to_owned()).collect(),
        }
    }

    fn new<'a>(
        question: &'a str,
        summary: &'a str,
        answer: &'a str,
        sources: &'a [QaSource],
        kind: AnswerKind,
        topic: &'a str,
    ) -> NewAnswer<'a> {
        NewAnswer {
            question,
            summary,
            short_question: "",
            explanation: "",
            answer,
            sources,
            model: "model-a",
            kind,
            topic,
        }
    }

    #[test]
    fn rewordings_match_but_different_questions_do_not() {
        assert_eq!(
            normalize_question("How do I undo the last Git commit?"),
            normalize_question("undo last git commit")
        );
        assert_eq!(normalize_question("What is C++?"), "c++");
        let a = normalize_question("How can I undo my last local git commit");
        let b = normalize_question("undo the last local git commit without losing changes");
        assert!(similarity(&a, &b) < SIMILARITY_THRESHOLD);
        let c = normalize_question("best way to undo last local git commit");
        assert!(similarity(&a, &c) >= SIMILARITY_THRESHOLD);
        assert_eq!(similarity("rust lifetimes", "rust traits"), 0.0);
    }

    #[test]
    fn rewordings_about_how_a_question_is_asked_still_match() {
        // Seen in a live chat: the model asked the same question two ways.
        let saved = normalize_question("how to rename a local git branch in command line");
        let asked = normalize_question("How do I rename a local Git branch?");
        assert!(similarity(&asked, &saved) >= SIMILARITY_THRESHOLD);
        let terminal =
            normalize_question("What's the easiest way to rename a git branch using the terminal?");
        let plain = normalize_question("rename a git branch");
        assert!(similarity(&terminal, &plain) >= SIMILARITY_THRESHOLD);
        // A different subject still does not match.
        let delete = normalize_question("How do I delete a local Git branch?");
        assert!(similarity(&asked, &delete) < SIMILARITY_THRESHOLD);
        let remote = normalize_question("How do I rename a remote Git branch?");
        assert!(similarity(&asked, &remote) < SIMILARITY_THRESHOLD);
    }

    #[test]
    fn kinds_decide_refreshing() {
        assert_eq!(AnswerKind::Factual.refresh_days(30), None);
        assert_eq!(AnswerKind::Changing.refresh_days(30), Some(30));
        assert_eq!(AnswerKind::TimeSensitive.refresh_days(30), Some(3));
        assert_eq!(AnswerKind::TimeSensitive.refresh_days(1), Some(1));
        assert_eq!(
            AnswerKind::parse("time-sensitive"),
            Some(AnswerKind::TimeSensitive)
        );
        assert_eq!(AnswerKind::parse("stable"), Some(AnswerKind::Factual));
        assert!(obviously_time_sensitive(
            "What is the latest stable Rust release?"
        ));
        assert!(obviously_time_sensitive("Bitcoin price"));
        assert!(obviously_time_sensitive("best laptops 2026"));
        assert!(!obviously_time_sensitive(
            "How do I undo the last git commit?"
        ));
        assert!(!obviously_time_sensitive("What is a monad?"));
    }

    #[test]
    fn answers_are_kept_as_organised_markdown_and_never_lost() {
        let dir = tempfile::tempdir().unwrap();
        let cache = QaCache::open(dir.path()).unwrap();
        let sources = [source(&["git"])];
        let first = cache
            .store(
                NewAnswer {
                    short_question: "Undo last git commit?",
                    explanation:
                        "--soft moves HEAD back one commit but leaves the index and files alone.",
                    ..new(
                        "How do I undo the last git commit?",
                        "Run git reset --soft HEAD~1.",
                        "Use `git reset --soft HEAD~1` [1].",
                        &sources,
                        AnswerKind::Factual,
                        "Git",
                    )
                },
                30,
            )
            .unwrap();
        assert_eq!(first.path, "git/how-do-i-undo-the-last-git-commit.md");
        assert_eq!(first.refresh_after, None, "factual answers never refresh");
        assert!(!first.needs_refresh());
        let file = cache.library_dir().join(&first.path);
        let text = fs::read_to_string(&file).unwrap();
        assert!(text.starts_with("---\nid: "));
        assert!(text.contains("question: \"How do I undo the last git commit?\""));
        assert!(text.contains("kind: factual\n"));
        assert!(text.contains("fact: \"Undo last git commit?\"\n"));
        assert!(text.contains("refresh: never\n"));
        assert!(text.contains(
            "> **Short answer:** Run git reset --soft HEAD~1.\n>\n> **Why:** --soft moves HEAD"
        ));
        assert!(text.contains("1. [How do I undo git](https://stackoverflow.com/questions/927358) — stackoverflow.com · read via the Stack Exchange API"));
        let index = fs::read_to_string(cache.library_dir().join("INDEX.md")).unwrap();
        assert!(index.contains("## git\n"));
        assert!(index.contains(&format!("({})", first.path)));

        // A refreshed answer keeps the file and moves the old answer to history.
        let second = cache
            .store(
                new(
                    "undo last git commit",
                    "",
                    "Newer answer.",
                    &[],
                    AnswerKind::Changing,
                    "other",
                ),
                30,
            )
            .unwrap();
        assert_eq!(second.id, first.id);
        assert_eq!(second.path, first.path);
        assert_eq!(second.history.len(), 1);
        assert!(second.refresh_after.is_some());
        let text = fs::read_to_string(&file).unwrap();
        assert!(text.contains("## Previous answers"));
        assert!(text.contains("by model-a"));
        assert!(text.contains("Use `git reset --soft HEAD~1` [1]."));

        // The record survives "--" and "-->" inside the content.
        let parsed = parse_markdown(&text).expect("record parses");
        assert_eq!(parsed.sources, second.sources);
        assert_eq!(
            parsed.history[0].answer,
            "Use `git reset --soft HEAD~1` [1]."
        );

        // Search covers questions, answers and sources.
        assert_eq!(cache.search("NEWER", 10).unwrap().len(), 1);
        assert!(cache.search("rebase", 10).unwrap().is_empty());
        assert_eq!(
            cache.search("50%_", 10).unwrap().len(),
            0,
            "LIKE wildcards are literal"
        );
    }

    #[test]
    fn settled_facts_live_in_the_facts_table_and_facts_md() {
        let dir = tempfile::tempdir().unwrap();
        let cache = QaCache::open(dir.path()).unwrap();
        let sources = [source(&["git"])];
        let entry = cache
            .store(
                NewAnswer {
                    short_question: "Undo last git commit?",
                    explanation: "It moves HEAD back one commit and keeps your changes staged.",
                    ..new(
                        "How do I undo the last git commit and keep my changes?",
                        "Run git reset --soft HEAD~1.",
                        "Use `git reset --soft HEAD~1` [1].",
                        &sources,
                        AnswerKind::Factual,
                        "git",
                    )
                },
                30,
            )
            .unwrap();
        let facts = cache.facts().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].question, "Undo last git commit?");
        assert_eq!(facts[0].answer, "Run git reset --soft HEAD~1.");
        assert_eq!(
            facts[0].sources,
            vec!["https://stackoverflow.com/questions/927358"]
        );
        assert_eq!(
            cache.search_facts("undo COMMIT", 5).unwrap()[0].id,
            entry.id
        );
        assert!(cache
            .search_facts("\"); DROP TABLE facts; --", 5)
            .unwrap()
            .is_empty());
        let md = fs::read_to_string(cache.library_dir().join("FACTS.md")).unwrap();
        assert!(md.contains("**Undo last git commit?** Run git reset --soft HEAD~1."));
        assert!(md.contains("It moves HEAD back one commit"));

        // No longer factual: removed from the facts table, kept in the library.
        cache.set_kind(&entry.id, AnswerKind::Changing, 30).unwrap();
        assert!(cache.facts().unwrap().is_empty());
        assert!(cache.get(&entry.id).unwrap().is_some());

        // Marked factual by hand without model-written fields: they are derived.
        let other = cache
            .store(
                new(
                    "What does git stash do?",
                    "",
                    "Git stash saves uncommitted changes. It restores a clean working tree. More detail.",
                    &[],
                    AnswerKind::Changing,
                    "git",
                ),
                30,
            )
            .unwrap();
        cache.set_kind(&other.id, AnswerKind::Factual, 30).unwrap();
        let fact = cache.search_facts("stash", 5).unwrap().remove(0);
        assert_eq!(fact.question, "What does git stash do?");
        assert_eq!(fact.answer, "Git stash saves uncommitted changes.");
        assert!(fact.explanation.contains("clean working tree"));
    }

    #[test]
    fn stale_answers_are_kept_and_flagged_and_the_index_rebuilds_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let sources = [source(&[])];
        {
            let cache = QaCache::open(dir.path()).unwrap();
            let entry = cache
                .store(
                    new(
                        "latest rust release version",
                        "1.99.1",
                        "1.99.1",
                        &sources,
                        AnswerKind::TimeSensitive,
                        "rust",
                    ),
                    30,
                )
                .unwrap();
            assert!(!entry.needs_refresh());
            // Pretend it was saved a week ago: due for refresh, still kept.
            let mut old = entry.clone();
            old.saved_at = (chrono::Utc::now() - chrono::Duration::days(7)).to_rfc3339();
            old.refresh_after = refresh_after(&old.saved_at, old.kind, 30);
            cache.save(&mut old).unwrap();
            let hit = cache
                .lookup("latest rust release version")
                .unwrap()
                .unwrap();
            assert!(hit.needs_refresh());
            assert_eq!(hit.answer, "1.99.1");
            // Reclassifying as factual stops refreshing.
            let fixed = cache.set_kind(&hit.id, AnswerKind::Factual, 30).unwrap();
            assert!(!fixed.needs_refresh());
        }
        // Lose the index: it is rebuilt from the Markdown files, facts included.
        for name in [
            "qa_cache.sqlite3",
            "qa_cache.sqlite3-wal",
            "qa_cache.sqlite3-shm",
        ] {
            let _ = fs::remove_file(dir.path().join(name));
        }
        let cache = QaCache::open(dir.path()).unwrap();
        let hit = cache
            .lookup("latest rust release version")
            .unwrap()
            .expect("rebuilt");
        assert_eq!(hit.kind, AnswerKind::Factual);
        assert_eq!(hit.path, "rust/latest-rust-release-version.md");
        assert_eq!(cache.facts().unwrap().len(), 1);
        assert!(cache.delete(&hit.id).unwrap());
        assert!(!cache
            .library_dir()
            .join("rust/latest-rust-release-version.md")
            .exists());
        assert!(cache.list(10).unwrap().is_empty());
        assert!(cache.facts().unwrap().is_empty());
    }

    #[test]
    fn slugs_are_safe_file_names() {
        assert_eq!(
            slug("How do I use C++ and C#?", 60),
            "how-do-i-use-cplusplus-and-csharp"
        );
        assert_eq!(slug("../../etc/passwd", 60), "etc-passwd");
        assert_eq!(slug("日本語", 60), "");
        assert!(slug(&"word ".repeat(40), 60).len() <= 60);
    }
}
