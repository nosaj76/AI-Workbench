//! Domain knowledge notes: one Markdown file per topic, built up over time.
//!
//! Essential sentences from conversations, web searches and research runs are
//! collected into `knowledge-notes/<topic>.md`. A file holds everything known
//! about one topic, each fact with its basis (web, research, or an unverified
//! model answer) and the sources it came from.
//!
//! No model writes the notes. A script picks the sentences that carry the
//! most information about the question (relevance, how central the sentence's
//! terms are to the text, concrete details such as numbers, identifiers and
//! definitions), drops chatter, questions and fragments that cannot stand
//! alone, and keeps each chosen sentence word for word, credited to the page or
//! conversation it came from.
//!
//! The Markdown files are the only record: they are plain text you can read,
//! edit (in Workbench, an editor or Obsidian) and back up without Workbench.
//! SQLite (`knowledge_notes.sqlite3`) is just a search index and is rebuilt
//! from the files whenever they change.
//!
//! Nothing is deleted or cut automatically. A repeated fact is merged and its
//! new source added as corroboration, an unverified fact that a web or
//! research source later confirms is upgraded, and a note you delete is moved
//! to `knowledge-notes/.deleted/` rather than erased.
use crate::{
    ollama::{ChatMessage, OllamaClient},
    qa_library::{normalize_question, slug},
    settings::Settings,
};
use anyhow::{bail, Context, Result};
use regex::Regex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

pub const NOTES_DIR: &str = "knowledge-notes";
pub const BASIS_WEB: &str = "web";
pub const BASIS_RESEARCH: &str = "research";
pub const BASIS_MODEL: &str = "model answer, unverified";

/// Facts this similar (shared significant words) are the same fact.
const DUPLICATE_FACT_SIMILARITY: f64 = 0.8;
/// Topic titles this similar are the same topic.
const SAME_TOPIC_SIMILARITY: f64 = 0.6;
const MAX_TITLE_CHARS: usize = 120;
const MIN_FACT_CHARS: usize = 15;
const MAX_FACT_CHARS: usize = 1000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteSource {
    pub id: String,
    /// `web`, `research` or `conversation`.
    pub kind: String,
    pub label: String,
    pub url: String,
    pub date: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteEntry {
    pub text: String,
    pub basis: String,
    /// Ids of the note's sources, e.g. `S1`.
    pub sources: Vec<String>,
    pub added: String,
    /// What several independent models said when asked whether it is true.
    pub consensus: Option<Consensus>,
}

/// Votes of the reviewing models on one fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consensus {
    pub agree: usize,
    pub disagree: usize,
    pub unsure: usize,
}

/// What the votes add up to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewStatus {
    /// Every reviewing model judged it true.
    Agreed,
    /// The models differ, but most did not call it wrong.
    Mixed,
    /// As many models or more called it wrong as called it true.
    Disputed,
}

/// Fewest models that must answer before a tally means anything.
pub const REVIEW_MIN_MODELS: usize = 2;

impl Consensus {
    pub fn responders(&self) -> usize {
        self.agree + self.disagree + self.unsure
    }

    pub fn status(&self) -> ReviewStatus {
        if self.responders() >= REVIEW_MIN_MODELS && self.agree == self.responders() {
            ReviewStatus::Agreed
        } else if self.disagree > 0 && self.disagree >= self.agree {
            ReviewStatus::Disputed
        } else {
            ReviewStatus::Mixed
        }
    }

    /// Plain wording for the chat model and the note viewer.
    pub fn label(&self) -> String {
        let total = self.responders();
        match self.status() {
            ReviewStatus::Agreed => format!("all {total} models agree"),
            ReviewStatus::Disputed => format!(
                "disputed: {} of {total} models say it is wrong",
                self.disagree
            ),
            ReviewStatus::Mixed => {
                let mut label = format!("{} of {total} models agree", self.agree);
                if self.disagree > 0 {
                    label.push_str(&format!(", {} disagree", self.disagree));
                }
                if self.unsure > 0 {
                    label.push_str(&format!(", {} unsure", self.unsure));
                }
                label
            }
        }
    }

    fn token(&self) -> String {
        format!(
            "{{review: {} agree, {} disagree, {} unsure}}",
            self.agree, self.disagree, self.unsure
        )
    }

    fn parse(inner: &str) -> Option<Self> {
        let mut numbers = inner
            .split(',')
            .map(|part| part.split_whitespace().next()?.parse::<usize>().ok());
        Some(Self {
            agree: numbers.next()??,
            disagree: numbers.next()??,
            unsure: numbers.next()??,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    pub title: String,
    pub slug: String,
    pub created: String,
    pub updated: String,
    pub entries: Vec<NoteEntry>,
    pub sources: Vec<NoteSource>,
    /// One line per review run: when, by which models, how many facts.
    pub reviews: Vec<String>,
}

impl Note {
    /// Facts judged by the reviewing models: (agreed, mixed, disputed).
    pub fn tally(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for status in self
            .entries
            .iter()
            .filter_map(|e| e.consensus.map(|c| c.status()))
        {
            match status {
                ReviewStatus::Agreed => counts.0 += 1,
                ReviewStatus::Mixed => counts.1 += 1,
                ReviewStatus::Disputed => counts.2 += 1,
            }
        }
        counts
    }
}

#[derive(Clone, Debug)]
pub struct NoteSummary {
    pub slug: String,
    pub title: String,
    pub entries: usize,
    pub updated: String,
    pub agreed: usize,
    pub mixed: usize,
    pub disputed: usize,
}

/// A source before it has been merged into a note and given an id.
#[derive(Clone, Debug)]
pub struct IncomingSource {
    pub kind: String,
    pub label: String,
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct IncomingFact {
    pub text: String,
    pub basis: String,
    /// Indexes into the `sources` slice given to `absorb`.
    pub source_ix: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Absorbed {
    pub slug: String,
    pub title: String,
    pub created: bool,
    pub added: usize,
    pub corroborated: usize,
    pub upgraded: usize,
    /// Facts refused for being too short or too long to be a single fact.
    pub rejected: usize,
}

/// Owned text and the index of the source it came from; see `Part`.
pub type OwnedPart = (String, Option<usize>);

/// One piece of text to take sentences from, and the source it came from.
pub struct Part<'a> {
    pub text: &'a str,
    /// Index into `Material::sources`; `None` credits every source.
    pub source: Option<usize>,
}

/// Text to extract notes from, and where it came from.
pub struct Material<'a> {
    /// What was asked; decides which sentences matter, and the topic unless
    /// `topic` is given.
    pub question: &'a str,
    /// The subject to file the facts under, when it differs from the question
    /// (a follow-up belongs to the subject its conversation started with).
    pub topic: Option<&'a str>,
    pub parts: &'a [Part<'a>],
    /// How trustworthy the facts are: `BASIS_WEB`, `BASIS_RESEARCH` or `BASIS_MODEL`.
    pub basis: &'a str,
    /// Where the material came from; cited on the facts taken from it.
    pub sources: &'a [IncomingSource],
}

fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

fn words(text: &str) -> BTreeSet<String> {
    normalize_question(text)
        .split(' ')
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// A word reduced to a rough stem (plural "s" dropped, first six letters), so
/// "configuring" and "configuration", or "emulators" and "emulator", match.
fn stem(word: &str) -> String {
    let base = word
        .strip_suffix('s')
        .filter(|rest| rest.chars().count() >= 3)
        .unwrap_or(word);
    base.chars().take(6).collect()
}

fn stems(text: &str) -> BTreeSet<String> {
    words(text).iter().map(|w| stem(w)).collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Statements about the assistant, its access or the conversation are not
/// domain knowledge, however the model phrased them.
fn looks_meta(text: &str) -> bool {
    let lower = text.to_lowercase();
    text.trim_end().ends_with('?')
        || [
            "the material",
            "this conversation",
            "the conversation",
            "the assistant",
            "the user",
            "the model ",
            "file access",
            "filesystem access",
            "tool access",
            "your project",
            "your workspace",
            "i can ",
            "i cannot",
            "i can't",
            "i do not",
            "i don't",
        ]
        .iter()
        .any(|p| has_phrase(&lower, p))
}

/// Whether `phrase` occurs in `lower` starting at a word boundary, so "here are"
/// does not match inside "there are" nor "i can " inside "AI can".
fn has_phrase(lower: &str, phrase: &str) -> bool {
    format!(" {lower}").contains(&format!(" {}", phrase.trim_start()))
}

fn basis_rank(basis: &str) -> u8 {
    match basis {
        BASIS_RESEARCH => 3,
        BASIS_WEB => 2,
        BASIS_MODEL => 0,
        _ => 1,
    }
}

// ------------------------------------------------------------ markdown ----

pub fn render_markdown(note: &Note) -> String {
    let mut out = String::new();
    out.push_str("---\n");
    out.push_str(&format!("title: {}\n", json!(note.title)));
    out.push_str(&format!(
        "created: {}\nupdated: {}\n",
        note.created, note.updated
    ));
    out.push_str(&format!(
        "entries: {}\nsources: {}\n---\n\n# {}\n\n## Notes\n\n",
        note.entries.len(),
        note.sources.len(),
        note.title
    ));
    for entry in &note.entries {
        out.push_str(&format!("- {} *({})*", entry.text, entry.basis));
        for id in &entry.sources {
            out.push_str(&format!(" [{id}]"));
        }
        if let Some(consensus) = &entry.consensus {
            out.push(' ');
            out.push_str(&consensus.token());
        }
        out.push_str(&format!(" · {}\n", entry.added));
    }
    out.push_str("\n## Sources\n\n");
    for source in &note.sources {
        out.push_str(&format!(
            "- **{}** · {} · {} · {} · {}\n",
            source.id,
            source.kind,
            source.label.replace(" · ", " - "),
            source.url,
            source.date
        ));
    }
    if !note.reviews.is_empty() {
        out.push_str("\n## Reviews\n\n");
        for review in &note.reviews {
            out.push_str(&format!("- {review}\n"));
        }
    }
    out
}

fn is_date(text: &str) -> bool {
    text.len() == 10
        && text.chars().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        })
}

fn parse_entry(line: &str) -> Option<NoteEntry> {
    let mut rest = line.strip_prefix("- ")?.trim().to_string();
    let mut added = String::new();
    if let Some(i) = rest.rfind(" · ") {
        if is_date(rest[i + " · ".len()..].trim()) {
            added = rest[i + " · ".len()..].trim().to_string();
            rest.truncate(i);
        }
    }
    let mut consensus = None;
    {
        let trimmed = rest.trim_end();
        if trimmed.ends_with('}') {
            if let Some(open) = trimmed.rfind(" {review: ") {
                let inner = &trimmed[open + " {review: ".len()..trimmed.len() - 1];
                if let Some(parsed) = Consensus::parse(inner) {
                    consensus = Some(parsed);
                    rest = trimmed[..open].to_string();
                }
            }
        }
    }
    let mut sources = Vec::new();
    loop {
        let trimmed = rest.trim_end();
        if !trimmed.ends_with(']') {
            break;
        }
        let Some(open) = trimmed.rfind('[') else {
            break;
        };
        let inner = &trimmed[open + 1..trimmed.len() - 1];
        if inner.len() < 2
            || !inner.starts_with('S')
            || !inner[1..].chars().all(|c| c.is_ascii_digit())
        {
            break;
        }
        sources.push(inner.to_string());
        rest = trimmed[..open].to_string();
    }
    sources.reverse();
    let mut basis = String::new();
    let trimmed = rest.trim_end();
    if trimmed.ends_with(")*") {
        if let Some(open) = trimmed.rfind(" *(") {
            basis = trimmed[open + 3..trimmed.len() - 2].to_string();
            rest = trimmed[..open].to_string();
        }
    }
    let text = rest.trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some(NoteEntry {
        text,
        basis,
        sources,
        added,
        consensus,
    })
}

fn parse_source(line: &str) -> Option<NoteSource> {
    let rest = line.strip_prefix("- **")?;
    let (id, rest) = rest.split_once("**")?;
    let parts: Vec<&str> = rest.trim_start_matches(" · ").split(" · ").collect();
    let get = |i: usize| {
        parts
            .get(i)
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    Some(NoteSource {
        id: id.trim().to_string(),
        kind: get(0),
        label: get(1),
        url: get(2),
        date: get(3),
    })
}

/// Parses a note file. `slug` comes from the file name, so renaming a file
/// renames the topic's identity, and hand-written files without a header work.
pub fn parse_markdown(text: &str, slug: &str) -> Note {
    let mut title = String::new();
    let mut created = String::new();
    let mut updated = String::new();
    let mut lines = text.lines().peekable();
    if lines.peek().is_some_and(|l| l.trim() == "---") {
        lines.next();
        for line in lines.by_ref() {
            if line.trim() == "---" {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                let value = value.trim();
                match key.trim() {
                    "title" => {
                        title = serde_json::from_str::<String>(value)
                            .unwrap_or_else(|_| value.trim_matches('"').to_string())
                    }
                    "created" => created = value.to_string(),
                    "updated" => updated = value.to_string(),
                    _ => {}
                }
            }
        }
    }
    let mut entries = Vec::new();
    let mut sources = Vec::new();
    let mut reviews = Vec::new();
    let mut section = "";
    for line in lines {
        if let Some(heading) = line.strip_prefix("## ") {
            section = match heading.trim() {
                "Notes" => "notes",
                "Sources" => "sources",
                "Reviews" => "reviews",
                _ => "",
            };
            continue;
        }
        if title.is_empty() {
            if let Some(h) = line.strip_prefix("# ") {
                title = h.trim().to_string();
            }
        }
        match section {
            "notes" => entries.extend(parse_entry(line)),
            "sources" => sources.extend(parse_source(line)),
            "reviews" => {
                if let Some(review) = line.strip_prefix("- ") {
                    reviews.push(review.trim().to_string());
                }
            }
            _ => {}
        }
    }
    if title.is_empty() {
        title = slug.replace('-', " ");
    }
    Note {
        title,
        slug: slug.to_string(),
        created,
        updated,
        entries,
        sources,
        reviews,
    }
}

// --------------------------------------------------------------- store ----

pub struct NoteStore {
    conn: Connection,
    dir: PathBuf,
}

fn fts_or_query(text: &str) -> String {
    let mut seen = BTreeSet::new();
    normalize_question(text)
        .split(' ')
        .filter(|w| w.len() > 1 && seen.insert(w.to_string()))
        .take(12)
        .map(|w| format!("\"{w}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

impl NoteStore {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let dir = data_dir.join(NOTES_DIR);
        fs::create_dir_all(&dir)?;
        let conn = Connection::open(data_dir.join("knowledge_notes.sqlite3"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS notes(
                slug TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                entries INTEGER NOT NULL,
                updated TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                slug UNINDEXED, title, body, tokenize='unicode61'
             );",
        )?;
        // Older indexes lack the review tallies; adding a column that already
        // exists is the expected, harmless failure.
        for column in ["agreed", "mixed", "disputed"] {
            let _ = conn.execute(
                &format!("ALTER TABLE notes ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"),
                [],
            );
        }
        let store = Self { conn, dir };
        store.sync()?;
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, slug: &str) -> PathBuf {
        self.dir.join(format!("{slug}.md"))
    }

    fn files(&self) -> Vec<(String, PathBuf)> {
        let mut out = Vec::new();
        if let Ok(read) = fs::read_dir(&self.dir) {
            for entry in read.flatten() {
                let path = entry.path();
                if !path.is_file() || path.extension().is_none_or(|e| e != "md") {
                    continue;
                }
                let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                    continue;
                };
                if stem != "INDEX" {
                    out.push((stem, path));
                }
            }
        }
        out.sort();
        out
    }

    fn mtime(path: &Path) -> i64 {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0)
    }

    /// Brings the search index in line with the files, including notes that
    /// were edited, added or removed outside Workbench.
    pub fn sync(&self) -> Result<usize> {
        let files = self.files();
        let known: Vec<(String, i64)> = {
            let mut statement = self.conn.prepare("SELECT slug, mtime FROM notes")?;
            let rows = statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        for (slug, _) in &known {
            if !files.iter().any(|(s, _)| s == slug) {
                self.conn
                    .execute("DELETE FROM notes WHERE slug=?1", [slug])?;
                self.conn
                    .execute("DELETE FROM notes_fts WHERE slug=?1", [slug])?;
            }
        }
        let mut changed = 0;
        for (slug, path) in &files {
            let mtime = Self::mtime(path);
            if known.iter().any(|(s, m)| s == slug && *m == mtime) {
                continue;
            }
            let Ok(text) = fs::read_to_string(path) else {
                continue;
            };
            self.reindex(&parse_markdown(&text, slug), mtime)?;
            changed += 1;
        }
        if changed > 0 || files.len() != known.len() {
            self.write_index()?;
        }
        Ok(changed)
    }

    fn reindex(&self, note: &Note, mtime: i64) -> Result<()> {
        let body = note
            .entries
            .iter()
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.conn
            .execute("DELETE FROM notes_fts WHERE slug=?1", [&note.slug])?;
        self.conn.execute(
            "INSERT INTO notes_fts(slug,title,body) VALUES(?1,?2,?3)",
            params![note.slug, note.title, body],
        )?;
        let (agreed, mixed, disputed) = note.tally();
        self.conn.execute(
            "INSERT INTO notes(slug,title,mtime,entries,updated,agreed,mixed,disputed) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(slug) DO UPDATE SET title=?2, mtime=?3, entries=?4, updated=?5, agreed=?6, mixed=?7, disputed=?8",
            params![
                note.slug,
                note.title,
                mtime,
                note.entries.len() as i64,
                note.updated,
                agreed as i64,
                mixed as i64,
                disputed as i64
            ],
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<NoteSummary>> {
        let mut statement = self
            .conn
            .prepare("SELECT slug,title,entries,updated,agreed,mixed,disputed FROM notes ORDER BY updated DESC, title")?;
        let rows = statement.query_map([], |r| {
            Ok(NoteSummary {
                slug: r.get(0)?,
                title: r.get(1)?,
                entries: r.get::<_, i64>(2)? as usize,
                updated: r.get(3)?,
                agreed: r.get::<_, i64>(4)? as usize,
                mixed: r.get::<_, i64>(5)? as usize,
                disputed: r.get::<_, i64>(6)? as usize,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    pub fn get(&self, slug: &str) -> Result<Option<Note>> {
        let path = self.path(slug);
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(parse_markdown(&fs::read_to_string(path)?, slug)))
    }

    /// The slug of the existing note for this topic title, if there is one.
    pub fn find_topic(&self, title: &str) -> Result<Option<String>> {
        let wanted = slug(title, 60);
        if wanted.is_empty() {
            return Ok(None);
        }
        let summaries = self.list()?;
        if let Some(found) = summaries.iter().find(|s| s.slug == wanted) {
            return Ok(Some(found.slug.clone()));
        }
        let target = stems(title);
        // A single word never matches loosely: one word changes the subject.
        if target.len() < 2 {
            return Ok(None);
        }
        let mut best: Option<(f64, String)> = None;
        for summary in &summaries {
            let score = jaccard(&target, &stems(&summary.title));
            if score >= SAME_TOPIC_SIMILARITY && best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, summary.slug.clone()));
            }
        }
        Ok(best.map(|(_, s)| s))
    }

    /// Adds facts to a topic's note, creating it when new.
    pub fn absorb(
        &self,
        topic: &str,
        facts: &[IncomingFact],
        sources: &[IncomingSource],
    ) -> Result<Absorbed> {
        let title = one_line(topic);
        if title.is_empty() || title.chars().count() > MAX_TITLE_CHARS {
            bail!("A topic title must be 1 to {MAX_TITLE_CHARS} characters");
        }
        let existing = self.find_topic(&title)?;
        let slug_name = match &existing {
            Some(s) => s.clone(),
            None => {
                let s = slug(&title, 60);
                if s.is_empty() {
                    bail!("A topic title needs letters or digits");
                }
                s
            }
        };
        let date = today();
        let mut note = match self.get(&slug_name)? {
            Some(note) => note,
            None => Note {
                title: title.clone(),
                slug: slug_name.clone(),
                created: date.clone(),
                updated: date.clone(),
                entries: Vec::new(),
                sources: Vec::new(),
                reviews: Vec::new(),
            },
        };
        let created = existing.is_none() && note.entries.is_empty();
        let mut out = Absorbed {
            slug: slug_name,
            title: note.title.clone(),
            created,
            ..Default::default()
        };
        // Merge sources first so facts can point at stable ids.
        let mut ids = Vec::new();
        for incoming in sources {
            let url = incoming.url.trim();
            let found = note.sources.iter().find(|s| {
                if url.is_empty() {
                    s.url.is_empty()
                        && s.kind == incoming.kind
                        && s.label == one_line(&incoming.label)
                } else {
                    s.url == url
                }
            });
            let id = match found {
                Some(s) => s.id.clone(),
                None => {
                    let next = note
                        .sources
                        .iter()
                        .filter_map(|s| s.id.strip_prefix('S')?.parse::<usize>().ok())
                        .max()
                        .unwrap_or(0)
                        + 1;
                    let id = format!("S{next}");
                    note.sources.push(NoteSource {
                        id: id.clone(),
                        kind: incoming.kind.clone(),
                        label: one_line(&incoming.label),
                        url: url.to_string(),
                        date: date.clone(),
                    });
                    id
                }
            };
            ids.push(id);
        }
        for fact in facts {
            let text = one_line(&fact.text);
            let length = text.chars().count();
            if !(MIN_FACT_CHARS..=MAX_FACT_CHARS).contains(&length) || looks_meta(&text) {
                out.rejected += 1;
                continue;
            }
            let mut fact_ids: Vec<String> = fact
                .source_ix
                .iter()
                .filter_map(|i| ids.get(*i).cloned())
                .collect();
            fact_ids.dedup();
            let new_words = words(&text);
            let same = note.entries.iter().position(|e| {
                e.text == text
                    || (new_words.len() >= 3
                        && jaccard(&new_words, &words(&e.text)) >= DUPLICATE_FACT_SIMILARITY)
            });
            match same {
                Some(i) => {
                    let entry = &mut note.entries[i];
                    let mut changed = false;
                    for id in fact_ids {
                        if !entry.sources.contains(&id) {
                            entry.sources.push(id);
                            changed = true;
                        }
                    }
                    if basis_rank(&fact.basis) > basis_rank(&entry.basis) {
                        entry.basis = fact.basis.clone();
                        out.upgraded += 1;
                        changed = true;
                    }
                    if changed {
                        out.corroborated += 1;
                    }
                }
                None => {
                    note.entries.push(NoteEntry {
                        text,
                        basis: fact.basis.clone(),
                        sources: fact_ids,
                        added: date.clone(),
                        consensus: None,
                    });
                    out.added += 1;
                }
            }
        }
        if out.added == 0 && out.corroborated == 0 && !out.created {
            return Ok(out);
        }
        if note.entries.is_empty() {
            // Nothing usable for a brand-new topic: do not leave an empty file.
            out.created = false;
            return Ok(out);
        }
        note.updated = date;
        // A search returns pages that contributed nothing (an unrelated "Grow"
        // site for "how to grow ginseng"); only pages a fact cites are sources.
        let cited: std::collections::HashSet<String> = note
            .entries
            .iter()
            .flat_map(|entry| entry.sources.iter().cloned())
            .collect();
        note.sources.retain(|source| cited.contains(&source.id));
        self.save(&note)?;
        Ok(out)
    }

    /// Writes a note file (replacing it atomically) and updates the index.
    pub fn save(&self, note: &Note) -> Result<()> {
        let path = self.path(&note.slug);
        let temp = path.with_extension("md.tmp");
        fs::write(&temp, render_markdown(note))?;
        fs::rename(&temp, &path)?;
        self.reindex(note, Self::mtime(&path))?;
        self.write_index()
    }

    /// Records review results by fact text, on the note as it is now (a chat
    /// may have added facts while the models were working). Returns how many
    /// facts were updated.
    pub fn apply_reviews(
        &self,
        slug: &str,
        results: &[(String, Consensus)],
        review_line: &str,
    ) -> Result<usize> {
        let Some(mut note) = self.get(slug)? else {
            return Ok(0);
        };
        let mut updated = 0;
        for entry in &mut note.entries {
            if let Some((_, consensus)) = results.iter().find(|(text, _)| *text == entry.text) {
                entry.consensus = Some(*consensus);
                updated += 1;
            }
        }
        if updated > 0 {
            note.reviews.push(review_line.to_string());
            self.save(&note)?;
        }
        Ok(updated)
    }

    /// Notes matching a question, best first.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Note>> {
        let q = fts_or_query(query);
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let mut statement = self.conn.prepare(
            "SELECT slug FROM notes_fts WHERE notes_fts MATCH ?1 ORDER BY bm25(notes_fts, 0.0, 4.0, 1.0) LIMIT ?2",
        )?;
        let slugs: Vec<String> = statement
            .query_map(params![q, limit as i64], |r| r.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        let mut notes = Vec::new();
        for s in slugs {
            if let Some(note) = self.get(&s)? {
                notes.push(note);
            }
        }
        Ok(notes)
    }

    /// A `<knowledge_notes>` block for the chat model, bounded by
    /// `max_chars`, that says exactly what it leaves out.
    pub fn context_for(
        &self,
        query: &str,
        max_chars: usize,
    ) -> Result<Option<(String, Vec<String>)>> {
        self.context_for_excluding(query, max_chars, &[])
    }

    /// `context_for`, leaving out every fact whose only sources are the
    /// conversations labelled in `own_conversation`: a chat must not be handed its
    /// own earlier answers back as evidence.
    pub fn context_for_excluding(
        &self,
        query: &str,
        max_chars: usize,
        own_conversation: &[String],
    ) -> Result<Option<(String, Vec<String>)>> {
        let hits = self.search(query, 6)?;
        if hits.is_empty() {
            return Ok(None);
        }
        let query_words = words(query);
        let mut body = String::new();
        let mut used: Vec<String> = Vec::new();
        let mut left_out: Vec<String> = Vec::new();
        for note in &hits {
            // Facts that the reviewing models judged wrong are never given to
            // the chat model as knowledge; they are counted instead.
            let is_disputed = |e: &NoteEntry| {
                e.consensus
                    .is_some_and(|c| c.status() == ReviewStatus::Disputed)
            };
            let disputed = note.entries.iter().filter(|e| is_disputed(e)).count();
            let visible = note.entries.len() - disputed;
            let circular = |e: &NoteEntry| {
                !own_conversation.is_empty()
                    && !e.sources.is_empty()
                    && e.sources.iter().all(|id| {
                        note.sources.iter().any(|s| {
                            &s.id == id
                                && s.kind == "conversation"
                                && own_conversation.contains(&s.label)
                        })
                    })
            };
            let mut ranked: Vec<(usize, &NoteEntry)> = note
                .entries
                .iter()
                .filter(|e| !is_disputed(e) && !circular(e))
                .map(|e| (words(&e.text).intersection(&query_words).count(), e))
                .collect();
            ranked.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
            let header = format!("## {} ({}/{}.md)\n", note.title, NOTES_DIR, note.slug);
            if body.len() + header.len() > max_chars {
                left_out.push(note.title.clone());
                continue;
            }
            let mut section = header;
            let mut shown = 0;
            for (score, entry) in &ranked {
                if *score == 0 && shown > 0 {
                    break;
                }
                let urls: Vec<&str> = entry
                    .sources
                    .iter()
                    .filter_map(|id| note.sources.iter().find(|s| &s.id == id))
                    .filter(|s| !s.url.is_empty())
                    .map(|s| s.url.as_str())
                    .take(2)
                    .collect();
                let cite = if urls.is_empty() {
                    String::new()
                } else {
                    format!(" <{}>", urls.join(", "))
                };
                let trust = match &entry.consensus {
                    Some(c) => format!("{}; {}", entry.basis, c.label()),
                    None => entry.basis.clone(),
                };
                let line = format!("- {} ({}){}\n", entry.text, trust, cite);
                if body.len() + section.len() + line.len() > max_chars {
                    break;
                }
                section.push_str(&line);
                shown += 1;
            }
            if shown == 0 {
                left_out.push(note.title.clone());
                continue;
            }
            if shown < visible {
                section.push_str(&format!(
                    "({} more entries in this note are not shown.)\n",
                    visible - shown
                ));
            }
            if disputed > 0 {
                section.push_str(&format!(
                    "({disputed} entries that the reviewing models judged wrong are not shown.)\n"
                ));
            }
            body.push_str(&section);
            used.push(note.title.clone());
        }
        if used.is_empty() {
            return Ok(None);
        }
        let mut text = String::from(
            "<knowledge_notes>\nThese are notes Workbench saved earlier from the user's conversations, web searches and research. \"web\" and \"research\" entries come from the cited sources; \"model answer, unverified\" entries are earlier model output and may be wrong. A note like \"all 3 models agree\" means that many independent local models each judged the statement true without seeing any source; it raises confidence but is not proof, and \"models disagree\" entries need checking. Use relevant notes instead of guessing, say when you rely on one, and verify unverified ones if it matters. Notes may be out of date.\n",
        );
        text.push_str(&body);
        if !left_out.is_empty() {
            text.push_str(&format!(
                "Other matching notes not shown: {}.\n",
                left_out.join(", ")
            ));
        }
        text.push_str("</knowledge_notes>");
        Ok(Some((text, used)))
    }

    /// Moves a note to `.deleted/` so it can still be recovered by hand.
    pub fn delete(&self, slug: &str) -> Result<bool> {
        let path = self.path(slug);
        if !path.is_file() {
            return Ok(false);
        }
        let trash = self.dir.join(".deleted");
        fs::create_dir_all(&trash)?;
        let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S");
        fs::rename(&path, trash.join(format!("{slug}-{stamp}.md")))?;
        self.sync()?;
        Ok(true)
    }

    pub fn write_index(&self) -> Result<()> {
        let mut out = String::from("# Knowledge notes\n\nOne file per topic, built from conversations, web searches and research.\n\n");
        for s in self.list()? {
            out.push_str(&format!(
                "- [{}]({}.md) · {} entries · updated {}\n",
                s.title, s.slug, s.entries, s.updated
            ));
        }
        fs::write(self.dir.join("INDEX.md"), out)?;
        Ok(())
    }
}

// ---------------------------------------------------------------- review --
//
// The script writes the facts; models only judge them. Several different
// installed models are each asked, separately and without seeing any source,
// whether each fact is true, and the votes are recorded on the fact.

/// Facts per question to one model.
const REVIEW_BATCH: usize = 10;

/// Counts chats started, so a background review can step aside for a new one.
static CHAT_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Called when a chat starts: a review running in the background stops at its
/// next step so the chat does not wait for the local model.
pub fn note_chat_started() {
    CHAT_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// The chat counter now. A review scheduled with this value stops when it
/// changes, i.e. when another chat starts.
pub fn current_chat_epoch() -> u64 {
    CHAT_EPOCH.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    True,
    False,
    Unsure,
}

/// The family of a model name ("qwen3.5" for "qwen3.5:9b"), so that reviewers
/// can be chosen from different families rather than sizes of one model.
fn family(model: &str) -> String {
    model.split(':').next().unwrap_or(model).to_lowercase()
}

/// The models that will review facts. Taken from the configured list, or from
/// the routing profile's general, coding, writing and reasoning models;
/// restricted to installed models; different families first; the model that
/// wrote the answer is left out whenever enough others remain.
pub fn pick_reviewers(
    settings: &Settings,
    installed: &[String],
    author: Option<&str>,
) -> Vec<String> {
    let wanted = settings.notes_review_count.clamp(REVIEW_MIN_MODELS, 5);
    let configured: Vec<String> = if settings.notes_review_models.is_empty() {
        crate::routing::config(settings)
            .map(|c| [c.general, c.coding, c.writing, c.reasoning].concat())
            .unwrap_or_default()
    } else {
        settings.notes_review_models.clone()
    };
    let mut pool: Vec<String> = Vec::new();
    for model in configured {
        if installed.contains(&model) && !pool.contains(&model) {
            pool.push(model);
        }
    }
    let others: Vec<String> = pool
        .iter()
        .filter(|m| Some(m.as_str()) != author)
        .cloned()
        .collect();
    if others.len() >= REVIEW_MIN_MODELS {
        pool = others;
    }
    let mut chosen: Vec<String> = Vec::new();
    let mut families = BTreeSet::new();
    for model in &pool {
        if chosen.len() < wanted && families.insert(family(model)) {
            chosen.push(model.clone());
        }
    }
    for model in &pool {
        if chosen.len() < wanted && !chosen.contains(model) {
            chosen.push(model.clone());
        }
    }
    chosen
}

const REVIEW_SYSTEM: &str = "You are a careful fact checker working without internet access. For each numbered statement about the topic, decide whether it is correct as a general statement of fact. Answer \"true\" only when you are confident it is accurate; \"false\" when it is wrong or misleading; \"unsure\" when you cannot tell or it depends on context. Output JSON only, in the form {\"verdicts\":[{\"id\":1,\"verdict\":\"true\"}]}, with one entry per statement and no explanations.";

fn parse_verdicts(content: &str, count: usize) -> Result<Vec<Verdict>> {
    let data: Value =
        serde_json::from_str(content.trim()).context("The reviewing model did not return JSON")?;
    let list = data["verdicts"]
        .as_array()
        .context("The reviewing model's reply has no verdicts")?;
    // A statement the model skipped counts as unsure, not as agreement.
    let mut out = vec![Verdict::Unsure; count];
    for item in list {
        let Some(id) = item["id"].as_u64().map(|n| n as usize) else {
            continue;
        };
        if (1..=count).contains(&id) {
            out[id - 1] = match item["verdict"].as_str().map(str::to_lowercase).as_deref() {
                Some("true") => Verdict::True,
                Some("false") => Verdict::False,
                _ => Verdict::Unsure,
            };
        }
    }
    Ok(out)
}

fn tally(votes: &[Verdict]) -> Consensus {
    let mut consensus = Consensus::default();
    for vote in votes {
        match vote {
            Verdict::True => consensus.agree += 1,
            Verdict::False => consensus.disagree += 1,
            Verdict::Unsure => consensus.unsure += 1,
        }
    }
    consensus
}

async fn ask_model(
    client: &OllamaClient,
    settings: &Settings,
    model: &str,
    title: &str,
    facts: &[String],
    cancel: Arc<AtomicBool>,
) -> Result<Vec<Verdict>> {
    let mut quiet = settings.clone();
    if crate::routing::config(settings).is_ok_and(|c| c.single_model_mode) {
        // Release the model afterwards, as every other background request does.
        quiet.keep_alive = "0".into();
    }
    let mut options = crate::runtime::options(&quiet)?;
    options.temperature = 0.0;
    options.num_predict = (80 + 30 * facts.len()) as i32;
    options.think = Some(Value::Bool(false));
    options.format = Some(json!({
        "type":"object",
        "properties":{"verdicts":{"type":"array","items":{"type":"object",
            "properties":{"id":{"type":"integer"},"verdict":{"type":"string","enum":["true","false","unsure"]}},
            "required":["id","verdict"]}}},
        "required":["verdicts"]
    }));
    let numbered = facts
        .iter()
        .enumerate()
        .map(|(i, fact)| format!("{}. {fact}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: REVIEW_SYSTEM.into(),
            ..Default::default()
        },
        ChatMessage {
            role: "user".into(),
            content: format!("Topic: {title}\n\nStatements:\n{numbered}"),
            ..Default::default()
        },
    ];
    // Hold the shared model permit only for this one call, so a chat that
    // starts meanwhile waits for at most one short answer.
    let permit = crate::routing::inference_permit(settings, cancel.clone()).await?;
    let response = client
        .chat_stream(model, &messages, &options, &[], cancel, |_| {})
        .await;
    drop(permit);
    parse_verdicts(&response?.content, facts.len())
}

/// What a review of one note found.
#[derive(Clone, Debug, Default)]
pub struct ReviewReport {
    pub title: String,
    pub models: Vec<String>,
    /// Models that could not answer, and why; they do not count as votes.
    pub failed: Vec<(String, String)>,
    pub reviewed: usize,
    pub agreed: usize,
    pub mixed: usize,
    pub disputed: usize,
    pub too_few_models: bool,
    pub stopped_for_chat: bool,
}

impl ReviewReport {
    pub fn summary(&self) -> String {
        let mut text = if self.too_few_models {
            format!("{}: needs at least {REVIEW_MIN_MODELS} different installed models to compare; none were reviewed.", self.title)
        } else if self.reviewed == 0 {
            format!("{}: nothing new to review.", self.title)
        } else {
            format!(
                "{}: {} fact(s) reviewed by {}: {} agreed, {} mixed, {} disputed.",
                self.title,
                self.reviewed,
                self.models.join(", "),
                self.agreed,
                self.mixed,
                self.disputed
            )
        };
        if self.stopped_for_chat {
            text.push_str(
                " Stopped early to give way to a chat; the rest will be reviewed next time.",
            );
        }
        for (model, why) in &self.failed {
            text.push_str(&format!(" {model} could not answer ({why})."));
        }
        text
    }
}

/// Has every not-yet-reviewed fact of `slug` judged by each of `models` in
/// turn and records the tally on the fact. With `yield_after` (a value of
/// `current_chat_epoch`) it stops as soon as another chat has started.
pub async fn review_note(
    data_dir: &Path,
    client: &OllamaClient,
    settings: &Settings,
    slug: &str,
    models: &[String],
    yield_after: Option<u64>,
    cancel: Arc<AtomicBool>,
) -> Result<ReviewReport> {
    let store = NoteStore::open(data_dir)?;
    let note = store.get(slug)?.context("No such note")?;
    let mut report = ReviewReport {
        title: note.title.clone(),
        models: models.to_vec(),
        ..Default::default()
    };
    let targets: Vec<String> = note
        .entries
        .iter()
        .filter(|e| e.consensus.is_none())
        .map(|e| e.text.clone())
        .collect();
    if targets.is_empty() {
        return Ok(report);
    }
    if models.len() < REVIEW_MIN_MODELS {
        report.too_few_models = true;
        return Ok(report);
    }
    let mut votes: Vec<Vec<Verdict>> = vec![Vec::new(); targets.len()];
    let mut answering: Vec<String> = Vec::new();
    'models: for model in models {
        let mut model_votes: Vec<Verdict> = Vec::new();
        for batch in targets.chunks(REVIEW_BATCH) {
            if cancel.load(Ordering::Relaxed) {
                break 'models;
            }
            if yield_after.is_some_and(|epoch| CHAT_EPOCH.load(Ordering::Relaxed) != epoch) {
                report.stopped_for_chat = true;
                break 'models;
            }
            match ask_model(client, settings, model, &note.title, batch, cancel.clone()).await {
                Ok(verdicts) => model_votes.extend(verdicts),
                Err(error) => {
                    report.failed.push((model.clone(), format!("{error:#}")));
                    model_votes.clear();
                    break;
                }
            }
        }
        // Only a model that judged every fact counts as having voted.
        if model_votes.len() == targets.len() {
            for (slot, vote) in votes.iter_mut().zip(model_votes) {
                slot.push(vote);
            }
            answering.push(model.clone());
        }
    }
    let results: Vec<(String, Consensus)> = targets
        .iter()
        .zip(&votes)
        .map(|(text, v)| (text.clone(), tally(v)))
        .filter(|(_, c)| c.responders() >= REVIEW_MIN_MODELS)
        .collect();
    for (_, consensus) in &results {
        match consensus.status() {
            ReviewStatus::Agreed => report.agreed += 1,
            ReviewStatus::Mixed => report.mixed += 1,
            ReviewStatus::Disputed => report.disputed += 1,
        }
    }
    report.reviewed = results.len();
    report.models = answering.clone();
    if !results.is_empty() {
        let line = format!(
            "{} · models: {} · {} fact(s): {} agreed, {} mixed, {} disputed",
            today(),
            answering.join(", "),
            results.len(),
            report.agreed,
            report.mixed,
            report.disputed
        );
        store.apply_reviews(slug, &results, &line)?;
    }
    Ok(report)
}

/// Reviews notes on a worker thread's own runtime: finds the installed
/// models, chooses reviewers, and reviews each note in `slugs`.
pub fn review_notes_blocking(
    settings: &Settings,
    data_dir: &Path,
    slugs: &[String],
    author: Option<&str>,
    yield_after: Option<u64>,
    cancel: Arc<AtomicBool>,
) -> Result<Vec<ReviewReport>> {
    let client = OllamaClient::new(&settings.base_url)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let installed: Vec<String> = client
            .list_models()
            .await?
            .into_iter()
            .map(|m| m.name)
            .collect();
        let models = pick_reviewers(settings, &installed, author);
        let mut reports = Vec::new();
        for slug in slugs {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let report = review_note(
                data_dir,
                &client,
                settings,
                slug,
                &models,
                yield_after,
                cancel.clone(),
            )
            .await?;
            let stop = report.stopped_for_chat;
            reports.push(report);
            if stop {
                break;
            }
        }
        Ok(reports)
    })
}

// ------------------------------------------------------------ extracting --
//
// A script, not a model, writes the notes: it picks essential sentences from
// the text and keeps them word for word.

const MAX_FACTS_ALONE: usize = 8;
const MAX_FACTS_PER_SOURCE: usize = 5;
const MIN_SENTENCE_CHARS: usize = 40;
const MAX_SENTENCE_CHARS: usize = 450;
const MIN_SENTENCE_WORDS: usize = 6;
/// Sentences scoring below this are not essential enough to keep.
const MIN_SCORE: f64 = 1.2;
/// Sentences this alike (shared significant words) count as one.
const NEAR_DUPLICATE_SENTENCE: f64 = 0.7;
const MAX_TOPIC_WORDS: usize = 4;

/// A sentence that opens with one of these leans on the sentence before it
/// ("It prevents…", "However, …") and means nothing alone.
const LEANS_ON_PREVIOUS: &[&str] = &[
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "he",
    "she",
    "its",
    "their",
    "there",
    "here",
    "then",
    "also",
    "but",
    "and",
    "so",
    "or",
    "however",
    "therefore",
    "thus",
    "additionally",
    "moreover",
    "furthermore",
    "instead",
    "otherwise",
    "similarly",
    "meanwhile",
    "which",
    "such",
    "both",
    "either",
    "for",
    "in",
    "as",
    "because",
    "since",
    "although",
    "though",
    "yet",
    "still",
    "otherwise",
    "conversely",
    "consequently",
];

/// Chat, advice and page furniture rather than knowledge.
const CHATTER: &[&str] = &[
    "let me",
    "i'd ",
    "i'll",
    "i will",
    "i can ",
    "i'm ",
    "i am ",
    "i think",
    "i recommend",
    "sure,",
    "great,",
    "here's",
    "here is",
    "here are",
    "feel free",
    "would you like",
    "let's",
    "hope this",
    "in summary",
    "in conclusion",
    "to summarize",
    "you might",
    "you may want",
    "you should",
    "you might",
    "you may",
    "if you ",
    "cookie",
    "subscribe",
    "sign up",
    "sign in",
    "log in",
    "all rights reserved",
    "privacy policy",
    "terms of",
    "click here",
    "newsletter",
    "share this",
    "read more",
    "advertisement",
    "skip to",
    "below is",
    "above is",
    "the following",
    "as follows",
    "step by step",
    "structured guide",
    "this guide",
    "this article",
    "in this section",
];

/// Phrases that mark a sentence as stating how something works or what it is.
const DEFINING: &[&str] = &[
    " is ",
    " are ",
    " means ",
    " refers to ",
    " allows ",
    " prevents ",
    " requires ",
    " uses ",
    " provides ",
    " stores ",
    " enables ",
    " ensures ",
    " determines ",
    " supports ",
    " contains ",
    " reads ",
    " defines ",
    " controls ",
    " creates ",
    " includes ",
    " consists of ",
    " runs ",
    " translates ",
    " compiles ",
    " reduces ",
    " increases ",
    " causes ",
];

/// Words of a request that say what the user wants done, not what it is about.
const REQUEST_WORDS: &[&str] = &[
    "tell",
    "explain",
    "describe",
    "show",
    "give",
    "list",
    "write",
    "find",
    "search",
    "research",
    "please",
    "want",
    "need",
    "know",
    "worth",
    "best",
    "better",
    "think",
    "using",
    "use",
    "look",
    "help",
    "does",
    "work",
    "works",
    "much",
    "many",
    "really",
    "actually",
    "something",
    "anything",
    "focus",
    "review",
    "add",
    "continue",
    "additional",
    "deeper",
    "more",
    "analyze",
    "analyse",
    "investigate",
    "explore",
    "check",
    "if",
    "including",
    "include",
    "includes",
];

/// Short words kept in capitals in a topic title.
const ACRONYMS: &[&str] = &[
    "cpu", "gpu", "ram", "os", "api", "sql", "ssh", "tls", "ssl", "usb", "dns", "tcp", "udp", "ip",
    "gui", "cli", "vm", "ide", "json", "xml", "http", "https", "html", "css", "ai", "ml", "llm",
    "arm", "bsd", "gnu", "nfs", "pci", "uefi", "bios", "iso", "ssd", "hdd", "tpm",
];

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("valid built-in pattern"))
}

/// Plain paragraphs of prose from Markdown or rendered web text: code, tables,
/// headings and rules are left out, bullets stand alone, wrapped lines are
/// joined, and links, tags and emphasis marks are removed.
fn prose_paragraphs(text: &str) -> Vec<(String, bool)> {
    static BULLET: OnceLock<Regex> = OnceLock::new();
    static HEADING: OnceLock<Regex> = OnceLock::new();
    static LINK: OnceLock<Regex> = OnceLock::new();
    static TAG: OnceLock<Regex> = OnceLock::new();
    let bullet = pattern(&BULLET, r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+");
    let heading = pattern(&HEADING, r"^\s*#{1,6}\s");
    let link = pattern(&LINK, r"\[([^\]]*)\]\([^)]*\)");
    let tag = pattern(&TAG, r"<[^>\n]{1,200}>");
    let mut paragraphs: Vec<(String, bool)> = Vec::new();
    let mut current = String::new();
    let mut current_is_bullet = false;
    let mut in_code = false;
    let flush = |current: &mut String, bullet: bool, out: &mut Vec<(String, bool)>| {
        let cleaned = one_line(&current.replace('*', ""));
        if !cleaned.is_empty() {
            out.push((cleaned, bullet));
        }
        current.clear();
    };
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            flush(&mut current, current_is_bullet, &mut paragraphs);
            in_code = !in_code;
            continue;
        }
        let rule_or_table = line.starts_with('|')
            || (!line.is_empty()
                && line
                    .chars()
                    .all(|c| matches!(c, '-' | '=' | '_' | '*' | ':' | '|' | ' ')));
        if in_code || line.is_empty() || rule_or_table || heading.is_match(line) {
            flush(&mut current, current_is_bullet, &mut paragraphs);
            continue;
        }
        if bullet.is_match(line) {
            flush(&mut current, current_is_bullet, &mut paragraphs);
            current_is_bullet = true;
        } else if current.is_empty() {
            current_is_bullet = false;
        }
        let body = bullet.replace(line, "");
        let body = link.replace_all(&body, "$1");
        let body = tag.replace_all(&body, "");
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(body.trim());
    }
    flush(&mut current, current_is_bullet, &mut paragraphs);
    paragraphs
}

/// Splits a paragraph at sentence ends, but not inside "e.g.", "vs." or a
/// version number such as 3.5.
fn split_sentences(paragraph: &str) -> Vec<String> {
    const ABBREVIATIONS: &[&str] = &[
        "e.g", "i.e", "vs", "etc", "approx", "fig", "no", "dr", "mr", "mrs", "st", "cf",
    ];
    let chars: Vec<char> = paragraph.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..chars.len() {
        if !matches!(chars[i], '.' | '!' | '?') {
            continue;
        }
        if i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
            continue;
        }
        let next = chars[i + 1..].iter().find(|c| !c.is_whitespace());
        if next.is_some_and(|c| {
            !(c.is_uppercase() || c.is_ascii_digit() || matches!(c, '"' | '\'' | '(' | '`'))
        }) {
            continue;
        }
        if chars[i] == '.' {
            let word: String = chars[start..i]
                .iter()
                .rev()
                .take_while(|c| !c.is_whitespace())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let lower = word.to_lowercase();
            if ABBREVIATIONS.contains(&lower.as_str())
                || (word.chars().count() == 1 && word.chars().all(char::is_uppercase))
            {
                continue;
            }
        }
        let sentence: String = chars[start..=i].iter().collect();
        out.push(sentence.trim().to_string());
        start = i + 1;
    }
    let tail: String = chars[start..].iter().collect();
    if !tail.trim().is_empty() {
        out.push(tail.trim().to_string());
    }
    out
}

/// Whether a sentence can stand alone as a fact: a complete declarative
/// statement of a sensible length that is not chat, advice or page furniture.
fn usable_sentence(sentence: &str, bullet: bool) -> bool {
    static LABEL: OnceLock<Regex> = OnceLock::new();
    let length = sentence.chars().count();
    if !(MIN_SENTENCE_CHARS..=MAX_SENTENCE_CHARS).contains(&length) {
        return false;
    }
    let trimmed = sentence.trim();
    // Only finished statements: a bullet like "Void Linux: Simple, no systemd"
    // or "Kernel Land → isolated" is a fragment, not a fact.
    let core = trimmed.trim_end_matches(['"', '\'', ')', '”', '’']);
    // A snippet cut off by the search engine ("…lessens the chance of...").
    if core.ends_with("...") || core.ends_with('…') {
        return false;
    }
    let finished = core.ends_with('.') || core.ends_with('!');
    // A bullet item often ends without a full stop; it still counts when it
    // clearly states something (it has a verb), not when it is a bare phrase.
    let stated =
        bullet && !core.ends_with('?') && !core.ends_with(':') && has_verb(&trimmed.to_lowercase());
    // A bullet is a fragment unless it says something with a verb, full stop or not.
    if (bullet && !has_verb(&trimmed.to_lowercase()))
        || !(finished || stated)
        || trimmed.contains('|')
        || trimmed.contains('→')
        || pattern(&LABEL, r"^[A-Z0-9][^:.!?]{0,40}:\s").is_match(trimmed)
    {
        return false;
    }
    let Some(first) = trimmed.chars().next() else {
        return false;
    };
    if !(first.is_uppercase() || first.is_ascii_digit() || first == '`') {
        return false;
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() < MIN_SENTENCE_WORDS {
        return false;
    }
    let letters = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let visible = trimmed.chars().filter(|c| !c.is_whitespace()).count();
    if letters * 10 < visible * 6 {
        return false;
    }
    let opener: String = tokens[0]
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if LEANS_ON_PREVIOUS.contains(&opener.as_str()) {
        return false;
    }
    let lower = trimmed.to_lowercase().replace(['’', '‘'], "'");
    // A page describing itself ("Discover how long…", "A complete guide to…").
    const PAGE_PITCH_OPENERS: &[&str] = &[
        "discover",
        "learn",
        "explore",
        "find",
        "read",
        "click",
        "check",
        "subscribe",
        "shop",
        "browse",
        "get",
    ];
    const PAGE_PITCH: &[&str] = &[
        "this guide",
        "this article",
        "this post",
        "complete guide",
        "growing guide",
        "step-by-step guide",
        "in this video",
        "affiliate",
        "many want to know",
        "shown in the list",
        "see the list",
        "see also",
        "list of commercial",
    ];
    if PAGE_PITCH_OPENERS.contains(&opener.as_str())
        || PAGE_PITCH.iter().any(|p| has_phrase(&lower, p))
    {
        return false;
    }
    !(CHATTER.iter().any(|p| has_phrase(&lower, p)) || looks_meta(trimmed))
}

/// Whether the text has a verb the script can recognise: an auxiliary or
/// modal, or a word that says what something is or does.
fn has_verb(lower: &str) -> bool {
    const AUXILIARIES: &[&str] = &[
        "is", "are", "was", "were", "be", "been", "has", "have", "had", "can", "could", "will",
        "would", "should", "must", "may", "might", "does", "do", "did",
    ];
    // A how-to step opens with its verb ("Use `debootstrap` to create…").
    const ACTIONS: &[&str] = &[
        "use",
        "run",
        "install",
        "add",
        "create",
        "edit",
        "set",
        "configure",
        "enable",
        "disable",
        "copy",
        "build",
        "start",
        "stop",
        "check",
        "open",
        "select",
        "choose",
        "update",
        "remove",
        "download",
        "mount",
        "compile",
        "write",
        "make",
        "restart",
        "reload",
        "verify",
        "test",
        "define",
        "specify",
        "ensure",
        "avoid",
        "prefer",
        "keep",
        "store",
        "save",
        "load",
        "boot",
        "partition",
        "format",
        "generate",
        "apply",
        "assign",
        "replace",
        "restrict",
        "limit",
    ];
    let padded = format!(" {lower} ");
    let first = lower
        .split(|c: char| !c.is_alphanumeric())
        .find(|w| !w.is_empty())
        .unwrap_or("");
    ACTIONS.contains(&first)
        || DEFINING.iter().any(|p| padded.contains(p))
        || lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| AUXILIARIES.contains(&w))
}

/// The words a question is about, minus words that only say what is wanted.
fn question_words(question: &str) -> BTreeSet<String> {
    words(question)
        .into_iter()
        .filter(|w| !REQUEST_WORDS.contains(&w.as_str()))
        .collect()
}

/// The most essential sentences of `text` for `question`, at most `max`, in
/// the order they appear. A sentence scores for matching the question, for its
/// terms being repeated through the text (central to it), and for concrete
/// detail: numbers, identifiers in backticks, and statements of what
/// something is or does.
fn essential_sentences(
    text: &str,
    question: &BTreeSet<String>,
    max: usize,
    require_match: bool,
) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    for (paragraph, bullet) in prose_paragraphs(text) {
        for sentence in split_sentences(&paragraph) {
            if usable_sentence(&sentence, bullet) && !sentences.contains(&sentence) {
                sentences.push(sentence);
            }
        }
    }
    if sentences.is_empty() {
        return Vec::new();
    }
    let sets: Vec<BTreeSet<String>> = sentences.iter().map(|s| words(s)).collect();
    let mut document_frequency: BTreeMap<&str, usize> = BTreeMap::new();
    for set in &sets {
        for word in set {
            *document_frequency.entry(word.as_str()).or_default() += 1;
        }
    }
    let centrality: Vec<f64> = sets
        .iter()
        .map(|set| {
            if set.is_empty() {
                return 0.0;
            }
            let shared: usize = set
                .iter()
                .map(|w| document_frequency[w.as_str()].saturating_sub(1))
                .sum();
            shared as f64 / set.len() as f64
        })
        .collect();
    let top_centrality = centrality.iter().cloned().fold(0.0, f64::max).max(1e-9);
    // Matching on stems lets "emulators" answer a question about an "emulator".
    let question_stems: BTreeSet<String> = question.iter().map(|w| stem(w)).collect();
    let mut scored: Vec<(usize, f64)> = sentences
        .iter()
        .enumerate()
        .map(|(i, sentence)| {
            let sentence_stems: BTreeSet<String> = sets[i].iter().map(|w| stem(w)).collect();
            let matched = sentence_stems.intersection(&question_stems).count();
            let relevance = if question_stems.is_empty() {
                0.0
            } else {
                matched as f64 / question_stems.len() as f64
            };
            let lower = format!(" {} ", sentence.to_lowercase());
            let mut detail = 0.0;
            if sentence.chars().any(|c| c.is_ascii_digit()) {
                detail += 0.4;
            }
            if sentence.contains('`') {
                detail += 0.5;
            }
            if DEFINING.iter().any(|p| lower.contains(p)) {
                detail += 0.5;
            }
            // Text from a web page must be about the question's subject: matching
            // at least one of its words is required, however central the sentence
            // looks, because pages carry navigation and unrelated text. A model's
            // own answer to the question is on topic as a whole.
            let on_topic = !require_match || question_stems.is_empty() || matched > 0;
            let score = 3.0 * relevance.min(1.0) + 2.0 * (centrality[i] / top_centrality) + detail;
            (i, if on_topic { score } else { 0.0 })
        })
        .filter(|(_, score)| *score >= MIN_SCORE)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut chosen: Vec<usize> = Vec::new();
    for (i, _) in scored {
        if chosen.len() >= max {
            break;
        }
        if chosen
            .iter()
            .all(|c| jaccard(&sets[i], &sets[*c]) < NEAR_DUPLICATE_SENTENCE)
        {
            chosen.push(i);
        }
    }
    chosen.sort_unstable();
    chosen.into_iter().map(|i| sentences[i].clone()).collect()
}

/// A topic title from the question, built from the words of the question that
/// the text is most about (so "if I create three user lands from the linux
/// kernel…" becomes "User Lands Linux Kernel", not its first words). With no
/// text, the question's first subject words are used. Titles that share most
/// of their words are merged into one note later.
pub fn topic_for(question: &str, texts: &[&str]) -> Option<String> {
    let mut seen = BTreeSet::new();
    let mut candidates: Vec<(String, String)> = Vec::new();
    for token in question
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '+' | '#')))
        .filter(|t| !t.is_empty())
    {
        let lower = token.to_lowercase();
        if normalize_question(&lower).is_empty()
            || REQUEST_WORDS.contains(&lower.as_str())
            || !seen.insert(lower.clone())
        {
            continue;
        }
        let shown = if ACRONYMS.contains(&lower.as_str()) {
            lower.to_uppercase()
        } else {
            let mut chars = lower.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        };
        candidates.push((lower, shown));
    }
    if !candidates
        .iter()
        .any(|(w, _)| w.chars().any(char::is_alphabetic))
    {
        return None;
    }
    if !texts.is_empty() {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for text in texts {
            for token in text
                .split(|c: char| !(c.is_alphanumeric() || matches!(c, '+' | '#')))
                .filter(|t| !t.is_empty())
            {
                *counts.entry(stem(&token.to_lowercase())).or_default() += 1;
            }
        }
        let count_of = |w: &str| counts.get(&stem(w)).copied().unwrap_or(0);
        let mentioned: Vec<(String, String)> = candidates
            .iter()
            .filter(|(w, _)| count_of(w) > 0)
            .cloned()
            .collect();
        if !mentioned.is_empty() {
            candidates = mentioned;
        }
        // Keep the words the text talks about most, in the order asked.
        // Words asked first are usually the subject, so they count for more;
        // the extra 1 keeps a leading word the text hardly repeats in the running.
        let weight = |i: usize| (count_of(&candidates[i].0) + 1) as f64 / (1.0 + 0.25 * i as f64);
        let mut ranked: Vec<usize> = (0..candidates.len()).collect();
        ranked.sort_by(|a, b| weight(*b).total_cmp(&weight(*a)));
        ranked.truncate(MAX_TOPIC_WORDS);
        ranked.sort_unstable();
        candidates = ranked.into_iter().map(|i| candidates[i].clone()).collect();
    }
    candidates.truncate(MAX_TOPIC_WORDS);
    // A title must not end on half of a phrase ("Lisp Programming Language Open"
    // from "open source"): drop a trailing modifier word.
    const DANGLING: &[&str] = &[
        "open", "free", "new", "best", "modern", "simple", "basic", "advanced", "good", "top",
        "main", "first", "full", "complete", "easy", "old", "common", "high", "low", "real",
    ];
    while candidates.len() > 1
        && candidates
            .last()
            .is_some_and(|(word, _)| DANGLING.contains(&word.as_str()))
    {
        candidates.pop();
    }
    (!candidates.is_empty()).then(|| {
        candidates
            .into_iter()
            .map(|(_, shown)| shown)
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// A topic title from the question alone; see `topic_for`.
pub fn topic_from_question(question: &str) -> Option<String> {
    topic_for(question, &[])
}

/// The facts of one piece of material: essential sentences from each part,
/// each credited to the part's own source.
/// Search snippets often start with the page's date ("Jun 22, 2026 · …"). The
/// date belongs to the listing, not to the statement after it.
fn strip_snippet_dates(text: &str) -> String {
    static DATE: OnceLock<Regex> = OnceLock::new();
    pattern(
        &DATE,
        r"(?m)(^|[\n.!?]\s*)[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}\s*[·•—-]\s*",
    )
    .replace_all(text, "$1")
    .into_owned()
}

pub fn extract_facts(material: &Material<'_>) -> Vec<IncomingFact> {
    let question = question_words(material.question);
    let per_part = if material.parts.len() > 1 {
        MAX_FACTS_PER_SOURCE
    } else {
        MAX_FACTS_ALONE
    };
    let mut facts = Vec::new();
    for part in material.parts {
        let cleaned = strip_snippet_dates(part.text);
        for sentence in
            essential_sentences(&cleaned, &question, per_part, material.basis != BASIS_MODEL)
        {
            if crate::memory::sensitive(&sentence) {
                continue;
            }
            facts.push(IncomingFact {
                text: sentence,
                basis: material.basis.to_string(),
                source_ix: match part.source {
                    Some(i) => vec![i],
                    None => (0..material.sources.len()).collect(),
                },
            });
        }
    }
    facts
}

/// Extracts essential sentences from `material` and merges them into the
/// notes under `data_dir`. Fast and local: no model is involved.
pub fn capture(
    data_dir: &Path,
    settings: &Settings,
    material: &Material<'_>,
) -> Result<Vec<Absorbed>> {
    if !settings.knowledge_notes || crate::memory::sensitive(material.question) {
        return Ok(Vec::new());
    }
    let texts: Vec<&str> = material.parts.iter().map(|p| p.text).collect();
    let Some(topic) = topic_for(material.topic.unwrap_or(material.question), &texts) else {
        return Ok(Vec::new());
    };
    let facts = extract_facts(material);
    if facts.is_empty() {
        return Ok(Vec::new());
    }
    let absorbed = NoteStore::open(data_dir)?.absorb(&topic, &facts, material.sources)?;
    Ok(
        if absorbed.added > 0 || absorbed.corroborated > 0 || absorbed.created {
            vec![absorbed]
        } else {
            Vec::new()
        },
    )
}

/// Page text of a finished research run, one part per web page it retrieved,
/// with those pages as the sources. The retrieved page itself is used, not the
/// model's analysis of it. Local-library sources are private and never become
/// notes.
pub fn research_material(
    run: &crate::research::ResearchRun,
) -> Option<(Vec<OwnedPart>, Vec<IncomingSource>)> {
    let mut parts = Vec::new();
    let mut sources = Vec::new();
    for source in &run.sources {
        if source.kind == "local"
            || !source.url.starts_with("http")
            || source.content.trim().is_empty()
        {
            continue;
        }
        parts.push((source.content.clone(), Some(sources.len())));
        sources.push(IncomingSource {
            kind: "web".into(),
            label: if source.title.trim().is_empty() {
                source.url.clone()
            } else {
                source.title.clone()
            },
            url: source.url.clone(),
        });
    }
    (!sources.is_empty()).then_some((parts, sources))
}

/// Turns a finished research run into notes. Fast and local: no model is
/// involved, so it is safe to call from any worker thread.
pub fn capture_research(
    settings: &Settings,
    data_dir: &Path,
    run: &crate::research::ResearchRun,
) -> Result<Vec<Absorbed>> {
    let Some((owned, sources)) = research_material(run) else {
        return Ok(Vec::new());
    };
    let parts: Vec<Part<'_>> = owned
        .iter()
        .map(|(text, source)| Part {
            text,
            source: *source,
        })
        .collect();
    let material = Material {
        topic: None,
        question: &run.question,
        parts: &parts,
        basis: BASIS_RESEARCH,
        sources: &sources,
    };
    capture(data_dir, settings, &material)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(text: &str, basis: &str, ix: &[usize]) -> IncomingFact {
        IncomingFact {
            text: text.into(),
            basis: basis.into(),
            source_ix: ix.to_vec(),
        }
    }
    fn web(url: &str) -> IncomingSource {
        IncomingSource {
            kind: "web".into(),
            label: "A page".into(),
            url: url.into(),
        }
    }

    #[test]
    fn a_topic_gets_one_readable_file_that_round_trips_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let out = store
            .absorb(
                "Linux Kernel Configuration",
                &[
                    fact(
                        "make menuconfig opens an interactive kernel configuration menu.",
                        BASIS_WEB,
                        &[0],
                    ),
                    fact(
                        "The .config file stores the selected kernel options.",
                        BASIS_MODEL,
                        &[],
                    ),
                ],
                &[web("https://kernel.org/doc")],
            )
            .unwrap();
        assert!(out.created);
        assert_eq!((out.added, out.rejected), (2, 0));
        let text = fs::read_to_string(
            dir.path()
                .join("knowledge-notes/linux-kernel-configuration.md"),
        )
        .unwrap();
        assert!(
            text.contains("## Notes") && text.contains("## Sources") && text.contains("[S1]"),
            "{text}"
        );
        let note = store.get("linux-kernel-configuration").unwrap().unwrap();
        assert_eq!(parse_markdown(&render_markdown(&note), &note.slug), note);
        assert_eq!(note.entries[0].sources, ["S1"]);
        assert_eq!(note.entries[1].basis, BASIS_MODEL);
        assert!(dir.path().join("knowledge-notes/INDEX.md").is_file());
    }

    #[test]
    fn repeated_facts_are_corroborated_not_duplicated_and_web_upgrades_unverified() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let text = "The kernel build system reads options from the .config file.";
        store
            .absorb("Kernel Build", &[fact(text, BASIS_MODEL, &[])], &[])
            .unwrap();
        let again = store
            .absorb(
                "Kernel Build",
                &[fact(
                    "The kernel build system reads options from the .config file",
                    BASIS_WEB,
                    &[0],
                )],
                &[web("https://example.org/a")],
            )
            .unwrap();
        assert_eq!((again.added, again.corroborated, again.upgraded), (0, 1, 1));
        let note = store.get("kernel-build").unwrap().unwrap();
        assert_eq!(note.entries.len(), 1);
        assert_eq!(note.entries[0].basis, BASIS_WEB);
        assert_eq!(note.entries[0].sources, ["S1"]);
        // A different source for the same fact adds a second citation.
        store
            .absorb(
                "Kernel Build",
                &[fact(text, BASIS_WEB, &[0])],
                &[web("https://example.org/b")],
            )
            .unwrap();
        assert_eq!(
            store.get("kernel-build").unwrap().unwrap().entries[0].sources,
            ["S1", "S2"]
        );
    }

    #[test]
    fn near_identical_topic_titles_share_one_note_but_unrelated_ones_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store
            .absorb(
                "Linux Kernel Configuration",
                &[fact(
                    "Kernel options live in the .config file.",
                    BASIS_WEB,
                    &[],
                )],
                &[],
            )
            .unwrap();
        let same = store
            .absorb(
                "Configuring the Linux Kernel",
                &[fact(
                    "Menuconfig is a text based configuration interface.",
                    BASIS_WEB,
                    &[],
                )],
                &[],
            )
            .unwrap();
        assert!(!same.created);
        assert_eq!(same.slug, "linux-kernel-configuration");
        let other = store
            .absorb(
                "Linux Distribution Building",
                &[fact(
                    "A distribution needs a package manager and a base system.",
                    BASIS_WEB,
                    &[],
                )],
                &[],
            )
            .unwrap();
        assert!(other.created);
        assert_eq!(store.list().unwrap().len(), 2);
    }

    #[test]
    fn unusable_facts_are_counted_and_an_empty_topic_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let out = store
            .absorb("Nothing Useful", &[fact("too short", BASIS_WEB, &[])], &[])
            .unwrap();
        assert_eq!((out.added, out.rejected, out.created), (0, 1, false));
        assert!(store.list().unwrap().is_empty());
        assert!(!store.path("nothing-useful").exists());
        assert!(store.absorb("   ", &[], &[]).is_err());
    }

    #[test]
    fn notes_are_searchable_and_edits_made_outside_workbench_are_picked_up() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store
            .absorb(
                "Rust Ownership",
                &[fact(
                    "Each value in Rust has a single owner at a time.",
                    BASIS_WEB,
                    &[],
                )],
                &[],
            )
            .unwrap();
        assert_eq!(
            store.search("who owns a value in rust", 5).unwrap()[0].slug,
            "rust-ownership"
        );
        assert!(store
            .search("quantum chromodynamics", 5)
            .unwrap()
            .is_empty());
        // A hand-written file with no header or tags still becomes a searchable note.
        fs::write(
            store.dir().join("zig-comptime.md"),
            "# Zig Comptime\n\n## Notes\n\n- Zig evaluates comptime blocks while compiling.\n",
        )
        .unwrap();
        assert_eq!(store.sync().unwrap(), 1);
        let hit = &store.search("zig comptime compiling", 5).unwrap()[0];
        assert_eq!((hit.title.as_str(), hit.entries.len()), ("Zig Comptime", 1));
        // Reopening in a fresh process sees the same notes without re-reading all of them.
        let reopened = NoteStore::open(dir.path()).unwrap();
        assert_eq!(reopened.list().unwrap().len(), 2);
        assert_eq!(reopened.sync().unwrap(), 0);
    }

    #[test]
    fn deleting_moves_the_file_aside_and_removes_it_from_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store
            .absorb(
                "Temporary Topic",
                &[fact(
                    "This topic exists only for the deletion test.",
                    BASIS_WEB,
                    &[],
                )],
                &[],
            )
            .unwrap();
        assert!(store.delete("temporary-topic").unwrap());
        assert!(store.list().unwrap().is_empty());
        assert!(store.search("deletion test", 5).unwrap().is_empty());
        let kept = fs::read_dir(store.dir().join(".deleted")).unwrap().count();
        assert_eq!(kept, 1);
        assert!(!store.delete("temporary-topic").unwrap());
    }

    #[test]
    fn search_listing_noise_is_not_saved_and_unused_pages_are_not_sources() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let parts = [
            Part {
                text: "Jun 22, 2026 · Ginseng roots usually need five to seven years before they are harvested.",
                source: Some(0),
            },
            Part {
                text: "Discover how long ginseng takes to grow, from seed germination to root maturity.",
                source: Some(1),
            },
            Part {
                text: "The method described below requires seven years of growth before the harvesting, but produces...",
                source: Some(2),
            },
        ];
        let sources = [
            web("https://garden.example/ginseng"),
            web("https://pitch.example/guide"),
            web("https://cut.example/snippet"),
            web("https://grow.example/unrelated"),
        ];
        let facts = extract_facts(&Material {
            topic: None,
            question: "how long does ginseng take to grow",
            parts: &parts,
            basis: BASIS_WEB,
            sources: &sources,
        });
        let texts: Vec<&str> = facts.iter().map(|f| f.text.as_str()).collect();
        assert_eq!(
            texts,
            ["Ginseng roots usually need five to seven years before they are harvested."]
        );
        store.absorb("Growing Ginseng", &facts, &sources).unwrap();
        let note = store.get("growing-ginseng").unwrap().unwrap();
        let urls: Vec<&str> = note.sources.iter().map(|s| s.url.as_str()).collect();
        assert_eq!(urls, ["https://garden.example/ginseng"]);
    }

    #[test]
    fn a_topic_title_does_not_end_on_half_a_phrase() {
        let title = topic_for(
            "research the lisp programming language, open source versions",
            &["Lisp is a programming language. Open source Lisp language versions exist."],
        )
        .unwrap();
        assert!(!title.ends_with("Open"), "{title}");
        assert!(title.starts_with("Lisp"), "{title}");
    }

    #[test]
    fn a_follow_up_is_filed_under_the_conversation_subject() {
        let dir = tempfile::tempdir().unwrap();
        let settings = crate::settings::Settings {
            knowledge_notes: true,
            ..Default::default()
        };
        let parts = [Part {
            text: "Ginseng is grown best in well drained, slightly acidic soil under deep shade.",
            source: Some(0),
        }];
        let sources = [web("https://garden.example/soil")];
        let done = capture(
            dir.path(),
            &settings,
            &Material {
                question: "soil requirements and environment conditions",
                topic: Some("tell me about growing ginseng"),
                parts: &parts,
                basis: BASIS_WEB,
                sources: &sources,
            },
        )
        .unwrap();
        assert_eq!(done.len(), 1);
        assert!(
            done[0].title.to_lowercase().contains("ginseng"),
            "{}",
            done[0].title
        );
    }

    #[test]
    fn a_conversation_is_not_given_its_own_answers_back_as_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let own = IncomingSource {
            kind: "conversation".into(),
            label: "Conversation: tell me about dandelion watering".into(),
            url: String::new(),
        };
        store
            .absorb(
                "Dandelion Watering",
                &[
                    fact(
                        "Dandelion watering needs are moderate and they tolerate drought.",
                        BASIS_MODEL,
                        &[0],
                    ),
                    fact(
                        "Dandelion watering guides say a deep taproot survives dry spells.",
                        BASIS_WEB,
                        &[1],
                    ),
                ],
                &[own, web("https://extension.example/dandelion")],
            )
            .unwrap();
        let mine = vec!["Conversation: tell me about dandelion watering".to_string()];
        let (text, _) = store
            .context_for_excluding("dandelion watering", 4000, &mine)
            .unwrap()
            .unwrap();
        assert!(!text.contains("tolerate drought"), "{text}");
        assert!(text.contains("deep taproot"), "{text}");
        // Another conversation still gets both facts.
        let (other, _) = store
            .context_for("dandelion watering", 4000)
            .unwrap()
            .unwrap();
        assert!(other.contains("tolerate drought"), "{other}");
    }

    #[test]
    fn chat_context_labels_basis_cites_sources_and_says_what_it_omits() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let facts: Vec<IncomingFact> = (0..30)
            .map(|n| {
                fact(
                    &format!("Kernel configuration fact number {n} about menuconfig options."),
                    if n == 0 { BASIS_WEB } else { BASIS_MODEL },
                    if n == 0 { &[0][..] } else { &[] },
                )
            })
            .collect();
        store
            .absorb(
                "Kernel Configuration",
                &facts,
                &[web("https://kernel.org/x")],
            )
            .unwrap();
        let (text, used) = store
            .context_for("kernel configuration menuconfig", 600)
            .unwrap()
            .unwrap();
        assert_eq!(used, ["Kernel Configuration"]);
        assert!(text.starts_with("<knowledge_notes>") && text.ends_with("</knowledge_notes>"));
        assert!(text.contains("model answer, unverified"));
        assert!(
            text.contains("more entries in this note are not shown"),
            "{text}"
        );
        assert!(text.len() < 1500, "{}", text.len());
        let (roomy, _) = store
            .context_for("kernel configuration menuconfig", 8000)
            .unwrap()
            .unwrap();
        assert!(roomy.contains("<https://kernel.org/x>"));
        assert!(!roomy.contains("not shown"), "{roomy}");
        assert!(store
            .context_for("unrelated gardening question", 600)
            .unwrap()
            .is_none());
    }

    const ANSWER: &str = "Running 32-bit code emulation on modern 64-bit CPUs is generally **not worth it** for most use cases, though there are specific scenarios where the trade-off makes sense. Here's what you should consider:

## Performance Overhead (Significant)

| Aspect | Native Execution | Emulated Execution |
|--------|------------------|--------------------|
| Speed | Baseline (100%) | 5-30x slower |

**Why the slowdown?** Emulators such as QEMU translate 32-bit instructions dynamically, which adds interpretation layers that slow compute-intensive tasks.

Most 64-bit x86 processors already run 32-bit code natively in compatibility mode, so emulation is rarely needed for ordinary programs.

## Recommendations

- Use `qemu-user` only for architectures that your CPU cannot execute natively.
- Let me know if you have a specific application in mind.

```bash
sudo apt install qemu-user
```

It also increases memory use. Would you like more detail?";

    fn question() -> &'static str {
        "is it worth running 32 bit cpu emulator on 64 bit cpu?"
    }

    #[test]
    fn the_script_keeps_essential_sentences_word_for_word_and_drops_chatter() {
        let parts = [Part {
            text: ANSWER,
            source: Some(0),
        }];
        let sources = [web("https://example.org")];
        let facts = extract_facts(&Material {
            topic: None,
            question: question(),
            parts: &parts,
            basis: BASIS_MODEL,
            sources: &sources,
        });
        let texts: Vec<&str> = facts.iter().map(|f| f.text.as_str()).collect();
        let all = texts.join("\n");
        assert!(
            texts.iter().any(|t| t.starts_with(
                "Running 32-bit code emulation on modern 64-bit CPUs is generally not worth it"
            )),
            "{all}"
        );
        assert!(
            all.contains("Emulators such as QEMU translate 32-bit instructions dynamically"),
            "{all}"
        );
        assert!(
            all.contains("Most 64-bit x86 processors already run 32-bit code natively"),
            "{all}"
        );
        // A how-to step that states something is kept; the code block under it is not.
        assert!(
            all.contains("Use `qemu-user` only for architectures"),
            "{all}"
        );
        for junk in [
            "Let me know",
            "sudo",
            "Speed",
            "Would you like",
            "It also",
            "Here's",
            "slowdown?",
        ] {
            assert!(!all.contains(junk), "{junk} leaked into: {all}");
        }
        assert!(texts.len() <= MAX_FACTS_ALONE);
        // Every fact is a sentence that really appears in the answer, with emphasis marks only removed.
        let plain = ANSWER.replace('*', "");
        assert!(texts.iter().all(|t| plain.contains(t)), "{all}");
    }

    #[test]
    fn each_fact_is_credited_to_its_own_page_and_the_same_fact_from_two_pages_cites_both() {
        let one = "The kernel reads its build options from the .config file in the source tree.";
        let two = "Menuconfig is a text menu that edits the kernel .config file interactively.";
        let sources = [web("https://a.example/x"), web("https://b.example/y")];
        let a = format!("{one} {two}");
        let parts = [
            Part {
                text: &a,
                source: Some(0),
            },
            Part {
                text: one,
                source: Some(1),
            },
        ];
        let facts = extract_facts(&Material {
            topic: None,
            question: "how is the kernel .config file edited",
            parts: &parts,
            basis: BASIS_WEB,
            sources: &sources,
        });
        assert!(
            facts.iter().any(|f| f.text == two && f.source_ix == [0]),
            "{facts:?}"
        );
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store.absorb("Kernel Config", &facts, &sources).unwrap();
        let note = store.get("kernel-config").unwrap().unwrap();
        let shared = note.entries.iter().find(|e| e.text == one).unwrap();
        assert_eq!(shared.sources, ["S1", "S2"], "{note:?}");
    }

    #[test]
    fn sentences_that_cannot_stand_alone_or_are_not_knowledge_are_refused() {
        assert!(usable_sentence(
            "The borrow checker rejects programs that could alias a mutable reference.",
            false
        ));
        for bad in [
            "It prevents data races at compile time in every case.",
            "However, the compiler still checks every reference carefully.",
            "Would you like more detail about this particular setup today?",
            "What does the borrow checker actually verify in a program?",
            "Here is the list of things the compiler checks for you:",
            "You should always profile before choosing an emulator for this.",
            "Short sentence.",
            "Void Linux: Simple, no systemd by default.",
            "Interactive text-based GUI with tabs and menus",
            "Kernel Land → isolated process trees per environment today.",
            "Below is a structured guide covering the most common configurations.",
            "I\u{2019}ll focus on Debian based systems since they are widely used everywhere.",
            "| Aspect | Native Execution | Emulated Execution | Notes here |",
            "I don't have filesystem access to inspect that folder in this session.",
        ] {
            assert!(!usable_sentence(bad, false), "{bad}");
        }
    }

    #[test]
    fn a_bullet_without_a_full_stop_counts_only_when_it_states_something() {
        assert!(usable_sentence(
            "Can save configuration to `.config` file automatically",
            true
        ));
        assert!(!usable_sentence(
            "Can save configuration to `.config` file automatically",
            false
        ));
        assert!(!usable_sentence(
            "Interactive text-based GUI with tabs and menus and shortcuts",
            true
        ));
        assert!(!usable_sentence(
            "Dependency errors: run olddefconfig after major changes",
            true
        ));
    }

    #[test]
    fn chatter_phrases_match_whole_words_only() {
        assert!(usable_sentence(
            "Most modern CPUs have specific instructions where there are no safe equivalents.",
            false
        ));
        assert!(usable_sentence(
            "AI can classify images faster than a hand written rule set could manage.",
            false
        ));
        assert!(!usable_sentence(
            "Here are the main things that the compiler checks for each reference.",
            false
        ));
    }

    #[test]
    fn sentence_splitting_respects_abbreviations_and_version_numbers() {
        let parts = split_sentences("Python 3.12 adds faster startup, e.g. for short scripts. The GIL is unchanged in this release! Is it?");
        assert_eq!(parts.len(), 3, "{parts:?}");
        assert!(parts[0].ends_with("short scripts."), "{parts:?}");
    }

    #[test]
    fn topics_come_from_the_subject_words_of_the_question() {
        let title = topic_for(question(), &[ANSWER]).unwrap();
        assert!(
            title.contains("Emulator") && title.contains("Bit"),
            "{title}"
        );
        assert!(
            !title.to_lowercase().contains("worth") && title.split(' ').count() <= MAX_TOPIC_WORDS,
            "{title}"
        );
        assert!(topic_from_question("tell me how to configure linux")
            .unwrap()
            .contains("Linux"));
        // Words the text is not about are not title words, however early they come.
        let answer = "Userland is everything outside the kernel. Each user land has its own binaries and libraries, while the kernel stays shared between every user land.";
        let title = topic_for(
            "if I create three user lands from the linux kernel, which directories move?",
            &[answer],
        )
        .unwrap();
        assert!(
            title.contains("User")
                && title.contains("Kernel")
                && !title.contains("If")
                && !title.contains("Three"),
            "{title}"
        );
        let rust = "Rust's borrow checker prevents data races. The borrow checker enforces ownership, so each value has one owner. Shared references allow reads, mutable references allow writes, and shared data is never mutated through a shared reference. Data crossing threads must be Send, and data shared between threads must be Sync.";
        let title = topic_for("Explain how the Rust borrow checker prevents data races, including ownership, shared and mutable references, and the Send and Sync traits.", &[rust]).unwrap();
        assert!(
            title.contains("Rust") && title.contains("Borrow") && title.contains("Checker"),
            "{title}"
        );
        // Nothing to be about: numbers only, or only words that ask for something.
        assert_eq!(topic_from_question("?!"), None);
        assert_eq!(topic_from_question("1 and 3"), None);
        assert_eq!(topic_from_question("please explain more"), None);
    }

    #[test]
    fn capture_builds_a_note_without_any_model_and_repeating_it_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings {
            knowledge_notes: true,
            ..Default::default()
        };
        let parts = [Part {
            text: ANSWER,
            source: Some(0),
        }];
        let sources = [IncomingSource {
            kind: "conversation".into(),
            label: "Conversation: emulation".into(),
            url: String::new(),
        }];
        let material = Material {
            topic: None,
            question: question(),
            parts: &parts,
            basis: BASIS_MODEL,
            sources: &sources,
        };
        let done = capture(dir.path(), &settings, &material).unwrap();
        assert_eq!(done.len(), 1);
        assert!(done[0].created && done[0].added >= 2, "{done:?}");
        let store = NoteStore::open(dir.path()).unwrap();
        let note = store.get(&done[0].slug).unwrap().unwrap();
        assert!(note
            .entries
            .iter()
            .all(|e| e.basis == BASIS_MODEL && e.sources == ["S1"]));
        assert!(capture(dir.path(), &settings, &material)
            .unwrap()
            .is_empty());
        let off = Settings {
            knowledge_notes: false,
            ..Default::default()
        };
        assert!(capture(dir.path(), &off, &material).unwrap().is_empty());
        let secret = Material {
            topic: None,
            question: "my password is hunter2 what now",
            parts: &parts,
            basis: BASIS_MODEL,
            sources: &sources,
        };
        assert!(capture(dir.path(), &settings, &secret).unwrap().is_empty());
    }

    fn source(kind: &str, url: &str, content: &str) -> crate::research::ResearchSource {
        crate::research::ResearchSource {
            id: "R1".into(),
            kind: kind.into(),
            title: "A page".into(),
            url: url.into(),
            path: None,
            start_line: None,
            end_line: None,
            content: content.into(),
            content_sha256: String::new(),
            truncated: false,
            fetched_at: String::new(),
            notes: "an analysis the model wrote".into(),
            analysis_error: None,
        }
    }

    fn run_with(sources: Vec<crate::research::ResearchSource>) -> crate::research::ResearchRun {
        crate::research::ResearchRun {
            id: "r".into(),
            question: "How are kernels configured?".into(),
            workspace: None,
            model: "m".into(),
            state: "complete".into(),
            rounds_completed: 1,
            queries: vec![],
            sources,
            answer: "Kernels are configured with menuconfig.".into(),
            warnings: vec![],
            stop_reason: String::new(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn research_notes_use_the_retrieved_pages_not_the_models_analysis_and_never_local_content() {
        let page = "Menuconfig is a text menu that edits the kernel .config file interactively.";
        let web_only = run_with(vec![source("web", "https://kernel.org/doc", page)]);
        let (parts, sources) = research_material(&web_only).unwrap();
        assert_eq!((parts.len(), sources.len()), (1, 1));
        assert_eq!(parts[0], (page.to_string(), Some(0)));
        assert!(!parts[0].0.contains("an analysis the model wrote"));

        let mixed = run_with(vec![
            source("web", "https://kernel.org/doc", page),
            source("local", "", "private internal design notes"),
        ]);
        let (parts, sources) = research_material(&mixed).unwrap();
        assert_eq!(sources.len(), 1);
        assert!(parts.iter().all(|(t, _)| !t.contains("private internal")));

        let local_only = run_with(vec![source("local", "", "private")]);
        assert!(research_material(&local_only).is_none());
        let empty_page = run_with(vec![source("web", "https://x.org", "  ")]);
        assert!(research_material(&empty_page).is_none());
    }

    #[test]
    fn a_research_run_becomes_a_note_credited_to_its_pages() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings {
            knowledge_notes: true,
            ..Default::default()
        };
        let run = run_with(vec![source(
            "web",
            "https://kernel.org/doc",
            "Menuconfig is a text menu that edits the kernel configuration file interactively. The selected options are saved to the .config file in the source tree.",
        )]);
        let done = capture_research(&settings, dir.path(), &run).unwrap();
        assert_eq!(done.len(), 1, "{done:?}");
        let note = NoteStore::open(dir.path())
            .unwrap()
            .get(&done[0].slug)
            .unwrap()
            .unwrap();
        assert!(
            note.entries
                .iter()
                .all(|e| e.basis == BASIS_RESEARCH && e.sources == ["S1"]),
            "{note:?}"
        );
        assert_eq!(note.sources[0].url, "https://kernel.org/doc");
    }

    #[test]
    fn statements_about_the_assistant_or_its_access_are_not_saved_as_knowledge() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        let out = store
            .absorb(
                "32-Bit Emulation",
                &[
                    fact("The material indicates limited code analysis capabilities without direct file access.", BASIS_MODEL, &[]),
                    fact("I don't have filesystem access in this session unless a folder is connected.", BASIS_MODEL, &[]),
                    fact("Is emulation worth the overhead on a modern CPU?", BASIS_MODEL, &[]),
                    fact("Emulating 32-bit code on a 64-bit CPU adds translation overhead compared with native execution.", BASIS_MODEL, &[]),
                ],
                &[],
            )
            .unwrap();
        assert_eq!((out.added, out.rejected), (1, 3));
        assert_eq!(
            store
                .get("32-bit-emulation")
                .unwrap()
                .unwrap()
                .entries
                .len(),
            1
        );
    }

    #[test]
    fn votes_add_up_to_agreed_mixed_or_disputed() {
        let c = |agree, disagree, unsure| Consensus {
            agree,
            disagree,
            unsure,
        };
        assert_eq!(c(3, 0, 0).status(), ReviewStatus::Agreed);
        assert_eq!(c(2, 0, 0).status(), ReviewStatus::Agreed);
        // One model alone cannot agree with anyone.
        assert_eq!(c(1, 0, 0).status(), ReviewStatus::Mixed);
        assert_eq!(c(2, 0, 1).status(), ReviewStatus::Mixed);
        assert_eq!(c(2, 1, 0).status(), ReviewStatus::Mixed);
        assert_eq!(c(1, 1, 1).status(), ReviewStatus::Disputed);
        assert_eq!(c(0, 2, 1).status(), ReviewStatus::Disputed);
        assert_eq!(c(3, 0, 0).label(), "all 3 models agree");
        assert_eq!(c(2, 1, 0).label(), "2 of 3 models agree, 1 disagree");
        assert!(c(1, 2, 0).label().starts_with("disputed: 2 of 3"));
    }

    #[test]
    fn verdicts_are_parsed_strictly_and_a_skipped_statement_is_unsure_not_agreement() {
        let v = parse_verdicts(
            r#"{"verdicts":[{"id":1,"verdict":"true"},{"id":3,"verdict":"FALSE"}]}"#,
            3,
        )
        .unwrap();
        assert_eq!(v, [Verdict::True, Verdict::Unsure, Verdict::False]);
        assert!(parse_verdicts("I think they are all true", 2).is_err());
        assert!(parse_verdicts(r#"{"answer":"true"}"#, 2).is_err());
        let out_of_range =
            parse_verdicts(r#"{"verdicts":[{"id":9,"verdict":"true"}]}"#, 2).unwrap();
        assert_eq!(out_of_range, [Verdict::Unsure, Verdict::Unsure]);
        assert_eq!(
            tally(&[
                Verdict::True,
                Verdict::True,
                Verdict::False,
                Verdict::Unsure
            ]),
            Consensus {
                agree: 2,
                disagree: 1,
                unsure: 1
            }
        );
    }

    #[test]
    fn the_review_tally_and_log_survive_the_markdown_file_and_the_index() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store
            .absorb(
                "Rust Ownership",
                &[
                    fact(
                        "Each value in Rust has a single owner at a time.",
                        BASIS_MODEL,
                        &[],
                    ),
                    fact(
                        "Rust frees memory with a tracing garbage collector at runtime.",
                        BASIS_MODEL,
                        &[],
                    ),
                    fact(
                        "Moving a value transfers ownership to the new binding.",
                        BASIS_MODEL,
                        &[],
                    ),
                ],
                &[],
            )
            .unwrap();
        let results = vec![
            (
                "Each value in Rust has a single owner at a time.".to_string(),
                Consensus {
                    agree: 3,
                    disagree: 0,
                    unsure: 0,
                },
            ),
            (
                "Rust frees memory with a tracing garbage collector at runtime.".to_string(),
                Consensus {
                    agree: 0,
                    disagree: 3,
                    unsure: 0,
                },
            ),
            (
                "Moving a value transfers ownership to the new binding.".to_string(),
                Consensus {
                    agree: 2,
                    disagree: 0,
                    unsure: 1,
                },
            ),
        ];
        assert_eq!(
            store
                .apply_reviews(
                    "rust-ownership",
                    &results,
                    "2026-11-09 · models: a, b, c · 3 fact(s)"
                )
                .unwrap(),
            3
        );
        let text = fs::read_to_string(store.path("rust-ownership")).unwrap();
        assert!(
            text.contains("{review: 3 agree, 0 disagree, 0 unsure} · "),
            "{text}"
        );
        assert!(
            text.contains("## Reviews") && text.contains("models: a, b, c"),
            "{text}"
        );
        let note = store.get("rust-ownership").unwrap().unwrap();
        assert_eq!(parse_markdown(&render_markdown(&note), &note.slug), note);
        assert_eq!(note.tally(), (1, 1, 1));
        let summary = &NoteStore::open(dir.path()).unwrap().list().unwrap()[0];
        assert_eq!((summary.agreed, summary.mixed, summary.disputed), (1, 1, 1));
        // A fact that was edited since the models judged it is not given their verdict.
        assert_eq!(
            store
                .apply_reviews(
                    "rust-ownership",
                    &[("some other text".into(), Consensus::default())],
                    "x"
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn reviewers_are_different_installed_models_and_leave_out_the_author() {
        let mut settings = Settings::default();
        let installed: Vec<String> = [
            "qwen3.5:4b",
            "qwen3.5:9b",
            "ministral-3:8b",
            "gemma4:e4b",
            "deepseek-ocr:latest",
        ]
        .map(String::from)
        .to_vec();
        settings.notes_review_models = vec![
            "qwen3.5:9b".into(),
            "qwen3.5:4b".into(),
            "ministral-3:8b".into(),
            "gemma4:e4b".into(),
            "not-installed:1b".into(),
        ];
        settings.notes_review_count = 3;
        // Different families first, even though both qwen sizes come earlier in the list.
        assert_eq!(
            pick_reviewers(&settings, &installed, None),
            ["qwen3.5:9b", "ministral-3:8b", "gemma4:e4b"]
        );
        // The author never reviews its own answer while enough others remain.
        assert_eq!(
            pick_reviewers(&settings, &installed, Some("qwen3.5:9b")),
            ["qwen3.5:4b", "ministral-3:8b", "gemma4:e4b"]
        );
        // With too few others, the author is kept so a comparison is still possible.
        settings.notes_review_models = vec!["qwen3.5:9b".into(), "ministral-3:8b".into()];
        assert_eq!(
            pick_reviewers(&settings, &installed, Some("qwen3.5:9b")),
            ["qwen3.5:9b", "ministral-3:8b"]
        );
        settings.notes_review_count = 99;
        settings.notes_review_models = installed.clone();
        assert_eq!(pick_reviewers(&settings, &installed, None).len(), 5);
        // Nothing configured and nothing installed in the routing profile: no reviewers.
        settings.notes_review_models.clear();
        assert!(pick_reviewers(&settings, &["x:1b".to_string()], None).is_empty());
    }

    #[test]
    fn chat_context_shows_consensus_and_withholds_what_the_models_called_wrong() {
        let dir = tempfile::tempdir().unwrap();
        let store = NoteStore::open(dir.path()).unwrap();
        store
            .absorb(
                "Kernel Build",
                &[
                    fact(
                        "The kernel build reads its options from the .config file.",
                        BASIS_MODEL,
                        &[],
                    ),
                    fact(
                        "The kernel build reads its options from the registry hive.",
                        BASIS_MODEL,
                        &[],
                    ),
                    fact(
                        "The kernel build compiles object files with the system compiler.",
                        BASIS_MODEL,
                        &[],
                    ),
                ],
                &[],
            )
            .unwrap();
        store
            .apply_reviews(
                "kernel-build",
                &[
                    (
                        "The kernel build reads its options from the .config file.".into(),
                        Consensus {
                            agree: 3,
                            disagree: 0,
                            unsure: 0,
                        },
                    ),
                    (
                        "The kernel build reads its options from the registry hive.".into(),
                        Consensus {
                            agree: 0,
                            disagree: 3,
                            unsure: 0,
                        },
                    ),
                ],
                "log",
            )
            .unwrap();
        let (text, _) = store
            .context_for("how does the kernel build read options", 4000)
            .unwrap()
            .unwrap();
        assert!(
            text.contains(".config file. (model answer, unverified; all 3 models agree)"),
            "{text}"
        );
        assert!(
            !text.contains("registry hive"),
            "a disputed fact reached the chat model: {text}"
        );
        assert!(
            text.contains("1 entries that the reviewing models judged wrong are not shown"),
            "{text}"
        );
        assert!(
            text.contains("system compiler. (model answer, unverified)"),
            "{text}"
        );
    }
}
