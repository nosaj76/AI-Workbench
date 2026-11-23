//! Repo map: a ranked outline of a project's code that fits a small budget.
//!
//! A model that has to explore a large project with file listings and searches
//! wastes its few tool rounds. The map gives it the project's shape up front:
//! the files that the rest of the code relies on most, each with its key
//! definitions (functions, types, classes) and their line numbers.
//!
//! How it works: every source file is scanned for definitions (by language),
//! and every file is scanned for mentions of definitions made elsewhere. That
//! makes a graph of which files rely on which; files are ranked with a
//! personalized PageRank, so the files and names the user mentions rank
//! higher. The map lists files in rank order until the budget is used, and says
//! how many files it did not show and how to page on.
//!
//! The approach follows aider's repository map (Apache-2.0); this is an
//! independent implementation that uses simple per-language patterns instead of
//! a syntax-tree parser, so it can miss unusual declarations.
use crate::files::Workspace;
use anyhow::Result;
use regex::Regex;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

const EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "go", "java", "kt", "kts", "cs", "scala",
    "swift", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "rb", "php", "sh", "bash", "lua", "ex",
    "exs", "dart", "zig",
];

/// Directories holding other people's code (dependency copies, generated
/// builds): never entered, because mapping them says nothing about the project.
const VENDORED: &[&str] = &[
    "vendor",
    "node_modules",
    "target",
    "dist",
    "registry",
    "third_party",
    "third-party",
    "site-packages",
    "dist-packages",
    ".cargo",
    "bower_components",
    "Pods",
    "_deps",
    "extern",
    "external",
    "vendored",
    ".tox",
    ".mypy_cache",
    "__pypackages__",
];

/// How long a scan that a limit cut short is reused before scanning again.
const PARTIAL_SCAN_LIFETIME: Duration = Duration::from_secs(120);

/// A name defined in this many files or more (`new`, `main`, `get`…) says
/// nothing about which file matters, so it is not used to link files.
const GENERIC_DEFINITION_FILES: usize = 8;
const MAX_SIGNATURE_CHARS: usize = 160;
const SYMBOLS_PER_FILE: usize = 8;
const DAMPING: f64 = 0.85;
const ITERATIONS: usize = 40;
const FOCUS_BOOST: f64 = 50.0;

/// What bounds the scan. A project beyond these is mapped in part, and the map
/// says so.
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_visited: usize,
    pub max_files: usize,
    pub max_file_bytes: u64,
    /// Wall-clock allowance for the whole scan.
    pub max_seconds: u64,
}

impl Default for Limits {
    /// For the `repo_map` tool, which a model asks for on purpose.
    fn default() -> Self {
        Self {
            max_visited: 300_000,
            max_files: 6_000,
            max_file_bytes: 256 * 1024,
            max_seconds: 12,
        }
    }
}

impl Limits {
    /// For the map added to every chat, where a long scan would delay the reply.
    pub fn quick() -> Self {
        Self {
            max_visited: 60_000,
            max_files: 2_500,
            max_file_bytes: 128 * 1024,
            max_seconds: 3,
        }
    }
}

pub struct MapRequest<'a> {
    /// Text naming what the user cares about; file names and defined names in
    /// it raise the rank of the files concerned.
    pub focus: &'a str,
    /// Only list files under this project-relative directory.
    pub under: Option<&'a str>,
    pub max_chars: usize,
    /// Ranked files to skip, to page through a larger map.
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct RepoMap {
    pub text: String,
    pub files_shown: usize,
    /// Files that have at least one definition and match the request.
    pub files_with_definitions: usize,
    pub files_scanned: usize,
    /// False when a scan limit ended the walk, so more source files exist.
    pub scan_complete: bool,
    pub next_offset: Option<usize>,
}

#[derive(Clone, Debug)]
struct Def {
    name: String,
    kind: String,
    line: usize,
    signature: String,
    /// Part of the file's public surface (or, for languages without a keyword, named like it).
    public: bool,
}

struct Index {
    files: Vec<String>,
    defs: Vec<Vec<Def>>,
    /// (referencing file, defining file, weight)
    edges: Vec<(usize, usize, f64)>,
    name_files: BTreeMap<String, Vec<usize>>,
    /// Mentions of each defined name in files that do not define it.
    mentions: BTreeMap<String, u32>,
    complete: bool,
}

// ------------------------------------------------------------ definitions --

enum Kind {
    Fixed(&'static str),
    Group(usize),
}

struct Pattern {
    re: Regex,
    kind: Kind,
    name: usize,
}

fn pat(source: &str, kind: Kind, name: usize) -> Pattern {
    Pattern {
        re: Regex::new(source).expect("valid built-in pattern"),
        kind,
        name,
    }
}

fn languages() -> &'static HashMap<&'static str, Arc<Vec<Pattern>>> {
    static LANGUAGES: OnceLock<HashMap<&'static str, Arc<Vec<Pattern>>>> = OnceLock::new();
    LANGUAGES.get_or_init(|| {
        let rust = Arc::new(vec![
            pat(r#"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+|async\s+|unsafe\s+|const\s+|extern\s+"[^"]*"\s+)*(fn|struct|enum|trait|type|mod|union)\s+([A-Za-z_]\w*)"#, Kind::Group(1), 2),
            pat(r"^\s*impl(?:<[^>]*>)?\s+(?:[^{]*?\sfor\s+)?([A-Za-z_]\w*)", Kind::Fixed("impl"), 1),
            pat(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?([A-Z_][A-Z0-9_]*)\s*:", Kind::Fixed("const"), 1),
        ]);
        let python = Arc::new(vec![pat(r"^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)", Kind::Group(1), 2)]);
        let script = Arc::new(vec![
            pat(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(function\*?|class|interface|enum|type)\s+([A-Za-z_$][\w$]*)", Kind::Group(1), 2),
            pat(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>", Kind::Fixed("function"), 1),
        ]);
        let go = Arc::new(vec![
            pat(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", Kind::Fixed("func"), 1),
            pat(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", Kind::Fixed("type"), 1),
        ]);
        let jvm = Arc::new(vec![pat(
            r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|open|sealed|data|override|partial|async|suspend|inline|case)\s+)*(class|interface|enum|record|struct|trait|object|fun|func|def|protocol)\s+([A-Za-z_]\w*)",
            Kind::Group(1),
            2,
        )]);
        let java = Arc::new(vec![
            pat(
                r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|open|sealed|data|override|partial|async|suspend|inline|case)\s+)*(class|interface|enum|record|struct|trait|object|fun|func|def|protocol)\s+([A-Za-z_]\w*)",
                Kind::Group(1),
                2,
            ),
            pat(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|override|virtual|async|synchronized)\s+)+[\w<>\[\],\s?]+?\s+([A-Za-z_]\w*)\s*\(", Kind::Fixed("method"), 1),
        ]);
        let c = Arc::new(vec![
            pat(r"^(?:typedef\s+)?(struct|class|enum|union|namespace)\s+([A-Za-z_]\w*)", Kind::Group(1), 2),
            pat(r"^[A-Za-z_][\w\s\*:<>,&~]*?[\s\*&]([A-Za-z_~][\w:~]*)\s*\([^;]*$", Kind::Fixed("function"), 1),
        ]);
        let ruby = Arc::new(vec![pat(r"^\s*(def|class|module)\s+(?:self\.)?([A-Za-z_]\w*[?!]?)", Kind::Group(1), 2)]);
        let php = Arc::new(vec![pat(
            r"^\s*(?:abstract\s+|final\s+)?(?:(?:public|private|protected|static)\s+)*(function|class|interface|trait)\s+([A-Za-z_]\w*)",
            Kind::Group(1),
            2,
        )]);
        let shell = Arc::new(vec![
            pat(r"^\s*function\s+([A-Za-z_][\w-]*)", Kind::Fixed("function"), 1),
            pat(r"^\s*([A-Za-z_][\w-]*)\s*\(\)\s*\{?", Kind::Fixed("function"), 1),
        ]);
        let lua = Arc::new(vec![pat(r"^\s*(?:local\s+)?function\s+([A-Za-z_][\w.:]*)", Kind::Fixed("function"), 1)]);
        let elixir = Arc::new(vec![pat(r"^\s*(defmodule|defp|def|defmacro)\s+([A-Za-z_][\w.?!]*)", Kind::Group(1), 2)]);
        let dart = Arc::new(vec![pat(r"^\s*(?:abstract\s+)?(class|enum|mixin|extension)\s+([A-Za-z_]\w*)", Kind::Group(1), 2)]);
        let zig = Arc::new(vec![pat(r"^\s*(?:pub\s+)?(fn|const)\s+([A-Za-z_]\w*)\s*(?:\(|=\s*(?:struct|enum|union))", Kind::Group(1), 2)]);
        let mut map: HashMap<&'static str, Arc<Vec<Pattern>>> = HashMap::new();
        map.insert("rs", rust);
        map.insert("py", python);
        for e in ["js", "jsx", "ts", "tsx", "mjs", "cjs"] {
            map.insert(e, script.clone());
        }
        map.insert("go", go);
        for e in ["kt", "kts", "scala", "swift"] {
            map.insert(e, jvm.clone());
        }
        for e in ["java", "cs"] {
            map.insert(e, java.clone());
        }
        for e in ["c", "h", "cc", "cpp", "cxx", "hpp", "hh"] {
            map.insert(e, c.clone());
        }
        map.insert("rb", ruby);
        map.insert("php", php);
        for e in ["sh", "bash"] {
            map.insert(e, shell.clone());
        }
        map.insert("lua", lua);
        for e in ["ex", "exs"] {
            map.insert(e, elixir.clone());
        }
        map.insert("dart", dart);
        map.insert("zig", zig);
        map
    })
}

/// Words that look like a function name to the C pattern but are statements.
const NOT_FUNCTIONS: &[&str] = &[
    "if", "for", "while", "switch", "return", "else", "sizeof", "do", "catch", "defined",
];

fn signature(line: &str) -> String {
    let trimmed = line.trim();
    // A one-line function shows its header, not its body.
    let header = match trimmed.find('{') {
        Some(at) if at > 0 => &trimmed[..at],
        _ => trimmed,
    };
    let mut compact = header
        .trim_end_matches(['{', ':', ' '])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    // Python: `def f(x) -> int: return x` ends at the header's colon.
    if compact.starts_with("def ")
        || compact.starts_with("async def ")
        || compact.starts_with("class ")
    {
        if let Some(arrow) = compact.find(")->").or_else(|| compact.find(") ->")) {
            if let Some(colon) = compact[arrow..].find(':') {
                compact.truncate(arrow + colon);
            }
        } else if let Some(end) = compact.find("):") {
            compact.truncate(end + 1);
        }
    }
    if compact.chars().count() > MAX_SIGNATURE_CHARS {
        let cut: String = compact.chars().take(MAX_SIGNATURE_CHARS - 1).collect();
        format!("{cut}…")
    } else {
        compact
    }
}

fn definitions(extension: &str, text: &str) -> Vec<Def> {
    let Some(patterns) = languages().get(extension.to_ascii_lowercase().as_str()) else {
        return Vec::new();
    };
    let c_family = matches!(extension, "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh");
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        // Rust unit tests sit in a trailing test module; they are not the file's shape.
        if extension == "rs" && line.trim_start().starts_with("#[cfg(test)]") {
            break;
        }
        if line.len() > 400 {
            continue;
        }
        for p in patterns.iter() {
            let Some(captures) = p.re.captures(line) else {
                continue;
            };
            let name = captures[p.name].to_string();
            if c_family && NOT_FUNCTIONS.contains(&name.as_str()) {
                continue;
            }
            let kind = match &p.kind {
                Kind::Fixed(k) => (*k).to_string(),
                Kind::Group(g) => captures[*g].to_string(),
            };
            let signature = signature(line);
            // Prose in a comment can look like `word word name(…`; a declaration
            // line ends the way a declaration does.
            if c_family && p.name == 1 && matches!(&p.kind, Kind::Fixed("function")) {
                let ends = [")", "{", ",", "const", "noexcept", "override", "\\"];
                if !ends.iter().any(|e| signature.ends_with(e)) && !line.trim_end().ends_with('{') {
                    continue;
                }
            }
            // `mod name;` only declares a file; it says nothing about the code.
            if kind == "mod" && signature.ends_with(';') {
                break;
            }
            if name.starts_with("test_") || name.starts_with("Test") {
                break;
            }
            let trimmed = line.trim_start();
            let public = trimmed.starts_with("pub")
                || trimmed.starts_with("export")
                || match extension {
                    "py" | "rb" | "lua" | "sh" | "bash" | "ex" | "exs" => !name.starts_with('_'),
                    "go" => name.chars().next().is_some_and(char::is_uppercase),
                    "java" | "cs" | "kt" | "scala" | "swift" | "php" => {
                        trimmed.starts_with("public")
                    }
                    _ => false,
                };
            out.push(Def {
                name,
                kind,
                line: index + 1,
                signature,
                public,
            });
            break;
        }
    }
    out
}

// ------------------------------------------------------------------ index --

fn read_source(workspace: &Workspace, relative: &str) -> Option<String> {
    let bytes = fs::read(workspace.root().join(relative)).ok()?;
    let text = String::from_utf8(bytes).ok()?;
    // Minified or generated code has no useful outline.
    let lines = text.lines().count().max(1);
    if text.len() > 4000 && text.len() / lines > 300 {
        return None;
    }
    Some(text)
}

fn is_test_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.contains("/tests/")
        || lower.starts_with("tests/")
        || lower.contains("/test/")
        || lower.starts_with("test/")
        || lower.contains("/__tests__/")
        || lower.contains("_test.")
        || lower.contains(".test.")
        || lower.contains(".spec.")
        || lower
            .rsplit('/')
            .next()
            .is_some_and(|f| f.starts_with("test_"))
}

fn weight_multiplier(name: &str) -> f64 {
    let mixed = name.chars().any(char::is_lowercase) && name.chars().any(char::is_uppercase);
    if name.starts_with('_') {
        0.1
    } else if name.len() >= 8 && (name.contains('_') || mixed) {
        4.0
    } else {
        1.0
    }
}

fn build_index(
    workspace: &Workspace,
    paths: &[String],
    mut complete: bool,
    deadline: Instant,
) -> Index {
    let mut files = Vec::new();
    let mut defs = Vec::new();
    for path in paths {
        if Instant::now() >= deadline {
            // Out of time: the files read so far are mapped, and the map says it is partial.
            complete = false;
            break;
        }
        let Some(text) = read_source(workspace, path) else {
            continue;
        };
        let extension = path.rsplit('.').next().unwrap_or("");
        files.push(path.clone());
        defs.push(definitions(extension, &text));
    }
    let mut name_files: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, file_defs) in defs.iter().enumerate() {
        for d in file_defs {
            let entry = name_files.entry(d.name.clone()).or_default();
            if !entry.contains(&i) {
                entry.push(i);
            }
        }
    }
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    let token =
        TOKEN.get_or_init(|| Regex::new(r"[A-Za-z_$][A-Za-z0-9_$]{2,}").expect("valid pattern"));
    let mut weights: HashMap<(usize, usize), f64> = HashMap::new();
    let mut mentions: BTreeMap<String, u32> = BTreeMap::new();
    for (a, path) in files.iter().enumerate() {
        if Instant::now() >= deadline {
            complete = false;
            break;
        }
        let Some(text) = read_source(workspace, path) else {
            continue;
        };
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for m in token.find_iter(&text) {
            let word = m.as_str();
            if name_files
                .get(word)
                .is_some_and(|d| d.len() < GENERIC_DEFINITION_FILES)
            {
                *counts.entry(word).or_default() += 1;
            }
        }
        for (word, count) in counts {
            let definers = &name_files[word];
            if definers.contains(&a) {
                continue;
            }
            *mentions.entry(word.to_string()).or_default() += count;
            let each = weight_multiplier(word) * f64::from(count).sqrt() / definers.len() as f64;
            for b in definers {
                *weights.entry((a, *b)).or_default() += each;
            }
        }
    }
    Index {
        files,
        defs,
        edges: weights.into_iter().map(|((a, b), w)| (a, b, w)).collect(),
        name_files,
        mentions,
        complete,
    }
}

fn cached_index(
    workspace: &Workspace,
    limits: &Limits,
    only: Option<&str>,
) -> Result<(Arc<Index>, usize)> {
    // Keyed by the limits too: a small scan made for a chat must not stand in
    // for the larger one a model asks for on purpose.
    type Key = (std::path::PathBuf, usize, u64, String);
    type Cache = Mutex<HashMap<Key, (u64, Instant, Arc<Index>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key: Key = (
        workspace.root().to_path_buf(),
        limits.max_files,
        limits.max_seconds,
        only.unwrap_or("").to_string(),
    );
    // A scan that a limit cut short is expensive to repeat and would differ each
    // time; reuse it briefly instead of rescanning on every chat message.
    if let Ok(guard) = cache.lock() {
        if let Some((_, built, index)) = guard.get(&key) {
            if !index.complete && built.elapsed() < PARTIAL_SCAN_LIFETIME {
                return Ok((index.clone(), index.files.len()));
            }
        }
    }
    let started = Instant::now();
    let deadline = started + Duration::from_secs(limits.max_seconds);
    // Walking gets part of the time; reading and ranking the files gets the rest.
    let walk_deadline = started + Duration::from_millis(limits.max_seconds * 400);
    // Shallow first: in a huge tree the project's own code sits near the top, and
    // a deeper pass is made only while time remains. Only a finished unlimited
    // pass counts as a complete scan.
    let mut chosen: Option<crate::files::SourceListing> = None;
    for depth in [Some(3), Some(6), None] {
        let mut pass = workspace.source_files(&crate::files::SourceQuery {
            extensions: EXTENSIONS,
            max_visited: limits.max_visited,
            max_files: limits.max_files,
            max_bytes: limits.max_file_bytes,
            prune: VENDORED,
            max_depth: depth,
            only,
            deadline: Some(walk_deadline),
        })?;
        let whole = depth.is_none() && pass.complete;
        if depth.is_some() {
            // A shallower pass has not seen the deeper files.
            pass.complete = false;
        }
        // A pass the clock cut short can hold fewer files than the shallower one
        // that finished; keep whichever found more.
        if chosen
            .as_ref()
            .is_none_or(|c| pass.files.len() >= c.files.len())
        {
            chosen = Some(pass);
        }
        if whole || Instant::now() >= walk_deadline {
            break;
        }
    }
    let mut listing = chosen.expect("at least one pass runs");
    let scanned = listing.files.len();
    // The index is rebuilt only when a source file was added, removed or changed.
    let signature = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for f in &listing.files {
            (&f.path, f.size, f.modified, f.change).hash(&mut hasher);
            // Where both clocks expose only whole seconds, identical metadata
            // can conceal a same-size rewrite. Bound this fallback by the same
            // deadline as parsing; precise ctime avoids it on Linux filesystems.
            let fractional_mtime = f.modified
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .is_some_and(|t| t.subsec_nanos() != 0);
            if !fractional_mtime && !f.change.is_some_and(|(_, _, ns)| ns != 0) {
                if Instant::now() >= deadline {
                    listing.complete = false;
                    break;
                }
                workspace.read_text(&f.path, limits.max_file_bytes as usize)
                    .ok().map(|doc| doc.text).hash(&mut hasher);
            }
        }
        listing.complete.hash(&mut hasher);
        hasher.finish()
    };
    if let Ok(guard) = cache.lock() {
        if let Some((known, _, index)) = guard.get(&key) {
            if *known == signature && index.complete == listing.complete {
                return Ok((index.clone(), scanned));
            }
        }
    }
    let paths: Vec<String> = listing.files.iter().map(|f| f.path.clone()).collect();
    let index = Arc::new(build_index(workspace, &paths, listing.complete, deadline));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, (signature, Instant::now(), index.clone()));
    }
    Ok((index, scanned))
}

// ---------------------------------------------------------------- ranking --

fn focus_files(index: &Index, focus: &str) -> Vec<usize> {
    let mut hit = Vec::new();
    for token in focus
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '-' | '/' | '$')))
        .map(|t| t.trim_matches(['.', '-', '/']))
        .filter(|t| t.chars().count() >= 3)
    {
        let lower = token.to_lowercase();
        for (i, path) in index.files.iter().enumerate() {
            let path_lower = path.to_lowercase();
            let stem = path_lower
                .rsplit('/')
                .next()
                .map(|f| f.rsplit_once('.').map_or(f, |(s, _)| s))
                .unwrap_or("");
            if path_lower == lower || path_lower.ends_with(&format!("/{lower}")) || stem == lower {
                hit.push(i);
            }
        }
        if let Some(definers) = index.name_files.get(token) {
            hit.extend(definers);
        }
    }
    hit.sort_unstable();
    hit.dedup();
    hit
}

/// Personalized PageRank over the file graph: rank flows from a file to the
/// files it relies on, and restarts at the focus files.
fn rank(index: &Index, focus: &[usize]) -> Vec<f64> {
    let n = index.files.len();
    if n == 0 {
        return Vec::new();
    }
    let mut restart = vec![1.0 / n as f64; n];
    for &f in focus {
        restart[f] += FOCUS_BOOST / n as f64;
    }
    let total: f64 = restart.iter().sum();
    restart.iter_mut().for_each(|r| *r /= total);
    let mut out_weight = vec![0.0; n];
    for &(a, _, w) in &index.edges {
        out_weight[a] += w;
    }
    let mut rank = restart.clone();
    for _ in 0..ITERATIONS {
        let mut next: Vec<f64> = restart.iter().map(|r| (1.0 - DAMPING) * r).collect();
        let mut dangling = 0.0;
        for (i, w) in out_weight.iter().enumerate() {
            if *w == 0.0 {
                dangling += rank[i];
            }
        }
        for &(a, b, w) in &index.edges {
            next[b] += DAMPING * rank[a] * w / out_weight[a];
        }
        for (i, r) in restart.iter().enumerate() {
            next[i] += DAMPING * dangling * r;
        }
        rank = next;
    }
    rank
}

// -------------------------------------------------------------- rendering --

fn render(index: &Index, rank: &[f64], request: &MapRequest<'_>, scanned: usize) -> RepoMap {
    let under = request
        .under
        .map(|u| u.trim_matches('/').to_string())
        .filter(|u| !u.is_empty() && u != ".");
    let mut order: Vec<usize> = (0..index.files.len())
        .filter(|&i| !index.defs[i].is_empty())
        .filter(|&i| {
            under.as_ref().is_none_or(|u| {
                index.files[i] == *u || index.files[i].starts_with(&format!("{u}/"))
            })
        })
        .collect();
    // Test code explains little about how a project is built; it comes after the rest.
    let weight = |i: usize| {
        rank[i]
            * if is_test_path(&index.files[i]) {
                0.25
            } else {
                1.0
            }
    };
    order.sort_by(|&a, &b| {
        weight(b)
            .total_cmp(&weight(a))
            .then(index.files[a].cmp(&index.files[b]))
    });
    let total = order.len();
    let mut text = String::new();
    let mut shown = 0;
    let mut next_offset = None;
    for (position, &i) in order.iter().enumerate().skip(request.offset) {
        let mut symbols: Vec<&Def> = index.defs[i].iter().collect();
        // The symbols the rest of the code mentions most come first, then file order.
        symbols.sort_by(|a, b| {
            // Types and classes outline a file better than its functions do.
            let types = [
                "struct",
                "enum",
                "trait",
                "class",
                "interface",
                "type",
                "impl",
                "record",
                "module",
                "object",
                "protocol",
                "union",
            ];
            // A distinctive, public, type-like name that other files mention says
            // the most about a file; short names like `path` or `get` say little.
            let score = |d: &Def| {
                let specific = d.name.chars().count().min(14) as f64 / 14.0;
                f64::from(*index.mentions.get(&d.name).unwrap_or(&0) + 1).ln() * specific
                    + if types.contains(&d.kind.as_str()) {
                        0.8
                    } else {
                        0.0
                    }
                    + if d.public { 0.5 } else { 0.0 }
            };
            score(b).total_cmp(&score(a)).then(a.line.cmp(&b.line))
        });
        let extra = symbols.len().saturating_sub(SYMBOLS_PER_FILE);
        symbols.truncate(SYMBOLS_PER_FILE);
        symbols.sort_by_key(|d| d.line);
        let mut block = format!("{}\n", index.files[i]);
        for d in &symbols {
            block.push_str(&format!("  {}: {}\n", d.line, d.signature));
        }
        if extra > 0 {
            block.push_str(&format!("  (+{extra} more definitions)\n"));
        }
        if !text.is_empty() && text.len() + block.len() > request.max_chars {
            next_offset = Some(position);
            break;
        }
        text.push_str(&block);
        shown += 1;
    }
    RepoMap {
        text,
        files_shown: shown,
        files_with_definitions: total,
        files_scanned: scanned,
        scan_complete: index.complete,
        next_offset,
    }
}

/// Builds the map of the project. Cheap to call again: the scan is cached until
/// a source file changes, and only the ranking is recomputed.
pub fn build(workspace: &Workspace, request: &MapRequest<'_>, limits: &Limits) -> Result<RepoMap> {
    // A directory scopes the scan itself, so on a huge project it is mapped in full
    // instead of being filtered out of a scan that ran out of time elsewhere.
    let only = request
        .under
        .map(|u| u.trim_matches('/'))
        .filter(|u| !u.is_empty() && *u != ".");
    let (index, scanned) = cached_index(workspace, limits, only)?;
    let focus = focus_files(&index, request.focus);
    let ranks = rank(&index, &focus);
    Ok(render(&index, &ranks, request, scanned))
}

impl RepoMap {
    /// One sentence on how much of the project the map covers.
    pub fn coverage(&self, offset: usize) -> String {
        let mut note = format!(
            "{} of {} files with definitions are shown (from rank {}); {} source files were scanned",
            self.files_shown,
            self.files_with_definitions,
            offset + 1,
            self.files_scanned
        );
        if !self.scan_complete {
            note.push_str(", but the scan stopped at its limit so more source files exist that are not mapped");
        }
        match self.next_offset {
            Some(next) => note.push_str(&format!(
                ". More ranked files follow: ask again with offset={next}."
            )),
            None => note.push('.'),
        }
        note
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn write(root: &Path, path: &str, text: &str) {
        let full = root.join(path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        fs::write(full, text).unwrap();
    }

    fn request<'a>(focus: &'a str, max_chars: usize) -> MapRequest<'a> {
        MapRequest {
            focus,
            under: None,
            max_chars,
            offset: 0,
        }
    }

    fn project() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        // `core` is used by every other file; `leaf` by none.
        write(dir.path(), "src/core.rs", "pub struct Engine {\n    id: u32,\n}\n\npub fn start_engine_loop(e: &Engine) {}\n\nimpl Engine {\n    pub fn tick(&self) {}\n}\n");
        write(dir.path(), "src/leaf.rs", "pub fn lonely_helper() {}\n");
        write(
            dir.path(),
            "src/a.rs",
            "use crate::core::Engine;\nfn run_a(e: &Engine) { start_engine_loop(e); }\n",
        );
        write(
            dir.path(),
            "src/b.rs",
            "fn run_b(e: &Engine) { start_engine_loop(e); let _ = Engine::default(); }\n",
        );
        write(
            dir.path(),
            "src/c.rs",
            "fn run_c(e: &Engine) { start_engine_loop(e); }\n",
        );
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        (dir, workspace)
    }

    #[test]
    fn definitions_are_found_for_the_common_languages() {
        let names = |ext: &str, text: &str| -> Vec<String> {
            definitions(ext, text)
                .into_iter()
                .map(|d| format!("{}:{}", d.kind, d.name))
                .collect()
        };
        assert_eq!(names("rs", "pub async fn load(x: u8) -> u8 {\n}\nstruct Point;\nimpl<T> Show for Point<T> {}\nconst MAX_ITEMS: usize = 3;\n// fn commented() {}\n"), ["fn:load", "struct:Point", "impl:Point", "const:MAX_ITEMS"]);
        // A one-line function shows its header, not its body.
        let sig = |ext: &str, text: &str| definitions(ext, text).remove(0).signature;
        assert_eq!(
            sig(
                "rs",
                "fn version(s:&str)->bool{!s.is_empty()&&s.len()<=32}\n"
            ),
            "fn version(s:&str)->bool"
        );
        assert_eq!(
            sig("py", "def data():return POLICY.read_bytes()\n"),
            "def data()"
        );
        assert_eq!(
            sig("py", "    def load(self, path: str) -> dict: return {}\n"),
            "def load(self, path: str) -> dict"
        );
        assert_eq!(sig("rs", "pub struct Point<T> {\n"), "pub struct Point<T>");
        assert!(sig(
            "rs",
            &format!("pub fn long({}) {{}}\n", "a: u8, ".repeat(40))
        )
        .ends_with('…'));
        // Module declarations, unit tests and test helpers are not the file's shape.
        assert_eq!(names("rs", "mod chat;\npub mod inline {\n}\nfn real() {}\n#[cfg(test)]\nmod tests {\n    fn helper() {}\n}\n"), ["mod:inline", "fn:real"]);
        assert_eq!(names("py", "def test_something():\n    pass\nclass TestThing:\n    pass\ndef build():\n    pass\n"), ["def:build"]);
        assert_eq!(
            names(
                "py",
                "class Foo(Base):\n    async def bar(self):\n        pass\n# def nope():\n"
            ),
            ["class:Foo", "def:bar"]
        );
        assert_eq!(names("ts", "export default class App {}\nexport const handler = async (req: Req) => {};\ninterface Props {}\n"), ["class:App", "function:handler", "interface:Props"]);
        assert_eq!(
            names(
                "go",
                "func (s *Server) Serve() error {\ntype Config struct {\n"
            ),
            ["func:Serve", "type:Config"]
        );
        assert_eq!(names("c", "static int parse_header(char *buf, int n)\n{\nif (x) {\nreturn 1;\nstruct packet {\n"), ["function:parse_header", "struct:packet"]);
        assert_eq!(
            names(
                "java",
                "public class Order {\n    public static double total(List<Item> items) {\n"
            ),
            ["class:Order", "method:total"]
        );
        assert!(names("rs", "let x = 5;\nprintln!(\"fn not_a_def\");\n").is_empty());
        assert_eq!(
            names("sh", "build_all() {\n  echo hi\n}\nfunction clean_up {\n"),
            ["function:build_all", "function:clean_up"]
        );
    }

    #[test]
    fn the_file_everything_relies_on_ranks_first_and_unused_code_last() {
        let (_dir, workspace) = project();
        let map = build(&workspace, &request("", 4000), &Limits::default()).unwrap();
        let order: Vec<&str> = map.text.lines().filter(|l| !l.starts_with(' ')).collect();
        assert_eq!(order[0], "src/core.rs", "{}", map.text);
        assert_eq!(*order.last().unwrap(), "src/leaf.rs", "{}", map.text);
        // Definitions carry their real line numbers and signatures.
        assert!(map.text.contains("  1: pub struct Engine"), "{}", map.text);
        assert!(
            map.text
                .contains("  5: pub fn start_engine_loop(e: &Engine)"),
            "{}",
            map.text
        );
        assert!(map.scan_complete && map.files_scanned == 5 && map.next_offset.is_none());
    }

    #[test]
    fn files_and_names_the_user_mentions_rank_higher() {
        let (_dir, workspace) = project();
        let by_file = build(
            &workspace,
            &request("what does src/leaf.rs do?", 4000),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            by_file.text.lines().next().unwrap(),
            "src/leaf.rs",
            "{}",
            by_file.text
        );
        let by_name = build(
            &workspace,
            &request("where is lonely_helper used", 4000),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            by_name.text.lines().next().unwrap(),
            "src/leaf.rs",
            "{}",
            by_name.text
        );
        let by_stem = build(
            &workspace,
            &request("explain leaf", 4000),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(by_stem.text.lines().next().unwrap(), "src/leaf.rs");
    }

    #[test]
    fn a_small_budget_pages_and_every_file_is_reachable_exactly_once() {
        let (_dir, workspace) = project();
        let mut seen = Vec::new();
        let mut offset = 0;
        loop {
            let map = build(
                &workspace,
                &MapRequest {
                    focus: "",
                    under: None,
                    max_chars: 120,
                    offset,
                },
                &Limits::default(),
            )
            .unwrap();
            assert!(map.files_shown >= 1);
            seen.extend(
                map.text
                    .lines()
                    .filter(|l| !l.starts_with(' '))
                    .map(str::to_string),
            );
            match map.next_offset {
                Some(next) => {
                    assert!(map.coverage(offset).contains(&format!("offset={next}")));
                    offset = next;
                }
                None => break,
            }
            assert!(seen.len() < 20, "paging never ended");
        }
        seen.sort();
        assert_eq!(
            seen,
            [
                "src/a.rs",
                "src/b.rs",
                "src/c.rs",
                "src/core.rs",
                "src/leaf.rs"
            ]
        );
    }

    #[test]
    fn distinctive_public_types_outrank_short_generic_names_within_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut body =
            String::from("pub struct ConsensusReview;\npub fn run_consensus_review() {}\n");
        for n in 0..10 {
            body.push_str(&format!("fn p{n}(path: &str) {{}}\n"));
        }
        body.push_str("fn path() {}\nfn get() {}\n");
        write(dir.path(), "src/notes.rs", &body);
        write(
            dir.path(),
            "src/other.rs",
            "fn x() { run_consensus_review(); let _ = ConsensusReview; path(); get(); }\n",
        );
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let map = build(&workspace, &request("", 4000), &Limits::default()).unwrap();
        let notes: Vec<&str> = map
            .text
            .lines()
            .skip_while(|l| *l != "src/notes.rs")
            .skip(1)
            .take_while(|l| l.starts_with(' '))
            .collect();
        assert!(
            notes[0].contains("ConsensusReview") && notes[1].contains("run_consensus_review"),
            "{notes:?}"
        );
        assert!(notes.len() <= SYMBOLS_PER_FILE + 1);
    }

    #[test]
    fn test_files_come_after_the_code_they_test() {
        let (dir, workspace) = project();
        write(dir.path(), "tests/engine_test.rs", "fn check_everything() { start_engine_loop(); let _ = Engine; }\nfn check_more() { start_engine_loop(); }\n");
        let map = build(&workspace, &request("", 4000), &Limits::default()).unwrap();
        let order: Vec<&str> = map.text.lines().filter(|l| !l.starts_with(' ')).collect();
        assert_eq!(*order.last().unwrap(), "tests/engine_test.rs", "{order:?}");
    }

    #[test]
    fn prose_in_a_c_comment_is_not_mistaken_for_a_function() {
        let text = "/*\n with a form of backtracking (early exit). No state needs to be saved\n*/\nstatic int real_one(int a)\n{\n}\nvoid other(int x,\n           int y);\nint wrapped(int a,\n";
        let names: Vec<String> = definitions("c", text).into_iter().map(|d| d.name).collect();
        // The prose line is gone; the declaration that continues on the next line stays.
        assert_eq!(names, ["real_one", "other", "wrapped"], "{names:?}");
    }

    #[test]
    fn a_directory_scopes_the_scan_itself_so_a_huge_project_can_be_mapped_part_by_part() {
        let dir = tempfile::tempdir().unwrap();
        for n in 0..6 {
            write(
                dir.path(),
                &format!("aaa/f{n}.rs"),
                &format!("pub fn from_aaa_{n}() {{}}\n"),
            );
        }
        write(
            dir.path(),
            "zzz/deep/target_code.rs",
            "pub fn the_code_i_want() {}\n",
        );
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let limits = Limits {
            max_files: 3,
            ..Limits::default()
        };
        // The whole-project scan spends its three files on `aaa`.
        let whole = build(&workspace, &request("", 4000), &limits).unwrap();
        assert!(
            !whole.text.contains("the_code_i_want") && !whole.scan_complete,
            "{}",
            whole.text
        );
        // Scoping to `zzz` reaches the code, and that scan is complete.
        let scoped = build(
            &workspace,
            &MapRequest {
                focus: "",
                under: Some("zzz"),
                max_chars: 4000,
                offset: 0,
            },
            &limits,
        )
        .unwrap();
        assert!(
            scoped.text.contains("zzz/deep/target_code.rs") && scoped.scan_complete,
            "{}",
            scoped.text
        );
        assert!(build(
            &workspace,
            &MapRequest {
                focus: "",
                under: Some("../x"),
                max_chars: 4000,
                offset: 0
            },
            &limits
        )
        .is_err());
    }

    #[test]
    fn vendored_dependency_code_is_never_entered() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/main.rs", "pub fn real_entry_point() {}\n");
        write(
            dir.path(),
            "registry/src/index.crates.io-1/dep-1.0/lib.rs",
            "pub fn dependency_internal() {}\n",
        );
        write(
            dir.path(),
            "third_party/lib/x.py",
            "def vendored():\n    pass\n",
        );
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let map = build(&workspace, &request("", 4000), &Limits::default()).unwrap();
        assert!(
            map.text.contains("src/main.rs")
                && !map.text.contains("dependency_internal")
                && !map.text.contains("vendored"),
            "{}",
            map.text
        );
        assert_eq!(map.files_scanned, 1);
    }

    #[test]
    fn a_scan_out_of_time_is_partial_and_says_so() {
        let (_dir, workspace) = project();
        let limits = Limits {
            max_seconds: 0,
            ..Limits::default()
        };
        let map = build(&workspace, &request("", 4000), &limits).unwrap();
        assert!(!map.scan_complete, "{}", map.coverage(0));
        assert!(map.coverage(0).contains("scan stopped at its limit"));
        // A cut-off scan is reused briefly, not repeated for every call.
        let again = build(&workspace, &request("", 4000), &limits).unwrap();
        assert_eq!(again.text, map.text);
    }

    #[test]
    fn a_directory_filter_lists_only_that_part() {
        let (dir, workspace) = project();
        write(dir.path(), "tools/script.py", "def helper():\n    pass\n");
        let map = build(
            &workspace,
            &MapRequest {
                focus: "",
                under: Some("tools"),
                max_chars: 4000,
                offset: 0,
            },
            &Limits::default(),
        )
        .unwrap();
        assert!(
            map.text.starts_with("tools/script.py") && !map.text.contains("src/"),
            "{}",
            map.text
        );
        assert_eq!(map.files_with_definitions, 1);
    }

    #[test]
    fn a_scan_that_hits_its_limit_says_the_map_is_partial() {
        let (_dir, workspace) = project();
        let limits = Limits {
            max_files: 2,
            ..Limits::default()
        };
        let map = build(&workspace, &request("", 4000), &limits).unwrap();
        assert!(!map.scan_complete && map.files_scanned == 2);
        assert!(
            map.coverage(0).contains("scan stopped at its limit"),
            "{}",
            map.coverage(0)
        );
    }

    #[test]
    fn generic_names_defined_in_many_files_do_not_link_files() {
        let dir = tempfile::tempdir().unwrap();
        for n in 0..10 {
            write(dir.path(), &format!("m{n}.rs"), "pub fn new() -> Self {}\n");
        }
        write(dir.path(), "user.rs", "fn go() { new(); new(); new(); }\n");
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let (index, _) = cached_index(&workspace, &Limits::default(), None).unwrap();
        assert!(index.edges.is_empty(), "{:?}", index.edges);
    }

    #[test]
    fn the_scan_is_cached_until_a_source_file_changes() {
        let (dir, workspace) = project();
        let (first, _) = cached_index(&workspace, &Limits::default(), None).unwrap();
        let (again, _) = cached_index(&workspace, &Limits::default(), None).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write(dir.path(), "src/new.rs", "pub fn brand_new_function() {}\n");
        let (rebuilt, _) = cached_index(&workspace, &Limits::default(), None).unwrap();
        assert!(!Arc::ptr_eq(&first, &rebuilt));
        assert!(rebuilt.files.contains(&"src/new.rs".to_string()));
    }

    #[test]
    fn same_length_rename_with_identical_coarse_mtime_invalidates_map() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let path = dir.path().join("module.rs");
        let time = std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        for name in ["first", "later"] {
            fs::write(&path, format!("pub fn {name}() {{}}")).unwrap();
            fs::File::open(&path).unwrap().set_modified(time).unwrap();
            let (index, _) = cached_index(&workspace, &Limits::default(), None).unwrap();
            assert!(index.name_files.contains_key(name));
        }
    }

    #[test]
    fn binary_minified_and_ignored_files_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "good.rs", "pub fn good_one() {}\n");
        write(
            dir.path(),
            "min.js",
            &format!("function minified(){{}}{}", "x=1;".repeat(2000)),
        );
        fs::write(dir.path().join("blob.rs"), [0xff, 0xfe, 0x00, 0x80]).unwrap();
        write(dir.path(), ".gitignore", "ignored.rs\n");
        write(dir.path(), "ignored.rs", "pub fn ignored_one() {}\n");
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let map = build(&workspace, &request("", 4000), &Limits::default()).unwrap();
        assert!(
            map.text.contains("good.rs")
                && !map.text.contains("min.js")
                && !map.text.contains("blob.rs")
                && !map.text.contains("ignored.rs"),
            "{}",
            map.text
        );
    }
}
