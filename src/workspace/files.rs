//! Local workspace discovery and bounded document extraction. No network access.
use anyhow::{bail, Context, Result};
use ignore::WalkBuilder;
use quick_xml::{events::Event, Reader};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EXTRACT_BYTES: usize = 8 * 1024 * 1024;
const MAX_XML_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ARCHIVE_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}
const MAX_DIRECTORY_ENTRIES: usize = 200_000;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub path: String,
    pub text: String,
    pub truncated: bool,
    pub kind: String,
}
/// One page of whole lines from a file; see `Workspace::read_line_page`.
#[derive(Debug, Clone)]
pub struct LinePage {
    pub kind: String,
    pub text: String,
    pub total_lines: usize,
    pub start_line: usize,
    pub end_line: usize,
    /// First line not returned, or None when the file end was reached.
    pub next_start_line: Option<usize>,
    /// When `next_start_line` is the line that was only partly returned: the 0-based
    /// character offset inside that line where the next page must resume.
    pub next_column: Option<usize>,
    /// SHA-256 of the extracted text this page came from. Pass it back as
    /// `expected_sha256` to detect that the file changed between pages.
    pub source_sha256: String,
    pub line_truncated: bool,
    /// The extractor itself stopped early (size limit); lines beyond are unreadable here.
    pub source_truncated: bool,
}
/// What `Workspace::source_files` should look for, and when to stop.
#[derive(Clone, Copy, Debug)]
pub struct SourceQuery<'a> {
    pub extensions: &'a [&'a str],
    /// Directory entries to look at before giving up.
    pub max_visited: usize,
    pub max_files: usize,
    pub max_bytes: u64,
    /// Directory names never entered.
    pub prune: &'static [&'static str],
    pub max_depth: Option<usize>,
    /// Confine the walk to this project-relative directory.
    pub only: Option<&'a str>,
    pub deadline: Option<std::time::Instant>,
}

/// A source file found by `Workspace::source_files`.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub size: u64,
    /// Modification time in seconds since the epoch.
    pub modified: u64,
}

/// Result of `Workspace::source_files`.
#[derive(Debug, Clone)]
pub struct SourceListing {
    pub files: Vec<SourceFile>,
    /// Directory entries looked at, files or not.
    pub visited: usize,
    /// False when a limit ended the walk early.
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: PathBuf) -> Result<Self> {
        let root = root.canonicalize().context("Open workspace folder")?;
        if !root.is_dir() {
            bail!("The workspace must be a directory");
        }
        Ok(Self { root })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path without following symlinks. The root is canonical.
    /// Ignored files are excluded from discovery; explicit paths remain readable
    /// unless they are secrets, protected folders, symlinks, or oversized files.
    pub fn resolve(&self, relative: &str, allow_missing: bool) -> Result<PathBuf> {
        if relative == "~" || relative.starts_with("~/") {
            bail!("Workspace paths do not expand '~'. The connected root is {}. Use '.' for that directory or an absolute path inside it; do not repeat a shell-style cwd.",self.root.display());
        }
        // An absolute path that points inside the connected folder (for example
        // one the user pasted) is the same file as its relative form. Every
        // check below still applies; a path outside the folder stays refused.
        let inside;
        let relative = if relative.starts_with('/') {
            match Path::new(relative).strip_prefix(&self.root) {
                Ok(rest) => {
                    inside = rest.to_str().context("Paths must be valid UTF-8")?.to_string();
                    if inside.is_empty() {
                        "."
                    } else {
                        inside.as_str()
                    }
                }
                Err(_) => bail!(
                    "That absolute path is outside the connected project folder ({}). Use a path relative to it, or connect that folder under Local files.",
                    self.root.display()
                ),
            }
        } else {
            relative
        };
        if relative.contains('\0')
            || relative.contains('\\')
            || relative.split('/').next().is_some_and(|p| p.contains(':'))
        {
            bail!("Use a workspace-relative path with forward slashes");
        }
        let mut result = self.root.clone();
        for component in Path::new(relative).components() {
            match component {
                Component::CurDir => {}
                Component::Normal(name) => {
                    let name = name.to_str().context("Paths must be valid UTF-8")?;
                    if denied_name(name) {
                        bail!("Access to protected or secret path component is denied: {name}");
                    }
                    result.push(name);
                    match fs::symlink_metadata(&result) {
                        Ok(meta) if meta.file_type().is_symlink() => {
                            bail!("Symlinks are excluded from workspace access: {relative}")
                        }
                        Ok(_) => {}
                        Err(e) if allow_missing && e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => {
                            return Err(e)
                                .with_context(|| format!("Resolve workspace path: {relative}"))
                        }
                    }
                }
                _ => bail!("Absolute paths and parent traversal are not permitted"),
            }
        }
        if !result.starts_with(&self.root) {
            bail!("Path is outside the workspace");
        }
        Ok(result)
    }

    fn walker(&self) -> WalkBuilder {
        self.walker_pruning(&[], None)
    }

    /// The project walker, additionally never entering directories with these
    /// names (pruned, so their contents are not even visited).
    fn walker_pruning(&self, prune: &'static [&'static str], only: Option<PathBuf>) -> WalkBuilder {
        let mut builder = WalkBuilder::new(&self.root);
        builder
            .hidden(false)
            .follow_links(false)
            .parents(false)
            .git_global(false)
            .git_ignore(true)
            .git_exclude(true)
            .require_git(false)
            .sort_by_file_name(|a, b| a.cmp(b));
        let root = self.root.clone();
        builder.filter_entry(move |entry| {
            if entry.depth() == 0 {
                return true;
            }
            if entry.file_type().is_some_and(|ft| ft.is_symlink()) {
                return false;
            }
            let Ok(relative) = entry.path().strip_prefix(&root) else {
                return false;
            };
            if relative
                .components()
                .any(|c| denied_name(&c.as_os_str().to_string_lossy()))
            {
                return false;
            }
            // Confined to one subtree: only it, and the directories leading to it.
            if let Some(only) = &only {
                if !(entry.path().starts_with(only) || only.starts_with(entry.path())) {
                    return false;
                }
            }
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                return !prune.contains(&entry.file_name().to_string_lossy().as_ref());
            }
            // Keep size filtering here: ignore's serial walker can return from
            // max_filesize before evaluating filter_entry for ordinary files.
            // Protected names and binary exclusions must always be evaluated.
            !binary_extension(entry.path())
                && entry
                    .metadata()
                    .map_or(true, |m| m.is_file() && m.len() <= MAX_FILE_BYTES)
        });
        builder
    }

    pub fn tree(&self, limit: usize) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        for entry in self.walker().build() {
            let entry = entry.context("Read workspace tree")?;
            if entry.depth() == 0 {
                continue;
            }
            if entries.len() >= limit.min(100_000) {
                break;
            }
            let meta = entry.metadata()?;
            if !meta.is_dir() && !meta.is_file() {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/");
            if self.resolve(&relative, false).is_err()
                || (meta.is_file() && meta.len() > MAX_FILE_BYTES)
            {
                continue;
            }
            entries.push(FileEntry {
                path: relative,
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: meta.is_dir(),
                size: meta.len(),
            });
        }
        Ok(entries)
    }

    /// Source files for code analysis, under the same ignore, secret and size
    /// rules as the tree. `complete` is false when a limit stopped the walk, so
    /// callers can say the listing is partial instead of implying it is whole.
    pub fn source_files(&self, query: &SourceQuery<'_>) -> Result<SourceListing> {
        let SourceQuery {
            extensions,
            max_visited,
            max_files,
            max_bytes,
            prune,
            max_depth,
            only,
            deadline,
        } = *query;
        let mut listing = SourceListing {
            files: Vec::new(),
            visited: 0,
            complete: true,
        };
        let only = match only {
            Some(relative) => Some(self.resolve(relative, false)?),
            None => None,
        };
        let mut walker = self.walker_pruning(prune, only);
        walker.max_depth(max_depth);
        for entry in walker.build() {
            let Ok(entry) = entry else { continue };
            if entry.depth() == 0 {
                continue;
            }
            listing.visited += 1;
            if listing.visited > max_visited
                || listing.files.len() >= max_files
                || deadline.is_some_and(|d| {
                    listing.visited.is_multiple_of(256) && std::time::Instant::now() >= d
                })
            {
                listing.complete = false;
                break;
            }
            let Some(extension) = entry.path().extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !extensions.iter().any(|e| e.eq_ignore_ascii_case(extension)) {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() || meta.len() > max_bytes.min(MAX_FILE_BYTES) {
                continue;
            }
            let Ok(relative) = entry.path().strip_prefix(&self.root) else {
                continue;
            };
            let relative = relative.to_string_lossy().replace('\\', "/");
            if self.resolve(&relative, false).is_err() {
                continue;
            }
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            listing.files.push(SourceFile {
                path: relative,
                size: meta.len(),
                modified,
            });
        }
        Ok(listing)
    }

    pub fn list(&self, relative: &str) -> Result<Vec<FileEntry>> {
        let directory = self.resolve(relative, false)?;
        if !directory.is_dir() {
            bail!("{} is a file, not a directory. Use read_file for its contents; list_files requires a directory such as \".\".",relative);
        }
        let mut entries = Vec::new();
        // Traversing from the root preserves nested .gitignore semantics, but
        // only the directory's direct children matter: never walk deeper.
        let child_depth = directory
            .strip_prefix(&self.root)
            .map(|p| p.components().count())
            .unwrap_or(0)
            + 1;
        let mut builder = self.walker();
        builder.max_depth(Some(child_depth));
        for entry in builder.build() {
            let entry = entry.context("List workspace directory")?;
            if entry.path().parent() != Some(directory.as_path()) {
                continue;
            }
            let meta = entry.metadata()?;
            if !meta.is_dir() && !meta.is_file() {
                continue;
            }
            let path = entry
                .path()
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/");
            if self.resolve(&path, false).is_err()
                || (meta.is_file() && meta.len() > MAX_FILE_BYTES)
            {
                continue;
            }
            entries.push(FileEntry {
                path,
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: meta.is_dir(),
                size: meta.len(),
            });
            if entries.len() >= MAX_DIRECTORY_ENTRIES {
                bail!("Directory has more than {MAX_DIRECTORY_ENTRIES} entries; list a subdirectory or search by name instead of listing it");
            }
        }
        entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.path.cmp(&b.path)));
        Ok(entries)
    }

    pub fn read_text(&self, relative: &str, max_chars: usize) -> Result<Document> {
        let path = self.resolve(relative, false)?;
        let meta = fs::metadata(&path)?;
        if !meta.is_file() {
            bail!("Path is not a regular file");
        }
        if meta.len() > MAX_FILE_BYTES {
            bail!("Files larger than 16 MiB are excluded");
        }
        let extension = extension(&path);
        // External extractors and the archive reader open a path themselves. Give
        // them a private copy taken through a descriptor opened beneath the root,
        // so a directory swapped after the checks cannot make them read elsewhere.
        let snapshot = if matches!(
            extension.as_str(),
            "docx"
                | "odt"
                | "pptx"
                | "xlsx"
                | "epub"
                | "pdf"
                | "png"
                | "jpg"
                | "jpeg"
                | "tif"
                | "tiff"
                | "bmp"
                | "webp"
                | "gif"
        ) {
            let relative = path.strip_prefix(&self.root).unwrap_or(&path);
            match crate::confined::open_read(&self.root, relative)? {
                Some(file) => Some(PrivateCopy::of(file, &extension)?),
                None => None,
            }
        } else {
            None
        };
        let source: &Path = snapshot
            .as_ref()
            .map_or(path.as_path(), |copy| copy.0.as_path());
        let (text, kind, extractor_truncated) = match extension.as_str() {
            "docx" | "odt" | "pptx" | "xlsx" | "epub" => (
                extract_zip(source, &extension)?,
                format!("{} (extracted lines)", extension.to_uppercase()),
                false,
            ),
            "pdf" => {
                let (s, t) =
                    run_extractor("pdftotext", &["-layout", "-enc", "UTF-8"], source, &["-"])?;
                (s, "PDF (extracted lines; requires pdftotext)".into(), t)
            }
            "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" | "gif" => {
                let (s, t) = run_extractor("tesseract", &[], source, &["stdout"])?;
                (
                    s,
                    "Image OCR (extracted lines; requires tesseract)".into(),
                    t,
                )
            }
            _ => {
                if binary_extension(&path) {
                    bail!("This binary format cannot be read as text");
                }
                let mut bytes = Vec::new();
                // Opened beneath the root through descriptors where the kernel allows,
                // so a directory swapped after the path check cannot redirect the read.
                let relative = path.strip_prefix(&self.root).unwrap_or(&path);
                let file = match crate::confined::open_read(&self.root, relative)? {
                    Some(file) => file,
                    None => File::open(&path)?,
                };
                file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)?;
                if bytes.len() as u64 > MAX_FILE_BYTES {
                    bail!("File grew beyond the 16 MiB limit");
                }
                if bytes.contains(&0) {
                    bail!("Binary or UTF-16 content is not supported; export as UTF-8 text");
                }
                let text = String::from_utf8(bytes).context("The file is not UTF-8 text")?;
                if text
                    .chars()
                    .filter(|c| c.is_control() && !matches!(*c, '\n' | '\r' | '\t' | '\u{c}'))
                    .take(8)
                    .count()
                    >= 8
                {
                    bail!("File appears to contain binary data");
                }
                (
                    text.trim_start_matches('\u{feff}').to_string(),
                    "UTF-8 text".into(),
                    false,
                )
            }
        };
        let (text, truncated) = truncate_chars(&text, max_chars.min(MAX_EXTRACT_BYTES));
        Ok(Document {
            path: path
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/"),
            text,
            truncated: truncated || extractor_truncated,
            kind,
        })
    }

    pub fn read_lines(
        &self,
        relative: &str,
        start: usize,
        end: usize,
        max_chars: usize,
    ) -> Result<Document> {
        if start == 0 || end < start {
            bail!("Line ranges are 1-based and end must be at least start");
        }
        let mut document = self.read_text(relative, MAX_EXTRACT_BYTES)?;
        let lines: Vec<_> = document.text.lines().collect();
        let text = lines
            .iter()
            .enumerate()
            .skip(start - 1)
            .take(end.saturating_sub(start) + 1)
            .map(|(i, line)| format!("{}: {}", i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        let (text, cut) = truncate_chars(&text, max_chars.min(MAX_EXTRACT_BYTES));
        document.truncated |= cut || start > 1 || end < lines.len();
        document.text = text;
        Ok(document)
    }

    /// Reads a page of whole lines. Unlike `read_lines`, the character budget
    /// never cuts a line in half (except one single line longer than the whole
    /// budget, which is flagged) and the result states exactly what was
    /// returned, how many lines exist and where to continue.
    pub fn read_line_page(
        &self,
        relative: &str,
        start: usize,
        start_column: usize,
        end: usize,
        max_chars: usize,
        expected_sha256: Option<&str>,
    ) -> Result<LinePage> {
        if start == 0 || end < start {
            bail!("Line ranges are 1-based and end must be at least start");
        }
        let document = self.read_text(relative, MAX_EXTRACT_BYTES)?;
        let source_sha256 = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(document.text.as_bytes()))
        };
        if let Some(expected) = expected_sha256 {
            if !expected.eq_ignore_ascii_case(&source_sha256) {
                bail!("The file changed since the earlier page was read (expected_sha256 does not match). Restart from start_line=1 so no content is skipped or repeated.");
            }
        }
        let lines: Vec<&str> = document.text.lines().collect();
        let total_lines = lines.len();
        if start > total_lines.max(1) {
            bail!("start_line {start} is past the end of the file ({total_lines} lines)");
        }
        let last_wanted = end.min(total_lines);
        let mut text = String::new();
        let mut used = 0usize;
        let mut returned_end = start.saturating_sub(1);
        let mut line_truncated = false;
        let mut next_column = None;
        let mut partial_line = None;
        for number in start..=last_wanted {
            let full = lines[number - 1];
            let skip = if number == start { start_column } else { 0 };
            // Resume inside a long line without ever splitting a UTF-8 character.
            let from = match full.char_indices().nth(skip) {
                Some((index, _)) => index,
                None if skip == full.chars().count() => full.len(),
                None => bail!("start_column {skip} is past the end of line {number}"),
            };
            let line = &full[from..];
            let prefix = if skip > 0 {
                format!("{number}@{skip}: ")
            } else {
                format!("{number}: ")
            };
            let cost = prefix.chars().count() + line.chars().count() + 1;
            if used + cost > max_chars && returned_end >= start {
                break;
            }
            if used + cost > max_chars {
                // One line longer than the whole budget: return as much as fits and
                // say exactly where to resume.
                let room = max_chars.saturating_sub(prefix.chars().count() + 1).max(1);
                let (piece, _) = truncate_chars(line, room);
                next_column = Some(skip + piece.chars().count());
                text.push_str(&prefix);
                text.push_str(&piece);
                text.push('\n');
                line_truncated = true;
                partial_line = Some(number);
                break;
            }
            text.push_str(&prefix);
            text.push_str(line);
            text.push('\n');
            used += cost;
            returned_end = number;
        }
        let text = text.trim_end_matches('\n').to_string();
        let (end_line, next_start_line) = match partial_line {
            Some(number) => (number, Some(number)),
            None => (
                returned_end,
                (returned_end < total_lines).then_some(returned_end + 1),
            ),
        };
        Ok(LinePage {
            kind: document.kind,
            text,
            total_lines,
            start_line: start,
            end_line,
            next_start_line,
            next_column,
            source_sha256,
            line_truncated,
            source_truncated: document.truncated,
        })
    }

    /// Literal, case-insensitive content search. Extraction failures are skipped.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        if query.trim().is_empty() {
            bail!("Enter a nonempty search query");
        }
        let needle = query.to_lowercase();
        let mut hits = Vec::new();
        for entry in self.tree(100_000)?.into_iter().filter(|e| !e.is_dir) {
            if hits.len() >= limit.min(1000) {
                break;
            }
            // Avoid starting potentially expensive OCR/PDF tools for broad search.
            if requires_external_extractor(Path::new(&entry.path)) {
                continue;
            }
            let Ok(document) = self.read_text(&entry.path, MAX_EXTRACT_BYTES) else {
                continue;
            };
            for (i, line) in document.text.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    hits.push(SearchHit {
                        path: entry.path.clone(),
                        line: i + 1,
                        text: truncate_chars(line, 1000).0,
                    });
                    if hits.len() >= limit.min(1000) {
                        return Ok(hits);
                    }
                }
            }
        }
        Ok(hits)
    }

    /// Enumerate readable candidates without an arbitrary UI tree cap.
    pub(crate) fn candidates(&self) -> impl Iterator<Item = Result<FileEntry>> + '_ {
        self.walker().build().filter_map(move |entry| {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => return Some(Err(e.into())),
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                return None;
            }
            Some((|| {
                let meta = entry.metadata()?;
                let path = entry
                    .path()
                    .strip_prefix(&self.root)?
                    .to_string_lossy()
                    .replace('\\', "/");
                self.resolve(&path, false)?;
                if meta.len() > MAX_FILE_BYTES {
                    bail!("File grew beyond the 16 MiB limit");
                }
                Ok(FileEntry {
                    path,
                    name: entry.file_name().to_string_lossy().into_owned(),
                    is_dir: false,
                    size: meta.len(),
                })
            })())
        })
    }
}

fn denied_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    matches!(
        n.as_str(),
        ".git" | ".svn" | ".hg" | "target" | "node_modules" | "__pycache__" | ".venv" | "venv"
    ) || secret_name(name)
}

/// Names of credentials and private data. File tools refuse them, and the
/// command sandbox hides them inside the project folder.
pub fn secret_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    matches!(
        n.as_str(),
        ".ssh"
            | ".aws"
            | ".gnupg"
            | ".ollama-workbench"
            | "secrets"
            | ".secrets"
            | "credentials"
            | ".git-credentials"
            | ".netrc"
            | ".npmrc"
            | ".pypirc"
            | "credentials.json"
            | "secrets.json"
            | "secrets.yaml"
            | "secrets.yml"
            | "id_rsa"
            | "id_dsa"
            | "id_ecdsa"
            | "id_ed25519"
    ) || n == ".env"
        || n.starts_with(".env.")
        || n.ends_with(".pem")
        || n.ends_with(".key")
        || n.ends_with(".p12")
        || n.ends_with(".pfx")
        || n.ends_with(".kdbx")
}
fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}
fn binary_extension(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "exe"
            | "dll"
            | "so"
            | "dylib"
            | "a"
            | "o"
            | "rlib"
            | "rmeta"
            | "class"
            | "pyc"
            | "pyo"
            | "wasm"
            | "bin"
            | "gguf"
            | "ggml"
            | "safetensors"
            | "pt"
            | "pth"
            | "onnx"
            | "zip"
            | "tar"
            | "gz"
            | "bz2"
            | "xz"
            | "7z"
            | "rar"
            | "iso"
            | "sqlite"
            | "sqlite3"
            | "db"
            | "mp3"
            | "mp4"
            | "mkv"
            | "avi"
            | "mov"
            | "flac"
            | "wav"
            | "ogg"
            | "woff"
            | "woff2"
            | "ttf"
            | "ico"
            | "icns"
    )
}
pub(crate) fn requires_external_extractor(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "pdf" | "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" | "gif"
    )
}
/// A private temporary copy of a workspace file for tools that need a path.
/// Created exclusively (mode 0600, never through a symlink) and removed on drop.
struct PrivateCopy(PathBuf);

impl PrivateCopy {
    fn of(file: File, extension: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            ".workbench-extract-{}.{extension}",
            uuid::Uuid::new_v4()
        ));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut out = options.open(&path)?;
        let copy = Self(path);
        let copied = std::io::copy(&mut file.take(MAX_FILE_BYTES + 1), &mut out)?;
        if copied > MAX_FILE_BYTES {
            bail!("File grew beyond the 16 MiB limit");
        }
        Ok(copy)
    }
}

impl Drop for PrivateCopy {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(crate) fn truncate_chars(text: &str, max: usize) -> (String, bool) {
    match text.char_indices().nth(max) {
        Some((i, _)) => (text[..i].to_string(), true),
        None => (text.to_string(), false),
    }
}

fn run_extractor(
    program: &str,
    before: &[&str],
    path: &Path,
    after: &[&str],
) -> Result<(String, bool)> {
    let mut child = Command::new(program)
        .args(before)
        .arg(path)
        .args(after)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!("Install {program} and make it available on PATH to read this format")
        })?;
    let output = child
        .stdout
        .take()
        .context("Extractor stdout unavailable")?;
    let error = child
        .stderr
        .take()
        .context("Extractor stderr unavailable")?;
    let exceeded = Arc::new(AtomicBool::new(false));
    let flag = exceeded.clone();
    let reader = thread::spawn(move || bounded_read(output, MAX_EXTRACT_BYTES, flag));
    let errors = thread::spawn(move || {
        let mut v = Vec::new();
        let _ = error.take(16 * 1024).read_to_end(&mut v);
        v
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if exceeded.load(Ordering::Relaxed) || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let bytes = reader
        .join()
        .map_err(|_| anyhow::anyhow!("Extractor reader failed"))??;
    let stderr = errors.join().unwrap_or_default();
    if status.is_none() && !exceeded.load(Ordering::Relaxed) {
        bail!("{program} exceeded the 30-second extraction limit");
    }
    if status.is_some_and(|s| !s.success()) && !exceeded.load(Ordering::Relaxed) {
        bail!("{program} failed: {}", String::from_utf8_lossy(&stderr));
    }
    Ok((
        String::from_utf8_lossy(&bytes).into_owned(),
        exceeded.load(Ordering::Relaxed),
    ))
}
fn bounded_read(mut reader: impl Read, limit: usize, exceeded: Arc<AtomicBool>) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        let remaining = limit.saturating_sub(result.len());
        result.extend_from_slice(&buffer[..n.min(remaining)]);
        if n > remaining {
            exceeded.store(true, Ordering::Relaxed);
            break;
        }
    }
    Ok(result)
}

fn zip_text(archive: &mut zip::ZipArchive<File>, name: &str, total: &mut u64) -> Result<String> {
    let entry = archive
        .by_name(name)
        .with_context(|| format!("Missing document part: {name}"))?;
    if entry.size() > MAX_XML_BYTES || total.saturating_add(entry.size()) > MAX_ARCHIVE_BYTES {
        bail!("Document expands beyond the extraction limit");
    }
    let mut data = Vec::new();
    entry.take(MAX_XML_BYTES + 1).read_to_end(&mut data)?;
    if data.len() as u64 > MAX_XML_BYTES {
        bail!("Document XML exceeds the extraction limit");
    }
    *total += data.len() as u64;
    String::from_utf8(data).context("Document XML is not UTF-8")
}
fn extract_zip(path: &Path, kind: &str) -> Result<String> {
    let mut archive = zip::ZipArchive::new(File::open(path)?).context("Open document archive")?;
    if archive.len() > 10_000 {
        bail!("Document contains too many archive entries");
    }
    let mut total = 0;
    let mut output = String::new();
    match kind {
        "docx" => output = xml_text(&zip_text(&mut archive, "word/document.xml", &mut total)?)?,
        "odt" => output = xml_text(&zip_text(&mut archive, "content.xml", &mut total)?)?,
        "pptx" => {
            let mut names: Vec<String> = archive
                .file_names()
                .filter(|n| {
                    n.starts_with("ppt/slides/slide")
                        && n.ends_with(".xml")
                        && !n[11..].contains('/')
                })
                .map(str::to_owned)
                .collect();
            names.sort_by_key(|n| {
                n.trim_start_matches("ppt/slides/slide")
                    .trim_end_matches(".xml")
                    .parse::<usize>()
                    .unwrap_or(usize::MAX)
            });
            for (i, name) in names.iter().enumerate() {
                output.push_str(&format!(
                    "\n## Slide {}\n{}\n",
                    i + 1,
                    xml_text(&zip_text(&mut archive, name, &mut total)?)?
                ));
            }
        }
        "xlsx" => {
            let shared = if archive.file_names().any(|n| n == "xl/sharedStrings.xml") {
                shared_strings(&zip_text(&mut archive, "xl/sharedStrings.xml", &mut total)?)?
            } else {
                Vec::new()
            };
            let mut names: Vec<_> = archive
                .file_names()
                .filter(|n| n.starts_with("xl/worksheets/sheet") && n.ends_with(".xml"))
                .map(str::to_owned)
                .collect();
            names.sort();
            for name in names {
                output.push_str(&format!(
                    "\n## {name}\n{}\n",
                    sheet_text(&zip_text(&mut archive, &name, &mut total)?, &shared)?
                ));
            }
        }
        "epub" => {
            let names = epub_spine(&mut archive, &mut total).unwrap_or_default();
            let mut names = if names.is_empty() {
                archive
                    .file_names()
                    .filter(|n| {
                        matches!(extension(Path::new(n)).as_str(), "html" | "xhtml" | "htm")
                    })
                    .map(str::to_owned)
                    .collect()
            } else {
                names
            };
            if names.len() > 1000 {
                bail!("EPUB contains too many chapters");
            }
            // Preserve OPF reading order. Fallback archive order is deterministic.
            names.dedup();
            for name in names {
                output.push_str(&format!(
                    "\n## {name}\n{}\n",
                    xml_text(&zip_text(&mut archive, &name, &mut total)?)?
                ));
            }
        }
        _ => bail!("Unsupported archive document"),
    }
    if output.len() > MAX_EXTRACT_BYTES {
        bail!("Extracted document exceeds the 8 MiB limit");
    }
    Ok(output.trim().to_owned())
}
fn text_event(text: quick_xml::events::BytesText<'_>) -> String {
    text.unescape()
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| String::from_utf8_lossy(text.as_ref()).replace("&nbsp;", " "))
}
fn xml_text(xml: &str) -> Result<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().check_end_names = false;
    let mut output = String::new();
    let mut skip = 0usize;
    loop {
        match reader.read_event()? {
            Event::Start(e) => {
                let name = e.local_name();
                if matches!(name.as_ref(), b"script" | b"style") {
                    skip += 1;
                }
            }
            Event::End(e) => {
                let name = e.local_name();
                if matches!(name.as_ref(), b"script" | b"style") {
                    skip = skip.saturating_sub(1);
                } else if skip == 0
                    && matches!(
                        name.as_ref(),
                        b"p" | b"h" | b"h1" | b"h2" | b"h3" | b"div" | b"li" | b"tr" | b"table-row"
                    )
                {
                    output.push('\n');
                }
            }
            Event::Empty(e) if skip == 0 => {
                let name = e.local_name();
                if matches!(name.as_ref(), b"br" | b"line-break") {
                    output.push('\n');
                } else if matches!(name.as_ref(), b"tab" | b"s") {
                    output.push(' ');
                }
            }
            Event::Text(t) if skip == 0 => output.push_str(&text_event(t)),
            Event::CData(t) if skip == 0 => output.push_str(&String::from_utf8_lossy(t.as_ref())),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(output)
}
fn shared_strings(xml: &str) -> Result<Vec<String>> {
    let mut r = Reader::from_str(xml);
    let mut strings = Vec::new();
    let mut current = String::new();
    let mut in_text = false;
    loop {
        match r.read_event()? {
            Event::Start(e) if e.local_name().as_ref() == b"si" => current.clear(),
            Event::Start(e) if e.local_name().as_ref() == b"t" => in_text = true,
            Event::End(e) if e.local_name().as_ref() == b"t" => in_text = false,
            Event::End(e) if e.local_name().as_ref() == b"si" => strings.push(current.clone()),
            Event::Text(t) if in_text => current.push_str(&text_event(t)),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(strings)
}
fn sheet_text(xml: &str, shared: &[String]) -> Result<String> {
    let mut r = Reader::from_str(xml);
    let mut out = String::new();
    let (mut reference, mut kind, mut value) = (String::new(), String::new(), String::new());
    let mut capture = false;
    loop {
        match r.read_event()? {
            Event::Start(e) if e.local_name().as_ref() == b"c" => {
                reference.clear();
                kind.clear();
                value.clear();
                for a in e.attributes().flatten() {
                    match a.key.as_ref() {
                        b"r" => reference = a.unescape_value()?.into_owned(),
                        b"t" => kind = a.unescape_value()?.into_owned(),
                        _ => {}
                    }
                }
            }
            Event::Start(e) if matches!(e.local_name().as_ref(), b"v" | b"t") => capture = true,
            Event::End(e) if matches!(e.local_name().as_ref(), b"v" | b"t") => capture = false,
            Event::Text(t) if capture => value.push_str(&text_event(t)),
            Event::End(e) if e.local_name().as_ref() == b"c" => {
                let v = if kind == "s" {
                    value
                        .parse::<usize>()
                        .ok()
                        .and_then(|i| shared.get(i))
                        .cloned()
                        .unwrap_or_else(|| "[missing shared string]".into())
                } else {
                    value.clone()
                };
                if !v.is_empty() {
                    if !out.ends_with('\n') && !out.is_empty() {
                        out.push('\t');
                    }
                    out.push_str(&format!("{reference}: {v}"));
                }
            }
            Event::End(e) if e.local_name().as_ref() == b"row" => out.push('\n'),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}
fn epub_spine(archive: &mut zip::ZipArchive<File>, total: &mut u64) -> Result<Vec<String>> {
    let container = zip_text(archive, "META-INF/container.xml", total)?;
    let mut r = Reader::from_str(&container);
    let mut opf = None;
    loop {
        match r.read_event()? {
            Event::Empty(e) | Event::Start(e) if e.local_name().as_ref() == b"rootfile" => {
                for a in e.attributes().flatten() {
                    if a.key.as_ref() == b"full-path" {
                        opf = Some(a.unescape_value()?.into_owned());
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    let opf = opf.context("Missing EPUB package")?;
    let package = zip_text(archive, &opf, total)?;
    let base = Path::new(&opf).parent().unwrap_or(Path::new(""));
    let mut r = Reader::from_str(&package);
    let mut manifest = HashMap::new();
    let mut spine = Vec::new();
    loop {
        match r.read_event()? {
            Event::Empty(e) | Event::Start(e) => {
                let mut attrs = HashMap::new();
                for a in e.attributes().flatten() {
                    attrs.insert(
                        String::from_utf8_lossy(a.key.as_ref()).into_owned(),
                        a.unescape_value()?.into_owned(),
                    );
                }
                match e.local_name().as_ref() {
                    b"item" => {
                        if let (Some(id), Some(href)) = (attrs.get("id"), attrs.get("href")) {
                            manifest.insert(id.clone(), href.clone());
                        }
                    }
                    b"itemref" => {
                        if let Some(id) = attrs.get("idref") {
                            spine.push(id.clone());
                        }
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(spine
        .into_iter()
        .filter_map(|id| {
            manifest
                .get(&id)
                .map(|href| base.join(href).to_string_lossy().replace('\\', "/"))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn documents_are_extracted_from_a_private_copy_that_is_removed_afterwards() {
        let dir = tempfile::tempdir().unwrap();
        let file = File::create(dir.path().join("note.docx")).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        std::io::Write::write_all(
            &mut zip,
            br#"<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Hello from docx</w:t></w:r></w:p></w:body></w:document>"#,
        )
        .unwrap();
        zip.finish().unwrap();
        let original = fs::read(dir.path().join("note.docx")).unwrap();
        let w = Workspace::new(dir.path().to_path_buf()).unwrap();
        let document = w.read_text("note.docx", 10_000).unwrap();
        assert!(
            document.text.contains("Hello from docx"),
            "{}",
            document.text
        );
        // Other tests may be extracting their own files at the same time, so
        // look for a leftover copy of this document by its content.
        let leftover = fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(".workbench-extract-")
            })
            .any(|e| fs::read(e.path()).is_ok_and(|bytes| bytes == original));
        assert!(!leftover, "the private copy must be removed");
    }

    #[test]
    fn paths_and_secrets_are_confined() {
        let t = tempfile::tempdir().unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        assert!(w.resolve("../outside", true).is_err());
        assert!(w.resolve("/etc/passwd", false).is_err());
        assert!(w.resolve("C:\\secret", true).is_err());
        assert!(w.resolve(".env", true).is_err());
        assert!(w.resolve("a/id_rsa", true).is_err());
        assert_eq!(w.resolve("a/b.rs", true).unwrap(), t.path().join("a/b.rs"));
    }
    #[test]
    fn unicode_truncation_and_lines() {
        let t = tempfile::tempdir().unwrap();
        fs::write(t.path().join("a.txt"), "αβγ\nsecond\nthird").unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        let d = w.read_text("a.txt", 2).unwrap();
        assert_eq!(d.text, "αβ");
        assert!(d.truncated);
        let d = w.read_lines("a.txt", 2, 2, 100).unwrap();
        assert_eq!(d.text, "2: second");
        assert!(d.truncated);
    }
    #[test]
    fn ignore_and_search() {
        let t = tempfile::tempdir().unwrap();
        fs::write(t.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(t.path().join("ignored.txt"), "needle").unwrap();
        fs::write(t.path().join("visible.txt"), "a\nNeedle here").unwrap();
        fs::write(t.path().join(".env"), "needle").unwrap();
        fs::create_dir(t.path().join("node_modules")).unwrap();
        fs::write(t.path().join("node_modules/a.js"), "needle").unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        let hits = w.search("needle", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "visible.txt");
        assert_eq!(hits[0].line, 2);
    }
    #[test]
    fn discovery_never_returns_secret_binary_or_oversized_file_names() {
        let t = tempfile::tempdir().unwrap();
        for name in [
            ".env",
            ".env.production",
            "id_rsa",
            "private.pem",
            "model.gguf",
            "visible.txt",
        ] {
            fs::write(t.path().join(name), "small file").unwrap();
        }
        let oversized = File::create(t.path().join("huge.txt")).unwrap();
        oversized.set_len(MAX_FILE_BYTES + 1).unwrap();
        fs::create_dir(t.path().join("child")).unwrap();
        fs::write(t.path().join("child/.env"), "secret").unwrap();
        fs::write(t.path().join("child/visible.txt"), "visible").unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        let listed: Vec<_> = w.list(".").unwrap().into_iter().map(|e| e.path).collect();
        assert_eq!(listed, vec!["child", "visible.txt"]);
        let nested: Vec<_> = w
            .list("child")
            .unwrap()
            .into_iter()
            .map(|e| e.path)
            .collect();
        assert_eq!(nested, vec!["child/visible.txt"]);
        let tree: Vec<_> = w.tree(100).unwrap().into_iter().map(|e| e.path).collect();
        assert_eq!(tree, vec!["child", "child/visible.txt", "visible.txt"]);
        let candidates: Vec<_> = w.candidates().map(|e| e.unwrap().path).collect();
        assert_eq!(candidates, vec!["child/visible.txt", "visible.txt"]);
    }
    #[cfg(unix)]
    #[test]
    fn symlinks_are_denied() {
        use std::os::unix::fs::symlink;
        let t = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("s.txt"), "secret").unwrap();
        symlink(outside.path(), t.path().join("link")).unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        assert!(w.resolve("link/s.txt", false).is_err());
        assert!(w.tree(100).unwrap().is_empty());
    }
    #[test]
    fn office_xml_extraction() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("sample.docx");
        let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        use std::io::Write;
        zip.write_all(br#"<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Hello &amp; world</w:t></w:r></w:p></w:body></w:document>"#).unwrap();
        zip.finish().unwrap();
        let w = Workspace::new(t.path().to_path_buf()).unwrap();
        assert_eq!(
            w.read_text("sample.docx", 100).unwrap().text,
            "Hello & world"
        );
    }
}
