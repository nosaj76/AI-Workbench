//! ToolEngine reading responsibilities.
use super::*;

/// Folder names the walker never enters. Reported in search results so the
/// model knows exactly what was not covered.
const EXCLUDED_DIRECTORY_NAMES: [&str; 10] = [
    "target",
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ollama-workbench",
];

/// Whether a literal-search query looks like it was written as a regular
/// expression, which the model often does and which would silently match
/// nothing as plain text.
fn looks_like_regex(query: &str) -> bool {
    let single_pipe = query
        .char_indices()
        .any(|(i, c)| c == '|' && !query[..i].ends_with('|') && !query[i + 1..].starts_with('|'));
    single_pipe
        || query.contains(".*")
        || query.contains(".+")
        || query.contains("(?")
        || ["\\b", "\\d", "\\w", "\\s"]
            .iter()
            .any(|e| query.contains(e))
}

impl ToolEngine {
    pub(super) fn list(&self, args: &Value) -> Result<Value> {
        let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
        let limit = uint(args, "limit", 60, 1, 1000);
        let offset = uint(args, "offset", 0, 0, 20_000);
        let entries = self.workspace.list(path)?;
        let total = entries.len();
        let list = entries.into_iter().skip(offset).take(limit).map(|entry| {
            json!({"name":entry.name,"path":entry.path,"bytes":entry.size,"kind":if entry.is_dir{"directory"}else{"file"}})
        }).collect::<Vec<_>>();
        let next = offset.saturating_add(list.len());
        let has_more = next < total;
        Ok(
            json!({"path":path,"entries":list,"total":total,"offset":offset,
            "next_offset":if has_more {Some(next)} else {None},"truncated":has_more}),
        )
    }

    pub(super) fn tree(&self, args: &Value) -> Result<Value> {
        let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
        let root = self.workspace.resolve(path, false)?;
        let depth = uint(args, "max_depth", 4, 1, 12);
        let limit = uint(args, "limit", 60, 1, 2000);
        let offset = uint(args, "offset", 0, 0, 20_000);
        let mut entries = Vec::new();
        let mut truncated = false;
        let mut eligible = 0usize;
        for item in self.walker(&root, Some(depth)) {
            let entry = match item {
                Ok(e) => e,
                Err(_) => continue,
            };
            if entry.path() == root || !entry.path().starts_with(&root) {
                continue;
            }
            if eligible < offset {
                eligible += 1;
                continue;
            }
            if entries.len() >= limit {
                truncated = true;
                break;
            }
            eligible += 1;
            entries.push(json!({"path":relative(self.root(),entry.path()),"kind":if entry.file_type().is_some_and(|t|t.is_dir()){"directory"}else{"file"}}));
        }
        let next = offset.saturating_add(entries.len());
        Ok(json!({"path":path,"entries":entries,"offset":offset,
            "next_offset":if truncated {Some(next)} else {None},"truncated":truncated}))
    }

    pub(super) fn repo_map(&self, args: &Value) -> Result<Value> {
        let focus = args.get("focus").and_then(Value::as_str).unwrap_or("");
        let under = args
            .get("path")
            .and_then(Value::as_str)
            .filter(|p| *p != ".");
        let max_chars = uint(args, "max_chars", 6000, 1500, 40_000);
        let offset = uint(args, "offset", 0, 0, 100_000);
        if let Some(under) = under {
            // The same confinement as every other path argument.
            self.workspace.resolve(under, false)?;
        }
        let map = crate::repomap::build(
            &self.workspace,
            &crate::repomap::MapRequest {
                focus,
                under,
                max_chars,
                offset,
            },
            &crate::repomap::Limits::default(),
        )?;
        Ok(json!({
            "map": map.text,
            "files_shown": map.files_shown,
            "files_with_definitions": map.files_with_definitions,
            "files_scanned": map.files_scanned,
            "scan_complete": map.scan_complete,
            "next_offset": map.next_offset,
            "complete": map.next_offset.is_none() && map.scan_complete,
            "coverage": map.coverage(offset),
        }))
    }

    pub(super) fn read_file(&self, args: &Value) -> Result<Value> {
        let path = required(args, "path")?;
        let first = uint(args, "start_line", 1, 1, usize::MAX);
        // The whole rest of the file by default; the character budget decides
        // where a file too large for the model's context is split.
        let last = uint(args, "end_line", usize::MAX, first, usize::MAX);
        let cap = uint(args, "max_chars", 6000, 1000, 4_000_000);
        let column = uint(args, "start_column", 0, 0, usize::MAX);
        let expected = args.get("expected_sha256").and_then(Value::as_str);
        let page = self
            .workspace
            .read_line_page(&path, first, column, last, cap, expected)?;
        let requested_end = last.min(page.total_lines);
        // Complete means: every line asked for was returned and nothing was cut.
        let complete =
            page.end_line >= requested_end && !page.line_truncated && !page.source_truncated;
        let note = if complete {
            match page.next_start_line {
                Some(next) => format!("Complete for the requested range. The file has {} lines; continue with start_line={next} for the rest.", page.total_lines),
                None => format!("Complete: lines {}-{} of {} (end of file).", page.start_line, page.end_line, page.total_lines),
            }
        } else if page.source_truncated {
            "INCOMPLETE: the text extractor stopped early (size limit), so later content of this document cannot be read here. Use the Review queue or an external tool for the rest.".to_string()
        } else if page.line_truncated {
            format!("INCOMPLETE: line {} is longer than the character budget; its first part was returned. Continue with start_line={} and start_column={} to read the rest of that line.", page.end_line, page.end_line, page.next_column.unwrap_or(0))
        } else {
            format!("INCOMPLETE: returned lines {}-{} of the {} requested. Continue with start_line={}; do not treat this as the whole file.", page.start_line, page.end_line, requested_end, page.next_start_line.unwrap_or(page.end_line + 1))
        };
        Ok(json!({
            "path":path,"kind":page.kind,
            "requested_start_line":first,"requested_end_line":last,
            "returned_start_line":page.start_line,"returned_end_line":page.end_line,
            "total_lines":page.total_lines,"next_start_line":page.next_start_line,
            "next_column":page.next_column,"source_sha256":page.source_sha256,
            "complete":complete,"truncated":!complete,"coverage":note,
            "content":page.text,
            "line_basis":"Extracted text; source-line numbers for plain text files"
        }))
    }

    pub(super) fn search(&self, args: &Value) -> Result<Value> {
        /// Text files read per call. A bigger workspace is continued with
        /// `resume_after`; nothing is silently left unsearched.
        const FILES_PER_CALL: usize = 10_000;
        const MATCHES_PER_FILE: usize = 200;
        const SHOWN_EXAMPLES: usize = 25;
        let query = required(args, "query")?;
        if query.len() > 4096 {
            bail!("Query exceeds 4096 bytes");
        }
        let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
        let root = self.workspace.resolve(path, false)?;
        let sensitive = args
            .get("case_sensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let needle = if sensitive {
            query.clone()
        } else {
            query.to_lowercase()
        };
        let use_regex = args.get("regex").and_then(Value::as_bool).unwrap_or(false);
        let pattern = if use_regex {
            Some(
                regex::RegexBuilder::new(&query)
                    .case_insensitive(!sensitive)
                    .size_limit(1 << 20)
                    .build()
                    .map_err(|e| anyhow::anyhow!("Invalid regular expression: {e}"))?,
            )
        } else {
            None
        };
        let regex_like = !use_regex && looks_like_regex(&query);
        let limit = uint(args, "limit", 50, 1, 200);
        let resume_after = args
            .get("resume_after")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .map(str::to_string);
        let mut resuming = resume_after.is_some();
        let mut matches = Vec::new();
        let mut files = 0usize;
        let mut skipped_unreadable = 0usize;
        let mut needs_extractor = Vec::new();
        let mut needs_extractor_count = 0usize;
        let mut partly_read = Vec::new();
        let mut partly_read_count = 0usize;
        let mut dense_files = Vec::new();
        let mut last_path: Option<String> = None;
        let mut stopped_for: Option<&str> = None;
        for item in self.walker(&root, None) {
            let entry = match item {
                Ok(e) => e,
                Err(_) => {
                    skipped_unreadable += 1;
                    continue;
                }
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let rel = relative(self.root(), entry.path());
            if resuming {
                if resume_after.as_deref() == Some(rel.as_str()) {
                    resuming = false;
                }
                continue;
            }
            // Stop only between files, so a file's matches are never cut off.
            if matches.len() >= limit {
                stopped_for = Some("match_limit");
                break;
            }
            if files >= FILES_PER_CALL {
                stopped_for = Some("file_limit");
                break;
            }
            if self.workspace.resolve(&rel, false).is_err() {
                skipped_unreadable += 1;
                continue;
            }
            if crate::files::requires_external_extractor(entry.path()) {
                needs_extractor_count += 1;
                if needs_extractor.len() < SHOWN_EXAMPLES {
                    needs_extractor.push(rel);
                }
                continue;
            }
            let document = match self.workspace.read_text(&rel, 1024 * 1024) {
                Ok(doc) => doc,
                Err(_) => {
                    skipped_unreadable += 1;
                    continue;
                }
            };
            if document.truncated {
                partly_read_count += 1;
                if partly_read.len() < SHOWN_EXAMPLES {
                    partly_read.push(rel.clone());
                }
            }
            files += 1;
            last_path = Some(rel.clone());
            let mut in_file = 0usize;
            for (index, line) in document.text.lines().enumerate() {
                let hit = match &pattern {
                    Some(re) => re.is_match(line),
                    None if sensitive => line.contains(&needle),
                    None => line.to_lowercase().contains(&needle),
                };
                if !hit {
                    continue;
                }
                if in_file >= MATCHES_PER_FILE {
                    dense_files.push(rel.clone());
                    break;
                }
                in_file += 1;
                let chars = line.chars().count();
                let mut found = json!({"path":rel,"line":index+1,"text":line.chars().take(600).collect::<String>()});
                if chars > 600 {
                    found["text_truncated"] = json!(true);
                    found["text_chars"] = json!(chars);
                }
                matches.push(found);
            }
        }
        if resume_after.is_some() && resuming {
            bail!("resume_after path was not found while walking; pass the exact `resume_after` value from the previous result");
        }
        let complete = stopped_for.is_none();
        let coverage = if complete {
            let mut note = format!("COMPLETE: every searchable text file under `{path}` was searched ({files} files read). ");
            note.push_str(if matches.is_empty() {
                "Zero matches means the text does not appear in them."
            } else {
                "All matches are listed."
            });
            if needs_extractor_count > 0 {
                note.push_str(&format!(" {needs_extractor_count} file(s) (PDF/scanned/image types) were NOT searched because they need an external extractor; read them with read_file."));
            }
            if partly_read_count > 0 {
                note.push_str(&format!(" {partly_read_count} large file(s) were only searched in part (see partly_read_files)."));
            }
            note
        } else {
            let reason = if stopped_for == Some("match_limit") {
                "the match limit was reached"
            } else {
                "the per-call file limit was reached"
            };
            format!("INCOMPLETE: stopped because {reason}; searched {files} files so far and NOT the rest. Zero or few matches here does NOT mean the text is absent elsewhere. Call search_files again with the same query and resume_after=\"{}\", or narrow `path`.", last_path.clone().unwrap_or_default())
        };
        let warning = regex_like.then(|| {
            "The query contains regular-expression syntax (such as | alternation) but was matched as LITERAL text, so these results do not reflect the pattern you meant. Repeat with \"regex\": true, or run one search per alternative.".to_string()
        });
        let coverage = match &warning {
            Some(w) if matches.is_empty() => format!("NOT CONCLUSIVE: {w} {coverage}"),
            Some(w) => format!("{w} {coverage}"),
            None => coverage,
        };
        Ok(json!({
            "query":query,"path":path,"matches":matches,
            "query_mode":if use_regex {"regex"} else {"literal"},"warning":warning,
            "complete":complete,"truncated":!complete,
            "resume_after":if complete {None} else {last_path},
            "coverage":coverage,
            "files_searched":files,"files_skipped":skipped_unreadable,
            "skipped_needs_external_extractor":needs_extractor_count,
            "needs_external_extractor_examples":needs_extractor,
            "partial_files":partly_read_count,"partly_read_files":partly_read,
            "files_with_more_matches_than_shown":dense_files,
            "excluded_directory_names":EXCLUDED_DIRECTORY_NAMES
        }))
    }

    pub(super) fn walker(&self, scope: &Path, depth: Option<usize>) -> ignore::Walk {
        let scope = scope.to_path_buf();
        let workspace = self.workspace.clone();
        let offset = scope
            .strip_prefix(self.root())
            .map(|p| p.components().count())
            .unwrap_or(0);
        let mut builder = ignore::WalkBuilder::new(self.root());
        builder
            .hidden(false)
            .follow_links(false)
            .git_ignore(true)
            .git_exclude(true)
            .require_git(false)
            .git_global(false)
            .parents(false)
            .max_depth(depth.map(|d| d.saturating_add(offset)))
            .sort_by_file_name(|a, b| a.cmp(b))
            .filter_entry(move |entry| {
                if entry.depth() == 0 {
                    return true;
                }
                if !(entry.path().starts_with(&scope) || scope.starts_with(entry.path())) {
                    return false;
                }
                let name = entry.file_name().to_string_lossy();
                if EXCLUDED_DIRECTORY_NAMES.contains(&name.as_ref()) {
                    return false;
                }
                let rel = relative(workspace.root(), entry.path());
                workspace.resolve(&rel, false).is_ok()
            });
        builder.build()
    }

    /// For image-capable UI/API callers. Not advertised as a text tool because
    /// base64 must be placed in Ollama message.images, not in textual tool output.
    pub fn read_image(&self, path: &str) -> Result<Value> {
        let resolved = self.workspace.resolve(path, false)?;
        let bytes = read_limited_beneath(self.root(), &resolved, 8 * 1024 * 1024)?;
        let format = image::guess_format(&bytes).context("Unsupported image")?;
        let mime = match format {
            image::ImageFormat::Png => "image/png",
            image::ImageFormat::Jpeg => "image/jpeg",
            image::ImageFormat::Gif => "image/gif",
            image::ImageFormat::WebP => "image/webp",
            _ => bail!("Use a PNG, JPEG, GIF or WebP image"),
        };
        Ok(
            json!({"path":path,"mime_type":mime,"base64":base64::engine::general_purpose::STANDARD.encode(bytes)}),
        )
    }
}
