//! Export complete fenced code blocks only after a separate, explicit save action.
//! This never writes to the selected workspace through an agent tool or executes code.
use anyhow::{bail, Context, Result};
use std::{collections::HashSet, fs::OpenOptions, io::Write, path::Path};

pub(super) const MAX_CODE_BLOCKS: usize = 64;

/// Never let an export dialog outlive the answer or conversation that produced it.
#[derive(Clone)]
pub(super) struct CodeExportSelection {
    pub session_id: String,
    pub entry_id: String,
    pub blocks: Vec<CodeArtifact>,
    pub selected: usize,
}
const MAX_BLOCK_BYTES: usize = 1024 * 1024;
const MAX_PACKAGE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CodeArtifact {
    pub name: String,
    pub language: String,
    pub text: String,
}

fn parse_fence(line: &str) -> Option<(char, usize, &str)> {
    let line = line.trim_end_matches(['\r', '\n']);
    let indent = line.bytes().take_while(|b| *b == b' ').count();
    if indent > 3 {
        return None;
    }
    let tail = &line[indent..];
    let marker = tail.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let count = tail.chars().take_while(|c| *c == marker).count();
    (count >= 3).then_some((marker, count, &tail[count..]))
}

fn extension(language: &str) -> &'static str {
    match language.to_ascii_lowercase().as_str() {
        "rust" | "rs" => "rs",
        "python" | "py" => "py",
        "bash" | "sh" | "shell" => "sh",
        "javascript" | "js" => "js",
        "typescript" | "ts" => "ts",
        "html" => "html",
        "css" => "css",
        "json" => "json",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "markdown" | "md" => "md",
        "c" => "c",
        "cpp" | "c++" | "cxx" => "cpp",
        "java" => "java",
        "go" => "go",
        "sql" => "sql",
        "xml" => "xml",
        "text" | "txt" => "txt",
        _ => "txt",
    }
}

fn safe_name(info: &str, ordinal: usize, used: &mut HashSet<String>) -> (String, String) {
    let mut parts = info.split_whitespace();
    let first = parts.next().unwrap_or("");
    let (language, filename) = if first.contains('=') || first.contains('/') || first.contains('\\')
    {
        ("text", Some(first))
    } else {
        (first, parts.next())
    };
    let raw = filename
        .and_then(|part| part.split_once('=').map(|(_, value)| value).or(Some(part)))
        .map(|value| value.trim_matches(['\'', '"']))
        .unwrap_or("");
    // Flatten every path and strip undesirable characters. Even a malicious model
    // filename is never used as a directory or an absolute destination.
    let basename = raw
        .replace('\\', "/")
        .split('/')
        .next_back()
        .unwrap_or("")
        .to_string();
    let safe: String = basename
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        .take(100)
        .collect();
    let stem = if safe.trim_matches('.').is_empty() || safe.starts_with('.') {
        format!("snippet_{ordinal:03}.{}", extension(language))
    } else {
        safe
    };
    let mut unique = stem.clone();
    let mut copy = 2;
    while !used.insert(unique.to_ascii_lowercase()) {
        let (base, ext) = stem.rsplit_once('.').unwrap_or((&stem, ""));
        unique = if ext.is_empty() {
            format!("{base}_{copy}")
        } else {
            format!("{base}_{copy}.{ext}")
        };
        copy += 1;
    }
    (language.to_string(), unique)
}

/// Extract complete Markdown code fences (including tilde fences), in display order.
/// Incomplete streaming fences are deliberately not exportable.
pub(super) fn extract_code_blocks(markdown: &str) -> Result<Vec<CodeArtifact>> {
    if markdown.len() > 16 * 1024 * 1024 {
        bail!("Answer is too large to scan for code");
    }
    let mut used = HashSet::new();
    let mut result = Vec::new();
    let mut active: Option<(char, usize, String, String, String)> = None;
    let mut total = 0_usize;
    for line in markdown.split_inclusive('\n') {
        if let Some((marker, count, language, name, body)) = active.as_mut() {
            if let Some((end_marker, end_count, suffix)) = parse_fence(line) {
                if end_marker == *marker && end_count >= *count && suffix.trim().is_empty() {
                    if !body.is_empty() {
                        result.push(CodeArtifact {
                            name: name.clone(),
                            language: language.clone(),
                            text: std::mem::take(body),
                        });
                    }
                    active = None;
                    continue;
                }
            }
            if body.len().saturating_add(line.len()) > MAX_BLOCK_BYTES {
                bail!("Code block exceeds the 1 MiB export limit");
            }
            total = total.saturating_add(line.len());
            if total > MAX_PACKAGE_BYTES {
                bail!("Code blocks exceed the 8 MiB export limit");
            }
            body.push_str(line);
        } else if let Some((marker, count, suffix)) = parse_fence(line) {
            if result.len() >= MAX_CODE_BLOCKS {
                bail!("Answer contains more than 64 code blocks");
            }
            let (language, name) = safe_name(suffix.trim(), result.len() + 1, &mut used);
            active = Some((marker, count, language, name, String::new()));
        }
    }
    // Intentionally ignore unfinished streaming output.
    Ok(result)
}

/// Single extracted file; an existing destination is never overwritten.
pub(super) fn save_single(path: &Path, block: &CodeArtifact) -> Result<()> {
    if block.text.len() > MAX_BLOCK_BYTES {
        bail!("Code block exceeds the 1 MiB limit");
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "Cannot create {} (existing files are not overwritten)",
                path.display()
            )
        })?;
    if let Err(error) = file.write_all(block.text.as_bytes()) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}

fn valid_archive_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name.len() <= 100
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Multi-file TAR: flat names under snippets/, no symlinks or executable entries.
pub(super) fn save_archive(path: &Path, blocks: &[CodeArtifact]) -> Result<()> {
    if blocks.is_empty() || blocks.len() > MAX_CODE_BLOCKS {
        bail!("Choose 1 to 64 completed code blocks");
    }
    let mut names = HashSet::new();
    for block in blocks {
        if !valid_archive_name(&block.name) || !names.insert(block.name.to_ascii_lowercase()) {
            bail!("Invalid or duplicate code artifact name");
        }
        if block.text.len() > MAX_BLOCK_BYTES {
            bail!("Code block exceeds the 1 MiB limit");
        }
    }
    if blocks
        .iter()
        .try_fold(0_usize, |n, b| n.checked_add(b.text.len()))
        .filter(|n| *n <= MAX_PACKAGE_BYTES)
        .is_none()
    {
        bail!("Code package exceeds 8 MiB");
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "Cannot create {} (existing files are not overwritten)",
                path.display()
            )
        })?;
    let written = (|| -> Result<()> {
        let mut archive = tar::Builder::new(file);
        for block in blocks {
            let mut header = tar::Header::new_gnu();
            header.set_path(format!("snippets/{}", block.name))?;
            header.set_mode(0o644);
            header.set_size(block.text.len() as u64);
            header.set_cksum();
            archive.append(&header, block.text.as_bytes())?;
        }
        archive.finish()?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(path);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exports_complete_fences_and_ignores_incomplete_streaming_code() {
        let code = extract_code_blocks("Hi\n```rust src/main.rs\nfn main() {}\n```\n~~~python app.py\nprint(1)\n~~~\n```js\nunfinished").unwrap();
        assert_eq!(code.len(), 2);
        assert_eq!(code[0].name, "main.rs");
        assert_eq!(code[0].text, "fn main() {}\n");
        assert_eq!(code[1].name, "app.py");
    }
    #[test]
    fn dangerous_paths_flatten_and_duplicate_names_are_disambiguated() {
        let blocks = extract_code_blocks("```rust ../../outside.rs\n1\n```\n```rust /root/outside.rs\n2\n```\n```rust ..\\..\\outside.rs\n3\n```").unwrap();
        assert_eq!(
            blocks.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
            ["outside.rs", "outside_2.rs", "outside_3.rs"]
        );
    }
    #[test]
    fn code_files_use_exclusive_creation_and_tar_has_only_regular_files() {
        let temp = tempfile::tempdir().unwrap();
        let blocks =
            extract_code_blocks("```rust first.rs\na\n```\n```bash script.sh\nb\n```").unwrap();
        let path = temp.path().join("answer.tar");
        save_archive(&path, &blocks).unwrap();
        assert!(save_archive(&path, &blocks).is_err());
        let mut archive = tar::Archive::new(std::fs::File::open(&path).unwrap());
        let names = archive
            .entries()
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                assert!(entry.header().entry_type().is_file());
                entry.path().unwrap().to_string_lossy().to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(names, ["snippets/first.rs", "snippets/script.sh"]);
        let one = temp.path().join("first.rs");
        save_single(&one, &blocks[0]).unwrap();
        assert!(save_single(&one, &blocks[0]).is_err());
    }
    #[test]
    fn archive_rejects_untrusted_external_names_before_writing() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("out.tar");
        for candidate in ["../../out.rs", "../x.rs", ".env", "a\\b.rs", "x/y.rs"] {
            let block = CodeArtifact {
                name: candidate.into(),
                language: "rust".into(),
                text: "test".into(),
            };
            assert!(save_archive(&output, &[block]).is_err());
            assert!(!output.exists());
        }
    }
    #[test]
    fn oversized_blocks_fail_before_writing_any_file() {
        let large = format!("```rust\n{}\n```", "X".repeat(MAX_BLOCK_BYTES));
        assert!(extract_code_blocks(&large).is_err());
    }
}
