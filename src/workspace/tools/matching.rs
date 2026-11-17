//! Tolerant location of text for `replace_text`.
//!
//! Small models often copy code with the wrong indentation, trailing spaces or
//! line endings, and an exact-match edit then fails for no useful reason. This
//! finds the one place the text really is, and returns the exact text the file
//! contains so the approval preview and the write use the real bytes. When
//! nothing matches it names the closest real lines instead of only failing.
//!
//! The approach (try exact, then tolerate whitespace, then show the closest
//! chunk) follows the search/replace editing of aider (Apache-2.0); this is an
//! independent implementation.
use anyhow::{bail, Result};

/// Where the text was found, as it exists in the file.
#[derive(Debug, PartialEq, Eq)]
pub struct Located {
    /// The exact text in the file to replace.
    pub find: String,
    /// The replacement, re-indented when the match differed in indentation.
    pub replacement: String,
    pub occurrences: usize,
    /// Set when the match was not exact, saying how it differed.
    pub note: Option<String>,
}

const CLOSE_ENOUGH: f64 = 0.6;
const SHOWN_LINES: usize = 20;

/// Lines of `text` with the byte offset of each line start.
fn lines_with_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = 0;
    for piece in text.split_inclusive('\n') {
        out.push((start, piece.trim_end_matches(['\n', '\r'])));
        start += piece.len();
    }
    out
}

fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Dice similarity of two lines' character pairs, ignoring surrounding space.
fn line_similarity(a: &str, b: &str) -> f64 {
    let (a, b) = (a.trim(), b.trim());
    if a == b {
        return 1.0;
    }
    let pairs = |s: &str| -> Vec<(char, char)> {
        let chars: Vec<char> = s.chars().collect();
        chars.windows(2).map(|w| (w[0], w[1])).collect()
    };
    let (pa, mut pb) = (pairs(a), pairs(b));
    if pa.is_empty() || pb.is_empty() {
        return 0.0;
    }
    let total = pa.len() + pb.len();
    let mut shared = 0;
    for pair in &pa {
        if let Some(i) = pb.iter().position(|q| q == pair) {
            pb.swap_remove(i);
            shared += 1;
        }
    }
    2.0 * shared as f64 / total as f64
}

/// Byte range of the file text covering lines `first..first + count`, without
/// the final line break (so it matches a `find` that has none).
fn span(file_lines: &[(usize, &str)], first: usize, count: usize, text: &str) -> (usize, usize) {
    let start = file_lines[first].0;
    let (last_start, last) = file_lines[first + count - 1];
    let end = last_start + last.len();
    debug_assert!(end <= text.len());
    (start, end)
}

/// Windows of the file whose lines equal the wanted lines under `same`.
fn windows(file: &[(usize, &str)], want: &[&str], same: impl Fn(&str, &str) -> bool) -> Vec<usize> {
    if want.is_empty() || file.len() < want.len() {
        return Vec::new();
    }
    (0..=file.len() - want.len())
        .filter(|&i| want.iter().enumerate().all(|(j, w)| same(file[i + j].1, w)))
        .collect()
}

/// Re-indents `replacement` when the file's text is indented differently from
/// the text the model wrote, but only when the difference is one uniform shift.
fn reindent(file_window: &[(usize, &str)], want: &[&str], replacement: &str) -> Option<String> {
    let pairs: Vec<(&str, &str)> = file_window
        .iter()
        .map(|(_, line)| *line)
        .zip(want.iter().copied())
        .filter(|(f, w)| !f.trim().is_empty() && !w.trim().is_empty())
        .collect();
    let (first_file, first_want) = *pairs.first()?;
    let (file_prefix, want_prefix) = (indent_of(first_file), indent_of(first_want));
    for (f, w) in &pairs {
        let (fi, wi) = (indent_of(f), indent_of(w));
        let tail = wi.strip_prefix(want_prefix)?;
        if fi != format!("{file_prefix}{tail}") {
            return None;
        }
    }
    let mut out = String::new();
    for piece in replacement.split_inclusive('\n') {
        let body = piece.trim_end_matches(['\n', '\r']);
        let ending = &piece[body.len()..];
        if body.trim().is_empty() {
            out.push_str(piece);
            continue;
        }
        // A replacement line that does not carry the model's own indentation is
        // left alone rather than guessed at.
        match body.strip_prefix(want_prefix) {
            Some(rest) => out.push_str(&format!("{file_prefix}{rest}{ending}")),
            None => out.push_str(piece),
        }
    }
    Some(out)
}

fn message_for_no_match(file: &[(usize, &str)], want: &[&str]) -> String {
    let mut best: Option<(f64, usize)> = None;
    if !want.is_empty() && file.len() >= want.len() {
        for i in 0..=file.len() - want.len() {
            let score: f64 = want
                .iter()
                .enumerate()
                .map(|(j, w)| line_similarity(file[i + j].1, w))
                .sum::<f64>()
                / want.len() as f64;
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, i));
            }
        }
    }
    match best {
        Some((score, at)) if score >= CLOSE_ENOUGH => {
            let shown: Vec<String> = file[at..at + want.len()]
                .iter()
                .take(SHOWN_LINES)
                .enumerate()
                .map(|(j, (_, line))| format!("{}: {line}", at + j + 1))
                .collect();
            let more = want.len().saturating_sub(SHOWN_LINES);
            format!(
                "No match for the text to replace. The closest real lines are {}-{} ({:.0}% similar):\n{}{}\nCopy the lines exactly as they are in the file into `find`.",
                at + 1,
                at + want.len(),
                score * 100.0,
                shown.join("\n"),
                if more > 0 { format!("\n({more} more lines in that range are not shown.)") } else { String::new() }
            )
        }
        _ => "No match for the text to replace, and nothing in the file is close to it. Re-read the file with read_file and copy the lines exactly.".to_string(),
    }
}

/// Finds `find` in `before` for replacement. `replace_all` allows several
/// exact matches; otherwise there must be exactly one place.
pub fn locate(before: &str, find: &str, replacement: &str, replace_all: bool) -> Result<Located> {
    let exact = before.matches(find).count();
    if exact == 1 || (exact > 1 && replace_all) {
        return Ok(Located {
            find: find.to_string(),
            replacement: replacement.to_string(),
            occurrences: exact,
            note: None,
        });
    }
    let file = lines_with_offsets(before);
    if exact > 1 {
        let at: Vec<String> = before
            .match_indices(find)
            .map(|(offset, _)| (before[..offset].matches('\n').count() + 1).to_string())
            .collect();
        bail!(
            "The text to replace matches {exact} places (lines {}); include more surrounding lines so exactly one matches, or set replace_all=true",
            at.join(", ")
        );
    }
    if find.trim().is_empty() {
        bail!("The text to replace is empty");
    }
    let want: Vec<&str> = find
        .trim_end_matches(['\n', '\r'])
        .split('\n')
        .map(|l| l.trim_end_matches('\r'))
        .collect();
    // Each step tolerates one more kind of difference; the first that finds the
    // text in exactly one place wins.
    type Same = fn(&str, &str) -> bool;
    let steps: [(&str, Same); 2] = [
        ("trailing spaces or line endings differed", |f, w| {
            f.trim_end() == w.trim_end()
        }),
        ("indentation differed", |f, w| f.trim() == w.trim()),
    ];
    for (what, same) in steps {
        let found = windows(&file, &want, same);
        match found.as_slice() {
            [] => continue,
            [at] => {
                let window = &file[*at..*at + want.len()];
                let (start, end) = span(&file, *at, want.len(), before);
                let new_text = if what.starts_with("indentation") {
                    match reindent(window, &want, replacement) {
                        Some(text) => text,
                        // Not one uniform shift: the match is a guess, so refuse.
                        None => continue,
                    }
                } else {
                    replacement.trim_end_matches('\r').to_string()
                };
                return Ok(Located {
                    find: before[start..end].to_string(),
                    // The matched text ends at its last line's content, so the
                    // replacement does too.
                    replacement: new_text.trim_end_matches(['\n', '\r']).to_string(),
                    occurrences: 1,
                    note: Some(format!(
                        "matched lines {}-{} although {what}",
                        at + 1,
                        at + want.len()
                    )),
                });
            }
            many => bail!(
                "The text to replace matches {} places once whitespace is ignored (starting at lines {}); include more surrounding lines so exactly one matches",
                many.len(),
                many.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    bail!("{}", message_for_no_match(&file, &want))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "fn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{}\", x + y);\n}\n\nfn other() {\n    let x = 1;\n}\n";

    #[test]
    fn an_exact_unique_match_is_unchanged() {
        let r = locate(FILE, "    let y = 2;\n", "    let y = 3;\n", false).unwrap();
        assert_eq!(
            (r.find.as_str(), r.occurrences, r.note.is_none()),
            ("    let y = 2;\n", 1, true)
        );
    }

    #[test]
    fn two_exact_matches_name_their_lines_and_replace_all_allows_them() {
        let err = locate(FILE, "let x = 1;", "let x = 9;", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("matches 2 places (lines 2, 8)"), "{err}");
        assert_eq!(
            locate(FILE, "let x = 1;", "let x = 9;", true)
                .unwrap()
                .occurrences,
            2
        );
    }

    #[test]
    fn wrong_indentation_still_finds_the_one_place_and_reindents_the_replacement() {
        // The model wrote the block with no indentation.
        let find = "let y = 2;\nprintln!(\"{}\", x + y);";
        let replacement = "let y = 3;\nprintln!(\"{}\", x * y);";
        let r = locate(FILE, find, replacement, false).unwrap();
        assert_eq!(r.find, "    let y = 2;\n    println!(\"{}\", x + y);");
        assert_eq!(
            r.replacement,
            "    let y = 3;\n    println!(\"{}\", x * y);"
        );
        assert!(r.note.unwrap().contains("indentation differed"));
        // Applying it gives a correctly indented file.
        let after = FILE.replacen(&r.find, &r.replacement, 1);
        assert!(
            after.contains("    let y = 3;\n    println!(\"{}\", x * y);\n}"),
            "{after}"
        );
    }

    #[test]
    fn trailing_spaces_and_crlf_do_not_prevent_a_match() {
        let file = "alpha  \r\nbeta\r\ngamma\r\n";
        let r = locate(file, "alpha\nbeta", "ALPHA\nBETA", false).unwrap();
        assert_eq!(r.find, "alpha  \r\nbeta");
        assert!(r.note.unwrap().contains("trailing spaces or line endings"));
    }

    #[test]
    fn an_ambiguous_loose_match_is_refused_not_guessed() {
        let file = "if a {\n    run();\n}\nif b {\n  run();\n}\n";
        let err = locate(file, "run();", "stop();", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("matches 2 places"), "{err}");
        let err = locate("    run();\n  run();\n", "run();", "stop();", false);
        assert!(err.is_err());
    }

    #[test]
    fn a_non_uniform_indentation_difference_is_not_treated_as_a_match() {
        // Lines are indented differently from each other, so shifting is a guess.
        let file = "    a();\n b();\n";
        let err = locate(file, "a();\nb();", "x();", false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("No match") || err.contains("closest"), "{err}");
    }

    #[test]
    fn a_failure_shows_the_closest_real_lines() {
        let err = locate(
            FILE,
            "    let y = 20;\n    println!(\"{}\", x - y);",
            "z",
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("closest real lines are 3-4"), "{err}");
        assert!(
            err.contains("3:     let y = 2;") && err.contains("4:     println!"),
            "{err}"
        );
        let nothing = locate(FILE, "completely unrelated text here\nand more", "z", false)
            .unwrap_err()
            .to_string();
        assert!(
            nothing.contains("nothing in the file is close"),
            "{nothing}"
        );
        assert!(locate(FILE, "  \n", "z", false).is_err());
    }
}
