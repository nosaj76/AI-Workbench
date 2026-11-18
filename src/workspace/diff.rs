//! Line diffs for previews of file changes.
//!
//! Common leading and trailing lines are trimmed first, then the middle is
//! diffed with Myers' algorithm. Very large rewrites fall back to "remove the
//! old middle, add the new middle", which is still exact, only less minimal.

/// One line of a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    Context,
    Removed,
    Added,
}

/// A run of changes with surrounding context, numbered from 1 like `diff -u`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    pub old_start: usize,
    pub old_len: usize,
    pub new_start: usize,
    pub new_len: usize,
    pub lines: Vec<(LineKind, String)>,
}

/// Summary counts for a change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiffStats {
    pub added: usize,
    pub removed: usize,
}

/// Edit budget for Myers' algorithm; beyond it the middle is replaced wholesale.
const MAX_TRACE_CELLS: usize = 20_000_000;

fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split_inclusive('\n').collect()
}

/// The edit script turning `old` into `new`, one entry per line.
fn edit_script<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<(LineKind, &'a str)> {
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let a = &old[prefix..old.len() - suffix];
    let b = &new[prefix..new.len() - suffix];
    let mut script: Vec<(LineKind, &str)> = old[..prefix]
        .iter()
        .map(|l| (LineKind::Context, *l))
        .collect();
    script.extend(myers(a, b));
    script.extend(
        old[old.len() - suffix..]
            .iter()
            .map(|l| (LineKind::Context, *l)),
    );
    script
}

fn replace_all<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(LineKind, &'a str)> {
    a.iter()
        .map(|l| (LineKind::Removed, *l))
        .chain(b.iter().map(|l| (LineKind::Added, *l)))
        .collect()
}

fn myers<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(LineKind, &'a str)> {
    let (n, m) = (a.len() as isize, b.len() as isize);
    if n == 0 || m == 0 {
        return replace_all(a, b);
    }
    let max = (n + m) as usize;
    let width = 2 * max + 1;
    let max_d = (MAX_TRACE_CELLS / width).min(max);
    let offset = max as isize;
    let mut v = vec![0isize; width];
    let mut trace: Vec<Vec<isize>> = Vec::new();
    let mut found = false;
    'search: for d in 0..=max_d as isize {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let index = (k + offset) as usize;
            let mut x = if k == -d || (k != d && v[index - 1] < v[index + 1]) {
                v[index + 1]
            } else {
                v[index - 1] + 1
            };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[index] = x;
            if x >= n && y >= m {
                found = true;
                break 'search;
            }
            k += 2;
        }
    }
    if !found {
        return replace_all(a, b);
    }
    // Walk the trace backwards from (n, m).
    let mut script = Vec::new();
    let (mut x, mut y) = (n, m);
    for d in (0..trace.len() as isize).rev() {
        let v = &trace[d as usize];
        let k = x - y;
        let previous_k =
            if k == -d || (k != d && v[(k - 1 + offset) as usize] < v[(k + 1 + offset) as usize]) {
                k + 1
            } else {
                k - 1
            };
        let previous_x = if d == 0 {
            0
        } else {
            v[(previous_k + offset) as usize]
        };
        let previous_y = previous_x - previous_k;
        while x > previous_x && y > previous_y {
            x -= 1;
            y -= 1;
            script.push((LineKind::Context, a[x as usize]));
        }
        if d > 0 {
            if x == previous_x {
                y -= 1;
                script.push((LineKind::Added, b[y as usize]));
            } else {
                x -= 1;
                script.push((LineKind::Removed, a[x as usize]));
            }
        }
    }
    script.reverse();
    script
}

/// Hunks with `context` unchanged lines around each change.
pub fn hunks(before: &str, after: &str, context: usize) -> Vec<Hunk> {
    let old = split_lines(before);
    let new = split_lines(after);
    let script = edit_script(&old, &new);
    let changed: Vec<usize> = script
        .iter()
        .enumerate()
        .filter(|(_, (kind, _))| *kind != LineKind::Context)
        .map(|(i, _)| i)
        .collect();
    let mut result = Vec::new();
    let mut i = 0;
    while i < changed.len() {
        let start = changed[i].saturating_sub(context);
        let mut end = changed[i];
        while i + 1 < changed.len() && changed[i + 1] <= end + 2 * context + 1 {
            i += 1;
            end = changed[i];
        }
        let end = (end + context + 1).min(script.len());
        // Line numbers at `start`.
        let (mut old_line, mut new_line) = (1, 1);
        for (kind, _) in &script[..start] {
            match kind {
                LineKind::Context => {
                    old_line += 1;
                    new_line += 1;
                }
                LineKind::Removed => old_line += 1,
                LineKind::Added => new_line += 1,
            }
        }
        let slice = &script[start..end];
        let old_len = slice.iter().filter(|(k, _)| *k != LineKind::Added).count();
        let new_len = slice
            .iter()
            .filter(|(k, _)| *k != LineKind::Removed)
            .count();
        result.push(Hunk {
            old_start: old_line,
            old_len,
            new_start: new_line,
            new_len,
            lines: slice
                .iter()
                .map(|(kind, line)| (*kind, line.trim_end_matches(['\n', '\r']).to_string()))
                .collect(),
        });
        i += 1;
    }
    result
}

pub fn stats(hunks: &[Hunk]) -> DiffStats {
    let mut stats = DiffStats::default();
    for (kind, _) in hunks.iter().flat_map(|h| &h.lines) {
        match kind {
            LineKind::Added => stats.added += 1,
            LineKind::Removed => stats.removed += 1,
            LineKind::Context => {}
        }
    }
    stats
}

/// A `diff -u` style rendering (without file headers).
pub fn unified(before: &str, after: &str, context: usize) -> String {
    let mut out = String::new();
    for hunk in hunks(before, after, context) {
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            hunk.old_start, hunk.old_len, hunk.new_start, hunk.new_len
        ));
        for (kind, line) in &hunk.lines {
            out.push(match kind {
                LineKind::Context => ' ',
                LineKind::Removed => '-',
                LineKind::Added => '+',
            });
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(before: &str, hunks: &[Hunk]) -> String {
        // Rebuild "after" from the script to prove the diff is exact.
        let old: Vec<&str> = before.lines().collect();
        let mut out = Vec::new();
        let mut next = 0usize;
        for hunk in hunks {
            while next + 1 < hunk.old_start {
                out.push(old[next].to_string());
                next += 1;
            }
            for (kind, line) in &hunk.lines {
                match kind {
                    LineKind::Context => {
                        out.push(line.clone());
                        next += 1;
                    }
                    LineKind::Removed => next += 1,
                    LineKind::Added => out.push(line.clone()),
                }
            }
        }
        out.extend(old[next..].iter().map(|s| s.to_string()));
        out.join("\n")
    }

    #[test]
    fn a_one_line_change_shows_context_and_numbers() {
        let before = "a\nb\nc\nd\ne\nf\ng\n";
        let after = "a\nb\nc\nD\ne\nf\ng\n";
        assert_eq!(
            unified(before, after, 2),
            "@@ -2,5 +2,5 @@\n b\n c\n-d\n+D\n e\n f\n"
        );
    }

    #[test]
    fn new_and_emptied_files() {
        assert_eq!(unified("", "x\ny\n", 3), "@@ -1,0 +1,2 @@\n+x\n+y\n");
        assert_eq!(unified("x\n", "", 3), "@@ -1,1 +1,0 @@\n-x\n");
        assert!(hunks("same\n", "same\n", 3).is_empty());
    }

    #[test]
    fn distant_changes_make_separate_hunks_and_rebuild_exactly() {
        let before: String = (0..60).map(|i| format!("line {i}\n")).collect();
        let after = before
            .replace("line 5\n", "line five\n")
            .replace("line 50\n", "")
            .replace("line 30\n", "line 30\ninserted\n");
        let h = hunks(&before, &after, 3);
        assert_eq!(h.len(), 3);
        assert_eq!(
            stats(&h),
            DiffStats {
                added: 2,
                removed: 2
            }
        );
        assert_eq!(apply(&before, &h), after.trim_end());
    }

    #[test]
    fn interleaved_edits_are_minimal() {
        let h = hunks("a\nb\nc\nd\n", "a\nx\nc\ny\n", 0);
        assert_eq!(
            stats(&h),
            DiffStats {
                added: 2,
                removed: 2
            }
        );
    }

    #[test]
    fn huge_rewrites_fall_back_to_an_exact_replacement() {
        let before: String = (0..40_000).map(|i| format!("{i}\n")).collect();
        let after: String = (0..40_000).map(|i| format!("{}\n", i * 7 + 1)).collect();
        let h = hunks(&before, &after, 0);
        let s = stats(&h);
        assert!(s.added > 0 && s.removed > 0);
        assert_eq!(apply(&before, &h), after.trim_end());
    }
}
