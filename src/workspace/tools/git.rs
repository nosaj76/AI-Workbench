//! Git awareness for the model: read-only status, diff and log, and a commit
//! that needs approval.
//!
//! A project's own `.git/config` is not trusted: every git invocation disables
//! hooks, fsmonitor, external diff programs, textconv and pagers, and a commit
//! is refused while the repository configures filter programs (which `git add`
//! would run). Read-only commands run in the command sandbox when it is
//! available, where `.git` is mounted read-only.

use super::terminal::optional_string;
use super::*;

const MAX_GIT_OUTPUT: usize = 256 * 1024;
const MAX_DIFF_CHARS: usize = 60_000;
const MAX_MESSAGE: usize = 4_000;

/// Settings that keep git from running programs named by repository config.
const HARDENING: [&str; 15] = [
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.pager=cat",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "core.sshCommand=false",
    "--no-pager",
    "--no-optional-locks",
    "--literal-pathspecs",
];

// Compute the actual prospective tree in a disposable index and object store.
// Git supplies the same pathspec, attributes, mode and ignore semantics as the
// final commit. Even an unborn branch can be previewed without writing .git.
const SNAPSHOT_ALIAS: &str = r#"alias.workbench-preview=!f() {
    set -eu
    scratch=$(mktemp -d)
    trap 'rm -rf -- "$scratch"' EXIT
    # Indexing may run a user/system clean filter even when .git/config is
    # clean. It must never run a helper merely to show an approval prompt.
    if git config --get-regexp '^filter\..*\.(clean|process)$' >/dev/null; then
        git ls-files --cached --others --exclude-standard -- "$@" > "$scratch/paths"
        git check-attr --stdin filter < "$scratch/paths" > "$scratch/attributes"
        sed 's/^.*: filter: //' "$scratch/attributes" | sort -u > "$scratch/filters"
        while IFS= read -r name; do
            case "$name" in unspecified|unset|'') continue ;; esac
            for field in clean process; do
                helper=$(git config --get "filter.$name.$field") || helper=
                if [ -n "$helper" ]; then
                    echo "A selected file uses a Git clean filter. Commit it manually or select other paths." >&2
                    exit 1
                fi
            done
        done < "$scratch/filters"
    fi
    head=$(git rev-parse --verify -q HEAD) || head=
    ref=$(git symbolic-ref -q HEAD) || ref=HEAD
    index=$(git rev-parse --path-format=absolute --git-path index)
    objects=$(git rev-parse --path-format=absolute --git-path objects)
    mkdir "$scratch/objects"
    export GIT_INDEX_FILE="$scratch/index" GIT_OBJECT_DIRECTORY="$scratch/objects"
    export GIT_ALTERNATE_OBJECT_DIRECTORIES="$objects${GIT_ALTERNATE_OBJECT_DIRECTORIES:+:$GIT_ALTERNATE_OBJECT_DIRECTORIES}"
    if [ "$#" -eq 0 ]; then
        if [ -f "$index" ]; then cp -- "$index" "$GIT_INDEX_FILE"; fi
        git add --all -- .
    else
        if [ -n "$head" ]; then git read-tree "$head"; else git read-tree --empty; fi
        git add --all -- "$@"
    fi
    tree=$(git write-tree)
    printf '%s\n' "$head" "$ref" "$tree"
    git diff --cached --no-color --no-ext-diff --no-textconv --stat --patch
}; f"#;

// This is the only hook in a fresh private directory: repository/user hooks
// remain disabled. Git invokes it on the *actual locked commit index*, after
// --only has assembled its tree, before publishing any commit.
const VERIFY_COMMIT: &str = r#"#!/bin/sh
set -eu
dir=${0%/*}
head=$(git rev-parse --verify -q HEAD) || head=
ref=$(git symbolic-ref -q HEAD) || ref=HEAD
if [ "$head" != "$(cat "$dir/head")" ] || [ "$ref" != "$(cat "$dir/ref")" ] ||
   [ "$(git write-tree)" != "$(cat "$dir/tree")" ]; then
    echo 'The Git branch, HEAD or proposed contents changed. Request a new commit preview.' >&2
    exit 1
fi
"#;

/// Other Git processes use this same lock for staging, reset and checkout.
struct GitIndexLock(PathBuf);

impl GitIndexLock {
    fn acquire(path: PathBuf) -> Result<Self> {
        OpenOptions::new().write(true).create_new(true).open(&path)
            .context("Could not lock the Git index; finish the other Git operation and retry")?;
        Ok(Self(path))
    }
}

impl Drop for GitIndexLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(super) fn definitions() -> Vec<Value> {
    let path = || json!({"type":"string"});
    vec![
        definition(
            "git_status",
            "Read-only: branch and changed files in the project's git repository.",
            json!({}),
            &[],
        ),
        definition(
            "git_diff",
            "Read-only: unified diff of uncommitted changes (staged=true for the index), optionally for one path.",
            json!({"path":path(),"staged":{"type":"boolean"}}),
            &[],
        ),
        definition(
            "git_log",
            "Read-only: recent commits, newest first.",
            json!({"limit":{"type":"integer","minimum":1,"maximum":100}}),
            &[],
        ),
        definition(
            "git_commit",
            "Approval: stage the given paths (default: all changes) and commit with message; optional new branch first.",
            json!({"message":{"type":"string"},"paths":{"type":"array","items":path()},"branch":{"type":"string"}}),
            &["message"],
        ),
    ]
}

impl ToolEngine {
    /// Whether the project root is a git working tree (a `.git` folder or file).
    pub(super) fn is_git_repository(&self) -> bool {
        std::fs::symlink_metadata(self.root().join(".git")).is_ok_and(|m| m.is_dir() || m.is_file())
    }

    fn git(&self, args: &[&str], read_only: bool) -> Result<integrations::CommandResult> {
        self.git_in_index(args, read_only, None)
    }

    fn git_in_index(&self, args: &[&str], read_only: bool, index: Option<&Path>)
        -> Result<integrations::CommandResult> {
        if !self.is_git_repository() {
            bail!("The project folder is not a git repository");
        }
        let mut env = BTreeMap::new();
        if let Some(index) = index {
            env.insert("GIT_INDEX_FILE".into(), index.to_str().context("Git index path is not UTF-8")?.into());
        }
        env.insert("GIT_OPTIONAL_LOCKS".into(), "0".into());
        env.insert("GIT_TERMINAL_PROMPT".into(), "0".into());
        env.insert("GIT_PAGER".into(), "cat".into());
        let spec = CommandSpec {
            program: "git".into(),
            args: HARDENING
                .iter()
                .chain(args)
                .map(|s| s.to_string())
                .collect(),
            cwd: Some(self.root().to_path_buf()),
            timeout_secs: 30,
            max_output_bytes: MAX_GIT_OUTPUT,
            env,
            sandbox: if read_only {
                self.command_sandbox()?
            } else {
                None
            },
            ..Default::default()
        };
        integrations::run_command(&spec, self.root())
    }

    fn git_ok(&self, args: &[&str], read_only: bool) -> Result<String> {
        let result = self.git(args, read_only)?;
        Self::git_output(args, result)
    }

    fn git_output(args: &[&str], result: integrations::CommandResult) -> Result<String> {
        if result.exit_code != Some(0) {
            bail!(
                "git {} failed: {}",
                args.first().unwrap_or(&""),
                result.stderr.trim()
            );
        }
        Ok(result.stdout)
    }

    pub(super) fn git_status(&self) -> Result<Value> {
        let out = self.git_ok(
            &[
                "status",
                "--porcelain=v1",
                "--branch",
                "--untracked-files=normal",
            ],
            true,
        )?;
        let mut lines = out.lines();
        let branch = lines
            .next()
            .and_then(|l| l.strip_prefix("## "))
            .unwrap_or("")
            .to_string();
        let files: Vec<Value> = lines
            .take(500)
            .map(|line| {
                let (code, path) = line.split_at(line.len().min(3));
                json!({"status": code.trim(), "path": path})
            })
            .collect();
        Ok(json!({"branch": branch, "clean": files.is_empty(), "changes": files}))
    }

    pub(super) fn git_diff(&self, args: &Value) -> Result<Value> {
        let staged = args.get("staged").and_then(Value::as_bool).unwrap_or(false);
        let mut command = vec!["diff", "--no-ext-diff", "--no-textconv", "--no-color"];
        if staged {
            command.push("--cached");
        }
        let path = optional_string(args, "path")?;
        if let Some(path) = &path {
            // The same confinement as file tools (no '..', no protected names).
            self.workspace.resolve(path, true)?;
            command.extend(["--", path.as_str()]);
        }
        let mut diff = self.git_ok(&command, true)?;
        let truncated = diff.chars().count() > MAX_DIFF_CHARS;
        if truncated {
            diff = diff.chars().take(MAX_DIFF_CHARS).collect();
        }
        Ok(
            json!({"staged": staged, "path": path, "diff": diff, "truncated": truncated, "empty": diff.is_empty()}),
        )
    }

    pub(super) fn git_log(&self, args: &Value) -> Result<Value> {
        let limit = uint(args, "limit", 15, 1, 100).to_string();
        let out = self.git_ok(
            &[
                "log",
                "--no-color",
                "-n",
                &limit,
                "--format=%h%x09%an%x09%ad%x09%s",
                "--date=short",
            ],
            true,
        )?;
        let commits: Vec<Value> = out
            .lines()
            .map(|line| {
                let mut parts = line.splitn(4, '\t');
                json!({"commit":parts.next(),"author":parts.next(),"date":parts.next(),"subject":parts.next()})
            })
            .collect();
        Ok(json!({ "commits": commits }))
    }

    /// Filter drivers in the repository's own configuration (not the user's
    /// global one, which they wrote) run during `git add`.
    fn configured_filter_programs(&self) -> Result<Vec<String>> {
        let result = self.git(
            &[
                "config",
                "--local",
                "--includes",
                "--get-regexp",
                r"^filter\..*\.(clean|smudge|process)$",
            ],
            true,
        )?;
        if result.exit_code == Some(1) {
            return Ok(Vec::new());
        }
        let config = Self::git_output(&["config"], result)?;
        Ok(config.lines().map(str::to_owned).collect())
    }

    fn prospective_commit(&self, paths: &[String]) -> Result<(GitCommitSnapshot, String)> {
        let mut command = vec!["-c", "core.splitIndex=false", "-c", SNAPSHOT_ALIAS, "workbench-preview"];
        command.extend(paths.iter().map(String::as_str));
        let output = self.git_ok(&command, true)?;
        let mut fields = output.splitn(4, '\n');
        let head = fields.next().unwrap_or_default().to_owned();
        let head_ref = fields.next().context("Missing preview branch")?.to_owned();
        let tree = fields.next().context("Missing preview tree")?.to_owned();
        if head_ref.is_empty() || !matches!(tree.len(), 40 | 64)
            || !tree.bytes().all(|c| c.is_ascii_hexdigit()) {
            bail!("Git returned an incomplete commit preview");
        }
        let diff = fields.next().unwrap_or_default().chars().take(MAX_DIFF_CHARS).collect();
        Ok((GitCommitSnapshot { head, head_ref, tree }, diff))
    }

    pub(super) fn prepare_git_commit(
        &self,
        args: &Value,
        expected: &mut Vec<PathState>,
    ) -> Result<(String, String, Value, Operation)> {
        let message = required(args, "message")?;
        if message.trim().is_empty() || message.len() > MAX_MESSAGE {
            bail!("A commit message of 1 to {MAX_MESSAGE} bytes is required");
        }
        let paths = if args.get("paths").is_some() {
            parse_string_array(args.get("paths"))?
        } else {
            Vec::new()
        };
        for path in &paths {
            let state = self.state(path)?;
            expected.push(state);
        }
        let branch = optional_string(args, "branch")?;
        if let Some(branch) = &branch {
            let valid = self.git(&["check-ref-format", "--branch", branch], true)?;
            if valid.exit_code != Some(0) || branch.starts_with('-') {
                bail!("{branch:?} is not a valid branch name");
            }
        }
        let filters = self.configured_filter_programs()?;
        if !filters.is_empty() {
            bail!("This repository configures filter programs that git add would run ({}). Commit it yourself; nothing was staged.", filters.join(", "));
        }
        let status = self.git_status()?;
        if status["clean"] == true {
            bail!("There is nothing to commit");
        }
        let (snapshot, diff) = self.prospective_commit(&paths)?;
        let details = json!({
            "message": message,
            "branch": branch,
            "paths": if paths.is_empty() { json!("all changes, including untracked files") } else { json!(paths) },
            "current_branch": status["branch"],
            "changes": status["changes"],
            "tracked_diff": diff,
        });
        Ok((
            match &branch {
                Some(branch) => format!("Commit to new branch {branch}"),
                None => "Commit changes".into(),
            },
            "Stage the listed changes and create a git commit. Hooks and signing are disabled; the commit can be undone with git reset.".into(),
            details,
            Operation::GitCommit { message, paths, branch, snapshot },
        ))
    }

    pub(super) fn execute_git_commit(
        &self,
        message: &str,
        paths: &[String],
        branch: Option<&str>,
        snapshot: &GitCommitSnapshot,
    ) -> Result<Value> {
        if !self.configured_filter_programs()?.is_empty() {
            bail!(
                "The repository configured filter programs after the preview; nothing was staged"
            );
        }
        if self.prospective_commit(paths)?.0 != *snapshot {
            bail!("The Git branch, HEAD or proposed contents changed. Request a new commit preview.");
        }
        let index_path = self.git_ok(&["rev-parse","--path-format=absolute","--git-path","index"], true)?;
        let index = PathBuf::from(index_path.trim_end_matches('\n'));
        let mut lock_path = index.as_os_str().to_owned();
        lock_path.push(".lock");
        let lock_path = PathBuf::from(lock_path);
        let _index_lock = GitIndexLock::acquire(lock_path)?;
        let directory = tempfile::tempdir_in(index.parent().context("Missing Git index directory")?)?;
        let shadow = directory.path().join("index");
        if index.try_exists()? {
            fs::copy(&index, &shadow)?;
        }
        // Every staging operation and Git's commit use this private index. Failed
        // approvals leave the user's real index and unrelated staged work intact.
        let run = |args: &[&str]| -> Result<String> {
            Self::git_output(args, self.git_in_index(args, false, Some(&shadow))?)
        };
        fs::write(directory.path().join("head"), &snapshot.head)?;
        fs::write(directory.path().join("tree"), &snapshot.tree)?;
        let expected_ref = branch.map(|b| format!("refs/heads/{b}"))
            .unwrap_or_else(|| snapshot.head_ref.clone());
        fs::write(directory.path().join("ref"), &expected_ref)?;
        let hook = directory.path().join("pre-commit");
        fs::write(&hook, VERIFY_COMMIT)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&hook, fs::Permissions::from_mode(0o700))?;
        }
        let guard_config = format!("core.hooksPath={}", directory.path().display());
        let mut commit = vec!["-c", &guard_config, "commit", "--quiet", "-m", message];
        if paths.is_empty() {
            run(&["add", "--all", "--", "."])?;
        } else {
            let mut register = vec!["add", "--intent-to-add", "--"];
            for path in paths {
                if self.workspace.resolve(path, true)?.try_exists()? {
                    register.push(path);
                }
            }
            if register.len() > 3 {
                run(&register)?;
            }
            commit.extend(["--only", "--"]);
            commit.extend(paths.iter().map(String::as_str));
        }
        if let Some(branch) = branch {
            run(&["switch", "--create", branch])?;
        }
        if let Err(error) = run(&commit) {
            if branch.is_some() {
                let rollback = self.restore_commit_branch(snapshot, &expected_ref);
                if let Err(rollback) = rollback {
                    bail!("{error:#}. Original staging is intact. Check the current branch before retrying: {rollback:#}");
                }
            }
            return Err(error);
        }
        if let Err(error) = fs::rename(&shadow, &index) {
            let saved_index = directory.keep().join("index");
            bail!("Git created the commit, but updating the index failed: {error}. The committed index is preserved at {}. Check git status before retrying.", saved_index.display());
        }
        let commit = self.git_ok(&["log", "-1", "--format=%h %s"], true)?;
        Ok(json!({"ok": true, "commit": commit.trim(), "branch": branch}))
    }

    fn restore_commit_branch(&self, snapshot: &GitCommitSnapshot, created: &str) -> Result<()> {
        let current = self.git_ok(&["symbolic-ref","-q","HEAD"], false)?;
        if current.trim_end() != created {
            bail!("HEAD has since changed; both branches were preserved");
        }
        for reference in ["HEAD", snapshot.head_ref.as_str()] {
            let value = self.git(&["rev-parse","--verify","-q",reference], false)?;
            let valid = if snapshot.head.is_empty() {
                value.exit_code == Some(1)
            } else {
                value.exit_code == Some(0) && value.stdout.trim_end() == snapshot.head
            };
            if !valid {
                bail!("A branch tip changed; the checkout was preserved");
            }
        }
        // Both branches still point at the same commit. Nothing in the working
        // tree needs to be checked out; preserve edits made during approval.
        if snapshot.head_ref == "HEAD" {
            self.git_ok(&["update-ref","--no-deref","HEAD",&snapshot.head,&snapshot.head], false)?;
        } else {
            self.git_ok(&["symbolic-ref","HEAD",&snapshot.head_ref], false)?;
        }
        if !snapshot.head.is_empty() {
            self.git_ok(&["update-ref","-d",created,&snapshot.head], false)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .output()
            .unwrap();
        assert!(status.status.success(), "{:?}", status);
    }

    fn repo() -> (tempfile::TempDir, tempfile::TempDir, ToolEngine) {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q", "-b", "main"]);
        git(
            dir.path(),
            &["config", "user.email", "test@example.invalid"],
        );
        git(dir.path(), &["config", "user.name", "Test"]);
        std::fs::write(dir.path().join("a.txt"), "one\n").unwrap();
        git(dir.path(), &["add", "a.txt"]);
        git(dir.path(), &["commit", "-q", "-m", "first"]);
        let engine = ToolEngine::with_backup_dir(
            Workspace::new(dir.path().to_path_buf()).unwrap(),
            backups.path().to_path_buf(),
        );
        (dir, backups, engine)
    }

    fn immediate(engine: &ToolEngine, name: &str, args: Value) -> Value {
        match engine.prepare(name, &args).unwrap() {
            PreparedTool::Immediate(v) => v,
            PreparedTool::Approval(_) => panic!("{name} must not need approval"),
        }
    }

    #[test]
    fn selected_commit_leaves_other_staged_work_in_the_index() {
        let (dir, _backups, engine) = repo();
        std::fs::write(dir.path().join("unrelated"), "work in progress").unwrap();
        git(dir.path(), &["add", "unrelated"]);
        std::fs::write(dir.path().join("selected"), "new file").unwrap();
        git(dir.path(), &["rm", "a.txt"]);
        let PreparedTool::Approval(action) = engine.prepare(
            "git_commit", &json!({"message":"selected only","paths":["a.txt", "selected"]})
        ).unwrap() else { panic!("expected preview") };
        engine.approve(&action).unwrap();
        assert_eq!(engine.git_ok(&["diff", "--name-only", "HEAD^", "HEAD"], true).unwrap().trim(), "a.txt\nselected");
        assert_eq!(engine.git_ok(&["diff", "--cached", "--name-only"], true).unwrap().trim(), "unrelated");
    }

    #[test]
    fn approval_rejects_new_files_same_size_edits_and_branch_or_head_changes() {
        let (dir, _backups, engine) = repo();
        let prepare = || match engine.prepare("git_commit", &json!({"message":"reviewed"})).unwrap() {
            PreparedTool::Approval(a) => a,
            _ => panic!("expected approval"),
        };
        fs::write(dir.path().join("a.txt"), "two\n").unwrap();
        let preview = prepare();
        fs::write(dir.path().join("later.txt"), "later").unwrap();
        assert!(engine.approve(&preview).is_err());
        assert_eq!(engine.git_ok(&["diff", "--cached", "--name-only"], true).unwrap(), "");
        let preview = prepare();
        fs::write(dir.path().join("later.txt"), "other").unwrap();
        assert!(engine.approve(&preview).is_err());
        let preview = prepare();
        git(dir.path(), &["switch","--create","other"]);
        assert!(engine.approve(&preview).is_err());
        let preview = prepare();
        git(dir.path(), &["commit","--allow-empty","-qm","advance HEAD"]);
        assert!(engine.approve(&preview).is_err());
    }

    #[test]
    fn a_failed_selected_commit_leaves_no_intent_to_add_entries() {
        let (dir, _backups, engine) = repo();
        std::fs::write(dir.path().join("new.txt"), "new file").unwrap();
        let PreparedTool::Approval(action) = engine.prepare(
            "git_commit", &json!({"message":"fails","paths":["new.txt"],"branch":"attempt"})
        ).unwrap() else { panic!("expected preview") };
        git(dir.path(), &["config","user.name",""]);
        git(dir.path(), &["config","user.email",""]);
        assert!(engine.approve(&action).is_err());
        assert_eq!(engine.git_ok(&["symbolic-ref","--short","HEAD"], true).unwrap().trim(), "main");
        assert!(engine.git_ok(&["branch","--list","attempt"], true).unwrap().is_empty());
        assert_eq!(engine.git_ok(&["ls-files", "--cached", "--", "new.txt"], true).unwrap(), "");
        assert!(dir.path().join("new.txt").exists());
    }

    #[test]
    fn git_tools_appear_only_in_repositories() {
        let plain = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let engine = ToolEngine::with_backup_dir(
            Workspace::new(plain.path().to_path_buf()).unwrap(),
            backups.path().to_path_buf(),
        );
        let names = |e: &ToolEngine| {
            e.definitions()
                .iter()
                .map(|d| d["function"]["name"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert!(!names(&engine).contains(&"git_status".to_string()));
        assert!(engine.prepare("git_status", &json!({})).is_err());
        let (_dir, _b, engine) = repo();
        assert!(names(&engine).contains(&"git_status".to_string()));
    }

    #[test]
    fn status_diff_log_and_an_approved_commit_on_a_new_branch() {
        let (dir, _b, engine) = repo();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "new\n").unwrap();
        let status = immediate(&engine, "git_status", json!({}));
        assert_eq!(status["clean"], false);
        assert!(status["branch"].as_str().unwrap().starts_with("main"));
        let diff = immediate(&engine, "git_diff", json!({"path":"a.txt"}));
        assert!(diff["diff"].as_str().unwrap().contains("+two"), "{diff}");
        let log = immediate(&engine, "git_log", json!({"limit":5}));
        assert_eq!(log["commits"][0]["subject"], "first");

        let action = match engine
            .prepare(
                "git_commit",
                &json!({"message":"add two","branch":"ai/two"}),
            )
            .unwrap()
        {
            PreparedTool::Approval(a) => *a,
            PreparedTool::Immediate(_) => panic!("commit needs approval"),
        };
        assert_eq!(action.operation.approval_kind(), ApprovalKind::FileChange);
        assert!(action.details["tracked_diff"]
            .as_str()
            .unwrap()
            .contains("+two"));
        let result = engine.approve(&action).unwrap();
        assert_eq!(result["ok"], true, "{result}");
        let status = immediate(&engine, "git_status", json!({}));
        assert_eq!(status["clean"], true, "{status}");
        assert!(status["branch"].as_str().unwrap().starts_with("ai/two"));
        assert!(engine
            .prepare("git_commit", &json!({"message":"again"}))
            .is_err());
    }

    #[test]
    fn repository_config_cannot_make_git_run_programs() {
        let (dir, _b, engine) = repo();
        let marker = dir.path().join("pwned");
        let script = dir.path().join("evil.sh");
        std::fs::write(&script, format!("#!/bin/sh\ntouch {}\n", marker.display())).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let s = script.display().to_string();
        git(dir.path(), &["config", "core.fsmonitor", &s]);
        git(dir.path(), &["config", "core.pager", &s]);
        git(dir.path(), &["config", "diff.external", &s]);
        std::fs::create_dir_all(dir.path().join(".git/hooks")).unwrap();
        std::fs::copy(&script, dir.path().join(".git/hooks/pre-commit")).unwrap();
        std::fs::copy(&script, dir.path().join(".git/hooks/post-commit")).unwrap();
        std::fs::write(dir.path().join("a.txt"), "changed\n").unwrap();
        immediate(&engine, "git_status", json!({}));
        immediate(&engine, "git_diff", json!({}));
        immediate(&engine, "git_log", json!({}));
        let action = match engine
            .prepare("git_commit", &json!({"message":"m","paths":["a.txt"]}))
            .unwrap()
        {
            PreparedTool::Approval(a) => *a,
            PreparedTool::Immediate(_) => unreachable!(),
        };
        engine.approve(&action).unwrap();
        assert!(!marker.exists(), "a repository-configured program ran");

        // Filter drivers would run during `git add`: refuse to stage.
        git(dir.path(), &["config", "filter.x.clean", &s]);
        std::fs::write(dir.path().join("a.txt"), "again\n").unwrap();
        let error = engine
            .prepare("git_commit", &json!({"message":"m"}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("filter programs"), "{error}");
        assert!(!marker.exists());
    }
}
