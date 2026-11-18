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
const HARDENING: [&str; 14] = [
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
];

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
        if !self.is_git_repository() {
            bail!("The project folder is not a git repository");
        }
        let mut env = BTreeMap::new();
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
        Ok(result.stdout.lines().map(str::to_owned).collect())
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
        // Preview: everything that would be staged, against HEAD.
        let mut preview = vec![
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "HEAD",
            "--stat",
            "--patch",
        ];
        if !paths.is_empty() {
            preview.push("--");
            preview.extend(paths.iter().map(String::as_str));
        }
        let diff = self
            .git(&preview, true)
            .map(|r| r.stdout)
            .unwrap_or_default();
        let diff: String = diff.chars().take(MAX_DIFF_CHARS).collect();
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
            Operation::GitCommit { message, paths, branch },
        ))
    }

    pub(super) fn execute_git_commit(
        &self,
        message: &str,
        paths: &[String],
        branch: Option<&str>,
    ) -> Result<Value> {
        if !self.configured_filter_programs()?.is_empty() {
            bail!(
                "The repository configured filter programs after the preview; nothing was staged"
            );
        }
        if let Some(branch) = branch {
            self.git_ok(&["switch", "--create", branch], false)?;
        }
        let mut add = vec!["add", "--"];
        if paths.is_empty() {
            add = vec!["add", "--all", "--"];
            add.push(".");
        } else {
            add.extend(paths.iter().map(String::as_str));
        }
        self.git_ok(&add, false)?;
        self.git_ok(&["commit", "--no-verify", "--quiet", "-m", message], false)?;
        let commit = self.git_ok(&["log", "-1", "--format=%h %s"], true)?;
        Ok(json!({"ok": true, "commit": commit.trim(), "branch": branch}))
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
