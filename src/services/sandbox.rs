//! Bubblewrap isolation for commands and terminals the model proposes.
//!
//! A sandboxed process sees the operating system read-only, the project folder
//! (and any other `writable` path) read-write, a few chosen tool folders
//! read-only, and an empty home directory. It gets its own PID, IPC and UTS
//! namespaces, a private `/tmp`, a cleared environment, and no network unless
//! the policy allows it. Commands the user configured (adapters, MCP servers,
//! external agents) and terminals the user opens are not wrapped.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// How model-proposed commands are isolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SandboxMode {
    /// Sandbox whenever bubblewrap works; otherwise run unsandboxed and say so.
    #[default]
    Auto,
    /// Refuse to run model-proposed commands without a working sandbox.
    Required,
    /// Run commands with the user's ordinary permissions.
    Off,
}

impl SandboxMode {
    pub fn parse(value: &str) -> Result<Self> {
        Ok(match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "required" => Self::Required,
            "off" => Self::Off,
            other => bail!("Command sandbox must be auto, required or off (got {other:?})"),
        })
    }
}

/// The filesystem and network access granted to one sandboxed process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Bound read-write at the same path (normally just the project folder).
    pub writable: Vec<PathBuf>,
    /// Bound read-only at the same path when they exist (tool folders).
    pub read_only: Vec<PathBuf>,
    /// Share the host network namespace.
    pub network: bool,
}

const SYSTEM_DIRECTORIES: [&str; 6] = ["/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"];
const PASSED_ENVIRONMENT: [&str; 6] = ["PATH", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME"];

impl SandboxPolicy {
    /// One project folder, read-write, plus read-only tool folders.
    pub fn for_workspace(root: &Path, read_only: &[PathBuf], network: bool) -> Self {
        Self {
            writable: vec![root.to_path_buf()],
            read_only: read_only.to_vec(),
            network,
        }
    }

    /// The bubblewrap command line that runs `program args` under this policy.
    /// `interactive` keeps the controlling terminal (for PTY sessions).
    pub fn wrap(
        &self,
        program: &str,
        args: &[String],
        cwd: &Path,
        env: &BTreeMap<String, String>,
        interactive: bool,
    ) -> Result<(String, Vec<String>)> {
        let bwrap = bwrap_path().context(
            "The command sandbox needs bubblewrap (bwrap); install it or set the command sandbox to off",
        )?;
        if self.writable.is_empty() {
            bail!("A sandbox needs at least one writable folder");
        }
        if !self
            .writable
            .iter()
            .chain(self.read_only.iter())
            .any(|root| cwd.starts_with(root))
        {
            bail!(
                "Sandboxed working directory {} is outside the folders the sandbox can see",
                cwd.display()
            );
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .unwrap_or_else(|| PathBuf::from("/tmp/home"));
        let mut out: Vec<String> = vec!["--die-with-parent".into()];
        if !interactive {
            // Detach from the caller's terminal so the process cannot inject input.
            out.push("--new-session".into());
        }
        out.push("--unshare-all".into());
        if self.network {
            out.push("--share-net".into());
        }
        push(&mut out, ["--ro-bind", "/usr", "/usr"]);
        for directory in SYSTEM_DIRECTORIES {
            let path = Path::new(directory);
            match std::fs::symlink_metadata(path) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    let target = std::fs::read_link(path)?;
                    push(
                        &mut out,
                        ["--symlink", &target.to_string_lossy(), directory],
                    );
                }
                Ok(meta) if meta.is_dir() => push(&mut out, ["--ro-bind", directory, directory]),
                _ => {}
            }
        }
        push(&mut out, ["--ro-bind", "/etc", "/etc"]);
        push(&mut out, ["--ro-bind-try", "/opt", "/opt"]);
        if self.network {
            // Ubuntu's resolv.conf points into /run for DNS.
            push(
                &mut out,
                [
                    "--ro-bind-try",
                    "/run/systemd/resolve",
                    "/run/systemd/resolve",
                ],
            );
        }
        push(&mut out, ["--proc", "/proc"]);
        push(&mut out, ["--dev", "/dev"]);
        push(&mut out, ["--tmpfs", "/tmp"]);
        // Every home directory is replaced by an empty one; only granted
        // folders below are mounted back.
        push(&mut out, ["--tmpfs", "/home"]);
        if !home.starts_with("/home") {
            push(&mut out, ["--tmpfs", &home.to_string_lossy()]);
        } else {
            push(&mut out, ["--dir", &home.to_string_lossy()]);
        }
        for path in &self.read_only {
            let path = path.to_string_lossy();
            push(&mut out, ["--ro-bind-try", &path, &path]);
        }
        for path in &self.writable {
            if !path.is_dir() {
                bail!("Sandbox folder {} does not exist", path.display());
            }
            let path = path.to_string_lossy();
            push(&mut out, ["--bind", &path, &path]);
        }
        // Later mounts cover earlier ones. Protect what the writable folders
        // contain but sandboxed code must not change or read.
        for root in &self.writable {
            protect_writable_root(&mut out, root);
        }
        push(&mut out, ["--chdir", &cwd.to_string_lossy()]);
        out.push("--clearenv".into());
        push(&mut out, ["--setenv", "HOME", &home.to_string_lossy()]);
        for name in PASSED_ENVIRONMENT {
            if let Some(value) = std::env::var_os(name) {
                push(&mut out, ["--setenv", name, &value.to_string_lossy()]);
            }
        }
        for (name, value) in env {
            push(&mut out, ["--setenv", name, value]);
        }
        out.push("--".into());
        out.push(program.into());
        out.extend(args.iter().cloned());
        Ok((bwrap.to_string_lossy().into_owned(), out))
    }

    /// A short description for approval previews.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "sandboxed": true,
            "writable": self.writable,
            "read_only": self.read_only,
            "network": self.network,
            "home": "empty",
        })
    }
}

/// Entries scanned per writable folder when looking for secrets to hide.
const MAX_SCANNED_ENTRIES: usize = 200_000;
/// Hidden paths per writable folder; bubblewrap's argument list stays bounded.
const MAX_HIDDEN_PATHS: usize = 512;

/// Make `root/.git` read-only and hide secret files below `root`.
///
/// A writable `.git` lets sandboxed code add hooks or `core.fsmonitor`, which
/// then run unsandboxed on the user's next git command. Secret names follow the
/// file tools' deny list ([`crate::files::secret_name`]): files are replaced by
/// `/dev/null` and directories by an empty tmpfs. Symlinks are left alone; their
/// targets are mounted (or not) on their own terms.
fn protect_writable_root(out: &mut Vec<String>, root: &Path) {
    let git = root.join(".git");
    if std::fs::symlink_metadata(&git).is_ok_and(|m| m.is_dir() || m.is_file()) {
        let git = git.to_string_lossy();
        push(out, ["--ro-bind", &git, &git]);
    }
    let mut hidden = 0usize;
    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(false)
        .follow_links(false)
        .filter_entry(|entry| {
            // Generated trees and installed packages hold no project secrets,
            // and their public files (such as certifi's cacert.pem CA bundle)
            // must stay readable for tools to work.
            let generated = matches!(
                entry.file_name().to_str(),
                Some(
                    ".git"
                        | "target"
                        | "node_modules"
                        | ".venv"
                        | "venv"
                        | "__pycache__"
                        | "site-packages"
                        | "dist-packages"
                )
            );
            let virtualenv = entry.file_type().is_some_and(|t| t.is_dir())
                && entry.path().join("pyvenv.cfg").is_file();
            !generated && !virtualenv
        })
        .build();
    let mut skip_below: Vec<PathBuf> = Vec::new();
    for entry in walker.flatten().take(MAX_SCANNED_ENTRIES) {
        let path = entry.path();
        if path == root || skip_below.iter().any(|dir| path.starts_with(dir)) {
            continue;
        }
        let Some(name) = entry.file_name().to_str() else {
            continue;
        };
        if !crate::files::secret_name(name) {
            continue;
        }
        let Some(kind) = entry.file_type() else {
            continue;
        };
        let target = path.to_string_lossy();
        if kind.is_dir() {
            push(out, ["--tmpfs", &target]);
            skip_below.push(path.to_path_buf());
        } else if kind.is_file() {
            push(out, ["--ro-bind", "/dev/null", &target]);
        } else {
            continue;
        }
        hidden += 1;
        if hidden >= MAX_HIDDEN_PATHS {
            break;
        }
    }
}

fn push<const N: usize>(out: &mut Vec<String>, items: [&str; N]) {
    out.extend(items.iter().map(|item| item.to_string()));
}

/// The bubblewrap executable, if installed.
pub fn bwrap_path() -> Option<PathBuf> {
    crate::diagnostics::find_executable("bwrap")
}

/// Whether bubblewrap can actually create a sandbox here (user namespaces can
/// be disabled even when it is installed). Checked once per run.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let Some(bwrap) = bwrap_path() else {
            return false;
        };
        std::process::Command::new(bwrap)
            .args([
                "--die-with-parent",
                "--unshare-all",
                "--ro-bind",
                "/",
                "/",
                "--",
                "true",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
}

/// Expand `~/` and drop blank lines from a newline-separated path list.
pub fn parse_paths(text: &str) -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| match (line.strip_prefix("~/"), &home) {
            (Some(rest), Some(home)) => Some(home.join(rest)),
            (Some(_), None) => None,
            (None, _) => Some(PathBuf::from(line)),
        })
        .filter(|path| path.is_absolute())
        .collect()
}

pub const DEFAULT_READ_ONLY_PATHS: &str = "~/.cargo/bin\n~/.rustup\n~/.local/bin";

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn run(policy: &SandboxPolicy, cwd: &Path, script: &str) -> (bool, String) {
        let (program, args) = policy
            .wrap(
                "sh",
                &["-c".into(), script.into()],
                cwd,
                &BTreeMap::new(),
                false,
            )
            .unwrap();
        let output = Command::new(program).args(args).output().unwrap();
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).into_owned()
                + &String::from_utf8_lossy(&output.stderr),
        )
    }

    #[test]
    fn modes_parse_and_reject_unknown_values() {
        assert_eq!(SandboxMode::parse("").unwrap(), SandboxMode::Auto);
        assert_eq!(
            SandboxMode::parse("Required").unwrap(),
            SandboxMode::Required
        );
        assert_eq!(SandboxMode::parse("off").unwrap(), SandboxMode::Off);
        assert!(SandboxMode::parse("sometimes").is_err());
    }

    #[test]
    fn working_directory_must_be_visible_inside_the_sandbox() {
        if bwrap_path().is_none() {
            return;
        }
        let project = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let policy = SandboxPolicy::for_workspace(project.path(), &[], false);
        let error = policy
            .wrap("true", &[], outside.path(), &BTreeMap::new(), false)
            .unwrap_err();
        assert!(error.to_string().contains("outside"), "{error}");
    }

    #[test]
    fn sandbox_keeps_git_read_only_and_hides_project_secrets() {
        if !available() {
            eprintln!("bubblewrap unavailable; skipping");
            return;
        }
        let project = tempfile::tempdir().unwrap();
        let root = project.path();
        std::fs::create_dir_all(root.join(".git/hooks")).unwrap();
        std::fs::write(root.join(".git/config"), "[core]\n").unwrap();
        std::fs::write(root.join(".env"), "TOKEN=top-secret-env").unwrap();
        std::fs::create_dir_all(root.join("deploy/secrets")).unwrap();
        std::fs::write(root.join("deploy/secrets/db.txt"), "top-secret-db").unwrap();
        std::fs::write(root.join("deploy/server.pem"), "top-secret-pem").unwrap();
        std::fs::write(root.join("notes.txt"), "ordinary").unwrap();
        // Public CA bundles in installed packages stay readable.
        std::fs::create_dir_all(root.join("env/lib/site-packages/certifi")).unwrap();
        std::fs::write(root.join("env/pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        std::fs::write(
            root.join("env/lib/site-packages/certifi/cacert.pem"),
            "public-ca",
        )
        .unwrap();
        let policy = SandboxPolicy::for_workspace(root, &[], false);
        let (_, output) = run(
            &policy,
            root,
            "echo 'exit 0' > .git/hooks/pre-commit 2>&1; \
             echo '[core] fsmonitor = evil' >> .git/config 2>&1; \
             cat .env deploy/server.pem deploy/secrets/db.txt 2>&1; ls deploy/secrets; \
             cat notes.txt env/lib/site-packages/certifi/cacert.pem; echo changed > notes.txt",
        );
        assert!(!root.join(".git/hooks/pre-commit").exists(), "{output}");
        assert_eq!(
            std::fs::read_to_string(root.join(".git/config")).unwrap(),
            "[core]\n"
        );
        assert!(!output.contains("top-secret"), "{output}");
        assert!(
            output.contains("ordinary") && output.contains("public-ca"),
            "{output}"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).unwrap(),
            "changed\n"
        );
        // Hiding never alters the real files.
        assert_eq!(
            std::fs::read_to_string(root.join(".env")).unwrap(),
            "TOKEN=top-secret-env"
        );
    }

    #[test]
    fn sandbox_writes_only_the_project_and_hides_home_and_network() {
        if !available() {
            eprintln!("bubblewrap unavailable; skipping");
            return;
        }
        let project = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let secret = elsewhere.path().join("secret.txt");
        std::fs::write(&secret, "do not read").unwrap();
        let policy = SandboxPolicy::for_workspace(project.path(), &[], false);
        let (ok, output) = run(
            &policy,
            project.path(),
            &format!(
                "echo made > inside.txt; cat {} 2>&1; echo hostile > {}/x 2>&1; \
                 ls \"$HOME\" | wc -l; (getent hosts example.com >/dev/null && echo NET) || echo NONET",
                secret.display(),
                elsewhere.path().display()
            ),
        );
        assert!(ok || !output.is_empty(), "{output}");
        assert_eq!(
            std::fs::read_to_string(project.path().join("inside.txt")).unwrap(),
            "made\n"
        );
        assert!(!output.contains("do not read"), "{output}");
        assert!(
            !elsewhere.path().join("x").exists(),
            "wrote outside the project"
        );
        assert!(output.contains("NONET"), "{output}");
    }

    #[test]
    fn environment_is_cleared_except_safe_names_and_explicit_values() {
        if !available() {
            return;
        }
        let project = tempfile::tempdir().unwrap();
        let policy = SandboxPolicy::for_workspace(project.path(), &[], false);
        let mut env = BTreeMap::new();
        env.insert("REQUESTED".to_string(), "yes".to_string());
        let (program, args) = policy
            .wrap(
                "sh",
                &[
                    "-c".into(),
                    "echo ${REQUESTED:-none} ${LEAKED_TOKEN:-clear}".into(),
                ],
                project.path(),
                &env,
                false,
            )
            .unwrap();
        let output = Command::new(program)
            .args(args)
            .env("LEAKED_TOKEN", "secret")
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "yes clear");
    }

    #[test]
    fn read_only_paths_are_visible_but_not_writable() {
        if !available() {
            return;
        }
        let project = tempfile::tempdir().unwrap();
        let tools = tempfile::tempdir().unwrap();
        std::fs::write(tools.path().join("tool"), "present").unwrap();
        let policy =
            SandboxPolicy::for_workspace(project.path(), &[tools.path().to_path_buf()], false);
        let (_, output) = run(
            &policy,
            project.path(),
            &format!(
                "cat {0}/tool; echo; echo x > {0}/new 2>/dev/null || echo READONLY",
                tools.path().display()
            ),
        );
        assert!(output.contains("present"), "{output}");
        assert!(output.contains("READONLY"), "{output}");
        assert!(!tools.path().join("new").exists());
    }

    #[test]
    fn path_lists_expand_home_and_skip_comments() {
        let home = std::env::var("HOME").unwrap();
        let paths = parse_paths("~/.cargo/bin\n\n# note\n/opt/tools\nrelative/ignored");
        assert_eq!(
            paths,
            vec![
                PathBuf::from(format!("{home}/.cargo/bin")),
                PathBuf::from("/opt/tools")
            ]
        );
    }
}
