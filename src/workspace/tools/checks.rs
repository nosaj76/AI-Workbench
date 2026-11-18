//! Automatic checks after approved file changes.
//!
//! When a change is applied, the project's compiler or a per-file syntax check
//! runs in the command sandbox (never outside it, never with network) and its
//! diagnostics are attached to the tool result, so the model sees errors
//! without having to ask for them.

use super::*;

const CHECK_TIMEOUT_SECS: u64 = 90;
const MAX_CHECK_OUTPUT: usize = 128 * 1024;
const MAX_DIAGNOSTIC_LINES: usize = 40;
const MAX_DIAGNOSTIC_CHARS: usize = 6_000;

/// What to run after file changes, from the `check_after_edits` setting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CheckMode {
    #[default]
    Off,
    /// Detect the project type: `cargo check`, `go build`, or a syntax check
    /// of the changed Python or JavaScript files.
    Auto,
    /// The user's own argv, run in the project folder.
    Custom(Vec<String>),
}

impl CheckMode {
    /// `auto`, `off`/empty, or a JSON string array such as `["make","test"]`.
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        Ok(match value.to_ascii_lowercase().as_str() {
            "" | "off" => Self::Off,
            "auto" => Self::Auto,
            _ => {
                let argv: Vec<String> = serde_json::from_str(value).context(
                    "The check after edits must be auto, off, or a JSON string array such as [\"make\",\"check\"]",
                )?;
                if argv.is_empty() || argv[0].trim().is_empty() {
                    bail!("The custom check command is empty");
                }
                Self::Custom(argv)
            }
        })
    }
}

/// Toolchain caches a check may read (dependencies are already downloaded;
/// the sandbox has no network).
fn toolchain_caches() -> Vec<PathBuf> {
    crate::sandbox::parse_paths("~/.cargo/registry\n~/.cargo/git\n~/go/pkg/mod")
}

fn changed_paths(operation: &Operation) -> Vec<String> {
    match operation {
        Operation::Write { path, .. } | Operation::Replace { path, .. } => vec![path.clone()],
        Operation::Copy { destination, .. } | Operation::Move { destination, .. } => {
            vec![destination.clone()]
        }
        Operation::Delete { path } => vec![path.clone()],
        _ => Vec::new(),
    }
}

/// Keep errors first, then warnings, then anything else, within a budget.
fn summarize(output: &str) -> (String, usize, usize) {
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    let is_error = |l: &str| {
        let lower = l.to_ascii_lowercase();
        lower.contains("error") || lower.contains("syntaxerror")
    };
    let is_warning = |l: &str| l.to_ascii_lowercase().contains("warning");
    let errors = lines.iter().filter(|l| is_error(l)).count();
    let warnings = lines
        .iter()
        .filter(|l| !is_error(l) && is_warning(l))
        .count();
    let mut chosen: Vec<&str> = lines.iter().copied().filter(|l| is_error(l)).collect();
    chosen.extend(
        lines
            .iter()
            .copied()
            .filter(|l| !is_error(l) && is_warning(l)),
    );
    if chosen.is_empty() {
        chosen = lines
            .iter()
            .rev()
            .take(MAX_DIAGNOSTIC_LINES)
            .rev()
            .copied()
            .collect();
    }
    let mut text = String::new();
    for line in chosen.into_iter().take(MAX_DIAGNOSTIC_LINES) {
        if text.len() + line.len() > MAX_DIAGNOSTIC_CHARS {
            text.push_str("…\n");
            break;
        }
        text.push_str(line);
        text.push('\n');
    }
    (text, errors, warnings)
}

impl ToolEngine {
    pub fn set_check_mode(&self, mode: CheckMode) {
        if let Ok(mut current) = self.check_mode.lock() {
            *current = mode;
        }
    }

    /// The argv to run for a change to `changed`, if any applies.
    fn check_command(&self, changed: &[String]) -> Option<(Vec<String>, BTreeMap<String, String>)> {
        let mode = self.check_mode.lock().ok()?.clone();
        let mut env = BTreeMap::new();
        let installed = |program: &str| crate::diagnostics::find_executable(program).is_some();
        let argv: Vec<String> = match mode {
            CheckMode::Off => return None,
            CheckMode::Custom(argv) => argv,
            CheckMode::Auto => {
                let root = self.root();
                let files_with = |extensions: &[&str]| -> Vec<String> {
                    changed
                        .iter()
                        .filter(|p| extensions.iter().any(|e| p.ends_with(e)))
                        .filter(|p| self.workspace.resolve(p, false).is_ok_and(|r| r.is_file()))
                        .cloned()
                        .collect()
                };
                if root.join("Cargo.toml").is_file() && installed("cargo") {
                    env.insert("CARGO_TERM_COLOR".into(), "never".into());
                    [
                        "cargo",
                        "check",
                        "--offline",
                        "--quiet",
                        "--all-targets",
                        "--message-format=short",
                    ]
                    .map(String::from)
                    .to_vec()
                } else if root.join("go.mod").is_file() && installed("go") {
                    env.insert("GOFLAGS".into(), "-mod=mod".into());
                    env.insert("GOPROXY".into(), "off".into());
                    env.insert("GOCACHE".into(), "/tmp/go-build".into());
                    ["go", "build", "./..."].map(String::from).to_vec()
                } else {
                    let python = files_with(&[".py"]);
                    let javascript = files_with(&[".js", ".mjs", ".cjs"]);
                    if !python.is_empty() && installed("python3") {
                        let mut argv: Vec<String> = ["python3", "-I", "-B", "-m", "py_compile"]
                            .map(String::from)
                            .to_vec();
                        argv.extend(python);
                        argv
                    } else if let (Some(file), true) = (javascript.first(), installed("node")) {
                        vec!["node".into(), "--check".into(), file.clone()]
                    } else {
                        return None;
                    }
                }
            }
        };
        Some((argv, env))
    }

    /// Run the configured check after a successful file change and attach its
    /// result. Failures to run the check never turn a successful change into an
    /// error.
    pub(super) fn with_post_change_check(
        &self,
        operation: &Operation,
        result: Result<Value>,
    ) -> Result<Value> {
        let changed = changed_paths(operation);
        let Ok(mut value) = result else {
            return result;
        };
        if changed.is_empty() || !value.is_object() {
            return Ok(value);
        }
        if let Some(check) = self.post_change_check(&changed) {
            value["check"] = check;
        }
        Ok(value)
    }

    fn post_change_check(&self, changed: &[String]) -> Option<Value> {
        let (argv, env) = self.check_command(changed)?;
        // Only ever inside the sandbox: a check runs code (build scripts,
        // macros) from files the model just changed.
        let mut policy = match self.command_sandbox() {
            Ok(Some(policy)) => policy,
            _ => return None,
        };
        policy.network = false;
        policy.read_only.extend(toolchain_caches());
        let spec = CommandSpec {
            program: argv[0].clone(),
            args: argv[1..].to_vec(),
            cwd: Some(self.root().to_path_buf()),
            timeout_secs: CHECK_TIMEOUT_SECS,
            max_output_bytes: MAX_CHECK_OUTPUT,
            env,
            sandbox: Some(policy),
            ..Default::default()
        };
        let command = argv.join(" ");
        Some(match integrations::run_command(&spec, self.root()) {
            Ok(result) => {
                let combined = format!("{}\n{}", result.stdout, result.stderr);
                let (diagnostics, errors, warnings) = summarize(&combined);
                let passed = result.exit_code == Some(0) && !result.timed_out;
                json!({
                    "command": command,
                    "passed": passed,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "error_lines": errors,
                    "warning_lines": warnings,
                    "diagnostics": diagnostics,
                    "note": if passed { "The project check passed after this change." } else { "The project check reported problems after this change; fix them before continuing." },
                })
            }
            Err(error) => {
                json!({"command": command, "passed": null, "error": format!("{error:#}")})
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse() {
        assert_eq!(CheckMode::parse("").unwrap(), CheckMode::Off);
        assert_eq!(CheckMode::parse("Auto").unwrap(), CheckMode::Auto);
        assert_eq!(
            CheckMode::parse(r#"["make","check"]"#).unwrap(),
            CheckMode::Custom(vec!["make".into(), "check".into()])
        );
        assert!(CheckMode::parse("make check").is_err());
        assert!(CheckMode::parse("[]").is_err());
    }

    #[test]
    fn summaries_put_errors_first() {
        let (text, errors, warnings) = summarize(
            "Compiling x\nsrc/a.rs:1:1: warning: unused\nsrc/b.rs:2:2: error[E0425]: missing\n",
        );
        assert_eq!((errors, warnings), (1, 1));
        assert!(text.starts_with("src/b.rs:2:2: error"), "{text}");
    }

    fn approve_write(engine: &ToolEngine, path: &str, content: &str) -> Value {
        let action = match engine
            .prepare("write_file", &json!({"path":path,"content":content}))
            .unwrap()
        {
            PreparedTool::Approval(a) => *a,
            PreparedTool::Immediate(_) => unreachable!(),
        };
        engine.approve(&action).unwrap()
    }

    #[test]
    fn a_python_syntax_error_is_reported_after_the_write() {
        if !crate::sandbox::available() || crate::diagnostics::find_executable("python3").is_none()
        {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let engine = ToolEngine::with_backup_dir(
            Workspace::new(dir.path().to_path_buf()).unwrap(),
            backups.path().to_path_buf(),
        );
        engine.set_sandbox(SandboxSettings::default());
        engine.set_check_mode(CheckMode::Auto);
        let bad = approve_write(&engine, "app.py", "def broken(:\n    pass\n");
        assert_eq!(bad["ok"], true);
        assert_eq!(bad["check"]["passed"], false, "{bad}");
        assert!(
            bad["check"]["diagnostics"]
                .as_str()
                .unwrap()
                .contains("SyntaxError"),
            "{bad}"
        );
        let good = approve_write(&engine, "app.py", "def fixed():\n    pass\n");
        assert_eq!(good["check"]["passed"], true, "{good}");
        // Off, or without the sandbox, nothing runs.
        engine.set_check_mode(CheckMode::Off);
        assert!(approve_write(&engine, "app.py", "x = 1\n")
            .get("check")
            .is_none());
        engine.set_check_mode(CheckMode::Auto);
        engine.set_sandbox(SandboxSettings {
            mode: crate::sandbox::SandboxMode::Off,
            ..Default::default()
        });
        assert!(approve_write(&engine, "app.py", "x = 2\n")
            .get("check")
            .is_none());
    }

    #[test]
    fn a_rust_type_error_is_reported_by_cargo_check() {
        if !crate::sandbox::available() || crate::diagnostics::find_executable("cargo").is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        let engine = ToolEngine::with_backup_dir(
            Workspace::new(dir.path().to_path_buf()).unwrap(),
            backups.path().to_path_buf(),
        );
        engine.set_sandbox(SandboxSettings {
            read_only: crate::sandbox::parse_paths(crate::sandbox::DEFAULT_READ_ONLY_PATHS),
            ..Default::default()
        });
        engine.set_check_mode(CheckMode::Auto);
        let bad = approve_write(&engine, "src/main.rs", "fn main() { let x: u8 = \"no\"; }\n");
        assert_eq!(bad["check"]["passed"], false, "{bad}");
        assert!(bad["check"]["diagnostics"].as_str().unwrap().contains("mismatched types"), "{bad}");
        let good = approve_write(&engine, "src/main.rs", "fn main() { let _x: u8 = 1; }\n");
        assert_eq!(good["check"]["passed"], true, "{good}");
    }
}
