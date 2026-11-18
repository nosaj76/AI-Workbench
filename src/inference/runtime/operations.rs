//! One accountable record per tool call: what the model requested, whether it was
//! authorized, whether it ran, and, for file changes, whether the change is really
//! on disk afterwards. A model's statement that it did something is never the proof;
//! this record and the filesystem check are.

use serde_json::{json, Map, Value};
use std::path::{Component, Path};

/// How the request was authorized before it ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Authorization {
    /// Read-only tools run without asking.
    NotRequired,
    Approved,
    Denied,
    /// Refused by a turn policy (wrong scope, guessed URL, provider block).
    RefusedByPolicy,
    /// An identical request already ran in this turn.
    Duplicate,
}

impl Authorization {
    fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::RefusedByPolicy => "refused_by_policy",
            Self::Duplicate => "duplicate",
        }
    }
}

/// Builds the record for a finished tool call. For a successful file change the
/// workspace is checked; if the change is not there, `result` is marked failed so
/// neither the model nor the user can treat it as done.
pub(super) fn finish(
    tool: &str,
    model: &str,
    arguments: &Value,
    authorization: Authorization,
    root: Option<&Path>,
    result: &mut Value,
) -> Value {
    let mut stages = vec!["requested", "validated"];
    let state;
    let mut verification = Value::Null;
    if result.get("cancelled").and_then(Value::as_bool) == Some(true) {
        state = "cancelled";
    } else if authorization == Authorization::Duplicate {
        state = "skipped_duplicate";
    } else if authorization == Authorization::RefusedByPolicy {
        state = "refused";
    } else if authorization == Authorization::Denied
        || result.get("denied").and_then(Value::as_bool) == Some(true)
    {
        state = "denied";
    } else {
        stages.push("authorized");
        stages.push("executing");
        if super::turn_helpers::tool_succeeded(result) {
            stages.push("executed");
            match verify(tool, arguments, root) {
                Some(Ok(evidence)) => {
                    stages.push("verified");
                    state = "verified";
                    verification = json!({"passed": true, "evidence": evidence});
                }
                Some(Err(problem)) => {
                    state = "verification_failed";
                    verification = json!({"passed": false, "problem": problem});
                    if let Some(object) = result.as_object_mut() {
                        object.insert("ok".into(), json!(false));
                        object.insert(
                            "verification_error".into(),
                            json!(format!("The tool reported success, but Workbench could not confirm the change on disk: {problem}. Do not report this operation as completed.")),
                        );
                    }
                }
                None => state = "executed",
            }
        } else {
            state = "failed";
        }
    }
    let record = json!({
        "tool": tool,
        "requested_by": model,
        "arguments": redact(arguments),
        "authorization": authorization.as_str(),
        "stages": stages,
        "state": state,
        "verification": verification,
        "error": result.get("error").cloned().unwrap_or(Value::Null),
    });
    if let Some(object) = result.as_object_mut() {
        object.insert(
            "operation".into(),
            json!({"state": state, "verified": state == "verified"}),
        );
    }
    record
}

/// Checks that a reported file change is really present. None when the tool does
/// not change files (or there is no workspace to look in).
fn verify(tool: &str, arguments: &Value, root: Option<&Path>) -> Option<Result<String, String>> {
    let root = root?;
    let at = |key: &str| -> Result<std::path::PathBuf, String> {
        let relative = arguments
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("the request has no {key}"))?;
        let path = Path::new(relative);
        if path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(format!("{relative} is not a workspace-relative path"));
        }
        if path.is_absolute() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| format!("{relative} is outside the connected workspace"))?;
            Ok(root.join(relative))
        } else {
            Ok(root.join(path))
        }
    };
    let check = || -> Result<String, String> {
        match tool {
            "write_file" => {
                let path = at("path")?;
                let actual = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{} cannot be read back: {e}", path.display()))?;
                let wanted = arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if actual == wanted {
                    Ok(format!(
                        "read back {} bytes; content matches the request",
                        actual.len()
                    ))
                } else {
                    Err("the file content on disk differs from the requested content".into())
                }
            }
            "replace_text" => {
                let path = at("path")?;
                let actual = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{} cannot be read back: {e}", path.display()))?;
                let replacement = arguments
                    .get("replacement")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                // Replacements are matched tolerantly (indentation, line endings), so
                // compare the trimmed replacement lines rather than raw bytes.
                let missing = replacement
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .find(|line| !actual.contains(line));
                match missing {
                    None => Ok("the replacement text is present in the file".into()),
                    Some(line) => Err(format!("the replacement line `{line}` is not in the file")),
                }
            }
            "copy_file" => {
                let destination = at("destination")?;
                if destination.is_file() {
                    Ok(format!("{} exists", destination.display()))
                } else {
                    Err(format!("{} does not exist", destination.display()))
                }
            }
            "move_file" => {
                let (source, destination) = (at("source")?, at("destination")?);
                if !destination.is_file() {
                    Err(format!("{} does not exist", destination.display()))
                } else if source.exists() && source != destination {
                    Err(format!("{} still exists", source.display()))
                } else {
                    Ok(format!(
                        "{} exists and the source is gone",
                        destination.display()
                    ))
                }
            }
            "delete_file" => {
                let path = at("path")?;
                if path.exists() {
                    Err(format!("{} still exists", path.display()))
                } else {
                    Ok(format!("{} is gone", path.display()))
                }
            }
            "make_directory" => {
                let path = at("path")?;
                if path.is_dir() {
                    Ok(format!("{} is a directory", path.display()))
                } else {
                    Err(format!("{} is not a directory", path.display()))
                }
            }
            _ => unreachable!(),
        }
    };
    matches!(
        tool,
        "write_file"
            | "replace_text"
            | "copy_file"
            | "move_file"
            | "delete_file"
            | "make_directory"
    )
    .then(check)
}

/// Arguments for the record: secrets are hidden and long text is summarized (the
/// full request stays in the saved conversation).
fn redact(arguments: &Value) -> Value {
    const SECRET: [&str; 7] = [
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
    ];
    match arguments {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                let lower = key.to_ascii_lowercase();
                let shown = if SECRET.iter().any(|s| lower.contains(s)) {
                    json!("[redacted]")
                } else {
                    redact(value)
                };
                out.insert(key.clone(), shown);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        Value::String(text) if text.chars().count() > 300 => json!(format!(
            "{}… [{} characters in total; full text in the conversation]",
            text.chars().take(120).collect::<String>(),
            text.chars().count()
        )),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_that_is_really_on_disk_is_verified() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let mut result = json!({"ok": true});
        let record = finish(
            "write_file",
            "m",
            &json!({"path": "a.txt", "content": "hello"}),
            Authorization::Approved,
            Some(dir.path()),
            &mut result,
        );
        assert_eq!(record["state"], "verified");
        assert_eq!(result["operation"]["verified"], true);
    }

    #[test]
    fn a_reported_write_that_is_not_on_disk_fails_verification_and_the_result() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "old").unwrap();
        let mut result = json!({"ok": true});
        let record = finish(
            "write_file",
            "m",
            &json!({"path": "a.txt", "content": "new"}),
            Authorization::Approved,
            Some(dir.path()),
            &mut result,
        );
        assert_eq!(record["state"], "verification_failed");
        assert_eq!(result["ok"], false);
        assert!(!super::super::turn_helpers::tool_succeeded(&result));
    }

    #[test]
    fn denied_refused_duplicate_and_read_only_calls_get_distinct_states() {
        let mut denied = json!({"denied": true});
        assert_eq!(
            finish(
                "write_file",
                "m",
                &json!({}),
                Authorization::Denied,
                None,
                &mut denied
            )["state"],
            "denied"
        );
        let mut refused = json!({"denied": true, "executed": false});
        assert_eq!(
            finish(
                "fetch_url",
                "m",
                &json!({}),
                Authorization::RefusedByPolicy,
                None,
                &mut refused
            )["state"],
            "refused"
        );
        let mut duplicate = json!({"duplicate": true, "executed": false});
        assert_eq!(
            finish(
                "read_file",
                "m",
                &json!({}),
                Authorization::Duplicate,
                None,
                &mut duplicate
            )["state"],
            "skipped_duplicate"
        );
        let mut read = json!({"content": "x"});
        assert_eq!(
            finish(
                "read_file",
                "m",
                &json!({}),
                Authorization::NotRequired,
                None,
                &mut read
            )["state"],
            "executed"
        );
        let mut failed = json!({"error": "boom"});
        assert_eq!(
            finish(
                "read_file",
                "m",
                &json!({}),
                Authorization::NotRequired,
                None,
                &mut failed
            )["state"],
            "failed"
        );
    }

    #[test]
    fn secrets_are_redacted_and_long_text_is_summarized() {
        let long = "x".repeat(1000);
        let shown = redact(&json!({"api_key": "abc", "content": long, "path": "a"}));
        assert_eq!(shown["api_key"], "[redacted]");
        assert!(shown["content"]
            .as_str()
            .unwrap()
            .contains("1000 characters"));
        assert_eq!(shown["path"], "a");
    }
}

#[cfg(test)]
mod absolute_path_regressions {
    use super::*;
    #[test]
    fn verification_accepts_in_scope_absolute_paths_and_rejects_escape() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("ok.txt");
        std::fs::write(&file, "ok").unwrap();
        assert!(verify(
            "write_file",
            &json!({"path":file,"content":"ok"}),
            Some(root.path())
        )
        .unwrap()
        .is_ok());
        assert!(verify(
            "write_file",
            &json!({"path":"/outside/ok.txt","content":"ok"}),
            Some(root.path())
        )
        .unwrap()
        .is_err());
        assert!(verify(
            "write_file",
            &json!({"path":"../ok.txt","content":"ok"}),
            Some(root.path())
        )
        .unwrap()
        .is_err());
    }
}
