//! Non-destructive readiness inspection. No subprocess, network, model download,
//! authentication request, service registration or filesystem mutation is performed.
use crate::settings::Settings;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    Available,
    Missing,
    Invalid,
    NotChecked,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Check {
    pub feature: String,
    pub state: State,
    pub detail: String,
}

fn executable(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub fn find_executable(program: &str) -> Option<PathBuf> {
    if program.trim().is_empty() || program.contains('\0') {
        return None;
    }
    let path = Path::new(program);
    if path.components().count() > 1 || path.is_absolute() {
        return executable(path).then(|| path.to_path_buf());
    }
    std::env::var_os("PATH").and_then(|value| {
        std::env::split_paths(&value)
            .filter(|p| !p.as_os_str().is_empty())
            .map(|base| base.join(program))
            .find(|path| executable(path))
    })
}

pub fn inspect(settings: &Settings, dir: &Path, workspace: Option<&Path>) -> Vec<Check> {
    let mut checks = Vec::new();
    let mut push = |feature: &str, state: State, detail: String| {
        checks.push(Check {
            feature: feature.into(),
            state,
            detail,
        });
    };
    push(
        "Application build",
        State::Available,
        format!(
            "{}; source {}; {} / {}",
            env!("CARGO_PKG_VERSION"),
            env!("WORKBENCH_SOURCE_ID"),
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
    );
    match settings.validate() {
        Ok(()) => push(
            "Saved configuration",
            State::Available,
            "Syntax and policy limits pass validation. This does not test external services."
                .into(),
        ),
        Err(error) => push("Saved configuration", State::Invalid, format!("{error:#}")),
    }
    push("Ollama inference and model downloads", State::NotChecked,
        format!("Endpoint: {}. Use Check connection, then a real prompt, to test the server and selected model.", settings.base_url));
    push(
        "Application data",
        if dir.is_dir() {
            State::Available
        } else {
            State::Missing
        },
        dir.display().to_string(),
    );
    match workspace {
        Some(root) => match crate::files::Workspace::new(root.to_path_buf()) {
            Ok(ws) => push(
                "Workspace",
                State::Available,
                ws.root().display().to_string(),
            ),
            Err(error) => push("Workspace", State::Invalid, format!("{error:#}")),
        },
        None => push(
            "Workspace",
            State::Missing,
            "Open a project folder before file tools, indexing, or project review.".into(),
        ),
    }
    for (feature, program) in [
        ("PDF text extraction", "pdftotext"),
        ("PDF page rendering", "pdftoppm"),
        ("OCR", "tesseract"),
        ("Audio/video conversion", "ffmpeg"),
        ("Media metadata", "ffprobe"),
        ("LaTeX reports", "pdflatex"),
        ("Chat speech adapter", settings.tts_program.as_str()),
        ("Chat transcription adapter", settings.stt_program.as_str()),
    ] {
        match find_executable(program) {
            Some(path) => push(feature, State::Available, format!("Found {}. Engine version, arguments, voices and weights have not been tested.", path.display())),
            None => push(feature, State::Missing, format!("Executable {:?} not found. Install or configure this optional program to use the feature.", program)),
        }
    }
    push("Studio image and media engines", State::NotChecked, "Studio has its own engine paths and endpoints. Review Studio configuration and run an activity; model/voice weights are separate installations.".into());
    let mcp_servers = serde_json::from_str::<Vec<crate::mcp::McpServerConfig>>(&settings.mcp_json);
    let mapped_mcp = mcp_servers.as_ref().ok().and_then(|servers| {
        servers
            .iter()
            .find(|server| server.enabled && server.web_search.is_some())
    });
    if !settings.network_tools {
        push(
            "Web search",
            State::Missing,
            "Network tools are disabled. Enable them before any search query can be proposed."
                .into(),
        );
    } else if let Some(server) = mapped_mcp {
        if !settings.mcp_enabled {
            push(
                "Web search",
                State::Missing,
                format!(
                    "Mapped MCP provider {} is enabled in JSON, but MCP servers are globally disabled.",
                    server.name
                ),
            );
        } else if let Some(path) = find_executable(&server.program) {
            push(
                "Web search",
                State::NotChecked,
                format!(
                    "MCP provider {} is configured and {} was found. Diagnostics did not launch it or send a query.",
                    server.name,
                    path.display()
                ),
            );
        } else {
            push(
                "Web search",
                State::Missing,
                format!(
                    "MCP provider {} is configured, but executable {:?} was not found on PATH.",
                    server.name, server.program
                ),
            );
        }
    } else if !settings.searxng_url.trim().is_empty() {
        push(
            "Web search",
            State::NotChecked,
            format!(
                "SearXNG fallback {} is configured. Diagnostics did not contact it.",
                settings.searxng_url
            ),
        );
    } else if let Some(browser) = crate::integrations::resolve_text_browser(&settings.text_browser)
    {
        push(
            "Web search",
            State::NotChecked,
            format!(
                "Text browser {} ({}) will query DuckDuckGo. Diagnostics did not send a query.",
                browser.name,
                browser.program.display()
            ),
        );
    } else {
        push(
            "Web search",
            State::Missing,
            "No MCP web-search mapping, SearXNG URL or installed text browser (lynx, w3m, elinks, links) is available.".into(),
        );
    }
    push("MCP servers", State::NotChecked, "Only an explicitly approved connection starts a configured server. No server was started by diagnostics.".into());
    push("Scheduler and managed Ollama", State::NotChecked, "Inspect Services for actual process state. Diagnostics does not start, stop, install or enable a service.".into());
    push("External agent CLIs", State::NotChecked, "External Agents requires installed/authenticated CLI programs, explicit network consent where applicable, and review of each launch.".into());
    checks
}

pub fn report(checks: &[Check]) -> String {
    let mut text = String::from(
        "AI Workbench readiness report\nExecutable discovery is not an end-to-end engine test.\n\n",
    );
    for check in checks {
        text.push_str(&format!(
            "[{:?}] {}: {}\n",
            check.state, check.feature, check.detail
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_program_does_not_report_available() {
        assert!(find_executable("workbench-deliberately-missing-executable-7812").is_none());
        assert!(find_executable("").is_none());
        let dir = tempfile::tempdir().unwrap();
        assert!(find_executable(dir.path().to_str().unwrap()).is_none());
    }
    #[test]
    fn diagnostic_inspection_does_not_create_data_or_claim_inference() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("absent-data");
        let checks = inspect(&Settings::default(), &dir, None);
        assert!(!dir.exists());
        assert!(checks
            .iter()
            .any(|c| c.feature.contains("Ollama inference") && c.state == State::NotChecked));
        assert!(report(&checks).contains("not an end-to-end"));
    }

    #[test]
    fn web_search_diagnostics_report_configuration_without_launching_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = Settings {
            network_tools: true,
            mcp_enabled: true,
            ..Settings::default()
        };
        let mut server = crate::mcp::McpServerConfig::brave_search_preset();
        server.id = "fixture-search".into();
        server.name = "Fixture Search".into();
        server.program = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        server.args.clear();
        server.env.clear();
        server.enabled = true;
        settings.mcp_json = serde_json::to_string(&[server]).unwrap();
        let checks = inspect(&settings, dir.path(), None);
        let search = checks
            .iter()
            .find(|check| check.feature == "Web search")
            .unwrap();
        assert_eq!(search.state, State::NotChecked);
        assert!(search.detail.contains("did not launch"));
    }
}
