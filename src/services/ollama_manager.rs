//! User-scoped management of the official Ollama Linux distribution. Downloads
//! are verified before extraction; installation never runs a downloaded script
//! or overwrites an administrator-managed Ollama installation.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const RELEASE_API: &str = "https://api.github.com/repos/ollama/ollama/releases/latest";
const MAX_ARCHIVE: u64 = 16 * 1024 * 1024 * 1024;
const MAX_EXPANDED: u64 = 48 * 1024 * 1024 * 1024;
const MAX_FILES: usize = 100_000;
const MAX_METADATA: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseInfo {
    pub version: String,
    pub published_at: String,
    pub url: String,
    pub notes: String,
    pub assets: Vec<ReleaseAsset>,
}
impl ReleaseInfo {
    pub fn linux_asset(&self) -> Result<&ReleaseAsset> {
        let architecture = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            other => bail!("Managed Ollama packages are unavailable for {other}"),
        };
        self.assets
            .iter()
            .find(|a| a.name == format!("ollama-linux-{architecture}.tar.zst"))
            .or_else(|| {
                self.assets
                    .iter()
                    .find(|a| a.name == format!("ollama-linux-{architecture}.tgz"))
            })
            .context("Official release does not contain a supported Linux archive")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ActiveVersions {
    current: Option<String>,
    previous: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedProcess {
    pub pid: u32,
    pub start_ticks: u64,
    pub version: String,
    pub port: u16,
    pub models: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStatus {
    pub current_version: Option<String>,
    pub previous_version: Option<String>,
    pub binary: Option<PathBuf>,
    pub running: bool,
    pub process: Option<ManagedProcess>,
    pub log_path: PathBuf,
    pub models_directory: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ManagerProgress {
    pub stage: String,
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct OllamaManager {
    root: PathBuf,
}

impl OllamaManager {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let root = data_dir.join("managed-ollama");
        if fs::symlink_metadata(&root).is_ok_and(|m| m.file_type().is_symlink()) {
            bail!("Managed runtime directory must not be a symlink");
        }
        fs::create_dir_all(&root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        }
        let root = root.canonicalize()?;
        for name in ["releases", "downloads", "models"] {
            let path = root.join(name);
            if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
                bail!("Managed runtime subdirectory must not be a symlink");
            }
            fs::create_dir_all(path)?;
        }
        Ok(Self { root })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn latest_release(&self) -> Result<ReleaseInfo> {
        let bytes = read_https(RELEASE_API, MAX_METADATA)?;
        parse_release(&serde_json::from_slice(&bytes).context("Invalid official release metadata")?)
    }
    pub fn release(&self, version: &str) -> Result<ReleaseInfo> {
        validate_version(version)?;
        let url = format!("https://api.github.com/repos/ollama/ollama/releases/tags/{version}");
        parse_release(&serde_json::from_slice(&read_https(&url, MAX_METADATA)?)?)
    }
    pub fn status(&self) -> Result<ManagerStatus> {
        let active = self.active()?;
        let process = self.process()?;
        let running = process.as_ref().is_some_and(|p| self.process_matches(p));
        let binary = active
            .current
            .as_ref()
            .map(|v| self.binary_for(v))
            .transpose()?;
        Ok(ManagerStatus {
            current_version: active.current,
            previous_version: active.previous,
            binary,
            running,
            process: if running { process } else { None },
            log_path: self.root.join("server.log"),
            models_directory: self.root.join("models"),
        })
    }
    /// The worker must show ReleaseInfo/version/asset size to the user before
    /// calling this explicit install operation. It changes the selected managed
    /// binary atomically but leaves any currently running server undisturbed.
    pub fn install_release<F>(
        &self,
        release: &ReleaseInfo,
        cancel: Arc<AtomicBool>,
        mut progress: F,
    ) -> Result<ManagerStatus>
    where
        F: FnMut(ManagerProgress),
    {
        let _lock = ManagerLock::acquire(&self.root)?;
        if cancel.load(Ordering::Relaxed) {
            bail!("Runtime installation cancelled");
        }
        validate_version(&release.version)?;
        let asset = release.linux_asset()?;
        validate_asset_url(&asset.url)?;
        if asset.size == 0 || asset.size > MAX_ARCHIVE {
            bail!("Official asset size is unavailable or exceeds 16 GiB");
        }
        let checksum = if let Some(digest) = &asset.sha256 {
            validate_digest(digest)?;
            digest.to_ascii_lowercase()
        } else {
            let checksums=release.assets.iter().find(|a|matches!(a.name.as_str(),"sha256sum.txt"|"sha256sums.txt"|"SHA256SUMS"|"checksums.txt")).context("Release provides no SHA-256 digest or checksum manifest; no unverified executable was installed")?;
            validate_asset_url(&checksums.url)?;
            checksum_from_manifest(
                &String::from_utf8(read_https(&checksums.url, 2 * 1024 * 1024)?)?,
                &asset.name,
            )?
        };
        if cancel.load(Ordering::Relaxed) {
            bail!("Runtime installation cancelled before activation");
        }
        let existing = self.root.join("releases").join(&release.version);
        if existing.exists() {
            let manifest = existing.join("workbench-install.json");
            let data: Value = serde_json::from_slice(
                &read_small_file(&manifest, 64_000)
                    .context("An unrecognized directory already exists for that version")?,
            )?;
            if data.get("sha256").and_then(Value::as_str) != Some(checksum.as_str()) {
                bail!("Existing managed version has a different checksum; it was preserved");
            }
            self.activate(&release.version)?;
            return self.status();
        }
        let download =
            self.root
                .join("downloads")
                .join(format!("{}-{}", uuid::Uuid::new_v4(), asset.name));
        let stage = self
            .root
            .join("releases")
            .join(format!(".install-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&stage)?;
        let result = (|| -> Result<()> {
            progress(ManagerProgress {
                stage: format!("Downloading {}", asset.name),
                completed: 0,
                total: Some(asset.size),
            });
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(download_verified(
                asset,
                &download,
                &checksum,
                cancel.clone(),
                |done| {
                    progress(ManagerProgress {
                        stage: "Downloading verified official package".into(),
                        completed: done,
                        total: Some(asset.size),
                    })
                },
            ))?;
            if cancel.load(Ordering::Relaxed) {
                bail!("Runtime installation cancelled before extraction");
            }
            progress(ManagerProgress {
                stage: "Extracting into an isolated staging directory".into(),
                completed: 0,
                total: None,
            });
            extract_archive(&download, &stage, cancel.as_ref())?;
            verify_linux_binary(&stage.join("bin/ollama"))?;
            if cancel.load(Ordering::Relaxed) {
                bail!("Runtime installation cancelled before activation");
            }
            crate::scheduler::atomic_write(
                &stage.join("workbench-install.json"),
                &serde_json::to_vec_pretty(
                    &serde_json::json!({"version":release.version,"asset":asset.name,"source_url":asset.url,"sha256":checksum,"installed_at":chrono::Utc::now().to_rfc3339()}),
                )?,
            )?;
            fs::rename(&stage, &existing)?;
            self.activate(&release.version)?;
            progress(ManagerProgress {
                stage: format!(
                    "Installed {}. Existing running servers were not interrupted.",
                    release.version
                ),
                completed: asset.size,
                total: Some(asset.size),
            });
            Ok(())
        })();
        let _ = fs::remove_file(&download);
        if result.is_err() {
            let _ = fs::remove_dir_all(&stage);
        }
        result?;
        self.status()
    }
    pub fn rollback(&self) -> Result<ManagerStatus> {
        let _lock = ManagerLock::acquire(&self.root)?;
        let status = self.status()?;
        if status.running {
            bail!("Stop the managed server before rolling back its selected runtime");
        }
        let mut active = self.active()?;
        let previous = active
            .previous
            .clone()
            .context("No previous managed runtime is available")?;
        verify_linux_binary(&self.binary_for(&previous)?)?;
        std::mem::swap(&mut active.current, &mut active.previous);
        crate::scheduler::atomic_write(
            &self.root.join("active.json"),
            &serde_json::to_vec_pretty(&active)?,
        )?;
        self.status()
    }
    pub fn start(&self, port: u16) -> Result<ManagerStatus> {
        self.start_with_models(port, None)
    }
    pub fn start_with_models(&self, port: u16, models: Option<&Path>) -> Result<ManagerStatus> {
        let _lock = ManagerLock::acquire(&self.root)?;
        self.start_unlocked(port, models)
    }
    fn start_unlocked(&self, port: u16, models: Option<&Path>) -> Result<ManagerStatus> {
        if port < 1024 {
            bail!("Choose a non-privileged local port (1024–65535)");
        }
        let status = self.status()?;
        if status.running {
            bail!("The managed Ollama server is already running");
        }
        let version = status
            .current_version
            .context("Install a managed Ollama runtime first")?;
        let binary = self.binary_for(&version)?;
        verify_linux_binary(&binary)?;
        let socket =
            TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).with_context(|| {
                format!("Port {port} is already in use; the existing service was left alone")
            })?;
        drop(socket);
        let models = models
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.root.join("models"));
        fs::create_dir_all(&models)?;
        let models = models.canonicalize()?;
        let mut log_options = OpenOptions::new();
        log_options.append(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            log_options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let log = log_options.open(self.root.join("server.log"))?;
        let mut command = Command::new(&binary);
        command
            .arg("serve")
            .current_dir(&self.root)
            .env("OLLAMA_HOST", format!("127.0.0.1:{port}"))
            .env("OLLAMA_MODELS", &models)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        crate::integrations::configure_process_group(&mut command);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(600))
            .no_proxy()
            .build()?;
        let mut child = command.spawn().context("Start managed Ollama server")?;
        let start_ticks = match process_start_ticks(child.id()) {
            Ok(ticks) => ticks,
            Err(error) => {
                crate::integrations::kill_process_group(&mut child);
                let _ = child.wait();
                return Err(error).context("Identify managed server process");
            }
        };
        let process = ManagedProcess {
            pid: child.id(),
            start_ticks,
            version,
            port,
            models,
        };
        if let Err(error) = crate::scheduler::atomic_write(
            &self.root.join("process.json"),
            &serde_json::to_vec_pretty(&process)?,
        ) {
            crate::integrations::kill_process_group(&mut child);
            let _ = child.wait();
            return Err(error);
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut ready = false;
        while Instant::now() < deadline {
            if let Some(exit) = child.try_wait()? {
                let _ = fs::remove_file(self.root.join("process.json"));
                bail!("Managed Ollama exited during startup ({exit}); inspect its log");
            }
            if client
                .get(format!("http://127.0.0.1:{port}/api/version"))
                .send()
                .ok()
                .filter(|r| r.status().is_success())
                .and_then(|r| r.json::<Value>().ok())
                .is_some_and(|v| {
                    v.get("version")
                        .and_then(Value::as_str)
                        .is_some_and(|version| {
                            version.trim_start_matches('v')
                                == process.version.trim_start_matches('v')
                        })
                })
            {
                ready = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if !ready {
            crate::integrations::kill_process_group(&mut child);
            let _ = child.wait();
            let _ = fs::remove_file(self.root.join("process.json"));
            bail!("Managed Ollama did not become ready within 20 seconds; inspect its log");
        }
        // Reap normal exits while the GUI is alive. Dropping the application
        // leaves this separately grouped server running for background jobs.
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        self.status()
    }
    pub fn stop(&self) -> Result<ManagerStatus> {
        let _lock = ManagerLock::acquire(&self.root)?;
        self.stop_unlocked()
    }
    fn stop_unlocked(&self) -> Result<ManagerStatus> {
        let Some(process) = self.process()? else {
            return self.status();
        };
        if !self.process_matches(&process) {
            let _ = fs::remove_file(self.root.join("process.json"));
            return self.status();
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(-(process.pid as i32), libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            bail!("Managed process control is currently implemented for Unix");
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && self.process_matches(&process) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if self.process_matches(&process) {
            #[cfg(unix)]
            unsafe {
                libc::kill(-(process.pid as i32), libc::SIGKILL);
            }
        }
        let _ = fs::remove_file(self.root.join("process.json"));
        self.status()
    }
    pub fn restart(&self) -> Result<ManagerStatus> {
        let _lock = ManagerLock::acquire(&self.root)?;
        let status = self.status()?;
        let process = status.process.context("Managed server is not running")?;
        self.stop_unlocked()?;
        self.start_unlocked(process.port, Some(&process.models))
    }
    pub fn logs(&self, max_bytes: usize) -> Result<String> {
        use std::io::{Seek, SeekFrom};
        let path = self.root.join("server.log");
        if !path.exists() {
            return Ok(String::new());
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(path)?;
        let size = file.metadata()?.len();
        let count = (max_bytes.min(2 * 1024 * 1024) as u64).min(size);
        file.seek(SeekFrom::Start(size - count))?;
        let mut bytes = Vec::new();
        file.take(count).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    pub fn installed_versions(&self) -> Result<Vec<String>> {
        let mut versions = Vec::new();
        for entry in fs::read_dir(self.root.join("releases"))? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type()?.is_dir()
                && validate_version(&name).is_ok()
                && entry.path().join("workbench-install.json").is_file()
            {
                versions.push(name);
            }
        }
        versions.sort();
        Ok(versions)
    }
    fn active(&self) -> Result<ActiveVersions> {
        let path = self.root.join("active.json");
        if !path.exists() {
            return Ok(ActiveVersions::default());
        }
        let data = read_small_file(&path, 64_000)?;
        let active: ActiveVersions = serde_json::from_slice(&data)?;
        for version in [active.current.as_deref(), active.previous.as_deref()]
            .into_iter()
            .flatten()
        {
            validate_version(version)?;
        }
        Ok(active)
    }
    fn activate(&self, version: &str) -> Result<()> {
        validate_version(version)?;
        verify_linux_binary(&self.binary_for(version)?)?;
        let mut active = self.active()?;
        if active.current.as_deref() != Some(version) {
            active.previous = active.current;
            active.current = Some(version.into());
            crate::scheduler::atomic_write(
                &self.root.join("active.json"),
                &serde_json::to_vec_pretty(&active)?,
            )?;
        }
        Ok(())
    }
    fn binary_for(&self, version: &str) -> Result<PathBuf> {
        validate_version(version)?;
        Ok(self.root.join("releases").join(version).join("bin/ollama"))
    }
    fn process(&self) -> Result<Option<ManagedProcess>> {
        let path = self.root.join("process.json");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&read_small_file(
            &path, 64_000,
        )?)?))
    }
    fn process_matches(&self, process: &ManagedProcess) -> bool {
        if process.pid < 2
            || validate_version(&process.version).is_err()
            || process_start_ticks(process.pid).ok() != Some(process.start_ticks)
        {
            return false;
        }
        let Ok(binary) = self
            .binary_for(&process.version)
            .and_then(|p| p.canonicalize().map_err(Into::into))
        else {
            return false;
        };
        #[cfg(target_os = "linux")]
        {
            fs::read_link(format!("/proc/{}/exe", process.pid)).is_ok_and(|actual| actual == binary)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = binary;
            false
        }
    }
}

/// Serialize state-changing operations across separate desktop processes.
/// Read-only status and logs remain available while a package downloads.
struct ManagerLock(File);
impl ManagerLock {
    fn acquire(root: &Path) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let file = options.open(root.join("operation.lock"))?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                bail!("Another managed-runtime operation is active; wait for it to finish");
            }
        }
        Ok(Self(file))
    }
}
impl Drop for ManagerLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

fn parse_release(value: &Value) -> Result<ReleaseInfo> {
    let version = value
        .get("tag_name")
        .and_then(Value::as_str)
        .context("Release has no version tag")?
        .to_string();
    validate_version(&version)?;
    if value.get("draft").and_then(Value::as_bool) == Some(true) {
        bail!("Draft releases cannot be installed");
    }
    let mut assets = Vec::new();
    for asset in value
        .get("assets")
        .and_then(Value::as_array)
        .context("Release has no asset list")?
    {
        let name = asset
            .get("name")
            .and_then(Value::as_str)
            .context("Release asset has no name")?;
        if name.len() > 256 || name.contains(['/', '\\']) {
            bail!("Invalid release asset name");
        }
        let url = asset
            .get("browser_download_url")
            .and_then(Value::as_str)
            .context("Release asset has no download URL")?;
        validate_asset_url(url)?;
        let sha256 = asset
            .get("digest")
            .and_then(Value::as_str)
            .and_then(|s| s.strip_prefix("sha256:"))
            .map(String::from);
        if let Some(digest) = &sha256 {
            validate_digest(digest)?;
        }
        assets.push(ReleaseAsset {
            name: name.into(),
            url: url.into(),
            size: asset.get("size").and_then(Value::as_u64).unwrap_or(0),
            sha256,
        });
    }
    Ok(ReleaseInfo {
        version,
        published_at: value
            .get("published_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        url: value
            .get("html_url")
            .and_then(Value::as_str)
            .unwrap_or("https://github.com/ollama/ollama/releases")
            .into(),
        notes: value
            .get("body")
            .and_then(Value::as_str)
            .unwrap_or("")
            .chars()
            .take(100_000)
            .collect(),
        assets,
    })
}

fn validate_version(value: &str) -> Result<()> {
    if value.len() > 80
        || !value.starts_with('v')
        || !value[1..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        || value.len() < 2
        || value.contains("..")
    {
        bail!("Invalid official release version");
    }
    Ok(())
}
fn validate_digest(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("Release SHA-256 digest is invalid");
    }
    Ok(())
}
fn validate_asset_url(value: &str) -> Result<()> {
    let url = url::Url::parse(value)?;
    if url.scheme() != "https"
        || url.host_str() != Some("github.com")
        || !url.path().starts_with("/ollama/ollama/releases/download/")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        bail!("Runtime assets must come from the official Ollama GitHub releases");
    }
    Ok(())
}
fn https_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("Ollama-Workbench/0.2 native-runtime-manager")
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(45))
        .https_only(true)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?)
}
fn read_https(url: &str, max: usize) -> Result<Vec<u8>> {
    let response = https_client()?.get(url).send()?.error_for_status()?;
    if response.content_length().is_some_and(|n| n > max as u64) {
        bail!("Remote metadata exceeds the size limit");
    }
    let mut bytes = Vec::new();
    response.take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        bail!("Remote metadata exceeds the size limit");
    }
    Ok(bytes)
}

fn checksum_from_manifest(text: &str, name: &str) -> Result<String> {
    for line in text.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() == 2 && parts[1].trim_start_matches('*') == name {
            validate_digest(parts[0])?;
            return Ok(parts[0].to_ascii_lowercase());
        }
    }
    bail!("Official checksum manifest does not list {name}")
}

async fn download_verified<F>(
    asset: &ReleaseAsset,
    path: &Path,
    expected: &str,
    cancel: Arc<AtomicBool>,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(u64),
{
    validate_asset_url(&asset.url)?;
    validate_digest(expected)?;
    let client = reqwest::Client::builder()
        .user_agent("Ollama-Workbench/0.2 native-runtime-manager")
        .connect_timeout(Duration::from_secs(15))
        .https_only(true)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?;
    let response = cancellable(client.get(&asset.url).send(), cancel.as_ref())
        .await??
        .error_for_status()?;
    if response.content_length().is_some_and(|n| n != asset.size) {
        bail!("Download size differs from official release metadata");
    }
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut stream = response.bytes_stream();
    let mut last = Instant::now();
    while let Some(chunk) = cancellable(stream.next(), cancel.as_ref()).await? {
        let chunk = chunk?;
        size = size
            .checked_add(chunk.len() as u64)
            .context("Download size overflow")?;
        if size > asset.size || size > MAX_ARCHIVE {
            bail!("Download exceeded its declared size");
        }
        digest.update(&chunk);
        file.write_all(&chunk)?;
        if last.elapsed() > Duration::from_millis(100) {
            progress(size);
            last = Instant::now();
        }
    }
    if size != asset.size {
        bail!("Official runtime download was incomplete");
    }
    let actual = format!("{:x}", digest.finalize());
    if !actual.eq_ignore_ascii_case(expected) {
        bail!("SHA-256 verification failed; the downloaded executable was not installed");
    }
    file.sync_all()?;
    progress(size);
    Ok(())
}

async fn cancellable<T>(
    future: impl std::future::Future<Output = T>,
    cancel: &AtomicBool,
) -> Result<T> {
    let future = tokio::time::timeout(Duration::from_secs(60), future);
    tokio::pin!(future);
    loop {
        if cancel.load(Ordering::Relaxed) {
            bail!("Runtime download cancelled");
        }
        tokio::select! {result=&mut future=>return result.context("Runtime download made no progress for 60 seconds"),_=tokio::time::sleep(Duration::from_millis(100))=>{}}
    }
}

/// Parse archive entries manually: no extraction of devices, FIFOs, absolute or
/// escaping paths. Relative internal symlinks are deferred until all regular
/// files exist, then resolved and confined to the new installation root.
pub fn extract_archive(path: &Path, destination: &Path, cancel: &AtomicBool) -> Result<()> {
    if fs::symlink_metadata(destination).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        bail!("Runtime extraction destination must not be a symlink");
    }
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_ARCHIVE {
        bail!("Runtime archive exceeds 16 GiB");
    }
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let reader: Box<dyn Read> = if name.ends_with(".tar.zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else if name.ends_with(".tgz") || name.ends_with(".tar.gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else if name.ends_with(".tar") {
        Box::new(file)
    } else {
        bail!("Unsupported runtime archive format")
    };
    let mut archive = tar::Archive::new(reader);
    fs::create_dir_all(destination)?;
    let root = destination.canonicalize()?;
    let mut names = HashSet::new();
    let mut links = Vec::new();
    let mut expanded = 0u64;
    for (index, entry) in archive.entries()?.enumerate() {
        if cancel.load(Ordering::Relaxed) {
            bail!("Runtime extraction cancelled");
        }
        if index >= MAX_FILES {
            bail!("Runtime archive has too many entries");
        }
        let mut entry = entry?;
        let relative = clean_relative(&entry.path()?)?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        let kind = entry.header().entry_type();
        let path = root.join(&relative);
        if kind.is_dir() {
            create_safe_directory(&root, &relative)?;
            continue;
        }
        if !names.insert(relative.clone()) {
            bail!("Duplicate runtime archive entry: {}", relative.display());
        }
        if kind.is_symlink() || kind.is_hard_link() {
            let target = entry
                .link_name()?
                .context("Runtime archive link has no target")?
                .into_owned();
            links.push((relative, target, kind.is_hard_link()));
            continue;
        }
        if !kind.is_file() {
            bail!(
                "Runtime archive contains a special file: {}",
                relative.display()
            );
        }
        let size = entry.size();
        expanded = expanded
            .checked_add(size)
            .context("Expanded archive size overflow")?;
        if size > MAX_ARCHIVE || expanded > MAX_EXPANDED {
            bail!("Runtime archive expands beyond its configured limit");
        }
        if let Some(parent) = relative.parent() {
            create_safe_directory(&root, parent)?;
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut output = options.open(&path)?;
        let mut buffer = [0u8; 64 * 1024];
        let mut written = 0u64;
        loop {
            if cancel.load(Ordering::Relaxed) {
                bail!("Runtime extraction cancelled");
            }
            let n = entry.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            written += n as u64;
            if written > size {
                bail!("Runtime archive entry exceeds its declared size");
            }
            output.write_all(&buffer[..n])?;
        }
        if written != size {
            bail!("Truncated runtime archive entry");
        }
        output.sync_all()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = entry.header().mode().unwrap_or(0o644);
            fs::set_permissions(
                &path,
                fs::Permissions::from_mode(if mode & 0o111 != 0 { 0o755 } else { 0o644 }),
            )?;
        }
    }
    // Link chains are permitted only if each target resolves to an existing
    // in-root entry. Cycles and dangling/outside links fail the staged install.
    for _ in 0..128 {
        if links.is_empty() {
            break;
        }
        let mut pending = Vec::new();
        let mut made = 0;
        for (relative, target, hard) in links {
            let target_relative = if hard {
                normalize_link(Path::new(""), &target)?
            } else {
                normalize_link(relative.parent().unwrap_or_else(|| Path::new("")), &target)?
            };
            let target_path = root.join(&target_relative);
            if !target_path.exists() {
                pending.push((relative, target, hard));
                continue;
            }
            let canonical = target_path.canonicalize()?;
            if !canonical.starts_with(&root) {
                bail!("Runtime link escapes its installation");
            }
            if let Some(parent) = relative.parent() {
                create_safe_directory(&root, parent)?;
            }
            let path = root.join(&relative);
            if hard {
                if !canonical.is_file() {
                    bail!("Runtime hard link must target a regular file");
                }
                fs::hard_link(canonical, path)?;
            } else {
                #[cfg(unix)]
                std::os::unix::fs::symlink(&target, path)?;
                #[cfg(not(unix))]
                bail!("Runtime archive symlinks need a Unix filesystem");
            }
            made += 1;
        }
        links = pending;
        if made == 0 {
            break;
        }
    }
    if !links.is_empty() {
        bail!("Runtime archive contains dangling or cyclic links");
    }
    Ok(())
}

fn clean_relative(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().len() > 4096 {
        bail!("Runtime archive path is too long");
    }
    let mut relative = PathBuf::new();
    for (i, component) in path.components().enumerate() {
        if i > 128 {
            bail!("Runtime archive path is too deep");
        }
        match component {
            Component::Normal(s) => {
                if s.to_string_lossy().contains(['\\', ':']) {
                    bail!("Invalid runtime archive path");
                }
                relative.push(s);
            }
            Component::CurDir => {}
            _ => bail!("Runtime archive path escapes its installation"),
        }
    }
    Ok(relative)
}
fn normalize_link(parent: &Path, target: &Path) -> Result<PathBuf> {
    let mut out = parent.to_path_buf();
    for part in target.components() {
        match part {
            Component::Normal(s) => {
                if s.to_string_lossy().contains(['\\', ':']) {
                    bail!("Invalid runtime link path");
                }
                out.push(s);
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    bail!("Runtime link escapes its installation");
                }
            }
            _ => bail!("Absolute runtime archive links are not permitted"),
        }
    }
    Ok(out)
}
fn create_safe_directory(root: &Path, relative: &Path) -> Result<()> {
    let mut path = root.to_path_buf();
    for component in relative.components() {
        if let Component::Normal(name) = component {
            path.push(name);
            match fs::symlink_metadata(&path) {
                Ok(metadata) => {
                    if !metadata.is_dir() || metadata.file_type().is_symlink() {
                        bail!("Runtime archive parent is not a normal directory");
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&path)?,
                Err(e) => return Err(e.into()),
            }
        } else {
            bail!("Invalid runtime archive parent");
        }
    }
    Ok(())
}
fn verify_linux_binary(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).context("Runtime archive is missing bin/ollama")?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        bail!("Managed Ollama executable must be a regular file");
    }
    let mut header = [0u8; 20];
    File::open(path)?.read_exact(&mut header)?;
    if &header[..4] != b"\x7fELF" || header[4] != 2 || header[5] != 1 {
        bail!("Managed runtime is not a little-endian 64-bit ELF executable");
    }
    let machine = u16::from_le_bytes([header[18], header[19]]);
    let expected = match std::env::consts::ARCH {
        "x86_64" => 62,
        "aarch64" => 183,
        _ => bail!("Unsupported managed runtime architecture"),
    };
    if machine != expected {
        bail!("Managed runtime does not match this CPU architecture");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            bail!("Managed runtime has no executable permission");
        }
    }
    Ok(())
}
fn read_small_file(path: &Path, limit: usize) -> Result<Vec<u8>> {
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        bail!("Managed runtime state must not be a symlink");
    }
    let mut data = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut data)?;
    if data.len() > limit {
        bail!("Managed runtime state is oversized");
    }
    Ok(data)
}
fn process_start_ticks(pid: u32) -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
        let rest = stat.rsplit_once(')').context("Invalid process status")?.1;
        rest.split_whitespace()
            .nth(19)
            .context("Process status has no start time")?
            .parse()
            .context("Invalid process start time")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        bail!("Managed runtime process identity is implemented for Linux")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn archive(path: &Path, entries: &[(&str, &[u8])]) {
        let mut builder = tar::Builder::new(File::create(path).unwrap());
        for (name, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            builder.append_data(&mut h, *name, *data).unwrap();
        }
        builder.finish().unwrap();
    }
    #[test]
    fn extraction_rejects_escape_paths_and_external_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.tar");
        {
            let mut builder = tar::Builder::new(File::create(&path).unwrap());
            let mut header = tar::Header::new_gnu();
            let name = b"../outside";
            header.as_mut_bytes()[..name.len()].copy_from_slice(name);
            header.set_size(0);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, &b""[..]).unwrap();
            builder.finish().unwrap();
        }
        assert!(
            extract_archive(&path, &dir.path().join("stage"), &AtomicBool::new(false)).is_err()
        );
        assert!(!dir.path().join("outside").exists());
        {
            let mut builder = tar::Builder::new(File::create(&path).unwrap());
            let mut header = tar::Header::new_gnu();
            header.set_path("lib/escape").unwrap();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_link_name("../../outside").unwrap();
            header.set_size(0);
            header.set_mode(0o777);
            header.set_cksum();
            builder.append(&header, &b""[..]).unwrap();
            builder.finish().unwrap();
        }
        assert!(
            extract_archive(&path, &dir.path().join("stage2"), &AtomicBool::new(false)).is_err()
        );
    }
    #[test]
    fn extraction_preserves_internal_links_and_checks_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runtime.tar");
        {
            let mut builder = tar::Builder::new(File::create(&path).unwrap());
            let mut h = tar::Header::new_gnu();
            h.set_size(4);
            h.set_mode(0o644);
            h.set_cksum();
            builder
                .append_data(&mut h, "lib/version.so", &b"data"[..])
                .unwrap();
            let mut h = tar::Header::new_gnu();
            h.set_path("lib/current.so").unwrap();
            h.set_entry_type(tar::EntryType::Symlink);
            h.set_link_name("version.so").unwrap();
            h.set_size(0);
            h.set_mode(0o777);
            h.set_cksum();
            builder.append(&h, &b""[..]).unwrap();
            builder.finish().unwrap();
        }
        let stage = dir.path().join("stage");
        extract_archive(&path, &stage, &AtomicBool::new(false)).unwrap();
        assert_eq!(fs::read(stage.join("lib/current.so")).unwrap(), b"data");
        assert!(
            extract_archive(&path, &dir.path().join("stopped"), &AtomicBool::new(true)).is_err()
        );
    }
    #[test]
    fn release_metadata_requires_official_origin_and_valid_digest() {
        let good = serde_json::json!({"tag_name":"v1.2.3","assets":[{"name":"ollama-linux-amd64.tar.zst","browser_download_url":"https://github.com/ollama/ollama/releases/download/v1.2.3/ollama-linux-amd64.tar.zst","size":100,"digest":format!("sha256:{}","a".repeat(64))}]});
        let release = parse_release(&good).unwrap();
        assert_eq!(
            release.assets[0].sha256.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        let mut bad = good.clone();
        bad["assets"][0]["browser_download_url"] = Value::String("https://example.org/run".into());
        assert!(parse_release(&bad).is_err());
        assert!(validate_version("v../../escape").is_err());
        assert_eq!(
            checksum_from_manifest(
                &format!("{}  file.tar.zst\n", "b".repeat(64)),
                "file.tar.zst"
            )
            .unwrap(),
            "b".repeat(64)
        );
        assert!(checksum_from_manifest("bad file.tar.zst", "file.tar.zst").is_err());
    }
    #[test]
    fn concurrent_manager_operations_are_serialized() {
        let dir = tempfile::tempdir().unwrap();
        let manager = OllamaManager::new(dir.path()).unwrap();
        let lock = ManagerLock::acquire(manager.root()).unwrap();
        assert!(ManagerLock::acquire(manager.root()).is_err());
        assert!(manager.stop().is_err());
        drop(lock);
        assert!(manager.stop().is_ok());
    }
    #[test]
    fn activation_and_rollback_preserve_previous_version() {
        let dir = tempfile::tempdir().unwrap();
        let manager = OllamaManager::new(dir.path()).unwrap();
        for version in ["v1.0.0", "v1.1.0"] {
            let path = manager.binary_for(version).unwrap();
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::copy("/bin/true", &path).unwrap();
            manager.activate(version).unwrap();
        }
        let status = manager.status().unwrap();
        assert_eq!(status.current_version.as_deref(), Some("v1.1.0"));
        assert_eq!(status.previous_version.as_deref(), Some("v1.0.0"));
        assert_eq!(
            manager.rollback().unwrap().current_version.as_deref(),
            Some("v1.0.0")
        );
        assert!(manager.activate("v9.0.0").is_err());
        assert_eq!(
            manager.status().unwrap().current_version.as_deref(),
            Some("v1.0.0")
        );
    }
    #[test]
    fn invalid_archive_does_not_change_active_installation() {
        let dir = tempfile::tempdir().unwrap();
        let manager = OllamaManager::new(dir.path()).unwrap();
        let binary = manager.binary_for("v1.0.0").unwrap();
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::copy("/bin/true", binary).unwrap();
        manager.activate("v1.0.0").unwrap();
        let bad = dir.path().join("bad.tar");
        archive(&bad, &[("bin/ollama", b"not an ELF")]);
        let stage = dir.path().join("staging");
        extract_archive(&bad, &stage, &AtomicBool::new(false)).unwrap();
        assert!(verify_linux_binary(&stage.join("bin/ollama")).is_err());
        assert_eq!(
            manager.status().unwrap().current_version.as_deref(),
            Some("v1.0.0")
        );
        assert!(!manager.process_matches(&ManagedProcess {
            pid: std::process::id(),
            start_ticks: process_start_ticks(std::process::id()).unwrap(),
            version: "v1.0.0".into(),
            port: 11434,
            models: dir.path().into()
        }));
    }
}
