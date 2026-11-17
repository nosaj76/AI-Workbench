//! Ollama's native HTTP API. Every streamed operation is cancellable while
//! connecting and reading, and successful EOF requires an explicit completion.
use anyhow::{anyhow, bail, Context, Result};
use reqwest::{Client, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    future::Future,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use url::Url;

const CANCEL_POLL: Duration = Duration::from_millis(50);
const IDLE_TIMEOUT: Duration = Duration::from_secs(180);
const JSON_LIMIT: usize = 32 * 1024 * 1024;
const LINE_LIMIT: usize = 8 * 1024 * 1024;
const STREAM_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default, deserialize_with = "arguments_from_wire")]
    pub arguments: Value,
}

fn arguments_from_wire<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Value, D::Error> {
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(raw) => serde_json::from_str(&raw).map_err(serde::de::Error::custom),
        other => Ok(other),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatOptions {
    pub temperature: f32,
    pub num_ctx: u32,
    pub num_predict: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: Option<i64>,
    pub think: Option<Value>,
    pub format: Option<Value>,
    pub keep_alive: String,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            num_ctx: 8192,
            num_predict: -1,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            think: None,
            format: None,
            keep_alive: "5m".into(),
        }
    }
}

impl ChatOptions {
    fn parameters(&self) -> Result<Value> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            bail!("Temperature must be a finite, nonnegative number");
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            bail!("Top-p must be between 0 and 1");
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            bail!("Repeat penalty must be positive and finite");
        }
        if self.num_ctx == 0 {
            bail!("Context size must be greater than zero");
        }
        if let Some(think) = &self.think {
            if !think.is_null() && !think.is_boolean() && !think.is_string() {
                bail!("Thinking must be a boolean or a model-supported string");
            }
        }
        if let Some(format) = &self.format {
            if !format.is_object() && format.as_str() != Some("json") {
                bail!("Response format must be json or a JSON schema object");
            }
        }
        let mut options = json!({"temperature":self.temperature,"num_ctx":self.num_ctx,"num_predict":self.num_predict,"top_p":self.top_p,"repeat_penalty":self.repeat_penalty});
        if let Some(seed) = self.seed {
            options["seed"] = json!(seed);
        }
        Ok(options)
    }
}

#[derive(Clone, Debug)]
pub enum StreamEvent {
    Content(String),
    Thinking(String),
    Stats(Value),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub remote_model: Option<String>,
    #[serde(default)]
    pub remote_host: Option<String>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ModelInfo {
    pub fn is_remote(&self) -> bool {
        self.remote_model.as_ref().is_some_and(|s| !s.is_empty())
            || self.remote_host.as_ref().is_some_and(|s| !s.is_empty())
            || self.name.to_ascii_lowercase().ends_with("-cloud")
            || self.name.to_ascii_lowercase().contains(":cloud")
    }
}

#[derive(Clone, Debug)]
pub struct OllamaClient {
    client: Client,
    base: Url,
    idle_timeout: Duration,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let base = crate::endpoint::normalize(base_url)?;
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .user_agent(concat!("ollama-workbench/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("Build Ollama HTTP client")?;
        Ok(Self {
            client,
            base,
            idle_timeout: IDLE_TIMEOUT,
        })
    }

    fn url(&self, endpoint: &str) -> Result<Url> {
        self.base.join(endpoint).context("Invalid API URL")
    }

    async fn wait_io<T, F>(&self, cancel: &Arc<AtomicBool>, future: F, operation: &str) -> Result<T>
    where
        F: Future<Output = std::result::Result<T, reqwest::Error>>,
    {
        if cancel.load(Ordering::Relaxed) {
            bail!("Request cancelled");
        }
        tokio::select! {
            biased;
            _ = cancelled(cancel) => bail!("Request cancelled"),
            result = tokio::time::timeout(self.idle_timeout,future) => {
                result.with_context(||format!("Ollama timed out while {operation}; no network progress for {} seconds",self.idle_timeout.as_secs()))?
                    .with_context(||format!("Ollama connection failed while {operation}"))
            }
        }
    }

    async fn send(&self, request: RequestBuilder, cancel: &Arc<AtomicBool>) -> Result<Response> {
        let response = self
            .wait_io(cancel, request.send(), "waiting for response headers")
            .await?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status();
        let body = self.read_body(response, cancel, 64 * 1024).await?;
        let description = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| {
                v.get("error").map(|e| {
                    e.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| e.to_string())
                })
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&body).chars().take(2048).collect());
        if status.is_redirection() {
            bail!("Ollama returned HTTP {status}. Redirects are disabled to keep conversation data at the configured endpoint");
        }
        bail!("Ollama returned HTTP {status}: {}", description.trim())
    }

    async fn read_body(
        &self,
        mut response: Response,
        cancel: &Arc<AtomicBool>,
        limit: usize,
    ) -> Result<Vec<u8>> {
        if response.content_length().is_some_and(|n| n > limit as u64) {
            bail!(
                "Ollama response exceeds the {} MiB limit",
                limit / (1024 * 1024)
            );
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = self
            .wait_io(cancel, response.chunk(), "reading response")
            .await?
        {
            if bytes.len().saturating_add(chunk.len()) > limit {
                bail!("Ollama response exceeds the allowed size");
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    async fn request_json(
        &self,
        request: RequestBuilder,
        cancel: &Arc<AtomicBool>,
    ) -> Result<Value> {
        let response = self.send(request, cancel).await?;
        let bytes = self.read_body(response, cancel, JSON_LIMIT).await?;
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Ok(Value::Null);
        }
        let value: Value =
            serde_json::from_slice(&bytes).context("Ollama returned invalid JSON")?;
        api_error(&value)?;
        Ok(value)
    }

    async fn stream<F>(
        &self,
        request: RequestBuilder,
        cancel: &Arc<AtomicBool>,
        mut on_record: F,
    ) -> Result<()>
    where
        F: FnMut(Value) -> Result<bool>,
    {
        let mut response = self.send(request, cancel).await?;
        let mut pending = Vec::new();
        let mut received = 0usize;
        while let Some(chunk) = self
            .wait_io(cancel, response.chunk(), "reading stream")
            .await?
        {
            received = received.saturating_add(chunk.len());
            if received > STREAM_LIMIT {
                bail!("Ollama stream exceeded the 64 MiB response limit");
            }
            pending.extend_from_slice(&chunk);
            while let Some(end) = pending.iter().position(|&b| b == b'\n') {
                if end > LINE_LIMIT {
                    bail!("An Ollama stream record exceeded the 8 MiB limit");
                }
                let line: Vec<u8> = pending.drain(..=end).collect();
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                if cancel.load(Ordering::Relaxed) {
                    bail!("Request cancelled");
                }
                let value: Value =
                    serde_json::from_slice(&line).context("Invalid JSON in Ollama stream")?;
                api_error(&value)?;
                if on_record(value)? {
                    return Ok(());
                }
            }
            if pending.len() > LINE_LIMIT {
                bail!("An Ollama stream record exceeded the 8 MiB limit");
            }
        }
        if !pending.iter().all(u8::is_ascii_whitespace) {
            let value: Value = serde_json::from_slice(&pending)
                .context("Incomplete JSON at the end of the Ollama stream")?;
            api_error(&value)?;
            if on_record(value)? {
                return Ok(());
            }
        }
        bail!("Ollama stream ended before reporting completion; the response may be incomplete")
    }

    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        tools: &[Value],
        cancel: Arc<AtomicBool>,
        mut on_event: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(StreamEvent),
    {
        require_model(model)?;
        let mut body = json!({"model":model,"messages":messages,"stream":true,"options":options.parameters()?,"keep_alive":options.keep_alive});
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(think) = &options.think {
            body["think"] = think.clone();
        }
        if let Some(format) = &options.format {
            body["format"] = format.clone();
        }
        let mut answer = ChatMessage {
            role: "assistant".into(),
            ..Default::default()
        };
        self.stream(
            self.client.post(self.url("api/chat")?).json(&body),
            &cancel,
            |mut value| {
                if let Some(message) = value.get("message") {
                    let part: ChatMessage = serde_json::from_value(message.clone())
                        .context("Invalid chat message in Ollama stream")?;
                    if !part.role.is_empty() && part.role != "assistant" {
                        bail!("Ollama returned an unexpected chat role");
                    }
                    if !part.content.is_empty() {
                        answer.content.push_str(&part.content);
                        on_event(StreamEvent::Content(part.content));
                    }
                    if !part.thinking.is_empty() {
                        answer.thinking.push_str(&part.thinking);
                        on_event(StreamEvent::Thinking(part.thinking));
                    }
                    for call in part.tool_calls {
                        if call.function.name.trim().is_empty()
                            || !call.function.arguments.is_object()
                        {
                            bail!("Ollama returned an invalid tool name or arguments object");
                        }
                        if answer.tool_calls.len() >= 128 {
                            bail!("Ollama exceeded the limit of 128 tool calls per response");
                        }
                        answer.tool_calls.push(call);
                    }
                }
                let done = value.get("done").and_then(Value::as_bool) == Some(true);
                if done {
                    if let Some(object) = value.as_object_mut() {
                        object.remove("message");
                    }
                    on_event(StreamEvent::Stats(value));
                }
                Ok(done)
            },
        )
        .await?;
        Ok(answer)
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let value = self
            .request_json(self.client.get(self.url("api/tags")?), &uncancelled())
            .await?;
        serde_json::from_value(
            value
                .get("models")
                .cloned()
                .ok_or_else(|| anyhow!("Ollama model list is missing models"))?,
        )
        .context("Invalid model list")
    }

    pub async fn running_models(&self) -> Result<Value> {
        self.request_json(self.client.get(self.url("api/ps")?), &uncancelled())
            .await
    }

    pub async fn version(&self) -> Result<String> {
        let value = self
            .request_json(self.client.get(self.url("api/version")?), &uncancelled())
            .await?;
        value
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Ollama version response is missing version"))
    }

    pub async fn show(&self, model: &str) -> Result<Value> {
        require_model(model)?;
        self.request_json(
            self.client
                .post(self.url("api/show")?)
                .json(&json!({"model":model})),
            &uncancelled(),
        )
        .await
    }

    /// Preflight local-only inference with model metadata, including renamed
    /// cloud aliases. The request sends a model name, never conversation data.
    /// Failure to obtain metadata is deliberately not treated as local proof.
    pub async fn ensure_local_model(&self, model: &str, cancel: Arc<AtomicBool>) -> Result<()> {
        require_model(model)?;
        let name = model.to_ascii_lowercase();
        if name.contains(":cloud")
            || name
                .split(':')
                .next()
                .is_some_and(|s| s.ends_with("-cloud"))
        {
            bail!("This is a cloud-backed model. Enable remote inference in Settings to use it");
        }
        let metadata = self
            .request_json(
                self.client
                    .post(self.url("api/show")?)
                    .json(&json!({"model":model})),
                &cancel,
            )
            .await
            .context("Cannot verify that this model runs locally; no conversation data was sent")?;
        if !metadata.is_object() {
            bail!("Ollama returned invalid model metadata; local execution could not be verified");
        }
        for field in ["remote_model", "remote_host"] {
            if metadata.get(field).is_some_and(|value| match value {
                Value::Null => false,
                Value::String(s) => !s.trim().is_empty(),
                _ => true,
            }) {
                bail!("Model {model} routes inference to a remote service ({field}). Remote inference is disabled in Settings");
            }
        }
        Ok(())
    }

    pub async fn embeddings(
        &self,
        model: &str,
        inputs: &[String],
        cancel: Arc<AtomicBool>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embeddings_with_keep_alive(model, inputs, None, cancel)
            .await
    }

    /// Request embeddings with an optional model residency policy. Omitting
    /// the policy preserves Ollama's existing behavior; "0" releases the model
    /// after the request so a different specialist can use the same memory.
    pub async fn embeddings_with_keep_alive(
        &self,
        model: &str,
        inputs: &[String],
        keep_alive: Option<&str>,
        cancel: Arc<AtomicBool>,
    ) -> Result<Vec<Vec<f32>>> {
        require_model(model)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut body = json!({"model":model,"input":inputs,"truncate":false});
        if let Some(keep_alive) = keep_alive {
            body["keep_alive"] = json!(keep_alive);
        }
        let value = self
            .request_json(
                self.client.post(self.url("api/embed")?).json(&body),
                &cancel,
            )
            .await?;
        let vectors: Vec<Vec<f32>> = serde_json::from_value(
            value
                .get("embeddings")
                .cloned()
                .ok_or_else(|| anyhow!("Ollama embedding response is missing embeddings"))?,
        )
        .context("Invalid embedding vectors")?;
        if vectors.len() != inputs.len() {
            bail!(
                "Ollama returned {} embeddings for {} inputs",
                vectors.len(),
                inputs.len()
            );
        }
        let dimension = vectors.first().map(Vec::len).unwrap_or_default();
        if dimension == 0
            || vectors
                .iter()
                .any(|v| v.len() != dimension || v.iter().any(|n| !n.is_finite()))
        {
            bail!("Ollama returned empty, inconsistent or non-finite embedding vectors");
        }
        Ok(vectors)
    }

    pub async fn pull<F>(&self, model: &str, cancel: Arc<AtomicBool>, on_progress: F) -> Result<()>
    where
        F: FnMut(Value),
    {
        require_model(model)?;
        self.progress(
            "api/pull",
            json!({"model":model,"stream":true}),
            cancel,
            on_progress,
        )
        .await
    }

    /// `files` may map GGUF filenames to previously uploaded sha256 digests.
    pub async fn create<F>(
        &self,
        mut body: Value,
        cancel: Arc<AtomicBool>,
        on_progress: F,
    ) -> Result<()>
    where
        F: FnMut(Value),
    {
        if !body.is_object() {
            bail!("Create model requires a JSON object");
        }
        require_model(
            body.get("model")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )?;
        body["stream"] = Value::Bool(true);
        self.progress("api/create", body, cancel, on_progress).await
    }

    async fn progress<F>(
        &self,
        endpoint: &str,
        body: Value,
        cancel: Arc<AtomicBool>,
        mut on_progress: F,
    ) -> Result<()>
    where
        F: FnMut(Value),
    {
        self.stream(
            self.client.post(self.url(endpoint)?).json(&body),
            &cancel,
            |value| {
                let done = value.get("status").and_then(Value::as_str) == Some("success");
                on_progress(value);
                Ok(done)
            },
        )
        .await
    }

    pub async fn delete(&self, model: &str) -> Result<()> {
        require_model(model)?;
        self.request_json(
            self.client
                .delete(self.url("api/delete")?)
                .json(&json!({"model":model})),
            &uncancelled(),
        )
        .await?;
        Ok(())
    }

    pub async fn copy(&self, source: &str, destination: &str) -> Result<()> {
        require_model(source)?;
        require_model(destination)?;
        self.request_json(
            self.client
                .post(self.url("api/copy")?)
                .json(&json!({"source":source,"destination":destination})),
            &uncancelled(),
        )
        .await?;
        Ok(())
    }

    pub async fn unload(&self, model: &str) -> Result<()> {
        require_model(model)?;
        self.request_json(
            self.client
                .post(self.url("api/generate")?)
                .json(&json!({"model":model,"keep_alive":0,"stream":false})),
            &uncancelled(),
        )
        .await?;
        Ok(())
    }

    /// Hash a file and stream its bytes to Ollama's blob store. This does not
    /// create or select a model; pass the returned digest in `create.files`.
    /// Progress reports bytes read by HTTP transport, not server-side storage.
    pub async fn upload_blob<F>(
        &self,
        path: &Path,
        cancel: Arc<AtomicBool>,
        on_progress: F,
    ) -> Result<String>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        if cancel.load(Ordering::Relaxed) {
            bail!("Request cancelled");
        }
        let path = path.to_owned();
        let hash_cancel = cancel.clone();
        let mut hash_job = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut file = std::fs::File::open(&path).context("Open model file for upload")?;
            let metadata = file.metadata()?;
            if !metadata.is_file() || metadata.len() == 0 {
                bail!("Model upload requires a nonempty regular file");
            }
            let size = metadata.len();
            let mut hash = Sha256::new();
            let mut buffer = vec![0u8; 256 * 1024];
            let mut hashed = 0u64;
            loop {
                if hash_cancel.load(Ordering::Relaxed) {
                    bail!("Request cancelled");
                }
                let count = file.read(&mut buffer).context("Read model for SHA-256")?;
                if count == 0 {
                    break;
                }
                hashed = hashed.saturating_add(count as u64);
                if hashed > size {
                    bail!("Model file grew during hashing; retry after it stops changing");
                }
                hash.update(&buffer[..count]);
            }
            if hashed != size {
                bail!("Model file changed during hashing; retry after it stops changing");
            }
            file.seek(SeekFrom::Start(0))?;
            Ok((file, size, format!("sha256:{:x}", hash.finalize())))
        });
        let (file, size, digest) = tokio::select! {
            biased;
            _=cancelled(&cancel)=>bail!("Request cancelled"),
            result=&mut hash_job=>result.context("Model hashing worker failed")??,
        };
        let read_cancel = cancel.clone();
        let stream = futures_util::stream::try_unfold(
            (file, 0u64, on_progress),
            move |(mut file, position, mut on_progress)| {
                let read_cancel = read_cancel.clone();
                async move {
                    if read_cancel.load(Ordering::Relaxed) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "Upload cancelled",
                        ));
                    }
                    let (file, buffer) =
                        tokio::task::spawn_blocking(move || -> std::io::Result<_> {
                            let mut buffer = vec![0u8; 256 * 1024];
                            let count = file.read(&mut buffer)?;
                            buffer.truncate(count);
                            Ok((file, buffer))
                        })
                        .await
                        .map_err(std::io::Error::other)??;
                    if buffer.is_empty() {
                        if position != size {
                            return Err(std::io::Error::other("Model file changed during upload"));
                        }
                        return Ok(None);
                    }
                    let next = position + buffer.len() as u64;
                    if next > size {
                        return Err(std::io::Error::other("Model file grew during upload"));
                    }
                    on_progress(next, size);
                    Ok(Some((buffer, (file, next, on_progress))))
                }
            },
        );
        let request = self
            .client
            .post(self.url(&format!("api/blobs/{digest}"))?)
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .header(reqwest::header::CONTENT_LENGTH, size)
            .body(reqwest::Body::wrap_stream(stream));
        let response = self.send(request, &cancel).await?;
        // Drain any small response so malformed HTTP or API errors are visible.
        let bytes = self.read_body(response, &cancel, 64 * 1024).await?;
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            api_error(&value)?;
        }
        Ok(digest)
    }
}

fn require_model(model: &str) -> Result<()> {
    if model.trim().is_empty() || model.contains(['\r', '\n', '\0']) {
        bail!("A nonempty model name is required");
    }
    Ok(())
}

fn api_error(value: &Value) -> Result<()> {
    if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
        bail!(
            "Ollama: {}",
            error
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| error.to_string())
        );
    }
    Ok(())
}

fn uncancelled() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

async fn cancelled(cancel: &AtomicBool) {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(CANCEL_POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        task::JoinHandle,
    };

    async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
        let mut data = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let count = socket.read(&mut buffer).await.unwrap();
            if count == 0 {
                break;
            }
            data.extend_from_slice(&buffer[..count]);
            if let Some(header_end) = data.windows(4).position(|b| b == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&data[..header_end]);
                let length = headers
                    .lines()
                    .find_map(|l| {
                        l.split_once(':')
                            .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                            .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if data.len() >= header_end + 4 + length {
                    break;
                }
            }
        }
        data
    }

    async fn serve(
        status: &str,
        chunks: Vec<Vec<u8>>,
        extra_headers: &str,
    ) -> (String, JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let status = status.to_owned();
        let extra_headers = extra_headers.to_owned();
        let worker = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            let length: usize = chunks.iter().map(Vec::len).sum();
            socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: application/x-ndjson\r\nContent-Length: {length}\r\nConnection: close\r\n{extra_headers}\r\n").as_bytes()).await.unwrap();
            for chunk in chunks {
                if socket.write_all(&chunk).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            request
        });
        (base, worker)
    }

    fn request_body(request: &[u8]) -> Value {
        let end = request.windows(4).position(|b| b == b"\r\n\r\n").unwrap() + 4;
        serde_json::from_slice(&request[end..]).unwrap()
    }

    async fn chat(client: &OllamaClient) -> Result<ChatMessage> {
        client
            .chat_stream(
                "test",
                &[],
                &ChatOptions::default(),
                &[],
                uncancelled(),
                |_| {},
            )
            .await
    }

    #[test]
    fn endpoint_validation_and_path_prefix() {
        for invalid in [
            "",
            "localhost:11434",
            "file:///etc/passwd",
            "ftp://localhost",
            "http://user:password@localhost",
            "http://localhost/?key=x",
            "http://localhost/#fragment",
        ] {
            assert!(OllamaClient::new(invalid).is_err(), "accepted {invalid}");
        }
        let client = OllamaClient::new("http://127.0.0.1:11434/proxy/ollama").unwrap();
        assert_eq!(
            client.url("api/chat").unwrap().as_str(),
            "http://127.0.0.1:11434/proxy/ollama/api/chat"
        );
        for suffix in ["/api", "/api/"] {
            let client =
                OllamaClient::new(&format!("http://localhost:11434/proxy{suffix}")).unwrap();
            assert_eq!(
                client.url("api/chat").unwrap().as_str(),
                "http://localhost:11434/proxy/api/chat"
            );
        }
    }

    #[tokio::test]
    async fn fragmented_ndjson_preserves_unicode_thinking_tools_and_final_content() {
        let records=concat!(
            "{\"message\":{\"role\":\"assistant\",\"thinking\":\"Consider 🦀\"},\"done\":false}\r\n",
            "\n{\"message\":{\"content\":\"Café \"},\"done\":false}\n",
            "{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"main.rs\"}}}]},\"done\":false}\n",
            "{\"message\":{\"content\":\"ready\"},\"done\":true,\"eval_count\":3}"
        );
        let chunks = records.as_bytes().chunks(3).map(|c| c.to_vec()).collect();
        let (base, worker) = serve("200 OK", chunks, "").await;
        let client = OllamaClient::new(&base).unwrap();
        let options = ChatOptions {
            think: Some(json!(true)),
            format: Some(json!("json")),
            seed: Some(42),
            ..Default::default()
        };
        let mut events = Vec::new();
        let answer = client
            .chat_stream(
                "test",
                &[ChatMessage {
                    role: "user".into(),
                    content: "review".into(),
                    ..Default::default()
                }],
                &options,
                &[json!({"type":"function","function":{"name":"read_file"}})],
                uncancelled(),
                |e| events.push(e),
            )
            .await
            .unwrap();
        assert_eq!(answer.content, "Café ready");
        assert_eq!(answer.thinking, "Consider 🦀");
        assert_eq!(answer.tool_calls[0].function.arguments["path"], "main.rs");
        assert!(matches!(&events[0],StreamEvent::Thinking(s) if s=="Consider 🦀"));
        assert!(matches!(events.last(),Some(StreamEvent::Stats(v)) if v["eval_count"]==3));
        let request = worker.await.unwrap();
        assert!(String::from_utf8_lossy(&request).starts_with("POST /api/chat "));
        let body = request_body(&request);
        assert_eq!(body["stream"], true);
        assert_eq!(body["options"]["seed"], 42);
        assert_eq!(body["think"], true);
        assert_eq!(body["format"], "json");
        assert!(body["options"].get("think").is_none());
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    }

    #[tokio::test]
    async fn truncated_and_malformed_streams_never_report_success() {
        for data in [
            "{\"message\":{\"content\":\"partial\"},\"done\":false}\n",
            "{\"message\":{\"content\":\"partial",
            "not json\n",
            "",
        ] {
            let (base, worker) = serve("200 OK", vec![data.as_bytes().to_vec()], "").await;
            assert!(
                chat(&OllamaClient::new(&base).unwrap()).await.is_err(),
                "accepted {data}"
            );
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn http_and_midstream_api_errors_are_preserved() {
        for (status,data) in [
            ("404 Not Found","{\"error\":\"model not found\"}"),
            ("200 OK","{\"message\":{\"content\":\"partial\"},\"done\":false}\n{\"error\":\"out of memory\"}\n"),
        ] {
            let (base,worker)=serve(status,vec![data.as_bytes().to_vec()],"").await;
            let error=chat(&OllamaClient::new(&base).unwrap()).await.unwrap_err().to_string();
            assert!(error.contains(if status.starts_with("404"){"model not found"}else{"out of memory"}),"{error}");
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn cancellation_interrupts_waiting_for_headers_and_stream_bytes() {
        for send_headers in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let client =
                OllamaClient::new(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                if send_headers {
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n")
                        .await
                        .unwrap();
                }
                tokio::time::sleep(Duration::from_secs(10)).await;
            });
            let cancel = uncancelled();
            let setter = cancel.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                setter.store(true, Ordering::Relaxed);
            });
            let start = std::time::Instant::now();
            let error = client
                .chat_stream("test", &[], &ChatOptions::default(), &[], cancel, |_| {})
                .await
                .unwrap_err();
            assert!(error.to_string().contains("cancelled"));
            assert!(
                start.elapsed() < Duration::from_millis(200),
                "cancellation took {:?}",
                start.elapsed()
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn finite_idle_timeout_covers_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client =
            OllamaClient::new(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
        client.idle_timeout = Duration::from_millis(40);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        });
        assert!(chat(&client)
            .await
            .unwrap_err()
            .to_string()
            .contains("timed out"));
        server.abort();
    }

    #[tokio::test]
    async fn redirects_do_not_forward_chat_data() {
        let destination = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let location = format!(
            "Location: http://{}/api/chat\r\n",
            destination.local_addr().unwrap()
        );
        let (base, worker) = serve("307 Temporary Redirect", vec![], &location).await;
        let error = chat(&OllamaClient::new(&base).unwrap())
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("Redirects are disabled"));
        assert!(
            tokio::time::timeout(Duration::from_millis(30), destination.accept())
                .await
                .is_err()
        );
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn model_metadata_retains_cloud_routing_fields() {
        let body=json!({"models":[{"name":"remote:latest","size":42,"details":{"family":"test"},"remote_host":"https://ollama.com","remote_model":"remote","digest":"abc"},{"name":"local:latest"}]}).to_string();
        let (base, worker) = serve("200 OK", vec![body.into_bytes()], "").await;
        let models = OllamaClient::new(&base)
            .unwrap()
            .list_models()
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
        assert!(models[0].is_remote());
        assert_eq!(models[0].extra["digest"], "abc");
        assert!(!models[1].is_remote());
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn local_model_preflight_blocks_renamed_cloud_models_and_lookup_failures() {
        for (status, data) in [
            (
                "200 OK",
                r#"{"remote_model":"cloud-model","remote_host":"https://ollama.com"}"#,
            ),
            ("404 Not Found", r#"{"error":"model not found"}"#),
            ("200 OK", "null"),
        ] {
            let (base, worker) = serve(status, vec![data.as_bytes().to_vec()], "").await;
            assert!(OllamaClient::new(&base)
                .unwrap()
                .ensure_local_model("innocent-alias", uncancelled())
                .await
                .is_err());
            let request = worker.await.unwrap();
            assert!(String::from_utf8_lossy(&request).starts_with("POST /api/show "));
            assert_eq!(request_body(&request), json!({"model":"innocent-alias"}));
        }
        let (base, worker) = serve(
            "200 OK",
            vec![br#"{"details":{"family":"test"},"remote_model":"","remote_host":null}"#.to_vec()],
            "",
        )
        .await;
        OllamaClient::new(&base)
            .unwrap()
            .ensure_local_model("local", uncancelled())
            .await
            .unwrap();
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn embedding_batch_is_validated_and_never_silently_truncated() {
        let (base, worker) = serve(
            "200 OK",
            vec![br#"{"embeddings":[[0.5,1.0],[0.25,-1.0]]}"#.to_vec()],
            "",
        )
        .await;
        let vectors = OllamaClient::new(&base)
            .unwrap()
            .embeddings("embed", &["one".into(), "two".into()], uncancelled())
            .await
            .unwrap();
        assert_eq!(vectors, vec![vec![0.5, 1.0], vec![0.25, -1.0]]);
        let body = request_body(&worker.await.unwrap());
        assert_eq!(body["truncate"], false);
        assert_eq!(body["input"], json!(["one", "two"]));
        assert!(body.get("keep_alive").is_none());
        for invalid in [r#"{"embeddings":[]}"#, r#"{"embeddings":[[1],[1,2]]}"#] {
            let (base, worker) = serve("200 OK", vec![invalid.as_bytes().to_vec()], "").await;
            assert!(OllamaClient::new(&base)
                .unwrap()
                .embeddings("embed", &["one".into(), "two".into()], uncancelled())
                .await
                .is_err());
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn embedding_request_can_release_the_model_after_inference() {
        let (base, worker) = serve(
            "200 OK",
            vec![br#"{"embeddings":[[0.5,1.0]]}"#.to_vec()],
            "",
        )
        .await;
        let vectors = OllamaClient::new(&base)
            .unwrap()
            .embeddings_with_keep_alive("embed", &["one".into()], Some("0"), uncancelled())
            .await
            .unwrap();
        assert_eq!(vectors, vec![vec![0.5, 1.0]]);
        let request = worker.await.unwrap();
        assert!(String::from_utf8_lossy(&request).starts_with("POST /api/embed "));
        let body = request_body(&request);
        assert_eq!(body["keep_alive"], "0");
        assert_eq!(body["model"], "embed");
        assert_eq!(body["input"], json!(["one"]));
        assert_eq!(body["truncate"], false);
    }

    #[tokio::test]
    async fn model_progress_requires_success_and_create_preserves_fields() {
        let records =
            b"{\"status\":\"pulling\",\"completed\":10,\"total\":20}\n{\"status\":\"success\"}\n";
        let (base, worker) = serve("200 OK", vec![records.to_vec()], "").await;
        let mut progress = Vec::new();
        OllamaClient::new(&base)
            .unwrap()
            .pull("test", uncancelled(), |v| progress.push(v))
            .await
            .unwrap();
        assert_eq!(progress.len(), 2);
        assert_eq!(request_body(&worker.await.unwrap())["model"], "test");
        let (base, worker) = serve("200 OK", vec![b"{\"status\":\"success\"}".to_vec()], "").await;
        OllamaClient::new(&base)
            .unwrap()
            .create(
                json!({"model":"custom","from":"test","system":"Be concise","stream":false}),
                uncancelled(),
                |_| {},
            )
            .await
            .unwrap();
        let body = request_body(&worker.await.unwrap());
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "Be concise");
        let (base, worker) =
            serve("200 OK", vec![b"{\"status\":\"pulling\"}\n".to_vec()], "").await;
        assert!(OllamaClient::new(&base)
            .unwrap()
            .pull("test", uncancelled(), |_| {})
            .await
            .is_err());
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn model_mutation_methods_send_native_payloads() {
        let (base, worker) = serve("200 OK", vec![], "").await;
        OllamaClient::new(&base)
            .unwrap()
            .delete("test")
            .await
            .unwrap();
        let request = worker.await.unwrap();
        assert!(String::from_utf8_lossy(&request).starts_with("DELETE /api/delete "));
        assert_eq!(request_body(&request)["model"], "test");
        let (base, worker) = serve("200 OK", vec![], "").await;
        OllamaClient::new(&base)
            .unwrap()
            .copy("test", "copy")
            .await
            .unwrap();
        assert_eq!(
            request_body(&worker.await.unwrap()),
            json!({"source":"test","destination":"copy"})
        );
        let (base, worker) = serve("200 OK", vec![b"{\"done\":true}".to_vec()], "").await;
        OllamaClient::new(&base)
            .unwrap()
            .unload("test")
            .await
            .unwrap();
        let body = request_body(&worker.await.unwrap());
        assert_eq!(body["keep_alive"], 0);
        assert_eq!(body["stream"], false);
    }

    #[tokio::test]
    async fn blob_upload_streams_exact_bytes_with_sha256_and_progress() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let data = vec![42u8; 600_000];
        std::fs::write(&path, &data).unwrap();
        let expected = format!("sha256:{:x}", Sha256::digest(&data));
        let (base, worker) = serve("201 Created", vec![], "").await;
        let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = progress.clone();
        let digest = OllamaClient::new(&base)
            .unwrap()
            .upload_blob(&path, uncancelled(), move |read, total| {
                observed.lock().unwrap().push((read, total))
            })
            .await
            .unwrap();
        assert_eq!(digest, expected);
        assert_eq!(progress.lock().unwrap().last(), Some(&(600_000, 600_000)));
        let request = worker.await.unwrap();
        assert!(String::from_utf8_lossy(&request[..100])
            .starts_with(&format!("POST /api/blobs/{expected} ")));
        let body_start = request.windows(4).position(|b| b == b"\r\n\r\n").unwrap() + 4;
        assert_eq!(&request[body_start..], data.as_slice());
    }
}
