//! Native image-generation jobs backed by an installed local diffusion engine.
//! The desktop owns prompts, parameters, cancellation, images and job records;
//! no engine web frontend is embedded or opened.

use crate::{
    integrations::{run_command_cancellable, CommandSpec},
    media,
};
use anyhow::{bail, Context, Result};
use base64::Engine as _;
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use url::Url;

const JSON_LIMIT: usize = 128 * 1024 * 1024;
const API_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageBackend {
    #[default]
    ComfyUi,
    Automatic1111,
    StableDiffusionCpp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageConfig {
    pub backend: ImageBackend,
    pub endpoint: String,
    pub allow_remote: bool,
    pub checkpoint: String,
    pub sd_program: String,
    pub sd_model: PathBuf,
    pub sd_extra_args: Vec<String>,
    pub allow_custom_nodes: bool,
    pub timeout_secs: u64,
}
impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            backend: ImageBackend::ComfyUi,
            endpoint: "http://127.0.0.1:8188".into(),
            allow_remote: false,
            checkpoint: String::new(),
            sd_program: "sd-cli".into(),
            sd_model: PathBuf::new(),
            sd_extra_args: vec![],
            allow_custom_nodes: false,
            timeout_secs: 900,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    /// -1 chooses and records a fresh nonnegative seed before submission.
    pub seed: i64,
    pub batch_size: u32,
    pub sampler: String,
    pub scheduler: String,
    pub input_image: Option<PathBuf>,
    pub strength: f32,
    /// An API-format Comfy graph; this is not its editor's node/link format.
    /// Exact string placeholders such as {{prompt}} are replaced as JSON values.
    pub workflow: Option<Value>,
}
impl Default for ImageRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg_scale: 7.0,
            seed: -1,
            batch_size: 1,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            input_image: None,
            strength: 0.65,
            workflow: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOutput {
    pub id: String,
    pub directory: PathBuf,
    pub images: Vec<ImageArtifact>,
    pub metadata: PathBuf,
    pub seed: i64,
    pub warnings: Vec<String>,
}

pub fn preview(config: &ImageConfig, request: &ImageRequest) -> Result<Value> {
    validate(config, request)?;
    Ok(
        json!({"engine":config.backend,"endpoint":if config.backend==ImageBackend::StableDiffusionCpp {None}else{Some(endpoint(config)?.to_string())},"checkpoint":config.checkpoint,"local_executable":if config.backend==ImageBackend::StableDiffusionCpp {Some(&config.sd_program)}else{None},"model_file":config.sd_model,"request":request,"extra_arguments":config.sd_extra_args,"custom_nodes_allowed":config.allow_custom_nodes,"output":"Fresh local images activity directory; no source overwrite","network":"Remote engine endpoints require explicit opt-in. Trusted custom engine nodes retain their own permissions."}),
    )
}

pub fn list_models(config: &ImageConfig, cancel: Arc<AtomicBool>) -> Result<Vec<String>> {
    media::check_cancel(&cancel)?;
    if config.backend == ImageBackend::StableDiffusionCpp {
        return Ok(if config.sd_model.is_file() {
            vec![config.sd_model.display().to_string()]
        } else {
            vec![]
        });
    }
    let base = endpoint(config)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let client = http_client()?;
        let (route, comfy) = if config.backend == ImageBackend::ComfyUi {
            ("object_info/CheckpointLoaderSimple", true)
        } else {
            ("sdapi/v1/sd-models", false)
        };
        let value = json_response(
            client
                .get(base.join(route)?)
                .timeout(Duration::from_secs(15)),
            &cancel,
            API_LIMIT,
        )
        .await?;
        let models = if comfy {
            value["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                .as_array()
                .context("The engine did not expose a checkpoint list")?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        } else {
            value
                .as_array()
                .context("The engine did not return a model array")?
                .iter()
                .filter_map(|m| m["title"].as_str().or(m["model_name"].as_str()))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        if models.len() > 5000 {
            bail!("The engine returned an excessive number of checkpoints");
        }
        Ok(models)
    })
}

/// Run on a desktop worker thread. Output files are created only in a new UUID
/// directory, and a result is successful only after real images were validated.
pub fn generate(
    config: &ImageConfig,
    request: &ImageRequest,
    data_dir: &Path,
    cancel: Arc<AtomicBool>,
    mut progress: impl FnMut(String),
) -> Result<ImageOutput> {
    validate(config, request)?;
    media::check_cancel(&cancel)?;
    let directory = media::create_run_dir(data_dir, "images")?;
    let id = directory
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut request = request.clone();
    if request.seed == -1 {
        request.seed =
            (u128::from_le_bytes(*uuid::Uuid::new_v4().as_bytes()) & (i64::MAX as u128)) as i64;
    }
    media::write_new(
        &directory.join("request.json"),
        &serde_json::to_vec_pretty(&json!({"engine":config,"request":request}))?,
    )?;
    let result = match config.backend {
        ImageBackend::StableDiffusionCpp => {
            generate_cpp(config, &request, &directory, &cancel, &mut progress)
        }
        _ => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                let client = http_client()?;
                let base = endpoint(config)?;
                let work = async {
                    if config.backend == ImageBackend::ComfyUi {
                        generate_comfy(
                            config,
                            &request,
                            &directory,
                            &client,
                            &base,
                            &cancel,
                            &mut progress,
                        )
                        .await
                    } else {
                        generate_a1111(
                            config,
                            &request,
                            &directory,
                            &client,
                            &base,
                            &cancel,
                            &mut progress,
                        )
                        .await
                    }
                };
                // Each backend owns its engine cleanup; this outer limit bounds only
                // nonqueued A1111 requests. Comfy has its own deadline after submission.
                work.await
            }),
    };
    let (images, details, warnings) = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = media::write_new(
                &directory.join("failed.json"),
                &serde_json::to_vec_pretty(
                    &json!({"error":format!("{error:#}"),"cancelled":cancel.load(Ordering::Relaxed)}),
                )?,
            );
            return Err(error)
                .with_context(|| format!("Image job stopped; record: {}", directory.display()));
        }
    };
    media::check_cancel(&cancel)?;
    if images.is_empty() {
        bail!("The image engine completed without producing any images");
    }
    let metadata = directory.join("generation.json");
    media::write_new(
        &metadata,
        &serde_json::to_vec_pretty(
            &json!({"id":id,"engine":config.backend,"seed":request.seed,"images":images,"details":details,"warnings":warnings}),
        )?,
    )?;
    progress(format!("Saved {} generated image(s)", images.len()));
    Ok(ImageOutput {
        id,
        directory,
        images,
        metadata,
        seed: request.seed,
        warnings,
    })
}

fn validate(config: &ImageConfig, request: &ImageRequest) -> Result<()> {
    if request.prompt.trim().is_empty()
        || request.prompt.len() > 32_000
        || request.negative_prompt.len() > 16_000
    {
        bail!("Enter a prompt up to 32,000 bytes and a negative prompt up to 16,000 bytes");
    }
    if !(64..=2048).contains(&request.width)
        || !(64..=2048).contains(&request.height)
        || !request.width.is_multiple_of(8)
        || !request.height.is_multiple_of(8)
    {
        bail!("Image width/height must be multiples of eight between 64 and 2048");
    }
    if !(1..=200).contains(&request.steps)
        || !(1..=4).contains(&request.batch_size)
        || request.seed < -1
    {
        bail!("Choose 1–200 steps, 1–4 images and a nonnegative seed (or -1 for random)");
    }
    if !request.cfg_scale.is_finite()
        || !(0.0..=30.0).contains(&request.cfg_scale)
        || !request.strength.is_finite()
        || !(0.0..=1.0).contains(&request.strength)
    {
        bail!("CFG/strength are outside their supported ranges");
    }
    for name in [&request.sampler, &request.scheduler] {
        if name.is_empty() || name.len() > 128 || name.contains('\0') {
            bail!("Choose a valid sampler and scheduler");
        }
    }
    if let Some(path) = &request.input_image {
        media::image_input(path)?;
    }
    if config.backend == ImageBackend::StableDiffusionCpp {
        media::regular_input(&config.sd_model, 64 * 1024 * 1024 * 1024)
            .context("Choose an installed diffusion model file")?;
        if config.sd_program.trim().is_empty() {
            bail!("Configure the sd-cli executable");
        }
        if request.input_image.is_some() || request.workflow.is_some() {
            bail!("Use ComfyUI or AUTOMATIC1111 for image-to-image/custom workflow requests");
        }
        if config.sd_extra_args.len() > 100
            || config.sd_extra_args.iter().any(|a| {
                a.contains('\0')
                    || a.len() > 8192
                    || matches!(a.as_str(), "-o" | "--output" | "--output-dir")
            })
        {
            bail!("Extra diffusion arguments are invalid or override the managed output path");
        }
    } else {
        endpoint(config)?;
        if config.backend == ImageBackend::ComfyUi
            && request.workflow.is_none()
            && config.checkpoint.trim().is_empty()
        {
            bail!("Choose an installed ComfyUI checkpoint or provide an API workflow");
        }
    }
    if let Some(workflow) = &request.workflow {
        if config.backend != ImageBackend::ComfyUi {
            bail!("Custom workflow JSON is supported by ComfyUI");
        }
        validate_workflow(workflow, config.allow_custom_nodes)?;
    }
    Ok(())
}

fn endpoint(config: &ImageConfig) -> Result<Url> {
    let mut url = Url::parse(config.endpoint.trim()).context("Image engine URL is invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("Use an HTTP(S) image engine URL without credentials, query or fragment");
    }
    let host = url.host_str().context("Image engine URL has no host")?;
    let local = host == "localhost"
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if !local && !config.allow_remote {
        bail!("Remote image generation is disabled; choose a loopback engine URL or explicitly allow remote engines");
    }
    if host == "localhost" {
        url.set_host(Some("127.0.0.1"))
            .map_err(|_| anyhow::anyhow!("Invalid local host"))?;
    }
    let path = format!("{}/", url.path().trim_end_matches('/'));
    url.set_path(&path);
    Ok(url)
}

fn http_client() -> Result<Client> {
    Ok(Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()?)
}

async fn cancellable<T>(future: impl Future<Output = Result<T>>, cancel: &AtomicBool) -> Result<T> {
    tokio::pin!(future);
    loop {
        media::check_cancel(cancel)?;
        tokio::select! {result=&mut future=>return result,_=tokio::time::sleep(Duration::from_millis(50))=>{}}
    }
}

async fn response_bytes(
    request: RequestBuilder,
    cancel: &AtomicBool,
    limit: usize,
) -> Result<Vec<u8>> {
    cancellable(
        async {
            let response = request.send().await.context("Connect to image engine")?;
            let status = response.status();
            if response.content_length().is_some_and(|n| n > limit as u64) {
                bail!("Image engine response exceeds its size limit");
            }
            let mut stream = response.bytes_stream();
            let mut data = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if data.len().saturating_add(chunk.len()) > limit {
                    bail!("Image engine response exceeds its size limit");
                }
                data.extend_from_slice(&chunk);
            }
            if !status.is_success() {
                bail!(
                    "Image engine returned HTTP {status}: {}",
                    String::from_utf8_lossy(&data)
                        .chars()
                        .take(3000)
                        .collect::<String>()
                );
            }
            Ok(data)
        },
        cancel,
    )
    .await
}
async fn json_response(
    request: RequestBuilder,
    cancel: &AtomicBool,
    limit: usize,
) -> Result<Value> {
    serde_json::from_slice(&response_bytes(request, cancel, limit).await?)
        .context("Image engine returned invalid JSON")
}

type GenerationResult = (Vec<ImageArtifact>, Value, Vec<String>);

async fn generate_a1111(
    config: &ImageConfig,
    request: &ImageRequest,
    directory: &Path,
    client: &Client,
    base: &Url,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(String),
) -> Result<GenerationResult> {
    // Core Comfy/sd-cli names differ from AUTOMATIC1111's display names.
    // Preserve explicit engine names while accepting the cross-engine defaults.
    let sampler = match request.sampler.as_str() {
        "euler" => "Euler",
        "euler_a" | "euler_ancestral" => "Euler a",
        other => other,
    };
    let mut body = json!({"prompt":request.prompt,"negative_prompt":request.negative_prompt,"width":request.width,"height":request.height,"steps":request.steps,"cfg_scale":request.cfg_scale,"seed":request.seed,"batch_size":request.batch_size,"n_iter":1,"sampler_name":sampler,"send_images":true,"save_images":false});
    if !config.checkpoint.trim().is_empty() {
        body["override_settings"] = json!({"sd_model_checkpoint":config.checkpoint});
        body["override_settings_restore_afterwards"] = json!(true);
    }
    let route = if let Some(input) = &request.input_image {
        let bytes = media::read_bytes_bounded(input, media::MAX_IMAGE_BYTES)?;
        body["init_images"] = json!([base64::engine::general_purpose::STANDARD.encode(bytes)]);
        body["denoising_strength"] = json!(request.strength);
        "sdapi/v1/img2img"
    } else {
        "sdapi/v1/txt2img"
    };
    progress("Generating with AUTOMATIC1111; waiting for real image output".into());
    let response = json_response(
        client
            .post(base.join(route)?)
            .json(&body)
            .timeout(Duration::from_secs(config.timeout_secs.clamp(1, 3600))),
        cancel,
        JSON_LIMIT,
    )
    .await;
    if cancel.load(Ordering::Relaxed) {
        bail!("Cancelled the client request. AUTOMATIC1111 has no per-request cancellation API; its engine may finish this request. No global interrupt was sent to other clients.");
    }
    let response = response?;
    let encoded = response["images"]
        .as_array()
        .context("The image engine response contains no images array")?;
    if encoded.is_empty() || encoded.len() > 4 {
        bail!("The engine returned zero images or exceeded the four-image limit");
    }
    let mut images = vec![];
    for (index, image) in encoded.iter().enumerate() {
        media::check_cancel(cancel)?;
        let encoded = image.as_str().context("Invalid encoded image")?;
        let encoded = encoded
            .split_once(",")
            .filter(|(prefix, _)| prefix.starts_with("data:image/"))
            .map(|(_, bytes)| bytes)
            .unwrap_or(encoded);
        if encoded.len() > 48 * 1024 * 1024 {
            bail!("Encoded image exceeds the file limit");
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .context("Invalid image base64")?;
        images.push(save_image(directory, index, &bytes)?);
    }
    Ok((
        images,
        json!({"parameters":response.get("parameters"),"info":response.get("info")}),
        vec![],
    ))
}

async fn generate_comfy(
    config: &ImageConfig,
    request: &ImageRequest,
    directory: &Path,
    client: &Client,
    base: &Url,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(String),
) -> Result<GenerationResult> {
    let id = directory
        .file_name()
        .and_then(|name| name.to_str())
        .context("Invalid image job directory")?;
    let uploaded = if let Some(input) = &request.input_image {
        progress("Sending the selected image to the configured local image engine".into());
        let bytes = media::read_bytes_bounded(input, media::MAX_IMAGE_BYTES)?;
        let extension = image_extension(&bytes)?;
        let part =
            reqwest::multipart::Part::bytes(bytes).file_name(format!("workbench-{id}.{extension}"));
        let form = reqwest::multipart::Form::new()
            .part("image", part)
            .text("type", "input")
            .text("overwrite", "false");
        let response = json_response(
            client
                .post(base.join("upload/image")?)
                .multipart(form)
                .timeout(Duration::from_secs(30)),
            cancel,
            API_LIMIT,
        )
        .await?;
        let name = response["name"]
            .as_str()
            .context("Image upload did not return a name")?;
        let subfolder = response["subfolder"].as_str().unwrap_or("");
        safe_remote_name(name, false)?;
        safe_remote_name(subfolder, true)?;
        Some(if subfolder.is_empty() {
            name.into()
        } else {
            format!("{subfolder}/{name}")
        })
    } else {
        None
    };
    let workflow = if let Some(workflow) = &request.workflow {
        let mut graph = workflow.clone();
        substitute(
            &mut graph,
            request,
            &config.checkpoint,
            uploaded.as_deref(),
            id,
            0,
        )?;
        graph
    } else {
        default_workflow(config, request, uploaded.as_deref(), id)
    };
    validate_workflow(&workflow, config.allow_custom_nodes)?;
    media::write_new(
        &directory.join("workflow.json"),
        &serde_json::to_vec_pretty(&workflow)?,
    )?;
    if config.allow_custom_nodes && !config.allow_remote {
        for node in workflow.as_object().unwrap().values() {
            let class = node["class_type"].as_str().unwrap();
            let mut url = base.clone();
            url.path_segments_mut()
                .map_err(|_| anyhow::anyhow!("Invalid image engine URL"))?
                .pop_if_empty()
                .push("object_info")
                .push(class);
            let info = json_response(
                client.get(url).timeout(Duration::from_secs(10)),
                cancel,
                API_LIMIT,
            )
            .await?;
            if info[class]["api_node"].as_bool() == Some(true) {
                bail!("Workflow includes a remote API node ({class}); remote image engines are disabled");
            }
        }
    }
    progress("Submitting the image workflow".into());
    let submission = json_response(
        client
            .post(base.join("prompt")?)
            .json(&json!({"prompt":workflow,"client_id":id,"prompt_id":id}))
            .timeout(Duration::from_secs(30)),
        cancel,
        API_LIMIT,
    )
    .await?;
    if submission.get("error").is_some() {
        bail!("Image workflow validation failed: {submission}");
    }
    let prompt_id = submission["prompt_id"]
        .as_str()
        .context("Image engine did not return a prompt ID")?
        .to_owned();
    if prompt_id.len() > 128
        || !prompt_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        bail!("Image engine returned an invalid job ID");
    }
    let started = Instant::now();
    let result=async {
        loop {
            media::check_cancel(cancel)?;
            if started.elapsed()>=Duration::from_secs(config.timeout_secs.clamp(1,3600)){bail!("Image generation timed out");}
            let history=json_response(client.get(base.join(&format!("history/{prompt_id}"))?).timeout(Duration::from_secs(15)),cancel,API_LIMIT).await?;
            if let Some(record)=history.get(&prompt_id) {
                let status=record["status"]["status_str"].as_str().unwrap_or("");
                let messages=record["status"]["messages"].as_array();
                if status=="error"||messages.is_some_and(|items|items.iter().any(|m|m[0].as_str().is_some_and(|s|matches!(s,"execution_error"|"execution_interrupted")))) {bail!("The image workflow failed: {}",record["status"]);}
                if record["status"]["completed"].as_bool()==Some(true)||status=="success" {
                    let outputs=record["outputs"].as_object().context("Completed image workflow has no outputs")?;
                    let mut images=vec![];
                    for node in outputs.values(){if let Some(files)=node["images"].as_array(){for file in files {
                        if images.len()>=4 {bail!("Custom workflow exceeded the four-image output limit");}
                        let filename=file["filename"].as_str().context("Missing image filename")?;let subfolder=file["subfolder"].as_str().unwrap_or("");let kind=file["type"].as_str().unwrap_or("output");
                        safe_remote_name(filename,false)?;safe_remote_name(subfolder,true)?;if !matches!(kind,"output"|"temp"){bail!("Workflow output was not an engine output/temp image");}
                        let mut url=base.join("view")?;url.query_pairs_mut().append_pair("filename",filename).append_pair("subfolder",subfolder).append_pair("type",kind);
                        progress(format!("Saving generated image {}",images.len()+1));
                        let bytes=response_bytes(client.get(url).timeout(Duration::from_secs(30)),cancel,media::MAX_IMAGE_BYTES as usize).await?;
                        images.push(save_image(directory,images.len(),&bytes)?);
                    }}}
                    if images.is_empty(){bail!("The completed workflow produced no image outputs; use a SaveImage output node");}
                    let warnings=if config.allow_custom_nodes{vec!["Custom ComfyUI nodes run with the engine's normal permissions; a native client cannot sandbox installed engine plugins.".into()]}else{vec![]};
                    return Ok((images,json!({"prompt_id":prompt_id,"status":record["status"]}),warnings));
                }
            }
            progress(format!("Image job {}: waiting for engine completion ({} s)",prompt_id,started.elapsed().as_secs()));
            cancellable(async{tokio::time::sleep(Duration::from_millis(400)).await;Ok(())},cancel).await?;
        }
    }.await;
    if let Err(error) = result {
        // Current ComfyUI exposes a targeted cancellation route. Older engines
        // get a targeted queue deletion only; never interrupt somebody else's job.
        let quiet = AtomicBool::new(false);
        let targeted = response_bytes(
            client
                .post(base.join(&format!("api/jobs/{prompt_id}/cancel"))?)
                .json(&json!({}))
                .timeout(Duration::from_secs(3)),
            &quiet,
            API_LIMIT,
        )
        .await;
        if targeted.is_err() {
            let _ = response_bytes(
                client
                    .post(base.join("queue")?)
                    .json(&json!({"delete":[prompt_id]}))
                    .timeout(Duration::from_secs(3)),
                &quiet,
                API_LIMIT,
            )
            .await;
            return Err(error).context("Per-job engine cancellation was unavailable; the queued job was removed if possible, but an older engine may finish a running job");
        }
        return Err(error);
    }
    result
}

fn safe_remote_name(name: &str, subfolder: bool) -> Result<()> {
    if (!subfolder && name.is_empty())
        || name.len() > 1024
        || name.contains(['\0', '\\'])
        || name.starts_with('/')
        || (!subfolder && name.contains('/'))
        || name.split('/').any(|p| p == ".." || p == ".")
    {
        bail!("Image engine returned an unsafe file reference");
    }
    Ok(())
}

pub fn default_workflow(
    config: &ImageConfig,
    request: &ImageRequest,
    uploaded_image: Option<&str>,
    id: &str,
) -> Value {
    let mut graph = json!({
        "checkpoint":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":config.checkpoint}},
        "positive":{"class_type":"CLIPTextEncode","inputs":{"text":request.prompt,"clip":["checkpoint",1]}},
        "negative":{"class_type":"CLIPTextEncode","inputs":{"text":request.negative_prompt,"clip":["checkpoint",1]}},
        "latent":{"class_type":"EmptyLatentImage","inputs":{"width":request.width,"height":request.height,"batch_size":request.batch_size}},
        "sample":{"class_type":"KSampler","inputs":{"model":["checkpoint",0],"positive":["positive",0],"negative":["negative",0],"latent_image":["latent",0],"seed":request.seed,"steps":request.steps,"cfg":request.cfg_scale,"sampler_name":request.sampler,"scheduler":request.scheduler,"denoise":1.0}},
        "decode":{"class_type":"VAEDecode","inputs":{"samples":["sample",0],"vae":["checkpoint",2]}},
        "save":{"class_type":"SaveImage","inputs":{"images":["decode",0],"filename_prefix":format!("workbench/{id}")}}
    });
    if let Some(input) = uploaded_image {
        graph["input"] = json!({"class_type":"LoadImage","inputs":{"image":input}});
        graph["resize"] = json!({"class_type":"ImageScale","inputs":{"image":["input",0],"upscale_method":"lanczos","width":request.width,"height":request.height,"crop":"center"}});
        graph["encode"] = json!({"class_type":"VAEEncode","inputs":{"pixels":["resize",0],"vae":["checkpoint",2]}});
        graph["latent"] = json!({"class_type":"RepeatLatentBatch","inputs":{"samples":["encode",0],"amount":request.batch_size}});
        graph["sample"]["inputs"]["denoise"] = json!(request.strength);
    }
    graph
}

fn validate_workflow(workflow: &Value, custom: bool) -> Result<()> {
    let graph = workflow
        .as_object()
        .context("Use an API-format workflow object keyed by node IDs")?;
    if graph.is_empty() || graph.len() > 512 || serde_json::to_vec(workflow)?.len() > 1024 * 1024 {
        bail!("Workflow must have 1–512 nodes and be at most 1 MiB");
    }
    const CORE: &[&str] = &[
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "LoadImage",
        "ImageScale",
        "VAEEncode",
        "RepeatLatentBatch",
        "PreviewImage",
    ];
    for (id, node) in graph {
        if id.len() > 128 {
            bail!("Workflow node ID is too long");
        }
        let class = node["class_type"]
            .as_str()
            .context("Each workflow node needs class_type")?;
        if !node["inputs"].is_object() {
            bail!("Each workflow node needs an inputs object");
        }
        if !custom && !CORE.contains(&class) {
            bail!("Workflow node {class} requires explicitly enabling trusted custom workflows");
        }
    }
    Ok(())
}

fn substitute(
    value: &mut Value,
    request: &ImageRequest,
    checkpoint: &str,
    input: Option<&str>,
    id: &str,
    depth: usize,
) -> Result<()> {
    if depth > 48 {
        bail!("Workflow nesting exceeds 48 levels");
    }
    match value {
        Value::Object(map) => {
            for child in map.values_mut() {
                substitute(child, request, checkpoint, input, id, depth + 1)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                substitute(child, request, checkpoint, input, id, depth + 1)?;
            }
        }
        Value::String(text) => {
            let replacement = match text.as_str() {
                "{{prompt}}" => Some(json!(request.prompt)),
                "{{negative_prompt}}" => Some(json!(request.negative_prompt)),
                "{{checkpoint}}" => Some(json!(checkpoint)),
                "{{seed}}" => Some(json!(request.seed)),
                "{{width}}" => Some(json!(request.width)),
                "{{height}}" => Some(json!(request.height)),
                "{{steps}}" => Some(json!(request.steps)),
                "{{cfg_scale}}" => Some(json!(request.cfg_scale)),
                "{{batch_size}}" => Some(json!(request.batch_size)),
                "{{strength}}" => Some(json!(request.strength)),
                "{{input_image}}" => Some(json!(
                    input.context("This workflow requires an input image")?
                )),
                "{{output_prefix}}" => Some(json!(format!("workbench/{id}"))),
                _ => None,
            };
            if let Some(replacement) = replacement {
                *value = replacement;
            }
        }
        _ => {}
    }
    Ok(())
}

fn generate_cpp(
    config: &ImageConfig,
    request: &ImageRequest,
    directory: &Path,
    cancel: &Arc<AtomicBool>,
    progress: &mut impl FnMut(String),
) -> Result<GenerationResult> {
    let model = media::regular_input(&config.sd_model, 64 * 1024 * 1024 * 1024)?;
    let mut images = vec![];
    let mut commands = vec![];
    for index in 0..request.batch_size {
        media::check_cancel(cancel)?;
        let output = directory.join(format!("image-{:03}.png", index + 1));
        let seed = request
            .seed
            .checked_add(index as i64)
            .context("Image batch seed overflow")?;
        let mut args = config.sd_extra_args.clone();
        args.extend([
            "-m".into(),
            media::path_text(&model)?,
            "-p".into(),
            request.prompt.clone(),
            "-n".into(),
            request.negative_prompt.clone(),
            "-W".into(),
            request.width.to_string(),
            "-H".into(),
            request.height.to_string(),
            "--steps".into(),
            request.steps.to_string(),
            "--cfg-scale".into(),
            request.cfg_scale.to_string(),
            "--sampling-method".into(),
            request.sampler.clone(),
            "-s".into(),
            seed.to_string(),
            "-o".into(),
            media::path_text(&output)?,
        ]);
        let command = CommandSpec {
            program: config.sd_program.clone(),
            args,
            cwd: Some(directory.into()),
            timeout_secs: config.timeout_secs.clamp(1, 3600),
            max_output_bytes: 512 * 1024,
            ..Default::default()
        };
        progress(format!(
            "Generating local image {} of {} with sd-cli",
            index + 1,
            request.batch_size
        ));
        let result = run_command_cancellable(&command, directory, Some(cancel.clone()))?;
        commands.push(json!({"command":command,"result":result}));
        media::write_new(
            &directory.join(format!("engine-{:03}.json", index + 1)),
            &serde_json::to_vec_pretty(&commands.last())?,
        )?;
        if result.cancelled {
            bail!("Image generation cancelled; sd-cli process group stopped");
        }
        if result.timed_out {
            bail!("Image generation timed out; sd-cli process group stopped");
        }
        if result.exit_code != Some(0) {
            bail!(
                "sd-cli failed: {}",
                result.stderr.chars().take(4000).collect::<String>()
            );
        }
        let bytes = media::read_bytes_bounded(&output, media::MAX_IMAGE_BYTES)?;
        let (width, height) = validate_image_bytes(&bytes)?;
        images.push(ImageArtifact {
            path: output,
            width,
            height,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        });
    }
    Ok((images, json!({"commands":commands}), vec![]))
}

fn image_extension(bytes: &[u8]) -> Result<&'static str> {
    Ok(match image::guess_format(bytes)? {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpg",
        image::ImageFormat::WebP => "webp",
        image::ImageFormat::Gif => "gif",
        _ => bail!("The engine image format is unsupported"),
    })
}
fn validate_image_bytes(bytes: &[u8]) -> Result<(u32, u32)> {
    let dimensions = media::image_dimensions(bytes)?;
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .context("The engine returned damaged image data")?;
    Ok(dimensions)
}
fn save_image(directory: &Path, index: usize, bytes: &[u8]) -> Result<ImageArtifact> {
    let (width, height) = validate_image_bytes(bytes)?;
    let path = directory.join(format!(
        "image-{:03}.{}",
        index + 1,
        image_extension(bytes)?
    ));
    media::write_new(&path, bytes)?;
    Ok(ImageArtifact {
        path,
        width,
        height,
        sha256: format!("{:x}", Sha256::digest(bytes)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    fn png() -> Vec<u8> {
        let mut out = Cursor::new(vec![]);
        image::DynamicImage::new_rgb8(2, 3)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }
    fn server(responses: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut requests = vec![];
            for (mime, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = vec![];
                let end;
                loop {
                    let mut b = [0u8; 1];
                    stream.read_exact(&mut b).unwrap();
                    bytes.push(b[0]);
                    if bytes.ends_with(b"\r\n\r\n") {
                        end = bytes.len();
                        break;
                    }
                }
                let header = String::from_utf8_lossy(&bytes).to_string();
                let length = header
                    .lines()
                    .find_map(|l| {
                        l.to_lowercase()
                            .strip_prefix("content-length:")
                            .map(|n| n.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                bytes.resize(end + length, 0);
                stream.read_exact(&mut bytes[end..]).unwrap();
                requests.push(String::from_utf8_lossy(&bytes).into());
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                stream.write_all(&body).unwrap();
            }
            requests
        });
        (format!("http://{address}"), handle)
    }

    #[test]
    fn a1111_saves_actual_image_and_records_request_parameters() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(png());
        let (url, server) = server(vec![(
            "application/json".into(),
            serde_json::to_vec(&json!({"images":[encoded],"info":"fixture"})).unwrap(),
        )]);
        let dir = tempfile::tempdir().unwrap();
        let config = ImageConfig {
            backend: ImageBackend::Automatic1111,
            endpoint: url,
            ..Default::default()
        };
        let request = ImageRequest {
            prompt: "A test image".into(),
            seed: 42,
            ..Default::default()
        };
        let output = generate(
            &config,
            &request,
            dir.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.images.len(), 1);
        assert_eq!((output.images[0].width, output.images[0].height), (2, 3));
        assert_eq!(std::fs::read(&output.images[0].path).unwrap(), png());
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("POST /sdapi/v1/txt2img"));
        assert!(requests[0].contains("\"seed\":42"));
        assert!(requests[0].contains("\"sampler_name\":\"Euler\""));
    }

    #[test]
    fn comfy_requires_success_before_downloading_its_own_images() {
        let prompt_id = "fixture-job";
        let history = json!({prompt_id:{"status":{"completed":true,"status_str":"success"},"outputs":{"save":{"images":[{"filename":"result.png","subfolder":"workbench","type":"output"}]}}}});
        let (url, server) = server(vec![
            (
                "application/json".into(),
                serde_json::to_vec(&json!({"prompt_id":prompt_id})).unwrap(),
            ),
            (
                "application/json".into(),
                serde_json::to_vec(&history).unwrap(),
            ),
            ("image/png".into(), png()),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let config = ImageConfig {
            endpoint: url,
            checkpoint: "fixture.safetensors".into(),
            ..Default::default()
        };
        let request = ImageRequest {
            prompt: "A test image".into(),
            seed: 4,
            ..Default::default()
        };
        let output = generate(
            &config,
            &request,
            dir.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.images.len(), 1);
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("POST /prompt"));
        assert!(requests[1].starts_with("GET /history/fixture-job"));
        assert!(requests[2].contains("filename=result.png"));
    }

    #[test]
    fn remote_and_untrusted_workflows_are_explicit() {
        let request = ImageRequest {
            prompt: "A picture".into(),
            ..Default::default()
        };
        let config = ImageConfig {
            endpoint: "https://example.com".into(),
            checkpoint: "model".into(),
            ..Default::default()
        };
        assert!(preview(&config, &request).is_err());
        assert!(
            validate_workflow(&json!({"1":{"class_type":"RunShell","inputs":{}}}), false).is_err()
        );
        assert!(safe_remote_name("../../secret", true).is_err());
        let mut workflow = json!({"prompt":"{{prompt}}","seed":"{{seed}}"});
        substitute(&mut workflow, &request, "model", None, "id", 0).unwrap();
        assert_eq!(workflow["prompt"], request.prompt);
        assert!(workflow["seed"].is_i64());
    }

    #[test]
    fn invalid_engine_images_never_report_success() {
        let dir = tempfile::tempdir().unwrap();
        assert!(save_image(dir.path(), 0, b"not a generated image").is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
