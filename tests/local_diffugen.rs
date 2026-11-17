//! Explicit real-engine acceptance for the user's bundled DiffuGen project.
use ollama_workbench::{
    image_generation::{self, ImageBackend, ImageConfig, ImageRequest},
    mcp::{McpClient, McpServerConfig},
};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

fn generated(client: &mut McpClient, name: &str, args: Value) -> PathBuf {
    let result = client.call_tool(name, args).unwrap();
    assert_ne!(result["isError"], json!(true), "{result}");
    let value = result
        .get("structuredContent")
        .cloned()
        .or_else(|| {
            result["content"]
                .as_array()?
                .iter()
                .find_map(|c| serde_json::from_str::<Value>(c["text"].as_str()?).ok())
        })
        .expect("DiffuGen must return a structured generation result");
    assert_eq!(value["success"], json!(true), "{value}");
    let path = PathBuf::from(value["image_path"].as_str().unwrap());
    assert_eq!(image::image_dimensions(&path).unwrap(), (512, 512));
    eprintln!("{name}: {}", path.display());
    path
}

#[test]
#[ignore = "requires copied DiffuGen runtime/models and a usable Vulkan device"]
fn bundled_diffugen_works_in_studio() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bundle = root.join("DiffuGen");
    let data = std::env::var("WORKBENCH_DIFFUGEN_ACCEPTANCE_DIR")
        .map(PathBuf::from)
        .expect("set WORKBENCH_DIFFUGEN_ACCEPTANCE_DIR for persistent test outputs");
    std::fs::create_dir_all(&data).unwrap();
    let native = image_generation::generate(
        &ImageConfig {
            backend: ImageBackend::StableDiffusionCpp,
            sd_program: bundle.join("runtime/vulkan/sd-cli").display().to_string(),
            sd_model: bundle.join("stable-diffusion.cpp/models/sd15.safetensors"),
            ..Default::default()
        },
        &ImageRequest {
            prompt: "a small red cabin in a snowy pine forest, watercolor illustration".into(),
            steps: 12,
            seed: 1401,
            ..Default::default()
        },
        &data,
        Arc::new(AtomicBool::new(false)),
        |s| eprintln!("{s}"),
    )
    .unwrap();
    assert_eq!(native.images.len(), 1);
    assert_eq!(
        (native.images[0].width, native.images[0].height),
        (512, 512)
    );
    eprintln!("Studio output: {}", native.images[0].path.display());
}

#[test]
#[ignore = "requires copied DiffuGen runtime/models, Python dependencies and Vulkan"]
fn bundled_diffugen_mcp_generates_and_edits() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bundle = root.join("DiffuGen");
    let data = std::env::var("WORKBENCH_DIFFUGEN_ACCEPTANCE_DIR")
        .map(PathBuf::from)
        .expect("set WORKBENCH_DIFFUGEN_ACCEPTANCE_DIR for persistent test outputs");
    std::fs::create_dir_all(&data).unwrap();
    let config: McpServerConfig = serde_json::from_value(json!({
        "id":"diffugen", "name":"DiffuGen", "program":bundle.join("run_diffugen.sh"),
        "args":[], "cwd":bundle, "env":{"DIFFUGEN_OUTPUT_DIR":data},
        "enabled":true, "timeout_secs":600
    }))
    .unwrap();
    let mut client = McpClient::connect(&config, root).unwrap();
    let names: Vec<_> = client
        .refresh_tools()
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    eprintln!("MCP tools: {names:?}");
    for name in [
        "generate_stable_diffusion_image",
        "generate_img2img_image",
        "generate_inpaint_image",
    ] {
        assert!(names.iter().any(|n| n == name));
    }
    let source = generated(
        &mut client,
        "generate_stable_diffusion_image",
        json!({
            "prompt":"a small red cabin in a snowy pine forest, watercolor illustration", "model":"sd15",
            "output_dir":data, "width":512, "height":512, "steps":12, "seed":1402
        }),
    );
    let original = std::fs::read(&source).unwrap();
    generated(
        &mut client,
        "generate_img2img_image",
        json!({
            "prompt":"a small red cabin in a snowy pine forest at sunrise, watercolor illustration",
            "init_image":source, "output_dir":data, "width":512, "height":512, "steps":12,
            "strength":0.5, "seed":1403
        }),
    );
    let mask_path = data.join("acceptance-mask.png");
    let mask = image::GrayImage::from_fn(512, 512, |x, y| {
        image::Luma([if (180..332).contains(&x) && (180..332).contains(&y) {
            255
        } else {
            0
        }])
    });
    mask.save(&mask_path).unwrap();
    generated(
        &mut client,
        "generate_inpaint_image",
        json!({
            "prompt":"a small blue cabin, watercolor illustration", "init_image":source,
            "mask_image":mask_path, "output_dir":data, "width":512, "height":512,
            "steps":12, "strength":0.75, "seed":1404
        }),
    );
    assert_eq!(std::fs::read(&source).unwrap(), original);
    client.shutdown();
}

#[test]
#[ignore = "reads the local Workbench configuration and starts its configured DiffuGen server"]
fn saved_workbench_diffugen_connections_are_usable() {
    let dir = ollama_workbench::settings::data_dir().unwrap();
    let settings = ollama_workbench::settings::Settings::load(&dir).unwrap();
    assert!(settings.mcp_enabled);
    let servers: Vec<McpServerConfig> = serde_json::from_str(&settings.mcp_json).unwrap();
    let server = servers
        .iter()
        .find(|s| s.id == "diffugen" && s.enabled)
        .unwrap();
    let mut client = McpClient::connect(server, Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
    let names: Vec<_> = client
        .refresh_tools()
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    for name in [
        "generate_stable_diffusion_image",
        "generate_img2img_image",
        "generate_inpaint_image",
    ] {
        assert!(names.iter().any(|n| n == name));
    }
    client.shutdown();
    let studio: Value =
        serde_json::from_slice(&std::fs::read(dir.join("studio-settings.json")).unwrap()).unwrap();
    let config: ImageConfig = serde_json::from_value(studio["image"].clone()).unwrap();
    assert_eq!(config.backend, ImageBackend::StableDiffusionCpp);
    assert_eq!(
        image_generation::list_models(&config, Arc::new(AtomicBool::new(false)))
            .unwrap()
            .len(),
        1
    );
    image_generation::preview(
        &config,
        &ImageRequest {
            prompt: "configuration acceptance".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let media: ollama_workbench::media::MediaTools =
        serde_json::from_value(studio["media"].clone()).unwrap();
    assert!(
        media.whisper_model.is_file(),
        "image setup must preserve the transcription model"
    );
}
