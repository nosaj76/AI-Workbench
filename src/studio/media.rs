//! Explicit, local document/media activities. Each activity owns a fresh output
//! directory, records its actual outputs, and never overwrites its input.
//! External programs are launched as direct argument arrays, not shell strings.

use crate::integrations::{run_command_cancellable, CommandResult, CommandSpec};
use anyhow::{bail, Context, Result};
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{Cursor, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub(crate) const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_IMAGE_PIXELS: u64 = 16_777_216;
const MAX_MEDIA_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaTools {
    pub ffmpeg: String,
    pub ffprobe: String,
    pub pdftoppm: String,
    pub pdfinfo: String,
    pub tesseract: String,
    pub whisper: String,
    pub whisper_model: PathBuf,
    pub espeak: String,
    pub piper: String,
    pub piper_model: PathBuf,
    pub pdflatex: String,
    pub rembg: String,
    pub rembg_model_directory: PathBuf,
    pub timeout_secs: u64,
}

impl Default for MediaTools {
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            pdftoppm: "pdftoppm".into(),
            pdfinfo: "pdfinfo".into(),
            tesseract: "tesseract".into(),
            whisper: "whisper-cli".into(),
            whisper_model: PathBuf::new(),
            espeak: "espeak-ng".into(),
            piper: "piper".into(),
            piper_model: PathBuf::new(),
            pdflatex: "pdflatex".into(),
            rembg: "rembg".into(),
            rembg_model_directory: PathBuf::new(),
            timeout_secs: 900,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpeechEngine {
    #[default]
    Espeak,
    Piper,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "activity", rename_all = "snake_case")]
pub enum MediaRequest {
    PdfOcr {
        input: PathBuf,
        first_page: u32,
        last_page: u32,
        dpi: u32,
        language: String,
    },
    ImageOcr {
        input: PathBuf,
        language: String,
    },
    Transcribe {
        input: PathBuf,
        language: String,
        translate_to_english: bool,
        max_seconds: u32,
    },
    VideoFrames {
        input: PathBuf,
        every_seconds: f64,
        max_frames: u32,
    },
    Speech {
        text: String,
        engine: SpeechEngine,
        voice: String,
        words_per_minute: u32,
    },
    Podcast {
        script: String,
        engine: SpeechEngine,
        host_voice: String,
        guest_voice: String,
        words_per_minute: u32,
    },
    RemoveBackground {
        input: PathBuf,
        color: Option<[u8; 3]>,
        tolerance: u8,
        feather: u8,
        semantic_model: bool,
    },
    LatexPdf {
        source: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityOutput {
    pub id: String,
    pub directory: PathBuf,
    pub text: String,
    pub files: Vec<PathBuf>,
    pub warnings: Vec<String>,
    pub metadata: Value,
}

impl ActivityOutput {
    pub(crate) fn new(directory: PathBuf) -> Self {
        Self {
            id: directory
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            directory,
            text: String::new(),
            files: vec![],
            warnings: vec![],
            metadata: json!({}),
        }
    }

    pub(crate) fn finish(mut self) -> Result<Self> {
        let path = self.directory.join("activity.json");
        write_new(&path, &serde_json::to_vec_pretty(&self)?)?;
        self.files.push(path);
        Ok(self)
    }
}

/// A concrete description for the native approval/launch panel. This does not
/// run commands, read the input contents, or create an output directory.
pub fn preview(tools: &MediaTools, request: &MediaRequest) -> Result<Value> {
    let programs = match request {
        MediaRequest::PdfOcr {
            input,
            first_page,
            last_page,
            dpi,
            language,
        } => {
            regular_input(input, 128 * 1024 * 1024)?;
            if *first_page == 0
                || last_page < first_page
                || last_page - first_page >= 100
                || !(50..=300).contains(dpi)
            {
                bail!("Choose 1–100 PDF pages and a resolution of 50–300 DPI");
            }
            language_name(language)?;
            vec![tools.pdfinfo.clone(), tools.pdftoppm.clone(), tools.tesseract.clone()]
        }
        MediaRequest::ImageOcr { input, language } => {
            regular_input(input, MAX_IMAGE_BYTES)?;
            language_name(language)?;
            vec![tools.tesseract.clone()]
        }
        MediaRequest::Transcribe {
            input,
            language,
            max_seconds,
            ..
        } => {
            regular_input(input, MAX_MEDIA_BYTES)?;
            regular_input(&tools.whisper_model, MAX_MEDIA_BYTES)
                .context("Choose an installed Whisper model")?;
            language_name(language)?;
            if !(1..=21_600).contains(max_seconds) {
                bail!("Audio duration must be 1–21,600 seconds");
            }
            vec![tools.ffmpeg.clone(), tools.whisper.clone()]
        }
        MediaRequest::VideoFrames {
            input,
            every_seconds,
            max_frames,
        } => {
            regular_input(input, MAX_MEDIA_BYTES)?;
            if !every_seconds.is_finite()
                || !(0.1..=3600.0).contains(every_seconds)
                || !(1..=120).contains(max_frames)
            {
                bail!("Choose a 0.1–3,600 second interval and 1–120 frames");
            }
            vec![tools.ffprobe.clone(), tools.ffmpeg.clone()]
        }
        MediaRequest::Speech {
            text,
            engine,
            voice,
            words_per_minute,
        } => {
            speech_parameters(tools, text, *engine, voice, *words_per_minute)?;
            vec![speech_program(tools, *engine).into()]
        }
        MediaRequest::Podcast {
            script,
            engine,
            host_voice,
            guest_voice,
            words_per_minute,
        } => {
            speech_parameters(tools, script, *engine, host_voice, *words_per_minute)?;
            voice_name(guest_voice)?;
            parse_podcast_script(script)?;
            vec![speech_program(tools, *engine).into(), tools.ffmpeg.clone()]
        }
        MediaRequest::RemoveBackground {
            input,
            semantic_model,
            ..
        } => {
            image_input(input)?;
            if *semantic_model {
                regular_input(&tools.rembg_model_directory.join("u2net.onnx"), MAX_MEDIA_BYTES)
                    .context("Semantic cutout requires a local u2net.onnx in the configured rembg model directory")?;
                vec![tools.rembg.clone()]
            } else {
                vec![]
            }
        }
        MediaRequest::LatexPdf { source } => {
            if source.trim().is_empty() || source.len() > 256 * 1024 {
                bail!("LaTeX source must contain 1–262,144 bytes");
            }
            vec![tools.pdflatex.clone()]
        }
    };
    if programs
        .iter()
        .any(|p| p.trim().is_empty() || p.contains('\0'))
    {
        bail!("Configure every required executable");
    }
    Ok(
        json!({"request":request,"programs":programs,"outputs":"Fresh directory under the Workbench data directory; source files are not replaced", "timeout_per_command_seconds":tools.timeout_secs.clamp(1, 3600),"execution":"Local executables have normal account permissions. LaTeX shell escape is disabled."}),
    )
}

/// Call after the user has selected/reviewed this activity. It is deliberately a
/// blocking worker API so the desktop can run it off the UI thread.
pub fn run(
    tools: &MediaTools,
    request: &MediaRequest,
    data_dir: &Path,
    cancel: Arc<AtomicBool>,
    mut progress: impl FnMut(String),
) -> Result<ActivityOutput> {
    preview(tools, request)?;
    check_cancel(&cancel)?;
    let dir = create_run_dir(data_dir, "media")?;
    write_new(
        &dir.join("request.json"),
        &serde_json::to_vec_pretty(request)?,
    )?;
    let mut output = ActivityOutput::new(dir.clone());
    output.files.push(dir.join("request.json"));
    let mut commands = Vec::<Value>::new();
    let result = (|| -> Result<()> {
        match request {
            MediaRequest::PdfOcr {
                input,
                first_page,
                last_page,
                dpi,
                language,
            } => {
                let input = regular_input(input, 128 * 1024 * 1024)?;
                progress(format!("Rendering PDF pages {first_page}–{last_page}"));
                let info = execute(
                    tools, &dir, &cancel, &mut commands, &tools.pdfinfo,
                    vec![
                        "-f".into(), first_page.to_string(),
                        "-l".into(), last_page.to_string(),
                        "-box".into(), path_text(&input)?,
                    ],
                    None,
                )?;
                let mut pages = Vec::new();
                let mut total_bytes = 0;
                for line in info.stdout.lines() {
                    let fields: Vec<_> = line.split_whitespace().collect();
                    let ["Page", page, "MediaBox:", left, bottom, right, top] = fields.as_slice() else {
                        continue;
                    };
                    let number: u32 = page.parse()?;
                    if number < *first_page || number > *last_page || pages.len() >= 100 {
                        bail!("The PDF inspector returned an unexpected page number: {number}");
                    }
                    let width = right.parse::<f64>()? - left.parse::<f64>()?;
                    let height = top.parse::<f64>()? - bottom.parse::<f64>()?;
                    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
                        bail!("The PDF inspector reported invalid dimensions for page {number}");
                    }
                    // Poppler's -scale-to overrides -r. Calculate the requested
                    // raster size per page, then cap the longest side before rendering.
                    let longest_side = width.max(height) * f64::from(*dpi) / 72.0;
                    let pixels = longest_side.ceil().clamp(1.0, 2000.0) as u32;
                    let prefix = dir.join(format!("page-{number}"));
                    execute(
                        tools, &dir, &cancel, &mut commands, &tools.pdftoppm,
                        vec![
                            "-png".into(), "-singlefile".into(), "-scale-to".into(), pixels.to_string(),
                            "-f".into(), number.to_string(), "-l".into(), number.to_string(),
                            path_text(&input)?, path_text(&prefix)?,
                        ],
                        None,
                    )?;
                    let image = prefix.with_extension("png");
                    total_bytes += fs::metadata(&image)?.len();
                    if total_bytes > 512 * 1024 * 1024 {
                        bail!("Rendered PDF pages exceed 512 MiB");
                    }
                    pages.push((number, image));
                }
                if pages.is_empty() {
                    bail!("The PDF renderer did not produce any requested pages");
                }
                for (number, image) in &pages {
                    check_cancel(&cancel)?;
                    image_input(image)?;
                    progress(format!("Recognizing PDF page {number}"));
                    let text = ocr(tools, &dir, image, language, &cancel, &mut commands)?;
                    output.text.push_str(&format!(
                        "\n\n## {} — page {}\n\n{}",
                        input.file_name().unwrap_or_default().to_string_lossy(),
                        number,
                        text.trim()
                    ));
                    output.files.push(image.clone());
                    if output.text.len() > 8 * 1024 * 1024 {
                        bail!("OCR text exceeds the 8 MiB activity limit");
                    }
                }
                if pages.len() != (last_page - first_page + 1) as usize {
                    output.warnings.push("Only pages actually produced by the renderer were recognized; the requested range may extend beyond the PDF.".into());
                }
                output.metadata = json!({"pages":pages.iter().map(|(page,path)|json!({"page":page,"image":path})).collect::<Vec<_>>(),"citation_units":"original PDF page numbers"});
                let text_path = dir.join("ocr.md");
                write_new(&text_path, output.text.as_bytes())?;
                output.files.push(text_path);
            }
            MediaRequest::ImageOcr { input, language } => {
                let input = regular_input(input, MAX_IMAGE_BYTES)?;
                progress("Recognizing image text".into());
                output.text = ocr(tools, &dir, &input, language, &cancel, &mut commands)?;
                let text_path = dir.join("ocr.txt");
                write_new(&text_path, output.text.as_bytes())?;
                output.files.push(text_path);
            }
            MediaRequest::Transcribe {
                input,
                language,
                translate_to_english,
                max_seconds,
            } => {
                let input = regular_input(input, MAX_MEDIA_BYTES)?;
                let model = regular_input(&tools.whisper_model, MAX_MEDIA_BYTES)?;
                let wave = dir.join("audio.wav");
                progress("Converting audio locally to 16 kHz mono WAV".into());
                execute(
                    tools,
                    &dir,
                    &cancel,
                    &mut commands,
                    &tools.ffmpeg,
                    vec![
                        "-nostdin".into(),
                        "-v".into(),
                        "error".into(),
                        "-protocol_whitelist".into(),
                        "file,pipe".into(),
                        "-i".into(),
                        path_text(&input)?,
                        "-map".into(),
                        "0:a:0".into(),
                        "-t".into(),
                        max_seconds.to_string(),
                        "-ac".into(),
                        "1".into(),
                        "-ar".into(),
                        "16000".into(),
                        "-c:a".into(),
                        "pcm_s16le".into(),
                        "-n".into(),
                        path_text(&wave)?,
                    ],
                    None,
                )?;
                require_wave(&wave)?;
                progress("Transcribing with the installed Whisper model".into());
                let prefix = dir.join("transcript");
                let mut args = vec![
                    "-m".into(),
                    path_text(&model)?,
                    "-f".into(),
                    path_text(&wave)?,
                    "-l".into(),
                    language.clone(),
                    "-otxt".into(),
                    "-osrt".into(),
                    "-oj".into(),
                    "-of".into(),
                    path_text(&prefix)?,
                ];
                if *translate_to_english {
                    args.push("-tr".into());
                }
                execute(
                    tools,
                    &dir,
                    &cancel,
                    &mut commands,
                    &tools.whisper,
                    args,
                    None,
                )?;
                output.text = read_text_bounded(&dir.join("transcript.txt"), 4 * 1024 * 1024)?;
                if output.text.trim().is_empty() {
                    bail!("Transcription produced no text");
                }
                output.files.push(wave);
                for name in ["transcript.txt", "transcript.srt", "transcript.json"] {
                    let p = dir.join(name);
                    if p.exists() {
                        regular_input(&p, 16 * 1024 * 1024)?;
                        output.files.push(p);
                    }
                }
                output.warnings.push(format!("Audio processing was limited to the first {max_seconds} seconds. Transcription accuracy depends on the selected local model."));
                output.metadata = json!({"source":input,"language":language,"translation":translate_to_english,"maximum_seconds":max_seconds});
            }
            MediaRequest::VideoFrames {
                input,
                every_seconds,
                max_frames,
            } => {
                let input = regular_input(input, MAX_MEDIA_BYTES)?;
                progress("Reading video duration".into());
                let info = execute(
                    tools,
                    &dir,
                    &cancel,
                    &mut commands,
                    &tools.ffprobe,
                    vec![
                        "-v".into(),
                        "error".into(),
                        "-protocol_whitelist".into(),
                        "file,pipe".into(),
                        "-show_entries".into(),
                        "format=duration".into(),
                        "-of".into(),
                        "json".into(),
                        path_text(&input)?,
                    ],
                    None,
                )?;
                let info: Value =
                    serde_json::from_str(&info.stdout).context("ffprobe returned invalid JSON")?;
                let duration = info["format"]["duration"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|d| d.is_finite() && *d > 0.0)
                    .context("The selected file has no finite video duration")?;
                let mut frames = vec![];
                for index in 0..*max_frames {
                    check_cancel(&cancel)?;
                    let time = *every_seconds * index as f64;
                    if time >= duration {
                        break;
                    }
                    progress(format!(
                        "Extracting frame {} at {:.2} seconds",
                        index + 1,
                        time
                    ));
                    let path = dir.join(format!("frame-{:04}.png", index + 1));
                    execute(
                        tools,
                        &dir,
                        &cancel,
                        &mut commands,
                        &tools.ffmpeg,
                        vec![
                            "-nostdin".into(),
                            "-v".into(),
                            "error".into(),
                            "-protocol_whitelist".into(),
                            "file,pipe".into(),
                            "-ss".into(),
                            format!("{time:.6}"),
                            "-i".into(),
                            path_text(&input)?,
                            "-map".into(),
                            "0:v:0".into(),
                            "-frames:v".into(),
                            "1".into(),
                            "-vf".into(),
                            "scale=1280:1280:force_original_aspect_ratio=decrease".into(),
                            "-n".into(),
                            path_text(&path)?,
                        ],
                        None,
                    )?;
                    // FFmpeg exits successfully without writing a file when the
                    // selected video stream ends before the audio/container.
                    if !path.try_exists()? {
                        break;
                    }
                    image_input(&path)?;
                    frames.push(json!({"path":path,"requested_timestamp_seconds":time}));
                    output.files.push(path);
                }
                if frames.is_empty() {
                    bail!("No video frames were produced");
                }
                output.text = format!("Extracted {} sampled frames from {} (duration {:.2} seconds), at requested {}-second intervals. Samples do not constitute a complete video review.",frames.len(),input.display(),duration,every_seconds);
                output.metadata =
                    json!({"source":input,"duration_seconds":duration,"frames":frames});
                let path = dir.join("frames.json");
                write_new(&path, &serde_json::to_vec_pretty(&output.metadata)?)?;
                output.files.push(path);
            }
            MediaRequest::Speech {
                text,
                engine,
                voice,
                words_per_minute,
            } => {
                progress("Synthesizing a local audio file".into());
                let wave = dir.join("speech.wav");
                speak(
                    tools,
                    text,
                    *engine,
                    voice,
                    *words_per_minute,
                    &wave,
                    &cancel,
                    &mut commands,
                )?;
                let script = dir.join("speech.txt");
                write_new(&script, text.as_bytes())?;
                output.text = text.clone();
                output.files.extend([wave, script]);
            }
            MediaRequest::Podcast {
                script,
                engine,
                host_voice,
                guest_voice,
                words_per_minute,
            } => {
                let turns = parse_podcast_script(script)?;
                let mut segments = vec![];
                for (index, turn) in turns.iter().enumerate() {
                    check_cancel(&cancel)?;
                    progress(format!(
                        "Synthesizing podcast turn {} of {}",
                        index + 1,
                        turns.len()
                    ));
                    let wave = dir.join(format!("turn-{:03}.wav", index + 1));
                    let voice = if turn.speaker == "GUEST" {
                        guest_voice
                    } else {
                        host_voice
                    };
                    speak(
                        tools,
                        &turn.text,
                        *engine,
                        voice,
                        *words_per_minute,
                        &wave,
                        &cancel,
                        &mut commands,
                    )?;
                    segments.push(wave);
                }
                let episode = dir.join("podcast.wav");
                if segments.len() == 1 {
                    let bytes = read_bytes_bounded(&segments[0], 256 * 1024 * 1024)?;
                    write_new(&episode, &bytes)?;
                } else {
                    progress("Joining podcast audio segments".into());
                    let mut args = vec!["-nostdin".into(), "-v".into(), "error".into()];
                    for segment in &segments {
                        args.extend(["-i".into(), path_text(segment)?]);
                    }
                    let inputs = (0..segments.len())
                        .map(|i| format!("[{i}:a]"))
                        .collect::<String>();
                    args.extend([
                        "-filter_complex".into(),
                        format!("{inputs}concat=n={}:v=0:a=1[episode]", segments.len()),
                        "-map".into(),
                        "[episode]".into(),
                        "-c:a".into(),
                        "pcm_s16le".into(),
                        "-n".into(),
                        path_text(&episode)?,
                    ]);
                    execute(
                        tools,
                        &dir,
                        &cancel,
                        &mut commands,
                        &tools.ffmpeg,
                        args,
                        None,
                    )?;
                }
                require_wave(&episode)?;
                let script_path = dir.join("podcast-script.txt");
                write_new(&script_path, script.as_bytes())?;
                output.text = script.clone();
                output.files.extend([episode, script_path]);
                output.files.extend(segments);
                output.metadata = json!({"turns":turns,"engine":engine,"host_voice":host_voice,"guest_voice":guest_voice});
                if *engine == SpeechEngine::Piper {
                    output.warnings.push("This Piper configuration uses one installed voice model for both speakers. Choose eSpeak voices or configure separate models externally for different voices.".into());
                }
            }
            MediaRequest::RemoveBackground {
                input,
                color,
                tolerance,
                feather,
                semantic_model,
            } => {
                let input = regular_input(input, MAX_IMAGE_BYTES)?;
                let path = dir.join("cutout.png");
                progress("Removing the image background".into());
                if *semantic_model {
                    let mut spec = command(
                        tools,
                        &dir,
                        &tools.rembg,
                        vec![
                            "i".into(),
                            "-m".into(),
                            "u2net".into(),
                            path_text(&input)?,
                            path_text(&path)?,
                        ],
                        None,
                    );
                    spec.env.insert(
                        "U2NET_HOME".into(),
                        path_text(&tools.rembg_model_directory.canonicalize()?)?,
                    );
                    execute_spec(&spec, &dir, &cancel, &mut commands)?;
                    image_input(&path)?;
                } else {
                    let image = load_image(&input)?;
                    let cutout =
                        remove_border_background(image, *color, *tolerance, *feather, &cancel)?;
                    let mut encoded = Cursor::new(Vec::new());
                    cutout.write_to(&mut encoded, ImageFormat::Png)?;
                    write_new(&path, &encoded.into_inner())?;
                    output.warnings.push("Native color cutout removes matching background connected to the image border. It is intended for plain backgrounds; it is not semantic subject segmentation.".into());
                }
                output.text = format!("Saved transparent-background cutout: {}", path.display());
                output.files.push(path);
            }
            MediaRequest::LatexPdf { source } => {
                let source_path = dir.join("document.tex");
                write_new(&source_path, source.as_bytes())?;
                for pass in 1..=2 {
                    progress(format!("Compiling LaTeX, pass {pass}/2"));
                    let mut spec = command(
                        tools,
                        &dir,
                        &tools.pdflatex,
                        vec![
                            "-no-shell-escape".into(),
                            "-interaction=nonstopmode".into(),
                            "-halt-on-error".into(),
                            "-file-line-error".into(),
                            "-output-directory".into(),
                            ".".into(),
                            // Restricted TeX input/output policies reject absolute paths.
                            // The command already runs inside this activity directory.
                            "document.tex".into(),
                        ],
                        None,
                    );
                    spec.env.insert("openin_any".into(), "p".into());
                    spec.env.insert("openout_any".into(), "p".into());
                    execute_spec(&spec, &dir, &cancel, &mut commands)?;
                }
                let pdf = dir.join("document.pdf");
                let mut header = [0; 5];
                File::open(regular_input(&pdf, 128 * 1024 * 1024)?)?.read_exact(&mut header)?;
                if &header != b"%PDF-" {
                    bail!("LaTeX did not produce a PDF file");
                }
                output.text = "Compiled LaTeX source to PDF with shell escape disabled.".into();
                output.files.extend([source_path, pdf]);
            }
        }
        check_cancel(&cancel)?;
        Ok(())
    })();
    let log = dir.join("commands.json");
    write_new(&log, &serde_json::to_vec_pretty(&commands)?)?;
    output.files.push(log);
    if let Err(error) = result {
        let _ = write_new(
            &dir.join("failed.json"),
            &serde_json::to_vec_pretty(
                &json!({"error":format!("{error:#}"),"cancelled":cancel.load(Ordering::Relaxed)}),
            )?,
        );
        return Err(error).with_context(|| {
            format!(
                "Activity stopped; partial outputs and command log: {}",
                dir.display()
            )
        });
    }
    progress("Activity completed; outputs saved locally".into());
    output.finish()
}

fn command(
    tools: &MediaTools,
    cwd: &Path,
    program: &str,
    args: Vec<String>,
    stdin_text: Option<String>,
) -> CommandSpec {
    CommandSpec {
        program: program.into(),
        args,
        cwd: Some(cwd.into()),
        timeout_secs: tools.timeout_secs.clamp(1, 3600),
        max_output_bytes: 2 * 1024 * 1024,
        stdin_text,
        ..Default::default()
    }
}

fn execute(
    tools: &MediaTools,
    cwd: &Path,
    cancel: &Arc<AtomicBool>,
    log: &mut Vec<Value>,
    program: &str,
    args: Vec<String>,
    stdin: Option<String>,
) -> Result<CommandResult> {
    execute_spec(&command(tools, cwd, program, args, stdin), cwd, cancel, log)
}

fn execute_spec(
    spec: &CommandSpec,
    cwd: &Path,
    cancel: &Arc<AtomicBool>,
    log: &mut Vec<Value>,
) -> Result<CommandResult> {
    check_cancel(cancel)?;
    let result = run_command_cancellable(spec, cwd, Some(cancel.clone()))?;
    log.push(json!({"command":spec,"result":result}));
    if result.cancelled {
        bail!("Activity cancelled");
    }
    if result.timed_out {
        bail!("{} exceeded the activity timeout", spec.program);
    }
    if result.exit_code != Some(0) {
        bail!(
            "{} failed with exit {:?}: {}",
            spec.program,
            result.exit_code,
            result.stderr.chars().take(4000).collect::<String>()
        );
    }
    Ok(result)
}

fn ocr(
    tools: &MediaTools,
    cwd: &Path,
    input: &Path,
    language: &str,
    cancel: &Arc<AtomicBool>,
    log: &mut Vec<Value>,
) -> Result<String> {
    let result = execute(
        tools,
        cwd,
        cancel,
        log,
        &tools.tesseract,
        vec![
            path_text(input)?,
            "stdout".into(),
            "-l".into(),
            language.into(),
            "--psm".into(),
            "3".into(),
        ],
        None,
    )?;
    if result.truncated {
        bail!("OCR output exceeded the 2 MiB per-page limit");
    }
    Ok(result.stdout)
}

fn speech_program(tools: &MediaTools, engine: SpeechEngine) -> &str {
    if engine == SpeechEngine::Espeak {
        &tools.espeak
    } else {
        &tools.piper
    }
}

fn speech_parameters(
    tools: &MediaTools,
    text: &str,
    engine: SpeechEngine,
    voice: &str,
    rate: u32,
) -> Result<()> {
    if text.trim().is_empty() || text.len() > 100_000 {
        bail!("Speech/script text must contain 1–100,000 bytes");
    }
    voice_name(voice)?;
    if !(80..=450).contains(&rate) {
        bail!("Speech rate must be 80–450 words per minute");
    }
    if engine == SpeechEngine::Piper {
        regular_input(&tools.piper_model, MAX_MEDIA_BYTES)
            .context("Choose an installed Piper voice model")?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "The synthesis boundary keeps voice/rate, output, cancellation and logging explicit"
)]
fn speak(
    tools: &MediaTools,
    text: &str,
    engine: SpeechEngine,
    voice: &str,
    rate: u32,
    wave: &Path,
    cancel: &Arc<AtomicBool>,
    log: &mut Vec<Value>,
) -> Result<()> {
    speech_parameters(tools, text, engine, voice, rate)?;
    let cwd = wave.parent().context("Missing audio output folder")?;
    let args = if engine == SpeechEngine::Espeak {
        vec![
            "--stdin".into(),
            "-w".into(),
            path_text(wave)?,
            "-s".into(),
            rate.to_string(),
            "-v".into(),
            voice.into(),
        ]
    } else {
        vec![
            "--model".into(),
            path_text(&tools.piper_model.canonicalize()?)?,
            "--output_file".into(),
            path_text(wave)?,
            "--length_scale".into(),
            format!("{:.4}", 175.0 / rate as f64),
        ]
    };
    execute(
        tools,
        cwd,
        cancel,
        log,
        speech_program(tools, engine),
        args,
        Some(text.into()),
    )?;
    require_wave(wave)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodcastTurn {
    pub speaker: String,
    pub text: String,
}

pub fn parse_podcast_script(script: &str) -> Result<Vec<PodcastTurn>> {
    if script.len() > 100_000 {
        bail!("Podcast script exceeds 100,000 bytes");
    }
    let mut turns = Vec::<PodcastTurn>::new();
    for line in script.lines() {
        let line = line.trim().trim_matches('*').trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("```") {
            continue;
        }
        let tagged = line.split_once(':').filter(|(speaker, _)| {
            matches!(
                speaker.trim().to_ascii_uppercase().as_str(),
                "HOST" | "GUEST" | "NARRATOR"
            )
        });
        if let Some((speaker, text)) = tagged {
            turns.push(PodcastTurn {
                speaker: speaker.trim().to_ascii_uppercase(),
                text: text.trim().into(),
            });
        } else if let Some(turn) = turns.last_mut() {
            turn.text.push(' ');
            turn.text.push_str(line);
        } else {
            bail!("Podcast speech must start with HOST:, GUEST:, or NARRATOR:");
        }
        if turns.len() > 64 {
            bail!("Podcast is limited to 64 speech turns");
        }
    }
    if turns.is_empty()
        || turns
            .iter()
            .any(|t| t.text.is_empty() || t.text.len() > 16_000)
    {
        bail!("Each podcast turn must contain 1–16,000 bytes of speech");
    }
    Ok(turns)
}

fn voice_name(voice: &str) -> Result<()> {
    if voice.is_empty()
        || voice.len() > 128
        || !voice
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '+' | '/'))
    {
        bail!("Invalid local voice identifier");
    }
    Ok(())
}
fn language_name(language: &str) -> Result<()> {
    if language.is_empty()
        || language.len() > 64
        || !language
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '_'))
    {
        bail!("Invalid language identifier");
    }
    Ok(())
}

pub(crate) fn path_text(path: &Path) -> Result<String> {
    Ok(path.to_str().context("Media paths must be UTF-8")?.into())
}
pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("Activity cancelled");
    }
    Ok(())
}

pub(crate) fn regular_input(path: &Path, max: u64) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("Select an absolute local file path: {}", path.display());
    }
    let mut checked = PathBuf::new();
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            bail!("Parent traversal is not allowed in media paths");
        }
        checked.push(component.as_os_str());
        if fs::symlink_metadata(&checked)?.file_type().is_symlink() {
            bail!(
                "Symlinks are excluded from media activities: {}",
                path.display()
            );
        }
    }
    let meta = fs::metadata(path)?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > max {
        bail!(
            "Input must be a nonempty regular file no larger than {max} bytes: {}",
            path.display()
        );
    }
    path.canonicalize().context("Resolve media input")
}

pub(crate) fn create_run_dir(data_dir: &Path, category: &str) -> Result<PathBuf> {
    if !matches!(category, "media" | "images" | "documents") {
        bail!("Invalid activity category");
    }
    if !data_dir.is_absolute() {
        bail!("Activity data directory must be absolute");
    }
    let mut checked = PathBuf::new();
    for component in data_dir.components() {
        if matches!(component, Component::ParentDir) {
            bail!("Parent traversal is not allowed in activity data directories");
        }
        checked.push(component.as_os_str());
        if let Ok(meta) = fs::symlink_metadata(&checked) {
            if meta.file_type().is_symlink() {
                bail!("Activity data directory must not contain symlinks");
            }
        }
    }
    fs::create_dir_all(data_dir)?;
    let base = data_dir.canonicalize()?.join("activities");
    for directory in [&base, &base.join(category)] {
        if let Ok(meta) = fs::symlink_metadata(directory) {
            if !meta.is_dir() || meta.file_type().is_symlink() {
                bail!("Activity output location is not a real directory");
            }
        } else {
            fs::create_dir(directory)?;
        }
    }
    let path = base.join(category).join(uuid::Uuid::new_v4().to_string());
    fs::create_dir(&path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(path)
}

pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("Create new activity output {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn read_bytes_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let path = regular_input(path, max)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut bytes = Vec::new();
    options.open(path)?.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        bail!("Input grew past its size limit");
    }
    Ok(bytes)
}
pub(crate) fn read_text_bounded(path: &Path, max: u64) -> Result<String> {
    String::from_utf8(read_bytes_bounded(path, max)?).context("Activity text output is not UTF-8")
}

pub(crate) fn image_input(path: &Path) -> Result<(u32, u32)> {
    let bytes = read_bytes_bounded(path, MAX_IMAGE_BYTES)?;
    image_dimensions(&bytes)
}
pub(crate) fn image_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_IMAGE_BYTES {
        bail!("Image exceeds the 32 MiB limit");
    }
    let (width, height) = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    if width == 0 || height == 0 || width as u64 * height as u64 > MAX_IMAGE_PIXELS {
        bail!("Image exceeds the 16 megapixel limit");
    }
    Ok((width, height))
}
pub(crate) fn load_image(path: &Path) -> Result<DynamicImage> {
    let bytes = read_bytes_bounded(path, MAX_IMAGE_BYTES)?;
    image_dimensions(&bytes)?;
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
}

fn require_wave(path: &Path) -> Result<()> {
    let path = regular_input(path, 768 * 1024 * 1024)?;
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        bail!("Speech/conversion engine did not produce a RIFF WAV file");
    }
    let declared = u32::from_le_bytes(header[4..8].try_into()?) as u64 + 8;
    if declared > length || declared < 12 {
        bail!("Speech/conversion engine produced an incomplete WAV container");
    }
    let mut position = 12u64;
    let mut has_format = false;
    let mut has_audio = false;
    while position + 8 <= declared {
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk)?;
        let size = u32::from_le_bytes(chunk[4..8].try_into()?) as u64;
        let end = position + 8 + size;
        if end > declared {
            bail!("Speech/conversion engine produced a truncated WAV chunk");
        }
        if &chunk[..4] == b"fmt " {
            if size < 16 {
                bail!("Speech/conversion engine produced an invalid WAV format");
            }
            let mut format = [0u8; 16];
            file.read_exact(&mut format)?;
            let encoding = u16::from_le_bytes(format[..2].try_into()?);
            let channels = u16::from_le_bytes(format[2..4].try_into()?);
            let sample_rate = u32::from_le_bytes(format[4..8].try_into()?);
            if !matches!(encoding, 1 | 3 | 0xfffe) || channels == 0 || sample_rate == 0 {
                bail!("Speech/conversion engine returned unsupported or invalid WAV audio");
            }
            has_format = true;
        } else if &chunk[..4] == b"data" && size > 0 {
            has_audio = true;
        }
        position = end + (size & 1);
        file.seek(SeekFrom::Start(position))?;
    }
    if !has_format || !has_audio {
        bail!("Speech/conversion engine produced no playable WAV samples");
    }
    Ok(())
}

pub fn remove_border_background(
    image: DynamicImage,
    color: Option<[u8; 3]>,
    tolerance: u8,
    feather: u8,
    cancel: &AtomicBool,
) -> Result<DynamicImage> {
    let mut rgba = image.into_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 || width as u64 * height as u64 > MAX_IMAGE_PIXELS {
        bail!("Image dimensions are out of range");
    }
    let background = color.unwrap_or_else(|| {
        let corners = [
            rgba.get_pixel(0, 0),
            rgba.get_pixel(width - 1, 0),
            rgba.get_pixel(0, height - 1),
            rgba.get_pixel(width - 1, height - 1),
        ];
        std::array::from_fn(|c| (corners.iter().map(|p| p[c] as u32).sum::<u32>() / 4) as u8)
    });
    let mut seen = vec![false; width as usize * height as usize];
    let mut queue = VecDeque::new();
    for x in 0..width {
        queue.push_back((x, 0));
        queue.push_back((x, height - 1));
    }
    for y in 1..height.saturating_sub(1) {
        queue.push_back((0, y));
        queue.push_back((width - 1, y));
    }
    let mut visited = 0usize;
    while let Some((x, y)) = queue.pop_front() {
        let index = (y * width + x) as usize;
        if seen[index] {
            continue;
        }
        seen[index] = true;
        visited += 1;
        if visited.is_multiple_of(4096) {
            check_cancel(cancel)?;
        }
        let pixel = rgba.get_pixel_mut(x, y);
        let distance = (0..3)
            .map(|c| pixel[c].abs_diff(background[c]))
            .max()
            .unwrap_or(0) as u16;
        let outer = tolerance as u16 + feather as u16;
        if distance > outer {
            continue;
        }
        let alpha = if distance <= tolerance as u16 || feather == 0 {
            0
        } else {
            (pixel[3] as u32 * (distance - tolerance as u16) as u32 / feather as u32) as u8
        };
        pixel[3] = alpha;
        if x > 0 {
            queue.push_back((x - 1, y));
        }
        if x + 1 < width {
            queue.push_back((x + 1, y));
        }
        if y > 0 {
            queue.push_back((x, y - 1));
        }
        if y + 1 < height {
            queue.push_back((x, y + 1));
        }
    }
    check_cancel(cancel)?;
    Ok(DynamicImage::ImageRgba8(rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn fixture_executable(directory: &Path, name: &str, body: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join(name);
        fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path.display().to_string()
    }

    #[cfg(unix)]
    fn quoted_fixture(path: &Path) -> String {
        format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
    }

    #[cfg(unix)]
    fn wave_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend(38u32.to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(16_000u32.to_le_bytes());
        bytes.extend(32_000u32.to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(0i16.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn transcription_pipeline_saves_real_outputs_and_duration_contract() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("recording with spaces.wav");
        wave_fixture(&input);
        let model = temp.path().join("whisper model.bin");
        fs::write(&model, b"model fixture").unwrap();
        let ffmpeg = fixture_executable(
            temp.path(),
            "audio-converter",
            &format!(
                "for argument do output=$argument; done\ncp {} \"$output\"",
                quoted_fixture(&input)
            ),
        );
        let whisper = fixture_executable(
            temp.path(),
            "transcriber",
            r#"
prefix=
while [ "$#" -gt 0 ]; do
    if [ "$1" = '-of' ]; then shift; prefix=$1; fi
    shift
done
printf 'Recorded project decisions.\n' > "${prefix}.txt"
printf '1\n00:00:00,000 --> 00:00:01,000\nRecorded project decisions.\n' > "${prefix}.srt"
printf '{"transcription":[{"text":"Recorded project decisions."}]}' > "${prefix}.json"
"#,
        );
        let tools = MediaTools {
            ffmpeg,
            whisper,
            whisper_model: model,
            ..Default::default()
        };
        let output = run(
            &tools,
            &MediaRequest::Transcribe {
                input: input.clone(),
                language: "auto".into(),
                translate_to_english: true,
                max_seconds: 90,
            },
            temp.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        assert_eq!(output.text, "Recorded project decisions.\n");
        assert!(output
            .files
            .iter()
            .any(|path| path.ends_with("transcript.srt")));
        require_wave(&output.directory.join("audio.wav")).unwrap();
        let commands: Value =
            serde_json::from_slice(&fs::read(output.directory.join("commands.json")).unwrap())
                .unwrap();
        let conversion = commands[0]["command"]["args"].as_array().unwrap();
        assert!(conversion
            .windows(2)
            .any(|args| args[0] == "-t" && args[1] == "90"));
        assert!(conversion.iter().any(|arg| arg == input.to_str().unwrap()));
        assert!(commands[1]["command"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|arg| arg == "-tr"));
        assert_eq!(output.metadata["maximum_seconds"], 90);
        assert_eq!(fs::read(input).unwrap().len(), 46);
    }

    #[cfg(unix)]
    #[test]
    fn pdf_ocr_keeps_original_page_numbers_and_only_claims_rendered_pages() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("source.pdf");
        fs::write(&input, b"%PDF-fixture").unwrap();
        let image = temp.path().join("page-fixture.png");
        image::RgbImage::from_pixel(4, 4, image::Rgb([255, 255, 255]))
            .save(&image)
            .unwrap();
        let inspector = fixture_executable(temp.path(), "pdf-inspector",
            "printf 'Page 2 MediaBox: 0 0 612 792\\nPage 3 MediaBox: 0 0 1500 1800\\n'");
        let renderer = fixture_executable(temp.path(), "pdf-renderer", &format!(
            "for argument do prefix=$argument; done\ncp {} \"${{prefix}}.png\"", quoted_fixture(&image)));
        let ocr = fixture_executable(temp.path(), "ocr", "printf 'Recognized page text.\\n'");
        let output = run(
            &MediaTools {
                pdftoppm: renderer,
                pdfinfo: inspector,
                tesseract: ocr,
                ..Default::default()
            },
            &MediaRequest::PdfOcr {
                input,
                first_page: 2,
                last_page: 4,
                dpi: 150,
                language: "eng".into(),
            },
            temp.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        let log: Value = serde_json::from_slice(&fs::read(output.directory.join("commands.json")).unwrap()).unwrap();
        assert_eq!(log[1]["command"]["args"][3], "1650"); // Letter at 150 DPI
        assert_eq!(log[2]["command"]["args"][3], "2000"); // Poster hits the cap
        assert!(output.text.contains("page 2"));
        assert!(output.text.contains("page 3"));
        assert!(!output.text.contains("page 4"));
        assert_eq!(output.metadata["pages"].as_array().unwrap().len(), 2);
        assert_eq!(output.warnings.len(), 1);
        assert!(output.files.iter().any(|path| path.ends_with("ocr.md")));
    }

    #[cfg(unix)]
    #[test]
    fn video_frames_keep_requested_times_and_stop_at_duration() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("clip.mp4");
        fs::write(&input, b"video fixture").unwrap();
        let image = temp.path().join("frame-fixture.png");
        image::RgbImage::from_pixel(4, 3, image::Rgb([200, 20, 20]))
            .save(&image)
            .unwrap();
        let ffprobe = fixture_executable(
            temp.path(),
            "video-probe",
            "printf '{\"format\":{\"duration\":\"2.5\"}}'",
        );
        let ffmpeg = fixture_executable(
            temp.path(),
            "frame-extractor",
            &format!(
                "for argument do output=$argument; done\ncp {} \"$output\"",
                quoted_fixture(&image)
            ),
        );
        let output = run(
            &MediaTools {
                ffprobe,
                ffmpeg,
                ..Default::default()
            },
            &MediaRequest::VideoFrames {
                input,
                every_seconds: 1.0,
                max_frames: 10,
            },
            temp.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        let frames = output.metadata["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[2]["requested_timestamp_seconds"], 2.0);
        for frame in frames {
            image_input(Path::new(frame["path"].as_str().unwrap())).unwrap();
        }
        assert!(output
            .text
            .contains("Samples do not constitute a complete video review"));
    }

    #[cfg(unix)]
    #[test]
    fn video_manifest_keeps_valid_frames_when_audio_outlasts_video() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("clip.mkv");
        fs::write(&input, b"video fixture").unwrap();
        let image = temp.path().join("fixture.png");
        image::RgbImage::new(2, 2).save(&image).unwrap();
        let tools = MediaTools {
            ffprobe: fixture_executable(temp.path(), "probe", r#"printf '{"format":{"duration":"4.0"}}'"#),
            ffmpeg: fixture_executable(temp.path(), "extract", &format!(
                "for argument do path=$argument; done\ncase \"$path\" in *frame-0001.png) cp {} \"$path\";; esac",
                quoted_fixture(&image))),
            ..Default::default()
        };
        let result = run(&tools, &MediaRequest::VideoFrames {input, every_seconds: 1.0, max_frames: 4},
            temp.path(), Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
        let manifest: Value = serde_json::from_slice(&fs::read(result.directory.join("frames.json")).unwrap()).unwrap();
        assert_eq!(manifest["frames"].as_array().unwrap().len(), 1);
        assert_eq!(manifest["frames"][0]["requested_timestamp_seconds"], 0.0);
    }

    #[cfg(unix)]
    #[test]
    fn podcast_pipeline_passes_each_turn_to_tts_and_creates_joined_audio() {
        let temp = tempfile::tempdir().unwrap();
        let fixture = temp.path().join("voice.wav");
        wave_fixture(&fixture);
        let tts = fixture_executable(
            temp.path(),
            "speech-engine",
            &format!(
                r#"
output=
while [ "$#" -gt 0 ]; do
    if [ "$1" = '-w' ]; then shift; output=$1; fi
    shift
done
cat > "${{output}}.input.txt"
cp {} "$output"
"#,
                quoted_fixture(&fixture)
            ),
        );
        let ffmpeg = fixture_executable(
            temp.path(),
            "join-audio",
            &format!(
                "for argument do output=$argument; done\ncp {} \"$output\"",
                quoted_fixture(&fixture)
            ),
        );
        let output = run(
            &MediaTools {
                espeak: tts,
                ffmpeg,
                ..Default::default()
            },
            &MediaRequest::Podcast {
                script: "HOST: Our first finding.\nGUEST: A second perspective.".into(),
                engine: SpeechEngine::Espeak,
                host_voice: "en-us".into(),
                guest_voice: "en-sc".into(),
                words_per_minute: 160,
            },
            temp.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap();
        require_wave(&output.directory.join("podcast.wav")).unwrap();
        assert_eq!(
            fs::read_to_string(output.directory.join("turn-001.wav.input.txt")).unwrap(),
            "Our first finding."
        );
        assert_eq!(
            fs::read_to_string(output.directory.join("turn-002.wav.input.txt")).unwrap(),
            "A second perspective."
        );
        let commands: Value =
            serde_json::from_slice(&fs::read(output.directory.join("commands.json")).unwrap())
                .unwrap();
        assert_eq!(commands.as_array().unwrap().len(), 3);
        assert!(commands[2]["command"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|arg| arg == "[0:a][1:a]concat=n=2:v=0:a=1[episode]"));
    }

    #[test]
    fn native_cutout_preserves_enclosed_foreground_and_existing_alpha() {
        let mut image = image::RgbaImage::from_pixel(7, 7, image::Rgba([255, 255, 255, 255]));
        for x in 2..5 {
            for y in 2..5 {
                image.put_pixel(x, y, image::Rgba([220, 20, 20, 160]));
            }
        }
        image.put_pixel(3, 3, image::Rgba([255, 255, 255, 200]));
        let result = remove_border_background(
            DynamicImage::ImageRgba8(image),
            Some([255, 255, 255]),
            5,
            5,
            &AtomicBool::new(false),
        )
        .unwrap()
        .into_rgba8();
        assert_eq!(result.get_pixel(0, 0)[3], 0);
        assert_eq!(result.get_pixel(2, 2)[3], 160);
        assert_eq!(result.get_pixel(3, 3)[3], 200);
    }

    #[test]
    fn podcast_parsing_is_explicit_and_bounded() {
        let turns = parse_podcast_script(
            "# Episode\nHOST: First idea.\nMore explanation.\nGUEST: A reply.",
        )
        .unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].text, "First idea. More explanation.");
        assert!(parse_podcast_script("Unlabeled unsupported script").is_err());
        assert!(parse_podcast_script(&"HOST: hi\n".repeat(65)).is_err());
    }

    #[test]
    fn wave_magic_without_format_and_audio_is_not_a_successful_output() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty.wav");
        fs::write(&path, b"RIFF\x04\x00\x00\x00WAVE").unwrap();
        assert!(require_wave(&path).is_err());
    }

    #[test]
    fn output_creation_never_overwrites_and_rejects_escape() {
        let temp = tempfile::tempdir().unwrap();
        let dir = create_run_dir(temp.path(), "media").unwrap();
        let file = dir.join("result.txt");
        write_new(&file, b"original").unwrap();
        assert!(write_new(&file, b"replacement").is_err());
        assert_eq!(fs::read(file).unwrap(), b"original");
        assert!(create_run_dir(temp.path(), "../escape").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path(), dir.join("link")).unwrap();
            assert!(create_run_dir(&dir.join("link"), "media").is_err());
        }
    }

    #[test]
    fn cancelled_native_activity_creates_no_output() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.png");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
            .save(&input)
            .unwrap();
        let request = MediaRequest::RemoveBackground {
            input,
            color: None,
            tolerance: 10,
            feather: 2,
            semantic_model: false,
        };
        assert!(run(
            &MediaTools::default(),
            &request,
            temp.path(),
            Arc::new(AtomicBool::new(true)),
            |_| {}
        )
        .is_err());
        assert!(!temp.path().join("activities").exists());
    }

    #[cfg(unix)]
    #[test]
    fn speech_requires_a_real_output_not_just_zero_exit() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let fake = temp.path().join("tts");
        fs::write(&fake, "#!/bin/sh\ncat >/dev/null\nexit 0\n").unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
        let tools = MediaTools {
            espeak: fake.display().to_string(),
            ..Default::default()
        };
        let request = MediaRequest::Speech {
            text: "Hello".into(),
            engine: SpeechEngine::Espeak,
            voice: "en".into(),
            words_per_minute: 175,
        };
        let error = run(
            &tools,
            &request,
            temp.path(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("partial outputs"));
    }
}
