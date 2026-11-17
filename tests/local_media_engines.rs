//! Opt-in acceptance tests using real locally installed media executables.
use ollama_workbench::media::{self, ActivityOutput, MediaRequest, MediaTools, SpeechEngine};
use std::{
    path::Path,
    sync::{atomic::AtomicBool, Arc},
};

fn run(dir: &Path, request: MediaRequest) -> ActivityOutput {
    media::run(
        &MediaTools::default(),
        &request,
        dir,
        Arc::new(AtomicBool::new(false)),
        |s| eprintln!("{s}"),
    )
    .unwrap()
}

#[test]
#[ignore = "requires pdflatex, pdftoppm and tesseract with English language data"]
fn installed_latex_and_ocr_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(dir.path(), MediaRequest::LatexPdf {
        source: r"\documentclass{article}\begin{document}\Huge Workbench engine acceptance\end{document}".into(),
    });
    let pdf = output
        .files
        .iter()
        .find(|p| p.extension().is_some_and(|e| e == "pdf"))
        .unwrap();
    let bytes = std::fs::read(pdf).unwrap();
    assert!(bytes.starts_with(b"%PDF-"));
    let ocr = run(
        dir.path(),
        MediaRequest::PdfOcr {
            input: pdf.clone(),
            first_page: 1,
            last_page: 1,
            dpi: 150,
            language: "eng".into(),
        },
    );
    assert!(
        ocr.text.contains("Workbench engine acceptance"),
        "{}",
        ocr.text
    );
    let png = ocr
        .files
        .iter()
        .find(|p| p.extension().is_some_and(|e| e == "png"))
        .unwrap();
    let image = run(
        dir.path(),
        MediaRequest::ImageOcr {
            input: png.clone(),
            language: "eng".into(),
        },
    );
    assert!(
        image.text.contains("Workbench engine acceptance"),
        "{}",
        image.text
    );
    assert_eq!(
        std::fs::read(pdf).unwrap(),
        bytes,
        "OCR must preserve its source"
    );
}

#[test]
#[ignore = "requires espeak-ng, ffmpeg and ffprobe"]
fn installed_speech_and_video_engines_produce_valid_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let speech = run(
        dir.path(),
        MediaRequest::Speech {
            text: "Workbench speech engine acceptance.".into(),
            engine: SpeechEngine::Espeak,
            voice: "en".into(),
            words_per_minute: 160,
        },
    );
    let wav = speech
        .files
        .iter()
        .find(|p| p.extension().is_some_and(|e| e == "wav"))
        .unwrap();
    let bytes = std::fs::read(wav).unwrap();
    assert!(bytes.starts_with(b"RIFF") && bytes.len() > 1000);
    let video = dir.path().join("fixture.mkv");
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=2",
            "-t",
            "2",
            "-c:v",
            "ffv1",
            "-n",
        ])
        .arg(&video)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let frames = run(
        dir.path(),
        MediaRequest::VideoFrames {
            input: video,
            every_seconds: 1.0,
            max_frames: 2,
        },
    );
    assert_eq!(frames.metadata["frames"].as_array().unwrap().len(), 2);
    for file in frames
        .files
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
    {
        assert!(std::fs::read(file)
            .unwrap()
            .starts_with(b"\x89PNG\r\n\x1a\n"));
    }
}

#[test]
#[ignore = "requires whisper-cli, a Whisper model, ffmpeg and the upstream jfk.wav sample"]
fn installed_transcription_recognizes_known_speech() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools {
        whisper: std::env::var("WORKBENCH_WHISPER_BIN").expect("set WORKBENCH_WHISPER_BIN"),
        whisper_model: std::env::var("WORKBENCH_WHISPER_MODEL")
            .expect("set WORKBENCH_WHISPER_MODEL")
            .into(),
        ..Default::default()
    };
    let output = media::run(
        &tools,
        &MediaRequest::Transcribe {
            input: std::env::var("WORKBENCH_WHISPER_SAMPLE")
                .expect("set WORKBENCH_WHISPER_SAMPLE to upstream jfk.wav")
                .into(),
            language: "en".into(),
            translate_to_english: false,
            max_seconds: 30,
        },
        dir.path(),
        Arc::new(AtomicBool::new(false)),
        |s| eprintln!("{s}"),
    )
    .unwrap();
    let text = output.text.to_lowercase();
    eprintln!("transcript: {text}");
    assert!(text.contains("ask not what your country can do for you"));
    assert!(text.contains("what you can do for your country"));
    assert!(output
        .files
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "srt")));
}
