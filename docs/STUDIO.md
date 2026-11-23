# Native Studio: images, documents and media

Studio is part of the Rust desktop application. Open **Workbench → Studio** to
generate images, work through local documents, or run an explicit media activity.
The application does not host a website. All panels, file dialogs, progress,
thumbnails and reports are native desktop UI.

Each activity has two steps: **Review activity** validates the selected request
and displays the concrete parameters; **Run this activity** runs that reviewed
snapshot on a background worker. Editing a form does not silently alter an
already reviewed snapshot. There is one active Studio worker at a time.

## What is included

| Area | Implemented workflow | Required local component |
|---|---|---|
| Image generation | Text-to-image, input-image variation, negative prompts, seeds, size, steps, batches, model selection, saved real images and thumbnails | ComfyUI API, AUTOMATIC1111 API, or `sd-cli`; matching model weights |
| Document activities | Summary, comparison, study guide, podcast script, rewrite, translation | A selected local Ollama text model |
| Report exports | Markdown, plain text and DOCX, with source/chunk coverage | Built into Rust |
| PDF OCR | Render selected pages, recognize each page, retain page images and original page numbers | Poppler `pdfinfo` and `pdftoppm`, Tesseract and its language data |
| Image OCR | Recognize an explicitly selected local image | Tesseract and its language data |
| Transcription | Convert audio/video audio to mono WAV, transcribe, save text/SRT/JSON, then summarize the transcript | FFmpeg, whisper.cpp `whisper-cli`, a local Whisper model; Ollama for the optional summary |
| Speech | Text to a saved WAV file | eSpeak NG, or Piper and a local voice model |
| Podcast | Source documents to a reviewed HOST/GUEST script; synthesize turns and join an episode WAV | Ollama for the script; eSpeak NG/Piper and FFmpeg for audio |
| Video frames | Extract timestamped still frames, display thumbnails, optionally review each frame with a vision model | FFmpeg/ffprobe; an Ollama vision model for image review |
| Background removal | Border-connected plain-color cutout with alpha preservation, or semantic U²-Net cutout | Plain-color mode is built in; semantic mode uses installed `rembg` and cached `u2net.onnx` |
| LaTeX | Compile explicit source into a PDF, with a second pass for references | A local `pdflatex` installation |

Engine executables and model weights are not bundled. A missing program/model
produces an actionable activity error; it does not produce a fabricated success
or substitute a cloud service.

## Image generation

### Choosing an engine

**ComfyUI** is the default API engine. The default address is
`http://127.0.0.1:8188`. Run a local ComfyUI instance, install a checkpoint it
supports, click **Refresh models**, and select that checkpoint. Studio submits
its own API graph, polls the returned prompt ID, downloads the actual output
images, and records the graph and generation parameters. You do not have to use
ComfyUI's browser editor for the built-in graph.

The built-in graph is a conventional checkpoint → text conditioning → sampler →
VAE decode → saved image workflow. It supports ordinary compatible checkpoints.
Architectures requiring separate text encoders, unusual loaders or different
sampling graphs need a matching custom API workflow. An installed engine model
is not automatically interchangeable with every graph.

**AUTOMATIC1111** uses its local API, normally at
`http://127.0.0.1:7860`. Its upstream `--nowebui` mode runs the API without its web
UI; follow that project's installation instructions for your hardware. Studio
uses `/sdapi/v1/txt2img` and `/sdapi/v1/img2img`, optionally requests a checkpoint
override, and saves the returned images itself. Refreshing models queries
`/sdapi/v1/sd-models`. Common Euler sampler names are translated to this engine's
display names; other sampler names must match your engine's installed samplers.

**stable-diffusion.cpp** launches your configured `sd-cli` executable directly.
It requires no image HTTP service or browser. Select an actual local model file.
Studio constructs a direct argument vector with the prompt, negative prompt,
width, height, steps, CFG scale, sampler, seed and managed output filename. An
additional-arguments field accepts a JSON array, for example separate loader
arguments required by a particular model. It is an argument array, not a shell
command. This integration supports text-to-image; choose ComfyUI or
AUTOMATIC1111 for Studio's input-image workflow.

Ollama remains the document/chat/vision inference service. At implementation
time, the current official Ollama `/api/generate` documentation describes
`images` as **input** images, and the official server source rejects image-model
generation with `image generation models are not currently supported`. Studio
therefore uses the concrete local image engines above. It does not send an
invented image-generation schema to Ollama. Upstream support should be rechecked
before adding a direct Ollama image backend.

### Parameters and output

- Width and height: multiples of eight, from 64 through 2,048 pixels.
- Steps: 1–200; batch size: 1–4; CFG: 0–30.
- Seed: `-1` selects a random seed before submission; nonnegative values are
  preserved. The actual selected seed is saved in the run metadata. The
  `sd-cli` batch implementation runs one output per process with successive
  seeds.
- An input image enables ComfyUI/AUTOMATIC1111 image-to-image. Strength controls
  denoising. The original input remains unchanged.
- Image data is size-limited and decoded before a run can be reported as
  successful. Every saved artifact includes dimensions and a SHA-256 hash.
- Progress reflects actual pipeline stages and engine completion. The UI does
  not simulate a percentage or estimate unreported sampling steps.

Local loopback endpoints are allowed by default. An explicit **Allow remote
image engine** setting is required for other hosts. HTTP redirects and proxy
inheritance are disabled. Enabling a remote engine sends that activity's prompt
and input image to the configured endpoint.

### Custom ComfyUI workflows

Paste an **API-format graph**, whose top-level keys identify nodes and whose
node objects contain `class_type` and `inputs`. A normal browser/editor workflow
export is a different format and is not accepted. The **Load text-to-image template**
button provides a compatible starting graph for text-to-image.

Exact JSON string values can be replaced with typed request parameters:

```text
{{prompt}}            {{negative_prompt}}    {{checkpoint}}
{{seed}}              {{width}}              {{height}}
{{steps}}             {{cfg_scale}}          {{batch_size}}
{{strength}}          {{input_image}}        {{output_prefix}}
```

For example, `"seed": "{{seed}}"` becomes a JSON integer. Substitution only
occurs when the entire string equals a placeholder; it is not a template
language or an expression evaluator. The uploaded input-image name is supplied
by the engine, and the output prefix is unique to the current run.

Graphs are limited to 1 MiB, 512 nodes and 48 levels of placeholder nesting. Core
nodes are allowed by default. **Allow trusted custom nodes** is an explicit
choice for installed extensions; it does not sandbox those extensions. When
remote use is disabled, known ComfyUI API nodes advertised by `object_info` are
rejected. A custom node can have arbitrary behavior under the engine's account,
so only use graphs and extensions you trust.

### Cancelling image work

Cancellation stops client polling immediately. For current ComfyUI servers,
Studio requests cancellation of the specific job through
`/api/jobs/{prompt_id}/cancel`. On older servers lacking that endpoint, it
requests deletion of that prompt from the queue; an already running job may
finish. It never sends a global interrupt that could stop another client's job.

AUTOMATIC1111 has no per-request interrupt in this integration. Cancellation
drops Studio's request, reports that the engine may finish it, and does not
interrupt unrelated clients. A directly launched `sd-cli` process group is
terminated on cancellation or timeout. Successfully saved partial files remain
available in the run directory.

## Document activities and coverage

Select a workspace first, then choose one or more documents from inside it.
Paths in the form are workspace-relative, one per line. A comparison needs at
least two documents. A translation needs the target language in its instruction.
The model field can override the currently selected Ollama model.

The existing local document extractor handles plain text/code, DOCX, ODT, PPTX,
XLSX and EPUB. Text-based PDF extraction uses local `pdftotext`. For scanned
PDFs, use **PDF OCR** first and then **Summarize extracted text**. Office/PDF
extraction reads textual content; it is not a reconstruction of original visual
layout, formulas, slide animation, or every embedded object.

The activity splits extracted text into Unicode-safe chunks, retaining source
names and original extracted-text line ranges. It processes **every admitted
chunk**. If the requested documents exceed the explicit chunk limit, it stops
before any model review begins. It does not select a handful of retrieval hits
and label that a review of the whole document.

Summaries, comparisons, study guides and podcast scripts first produce source
notes for each chunk. Bounded reduction combines all notes when they cannot fit
one context window. Rewrite and translation preserve each transformed chunk in
sequence; they are not reduced into a short summary. Model output remains
model-generated and can be incomplete or mistaken; the source map makes the
reviewed evidence and boundaries inspectable.

The saved coverage record includes:

- source path and extraction kind;
- SHA-256 of the file read;
- extracted character/line counts and number of processed chunks;
- whether extraction hit its content limit;
- whether the source changed after it was read.

If a source changes during extraction, the activity fails and asks you to retry
with a stable file. A later change is reported in final coverage. Empty sources
are recorded; an all-empty selection cannot produce a report. Extractor
truncation and configured limits are disclosed instead of being treated as full
file coverage.

The request accepts 1–64 documents, 512–32,000 requested characters per chunk,
1–512 chunks and 128–8,192 requested output tokens. The effective chunk size is
reduced to fit the selected context/output budget. Source file and extracted
text limits also apply. Oversized model responses and failed context fits stop
the activity rather than silently discarding the source being processed.

Every completed activity exports `report.md`, `answer.txt`, `report.docx`,
`source-coverage.json`, `source-chunks.json` and `chunk-notes.json`. The initial
coverage record is also retained as `source-coverage-initial.json`; final
`source-coverage.json` includes changes detected after inference. Individual
chunk notes are also written as the run progresses. The DOCX export preserves
text, headings and code paragraphs; it is a report document, not a full Markdown
typesetting engine or an edited copy of the original document.

Studio's model requests use no tool definitions. They do not start autonomous
subagents, collect new long-term memories or route away from an explicitly
selected model. Document text is treated as evidence, not executable commands.

## OCR, audio, video and speech

### PDF and image OCR

Choose the first and last PDF page, DPI and installed Tesseract language code.
At most 100 pages are rendered per run, at 50–300 DPI with a 2,000-pixel maximum
page dimension. Studio reads the dimensions of each requested page before rendering
so smaller pages retain the selected DPI. Only actual rendered pages are recognized, and headings retain
the PDF's original page numbers. A request extending beyond the end of the PDF
produces a coverage warning. Page images are retained for inspection, and OCR
text is saved as `ocr.md`. Single-image OCR saves `ocr.txt`.

**Summarize extracted text** creates a separate reviewed document activity. It
uses the saved OCR/transcript file, so both the recognized text and the later
model summary remain inspectable.

### Transcription

Configure `whisper-cli` and an existing local Whisper model. Pick an audio file
or a video with an audio track, choose the language (`auto` is supported by
whisper.cpp), and set the maximum duration. The allowed range is 1–21,600
seconds. The UI and output state that only this initial duration is processed.

FFmpeg creates a 16 kHz mono PCM WAV. whisper.cpp produces `transcript.txt`,
`transcript.srt` and `transcript.json`; successful files are preserved along with
the normalized audio and command log. The translate option requests translation
to English from Whisper, not arbitrary-target translation. Use the document
translation activity for another target language. Recognition and translation
quality depend on the installed model and audio.

### Speech and podcast production

eSpeak NG needs an installed voice identifier, such as `en-us`. Piper needs an
installed model; keep that model's required adjacent configuration file in
place. The configurable speech rate is 80–450 words per minute. Piper receives
a corresponding length-scale setting; its exact audible rate depends on the
voice model.

A podcast script uses explicit `HOST:`, `GUEST:` or `NARRATOR:` turns. Continuation
lines belong to the current speaker. There can be at most 64 turns, each with
1–16,000 bytes of speech and at most 100,000 bytes in the complete script.
Studio generates separate segment WAV files and combines them through FFmpeg
into `podcast.wav`. It also keeps the exact script. eSpeak can use different host
and guest voice IDs. The current Piper configuration uses one installed voice
model for both speakers and reports that limitation.

For a source-grounded episode, run **Documents → Podcast script**, inspect the
script, then choose **Use script to produce podcast audio**. Audio generation is
a separate reviewable action, so the script can be edited first.

### Video frames and vision review

Choose a frame interval from 0.1 to 3,600 seconds and at most 120 frames. The
activity reads duration with ffprobe, extracts actual PNG frames at requested
times, stops at the reported duration, and writes `frames.json` with those
timestamps. Frames are fitted within 1,280 × 1,280 pixels.

The completed result offers **Review images with Ollama**. This submits up to
32 explicitly selected result images to the current vision model, one image per
request, and saves their labeled observations in Markdown/DOCX. Images are
resized to a maximum 1,280 × 1,280 pixels for inference. Each image is reviewed
independently. This is a review of sampled still images: it does not establish
what happened in unsampled frames, infer unseen motion, or analyze audio. To
review speech, separately transcribe the video's audio.

### Background removal

The built-in mode estimates a plain background from the corners, then removes
matching pixels connected to the image border. Tolerance and feathering control
the edge; existing transparency and enclosed foreground regions are preserved.
It works best for clean, relatively uniform backgrounds and does not claim
semantic subject segmentation.

Semantic mode runs installed `rembg` with U²-Net. It requires a local cached
`u2net.onnx` in the configured model directory before starting; Studio does not
silently download weights. Both modes save a new transparent PNG and retain the
source unchanged.

### LaTeX PDF

Paste the source, review it and run. Studio writes a fresh `document.tex`, then
runs two `pdflatex` passes with `-no-shell-escape`, restricted TeX input/output
settings and a timeout. It validates that `document.pdf` exists and has a PDF
header before reporting success. Your installed TeX distribution must contain
the packages referenced by the source. The activity does not install packages.

## Output storage, limits and recovery

Runs are saved under the application data directory:

```text
activities/images/<unique-id>/
activities/documents/<unique-id>/
activities/media/<unique-id>/
```

Each run has a fresh directory. Managed output writes use exclusive creation;
source files are never overwritten. Input paths must refer to actual regular
files, and traversal/symlink paths are rejected. Generated image limits include
32 MiB per file and 16,777,216 pixels. General audio/video inputs are limited to
4 GiB; per-command execution time is configurable up to one hour. PDF rendering
and transcript/OCR response caps provide additional bounds.

These are operational limits, not an OS sandbox. External tools and ComfyUI
extensions have the normal permissions of their process account. Activities
use direct argument arrays rather than shell commands. FFmpeg media input
protocols are restricted to local file/pipe use. Local child process groups are
stopped on cancellation/timeout. Programs that are killed may leave partial
files, and those files remain inspectable in the unique run directory.

Media runs keep `request.json`, `commands.json` and `activity.json`; failed runs
record `failed.json` when possible. Image runs save the resolved request,
workflow/engine details and validated image metadata. Document runs save source
coverage and intermediate notes. The UI offers output-folder/file opening,
thumbnails, rendered reports, copy, and **Attach report to chat**.

Engine/program configuration is stored separately in `studio-settings.json`
with a bounded atomic write. An unreadable configuration is reported and left
in place; it is replaced only when you explicitly save Studio settings. Paths
and programs are configurable so existing installations can be used without
changing the desktop app.

## Verification and upstream references

Automated tests exercise native background removal, bounded podcast parsing,
exclusive file creation, cancellation, invalid configuration preservation,
Unicode-complete document chunking, DOCX readback, source limits, and a complete
multi-chunk document model exchange against a local HTTP fixture. Image tests
exercise actual ComfyUI/AUTOMATIC1111 request/response flows and saved PNG bytes.
Executable fixtures verify PDF page coverage, transcription outputs and duration
arguments, video timestamps/EOF, TTS stdin, and podcast concatenation. An engine
that exits successfully without creating its promised output fails the activity.

These tests establish the application's API/CLI contracts and output handling.
They are not claims that every external engine release, GPU or model was tested.
Use the installed engine's own diagnostics for model compatibility and hardware
setup.

Primary implementation references:

- [Ollama generate API](https://docs.ollama.com/api/generate) and
  [official server routes](https://github.com/ollama/ollama/blob/main/server/routes.go).
- [ComfyUI server routes](https://docs.comfy.org/development/comfyui-server/comms_routes),
  [official API example](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/basic_api_example.py),
  and [job cancellation source](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py).
- [AUTOMATIC1111 API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)
  and [command-line options](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings).
- [stable-diffusion.cpp CLI](https://github.com/leejet/stable-diffusion.cpp/tree/master/examples/cli)
  and [argument definitions](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/common/common.cpp).
- [whisper.cpp CLI](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli),
  [Tesseract command-line usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html),
  and [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html).

These integrations are independent implementations using public interfaces;
upstream GPL application code was not copied into the Rust application.
