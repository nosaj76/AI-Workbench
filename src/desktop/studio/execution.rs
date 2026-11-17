//! StudioUi execution responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn prepare(&mut self, job: StudioJob) {
        self.prepared = None;
        self.start_work("Preparing an activity preview",move|tx,cancel|{
            crate::media::check_cancel(&cancel)?;
            let description=match &job{
                StudioJob::Image(data)=>image_generation::preview(&data.0,&data.1)?,
                StudioJob::Documents(data)=>{let workspace=Workspace::new(data.2.clone())?;document_activities::preview(&data.1,&workspace)?},
                StudioJob::Media(data)=>crate::media::preview(&data.0,&data.1)?,
                StudioJob::ImageReview(data)=>{if data.1.is_empty()||data.1.len()>32{bail!("Choose 1–32 images for vision review");}for path in &data.1{crate::media::image_input(path)?;}json!({"activity":"vision review of selected output images","model":data.0.model,"images":data.1,"question":data.2,"processing":"Images are resized to fit 1280×1280 and sent to the configured Ollama endpoint; one request per selected image."})},
            };
            tx.send(StudioEvent::Prepared(Box::new((job,description))))?;Ok(())
        });
    }

    pub(super) fn preview_panel(&mut self, ui: &mut egui::Ui) {
        let mut execute = false;
        let mut dismiss = false;
        if let Some((_, preview)) = &self.prepared {
            ui.separator();
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.heading("Review this activity");
                let mut text = serde_json::to_string_pretty(preview).unwrap_or_default();
                egui::ScrollArea::vertical()
                    .id_salt("studio_preview")
                    .max_height(250.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .code_editor()
                                .interactive(false)
                                .desired_width(f32::INFINITY),
                        );
                    });
                ui.add_enabled_ui(!self.busy, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .button(RichText::new("Run this activity").strong())
                            .clicked()
                        {
                            execute = true;
                        }
                        if ui.button("Dismiss preview").clicked() {
                            dismiss = true;
                        }
                    });
                });
            });
        }
        if dismiss {
            self.prepared = None;
        }
        if execute {
            if let Some((job, _)) = self.prepared.take() {
                self.launch(job);
            }
        }
    }

    pub(super) fn launch(&mut self, job: StudioJob) {
        let directory = self.data_dir.clone();
        self.start_work("Starting the reviewed activity", move |tx, cancel| {
            let updates = tx.clone();
            let progress = move |status: String| {
                let _ = updates.send(StudioEvent::Status(status));
            };
            let output = match job {
                StudioJob::Image(data) => StudioOutput::Images(Box::new(
                    image_generation::generate(&data.0, &data.1, &directory, cancel, progress)?,
                )),
                StudioJob::Documents(data) => {
                    let workspace = Workspace::new(data.2.clone())?;
                    StudioOutput::Activity(Box::new(document_activities::run(
                        &data.0, &data.1, &workspace, &directory, cancel, progress,
                    )?))
                }
                StudioJob::Media(data) => StudioOutput::Activity(Box::new(crate::media::run(
                    &data.0, &data.1, &directory, cancel, progress,
                )?)),
                StudioJob::ImageReview(data) => {
                    StudioOutput::Activity(Box::new(document_activities::review_images(
                        &data.0, &data.1, &data.2, &directory, cancel, progress,
                    )?))
                }
            };
            let thumbnails = thumbnails(&output.files());
            tx.send(StudioEvent::Finished(Box::new(output), thumbnails))?;
            Ok(())
        });
    }

    pub(super) fn start_work(
        &mut self,
        status: &str,
        work: impl FnOnce(Sender<StudioEvent>, Arc<AtomicBool>) -> Result<()> + Send + 'static,
    ) {
        if self.busy {
            return;
        }
        if let Some(worker) = self.worker.take() {
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
        self.busy = true;
        self.error = None;
        self.status = status.into();
        self.started = Some(Instant::now());
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        self.worker = Some(thread::spawn(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(tx.clone(), cancel)));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let _ = tx.send(StudioEvent::Error(format!("{error:#}")));
                }
                Err(_) => {
                    let _ = tx.send(StudioEvent::Error(
                        "Activity worker stopped unexpectedly; saved partial files were preserved."
                            .into(),
                    ));
                }
            }
            let _ = tx.send(StudioEvent::Done);
        }));
    }
}
