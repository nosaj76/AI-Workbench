//! StudioUi files responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn pick_input(&mut self, image: bool, workspace: Option<&Path>) {
        let directory = workspace.map(Path::to_path_buf);
        self.start_work("Choosing a local file", move |tx, _| {
            let mut dialog = rfd::FileDialog::new().set_title(if image {
                "Choose an input image"
            } else {
                "Choose a local media file"
            });
            if let Some(directory) = directory {
                dialog = dialog.set_directory(directory);
            }
            if image {
                dialog = dialog.add_filter("Images", &["png", "jpg", "jpeg", "webp", "gif"]);
            }
            if let Some(path) = dialog.pick_file() {
                tx.send(StudioEvent::PickedInput(path.display().to_string(), image))?;
            }
            Ok(())
        });
    }

    pub(super) fn pick_model(&mut self, workspace: Option<&Path>) {
        let directory = workspace.map(Path::to_path_buf);
        self.start_work("Choosing diffusion model weights", move |tx, _| {
            let mut dialog = rfd::FileDialog::new()
                .set_title("Choose local diffusion model weights")
                .add_filter("Model weights", &["gguf", "safetensors", "ckpt"]);
            if let Some(directory) = directory {
                dialog = dialog.set_directory(directory);
            }
            if let Some(path) = dialog.pick_file() {
                tx.send(StudioEvent::PickedModel(path.display().to_string()))?;
            }
            Ok(())
        });
    }

    pub(super) fn pick_documents(&mut self, workspace: &Path) {
        let workspace = workspace.to_path_buf();
        self.start_work("Choosing local documents", move |tx, _| {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title("Choose documents inside the workspace")
                .set_directory(&workspace)
                .pick_files()
            {
                let mut relative = vec![];
                for path in paths {
                    relative.push(
                        path.strip_prefix(&workspace)
                            .context("Choose documents inside the selected workspace")?
                            .to_str()
                            .context("Document paths must be UTF-8")?
                            .replace('\\', "/"),
                    );
                }
                tx.send(StudioEvent::PickedDocuments(relative))?;
            }
            Ok(())
        });
    }

    pub(super) fn save_settings(&mut self) {
        match serde_json::from_str::<Vec<String>>(&self.extra_args_json)
            .context("Additional sd-cli arguments must be a JSON array")
            .and_then(|args| {
                self.config.image.sd_extra_args = args;
                save_config(&self.data_dir, &self.config)
            }) {
            Ok(()) => self.status = "Studio engine/program settings saved".into(),
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }
}
