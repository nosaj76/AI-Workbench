//! Workbench knowledge responsibilities.
use super::*;

impl Workbench {
    pub(super) fn knowledge_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Workspace knowledge");
        ui.label("Build a local index, retrieve relevant excerpts, and cite the exact source paths. Indexing is explicit; refresh after changing project files.");
        let Some(root) = self.session.project.clone() else {
            ui.add_space(25.0);
            if ui.button("Choose a workspace").clicked() {
                self.browse_workspace();
            }
            return;
        };
        ui.label(
            RichText::new(&root)
                .monospace()
                .color(ui.visuals().hyperlink_color),
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.work_busy,
                    egui::Button::new("Build / refresh text index"),
                )
                .clicked()
            {
                self.index_workspace(false);
            }
            if ui
                .add_enabled(
                    !self.work_busy && !self.settings.embedding_model.trim().is_empty(),
                    egui::Button::new("Build / refresh embeddings"),
                )
                .clicked()
            {
                self.index_workspace(true);
            }
            if self.work_busy && ui.button("Cancel indexing").clicked() {
                self.work_cancel.store(true, Ordering::Relaxed);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Embedding model");
            ui.text_edit_singleline(&mut self.settings.embedding_model);
        });
        ui.label(RichText::new("Text retrieval uses SQLite FTS5. Embedding retrieval combines lexical matches and cosine similarity; vectors stay in the local index.").small());
        ui.separator();
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.knowledge_query)
                    .hint_text("Find evidence about…")
                    .desired_width((ui.available_width() - 100.0).max(200.0)),
            );
            if ui
                .add_enabled(
                    !self.work_busy && !self.knowledge_query.trim().is_empty(),
                    egui::Button::new("Search"),
                )
                .clicked()
            {
                self.search_knowledge();
            }
        });
        if !self.knowledge_hits.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Attach these excerpts").clicked() {
                    self.attachments.push(Attachment {
                        name: format!("{} retrieved excerpts", self.knowledge_hits.len()),
                        source: format!("Knowledge search: {}", self.knowledge_query),
                        text: self.knowledge_results.clone(),
                        image: None,
                    });
                    self.page = Page::Chat;
                }
                if ui.button("Ask with evidence").clicked() {
                    self.attachments.push(Attachment {
                        name: "Knowledge evidence".into(),
                        source: format!("Knowledge search: {}", self.knowledge_query),
                        text: self.knowledge_results.clone(),
                        image: None,
                    });
                    self.input = self.knowledge_query.clone();
                    self.page = Page::Chat;
                }
            });
        }
        let mut open = None;
        egui::ScrollArea::vertical().id_salt("knowledge_results").show(ui,|ui|{
            for evidence in &self.knowledge_hits{egui::Frame::default().fill(ui.visuals().faint_bg_color).inner_margin(egui::Margin::same(12)).corner_radius(egui::CornerRadius::same(8)).show(ui,|ui|{ui.horizontal(|ui|{ui.label(RichText::new(format!("{}:{}–{}",evidence.path,evidence.start_line,evidence.end_line)).strong().monospace().color(ui.visuals().hyperlink_color));if ui.small_button("Open source").clicked(){open=Some(evidence.path.clone());}});ui.add(egui::Label::new(RichText::new(&evidence.text).monospace().size(13.0)).selectable(true));});ui.add_space(8.0);}
            if self.knowledge_hits.is_empty(){ui.add_space(25.0);ui.label("Build an index and search to inspect the excerpts that would be sent to the model.");}
        });
        if let Some(path) = open {
            self.read_file(path);
            self.page = Page::Files;
        }
    }

    pub(super) fn index_workspace(&mut self, embeddings: bool) {
        let Some(root) = self.session.project.clone() else {
            return;
        };
        let db = self.dir.join("knowledge.sqlite3");
        let settings = self.settings.clone();
        let model = self.settings.embedding_model.clone();
        if embeddings {
            if let Err(e) = self.settings.validate_endpoint() {
                self.error = Some(e.to_string());
                return;
            }
        }
        self.work("Indexing workspace…",move |tx,cancel|{
            let workspace=Workspace::new(PathBuf::from(&root))?;let mut index=crate::retrieval::RetrievalIndex::open(db)?;
            let stats=index.index_workspace(&workspace,&cancel,|p|{let _=tx.send(WorkEvent::Status(format!("Indexing: {} scanned · {} updated · {}",p.scanned,p.indexed,p.current_path)));})?;
            if stats.cancelled{tx.send(WorkEvent::Status("Indexing cancelled; committed data remains available".into()))?;return Ok(());}
            if embeddings{
                let rt=tokio::runtime::Builder::new_current_thread().enable_all().build()?;let client=OllamaClient::new(&settings.base_url)?;let mut count=0;
                loop{if cancel.load(Ordering::Relaxed){bail!("Embedding indexing cancelled; completed batches are retained");}let chunks=index.chunks_for_embedding(&workspace,&model,16)?;if chunks.is_empty(){break;}let inputs=chunks.iter().map(|c|c.text.clone()).collect::<Vec<_>>();let vectors=rt.block_on(crate::routing::embeddings_with_policy(&settings,&client,&model,&inputs,cancel.clone()))?;if vectors.len()!=chunks.len(){bail!("Ollama returned a different number of embeddings than inputs");}for(chunk,vector)in chunks.iter().zip(vectors.iter()){index.set_embedding(chunk.id,&chunk.content_hash,&model,vector)?;count+=1;}tx.send(WorkEvent::Status(format!("Embedded {count} chunks with {model}")))?;}
                tx.send(WorkEvent::Status(format!("Index ready · {} updated · {} unchanged · {} removed · {} skipped · {count} vectors added",stats.indexed,stats.unchanged,stats.removed,stats.skipped)))?;
            }else{tx.send(WorkEvent::Status(format!("Text index ready · {} updated · {} unchanged · {} removed · {} skipped",stats.indexed,stats.unchanged,stats.removed,stats.skipped)))?;}
            Ok(())
        });
    }

    pub(super) fn search_knowledge(&mut self) {
        let Some(root) = self.session.project.clone() else {
            return;
        };
        let db = self.dir.join("knowledge.sqlite3");
        let query = self.knowledge_query.clone();
        let model = self.settings.embedding_model.clone();
        let settings = self.settings.clone();
        if !model.is_empty() {
            if let Err(e) = self.settings.validate_endpoint() {
                self.error = Some(e.to_string());
                return;
            }
        }
        self.work("Searching knowledge…", move |tx, cancel| {
            let ws = Workspace::new(PathBuf::from(&root))?;
            let index = crate::retrieval::RetrievalIndex::open(db)?;
            let hits = if model.trim().is_empty() {
                index.search(&ws, &query, 8)?
            } else {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let client = OllamaClient::new(&settings.base_url)?;
                match rt.block_on(crate::routing::embeddings_with_policy(
                    &settings,
                    &client,
                    &model,
                    std::slice::from_ref(&query),
                    cancel,
                )) {
                    Ok(vectors) => {
                        let vector = vectors.first().context("No query embedding returned")?;
                        index.search_hybrid(&ws, &query, 8, &model, vector)?
                    }
                    Err(e) => {
                        tx.send(WorkEvent::Status(format!(
                            "Embedding query failed; lexical results shown: {e}"
                        )))?;
                        index.search(&ws, &query, 8)?
                    }
                }
            };
            tx.send(WorkEvent::Status(format!(
                "Found {} relevant excerpts",
                hits.len()
            )))?;
            tx.send(WorkEvent::Knowledge { root, hits })?;
            Ok(())
        });
    }
}
