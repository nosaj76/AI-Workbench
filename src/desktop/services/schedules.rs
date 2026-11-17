//! ServicesUi schedules responsibilities.
use super::*;

impl ServicesUi {
    pub(super) fn schedules_page(&mut self, ui: &mut egui::Ui, workspace: Option<&Path>) {
        ui.heading("Background scheduler");
        let running = self.daemon.as_ref().is_some_and(|d| d.running);
        ui.label(if let Some(daemon) = &self.daemon {
            format!(
                "{} · {}",
                if daemon.running { "Running" } else { "Stopped" },
                daemon.message
            )
        } else {
            "No scheduler process is running".into()
        });
        ui.label("Jobs continue after this window closes. Background runs can inspect files; requested file changes, commands and external calls are saved as Needs approval and are not executed.");
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.busy && !running,
                    egui::Button::new("Start background process"),
                )
                .clicked()
            {
                let dir = self.dir.clone();
                self.work("Starting background scheduler", move |tx, _| {
                    let state = scheduler::start_background(&dir)?;
                    tx.send(Event::Text(state.message))?;
                    tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
                    Ok(())
                });
            }
            if ui
                .add_enabled(
                    !self.busy && running,
                    egui::Button::new("Stop background process"),
                )
                .clicked()
            {
                let dir = self.dir.clone();
                self.work("Stopping background scheduler", move |tx, _| {
                    scheduler::stop_background(&dir)?;
                    tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
                    tx.send(Event::Text("Background scheduler stopped".into()))?;
                    Ok(())
                });
            }
            if ui.button("Startup service details").clicked() {
                self.show_unit = !self.show_unit;
            }
        });
        if self.show_unit {
            ui.group(|ui|{
                ui.label("The user service starts at login and uses this executable and data directory. Run the application's installer first if you want a stable installed path. Running before login requires the account's normal systemd lingering configuration.");
                match std::env::current_exe().and_then(|p|scheduler::service_unit(&p,&self.dir).map_err(std::io::Error::other)){Ok(unit)=>{egui::CollapsingHeader::new("Unit preview").show(ui,|ui|{ui.add(egui::Label::new(RichText::new(unit).monospace().size(12.0)).selectable(true));});},Err(e)=>{ui.label(e.to_string());}}
                ui.horizontal_wrapped(|ui|{
                    if ui.add_enabled(!self.busy,egui::Button::new("Install / update user service")).clicked(){let dir=self.dir.clone();self.work("Installing user service",move|tx,_|{let path=scheduler::install_user_service(&std::env::current_exe()?,&dir)?;tx.send(Event::Text(format!("Service unit saved: {}",path.display())))?;Ok(())});}
                    for (action,label) in [(ServiceAction::Enable,"Enable at login & start"),(ServiceAction::Start,"Start service"),(ServiceAction::Stop,"Stop service"),(ServiceAction::Disable,"Disable at login & stop"),(ServiceAction::Status,"Service status")]{if ui.add_enabled(!self.busy,egui::Button::new(label)).on_hover_text(action.preview()).clicked(){self.systemd_action(action);}}
                });
            });
        }
        if !self.logs.is_empty() {
            egui::CollapsingHeader::new("Last service output")
                .id_salt("scheduler_service_output")
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(RichText::new(&self.logs).monospace().size(12.0))
                            .selectable(true),
                    );
                });
        }
        ui.separator();
        ui.heading(if self.edit_id.is_some() {
            "Edit scheduled job"
        } else {
            "New scheduled job"
        });
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(300.0));
            ui.checkbox(&mut self.enabled, "Enabled");
        });
        ui.horizontal(|ui| {
            ui.label("Model");
            ui.add(
                egui::TextEdit::singleline(&mut self.model)
                    .hint_text("Installed Ollama model")
                    .desired_width(300.0),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Workspace");
            ui.add(egui::TextEdit::singleline(&mut self.workspace).desired_width(430.0));
            if ui
                .add_enabled(!self.busy, egui::Button::new("Browse…"))
                .clicked()
            {
                self.work("Choose scheduled workspace", |tx, _| {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        tx.send(Event::Workspace(path))?;
                    }
                    Ok(())
                });
            }
            if let Some(path) = workspace {
                if ui.button("Use current workspace").clicked() {
                    self.workspace = path.display().to_string();
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.timing_mode, 0, "Cron");
            ui.selectable_value(&mut self.timing_mode, 1, "Once");
            ui.selectable_value(&mut self.timing_mode, 2, "Interval");
        });
        match self.timing_mode {
            0 => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Expression");
                    ui.add(egui::TextEdit::singleline(&mut self.cron).desired_width(200.0));
                    ui.label("Timezone");
                    ui.add(egui::TextEdit::singleline(&mut self.timezone).desired_width(220.0));
                });
                ui.label(RichText::new("minute hour day-of-month month day-of-week · lists, ranges, steps, names and @daily/@reboot · Example: 0 9 * * MON-FRI").small());
                if let Ok(cron) = scheduler::CronSchedule::parse(&self.cron) {
                    if cron.is_reboot() {
                        ui.label("Runs once when the daemon observes a new operating-system boot.");
                    } else if let Ok(next) = cron.next_after(Utc::now(), &self.timezone) {
                        ui.label(format!("Next: {}", display_time(next, &self.timezone)));
                    }
                }
            }
            1 => {
                ui.label("Date and time with UTC offset (RFC3339)");
                ui.add(egui::TextEdit::singleline(&mut self.once).desired_width(380.0));
            }
            _ => {
                ui.horizontal(|ui| {
                    ui.label("Repeat every");
                    ui.add(egui::DragValue::new(&mut self.interval).range(1..=5_256_000));
                    ui.label("minutes");
                });
            }
        }
        ui.horizontal(|ui| {
            ui.label("Maximum run time");
            ui.add(
                egui::DragValue::new(&mut self.timeout)
                    .range(30..=86400)
                    .speed(30),
            );
            ui.label("seconds");
        });
        ui.label("Prompt");
        ui.add(
            egui::TextEdit::multiline(&mut self.prompt)
                .desired_rows(4)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!self.busy, egui::Button::new("Save job"))
                .clicked()
            {
                match self.form_job() {
                    Ok(job) => {
                        let dir = self.dir.clone();
                        self.work("Saving scheduled job", move |tx, _| {
                            SchedulerStore::open(&dir)?.upsert(&job)?;
                            tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
                            tx.send(Event::Text("Scheduled job saved".into()))?;
                            Ok(())
                        });
                        self.edit_id = None;
                    }
                    Err(error) => self.error = Some(format!("{error:#}")),
                }
            }
            if self.edit_id.is_some() && ui.button("Cancel editing").clicked() {
                self.edit_id = None;
            }
        });
        ui.separator();
        ui.heading("Saved jobs");
        let mut edit = None;
        let mut change = None;
        let mut remove = None;
        for job in &self.jobs {
            ui.group(|ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(&job.name);
                    ui.label(if job.enabled { "Enabled" } else { "Paused" });
                    ui.label(&job.model);
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("Edit").small())
                        .clicked()
                    {
                        edit = Some(job.clone());
                    }
                    if ui
                        .add_enabled(
                            !self.busy,
                            egui::Button::new(if job.enabled { "Pause" } else { "Enable" }).small(),
                        )
                        .clicked()
                    {
                        let mut changed = job.clone();
                        changed.enabled = !changed.enabled;
                        if changed.enabled {
                            if let Err(error) = changed.recalculate(Utc::now()) {
                                self.error = Some(error.to_string());
                            } else {
                                change = Some(changed);
                            }
                        } else {
                            change = Some(changed);
                        }
                    }
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("Queue now").small())
                        .clicked()
                    {
                        let mut changed = job.clone();
                        changed.enabled = true;
                        changed.next_run = Utc::now();
                        change = Some(changed);
                    }
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("Remove schedule").small())
                        .clicked()
                    {
                        remove = Some(job.id.clone());
                    }
                });
                ui.label(job.timing_label());
                ui.label(format!("Workspace: {}", job.workspace.display()));
                if job.enabled {
                    let next = match &job.timing {
                        ScheduleTiming::Cron { expression, .. }
                            if expression.trim() == "@reboot"
                                && job.next_run == DateTime::<Utc>::MAX_UTC =>
                        {
                            "Next operating-system boot observed by the scheduler".into()
                        }
                        ScheduleTiming::Cron { timezone, .. } => {
                            display_time(job.next_run, timezone)
                        }
                        _ => job.next_run.to_rfc3339(),
                    };
                    ui.label(format!("Next: {next}"));
                }
                egui::CollapsingHeader::new("Prompt")
                    .id_salt(&job.id)
                    .show(ui, |ui| {
                        ui.label(&job.prompt);
                    });
            });
        }
        if self.jobs.is_empty() {
            ui.label("No scheduled jobs. Save one above, then start the background scheduler.");
        }
        if let Some(job) = edit {
            self.load_form(job);
        }
        if let Some(job) = change {
            let dir = self.dir.clone();
            self.work("Updating scheduled job", move |tx, _| {
                SchedulerStore::open(&dir)?.upsert(&job)?;
                tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
                Ok(())
            });
        }
        if let Some(id) = remove {
            let dir = self.dir.clone();
            self.work(
                "Removing scheduled job; keeping run history",
                move |tx, _| {
                    SchedulerStore::open(&dir)?.remove(&id)?;
                    tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
                    Ok(())
                },
            );
        }
        ui.separator();
        ui.heading("Run history");
        for run in &self.runs {
            egui::CollapsingHeader::new(format!(
                "{} · {} · {}",
                run.job_name, run.state, run.started_at
            ))
            .id_salt(&run.id)
            .show(ui, |ui| {
                ui.label(format!("Scheduled for {}", run.scheduled_for));
                if !run.error.is_empty() {
                    ui.colored_label(ui.visuals().warn_fg_color, &run.error);
                }
                if let Some(id) = &run.session_id {
                    if ui.button("Open saved conversation").clicked() {
                        self.session_request = Some(id.clone());
                    }
                }
                if !run.output.is_empty() {
                    ui.add(egui::Label::new(&run.output).selectable(true));
                }
                for action in &run.blocked_actions {
                    ui.label(
                        RichText::new(serde_json::to_string_pretty(action).unwrap_or_default())
                            .monospace()
                            .size(12.0),
                    );
                }
            });
        }
    }

    pub(super) fn form_job(&self) -> Result<ScheduledJob> {
        let workspace = PathBuf::from(self.workspace.trim())
            .canonicalize()
            .context("Choose an existing scheduled workspace")?;
        if !workspace.is_dir() {
            bail!("Scheduled workspace must be a directory");
        }
        let timing = match self.timing_mode {
            0 => ScheduleTiming::Cron {
                expression: self.cron.trim().into(),
                timezone: self.timezone.trim().into(),
            },
            1 => ScheduleTiming::Once {
                at: DateTime::parse_from_rfc3339(self.once.trim())
                    .context(
                        "Use a date/time with UTC offset, for example 2026-11-07T09:00:00-05:00",
                    )?
                    .with_timezone(&Utc),
            },
            _ => ScheduleTiming::Interval {
                minutes: self.interval,
            },
        };
        let mut job = ScheduledJob {
            id: self
                .edit_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            name: self.name.trim().into(),
            prompt: self.prompt.clone(),
            model: self.model.trim().into(),
            workspace,
            timing,
            enabled: self.enabled,
            next_run: Utc::now(),
            timeout_seconds: self.timeout,
            last_run: None,
            last_boot_id: None,
        };
        if let Some(old) = self.jobs.iter().find(|j| j.id == job.id) {
            job.last_run = old.last_run;
            job.last_boot_id = old.last_boot_id.clone();
        }
        job.recalculate(Utc::now())?;
        Ok(job)
    }

    pub(super) fn load_form(&mut self, job: ScheduledJob) {
        self.edit_id = Some(job.id);
        self.name = job.name;
        self.prompt = job.prompt;
        self.model = job.model;
        self.workspace = job.workspace.display().to_string();
        self.enabled = job.enabled;
        self.timeout = job.timeout_seconds;
        match job.timing {
            ScheduleTiming::Cron {
                expression,
                timezone,
            } => {
                self.timing_mode = 0;
                self.cron = expression;
                self.timezone = timezone;
            }
            ScheduleTiming::Once { at } => {
                self.timing_mode = 1;
                self.once = at.to_rfc3339();
            }
            ScheduleTiming::Interval { minutes } => {
                self.timing_mode = 2;
                self.interval = minutes;
            }
        }
    }

    pub(super) fn systemd_action(&mut self, action: ServiceAction) {
        let dir = self.dir.clone();
        self.work(&action.preview(), move |tx, _| {
            let result = scheduler::service_control(action, &dir)?;
            let output = format!("{}\n{}", result.stdout, result.stderr);
            if result.exit_code != Some(0) {
                bail!(
                    "{} returned {:?}: {output}",
                    action.preview(),
                    result.exit_code
                );
            }
            tx.send(Event::Text(output))?;
            tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
            Ok(())
        });
    }
}
