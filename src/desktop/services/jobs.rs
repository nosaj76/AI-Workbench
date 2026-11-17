//! ServicesUi jobs responsibilities.
use super::*;

impl ServicesUi {
    pub(super) fn refresh(&mut self) {
        let error = self.error.clone();
        let dir = self.dir.clone();
        self.work("Refreshing local service state", move |tx, _| {
            tx.send(Event::Overview(Box::new(load_overview(&dir)?)))?;
            Ok(())
        });
        // Periodic refreshes must not silently dismiss an operation failure.
        self.error = error;
    }

    pub(super) fn work<F>(&mut self, label: &str, worker: F)
    where
        F: FnOnce(mpsc::Sender<Event>, Arc<AtomicBool>) -> Result<()> + Send + 'static,
    {
        if self.busy {
            return;
        }
        self.busy = true;
        self.status = label.into();
        self.error = None;
        self.cancel = Arc::new(AtomicBool::new(false));
        self.last_refresh = Instant::now();
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            if let Err(error) = crate::jobs::guarded(|| worker(tx.clone(), cancel)) {
                let _ = tx.send(Event::Error(format!("{error:#}")));
            }
            let _ = tx.send(Event::Done);
        });
    }
}
