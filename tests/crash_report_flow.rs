//! The panic hook is process-global, so it is exercised in its own test binary.

#[test]
fn a_panic_on_any_thread_writes_a_report_with_location_and_backtrace() {
    let dir = tempfile::tempdir().unwrap();
    ollama_workbench::crash_reports::install_panic_hook(dir.path());
    let joined = std::thread::Builder::new()
        .name("crash-report-probe".into())
        .spawn(|| panic!("probe panic for the crash report"))
        .unwrap()
        .join();
    assert!(joined.is_err());

    let reports = ollama_workbench::crash_reports::recent_reports(dir.path(), 5);
    assert_eq!(reports.len(), 1, "{reports:?}");
    let text = std::fs::read_to_string(&reports[0]).unwrap();
    assert!(text.contains("probe panic for the crash report"), "{text}");
    assert!(text.contains("Thread: crash-report-probe"), "{text}");
    assert!(text.contains("crash_report_flow.rs"), "{text}");
    assert!(text.contains("Backtrace:"), "{text}");
}
