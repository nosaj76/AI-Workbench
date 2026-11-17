//! Explicit opt-in test against installed Poppler programs, not mocks.
#[test]
#[ignore = "requires installed pdftotext and pdftoppm"]
fn installed_pdf_engines_extract_and_render() {
    let dir = tempfile::tempdir().unwrap();
    let stream = "BT /F1 18 Tf 40 100 Td (Workbench PDF acceptance) Tj ET";
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 180] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_owned(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
        format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
    ];
    let mut pdf = "%PDF-1.4\n".to_owned();
    let mut offsets = vec![0];
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{object}\nendobj\n", i + 1));
    }
    let xref = pdf.len();
    pdf.push_str("xref\n0 6\n0000000000 65535 f \n");
    for offset in &offsets[1..] {
        pdf.push_str(&format!("{offset:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ));
    let source = dir.path().join("fixture.pdf");
    std::fs::write(&source, pdf).unwrap();
    let workspace = ollama_workbench::files::Workspace::new(dir.path().to_path_buf()).unwrap();
    let extracted = workspace.read_text("fixture.pdf", 10000).unwrap();
    assert!(extracted.text.contains("Workbench PDF acceptance"));
    assert!(!extracted.truncated);
    let prefix = dir.path().join("page");
    let output = std::process::Command::new("pdftoppm")
        .args(["-f", "1", "-singlefile", "-scale-to", "320", "-png"])
        .arg(&source)
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let png = std::fs::read(prefix.with_extension("png")).unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert!(png.len() > 100);
}
