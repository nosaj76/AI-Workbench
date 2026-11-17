//! A diagnostic source identity, not a cryptographic signature.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("read source directory") {
            collect(&entry.expect("source entry").path(), files);
        }
    } else if path.extension().is_some_and(|s| s == "rs") {
        files.push(path.to_owned());
    }
}
fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("build.rs"),
    ];
    collect(&root.join("src"), &mut files);
    files.sort();
    let mut hash = 0xcbf29ce484222325u64;
    for file in files {
        for byte in file
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .as_bytes()
            .iter()
            .copied()
            .chain(fs::read(&file).expect("read source"))
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    println!("cargo:rustc-env=WORKBENCH_SOURCE_ID={hash:016x}");
}
