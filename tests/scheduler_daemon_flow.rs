#![cfg(target_os = "linux")]

use ollama_workbench::scheduler::{daemon_status, stop_background};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn gui_recognizes_the_daemons_effective_data_directory() {
    for explicit in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_ollama-workbench"));
        command
            .arg("--daemon")
            .env_remove("OLLAMA_WORKBENCH_DATA_DIR")
            .env("XDG_DATA_HOME", temp.path())
            .stdout(Stdio::null());
        let dir = if explicit {
            let path = temp.path().join("explicit");
            command.arg("--data-dir").arg(&path);
            path
        } else {
            temp.path().join("ollama-workbench")
        };
        let mut daemon = Daemon(command.spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !daemon_status(&dir)
            .unwrap()
            .is_some_and(|s| s.running && s.pid == daemon.0.id())
        {
            assert!(daemon.0.try_wait().unwrap().is_none());
            assert!(
                Instant::now() < deadline,
                "daemon was not recognized, explicit={explicit}"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
        // A copied status record cannot authorize stopping the real process.
        let other = tempfile::tempdir().unwrap();
        std::fs::copy(
            dir.join("scheduler-status.json"),
            other.path().join("scheduler-status.json"),
        )
        .unwrap();
        std::fs::File::create(other.path().join("scheduler.lock")).unwrap();
        assert!(!daemon_status(other.path()).unwrap().unwrap().running);
        stop_background(&dir).unwrap();
        assert!(daemon.0.wait().unwrap().success());
        assert!(!daemon_status(&dir).unwrap().unwrap().running);
    }
}
