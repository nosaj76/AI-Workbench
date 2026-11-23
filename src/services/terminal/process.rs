//! One owner spawns, supervises and reaps each PTY shell. Keeping the spawning
//! thread alive also preserves bubblewrap's Linux parent-death contract.
#[cfg(unix)]
use rustix::{
    event::{poll, PollFd, PollFlags, Timespec},
    io::{retry_on_intr, Errno},
    process::{kill_process_group, Pid, Signal},
};
use std::io;
use std::process::{Child, Command};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
#[cfg(unix)]
use std::{fs::File, os::unix::net::UnixStream};

pub(super) struct TerminalProcess {
    pub(super) pid: u32,
    pub(super) exit_code: Arc<OnceLock<i32>>,
    #[cfg(unix)]
    close: UnixStream,
    worker: Mutex<Option<JoinHandle<io::Result<()>>>>,
}

impl TerminalProcess {
    #[cfg(unix)]
    pub(super) fn spawn(mut command: Command, master: &File) -> io::Result<Self> {
        let master = master.try_clone()?;
        let (started, startup) = mpsc::sync_channel(1);
        let (close, requests) = UnixStream::pair()?;
        let exit_code = Arc::new(OnceLock::new());
        let status = exit_code.clone();
        let worker = thread::Builder::new()
            .name("workbench-pty-supervisor".into())
            .spawn(move || {
                let mut child = match command.spawn() {
                    Ok(child) => child,
                    Err(error) => {
                        let _ = started.send(Err(error));
                        return Ok(());
                    }
                };
                drop(command); // release the spawning side's slave descriptors
                let _ = started.send(Ok(child.id()));
                let result = supervise(&mut child, &requests, &master);
                if result.is_err() {
                    let _ = child.kill();
                }
                let reaped = child.wait();
                let _ = status.set(reaped.as_ref().ok().and_then(|s| s.code()).unwrap_or(-1));
                result.and(reaped.map(|_| ()))
            })?;
        let pid = startup
            .recv()
            .map_err(|_| io::Error::other("Terminal supervisor stopped during startup"))??;
        Ok(Self {
            pid,
            exit_code,
            close,
            worker: Mutex::new(Some(worker)),
        })
    }

    pub(super) fn request_close(&self) {
        #[cfg(unix)]
        let _ = self.close.shutdown(std::net::Shutdown::Write);
    }

    pub(super) fn finished(&self) -> bool {
        match self.worker.try_lock() {
            Ok(worker) => worker.as_ref().is_none_or(JoinHandle::is_finished),
            Err(std::sync::TryLockError::WouldBlock) => false,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned
                .into_inner()
                .as_ref()
                .is_none_or(JoinHandle::is_finished),
        }
    }

    pub(super) fn join(&self) -> io::Result<()> {
        let mut guard = self.worker.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(worker) = guard.take() {
            worker
                .join()
                .map_err(|_| io::Error::other("Terminal supervisor panicked"))??;
        }
        Ok(())
    }
}

#[cfg(unix)]
fn supervise(child: &mut Child, close: &UnixStream, master: &File) -> io::Result<()> {
    let sid = Pid::from_child(child);
    #[cfg(target_os = "linux")]
    let shell = match rustix::process::pidfd_open(sid, rustix::process::PidfdFlags::empty()) {
        Ok(fd) => Some(fd),
        Err(Errno::NOSYS | Errno::PERM) => None,
        Err(error) => return Err(error.into()),
    };
    #[cfg(not(target_os = "linux"))]
    let shell: Option<std::os::fd::OwnedFd> = None;

    let closing = if let Some(fd) = &shell {
        // No polling timer for idle Linux terminals. Either shell exit or a
        // close request wakes the supervisor.
        let mut fds = [
            PollFd::new(fd, PollFlags::IN),
            PollFd::new(close, PollFlags::IN),
        ];
        retry_on_intr(|| poll(&mut fds, None))?;
        fds[0].revents().is_empty()
    } else {
        let tick = Timespec {
            tv_sec: 0,
            tv_nsec: 40_000_000,
        };
        let mut request = [PollFd::new(close, PollFlags::IN)];
        while !exited(child)? && retry_on_intr(|| poll(&mut request, Some(&tick)))? == 0 {}
        !exited(child)?
    };
    if closing {
        if shell.is_none() {
            use std::os::fd::AsRawFd;
            let mut group = 0;
            if unsafe { libc::ioctl(master.as_raw_fd(), libc::TIOCGPGRP as _, &mut group) } == 0
                && group > 1
                && group != sid.as_raw_pid()
            {
                if let Some(foreground) = Pid::from_raw(group) {
                    let _ = kill_process_group(foreground, Signal::KILL);
                }
            }
        }
        let _ = kill_process_group(sid, Signal::HUP);
        let deadline = Instant::now() + Duration::from_millis(400);
        while !exited(child)? {
            if Instant::now() >= deadline {
                crate::integrations::kill_process_group(child);
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
    #[cfg(target_os = "linux")]
    {
        wait_for_exit(sid, rustix::process::WaitIdOptions::empty())?;
        stop_session_jobs(sid)?;
    }
    Ok(())
}

#[cfg(unix)]
fn exited(child: &mut Child) -> io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        wait_for_exit(
            Pid::from_child(child),
            rustix::process::WaitIdOptions::NOHANG,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(child.try_wait()?.is_some())
    }
}

// Only this supervisor reaps the shell. Keeping the leader waitable reserves
// its session ID until cleanup finishes, including after natural shell exit.
#[cfg(target_os = "linux")]
fn wait_for_exit(pid: Pid, options: rustix::process::WaitIdOptions) -> io::Result<bool> {
    use rustix::process::{waitid, WaitId, WaitIdOptions};
    let flags = options | WaitIdOptions::EXITED | WaitIdOptions::NOWAIT;
    Ok(retry_on_intr(|| waitid(WaitId::Pid(pid), flags))?.is_some())
}

#[cfg(target_os = "linux")]
fn stop_session_jobs(sid: Pid) -> io::Result<()> {
    use rustix::process::getsid;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let mut signalled = false;
        let mut failure = None;
        for entry in std::fs::read_dir("/proc")? {
            let name = entry?.file_name();
            let Some(pid) = name
                .to_str()
                .and_then(|s| s.parse().ok())
                .and_then(Pid::from_raw)
            else {
                continue;
            };
            if pid == sid || getsid(Some(pid)) != Ok(sid) {
                continue;
            }
            match stop_member(pid, sid) {
                Ok(sent) => signalled |= sent,
                Err(Errno::SRCH) => {} // exited during enumeration
                Err(error) => failure = Some(error),
            }
        }
        if !signalled {
            return failure.map_or(Ok(()), |error| Err(error.into()));
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Terminal jobs did not exit before the cleanup deadline",
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(target_os = "linux")]
fn stop_member(pid: Pid, sid: Pid) -> rustix::io::Result<bool> {
    use rustix::process::{getsid, pidfd_open, pidfd_send_signal, PidfdFlags};
    let fd = pidfd_open(pid, PidfdFlags::empty())?;
    // Recheck after opening the stable handle. A readable pidfd already exited.
    let mut ready = [PollFd::new(&fd, PollFlags::IN)];
    if getsid(Some(pid))? != sid
        || retry_on_intr(|| poll(&mut ready, Some(&Timespec::default())))? != 0
    {
        return Ok(false);
    }
    pidfd_send_signal(&fd, Signal::KILL)?;
    Ok(true)
}
