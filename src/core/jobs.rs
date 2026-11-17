//! Worker failure boundary: a panicking job must still deliver its completion event.
use anyhow::{anyhow, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub fn guarded<T>(work: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            Err(anyhow!("Worker stopped unexpectedly: {message}. Inspect saved partial results before retrying."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_error_and_panic_are_reported_not_success() {
        assert_eq!(guarded(|| Ok(7)).unwrap(), 7);
        assert!(guarded::<()>(|| Err(anyhow!("failure")))
            .unwrap_err()
            .to_string()
            .contains("failure"));
        assert!(guarded::<()>(|| panic!("fixture panic"))
            .unwrap_err()
            .to_string()
            .contains("fixture panic"));
    }
}
