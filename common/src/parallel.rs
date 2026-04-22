use std::sync::atomic::{AtomicUsize, Ordering};

static PAR_DISABLE_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// RAII guard that disables shared parallel execution for the current scope.
pub struct ParallelFlagGuard;

impl ParallelFlagGuard {
    /// Disables shared parallel execution for the current scope.
    pub fn disabled() -> Self {
        PAR_DISABLE_DEPTH.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for ParallelFlagGuard {
    fn drop(&mut self) {
        let previous = PAR_DISABLE_DEPTH.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(previous > 0, "parallel disable depth underflow");
    }
}

/// Returns the shared minimum parallel chunk length.
///
/// When parallel execution is enabled, this returns `1`, which is effectively
/// Rayon default behavior. When disabled, this returns `usize::MAX`, which
/// prevents further splitting for indexed iterators.
pub fn par_enabled() -> usize {
    if PAR_DISABLE_DEPTH.load(Ordering::SeqCst) == 0 {
        1
    } else {
        usize::MAX
    }
}

/// Returns the effective minimum parallel chunk length for the given enabled case.
///
/// When parallel execution is enabled, this returns `enabled_min_len`. When disabled,
/// this returns `usize::MAX`, which prevents further splitting for indexed iterators.
pub fn par_enabled_with(enabled_min_len: usize) -> usize {
    if PAR_DISABLE_DEPTH.load(Ordering::SeqCst) == 0 {
        enabled_min_len
    } else {
        usize::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::{PAR_DISABLE_DEPTH, ParallelFlagGuard, par_enabled, par_enabled_with};
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    static PARALLEL_TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn test_lock() -> MutexGuard<'static, ()> {
        PARALLEL_TEST_MUTEX.lock().unwrap()
    }

    fn reset_depth() {
        PAR_DISABLE_DEPTH.store(0, Ordering::SeqCst);
    }

    #[test]
    fn parallel_flag_guard_disables_and_restores() {
        let _lock = test_lock();
        reset_depth();

        assert_eq!(par_enabled(), 1);
        assert_eq!(par_enabled_with(4096), 4096);

        let _guard = ParallelFlagGuard::disabled();
        assert_eq!(par_enabled(), usize::MAX);
        assert_eq!(par_enabled_with(4096), usize::MAX);
    }

    #[test]
    fn parallel_flag_guard_restores_previous_value() {
        let _lock = test_lock();
        reset_depth();

        {
            let _guard = ParallelFlagGuard::disabled();
            assert_eq!(par_enabled(), usize::MAX);
        }

        assert_eq!(par_enabled(), 1);
        assert_eq!(par_enabled_with(4096), 4096);
    }

    #[test]
    fn parallel_flag_guard_is_reference_counted() {
        let _lock = test_lock();
        reset_depth();

        let outer = ParallelFlagGuard::disabled();
        assert_eq!(par_enabled(), usize::MAX);

        {
            let _inner = ParallelFlagGuard::disabled();
            assert_eq!(par_enabled(), usize::MAX);
        }

        assert_eq!(par_enabled(), usize::MAX);
        drop(outer);
        assert_eq!(par_enabled(), 1);
    }
}
