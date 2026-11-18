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

/// The fewest items a parallel loop gives one task when parallel execution is enabled.
/// Most of the loops that use it run a few field operations per item, and a task costs more
/// to hand out than a few items cost to run; a loop whose items are large passes its own
/// length with [`par_enabled_with`].
pub const PAR_MIN_LEN: usize = 128;

/// Returns the shared minimum parallel chunk length.
///
/// When parallel execution is enabled, this returns [`PAR_MIN_LEN`]. When disabled, this
/// returns `usize::MAX`, which prevents further splitting for indexed iterators.
pub fn par_enabled() -> usize {
    if PAR_DISABLE_DEPTH.load(Ordering::SeqCst) == 0 {
        PAR_MIN_LEN
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
    use super::{PAR_DISABLE_DEPTH, PAR_MIN_LEN, ParallelFlagGuard, par_enabled, par_enabled_with};
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

        assert_eq!(par_enabled(), PAR_MIN_LEN);
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

        assert_eq!(par_enabled(), PAR_MIN_LEN);
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
        assert_eq!(par_enabled(), PAR_MIN_LEN);
    }
}
