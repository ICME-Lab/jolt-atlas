use std::sync::atomic::{AtomicBool, Ordering};

static PAR_ENABLED: AtomicBool = AtomicBool::new(true);

/// RAII guard that restores the shared parallel execution flag on drop.
pub struct ParallelFlagGuard {
    previous: bool,
}

impl ParallelFlagGuard {
    /// Sets the shared parallel execution flag for the current scope.
    pub fn set(enabled: bool) -> Self {
        Self {
            previous: swap_par_enabled(enabled),
        }
    }

    /// Disables shared parallel execution for the current scope.
    pub fn disabled() -> Self {
        Self::set(false)
    }
}

impl Drop for ParallelFlagGuard {
    fn drop(&mut self) {
        swap_par_enabled(self.previous);
    }
}

/// Returns the shared minimum parallel chunk length.
///
/// When parallel execution is enabled, this returns `1`, which is effectively
/// Rayon default behavior. When disabled, this returns `usize::MAX`, which
/// prevents further splitting for indexed iterators.
pub fn par_enabled() -> usize {
    if PAR_ENABLED.load(Ordering::Relaxed) {
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
    if PAR_ENABLED.load(Ordering::Relaxed) {
        enabled_min_len
    } else {
        usize::MAX
    }
}

/// Sets whether shared parallel execution is enabled.
pub fn set_par_enabled(enabled: bool) {
    PAR_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Swaps the shared parallel execution flag and returns the previous value.
pub fn swap_par_enabled(enabled: bool) -> bool {
    PAR_ENABLED.swap(enabled, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{
        ParallelFlagGuard, par_enabled, par_enabled_with, set_par_enabled, swap_par_enabled,
    };

    #[test]
    fn parallel_flag_round_trips() {
        let original = swap_par_enabled(true);

        set_par_enabled(false);
        assert_eq!(par_enabled(), usize::MAX);
        assert_eq!(par_enabled_with(4096), usize::MAX);

        assert!(!swap_par_enabled(true));
        assert_eq!(par_enabled(), 1);
        assert_eq!(par_enabled_with(4096), 4096);

        set_par_enabled(original);
    }

    #[test]
    fn parallel_flag_guard_restores_previous_value() {
        let original = swap_par_enabled(true);
        set_par_enabled(true);

        {
            let _guard = ParallelFlagGuard::disabled();
            assert_eq!(par_enabled(), usize::MAX);
        }

        assert_eq!(par_enabled(), 1);
        set_par_enabled(original);
    }
}
