//! Timing/repetition/peak-memory primitives shared by every `examples/bench/bench_*.rs` binary.
//! No new deps: plain `std::time::Instant` and `/proc` reads.
#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Durations collected from repeated timed runs of the same closure.
pub struct RepeatedTiming {
    pub samples: Vec<Duration>,
}

impl RepeatedTiming {
    pub fn min(&self) -> Duration {
        *self.samples.iter().min().expect("at least one sample")
    }

    pub fn max(&self) -> Duration {
        *self.samples.iter().max().expect("at least one sample")
    }

    pub fn mean(&self) -> Duration {
        self.samples.iter().sum::<Duration>() / self.samples.len() as u32
    }

    pub fn median(&self) -> Duration {
        let mut sorted = self.samples.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }
}

/// Runs `f` `warmup` times (untimed) then `measured` times, running `settle` after every
/// iteration — pass `wait_for_background_drops` so the next iteration's allocations don't race
/// this one's freeing background threads (this OOM'd an earlier version on GPT-2/Qwen proofs).
pub fn time_repeated_with_settle<F: FnMut(), S: FnMut()>(
    warmup: usize,
    measured: usize,
    mut f: F,
    mut settle: S,
) -> RepeatedTiming {
    for _ in 0..warmup {
        f();
        settle();
    }
    let mut samples = Vec::with_capacity(measured);
    for _ in 0..measured {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
        settle();
    }
    RepeatedTiming { samples }
}

/// Peak RSS in bytes since process start (`VmHWM`). Linux-only; `None` elsewhere so callers
/// degrade the table field to `null` rather than fail.
#[cfg(target_os = "linux")]
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix("VmHWM:")?;
        let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    })
}

#[cfg(not(target_os = "linux"))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}
