//! A `tracing_subscriber::Layer` recording each named span's wall-clock duration (enter→exit),
//! with no `fmt` layer composed in. Only meaningful for synchronous, non-reentrant spans (true
//! of every `#[tracing::instrument]` in the proving path).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::span;
use tracing_subscriber::{layer::Context, Layer};

#[derive(Clone, Default)]
pub struct StageTimingLayer {
    starts: Arc<Mutex<HashMap<span::Id, Instant>>>,
    durations: Arc<Mutex<HashMap<&'static str, Duration>>>,
}

impl StageTimingLayer {
    /// Drains recorded durations. Call once per measured iteration so durations from different
    /// iterations don't get summed together.
    pub fn take(&self) -> HashMap<&'static str, Duration> {
        std::mem::take(&mut *self.durations.lock().unwrap())
    }
}

impl<S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>> Layer<S>
    for StageTimingLayer
{
    fn on_enter(&self, id: &span::Id, _ctx: Context<'_, S>) {
        self.starts
            .lock()
            .unwrap()
            .insert(id.clone(), Instant::now());
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        let Some(start) = self.starts.lock().unwrap().remove(id) else {
            return;
        };
        let Some(span) = ctx.span(id) else {
            return;
        };
        *self
            .durations
            .lock()
            .unwrap()
            .entry(span.name())
            .or_default() += start.elapsed();
    }
}
