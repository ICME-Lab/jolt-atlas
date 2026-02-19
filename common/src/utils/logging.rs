//! Tracing utility for examples and tests
//!
//! This module provides a simple interface for setting up tracing in examples,
//! supporting both Chrome Trace JSON output and terminal output with timing.

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{
    EnvFilter, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt,
};

#[allow(dead_code)]
/// Guard that ensures trace files are written when dropped.
/// This is an opaque wrapper around the optional ChromeLayer guard.
pub struct TracingGuard(Option<tracing_chrome::FlushGuard>);

/// Setup tracing based on command-line arguments.
///
/// Looks for `--trace` (Chrome JSON output) or `--trace-terminal` (terminal output)
/// in the command-line arguments and sets up tracing accordingly.
///
/// # Arguments
/// * `title` - The title to display when tracing is enabled
///
/// # Returns
/// A tuple of:
/// * `TracingGuard` - Must be kept alive until the end of main() to ensure trace output
/// * `bool` - Whether any tracing was enabled
pub fn setup_tracing(title: &str) -> (TracingGuard, bool) {
    let args: Vec<String> = std::env::args().collect();
    let enable_trace_json = args.contains(&"--trace".to_string());
    let enable_trace_terminal = args.contains(&"--trace-terminal".to_string());

    if enable_trace_json {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();

        println!("=== {title} (Chrome Tracing enabled) ===");
        println!("Trace output: trace-<timestamp>.json (viewable in chrome://tracing)\n");
        (TracingGuard(Some(guard)), true)
    } else if enable_trace_terminal {
        let fmt_layer = tracing_subscriber::fmt::layer().with_span_events(FmtSpan::CLOSE);
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();

        println!("=== {title} (Terminal Tracing enabled) ===");
        println!("Log level: Set via RUST_LOG environment variable (default: info)\n");
        (TracingGuard(None), true)
    } else {
        println!("=== {title} ===");
        println!("(Run with --trace for JSON output or --trace-terminal for terminal output)\n");
        (TracingGuard(None), false)
    }
}
