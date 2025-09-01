#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use colored::*;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use env_logger::Builder;
use log::{Level, LevelFilter, Record};
use std::{env, fmt::Formatter, io::Write};

/// sets the log level color
#[allow(dead_code)]
pub fn level_color(level: &log::Level, msg: &str) -> String {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        match level {
            Level::Error => msg.red(),
            Level::Warn => msg.yellow(),
            Level::Info => msg.blue(),
            Level::Debug => msg.green(),
            Level::Trace => msg.magenta(),
        }
        .bold()
        .to_string()
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        msg.to_string()
    }
}

/// sets the log level text color
pub fn level_text_color(level: &log::Level, msg: &str) -> String {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        match level {
            Level::Error => msg.red(),
            Level::Warn => msg.yellow(),
            Level::Info => msg.white(),
            Level::Debug => msg.white(),
            Level::Trace => msg.white(),
        }
        .bold()
        .to_string()
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        msg.to_string()
    }
}

/// sets the log level token
fn level_token(level: &Level) -> &str {
    match *level {
        Level::Error => "E",
        Level::Warn => "W",
        Level::Info => "*",
        Level::Debug => "D",
        Level::Trace => "T",
    }
}

/// sets the log level prefix token
fn prefix_token(level: &Level) -> String {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        format!(
            "{}{}{}",
            "[".blue().bold(),
            level_color(level, level_token(level)),
            "]".blue().bold()
        )
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        format!("[{}]", level_token(level))
    }
}

/// formats the log
pub fn format(buf: &mut Formatter, record: &Record<'_>) -> Result<(), std::fmt::Error> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let sep = format!("\n{} ", " | ".white().bold());
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let sep = format!("\n{} ", " | ");

    let level = record.level();
    writeln!(
        buf,
        "{} {}",
        prefix_token(&level),
        level_color(&level, record.args().as_str().unwrap()).replace('\n', &sep),
    )
}

/// initializes the logger
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn init_logger() {
    let mut builder = Builder::new();

    builder.format(move |buf, record| {
        writeln!(
            buf,
            "{} [{}, {}] - {}",
            prefix_token(&record.level()),
            //    pretty print UTC time
            chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
                .bright_magenta(),
            record.metadata().target(),
            level_text_color(&record.level(), &format!("{}", record.args()))
                .replace('\n', &format!("\n{} ", " | ".white().bold()))
        )
    });
    builder.target(env_logger::Target::Stdout);
    builder.filter(None, LevelFilter::Info);
    if env::var("RUST_LOG").is_ok() {
        builder.parse_filters(&env::var("RUST_LOG").unwrap());
    }
    builder.init();
}

/// initializes the logger for WASM (no-op)
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub fn init_logger() {
    // No-op for WASM
}
