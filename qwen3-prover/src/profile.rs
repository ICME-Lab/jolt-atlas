pub(crate) fn measure<T>(name: &str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }

    let started = std::time::Instant::now();
    let result = f();
    eprintln!("profile {name}: {:.6}s", started.elapsed().as_secs_f64());
    result
}

pub(crate) fn measure_detail<T>(name: &str, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "profile-detail")]
    {
        measure(name, f)
    }

    #[cfg(not(feature = "profile-detail"))]
    {
        let _ = name;
        f()
    }
}

fn enabled() -> bool {
    std::env::var_os("QWEN3_PROFILE").is_some()
}
