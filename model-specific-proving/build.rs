use std::{env, path::PathBuf, process::Command};

fn main() {
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("projection.metallib");
    let status = Command::new("xcrun")
        .args(["metal", "-O3", "src/projection.metal", "-o"])
        .arg(output)
        .status()
        .expect("run Xcode Metal compiler");
    assert!(status.success(), "compile Metal projection");
    cc::Build::new()
        .file("src/projection.m")
        .flag("-fobjc-arc")
        .opt_level(3)
        .compile("atlas_projection");
    println!("cargo:rerun-if-changed=src/projection.metal");
    println!("cargo:rerun-if-changed=src/projection.m");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
}
