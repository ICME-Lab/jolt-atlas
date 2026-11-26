use std::{env, path::PathBuf, process::Command};

fn run(args: &[&str]) {
    let mut command = Command::new("xcrun");
    command.args(args);
    assert!(
        command.status().expect("run the Xcode toolchain").success(),
        "{command:?}"
    );
}

fn main() {
    if env::var_os("CARGO_FEATURE_METAL").is_none()
        || env::var("TARGET").as_deref() != Ok("aarch64-apple-darwin")
    {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/leaf.metal");
    println!("cargo:rerun-if-changed=src/metal.m");
    let air = out.join("leaf.air");
    let library = out.join("leaf.metallib");
    let object = out.join("metal.o");
    let archive = out.join("libbitz_metal.a");
    run(&["metal", "-c", "src/leaf.metal", "-o", air.to_str().unwrap()]);
    run(&[
        "metallib",
        air.to_str().unwrap(),
        "-o",
        library.to_str().unwrap(),
    ]);
    run(&[
        "clang",
        "-arch",
        "arm64",
        "-O3",
        "-fobjc-arc",
        "-c",
        "src/metal.m",
        "-o",
        object.to_str().unwrap(),
    ]);
    run(&[
        "ar",
        "crs",
        archive.to_str().unwrap(),
        object.to_str().unwrap(),
    ]);
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=bitz_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=objc");
}
