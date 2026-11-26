use std::{env, path::Path, process::Command};

fn xcrun(out: &Path, name: &str, args: &[&str], output: &str) {
    let status = Command::new("xcrun")
        .arg(name)
        .args(args)
        .arg("-o")
        .arg(out.join(output))
        .status()
        .expect("could not run xcrun for the metal feature");
    assert!(status.success(), "failed to build {output}");
}

fn main() {
    if env::var_os("CARGO_FEATURE_METAL").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
    {
        return;
    }
    let out = env::var_os("OUT_DIR").unwrap();
    let out = Path::new(&out);
    let source = "src/protocol/metal";
    println!("cargo:rerun-if-changed={source}");
    xcrun(
        out,
        "metal",
        &["-O3", "-c", "src/protocol/metal/round.metal"],
        "round.air",
    );
    xcrun(
        out,
        "metallib",
        &[out.join("round.air").to_str().unwrap()],
        "round.metallib",
    );
    xcrun(
        out,
        "clang++",
        &[
            "-std=c++17",
            "-O3",
            "-fobjc-arc",
            "-c",
            "src/protocol/metal/device.mm",
        ],
        "device.o",
    );
    xcrun(
        out,
        "libtool",
        &["-static", out.join("device.o").to_str().unwrap()],
        "libatlas_metal.a",
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=atlas_metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=c++");
}
