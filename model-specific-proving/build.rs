use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let air = out.join("projection.air");
    let metal = out.join("projection.metallib");
    let object = out.join("projection.o");
    let library = out.join("libatlas_projection.a");
    for args in [
        vec![
            "metal",
            "-O3",
            "-c",
            "src/projection.metal",
            "-o",
            air.to_str().unwrap(),
        ],
        vec![
            "metallib",
            air.to_str().unwrap(),
            "-o",
            metal.to_str().unwrap(),
        ],
        vec![
            "clang",
            "-arch",
            "arm64",
            "-O3",
            "-fobjc-arc",
            "-c",
            "src/projection.m",
            "-o",
            object.to_str().unwrap(),
        ],
        vec![
            "ar",
            "crs",
            library.to_str().unwrap(),
            object.to_str().unwrap(),
        ],
    ] {
        assert!(
            Command::new("xcrun")
                .args(&args)
                .status()
                .expect("run the Xcode toolchain")
                .success(),
            "{args:?}"
        );
    }
    println!("cargo:rerun-if-changed=src/projection.metal");
    println!("cargo:rerun-if-changed=src/projection.m");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=atlas_projection");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
}
