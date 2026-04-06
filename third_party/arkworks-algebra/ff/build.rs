fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let use_s2n_mul = std::env::var_os("CARGO_FEATURE_S2N_BIGNUM_AARCH64_MUL").is_some();
    let use_s2n_square = std::env::var_os("CARGO_FEATURE_S2N_BIGNUM_AARCH64_SQUARE").is_some();

    if target_arch == "aarch64" && (use_s2n_mul || use_s2n_square) {
        let s2n_root = std::path::Path::new("../../s2n-bignum");
        let include_dir = s2n_root.join("include");
        let mut build = cc::Build::new();
        build.include(include_dir);

        if use_s2n_mul {
            build.file(s2n_root.join("arm/generic/bignum_montmul.S"));
        }
        if use_s2n_square {
            build.file(s2n_root.join("arm/generic/bignum_montsqr.S"));
        }
        build.compile("s2n_bignum_arm_generic");
    }
}
