//! # The audit surface: deterministic integer semantics
//!
//! Everything in this module is a **pure function of integer inputs and
//! `(Scale, SemVersion)`**. The contract:
//!
//! - no `f32`/`f64` and no platform math intrinsics in any value that can
//!   reach a trace or a lookup table (the test suite may use floats as
//!   oracles);
//! - no `rayon`, no allocator-dependent iteration order, no `OnceLock`
//!   state — callers own caching;
//! - these functions ARE the proof specification: the prover materializes
//!   its lookup tables by calling them, and golden-hash tests pin every
//!   table. **Any change to any output for any input requires a new
//!   [`crate::id::SemVersion`].**
//!
//! A guard test (`no_float_outside_tests`) mechanically rejects float types
//! and rayon in this module's non-test code.
//!
//! Layout:
//! - [`fixed`] — the fused-rescale split, saturation, floor division,
//!   rsqrt, and the trig period constants;
//! - [`exp`] — the one fixed-point exponential core (`e^-x`, Q2.62);
//! - [`trig`] — fixed-point sin/cos (quadrant reduction + Taylor);
//! - [`tables`] — lookup-table generation for every nonlinear op;
//! - [`coeffs`] — the five committed constants with provenance.

pub mod coeffs;
pub mod exp;
pub mod fixed;
pub mod tables;
pub mod trig;

#[cfg(test)]
mod guard {
    /// Reject `f64`/`f32`/`rayon` in the non-test portion of every
    /// semantics source file. The test portion (after `#[cfg(test)]`) is
    /// exempt — oracles there are deliberate.
    #[test]
    fn no_float_outside_tests() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/semantics");
        for entry in std::fs::read_dir(dir).expect("semantics dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read source");
            let non_test = source.split("#[cfg(test)]").next().unwrap_or("");
            // Doc text may mention floats; only code lines count.
            let code: String = non_test
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            for needle in ["f64", "f32", "rayon"] {
                assert!(
                    !code.contains(needle),
                    "{needle} found outside tests in {}",
                    path.display()
                );
            }
        }
    }
}
