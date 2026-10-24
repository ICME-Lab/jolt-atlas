//! Committed fixed-point constants.
//!
//! Every constant is `round(value · 2^fractional_bits)`, computed in
//! arbitrary-precision decimal arithmetic by `scripts/gen_constants.py`
//! (Python standard library only; 60 significant digits). Regenerate with:
//!
//! ```text
//! python3 scripts/gen_constants.py
//! ```
//!
//! Tests in this module cross-check each constant against `libm` (an
//! f64-precision oracle, so the check tolerates ~2^-52 relative error).
//! The committed value — not the oracle — is the specification.

/// π in Q64.64: `round(π · 2^64)`.
///
/// Identical to the constant the old crate committed in
/// `common/src/consts/trig.rs`, which independently validates both.
pub const PI_Q64: u128 = 57952155664616982739;

/// π/2 in Q64.64: `round(π/2 · 2^64)`.
pub const HALF_PI_Q64: u128 = 28976077832308491370;

/// e^-1 in Q0.62: `round(e^-1 · 2^62)`.
pub const EXP_NEG1_Q62: u64 = 1696544475317221319;

/// ln 2 in Q0.62: `round(ln 2 · 2^62)`.
pub const LN2_Q62: u64 = 3196577161300663915;

/// 2/√π in Q2.62: `round(2/√π · 2^62)`.
pub const TWO_OVER_SQRT_PI_Q62: u64 = 5203730428379116615;

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `constant ≈ value · 2^frac_bits` within `tol` units.
    fn check(constant: u128, value: f64, frac_bits: u32, tol: f64) {
        let expected = value * (frac_bits as f64).exp2();
        let diff = (constant as f64 - expected).abs();
        assert!(
            diff <= tol,
            "constant {constant} vs oracle {expected}: diff {diff} > tol {tol}"
        );
    }

    #[test]
    fn constants_match_f64_oracle() {
        // f64 has 52 fraction bits, so the oracle itself is only good to
        // ~|value|·2^-52; tolerances are sized accordingly.
        check(PI_Q64, std::f64::consts::PI, 64, 16384.0);
        check(HALF_PI_Q64, std::f64::consts::FRAC_PI_2, 64, 8192.0);
        check(EXP_NEG1_Q62.into(), (-1.0f64).exp(), 62, 1024.0);
        check(LN2_Q62.into(), std::f64::consts::LN_2, 62, 1024.0);
        check(
            TWO_OVER_SQRT_PI_Q62.into(),
            std::f64::consts::FRAC_2_SQRT_PI,
            62,
            1024.0,
        );
    }

    #[test]
    fn pi_matches_old_crate_commitment() {
        // The old crate committed round(π·2^64) in common/src/consts/trig.rs.
        assert_eq!(PI_Q64, 57_952_155_664_616_982_739);
    }
}
