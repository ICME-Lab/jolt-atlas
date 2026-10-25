//! The one fixed-point exponential core: `e^-x` in pure integer arithmetic.
//!
//! Every exponential-family table entry (tanh, sigmoid, the softmax exp
//! sub-tables, erf's `e^{-u²}` factor) is generated through this function,
//! so its error analysis covers them all.
//!
//! ## Method
//!
//! Split `x = n + f` with `n` integer and `f ∈ [0, 1)`:
//! - `e^-f` by the Taylor series `Σ (-f)^k / k!` in Q2.62 (`i128`).
//!   Terms decrease strictly in magnitude, so stopping at the first zero
//!   term bounds the truncation error by one Q62 unit; ≤ ~25 terms.
//! - `e^-n` by binary exponentiation of the committed constant
//!   [`EXP_NEG1_Q62`], ≤ 2·log₂(n) Q62 multiplications.
//!
//! ## Error bound
//!
//! Each Q62 multiplication/truncation contributes < 2^-62 relative error,
//! and the series and exponentiation together perform < 50 of them, so the
//! result is within 2^-56 relative of the exact value — about 2^40× finer
//! than the coarsest table quantum (2^-15). The committed table (not ℝ) is
//! the spec; this bound only says rounding ties are the sole possible
//! difference from exact rounding.

use super::coeffs::EXP_NEG1_Q62;

/// `e^-x` for `x ≥ 0` given in Q32.32, returned in Q0.62.
///
/// Returns 0 for `x ≥ 45` (`e^-45 < 2^-64`, below Q62 resolution).
pub fn exp_neg_q62(x_q32: u64) -> u64 {
    let n = (x_q32 >> 32) as u32;
    if n >= 45 {
        return 0;
    }
    let f_q62 = ((x_q32 & 0xFFFF_FFFF) as i128) << 30;

    // e^-f via Taylor: acc = Σ_k (-f)^k / k!, term_k = (-f)^k / k!.
    let mut acc: i128 = 1 << 62;
    let mut term: i128 = 1 << 62;
    let mut k: i128 = 1;
    loop {
        term = -((term * f_q62) >> 62) / k;
        if term == 0 {
            break;
        }
        acc += term;
        k += 1;
    }

    // Multiply by (e^-1)^n via binary exponentiation.
    let mut result = acc as u128;
    let mut base = EXP_NEG1_Q62 as u128;
    let mut n = n;
    while n > 0 {
        if n & 1 == 1 {
            result = (result * base) >> 62;
        }
        base = (base * base) >> 62;
        n >>= 1;
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert an f64 in [0, 64) to Q32.32.
    fn q32(x: f64) -> u64 {
        (x * (1u64 << 32) as f64).round() as u64
    }

    #[test]
    fn matches_f64_oracle() {
        // The oracle is only f64-accurate (~2^-52 relative), so compare with
        // a tolerance of a few thousand Q62 units scaled by the value.
        for &x in &[
            0.0, 1e-9, 0.1, 0.5, 0.9999, 1.0, 1.5, 2.0, 3.17, 8.0, 11.0, 16.0, 20.0, 30.0,
        ] {
            let got = exp_neg_q62(q32(x)) as f64;
            // The Q32.32 input quantization itself shifts the true answer:
            // evaluate the oracle at the quantized x.
            let xq = q32(x) as f64 / (1u64 << 32) as f64;
            let expected = (-xq).exp() * (1u64 << 62) as f64;
            let tol = expected * 1e-12 + 4.0;
            assert!(
                (got - expected).abs() <= tol,
                "x={x}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn exact_at_zero_and_underflow() {
        assert_eq!(exp_neg_q62(0), 1 << 62);
        assert_eq!(exp_neg_q62(45u64 << 32), 0);
        assert_eq!(exp_neg_q62(u64::MAX), 0);
    }

    #[test]
    fn monotone_decreasing() {
        let mut prev = u64::MAX;
        for i in 0..4096u64 {
            // Sweep x from 0 to 16 in steps of 1/256.
            let v = exp_neg_q62(i << 24);
            assert!(v <= prev, "not monotone at step {i}");
            prev = v;
        }
    }
}
