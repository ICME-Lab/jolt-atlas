//! Fixed-point sine and cosine: quadrant reduction against the committed
//! π/2 constant, then Taylor series in Q2.62 (`i128`).
//!
//! ## Error bound
//!
//! The reduced argument `r ∈ [0, π/2]` is exact up to `q` rounding units of
//! [`HALF_PI_Q64`] (each < 2^-65), with `q` ≤ a few hundred for every
//! supported scale, so the argument error is < 2^-55. The alternating
//! Taylor terms for sin/cos on `[0, π/2]` decrease in magnitude from the
//! second term on; stopping at the first zero term bounds truncation by one
//! Q62 unit per term over ≤ ~20 terms. Total error ≪ 2^-50, against a
//! coarsest table quantum of 2^-15.

use super::coeffs::HALF_PI_Q64;

/// `(sin θ, cos θ)` for `θ ≥ 0` given in Q64.64, each returned in signed
/// Q2.62 (values in `[-1, 1]`).
pub fn sin_cos_q62(theta_q64: u128) -> (i64, i64) {
    let q = theta_q64 / HALF_PI_Q64;
    let r_q64 = theta_q64 - q * HALF_PI_Q64;
    let (s, c) = sin_cos_taylor((r_q64 >> 2) as i128);
    match q % 4 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Taylor evaluation of `(sin r, cos r)` for `r ∈ [0, π/2]` in Q2.62.
fn sin_cos_taylor(r: i128) -> (i64, i64) {
    let r2 = (r * r) >> 62;

    // sin r = Σ (-1)^n r^(2n+1) / (2n+1)!
    let mut sin = r;
    let mut term = r;
    let mut n: i128 = 0;
    loop {
        term = -(((term * r2) >> 62) / ((2 * n + 2) * (2 * n + 3)));
        if term == 0 {
            break;
        }
        sin += term;
        n += 1;
    }

    // cos r = Σ (-1)^n r^(2n) / (2n)!
    let mut cos: i128 = 1 << 62;
    let mut term: i128 = 1 << 62;
    let mut n: i128 = 0;
    loop {
        term = -(((term * r2) >> 62) / ((2 * n + 1) * (2 * n + 2)));
        if term == 0 {
            break;
        }
        cos += term;
        n += 1;
    }

    (sin as i64, cos as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(theta: f64) {
        let theta_q64 = (theta * (1u128 << 64) as f64) as u128;
        let (s, c) = sin_cos_q62(theta_q64);
        // Evaluate the oracle at the quantized angle.
        let tq = theta_q64 as f64 / (1u128 << 64) as f64;
        let unit = (1u64 << 62) as f64;
        let tol = unit * 1e-12 + 8.0;
        assert!(
            (s as f64 - tq.sin() * unit).abs() <= tol,
            "sin({theta}): got {s}, expected {}",
            tq.sin() * unit
        );
        assert!(
            (c as f64 - tq.cos() * unit).abs() <= tol,
            "cos({theta}): got {c}, expected {}",
            tq.cos() * unit
        );
    }

    #[test]
    fn matches_f64_oracle_over_many_periods() {
        for i in 0..500 {
            check(i as f64 * 0.61803398875); // golden-ratio stride, hits all quadrants
        }
        for &t in &[
            0.0,
            1e-9,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            100.0,
            314.159,
        ] {
            check(t);
        }
    }

    #[test]
    fn pythagorean_identity() {
        let unit = 1i128 << 62;
        for i in 0..1000u32 {
            let theta_q64 = (i as u128) << 56; // 0 .. ~3.9 in steps of 2^-8
            let (s, c) = sin_cos_q62(theta_q64);
            let ss_cc = ((s as i128 * s as i128) >> 62) + ((c as i128 * c as i128) >> 62);
            let diff = (ss_cc - unit).abs();
            assert!(diff < 1 << 14, "sin²+cos² off by {diff} at step {i}");
        }
    }
}
