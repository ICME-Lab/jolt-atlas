//! Log-partition softmax: the tables and the row shift.
//!
//! Softmax over a row is `exp(x_j − L)` with `L` the logarithm of the row's partition sum. In
//! integers at scale `S`, with `S' = S·2^SUM_EXTRA_BITS` and `top` the last table address:
//!
//! ```text
//! T_out[u] = round(S ·e^(−u/S))                 the output table
//! T_sum[u] = round(S'·e^(−u/S))                 the same function with more fractional bits
//! f(c)     = Σ_j T_sum[min(c − x_j, top)]       the row sum at shift c
//! c        = min { c ≥ max_j x_j : f(c) ≤ S' }  the row's shift
//! y_j      = T_out[min(c − x_j, top)]           the output
//! ```
//!
//! `f` is nonincreasing in `c` and zero for `c ≥ max_j x_j + top`, so the shift exists and is
//! unique: the output is a function of the row. The shift is pinned by the finer table and the
//! output read from the coarser one so that the rounding of many small entries does not move
//! the large ones.

use super::exp::exp_neg_q62;
use crate::id::{Scale, SemVersion};

/// The extra fractional bits of the sum table over the output table.
pub const SUM_EXTRA_BITS: u32 = 8;

/// The two exponential tables of the log-partition softmax, over addresses `[0, 2^address_bits)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SoftmaxTables {
    /// `T_out[u] = round(S·e^(−u/S))`.
    pub out: Vec<i32>,
    /// `T_sum[u] = round(S·2^SUM_EXTRA_BITS·e^(−u/S))`.
    pub sum: Vec<i32>,
    /// log2 of the tables' length: the smallest width at which `T_sum` is zero on its two top
    /// entries.
    pub address_bits: u32,
    /// `S' = S·2^SUM_EXTRA_BITS`, the scale the row sum is compared with.
    pub sum_scale: i64,
}

/// Generate the softmax tables for `scale`, version `version`.
pub fn softmax_tables(scale: Scale, version: SemVersion) -> SoftmaxTables {
    match version {
        SemVersion::V1 | SemVersion::V2 => {}
    }
    let s = scale.denom_i64();
    let sum_scale = s << SUM_EXTRA_BITS;
    let shift = 32 - scale.log2();
    // round(v·e^(−u/S)), ties upward.
    let entry = |u: u64, v: i64| -> i32 {
        let e = exp_neg_q62(u << shift) as u128;
        (((e * v as u128) + (1 << 61)) >> 62) as i32
    };
    let mut address_bits = 1u32;
    while entry((1u64 << address_bits) - 2, sum_scale) != 0 {
        address_bits += 1;
    }
    let len = 1u64 << address_bits;
    SoftmaxTables {
        out: (0..len).map(|u| entry(u, s)).collect(),
        sum: (0..len).map(|u| entry(u, sum_scale)).collect(),
        address_bits,
        sum_scale,
    }
}

impl SoftmaxTables {
    /// The last table address; larger addresses clamp to it and read zero.
    pub fn top(&self) -> i64 {
        (1i64 << self.address_bits) - 1
    }

    /// The table address of logit `x` at shift `c`: `min(c − x, top)`. Requires `c ≥ x`.
    pub fn address(&self, c: i64, x: i32) -> usize {
        (c - x as i64).min(self.top()) as usize
    }

    /// `f(c)`: the row's sum-table reads at shift `c`. Requires `c ≥ x` for every logit.
    pub fn row_sum(&self, c: i64, row: &[i32]) -> i64 {
        row.iter()
            .map(|&x| self.sum[self.address(c, x)] as i64)
            .sum()
    }

    /// The row's shift: the smallest `c` at or above the row's maximum with `f(c) ≤ S'`.
    /// `None` for an empty row.
    pub fn shift(&self, row: &[i32]) -> Option<i64> {
        let max = *row.iter().max()? as i64;
        // f(max + top) = 0, since every address clamps to `top`.
        let (mut lo, mut hi) = (max, max + self.top());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.row_sum(mid, row) <= self.sum_scale {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Some(lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(scale: u8) -> SoftmaxTables {
        softmax_tables(Scale::new(scale).unwrap(), SemVersion::V2)
    }

    /// A small deterministic generator, so the tests need no dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn uniform(&mut self) -> f64 {
            (self.next() as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn gaussian(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    #[test]
    fn table_shape() {
        let t = tables(14);
        let s = 1i32 << 14;
        assert_eq!(t.address_bits, 18);
        assert_eq!(t.out[0], s);
        assert_eq!(t.sum[0], s << SUM_EXTRA_BITS);
        let len = t.sum.len();
        assert_eq!(&t.sum[len - 2..], &[0, 0]);
        assert_ne!(t.sum[(len >> 1) - 2], 0, "one bit fewer would not do");
        assert!(t.sum.windows(2).all(|w| w[0] >= w[1]), "nonincreasing");
        assert!(t.out.windows(2).all(|w| w[0] >= w[1]), "nonincreasing");
        // A zero sum entry is a zero output entry.
        assert!(
            t.sum
                .iter()
                .zip(&t.out)
                .all(|(&sum, &out)| sum != 0 || out == 0)
        );
    }

    #[test]
    fn entries_match_f64_oracle_within_one() {
        let t = tables(14);
        let s = (1i64 << 14) as f64;
        for (u, (&out, &sum)) in t.out.iter().zip(&t.sum).enumerate() {
            let e = (-(u as f64) / s).exp();
            assert!((out as f64 - s * e).abs() <= 1.0, "T_out[{u}]");
            assert!((sum as f64 - 256.0 * s * e).abs() <= 1.0, "T_sum[{u}]");
        }
    }

    /// The shift by definition: scan upward from the row maximum.
    fn shift_by_scan(t: &SoftmaxTables, row: &[i32]) -> i64 {
        let mut c = *row.iter().max().unwrap() as i64;
        while t.row_sum(c, row) > t.sum_scale {
            c += 1;
        }
        c
    }

    #[test]
    fn shift_is_the_smallest_admissible() {
        let t = tables(14);
        let mut rng = Lcg(0x50F7);
        let mut rows: Vec<Vec<i32>> = vec![
            vec![0; 16],
            vec![7],
            vec![1 << 20, 0, 0, 0],
            vec![i32::MAX, i32::MIN, 0],
            vec![5, 5],
            vec![0, -300_000, -300_001],
        ];
        for n in [2usize, 3, 16, 64] {
            for spread in [200.0, 8_000.0, 60_000.0] {
                rows.push((0..n).map(|_| (rng.gaussian() * spread) as i32).collect());
            }
        }
        for row in &rows {
            let c = t.shift(row).unwrap();
            assert_eq!(c, shift_by_scan(&t, row), "{row:?}");
            let max = *row.iter().max().unwrap() as i64;
            assert!(c >= max);
            assert!(t.row_sum(c, row) <= t.sum_scale);
            assert!(c == max || t.row_sum(c - 1, row) > t.sum_scale);
        }
        assert_eq!(t.shift(&[]), None);
    }

    #[test]
    fn output_is_within_two_ulp_of_float_softmax() {
        let t = tables(14);
        let s = (1i64 << 14) as f64;
        let mut rng = Lcg(0xACC);
        let (mut worst_entry, mut worst_sum) = (0f64, 0f64);
        for n in [16usize, 64, 256, 1024] {
            for sigma in [0.5, 2.0, 6.0] {
                for _ in 0..20 {
                    let row: Vec<i32> = (0..n)
                        .map(|_| (rng.gaussian() * sigma * s) as i32)
                        .collect();
                    let c = t.shift(&row).unwrap();
                    let out: Vec<i32> = row.iter().map(|&x| t.out[t.address(c, x)]).collect();

                    let max = *row.iter().max().unwrap() as f64;
                    let exps: Vec<f64> =
                        row.iter().map(|&x| ((x as f64 - max) / s).exp()).collect();
                    let total: f64 = exps.iter().sum();
                    for (&y, e) in out.iter().zip(&exps) {
                        let err = (y as f64 - s * e / total).abs();
                        worst_entry = worst_entry.max(err);
                        assert!(err <= 2.0, "entry off by {err} ulp at n={n} sigma={sigma}");
                    }
                    let sum_err = (out.iter().map(|&y| y as i64).sum::<i64>() - (1 << 14)) as f64;
                    worst_sum = worst_sum.max(sum_err.abs() / n as f64);
                    assert!(
                        sum_err.abs() <= n as f64 / 2.0 + 2.0,
                        "row sum off by {sum_err} at n={n} sigma={sigma}"
                    );
                }
            }
        }
        println!(
            "worst entry error {worst_entry:.2} ulp; worst row-sum error {worst_sum:.3} per lane"
        );
    }
}
