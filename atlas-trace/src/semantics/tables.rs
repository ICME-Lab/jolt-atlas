//! Lookup-table generation: THE specification of every nonlinear operator.
//!
//! The runtime kernels are plain lookups into these tables, and the prover
//! materializes its proof tables by calling the same functions, so tracer
//! and prover agree bit for bit by construction.
//!
//! Every entry is a pure function of `(kind, scale, version)`, computed in
//! integer arithmetic only (see the parent module's contract). Entries are
//! rounded from Q2.62 approximations whose error is < 2^-50 relative, far
//! below the half-ulp of the coarsest output quantum (2^-16), so an entry
//! can differ from the mathematically rounded value only on a tie within
//! 2^-50 of half — and on such ties the committed table, enforced by golden
//! hashes in `tests/`, is the spec.

use crate::id::{Scale, SemVersion};

use super::{
    coeffs::TWO_OVER_SQRT_PI_Q62,
    exp::exp_neg_q62,
    fixed::{trig_downscale_bits, trig_period_modulus},
    trig::sin_cos_q62,
};

/// Which activation table to generate.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum TableKind {
    /// `tanh(x/S)·S` over the symmetric clamped domain.
    Tanh,
    /// `S / (1 + e^(-x/S))` over the symmetric clamped domain.
    Sigmoid,
    /// `erf(x/S)·S` over the symmetric clamped domain.
    Erf,
    /// `sin` over the period-reduced, downshifted domain.
    Sin,
    /// `cos` over the period-reduced, downshifted domain.
    Cos,
}

/// How a quantized input maps to a table index. Part of the kernel spec.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum IndexRule {
    /// `idx = clamp(x, -2^bound_log, 2^bound_log - 1) + 2^bound_log`;
    /// table size `2^(bound_log+1)`.
    SymClamp {
        /// log2 of the symmetric clamp bound (`scale + 3`).
        bound_log: u32,
    },
    /// `idx = (x mod modulus) >> downshift` (Euclidean mod, so the index is
    /// always in range); table size `((modulus-1) >> downshift) + 1`.
    TrigReduced {
        /// The period modulus `round(k·2π·2^scale)`.
        modulus: u32,
        /// Right-shift applied after reduction to bound the table size.
        downshift: u32,
    },
}

impl IndexRule {
    /// Map a quantized input to its table index.
    pub fn apply(&self, x: i32) -> usize {
        match *self {
            IndexRule::SymClamp { bound_log } => {
                let bound = 1i32 << bound_log;
                (x.clamp(-bound, bound - 1) + bound) as usize
            }
            IndexRule::TrigReduced { modulus, downshift } => {
                (x.rem_euclid(modulus as i32) as u32 >> downshift) as usize
            }
        }
    }
}

/// A generated activation table: the entries plus the index rule that
/// addresses them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ActivationTable {
    /// Quantized output per index.
    pub entries: Vec<i32>,
    /// How inputs map to indices.
    pub index: IndexRule,
}

impl ActivationTable {
    /// Evaluate the activation on a quantized input.
    pub fn lookup(&self, x: i32) -> i32 {
        self.entries[self.index.apply(x)]
    }
}

/// Generate the table for `kind` at `scale`, version `version`.
///
/// This function IS the proof spec for the nonlinear operators: any change
/// to any entry for any input requires a new [`SemVersion`].
pub fn table(kind: TableKind, scale: Scale, version: SemVersion) -> ActivationTable {
    match version {
        SemVersion::V1 => {}
    }
    match kind {
        TableKind::Tanh => sym_clamp_table(scale, tanh_entry),
        TableKind::Sigmoid => sym_clamp_table(scale, sigmoid_entry),
        TableKind::Erf => sym_clamp_table(scale, erf_entry),
        TableKind::Sin => trig_table(scale, true),
        TableKind::Cos => trig_table(scale, false),
    }
}

/// `1.0` in Q2.62.
const ONE_Q62: u128 = 1 << 62;

/// Round a non-negative Q2.62 value to the quantized output scale `S`:
/// `round(v · S)` with ties away from zero resolved upward.
fn round_q62(v_q62: u128, s: i64) -> i32 {
    (((v_q62 * s as u128) + (1 << 61)) >> 62) as i32
}

/// Build a symmetric-clamp table (`tanh`/`sigmoid`/`erf` family): domain
/// `[-2^(s+3), 2^(s+3))`, one entry per quantized input.
fn sym_clamp_table(scale: Scale, entry: impl Fn(i32, Scale) -> i32) -> ActivationTable {
    let bound_log = scale.log2() + 3;
    let bound = 1i32 << bound_log;
    let entries = (-bound..bound).map(|x| entry(x, scale)).collect();
    ActivationTable {
        entries,
        index: IndexRule::SymClamp { bound_log },
    }
}

/// `round(S · tanh(x/S))` via `tanh(u) = sgn(u)·(1-E)/(1+E)` with
/// `E = e^(-2|u|)`.
fn tanh_entry(x: i32, scale: Scale) -> i32 {
    let two_abs_u_q32 = (2 * x.unsigned_abs() as u64) << (32 - scale.log2());
    let e = exp_neg_q62(two_abs_u_q32) as u128;
    let t_q62 = ((ONE_Q62 - e) << 62) / (ONE_Q62 + e);
    let magnitude = round_q62(t_q62, scale.denom_i64());
    if x < 0 { -magnitude } else { magnitude }
}

/// `round(S / (1 + e^(-x/S)))`: for `x ≥ 0`, `1/(1+E)`; for `x < 0`,
/// `E/(1+E)` with `E = e^(-|u|)`.
fn sigmoid_entry(x: i32, scale: Scale) -> i32 {
    let abs_u_q32 = (x.unsigned_abs() as u64) << (32 - scale.log2());
    let e = exp_neg_q62(abs_u_q32) as u128;
    let v_q62 = if x >= 0 {
        (ONE_Q62 << 62) / (ONE_Q62 + e)
    } else {
        (e << 62) / (ONE_Q62 + e)
    };
    round_q62(v_q62, scale.denom_i64())
}

/// `round(S · erf(x/S))`, saturating to `±S` for `|x/S| ≥ 4`
/// (`1 - erf(4) < 2^-26`, far below the half-ulp `2^-16` at the largest
/// supported scale).
fn erf_entry(x: i32, scale: Scale) -> i32 {
    let s = scale.denom_i32();
    let abs = x.unsigned_abs();
    let magnitude = if abs >= 4 * s as u32 {
        s
    } else {
        let u_q32 = (abs as u64) << (32 - scale.log2());
        round_q62(erf_q62(u_q32), scale.denom_i64())
    };
    if x < 0 { -magnitude } else { magnitude }
}

/// `erf(u)` in Q0.62 for `u ∈ [0, 4)` given in Q32.32, by the positive-term
/// series (Abramowitz & Stegun 7.1.6):
///
/// ```text
/// erf(u) = (2/√π) · u · e^(-u²) · Σ_{n≥0} (2u²)^n / (1·3·5···(2n+1))
/// ```
///
/// All terms are positive — no cancellation — and for `u < 4` the running
/// term fits `u128` with room to spare (peak partial sum < 2^22 · 2^62).
/// The series is cut at the first zero term; combined truncation and Q62
/// rounding error is < 2^-50 relative.
fn erf_q62(u_q32: u64) -> u128 {
    let u_q62 = (u_q32 as u128) << 30;
    let u2_q62 = (u_q62 * u_q62) >> 62;
    let two_u2_q32 = u2_q62 >> 29;
    let e = exp_neg_q62((u2_q62 >> 30) as u64) as u128;

    // sum = Σ (2u²)^n / (2n+1)!!  in Q62; term recurrence t ← t·2u²/(2n+3).
    let mut sum: u128 = ONE_Q62;
    let mut term: u128 = ONE_Q62;
    let mut n: u128 = 0;
    loop {
        term = ((term * two_u2_q32) >> 32) / (2 * n + 3);
        if term == 0 {
            break;
        }
        sum += term;
        n += 1;
    }

    // (2/√π) · u · e^(-u²) · sum, ordered so no product exceeds u128:
    // sum·e ≈ (√π/2)·erf(u)/u · 2^124 ≤ ~2^124.
    let a = (sum * e) >> 62;
    let b = (a * u_q62) >> 62;
    (b * TWO_OVER_SQRT_PI_Q62 as u128) >> 62
}

/// Build the sin/cos table: domain is the period-reduced remainder
/// `[0, modulus)` downshifted by `d`; the entry for index `v` is
/// `round(S'·f(v/S'))·2^d` with `S' = 2^(scale-d)` — exactly the value the
/// old crate's two-stage `eval_trig` kernel produced.
fn trig_table(scale: Scale, sin: bool) -> ActivationTable {
    let modulus = trig_period_modulus(scale.log2());
    let downshift = trig_downscale_bits(scale.log2());
    assert!(
        downshift < scale.log2(),
        "trig downshift {downshift} must be below scale {scale}"
    );
    let reduced_log = scale.log2() - downshift;
    // Table size is the power of two covering the reduced modulus: runtime
    // lookups stay below `((modulus-1) >> downshift) + 1`, but the committed
    // table the prover materializes is power-of-two sized, and its tail
    // entries follow the same generation rule.
    let size = (modulus.next_power_of_two() >> downshift) as usize;
    let s_reduced = 1i64 << reduced_log;

    let entries = (0..size)
        .map(|v| {
            // θ = v / S' exactly, in Q64.64.
            let theta_q64 = (v as u128) << (64 - reduced_log);
            let (sin_q62, cos_q62) = sin_cos_q62(theta_q64);
            let f_q62 = if sin { sin_q62 } else { cos_q62 };
            // round(S'·f), ties upward (arithmetic shift), then ×2^d.
            let reduced = (((f_q62 as i128 * s_reduced as i128) + (1 << 61)) >> 62) as i32;
            reduced << downshift
        })
        .collect();

    ActivationTable {
        entries,
        index: IndexRule::TrigReduced { modulus, downshift },
    }
}

/// Decomposed exp sub-tables for softmax, exploiting `e^(a+b) = e^a·e^b`:
/// the centered logit `z` splits into digits `z = z_hi·B + z_lo`, and
/// `exp_q = ⌊LUT_hi[z_hi]·LUT_lo[z_lo]/S⌋`. Same structure and sizing as
/// the old crate (its `generate_exp_lut_decomposed`), with entries from the
/// integer exp core instead of platform f64.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpLutDecomposed {
    /// `LUT_hi[h] = round(S·e^(-h·B/S))`, `h ∈ [0, hi_size)`.
    pub lut_hi: Vec<i32>,
    /// `LUT_lo[l] = round(S·e^(-l/S))`, `l ∈ [0, B)`.
    pub lut_lo: Vec<i32>,
    /// Digit base `B` (power of two).
    pub base: usize,
    /// `log2(B)`, for bit-shift digit extraction.
    pub log2_base: u32,
}

/// Generate the softmax exp sub-tables for `scale`, version `version`.
///
/// Sizing (all integer arithmetic, matching the old f64 computation exactly
/// for every supported scale):
/// - cutoff `needed = ⌈S·ln(2S)⌉ + 2 = ⌈S·(s+1)·ln2⌉ + 2` — beyond it
///   `e^(-z/S)·S` rounds to 0;
/// - digit base `B` = the smallest power of four's square root covering
///   `needed` (`B = 2^b` with the smallest `b` such that `4^b ≥ needed`);
/// - `hi_size = needed/B + 2`.
pub fn exp_lut_decomposed(scale: Scale, version: SemVersion) -> ExpLutDecomposed {
    match version {
        SemVersion::V1 => {}
    }
    let s = scale.denom_i64() as u128;
    let s_log = scale.log2();

    // needed = ceil(S·(s+1)·ln2) + 2, in Q62.
    let t = s * (s_log as u128 + 1) * super::coeffs::LN2_Q62 as u128;
    let needed = ((t + (1u128 << 62) - 1) >> 62) as usize + 2;

    // Smallest b with 4^b >= needed.
    let mut log2_base = 0u32;
    while (1usize << (2 * log2_base)) < needed {
        log2_base += 1;
    }
    let base = 1usize << log2_base;
    let hi_size = needed / base + 2;

    let entry = |z: u64| -> i32 {
        let e = exp_neg_q62(z << (32 - s_log)) as u128;
        round_q62(e, s as i64)
    };
    ExpLutDecomposed {
        lut_hi: (0..hi_size).map(|h| entry((h * base) as u64)).collect(),
        lut_lo: (0..base).map(|l| entry(l as u64)).collect(),
        base,
        log2_base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s14() -> Scale {
        Scale::new(14).unwrap()
    }

    #[test]
    fn tanh_table_properties() {
        let t = table(TableKind::Tanh, s14(), SemVersion::V1);
        let s = 1 << 14;
        let bound = 1i32 << 17;
        assert_eq!(t.entries.len(), 1 << 18);
        // Odd symmetry: tanh(-x) = -tanh(x) (skip the unpaired -bound entry).
        for x in 1..bound {
            assert_eq!(t.lookup(-x), -t.lookup(x), "odd symmetry at {x}");
        }
        assert_eq!(t.lookup(0), 0);
        // Range and saturation.
        assert!(t.entries.iter().all(|&e| e.abs() <= s));
        assert_eq!(t.lookup(bound - 1), s);
        // Monotone non-decreasing.
        assert!(t.entries.windows(2).all(|w| w[0] <= w[1]));
        // Clamp rule: everything beyond the bound saturates.
        assert_eq!(t.lookup(i32::MAX), t.lookup(bound - 1));
        assert_eq!(t.lookup(i32::MIN), t.lookup(-bound));
    }

    #[test]
    fn sigmoid_table_properties() {
        let t = table(TableKind::Sigmoid, s14(), SemVersion::V1);
        let s = 1i32 << 14;
        // sigmoid(0) = 1/2.
        assert_eq!(t.lookup(0), s / 2);
        // Complement symmetry: sigmoid(x) + sigmoid(-x) = S (±1 on ties).
        for x in 1..(1i32 << 17) {
            let sum = t.lookup(x) + t.lookup(-x);
            assert!((sum - s).abs() <= 1, "complement off at {x}: {sum}");
        }
        assert!(t.entries.windows(2).all(|w| w[0] <= w[1]));
        assert!(t.entries.iter().all(|&e| (0..=s).contains(&e)));
    }

    #[test]
    fn sigmoid_matches_old_doc_examples() {
        // Old crate doctest: sigmoid(256 at scale 8) = 187.
        let t = table(TableKind::Sigmoid, Scale::new(8).unwrap(), SemVersion::V1);
        assert_eq!(t.lookup(256), 187);
    }

    #[test]
    fn erf_table_properties() {
        let t = table(TableKind::Erf, s14(), SemVersion::V1);
        let s = 1i32 << 14;
        assert_eq!(t.lookup(0), 0);
        for x in 1..(1i32 << 17) {
            assert_eq!(t.lookup(-x), -t.lookup(x), "odd symmetry at {x}");
        }
        // Saturation at |u| >= 4.
        assert_eq!(t.lookup(4 * s), s);
        assert!(t.entries.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn activation_tables_match_f64_oracle_within_one() {
        // Entry-by-entry comparison against the f64 definition the old
        // crate used; ≤1 ulp differences allowed (rounding ties).
        let scale = s14();
        let s = (1u32 << 14) as f64;
        type Oracle = fn(f64) -> f64;
        let cases: [(TableKind, Oracle); 3] = [
            (TableKind::Tanh, |u| u.tanh()),
            (TableKind::Sigmoid, |u| 1.0 / (1.0 + (-u).exp())),
            (TableKind::Erf, libm::erf),
        ];
        for (kind, f) in cases {
            let t = table(kind, scale, SemVersion::V1);
            let bound = 1i32 << 17;
            let mut diffs = 0u32;
            for x in -bound..bound {
                let expected = (s * f(x as f64 / s)).round() as i32;
                let got = t.lookup(x);
                assert!(
                    (got - expected).abs() <= 1,
                    "{kind:?}[{x}]: got {got}, f64 oracle {expected}"
                );
                diffs += (got != expected) as u32;
            }
            // Ties should be rare: well under 0.1% of entries.
            assert!(diffs < (1 << 18) / 1000, "{kind:?}: {diffs} tie diffs");
        }
    }

    #[test]
    fn trig_tables_match_f64_oracle_within_one() {
        let scale = s14();
        for (kind, f) in [
            (TableKind::Sin, f64::sin as fn(f64) -> f64),
            (TableKind::Cos, f64::cos as fn(f64) -> f64),
        ] {
            let t = table(kind, scale, SemVersion::V1);
            let IndexRule::TrigReduced { modulus, downshift } = t.index else {
                panic!("wrong index rule");
            };
            let s_reduced = (1u32 << (14 - downshift)) as f64;
            let size = (modulus.next_power_of_two() >> downshift) as usize;
            assert_eq!(t.entries.len(), size);
            for v in 0..size {
                let expected = ((s_reduced * f(v as f64 / s_reduced)).round() as i32) << downshift;
                let got = t.entries[v];
                assert!(
                    (got - expected).abs() <= 1 << downshift,
                    "{kind:?}[{v}]: got {got}, f64 oracle {expected}"
                );
            }
        }
    }

    #[test]
    fn exp_lut_structure_matches_old_sizing() {
        // Old f64 sizing at scale 14: needed = ceil(16384·ln(32768)) + 2.
        let lut = exp_lut_decomposed(s14(), SemVersion::V1);
        let needed = (16384.0f64 * (2.0 * 16384.0f64).ln()).ceil() as usize + 2;
        let log2_b = ((needed as f64).log2() / 2.0).ceil() as u32;
        assert_eq!(lut.log2_base, log2_b);
        assert_eq!(lut.base, 1 << log2_b);
        assert_eq!(lut.lut_hi.len(), needed / lut.base + 2);
        assert_eq!(lut.lut_lo.len(), lut.base);
        // Entry oracle check (≤1 for ties).
        let s = 16384.0f64;
        for (h, &e) in lut.lut_hi.iter().enumerate() {
            let expected = (s * (-(h as f64 * lut.base as f64) / s).exp()).round() as i32;
            assert!((e - expected).abs() <= 1, "lut_hi[{h}]: {e} vs {expected}");
        }
        for (l, &e) in lut.lut_lo.iter().enumerate() {
            let expected = (s * (-(l as f64) / s).exp()).round() as i32;
            assert!((e - expected).abs() <= 1, "lut_lo[{l}]: {e} vs {expected}");
        }
        // First entries are exact: e^0 = 1.
        assert_eq!(lut.lut_hi[0], 16384);
        assert_eq!(lut.lut_lo[0], 16384);
    }

    #[test]
    fn exp_lut_sizing_integer_matches_f64_for_all_scales() {
        for s in 1..=15u8 {
            let scale = Scale::new(s).unwrap();
            let lut = exp_lut_decomposed(scale, SemVersion::V1);
            let sf = (1u32 << s) as f64;
            let needed = (sf * (2.0 * sf).ln()).ceil() as usize + 2;
            let log2_b = ((needed as f64).log2() / 2.0).ceil() as u32;
            assert_eq!(lut.log2_base, log2_b, "scale {s}");
            assert_eq!(lut.lut_hi.len(), needed / (1 << log2_b) + 2, "scale {s}");
        }
    }
}
