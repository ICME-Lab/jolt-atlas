//! Piecewise-linear tables for tanh, erf, sigmoid, sin and cos, with a prefix-suffix
//! decomposition of their MLEs. No op reads these tables.
//!
//! ## The table
//!
//! The input domain is cut into segments of `2^SEGMENT_BITS` inputs. A table is its *knots*, the
//! function's quantized value at every segment boundary, and an input `x = seg · 2^W + lo` reads
//! the chord between its segment's two knots:
//!
//! ```text
//! table(x) = 2^W · knot[seg] + (knot[seg + 1] - knot[seg]) · lo        (W = SEGMENT_BITS)
//! ```
//!
//! The entry is the interpolated value with `W` extra fractional bits: `table(x) / 2^W` is the
//! activation at the model scale, and the entry is exact at every knot. Keeping the numerator
//! unrounded is what makes the table linear in `lo`; a rounded output would need a remainder
//! range check on top of the lookup.
//!
//! Two domain shapes ([`PwlSpec::SIGNED`]):
//! - signed (tanh, erf, sigmoid): `x` is a two's-complement `XLEN`-bit integer and the domain is
//!   `[-2^D, 2^D)`. Outside it the table holds the boundary knot, so the input clamp is part of
//!   the table. The knots are entries of the `atlas-trace` semantics table.
//! - unsigned (sin, cos): `x` is an unsigned `XLEN`-bit integer and the domain is `[0, 2^D)`,
//!   the power of two covering the period modulus. Outside it the table is zero. The knots are
//!   rounded from the `atlas-trace` sine/cosine core at the model scale.
//!
//! ## The decomposition
//!
//! Split the `XLEN` index bits into a prefix (high bits) and a suffix of `L` low bits. For every
//! `L` the MLE is a short inner product `Σ_i P_i(prefix) · S_i(suffix)`, where each `P_i` is
//! multilinear in the prefix and each `S_i` is an integer function of the suffix bits
//! ([`PiecewiseLinearTable::prefix_evals`], [`PiecewiseLinearTable::suffix_evals`]):
//!
//! - `L <= W`: the prefix fixes the segment, so the two knots are prefix functions and the table
//!   is linear in the suffix. Two terms, with suffixes `1` and the suffix's value.
//! - `L > W`: the prefix fixes a *block* of `2^L` consecutive inputs. One term per block that
//!   meets the domain, with the indicator of that block as the prefix and the table restricted
//!   to the block as the suffix; for signed tables the two saturation values are one more term.
//!   With `L >= D` that is one block per sign; below it the count doubles per bit.
//!
//! The knot prefixes of the first case are dense: bound to field elements, they are the knot
//! vector folded once per segment bit, so a streaming prover carries a partially folded knot
//! vector rather than one scalar per prefix. [`crate::prefixes::PrefixCheckpoints`] holds one
//! scalar per prefix, which is why these tables implement [`JoltLookupTable`] but not
//! [`crate::PrefixSuffixDecompositionTrait`].

use crate::JoltLookupTable;
use atlas_algebra::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};
use atlas_trace::{
    semantics::{
        tables::{table, TableKind},
        trig::sin_cos_q62,
    },
    Scale, SemVersion,
};
use common::consts::{ACTIVATION_BOUND, MODEL_SCALE, TRIG_PERIOD_MODULUS};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::OnceLock};

/// log2 of a segment's width in inputs, and the number of extra fractional bits in an entry.
pub const SEGMENT_BITS: usize = 8;

/// log2 of the sin/cos domain: the power of two covering the period modulus.
pub const TRIG_DOMAIN_BITS: usize = TRIG_PERIOD_MODULUS.next_power_of_two().ilog2() as usize;

/// A piecewise-linear table's domain and knots.
pub trait PwlSpec: Debug + Clone + Default + Send + Sync + 'static {
    /// `true`: two's-complement inputs over `[-2^D, 2^D)`, saturating outside.
    /// `false`: unsigned inputs over `[0, 2^D)`, zero outside.
    const SIGNED: bool;
    /// `D`, the log2 of the domain bound.
    const DOMAIN_BITS: usize;
    /// The function's value at every segment boundary, both domain ends included, in input
    /// order: `2^(D - W + 1) + 1` knots for a signed table, `2^(D - W) + 1` for an unsigned one.
    fn knots() -> &'static [i32];
}

/// The knots of a symmetric-clamp activation: the semantics table read at every segment
/// boundary. The table's own index rule clamps the last boundary, `2^D`, to its last entry.
fn sym_clamp_knots(kind: TableKind) -> Vec<i32> {
    let scale = Scale::new(MODEL_SCALE as u8).expect("MODEL_SCALE is a scale the semantics define");
    let spec = table(kind, scale, SemVersion::V1);
    let bound = 1i32 << ACTIVATION_BOUND;
    (-bound..=bound)
        .step_by(1 << SEGMENT_BITS)
        .map(|x| spec.lookup(x))
        .collect()
}

/// The knots of sin or cos: `round(S · f(x / S))` at every segment boundary of `[0, 2^D]`, ties
/// upward, from the semantics' Q2.62 core.
fn trig_knots(sin: bool) -> Vec<i32> {
    (0..=1u128 << (TRIG_DOMAIN_BITS - SEGMENT_BITS))
        .map(|i| {
            let theta_q64 = (i << SEGMENT_BITS) << (64 - MODEL_SCALE);
            let (sin_q62, cos_q62) = sin_cos_q62(theta_q64);
            let f_q62 = if sin { sin_q62 } else { cos_q62 } as i128;
            (((f_q62 << MODEL_SCALE) + (1 << 61)) >> 62) as i32
        })
        .collect()
}

macro_rules! pwl_spec {
    ($(#[$doc:meta])* $name:ident, signed: $signed:expr, domain_bits: $bits:expr, knots: $knots:expr) => {
        $(#[$doc])*
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $name;

        impl PwlSpec for $name {
            const SIGNED: bool = $signed;
            const DOMAIN_BITS: usize = $bits;

            fn knots() -> &'static [i32] {
                static KNOTS: OnceLock<Vec<i32>> = OnceLock::new();
                KNOTS.get_or_init(|| $knots)
            }
        }
    };
}

pwl_spec!(
    /// `tanh` over `[-8, 8)`.
    TanhPwl, signed: true, domain_bits: ACTIVATION_BOUND, knots: sym_clamp_knots(TableKind::Tanh)
);
pwl_spec!(
    /// `erf` over `[-8, 8)`.
    ErfPwl, signed: true, domain_bits: ACTIVATION_BOUND, knots: sym_clamp_knots(TableKind::Erf)
);
pwl_spec!(
    /// The logistic sigmoid over `[-8, 8)`.
    SigmoidPwl, signed: true, domain_bits: ACTIVATION_BOUND, knots: sym_clamp_knots(TableKind::Sigmoid)
);
pwl_spec!(
    /// `sin` over the period-reduced domain.
    SinPwl, signed: false, domain_bits: TRIG_DOMAIN_BITS, knots: trig_knots(true)
);
pwl_spec!(
    /// `cos` over the period-reduced domain.
    CosPwl, signed: false, domain_bits: TRIG_DOMAIN_BITS, knots: trig_knots(false)
);

/// The piecewise-linear table of `S` over `XLEN`-bit inputs. Entries are signed, stored as
/// two's-complement `u64`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PiecewiseLinearTable<const XLEN: usize, S>(PhantomData<S>);

/// Piecewise-linear `tanh`.
pub type TanhPwlTable<const XLEN: usize> = PiecewiseLinearTable<XLEN, TanhPwl>;
/// Piecewise-linear `erf`.
pub type ErfPwlTable<const XLEN: usize> = PiecewiseLinearTable<XLEN, ErfPwl>;
/// Piecewise-linear sigmoid.
pub type SigmoidPwlTable<const XLEN: usize> = PiecewiseLinearTable<XLEN, SigmoidPwl>;
/// Piecewise-linear `sin`.
pub type SinPwlTable<const XLEN: usize> = PiecewiseLinearTable<XLEN, SinPwl>;
/// Piecewise-linear `cos`.
pub type CosPwlTable<const XLEN: usize> = PiecewiseLinearTable<XLEN, CosPwl>;

impl<const XLEN: usize, S: PwlSpec> PiecewiseLinearTable<XLEN, S> {
    /// Segment-index bits below the domain bound: `D - W`.
    const SEGMENT_INDEX_BITS: usize = S::DOMAIN_BITS - SEGMENT_BITS;
    /// Index bits of significance `>= 2^D`; a signed table's sign bit is the first of them.
    const HIGH_BITS: usize = XLEN - S::DOMAIN_BITS;

    /// The table at the integer input `x`.
    pub fn value(x: i64) -> i64 {
        let bound = 1i64 << S::DOMAIN_BITS;
        let offset = if S::SIGNED {
            x.clamp(-bound, bound) + bound
        } else if (0..bound).contains(&x) {
            x
        } else {
            return 0;
        };
        let knots = S::knots();
        let segment = (offset >> SEGMENT_BITS) as usize;
        let lo = offset & ((1 << SEGMENT_BITS) - 1);
        let left = knots[segment] as i64;
        if lo == 0 {
            return left << SEGMENT_BITS;
        }
        (left << SEGMENT_BITS) + (knots[segment + 1] as i64 - left) * lo
    }

    /// The values a signed table saturates to above and below its domain; zero for an unsigned
    /// table.
    fn saturation() -> (i64, i64) {
        if S::SIGNED {
            let bound = 1i64 << S::DOMAIN_BITS;
            (Self::value(bound), Self::value(-bound))
        } else {
            (0, 0)
        }
    }

    /// Blocks per sign that meet the domain when the suffix has `suffix_len` bits, as a log2.
    fn block_bits(suffix_len: usize) -> usize {
        S::DOMAIN_BITS.saturating_sub(suffix_len)
    }

    /// The prefix factors of the decomposition at the prefix point `prefix`, the high
    /// `prefix.len()` index variables, most significant first. Pairs with
    /// [`Self::suffix_evals`] on the remaining `XLEN - prefix.len()` bits.
    pub fn prefix_evals<F: JoltField>(prefix: &[F]) -> Vec<F> {
        assert!(!prefix.is_empty() && prefix.len() <= XLEN);
        let suffix_len = XLEN - prefix.len();
        if suffix_len > SEGMENT_BITS {
            Self::block_prefixes(prefix, suffix_len)
        } else {
            Self::linear_prefixes(prefix, suffix_len).to_vec()
        }
    }

    /// The suffix factors of the decomposition on the low `suffix.len()` index bits.
    pub fn suffix_evals(suffix: LookupBits) -> Vec<i64> {
        let suffix_len = suffix.len();
        assert!(suffix_len < XLEN);
        let s = u64::from(suffix) as i64;
        if suffix_len <= SEGMENT_BITS {
            return vec![1, s];
        }
        let blocks = 1i64 << Self::block_bits(suffix_len);
        let (sat_pos, sat_neg) = Self::saturation();
        let mut evals: Vec<i64> = (0..blocks)
            .map(|t| Self::value((t << suffix_len) + s) - sat_pos)
            .collect();
        if S::SIGNED {
            evals.extend(
                (0..blocks).map(|t| Self::value(((t - blocks) << suffix_len) + s) - sat_neg),
            );
            evals.push(1);
        }
        evals
    }

    /// `Σ_i prefixes[i] · suffixes[i]`: the table's MLE at the prefix point and suffix bits the
    /// factors were evaluated on.
    pub fn combine<F: JoltField>(prefixes: &[F], suffixes: &[i64]) -> F {
        assert_eq!(prefixes.len(), suffixes.len());
        prefixes
            .iter()
            .zip(suffixes)
            .fold(F::zero(), |acc, (&p, &s)| acc + p * F::from_i64(s))
    }

    /// Prefix factors for a suffix longer than a segment: the indicator of each block that
    /// meets the domain (non-negative blocks, then negative ones), then the saturation term.
    fn block_prefixes<F: JoltField>(prefix: &[F], suffix_len: usize) -> Vec<F> {
        let (head, tail) = prefix.split_at(prefix.len() - Self::block_bits(suffix_len));
        let all_zero = head.iter().fold(F::one(), |acc, &h| acc * (F::one() - h));
        let block = eq_evals(tail);
        let mut evals: Vec<F> = block.iter().map(|&e| all_zero * e).collect();
        if S::SIGNED {
            let all_one = head.iter().fold(F::one(), |acc, &h| acc * h);
            evals.extend(block.iter().map(|&e| all_one * e));
            let (sat_pos, sat_neg) = Self::saturation();
            let sign = head[0];
            evals.push((F::one() - sign) * F::from_i64(sat_pos) + sign * F::from_i64(sat_neg));
        }
        evals
    }

    /// Prefix factors for a suffix within a segment: `[constant, slope]` of the table as a
    /// linear function of the suffix's value.
    fn linear_prefixes<F: JoltField>(prefix: &[F], suffix_len: usize) -> [F; 2] {
        let (head, rest) = prefix.split_at(Self::HIGH_BITS);
        let (segment, lo) = rest.split_at(Self::SEGMENT_INDEX_BITS);
        let knots = S::knots();
        let half = 1usize << Self::SEGMENT_INDEX_BITS;
        let knot = |i: usize| F::from_i64(knots[i] as i64);
        let all_zero = head.iter().fold(F::one(), |acc, &h| acc * (F::one() - h));

        // The knot vector the remaining bits read: the non-negative half where the high bits
        // are all zero, the negative half where they are all one, and the boundary knot of the
        // input's sign everywhere else.
        let mut v: Vec<F> = if S::SIGNED {
            let all_one = head.iter().fold(F::one(), |acc, &h| acc * h);
            let sign = head[0];
            let saturated =
                (F::one() - sign - all_zero) * knot(2 * half) + (sign - all_one) * knot(0);
            (0..=half)
                .map(|i| all_zero * knot(half + i) + all_one * knot(i) + saturated)
                .collect()
        } else {
            (0..=half).map(|i| all_zero * knot(i)).collect()
        };

        // Each segment bit, most significant first, selects the left or right half of the knot
        // vector; the halves share their middle knot.
        for &t in segment {
            let n = (v.len() - 1) / 2;
            for i in 0..=n {
                v[i] = v[i] + t * (v[i + n] - v[i]);
            }
            v.truncate(n + 1);
        }

        let slope = v[1] - v[0];
        let lo_bound = lo.iter().fold(F::zero(), |acc, &b| acc + acc + b);
        let constant =
            v[0] * F::from_u64(1 << SEGMENT_BITS) + slope * lo_bound * F::from_u64(1 << suffix_len);
        [constant, slope]
    }
}

/// `eq(point, b)` for every Boolean `b`, with `point[0]` the most significant bit of the index.
fn eq_evals<F: JoltField>(point: &[F]) -> Vec<F> {
    point.iter().fold(vec![F::one()], |evals, &p| {
        evals
            .iter()
            .flat_map(|&e| {
                let hi = e * p;
                [e - hi, hi]
            })
            .collect()
    })
}

impl<const XLEN: usize, S: PwlSpec> JoltLookupTable for PiecewiseLinearTable<XLEN, S> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let x = if S::SIGNED {
            ((index << (64 - XLEN)) as i64) >> (64 - XLEN)
        } else {
            let index = index << (64 - XLEN) >> (64 - XLEN);
            if index >> S::DOMAIN_BITS != 0 {
                return 0;
            }
            index as i64
        };
        Self::value(x) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        assert_eq!(r.len(), XLEN);
        let r: Vec<F> = r.iter().map(|&c| c.into()).collect();
        Self::linear_prefixes(&r, 0)[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{lookup_table_mle_linearity_test, signed_lookup_table_mle_random_test};
    use atlas_algebra::field::Ring;
    use atlas_algebra::{field::fp128::Fp128 as Fr, utils::index_to_field_bitvector};
    use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

    const XLEN: usize = 32;
    type Table<S> = PiecewiseLinearTable<XLEN, S>;

    /// One table entry is `2^SEGMENT_BITS` output quanta.
    const ULP: i64 = 1 << SEGMENT_BITS;

    /// Half the indices land within twice the domain bound, where the table is not constant.
    fn random_index<S: PwlSpec>(rng: &mut StdRng) -> u64 {
        if rng.gen() {
            return rng.next_u32() as u64;
        }
        let reach = 1i64 << (S::DOMAIN_BITS + 1);
        let x = if S::SIGNED {
            rng.gen_range(-reach..reach)
        } else {
            rng.gen_range(0..reach)
        };
        x as u32 as u64
    }

    /// At every round of a prefix-suffix sumcheck with `rounds_per_phase`-bit phases, the
    /// prefix and suffix factors combine to the table's MLE at the partially bound point.
    fn decomposition_matches_mle<S: PwlSpec>(rounds_per_phase: usize, samples: usize) {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let table = Table::<S>::default();
        for _ in 0..samples {
            let index = random_index::<S>(&mut rng);
            let mut r: Vec<Fr> = vec![];
            for phase in 0..XLEN / rounds_per_phase {
                let suffix_len = XLEN - (phase + 1) * rounds_per_phase;
                let suffix_bits = LookupBits::new(index, suffix_len);
                let suffix_evals = Table::<S>::suffix_evals(suffix_bits);
                for _ in 0..rounds_per_phase {
                    let unbound = XLEN - suffix_len - r.len() - 1;
                    let c = if rng.gen() { 0 } else { 2 };
                    let mut prefix_point = r.clone();
                    prefix_point.push(Fr::from_u32(c));
                    prefix_point.extend(index_to_field_bitvector::<Fr>(
                        LookupBits::new(index >> suffix_len, unbound).into(),
                        unbound,
                    ));
                    let mut point = prefix_point.clone();
                    point.extend(index_to_field_bitvector::<Fr>(
                        suffix_bits.into(),
                        suffix_len,
                    ));

                    let prefix_evals = Table::<S>::prefix_evals(&prefix_point);
                    assert_eq!(
                        Table::<S>::combine(&prefix_evals, &suffix_evals),
                        table.evaluate_mle::<Fr, Fr>(&point),
                        "index {index}, suffix_len {suffix_len}, round {}",
                        r.len(),
                    );
                    r.push(Fr::from_u64(rng.next_u64()));
                }
            }
        }
    }

    /// The MLE agrees with the entries on Boolean points across and around the domain.
    fn mle_matches_entries<S: PwlSpec>() {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let table = Table::<S>::default();
        let bound = 1u64 << S::DOMAIN_BITS;
        let edges = [0, 1, bound - 1, bound, bound + 1, u32::MAX as u64]
            .into_iter()
            .chain([bound - 1, bound, bound + 1].map(|x| (x as u32).wrapping_neg() as u64));
        for index in edges.chain((0..2000).map(|_| random_index::<S>(&mut rng))) {
            assert_eq!(
                Fr::from_i64(table.materialize_entry(index) as i64),
                table.evaluate_mle::<Fr, Fr>(&index_to_field_bitvector(index, XLEN)),
                "index {index}",
            );
        }
    }

    /// A symmetric-clamp table against the semantics table it interpolates: exact at every
    /// knot, within 1.5 output quanta elsewhere (half a quantum of rounding in each table, and
    /// the chord's distance from the curve), clamped outside the domain, and non-decreasing.
    fn sym_clamp_tracks_spec<S: PwlSpec>(kind: TableKind) {
        let spec = table(kind, Scale::new(MODEL_SCALE as u8).unwrap(), SemVersion::V1);
        let bound = 1i64 << S::DOMAIN_BITS;
        let mut previous = i64::MIN;
        for x in -bound - 600..bound + 600 {
            let entry = Table::<S>::value(x);
            let expected = spec.lookup(x as i32) as i64 * ULP;
            let error = (entry - expected).abs();
            assert!(2 * error <= 3 * ULP, "x = {x}: error {error}");
            if x % ULP == 0 {
                assert_eq!(entry, expected, "knot x = {x}");
            }
            assert!(entry >= previous, "x = {x}: not monotone");
            previous = entry;
        }
        let table = Table::<S>::default();
        assert_eq!(
            table.materialize_entry(i32::MIN as u32 as u64) as i64,
            spec.lookup(i32::MIN) as i64 * ULP
        );
        assert_eq!(
            table.materialize_entry(i32::MAX as u64) as i64,
            spec.lookup(i32::MAX) as i64 * ULP
        );
    }

    /// A trig table against the real function: within one output quantum over the whole domain
    /// (half a quantum of knot rounding, half for the chord), and zero outside it.
    fn trig_tracks_function<S: PwlSpec>(f: fn(f64) -> f64) {
        let scale = (1u64 << MODEL_SCALE) as f64;
        for x in 0..1i64 << S::DOMAIN_BITS {
            let entry = Table::<S>::value(x) as f64 / ULP as f64;
            let exact = scale * f(x as f64 / scale);
            assert!(
                (entry - exact).abs() <= 1.001,
                "x = {x}: {entry} vs {exact}"
            );
        }
        let table = Table::<S>::default();
        assert_eq!(table.materialize_entry(1 << S::DOMAIN_BITS), 0);
        assert_eq!(table.materialize_entry(u32::MAX as u64), 0);
    }

    macro_rules! pwl_tests {
        ($module:ident, $spec:ty, $samples:expr, $accuracy:expr) => {
            mod $module {
                use super::*;

                #[test]
                fn prefix_suffix() {
                    decomposition_matches_mle::<$spec>(8, $samples);
                }

                #[test]
                fn prefix_suffix_wide_phases() {
                    decomposition_matches_mle::<$spec>(16, $samples);
                }

                #[test]
                fn mle_on_hypercube() {
                    mle_matches_entries::<$spec>();
                }

                #[test]
                fn mle_random() {
                    signed_lookup_table_mle_random_test::<Fr, PiecewiseLinearTable<64, $spec>>();
                }

                #[test]
                fn mle_linearity() {
                    lookup_table_mle_linearity_test::<XLEN, Fr, Table<$spec>>();
                }

                #[test]
                fn accuracy() {
                    $accuracy;
                }
            }
        };
    }

    pwl_tests!(
        tanh,
        TanhPwl,
        100,
        sym_clamp_tracks_spec::<TanhPwl>(TableKind::Tanh)
    );
    pwl_tests!(
        erf,
        ErfPwl,
        100,
        sym_clamp_tracks_spec::<ErfPwl>(TableKind::Erf)
    );
    pwl_tests!(
        sigmoid,
        SigmoidPwl,
        100,
        sym_clamp_tracks_spec::<SigmoidPwl>(TableKind::Sigmoid)
    );
    pwl_tests!(sin, SinPwl, 20, trig_tracks_function::<SinPwl>(f64::sin));
    pwl_tests!(cos, CosPwl, 20, trig_tracks_function::<CosPwl>(f64::cos));

    /// The term counts the module docs quote for 8-bit phases.
    #[test]
    fn term_counts() {
        let terms =
            |suffix_len| Table::<TanhPwl>::suffix_evals(LookupBits::new(0, suffix_len)).len();
        assert_eq!([24, 16, 8, 0].map(terms), [3, 5, 2, 2]);
    }
}
