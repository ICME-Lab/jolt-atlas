use crate::{
    field::JoltField,
    utils::{index_to_field_bitvector, interleave_bits, lookup_bits::LookupBits},
};

use common::consts::XLEN;
use num::Integer;
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use strum::IntoEnumIterator;

use super::{
    prefixes::{PrefixCheckpoints, Prefixes},
    suffixes::SuffixEval,
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};

/// Tests that a lookup table's MLE evaluates correctly at random points, converting
/// each materialized entry to `F` via `convert` before comparing against the MLE evaluation.
fn lookup_table_mle_random_test_inner<F: JoltField, T: JoltLookupTable + Default>(
    convert: impl Fn(u64) -> F,
) {
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let index = rng.gen();
        assert_eq!(
            convert(T::default().materialize_entry(index)),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(index, XLEN * 2)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

/// Tests that a lookup table's MLE evaluates correctly at random points.
///
/// Verifies that the MLE evaluation at random points matches the linear
/// combination of table entries with Lagrange basis polynomials.
pub fn lookup_table_mle_random_test<F: JoltField, T: JoltLookupTable + Default>() {
    lookup_table_mle_random_test_inner::<F, T>(F::from_u64);
}

/// Tests that a signed lookup table's MLE evaluates correctly at random points.
///
/// Verifies that the MLE evaluation at random points matches the linear
/// combination of table entries with Lagrange basis polynomials.
pub fn signed_lookup_table_mle_random_test<F: JoltField, T: JoltLookupTable + Default>() {
    lookup_table_mle_random_test_inner::<F, T>(|entry| F::from_i64(entry as i64));
}

/// Tests that a lookup table's MLE evaluates correctly over the full boolean hypercube,
/// converting each materialized entry to `F` via `convert` before comparing.
fn lookup_table_mle_full_hypercube_test_inner<F: JoltField, T: JoltLookupTable + Default>(
    convert: impl Fn(u64) -> F,
) {
    let materialized = T::default().materialize();
    for (i, entry) in materialized.iter().enumerate() {
        assert_eq!(
            convert(*entry),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(i as u64, 16)),
            "MLE did not match materialized table at index {i}",
        );
    }
}

/// Tests that a lookup table's MLE evaluates correctly over the full boolean hypercube.
///
/// Verifies MLE correctness by testing on all 2^n corners of the hypercube.
pub fn lookup_table_mle_full_hypercube_test<F: JoltField, T: JoltLookupTable + Default>() {
    lookup_table_mle_full_hypercube_test_inner::<F, T>(F::from_u64);
}

/// Tests that a signed lookup table's MLE evaluates correctly over the full boolean hypercube.
///
/// Verifies MLE correctness by testing on all 2^n corners of the hypercube.
pub fn signed_lookup_table_mle_full_hypercube_test<F: JoltField, T: JoltLookupTable + Default>() {
    lookup_table_mle_full_hypercube_test_inner::<F, T>(|entry| F::from_i16(entry as i16));
}

/// Tests that `evaluate_mle` is affine in each individual variable.
///
/// For each variable `x_i`, we freeze all other variables at random field points and
/// evaluate at `x_i = 0, 1, 2`. Affineness is equivalent to:
/// `f(2) - f(0) = 2 * (f(1) - f(0))`.
pub fn lookup_table_mle_linearity_test<
    const LOG_K: usize,
    F: JoltField,
    T: JoltLookupTable + Default,
>() {
    const NUM_RANDOM_ASSIGNMENTS: usize = 16;
    let mut rng = StdRng::seed_from_u64(12345);
    let table = T::default();
    let num_vars = LOG_K;
    let two = F::from_u64(2);

    for var_idx in 0..num_vars {
        for _ in 0..NUM_RANDOM_ASSIGNMENTS {
            let mut eval_point: Vec<F> =
                (0..num_vars).map(|_| F::from_u64(rng.next_u64())).collect();

            eval_point[var_idx] = F::zero();
            let y0 = table.evaluate_mle::<F, F>(&eval_point);

            eval_point[var_idx] = F::one();
            let y1 = table.evaluate_mle::<F, F>(&eval_point);

            eval_point[var_idx] = two;
            let y2 = table.evaluate_mle::<F, F>(&eval_point);

            let lhs = y2 - y0;
            let rhs = (y1 - y0) * two;
            assert_eq!(lhs, rhs, "evaluate_mle is not affine in variable {var_idx}");
        }
    }
}

/// Generates a lookup index where right operand is 111..000
pub fn gen_bitmask_lookup_index(rng: &mut StdRng) -> u64 {
    let x = rng.next_u32();
    let zeros = rng.gen_range(0..=XLEN);
    let y = (!0u32).wrapping_shl(zeros as u32);
    interleave_bits(x, y)
}

fn prefix_suffix_test_inner<
    const XLEN: usize,
    F: JoltField,
    T: PrefixSuffixDecompositionTrait<XLEN>,
>(
    rounds_per_phase: usize,
    total_addr_bits: usize,
) {
    let total_phases: usize = total_addr_bits / rounds_per_phase;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..300 {
        let mut prefix_checkpoints = PrefixCheckpoints::new();
        let lookup_index = T::random_lookup_index(&mut rng);
        let mut j = 0;
        let mut r: Vec<F> = vec![];
        let mut u_eval = F::one();
        for phase in 0..total_phases {
            let suffix_len = (total_phases - 1 - phase) * rounds_per_phase;
            let (mut prefix_bits, suffix_bits) =
                LookupBits::new(lookup_index, total_addr_bits - phase * rounds_per_phase)
                    .split(suffix_len);

            let suffix_evals: Vec<_> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u32(suffix.suffix_mle::<XLEN>(suffix_bits))))
                .collect();

            for _ in 0..rounds_per_phase {
                let mut eval_point = r.clone();
                let c = if rng.next_u64().is_even() { 0 } else { 2 };
                eval_point.push(F::from_u32(c));
                prefix_bits.pop_msb();

                eval_point
                    .extend(index_to_field_bitvector(prefix_bits.into(), prefix_bits.len()).iter());
                eval_point
                    .extend(index_to_field_bitvector(suffix_bits.into(), suffix_bits.len()).iter());

                let mle_eval = T::default().evaluate_mle(&eval_point);

                let r_x = if j % 2 == 1 {
                    Some(*r.last().unwrap())
                } else {
                    None
                };

                let prefix_evals: Vec<_> = T::default()
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F>(&prefix_checkpoints, r_x, c, prefix_bits, j)
                    })
                    .collect();

                let combined = T::default().combine(&prefix_evals, &suffix_evals);
                if combined != mle_eval {
                    println!("Lookup index: {lookup_index}");
                    println!("j: {j} {prefix_bits} {suffix_bits}");
                    for (i, x) in prefix_evals.iter().enumerate() {
                        println!("prefix_evals[{:?}] = {x}", Prefixes::iter().nth(i).unwrap());
                    }
                    for (i, x) in suffix_evals.iter().enumerate() {
                        println!("suffix_evals[{:?}] = {x}", T::default().suffixes()[i]);
                    }
                }

                assert_eq!(combined, mle_eval, "Prefix-suffix decomposition does not match MLE. (Multivariate Polynomials mismatch.)");

                // ps-shout suffixes includes the "eq", "ra" polynomial evaluations,
                // This is simulated here.
                let weighted_suffix_evals: Vec<_> = suffix_evals
                    .iter()
                    .map(|suffix_eval| *suffix_eval * u_eval)
                    .collect();
                let weighted_combined = T::default().combine(&prefix_evals, &weighted_suffix_evals);
                let weighted_expected = u_eval * mle_eval;
                assert_eq!(
                    weighted_combined, weighted_expected,
                    "Weighted suffix check failed at phase {phase}, round {j}.\n\
                    This commonly indicates an invalid suffix*suffix product in combine logic."
                );

                r.push(F::from_u64(rng.next_u64()));

                if r.len() % 2 == 0 {
                    Prefixes::update_checkpoints::<XLEN, F, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                        suffix_len,
                    );
                }

                j += 1;
            }

            // Simulating updating the "u_eval" at each phase, this is highly simplified from ps-shout.
            let phase_challenge = F::from_u64(rng.next_u64());
            u_eval *= phase_challenge;
        }
    }
}

/// Tests the prefix-suffix decomposition of a binary (2*XLEN address space) lookup table.
///
/// Verifies that combining prefix and suffix evaluations correctly reconstructs
/// the full lookup table value at random indices.
pub fn prefix_suffix_test<
    const XLEN: usize,
    F: JoltField,
    T: PrefixSuffixDecompositionTrait<XLEN>,
>() {
    prefix_suffix_test_inner::<XLEN, F, T>(16, XLEN * 2);
}

/// Tests the prefix-suffix decomposition of a unary (XLEN address space) lookup table.
///
/// Verifies that combining prefix and suffix evaluations correctly reconstructs
/// the full lookup table value at random indices.
pub fn prefix_suffix_test_unary<
    const XLEN: usize,
    F: JoltField,
    T: PrefixSuffixDecompositionTrait<XLEN>,
>() {
    prefix_suffix_test_inner::<XLEN, F, T>(8, XLEN);
}
