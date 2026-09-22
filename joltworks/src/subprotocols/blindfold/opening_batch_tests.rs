use super::*;
use crate::poly::opening_proof::SumcheckId;
use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use common::CommittedPoly;
use std::collections::HashMap;

fn opening(index: usize) -> OpeningId {
    let sumcheck = match index % 3 {
        0 => SumcheckId::NodeExecution(0),
        1 => SumcheckId::NodeExecution(1),
        _ => SumcheckId::RLC(0),
    };
    OpeningId::new(CommittedPoly::DivNodeQuotient(index / 3), sumcheck)
}

// Original vector-search implementation, retained as an independent reference.
fn reference(constraints: &[OutputClaimConstraint]) -> OutputClaimConstraint {
    let mut terms = Vec::new();
    let mut openings = Vec::new();
    let mut offset = constraints.len();
    for (index, constraint) in constraints.iter().enumerate() {
        let shift = |value: &ValueSource| match value {
            ValueSource::Challenge(i) => ValueSource::Challenge(i + offset),
            other => other.clone(),
        };
        for term in &constraint.terms {
            let mut factors = vec![ValueSource::Challenge(index)];
            factors.extend(term.factors.iter().map(shift));
            terms.push(ProductTerm::new(shift(&term.coeff), factors));
        }
        for id in &constraint.required_openings {
            if !openings.contains(id) {
                openings.push(*id);
            }
        }
        offset += constraint.num_challenges;
    }
    OutputClaimConstraint::new(terms, openings)
}

fn assert_same(actual: &OutputClaimConstraint, expected: &OutputClaimConstraint) {
    assert_eq!(actual.required_openings, expected.required_openings);
    assert_eq!(actual.num_challenges, expected.num_challenges);
    assert_eq!(actual.terms.len(), expected.terms.len());
    for (a, b) in actual.terms.iter().zip(&expected.terms) {
        assert_eq!(a.coeff, b.coeff);
        assert_eq!(a.factors, b.factors);
    }
    assert_eq!(
        actual.estimate_aux_var_count(),
        expected.estimate_aux_var_count()
    );
}

#[test]
fn opening_batch_preserves_first_occurrence_and_missing_constraint_behavior() {
    let mut first = OutputClaimConstraint::direct(opening(2));
    first.required_openings = vec![opening(2), opening(0), opening(2), opening(1)];
    let mut second = OutputClaimConstraint::direct(opening(3));
    second.required_openings = vec![opening(1), opening(3), opening(0), opening(4)];
    let constraints = vec![first.clone(), OutputClaimConstraint::default(), second];
    let expected = reference(&constraints);
    assert_eq!(
        expected.required_openings,
        [opening(2), opening(0), opening(1), opening(3), opening(4)]
    );
    let optional: Vec<_> = constraints.iter().cloned().map(Some).collect();
    for _ in 0..8 {
        assert_same(&OutputClaimConstraint::batch(&optional).unwrap(), &expected);
        assert_same(
            &InputClaimConstraint::batch_required(&constraints, constraints.len()),
            &expected,
        );
    }
    assert_same(&OutputClaimConstraint::batch(&[]).unwrap(), &reference(&[]));
    assert_same(
        &InputClaimConstraint::batch_required(&[], 0),
        &reference(&[]),
    );
    assert!(OutputClaimConstraint::batch(&[Some(first.clone()), None]).is_none());
    assert!(OutputClaimConstraint::batch(&[None, Some(first)]).is_none());
    assert!(
        std::panic::catch_unwind(|| InputClaimConstraint::batch_required(
            &constraints,
            constraints.len() + 1
        ))
        .is_err()
    );
}

#[test]
fn opening_batch_matches_independent_weighted_evaluation_and_challenge_offsets() {
    let mut rng = StdRng::seed_from_u64(771);
    let values: HashMap<_, _> = (0..96)
        .map(|i| (opening(i), Fr::from(rng.gen::<u64>())))
        .collect();
    for count in 0..24 {
        let mut constraints = Vec::new();
        let mut local_challenges = Vec::new();
        for i in 0..count {
            let ids: Vec<_> = (0..rng.gen_range(0..32))
                .map(|_| opening(rng.gen_range(0..96)))
                .collect();
            let mut terms = vec![ProductTerm::single(ValueSource::Constant(-7))];
            for (j, id) in ids.iter().enumerate() {
                terms.push(ProductTerm::new(
                    ValueSource::Challenge(j % 7),
                    vec![
                        ValueSource::Opening(*id),
                        ValueSource::Challenge((i + j) % 11),
                        ValueSource::Constant((j % 5) as i128 - 2),
                    ],
                ));
                if j % 3 == 0 {
                    terms.push(ProductTerm::new(
                        ValueSource::Opening(*id),
                        vec![ValueSource::Opening(*id)],
                    ));
                }
            }
            let constraint = OutputClaimConstraint::new(terms, ids);
            local_challenges.push(
                (0..constraint.num_challenges)
                    .map(|_| Fr::from(rng.gen::<u64>()))
                    .collect::<Vec<_>>(),
            );
            constraints.push(constraint);
        }
        let combined = InputClaimConstraint::batch_required(&constraints, count);
        assert_same(&combined, &reference(&constraints));
        let alphas: Vec<_> = (0..count).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let mut challenges = alphas.clone();
        challenges.extend(local_challenges.iter().flatten().copied());
        let get_values = |ids: &[OpeningId]| ids.iter().map(|id| values[id]).collect::<Vec<_>>();
        let expected: Fr = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| {
                alphas[i] * c.evaluate(&get_values(&c.required_openings), &local_challenges[i])
            })
            .sum();
        assert_eq!(
            combined.evaluate(&get_values(&combined.required_openings), &challenges),
            expected
        );
    }
}

fn large_fixture(count: usize) -> Vec<OutputClaimConstraint> {
    (0..count)
        .map(|i| {
            let id = opening((i * 8191) % count);
            let mut constraint = OutputClaimConstraint::direct(id);
            // Both duplicates and declared but unused openings must preserve order.
            constraint.required_openings.extend([opening(0), id]);
            constraint
        })
        .collect()
}

#[test]
fn opening_batch_large_unique_and_duplicate_lists_match_reference() {
    for count in [1, 127, 128, 129, 4096, 16384] {
        let constraints = large_fixture(count);
        let expected = reference(&constraints);
        let actual = InputClaimConstraint::batch_required(&constraints, count);
        assert_same(&actual, &expected);
        assert_eq!(actual.required_openings.len(), count);
    }
}

#[test]
fn opening_batch_preserves_complete_r1cs_matrices_and_opening_layout() {
    use crate::subprotocols::blindfold::{BakedPublicInputs, VerifierR1CSBuilder};
    use std::collections::BTreeMap;
    let constraints = large_fixture(129);
    let expected = reference(&constraints);
    let actual = InputClaimConstraint::batch_required(&constraints, constraints.len());
    let build = |constraint: &OutputClaimConstraint| {
        let baked = BakedPublicInputs::<Fr> {
            extra_constraint_challenges: (0..constraint.num_challenges)
                .map(|i| Fr::from((i + 7) as u64))
                .collect(),
            ..Default::default()
        };
        VerifierR1CSBuilder::new_with_extra(
            &[],
            std::slice::from_ref(constraint),
            &baked,
            vec![constraint.required_openings.clone()],
            BTreeMap::new(),
        )
        .with_row_width(4)
        .build()
    };
    let left = build(&actual);
    let right = build(&expected);
    assert_eq!(left.num_vars, right.num_vars);
    assert_eq!(left.num_constraints, right.num_constraints);
    assert_eq!(
        left.output_claims_opening_ids,
        right.output_claims_opening_ids
    );
    assert_eq!(left.opening_aliases, right.opening_aliases);
    for (a, b) in [
        (&left.a, &right.a),
        (&left.b, &right.b),
        (&left.c, &right.c),
    ] {
        assert_eq!(a.num_rows, b.num_rows);
        assert_eq!(a.num_cols, b.num_cols);
        assert_eq!(a.entries, b.entries);
    }
}

#[test]
#[ignore = "isolated opening deduplication benchmark"]
fn benchmark_opening_batch() {
    let constraints = large_fixture(65536);
    let mut expected = None;
    for mode in ["linear", "indexed", "indexed", "linear"] {
        let start = std::time::Instant::now();
        let result = if mode == "linear" {
            reference(&constraints)
        } else {
            InputClaimConstraint::batch_required(&constraints, constraints.len())
        };
        let elapsed = start.elapsed().as_secs_f64();
        if let Some(old) = &expected {
            assert_same(&result, old);
        } else {
            expected = Some(result);
        }
        println!(
            "OPENING_BATCH_BENCH mode={mode} constraints={} seconds={elapsed:.9}",
            constraints.len()
        );
    }
}
