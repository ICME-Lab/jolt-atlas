//! `OneHotPolynomial` against its dense equivalent, through the opening-reduction prover.

use crate::opening_reduction::{EqAddressState, EqCycleState, OneHotPolynomialProverOpening};
use ark_std::{test_rng, Zero};
use atlas_algebra::field::fp128::Fp128 as Fr;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::one_hot_polynomial::*;
use atlas_algebra::poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial};
use atlas_algebra::poly::{multilinear_polynomial::BindingOrder, unipoly::UniPoly};
use jolt_field::Field as _;
use jolt_field::Ring as _;
use rand_core::RngCore;
use std::sync::{Arc, RwLock};

fn to_dense_poly<F: JoltField>(p: &OneHotPolynomial<F>) -> DensePolynomial<F> {
    DensePolynomial::new(p.coeffs())
}

/// With `vanishing_address`, the first address coordinate is 1 and its challenge 0, so
/// `eq(r_address, ρ)` vanishes and every cycle-round message is zero.
fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>(vanishing_address: bool) {
    let K: usize = 1 << LOG_K;
    let T: usize = 1 << LOG_T;

    let mut rng = test_rng();

    let nonzero_indices: Vec<_> = (0..T)
        .map(|_| Some((rng.next_u64() % K as u64) as u16))
        .collect();
    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
    let mut dense_poly = to_dense_poly(&one_hot_poly);

    let mut r_address: Vec<Fr> =
        std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_K)
            .collect();
    if vanishing_address {
        r_address[0] = Fr::from_u64(1u64);
    }
    let r_cycle = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
        .take(LOG_T)
        .collect::<Vec<_>>();

    let eq_address_state = EqAddressState::new(&r_address);
    let eq_cycle_state = EqCycleState::new(&r_cycle);

    let mut one_hot_opening = OneHotPolynomialProverOpening::new(
        Arc::new(RwLock::new(eq_address_state)),
        Arc::new(RwLock::new(eq_cycle_state)),
    );
    one_hot_opening.initialize(vec![(one_hot_poly.clone(), Fr::from_u64(1u64))]);

    let r_cycle_field: Vec<Fr> = r_cycle.to_vec();
    let r_concat = [r_address.as_slice(), r_cycle_field.as_slice()].concat();
    let mut eq = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_concat));

    // Compute the initial input claim
    let input_claim: Fr = (0..dense_poly.len()).map(|i| dense_poly[i] * eq[i]).sum();
    let mut previous_claim = input_claim;

    for round in 0..LOG_K + LOG_T {
        let one_hot_message = one_hot_opening
            .compute_message(round, previous_claim)
            .expect("honest claim");
        // Evals at [0, 2].
        let mut expected_message = vec![Fr::zero(), Fr::zero()];
        let mle_half = dense_poly.len() / 2;

        // We bind first log_K vars HighToLow and then log_T vars LowToHigh
        // because the dense polynomial has address variables reversed
        if round < LOG_K {
            expected_message[0] = (0..mle_half).map(|i| dense_poly[i] * eq[i]).sum();
            expected_message[1] = (0..mle_half)
                .map(|i| {
                    let poly_bound_point =
                        dense_poly[i + mle_half] + dense_poly[i + mle_half] - dense_poly[i];
                    let eq_bound_point = eq[i + mle_half] + eq[i + mle_half] - eq[i];
                    poly_bound_point * eq_bound_point
                })
                .sum();
        } else {
            expected_message[0] = (0..mle_half).map(|i| dense_poly[i] * eq[i]).sum();
            expected_message[1] = (0..mle_half)
                .map(|i| {
                    let poly_bound_point =
                        dense_poly[i + mle_half] + dense_poly[i + mle_half] - dense_poly[i];
                    let eq_bound_point = eq[i + mle_half] + eq[i + mle_half] - eq[i];
                    poly_bound_point * eq_bound_point
                })
                .sum();
        }
        assert_eq!(
            [
                one_hot_message.evaluate(Fr::zero()),
                one_hot_message.evaluate(Fr::from_u64(2u64))
            ],
            *expected_message,
            "round {round} prover message mismatch"
        );

        let r = if vanishing_address && round == 0 {
            <Fr as JoltField>::Challenge::default()
        } else {
            <Fr as JoltField>::Challenge::random(&mut rng)
        };

        // Update previous_claim by evaluating the univariate polynomial at r
        let eval_at_1 = previous_claim - expected_message[0];
        let univariate_evals = vec![expected_message[0], eval_at_1, expected_message[1]];
        let univariate_poly = UniPoly::from_evals(&univariate_evals);
        previous_claim = univariate_poly.evaluate(&r);

        one_hot_opening.bind(r, round);
        dense_poly.bind_parallel(r, BindingOrder::HighToLow);
        eq.bind_parallel(r, BindingOrder::HighToLow);
    }
    assert_eq!(
        one_hot_opening.final_claim(),
        dense_poly[0],
        "final sumcheck claim"
    );
}

#[test]
fn sumcheck_K_less_than_T() {
    dense_polynomial_equivalence::<5, 6>(false);
}

#[test]
fn sumcheck_K_equals_T() {
    dense_polynomial_equivalence::<6, 6>(false);
}

#[test]
fn sumcheck_K_greater_than_T() {
    dense_polynomial_equivalence::<6, 5>(false);
}

#[test]
fn sumcheck_vanishing_address_eq() {
    dense_polynomial_equivalence::<5, 6>(true);
}

fn evaluate_test<const LOG_K: usize, const LOG_T: usize>() {
    let K: usize = 1 << LOG_K;
    let T: usize = 1 << LOG_T;

    let mut rng = test_rng();

    let nonzero_indices: Vec<_> = (0..T)
        .map(|_| Some(rng.next_u64() as u16 % K as u16))
        .collect();
    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
    let dense_poly = to_dense_poly(&one_hot_poly);

    let r: Vec<<Fr as JoltField>::Challenge> =
        std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_K + LOG_T)
            .collect();
    let r_one_hot = r.clone();

    assert_eq!(one_hot_poly.evaluate(&r_one_hot), dense_poly.evaluate(&r));
}

#[test]
fn evaluate_K_less_than_T() {
    evaluate_test::<5, 6>();
}

#[test]
fn evaluate_K_equals_T() {
    evaluate_test::<6, 6>();
}

#[test]
fn evaluate_K_greater_than_T() {
    evaluate_test::<6, 5>();
}
