#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        ra_poly::RaPolynomial,
    },
    utils::math::Math,
};
use allocative::Allocative;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the
/// commitment.
#[derive(Clone, Debug, Allocative)]
pub struct OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If empty, this polynomial is 0 for all j.
    pub nonzero_indices: Arc<Vec<Option<u16>>>,
    /// The number of variables that have been bound over the
    /// course of sumcheck so far.
    pub num_variables_bound: usize,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    pub G: Vec<F>,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    pub H: Arc<RwLock<RaPolynomial<u16, F>>>,
}

impl<F: JoltField> PartialEq for OneHotPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.K == other.K
            && self.nonzero_indices == other.nonzero_indices
            && self.num_variables_bound == other.num_variables_bound
            && self.G == other.G
            && *self.H.read().unwrap() == *other.H.read().unwrap()
    }
}

impl<F: JoltField> Default for OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            nonzero_indices: Arc::new(vec![]),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(RaPolynomial::None)),
        }
    }
}

impl<F: JoltField> OneHotPolynomial<F> {
    pub fn from_indices(nonzero_indices: Vec<Option<u16>>, K: usize) -> Self {
        assert!(K <= 1usize << u16::BITS, "K must be <= 65536 for indices");
        Self {
            K,
            nonzero_indices: Arc::new(nonzero_indices),
            ..Default::default()
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.K.log_2() + self.nonzero_indices.len().log_2()
    }

    /// # Note: r = (r_address, r_cycle) where r_address corresponds to the first log(K) variables and r_cycle corresponds to the last log(T) variables.
    pub fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F> + FieldChallengeOps<C>,
    {
        assert_eq!(
            r.len(),
            self.get_num_vars(),
            "Input length must match number of variables"
        );
        let (r_address, r_cycle) = r.split_at(self.K.log_2());
        let eq_r_address = EqPolynomial::evals(r_address);
        let poly = MultilinearPolynomial::from(
            self.nonzero_indices
                .par_iter()
                .map(|instruction| match instruction {
                    Some(index) => eq_r_address[*index as usize],
                    None => F::zero(),
                })
                .collect::<Vec<F>>(),
        );
        poly.evaluate(r_cycle)
    }

    #[cfg(test)]
    fn to_dense_poly(&self) -> DensePolynomial<F> {
        let T = self.nonzero_indices.len();
        let mut dense_coeffs: Vec<F> = vec![F::zero(); self.K * T];
        for (t, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                dense_coeffs[*k as usize * T + t] = F::one();
            }
        }
        DensePolynomial::new(dense_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{multilinear_polynomial::BindingOrder, unipoly::UniPoly},
        subprotocols::opening_reduction::{
            EqAddressState, EqCycleState, OneHotPolynomialProverOpening,
        },
    };
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;

    fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = (0..T)
            .map(|_| Some((rng.next_u64() % K as u64) as u16))
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let mut dense_poly = one_hot_poly.to_dense_poly();

        let r_address = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_K)
            .collect::<Vec<_>>();
        let r_cycle = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_T)
            .collect::<Vec<_>>();

        let eq_address_state = EqAddressState::new(&r_address);
        let eq_cycle_state = EqCycleState::new(&r_cycle);

        let mut one_hot_opening = OneHotPolynomialProverOpening::new(
            Arc::new(RwLock::new(eq_address_state)),
            Arc::new(RwLock::new(eq_cycle_state)),
        );
        one_hot_opening.initialize(one_hot_poly.clone());

        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let mut eq = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_concat));

        // Compute the initial input claim
        let input_claim: Fr = (0..dense_poly.len()).map(|i| dense_poly[i] * eq[i]).sum();
        let mut previous_claim = input_claim;

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_opening.compute_message(round, previous_claim);
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
                    one_hot_message.eval_at_zero(),
                    one_hot_message.evaluate::<Fr>(&Fr::from(2))
                ],
                *expected_message,
                "round {round} prover message mismatch"
            );

            let r = <Fr as JoltField>::Challenge::random(&mut rng);

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
            one_hot_opening.final_sumcheck_claim(),
            dense_poly[0],
            "final sumcheck claim"
        );
    }

    #[test]
    fn sumcheck_K_less_than_T() {
        dense_polynomial_equivalence::<5, 6>();
    }

    #[test]
    fn sumcheck_K_equals_T() {
        dense_polynomial_equivalence::<6, 6>();
    }

    #[test]
    fn sumcheck_K_greater_than_T() {
        dense_polynomial_equivalence::<6, 5>();
    }

    fn evaluate_test<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = (0..T)
            .map(|_| Some(rng.next_u64() as u16 % K as u16))
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let dense_poly = one_hot_poly.to_dense_poly();

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
}
