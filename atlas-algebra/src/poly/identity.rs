use std::sync::{Arc, RwLock};

use allocative::Allocative;
use num::Integer;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        multilinear::MultilinearPolynomial,
        prefix_suffix::{
            CachedPolynomial, Prefix, PrefixCheckpoints, PrefixPolynomial, PrefixRegistry,
            PrefixSuffixPolynomial, SuffixPolynomial,
        },
    },
    utils::{lookup_bits::LookupBits, math::Math},
};

use super::multilinear::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

#[derive(Clone, Debug, Allocative)]
pub struct IdentityPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> IdentityPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        IdentityPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for IdentityPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);

        match order {
            BindingOrder::LowToHigh => {
                self.bound_value += F::from_u128(1 << self.num_bound_vars) * r;
            }
            BindingOrder::HighToLow => {
                self.bound_value += self.bound_value;
                self.bound_value = self.bound_value + r;
            }
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for IdentityPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);
        (0..len)
            .map(|i| r[i].into().mul_u128(1 << (len - 1 - i)))
            .sum()
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let mut evals = vec![F::zero(); degree];
        let m = match order {
            BindingOrder::LowToHigh => {
                let m = F::from_u128(1 << self.num_bound_vars);
                evals[0] = self.bound_value + (m + m).mul_u64(index as u64);
                m
            }
            BindingOrder::HighToLow => {
                let m = F::from_u128(1 << (self.num_vars - 1 - self.num_bound_vars));
                evals[0] = self.bound_value * (m + m) + F::from_u64(index as u64);
                m
            }
        };

        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for IdentityPolynomial<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [Box::new(ShiftSuffixPolynomial), Box::new(self.clone())]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        if registry[Prefix::Identity].is_none() {
            registry[Prefix::Identity] = Some(Arc::new(RwLock::new(self.prefix_polynomial(
                &registry.checkpoints,
                chunk_len,
                phase,
            ))));
        }
        [registry[Prefix::Identity].clone(), None]
    }
}

impl<F: JoltField> PrefixPolynomial<F> for IdentityPolynomial<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        _phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::Identity].unwrap_or(F::zero());

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| bound_value.mul_u128(1 << chunk_len) + F::from_u64(i as u64))
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

impl<F: JoltField> SuffixPolynomial<F> for IdentityPolynomial<F> {
    fn suffix_mle(&self, b: LookupBits) -> u64 {
        debug_assert!(b.len().is_even());
        u64::from(b)
    }
}

pub struct ShiftSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u64 {
        debug_assert!(b.len().is_even());
        1u64 << b.len()
    }
}

pub struct ShiftHalfSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftHalfSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u64 {
        debug_assert!(b.len().is_even());
        1u64 << (b.len() / 2)
    }
}

#[cfg(test)]
mod tests {
    use crate::field::Field as _;

    use crate::poly::{
        multilinear::MultilinearPolynomial, prefix_suffix::tests::prefix_suffix_decomposition_test,
    };

    use super::*;
    use crate::field::fp128::Fp128 as Fr;

    use crate::utils::test_rng;

    #[test]
    fn identity_poly() {
        const NUM_VARS: usize = 10;
        const DEGREE: usize = 3;

        let mut rng = test_rng();
        let mut identity_poly: IdentityPolynomial<Fr> = IdentityPolynomial::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> =
            MultilinearPolynomial::from((0..(1 << NUM_VARS)).map(|i| i as u32).collect::<Vec<_>>());

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals =
                identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            let reference_poly_evals =
                reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            assert_eq!(identity_poly_evals, reference_poly_evals);
        }

        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                assert_eq!(identity_poly_evals, reference_poly_evals);
            }
        }

        assert_eq!(identity_poly.final_claim(), reference_poly.final_claim());
    }

    #[test]
    fn identity_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<8, 2, 2, false, _>(
            IdentityPolynomial::new(8),
            vec![Prefix::Identity],
        );
    }
}
