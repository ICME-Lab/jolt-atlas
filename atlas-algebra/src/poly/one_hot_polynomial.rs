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
use common::parallel::par_enabled;
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
                .with_min_len(par_enabled())
                .map(|instruction| match instruction {
                    Some(index) => eq_r_address[*index as usize],
                    None => F::zero(),
                })
                .collect::<Vec<F>>(),
        );
        poly.evaluate(r_cycle)
    }

    pub fn coeffs(&self) -> Vec<F> {
        let T = self.nonzero_indices.len();
        let mut coeffs = vec![F::zero(); self.K * T];
        for (t, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                coeffs[*k as usize * T + t] = F::one();
            }
        }
        coeffs
    }
}
