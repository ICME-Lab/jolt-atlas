//! Sparse Booleanity prover, using the original Atlas address and row phases.

use super::*;
use allocative::Allocative;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq::EqPolynomial,
        multilinear::{BindingOrder, PolynomialBinding},
        one_hot::OneHotPolynomial,
        ra::RaPolynomial,
        split_eq::{GruenSplitEqPolynomial, SplitEqUnreducedFold},
        univariate::UnivariatePoly,
    },
    utils::expanding_table::ExpandingTable,
};
use atlas_claims::SymbolicSumcheck;
use atlas_sumcheck::{
    engine::{prover::round_check_failed, prover::ProveRounds, SumcheckError},
    relation::{prover::RelationWitness, RelationChallenge, RelationSetup},
};
use common::parallel::par_enabled;
use num_traits::Zero;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Allocative)]
pub struct BooleanityProver<F: JoltField> {
    #[allocative(skip)]
    check: CoefficientCheck,
    #[allocative(skip)]
    setup: RelationSetup<F>,
    #[allocative(skip)]
    address_eq: GruenSplitEqPolynomial<F>,
    #[allocative(skip)]
    row_eq: GruenSplitEqPolynomial<F>,
    address_vars: usize,
    weights: Vec<F>,
    address_sums: Vec<Vec<F>>,
    indices: Vec<Arc<Vec<Option<u16>>>>,
    bound_addresses: ExpandingTable<F>,
    rows: Vec<RaPolynomial<u16, F>>,
}

impl<F: JoltField> BooleanityProver<F> {
    /// Build the check over the original one-hot tables, in the given polynomial order.
    pub fn new(
        check: CoefficientCheck,
        setup: RelationSetup<F>,
        witnesses: &[&OneHotPolynomial<F>],
    ) -> Result<Self, ProofVerifyError> {
        let Some(address_vars) = check.one_hot_address_vars() else {
            return Err(invalid("Booleanity requires a one-hot check"));
        };
        let columns = 1 << address_vars;
        let rows = 1 << (check.rounds() - address_vars);
        if setup.incoming_point.len() != check.rounds()
            || witnesses.len() != check.polynomials().len()
            || witnesses.iter().any(|poly| {
                poly.K != columns
                    || poly.nonzero_indices.len() != rows
                    || poly.num_variables_bound != 0
                    || poly
                        .nonzero_indices
                        .iter()
                        .flatten()
                        .any(|&index| usize::from(index) >= columns)
            })
        {
            return Err(invalid("wrong one-hot witness shape for Booleanity"));
        }

        let (address_point, row_point) = setup.incoming_point.split_at(address_vars);
        let address_eq = GruenSplitEqPolynomial::new(address_point, BindingOrder::LowToHigh);
        let row_eq = GruenSplitEqPolynomial::new(row_point, BindingOrder::LowToHigh);
        let row_weights = EqPolynomial::evals(row_point);
        let address_sums = witnesses
            .par_iter()
            .map(|poly| {
                let mut sums = vec![F::zero(); columns];
                for (&index, &weight) in poly.nonzero_indices.iter().zip(&row_weights) {
                    if let Some(index) = index {
                        sums[usize::from(index)] += weight;
                    }
                }
                sums
            })
            .collect();
        let weights = (0..witnesses.len())
            .map(|i| setup.challenge(&RelationChallenge::Batching(i)))
            .collect();
        let mut bound_addresses = ExpandingTable::new(columns, BindingOrder::LowToHigh);
        bound_addresses.reset(F::one());
        let mut prover = Self {
            check,
            setup,
            address_vars,
            address_eq,
            row_eq,
            weights,
            address_sums,
            indices: witnesses
                .iter()
                .map(|poly| poly.nonzero_indices.clone())
                .collect(),
            bound_addresses,
            rows: Vec::new(),
        };
        if address_vars == 0 {
            prover.initialize_rows();
        }
        Ok(prover)
    }

    fn initialize_rows(&mut self) {
        let addresses = self.bound_addresses.clone_values();
        self.rows = std::mem::take(&mut self.indices)
            .into_iter()
            .map(|indices| RaPolynomial::new(indices, addresses.clone()))
            .collect();
        self.address_sums.clear();
    }

    fn bind(&mut self, challenge: F, round: usize) {
        let r = F::challenge_from_field(challenge);
        if round < self.address_vars {
            self.address_eq.bind(challenge);
            self.bound_addresses.update(r);
            if round + 1 == self.address_vars {
                self.initialize_rows();
            }
        } else {
            self.row_eq.bind(challenge);
            self.rows
                .par_iter_mut()
                .with_min_len(par_enabled())
                .for_each(|poly| {
                    poly.bind_parallel(r, BindingOrder::LowToHigh);
                });
        }
    }

    fn address_round(&self, round: usize, claim: F) -> Result<UnivariatePoly<F>, F> {
        let half = 1 << round;
        let block_size = 2 * half;
        let [constant, leading] = self.address_eq.par_fold_out_in_unreduced::<2>(&|block| {
            self.address_sums.iter().zip(&self.weights).fold(
                [F::zero(); 2],
                |[mut constant, mut leading], (sums, &gamma)| {
                    let sums = &sums[block * block_size..][..block_size];
                    for (index, &weight) in sums.iter().enumerate() {
                        let bound = self.bound_addresses[index % half];
                        let term = gamma * bound * weight;
                        leading += term * bound;
                        if index < half {
                            constant += term * (bound - F::one());
                        }
                    }
                    [constant, leading]
                },
            )
        });
        self.address_eq.gruen_round_deg_3_from(
            constant,
            leading,
            &|block| {
                self.address_sums
                    .iter()
                    .zip(&self.weights)
                    .map(|(sums, &gamma)| {
                        (0..half)
                            .map(|index| {
                                let bound = self.bound_addresses[index];
                                gamma
                                    * sums[block * block_size + half + index]
                                    * bound
                                    * (bound - F::one())
                            })
                            .sum::<F>()
                    })
                    .sum()
            },
            claim,
        )
    }

    fn row_round(&self, claim: F) -> Result<UnivariatePoly<F>, F> {
        let address_factor = self.address_eq.current_scalar();
        let [constant, leading] = self.row_eq.par_fold_out_in_unreduced::<2>(&|index| {
            let mut constant = F::UnreducedProductAccum::zero();
            let mut leading = F::UnreducedProductAccum::zero();
            for (poly, &gamma) in self.rows.iter().zip(&self.weights) {
                let low = poly.get_bound_coeff(2 * index);
                let difference = poly.get_bound_coeff(2 * index + 1) - low;
                constant += (gamma * low).mul_to_product_accum(low - F::one());
                leading += (gamma * difference).mul_to_product_accum(difference);
            }
            [
                F::reduce_product_accum(constant),
                F::reduce_product_accum(leading),
            ]
        });
        self.row_eq.gruen_round_deg_3_from(
            address_factor * constant,
            address_factor * leading,
            &|index| {
                self.rows
                    .iter()
                    .zip(&self.weights)
                    .map(|(poly, &gamma)| {
                        let high = poly.get_bound_coeff(2 * index + 1);
                        address_factor * gamma * high * (high - F::one())
                    })
                    .sum()
            },
            claim,
        )
    }
}

impl<F: JoltField> ProveRounds<F> for BooleanityProver<F> {
    fn num_rounds(&self) -> usize {
        self.check.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge, round - 1);
        }
        let message = if round < self.address_vars {
            self.address_round(round, previous_claim)
        } else {
            self.row_round(previous_claim)
        };
        message.map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.check.rounds() - 1);
        Ok(())
    }
}

impl<F: JoltField> RelationWitness<F> for BooleanityProver<F> {
    type Relation = CoefficientCheck;

    fn relation(&self) -> &Self::Relation {
        &self.check
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.rows
            .iter()
            .map(PolynomialBinding::final_claim)
            .collect()
    }
}
