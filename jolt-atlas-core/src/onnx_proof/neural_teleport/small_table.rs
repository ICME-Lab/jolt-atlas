//! The prover of a teleported trigonometric op's table lookup.

use crate::onnx_proof::neural_teleport::utils::compute_ra_evals_from_usize_indices;
use crate::onnx_proof::relation::{RelationChallenge, RelationParams, RelationSetup};
#[cfg(test)]
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
};
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator, relation::prover::RelationWitness,
};
use atlas_trace::Tensor;
use common::parallel::par_enabled;
#[cfg(test)]
use common::VirtualPoly;
pub use jolt_atlas_verifier::onnx_proof::neural_teleport::small_table::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// Degree of the table-lookup sumcheck.
const DEGREE: usize = 2;

/// Read-raf prover for a trig table lookup: `output[i] = S::table[downscaled[i]]`.
pub struct TrigTableProver<F: JoltField, S: SmallTable> {
    relation: SmallTableRelation<S>,
    setup: RelationSetup<F>,
    gamma: F,
    table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField, S: SmallTable> TrigTableProver<F, S> {
    /// The prover for the downscaled input's lookup, and the lookup indices.
    pub fn initialize(
        downscaled: &Tensor<i32>,
        params: RelationParams<F, SmallTableRelation<S>>,
        _accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> (Self, Vec<usize>) {
        let (relation, setup) = params.into_parts();
        let gamma = setup.challenge(&RelationChallenge::Gamma);
        let table_size = 1 << S::VARS;
        assert!(downscaled
            .iter()
            .all(|&x| (0..table_size as i32).contains(&x)));

        let table = MultilinearPolynomial::from(S::materialize());
        let lookup_indices: Vec<usize> = downscaled
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&x| x as usize)
            .collect();
        let input_onehot = MultilinearPolynomial::from(compute_ra_evals_from_usize_indices(
            &setup.incoming_point,
            &lookup_indices,
            table_size,
        ));
        assert_eq!(input_onehot.len(), table.len());
        let identity = IdentityPolynomial::new(S::VARS);

        #[cfg(test)]
        {
            let accessor = AccOpeningAccessor::new(_accumulator, relation.node());
            let downscaled_claim = accessor.get_advice(VirtualPoly::TrigDownscaled).1;
            let rv_claim = accessor.get_reduced_opening().1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let a = input_onehot.get_bound_coeff(i);
                    let b = table.get_bound_coeff(i);
                    a * (b + gamma * F::from_u32(i as u32))
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(
                rv_claim + gamma * downscaled_claim,
                claim
            )
        }

        (
            Self {
                relation,
                setup,
                gamma,
                table,
                input_onehot,
                identity,
            },
            lookup_indices,
        )
    }

    fn round_message(&mut self, previous_claim: F) -> UnivariatePoly<F> {
        let Self {
            input_onehot,
            table,
            identity,
            gamma,
            ..
        } = self;

        let evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra = input_onehot.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let t = table.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let id = identity.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                [
                    ra[0] * (t[0] + id[0] * *gamma),
                    ra[1] * (t[1] + id[1] * *gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.input_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, S: SmallTable> RelationWitness<F> for TrigTableProver<F, S> {
    type Relation = SmallTableRelation<S>;

    fn relation(&self) -> &SmallTableRelation<S> {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.input_onehot.final_claim()]
    }
}

impl<F: JoltField, S: SmallTable> ProveRounds<F> for TrigTableProver<F, S> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        Ok(self.round_message(previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
