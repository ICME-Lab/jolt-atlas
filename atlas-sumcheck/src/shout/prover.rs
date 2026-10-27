//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::{
    booleanity::prover::BooleanitySumcheckProver, opening_proof::OpeningAccumulator,
    ra_virtual::prover::RaSumcheckProver, sumcheck::prover::SumcheckInstanceProver,
};
use atlas_algebra::poly::{
    identity_poly::IdentityPolynomial,
    multilinear_polynomial::{
        BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
    },
};
use atlas_algebra::{
    field::JoltField, poly::eq_poly::EqPolynomial, utils::thread::unsafe_allocate_zero_vec,
};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::array;

pub fn read_raf_prover<F: JoltField, T: Transcript>(
    provider: &impl ReadRafProvider<F>,
    lookup_indices: &[usize],
    table: &[i32],
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Box<dyn SumcheckInstanceProver<F, T>> {
    let params = ReadRafParams::new(provider, accumulator, transcript);
    Box::new(ReadRafProver::initialize(lookup_indices, table, params))
}

/// Prover state for lookups into small tables.
pub struct ReadRafProver<F: JoltField> {
    relation: ReadRafRelation,
    setup: RelationSetup<F>,
    gamma: F,
    val: MultilinearPolynomial<F>,
    G: MultilinearPolynomial<F>,
    int: IdentityPolynomial<F>,
}

impl<F: JoltField> ReadRafProver<F> {
    #[tracing::instrument(name = "ReadRafProver::initialize", skip_all)]
    /// Initialize the prover for lookups.
    pub fn initialize(lookup_indices: &[usize], table: &[i32], params: ReadRafParams<F>) -> Self {
        let table_size = 1 << params.log_K;
        let E = EqPolynomial::evals(&params.r);
        let G = lookup_indices
            .par_iter()
            .with_min_len(par_enabled())
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec::<F>(table_size),
                |mut local_G, (j, &lookup_index)| {
                    local_G[lookup_index] += E[j];
                    local_G
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec::<F>(table_size),
                |mut acc, local_G| {
                    for (i, &val) in local_G.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );
        let val = MultilinearPolynomial::from(table.to_vec());
        let int = IdentityPolynomial::new(params.log_K);
        let gamma = params.gamma;
        let (relation, setup) = params.into_parts(table.to_vec());
        Self {
            int,
            relation,
            setup,
            gamma,
            val,
            G: MultilinearPolynomial::from(G),
        }
    }
}

impl<F: JoltField> RelationWitness<F> for ReadRafProver<F> {
    type Relation = ReadRafRelation;

    fn relation(&self) -> &ReadRafRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.G.final_claim()]
    }
}

crate::relation_witness_instance!([F: JoltField] ReadRafProver<F>);

impl<F: JoltField> ReadRafProver<F> {
    #[tracing::instrument(name = "ReadRafProver::round_message", skip_all)]
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let Self { G, val, int, .. } = self;
        let half_poly_len = val.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let val_evals =
                    val.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let g_evals = G.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let int_evals =
                    int.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                [
                    g_evals[0] * (val_evals[0] + self.gamma * int_evals[0]),
                    g_evals[1] * (val_evals[1] + self.gamma * int_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UnivariatePoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.val.bind_parallel(r_j, BindingOrder::HighToLow);
        self.G.bind_parallel(r_j, BindingOrder::HighToLow);
        self.int.bind_parallel(r_j, BindingOrder::HighToLow);
    }
}

impl<F: JoltField> ProveRounds<F> for ReadRafProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}

/// Build the RA one-hot sumcheck **provers** (RaVirtual, HammingWeight,
/// Booleanity) from an [`RaOneHotEncoding`] and pre-computed `lookup_indices`.
///
/// Transcript challenge draw order:
/// 1. HammingWeight — `gamma_powers`
/// 2. Booleanity — `gammas`, then `r_address`
/// 3. RaVirtual — no challenges
///
/// Returns `[RaVirtual, Booleanity (+ Hamming weight)]`.
pub fn ra_onehot_provers<F: JoltField, T: Transcript>(
    encoding: &impl RaOneHotEncoding,
    lookup_indices: &[usize],
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> [Box<dyn SumcheckInstanceProver<F, T>>; 2] {
    let challenges = ra_onehot_challenges::<F, T>(encoding, transcript);
    ra_onehot_provers_with(encoding, lookup_indices, accumulator, challenges)
}

/// [`ra_onehot_provers`] with the challenges already drawn (no transcript).
pub fn ra_onehot_provers_with<F: JoltField, T: Transcript>(
    encoding: &impl RaOneHotEncoding,
    lookup_indices: &[usize],
    accumulator: &dyn OpeningAccumulator<F>,
    challenges: RaOneHotChallenges<F>,
) -> [Box<dyn SumcheckInstanceProver<F, T>>; 2] {
    let params = ra_onehot_params(encoding, accumulator, challenges);
    ra_onehot_provers_from_params(params, lookup_indices)
}

/// Build the RA one-hot provers from resolved parameters (no accumulator
/// or transcript access — safe to run in parallel across instances).
pub fn ra_onehot_provers_from_params<F: JoltField, T: Transcript>(
    params: RaOneHotParams<F>,
    lookup_indices: &[usize],
) -> [Box<dyn SumcheckInstanceProver<F, T>>; 2] {
    let (ra_params, booleanity_params) = params;
    let one_hot_params = &ra_params.one_hot_params;

    // --- Compute G and H_indices once (shared across both provers) ---
    let G = compute_ra_evals(lookup_indices, one_hot_params, &booleanity_params.r_cycle);
    let H_indices = compute_instruction_h_indices(lookup_indices, one_hot_params);

    [
        Box::new(RaSumcheckProver::gen(ra_params, H_indices.clone())),
        Box::new(BooleanitySumcheckProver::gen(
            booleanity_params,
            G,
            H_indices,
        )),
    ]
}
