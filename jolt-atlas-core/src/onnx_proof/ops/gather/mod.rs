use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::relation::{RelationChallenge, RelationParams, RelationSetup};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::IntoProverProvider;
use crate::{
    onnx_proof::{ProofId, ProofType, Prover},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, EvaluateMle, MultilinearPolynomial, PolynomialBinding,
            PolynomialEvaluation,
        },
        opening_point::OpeningPoint,
    },
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator,
    relation::prover::RelationWitness,
    sumcheck::{prover::SumcheckInstanceProver, SumcheckInstanceProof},
};
use atlas_trace::{trace::LayerData, Op, Tensor, Trace};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

/// Prover side of gather over a large dictionary.
pub mod large;
/// Prover side of gather over a small dictionary.
pub mod small;

/// Prover state for gather sumcheck protocol.
///
/// Implements a Read-Raf sumcheck asserting that each output[i] corresponds to a lookup
/// into the input dictionary at index input1[i].
pub struct GatherProver<F: JoltField> {
    relation: GatherRelation,
    setup: RelationSetup<F>,
    gamma: F,
    dictionary: MultilinearPolynomial<F>,
    index_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> GatherProver<F> {
    /// Initialize the prover with trace data, the set-up relation, accumulator, and transcript.
    pub fn initialize(
        trace: &Trace,
        params: RelationParams<F, GatherRelation>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (relation, setup) = params.into_parts();
        let gamma = setup.challenge(&RelationChallenge::Gamma);
        let num_words = relation.num_words();
        let LayerData { operands, .. } = TraceView::layer_data(trace, relation.node());
        let [dictionary, indexes] = operands[..] else {
            panic!("Expected two operands for Gather operation")
        };

        let dict_len = dictionary.dims()[0];
        let word_dim = dictionary.dims().iter().product::<usize>() / dict_len;

        let (r_index, r_word) = setup.incoming_point.split_at(relation.lookup_vars());
        assert_eq!(r_word.len(), word_dim.log_2());

        let padded_indexes = indexes.padded_pow2();
        let index_claim = padded_indexes.evaluate_mle(r_index);

        let mut provider = AccOpeningAccessor::new(accumulator, relation.node())
            .into_provider(transcript, OpeningPoint::new(r_index.to_vec()));
        provider.append_nodeio(Target::Input(1), index_claim);

        let index_onehot: Vec<F> = compute_ra_evals(r_index, &padded_indexes, num_words);
        let dict: Vec<F> = fold_dictionary(r_word, dictionary, num_words);

        let index_onehot = MultilinearPolynomial::from(index_onehot);
        let dictionary = MultilinearPolynomial::from(dict);
        assert_eq!(index_onehot.len(), dictionary.len());
        let identity = IdentityPolynomial::new(num_words.log_2());

        #[cfg(test)]
        {
            let rv_claim = provider.get_reduced_opening().1;
            let claim = (0..index_onehot.len())
                .map(|i| {
                    let a = index_onehot.get_bound_coeff(i);
                    let b = dictionary.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + gamma * int)
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(
                rv_claim + gamma * index_claim,
                claim
            )
        }
        Self {
            relation,
            setup,
            gamma,
            index_onehot,
            dictionary,
            identity,
        }
    }
}

impl<F: JoltField> RelationWitness<F> for GatherProver<F> {
    type Relation = GatherRelation;

    fn relation(&self) -> &GatherRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![
            self.index_onehot.final_claim(),
            self.dictionary.final_claim(),
        ]
    }
}

atlas_sumcheck::relation_witness_instance!([F: JoltField] GatherProver<F>);

impl<F: JoltField> GatherProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let Self {
            index_onehot,
            dictionary,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..index_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra_evals =
                    index_onehot.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let dict_evals =
                    dictionary.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals = identity.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (dict_evals[0] + id_evals[0] * self.gamma),
                    ra_evals[1] * (dict_evals[1] + id_evals[1] * self.gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UnivariatePoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.index_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.dictionary.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for GatherProver<F> {
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

// From the read indexes, computes the bound ra vector.
pub(crate) fn compute_ra_evals<F>(r: &[F], indexes: &Tensor<i32>, num_words: usize) -> Vec<F>
where
    F: JoltField,
{
    let e = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = indexes.len().div_ceil(num_threads);

    let indexes_usize = indexes
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();

    let partial_results: Vec<Vec<F>> = indexes_usize
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_ra = unsafe_allocate_zero_vec::<F>(num_words);
            let base_idx = chunk_idx * chunk_size;
            chunk.iter().enumerate().for_each(|(local_j, &k)| {
                let global_j = base_idx + local_j;
                local_ra[k] += e[global_j];
            });
            local_ra
        })
        .collect();
    let mut ra = unsafe_allocate_zero_vec::<F>(num_words);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .with_min_len(par_enabled())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}

// TODO: Assert correct behavior for axis != 0
pub(crate) fn fold_dictionary<F: JoltField>(
    r: &[F],
    dictionary: &Tensor<i32>,
    num_words: usize,
) -> Vec<F> {
    let eq = EqPolynomial::evals(r);

    let mut folded: Vec<F> = dictionary
        .par_chunks(eq.len())
        .map(|word_vector| {
            word_vector
                .iter()
                .zip(eq.iter())
                .map(|(&word_coeff, &e)| F::from_i32(word_coeff) * e)
                .sum()
        })
        .collect();
    folded.resize(num_words, F::zero());
    folded
}
