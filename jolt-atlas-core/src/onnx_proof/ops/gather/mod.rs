use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_id_builder::{OpeningIdBuilder, OpeningTarget},
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::parallel::par_enabled;
use common::VirtualPolynomial;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

mod large;
mod small;

const DEGREE_BOUND: usize = 2;

/// Parameters for proving gather (indexed lookup) operations.
///
/// Gather selects elements from an input dictionary tensor using an index tensor.
/// Uses one-hot encoding to prove correct lookups with a folding challenge gamma.
#[derive(Clone)]
pub struct GatherParams<F: JoltField> {
    gamma: F,
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    lookup_vars: usize,
    num_words: usize,
}

impl<F: JoltField> GatherParams<F> {
    /// Create new gather parameters from a computation node, graph, accumulator, and transcript.
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();

        let dict_len = match &computation_node.operator {
            Operator::GatherSmall(gather_op) => gather_op.dict_len,
            Operator::GatherLarge(gather_op) => gather_op.dict_len,
            _ => panic!("Expected Gather operator"),
        };

        let input_indices = &graph.nodes.get(&computation_node.inputs[1]).unwrap();
        let num_words = dict_len.next_power_of_two();
        let lookup_vars = input_indices.pow2_padded_num_output_elements().log_2();

        let r_node_output = accumulator
            .get_node_output_opening(computation_node.idx)
            .0
            .r;
        Self {
            gamma,
            r_node_output: r_node_output.into(),
            computation_node,
            num_words,
            lookup_vars,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for GatherParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let node = &self.computation_node;
        let builder = OpeningIdBuilder::new(node);
        let rv_claim = accumulator.get_node_output_opening(node.idx).1;

        let index_id = builder.node_io(OpeningTarget::Input(1));
        let index_claim = accumulator.get_virtual_polynomial_opening(index_id).1;

        rv_claim + self.gamma * index_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }
}

/// Prover state for gather sumcheck protocol.
///
/// Implements a Read-Raf sumcheck asserting that each output[i] corresponds to a lookup
/// into the input dictionary at index input1[i].
pub struct GatherProver<F: JoltField> {
    params: GatherParams<F>,
    dictionary: MultilinearPolynomial<F>,
    index_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> GatherProver<F> {
    /// Initialize the prover with trace data, parameters, accumulator, and transcript.
    pub fn initialize(
        trace: &Trace,
        params: GatherParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let [dictionary, indexes] = operands[..] else {
            panic!("Expected two operands for Gather operation")
        };

        let dict_len = dictionary.dims()[0];
        let word_dim = dictionary.dims().iter().product::<usize>() / dict_len;

        let (r_index, r_word) = params.r_node_output.r.split_at(params.lookup_vars);
        assert_eq!(r_word.len(), word_dim.log_2());

        let padded_indexes = indexes.padded_next_power_of_two();
        let index_claim = MultilinearPolynomial::from(padded_indexes.clone()).evaluate(r_index);
        let index_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
            SumcheckId::NodeExecution(params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            index_id,
            OpeningPoint::new(r_index.to_vec()),
            index_claim,
        );

        let index_onehot: Vec<F> = compute_ra_evals(r_index, &padded_indexes, params.num_words);
        let dict: Vec<F> = fold_dictionary(r_word, dictionary, params.num_words);

        let index_onehot = MultilinearPolynomial::from(index_onehot);
        let dictionary = MultilinearPolynomial::from(dict);
        assert_eq!(index_onehot.len(), dictionary.len());
        let identity = IdentityPolynomial::new(params.num_words.log_2());

        #[cfg(test)]
        {
            let rv_claim = accumulator
                .get_node_output_opening(params.computation_node.idx)
                .1;
            let claim = (0..index_onehot.len())
                .map(|i| {
                    let a = index_onehot.get_bound_coeff(i);
                    let b = dictionary.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            assert_eq!(rv_claim + params.gamma * index_claim, claim)
        }
        Self {
            params,
            index_onehot,
            dictionary,
            identity,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for GatherProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
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
                    ra_evals[0] * (dict_evals[0] + id_evals[0] * self.params.gamma),
                    ra_evals[1] * (dict_evals[1] + id_evals[1] * self.params.gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.index_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.dictionary.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let node = &self.params.computation_node;
        let builder = OpeningIdBuilder::new(node);
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let (r_index, r_word) = self
            .params
            .r_node_output
            .r
            .split_at(self.params.lookup_vars);

        let r_idx_onehot = [&opening_point.r, r_index].concat();
        let ra_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            ra_id,
            OpeningPoint::new(r_idx_onehot),
            self.index_onehot.final_sumcheck_claim(),
        );
        let r_dict = [&opening_point.r, r_word].concat();
        let dictionary_id = builder.node_io(OpeningTarget::Input(0));
        accumulator.append_virtual(
            transcript,
            dictionary_id,
            OpeningPoint::new(r_dict),
            self.dictionary.final_sumcheck_claim(),
        );
    }
}

/// Verifier for gather sumcheck protocol.
///
/// Verifies that the prover's claims are consistent with the gather (indexed lookup) operation.
pub struct GatherVerifier<F: JoltField> {
    params: GatherParams<F>,
}

impl<F: JoltField> GatherVerifier<F> {
    /// Create a new verifier for the gather operation, caching index operand openings.
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = GatherParams::new(computation_node, graph, accumulator, transcript);

        let (r_index, _) = params.r_node_output.r.split_at(params.lookup_vars);
        let index_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
            SumcheckId::NodeExecution(params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, index_id, OpeningPoint::new(r_index.to_vec()));

        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for GatherVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let ra_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let ra_claim = accumulator.get_virtual_polynomial_opening(ra_id).1;
        let int_eval =
            IdentityPolynomial::new(self.params.num_words.log_2()).evaluate(&opening_point.r);
        let dict_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        ra_claim * (dict_claim + self.params.gamma * int_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let node = &self.params.computation_node;
        let builder = OpeningIdBuilder::new(node);
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let (r_index, r_word) = self
            .params
            .r_node_output
            .r
            .split_at(self.params.lookup_vars);
        let r_idx_onehot = [&opening_point.r, r_index].concat();
        let ra_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, ra_id, OpeningPoint::new(r_idx_onehot));
        let r_dict = [&opening_point.r, r_word].concat();
        let dictionary_id = builder.node_io(OpeningTarget::Input(0));
        accumulator.append_virtual(transcript, dictionary_id, OpeningPoint::new(r_dict));
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

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::test::ModelBuilder, model::Model, ops::Operator, tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn gather_model(input_shape: &[usize], dictionnary_len: usize, word_dim: usize) -> Model {
        let mut b = ModelBuilder::new();
        let dictionnary = {
            let data = (0..dictionnary_len * word_dim)
                .map(|i| i as i32)
                .collect::<Vec<_>>();
            Tensor::construct(data, vec![dictionnary_len, word_dim])
        };
        let dict = b.constant(dictionnary);
        let indexes = b.input(input_shape.to_vec());
        let res = b.gather(
            dict,
            indexes,
            0,
            [input_shape.to_vec(), vec![word_dim]].concat(),
        );
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_gather() {
        let indices_dims = vec![4, 8];
        let dict_len = 32;
        let word_dim = 4;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_range(&mut rng, &indices_dims, 0..dict_len as i32);
        let model = gather_model(&indices_dims, dict_len, word_dim);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_gather_small_path() {
        let indices_dims = vec![4, 8];
        let dict_len = 32;
        let word_dim = 4;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_range(&mut rng, &indices_dims, 0..dict_len as i32);
        let model = gather_model(&indices_dims, dict_len, word_dim);
        assert!(matches!(
            model.graph.nodes[&2].operator,
            Operator::GatherSmall(_)
        ));
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_gather_large_path() {
        let indices_dims = vec![2, 4];
        let dict_len = 70000;
        let word_dim = 2;
        let mut rng = StdRng::seed_from_u64(0x88A);
        let input = Tensor::<i32>::random_range(&mut rng, &indices_dims, 0..dict_len as i32);
        let model = gather_model(&indices_dims, dict_len, word_dim);
        assert!(matches!(
            model.graph.nodes[&2].operator,
            Operator::GatherLarge(_)
        ));
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two gather path not fully validated yet"]
    fn test_gather_non_power_of_two_input_len() {
        let indices_dims = vec![5, 7];
        let dict_len = 33;
        let word_dim = 5;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_range(&mut rng, &indices_dims, 0..dict_len as i32);
        let model = gather_model(&indices_dims, dict_len, word_dim);
        unit_test_op(model, &[input]);
    }
}
