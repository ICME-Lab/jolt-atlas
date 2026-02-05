use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Gather,
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckVerifier, SmallBooleanitySumcheckProver,
        },
        hamming_booleanity::{
            HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver,
            HammingBooleanitySumcheckVerifier,
        },
        hamming_weight::{
            HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
        },
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Gather {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Execution proof (Read-Value and Raf)
        let params = GatherParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = GatherProver::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        // Stage 2 proofs (HammingBooleanity and Booleanity)

        let (hb_sumcheck, bool_sumcheck) = build_stage2_provers::<F>(node, prover);
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(hb_sumcheck), Box::new(bool_sumcheck)];

        let (stage2_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), stage2_proof));

        // Stage 3 proof (HammingWeight)
        let mut hw_sumcheck = build_stage3_prover::<F>(node, prover);
        let (hw_proof, _) = Sumcheck::prove(
            &mut hw_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        results.push((ProofId(node.idx, ProofType::RaHammingWeight), hw_proof));

        results
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution verification (Read-Value and Raf)
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = GatherVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 2 verification (HammingBooleanity and Booleanity)
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let (hb_sumcheck, bool_sumcheck) = build_stage2_verifiers::<F>(node, verifier);
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&hb_sumcheck, &bool_sumcheck];
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 3 verification (HammingWeight)
        let hw_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaHammingWeight))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let hw_sumcheck = build_stage3_verifier::<F>(node, verifier);
        Sumcheck::verify(
            hw_proof,
            &hw_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }
}

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
pub struct GatherParams<F: JoltField> {
    gamma: F,
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    lookup_vars: usize,
    num_words: usize,
}

impl<F: JoltField> GatherParams<F> {
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();

        let input_dict = &graph.nodes.get(&computation_node.inputs[0]).unwrap();
        let num_words = input_dict.output_dims[0];
        let lookup_vars = computation_node.output_dims[0].log_2();

        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            gamma,
            r_node_output,
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
        let rv_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        let index_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.inputs[1]),
                SumcheckId::Execution,
            )
            .1;

        rv_claim + self.gamma * index_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }
}

// This is essentially a Read-Raf sumcheck,
// Where we assert that each output[i] corresponds to a lookup into input0 at index input1[i]
pub struct GatherProver<F: JoltField> {
    params: GatherParams<F>,
    dictionary: MultilinearPolynomial<F>,
    index_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> GatherProver<F> {
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

        let word_dim = dictionary.dims().get(1).unwrap_or(&1usize);

        let (r_index, r_word) = params.r_node_output.split_at(params.lookup_vars);
        assert_eq!(r_word.len(), word_dim.log_2(),);

        let index_claim = MultilinearPolynomial::from(indexes.clone()).evaluate(r_index);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
            SumcheckId::Execution,
            r_index.to_vec().into(),
            index_claim,
        );

        let index_onehot: Vec<F> = compute_ra_evals(r_index, indexes, params.num_words);
        let dict: Vec<F> = fold_dictionary(r_word, dictionary);

        let index_onehot = MultilinearPolynomial::from(index_onehot);
        let dictionary = MultilinearPolynomial::from(dict);
        assert_eq!(index_onehot.len(), dictionary.len());
        let identity = IdentityPolynomial::new(params.num_words.log_2());

        #[cfg(test)]
        {
            let rv_claim = accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::NodeOutput(params.computation_node.idx),
                    SumcheckId::Execution,
                )
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let (r_index, r_word) = self.params.r_node_output.split_at(self.params.lookup_vars);

        let r_idx_onehot = [r_index, &opening_point.r].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            r_idx_onehot.into(),
            self.index_onehot.final_sumcheck_claim(),
        );
        let r_dict = [&opening_point.r, r_word].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            r_dict.into(),
            self.dictionary.final_sumcheck_claim(),
        );
    }
}

pub struct GatherVerifier<F: JoltField> {
    params: GatherParams<F>,
}

impl<F: JoltField> GatherVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = GatherParams::new(computation_node, graph, accumulator, transcript);

        let (r_index, _) = params.r_node_output.split_at(params.lookup_vars);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
            SumcheckId::Execution,
            r_index.to_vec().into(),
        );

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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let dict_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;
        let int_eval =
            IdentityPolynomial::new(self.params.num_words.log_2()).evaluate(&opening_point.r);

        ra_claim * (dict_claim + self.params.gamma * int_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let (r_index, r_word) = self.params.r_node_output.split_at(self.params.lookup_vars);
        let r_idx_onehot = [r_index, &opening_point.r].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            r_idx_onehot.into(),
        );
        let r_dict = [&opening_point.r, r_word].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            r_dict.into(),
        );
    }
}

// From the read indexes, computes the bound ra vector.
fn compute_ra_evals<F>(r: &[F::Challenge], indexes: &Tensor<i32>, num_words: usize) -> Vec<F>
where
    F: JoltField,
{
    let E = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = indexes.len().div_ceil(num_threads);

    let indexes_usize = indexes
        .par_iter()
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
                local_ra[k] += E[global_j];
            });
            local_ra
        })
        .collect();
    let mut ra = unsafe_allocate_zero_vec::<F>(num_words);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}

fn fold_dictionary<F: JoltField>(r: &[F::Challenge], dictionary: &Tensor<i32>) -> Vec<F> {
    let E = EqPolynomial::evals(r);

    dictionary
        .par_chunks(E.len())
        .map(|word_vector| {
            word_vector
                .iter()
                .zip(E.iter())
                .map(|(&word_coeff, &e)| F::from_i32(word_coeff) * e)
                .sum()
        })
        .collect()
}

fn build_stage2_provers<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckProver<F>,
    SmallBooleanitySumcheckProver<F>,
) {
    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        &prover.preprocessing.model.graph,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[0];

    let hw = {
        let mut lookup_vec = vec![F::one(); indexes.len()];
        lookup_vec.resize(indexes.len().next_power_of_two(), F::zero());
        lookup_vec
    };
    let ra_evals = compute_ra_evals(&bool_params.r_cycle, indexes, num_words);

    let indexes_u = indexes.iter().map(|&x| Some(x as u16)).collect();

    let hb_sumcheck = HammingBooleanitySumcheckProver::gen(hb_params, vec![hw]);
    let bool_sumcheck =
        SmallBooleanitySumcheckProver::gen(bool_params, vec![ra_evals], vec![indexes_u]);

    (hb_sumcheck, bool_sumcheck)
}

fn build_stage2_verifiers<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        &verifier.preprocessing.model.graph,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    let hb_sumcheck = HammingBooleanitySumcheckVerifier::new(hb_params);
    let bool_sumcheck = BooleanitySumcheckVerifier::new(bool_params);

    (hb_sumcheck, bool_sumcheck)
}

fn ra_hamming_bool_params<F: JoltField>(
    computation_node: &ComputationNode,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingBooleanitySumcheckParams<F> {
    let polynomial_types = vec![VirtualPolynomial::HammingWeight];

    let num_lookups = computation_node.output_dims[0];

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::Execution,
        )
        .0
        .r;

    HammingBooleanitySumcheckParams {
        d: 1,
        num_rounds: num_lookups.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,

        r_cycle: r_lookup,
        sumcheck_id: SumcheckId::RamHammingBooleanity,
    }
}

fn ra_booleanity_params<F: JoltField>(
    computation_node: &ComputationNode,
    graph: &ComputationGraph,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let num_words = graph
        .nodes
        .get(&computation_node.inputs[0])
        .unwrap()
        .output_dims[0];

    let polynomial_type = CommittedPolynomial::NodeOutputRaD(computation_node.idx, 0);

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::Execution,
        )
        .0
        .r;
    let r_address = transcript.challenge_vector_optimized::<F>(num_words.log_2());

    BooleanitySumcheckParams {
        d: 1,
        log_k_chunk: num_words.log_2(),
        log_t: computation_node.output_dims[0].log_2(),
        r_cycle: r_lookup,
        r_address,
        polynomial_types: vec![polynomial_type],
        gammas: vec![F::Challenge::from(1)],
        sumcheck_id: SumcheckId::Booleanity,
    }
}

fn build_stage3_prover<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> HammingWeightSumcheckProver<F> {
    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        &prover.preprocessing.model.graph,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[0];

    let ra_evals = compute_ra_evals(&hw_params.r_cycle, indexes, num_words);

    HammingWeightSumcheckProver::gen(hw_params, vec![ra_evals])
}

fn build_stage3_verifier<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> HammingWeightSumcheckVerifier<F> {
    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        &verifier.preprocessing.model.graph,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    HammingWeightSumcheckVerifier::new(hw_params)
}

fn ra_hamming_weight_params<F: JoltField>(
    computation_node: &ComputationNode,
    graph: &ComputationGraph,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let dict = graph.nodes.get(&computation_node.inputs[0]).unwrap();
    let num_words = dict.output_dims[0];

    let polynomial_types = vec![CommittedPolynomial::NodeOutputRaD(computation_node.idx, 0)];

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::Execution,
        )
        .0
        .r;

    HammingWeightSumcheckParams {
        d: 1,
        num_rounds: num_words.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle: r_lookup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::AtlasSharedPreprocessing;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::BTreeMap;

    #[test]
    fn test_gather() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x888);

        let dict_len = 32;
        let word_dim = 4;

        let input = Tensor::<i32>::random_range(&mut rng, &[T], 0..dict_len as i32);

        let model = model::test::gather_model(&[T], dict_len, word_dim);
        let trace = model.trace(&[input]);

        let prover_transcript = Blake2bTranscript::new(&[]);
        let preprocessing: AtlasSharedPreprocessing =
            AtlasSharedPreprocessing::preprocess(model.clone());
        let prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new(log_T);
        let mut prover = Prover {
            trace: trace.clone(),
            accumulator: prover_opening_accumulator,
            preprocessing,
            transcript: prover_transcript,
        };

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let r_node_output: Vec<<Fr as JoltField>::Challenge> = prover
            .transcript
            .challenge_vector_optimized::<Fr>(computation_node.num_output_elements().log_2());

        let gather_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            gather_claim,
        );

        let verifier_transcript = Blake2bTranscript::new(&[]);
        let verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new(log_T);

        let proofs = Gather { dim: 0 }.prove(computation_node, &mut prover);
        let proofs = BTreeMap::from_iter(proofs);

        let io = Trace::io(&trace, &model);

        let mut verifier = Verifier {
            proofs: &proofs,
            accumulator: verifier_opening_accumulator,
            preprocessing: &prover.preprocessing.clone(),
            io: &io,
            transcript: verifier_transcript,
        };
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> = verifier
            .transcript
            .challenge_vector_optimized::<Fr>(computation_node.num_output_elements().log_2());

        // Take claims
        for (key, (_, value)) in &prover.accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier
                .accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let res = Gather { dim: 0 }.verify(computation_node, &mut verifier);

        let r_prover: Fr = prover.transcript.challenge_scalar();
        let r_verifier: Fr = verifier.transcript.challenge_scalar();
        assert_eq!(r_prover, r_verifier);

        verifier.transcript.compare_to(prover.transcript);
        res.unwrap();
    }
}
