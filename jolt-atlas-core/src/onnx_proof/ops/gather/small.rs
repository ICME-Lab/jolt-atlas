use super::*;
use atlas_onnx_tracer::ops::GatherSmall;
use common::CommittedPolynomial;
use joltworks::subprotocols::{
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
    sumcheck::{BatchedSumcheck, Sumcheck},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for GatherSmall {
    #[tracing::instrument(skip_all, name = "GatherSmall::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

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

        let (hb_sumcheck, bool_sumcheck) = build_stage2_provers::<F>(node, prover);
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(hb_sumcheck), Box::new(bool_sumcheck)];
        let (stage2_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), stage2_proof));

        let mut hw_sumcheck = build_stage3_prover::<F>(node, prover);
        let (hw_proof, _) = Sumcheck::prove(
            &mut hw_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaHammingWeight), hw_proof));

        results
    }

    #[tracing::instrument(skip_all, name = "GatherSmall::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
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

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        vec![CommittedPolynomial::GatherRa(node.idx)]
    }
}

pub(crate) fn build_stage2_provers<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckProver<F>,
    SmallBooleanitySumcheckProver<F>,
) {
    let Operator::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];
    let num_lookups = indexes.len();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        num_words,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );

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

pub(crate) fn build_stage2_verifiers<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let Operator::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let graph = &verifier.preprocessing.model.graph;
    let dict = graph.nodes.get(&computation_node.inputs[0]).unwrap();
    let indices = graph.nodes.get(&computation_node.inputs[1]).unwrap();

    let num_words = dict.output_dims[gather_op.axis];
    let num_lookups = indices.pow2_padded_num_output_elements();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        num_words,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    let hb_sumcheck = HammingBooleanitySumcheckVerifier::new(hb_params);
    let bool_sumcheck = BooleanitySumcheckVerifier::new(bool_params);

    (hb_sumcheck, bool_sumcheck)
}

fn ra_hamming_bool_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingBooleanitySumcheckParams<F> {
    let polynomial_types = vec![VirtualPolynomial::HammingWeight];

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::NodeExecution(computation_node.idx),
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
    num_words: usize,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let polynomial_type = CommittedPolynomial::GatherRa(computation_node.idx);

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::NodeExecution(computation_node.idx),
        )
        .0
        .r;
    let r_address = transcript
        .challenge_vector_optimized::<F>(num_words.log_2())
        .into_opening();

    BooleanitySumcheckParams {
        d: 1,
        log_k_chunk: num_words.log_2(),
        log_t: num_lookups.log_2(),
        r_cycle: r_lookup,
        r_address,
        polynomial_types: vec![polynomial_type],
        gammas: vec![F::Challenge::from(1)],
        sumcheck_id: SumcheckId::Booleanity,
    }
}

pub(crate) fn build_stage3_prover<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> HammingWeightSumcheckProver<F> {
    let Operator::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        num_words,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let ra_evals = compute_ra_evals(&hw_params.r_cycle, indexes, num_words);

    HammingWeightSumcheckProver::gen(hw_params, vec![ra_evals])
}

pub(crate) fn build_stage3_verifier<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> HammingWeightSumcheckVerifier<F> {
    let Operator::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let graph = &verifier.preprocessing.model.graph;
    let dict = graph.nodes.get(&computation_node.inputs[0]).unwrap();

    let num_words = dict.output_dims[gather_op.axis];

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        num_words,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    HammingWeightSumcheckVerifier::new(hw_params)
}

fn ra_hamming_weight_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_words: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let polynomial_types = vec![CommittedPolynomial::GatherRa(computation_node.idx)];

    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::NodeExecution(computation_node.idx),
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
