use super::{TestField, TestPCS, TestTranscript};
use crate::{
    onnx_proof::{
        neural_teleport::division::{
            compute_division, TeleportDivisionParams, TeleportDivisionProver,
        },
        ops::{
            tanh::{TanhParams, TanhProver, TanhRaEncoding},
            NodeCommittedPolynomials, OperatorProver,
        },
        range_checking::{
            range_check_operands::{RangeCheckOperandsBase, RangeCheckingOperandsTrait},
            RangeCheckEncoding, RangeCheckProvider,
        },
        witness::WitnessGenerator,
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
        ProofId, ProofType, Prover, ProverDebugInfo,
    },
    utils::compute_lookup_indices_from_operands,
};
use atlas_onnx_tracer::{
    model::{
        trace::{ModelExecutionIO, Trace},
        test::ModelBuilder,
        Model,
    },
    node::ComputationNode,
    ops::{Operator, Tanh},
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{OpeningId, SumcheckId},
    },
    subprotocols::{
        shout,
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

type ProofOutput = (
    ONNXProof<TestField, TestTranscript, TestPCS>,
    ModelExecutionIO,
    Option<ProverDebugInfo<TestField, TestTranscript>>,
);

struct TauRangecheckBypassProof;

struct TauBypassTeleportRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

impl RangeCheckingOperandsTrait for TauBypassTeleportRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                input_operands: vec![
                    VirtualPolynomial::TeleportRemainder(node.idx),
                    VirtualPolynomial::NodeOutput(node.inputs[0]),
                ],
                virtual_ra: VirtualPolynomial::TeleportRangeCheckRa(node.idx),
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let tau = tau_override(node);
        let layer_data = Trace::layer_data(trace, node);
        let [input_tensor] = layer_data.operands[..] else {
            panic!("Expected exactly one input tensor for tanh teleport range-check");
        };
        let (_, remainder) = compute_division(input_tensor, tau);
        let divisor_tensor = Tensor::construct(vec![tau], vec![1])
            .expand(input_tensor.dims())
            .expect("tau override tensor should broadcast over input dims");
        (remainder, divisor_tensor)
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::TeleportRangeCheckRaD(self.base.node_idx, d)
    }
}

impl TauRangecheckBypassProof {
    fn prove(pp: &AtlasProverPreprocessing<TestField, TestPCS>, inputs: &[Tensor<i32>]) -> ProofOutput {
        let trace = pp.model().trace(inputs);
        let io = Trace::io(&trace, pp.model());

        let mut prover: Prover<TestField, TestTranscript> = Prover::new(pp.shared.clone(), trace);
        let mut proofs = BTreeMap::new();

        let (poly_map, commitments) = Self::commit_witness_polynomials(pp.model(), &prover.trace, &pp.generators, &mut prover.transcript);
        ONNXProof::<TestField, TestTranscript, TestPCS>::output_claim(&mut prover);
        Self::iop(pp.model().nodes(), &mut prover, &mut proofs);
        let reduced_opening_proof =
            ONNXProof::<TestField, TestTranscript, TestPCS>::prove_reduced_openings(
                &mut prover,
                &poly_map,
                &pp.generators,
            );
        ONNXProof::<TestField, TestTranscript, TestPCS>::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            reduced_opening_proof,
        )
    }

    fn commit_witness_polynomials(
        model: &Model,
        trace: &Trace,
        generators: &<TestPCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::ProverSetup,
        transcript: &mut TestTranscript,
    ) -> (
        BTreeMap<CommittedPolynomial, MultilinearPolynomial<TestField>>,
        Vec<<TestPCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment>,
    ) {
        let poly_map = Self::polynomial_map(model, trace);
        let commitments = poly_map
            .values()
            .map(|poly| TestPCS::commit(poly, generators).0)
            .collect::<Vec<_>>();
        for commitment in &commitments {
            transcript.append_serializable(commitment);
        }
        (poly_map, commitments)
    }

    fn polynomial_map(
        model: &Model,
        trace: &Trace,
    ) -> BTreeMap<CommittedPolynomial, MultilinearPolynomial<TestField>> {
        let tanh_node = find_tanh_node(model);
        model
            .graph
            .nodes
            .values()
            .flat_map(|node| NodeCommittedPolynomials::get_committed_polynomials::<TestField, TestTranscript>(node))
            .map(|poly| {
                let witness = match poly {
                    CommittedPolynomial::TeleportRangeCheckRaD(node_idx, d)
                        if node_idx == tanh_node.idx =>
                    {
                        build_tau_bypass_rangecheck_rad_witness(model, trace, node_idx, d)
                    }
                    _ => poly.generate_witness(model, trace),
                };
                (poly, witness)
            })
            .collect()
    }

    fn iop(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<TestField, TestTranscript>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<TestField, TestTranscript>>,
    ) {
        for (_, computation_node) in computation_nodes.iter().rev() {
            match &computation_node.operator {
                Operator::Tanh(op) => {
                    proofs.extend(malicious_tanh_tau_rangecheck_prove(computation_node, op, prover));
                }
                _ => proofs.extend(OperatorProver::prove(computation_node, prover)),
            }
        }
    }
}

fn malicious_tanh_tau_rangecheck_prove(
    node: &ComputationNode,
    op: &Tanh,
    prover: &mut Prover<TestField, TestTranscript>,
) -> Vec<(ProofId, SumcheckInstanceProof<TestField, TestTranscript>)> {
    let mut results = Vec::new();

    let div_params = TeleportDivisionParams::new(node.clone(), &prover.accumulator, op.tau);
    let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);
    let (div_proof, _) = Sumcheck::prove(
        &mut div_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::NeuralTeleport), div_proof));

    let params = TanhParams::new(
        node.clone(),
        &prover.preprocessing.model.graph,
        &prover.accumulator,
        &mut prover.transcript,
        op.clone(),
    );
    let mut exec_sumcheck = TanhProver::initialize(
        &prover.trace,
        params,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let (exec_proof, _) = Sumcheck::prove(
        &mut exec_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

    let layer_data = Trace::layer_data(&prover.trace, node);
    let [input] = layer_data.operands[..] else {
        panic!("Expected one operand for tanh");
    };
    let (quotient, _remainder) = compute_division(input, op.tau);
    let lookup_indices = quotient
        .par_iter()
        .map(|&x| crate::onnx_proof::neural_teleport::n_bits_to_usize(x, op.log_table))
        .collect::<Vec<usize>>();

    let rangecheck_provider = RangeCheckProvider::<TauBypassTeleportRangeCheckOperands>::new(node);
    let (rangecheck_sumcheck, rc_lookup_indices) = rangecheck_provider
        .read_raf_prove::<TestField, TestTranscript, UnsignedLessThanTable<32>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let tanh_encoding = TanhRaEncoding {
        node_idx: node.idx,
        log_table: op.log_table,
    };
    let ra_onehot_provers = shout::ra_onehot_provers(
        &tanh_encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![Box::new(rangecheck_sumcheck)];
    instances.extend(ra_onehot_provers);
    let (tanh_ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        tanh_ra_one_hot_proof,
    ));

    let rc_encoding = RangeCheckEncoding::<TauBypassTeleportRangeCheckOperands>::new(node);
    let [rc_ra, rc_hw, rc_bool] = shout::ra_onehot_provers(
        &rc_encoding,
        &rc_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![rc_ra, rc_hw, rc_bool];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::RaHammingWeight), ra_one_hot_proof));

    results
}

fn build_tau_bypass_rangecheck_rad_witness(
    model: &Model,
    trace: &Trace,
    node_idx: usize,
    d: usize,
) -> MultilinearPolynomial<TestField> {
    let node = &model.graph.nodes[&node_idx];
    let (left, right) = TauBypassTeleportRangeCheckOperands::get_operands_tensors(trace, node);
    let lookup_indices = compute_lookup_indices_from_operands(&[&left, &right], true);
    let one_hot_params = joltworks::config::OneHotParams::new(lookup_indices.len().log_2());
    let addresses: Vec<_> = lookup_indices
        .par_iter()
        .map(|lookup_index| Some(one_hot_params.lookup_index_chunk((*lookup_index).into(), d) as u16))
        .collect();
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        addresses,
        one_hot_params.k_chunk,
    ))
}

fn tau_override(node: &ComputationNode) -> i32 {
    match &node.operator {
        Operator::Tanh(op) => op.tau + 1,
        _ => panic!("tau override only implemented for tanh in this PoC"),
    }
}

fn tanh_model() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![1 << 8]);
    let out = b.tanh(i);
    b.mark_output(out);
    b.build()
}

fn find_tanh_node(model: &Model) -> ComputationNode {
    model
        .graph
        .nodes
        .values()
        .find(|node| matches!(node.operator, Operator::Tanh(_)))
        .cloned()
        .expect("tanh node should exist")
}

#[test]
#[ignore = "Known soundness issue: verifier does not independently bind teleport range-check tau"]
fn soundness_tanh_tau_rangecheck_bypass_is_rejected() {
    let model = tanh_model();
    let tanh_node = find_tanh_node(&model);
    let input_idx = tanh_node.inputs[0];
    let input_data = vec![5; 1 << 8];
    let input = Tensor::new(Some(&input_data), &[1 << 8]).expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let verifier_pp = AtlasVerifierPreprocessing::<TestField, TestPCS>::from(&prover_pp);

    let (honest_proof, _honest_io, _honest_debug) =
        ONNXProof::<TestField, TestTranscript, TestPCS>::prove(&prover_pp, std::slice::from_ref(&input));
    let (malicious_proof, io, _debug) =
        TauRangecheckBypassProof::prove(&prover_pp, std::slice::from_ref(&input));

    let key = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(input_idx),
        SumcheckId::Raf,
    );
    let honest_bound_claim = honest_proof
        .opening_claims
        .0
        .get(&key)
        .expect("honest proof should cache range-check bound claim")
        .1;
    let malicious_bound_claim = malicious_proof
        .opening_claims
        .0
        .get(&key)
        .expect("malicious proof should cache forged range-check bound claim")
        .1;

    assert_ne!(
        honest_bound_claim, malicious_bound_claim,
        "PoC precondition failed: malicious prover must change the RAF bound claim"
    );

    let res = malicious_proof.verify(&verifier_pp, &io, None);
    assert!(
        res.is_err(),
        "malicious tau-bypass proof still verifies: {res:?}"
    );
}
