use crate::{
    onnx_proof::{
        malicious_prover::MaliciousONNXProof,
        neural_teleport::{
            division::{compute_division, TeleportDivisionParams, TeleportDivisionProver},
            eval_shift::{EvalShiftParams, EvalShiftProver},
        },
        ops::{
            eval_reduction::NodeEvalReduction,
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
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{
        consts::FOUR_PI_APPROX,
        test::ModelBuilder,
        trace::{ModelExecutionIO, Trace},
        Model,
    },
    node::ComputationNode,
    ops::{Operator, Tanh},
    tensor::Tensor,
};
use common::{consts::LOG_K, CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
        multilinear_polynomial::MultilinearPolynomial,
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    },
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        shout,
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::math::Math,
};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn sub_model(rng: &mut StdRng, t: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(Tensor::random_small(rng, &[t]));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

fn sub_model_const_2() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![1]);
    let c = b.constant(Tensor::new(Some(&[2]), &[1]).expect("constant tensor should be valid"));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

fn find_sub_node(model: &Model) -> ComputationNode {
    model
        .graph
        .nodes
        .values()
        .find(|node| matches!(node.operator, Operator::Sub(_)))
        .cloned()
        .expect("sub node should exist")
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

fn fanout_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    let c1 = b.constant(Tensor::random_small(rng, &[t]));
    let c2 = b.constant(Tensor::random_small(rng, &[t]));

    // shared producer
    let x = b.sub(i, c0);
    // two consumers of x
    let y = b.sub(x, c1);
    let z = b.sub(x, c2);
    let o = b.add(y, z);
    b.mark_output(o);
    (b.build(), x, y, z)
}

fn duplicate_operand_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    // shared producer
    let x = b.sub(i, c0);
    // same consumer, same producer twice
    let y = b.sub(x, x);
    b.mark_output(y);
    (b.build(), x, y)
}

#[should_panic = "InvalidOpeningProof(\"Sub output claim should equal the difference of operand claims\")"]
#[test]
fn soundness_sub_virtual_operand_attack_is_rejected() {
    // This test demonstrates the virtual-operand-claim attack shape:
    // 1) malicious_sub forges the left operand opening (off by one) at the node's
    //    reduced output point, leaving the right operand honest.
    // 2) Sub is a no-sumcheck op, so the verifier checks `left - right == output`
    //    directly at that point and rejects the forged opening at the Sub node.
    let t = 1 << 12;
    let mut rng = StdRng::seed_from_u64(0xA77ACCEE);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let model = sub_model(&mut rng, t);
    let sub_node = find_sub_node(&model);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) =
        MaliciousONNXProof::prove::<Fr, Blake2bTranscript, HyperKZG<Bn254>>(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Confirm the malicious prover stored forged NodeOutput claims in opening_claims.
    for &input_idx in sub_node.inputs.iter() {
        let key = OpeningId::new(
            VirtualPoly::NodeOutput(input_idx),
            SumcheckId::NodeExecution(sub_node.idx),
        );
        assert!(
            proof.opening_claims.0.contains_key(&key),
            "sub node forged NodeOutput({input_idx}) claim should exist in opening_claims"
        );
    }

    // The verifier's direct `left - right == output` check rejects the forged
    // opening at the Sub node.
    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[should_panic = "InvalidOpeningProof(\"Sub output claim should equal the difference of operand claims\")"]
#[test]
fn soundness_sub_trace_tamper_3_minus_2_becomes_0_is_rejected() {
    let model = sub_model_const_2();
    let input = Tensor::new(Some(&[3]), &[1]).expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) = MaliciousONNXProof::prove_with_sub_trace_tamper_zero::<
        Fr,
        Blake2bTranscript,
        HyperKZG<Bn254>,
    >(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Tampered trace output is forced to zero (even though 3 - 2 should be 1).
    assert_eq!(io.outputs[0].data()[0], 0);
    // Verifier now correctly rejects the tampered proof.
    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[test]
fn soundness_fanout_nodeoutput_openings_should_be_reduced() {
    // #138 structural issue: one producer (x) consumed by two nodes (y, z)
    // produces two per-consumer openings for NodeOutput(x), keyed by
    // NodeExecution(y) and NodeExecution(z). These should be reduced to a
    // single opening via PAZK 4.5.2, but currently are not — only one gets
    // transitively verified against the committed polynomial.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0x138138);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, _y_idx, _z_idx) = fanout_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);

    let eval_reduction_proofs = proof.eval_reduction_proofs;

    // Desired property (#138): all per-consumer openings for NodeOutput(x)
    // should be reduced to a single opening via PAZK 4.5.2.
    // This assertion FAILS today (entries.len() == 2) because reduction is not implemented.
    assert!(
        eval_reduction_proofs.contains_key(&x_idx),
        "NodeOutput({x_idx}) should have an evaluation reduction proof, but none found"
    );
}

#[test]
fn soundness_same_consumer_duplicate_operand_should_track_both() {
    // y = sub(x, x): both operands write NodeOutput(x) + NodeExecution(y).
    // Both operand openings are at the same opening point and with the same
    // claimed value, so they can be represented as a single opening entry.
    // This test asserts that this deduped representation still verifies.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0xD0011CAA);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, y_idx) = duplicate_operand_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let y_node = &verifier_pp.model().graph.nodes[&y_idx];
    assert_eq!(
        y_node.inputs[0], y_node.inputs[1],
        "test precondition: y must consume x twice"
    );

    // Count entries for NodeOutput(x) in opening_claims.
    let lo = OpeningId::new(VirtualPoly::NodeOutput(x_idx), SumcheckId::NodeExecution(0));
    let hi = OpeningId::new(
        VirtualPoly::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

    // Duplicate-operand openings are expected to collapse into one entry.
    assert_eq!(
        entries.len(),
        1,
        "duplicate operand openings should collapse into one entry, but found {}",
        entries.len()
    );

    // The proving flow should still complete and verify successfully.
    proof.verify(&verifier_pp, &_io, None).unwrap();
}

#[test]
fn soundness_tanh_tau_rangecheck_bypass_is_rejected() {
    type ProofOutput = (
        ONNXProof<TestField, TestTranscript, TestPCS>,
        ModelExecutionIO,
        Option<ProverDebugInfo<TestField, TestTranscript>>,
    );

    struct TauRangecheckBypassProof;

    struct TauBypassTeleportRangeCheckOperands {
        base: RangeCheckOperandsBase,
    }

    type TestPCS = HyperKZG<Bn254>;
    type TestField = Fr;
    type TestTranscript = Blake2bTranscript;

    impl RangeCheckingOperandsTrait for TauBypassTeleportRangeCheckOperands {
        fn new(node: &ComputationNode) -> Self {
            Self {
                base: RangeCheckOperandsBase {
                    node_idx: node.idx,
                    remainder: VirtualPoly::TeleportRemainder(node.idx),
                    bound: None,
                    virtual_ra: VirtualPoly::TeleportRangeCheckRa(node.idx),
                    operator: node.operator.clone(),
                },
            }
        }

        fn base(&self) -> &RangeCheckOperandsBase {
            &self.base
        }

        fn get_operands_tensors(
            trace: &Trace,
            node: &ComputationNode,
        ) -> (Tensor<i32>, Tensor<i32>) {
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

        fn rad_poly(&self, d: usize) -> CommittedPoly {
            CommittedPoly::TeleportRangeCheckRaD(self.base.node_idx, d)
        }

        /// Extract the operand claims from the accumulator for the left and right operands.
        fn operand_claims<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> (F, F) {
            let operand_claims = self
                .get_input_operands()
                .iter()
                .map(|operand| {
                    let operand_id =
                        OpeningId::new(*operand, SumcheckId::NodeExecution(self.base.node_idx));
                    let (_, claim) = accumulator.get_virtual_polynomial_opening(operand_id);
                    claim
                })
                .collect::<Vec<_>>();
            let tau = match &self.base().operator {
                Operator::Tanh(inner) => inner.tau,
                Operator::Erf(inner) => inner.tau,
                Operator::Sigmoid(inner) => inner.tau,
                Operator::Cos(_) | Operator::Sin(_) => FOUR_PI_APPROX,
                _ => {
                    panic!(
                    "Expected Tanh, Erf, Sigmoid, Cos, or Sin operator for neural teleportation division"
                )
                }
            };
            (
                operand_claims[0],
                self.transform_right_claim(F::from_i32(tau)),
            )
        }
    }

    impl TauRangecheckBypassProof {
        fn prove(
            pp: &AtlasProverPreprocessing<TestField, TestPCS>,
            inputs: &[Tensor<i32>],
        ) -> ProofOutput {
            let trace = pp.model().trace(inputs);
            let io = Trace::io(&trace, pp.model());

            let mut prover: Prover<TestField, TestTranscript> =
                Prover::new(pp.shared.clone(), trace);
            let mut proofs = BTreeMap::new();
            let mut er_proofs = BTreeMap::new();

            let (poly_map, commitments) = Self::commit_witness_polynomials(
                pp.model(),
                &prover.trace,
                &pp.generators,
                &mut prover.transcript,
            );
            ONNXProof::<TestField, TestTranscript, TestPCS>::output_claim(&mut prover);
            Self::iop(pp.model().nodes(), &mut prover, &mut proofs, &mut er_proofs);
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
                er_proofs,
                reduced_opening_proof,
            )
        }

        fn commit_witness_polynomials(
            model: &Model,
            trace: &Trace,
            generators: &<TestPCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::ProverSetup,
            transcript: &mut TestTranscript,
        ) -> (
            BTreeMap<CommittedPoly, MultilinearPolynomial<TestField>>,
            Vec<<TestPCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment>,
        ){
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
        ) -> BTreeMap<CommittedPoly, MultilinearPolynomial<TestField>> {
            let tanh_node = find_tanh_node(model);
            model
            .graph
            .nodes
            .values()
            .flat_map(|node| {
                NodeCommittedPolynomials::get_committed_polynomials::<TestField, TestTranscript>(
                    node,
                )
            })
            .map(|poly| {
                let witness = match poly {
                    CommittedPoly::TeleportRangeCheckRaD(node_idx, d)
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
            eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<TestField>>,
        ) {
            for (_, computation_node) in computation_nodes.iter().rev() {
                match &computation_node.operator {
                    Operator::Tanh(op) => {
                        let eval_reduction_proof =
                            NodeEvalReduction::prove(prover, computation_node);
                        eval_reduction_proofs.insert(computation_node.idx, eval_reduction_proof);
                        proofs.extend(malicious_tanh_tau_rangecheck_prove(
                            computation_node,
                            op,
                            prover,
                        ));
                    }
                    _ => {
                        let (eval_reduction_proof, execution_proofs) =
                            OperatorProver::prove(computation_node, prover);
                        eval_reduction_proofs.insert(computation_node.idx, eval_reduction_proof);
                        proofs.extend(execution_proofs);
                    }
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

        let eval_shift_params = EvalShiftParams::new(node.clone(), &prover.accumulator);
        let mut eval_shift_sumcheck = EvalShiftProver::initialize(&prover.trace, eval_shift_params);

        let (div_proof, _) = BatchedSumcheck::prove(
            vec![&mut div_sumcheck as _, &mut eval_shift_sumcheck as _],
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
        let mut exec_sumcheck = TanhProver::initialize(&prover.trace, params);
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

        let rangecheck_provider =
            RangeCheckProvider::<TauBypassTeleportRangeCheckOperands>::new(node);
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
        results.push((
            ProofId(node.idx, ProofType::RaHammingWeight),
            ra_one_hot_proof,
        ));

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
        let one_hot_params =
            joltworks::config::OneHotParams::new(lookup_indices.len().log_2(), LOG_K);
        let addresses: Vec<_> = lookup_indices
            .par_iter()
            .map(|lookup_index| {
                Some(one_hot_params.lookup_index_chunk((*lookup_index).into(), d) as u16)
            })
            .collect();
        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
            addresses,
            one_hot_params.k_chunk,
        ))
    }

    fn tau_override(node: &ComputationNode) -> i32 {
        match &node.operator {
            // The PoC forges the teleport range-check bound by using tau + 1 instead of
            // the operator's real tau. Only the range-check path uses this override.
            Operator::Tanh(op) => op.tau + 1,
            _ => panic!("tau override only implemented for tanh in this PoC"),
        }
    }

    // This regression test builds a malicious prover that only changes the
    // teleport range-check path: inside get_operands_tensors, tau_override
    // returns tau + 1, so the prover effectively proves the bound with a forged
    // tau. The proof is then checked by the normal verifier implementation.
    //
    // If soundness is correct, the honest verifier must reject this proof.
    let model = tanh_model();
    let input_data = vec![5; 1 << 8];
    let input = Tensor::new(Some(&input_data), &[1 << 8]).expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let verifier_pp = AtlasVerifierPreprocessing::<TestField, TestPCS>::from(&prover_pp);

    let (malicious_proof, io, _debug) =
        TauRangecheckBypassProof::prove(&prover_pp, std::slice::from_ref(&input));

    let res = malicious_proof.verify(&verifier_pp, &io, None);
    assert!(
        res.is_err(),
        "malicious tau-bypass proof still verifies: {res:?}"
    );
}
