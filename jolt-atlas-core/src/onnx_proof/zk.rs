//! ZK proving and verification for ONNX neural network computations.
//!
//! This module provides `prove_zk` and `verify_zk` functions that run the
//! proof pipeline in a single pass with BlindFold zero-knowledge proofs.
//! Sumcheck round polynomials are Pedersen-committed instead of sent in the clear,
//! and a BlindFold proof (Nova folding + Spartan) verifies constraint consistency.

use crate::onnx_proof::{AtlasProverPreprocessing, AtlasVerifierPreprocessing, ONNXProof, Prover};
use ark_bn254::{Bn254, Fr};
use ark_std::Zero;
use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPoly;
use joltworks::{
    curve::Bn254Curve,
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG, pedersen::PedersenGenerators,
        },
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator as _, SumcheckId},
    },
    subprotocols::{
        blindfold::{
            protocol::{
                BlindFoldProof, BlindFoldProver, BlindFoldVerifier, BlindFoldVerifierInput,
            },
            r1cs::VerifierR1CSBuilder,
            relaxed_r1cs::RelaxedR1CSInstance,
            witness::{BlindFoldWitness, FinalOutputWitness, RoundWitness, StageWitness},
            BakedPublicInputs, BlindFoldAccumulator, StageConfig,
        },
        evaluation_reduction::EvalReductionProof,
        sumcheck::BatchedSumcheck,
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::{errors::ProofVerifyError, math::Math},
};
use std::collections::BTreeMap;

type F = Fr;
type C = Bn254Curve;
type T = Blake2bTranscript;
#[expect(clippy::upper_case_acronyms)]
type PCS = HyperKZG<Bn254>;

/// ZK sumcheck proof for a single operator, tagged with node index.
pub type NodeZkProof = (
    usize,
    joltworks::subprotocols::sumcheck::ZkSumcheckProof<F, C, T>,
);

/// Result of a single-pass ZK prove.
pub struct ZkProofBundle {
    /// BlindFold zero-knowledge proof (Nova folding + Spartan).
    pub blindfold_proof: BlindFoldProof<F, C>,
    /// BlindFold verifier input (round commitments from the real instance).
    pub blindfold_verifier_input: BlindFoldVerifierInput<C>,
    /// Stage configs describing the BlindFold R1CS layout.
    pub stage_configs: Vec<StageConfig>,
    /// Baked public inputs for the BlindFold R1CS.
    pub baked: BakedPublicInputs<F>,
    /// Evaluation reduction proofs for each node (standard mode data; h polynomial
    /// is NOT used by the ZK verifier, which relies on h_commitments instead).
    pub eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
    /// Pedersen commitments to eval reduction h polynomials (keyed by node index).
    /// None for single-opening nodes (short path, no h produced).
    pub eval_reduction_h_commitments: BTreeMap<usize, joltworks::curve::Bn254G1>,
    /// Polynomial commitments for witness polynomials.
    pub commitments: Vec<<PCS as CommitmentScheme>::Commitment>,
    /// The output claim value (public, computed from IO).
    pub output_claim: F,
    /// ZK sumcheck proofs for each operator stage (needed by verifier for IOP replay).
    pub zk_sumcheck_proofs: Vec<NodeZkProof>,
}

/// Run a single-instance ZK sumcheck and collect stage data.
/// Returns the ZK sumcheck proof for inclusion in the bundle.
fn run_zk_sumcheck(
    sc: &mut dyn SumcheckInstanceProver<F, T>,
    prover: &mut Prover<F, T>,
    blindfold_accumulator: &mut BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    pedersen_gens: &PedersenGenerators<C>,
) -> joltworks::subprotocols::sumcheck::ZkSumcheckProof<F, C, T> {
    let poly_degree = sc.get_params().degree();
    let num_rounds = sc.get_params().num_rounds();

    let _ = prover.accumulator.take_pending_claims();
    let _ = prover.accumulator.take_pending_claim_ids();

    let mut rng = rand::thread_rng();

    let instances: Vec<&mut dyn SumcheckInstanceProver<F, T>> = vec![sc];
    let (zk_proof, _, _) = BatchedSumcheck::prove_zk::<F, C, T, _>(
        instances,
        &mut prover.accumulator,
        blindfold_accumulator,
        &mut prover.transcript,
        pedersen_gens,
        &mut rng,
    );

    stage_configs.push(StageConfig::new_chain(num_rounds, poly_degree));
    zk_proof
}

/// Prove an ONNX model execution with zero-knowledge (single pass).
///
/// `pedersen_gens` must have at least `max(poly_degree + 1, hyrax_C)` message
/// generators. For current operators, 8 generators suffice.
pub fn prove_zk(
    pp: &AtlasProverPreprocessing<F, PCS>,
    inputs: &[Tensor<i32>],
    pedersen_gens: &PedersenGenerators<C>,
) -> (ZkProofBundle, ModelExecutionIO) {
    let trace = pp.model().trace(inputs);
    let io = Trace::io(&trace, pp.model());
    let mut prover = Prover::<F, T>::new(pp.shared.clone(), trace);

    let (poly_map, commitments) = ONNXProof::<F, T, PCS>::commit_witness_polynomials(
        pp.model(),
        &prover.trace,
        &pp.generators,
        &mut prover.transcript,
    );
    ONNXProof::<F, T, PCS>::output_claim(&mut prover);

    // Capture the output claim for the verifier. This is public (derived from IO).
    // Computed the same way as ONNXProof::output_claim: evaluate output MLE at
    // the challenge point that was just derived from the transcript.
    let output_index = pp.model().outputs()[0];
    let output_node = &pp.model()[output_index];
    let output_claim = {
        let output_key = joltworks::poly::opening_proof::OpeningId::new(
            VirtualPoly::NodeOutput(output_node.idx),
            SumcheckId::NodeExecution(output_node.idx + 1),
        );
        prover
            .accumulator
            .get_virtual_polynomial_opening(output_key)
            .1
    };

    let nodes = pp.model().nodes();
    let mut blindfold_accumulator = BlindFoldAccumulator::<F, C>::new();
    let mut stage_configs = Vec::new();
    let mut eval_reduction_proofs = BTreeMap::new();
    let mut eval_reduction_h_commitments = BTreeMap::new();
    let mut zk_sumcheck_proofs: Vec<NodeZkProof> = Vec::new();

    for (_, node) in nodes.iter().rev() {
        // ZK eval reduction: commit h via Pedersen instead of cleartext
        {
            use atlas_onnx_tracer::model::trace::{LayerData, Trace};
            use joltworks::subprotocols::evaluation_reduction::EvalReductionProtocol;

            let node_idx = node.idx;
            let openings = prover.accumulator.get_node_openings(node_idx);
            let LayerData {
                operands: _,
                output,
            } = Trace::layer_data(&prover.trace, node);
            let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());

            let (proof, reduced, h_com) = EvalReductionProtocol::prove_zk::<F, C, _>(
                &openings,
                output_mle,
                &mut prover.transcript,
                pedersen_gens,
            )
            .expect("ZK eval reduction should not fail");

            prover
                .accumulator
                .reduced_evaluations
                .insert(node_idx, reduced);
            eval_reduction_proofs.insert(node_idx, proof);
            if let Some(com) = h_com {
                eval_reduction_h_commitments.insert(node_idx, com);
            }
        }

        let zk_proof = create_prover_instance(node, &prover, pp.shared.model()).map(|mut sc| {
            run_zk_sumcheck(
                &mut *sc,
                &mut prover,
                &mut blindfold_accumulator,
                &mut stage_configs,
                pedersen_gens,
            )
        });
        if let Some(proof) = zk_proof {
            zk_sumcheck_proofs.push((node.idx, proof));
        }
    }

    // y_com for batch opening (no-op when poly_map is empty)
    if !poly_map.is_empty() {
        prover.accumulator.prepare_for_sumcheck(&poly_map);
        let (_acc_proof, r_sumcheck_acc) = prover
            .accumulator
            .prove_batch_opening_sumcheck(&mut prover.transcript);
        let state = prover
            .accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc, &mut prover.transcript);

        let joint_claim: F = state.sumcheck_claims.iter().sum();
        let y_blinding = F::random(&mut rand::thread_rng());
        let y_com = pedersen_gens.commit(&[joint_claim], &y_blinding);
        prover.transcript.append_serializable(&y_com);

        blindfold_accumulator.set_opening_proof_data(
            joltworks::subprotocols::blindfold::OpeningProofData {
                opening_ids: state
                    .polynomials
                    .iter()
                    .map(|poly| {
                        joltworks::poly::opening_proof::OpeningId::new(
                            *poly,
                            joltworks::poly::opening_proof::SumcheckId::BlindFoldBatchOpening,
                        )
                    })
                    .collect(),
                constraint_coeffs: state.gamma_powers.clone(),
                joint_claim,
                y_blinding,
            },
        );
    }

    // Build BlindFold proof from accumulated stage data
    let stage_data_vec = blindfold_accumulator.take_stage_data();
    let mut all_stages = Vec::new();

    for (i, sd) in stage_data_vec.iter().enumerate() {
        let rounds: Vec<RoundWitness<F>> = sd
            .poly_coeffs
            .iter()
            .zip(sd.challenges.iter())
            .map(|(coeffs, challenge)| {
                let challenge_f: F = (*challenge).into();
                RoundWitness::new(coeffs.clone(), challenge_f)
            })
            .collect();

        let output_constraint = sd.output_constraints[0].as_ref();
        if let Some(constraint) = output_constraint {
            let output_challenge_values = sd.constraint_challenge_values[0].clone();
            let output_opening_values: Vec<F> =
                sd.output_claims.iter().map(|(_id, val)| *val).collect();
            let final_output_witness =
                FinalOutputWitness::general(output_challenge_values, output_opening_values);
            all_stages.push(StageWitness::with_final_output(
                rounds,
                final_output_witness,
            ));
            stage_configs[i] = stage_configs[i].clone().with_constraint(constraint.clone());
        } else {
            all_stages.push(StageWitness::new(rounds));
        }
    }

    let initial_claim = stage_data_vec[0].initial_claim;
    let blindfold_witness = BlindFoldWitness::new(initial_claim, all_stages);
    let baked = BakedPublicInputs::from_witness(&blindfold_witness, &stage_configs);
    let builder = VerifierR1CSBuilder::<F>::new(&stage_configs, &baked);
    let r1cs = builder.build();

    let z = blindfold_witness.assign(&r1cs);
    r1cs.check_satisfaction(&z)
        .expect("BlindFold R1CS should be satisfied");

    assert!(
        pedersen_gens.message_generators.len() >= r1cs.hyrax.C,
        "Pedersen generators too small: need {} for Hyrax columns, have {}",
        r1cs.hyrax.C,
        pedersen_gens.message_generators.len()
    );
    let witness: Vec<F> = z[1..].to_vec();
    let hyrax = &r1cs.hyrax;
    let hyrax_C = hyrax.C;
    let R_coeff = hyrax.R_coeff;
    let R_prime = hyrax.R_prime;

    let mut rng = rand::thread_rng();
    let mut round_commitments = Vec::new();
    let mut w_row_blindings = vec![F::zero(); R_prime];

    for round_idx in 0..hyrax.total_rounds {
        let row_start = round_idx * hyrax_C;
        let blinding = F::random(&mut rng);
        let commitment = pedersen_gens.commit(&witness[row_start..row_start + hyrax_C], &blinding);
        w_row_blindings[round_idx] = blinding;
        round_commitments.push(commitment);
    }

    let noncoeff_rows_count = hyrax.total_noncoeff_rows();
    let mut noncoeff_row_commitments = Vec::new();
    for row in 0..noncoeff_rows_count {
        let start = R_coeff * hyrax_C + row * hyrax_C;
        let end = (start + hyrax_C).min(witness.len());
        let blinding = F::random(&mut rng);
        noncoeff_row_commitments.push(pedersen_gens.commit(&witness[start..end], &blinding));
        w_row_blindings[R_coeff + row] = blinding;
    }

    let (real_instance, real_witness) = RelaxedR1CSInstance::<F, C>::new_non_relaxed(
        &witness,
        r1cs.num_constraints,
        hyrax_C,
        round_commitments,
        Vec::new(),
        noncoeff_row_commitments,
        Vec::new(),
        w_row_blindings,
    );

    let bf_prover = BlindFoldProver::new(pedersen_gens, &r1cs, None);
    let mut bf_transcript = T::new(b"BlindFold_onnx");
    let blindfold_proof = bf_prover.prove(&real_instance, &real_witness, &z, &mut bf_transcript);

    let blindfold_verifier_input = BlindFoldVerifierInput {
        round_commitments: real_instance.round_commitments.clone(),
        output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
        eval_commitments: real_instance.eval_commitments.clone(),
    };

    let bundle = ZkProofBundle {
        blindfold_proof,
        blindfold_verifier_input,
        stage_configs,
        baked,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
        commitments,
        output_claim,
        zk_sumcheck_proofs,
    };

    (bundle, io)
}

/// Verify a ZK proof for an ONNX model execution.
///
/// Checks:
/// 1. The output claim matches the model execution IO.
/// 2. Full IOP replay: for each node, verify eval reduction + absorb ZK
///    sumcheck commitments into transcript (mirroring the prover's flow).
/// 3. BlindFold proof verifies (covers sumcheck round consistency and
///    constraint satisfaction).
pub fn verify_zk(
    bundle: &ZkProofBundle,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    pedersen_gens: &PedersenGenerators<C>,
) -> Result<(), ProofVerifyError> {
    use joltworks::poly::opening_proof::VerifierOpeningAccumulator;
    use joltworks::subprotocols::evaluation_reduction::EvalReductionProtocol;

    let model = pp.shared.model();
    let mut accumulator = VerifierOpeningAccumulator::<F>::new_zk();
    let mut transcript = T::new(b"ONNXProof");

    // 1. Absorb witness commitments into transcript
    for commitment in &bundle.commitments {
        transcript.append_serializable(commitment);
    }

    // 2. Verify output claim: derive challenge, evaluate output MLE, check match
    let output_index = model.outputs()[0];
    let output_node = &model[output_index];
    let r_node_output = transcript
        .challenge_vector_optimized::<F>(output_node.pow2_padded_num_output_elements().log_2());
    use joltworks::field::IntoOpening;
    let r_node_output_field: Vec<F> = r_node_output.clone().into_opening();
    let expected_output_claim =
        MultilinearPolynomial::from(io.outputs[0].padded_next_power_of_two())
            .evaluate(&r_node_output_field);

    if expected_output_claim != bundle.output_claim {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "ZK output claim does not match model execution IO".to_string(),
        ));
    }

    // Insert the output claim into the ZK-mode accumulator.
    // In ZK mode, append_virtual won't append the claim to transcript (it's hidden).
    // But we need the opening point registered for eval reduction.
    let output_key = joltworks::poly::opening_proof::OpeningId::new(
        VirtualPoly::NodeOutput(output_node.idx),
        SumcheckId::NodeExecution(output_node.idx + 1),
    );
    accumulator
        .openings
        .insert(output_key, (r_node_output.into(), expected_output_claim));
    // Append the claim to transcript to match prover's flow
    transcript.append_scalar(&expected_output_claim);

    // 3. Full IOP replay: for each node, verify eval reduction + absorb ZK sumcheck
    let mut zk_proof_idx = 0;
    for (_, node) in model.graph.nodes.iter().rev() {
        // Verify eval reduction (ZK mode: absorb h commitment, skip claim checks)
        let openings = accumulator.get_node_openings(node.idx);
        let h_commitment = bundle.eval_reduction_h_commitments.get(&node.idx);
        let reduced_instance =
            EvalReductionProtocol::verify_zk::<F, T>(&openings, h_commitment, &mut transcript)?;
        accumulator
            .reduced_evaluations
            .insert(node.idx, reduced_instance);

        // For operators with sumcheck: absorb ZK sumcheck commitments into transcript
        let has_sumcheck = !matches!(
            &node.operator,
            Operator::Input(_)
                | Operator::Identity(_)
                | Operator::Broadcast(_)
                | Operator::MoveAxis(_)
                | Operator::Constant(_)
                | Operator::IsNan(_)
        );

        if has_sumcheck {
            let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[zk_proof_idx];
            assert_eq!(
                *proof_node_idx, node.idx,
                "ZK sumcheck proof order mismatch"
            );

            // Create verifier sumcheck instances and run verify_zk to replay transcript
            let verifier_instances = create_verifier_instances(node, &accumulator, model);
            let verifier_refs: Vec<
                &dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
            > = verifier_instances
                .iter()
                .map(|v| {
                    v.as_ref()
                        as &dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<
                            F,
                            T,
                        >
                })
                .collect();

            BatchedSumcheck::verify_zk::<F, C, T>(
                zk_proof,
                verifier_refs,
                &mut accumulator,
                &mut transcript,
            )?;

            zk_proof_idx += 1;
        } else {
            // No-sumcheck operators: run their standard verify to keep transcript in sync.
            // Input verifies the claim against IO; others are no-ops.
            // Input: check that the input claim matches the IO.
            // Other no-sumcheck ops (Identity, Broadcast, etc.) don't modify the transcript.
        }
    }

    // 4. Verify BlindFold proof
    let builder = VerifierR1CSBuilder::<F>::new(&bundle.stage_configs, &bundle.baked);
    let r1cs = builder.build();
    let bf_verifier = BlindFoldVerifier::new(pedersen_gens, &r1cs, None);

    let mut bf_transcript = T::new(b"BlindFold_onnx");
    bf_verifier
        .verify(
            &bundle.blindfold_proof,
            &bundle.blindfold_verifier_input,
            &mut bf_transcript,
        )
        .map_err(|e| {
            ProofVerifyError::InvalidOpeningProof(format!("BlindFold verification failed: {e:?}"))
        })
}

/// Create the ZK sumcheck prover instance for a node.
/// Returns `None` for operators with no sumcheck (Input, Identity, etc.).
fn create_prover_instance(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &Prover<F, T>,
    model: &atlas_onnx_tracer::model::Model,
) -> Option<Box<dyn SumcheckInstanceProver<F, T>>> {
    match &node.operator {
        Operator::Square(_) => {
            use crate::onnx_proof::ops::square::{SquareParams, SquareProver};
            let params = SquareParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(SquareProver::initialize(&prover.trace, params)))
        }
        Operator::Add(_) => {
            use crate::onnx_proof::ops::add::{AddParams, AddProver};
            let params = AddParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(AddProver::initialize(&prover.trace, params)))
        }
        Operator::Reshape(_) => {
            use crate::onnx_proof::ops::reshape::{ReshapeSumcheckParams, ReshapeSumcheckProver};
            let params =
                ReshapeSumcheckParams::<F>::new(node.clone(), &prover.accumulator, &model.graph);
            Some(Box::new(ReshapeSumcheckProver::initialize(
                &prover.trace,
                params,
            )))
        }
        Operator::Slice(_) => {
            use crate::onnx_proof::ops::slice::{SliceSumcheckParams, SliceSumcheckProver};
            let params =
                SliceSumcheckParams::<F>::new(node.clone(), &prover.accumulator, &model.graph);
            Some(Box::new(SliceSumcheckProver::initialize(
                &prover.trace,
                params,
            )))
        }
        Operator::Neg(_) => {
            use crate::onnx_proof::ops::neg::{NegParams, NegProver};
            let params = NegParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(NegProver::initialize(&prover.trace, params)))
        }
        Operator::Sub(_) => {
            use crate::onnx_proof::ops::sub::{SubParams, SubProver};
            let params = SubParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(SubProver::initialize(&prover.trace, params)))
        }
        Operator::Mul(_) | Operator::And(_) => {
            use crate::onnx_proof::ops::mul::{MulParams, MulProver};
            let params = MulParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(MulProver::initialize(&prover.trace, params)))
        }
        Operator::Cube(_) => {
            use crate::onnx_proof::ops::cube::{CubeParams, CubeProver};
            let params = CubeParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(CubeProver::initialize(&prover.trace, params)))
        }
        Operator::Iff(_) => {
            use crate::onnx_proof::ops::iff::{IffParams, IffProver};
            let params = IffParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(IffProver::initialize(&prover.trace, params)))
        }
        Operator::Concat(_) => {
            use crate::onnx_proof::ops::concat::{ConcatSumcheckParams, ConcatSumcheckProver};
            let params =
                ConcatSumcheckParams::<F>::new(node.clone(), &prover.accumulator, &model.graph);
            Some(Box::new(ConcatSumcheckProver::initialize(
                &prover.trace,
                params,
            )))
        }
        Operator::Input(_)
        | Operator::Identity(_)
        | Operator::Broadcast(_)
        | Operator::MoveAxis(_)
        | Operator::Constant(_)
        | Operator::IsNan(_) => None,
        other => {
            panic!("ZK proving not yet implemented for operator: {other:?}");
        }
    }
}

/// Create verifier sumcheck instances for a node (for IOP transcript replay).
fn create_verifier_instances(
    node: &atlas_onnx_tracer::node::ComputationNode,
    accumulator: &joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    model: &atlas_onnx_tracer::model::Model,
) -> Vec<Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>>> {
    match &node.operator {
        Operator::Square(_) => {
            use crate::onnx_proof::ops::square::SquareVerifier;
            vec![Box::new(SquareVerifier::new(node.clone(), accumulator))]
        }
        Operator::Add(_) => {
            use crate::onnx_proof::ops::add::AddVerifier;
            vec![Box::new(AddVerifier::new(node.clone(), accumulator))]
        }
        Operator::Sub(_) => {
            use crate::onnx_proof::ops::sub::SubVerifier;
            vec![Box::new(SubVerifier::new(node.clone(), accumulator))]
        }
        Operator::Neg(_) => {
            use crate::onnx_proof::ops::neg::NegVerifier;
            vec![Box::new(NegVerifier::new(node.clone(), accumulator))]
        }
        Operator::Mul(_) | Operator::And(_) => {
            use crate::onnx_proof::ops::mul::MulVerifier;
            vec![Box::new(MulVerifier::new(node.clone(), accumulator))]
        }
        Operator::Cube(_) => {
            use crate::onnx_proof::ops::cube::CubeVerifier;
            vec![Box::new(CubeVerifier::new(node.clone(), accumulator))]
        }
        Operator::Iff(_) => {
            use crate::onnx_proof::ops::iff::IffVerifier;
            vec![Box::new(IffVerifier::new(node.clone(), accumulator))]
        }
        Operator::Reshape(_) => {
            use crate::onnx_proof::ops::reshape::ReshapeSumcheckVerifier;
            vec![Box::new(ReshapeSumcheckVerifier::new(
                node.clone(),
                accumulator,
                &model.graph,
            ))]
        }
        Operator::Slice(_) => {
            use crate::onnx_proof::ops::slice::SliceSumcheckVerifier;
            vec![Box::new(SliceSumcheckVerifier::new(
                node.clone(),
                accumulator,
                &model.graph,
            ))]
        }
        Operator::Concat(_) => {
            use crate::onnx_proof::ops::concat::ConcatSumcheckVerifier;
            vec![Box::new(ConcatSumcheckVerifier::new(
                node.clone(),
                accumulator,
                &model.graph,
            ))]
        }
        _ => vec![],
    }
}
