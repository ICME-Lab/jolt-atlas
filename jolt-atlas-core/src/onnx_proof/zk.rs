//! ZK proving and verification for ONNX neural network computations.
//!
//! This module provides `prove_zk` and `verify_zk` functions that wrap the
//! standard `ONNXProof` pipeline with BlindFold zero-knowledge proofs.
//! Sumcheck round polynomials are Pedersen-committed instead of sent in the clear,
//! and a BlindFold proof (Nova folding + Spartan) verifies constraint consistency.
//!
//! Currently only the `Square` operator is supported. Other operators will panic
//! at prove time until their BlindFold constraints are implemented.

use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProver},
    AtlasProverPreprocessing, AtlasSharedPreprocessing, ONNXProof, Prover,
};
use ark_bn254::{Bn254, Fr};
use ark_std::Zero;
use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use joltworks::{
    curve::Bn254Curve,
    field::{IntoOpening, JoltField},
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG, pedersen::PedersenGenerators,
        },
        opening_proof::ProverOpeningAccumulator,
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
        sumcheck::BatchedSumcheck,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceParams as _,
    },
    transcripts::{Blake2bTranscript, Transcript},
};
use std::collections::BTreeMap;

type F = Fr;
type C = Bn254Curve;
type T = Blake2bTranscript;
type PCS = HyperKZG<Bn254>;

/// Result of a ZK prove: the standard proof plus the BlindFold proof.
pub struct ZkProofBundle {
    /// Standard ONNX proof (sumcheck proofs, commitments, opening proof).
    pub proof: ONNXProof<F, T, PCS>,
    /// BlindFold zero-knowledge proof (Nova folding + Spartan).
    pub blindfold_proof: BlindFoldProof<F, C>,
    /// BlindFold verifier input (round commitments, etc.).
    pub blindfold_verifier_input: BlindFoldVerifierInput<C>,
    /// Pedersen generators used for commitments (needed by verifier).
    pub pedersen_gens: PedersenGenerators<C>,
    /// Stage configs describing the BlindFold R1CS layout.
    pub stage_configs: Vec<StageConfig>,
    /// Baked public inputs for the BlindFold R1CS.
    pub baked: BakedPublicInputs<F>,
}

/// Prove an ONNX model execution with zero-knowledge.
///
/// Runs the standard proof pipeline, then separately runs each operator's
/// sumcheck through `prove_zk` with Pedersen-committed rounds, and produces
/// a BlindFold proof over the accumulated stage data.
///
/// Currently only supports models with a single `Square` operator.
pub fn prove_zk(
    pp: &AtlasProverPreprocessing<F, PCS>,
    inputs: &[Tensor<i32>],
) -> (ZkProofBundle, ModelExecutionIO) {
    // 1. Run standard proof for the non-ZK parts (commitments, eval reduction, opening proof)
    let (standard_proof, io, _debug_info) = ONNXProof::<F, T, PCS>::prove(pp, inputs);

    // 2. Replay the flow with a fresh prover for the ZK sumcheck path.
    //    We need a second pass because the first consumed the prover state.
    let trace = pp.model().trace(inputs);
    let mut prover = Prover::<F, T>::new(pp.shared.clone(), trace);

    // Replay: commit witness polys to keep transcript in sync
    let (poly_map, _commitments) = ONNXProof::<F, T, PCS>::commit_witness_polynomials(
        pp.model(),
        &prover.trace,
        &pp.generators,
        &mut prover.transcript,
    );
    // Replay: output claim
    ONNXProof::<F, T, PCS>::output_claim(&mut prover);

    // 3. For each node, do eval reduction then ZK sumcheck
    let nodes = pp.model().nodes();
    let mut blindfold_accumulator = BlindFoldAccumulator::<F, C>::new();
    let mut max_degree = 0usize;
    let mut stage_configs = Vec::new();

    for (_, node) in nodes.iter().rev() {
        // Eval reduction (same as standard flow)
        NodeEvalReduction::prove(&mut prover, node);

        // Create sumcheck prover based on operator type
        match &node.operator {
            Operator::Square(_) => {
                use crate::onnx_proof::ops::square::{SquareParams, SquareProver};

                let params = SquareParams::<F>::new(node.clone(), &prover.accumulator);
                let poly_degree = params.degree();
                let num_rounds = params.num_rounds();
                max_degree = max_degree.max(poly_degree);

                let mut square_prover = SquareProver::initialize(&prover.trace, params);

                // Drain pre-sumcheck pending claims
                let _ = prover.accumulator.take_pending_claims();
                let _ = prover.accumulator.take_pending_claim_ids();

                // Size generators for max poly degree
                let pedersen_gens = PedersenGenerators::<C>::deterministic(poly_degree + 2);
                let mut rng = rand::thread_rng();

                let instances: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
                    vec![&mut square_prover];
                let _ = BatchedSumcheck::prove_zk::<F, C, T, _>(
                    instances,
                    &mut prover.accumulator,
                    &mut blindfold_accumulator,
                    &mut prover.transcript,
                    &pedersen_gens,
                    &mut rng,
                );

                stage_configs.push(StageConfig::new_chain(num_rounds, poly_degree));
            }
            // Input/Identity/Broadcast/MoveAxis/Constant: no sumcheck, just eval reduction
            Operator::Input(_)
            | Operator::Identity(_)
            | Operator::Broadcast(_)
            | Operator::MoveAxis(_)
            | Operator::Constant(_) => {
                // These operators produce no sumcheck proofs; eval reduction above is sufficient.
            }
            other => {
                panic!("ZK proving not yet implemented for operator: {other:?}");
            }
        }
    }

    // 4. Build BlindFold proof from accumulated stage data
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

    let gens = PedersenGenerators::<C>::deterministic(r1cs.hyrax.C + 1);
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
        let commitment = gens.commit(&witness[row_start..row_start + hyrax_C], &blinding);
        w_row_blindings[round_idx] = blinding;
        round_commitments.push(commitment);
    }

    let noncoeff_rows_count = hyrax.total_noncoeff_rows();
    let mut noncoeff_row_commitments = Vec::new();
    for row in 0..noncoeff_rows_count {
        let start = R_coeff * hyrax_C + row * hyrax_C;
        let end = (start + hyrax_C).min(witness.len());
        let blinding = F::random(&mut rng);
        noncoeff_row_commitments.push(gens.commit(&witness[start..end], &blinding));
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

    let bf_prover = BlindFoldProver::new(&gens, &r1cs, None);
    let mut bf_transcript = T::new(b"BlindFold_onnx");
    let blindfold_proof = bf_prover.prove(&real_instance, &real_witness, &z, &mut bf_transcript);

    let blindfold_verifier_input = BlindFoldVerifierInput {
        round_commitments: real_instance.round_commitments.clone(),
        output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
        eval_commitments: real_instance.eval_commitments.clone(),
    };

    let bundle = ZkProofBundle {
        proof: standard_proof,
        blindfold_proof,
        blindfold_verifier_input,
        pedersen_gens: gens,
        stage_configs,
        baked,
    };

    (bundle, io)
}

/// Verify a ZK proof for an ONNX model execution.
///
/// Verifies both the standard ONNX proof and the BlindFold proof.
pub fn verify_zk(
    bundle: &ZkProofBundle,
    pp: &crate::onnx_proof::AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    debug_info: Option<crate::onnx_proof::types::ProverDebugInfo<F, T>>,
) -> Result<(), joltworks::utils::errors::ProofVerifyError> {
    // 1. Verify standard proof
    bundle.proof.verify(pp, io, debug_info)?;

    // 2. Verify BlindFold proof
    let builder = VerifierR1CSBuilder::<F>::new(&bundle.stage_configs, &bundle.baked);
    let r1cs = builder.build();
    let bf_verifier = BlindFoldVerifier::new(&bundle.pedersen_gens, &r1cs, None);

    let mut bf_transcript = T::new(b"BlindFold_onnx");
    bf_verifier
        .verify(
            &bundle.blindfold_proof,
            &bundle.blindfold_verifier_input,
            &mut bf_transcript,
        )
        .map_err(|e| {
            joltworks::utils::errors::ProofVerifyError::InvalidOpeningProof(format!(
                "BlindFold verification failed: {e:?}"
            ))
        })
}
