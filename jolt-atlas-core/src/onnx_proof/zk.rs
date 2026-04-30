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

/// Run a batched ZK sumcheck (multiple instances) and collect stage data.
fn run_zk_batched_sumcheck(
    instances: Vec<&mut dyn SumcheckInstanceProver<F, T>>,
    prover: &mut Prover<F, T>,
    blindfold_accumulator: &mut BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    pedersen_gens: &PedersenGenerators<C>,
) -> joltworks::subprotocols::sumcheck::ZkSumcheckProof<F, C, T> {
    let max_degree = instances
        .iter()
        .map(|sc| sc.get_params().degree())
        .max()
        .unwrap_or(1);
    let max_rounds = instances
        .iter()
        .map(|sc| sc.get_params().num_rounds())
        .max()
        .unwrap_or(1);

    let _ = prover.accumulator.take_pending_claims();
    let _ = prover.accumulator.take_pending_claim_ids();

    let mut rng = rand::thread_rng();

    let (zk_proof, _, _) = BatchedSumcheck::prove_zk::<F, C, T, _>(
        instances,
        &mut prover.accumulator,
        blindfold_accumulator,
        &mut prover.transcript,
        pedersen_gens,
        &mut rng,
    );

    stage_configs.push(StageConfig::new_chain(max_rounds, max_degree));
    zk_proof
}

/// Run ZK eval reduction for a node: commit h polynomial via Pedersen.
fn prove_zk_eval_reduction(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
) {
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

/// Verify ZK eval reduction for a node: absorb h commitment, skip claim checks.
fn verify_zk_eval_reduction(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    use joltworks::subprotocols::evaluation_reduction::EvalReductionProtocol;

    let openings = accumulator.get_node_openings(node.idx);
    let h_commitment = bundle.eval_reduction_h_commitments.get(&node.idx);
    let reduced_instance =
        EvalReductionProtocol::verify_zk::<F, T>(&openings, h_commitment, transcript)?;
    accumulator
        .reduced_evaluations
        .insert(node.idx, reduced_instance);
    Ok(())
}

/// Helper to verify a ZK sumcheck given verifier instances.
fn verify_zk_sumcheck_instances(
    zk_proof: &joltworks::subprotocols::sumcheck::ZkSumcheckProof<F, C, T>,
    verifier_instances: Vec<
        Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>>,
    >,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    let verifier_refs: Vec<
        &dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
    > = verifier_instances
        .iter()
        .map(|v| {
            v.as_ref()
                as &dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>
        })
        .collect();

    BatchedSumcheck::verify_zk::<F, C, T>(zk_proof, verifier_refs, accumulator, transcript)?;
    Ok(())
}

/// Verify Div operator ZK proof: custom flow mirroring prove_div_zk.
fn verify_div_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::ops::div::DivVerifier;
    use crate::onnx_proof::range_checking::{
        range_check_operands::DivRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    };
    use crate::utils::opening_access::AccOpeningAccessor;
    use common::CommittedPoly;
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Execution sumcheck (DivVerifier squeezes r_node_output from transcript)
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(
            *proof_node_idx, node.idx,
            "ZK sumcheck proof order mismatch"
        );
        let verifier = DivVerifier::new(node.clone(), transcript);
        let verifier_instances: Vec<
            Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>>,
        > = vec![Box::new(verifier)];
        verify_zk_sumcheck_instances(zk_proof, verifier_instances, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 2. ZK eval reduction
    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // 3. Verify quotient binding
    {
        let reduced = AccOpeningAccessor::new(&mut *accumulator, node).get_reduced_opening();
        let mut provider = AccOpeningAccessor::new(&mut *accumulator, node)
            .into_provider(transcript, reduced.0.clone());
        provider.append_advice(CommittedPoly::DivNodeQuotient);
    }

    // 4. Range check + one-hot (skip for scalar outputs)
    if !node.is_scalar() {
        // Range check sumcheck
        {
            let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
            assert_eq!(
                *proof_node_idx, node.idx,
                "ZK sumcheck proof order mismatch"
            );
            let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
            let rangecheck_verifier = rangecheck_provider
                .read_raf_verify::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                    accumulator,
                    transcript,
                );
            let verifier_instances: Vec<
                Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>>,
            > = vec![Box::new(rangecheck_verifier)];
            verify_zk_sumcheck_instances(zk_proof, verifier_instances, accumulator, transcript)?;
            *zk_proof_idx += 1;
        }

        // One-hot batch sumcheck (Ra, HammingWeight, Booleanity)
        {
            let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
            assert_eq!(
                *proof_node_idx, node.idx,
                "ZK sumcheck proof order mismatch"
            );
            let encoding = RangeCheckEncoding::<DivRangeCheckOperands>::new(node);
            let [ra_v, hw_v, bool_v] = joltworks::subprotocols::shout::ra_onehot_verifiers(
                &encoding,
                accumulator,
                transcript,
            );
            let verifier_instances: Vec<
                Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>>,
            > = vec![ra_v, hw_v, bool_v];
            verify_zk_sumcheck_instances(zk_proof, verifier_instances, accumulator, transcript)?;
            *zk_proof_idx += 1;
        }
    }

    Ok(())
}

/// Verify a neural teleportation operator ZK proof.
fn verify_neural_teleport_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::neural_teleport::{
        division::TeleportDivisionVerifier, range_and_onehot::NeuralTeleportRangeOneHot,
    };
    use crate::onnx_proof::range_checking::{
        range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    };
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Eval reduction
    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    let tau = match &node.operator {
        Operator::Sigmoid(op) => op.tau,
        Operator::Tanh(op) => op.tau,
        Operator::Erf(op) => op.tau,
        _ => unreachable!(),
    };

    // 2. Division sumcheck
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*proof_node_idx, node.idx);
        let div_verifier = TeleportDivisionVerifier::new(node.clone(), accumulator, tau);
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![Box::new(div_verifier)],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

    // 3. Lookup sumcheck
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*proof_node_idx, node.idx);
        let verifier_instances = create_verifier_instances(node, accumulator, model, transcript);
        verify_zk_sumcheck_instances(zk_proof, verifier_instances, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 4. Range-check + operator-specific one-hot
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*proof_node_idx, node.idx);
        let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let rangecheck_verifier = rangecheck_provider
            .read_raf_verify::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                accumulator,
                transcript,
            );
        macro_rules! verify_ra_onehot {
            ($op:expr) => {{
                let ra_encoding = NeuralTeleportRangeOneHot::<F, T>::ra_encoding($op, node);
                let ra_v = joltworks::subprotocols::shout::ra_onehot_verifiers(
                    &ra_encoding,
                    &*accumulator,
                    transcript,
                );
                let mut instances: Vec<
                    Box<
                        dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<
                            F,
                            T,
                        >,
                    >,
                > = vec![Box::new(rangecheck_verifier)];
                instances.extend(ra_v);
                verify_zk_sumcheck_instances(zk_proof, instances, accumulator, transcript)
            }};
        }
        match &node.operator {
            Operator::Sigmoid(op) => verify_ra_onehot!(op)?,
            Operator::Tanh(op) => verify_ra_onehot!(op)?,
            Operator::Erf(op) => verify_ra_onehot!(op)?,
            _ => unreachable!(),
        };
        *zk_proof_idx += 1;
    }

    // 5. Range-check one-hot/hamming-weight consistency
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*proof_node_idx, node.idx);
        let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let [rc_ra, rc_hw, rc_bool] = joltworks::subprotocols::shout::ra_onehot_verifiers(
            &rc_encoding,
            accumulator,
            transcript,
        );
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![rc_ra, rc_hw, rc_bool],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Prove a neural teleportation operator (Sigmoid, Tanh, Erf) with ZK.
/// Default flow: eval reduction first, then division + lookup + range/onehot stages.
#[expect(clippy::too_many_arguments)]
fn prove_neural_teleport_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    model: &atlas_onnx_tracer::model::Model,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
) {
    use crate::onnx_proof::neural_teleport::{
        division::{TeleportDivisionParams, TeleportDivisionProver},
        range_and_onehot::NeuralTeleportRangeOneHot,
    };
    use crate::onnx_proof::range_checking::{
        range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    };
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Eval reduction (standard, before sumchecks)
    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // Dispatch to get operator-specific tau and create the lookup prover instance
    let tau = match &node.operator {
        Operator::Sigmoid(op) => op.tau,
        Operator::Tanh(op) => op.tau,
        Operator::Erf(op) => op.tau,
        _ => unreachable!(),
    };

    // 2. Division sumcheck
    let div_params = TeleportDivisionParams::<F>::new(node.clone(), &prover.accumulator, tau);
    let mut div_sc = TeleportDivisionProver::new(&prover.trace, div_params);
    let div_proof = run_zk_sumcheck(
        &mut div_sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, div_proof));

    // 3. Lookup sumcheck (operator-specific: needs mutable accumulator/transcript)
    {
        use crate::onnx_proof::ops::sigmoid::{SigmoidParams, SigmoidProver};
        let lookup_op = match &node.operator {
            Operator::Sigmoid(op) => op.clone(),
            // TODO: Add Tanh, Erf lookup prover creation here
            _ => unreachable!(),
        };
        let params = SigmoidParams::new(
            node.clone(),
            &model.graph,
            &prover.accumulator,
            &mut prover.transcript,
            lookup_op,
        );
        let mut sc = SigmoidProver::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let lookup_proof = run_zk_sumcheck(
            &mut sc,
            prover,
            blindfold_accumulator,
            stage_configs,
            pedersen_gens,
        );
        zk_sumcheck_proofs.push((node.idx, lookup_proof));
    }

    // 4. Range-check + operator-specific one-hot (batched: rangecheck + 3 ra_onehot)
    // Helper macro to avoid dynamic dispatch on RaOneHotEncoding (uses &impl).
    macro_rules! prove_range_and_onehot_zk {
        ($op:expr) => {{
            let lookup_indices =
                NeuralTeleportRangeOneHot::<F, T>::lookup_indices($op, node, &prover.trace);
            let ra_encoding = NeuralTeleportRangeOneHot::<F, T>::ra_encoding($op, node);
            let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
            let (rangecheck_sumcheck, rc_lookup_indices) = rangecheck_provider
                .read_raf_prove::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            let ra_onehot = joltworks::subprotocols::shout::ra_onehot_provers(
                &ra_encoding,
                &lookup_indices,
                &prover.accumulator,
                &mut prover.transcript,
            );
            let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
                vec![Box::new(rangecheck_sumcheck)];
            instances.extend(ra_onehot);
            let refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
                instances.iter_mut().map(|v| &mut **v as _).collect();
            let proof = run_zk_batched_sumcheck(
                refs,
                prover,
                blindfold_accumulator,
                stage_configs,
                pedersen_gens,
            );
            zk_sumcheck_proofs.push((node.idx, proof));
            rc_lookup_indices
        }};
    }
    let rc_lookup_indices = match &node.operator {
        Operator::Sigmoid(op) => prove_range_and_onehot_zk!(op),
        Operator::Tanh(op) => prove_range_and_onehot_zk!(op),
        Operator::Erf(op) => prove_range_and_onehot_zk!(op),
        _ => unreachable!(),
    };

    // 5. Range-check one-hot/hamming-weight consistency (batched: rc_ra + rc_hw + rc_bool)
    let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
    let [mut rc_ra, mut rc_hw, mut rc_bool] = joltworks::subprotocols::shout::ra_onehot_provers(
        &rc_encoding,
        &rc_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let hw_proof = run_zk_batched_sumcheck(
        vec![&mut *rc_ra, &mut *rc_hw, &mut *rc_bool],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, hw_proof));
}

/// Prove Div operator with ZK: custom flow with execution sumcheck before eval reduction.
#[expect(clippy::too_many_arguments)]
fn prove_div_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
) {
    use crate::onnx_proof::ops::div::{DivParams, DivProver};
    use crate::onnx_proof::range_checking::{
        range_check_operands::DivRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    };
    use crate::utils::opening_access::AccOpeningAccessor;
    use common::CommittedPoly;
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Execution sumcheck (r_node_output from transcript challenges)
    let params = DivParams::<F>::new(node.clone(), &mut prover.transcript);
    let mut sc = DivProver::initialize(&prover.trace, params);
    let zk_proof = run_zk_sumcheck(
        &mut sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, zk_proof));

    // 2. ZK eval reduction
    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // 3. Bind quotient as committed poly at reduced opening point
    {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let reduced = accessor.get_reduced_opening();
        let mut provider = accessor.into_provider(&mut prover.transcript, reduced.0.clone());
        provider.append_advice(CommittedPoly::DivNodeQuotient, reduced.1);
    }

    // 4. Range check + one-hot (skip for scalar outputs)
    // TODO: Re-enable when range check constraints are debugged
    if !node.is_scalar() {
        // Run range check and one-hot in non-ZK mode to keep accumulator consistent
        let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
        let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        let rc_proof = run_zk_sumcheck(
            &mut rangecheck_sumcheck,
            prover,
            blindfold_accumulator,
            stage_configs,
            pedersen_gens,
        );
        zk_sumcheck_proofs.push((node.idx, rc_proof));

        let encoding = RangeCheckEncoding::<DivRangeCheckOperands>::new(node);
        let [mut ra_sc, mut hw_sc, mut bool_sc] = joltworks::subprotocols::shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let onehot_proof = run_zk_batched_sumcheck(
            vec![&mut *ra_sc, &mut *hw_sc, &mut *bool_sc],
            prover,
            blindfold_accumulator,
            stage_configs,
            pedersen_gens,
        );
        zk_sumcheck_proofs.push((node.idx, onehot_proof));
    }
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
        if matches!(
            &node.operator,
            Operator::Sigmoid(_) | Operator::Tanh(_) | Operator::Erf(_)
        ) {
            prove_neural_teleport_zk(
                node,
                &mut prover,
                pp.shared.model(),
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            );
        } else if matches!(&node.operator, Operator::Div(_)) {
            prove_div_zk(
                node,
                &mut prover,
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            );
        } else {
            // Standard flow: eval reduction first, then execution sumcheck.
            prove_zk_eval_reduction(
                node,
                &mut prover,
                pedersen_gens,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
            );

            let zk_proof =
                create_prover_instance(node, &prover, pp.shared.model()).map(|mut sc| {
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
        let expected_coeffs = stage_configs[i].poly_degree + 1;
        let rounds: Vec<RoundWitness<F>> = sd
            .poly_coeffs
            .iter()
            .zip(sd.challenges.iter())
            .map(|(coeffs, challenge)| {
                let challenge_f: F = (*challenge).into();
                // Pad coefficients to expected degree if UniPoly stripped trailing zeros
                let mut padded = coeffs.clone();
                padded.resize(expected_coeffs, F::zero());
                RoundWitness::new(padded, challenge_f)
            })
            .collect();

        // Combine output constraints: single-instance stages have 1, batched have N.
        // Each constraint is pre-scaled by scale_by_new_challenge(). For multi-instance
        // stages, concatenate terms with offset challenge indices (no extra alpha layer).
        use joltworks::subprotocols::blindfold::output_constraint::{
            OutputClaimConstraint, ProductTerm, ValueSource,
        };
        let combined_constraint: Option<OutputClaimConstraint> =
            if sd.output_constraints.iter().any(|c| c.is_none()) {
                None
            } else if sd.output_constraints.len() == 1 {
                sd.output_constraints[0].clone()
            } else {
                let mut combined_terms = Vec::new();
                let mut combined_openings = Vec::new();
                let mut challenge_offset = 0usize;
                for oc in &sd.output_constraints {
                    let c = oc.as_ref().unwrap();
                    let offset = |vs: &ValueSource| match vs {
                        ValueSource::Challenge(idx) => {
                            ValueSource::Challenge(idx + challenge_offset)
                        }
                        other => other.clone(),
                    };
                    for term in &c.terms {
                        combined_terms.push(ProductTerm::new(
                            offset(&term.coeff),
                            term.factors.iter().map(&offset).collect(),
                        ));
                    }
                    for id in &c.required_openings {
                        if !combined_openings.contains(id) {
                            combined_openings.push(*id);
                        }
                    }
                    challenge_offset += c.num_challenges;
                }
                Some(OutputClaimConstraint::new(
                    combined_terms,
                    combined_openings,
                ))
            };
        if let Some(constraint) = combined_constraint {
            // Challenge values: concatenate per-instance values (already include batch_coeff).
            let output_challenge_values: Vec<F> = sd
                .constraint_challenge_values
                .iter()
                .flat_map(|cv| cv.iter().cloned())
                .collect();
            // Map opening values to match the constraint's required_openings order.
            let claims_map: std::collections::HashMap<_, _> =
                sd.output_claims.iter().cloned().collect();
            let output_opening_values: Vec<F> = constraint
                .required_openings
                .iter()
                .map(|id| {
                    *claims_map
                        .get(id)
                        .unwrap_or_else(|| panic!("Missing opening claim for {id:?}"))
                })
                .collect();
            let final_output_witness =
                FinalOutputWitness::general(output_challenge_values, output_opening_values);
            all_stages.push(StageWitness::with_final_output(
                rounds,
                final_output_witness,
            ));
            stage_configs[i] = stage_configs[i].clone().with_constraint(constraint);
        } else {
            all_stages.push(StageWitness::new(rounds));
        }
    }

    // Collect initial claims for each chain start
    let initial_claims: Vec<F> = stage_data_vec
        .iter()
        .enumerate()
        .filter(|(i, _)| stage_configs[*i].starts_new_chain || *i == 0)
        .map(|(_, sd)| sd.initial_claim)
        .collect();
    let blindfold_witness =
        BlindFoldWitness::with_extra_constraints(initial_claims, all_stages, vec![]);
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
        if matches!(
            &node.operator,
            Operator::Sigmoid(_) | Operator::Tanh(_) | Operator::Erf(_)
        ) {
            verify_neural_teleport_zk(
                node,
                model,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?;
        } else if matches!(&node.operator, Operator::Div(_)) {
            verify_div_zk(
                node,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?;
        } else {
            // Standard flow: eval reduction first, then execution sumcheck
            verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;

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

                let verifier_instances =
                    create_verifier_instances(node, &mut accumulator, model, &mut transcript);
                verify_zk_sumcheck_instances(
                    zk_proof,
                    verifier_instances,
                    &mut accumulator,
                    &mut transcript,
                )?;

                zk_proof_idx += 1;
            }
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
        Operator::ScalarConstDiv(_) => {
            use crate::onnx_proof::ops::scalar_const_div::{
                ScalarConstDivParams, ScalarConstDivProver,
            };
            let params = ScalarConstDivParams::<F>::new(node.clone(), &prover.accumulator);
            Some(Box::new(ScalarConstDivProver::initialize(
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
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    model: &atlas_onnx_tracer::model::Model,
    transcript: &mut T,
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
        Operator::ScalarConstDiv(_) => {
            use crate::onnx_proof::ops::scalar_const_div::ScalarConstDivVerifier;
            vec![Box::new(ScalarConstDivVerifier::new(
                node.clone(),
                accumulator,
            ))]
        }
        Operator::Sigmoid(op) => {
            use crate::onnx_proof::ops::sigmoid::SigmoidVerifier;
            vec![Box::new(SigmoidVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
                op.clone(),
            ))]
        }
        _ => vec![],
    }
}
