//! ZK proving and verification for ONNX neural network computations.
//!
//! This module provides `prove_zk` and `verify_zk` functions that run the
//! proof pipeline in a single pass with BlindFold zero-knowledge proofs.
//! Sumcheck round polynomials are Pedersen-committed instead of sent in the clear,
//! and a BlindFold proof (Nova folding + Spartan) verifies constraint consistency.

use crate::onnx_proof::{AtlasProverPreprocessing, AtlasVerifierPreprocessing, ONNXProof, Prover};
use ark_bn254::{Bn254, Fr};
use ark_std::{One, Zero};
use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    ops::Operator,
    tensor::Tensor,
    utils::quantize::scale_to_multiplier,
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
    /// Number of sumcheck instances per ZK proof (for transcript batching coeff replay).
    pub zk_sumcheck_num_instances: Vec<usize>,
    /// Pedersen commitments for inter-stage private claims (keyed by node index).
    /// Used by multi-stage operators (e.g. Softmax) that cache evaluations between stages.
    pub inter_stage_commitments: BTreeMap<usize, Vec<joltworks::curve::Bn254G1>>,
    /// Auxiliary opening claims that must be pre-populated in the verifier accumulator.
    /// In the non-ZK flow, these come from ONNXProof::opening_claims::populate_accumulator.
    pub auxiliary_claims: BTreeMap<joltworks::poly::opening_proof::OpeningId, F>,
    /// PCS opening proof on the joint (gamma-weighted RLC) polynomial at the
    /// batched-opening-reduction point `r_sumcheck`. Together with the
    /// homomorphically-combined commitment, this is what binds `bundle.commitments`
    /// to a specific evaluation. Reveals `joint_claim` in cleartext (one
    /// gamma-weighted aggregate scalar; per-poly y_P_i values remain hidden).
    /// `None` when `poly_map` is empty (no committed polynomials).
    ///
    /// NOTE: this is Step 8 of the wiki design. Full ZK for `joint_claim`
    /// would require a hiding/ZK extension to HyperKZG (none today).
    pub joint_opening_proof: Option<<PCS as CommitmentScheme>::Proof>,
    /// Cleartext joint claim revealed by the PCS opening.
    pub joint_claim: Option<F>,
    /// The opening point `r_sumcheck` used for `joint_opening_proof`,
    /// in `F::Challenge` form so it can feed `PCS::verify` directly.
    pub joint_opening_point: Option<Vec<<F as joltworks::field::JoltField>::Challenge>>,
    /// ZK proof for the batched-opening reduction sumcheck. The verifier
    /// re-absorbs its `round_commitments` into the main transcript so the
    /// transcript state at the y_com / PCS::verify points matches the prover.
    /// (BlindFold also re-verifies this stage internally via its R1CS; the
    /// bundle copy is purely for main-transcript replay.)
    pub batch_opening_zk_sumcheck:
        Option<joltworks::subprotocols::sumcheck::ZkSumcheckProof<F, C, T>>,
    /// Cleartext reduced-eval claim for each public-data node (Constant/Input).
    /// Mirror of the per-op `Constant::verify` / `Input::verify` check in non-zk:
    /// the verifier locally evaluates the public tensor's MLE at `r_reduced`
    /// and compares against this value. Catches honest-prover errors; full
    /// soundness against an active malicious prover additionally requires an
    /// R1CS constraint binding the BlindFold witness opening value to this
    /// cleartext (Jolt's `ValueSource::Constant`-style binding) — tracked as
    /// future work in `wiki/jolt-atlas/.../zk-prove-overhead.md`.
    pub public_node_reduced_claims: BTreeMap<usize, F>,
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
    if openings.is_empty() {
        // Mirror verify_zk_eval_reduction (zk.rs:204-209): nodes whose
        // consumers did not call cache_openings in the ZK path have no
        // openings to reduce. Skip; the verifier handles this symmetrically.
        // Safe for no-sumcheck operators (Input/Identity/Broadcast/MoveAxis/
        // Constant/IsNan/Clamp). For default-branch operators, the subsequent
        // sumcheck setup will panic looking up `reduced_evaluations[node]`;
        // that indicates a missing `cache_openings` call in the consumer's
        // ZK prove flow and is a real bug, not something to paper over here.
        return;
    }
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
    if openings.is_empty() {
        // No openings for this node (e.g. Input node whose consumer's cache_openings
        // wasn't called in the ZK verifier). Skip eval reduction.
        return Ok(());
    }
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

/// Verify SoftmaxLastAxis ZK proof.
///
/// Mirror of `prove_softmax_zk`: 4 batched stages with two private inter-stage
/// claims (R, r_exp) absorbed via Pedersen commitments. Each stage runs full
/// per-stage `verify_zk_sumcheck_instances` (BlindFold covers the *aggregate*
/// constraint, but per-stage transcript replay still has to match).
///
/// Public auxiliary claims (`SoftmaxSumOutput`, `SoftmaxMaxOutput`,
/// `SoftmaxMaxIndex`, `SoftmaxExpSum`) are pre-populated into the accumulator
/// from the bundle so `SoftmaxLastAxisVerifier::new` and `cache_exp_sum` see
/// the real values when they re-append them to the transcript.
fn verify_softmax_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::ops::softmax_last_axis::{
        SoftmaxLastAxisVerifier as SmVerifier, VerifierLookupTableData,
    };

    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    let Operator::SoftmaxLastAxis(op) = &node.operator else {
        unreachable!()
    };

    // Pre-populate public auxiliary claims so SmVerifier::new sees real values
    // when it re-appends them to the transcript.
    for (id, claim) in &bundle.auxiliary_claims {
        accumulator.openings.insert(
            *id,
            (
                joltworks::poly::opening_proof::OpeningPoint::default(),
                *claim,
            ),
        );
    }

    let inter_coms = bundle
        .inter_stage_commitments
        .get(&node.idx)
        .expect("Missing inter-stage commitments for Softmax node");

    // Auxiliary scalars are public, so we toggle off zk_mode while
    // SmVerifier::new and cache_exp_sum read/append them. They both rely on
    // the accumulator returning real claim values (not the zk_mode placeholder
    // zero) so the transcript appends match the prover's. We restore zk_mode
    // before the private inter-stage caches and the per-stage sumcheck builds.
    let saved_zk_mode = accumulator.zk_mode;
    accumulator.zk_mode = false;
    let scale = scale_to_multiplier(op.scale) as i32;
    let mut sm_v = SmVerifier::new(node, scale, accumulator, transcript);
    sm_v.cache_exp_sum(accumulator, transcript)
        .map_err(|e| ProofVerifyError::InvalidOpeningProof(format!("{e:?}")))?;
    accumulator.zk_mode = saved_zk_mode;

    let scale_bits = pp.shared.scale();

    // cache_R: insert explicit zero placeholder, absorb Pedersen commitment.
    sm_v.cache_R_zk(accumulator, transcript, &inter_coms[0]);

    // Stage 1
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = sm_v.build_stage1_verifiers(accumulator, transcript, scale_bits)?;
        verify_zk_sumcheck_instances(zk_proof, v, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    let lut = VerifierLookupTableData::new(scale);

    // cache_r_exp: same shape as cache_R.
    sm_v.cache_r_exp_zk(accumulator, transcript, &inter_coms[1]);

    // Stage 2
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = sm_v.build_stage2_verifiers(accumulator, transcript, scale_bits, &lut);
        verify_zk_sumcheck_instances(zk_proof, v, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // Stage 3
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = sm_v.build_stage3_verifiers(accumulator, transcript, scale_bits, &lut);
        verify_zk_sumcheck_instances(zk_proof, v, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // NOTE: operand_link (X(r2) == max_k - z_c - sat_diff, expanded as
    //   X(r2) - max_k(r2_lead) + z_hi(r2)*B + z_lo(r2) + sat_diff(r2) == 0
    // ) is intentionally skipped here.
    //
    // It is a verifier-side algebraic identity over openings X (input 0),
    // SoftmaxZHi, SoftmaxZLo, SoftmaxSatDiff, with `max_k(r2_lead)` derived by
    // the verifier from public auxiliary scalars and `B` the digit base from
    // the LUT. In the ZK pipeline all four openings are private and surface to
    // the verifier as the zk_mode placeholder zero, while max_k is non-zero,
    // so a direct verifier-side check would always fail.
    //
    // Closing this gap requires expressing the identity as a constraint inside
    // BlindFold's R1CS so it is checked as part of the aggregate proof. The
    // existing `extra_constraints` mechanism (currently dormant in prove_zk:
    // empty vec, eval_commitment_gens=None) emits `output_var = SoP` plus a
    // Pedersen binding on output_value, which is the wrong shape: an identity
    // wants `(SoP) * 1 == 0` with no output_var or commitment. The right
    // BlindFold-side change is a new `LayoutStep::AlgebraicIdentity` variant
    // (and parallel additions in layout.rs / r1cs.rs / witness.rs / folding.rs
    // / mod.rs) that allocates only the required opening vars, deduplicates
    // against earlier constraint allocations via the existing seen_openings
    // set, and emits a single R1CS row `(SoP) * 1 = 0`. Then this site
    // constructs an OutputClaimConstraint with terms
    //   1*X, (-max_k_eval)*1, B*z_hi, 1*z_lo, 1*sat_diff
    // (B as ValueSource::Constant, max_k_eval as a baked challenge), passes
    // the four opening_values via a new identity-witness collection on
    // BlindFoldWitness, and the prover commits no extra Pedersen output.
    //
    // Tracked in wiki/jolt-atlas/book/src/underway/pr-239-review.md as the
    // first remaining outstanding item.

    // Stage 4
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = sm_v.build_stage4_verifiers(accumulator, transcript, scale_bits, &lut);
        verify_zk_sumcheck_instances(zk_proof, v, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Verify GatherLarge ZK proof.
fn verify_gather_large_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::ops::gather::large::GatherRaEncoding;

    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // Execution sumcheck
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let verifier_instances = create_verifier_instances(node, accumulator, model, transcript);
        verify_zk_sumcheck_instances(zk_proof, verifier_instances, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // One-hot batch
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let encoding = GatherRaEncoding::new(node);
        let [ra, hw, b] =
            joltworks::subprotocols::shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![ra, hw, b], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Verify GatherSmall ZK proof.
fn verify_gather_small_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::ops::gather::small::{
        build_stage2_verifiers_zk, build_stage3_verifier_zk,
    };

    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // Stage 1: Execution
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let vi = create_verifier_instances(node, accumulator, model, transcript);
        verify_zk_sumcheck_instances(zk_proof, vi, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // Stage 2: HammingBooleanity + Booleanity
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let (hb_v, bool_v) =
            build_stage2_verifiers_zk::<F>(node, &model.graph, accumulator, transcript);
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![Box::new(hb_v), Box::new(bool_v)],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

    // Stage 3: HammingWeight
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let hw_v = build_stage3_verifier_zk::<F>(node, &model.graph, accumulator, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![Box::new(hw_v)], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Verify Relu ZK proof: default flow mirroring prove_relu_zk.
fn verify_relu_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::op_lookups::{OpLookupEncoding, OpLookupProvider};
    use joltworks::lookup_tables::relu::ReluTable;

    // 1. Eval reduction
    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // 2. Execution sumcheck
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let provider = OpLookupProvider::new(node.clone());
        let v = provider
            .read_raf_verify::<F, T, ReluTable<{ common::consts::XLEN }>>(accumulator, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![Box::new(v)], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 3. One-hot batch
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let encoding = OpLookupEncoding::new(node);
        let [ra, hw, b] =
            joltworks::subprotocols::shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![ra, hw, b], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Verify Rsqrt ZK proof: custom flow mirroring prove_rsqrt_zk.
fn verify_rsqrt_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::{
        ops::rsqrt::RsqrtVerifier,
        range_checking::{
            range_check_operands::{RiRangeCheckOperands, RsRangeCheckOperands},
            RangeCheckEncoding, RangeCheckProvider,
        },
    };
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Execution sumcheck
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = RsqrtVerifier::new(node.clone(), transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![Box::new(v)], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 2. Eval reduction
    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // 3. Two range checks batched
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let div_rc = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
        let div_v = div_rc
            .read_raf_verify::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                accumulator,
                transcript,
            );
        let sqrt_rc = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
        let sqrt_v = sqrt_rc
            .read_raf_verify::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                accumulator,
                transcript,
            );
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![Box::new(div_v), Box::new(sqrt_v)],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

    // 4. Six one-hot instances batched
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let div_enc = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
        let [d_ra, d_hw, d_bool] =
            joltworks::subprotocols::shout::ra_onehot_verifiers(&div_enc, accumulator, transcript);
        let sqrt_enc = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
        let [s_ra, s_hw, s_bool] =
            joltworks::subprotocols::shout::ra_onehot_verifiers(&sqrt_enc, accumulator, transcript);
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![d_ra, d_hw, d_bool, s_ra, s_hw, s_bool],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

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
    use crate::{
        onnx_proof::{
            ops::div::DivVerifier,
            range_checking::{
                range_check_operands::DivRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
            },
        },
        utils::opening_access::AccOpeningAccessor,
    };
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
    use crate::onnx_proof::{
        neural_teleport::{
            division::TeleportDivisionVerifier, eval_shift::EvalShiftVerifier,
            range_and_onehot::NeuralTeleportRangeOneHot,
        },
        range_checking::{
            range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding,
            RangeCheckProvider,
        },
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

    // 2. Division + Reduction sumchecks (batched, mirroring non-ZK path)
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*proof_node_idx, node.idx);
        let div_verifier = TeleportDivisionVerifier::new(node.clone(), accumulator, tau);
        let eval_shift_verifier = EvalShiftVerifier::new(node.clone(), accumulator);
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![Box::new(div_verifier), Box::new(eval_shift_verifier)],
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

/// Verify Cos/Sin ZK proof: custom flow mirroring prove_cos_sin_zk.
fn verify_cos_sin_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::{
        onnx_proof::{
            neural_teleport::{
                division::TeleportDivisionVerifier, range_and_onehot::NeuralTeleportRangeOneHot,
            },
            range_checking::{
                range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding,
                RangeCheckProvider,
            },
        },
        utils::opening_access::AccOpeningAccessor,
    };
    use common::CommittedPoly;
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    let tau = atlas_onnx_tracer::model::consts::FOUR_PI_APPROX;

    // 1. Division sumcheck (from transcript)
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let v = TeleportDivisionVerifier::new_from_transcript(node.clone(), tau, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![Box::new(v)], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 2. Lookup sumcheck
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let vi = create_verifier_instances(node, accumulator, model, transcript);
        verify_zk_sumcheck_instances(zk_proof, vi, accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // 3. Verify quotient binding
    {
        let accessor = AccOpeningAccessor::new(&mut *accumulator, node);
        let teleport_q = accessor.get_advice(VirtualPoly::TeleportQuotient);
        let mut provider = accessor.into_provider(transcript, teleport_q.0.clone());
        provider.append_advice(CommittedPoly::TeleportNodeQuotient);
    }

    // 4. Eval reduction
    verify_zk_eval_reduction(node, bundle, accumulator, transcript)?;

    // 5. Range+onehot
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let rc_prov = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let rc_v = rc_prov
            .read_raf_verify::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                accumulator,
                transcript,
            );
        macro_rules! verify_ra {
            ($op:expr) => {{
                let enc = NeuralTeleportRangeOneHot::<F, T>::ra_encoding($op, node);
                let ra = joltworks::subprotocols::shout::ra_onehot_verifiers(
                    &enc,
                    &*accumulator,
                    transcript,
                );
                let mut inst: Vec<
                    Box<
                        dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<
                            F,
                            T,
                        >,
                    >,
                > = vec![Box::new(rc_v)];
                inst.extend(ra);
                verify_zk_sumcheck_instances(zk_proof, inst, accumulator, transcript)
            }};
        }
        match &node.operator {
            Operator::Cos(op) => verify_ra!(op)?,
            Operator::Sin(op) => verify_ra!(op)?,
            _ => unreachable!(),
        };
        *zk_proof_idx += 1;
    }

    // 6. Hamming-weight
    {
        let (pid, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(*pid, node.idx);
        let rc_enc = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let [a, b, c] =
            joltworks::subprotocols::shout::ra_onehot_verifiers(&rc_enc, accumulator, transcript);
        verify_zk_sumcheck_instances(zk_proof, vec![a, b, c], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    Ok(())
}

/// Verify the fused-rescaling pre-stages (mirror of `prove_fused_rebase_pre_zk`):
/// remainder advice, clamp read-raf lookup, clamp one-hot batch.
fn verify_fused_rebase_pre_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::clamp_lookups::{is_scalar, ClampEncoding, ClampLookupProvider};

    // Mirror of the prover-side panic; see `prove_fused_rebase_pre_zk`.
    if is_scalar(node) {
        unimplemented!("ZK verification not yet implemented for scalar fused-rescale nodes");
    }

    crate::onnx_proof::fused_rebase::cache_remainder_verify(node, accumulator, transcript);

    // Clamp read-raf lookup.
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(
            *proof_node_idx, node.idx,
            "ZK sumcheck proof order mismatch"
        );
        let provider = ClampLookupProvider::new(node.clone());
        let execution_verifier = provider.read_raf_verify::<F, T>(accumulator, transcript);
        verify_zk_sumcheck_instances(
            zk_proof,
            vec![Box::new(execution_verifier)],
            accumulator,
            transcript,
        )?;
        *zk_proof_idx += 1;
    }

    // Clamp one-hot batch.
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(
            *proof_node_idx, node.idx,
            "ZK sumcheck proof order mismatch"
        );
        let encoding = ClampEncoding::new(node);
        let [ra, hw, b] = joltworks::subprotocols::shout::ra_onehot_verifiers(
            &encoding,
            &*accumulator,
            transcript,
        );
        verify_zk_sumcheck_instances(zk_proof, vec![ra, hw, b], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }
    Ok(())
}

/// Verify the fused-rescaling post-stages (mirror of `prove_fused_rebase_post_zk`):
/// remainder identity range check, remainder one-hot batch.
fn verify_fused_rebase_post_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    bundle: &ZkProofBundle,
    accumulator: &mut joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    zk_proof_idx: &mut usize,
) -> Result<(), ProofVerifyError> {
    use crate::onnx_proof::fused_rebase::{
        rebase_bits, RescaleRemainderRCProvider, RescaleRemainderRaEncoding,
    };
    use joltworks::subprotocols::identity_range_check::identity_rangecheck_verifier;

    let bits = rebase_bits(&node.operator).expect("fused op");

    // Remainder identity range check.
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(
            *proof_node_idx, node.idx,
            "ZK sumcheck proof order mismatch"
        );
        let rc_provider = RescaleRemainderRCProvider::new(node.clone(), bits);
        let rc = identity_rangecheck_verifier(&rc_provider, accumulator);
        verify_zk_sumcheck_instances(zk_proof, vec![Box::new(rc)], accumulator, transcript)?;
        *zk_proof_idx += 1;
    }

    // Remainder one-hot batch.
    {
        let (proof_node_idx, zk_proof) = &bundle.zk_sumcheck_proofs[*zk_proof_idx];
        assert_eq!(
            *proof_node_idx, node.idx,
            "ZK sumcheck proof order mismatch"
        );
        let encoding = RescaleRemainderRaEncoding::new(node.idx, bits);
        let [ra, hw, boolean] = joltworks::subprotocols::shout::ra_onehot_verifiers(
            &encoding,
            &*accumulator,
            transcript,
        );
        verify_zk_sumcheck_instances(zk_proof, vec![ra, hw, boolean], accumulator, transcript)?;
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
    use crate::onnx_proof::{
        neural_teleport::{
            division::{TeleportDivisionParams, TeleportDivisionProver},
            eval_shift::{EvalShiftParams, EvalShiftProver},
            range_and_onehot::NeuralTeleportRangeOneHot,
        },
        range_checking::{
            range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding,
            RangeCheckProvider,
        },
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

    // 2. Division + Reduction sumchecks (batched, mirroring non-ZK path)
    let div_params = TeleportDivisionParams::<F>::new(node.clone(), &prover.accumulator, tau);
    let mut div_sc = TeleportDivisionProver::new(&prover.trace, div_params);
    let eval_shift_params = EvalShiftParams::new(node.clone(), &prover.accumulator);
    let mut eval_shift_sc = EvalShiftProver::initialize(&prover.trace, eval_shift_params);
    let div_proof = run_zk_batched_sumcheck(
        vec![&mut div_sc, &mut eval_shift_sc],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, div_proof));

    // 3. Lookup sumcheck (operator-specific: needs mutable accumulator/transcript)
    // Extra trailing args are forwarded to `initialize`: `TanhProver::initialize`
    // additionally takes the opening accumulator + transcript to append the
    // clamped-input advice (`VirtualPoly::DummyClampedTanhInput`, #256); the
    // append is transcript-quiet in zk mode, mirroring `TanhVerifier::new`.
    macro_rules! prove_lookup_zk {
        ($Params:ty, $Prover:ty, $op:expr $(, $init_extra:expr)*) => {{
            let params = <$Params>::new(
                node.clone(),
                &model.graph,
                &prover.accumulator,
                &mut prover.transcript,
                $op.clone(),
            );
            let mut sc = <$Prover>::initialize(&prover.trace, params $(, $init_extra)*);
            let proof = run_zk_sumcheck(
                &mut sc,
                prover,
                blindfold_accumulator,
                stage_configs,
                pedersen_gens,
            );
            zk_sumcheck_proofs.push((node.idx, proof));
        }};
    }
    match &node.operator {
        Operator::Sigmoid(op) => {
            use crate::onnx_proof::ops::sigmoid::{SigmoidParams, SigmoidProver};
            prove_lookup_zk!(SigmoidParams::<F>, SigmoidProver::<F>, op);
        }
        Operator::Tanh(op) => {
            use crate::onnx_proof::ops::tanh::{TanhParams, TanhProver};
            prove_lookup_zk!(
                TanhParams::<F>,
                TanhProver::<F>,
                op,
                &mut prover.accumulator,
                &mut prover.transcript
            );
        }
        Operator::Erf(op) => {
            use crate::onnx_proof::ops::erf::{ErfParams, ErfProver};
            prove_lookup_zk!(ErfParams::<F>, ErfProver::<F>, op);
        }
        _ => unreachable!(),
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

/// Prove Cos/Sin with ZK: custom flow (division from transcript, then lookup,
/// then quotient binding, then eval reduction, then range+onehot).
#[expect(clippy::too_many_arguments)]
fn prove_cos_sin_zk(
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
    use crate::{
        onnx_proof::{
            neural_teleport::{
                division::{TeleportDivisionParams, TeleportDivisionProver},
                range_and_onehot::NeuralTeleportRangeOneHot,
            },
            range_checking::{
                range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding,
                RangeCheckProvider,
            },
        },
        utils::opening_access::AccOpeningAccessor,
    };
    use common::CommittedPoly;
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    let tau = atlas_onnx_tracer::model::consts::FOUR_PI_APPROX;

    // 1. Division sumcheck (r_node_output from transcript)
    let div_params =
        TeleportDivisionParams::<F>::new_from_transcript(node.clone(), &mut prover.transcript, tau);
    let mut div_sc = TeleportDivisionProver::new(&prover.trace, div_params);
    let div_proof = run_zk_sumcheck(
        &mut div_sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, div_proof));

    // 2. Lookup sumcheck
    macro_rules! prove_lookup {
        ($Params:ty, $Prover:ty) => {{
            let params = <$Params>::new(
                node.clone(),
                &model.graph,
                &prover.accumulator,
                &mut prover.transcript,
            );
            let mut sc = <$Prover>::initialize(
                &prover.trace,
                params,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
            let proof = run_zk_sumcheck(
                &mut sc,
                prover,
                blindfold_accumulator,
                stage_configs,
                pedersen_gens,
            );
            zk_sumcheck_proofs.push((node.idx, proof));
        }};
    }
    match &node.operator {
        Operator::Cos(_) => {
            use crate::onnx_proof::ops::cos::{CosParams, CosProver};
            prove_lookup!(CosParams::<F>, CosProver::<F>);
        }
        Operator::Sin(_) => {
            use crate::onnx_proof::ops::sin::{SinParams, SinProver};
            prove_lookup!(SinParams::<F>, SinProver::<F>);
        }
        _ => unreachable!(),
    }

    // 3. Bind TeleportNodeQuotient as committed poly
    {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let teleport_q = accessor.get_advice(VirtualPoly::TeleportQuotient);
        let mut provider = accessor.into_provider(&mut prover.transcript, teleport_q.0.clone());
        provider.append_advice(CommittedPoly::TeleportNodeQuotient, teleport_q.1);
    }

    // 4. Eval reduction
    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // 5-6. Range+onehot and hamming-weight
    macro_rules! prove_range_onehot {
        ($op:expr) => {{
            let indices =
                NeuralTeleportRangeOneHot::<F, T>::lookup_indices($op, node, &prover.trace);
            let enc = NeuralTeleportRangeOneHot::<F, T>::ra_encoding($op, node);
            let rc_prov = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
            let (rc_sc, rc_idx) = rc_prov
                .read_raf_prove::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            let ra = joltworks::subprotocols::shout::ra_onehot_provers(
                &enc,
                &indices,
                &prover.accumulator,
                &mut prover.transcript,
            );
            let mut inst: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![Box::new(rc_sc)];
            inst.extend(ra);
            let refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
                inst.iter_mut().map(|v| &mut **v as _).collect();
            let p = run_zk_batched_sumcheck(
                refs,
                prover,
                blindfold_accumulator,
                stage_configs,
                pedersen_gens,
            );
            zk_sumcheck_proofs.push((node.idx, p));
            rc_idx
        }};
    }
    let rc_indices = match &node.operator {
        Operator::Cos(op) => prove_range_onehot!(op),
        Operator::Sin(op) => prove_range_onehot!(op),
        _ => unreachable!(),
    };

    let rc_enc = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
    let [mut a, mut b, mut c] = joltworks::subprotocols::shout::ra_onehot_provers(
        &rc_enc,
        &rc_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let hw = run_zk_batched_sumcheck(
        vec![&mut *a, &mut *b, &mut *c],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, hw));
}

/// Prove SoftmaxLastAxis with ZK: default flow, 4 batched stages.
/// Mirrors SoftmaxLastAxisProver::prove but uses run_zk_batched_sumcheck.
#[expect(clippy::too_many_arguments)]
fn prove_softmax_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
    inter_stage_commitments: &mut BTreeMap<usize, Vec<joltworks::curve::Bn254G1>>,
    auxiliary_claims: &mut BTreeMap<joltworks::poly::opening_proof::OpeningId, F>,
) {
    use crate::onnx_proof::ops::softmax_last_axis::{
        pad_to_power_of_two, rc::sat_diff_rc_bits, to_indices, to_lookup_bits, LookupTableData,
        SoftmaxLastAxisProver as SmProver,
    };
    use atlas_onnx_tracer::ops::softmax::softmax_last_axis_decomposed;

    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    let Operator::SoftmaxLastAxis(op) = &node.operator else {
        unreachable!()
    };
    let softmax_input = prover.trace.operand_tensors(node)[0];
    let scale = scale_to_multiplier(op.scale) as i32;
    let trace = softmax_last_axis_decomposed(softmax_input, scale).1;
    let mut sm = SmProver::new(node, trace, scale);

    let scale_bits = prover.preprocessing.scale();
    let r_lookup_bits = to_lookup_bits(&sm.trace.R, scale_bits as usize);
    let r_indices = to_indices(&sm.trace.R);
    let r_exp_lookup_bits = to_lookup_bits(&sm.trace.decomposed_exp.r_exp, scale_bits as usize);
    let r_exp_indices = to_indices(&sm.trace.decomposed_exp.r_exp);
    let z_hi_indices = to_indices(&sm.trace.decomposed_exp.z_hi);
    let z_lo_indices = to_indices(&sm.trace.decomposed_exp.z_lo);
    let sat_diff_lookup_bits = to_lookup_bits(
        &sm.trace.decomposed_exp.sat_diff,
        sat_diff_rc_bits(scale_bits as usize),
    );
    let sat_diff_indices = to_indices(&sm.trace.decomposed_exp.sat_diff);

    let base = sm.trace.decomposed_exp.lut.base as u64;
    let mut table_hi = std::mem::take(&mut sm.trace.decomposed_exp.lut.lut_hi);
    let z_bound_minus_1 = (table_hi.len() as u64) * base - 1;
    pad_to_power_of_two(&mut table_hi);
    let mut table_lo = std::mem::take(&mut sm.trace.decomposed_exp.lut.lut_lo);
    pad_to_power_of_two(&mut table_lo);

    let lut_data = LookupTableData {
        table_hi,
        table_lo,
        z_hi_indices,
        z_lo_indices,
        z_bound_minus_1,
        base,
    };

    // Public auxiliary scalars: toggle off prover zk_mode so the cleartext
    // claims are appended to the transcript. The verifier toggles its own
    // zk_mode off symmetrically (see verify_softmax_zk).
    let saved_zk_mode = prover.accumulator.zk_mode;
    prover.accumulator.zk_mode = false;
    sm.send_auxiliary_vectors(prover);
    sm.cache_exp_sum(prover);
    prover.accumulator.zk_mode = saved_zk_mode;

    // Capture only PUBLIC auxiliary claims for the verifier.
    // These are verifier-computable from the auxiliary vectors sent via transcript.
    // Private claims (sumcheck outputs, cache_R, cache_r_exp) are NOT included.
    {
        use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
        let sid = SumcheckId::NodeExecution(node.idx);
        let [f, _] = sm.F_N;
        for k in 0..f {
            for vp_fn in [
                VirtualPoly::SoftmaxSumOutput as fn(usize, usize) -> VirtualPoly,
                VirtualPoly::SoftmaxMaxOutput,
                VirtualPoly::SoftmaxMaxIndex,
            ] {
                let id = OpeningId::new(vp_fn(node.idx, k), sid);
                if let Some((_, claim)) = prover.accumulator.openings.get(&id) {
                    auxiliary_claims.insert(id, *claim);
                }
            }
        }
        // SoftmaxExpSum: derived from public exp_sum_q auxiliary vector
        let exp_sum_id = OpeningId::new(VirtualPoly::SoftmaxExpSum(node.idx), sid);
        if let Some((_, claim)) = prover.accumulator.openings.get(&exp_sum_id) {
            auxiliary_claims.insert(exp_sum_id, *claim);
        }
    }

    // ZK cache_R: evaluate R polynomial, Pedersen-commit the claim, store in accumulator
    let cache_r_com = {
        use crate::utils::opening_access::AccOpeningAccessor;
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let r0 = accessor.get_reduced_opening().0;
        let eval = MultilinearPolynomial::from(sm.trace.R.clone()).evaluate(&r0.r);
        let blinding = F::random(&mut rand::thread_rng());
        let com = pedersen_gens.commit(&[eval], &blinding);
        prover.transcript.append_serializable(&com);
        // Store in accumulator (same as cache_R but without cleartext transcript append)
        let opening_id = joltworks::poly::opening_proof::OpeningId::new(
            VirtualPoly::SoftmaxRecipMultRemainder(node.idx),
            joltworks::poly::opening_proof::SumcheckId::NodeExecution(node.idx),
        );
        prover.accumulator.openings.insert(opening_id, (r0, eval));
        com
    };

    // Stage 1
    let mut s1 = sm.build_stage1_instances(prover, r_lookup_bits);
    let s1_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        s1.iter_mut().map(|v| &mut **v as _).collect();
    let s1_proof = run_zk_batched_sumcheck(
        s1_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s1_proof));

    // ZK cache_r_exp: same pattern
    let cache_r_exp_com = {
        use crate::utils::opening_access::AccOpeningAccessor;
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let eval =
            MultilinearPolynomial::from(sm.trace.decomposed_exp.r_exp.clone()).evaluate(&r1.r);
        let blinding = F::random(&mut rand::thread_rng());
        let com = pedersen_gens.commit(&[eval], &blinding);
        prover.transcript.append_serializable(&com);
        let opening_id = joltworks::poly::opening_proof::OpeningId::new(
            VirtualPoly::SoftmaxExpRemainder(node.idx),
            joltworks::poly::opening_proof::SumcheckId::NodeExecution(node.idx),
        );
        prover.accumulator.openings.insert(opening_id, (r1, eval));
        com
    };

    // Stage 2
    let mut s2 = sm.build_stage2_instances(prover, r_exp_lookup_bits, &r_indices, &lut_data);
    let s2_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        s2.iter_mut().map(|v| &mut **v as _).collect();
    let s2_proof = run_zk_batched_sumcheck(
        s2_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s2_proof));

    // Stage 3
    let mut s3 = sm.build_stage3_instances(prover, &lut_data, sat_diff_lookup_bits, &r_exp_indices);
    let s3_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        s3.iter_mut().map(|v| &mut **v as _).collect();
    let s3_proof = run_zk_batched_sumcheck(
        s3_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s3_proof));

    // Stage 4
    let mut s4 = sm.build_stage4_instances(prover, &lut_data, &sat_diff_indices);
    let s4_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        s4.iter_mut().map(|v| &mut **v as _).collect();
    let s4_proof = run_zk_batched_sumcheck(
        s4_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s4_proof));

    inter_stage_commitments.insert(node.idx, vec![cache_r_com, cache_r_exp_com]);
}

/// Prove GatherLarge with ZK: default flow (eval reduction, execution + shout one-hot).
#[expect(clippy::too_many_arguments)]
fn prove_gather_large_zk(
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
    use crate::onnx_proof::ops::gather::{large::GatherRaEncoding, GatherParams, GatherProver};

    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // Execution sumcheck
    let params = GatherParams::new(
        node.clone(),
        &model.graph,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut sc = GatherProver::initialize(
        &prover.trace,
        params,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let exec_proof = run_zk_sumcheck(
        &mut sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, exec_proof));

    // One-hot batch
    let encoding = GatherRaEncoding::new(node);
    let lookup_indices =
        crate::onnx_proof::ops::gather::large::gather_lookup_indices(node, &prover.trace);
    let [mut ra, mut hw, mut b] = joltworks::subprotocols::shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let oh_proof = run_zk_batched_sumcheck(
        vec![&mut *ra, &mut *hw, &mut *b],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, oh_proof));
}

/// Prove GatherSmall with ZK: default flow (eval reduction, execution, HB+bool batch, HW).
#[expect(clippy::too_many_arguments)]
fn prove_gather_small_zk(
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
    use crate::onnx_proof::ops::gather::{
        small::{build_stage2_provers, build_stage3_prover},
        GatherParams, GatherProver,
    };

    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // Stage 1: Execution
    let params = GatherParams::new(
        node.clone(),
        &model.graph,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut sc = GatherProver::initialize(
        &prover.trace,
        params,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let exec_proof = run_zk_sumcheck(
        &mut sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, exec_proof));

    // Stage 2: HammingBooleanity + Booleanity (batched)
    let (mut hb_sc, mut bool_sc) = build_stage2_provers::<F>(node, prover);
    let s2_proof = run_zk_batched_sumcheck(
        vec![
            &mut hb_sc as &mut dyn SumcheckInstanceProver<F, T>,
            &mut bool_sc,
        ],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s2_proof));

    // Stage 3: HammingWeight (single)
    let mut hw_sc = build_stage3_prover::<F>(node, prover);
    let s3_proof = run_zk_sumcheck(
        &mut hw_sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, s3_proof));
}

/// Prove Relu with ZK: default flow (eval reduction, then execution + one-hot).
#[expect(clippy::too_many_arguments)]
fn prove_relu_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
) {
    use crate::onnx_proof::op_lookups::{OpLookupEncoding, OpLookupProvider};
    use joltworks::lookup_tables::relu::ReluTable;

    // 1. Eval reduction (standard, before sumchecks)
    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // 2. Execution sumcheck (ps_shout read-raf)
    let provider = OpLookupProvider::new(node.clone());
    let (mut exec_sc, lookup_indices) = provider
        .read_raf_prove::<F, T, ReluTable<{ common::consts::XLEN }>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let exec_proof = run_zk_sumcheck(
        &mut exec_sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, exec_proof));

    // 3. One-hot batch (Ra, HammingWeight, Booleanity)
    let encoding = OpLookupEncoding::new(node);
    let [mut ra, mut hw, mut b] = joltworks::subprotocols::shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let oh_proof = run_zk_batched_sumcheck(
        vec![&mut *ra, &mut *hw, &mut *b],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, oh_proof));
}

/// Prove the fused-rescaling pre-stages for a fused arithmetic node (einsum /
/// `Mul` / `Square` / `Cube` with `scale > 0`), mirroring
/// `fused_rebase::prove_pre`: append the `RescaleRemainder` advice, then
/// discharge `output = SatClamp(rescaled)` via the 64-bit clamp read-raf
/// lookup and its one-hot checks. Must run *before* the operator's arithmetic
/// sumcheck so `fused_rebase::fused_input_claim` can read the
/// `ClampAcc`/`RescaleRemainder` advices. Both advice appends are
/// transcript-quiet in ZK mode; their values enter the proof via the baked
/// initial claims of the following stages (prover-supplied, like every
/// chain-start initial claim in this pipeline — the same treatment the Tanh
/// `DummyClampedTanhInput` advice gets), not as committed BlindFold output
/// claims. Binding them with a real `InputClaimConstraint` is future work
/// alongside the other baked-claim bindings.
fn prove_fused_rebase_pre_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
) -> atlas_onnx_tracer::tensor::Tensor<i32> {
    use crate::onnx_proof::clamp_lookups::{is_scalar, ClampEncoding, ClampLookupProvider};

    // Scalar fused nodes open `rescaled`/`R` in the clear and are checked by
    // the verifier directly (`fused_rebase::verify_post`); that cleartext
    // check has no ZK counterpart yet. Fail loudly rather than prove an
    // unbound clamp.
    if is_scalar(node) {
        unimplemented!("ZK proving not yet implemented for scalar fused-rescale nodes");
    }

    // Quotient + remainder from one accumulation pass; the remainder is
    // returned so `prove_fused_rebase_post_zk` can reuse it.
    let crate::onnx_proof::fused_rebase::RebaseIntermediates {
        quotient,
        remainder,
    } = crate::onnx_proof::fused_rebase::rebase_intermediates(node, &prover.trace);

    // Remainder advice at the reduced output point.
    crate::onnx_proof::fused_rebase::cache_remainder_prove(node, prover, &remainder);

    // Clamp read-raf lookup: output = SatClamp(acc); acc appended as ClampAcc.
    let provider = ClampLookupProvider::new(node.clone());
    let (mut exec_sc, lookup_indices) = provider.read_raf_prove::<F, T>(
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
        Some(quotient),
    );
    let exec_proof = run_zk_sumcheck(
        &mut exec_sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, exec_proof));

    // Clamp one-hot batch (Ra, HammingWeight, Booleanity) over `ClampRaD`.
    let encoding = ClampEncoding::new(node);
    let [mut ra, mut hw, mut b] = joltworks::subprotocols::shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let oh_proof = run_zk_batched_sumcheck(
        vec![&mut *ra, &mut *hw, &mut *b],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, oh_proof));

    remainder
}

/// Prove the fused-rescaling post-stages, mirroring
/// `fused_rebase::prove_remainder_rc`: the remainder range check
/// `R ∈ [0, 2^S)` plus its one-hot checks over `RescaleRemainderRaD`. Runs
/// *after* the operator's arithmetic sumcheck. `remainder` is the padded
/// tensor returned by `prove_fused_rebase_pre_zk`.
fn prove_fused_rebase_post_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
    remainder: &atlas_onnx_tracer::tensor::Tensor<i32>,
) {
    use crate::onnx_proof::fused_rebase::{
        rebase_bits, remainder_lookup_bits, RescaleRemainderRCProvider, RescaleRemainderRaEncoding,
    };
    use joltworks::subprotocols::identity_range_check::identity_rangecheck_prover;

    let bits = rebase_bits(&node.operator).expect("fused op");
    let lookup_bits = remainder_lookup_bits(remainder, bits);
    let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();

    // Remainder identity range check.
    let rc_provider = RescaleRemainderRCProvider::new(node.clone(), bits);
    let mut rc = identity_rangecheck_prover(&rc_provider, lookup_bits, &mut prover.accumulator);
    let rc_proof = run_zk_sumcheck(
        &mut rc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, rc_proof));

    // Remainder one-hot batch over `RescaleRemainderRaD`.
    let encoding = RescaleRemainderRaEncoding::new(node.idx, bits);
    let [mut ra, mut hw, mut boolean] = joltworks::subprotocols::shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let ra_proof = run_zk_batched_sumcheck(
        vec![&mut *ra, &mut *hw, &mut *boolean],
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, ra_proof));
}

/// Prove Rsqrt with ZK: custom flow (execution from transcript, eval reduction, range checks).
#[expect(clippy::too_many_arguments)]
fn prove_rsqrt_zk(
    node: &atlas_onnx_tracer::node::ComputationNode,
    prover: &mut Prover<F, T>,
    pedersen_gens: &PedersenGenerators<C>,
    blindfold_accumulator: &mut joltworks::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    stage_configs: &mut Vec<StageConfig>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    eval_reduction_h_commitments: &mut BTreeMap<usize, joltworks::curve::Bn254G1>,
    zk_sumcheck_proofs: &mut Vec<NodeZkProof>,
) {
    use crate::onnx_proof::{
        ops::rsqrt::{RsqrtParams, RsqrtProver},
        range_checking::{
            range_check_operands::{RiRangeCheckOperands, RsRangeCheckOperands},
            RangeCheckEncoding, RangeCheckProvider,
        },
    };
    use joltworks::lookup_tables::unsigned_less_than::UnsignedLessThanTable;

    // 1. Execution sumcheck (r_node_output + gamma from transcript)
    let params = RsqrtParams::<F>::new(node.clone(), &mut prover.transcript);
    let mut sc = RsqrtProver::initialize(&prover.trace, params);
    let exec_proof = run_zk_sumcheck(
        &mut sc,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, exec_proof));

    // 2. Eval reduction
    prove_zk_eval_reduction(
        node,
        prover,
        pedersen_gens,
        eval_reduction_proofs,
        eval_reduction_h_commitments,
    );

    // 3. Two range checks (Ri and Rs) batched together
    let div_rc = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let (div_rc_sc, div_rc_idx) = div_rc
        .read_raf_prove::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let sqrt_rc = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let (sqrt_rc_sc, sqrt_rc_idx) = sqrt_rc
        .read_raf_prove::<F, T, UnsignedLessThanTable<{ common::consts::XLEN }>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let mut rc_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
        vec![Box::new(div_rc_sc), Box::new(sqrt_rc_sc)];
    let rc_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        rc_instances.iter_mut().map(|v| &mut **v as _).collect();
    let rc_proof = run_zk_batched_sumcheck(
        rc_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, rc_proof));

    // 4. Six one-hot instances (3 per range check) batched together
    let div_enc = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
    let [div_ra, div_hw, div_bool] = joltworks::subprotocols::shout::ra_onehot_provers(
        &div_enc,
        &div_rc_idx,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let sqrt_enc = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
    let [sqrt_ra, sqrt_hw, sqrt_bool] = joltworks::subprotocols::shout::ra_onehot_provers(
        &sqrt_enc,
        &sqrt_rc_idx,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut oh_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
        vec![div_ra, div_hw, div_bool, sqrt_ra, sqrt_hw, sqrt_bool];
    let oh_refs: Vec<&mut dyn SumcheckInstanceProver<F, T>> =
        oh_instances.iter_mut().map(|v| &mut **v as _).collect();
    let oh_proof = run_zk_batched_sumcheck(
        oh_refs,
        prover,
        blindfold_accumulator,
        stage_configs,
        pedersen_gens,
    );
    zk_sumcheck_proofs.push((node.idx, oh_proof));
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
    use crate::{
        onnx_proof::{
            ops::div::{DivParams, DivProver},
            range_checking::{
                range_check_operands::DivRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
            },
        },
        utils::opening_access::AccOpeningAccessor,
    };
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
    // Bind public inputs into the transcript before any challenge (issue #230);
    // mirrored in `verify_zk_with_pcs_capture_and_extract`.
    super::append_inputs_to_transcript(&mut prover.transcript, &io);
    // Mirror the verifier's `VerifierOpeningAccumulator::new_zk` default: the
    // ZK pipeline hides cleartext opening claims behind Pedersen commitments
    // (emitted by BatchedSumcheck::prove_zk), so the prover must NOT append
    // raw claims to the transcript in `append_virtual` / `append_dense`. Code
    // paths that intentionally publish a claim in the clear (e.g. Softmax's
    // public auxiliary vectors) toggle this flag off and back on.
    prover.accumulator.zk_mode = true;

    // The ZK pipeline still proves `Add`/`Sub`/`Sum` un-clamped (the saturating
    // clamp lookup is not yet wired into `prove_zk` for them; per-node provers
    // are the un-clamped `AddProver`/`SubProver`/`SumAxisProver`). Those provers
    // never open the clamp one-hot decomposition (`ClampRaD`), so committing it
    // would leave it without an opening-reduction sumcheck. Filter those out so
    // the committed set matches what the zk sumchecks actually open. Fused
    // rescale ops (einsum/Mul/Square/Cube, `fuses_rebase`) DO open `ClampRaD`
    // in ZK via `prove_fused_rebase_pre_zk`, so theirs stay committed. (The
    // non-ZK path commits and opens `ClampRaD` for all clamped ops as usual.)
    let poly_map: std::collections::BTreeMap<
        common::CommittedPoly,
        joltworks::poly::multilinear_polynomial::MultilinearPolynomial<F>,
    > = ONNXProof::<F, T, PCS>::polynomial_map(pp.model(), &prover.trace)
        .into_iter()
        .filter(|(poly, _)| match poly {
            common::CommittedPoly::ClampRaD(node_idx, _) => {
                crate::onnx_proof::fused_rebase::fuses_rebase(&pp.model()[*node_idx].operator)
            }
            _ => true,
        })
        .collect();
    let commitments = ONNXProof::<F, T, PCS>::commit_to_polynomials(&poly_map, &pp.generators);
    for commitment in &commitments {
        prover.transcript.append_serializable(commitment);
    }
    // The output claim is a public scalar derived from IO; both prover and
    // verifier (`verify_zk` does `transcript.append_scalar(&expected_output_claim)`)
    // append it in the clear. Toggle prover zk_mode off so its append fires.
    {
        let saved = prover.accumulator.zk_mode;
        prover.accumulator.zk_mode = false;
        ONNXProof::<F, T, PCS>::output_claim(&mut prover);
        prover.accumulator.zk_mode = saved;
    }

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
    let mut inter_stage_commitments: BTreeMap<usize, Vec<joltworks::curve::Bn254G1>> =
        BTreeMap::new();
    let mut auxiliary_claims: BTreeMap<joltworks::poly::opening_proof::OpeningId, F> =
        BTreeMap::new();

    for (_, node) in nodes.iter().rev() {
        // Match is exhaustive: adding a new Operator variant fails to compile
        // until it is routed to a specific prove flow or the default branch.
        match &node.operator {
            Operator::SoftmaxLastAxis(_) => prove_softmax_zk(
                node,
                &mut prover,
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
                &mut inter_stage_commitments,
                &mut auxiliary_claims,
            ),
            Operator::GatherLarge(_) => prove_gather_large_zk(
                node,
                &mut prover,
                pp.shared.model(),
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            Operator::GatherSmall(_) => prove_gather_small_zk(
                node,
                &mut prover,
                pp.shared.model(),
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            Operator::ReLU(_) => prove_relu_zk(
                node,
                &mut prover,
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            Operator::Sigmoid(_) | Operator::Tanh(_) | Operator::Erf(_) => {
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
                )
            }
            Operator::Cos(_) | Operator::Sin(_) => prove_cos_sin_zk(
                node,
                &mut prover,
                pp.shared.model(),
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            Operator::Rsqrt(_) => prove_rsqrt_zk(
                node,
                &mut prover,
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            Operator::Div(_) => prove_div_zk(
                node,
                &mut prover,
                pedersen_gens,
                &mut blindfold_accumulator,
                &mut stage_configs,
                &mut eval_reduction_proofs,
                &mut eval_reduction_h_commitments,
                &mut zk_sumcheck_proofs,
            ),
            // No-sumcheck ops that don't register openings on inputs.
            // Constants and Inputs have no inputs to cache; their non-zk
            // `prove` is a no-op apart from a sanity read of `get_reduced_opening`.
            Operator::Input(_) | Operator::Constant(_) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
            }
            // No-sumcheck ops that DO register openings on their input(s).
            // The non-zk path runs eval reduction then the operator's `prove`,
            // which appends a `Target::Input(0)` opening so the producer's
            // openings list is non-empty when it's processed. Mirror that
            // here, otherwise default-branch producers (e.g. ScalarConstDiv
            // feeding a Broadcast) hit empty openings and the subsequent
            // sumcheck setup panics.
            // No-sumcheck ops that register an opening on their input. Each
            // op's prove flow appends `Target::Input(0)` via `append_nodeio`.
            // With `prover.accumulator.zk_mode = true`, those calls are
            // transcript-quiet on the prover; the verifier (also zk_mode on)
            // is symmetric, so no auxiliary_claims dance is needed here.
            Operator::Broadcast(_) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
                if prover
                    .accumulator
                    .reduced_evaluations
                    .contains_key(&node.idx)
                {
                    use crate::onnx_proof::ops::broadcast::{BroadcastParams, BroadcastProver};
                    let params = BroadcastParams::new(node.clone(), &prover.accumulator);
                    let bp = BroadcastProver::initialize(&prover.trace, params);
                    bp.prove(&mut prover.accumulator, &mut prover.transcript);
                }
            }
            Operator::MoveAxis(_) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
                if prover
                    .accumulator
                    .reduced_evaluations
                    .contains_key(&node.idx)
                {
                    use crate::onnx_proof::ops::moveaxis::{MoveAxisParams, MoveAxisProver};
                    let params = MoveAxisParams::new(node.clone(), &prover.accumulator);
                    let mp = MoveAxisProver::initialize(params);
                    mp.prove(&mut prover.accumulator, &mut prover.transcript);
                }
            }
            Operator::Identity(op) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
                if prover
                    .accumulator
                    .reduced_evaluations
                    .contains_key(&node.idx)
                {
                    use crate::onnx_proof::ops::OperatorProofTrait;
                    let _ = OperatorProofTrait::<F, T>::prove(op, node, &mut prover);
                }
            }
            Operator::IsNan(op) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
                if prover
                    .accumulator
                    .reduced_evaluations
                    .contains_key(&node.idx)
                {
                    use crate::onnx_proof::ops::OperatorProofTrait;
                    let _ = OperatorProofTrait::<F, T>::prove(op, node, &mut prover);
                }
            }
            Operator::Clamp(op) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );
                if prover
                    .accumulator
                    .reduced_evaluations
                    .contains_key(&node.idx)
                {
                    use crate::onnx_proof::ops::OperatorProofTrait;
                    let _ = OperatorProofTrait::<F, T>::prove(op, node, &mut prover);
                }
            }
            // MeanOfSquares is not yet implemented in `create_prover_instance`
            // (it falls into the panic catch-all). Surface that here as an
            // explicit `unimplemented!` so the dispatch reflects the gap.
            Operator::MeanOfSquares(_) => {
                unimplemented!("ZK proving not yet implemented for MeanOfSquares");
            }
            // Default flow: eval reduction first, then execution sumcheck.
            Operator::Add(_)
            | Operator::Sub(_)
            | Operator::Neg(_)
            | Operator::Mul(_)
            | Operator::Square(_)
            | Operator::Cube(_)
            | Operator::And(_)
            | Operator::Iff(_)
            | Operator::Concat(_)
            | Operator::Reshape(_)
            | Operator::Slice(_)
            | Operator::ScalarConstDiv(_)
            | Operator::Einsum(_)
            | Operator::Sum(_) => {
                prove_zk_eval_reduction(
                    node,
                    &mut prover,
                    pedersen_gens,
                    &mut eval_reduction_proofs,
                    &mut eval_reduction_h_commitments,
                );

                // Fused-rescale ops (einsum/Mul/Square/Cube with scale > 0)
                // mirror the non-zk `fused_rebase` flow: remainder advice +
                // clamp stages before the arithmetic sumcheck (its input
                // claim reads the `ClampAcc`/`RescaleRemainder` advices),
                // remainder range-check stages after.
                let fused = crate::onnx_proof::fused_rebase::fuses_rebase(&node.operator);
                let fused_remainder = fused.then(|| {
                    prove_fused_rebase_pre_zk(
                        node,
                        &mut prover,
                        pedersen_gens,
                        &mut blindfold_accumulator,
                        &mut stage_configs,
                        &mut zk_sumcheck_proofs,
                    )
                });

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

                if let Some(remainder) = &fused_remainder {
                    prove_fused_rebase_post_zk(
                        node,
                        &mut prover,
                        pedersen_gens,
                        &mut blindfold_accumulator,
                        &mut stage_configs,
                        &mut zk_sumcheck_proofs,
                        remainder,
                    );
                }
            }
        }
    }

    // y_com for batch opening (no-op when poly_map is empty).
    //
    // ZK path: route the batched-opening sumcheck through `BatchedSumcheck::prove_zk`
    // (via `prove_batch_opening_sumcheck_zk`) so its round polynomials are Pedersen-
    // committed and the stage data flows into `BlindFoldAccumulator::stage_data`,
    // letting the BlindFold R1CS prove sumcheck-validity instead of leaking
    // per-poly claims through the transcript. Output-claim binding to `y_com`
    // (eval_commitment_gens machinery) is wired in subsequent steps; the
    // current `Opening<F>::output_claim_constraint` returns `None`, so the
    // BlindFold R1CS does not yet enforce the sumcheck output. See
    // wiki/jolt-atlas/book/src/underway/batched-opening-sound-verifier.md.
    // Captured for the BlindFold `extra_constraint` that ties the y_com
    // commitment to `sum gamma_i * y_P_i` (where y_P_i = P_i(r_sumcheck)).
    // Empty when there are no committed polynomials (poly_map is empty).
    let mut joint_eval_commitment: Option<joltworks::curve::Bn254G1> = None;
    let mut batch_opening_extra_constraint: Option<
        joltworks::subprotocols::blindfold::output_constraint::OutputClaimConstraint,
    > = None;
    let mut batch_opening_extra_witness: Option<
        joltworks::subprotocols::blindfold::witness::ExtraConstraintWitness<F>,
    > = None;
    // Step 8 (PCS-binding): PCS opening proof for the joint polynomial at
    // r_sumcheck. Binds `bundle.commitments` to the joint evaluation.
    let mut joint_opening_proof: Option<<PCS as CommitmentScheme>::Proof> = None;
    let mut joint_claim_value: Option<F> = None;
    let mut joint_opening_point: Option<Vec<<F as joltworks::field::JoltField>::Challenge>> = None;
    let mut batch_opening_zk_sumcheck = None;
    if !poly_map.is_empty() {
        prover.accumulator.prepare_for_sumcheck(&poly_map);
        let mut rng = rand::thread_rng();
        let (zk_acc_proof, r_sumcheck_acc) = prover
            .accumulator
            .prove_batch_opening_sumcheck_zk::<T, C, _>(
                &mut blindfold_accumulator,
                &mut stage_configs,
                pedersen_gens,
                &mut rng,
                &mut prover.transcript,
            );
        batch_opening_zk_sumcheck = Some(zk_acc_proof);
        let state = prover
            .accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc, &mut prover.transcript);

        // Compute effective coefficients = gamma_i * lagrange_i, where
        // lagrange_i is the product over the high-order r_sumcheck variables
        // that polynomial i does not depend on. This matches the existing
        // non-zk `compute_joint_claim` so that
        //   joint_poly(r_sumcheck) = sum_i (gamma_i * lagrange_i) * y_P_i
        // i.e. the materialized RLC of `poly_map` (which pads short polys
        // with zeros) evaluated at `r_sumcheck` equals the algebraic
        // combination of per-poly evaluations weighted by the lagrange term.
        // Polynomials with `num_rounds == max_num_rounds` have empty
        // r_high, so lagrange_i = 1 (no effect).
        let max_num_rounds = state.r_sumcheck.len();
        use joltworks::field::OptimizedMul;
        let num_rounds_per_poly: Vec<usize> = state
            .polynomials
            .iter()
            .map(|poly| {
                let p = poly_map
                    .get(poly)
                    .expect("polynomial in state.polynomials must be in poly_map");
                match p {
                    joltworks::poly::multilinear_polynomial::MultilinearPolynomial::OneHot(oh) => {
                        (oh.K * oh.nonzero_indices.len()).log_2()
                    }
                    _ => p.original_len().log_2(),
                }
            })
            .collect();
        let effective_coeffs: Vec<F> = state
            .gamma_powers
            .iter()
            .zip(num_rounds_per_poly.iter())
            .map(|(g, num_rounds)| {
                let r_slice = &state.r_sumcheck[..max_num_rounds - num_rounds];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - *r).product();
                g.mul_01_optimized(lagrange_eval)
            })
            .collect();

        let joint_claim: F = effective_coeffs
            .iter()
            .zip(state.sumcheck_claims.iter())
            .map(|(c, claim)| *c * claim)
            .sum();
        let y_blinding = F::random(&mut rand::thread_rng());
        let y_com = pedersen_gens.commit(&[joint_claim], &y_blinding);
        prover.transcript.append_serializable(&y_com);

        // Build the BlindFold `extra_constraint`:
        //   joint_claim = sum_i (gamma_i * lagrange_i) * y_P_i
        // where each y_P_i is the opening registered by
        // `OpeningProofReductionSumcheckProver::cache_openings` at
        // (CommittedPoly_i, SumcheckId::BlindFoldBatchOpening).
        use joltworks::{
            poly::opening_proof::{OpeningId, SumcheckId as SId},
            subprotocols::blindfold::{
                output_constraint::{OutputClaimConstraint, ValueSource},
                witness::ExtraConstraintWitness,
            },
        };
        let opening_ids: Vec<OpeningId> = state
            .polynomials
            .iter()
            .map(|poly| OpeningId::new(*poly, SId::BlindFoldBatchOpening))
            .collect();
        let terms: Vec<(ValueSource, ValueSource)> = opening_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (ValueSource::Challenge(i), ValueSource::Opening(*id)))
            .collect();
        batch_opening_extra_constraint = Some(OutputClaimConstraint::linear(terms));
        batch_opening_extra_witness = Some(ExtraConstraintWitness {
            output_value: joint_claim,
            blinding: y_blinding,
            challenge_values: effective_coeffs.clone(),
            opening_values: state.sumcheck_claims.clone(),
        });
        joint_eval_commitment = Some(y_com);

        // Keep the legacy OpeningProofData hook populated for any consumer
        // that still reads it (none currently, but cheap).
        blindfold_accumulator.set_opening_proof_data(
            joltworks::subprotocols::blindfold::OpeningProofData {
                opening_ids,
                constraint_coeffs: effective_coeffs.clone(),
                joint_claim,
                y_blinding,
            },
        );

        // Step 8: produce the PCS opening proof on the joint polynomial.
        // Materialise the gamma-weighted RLC of poly_map and prove its opening
        // at r_sumcheck. The verifier homomorphically combines `bundle.commitments`
        // with `gamma_powers` to reconstruct the same joint commitment.
        let r_sumcheck_acc: Vec<<F as joltworks::field::JoltField>::Challenge> =
            state.r_sumcheck.clone();
        let joint_poly =
            joltworks::poly::rlc_polynomial::build_materialized_rlc(&state.gamma_powers, &poly_map);
        let opening_proof = PCS::prove(
            &pp.generators,
            &joint_poly,
            &r_sumcheck_acc,
            None,
            &mut prover.transcript,
        );
        joint_opening_proof = Some(opening_proof);
        joint_claim_value = Some(joint_claim);
        joint_opening_point = Some(r_sumcheck_acc);
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
    // Plumb the batched-opening extra constraint (if any) into both witness
    // and R1CS so the BlindFold R1CS enforces
    //   joint_claim = sum_i gamma_i * y_P_i   AND   y_com = g * joint_claim + h * y_blinding.
    // The latter is enforced via BlindFold's `eval_commitment_gens` check; the
    // y_com is placed in `RelaxedR1CSInstance::eval_commitments` below.
    let mut extra_constraints_for_r1cs: Vec<_> =
        batch_opening_extra_constraint.clone().into_iter().collect();
    let mut extra_constraint_witnesses: Vec<_> =
        batch_opening_extra_witness.clone().into_iter().collect();

    // Public-node opening binding: for each opening on a `Constant`/`Input`
    // node, add an R1CS extra-constraint that ties the BlindFold witness's
    // `Opening<F>` value to the verifier-known public value
    // `public_mle(opening_point)`. Mechanism (mirrors Jolt's
    // `ValueSource::Constant` baked binding for public data, adapted to
    // arbitrary field values via a Challenge slot):
    //   - Constraint: `output_var = Opening(id) - Challenge(0)`.
    //   - Challenge value (baked, public): `public_mle(opening_point)`.
    //   - Witness: `output_value = 0`, `blinding = 0`.
    //   - `y_com = Pedersen([0], 0) = identity_G1`.
    // The verifier independently computes the expected value, checks the
    // baked challenge matches it, checks the corresponding y_com is the
    // identity (which combined with BlindFold's R1CS algebra forces
    // `output_var = 0` and `blinding = 0`, hence `Opening = expected`).
    let mut public_node_binding_eval_commitments: Vec<joltworks::curve::Bn254G1> = Vec::new();
    {
        use common::VirtualPoly;
        use joltworks::{
            poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            subprotocols::blindfold::{
                output_constraint::{OutputClaimConstraint, ValueSource},
                witness::ExtraConstraintWitness,
            },
        };
        let model = pp.model();
        // BTreeMap iteration is deterministic; the verifier-side iteration
        // must use the same order so binding-constraint indices match.
        for (opening_id, (opening_point, opening_value)) in prover.accumulator.openings.iter() {
            let Some(VirtualPoly::NodeOutput(node_idx)) = opening_id.virtual_poly() else {
                continue;
            };
            // Only bind openings where the consumer opens the raw NodeOutput
            // directly (i.e., the cleartext value equals `node_mle(opening_point)`).
            // Internal protocols (Raf, RaVirtualization, RamHammingBooleanity,
            // ...) open transformed representations under the same NodeOutput
            // polynomial-id, so the witness value there is not the raw MLE
            // evaluation and a binding constraint against `public_mle(opening_point)`
            // would (correctly) reject. Those internal openings are bound by
            // their own protocol-specific algebraic constraints.
            if !matches!(
                opening_id.sumcheck,
                joltworks::poly::opening_proof::SumcheckId::NodeExecution(_)
            ) {
                continue;
            }
            let node = &model[node_idx];
            let expected_value: F = match &node.operator {
                atlas_onnx_tracer::ops::Operator::Constant(c) => {
                    MultilinearPolynomial::from(c.0.clone().padded_next_power_of_two())
                        .evaluate(&opening_point.r)
                }
                atlas_onnx_tracer::ops::Operator::Input(_) => {
                    let input_pos = io
                        .input_indices
                        .iter()
                        .position(|&i| i == node_idx)
                        .expect("Input node must be in io.input_indices");
                    MultilinearPolynomial::from(io.inputs[input_pos].padded_next_power_of_two())
                        .evaluate(&opening_point.r)
                }
                _ => continue,
            };
            let constraint = OutputClaimConstraint::linear(vec![
                (ValueSource::Constant(1), ValueSource::Opening(*opening_id)),
                (ValueSource::Constant(-1), ValueSource::Challenge(0)),
            ]);
            let witness = ExtraConstraintWitness {
                output_value: F::zero(),
                blinding: F::zero(),
                challenge_values: vec![expected_value],
                opening_values: vec![*opening_value],
            };
            let y_com = pedersen_gens.commit(&[F::zero()], &F::zero());
            extra_constraints_for_r1cs.push(constraint);
            extra_constraint_witnesses.push(witness);
            public_node_binding_eval_commitments.push(y_com);
        }
    }
    let blindfold_witness = BlindFoldWitness::with_extra_constraints(
        initial_claims,
        all_stages,
        extra_constraint_witnesses,
    );
    let baked = BakedPublicInputs::from_witness(&blindfold_witness, &stage_configs);
    let builder = VerifierR1CSBuilder::<F>::new_with_extra(
        &stage_configs,
        &extra_constraints_for_r1cs,
        &baked,
        Vec::new(),
        std::collections::BTreeMap::new(),
    );
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

    let mut eval_commitments: Vec<joltworks::curve::Bn254G1> =
        joint_eval_commitment.into_iter().collect();
    eval_commitments.extend(public_node_binding_eval_commitments);
    let (real_instance, real_witness) = RelaxedR1CSInstance::<F, C>::new_non_relaxed(
        &witness,
        r1cs.num_constraints,
        hyrax_C,
        round_commitments,
        Vec::new(),
        noncoeff_row_commitments,
        eval_commitments,
        w_row_blindings,
    );

    // `eval_commitment_gens = Some((g, h))` activates BlindFold's
    // eval-commitment check (protocol.rs:349-362): for every extra constraint
    // it verifies the folded `eval_commitment_i == g * folded_output_i + h *
    // folded_blinding_i`. With the batched-opening extra constraint in place,
    // this is what binds `y_com` to the BlindFold R1CS witness.
    let eval_commitment_gens = if !extra_constraints_for_r1cs.is_empty() {
        Some((
            pedersen_gens.message_generators[0],
            pedersen_gens.blinding_generator,
        ))
    } else {
        None
    };
    let bf_prover = BlindFoldProver::new(pedersen_gens, &r1cs, eval_commitment_gens);
    let mut bf_transcript = T::new(b"BlindFold_onnx");
    let blindfold_proof = bf_prover.prove(&real_instance, &real_witness, &z, &mut bf_transcript);

    let blindfold_verifier_input = BlindFoldVerifierInput {
        round_commitments: real_instance.round_commitments.clone(),
        output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
        eval_commitments: real_instance.eval_commitments.clone(),
    };

    let zk_sumcheck_num_instances: Vec<usize> = stage_data_vec
        .iter()
        .map(|sd| sd.batching_coefficients.len())
        .collect();

    // Collect cleartext reduced-eval claims for Constants and Inputs. The
    // verifier mirrors `Constant::verify` / `Input::verify` (non-zk) by locally
    // evaluating the public tensor's MLE at `r_reduced` and checking equality
    // against this value. See `verify_zk` for the matching check.
    let mut public_node_reduced_claims: BTreeMap<usize, F> = BTreeMap::new();
    for (_, node) in pp.model().nodes().iter() {
        if matches!(
            node.operator,
            atlas_onnx_tracer::ops::Operator::Constant(_)
                | atlas_onnx_tracer::ops::Operator::Input(_)
        ) {
            if let Some(reduced) = prover.accumulator.reduced_evaluations.get(&node.idx) {
                public_node_reduced_claims.insert(node.idx, reduced.claim);
            }
        }
    }

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
        zk_sumcheck_num_instances,
        inter_stage_commitments,
        auxiliary_claims,
        joint_opening_proof,
        joint_claim: joint_claim_value,
        joint_opening_point,
        batch_opening_zk_sumcheck,
        public_node_reduced_claims,
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
///
/// Snapshot of the inputs to the final HyperKZG verification step inside
/// [`verify_zk`], captured by [`verify_zk_with_pcs_capture`] right before
/// the call to `PCS::verify`. External tools (e.g., zkARc's pi-prime
/// extractor) use this to derive the verifier-side MSM trace at the exact
/// transcript state that the verifier reaches.
#[derive(Clone)]
pub struct PcsVerifyCapture {
    /// RLC of committed-polynomials commitments
    pub joint_commitment:
        <PCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment,
    /// PCS proof
    pub opening_proof:
        <PCS as joltworks::poly::commitment::commitment_scheme::CommitmentScheme>::Proof,
    /// Evaluation point for the PCS opening proof
    pub opening_point: Vec<<F as JoltField>::Challenge>,
    /// Claimed evaluation of the joint polynomial at `opening_point`
    pub joint_claim: F,
    /// Fiat shamir transcript
    pub transcript: T,
}

/// Verify a ZK proof bundle for an ONNX model execution.
pub fn verify_zk(
    bundle: &ZkProofBundle,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    pedersen_gens: &PedersenGenerators<C>,
) -> Result<(), ProofVerifyError> {
    verify_zk_with_pcs_capture(bundle, pp, io, pedersen_gens, None)
}

/// Test-only helper used by the soundness regression test. Runs the
/// `verify_zk` flow but, at the Input/Constant per-op check and the
/// public-node binding section, writes what the verifier WOULD compute
/// into `out` instead of comparing/rejecting. The function then returns
/// `Ok(())` regardless of whether the bundle would otherwise verify.
///
/// A real attacker doesn't need this helper — the values are a
/// deterministic function of the bundle and the public model — but
/// exposing it as a small public hook lets the test reproduce the
/// patching step concisely without re-implementing the transcript walk.
#[cfg(test)]
pub fn extract_verifier_expected_public_claims(
    bundle: &ZkProofBundle,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    pedersen_gens: &PedersenGenerators<C>,
    out: &mut BTreeMap<usize, F>,
) -> Result<(), ProofVerifyError> {
    verify_zk_with_pcs_capture_and_extract(bundle, pp, io, pedersen_gens, None, Some(out))
}

/// Same as [`verify_zk`] but exposes an optional slot for capturing the
/// inputs to the final HyperKZG verification. When `capture` is `Some`,
/// the slot is filled in right before `PCS::verify` runs; the verifier
/// then continues normally. Used by external auto-builders to extract
/// the verifier's MSM trace without having to mirror the entire
/// transcript walk.
pub fn verify_zk_with_pcs_capture(
    bundle: &ZkProofBundle,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    pedersen_gens: &PedersenGenerators<C>,
    capture: Option<&mut Option<PcsVerifyCapture>>,
) -> Result<(), ProofVerifyError> {
    verify_zk_with_pcs_capture_and_extract(bundle, pp, io, pedersen_gens, capture, None)
}

/// Internal: same as [`verify_zk_with_pcs_capture`] plus an optional
/// extract slot. When `extract_public_claims` is `Some`, the
/// Input/Constant per-op check writes the verifier-computed
/// `public_mle(r_reduced)` value into the map and SKIPS the equality
/// comparison, and the public-node binding section is bypassed
/// entirely. The function then returns `Ok(())` regardless of what the
/// bundle contains — used by the soundness regression test to recover
/// what the verifier would compute given a malicious bundle so that
/// `bundle.public_node_reduced_claims` can be patched before the real
/// verify is invoked.
fn verify_zk_with_pcs_capture_and_extract(
    bundle: &ZkProofBundle,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    pedersen_gens: &PedersenGenerators<C>,
    capture: Option<&mut Option<PcsVerifyCapture>>,
    mut extract_public_claims: Option<&mut BTreeMap<usize, F>>,
) -> Result<(), ProofVerifyError> {
    use joltworks::poly::opening_proof::VerifierOpeningAccumulator;

    let model = pp.shared.model();
    let mut accumulator = VerifierOpeningAccumulator::<F>::new_zk();
    let mut transcript = T::new(b"ONNXProof");

    // 0. Bind public inputs into the transcript before any challenge (issue
    //    #230); must mirror `prove_zk`'s first append.
    super::append_inputs_to_transcript(&mut transcript, io);

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
        // Match is exhaustive: adding a new Operator variant fails to compile
        // until it is routed to a specific verify flow or the default branch.
        match &node.operator {
            Operator::SoftmaxLastAxis(_) => verify_softmax_zk(
                node,
                pp,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::GatherLarge(_) => verify_gather_large_zk(
                node,
                model,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::GatherSmall(_) => verify_gather_small_zk(
                node,
                model,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::ReLU(_) => verify_relu_zk(
                node,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::Sigmoid(_) | Operator::Tanh(_) | Operator::Erf(_) => {
                verify_neural_teleport_zk(
                    node,
                    model,
                    bundle,
                    &mut accumulator,
                    &mut transcript,
                    &mut zk_proof_idx,
                )?
            }
            Operator::Cos(_) | Operator::Sin(_) => verify_cos_sin_zk(
                node,
                model,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::Rsqrt(_) => verify_rsqrt_zk(
                node,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            Operator::Div(_) => verify_div_zk(
                node,
                bundle,
                &mut accumulator,
                &mut transcript,
                &mut zk_proof_idx,
            )?,
            // Default flow: eval reduction always; execution sumcheck for ops
            // that have one. The no-sumcheck ops (including Clamp, which the
            // prover treats as a no-op via `create_prover_instance` returning
            // `None`) are listed explicitly below and skip the
            // sumcheck-instances call.
            // No-sumcheck ops that don't register openings on inputs.
            Operator::Input(_) | Operator::Constant(_) => {
                verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;
                // Mirror non-zk `Constant::verify` / `Input::verify` (ops/constant.rs:28-43,
                // ops/input.rs:27-49): locally evaluate the public MLE at the
                // reduced opening point and compare against the prover's
                // cleartext reduced claim from the bundle. Catches honest-prover
                // errors. Active-malice soundness needs an R1CS constraint
                // binding the BlindFold witness to this cleartext (future work;
                // see wiki/jolt-atlas/.../zk-prove-overhead.md).
                if let Some(reduced) = accumulator.reduced_evaluations.get(&node.idx).cloned() {
                    // In extract mode, skip the equality check (used by
                    // soundness tests to recover what the verifier WOULD
                    // compute, so they can patch a malicious bundle).
                    let prover_claim = if extract_public_claims.is_some() {
                        ark_std::Zero::zero()
                    } else {
                        bundle
                            .public_node_reduced_claims
                            .get(&node.idx)
                            .copied()
                            .ok_or_else(|| {
                                ProofVerifyError::InvalidOpeningProof(format!(
                                    "Missing public_node_reduced_claims for node {} ({})",
                                    node.idx,
                                    match node.operator {
                                        Operator::Input(_) => "Input",
                                        Operator::Constant(_) => "Constant",
                                        _ => unreachable!(),
                                    }
                                ))
                            })?
                    };
                    let r_field: Vec<F> = reduced.r.clone();
                    let expected_claim = match &node.operator {
                        Operator::Constant(c) => {
                            MultilinearPolynomial::from(c.0.clone().padded_next_power_of_two())
                                .evaluate(&r_field)
                        }
                        Operator::Input(_) => {
                            let input_pos = io
                                .input_indices
                                .iter()
                                .position(|&idx| idx == node.idx)
                                .ok_or_else(|| {
                                    ProofVerifyError::InvalidOpeningProof(format!(
                                        "Input node {} not in io.input_indices",
                                        node.idx
                                    ))
                                })?;
                            MultilinearPolynomial::from(
                                io.inputs[input_pos].padded_next_power_of_two(),
                            )
                            .evaluate(&r_field)
                        }
                        _ => unreachable!(),
                    };
                    if let Some(extract) = extract_public_claims.as_mut() {
                        extract.insert(node.idx, expected_claim);
                    } else if expected_claim != prover_claim {
                        return Err(ProofVerifyError::InvalidOpeningProof(format!(
                            "Public node {} ({}) reduced-eval mismatch: prover claim {} != public MLE eval {}",
                            node.idx,
                            match node.operator {
                                Operator::Input(_) => "Input",
                                Operator::Constant(_) => "Constant",
                                _ => unreachable!(),
                            },
                            prover_claim,
                            expected_claim
                        )));
                    }
                    // Insert the (now-verified) cleartext claim into the
                    // verifier's accumulator so downstream code that reads
                    // `reduced_evaluations` sees the correct value rather than
                    // the F::zero() placeholder. Has no effect on R1CS verify
                    // (which only uses bundle.baked) — see future-work note.
                    if let Some(slot) = accumulator.reduced_evaluations.get_mut(&node.idx) {
                        slot.claim = expected_claim;
                    }
                }
            }
            // No-sumcheck ops that DO register openings on their input(s).
            // Mirror the prover-side input caching (see the matching arms in
            // `prove_zk`) so the producer's openings list is populated.
            // No-sumcheck ops that register an opening on their input. With
            // `accumulator.zk_mode = true`, the verifier's `append_nodeio`
            // hits the placeholder branch and stays transcript-quiet,
            // mirroring the prover. Each op's verify() also internally
            // compares the input opening claim against the node's reduced
            // claim; in ZK both are the placeholder F::zero(), so the check
            // passes trivially (BlindFold enforces the real algebraic
            // identity via its R1CS).
            Operator::Broadcast(_) => {
                verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;
                if accumulator.reduced_evaluations.contains_key(&node.idx) {
                    use crate::onnx_proof::ops::broadcast::BroadcastVerifier;
                    let bv = BroadcastVerifier::new(node.clone(), &accumulator, &model.graph);
                    bv.verify(&mut accumulator, &mut transcript)?;
                }
            }
            Operator::MoveAxis(_) => {
                verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;
                if accumulator.reduced_evaluations.contains_key(&node.idx) {
                    use crate::onnx_proof::ops::moveaxis::MoveAxisVerifier;
                    let mv = MoveAxisVerifier::new(node.clone(), &accumulator);
                    mv.verify(&mut accumulator, &mut transcript)?;
                }
            }
            Operator::Identity(_) | Operator::IsNan(_) | Operator::Clamp(_) => {
                verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;
                if accumulator.reduced_evaluations.contains_key(&node.idx) {
                    use crate::utils::opening_access::{AccOpeningAccessor, Target};
                    let accessor = AccOpeningAccessor::new(&mut accumulator, node);
                    let (opening_point, _claim) = accessor.get_reduced_opening();
                    let mut provider = accessor.into_provider(&mut transcript, opening_point);
                    provider.append_nodeio(Target::Input(0));
                }
            }
            // MeanOfSquares is not yet implemented on the prover side
            // (`create_prover_instance` falls through to its panic arm). Mirror
            // the panic here so both sides fail loudly and symmetrically
            // rather than later at an opaque proof-index mismatch.
            Operator::MeanOfSquares(_) => {
                unimplemented!("ZK verification not yet implemented for MeanOfSquares");
            }
            Operator::Add(_)
            | Operator::Sub(_)
            | Operator::Neg(_)
            | Operator::Mul(_)
            | Operator::Square(_)
            | Operator::Cube(_)
            | Operator::And(_)
            | Operator::Iff(_)
            | Operator::Concat(_)
            | Operator::Reshape(_)
            | Operator::Slice(_)
            | Operator::ScalarConstDiv(_)
            | Operator::Einsum(_)
            | Operator::Sum(_) => {
                verify_zk_eval_reduction(node, bundle, &mut accumulator, &mut transcript)?;

                // Mirror of the prover-side fused-rescale staging.
                let fused = crate::onnx_proof::fused_rebase::fuses_rebase(&node.operator);
                if fused {
                    verify_fused_rebase_pre_zk(
                        node,
                        bundle,
                        &mut accumulator,
                        &mut transcript,
                        &mut zk_proof_idx,
                    )?;
                }

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

                if fused {
                    verify_fused_rebase_post_zk(
                        node,
                        bundle,
                        &mut accumulator,
                        &mut transcript,
                        &mut zk_proof_idx,
                    )?;
                }
            }
        }
    }

    // Extract mode: all the per-op Input/Constant checks have populated
    // `extract_public_claims` with the verifier-computed expected values.
    // The remaining steps (Step 8 batched-opening replay, BlindFold verify,
    // PCS verify, public-node R1CS binding checks) are skipped — the
    // soundness test that uses this hook only needs the per-node expected
    // values to patch a malicious bundle.
    if extract_public_claims.is_some() {
        return Ok(());
    }

    // Batched-opening reduction (Steps 6-8 of the wiki plan): mirror the
    // prover-side flow. Order matches the prover:
    //   1. derive gamma_powers from transcript (the cleartext per-poly claim
    //      append is suppressed on both sides in ZK mode),
    //   2. absorb y_com into the transcript,
    //   3. build the extra constraint linking joint_claim to per-poly y_P_i's,
    //   4. run BlindFold (R1CS proves sumcheck + extra constraint),
    //   5. run PCS::verify on the joint commitment so `bundle.commitments`
    //      bind to `joint_claim` at the reduced point (Step 8).
    let mut extra_constraints_for_r1cs: Vec<
        joltworks::subprotocols::blindfold::output_constraint::OutputClaimConstraint,
    > = Vec::new();
    let eval_commitment_gens_v;
    let gamma_powers_v: Vec<F>;
    // Step 8 fires iff the prover ran the batched-opening sumcheck (i.e., the
    // model had committed witness polynomials). The presence of the Step-8
    // ZK sumcheck proof in the bundle is the unambiguous gate; `eval_commitments`
    // is no longer reliable because the public-node binding y_coms (added
    // below) also occupy `eval_commitments` slots.
    if let Some(y_com) = bundle
        .batch_opening_zk_sumcheck
        .as_ref()
        .and(bundle.blindfold_verifier_input.eval_commitments.first())
    {
        // Replay the prover-side absorbs from `prove_batch_opening_sumcheck_zk`
        // and `finalize_batch_opening_sumcheck` into the main transcript:
        // (a) each round commitment from the batched-opening sumcheck (so
        //     batching coefficients and round challenges line up with the
        //     prover's BlindFold stage), then (b) the gamma_powers derivation,
        //     then (c) y_com. Order matters; deviating here desyncs the
        //     transcript at the PCS::verify point downstream.
        let batch_zk = bundle.batch_opening_zk_sumcheck.as_ref().ok_or_else(|| {
            ProofVerifyError::InvalidOpeningProof(
                "Missing batch_opening_zk_sumcheck in bundle".to_string(),
            )
        })?;
        let opening_ids: Vec<joltworks::poly::opening_proof::OpeningId> = accumulator
            .sumchecks_keys()
            .map(|poly| {
                joltworks::poly::opening_proof::OpeningId::new(
                    poly,
                    joltworks::poly::opening_proof::SumcheckId::BlindFoldBatchOpening,
                )
            })
            .collect();
        // (a) batching coefficients (consumes `opening_ids.len()` scalars).
        let _batching_coeffs: Vec<F> = transcript.challenge_vector(opening_ids.len());
        // round commitments + per-round challenge derivation
        for round_com in &batch_zk.round_commitments {
            transcript.append_serializable(round_com);
            let _r_j: <F as joltworks::field::JoltField>::Challenge =
                transcript.challenge_scalar_optimized::<F>();
        }
        // (output_claims_commitments are absorbed at the tail of BatchedSumcheck::prove_zk)
        for com in &batch_zk.output_claims_commitments {
            transcript.append_serializable(com);
        }
        // (b) gamma derivation -- cleartext claim append is suppressed on both
        //     sides in ZK mode, so we just sample.
        gamma_powers_v = transcript.challenge_scalar_powers(opening_ids.len());
        // (c) y_com.
        transcript.append_serializable(y_com);
        use joltworks::subprotocols::blindfold::output_constraint::{
            OutputClaimConstraint, ValueSource,
        };
        let terms: Vec<(ValueSource, ValueSource)> = opening_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (ValueSource::Challenge(i), ValueSource::Opening(*id)))
            .collect();
        extra_constraints_for_r1cs.push(OutputClaimConstraint::linear(terms));
        eval_commitment_gens_v = Some((
            pedersen_gens.message_generators[0],
            pedersen_gens.blinding_generator,
        ));
    } else {
        gamma_powers_v = Vec::new();
        eval_commitment_gens_v = None;
    }

    // Public-node opening binding (mirror of the prover-side block in
    // `prove_zk`): for each opening on a `Constant`/`Input` node, add an
    // R1CS extra-constraint `Opening(id) - Challenge(0) = output_var` and
    // verify two soundness invariants:
    //   1. `bundle.baked.extra_constraint_challenges[binding_idx]` equals
    //      the verifier-computed `public_mle(opening_point)`. This binds
    //      the matrix coefficient to the public expected value.
    //   2. The corresponding `bundle.blindfold_verifier_input.eval_commitments`
    //      entry is the identity G1 point. Combined with BlindFold's
    //      `eval_commitment_gens` check (which enforces
    //      `y_com == g*output_var + h*blinding`), identity forces
    //      `output_var = 0` AND `blinding = 0`. Then the R1CS row
    //      `output_var = Opening - Challenge` forces `Opening = Challenge
    //      = expected`.
    //
    // The challenge-value check is one `bundle.baked` lookup; the y_com
    // identity check is one G1 comparison.
    let binding_challenge_start = bundle
        .baked
        .extra_constraint_challenges
        .len()
        .saturating_sub(0);
    // The Step 8 constraint contributes `opening_ids.len()` challenges; binding
    // constraints contribute 1 challenge each. Step 8 (if present) is at
    // baked index 0..opening_ids.len(); bindings come after.
    let step8_num_challenges: usize = if !extra_constraints_for_r1cs.is_empty() {
        extra_constraints_for_r1cs[0].num_challenges
    } else {
        0
    };
    let binding_baked_start = step8_num_challenges;
    let step8_eval_commitments_count: usize = if !extra_constraints_for_r1cs.is_empty() {
        1
    } else {
        0
    };
    let mut binding_idx_in_baked = binding_baked_start;
    let mut binding_idx_in_eval = step8_eval_commitments_count;
    {
        use common::VirtualPoly;
        use joltworks::{
            poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            subprotocols::blindfold::output_constraint::{OutputClaimConstraint, ValueSource},
        };
        let model = pp.shared.model();
        let _ = binding_challenge_start;
        for (opening_id, (opening_point, _)) in accumulator.openings.iter() {
            let Some(VirtualPoly::NodeOutput(node_idx)) = opening_id.virtual_poly() else {
                continue;
            };
            // Mirror the prover-side filter: only bind `NodeExecution`
            // openings (raw NodeOutput evaluations); skip internal
            // protocol-specific sumchecks like `Raf`.
            if !matches!(
                opening_id.sumcheck,
                joltworks::poly::opening_proof::SumcheckId::NodeExecution(_)
            ) {
                continue;
            }
            let node = &model[node_idx];
            let expected_value: F = match &node.operator {
                Operator::Constant(c) => {
                    MultilinearPolynomial::from(c.0.clone().padded_next_power_of_two())
                        .evaluate(&opening_point.r)
                }
                Operator::Input(_) => {
                    let input_pos = io
                        .input_indices
                        .iter()
                        .position(|&i| i == node_idx)
                        .ok_or_else(|| {
                            ProofVerifyError::InvalidOpeningProof(format!(
                                "Input node {node_idx} missing from io.input_indices",
                            ))
                        })?;
                    MultilinearPolynomial::from(io.inputs[input_pos].padded_next_power_of_two())
                        .evaluate(&opening_point.r)
                }
                _ => continue,
            };
            // (1) baked challenge matches verifier-computed expected.
            let baked_challenge = bundle
                .baked
                .extra_constraint_challenges
                .get(binding_idx_in_baked)
                .copied()
                .ok_or_else(|| {
                    ProofVerifyError::InvalidOpeningProof(format!(
                        "Missing binding challenge at baked idx {binding_idx_in_baked}",
                    ))
                })?;
            if baked_challenge != expected_value {
                return Err(ProofVerifyError::InvalidOpeningProof(format!(
                    "Public-node binding mismatch for node {node_idx}: \
                     baked challenge {baked_challenge} != public MLE eval {expected_value}",
                )));
            }
            // (2) corresponding y_com is identity G1.
            let y_com = bundle
                .blindfold_verifier_input
                .eval_commitments
                .get(binding_idx_in_eval)
                .ok_or_else(|| {
                    ProofVerifyError::InvalidOpeningProof(format!(
                        "Missing binding y_com at eval_commitments idx {binding_idx_in_eval}",
                    ))
                })?;
            use joltworks::curve::JoltGroupElement;
            if !y_com.is_zero() {
                return Err(ProofVerifyError::InvalidOpeningProof(format!(
                    "Public-node binding y_com at idx {binding_idx_in_eval} is not identity G1 \
                     (would let the prover commit a non-zero output_var, decoupling \
                     witness opening from the public expected value)",
                )));
            }
            // Add the constraint to the R1CS structure (must mirror prover).
            extra_constraints_for_r1cs.push(OutputClaimConstraint::linear(vec![
                (ValueSource::Constant(1), ValueSource::Opening(*opening_id)),
                (ValueSource::Constant(-1), ValueSource::Challenge(0)),
            ]));
            binding_idx_in_baked += 1;
            binding_idx_in_eval += 1;
        }
    }
    // Activate eval_commitment_gens if we have any extra constraints (Step 8
    // OR binding constraints).
    let eval_commitment_gens_v = if !extra_constraints_for_r1cs.is_empty() {
        Some((
            pedersen_gens.message_generators[0],
            pedersen_gens.blinding_generator,
        ))
    } else {
        eval_commitment_gens_v
    };

    // 4. Verify BlindFold proof
    let builder = VerifierR1CSBuilder::<F>::new_with_extra(
        &bundle.stage_configs,
        &extra_constraints_for_r1cs,
        &bundle.baked,
        Vec::new(),
        std::collections::BTreeMap::new(),
    );
    let r1cs = builder.build();
    let bf_verifier = BlindFoldVerifier::new(pedersen_gens, &r1cs, eval_commitment_gens_v);

    let mut bf_transcript = T::new(b"BlindFold_onnx");
    bf_verifier
        .verify(
            &bundle.blindfold_proof,
            &bundle.blindfold_verifier_input,
            &mut bf_transcript,
        )
        .map_err(|e| {
            ProofVerifyError::InvalidOpeningProof(format!("BlindFold verification failed: {e:?}"))
        })?;

    // Step 8: PCS-binding. Verify that `bundle.commitments` actually evaluate
    // to `joint_claim` at the batched-opening reduction point. The verifier
    // homomorphically combines the polynomial commitments under `gamma_powers`
    // (matching the prover's RLC), then verifies the joint opening proof.
    //
    // This is what cryptographically ties the BlindFold R1CS witness (which
    // proves `joint_claim = sum gamma_i * y_P_i` and consistency with the
    // per-node sumchecks via the `y_P_i` openings) back to the actual
    // polynomial commitments. `joint_claim` is revealed in cleartext; a
    // future ZK extension to HyperKZG could hide it as well.
    if let (Some(opening_proof), Some(joint_claim), Some(opening_point)) = (
        bundle.joint_opening_proof.as_ref(),
        bundle.joint_claim.as_ref(),
        bundle.joint_opening_point.as_ref(),
    ) {
        let joint_commitment = PCS::combine_commitments(&bundle.commitments, &gamma_powers_v);
        if let Some(slot) = capture {
            *slot = Some(PcsVerifyCapture {
                joint_commitment: joint_commitment.clone(),
                opening_proof: opening_proof.clone(),
                opening_point: opening_point.clone(),
                joint_claim: *joint_claim,
                transcript: transcript.clone(),
            });
        }
        PCS::verify(
            opening_proof,
            &pp.generators,
            &mut transcript,
            opening_point,
            joint_claim,
            &joint_commitment,
        )?;
    }
    Ok(())
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
        Operator::Einsum(_) => {
            use crate::onnx_proof::ops::einsum::EinsumProver;
            Some(EinsumProver::sumcheck(
                model,
                &prover.trace,
                node.clone(),
                &prover.accumulator,
            ))
        }
        Operator::Sum(_) => {
            use crate::onnx_proof::ops::sum::create_sum_prover;
            Some(create_sum_prover(
                node,
                model,
                &prover.trace,
                &prover.accumulator,
            ))
        }
        Operator::Input(_)
        | Operator::Identity(_)
        | Operator::Broadcast(_)
        | Operator::MoveAxis(_)
        | Operator::Constant(_)
        | Operator::IsNan(_)
        | Operator::Clamp(_) => None,
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
        Operator::Tanh(op) => {
            use crate::onnx_proof::ops::tanh::TanhVerifier;
            vec![Box::new(TanhVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
                op.clone(),
            ))]
        }
        Operator::Erf(op) => {
            use crate::onnx_proof::ops::erf::ErfVerifier;
            vec![Box::new(ErfVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
                op.clone(),
            ))]
        }
        Operator::Cos(_) => {
            use crate::onnx_proof::ops::cos::CosVerifier;
            vec![Box::new(CosVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
            ))]
        }
        Operator::Sin(_) => {
            use crate::onnx_proof::ops::sin::SinVerifier;
            vec![Box::new(SinVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
            ))]
        }
        Operator::Einsum(_) => {
            use crate::onnx_proof::ops::einsum::EinsumVerifier;
            vec![EinsumVerifier::sumcheck(model, node.clone(), accumulator)]
        }
        Operator::Sum(_) => {
            use crate::onnx_proof::ops::sum::create_sum_verifier;
            vec![create_sum_verifier(node, model, accumulator)]
        }
        Operator::GatherLarge(_) | Operator::GatherSmall(_) => {
            use crate::onnx_proof::ops::gather::GatherVerifier;
            vec![Box::new(GatherVerifier::new(
                node.clone(),
                &model.graph,
                accumulator,
                transcript,
            ))]
        }
        _ => vec![],
    }
}
