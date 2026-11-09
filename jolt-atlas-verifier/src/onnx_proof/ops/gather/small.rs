use super::*;
use crate::node_view::ModelView;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::relation::{verify_relation, verify_relation_batch, DynRelationVerifier};
use atlas_sumcheck::{
    booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
    hamming_booleanity::{HammingBooleanitySumcheckParams, HammingBooleanitySumcheckVerifier},
    hamming_weight::{HammingWeightSumcheckParams, HammingWeightSumcheckVerifier},
};
use atlas_trace::op::GatherSmall;
use common::CommittedPoly;

impl OpLayout for GatherSmall {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        vec![CommittedPoly::GatherRa(node.idx)]
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for GatherSmall {
    #[tracing::instrument(skip_all, name = "GatherSmall::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::Execution))?;
        let exec_sumcheck = gather_verifier(node, verifier)?;
        verify_relation(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let ra_one_hot_proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::RaOneHotChecks))?;
        let (hb_sumcheck, bool_sumcheck) = build_stage2_verifiers::<F>(node, verifier)?;
        let instances: Vec<&dyn DynRelationVerifier<F, T>> = vec![&hb_sumcheck, &bool_sumcheck];
        verify_relation_batch(
            ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let hw_proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::RaHammingWeight))?;
        let hw_sumcheck = build_stage3_verifier::<F>(node, verifier)?;
        verify_relation(
            hw_proof,
            &hw_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }
}

/// The second-stage verifier instances (one-hot checks) of a small gather.
/// `Err` if the accumulator lacks a claim it reads.
pub fn build_stage2_verifiers<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> Result<
    (
        HammingBooleanitySumcheckVerifier<F>,
        BooleanitySumcheckVerifier<F>,
    ),
    ProofVerifyError,
> {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let model = &verifier.preprocessing.model;
    let dict = &model.view(computation_node.inputs[0]);
    let indices = model.view(computation_node.inputs[1]);

    let num_words = dict.output_dims[gather_op.axis];
    let num_lookups = indices.pow2_padded_num_output_elements();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    )?;
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        num_words,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let hb_sumcheck = hb_params.into_verifier();
    let bool_sumcheck = bool_params.into_verifier();

    Ok((hb_sumcheck, bool_sumcheck))
}

/// Hamming-booleanity parameters of the read addresses.
/// `Err` if the accumulator lacks a claim it reads.
pub fn ra_hamming_bool_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> Result<HammingBooleanitySumcheckParams<F>, ProofVerifyError> {
    let polynomial_types = vec![VirtualPoly::HammingWeight(computation_node.idx)];

    let r_lookup = AccOpeningAccessor::new(opening_accumulator, computation_node)
        .get_nodeio(Target::Input(1))?
        .0
        .r;

    Ok(HammingBooleanitySumcheckParams {
        d: 1,
        num_rounds: num_lookups.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,
        r_cycle: r_lookup,
        sumcheck_id: SumcheckId::RamHammingBooleanity,
    })
}

/// Booleanity parameters of the read addresses.
/// `Err` if the accumulator lacks a claim it reads.
pub fn ra_booleanity_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_words: usize,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> Result<BooleanitySumcheckParams<F>, ProofVerifyError> {
    let polynomial_type = CommittedPoly::GatherRa(computation_node.idx);

    let r_lookup = AccOpeningAccessor::new(opening_accumulator, computation_node)
        .get_nodeio(Target::Input(1))?
        .0
        .r;
    let r_address = transcript
        .challenge_vector_optimized::<F>(num_words.log_2())
        .into_opening();

    Ok(BooleanitySumcheckParams {
        linear: None,
        d: 1,
        log_k_chunk: num_words.log_2(),
        log_t: num_lookups.log_2(),
        r_cycle: r_lookup,
        r_address,
        polynomial_types: vec![polynomial_type],
        gammas: vec![F::Challenge::from(1)],
        sumcheck_id: SumcheckId::Booleanity,
    })
}

/// The third-stage verifier instance (Hamming weight) of a small gather.
/// `Err` if the accumulator lacks a claim it reads.
pub fn build_stage3_verifier<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> Result<HammingWeightSumcheckVerifier<F>, ProofVerifyError> {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let model = &verifier.preprocessing.model;
    let dict = &model.view(computation_node.inputs[0]);

    let num_words = dict.output_dims[gather_op.axis];

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        num_words,
        &verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(hw_params.into_verifier())
}

/// Hamming-weight parameters of the read addresses.
/// `Err` if the accumulator lacks a claim it reads.
pub fn ra_hamming_weight_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_words: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> Result<HammingWeightSumcheckParams<F>, ProofVerifyError> {
    let polynomial_types = vec![CommittedPoly::GatherRa(computation_node.idx)];

    let r_lookup = AccOpeningAccessor::new(opening_accumulator, computation_node)
        .get_nodeio(Target::Input(1))?
        .0
        .r;

    Ok(HammingWeightSumcheckParams {
        d: 1,
        num_rounds: num_words.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle: r_lookup,
    })
}
