use super::*;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator};
use atlas_sumcheck::{
    hamming_booleanity::{HammingBooleanitySumcheckParams, HammingBooleanitySumcheckVerifier},
    hamming_weight::{HammingWeightSumcheckParams, HammingWeightSumcheckVerifier},
};
use atlas_trace::op::GatherSmall;
use atlas_transcript::{Transcript, VerifyingTranscript};
use common::CommittedPoly;

/// The plan: the evaluation reduction, the dictionary read, then the one-hot checks of its `ra`
/// (Hamming-Booleanity and Hamming weight), proven in the one-hot sumcheck. That `ra` is
/// boolean is held by its commitment.
impl OpPlan for GatherSmall {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        let shape = GatherSmallShape::for_node(model, node)?;
        schedule.reduce();
        schedule.relation(OpRelation::Op(gather(model, node)?));
        schedule.one_hot(OpRelation::GatherSmallOneHot(shape.clone()));
        schedule.one_hot(OpRelation::GatherSmallHammingWeight(shape));
        Ok(())
    }
}

/// A small gather's one-hot layout: the node, its dictionary rows and its lookup count.
#[derive(Clone)]
pub struct GatherSmallShape {
    /// The gather node.
    pub node: ComputationNode,
    /// The dictionary rows (the one-hot width).
    pub num_words: usize,
    /// The padded number of lookups (the one-hot cycles).
    pub num_lookups: usize,
}

impl GatherSmallShape {
    /// The layout of a small gather `node` of `model`. `Err` if the node is not a small gather
    /// over a dictionary and an index operand.
    pub fn for_node(model: &Model, node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        Ok(Self {
            node: node.clone(),
            num_words: dictionary_words(model, node)?,
            num_lookups: operand(model, node, 1)?.pow2_padded_num_output_elements(),
        })
    }
}

/// The dictionary rows of a small gather: its dictionary operand's dim along the gather axis.
/// `Err` if the node is not a small gather over a dictionary with that axis.
fn dictionary_words(
    model: &Model,
    computation_node: &ComputationNode,
) -> Result<usize, ProofVerifyError> {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        return Err(unsupported(
            computation_node,
            "expected a GatherSmall operator",
        ));
    };
    operand(model, computation_node, 0)?
        .output_dims
        .get(gather_op.axis)
        .copied()
        .ok_or_else(|| unsupported(computation_node, "the gather axis is not a dictionary axis"))
}

/// The second-stage verifier instance (Hamming-Booleanity) of a small gather.
/// `Err` if the accumulator lacks a claim it reads.
pub fn build_stage2_verifier<F: JoltField>(
    shape: &GatherSmallShape,
    accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl VerifyingTranscript,
) -> Result<HammingBooleanitySumcheckVerifier<F>, ProofVerifyError> {
    let hb_params =
        ra_hamming_bool_params::<F>(&shape.node, shape.num_lookups, accumulator, transcript)?;
    Ok(hb_params.into_verifier())
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

/// The third-stage verifier instance (Hamming weight) of a small gather.
/// `Err` if the accumulator lacks a claim it reads.
pub fn build_stage3_verifier<F: JoltField>(
    shape: &GatherSmallShape,
    accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl VerifyingTranscript,
) -> Result<HammingWeightSumcheckVerifier<F>, ProofVerifyError> {
    let hw_params =
        ra_hamming_weight_params::<F>(&shape.node, shape.num_words, accumulator, transcript)?;
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
