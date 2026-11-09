use super::*;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::{
    booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
    hamming_booleanity::{HammingBooleanitySumcheckParams, HammingBooleanitySumcheckVerifier},
    hamming_weight::{HammingWeightSumcheckParams, HammingWeightSumcheckVerifier},
};
use atlas_trace::op::GatherSmall;
use common::CommittedPoly;

impl OpLayout for GatherSmall {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![CommittedPoly::GatherRa(node.idx)])
    }
}

/// The plan: the evaluation reduction, the dictionary read, the one-hot checks of its `ra`
/// (Hamming-Booleanity and Booleanity as one batch), then the Hamming-weight check.
impl OpPlan for GatherSmall {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        let shape = GatherSmallShape::for_node(model, node)?;
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Gather(GatherRelation::for_node(model, node.clone())?),
            },
            Step::Batch {
                proof: ProofType::RaOneHotChecks,
                relations: vec![OpRelation::GatherSmallOneHot(shape.clone())],
            },
            Step::Relation {
                proof: ProofType::RaHammingWeight,
                relation: OpRelation::GatherSmallHammingWeight(shape),
            },
        ]))
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

/// The second-stage verifier instances (one-hot checks) of a small gather.
/// `Err` if the accumulator lacks a claim it reads.
pub fn build_stage2_verifiers<F: JoltField>(
    shape: &GatherSmallShape,
    accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> Result<
    (
        HammingBooleanitySumcheckVerifier<F>,
        BooleanitySumcheckVerifier<F>,
    ),
    ProofVerifyError,
> {
    let hb_params =
        ra_hamming_bool_params::<F>(&shape.node, shape.num_lookups, accumulator, transcript)?;
    let bool_params = ra_booleanity_params::<F>(
        &shape.node,
        shape.num_words,
        shape.num_lookups,
        accumulator,
        transcript,
    )?;
    Ok((hb_params.into_verifier(), bool_params.into_verifier()))
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
    shape: &GatherSmallShape,
    accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
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
