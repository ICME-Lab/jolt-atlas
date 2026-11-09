use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpLookup, OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::{
    op_lookups::{DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding},
    ProofType,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_lookup_tables::clamp::CLAMP_BOUND;
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    shout::RaOneHotEncoding,
};
use atlas_trace::op::Clamp;
use common::{CommittedPoly, VirtualPoly};

/// Lookup helper for the ONNX `Clamp` op, backed by `atlas_lookup_tables::clamp`'s
/// symmetric clamp table (`[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`).
#[derive(Default)]
pub struct SymmetricClampOperands;

impl LookupOperandsTrait for SymmetricClampOperands {
    const LOG_K: usize = DefaultLookupOperands::LOG_K;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        DefaultLookupOperands::rv_claim(node, accumulator)
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SymmetricClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SymmetricClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        DefaultLookupOperands::witness_opening_id(node)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }
}

impl OpLayout for Clamp {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let encoding = OpLookupEncoding::<SymmetricClampOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        Ok((0..d)
            .map(|i| CommittedPoly::SymmetricClampRaD(node.idx, i))
            .collect())
    }
}

impl OpPlan for Clamp {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        // The clamp table is the symmetric one at this bound.
        if self.bound_log as usize != CLAMP_BOUND {
            return Err(ProofVerifyError::Malformed(format!(
                "model: clamp node {} has bound 2^{}, the table's is 2^{CLAMP_BOUND}",
                node.idx, self.bound_log
            )));
        }
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Lookup(OpLookup::SymmetricClamp, node.clone()),
            },
            Step::Batch {
                proof: ProofType::RaOneHotChecks,
                relations: vec![OpRelation::LookupOneHot(
                    OpLookup::SymmetricClamp,
                    node.clone(),
                )],
            },
        ]))
    }
}
