use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, ReductionFlow};
use crate::onnx_proof::plan::{OpLookup, OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::{op_lookups::OpLookupEncoding, ProofType};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::shout::RaOneHotEncoding;
use atlas_trace::op::ReLU;
use common::CommittedPoly;

impl OpLayout for ReLU {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let encoding: OpLookupEncoding = OpLookupEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        Ok((0..d)
            .map(|i| CommittedPoly::NodeOutputRaD(node.idx, i))
            .collect())
    }
}

crate::impl_plan_op!(@verify ReLU);

impl OpPlan for ReLU {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Lookup(OpLookup::Relu, node.clone()),
            },
            Step::Batch {
                proof: ProofType::RaOneHotChecks,
                relations: vec![OpRelation::LookupOneHot(OpLookup::Relu, node.clone())],
            },
        ]))
    }
}
