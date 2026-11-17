use crate::node_view::ComputationNode;
use crate::onnx_proof::deferred_lookups::DeferredBatch;
use crate::onnx_proof::op_lookups::OpLookupEncoding;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpLookup, OpPlan, OpRelation, Plan, Step};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::shout::RaOneHotEncoding;
use atlas_trace::op::ReLU;
use common::CommittedPoly;

impl OpLayout for ReLU {
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

impl OpPlan for ReLU {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Defer {
                batch: DeferredBatch::Lookups,
                relation: OpRelation::Lookup(OpLookup::Relu, node.clone()),
            },
            Step::Defer {
                batch: DeferredBatch::OneHot,
                relation: OpRelation::LookupOneHot(OpLookup::Relu, node.clone()),
            },
        ]))
    }
}
