use super::*;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::OpeningId,
    shout::RaOneHotEncoding,
};
use atlas_trace::op::GatherLarge;
use common::CommittedPoly;

crate::impl_plan_op!(@verify GatherLarge);

impl OpLayout for GatherLarge {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let encoding = GatherRaEncoding::new(node)?;
        let d = encoding.one_hot_params().instruction_d;
        Ok((0..d)
            .map(|i| CommittedPoly::GatherRaD(node.idx, i))
            .collect())
    }
}

/// The plan: the evaluation reduction, the dictionary read, then the one-hot checks of its
/// read addresses as one batch.
impl OpPlan for GatherLarge {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Gather(GatherRelation::for_node(model, node.clone())?),
            },
            Step::Batch {
                proof: ProofType::RaOneHotChecks,
                relations: vec![OpRelation::GatherLargeOneHot(node.clone())],
            },
        ]))
    }
}

/// One-hot encoding of a large gather's read addresses.
pub struct GatherRaEncoding {
    node_idx: usize,
    index_input_idx: usize,
    num_words: usize,
}

impl GatherRaEncoding {
    /// The encoding for the node's dictionary. `Err` if the node is not a large gather with
    /// an index operand.
    pub fn new(computation_node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        let Op::GatherLarge(gather_op) = &computation_node.operator else {
            return Err(unsupported(
                computation_node,
                "expected a GatherLarge operator",
            ));
        };
        let index_input_idx = *computation_node
            .inputs
            .get(1)
            .ok_or_else(|| unsupported(computation_node, "no index operand"))?;
        Ok(Self {
            node_idx: computation_node.idx,
            index_input_idx,
            num_words: gather_op.dict_len.next_power_of_two(),
        })
    }
}

impl RaOneHotEncoding for GatherRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::GatherRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(self.index_input_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutputRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.num_words.log_2()
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k())
    }
}
