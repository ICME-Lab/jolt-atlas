use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::OneHotParams, poly::opening_proof::SumcheckId, subprotocols::shout::RaOneHotEncoding,
};

pub mod read_raf_checking;

pub const LOG_K: usize = XLEN * 2;

// ---------------------------------------------------------------------------
// OpLookupEncoding — implements RaOneHotEncoding for op_lookups (ReLU, ULessThan, etc..)
// ---------------------------------------------------------------------------

pub struct OpLookupEncoding {
    pub node_idx: usize,
    pub log_t: usize,
}

impl OpLookupEncoding {
    pub fn new(computation_node: &ComputationNode) -> Self {
        use joltworks::utils::math::Math;
        Self {
            node_idx: computation_node.idx,
            log_t: computation_node.num_output_elements().log_2(),
        }
    }
}

impl RaOneHotEncoding for OpLookupEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::NodeOutputRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutput(self.node_idx),
            SumcheckId::Execution,
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutputRa(self.node_idx),
            SumcheckId::Execution,
        )
    }

    fn log_k(&self) -> usize {
        LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t)
    }
}

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for ComputationNode {
    fn is_interleaved_operands(&self) -> bool {
        match self.operator {
            Operator::ReLU(_) => false,
            _ => unimplemented!(),
        }
    }
}
