use atlas_onnx_tracer::node::ComputationNode;
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::OneHotParams, poly::opening_proof::SumcheckId, subprotocols::shout::RaOneHotEncoding,
    utils::math::Math,
};

use crate::onnx_proof::range_checking::sumcheck_instance::ReadRafSumcheckHelper;

pub mod read_raf_checking;
pub mod sumcheck_instance;

pub const LOG_K: usize = XLEN * 2;

// ---------------------------------------------------------------------------
// RangeCheckEncoding — implements RaOneHotEncoding for range-checking ops
// (Div, Rsqrt, Tanh)
// ---------------------------------------------------------------------------

pub struct RangeCheckEncoding<H: ReadRafSumcheckHelper> {
    pub helper: H,
    pub log_t: usize,
}

impl<H: ReadRafSumcheckHelper> RangeCheckEncoding<H> {
    pub fn new(computation_node: &ComputationNode) -> Self {
        Self {
            helper: H::new(computation_node),
            log_t: computation_node.num_output_elements().log_2(),
        }
    }
}

impl<H: ReadRafSumcheckHelper> RaOneHotEncoding for RangeCheckEncoding<H> {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        self.helper.rad_poly(d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutput(self.helper.node_idx()),
            SumcheckId::Execution,
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (self.helper.get_output_operand(), SumcheckId::Raf)
    }

    fn log_k(&self) -> usize {
        LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t)
    }
}
