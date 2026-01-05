use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
use common::consts::XLEN;

pub mod ra_virtual;
pub mod read_raf_checking;

pub const LOG_K: usize = XLEN * 2;

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for ComputationNode {
    fn is_interleaved_operands(&self) -> bool {
        matches!(self.operator, Operator::And2(_))
    }
}
