//! Operator lookup tables and Prefix suffix Read-raf checking protocols.
//!
//! This module provides infrastructure for proving operations using lookup tables,
//! particularly ReLU and comparison operations. The Prefix suffix Read-raf checking sum-check protocol
//! verifies correct reads from lookup tables and combines multiple claims using
//! gamma batching for efficiency.

use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::OneHotParams, poly::opening_proof::SumcheckId, subprotocols::shout::RaOneHotEncoding,
};

/// Prefix suffix Read-raf checking sum-check protocol implementation.
pub mod read_raf_checking;

/// Logarithm of K = XLEN * 2, the total number of address bits for lookup tables.
///
/// For XLEN=8, this is log₂(16) = 4; for XLEN=32, this is log₂(64) = 6.
pub const LOG_K: usize = XLEN * 2;

// ---------------------------------------------------------------------------
// OpLookupEncoding — implements RaOneHotEncoding for op_lookups (ReLU, ULessThan, etc..)
// ---------------------------------------------------------------------------

/// Encoding for proving reads into prefix-suffix operator lookup tables.
///
/// Implements the [`RaOneHotEncoding`] trait to provide ra one-hot checks for
/// prefix-suffix lookups in the ONNX proof system.
pub struct OpLookupEncoding {
    /// Index of the computation node using this lookup encoding.
    pub node_idx: usize,
    /// log₂(T): number of output elements in the node.
    pub log_t: usize,
}

impl OpLookupEncoding {
    /// Creates a new operation lookup encoding for the given computation node.
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

/// Trait for determining if a computation node uses interleaved operand bits.
///
/// Some operations store operands with interleaved bits for prefix-suffix decomposition
/// . This trait provides a method to check if a node uses this representation.
pub trait InterleavedBitsMarker {
    /// Returns `true` if the operands are stored with interleaved bits.
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
