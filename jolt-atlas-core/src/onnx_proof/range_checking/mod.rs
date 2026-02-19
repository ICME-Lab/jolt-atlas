//! Range-checking protocol for verifying bounds on remainder values.
//!
//! This module implements range-checking for operations that produce remainders (Div, Rsqrt, Tanh).
//! It uses the prefix-suffix read-checking sumcheck protocol from the Twist and Shout paper to
//! efficiently verify that remainder values are within valid bounds using lookup tables.

use atlas_onnx_tracer::node::ComputationNode;
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::OneHotParams, poly::opening_proof::SumcheckId, subprotocols::shout::RaOneHotEncoding,
    utils::math::Math,
};

use crate::onnx_proof::range_checking::sumcheck_instance::ReadRafSumcheckHelper;

/// Read-raf checking protocol for range-checking.
///
/// Implements the prefix-suffix read-checking sumcheck from the Twist and Shout paper,
/// used to verify that remainder values in division and square root operations are within
/// valid bounds using lookup tables.
pub mod read_raf_checking;
/// Sumcheck instance definitions for operations requiring range-checking.
///
/// Defines the [`ReadRafSumcheckHelper`] trait and implementations for different operations
/// (Div, Rsqrt, Tanh) that need range-checking in a trait-friendly manner.
pub mod sumcheck_instance;

/// Size of range-checking table
pub const LOG_K: usize = XLEN * 2;

// ---------------------------------------------------------------------------
// RangeCheckEncoding — implements RaOneHotEncoding for range-checking ops
// (Div, Rsqrt, Tanh)
// ---------------------------------------------------------------------------

/// Encoding for range-checking read-address one-hot checking.
///
/// This struct encapsulates the information needed to perform range-checking for
/// operations that produce remainders (Div, Rsqrt, Tanh). It implements the
/// [`RaOneHotEncoding`] trait to integrate with the one-hot checking protocol.
pub struct RangeCheckEncoding<H: ReadRafSumcheckHelper> {
    /// Helper providing operation-specific range-checking logic.
    pub helper: H,
    /// log2 of the number of elements in the computation (T).
    pub log_t: usize,
}

impl<H: ReadRafSumcheckHelper> RangeCheckEncoding<H> {
    /// Create a new range-check encoding from a computation node.
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
