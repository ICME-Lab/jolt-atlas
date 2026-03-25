//! Opening ID builder trait for ComputationNode.
//!
//! Provides convenient methods to construct scoped opening identifiers for proof generation.

use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPolynomial;
use joltworks::poly::opening_proof::{SumcheckId, VirtualOpeningId};

/// Selects the node index source used to build opening identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningTarget {
    /// Use the current node index.
    Current,
    /// Use the node index referenced by the current node input at the given position.
    Input(usize),
}

/// Trait for constructing opening IDs scoped to computation nodes.
pub trait OpeningIdBuilder {
    /// Build a virtual NodeOutput opening ID from a target.
    fn build_opening_id(&self, target: OpeningTarget) -> VirtualOpeningId;
}

impl OpeningIdBuilder for ComputationNode {
    fn build_opening_id(&self, target: OpeningTarget) -> VirtualOpeningId {
        let target_idx = match target {
            OpeningTarget::Current => self.idx,
            OpeningTarget::Input(position) => self.inputs[position],
        };
        VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(target_idx),
            SumcheckId::NodeExecution(self.idx),
        )
    }
}
