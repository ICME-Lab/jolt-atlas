//! Opening ID builder utilities for computation nodes.
//!
//! Uses a struct-based factory with closure-based constructors so callers can
//! create any virtual/committed opening ID without maintaining large template enums.

use atlas_onnx_tracer::node::ComputationNode;
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::poly::opening_proof::{CommittedOpeningId, OpeningId, SumcheckId, VirtualOpeningId};

/// Selects the node index source used to build opening identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningTarget {
    /// Use the current node index.
    Current,
    /// Use the node index referenced by the current node input at the given position.
    Input(usize),
    /// Use an explicit node index.
    Node(usize),
}

/// Struct-based opening ID factory scoped to one computation node.
#[derive(Debug, Clone, Copy)]
pub struct OpeningIdBuilder<'a> {
    node: &'a ComputationNode,
    sumcheck_id: SumcheckId,
}

impl<'a> OpeningIdBuilder<'a> {
    /// Create a builder scoped to one computation node.
    pub fn new(node: &'a ComputationNode) -> Self {
        Self {
            node,
            sumcheck_id: SumcheckId::NodeExecution(node.idx),
        }
    }

    /// Return a copy of this builder using the given default sumcheck ID.
    pub fn with_sumcheck_id(mut self, sumcheck_id: SumcheckId) -> Self {
        self.sumcheck_id = sumcheck_id;
        self
    }

    /// Return a copy of this builder using a target-resolved default sumcheck ID.
    pub fn with_sumcheck_target(
        mut self,
        sumcheck_target: OpeningTarget,
        sumcheck_ctor: impl FnOnce(usize) -> SumcheckId,
    ) -> Self {
        self.sumcheck_id = sumcheck_ctor(self.resolve_target(sumcheck_target));
        self
    }

    /// Return the current default sumcheck ID used by this builder.
    pub fn default_sumcheck_id(&self) -> SumcheckId {
        self.sumcheck_id
    }

    /// Convenience method for the common case of opening an input/current node output
    /// in this builder's default sumcheck context.
    pub fn node_io(&self, target: OpeningTarget) -> VirtualOpeningId {
        self.virtual_for_target(target, VirtualPolynomial::NodeOutput, self.sumcheck_id)
    }

    /// Build a virtual opening ID from already-materialized polynomial and sumcheck IDs.
    pub fn virtual_id(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> VirtualOpeningId {
        VirtualOpeningId::new(polynomial, sumcheck)
    }

    /// Build a committed opening ID from already-materialized polynomial and sumcheck IDs.
    pub fn committed_id(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> CommittedOpeningId {
        CommittedOpeningId::new(polynomial, sumcheck)
    }

    /// Build a generic opening ID wrapping a virtual opening identifier.
    pub fn opening_virtual(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> OpeningId {
        OpeningId::Virtual(self.virtual_id(polynomial, sumcheck))
    }

    /// Build a generic opening ID wrapping a committed opening identifier.
    pub fn opening_committed(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> OpeningId {
        OpeningId::Committed(self.committed_id(polynomial, sumcheck))
    }

    /// Resolve a target into a concrete node index.
    pub fn resolve_target(&self, target: OpeningTarget) -> usize {
        match target {
            OpeningTarget::Current => self.node.idx,
            OpeningTarget::Input(position) => self.node.inputs[position],
            OpeningTarget::Node(index) => index,
        }
    }

    /// Convenience helper when only the polynomial is target-dependent and sumcheck is fixed.
    fn virtual_for_target(
        &self,
        poly_target: OpeningTarget,
        poly_ctor: impl FnOnce(usize) -> VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> VirtualOpeningId {
        let polynomial = poly_ctor(self.resolve_target(poly_target));
        self.virtual_id(polynomial, sumcheck)
    }

    /// Convenience helper when only the polynomial is target-dependent and sumcheck is fixed.
    fn committed_for_target(
        &self,
        poly_target: OpeningTarget,
        poly_ctor: impl FnOnce(usize) -> CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> CommittedOpeningId {
        let polynomial = poly_ctor(self.resolve_target(poly_target));
        self.committed_id(polynomial, sumcheck)
    }

    /// Build a virtual advice opening for the current node in this builder's
    /// default sumcheck context.
    pub fn virtual_advice(
        &self,
        poly_ctor: impl FnOnce(usize) -> VirtualPolynomial,
    ) -> VirtualOpeningId {
        self.virtual_for_target(OpeningTarget::Current, poly_ctor, self.sumcheck_id)
    }

    /// Build a committed advice opening for the current node in this builder's
    /// default sumcheck context.
    pub fn committed_advice(
        &self,
        poly_ctor: impl FnOnce(usize) -> CommittedPolynomial,
    ) -> CommittedOpeningId {
        self.committed_for_target(OpeningTarget::Current, poly_ctor, self.sumcheck_id)
    }
}

/// Thin ergonomic extension trait for access from a computation node.
pub trait OpeningIdBuilderExt {
    /// Return an opening ID builder scoped to this node.
    fn openings(&self) -> OpeningIdBuilder<'_>;

    /// Return an opening ID builder scoped to this node and configured with a
    /// custom default sumcheck ID.
    fn openings_in_sumcheck(&self, sumcheck_id: SumcheckId) -> OpeningIdBuilder<'_> {
        self.openings().with_sumcheck_id(sumcheck_id)
    }

    /// Convenience wrapper around [`OpeningIdBuilder::build_opening_id`].
    fn build_opening_id(&self, target: OpeningTarget) -> VirtualOpeningId {
        self.openings().node_io(target)
    }

    /// Convenience wrapper to build a NodeOutput opening in a custom sumcheck.
    fn build_opening_id_in_sumcheck(
        &self,
        target: OpeningTarget,
        sumcheck_id: SumcheckId,
    ) -> VirtualOpeningId {
        self.openings_in_sumcheck(sumcheck_id).node_io(target)
    }
}

// Temporary disable: explicit builder usage only.
// impl OpeningIdBuilderExt for ComputationNode {
//     fn openings(&self) -> OpeningIdBuilder<'_> {
//         OpeningIdBuilder::new(self)
//     }
// }
