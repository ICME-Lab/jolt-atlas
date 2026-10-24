//! Typed errors for every stage of the pipeline. No entry point panics on
//! malformed input; panics are reserved for internal invariant violations.

use thiserror::Error;

use crate::id::NodeId;

/// Shape mismatches raised by tensor constructors and kernels.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Data length does not match the product of the dimensions.
    #[error("data length {len} does not match dims {dims:?} (expected {expected})")]
    LengthMismatch {
        /// Supplied data length.
        len: usize,
        /// Requested dimensions.
        dims: Vec<usize>,
        /// Product of the requested dimensions.
        expected: usize,
    },
    /// Two shapes cannot be broadcast together.
    #[error("shapes {lhs:?} and {rhs:?} are not broadcast-compatible")]
    BroadcastIncompatible {
        /// Left-hand shape.
        lhs: Vec<usize>,
        /// Right-hand shape.
        rhs: Vec<usize>,
    },
    /// A target shape is not a valid expansion of the source shape.
    #[error("cannot expand {from:?} to {to:?}")]
    ExpandIncompatible {
        /// Source shape.
        from: Vec<usize>,
        /// Target shape.
        to: Vec<usize>,
    },
    /// An axis index is out of range for the tensor's rank.
    #[error("axis {axis} out of range for rank {rank}")]
    AxisOutOfRange {
        /// Offending axis.
        axis: usize,
        /// Tensor rank.
        rank: usize,
    },
    /// A slice range is invalid for the dimension it addresses.
    #[error("slice {start}..{end} invalid for axis {axis} of size {size}")]
    SliceOutOfRange {
        /// Sliced axis.
        axis: usize,
        /// Slice start (inclusive).
        start: usize,
        /// Slice end (exclusive).
        end: usize,
        /// Size of the axis.
        size: usize,
    },
    /// An operator received the wrong number of inputs.
    #[error("expected {expected} input(s), got {got}")]
    WrongArity {
        /// Required input count.
        expected: usize,
        /// Supplied input count.
        got: usize,
    },
    /// A padding target is smaller than the source shape.
    #[error("cannot pad {from:?} to smaller target {to:?}")]
    PadTooSmall {
        /// Source shape.
        from: Vec<usize>,
        /// Target shape.
        to: Vec<usize>,
    },
    /// Generic shape constraint violation with a human-readable reason.
    #[error("{0}")]
    Invalid(String),
}

/// Errors raised while building or validating a [`crate::graph::Graph`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Scale outside the supported range.
    #[error("scale must be in 1..=15, got {scale}")]
    InvalidScale {
        /// Rejected log2 scale.
        scale: u8,
    },
    /// A node references an input at or after itself (breaks topological order).
    #[error("node {node} references input {input}, which is not an earlier node")]
    NotTopological {
        /// Offending node.
        node: NodeId,
        /// Referenced input id.
        input: NodeId,
    },
    /// An input/output list references a node id outside the graph.
    #[error("node id {id} out of range (graph has {len} nodes)")]
    IdOutOfRange {
        /// Offending id.
        id: NodeId,
        /// Number of nodes in the graph.
        len: usize,
    },
    /// A graph input must be an `Input` node, and vice versa.
    #[error("node {node}: {reason}")]
    InvalidIo {
        /// Offending node.
        node: NodeId,
        /// Why the IO wiring is invalid.
        reason: String,
    },
    /// An einsum equation string could not be parsed or is unsupported.
    #[error("invalid einsum equation {equation:?}: {reason}")]
    InvalidEinsum {
        /// Offending equation text.
        equation: String,
        /// Why it was rejected.
        reason: String,
    },
    /// An operator's static parameters are invalid.
    #[error("node {node} ({op}): {reason}")]
    InvalidOp {
        /// Offending node.
        node: NodeId,
        /// Operator display name.
        op: String,
        /// Why the parameters are invalid.
        reason: String,
    },
}

/// Errors raised by [`crate::model::Model::compile`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// Graph validation failed.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Shape inference failed at a node.
    #[error("node {node} ({op}): shape inference failed: {source}")]
    Shape {
        /// Offending node.
        node: NodeId,
        /// Operator display name.
        op: String,
        /// Underlying shape error.
        #[source]
        source: ShapeError,
    },
    /// Shape inference disagreed with the dims the graph carries.
    #[error("node {node} ({op}): inferred shape {inferred:?} != declared {declared:?}")]
    ShapeMismatch {
        /// Offending node.
        node: NodeId,
        /// Operator display name.
        op: String,
        /// Shape computed from the inputs.
        inferred: Vec<usize>,
        /// Shape the graph declared.
        declared: Vec<usize>,
    },
    /// The padding pass could not rewrite an op's shape-carrying fields.
    #[error("node {node} ({op}): padding failed: {reason}")]
    Padding {
        /// Offending node.
        node: NodeId,
        /// Operator display name.
        op: String,
        /// Why padding failed.
        reason: String,
    },
}

/// Errors raised while executing a compiled model.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ExecError {
    /// Wrong number of input tensors supplied.
    #[error("model expects {expected} input(s), got {got}")]
    WrongInputCount {
        /// Required input count.
        expected: usize,
        /// Supplied input count.
        got: usize,
    },
    /// An input tensor's shape does not match the model input.
    #[error("input {index}: shape {got:?} does not match expected {expected:?}")]
    InputShape {
        /// Positional input index.
        index: usize,
        /// Declared input dims.
        expected: Vec<usize>,
        /// Supplied dims.
        got: Vec<usize>,
    },
    /// A kernel failed at a node.
    #[error("node {node} ({op}): {source}")]
    Kernel {
        /// Offending node.
        node: NodeId,
        /// Operator display name.
        op: String,
        /// Underlying kernel error.
        #[source]
        source: KernelError,
    },
}

/// Errors raised inside an operator kernel (contextualized into
/// [`ExecError::Kernel`] by the executor).
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// A shape constraint failed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// Integer division by zero in `Div`.
    #[error("division by zero at element {element}")]
    DivByZero {
        /// Flat element index of the zero divisor.
        element: usize,
    },
    /// A gather index is out of range for the dictionary.
    #[error("gather index {index} out of range for dict of {dict_len}")]
    GatherOutOfRange {
        /// Offending index value.
        index: i64,
        /// Dictionary length.
        dict_len: usize,
    },
    /// An operand that must be boolean (0/1) held another value.
    #[error("{op} operand must be boolean (0/1), found {value} at element {element}")]
    NotBoolean {
        /// Operator name.
        op: &'static str,
        /// Offending value.
        value: i32,
        /// Flat element index.
        element: usize,
    },
    /// A computed witness value does not fit the witness's integer width.
    #[error("{op}: {what} {value} exceeds the i32 witness range")]
    WitnessOverflow {
        /// Operator name.
        op: &'static str,
        /// Which witness value overflowed.
        what: &'static str,
        /// The overflowing value.
        value: i64,
    },
}
