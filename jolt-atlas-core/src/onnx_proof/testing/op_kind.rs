//! Payload-free mirror of [`atlas_trace::Op`] for coverage bookkeeping.

use atlas_trace::Op;

/// One entry per `Op` variant. [`OpKind::of`] is a total match with no
/// wildcard arm, so adding a variant to `define_ops!` in atlas-trace is a
/// compile error in this crate's tests until the op-surface table declares a
/// stance for it (a proving case, an ignored case, or an expected rejection).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    strum_macros::EnumIter,
    strum_macros::Display,
)]
pub(crate) enum OpKind {
    Input,
    Constant,
    Identity,
    Add,
    Sub,
    Neg,
    Mul,
    Square,
    Cube,
    Div,
    ScalarConstDiv,
    Einsum,
    Sum,
    MeanOfSquares,
    SoftmaxLastAxis,
    Rsqrt,
    Tanh,
    Sigmoid,
    Erf,
    Sin,
    Cos,
    ReLU,
    Clamp,
    Iff,
    And,
    IsNan,
    GatherSmall,
    GatherLarge,
    Slice,
    Concat,
    Reshape,
    Broadcast,
    MoveAxis,
}

impl OpKind {
    /// The kind of an op. Exhaustive on purpose: no `_ =>` arm.
    pub(crate) fn of(op: &Op) -> OpKind {
        match op {
            Op::Input(_) => OpKind::Input,
            Op::Constant(_) => OpKind::Constant,
            Op::Identity(_) => OpKind::Identity,
            Op::Add(_) => OpKind::Add,
            Op::Sub(_) => OpKind::Sub,
            Op::Neg(_) => OpKind::Neg,
            Op::Mul(_) => OpKind::Mul,
            Op::Square(_) => OpKind::Square,
            Op::Cube(_) => OpKind::Cube,
            Op::Div(_) => OpKind::Div,
            Op::ScalarConstDiv(_) => OpKind::ScalarConstDiv,
            Op::Einsum(_) => OpKind::Einsum,
            Op::Sum(_) => OpKind::Sum,
            Op::MeanOfSquares(_) => OpKind::MeanOfSquares,
            Op::SoftmaxLastAxis(_) => OpKind::SoftmaxLastAxis,
            Op::Rsqrt(_) => OpKind::Rsqrt,
            Op::Tanh(_) => OpKind::Tanh,
            Op::Sigmoid(_) => OpKind::Sigmoid,
            Op::Erf(_) => OpKind::Erf,
            Op::Sin(_) => OpKind::Sin,
            Op::Cos(_) => OpKind::Cos,
            Op::ReLU(_) => OpKind::ReLU,
            Op::Clamp(_) => OpKind::Clamp,
            Op::Iff(_) => OpKind::Iff,
            Op::And(_) => OpKind::And,
            Op::IsNan(_) => OpKind::IsNan,
            Op::GatherSmall(_) => OpKind::GatherSmall,
            Op::GatherLarge(_) => OpKind::GatherLarge,
            Op::Slice(_) => OpKind::Slice,
            Op::Concat(_) => OpKind::Concat,
            Op::Reshape(_) => OpKind::Reshape,
            Op::Broadcast(_) => OpKind::Broadcast,
            Op::MoveAxis(_) => OpKind::MoveAxis,
        }
    }
}
