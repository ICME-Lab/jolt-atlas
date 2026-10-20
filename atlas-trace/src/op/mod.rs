//! The operator registry: one definition site per op.
//!
//! Each operator is a struct implementing [`OpSpec`], which covers everything
//! the rest of the crate needs to know about it: kernel ([`OpSpec::eval`]),
//! shape inference, clamp-width bound, padding behavior, rescale divisor, and
//! display. Adding an op means one struct, one `impl OpSpec`, and one line in
//! the [`define_ops!`] invocation — the compiler enforces completeness
//! everywhere else.

mod activation;
mod einsum;
mod fused;
mod linear;
mod softmax;
mod structural;

pub use einsum::EinsumEq;

use std::fmt;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{
    error::{KernelError, ShapeError},
    id::{Scale, SemVersion},
    semantics::tables::{self, ActivationTable, ExpLutDecomposed, TableKind},
    tensor::Tensor,
    trace::NodeAux,
};

/// Result of evaluating an operator: the output tensor plus any auxiliary
/// witness captured for the prover.
pub struct Eval {
    /// The node's quantized output.
    pub output: Tensor<i32>,
    /// Fused-rescale intermediates or the softmax witness, if the op
    /// produces one.
    pub aux: Option<NodeAux>,
}

impl Eval {
    /// An evaluation with no auxiliary witness.
    pub fn plain(output: Tensor<i32>) -> Self {
        Eval { output, aux: None }
    }
}

/// Everything a kernel may depend on besides its input tensors.
pub struct EvalCtx<'a> {
    /// The model's global fixed-point scale.
    pub scale: Scale,
    /// The semantics version tables are generated under.
    pub version: SemVersion,
    /// Memoized activation tables (generation is deterministic, so caching
    /// is purely an optimization).
    pub tables: &'a TableCache,
}

/// Inputs to the static clamp-width bound analysis.
pub struct BoundCtx<'a> {
    /// The model's global fixed-point scale.
    pub scale: Scale,
    /// Each operand's (padded) dims.
    pub input_dims: Vec<&'a [usize]>,
    /// Each operand's constant data, when the operand is a `Constant` node.
    pub input_consts: Vec<Option<&'a Tensor<i32>>>,
}

/// How an operator behaves under power-of-two padding of its operands.
///
/// Declared, not inferred: the padding pass refuses ops without a variant,
/// so a new op cannot silently miscompute on padded lanes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PadBehavior {
    /// Zero inputs in padded lanes produce zero outputs in padded lanes
    /// (elementwise arithmetic, einsum against zero-padded weights, …).
    ZeroPreserving,
    /// Pure index plumbing; the op's shape-carrying fields are rewritten by
    /// [`OpSpec::apply_padding`] (reshape, broadcast, slice, concat, …).
    Structural,
    /// Padded lanes would corrupt the result unless the op masks them
    /// internally (softmax via `active_len`, mean-of-squares via `count`
    /// vs `padded_count`).
    NeedsMask,
    /// Padded (zero) lanes map to the op's value at zero, which is not zero
    /// (sigmoid: `S/2`; cos: `S`). Deterministic and identical to the old
    /// crate's behavior; downstream multiplicative/structural ops absorb it,
    /// but a reduction over a padded axis would see these constants.
    ConstantFill,
}

/// What the padding pass tells an op when rewriting it.
pub struct PadCtx<'a> {
    /// Each operand's dims before padding.
    pub input_dims: Vec<&'a [usize]>,
    /// This node's output dims before padding.
    pub output_dims: &'a [usize],
}

/// The single per-op behavior trait. See the module docs.
pub trait OpSpec: fmt::Display {
    /// Execute the kernel. Pure given `(inputs, ctx)`.
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError>;

    /// Output dims from input dims. Also validates arity. Shape rules are
    /// dim-uniform, so the same function serves padded and unpadded graphs.
    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError>;

    /// Validate the op's static parameters (axis ranges, equation form, …).
    /// Runs once at graph validation.
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Sound bound on the magnitude of the pre-clamp value, if this op ends
    /// in a saturating clamp. `None` = no clamp lookup on this node.
    fn clamp_bound(&self, _ctx: &BoundCtx<'_>) -> Option<u128> {
        None
    }

    /// Padding behavior class (see [`PadBehavior`]).
    fn padding(&self) -> PadBehavior;

    /// Rewrite shape-carrying fields when the compile-time padding pass
    /// runs. Called before the node's `output_dims` are padded.
    fn apply_padding(&mut self, _ctx: &PadCtx<'_>) -> Result<(), String> {
        Ok(())
    }

    /// The fused-rescale divisor, if this op splits its accumulation into
    /// `(quotient, remainder)`: mul/square/einsum `2^s`, cube `2^(2s)`,
    /// mean-of-squares `count·2^s`.
    fn rebase_divisor(&self, _scale: Scale) -> Option<i64> {
        None
    }
}

/// Memoized activation tables for one `(scale, version)` pair. Generation is
/// deterministic, so this is purely an optimization; the executor owns one
/// per model.
#[derive(Default)]
pub struct TableCache {
    cells: [OnceLock<ActivationTable>; 5],
    exp: OnceLock<ExpLutDecomposed>,
}

impl TableCache {
    /// The table for `kind`, generated on first use.
    pub fn get(&self, kind: TableKind, scale: Scale, version: SemVersion) -> &ActivationTable {
        let slot = match kind {
            TableKind::Tanh => 0,
            TableKind::Sigmoid => 1,
            TableKind::Erf => 2,
            TableKind::Sin => 3,
            TableKind::Cos => 4,
        };
        self.cells[slot].get_or_init(|| tables::table(kind, scale, version))
    }

    /// The softmax exp sub-tables, generated on first use.
    pub fn exp_lut(&self, scale: Scale, version: SemVersion) -> &ExpLutDecomposed {
        self.exp
            .get_or_init(|| tables::exp_lut_decomposed(scale, version))
    }
}

impl fmt::Debug for TableCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TableCache")
    }
}

/// Check that every input has the same dims and return them; `expected`
/// fixes the arity (`None` = any arity ≥ 1).
pub(crate) fn same_dims(
    inputs: &[&[usize]],
    expected: Option<usize>,
) -> Result<Vec<usize>, ShapeError> {
    if let Some(n) = expected
        && inputs.len() != n
    {
        return Err(ShapeError::WrongArity {
            expected: n,
            got: inputs.len(),
        });
    }
    let first = *inputs.first().ok_or(ShapeError::WrongArity {
        expected: expected.unwrap_or(1),
        got: 0,
    })?;
    for d in &inputs[1..] {
        if *d != first {
            return Err(ShapeError::BroadcastIncompatible {
                lhs: first.to_vec(),
                rhs: d.to_vec(),
            });
        }
    }
    Ok(first.to_vec())
}

macro_rules! define_ops {
    ($($(#[doc = $doc:literal])* $name:ident $({ $($field:ident : $ty:ty),* $(,)? })? $(( $($tuple_ty:ty),* ))?),* $(,)?) => {
        $(
            define_ops!(@struct $(#[doc = $doc])* $name $({ $($field: $ty),* })? $(( $($tuple_ty),* ))?);
        )*

        /// The closed set of operators. The prover dispatches exhaustively
        /// over these variants, so the enum stays closed; per-op behavior
        /// lives in each struct's [`OpSpec`] impl.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub enum Op {
            $(
                #[doc = concat!("`", stringify!($name), "` operator.")]
                $name($name),
            )*
        }

        impl Op {
            /// The op's behavior as a trait object.
            pub fn spec(&self) -> &dyn OpSpec {
                match self { $( Op::$name(op) => op, )* }
            }

            /// Mutable access for the padding pass.
            pub(crate) fn spec_mut(&mut self) -> &mut dyn OpSpec {
                match self { $( Op::$name(op) => op, )* }
            }
        }

        impl fmt::Display for Op {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.spec().fmt(f)
            }
        }
    };

    (@struct $(#[doc = $doc:literal])* $name:ident) => {
        $(#[doc = $doc])*
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name;
    };
    (@struct $(#[doc = $doc:literal])* $name:ident { $($field:ident : $ty:ty),* }) => {
        $(#[doc = $doc])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            $(
                #[doc = concat!("`", stringify!($field), "` parameter.")]
                pub $field: $ty,
            )*
        }
    };
    (@struct $(#[doc = $doc:literal])* $name:ident ( $($tuple_ty:ty),* )) => {
        $(#[doc = $doc])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct $name($(pub $tuple_ty),*);
    };
}

define_ops! {
    /// Graph input placeholder; the executor substitutes the supplied tensor.
    Input,
    /// Embedded constant tensor (quantized weights or raw integers).
    Constant(Tensor<i32>),
    /// Pass-through.
    Identity,
    /// Saturating element-wise addition (i64 accumulate, clamp to i32).
    Add,
    /// Saturating element-wise subtraction.
    Sub,
    /// Element-wise negation.
    Neg,
    /// Fused element-wise product: i64 accumulate, floor-rescale by `2^s`,
    /// saturating clamp. `rebase: false` is the raw building-block product
    /// (no rescale, no clamp).
    Mul { rebase: bool },
    /// Fused element-wise square (`rebase` as in [`Mul`]).
    Square { rebase: bool },
    /// Fused element-wise cube; rescales by `2^(2s)` (`rebase` as in [`Mul`]).
    Cube { rebase: bool },
    /// Element-wise floor division.
    Div,
    /// Floor division by a constant scalar.
    ScalarConstDiv { divisor: i32 },
    /// Fused tensor contraction: i64 accumulate, floor-rescale by `2^s`,
    /// saturating clamp.
    Einsum { equation: EinsumEq },
    /// Sum over axes (i64 accumulate, saturating clamp).
    Sum { axes: Vec<usize> },
    /// Fused mean of squares over axes: `Σx²` in i64, floor-divide by
    /// `count·2^s`, saturating clamp. `count` is the semantic element count
    /// (the mean denominator); `padded_count` is the power-of-two-rounded
    /// product the prover's sumcheck cube uses.
    MeanOfSquares { axes: Vec<usize>, count: usize, padded_count: usize },
    /// Integer softmax over the last axis with full witness capture.
    /// `active_len` masks power-of-two padding on the last axis
    /// (`None` = the whole row is live).
    SoftmaxLastAxis { active_len: Option<usize> },
    /// Reciprocal square root `⌊√⌊S³/x⌋⌋` (non-positive inputs map to 0).
    Rsqrt,
    /// Table-lookup tanh over the clamped domain `[-2^(s+3), 2^(s+3))`.
    Tanh,
    /// Table-lookup logistic sigmoid over the clamped domain.
    Sigmoid,
    /// Table-lookup Gauss error function over the clamped domain.
    Erf,
    /// Table-lookup sine over the period-reduced domain.
    Sin,
    /// Table-lookup cosine over the period-reduced domain.
    Cos,
    /// Rectified linear unit `max(x, 0)`.
    ReLU,
    /// Symmetric clamp into `[-2^bound_log, 2^bound_log)`.
    Clamp { bound_log: u32 },
    /// Element-wise select `mask·a + (1-mask)·b`; mask must be 0/1.
    Iff,
    /// Boolean product; both operands must be 0/1.
    And,
    /// Always-zero output (the quantized domain has no NaN).
    IsNan,
    /// Embedding lookup on axis 0 against a dictionary of < 2^16 rows
    /// (after power-of-two rounding); proven with a small table.
    GatherSmall { axis: usize, dict_len: usize },
    /// Embedding lookup on axis 0 against a large dictionary.
    GatherLarge { axis: usize, dict_len: usize },
    /// Extract `start..end` along `axis`.
    Slice { axis: usize, start: usize, end: usize },
    /// Concatenate along `axis`.
    Concat { axis: usize },
    /// Reinterpret dims (same element count).
    Reshape { shape: Vec<usize> },
    /// Broadcast-expand to `shape` (NumPy semantics).
    Broadcast { shape: Vec<usize> },
    /// Move one axis to a new position.
    MoveAxis { source: usize, destination: usize },
}
