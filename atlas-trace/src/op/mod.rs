//! The operator registry: one definition site per op.
//!
//! Each operator is a struct implementing [`OpSpec`], which covers everything
//! the rest of the crate needs to know about it: kernel ([`OpSpec::eval`]),
//! shape inference, output range check, padding behavior, rescale divisor, and
//! display. Adding an op means one struct, one `impl OpSpec`, and one line in
//! the [`define_ops!`] invocation — the compiler enforces completeness
//! everywhere else.

mod activation;
mod einsum;
mod fused;
mod linear;
mod softmax;
mod structural;

pub use activation::SmoothReluForm;
pub use einsum::EinsumEq;

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};

use crate::{
    error::{KernelError, ShapeError},
    id::{Scale, SemVersion},
    semantics::{
        softmax::{SoftmaxTables, softmax_tables},
        tables::{self, ActivationTable, TableKind},
    },
    tensor::Tensor,
    trace::NodeAux,
};

/// The single per-op behavior trait. See the module docs.
pub trait OpSpec: fmt::Display {
    /// Execute the kernel. Pure given `(inputs, ctx)`.
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError>;

    /// Output dims from input dims. Also validates arity. Shape rules are
    /// dim-uniform, so the same function serves padded and unpadded graphs.
    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError>;

    /// Whether the kernel reads a generated table ([`Self::warm_tables`]).
    fn reads_tables(&self) -> bool {
        false
    }

    /// Generate the tables the kernel reads, so that an execution does not wait for them.
    fn warm_tables(&self, _ctx: &EvalCtx<'_>) {}

    /// Validate the op's static parameters (axis ranges, equation form, …).
    /// Runs once at graph validation.
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Whether the op's result is exact arithmetic narrowed to `i32` (a sum, difference, axis
    /// sum or rescaled quotient), so that a proof must show the output lies in `i32`.
    fn checks_output(&self) -> bool {
        false
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

    /// The level of the output given the operands' levels: a value at level `j` represents
    /// `v / 2^(j·S)`, level 1 is an `i32` and a higher level is wide. `Err` if the op has no
    /// form for operands at these levels; by default an op takes level-1 operands and gives
    /// a level-1 value.
    fn level(&self, operands: &[u8]) -> Result<u8, String> {
        if operands.iter().all(|&level| level == 1) {
            Ok(1)
        } else {
            Err(format!("no form for operands at levels {operands:?}"))
        }
    }

    /// The kernel when an operand or the output is wide ([`Self::level`]).
    fn eval_wide(
        &self,
        _inputs: &[Operand<'_>],
        levels: &[u8],
        _ctx: &EvalCtx<'_>,
    ) -> Result<Eval, KernelError> {
        Err(KernelError::NoWideForm {
            op: "op",
            levels: levels.to_vec(),
        })
    }

    /// `log₂` of the magnitude a wide output of operands at these levels must stay below
    /// ([`WIDE_LIMIT`], or less where the kernel accumulates in a narrower integer).
    fn wide_limit(&self, _operands: &[u8]) -> u32 {
        WIDE_LIMIT
    }

    /// A bound on the magnitude of a wide output's entries, given bounds on the operands',
    /// with the operands' levels and dims. Asked only of an op whose output is wide.
    fn wide_bound(&self, _operands: &[u128], _shape: &WideShape<'_>) -> u128 {
        u128::MAX
    }
}

/// `log₂` of the magnitude every wide value stays below: under it both sides of a relation
/// over wide values are integers of magnitude below half the 128-bit field's modulus, so
/// their equality in the field is their equality over the integers.
pub const WIDE_LIMIT: u32 = 125;

/// The highest level a value has: three deferred products of level-1 operands.
pub const MAX_LEVEL: u8 = 4;

/// What a wide output's bound depends on besides its operands' bounds.
pub struct WideShape<'a> {
    /// The operands' levels.
    pub levels: &'a [u8],
    /// The operands' dims.
    pub dims: &'a [&'a [usize]],
    /// The model's scale.
    pub scale: Scale,
}

/// `bound · 2^((to − from)·S)`: the bound of a value lifted from level `from` to level `to`.
pub fn lifted_bound(bound: u128, from: u8, to: u8, scale: Scale) -> u128 {
    let bits = u32::from(to.saturating_sub(from)) * scale.log2();
    if bits >= bound.leading_zeros() {
        u128::MAX
    } else {
        bound << bits
    }
}

/// An operand as a kernel with a wide form reads it.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    /// A level-1 value.
    Narrow(&'a Tensor<i32>),
    /// A wide value.
    Wide(&'a Tensor<i128>),
}

impl Operand<'_> {
    /// The operand's dims.
    pub fn dims(&self) -> &[usize] {
        match self {
            Operand::Narrow(tensor) => tensor.dims(),
            Operand::Wide(tensor) => tensor.dims(),
        }
    }

    /// Entry `index`, exactly.
    pub fn at(&self, index: usize) -> i128 {
        match self {
            Operand::Narrow(tensor) => tensor.data()[index] as i128,
            Operand::Wide(tensor) => tensor.data()[index],
        }
    }
}

/// Memoized activation tables for one `(scale, version)` pair. Generation is
/// deterministic, so this is purely an optimization; the executor owns one
/// per model. Cloning yields a fresh (empty) cache.
#[derive(Default)]
pub struct TableCache {
    cells: [OnceLock<Arc<ActivationTable>>; 8],
    softmax: OnceLock<SoftmaxTables>,
}

impl TableCache {
    /// The table for `kind`: the process's shared one, so a table the prover also reads is
    /// generated once.
    pub fn get(&self, kind: TableKind, scale: Scale, version: SemVersion) -> &ActivationTable {
        let slot = match kind {
            TableKind::Tanh => 0,
            TableKind::Sigmoid => 1,
            TableKind::Erf => 2,
            TableKind::Sin => 3,
            TableKind::Cos => 4,
            TableKind::GeluErf => 5,
            TableKind::GeluTanh => 6,
            TableKind::Silu => 7,
        };
        self.cells[slot].get_or_init(|| tables::shared_table(kind, scale, version))
    }

    /// The log-partition softmax tables, generated on first use.
    pub fn softmax_tables(&self, scale: Scale, version: SemVersion) -> &SoftmaxTables {
        self.softmax.get_or_init(|| softmax_tables(scale, version))
    }
}

impl Clone for TableCache {
    fn clone(&self) -> Self {
        TableCache::default()
    }
}

impl fmt::Debug for TableCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TableCache")
    }
}

/// Result of evaluating an operator: the output tensor plus any auxiliary
/// witness captured for the prover.
pub struct Eval {
    /// The node's quantized output; without data when the value is wide.
    pub output: Tensor<i32>,
    /// The node's exact value, when it is wide.
    pub wide: Option<Tensor<i128>>,
    /// The advice the op captures for the prover, if any: fused-rescale intermediates, the
    /// softmax witness, a division remainder, the rsqrt or trig-period intermediates.
    pub aux: Option<NodeAux>,
}

impl Eval {
    /// An evaluation with no auxiliary witness.
    pub fn plain(output: Tensor<i32>) -> Self {
        Eval {
            output,
            wide: None,
            aux: None,
        }
    }

    /// The evaluation of a node whose value is wide.
    pub fn wide(value: Tensor<i128>) -> Self {
        Eval {
            output: Tensor::absent(value.dims()),
            wide: Some(value),
            aux: None,
        }
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

/// How an operator behaves under power-of-two padding of its operands.
///
/// This classification is *declared documentation plus per-op masking*, not
/// a property the padding pass enforces: [`crate::model`]'s `pad_graph`
/// rewrites shapes via [`OpSpec::apply_padding`] but never consults this
/// variant. Each `NeedsMask` op masks itself (softmax via `active_len`,
/// mean-of-squares via `count`), and `Broadcast` zeroes the padded lanes of
/// the axes it expands (its `extent`); `ConstantFill` and the non-zero
/// padded lanes a `Gather` can produce rely on downstream absorption
/// (zero-padded weights, output cropping) that no analysis currently
/// verifies.
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
    /// Embedded constant tensor (quantized weights or raw integers),
    /// behind `Arc` so cloning an `Op` never deep-copies weights.
    Constant(std::sync::Arc<Tensor<i32>>),
    /// Pass-through.
    Identity,
    /// Element-wise addition; a sum outside `i32` is an execution error.
    Add,
    /// Element-wise subtraction; a difference outside `i32` is an execution error.
    Sub,
    /// Element-wise negation; `-i32::MIN` is an execution error.
    Neg,
    /// Fused element-wise product: i64 accumulate, floor-rescale by `2^s`; a
    /// quotient outside `i32` is an execution error. `rebase: false` is the raw
    /// building-block product (no rescale, wrapping). `deferred` is the exact product as a
    /// wide value, of operands at any levels, with no rescale.
    Mul { rebase: bool, deferred: bool },
    /// Fused element-wise square (`rebase` as in [`Mul`]).
    Square { rebase: bool },
    /// Fused element-wise cube; rescales by `2^(2s)` (`rebase` as in [`Mul`]).
    Cube { rebase: bool },
    /// Floor division by a constant scalar.
    ScalarConstDiv { divisor: i32 },
    /// Fused tensor contraction: i64 accumulate, floor-rescale by `2^s`; a
    /// quotient outside `i32` is an execution error. `deferred` is the exact contraction as
    /// a wide value, with no rescale.
    Einsum { equation: EinsumEq, deferred: bool },
    /// A wide value at level `j` taken to level 1: the Euclidean quotient by `2^((j−1)·s)`,
    /// which must lie in `i32`, with the remainder captured.
    Rescale,
    /// Sum over axes (i64 accumulate); a sum outside `i32` is an execution error.
    Sum { axes: Vec<usize> },
    /// Fused mean of squares over axes: `Σx²` in i64, floor-divide by
    /// `count·2^s`; a quotient outside `i32` is an execution error. `count` is the semantic element count
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
    /// A smooth activation `x·g(x)` with `f(x) − f(−x) = x` (GELU in its erf or tanh form,
    /// SiLU), computed as `max(x, 0)` plus a table-lookup correction that vanishes outside
    /// the table's clamped domain.
    SmoothRelu { form: SmoothReluForm },
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
    /// Broadcast-expand to `shape` (NumPy semantics). `extent` is the target shape before
    /// padding: on an expanded axis, the lanes at or past its extent are zero, as every
    /// padded lane is.
    Broadcast { shape: Vec<usize>, extent: Vec<usize> },
    /// Move one axis to a new position.
    MoveAxis { source: usize, destination: usize },
}
