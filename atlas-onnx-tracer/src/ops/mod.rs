//! Operator definitions and implementations for ONNX operations.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Element-wise addition operator.
pub mod add;
/// Logical AND operator.
pub mod and;
/// Broadcast operator for expanding tensor dimensions.
pub mod broadcast;
/// Clamp operator for limiting values to a range.
pub mod clamp;
/// Concatenation operator.
pub mod concat;
/// Constant tensor operator.
pub mod constant;
/// Element-wise cosine operator.
pub mod cos;
/// Element-wise cube (x^3) operator.
pub mod cube;
/// Element-wise division operator.
pub mod div;
/// Einstein summation (einsum) operator for tensor contractions.
pub mod einsum;
/// Error function (erf) operator.
pub mod erf;
/// Gather operator for indexing and embedding lookups.
pub mod gather;
/// Identity (pass-through) operator.
pub mod identity;
/// Conditional if-then-else operator.
pub mod iff;
/// Input placeholder operator.
pub mod input;
/// IsNaN check operator.
pub mod is_nan;
/// Fused mean-of-squares reduction operator.
pub mod mean_of_squares;
/// MoveAxis operator for transposing dimensions.
pub mod move_axis;
/// Element-wise multiplication operator.
pub mod mul;
/// Element-wise negation operator.
pub mod neg;
/// ReLU activation operator.
pub mod relu;
/// Reshape operator for changing tensor dimensions.
pub mod reshape;
/// Reciprocal square root operator.
pub mod rsqrt;
/// Division by a scalar constant operator.
pub mod scalar_const_div;
/// Sigmoid activation operator.
pub mod sigmoid;
/// Element-wise sine operator.
pub mod sin;
/// Slice operator for extracting tensor subregions.
pub mod slice;
/// Softmax activation operator.
pub mod softmax;
/// Element-wise square (x^2) operator.
pub mod square;
/// Element-wise subtraction operator.
pub mod sub;
/// Sum reduction operator.
pub mod sum;
/// Hyperbolic tangent activation operator.
pub mod tanh;

macro_rules! define_operators {
    // Internal rule to generate struct definition
    (@struct $operator:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        #[doc = concat!(stringify!($operator), " operator.")]
        pub struct $operator;
    };
    (@struct $operator:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[doc = concat!(stringify!($operator), " operator.")]
        pub struct $operator {
            $(
                #[doc = concat!(stringify!($field), " field.")]
                pub $field: $ty
            ),*
        }
    };
    (@struct $operator:ident ( $($ty:ty),* $(,)? )) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[doc = concat!(stringify!($operator), " operator.")]
        pub struct $operator($(pub $ty),*);
    };

    // Main entry point
    (operators: [$($operator:ident $({ $($field:ident : $ty:ty),* $(,)? })? $(( $($tuple_ty:ty),* $(,)? ))?),* $(,)?]) => {
        $(
            define_operators!(@struct $operator $({ $($field: $ty),* })? $(( $($tuple_ty),* ))?);
        )*

        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        /// An enum representing all supported ONNX operators.
        pub enum Operator {
            $(
                #[doc = concat!(stringify!($operator), " variant.")]
                $operator($operator),
            )*
        }

        impl Operator {
            /// Get a reference to the inner operator as a trait object.
            pub fn inner(&self) -> &dyn Op {
                match self {
                    $(
                        Operator::$operator(op) => op,
                    )*
                }
            }
        }
    };
}
define_operators! {
    operators: [
        Add,
        Broadcast { shape: Vec<usize> },
        And,
        Clamp { axes: usize, max_spread: i32 },
        Concat { axis: isize },
        Constant(Tensor<i32>),
        Cos { scale: i32 },
        Cube { scale: i32 },
        Div,
        Einsum { equation: String, scale: i32 },
        Erf { scale: i32, tau: i32, log_table: usize },
        GatherSmall { axis: usize, dict_len: usize },
        GatherLarge { axis: usize, dict_len: usize },
        Identity,
        Iff,
        Input,
        IsNan { out_dims: Vec<usize> },
        MeanOfSquares { axes: Vec<usize>, scale: i32, count: usize },
        MoveAxis { source: usize, destination: usize },
        Mul { scale: i32 },
        Neg,
        ReLU,
        Reshape { shape:Vec<usize> },
        Rsqrt { scale: i32 },
        ScalarConstDiv {divisor: i32},
        Sigmoid { scale: i32, tau: i32, log_table: usize },
        Sin { scale: i32 },
        Slice { axis: usize, start: usize, end: usize},
        SoftmaxLastAxis { scale: i32 },
        Square { scale: i32 },
        Sub,
        Sum { axes: Vec<usize> },
        Tanh { scale: i32, tau: i32, log_table: usize },
    ]
}

/// Trait for all operators - defines how to execute the operation on input tensors.
pub trait Op {
    /// Execute the operator on the given input tensors.
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32>;

    /// Returns true if this operator requires all inputs to have matching shapes.
    /// When true, broadcast nodes will be automatically inserted before this operator.
    fn requires_shape_equality(&self) -> bool {
        false
    }

    /// Returns the scale multiplier for rebasing after this operation.
    /// - `None` means no rebase is needed (e.g., Add, Sub)
    /// - `Some(1)` means divide by `1 << scale` (e.g., Mul, Square)
    /// - `Some(2)` means divide by `1 << (scale * 2)` (e.g., Cube)
    ///
    /// This is used to maintain fixed-point representation after operations
    /// that increase the scale factor.
    fn rebase_scale_factor(&self) -> Option<usize> {
        None
    }
}

/// Broadcast-expand `lhs` and `rhs`, apply `combine` element-wise in `i64`,
/// and return the unclamped intermediate as `Tensor<i64>`.
///
/// This is the re-executable kernel the proof system calls to recover the
/// pre-saturation intermediate without storing it in the trace.
pub(super) fn sat_accumulate_pair(
    lhs: &crate::tensor::Tensor<i32>,
    rhs: &crate::tensor::Tensor<i32>,
    op_name: &str,
    combine: impl Fn(i64, i64) -> i64 + Sync,
) -> crate::tensor::Tensor<i64> {
    use crate::tensor::get_broadcasted_shape;
    use common::parallel::par_enabled;
    use rayon::prelude::*;

    let shape = get_broadcasted_shape(lhs.dims(), rhs.dims())
        .unwrap_or_else(|_| panic!("{op_name}: incompatible broadcast shapes"));
    let lhs_exp = lhs
        .expand(&shape)
        .unwrap_or_else(|_| panic!("{op_name}: expand lhs"));
    let rhs_exp = rhs
        .expand(&shape)
        .unwrap_or_else(|_| panic!("{op_name}: expand rhs"));
    let data: Vec<i64> = lhs_exp
        .data()
        .par_iter()
        .zip(rhs_exp.data().par_iter())
        .with_min_len(par_enabled())
        .map(|(&a, &b)| combine(a as i64, b as i64))
        .collect();
    crate::tensor::Tensor::new(Some(&data), &shape)
        .unwrap_or_else(|_| panic!("{op_name}: Tensor::new"))
}

/// Floor-divide (Euclidean) each element of a raw i64 accumulation by `2^bits`,
/// returning the pre-clamp rescaled `i64` value.
///
/// Floor (rather than truncating) division is deliberate and shared by every
/// fused rescaling op (einsum, `Mul`, `Square`, `Cube`): it makes the rebase a
/// pure arithmetic right shift, so the remainder `R = acc mod 2^bits` (see
/// [`rebase_remainder_i32`]) always lands in `[0, 2^bits)` even when `acc` is
/// negative — directly range-checkable .
pub(super) fn floor_rebase_i64(
    acc: &crate::tensor::Tensor<i64>,
    bits: i32,
) -> crate::tensor::Tensor<i64> {
    let divisor = 1i64 << bits;
    let data: Vec<i64> = acc.data().iter().map(|&v| v.div_euclid(divisor)).collect();
    crate::tensor::Tensor::new(Some(&data), acc.dims())
        .unwrap_or_else(|e| panic!("floor_rebase_i64: {e:?}"))
}

/// The rescaling **remainder** `R = acc mod 2^bits ∈ [0, 2^bits)` (Euclidean),
/// where the fused identity is `acc = rescaled·2^bits + R`. `R` fits `i32` for
/// any `bits < 31`.
pub(super) fn rebase_remainder_i32(
    acc: &crate::tensor::Tensor<i64>,
    bits: i32,
) -> crate::tensor::Tensor<i32> {
    let divisor = 1i64 << bits;
    let data: Vec<i32> = acc
        .data()
        .iter()
        .map(|&v| v.rem_euclid(divisor) as i32)
        .collect();
    crate::tensor::Tensor::new(Some(&data), acc.dims())
        .unwrap_or_else(|e| panic!("rebase_remainder_i32: {e:?}"))
}

/// Floor-rebase a raw i64 accumulation by `2^bits` then saturate to `i32`.
///
/// This is the fused replacement for `op + ScalarConstDiv(2^bits)`: it avoids
/// the lossy wrap-then-divide path by accumulating in `i64`, floor-dividing, and
/// clamping . Shared by the `Mul`/`Square`/`Cube` kernels.
pub(super) fn floor_rebase_clamp_i32(
    acc: &crate::tensor::Tensor<i64>,
    bits: i32,
) -> crate::tensor::Tensor<i32> {
    clamp_to_i32(&floor_rebase_i64(acc, bits))
}

/// Clamp each element of a `Tensor<i64>` into `[i32::MIN, i32::MAX]`.
pub(super) fn clamp_to_i32(t: &crate::tensor::Tensor<i64>) -> crate::tensor::Tensor<i32> {
    use common::parallel::par_enabled;
    use rayon::prelude::*;

    let data: Vec<i32> = t
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();
    crate::tensor::Tensor::new(Some(&data), t.dims())
        .unwrap_or_else(|_| panic!("clamp_to_i32: Tensor::new"))
}

/// Saturating element-wise binary operation for [`Add`] and [`Sub`].
///
/// For each consecutive input pair: accumulates via [`sat_accumulate_pair`]
/// (in `i64`), then clamps to `i32` via [`clamp_to_i32`].  Per-step clamping
/// preserves correct saturation semantics for variadic inputs.
pub(super) fn sat_binop(
    inputs: Vec<&crate::tensor::Tensor<i32>>,
    op_name: &str,
    combine: impl Fn(i64, i64) -> i64 + Sync + Copy,
) -> crate::tensor::Tensor<i32> {
    let mut output = inputs[0].clone();
    for &rhs in &inputs[1..] {
        output = clamp_to_i32(&sat_accumulate_pair(&output, rhs, op_name, combine));
    }
    output
}

/// Re-execute a binary [`Add`]/[`Sub`] node's accumulation, returning the
/// pre-clamp `i64` intermediate.
///
/// The proof system uses this to recover the accumulation (lookup-index)
/// polynomial for the saturating-clamp lookup without storing it in the trace.
/// Panics on non-`Add`/`Sub` operators or a non-binary operand list.
pub fn sat_binop_intermediate(
    operator: &Operator,
    lhs: &crate::tensor::Tensor<i32>,
    rhs: &crate::tensor::Tensor<i32>,
) -> crate::tensor::Tensor<i64> {
    match operator {
        Operator::Add(_) => sat_accumulate_pair(lhs, rhs, "Add", |a, b| a + b),
        Operator::Sub(_) => sat_accumulate_pair(lhs, rhs, "Sub", |a, b| a - b),
        _ => panic!("sat_binop_intermediate: expected Add or Sub, got {operator:?}"),
    }
}

/// Shared evaluation for the periodic trig operators ([`Sin`], [`Cos`]).
///
/// Both reduce the input modulo a `4π` approximation before applying their
/// nonlinearity at the operator's scale, so the only thing that differs
/// between them is the `nonlinearity` function itself.
pub(crate) fn eval_trig(
    input: &Tensor<i32>,
    scale: i32,
    nonlinearity: fn(&Tensor<i32>, f64) -> Tensor<i32>,
) -> Tensor<i32> {
    use crate::{
        model::consts::FOUR_PI_APPROX, tensor::ops::nonlinearities::const_rem,
        utils::quantize::scale_to_multiplier,
    };
    let remainder = const_rem(input, FOUR_PI_APPROX);
    nonlinearity(&remainder, scale_to_multiplier(scale))
}

impl Op for Operator {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        self.inner().f(inputs)
    }

    fn requires_shape_equality(&self) -> bool {
        self.inner().requires_shape_equality()
    }
}
