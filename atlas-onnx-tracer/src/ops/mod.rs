//! Operator definitions and implementations for ONNX operations.

use crate::{tensor::Tensor, utils::f32::F32};
use serde::{Deserialize, Serialize};

/// Element-wise addition operator.
pub mod add;
/// Logical AND operator.
pub mod and;
/// Broadcast operator for expanding tensor dimensions.
pub mod broadcast;
/// Clamp operator for limiting values to a range.
pub mod clamp;
/// Constant tensor operator.
pub mod constant;
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
/// MoveAxis operator for transposing dimensions.
pub mod move_axis;
/// Element-wise multiplication operator.
pub mod mul;
/// ReLU activation operator.
pub mod relu;
/// Reshape operator for changing tensor dimensions.
pub mod reshape;
/// Reciprocal square root operator.
pub mod rsqrt;
/// Division by a scalar constant operator.
pub mod scalar_const_div;
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
        Add, And, Clamp { axes: usize, max_spread: i32 },
        Constant(Tensor<i32>), Cube, Div, Einsum { equation: String },
        Erf { scale: F32 }, Gather { axis: usize }, Identity, Iff, Input,
        IsNan { out_dims: Vec<usize> }, MoveAxis { source: usize, destination: usize },
        Mul, Broadcast { shape: Vec<usize> }, ReLU, Reshape { shape:Vec<usize> },
        Rsqrt { scale: F32 }, ScalarConstDiv {divisor: i32}, SoftmaxAxes { axes: usize, scale: F32 }, Square,
        Sub, Sum { axes: Vec<usize> }, Tanh { scale: F32, tau: i32, log_table: usize },
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

impl Op for Operator {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        self.inner().f(inputs)
    }

    fn requires_shape_equality(&self) -> bool {
        self.inner().requires_shape_equality()
    }
}
