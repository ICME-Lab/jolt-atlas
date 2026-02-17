use crate::{tensor::Tensor, utils::f32::F32};
use serde::{Deserialize, Serialize};

pub mod add;
pub mod and;
pub mod and2;
pub mod broadcast;
pub mod constant;
pub mod cube;
pub mod div;
pub mod einsum;
pub mod erf;
pub mod gather;
pub mod identity;
pub mod iff;
pub mod input;
pub mod is_nan;
pub mod lt;
pub mod move_axis;
pub mod mul;
pub mod noop;
pub mod relu;
pub mod reshape;
pub mod rsqrt;
pub mod scalar_const_div;
pub mod shr;
pub mod softmax;
pub mod square;
pub mod sub;
pub mod sum;
pub mod tanh;

macro_rules! define_operators {
    // Internal rule to generate struct definition
    (@struct $operator:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $operator;
    };
    (@struct $operator:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $operator {
            $(pub $field: $ty),*
        }
    };
    (@struct $operator:ident ( $($ty:ty),* $(,)? )) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $operator($(pub $ty),*);
    };

    // Main entry point
    (operators: [$($operator:ident $({ $($field:ident : $ty:ty),* $(,)? })? $(( $($tuple_ty:ty),* $(,)? ))?),* $(,)?]) => {
        $(
            define_operators!(@struct $operator $({ $($field: $ty),* })? $(( $($tuple_ty),* ))?);
        )*

        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub enum Operator {
            $(
                $operator($operator),
            )*
        }

        impl Operator {
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
        Add, And, And2, Constant(Tensor<i32>), Cube, Div, Einsum { equation: String },
        Erf { scale: F32 }, Gather { axis: usize }, Identity, Iff, Input,
        IsNan { out_dims: Vec<usize> }, MoveAxis { source: usize, destination: usize },
        Mul, Broadcast { shape: Vec<usize> }, Noop, ReLU, Reshape { shape:Vec<usize> },
        Rsqrt { scale: F32 }, ScalarConstDiv {divisor: i32}, Shr, SoftmaxAxes { axes: usize, scale: F32 }, Square,
        Sub, Sum { axes: Vec<usize> }, Tanh { scale: F32, tau: i32, log_table: usize }, ULessThan,
    ]
}

pub trait Op {
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
