//! Reduction operator handlers: Sum, MeanOfSquares
//!
//! This module provides handlers for reduction operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::tract_core::ops::nn::Reduce;

#[cfg(feature = "fused-ops")]
use crate::ops::MeanOfSquares;
use crate::{
    node::ComputationNode,
    ops::{Mul, Operator, ScalarConstDiv, Sum},
    utils::{
        handler_builder::HandlerBuilder,
        parser::{DecompositionBuilder, load_op},
    },
};

use super::{HandlerContext, OpHandlerFn};

/// Returns a map of reduction operator names to their handler functions.
pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Reduce<Sum>", handle_reduce_sum as OpHandlerFn),
        (
            "Reduce<MeanOfSquares>",
            handle_reduce_mean_of_squares as OpHandlerFn,
        ),
    ])
}

/// Reduce<Sum>: Sum reduction along specified axes.
fn handle_reduce_sum(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert_eq!(hctx.internal_input_indices.len(), 1);
    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes = op.axes.into_iter().collect();

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Sum(Sum { axes }))
        .build()
}

/// Reduce<MeanOfSquares>: Fused mean-of-squares when `fused-ops` is enabled,
/// otherwise decomposed into Square -> Sum -> Div(count).
///
/// HACK: When `pre_rebase_nonlinear` is true (non-fused path), decomposes into:
///   x' = x/S -> sq = Mul(x', x) -> Sum(sq) -> Div(count/S)
/// to avoid i32 overflow in large models.
/// TODO: Remove pre_rebase_nonlinear path once fused i64 ops are default.
fn handle_reduce_mean_of_squares(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert!(hctx.internal_input_indices.len() == 1);

    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes: Vec<usize> = op.axes.into_iter().collect();

    #[cfg(feature = "fused-ops")]
    {
        let scale = hctx.run_args.scale;
        HandlerBuilder::new(hctx)
            .simple_op(Operator::MeanOfSquares(MeanOfSquares { axes, scale }))
            .build()
    }

    #[cfg(not(feature = "fused-ops"))]
    {
        let input_dims = hctx.internal_input_nodes[0].output_dims.clone();
        let output_dims = hctx.output_dims.clone();
        let dividend_value =
            (input_dims.iter().product::<usize>() / output_dims.iter().product::<usize>()) as i32;

        // HACK: pre_rebase_nonlinear decomposes to avoid i32 overflow in Square.
        // TODO: Remove once fused i64-precision ops are the default path.
        if hctx.run_args.pre_rebase_nonlinear {
            let scale = hctx.run_args.scale;
            let s = 1i32 << scale;
            assert!(
                dividend_value % s == 0,
                "pre_rebase_nonlinear requires count ({dividend_value}) divisible by S ({s})"
            );

            let mut builder = DecompositionBuilder::new(hctx.ctx, 4);
            // Node 0: x' = x / S
            builder.add_node(ComputationNode {
                idx: builder.idx(0),
                operator: Operator::ScalarConstDiv(ScalarConstDiv { divisor: s }),
                inputs: hctx.internal_input_indices.clone(),
                output_dims: input_dims.clone(),
            });
            // Node 1: sq = Mul(x', x) = x²/S
            builder.add_node(ComputationNode {
                idx: builder.idx(1),
                operator: Operator::Mul(Mul { scale }),
                inputs: vec![builder.idx(0), hctx.internal_input_indices[0]],
                output_dims: input_dims,
            });
            // Node 2: sum = Sum(sq) = Σ(x²/S)
            builder.add_node(ComputationNode {
                idx: builder.idx(2),
                operator: Operator::Sum(Sum { axes }),
                inputs: vec![builder.idx(1)],
                output_dims: output_dims.clone(),
            });
            // Node 3: result = sum / (count/S) = Σ(x²)/count
            builder.add_node(ComputationNode {
                idx: builder.idx(3),
                operator: Operator::ScalarConstDiv(ScalarConstDiv {
                    divisor: dividend_value / s,
                }),
                inputs: vec![builder.idx(2)],
                output_dims,
            });
            return builder.finish();
        }

        HandlerBuilder::new(hctx)
            .pipe_with_dims(Operator::Square(Default::default()), input_dims)
            .pipe(Operator::Sum(Sum { axes }))
            .div_by_constant(dividend_value)
            .build()
    }
}
