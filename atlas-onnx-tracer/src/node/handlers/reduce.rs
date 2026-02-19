//! Reduction operator handlers: Sum, MeanOfSquares
//!
//! This module provides handlers for reduction operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::tract_core::ops::nn::Reduce;

use crate::{
    node::ComputationNode,
    ops::{Operator, Sum},
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

/// Reduce<MeanOfSquares>: Decomposed into Square -> Sum -> Div(count) -> Div(scale).
///
/// Pipeline: input -> Square(input_dims) -> Sum(output_dims) -> Div by count -> Div by scale
///
/// When `pre_rebase_nonlinear` is **true**, decomposes the Square into
/// ScalarConstDiv + Mul to avoid i32 overflow in large models:
///   x' = x / S → Mul(x', x) = x²/S → Sum → Div by (count/S)
/// This produces the same result: Σ(x²)/count.
fn handle_reduce_mean_of_squares(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert!(hctx.internal_input_indices.len() == 1);

    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes: Vec<usize> = op.axes.into_iter().collect();

    // Calculate the dividend (number of elements being averaged)
    let input_dims = hctx.internal_input_nodes[0].output_dims.clone();
    let output_dims = hctx.output_dims.clone();
    let dividend_value =
        (input_dims.iter().product::<usize>() / output_dims.iter().product::<usize>()) as i32;

    let scale = hctx.run_args.scale;

    if !hctx.run_args.pre_rebase_nonlinear {
        // Default path: Square -> Sum -> Div by count
        HandlerBuilder::new(hctx)
            .pipe_with_dims(Operator::Square(Default::default()), input_dims)
            .pipe(Operator::Sum(Sum { axes }))
            .div_by_constant(dividend_value)
            .build()
    } else {
        // Pre-rebase path: decompose Square into ScalarConstDiv + Mul to avoid overflow.
        // Mul(x/S, x) = x²/S, so Sum gives Σ(x²/S). Dividing by count/S yields Σ(x²)/count,
        // matching the non-decomposed output exactly.
        let s = 1_i32 << scale;
        assert!(
            dividend_value % s == 0,
            "MeanOfSquares pre_rebase: dividend {dividend_value} must be divisible by scale multiplier {s}"
        );
        let adjusted_divisor = dividend_value / s;
        let x_idx = hctx.internal_input_indices[0];

        let mut builder = DecompositionBuilder::new(hctx.ctx, 4);

        // Node 0: x' = x / S
        let x_div_idx = builder.idx(0);
        builder.add_node(ComputationNode {
            idx: x_div_idx,
            operator: Operator::ScalarConstDiv(crate::ops::ScalarConstDiv { divisor: s }),
            inputs: vec![x_idx],
            output_dims: input_dims.clone(),
        });

        // Node 1: sq = x' * x = x²/S
        let sq_idx = builder.idx(1);
        builder.add_node(ComputationNode {
            idx: sq_idx,
            operator: Operator::Mul(Default::default()),
            inputs: vec![x_div_idx, x_idx],
            output_dims: input_dims,
        });

        // Node 2: sum = Σ(x²/S)
        let sum_idx = builder.idx(2);
        builder.add_node(ComputationNode {
            idx: sum_idx,
            operator: Operator::Sum(Sum { axes }),
            inputs: vec![sq_idx],
            output_dims: output_dims.clone(),
        });

        // Node 3: result = Σ(x²/S) / (count/S) = Σ(x²)/count
        builder.add_node(ComputationNode {
            idx: builder.idx(3),
            operator: Operator::ScalarConstDiv(crate::ops::ScalarConstDiv {
                divisor: adjusted_divisor,
            }),
            inputs: vec![sum_idx],
            output_dims,
        });

        builder.finish()
    }
}
