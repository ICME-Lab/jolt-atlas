//! Reduction operator handlers: Sum, MeanOfSquares
//!
//! This module provides handlers for reduction operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::tract_core::ops::nn::Reduce;

use crate::{
    node::ComputationNode,
    ops::{MeanOfSquares, Operator, Sum},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
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

/// Reduce<MeanOfSquares>: monolithic fused mean-of-squares .
///
/// Emits a single [`MeanOfSquares`] node that accumulates `Σx²` in i64,
/// floor-divides by `D = N·2^S` (mean rebase), and saturating-clamps — replacing
/// the old Square → Sum → Div(N) decomposition (which squared in i32 and could
/// overflow, the reason `pre_rebase_nonlinear` existed). `count = N` is stored on
/// the operator so the prover/verifier can recover `D` from the node alone.
fn handle_reduce_mean_of_squares(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert!(hctx.internal_input_indices.len() == 1);

    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes: Vec<usize> = op.axes.into_iter().collect();
    let scale = hctx.run_args.scale;

    // N = product of the reduced-axis sizes (the mean denominator).
    let input_dims = &hctx.internal_input_nodes[0].output_dims;
    let count: usize = axes.iter().map(|&ax| input_dims[ax]).product();
    let count = if hctx.run_args.pad_to_power_of_2 {
        count.next_power_of_two()
    } else {
        count
    };
    HandlerBuilder::new(hctx)
        .simple_op(Operator::MeanOfSquares(MeanOfSquares {
            axes,
            scale,
            count,
        }))
        .build()
}
