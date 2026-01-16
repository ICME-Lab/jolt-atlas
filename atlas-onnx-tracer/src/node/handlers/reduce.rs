//! Reduction operator handlers: Sum, MeanOfSquares
//!
//! This module provides handlers for reduction operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::tract_core::ops::nn::Reduce;

use crate::{
    node::ComputationNode,
    ops::{Operator, Sum},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
};

use super::{HandlerContext, OpHandlerFn};

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
fn handle_reduce_mean_of_squares(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert!(hctx.internal_input_indices.len() == 1);

    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes: Vec<usize> = op.axes.into_iter().collect();

    // Calculate the dividend (number of elements being averaged)
    let input_dims = hctx.internal_input_nodes[0].output_dims.clone();
    let output_dims = hctx.output_dims.clone();
    let dividend_value =
        (input_dims.iter().product::<usize>() / output_dims.iter().product::<usize>()) as i32;

    // Pipeline: Square -> Sum -> Div by count -> Div by scale (rebase)
    HandlerBuilder::new(hctx)
        .pipe_with_dims(Operator::Square(Default::default()), input_dims)
        .pipe(Operator::Sum(Sum { axes }))
        .div_by_constant(dividend_value)
        .with_auto_rebase() // Square needs rebase
        .build()
}
