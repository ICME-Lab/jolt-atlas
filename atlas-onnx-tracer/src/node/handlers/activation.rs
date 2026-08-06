//! Activation operator handlers: ReLU (Max), Tanh, Softmax, Rsqrt, Erf
//!
//! This module provides handlers for activation functions, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Cos, Erf, Operator, Rsqrt, Sigmoid, Sin, SoftmaxLastAxis, Tanh},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
};

use super::{HandlerContext, OpHandlerFn};

/// Returns a map of activation operator names to their handler functions.
pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Cos", handle_cos as OpHandlerFn),
        ("Erf", handle_erf as OpHandlerFn),
        ("Max", handle_max as OpHandlerFn),
        ("Rsqrt", handle_rsqrt as OpHandlerFn),
        ("Sigmoid", handle_sigmoid as OpHandlerFn),
        ("Sin", handle_sin as OpHandlerFn),
        ("Softmax", handle_softmax as OpHandlerFn),
        ("Tanh", handle_tanh as OpHandlerFn),
    ])
}

/// Max: Special-cased to ReLU when comparing with 0.
fn handle_max(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // Extract the max value from constant input
    let max_value = hctx
        .internal_input_nodes
        .iter()
        .find_map(|input_node| {
            if let Operator::Constant(Constant(tensor)) = &input_node.operator {
                Some(tensor.data()[0])
            } else {
                None
            }
        })
        .expect("Max operator must have a constant input");

    // If max is 0, this is a ReLU operation
    if max_value == 0 {
        // Remove the constant input from the internal inputs
        hctx.internal_input_indices.remove(1);

        HandlerBuilder::new(hctx)
            .with_broadcast()
            .simple_op(Operator::ReLU(Default::default()))
            .build()
    } else {
        unimplemented!("Max operator with non-zero constant is not implemented");
    }
}

/// Tanh: Hyperbolic tangent activation.
fn handle_tanh(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Tanh(Tanh { scale }))
        .build()
}

/// Cos: Cosine activation.
fn handle_cos(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Cos(Cos { scale }))
        .build()
}

/// Sin: Sine activation.
fn handle_sin(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Sin(Sin { scale }))
        .build()
}

/// Erf: Error function activation.
fn handle_erf(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Erf(Erf { scale }))
        .build()
}

/// Sigmoid activation.
fn handle_sigmoid(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Sigmoid(Sigmoid { scale }))
        .build()
}

/// Softmax: Apply softmax along the last axis.
///
/// Uses the decomposed `SoftmaxLastAxis` prover which supports dynamic scales.
/// Panics if the ONNX softmax axis is not the last dimension.
fn handle_softmax(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::nn::Softmax>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let axes = op.axes.to_vec();
    assert!(axes.len() == 1, "Softmax must have exactly one axis");

    // Determine the rank of the input tensor so we can verify last-axis.
    let input_rank = hctx.internal_input_nodes[0].output_dims.len();
    let axis = axes[0];
    assert_eq!(
        axis,
        input_rank - 1,
        "SoftmaxLastAxis only supports the last axis (got axis={axis}, rank={input_rank}). \
         Transpose the input so the softmax dimension is last."
    );

    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale }))
        .build()
}

/// Rsqrt: Reciprocal square root.
fn handle_rsqrt(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Rsqrt(Rsqrt { scale }))
        .build()
}
