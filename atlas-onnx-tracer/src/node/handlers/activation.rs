//! Activation operator handlers: ReLU (Max), Tanh, Softmax, Rsqrt, Erf
//!
//! This module provides handlers for activation functions, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Erf, Operator, Rsqrt, SoftmaxAxes, Tanh},
    utils::{handler_builder::HandlerBuilder, parser::load_op, quantize::scale_to_multiplier},
};

use super::{HandlerContext, OpHandlerFn};

// TODO: These values should be finetuned based on input ranges and desired output precision.
const NEURAL_TELEPORT_TAU: i32 = 2;
pub const NEURAL_TELEPORT_LOG_TABLE_SIZE: usize = 10;

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Max", handle_max as OpHandlerFn),
        ("Tanh", handle_tanh as OpHandlerFn),
        ("Softmax", handle_softmax as OpHandlerFn),
        ("Rsqrt", handle_rsqrt as OpHandlerFn),
        ("Erf", handle_erf as OpHandlerFn),
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
    let scale = scale_to_multiplier(hctx.run_args.scale).into();
    let tau = NEURAL_TELEPORT_TAU;
    let log_table_size = NEURAL_TELEPORT_LOG_TABLE_SIZE;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Tanh(Tanh {
            scale,
            tau,
            log_table: log_table_size,
        }))
        .build()
}

/// Erf: Error function activation.
fn handle_erf(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = scale_to_multiplier(hctx.run_args.scale).into();

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Erf(Erf { scale }))
        .build()
}

/// Softmax: Softmax activation along specified axis.
fn handle_softmax(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::nn::Softmax>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let axes = op.axes.to_vec();
    assert!(axes.len() == 1);

    let scale = scale_to_multiplier(hctx.run_args.scale).into();

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::SoftmaxAxes(SoftmaxAxes {
            axes: axes[0],
            scale,
        }))
        .build()
}

/// Rsqrt: Reciprocal square root.
fn handle_rsqrt(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale as f32;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Rsqrt(Rsqrt {
            scale: scale.into(),
        }))
        .build()
}
