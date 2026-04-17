//! Activation operator handlers: ReLU (Max), Tanh, Softmax, Rsqrt, Erf
//!
//! This module provides handlers for activation functions, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Cos, Erf, Operator, Rsqrt, Sigmoid, Sin, SoftmaxLastAxis, Tanh},
    utils::{handler_builder::HandlerBuilder, parser::load_op, quantize::scale_to_multiplier},
};

use super::{HandlerContext, OpHandlerFn};

#[cfg(feature = "fused-ops")]
const NEURAL_TELEPORT_TAU: i32 = 1;

// TODO: These values should be finetuned based on input ranges and desired output precision.
#[cfg(not(feature = "fused-ops"))]
const NEURAL_TELEPORT_TAU: i32 = 2;

/// Log2 of the lookup table size used for activation functions.
/// With tau=2 and scale=256 (log_scale=8), this must cover quotients produced
/// by GELU's cubic polynomial.  16 gives a quotient range of ±32 768, i.e.
/// raw fixed-point inputs up to ±65 534 (≈ ±256 in real value).
pub const NEURAL_TELEPORT_LOG_TABLE_SIZE: usize = 16;

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

/// Cos: Cosine activation.
fn handle_cos(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = scale_to_multiplier(hctx.run_args.scale).into();

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Cos(Cos { scale }))
        .build()
}

/// Sin: Sine activation.
fn handle_sin(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = scale_to_multiplier(hctx.run_args.scale).into();

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Sin(Sin { scale }))
        .build()
}

/// Erf: Error function activation.
fn handle_erf(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = scale_to_multiplier(hctx.run_args.scale).into();
    let tau = NEURAL_TELEPORT_TAU;
    let log_table_size = NEURAL_TELEPORT_LOG_TABLE_SIZE;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Erf(Erf {
            scale,
            tau,
            log_table: log_table_size,
        }))
        .build()
}

/// Sigmoid activation.
fn handle_sigmoid(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = scale_to_multiplier(hctx.run_args.scale).into();
    let tau = NEURAL_TELEPORT_TAU;
    let log_table_size = NEURAL_TELEPORT_LOG_TABLE_SIZE;

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Sigmoid(Sigmoid {
            scale,
            tau,
            log_table: log_table_size,
        }))
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

    let scale = scale_to_multiplier(hctx.run_args.scale);

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::SoftmaxLastAxis(SoftmaxLastAxis {
            scale: scale as i32,
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
