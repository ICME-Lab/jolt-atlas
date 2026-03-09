//! Arithmetic operator handlers: Add, Sub, Mul, Div, Pow
//!
//! This module provides handlers for basic arithmetic operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Cube, Mul, Operator, ScalarConstDiv},
    simple_handler,
    utils::{handler_builder::HandlerBuilder, parser::DecompositionBuilder},
};

use super::{HandlerContext, OpHandlerFn};

/// Returns a map of arithmetic operator names to their handler functions.
pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Add", handle_add as OpHandlerFn),
        ("Sub", handle_sub as OpHandlerFn),
        ("Neg", handle_neg as OpHandlerFn),
        ("Mul", handle_mul as OpHandlerFn),
        ("Pow", handle_pow as OpHandlerFn),
        ("Square", handle_square as OpHandlerFn),
        ("And", handle_and as OpHandlerFn),
    ])
}

// Add: Simple element-wise addition, no rebase needed.
simple_handler!(handle_add, Operator::Add(Default::default()));

// Sub: Simple element-wise subtraction, no rebase needed.
simple_handler!(handle_sub, Operator::Sub(Default::default()));

// Neg: Simple element-wise negation, no rebase needed.
simple_handler!(handle_neg, Operator::Neg(Default::default()));

// // Mul: Element-wise multiplication, needs rebase (div by 1 << scale).
// simple_handler!(handle_mul, Operator::Mul(Default::default()), rebase);

#[inline]
fn build_mul(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    let builder = HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Mul(Mul { scale }));

    #[cfg(not(feature = "fused-ops"))]
    let builder = builder.with_auto_rebase();

    builder.build()
}

fn handle_mul(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    build_mul(hctx)
}

/// Pow: Power operation, dispatches to Square or Cube based on exponent.
fn handle_pow(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    let exponent = match &hctx.internal_input_nodes[1].operator {
        Operator::Constant(Constant(tensor)) => tensor.data()[0] >> scale,
        _ => panic!("Expected constant exponent for Pow operator"),
    };

    match exponent {
        2 => build_square(hctx),
        3 => build_cube(hctx),
        _ => unimplemented!(
            "Power operator with exponent {} is not implemented",
            exponent
        ),
    }
}

// And: Logical AND operation, no rebase needed.
simple_handler!(handle_and, Operator::And(Default::default()));

// Square handler (also used by Pow with exponent=2).
fn handle_square(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    build_square(hctx)
}

/// Builds Square(x) = x²/S to maintain scale S.
///
/// When `pre_rebase_nonlinear` is **false** (default):
///   Square(x) → ScalarConstDiv(x², S)
///
/// HACK: When `pre_rebase_nonlinear` is **true** (for large models like GPT-2):
///   Decomposes into existing ops to avoid i32 overflow:
///     x' = x / S          (ScalarConstDiv)
///     result = x' * x     (Mul, no rebase)
///   Since x' * x = x²/S, the result is already at scale S.
///   TODO: Remove pre_rebase_nonlinear path once fused i64 ops are default.
fn build_square(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // HACK: pre_rebase_nonlinear decomposes to avoid i32 overflow.
    // TODO: Remove once fused i64-precision ops are the default path.
    if hctx.run_args.pre_rebase_nonlinear {
        let scale = hctx.run_args.scale;
        let s = 1i32 << scale;
        let output_dims = hctx.output_dims.clone();
        let x_idx = hctx.internal_input_indices[0];

        let mut builder = DecompositionBuilder::new(hctx.ctx, 2);
        // Node 0: x' = x / S
        builder.add_node(ComputationNode {
            idx: builder.idx(0),
            operator: Operator::ScalarConstDiv(ScalarConstDiv { divisor: s }),
            inputs: vec![x_idx],
            output_dims: output_dims.clone(),
        });
        // Node 1: result = Mul(x', x) = x²/S — already at scale S, no rebase
        builder.add_node(ComputationNode {
            idx: builder.idx(1),
            operator: Operator::Mul(Mul { scale }),
            inputs: vec![builder.idx(0), x_idx],
            output_dims,
        });
        return builder.finish();
    }

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Square(Default::default()))
        .with_auto_rebase()
        .build()
}

/// Builds Cube(x) = x³/S² to maintain scale S.
///
/// HACK: When `pre_rebase_nonlinear` is true, decomposes into:
///   x' = x / S → sq = Mul(x', x) → result = Mul(sq, x')
/// to avoid i32 overflow in large models.
/// TODO: Remove pre_rebase_nonlinear path once fused i64 ops are default.
fn build_cube(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // HACK: pre_rebase_nonlinear decomposes to avoid i32 overflow.
    // TODO: Remove once fused i64-precision ops are the default path.
    #[cfg(not(feature = "fused-ops"))]
    if hctx.run_args.pre_rebase_nonlinear {
        let scale = hctx.run_args.scale;
        let s = 1i32 << scale;
        let output_dims = hctx.output_dims.clone();
        let x_idx = hctx.internal_input_indices[0];

        let mut builder = DecompositionBuilder::new(hctx.ctx, 3);
        // Node 0: x' = x / S
        builder.add_node(ComputationNode {
            idx: builder.idx(0),
            operator: Operator::ScalarConstDiv(ScalarConstDiv { divisor: s }),
            inputs: vec![x_idx],
            output_dims: output_dims.clone(),
        });
        // Node 1: sq = Mul(x', x) = x²/S
        builder.add_node(ComputationNode {
            idx: builder.idx(1),
            operator: Operator::Mul(Mul { scale }),
            inputs: vec![builder.idx(0), x_idx],
            output_dims: output_dims.clone(),
        });
        // Node 2: result = Mul(sq, x') = x³/S²
        builder.add_node(ComputationNode {
            idx: builder.idx(2),
            operator: Operator::Mul(Mul { scale }),
            inputs: vec![builder.idx(1), builder.idx(0)],
            output_dims,
        });
        return builder.finish();
    }

    let scale = hctx.run_args.scale;
    let builder = HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Cube(Cube { scale }));

    #[cfg(not(feature = "fused-ops"))]
    let builder = builder.with_auto_rebase();

    builder.build()
}
