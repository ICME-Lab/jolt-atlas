//! Arithmetic operator handlers: Add, Sub, Mul, Div, Pow
//!
//! This module provides handlers for basic arithmetic operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Operator},
    simple_handler,
    utils::{handler_builder::HandlerBuilder, parser::DecompositionBuilder},
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Add", handle_add as OpHandlerFn),
        ("Sub", handle_sub as OpHandlerFn),
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

// Mul: Element-wise multiplication, needs rebase (div by 1 << scale).
simple_handler!(handle_mul, Operator::Mul(Default::default()), rebase);

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
/// When `pre_rebase_nonlinear` is **true** (for large models like GPT-2):
///   Decomposes into existing ops to avoid i32 overflow:
///     x' = x / S          (ScalarConstDiv)
///     result = x' * x     (Mul, no rebase)
///   Since x' * x = x²/S, the result is already at scale S.
fn build_square(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    if !hctx.run_args.pre_rebase_nonlinear {
        // Default path: Square op + auto rebase
        return HandlerBuilder::new(hctx)
            .with_broadcast()
            .simple_op(Operator::Square(Default::default()))
            .with_auto_rebase()
            .build();
    }

    // Pre-rebase path: decompose into ScalarConstDiv + Mul
    let scale = hctx.run_args.scale;
    let dims = hctx.output_dims.clone();
    let x_idx = hctx.internal_input_indices[0];

    let mut builder = DecompositionBuilder::new(hctx.ctx, 2);

    // Node 0: x' = x / S
    let x_div_idx = builder.idx(0);
    builder.add_node(ComputationNode {
        idx: x_div_idx,
        operator: Operator::ScalarConstDiv(crate::ops::ScalarConstDiv {
            divisor: 1 << scale,
        }),
        inputs: vec![x_idx],
        output_dims: dims.clone(),
    });

    // Node 1: result = x' * x = x²/S (already at scale S, no rebase needed)
    builder.add_node(ComputationNode {
        idx: builder.idx(1),
        operator: Operator::Mul(Default::default()),
        inputs: vec![x_div_idx, x_idx],
        output_dims: dims,
    });

    builder.finish()
}

/// Builds Cube(x) = x³/S² to maintain scale S.
///
/// When `pre_rebase_nonlinear` is **false** (default):
///   Cube(x) → ScalarConstDiv(x³, S²)
///
/// When `pre_rebase_nonlinear` is **true** (for large models like GPT-2):
///   Decomposes into existing ops to avoid i32 overflow:
///     x' = x / S                (ScalarConstDiv)
///     sq = x' * x = x²/S       (Mul)
///     result = sq * x' = x³/S² (Mul)
///   Result is already at scale S, no rebase needed.
fn build_cube(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    if !hctx.run_args.pre_rebase_nonlinear {
        // Default path: Cube op + auto rebase
        return HandlerBuilder::new(hctx)
            .with_broadcast()
            .simple_op(Operator::Cube(Default::default()))
            .with_auto_rebase()
            .build();
    }

    // Pre-rebase path: decompose into ScalarConstDiv + Mul + Mul
    let scale = hctx.run_args.scale;
    let dims = hctx.output_dims.clone();
    let x_idx = hctx.internal_input_indices[0];

    let mut builder = DecompositionBuilder::new(hctx.ctx, 3);

    // Node 0: x' = x / S
    let x_div_idx = builder.idx(0);
    builder.add_node(ComputationNode {
        idx: x_div_idx,
        operator: Operator::ScalarConstDiv(crate::ops::ScalarConstDiv {
            divisor: 1 << scale,
        }),
        inputs: vec![x_idx],
        output_dims: dims.clone(),
    });

    // Node 1: sq = x' * x = x²/S
    let sq_idx = builder.idx(1);
    builder.add_node(ComputationNode {
        idx: sq_idx,
        operator: Operator::Mul(Default::default()),
        inputs: vec![x_div_idx, x_idx],
        output_dims: dims.clone(),
    });

    // Node 2: result = sq * x' = (x²/S) * (x/S) = x³/S²  (already at scale S)
    builder.add_node(ComputationNode {
        idx: builder.idx(2),
        operator: Operator::Mul(Default::default()),
        inputs: vec![sq_idx, x_div_idx],
        output_dims: dims,
    });

    builder.finish()
}
