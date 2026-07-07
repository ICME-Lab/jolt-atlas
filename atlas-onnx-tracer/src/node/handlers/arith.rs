//! Arithmetic operator handlers: Add, Sub, Mul, Div, Pow
//!
//! This module provides handlers for basic arithmetic operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Cube, Mul, Operator, Square},
    simple_handler,
    utils::handler_builder::HandlerBuilder,
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

// Mul: Element-wise multiplication. The fused i64 kernel accumulates, floor-
// rescales by `1 << scale`, and saturating-clamps in one node, so no
// separate ScalarConstDiv rebase node is emitted.
#[inline]
fn build_mul(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Mul(Mul { scale }))
        .build()
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
    // truncate internal_input_indices to 1
    hctx.internal_input_indices.truncate(1);
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
/// The fused i64 [`Square`] kernel accumulates, floor-rescales by `1 << scale`,
/// and saturating-clamps in one node — no separate ScalarConstDiv rebase.
fn build_square(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Square(Square { scale }))
        .build()
}

/// Builds Cube(x) = x³/S² to maintain scale S.
///
/// The fused i64 [`Cube`] kernel accumulates, floor-rescales by `1 << (scale*2)`,
/// and saturating-clamps in one node — no separate ScalarConstDiv rebase.
fn build_cube(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Cube(Cube { scale }))
        .build()
}
