//! Arithmetic operator handlers: Add, Sub, Mul, Div, Pow
//!
//! This module provides handlers for basic arithmetic operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Operator},
    simple_handler,
    utils::handler_builder::HandlerBuilder,
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
        2 => {
            // Square: x^2, needs rebase by (1 << scale)
            HandlerBuilder::new(hctx)
                .with_broadcast()
                .simple_op(Operator::Square(Default::default()))
                .with_auto_rebase()
                .build()
        }
        3 => {
            // Cube: x^3, needs rebase by (1 << (scale * 2))
            HandlerBuilder::new(hctx)
                .with_broadcast()
                .simple_op(Operator::Cube(Default::default()))
                .with_auto_rebase()
                .build()
        }
        _ => unimplemented!(
            "Power operator with exponent {} is not implemented",
            exponent
        ),
    }
}

// And: Logical AND operation, no rebase needed.
simple_handler!(handle_and, Operator::And(Default::default()));

// Square: x^2 operation, needs rebase (div by 1 << scale).
simple_handler!(handle_square, Operator::Square(Default::default()), rebase);
