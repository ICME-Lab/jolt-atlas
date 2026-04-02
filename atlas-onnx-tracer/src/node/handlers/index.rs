//! Indexing operator handlers: Gather, Slice
//!
//! This module provides handlers for indexing operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;
use tract_onnx::tract_hir::internal::DimLike;

use crate::{
    node::ComputationNode,
    ops::{GatherLarge, GatherSmall, Operator, Slice},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
};

use super::{HandlerContext, OpHandlerFn};

/// Returns a map of indexing operator names to their handler functions.
pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Gather", handle_gather as OpHandlerFn),
        ("Slice", handle_slice as OpHandlerFn),
    ])
}

/// Gather: Gathers values along an axis.
fn handle_gather(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert_eq!(hctx.internal_input_indices.len(), 2);
    let op = load_op::<tract_onnx::tract_core::ops::array::Gather>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );

    let dict_len = hctx.internal_input_nodes[0].output_dims[op.axis];

    let operator = if dict_len.next_power_of_two() <= 65536 {
        Operator::GatherSmall(GatherSmall {
            axis: op.axis,
            dict_len,
        })
    } else {
        Operator::GatherLarge(GatherLarge {
            axis: op.axis,
            dict_len,
        })
    };

    HandlerBuilder::new(hctx).simple_op(operator).build()
}

/// Slice: Single-axis slice using axis, start and end parameters.
fn handle_slice(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::array::Slice>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let start = op
        .start
        .eval(hctx.symbol_values)
        .to_usize()
        .expect("Slice start must resolve to usize");
    let end = op
        .end
        .eval(hctx.symbol_values)
        .to_usize()
        .expect("Slice end must resolve to usize");

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Slice(Slice {
            axis: op.axis,
            start,
            end,
        }))
        .build()
}
