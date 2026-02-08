//! Indexing operator handlers: Gather
//!
//! This module provides handlers for indexing operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Gather, Operator},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([("Gather", handle_gather as OpHandlerFn)])
}

/// Gather: Gathers values along an axis.
fn handle_gather(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert_eq!(hctx.internal_input_indices.len(), 2);
    let op = load_op::<tract_onnx::tract_core::ops::array::Gather>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Gather(Gather { axis: op.axis }))
        .build()
}
