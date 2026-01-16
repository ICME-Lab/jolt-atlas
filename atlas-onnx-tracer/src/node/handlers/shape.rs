//! Shape manipulation operator handlers: Reshape, MoveAxis, MultiBroadcast
//!
//! This module provides handlers for shape manipulation operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::{
    tract_core::ops::array::MultiBroadcastTo,
    tract_hir::internal::{AxisOp, DimLike},
};

use crate::{
    node::ComputationNode,
    ops::{Broadcast, MoveAxis, Operator, Reshape},
    utils::{handler_builder::HandlerBuilder, parser::load_op},
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Reshape", handle_reshape as OpHandlerFn),
        ("RmAxis", handle_reshape as OpHandlerFn),
        ("AddAxis", handle_reshape as OpHandlerFn),
        ("MoveAxis", handle_move_axis as OpHandlerFn),
        ("MultiBroadcastTo", handle_broadcast as OpHandlerFn),
    ])
}

/// Reshape: Changes tensor dimensions without changing data.
fn handle_reshape(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let shape = hctx.output_dims.clone();

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Reshape(Reshape { shape }))
        .build()
}

/// MoveAxis: Moves an axis from source position to destination position.
fn handle_move_axis(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<AxisOp>(hctx.node.op(), hctx.node.op().name().to_string());
    match op {
        AxisOp::Move(from, to) => {
            let source = from.to_usize().unwrap();
            let destination = to.to_usize().unwrap();

            HandlerBuilder::new(hctx)
                .simple_op(Operator::MoveAxis(MoveAxis {
                    source,
                    destination,
                }))
                .build()
        }
        _ => panic!("Expected MoveAxis operator"),
    }
}

/// MultiBroadcastTo: Broadcasts tensor to target shape.
fn handle_broadcast(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<MultiBroadcastTo>(hctx.node.op(), hctx.node.op().name().to_string());
    let shape = op
        .shape
        .iter()
        .map(|x| x.to_usize())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Broadcast(Broadcast { shape }))
        .build()
}
