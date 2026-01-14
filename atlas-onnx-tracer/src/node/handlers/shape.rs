//! Shape manipulation operator handlers: Reshape, MoveAxis, MultiBroadcast

use std::collections::HashMap;

use tract_onnx::{
    tract_core::ops::array::MultiBroadcastTo,
    tract_hir::internal::{AxisOp, DimLike},
};

use crate::{
    node::ComputationNode,
    ops::{Broadcast, MoveAxis, Operator, Reshape},
    utils::parser::{load_op, DecompositionBuilder},
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

fn handle_reshape(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Reshape(Reshape(hctx.output_dims.clone())),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_move_axis(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<AxisOp>(hctx.node.op(), hctx.node.op().name().to_string());
    match op {
        AxisOp::Move(from, to) => {
            let source = from.to_usize().unwrap();
            let destination = to.to_usize().unwrap();
            let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
            builder.add_node(ComputationNode {
                idx: builder.idx(0),
                operator: Operator::MoveAxis(MoveAxis {
                    source,
                    destination,
                }),
                inputs: hctx.internal_input_indices.clone(),
                output_dims: hctx.output_dims.clone(),
            });
            builder.finish()
        }
        _ => panic!("Expected MoveAxis operator"),
    }
}

fn handle_broadcast(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<MultiBroadcastTo>(hctx.node.op(), hctx.node.op().name().to_string());
    let shape = op.shape.clone();
    let shape = shape
        .iter()
        .map(|x| x.to_usize())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Broadcast(Broadcast { shape }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}
