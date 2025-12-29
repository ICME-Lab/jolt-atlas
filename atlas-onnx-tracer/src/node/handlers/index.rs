//! Indexing operator handlers: Gather

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Gather, Operator},
    utils::parser::{DecompositionBuilder, load_op},
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([("Gather", handle_gather as OpHandlerFn)])
}

fn handle_gather(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert_eq!(hctx.internal_input_indices.len(), 2);
    let op = load_op::<tract_onnx::tract_core::ops::array::Gather>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Gather(Gather { dim: op.axis }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}
