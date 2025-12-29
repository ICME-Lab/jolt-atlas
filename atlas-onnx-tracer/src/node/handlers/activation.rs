//! Activation operator handlers: ReLU (Max), Tanh, Softmax, Rsqrt

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Operator, Rsqrt, Softmax, Tanh},
    utils::{
        parser::{DecompositionBuilder, load_op},
        quantize::scale_to_multiplier,
    },
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Max", handle_max as OpHandlerFn),
        ("Tanh", handle_tanh as OpHandlerFn),
        ("Softmax", handle_softmax as OpHandlerFn),
        ("Rsqrt", handle_rsqrt as OpHandlerFn),
    ])
}

fn handle_max(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // --- Special case for relu ---
    // Extract the max value
    // first find the input that is a constant
    // and then extract the value
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

    if max_value == 0 {
        // If max is 0, this is a ReLU operation
        let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
        builder.add_node(ComputationNode {
            idx: builder.idx(0),
            operator: Operator::ReLU(Default::default()),
            inputs: hctx.internal_input_indices.clone(),
            output_dims: hctx.output_dims.clone(),
        });
        builder.finish()
    } else {
        unimplemented!("Max operator with non-zero constant is not implemented");
    }
}

fn handle_tanh(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Tanh(Tanh {
            scale: scale_to_multiplier(hctx.run_args.scale).into(),
        }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_softmax(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::nn::Softmax>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let axes = op.axes.to_vec();
    assert!(axes.len() == 1);

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Softmax(Softmax {
            axes: axes[0],
            scale: scale_to_multiplier(hctx.run_args.scale).into(),
        }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_rsqrt(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // TODO: implement Rsqrt decomposition
    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Rsqrt(Rsqrt {
            scale: scale_to_multiplier(hctx.run_args.scale).into(),
        }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}
