//! Other operator handlers: Const, Source, Einsum, Iff

use std::collections::HashMap;

use tract_onnx::{prelude::DatumType, tract_hir::ops::konst::Const};

use crate::{
    node::ComputationNode,
    ops::{Constant, Einsum, IsNan, Operator},
    tensor::Tensor,
    utils::{
        parser::{DecompositionBuilder, GraphParser, extract_tensor_value, load_op},
        quantize::quantize_tensor,
    },
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Const", handle_const as OpHandlerFn),
        ("Source", handle_source as OpHandlerFn),
        ("EinSum", handle_einsum as OpHandlerFn),
        ("Iff", handle_iff as OpHandlerFn),
        ("onnx.IsNan", handle_is_nan as OpHandlerFn),
        ("Cast", handle_cast as OpHandlerFn),
    ])
}

fn handle_const(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<Const>(hctx.node.op(), hctx.node.op().name().to_string());
    let raw_tensor = extract_tensor_value(op.val().clone()).unwrap();
    let quantized_tensor = quantize_tensor(raw_tensor, hctx.run_args.scale);

    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Constant(Constant(quantized_tensor)),
        inputs: vec![],
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_source(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // TODO: might need to maintain tract datum type info
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Input(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_is_nan(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::IsNan(IsNan {
            out_dims: hctx.output_dims.clone(),
        }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_einsum(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::einsum::EinSum>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let axes = &op.axes;
    let output_dims = hctx.output_dims.clone();
    let scale = hctx.run_args.scale;

    let mut builder = DecompositionBuilder::new(hctx.ctx, 3);

    // Map tract's einsum string to our simplified internal representation
    let tract_string = axes.to_string();
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Einsum(Einsum {
            equation: tract_string,
        }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(1),
        operator: Operator::Constant(Constant(Tensor::construct(
            vec![1 << scale; output_dims.iter().product()],
            output_dims.clone(),
        ))),
        inputs: vec![],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(2),
        operator: Operator::Div(Default::default()),
        inputs: vec![builder.idx(0), builder.idx(1)],
        output_dims,
    });

    builder.finish()
}

fn handle_iff(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Iff(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_cast(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::cast::Cast>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let dt = op.to;

    let input_node = hctx
        .ctx
        .nodes
        .get(&hctx.internal_input_indices[0])
        .expect("Input node not found");
    assert_eq!(input_node.output_dims, hctx.output_dims);

    match dt {
        DatumType::Bool
        | DatumType::TDim
        | DatumType::I64
        | DatumType::I32
        | DatumType::I16
        | DatumType::I8
        | DatumType::U8
        | DatumType::U16
        | DatumType::U32
        | DatumType::U64 => {
            let mut builder = DecompositionBuilder::new(hctx.ctx, 2);
            let output_dims = hctx.output_dims.clone();
            let scale = hctx.run_args.scale;
            builder.add_node(ComputationNode {
                idx: builder.idx(0),
                operator: Operator::Constant(Constant(Tensor::construct(
                    vec![scale; output_dims.iter().product()],
                    output_dims.clone(),
                ))),
                inputs: vec![],
                output_dims: output_dims.clone(),
            });
            builder.add_node(ComputationNode {
                idx: builder.idx(1),
                operator: Operator::Div(Default::default()),
                inputs: vec![hctx.internal_input_indices[0], builder.idx(0)],
                output_dims: hctx.output_dims.clone(),
            });
            builder.finish()
        }
        DatumType::F16 | DatumType::F32 | DatumType::F64 => {
            let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
            builder.add_node(ComputationNode {
                idx: builder.idx(0),
                operator: Operator::Identity(Default::default()),
                inputs: hctx.internal_input_indices.clone(),
                output_dims: hctx.output_dims.clone(),
            });
            builder.finish()
        }
        _ => panic!("unsupported data type"),
    }
}
