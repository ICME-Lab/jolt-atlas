//! Arithmetic operator handlers: Add, Sub, Mul, Div, Pow

use std::collections::HashMap;

use crate::{
    node::ComputationNode,
    ops::{Constant, Operator},
    tensor::Tensor,
    utils::parser::{DecompositionBuilder, GraphParser},
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

fn handle_add(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Add(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_sub(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }
    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Sub(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_mul(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let output_dims = hctx.output_dims.clone();
    let scale = hctx.run_args.scale;

    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 3 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Mul(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes + 1),
        operator: Operator::Constant(Constant(Tensor::construct(
            vec![1 << scale; output_dims.iter().product()],
            output_dims.clone(),
        ))),
        inputs: vec![],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes + 2),
        operator: Operator::Div(Default::default()),
        inputs: vec![builder.idx(bc_nodes), builder.idx(bc_nodes + 1)],
        output_dims,
    });

    builder.finish()
}

fn handle_pow(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let scale = hctx.run_args.scale;
    let exponent = match &hctx.internal_input_nodes[1].operator {
        Operator::Constant(Constant(tensor)) => tensor.data()[0] >> scale, // Shift by scale to get original exponent
        _ => panic!("Expected constant exponent for Pow operator"),
    };

    let output_dims = hctx.output_dims.clone();

    match exponent {
        2 => {
            // Special case for square operation
            let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
            let bc_nodes = broadcast_nodes.len();

            let mut builder = DecompositionBuilder::new(hctx.ctx, 3 + bc_nodes);
            for node in broadcast_nodes {
                builder.add_node(node);
            }

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes),
                operator: Operator::Square(Default::default()),
                inputs: hctx.internal_input_indices.clone(),
                output_dims: output_dims.clone(),
            });

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes + 1),
                operator: Operator::Constant(Constant(Tensor::construct(
                    vec![1 << scale; output_dims.iter().product()],
                    output_dims.clone(),
                ))),
                inputs: vec![],
                output_dims: output_dims.clone(),
            });

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes + 2),
                operator: Operator::Div(Default::default()),
                inputs: vec![builder.idx(bc_nodes), builder.idx(bc_nodes + 1)],
                output_dims,
            });

            builder.finish()
        }
        3 => {
            // Special case for cube operation
            let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
            let bc_nodes = broadcast_nodes.len();

            let mut builder = DecompositionBuilder::new(hctx.ctx, 3 + bc_nodes);
            for node in broadcast_nodes {
                builder.add_node(node);
            }

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes),
                operator: Operator::Cube(Default::default()),
                inputs: hctx.internal_input_indices.clone(),
                output_dims: output_dims.clone(),
            });

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes + 1),
                operator: Operator::Constant(Constant(Tensor::construct(
                    vec![1 << (scale * 2); output_dims.iter().product()],
                    output_dims.clone(),
                ))),
                inputs: vec![],
                output_dims: output_dims.clone(),
            });

            builder.add_node(ComputationNode {
                idx: builder.idx(bc_nodes + 2),
                operator: Operator::Div(Default::default()),
                inputs: vec![builder.idx(bc_nodes), builder.idx(bc_nodes + 1)],
                output_dims,
            });

            builder.finish()
        }
        _ => unimplemented!(
            "Power operator with exponent {} is not implemented",
            exponent
        ),
    }
}

fn handle_and(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::And(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_square(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    // Square operation: x^2
    // Decompose into: Square -> Div by (1 << scale) to maintain fixed-point scale
    let scale = hctx.run_args.scale;
    let output_dims = hctx.output_dims.clone();

    let broadcast_nodes = GraphParser::insert_broadcast_nodes(hctx);
    let bc_nodes = broadcast_nodes.len();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 3 + bc_nodes);
    for node in broadcast_nodes {
        builder.add_node(node);
    }

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes),
        operator: Operator::Square(Default::default()),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes + 1),
        operator: Operator::Constant(Constant(Tensor::construct(
            vec![1 << scale; output_dims.iter().product()],
            output_dims.clone(),
        ))),
        inputs: vec![],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(bc_nodes + 2),
        operator: Operator::Div(Default::default()),
        inputs: vec![builder.idx(bc_nodes), builder.idx(bc_nodes + 1)],
        output_dims,
    });

    builder.finish()
}
