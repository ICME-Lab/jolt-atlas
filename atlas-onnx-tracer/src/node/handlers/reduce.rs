//! Reduction operator handlers: Sum, MeanOfSquares

use std::collections::HashMap;

use tract_onnx::tract_core::ops::nn::Reduce;

use crate::{
    node::ComputationNode,
    ops::{Constant, Operator, Sum},
    tensor::Tensor,
    utils::parser::{DecompositionBuilder, load_op},
};

use super::{HandlerContext, OpHandlerFn};

pub fn handlers() -> HashMap<&'static str, OpHandlerFn> {
    HashMap::from([
        ("Reduce<Sum>", handle_reduce_sum as OpHandlerFn),
        (
            "Reduce<MeanOfSquares>",
            handle_reduce_mean_of_squares as OpHandlerFn,
        ),
    ])
}

fn handle_reduce_sum(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert_eq!(hctx.internal_input_indices.len(), 1);
    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes = op.axes.into_iter().collect();

    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Sum(Sum { axes }),
        inputs: hctx.internal_input_indices.clone(),
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

fn handle_reduce_mean_of_squares(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    assert!(hctx.internal_input_indices.len() == 1);
    // Decompose MeanOfSquares into
    // square, sum, div, div (and then rebase)
    let op = load_op::<Reduce>(hctx.node.op(), hctx.node.op().name().to_string());
    let axes = op.axes.into_iter().collect();

    // Clone data we need before creating builder (to avoid borrow issues)
    let input_dims = hctx.internal_input_nodes[0].output_dims.clone();
    let output_dims = hctx.output_dims.clone();
    let internal_input_indices = hctx.internal_input_indices.clone();
    let scale = hctx.run_args.scale;

    let dividend_value =
        (input_dims.iter().product::<usize>() / output_dims.iter().product::<usize>()) as i32;

    let mut builder = DecompositionBuilder::new(hctx.ctx, 6);

    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Square(Default::default()),
        inputs: internal_input_indices,
        output_dims: input_dims,
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(1),
        operator: Operator::Sum(Sum { axes }),
        inputs: vec![builder.idx(0)],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(2),
        operator: Operator::Constant(Constant(Tensor::construct(
            vec![dividend_value; output_dims.iter().product::<usize>()],
            output_dims.clone(),
        ))),
        inputs: vec![],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(3),
        operator: Operator::Div(Default::default()),
        inputs: vec![builder.idx(1), builder.idx(2)],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(4),
        operator: Operator::Constant(Constant(Tensor::construct(
            vec![scale; output_dims.iter().product()],
            output_dims.clone(),
        ))),
        inputs: vec![],
        output_dims: output_dims.clone(),
    });

    builder.add_node(ComputationNode {
        idx: builder.idx(5),
        operator: Operator::Shr(Default::default()),
        inputs: vec![builder.idx(3), builder.idx(4)],
        output_dims,
    });

    builder.finish()
}
