//! Other operator handlers: Const, Source, Einsum, Iff, IsNan, Cast
//!
//! This module provides handlers for miscellaneous operations, using the
//! `HandlerBuilder` for clean, declarative decomposition patterns.

use std::collections::HashMap;

use tract_onnx::{prelude::DatumType, tract_hir::ops::konst::Const};

use crate::{
    node::ComputationNode,
    ops::{Constant, Einsum, IsNan, Operator},
    tensor::TensorError,
    utils::{
        handler_builder::HandlerBuilder,
        parser::{DecompositionBuilder, extract_tensor_value, load_op},
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
        ("IsNan", handle_is_nan as OpHandlerFn),
        ("Cast", handle_cast as OpHandlerFn),
    ])
}

/// Const: Constant tensor value (quantized for fixed-point).
///
/// Boolean constants (e.g., attention masks) are converted directly to 0/1 integers
/// without quantization scaling, since they represent logical values, not numeric ones.
/// All other constants are quantized to fixed-point representation.
fn handle_const(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<Const>(hctx.node.op(), hctx.node.op().name().to_string());
    let datum_type = op.val().datum_type();
    let raw_tensor = extract_tensor_value(op.val().clone()).unwrap();

    let result_tensor = if datum_type == DatumType::Bool
        || datum_type == DatumType::I64
        || datum_type == DatumType::I32
        || datum_type == DatumType::I16
        || datum_type == DatumType::I8
        || datum_type == DatumType::U8
        || datum_type == DatumType::U16
        || datum_type == DatumType::U32
        || datum_type == DatumType::U64
    {
        // Boolean tensors (e.g., causal attention masks) must remain as 0/1 integers.
        // Integer tensors (e.g., gather indices, shapes) must remain as raw values.
        // Applying quantization scaling would turn 1s into 1<<scale (e.g., 128),
        // breaking downstream boolean checks in operations like Iff, or causing
        // out-of-bounds errors when used as Gather indices.
        raw_tensor
            .par_enum_map::<_, i32, TensorError>(|_, x| Ok(x as i32))
            .unwrap()
    } else {
        quantize_tensor(raw_tensor, hctx.run_args.scale)
    };

    // Constants have no inputs, use builder directly for this special case
    let mut builder = DecompositionBuilder::new(hctx.ctx, 1);
    builder.add_node(ComputationNode {
        idx: builder.idx(0),
        operator: Operator::Constant(Constant(result_tensor)),
        inputs: vec![],
        output_dims: hctx.output_dims.clone(),
    });
    builder.finish()
}

/// Source: Input tensor placeholder.
fn handle_source(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Input(Default::default()))
        .build()
}

/// IsNan: Check for NaN values in tensor.
fn handle_is_nan(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let out_dims = hctx.output_dims.clone();

    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::IsNan(IsNan { out_dims }))
        .build()
}

/// EinSum: Einstein summation with automatic rebase.
fn handle_einsum(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    let op = load_op::<tract_onnx::tract_core::ops::einsum::EinSum>(
        hctx.node.op(),
        hctx.node.op().name().to_string(),
    );
    let tract_string = op.axes.to_string();

    HandlerBuilder::new(hctx)
        .simple_op(Operator::Einsum(Einsum {
            equation: tract_string,
        }))
        .with_auto_rebase()
        .build()
}

/// Iff: Conditional selection (if-then-else).
fn handle_iff(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
    HandlerBuilder::new(hctx)
        .with_broadcast()
        .simple_op(Operator::Iff(Default::default()))
        .build()
}

/// Cast: Type casting with scale adjustment for integer types.
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
            // For integer types, divide by the quantization multiplier (2^scale)
            // to convert from fixed-point back to integer domain.
            let scale = hctx.run_args.scale;
            HandlerBuilder::new(hctx)
                .simple_op(Operator::Identity(Default::default()))
                .div_by_constant(1 << scale)
                .build()
        }
        DatumType::F16 | DatumType::F32 | DatumType::F64 => {
            // For float types, just pass through
            HandlerBuilder::new(hctx)
                .simple_op(Operator::Identity(Default::default()))
                .build()
        }
        _ => panic!("unsupported data type"),
    }
}
