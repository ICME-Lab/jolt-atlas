//! The rewrite pipeline: a fixed sequence of hand-written, individually
//! testable passes turning the analyzed ONNX graph into the atlas op
//! surface. No pattern DSL — each matcher is plain Rust with explicit
//! guards, because the passes define the statement that gets proven.
//!
//! Passes may introduce `atlas.*` ops (`atlas.Sum`, `atlas.MeanOfSquares`,
//! `atlas.Rsqrt`, `atlas.Einsum`, `atlas.MoveAxis`, `atlas.Broadcast`,
//! `atlas.Descale`); `lower.rs` is the single choke point that verifies the
//! final surface.

use crate::{
    error::ImportError,
    graph::{Attr, ConstData, ConstTensor, DType, Edge, EdgeId, IGraph, INode, TensorInfo},
};
use atlas_trace::Scale;

/// How many rewrites each pass performed (for logging and tests).
#[derive(Debug, Clone)]
pub struct PassReport {
    /// Pass name.
    pub name: &'static str,
    /// Number of rewrites applied.
    pub rewrites: usize,
}

type Pass = fn(&mut IGraph) -> Result<usize, ImportError>;

/// A pass in the fixed pipeline; `Scaled` passes need the fixed-point scale
/// (today only `div_rewrites`, which rejects sub-resolution reciprocals).
enum PassFn {
    Plain(Pass),
    Scaled(fn(&mut IGraph, Scale) -> Result<usize, ImportError>),
}

/// Run the full pipeline in its fixed order.
pub fn run_all(g: &mut IGraph, scale: Scale) -> Result<Vec<PassReport>, ImportError> {
    use PassFn::{Plain, Scaled};
    let mut reports = Vec::new();
    let passes: &[(&'static str, PassFn)] = &[
        ("elide", Plain(elide)),
        ("gather_norm", Plain(gather_norm)),
        ("reshape_family", Plain(reshape_family)),
        ("split_to_slices", Plain(split_to_slices)),
        ("pow_to_square_cube", Plain(pow_to_square_cube)),
        ("norm_fusion", Plain(norm_fusion)),
        ("div_rewrites", Scaled(div_rewrites)),
        ("rsqrt_eps_floor", Scaled(rsqrt_eps_floor)),
        ("matmul_gemm_to_einsum", Plain(matmul_gemm_to_einsum)),
        ("slice_through_projection", Plain(slice_through_projection)),
        ("transpose_to_moveaxis", Plain(transpose_to_moveaxis)),
        ("expand_to_broadcast", Plain(expand_to_broadcast)),
        ("repeat_to_gather", Plain(repeat_to_gather)),
        ("cast_policy", Plain(cast_policy)),
        ("elide_again", Plain(elide)),
    ];
    for (name, pass) in passes {
        let rewrites = match pass {
            Plain(f) => f(g)?,
            Scaled(f) => f(g, scale)?,
        };
        reports.push(PassReport { name, rewrites });
    }
    crate::analyze::dce(g);
    Ok(reports)
}

fn dims_of(g: &IGraph, e: EdgeId) -> Vec<usize> {
    g.dims(e).expect("analyzed edge").to_vec()
}

fn f32_info(dims: &[usize]) -> TensorInfo {
    TensorInfo {
        dtype: DType::F32,
        dims: dims.to_vec(),
    }
}

/// Create a scalar f32 constant edge.
fn f32_scalar_const(g: &mut IGraph, hint: &str, value: f32) -> EdgeId {
    let e = g.fresh_edge(hint, f32_info(&[1]));
    g.edge_mut(e).konst = Some(ConstTensor {
        dtype: DType::F32,
        dims: vec![1],
        data: ConstData::F32(vec![value]),
    });
    e
}

/// Drop Identity nodes, same-dtype Casts, and zero-size Concat operands
/// (KV-cache concats carry an empty `past` tensor when past length is 0).
fn elide(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op == "Concat" {
            let live: Vec<EdgeId> = node
                .inputs
                .iter()
                .copied()
                .filter(|&e| g.dims(e).is_none_or(|d| d.iter().product::<usize>() > 0))
                .collect();
            if live.len() < node.inputs.len() {
                rewrites += 1;
                if live.len() == 1 {
                    let (from, to) = (node.outputs[0], live[0]);
                    g.redirect_uses(from, to);
                    g.remove_node(idx);
                    continue;
                }
                g.nodes[idx].as_mut().expect("live").inputs = live;
            }
        }
        let node = g.nodes[idx].as_ref().expect("live");
        let is_noop = match node.op.as_str() {
            "Identity" => true,
            "Cast" => {
                let src = g.info(node.inputs[0]).map(|i| i.dtype);
                let dst = g.info(node.outputs[0]).map(|i| i.dtype);
                src.is_some() && src == dst
            }
            _ => false,
        };
        if is_noop {
            let node = g.nodes[idx].as_ref().expect("live");
            let (from, to) = (node.outputs[0], node.inputs[0]);
            g.redirect_uses(from, to);
            g.remove_node(idx);
            rewrites += 1;
        }
    }
    Ok(rewrites)
}

/// Squeeze/Unsqueeze/Flatten → Reshape (target shape = the already-inferred
/// output dims).
fn reshape_family(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if !matches!(
            node.op.as_str(),
            "Squeeze" | "Unsqueeze" | "Flatten" | "Reshape"
        ) {
            continue;
        }
        let shape = dims_of(g, node.outputs[0]);
        let node = g.nodes[idx].as_mut().expect("live");
        node.op = "Reshape".into();
        node.inputs.truncate(1); // drop a shape operand if present
        node.attrs.clear();
        node.attrs.insert(
            "shape".into(),
            Attr::Ints(shape.iter().map(|&d| d as i64).collect()),
        );
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Split → one Slice per output.
fn split_to_slices(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Split" {
            continue;
        }
        let input = node.inputs[0];
        let in_dims = dims_of(g, input);
        let axis = crate::shape_fns::norm_axis(node.attr_int("axis").unwrap_or(0), in_dims.len())
            .map_err(|reason| ImportError::Pass {
            pass: "split_to_slices",
            node: node.name.clone(),
            reason,
        })?;
        let outputs = node.outputs.clone();
        let name = node.name.clone();
        let mut start = 0usize;
        for out in outputs {
            let len = dims_of(g, out)[axis];
            g.push_node(INode {
                name: format!("{name}/slice@{start}"),
                op: "atlas.Slice".into(),
                attrs: [
                    ("axis".to_string(), Attr::Int(axis as i64)),
                    ("start".to_string(), Attr::Int(start as i64)),
                    ("end".to_string(), Attr::Int((start + len) as i64)),
                ]
                .into(),
                inputs: vec![input],
                outputs: vec![out],
            });
            start += len;
        }
        g.remove_node(idx);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Pow(x, const 2|3) → atlas.Square / atlas.Cube.
/// Normalize Gather: resolve a negative axis attribute and rebase negative
/// constant indices (ONNX from-the-end indexing) by the axis length — the
/// prover's gather kernels only accept non-negative indices.
fn gather_norm(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Gather" {
            continue;
        }
        let name = node.name.clone();
        let (data, indices) = (node.inputs[0], node.inputs[1]);
        let axis_attr = node.attr_int("axis").unwrap_or(0);
        let rank = dims_of(g, data).len();
        let fail = |reason: String| ImportError::Pass {
            pass: "gather_norm",
            node: name.clone(),
            reason,
        };
        let axis = crate::shape_fns::norm_axis(axis_attr, rank).map_err(&fail)?;
        if axis as i64 != axis_attr {
            let node = g.nodes[idx].as_mut().expect("live");
            node.attrs.insert("axis".into(), Attr::Int(axis as i64));
            rewrites += 1;
        }
        let axis_len = dims_of(g, data)[axis] as i64;
        let Some(k) = g.konst(indices) else {
            continue;
        };
        let Some(vals) = k.as_i64s() else {
            continue; // non-integer indices fail shape/lowering loudly
        };
        if vals.iter().all(|&v| v >= 0) {
            continue;
        }
        let (dtype, dims) = (k.dtype, k.dims.clone());
        let resolved: Vec<i64> = vals
            .iter()
            .map(|&v| {
                let r = if v < 0 { v + axis_len } else { v };
                if (0..axis_len).contains(&r) {
                    Ok(r)
                } else {
                    Err(format!(
                        "Gather index {v} out of range for axis length {axis_len}"
                    ))
                }
            })
            .collect::<Result<_, _>>()
            .map_err(&fail)?;
        // The index constant may feed other nodes; rebase on a fresh edge.
        let re = g.fresh_edge(
            &format!("{name}/indices"),
            TensorInfo {
                dtype,
                dims: dims.clone(),
            },
        );
        g.edge_mut(re).konst = Some(ConstTensor {
            dtype,
            dims,
            data: ConstData::I64(resolved),
        });
        g.nodes[idx].as_mut().expect("live").inputs[1] = re;
        rewrites += 1;
    }
    Ok(rewrites)
}

fn pow_to_square_cube(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Pow" {
            continue;
        }
        let name = node.name.clone();
        let exponent = node.inputs.get(1).and_then(|&e| g.konst(e)).and_then(|k| {
            k.scalar_f32()
                .map(|f| f as f64)
                .or_else(|| k.scalar_i64().map(|i| i as f64))
        });
        let op = match exponent {
            Some(2.0) => "atlas.Square",
            Some(3.0) => "atlas.Cube",
            other => {
                return Err(ImportError::Pass {
                    pass: "pow_to_square_cube",
                    node: name,
                    reason: format!("only integer exponents 2 and 3 are supported, got {other:?}"),
                });
            }
        };
        let node = g.nodes[idx].as_mut().expect("live");
        node.op = op.into();
        node.inputs.truncate(1);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Normalization lowering:
/// (a) fused `LayerNormalization` → Sum·(1/N) mean, Sub, MeanOfSquares,
///     +eps, Rsqrt, ·γ, +β — the same decomposition tract produced;
/// (b) `ReduceMean(Pow(x,2))` / `ReduceMean(Mul(x,x))` → atlas.MeanOfSquares;
/// (c) leftover `ReduceMean` → atlas.Sum + Mul(1/N);
/// (d) `ReduceSum` → atlas.Sum.
fn norm_fusion(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;

    // (a) Fused LayerNormalization.
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "LayerNormalization" {
            continue;
        }
        let name = node.name.clone();
        let x = node.inputs[0];
        let gamma = node.inputs.get(1).copied();
        let beta = node.inputs.get(2).copied();
        let out = node.outputs[0];
        let eps = node.attr_float("epsilon").unwrap_or(1e-5);
        let x_dims = dims_of(g, x);
        let rank = x_dims.len();
        let axis = crate::shape_fns::norm_axis(node.attr_int("axis").unwrap_or(-1), rank).map_err(
            |reason| ImportError::Pass {
                pass: "norm_fusion",
                node: name.clone(),
                reason,
            },
        )?;
        if axis != rank - 1 {
            return Err(ImportError::Pass {
                pass: "norm_fusion",
                node: name,
                reason: format!("LayerNormalization axis {axis} must be the last axis"),
            });
        }
        let n = x_dims[rank - 1];
        if n == 0 {
            // 1/N below would be inf (and quantize to a silent saturation).
            return Err(ImportError::Pass {
                pass: "norm_fusion",
                node: name,
                reason: "LayerNormalization over a zero-size axis".into(),
            });
        }
        let mut red_dims = x_dims.clone();
        red_dims[rank - 1] = 1;

        let sum = g.fresh_edge(&format!("{name}/sum"), f32_info(&red_dims));
        let mean = g.fresh_edge(&format!("{name}/mean"), f32_info(&red_dims));
        let centered = g.fresh_edge(&format!("{name}/centered"), f32_info(&x_dims));
        let mos = g.fresh_edge(&format!("{name}/mos"), f32_info(&red_dims));
        let var_eps = g.fresh_edge(&format!("{name}/var_eps"), f32_info(&red_dims));
        let inv = g.fresh_edge(&format!("{name}/inv"), f32_info(&red_dims));
        let normed = g.fresh_edge(&format!("{name}/normed"), f32_info(&x_dims));
        let recip_n = f32_scalar_const(g, &format!("{name}/recip_n"), 1.0 / n as f32);
        let eps_c = f32_scalar_const(g, &format!("{name}/eps"), eps);

        let axes = Attr::Ints(vec![(rank - 1) as i64]);
        let mut push = |op: &str, attrs: Vec<(&str, Attr)>, inputs: Vec<EdgeId>, output: EdgeId| {
            g.push_node(INode {
                name: format!("{}/{}", name, op),
                op: op.into(),
                attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
                inputs,
                outputs: vec![output],
            });
        };
        push("atlas.Sum", vec![("axes", axes.clone())], vec![x], sum);
        push("Mul", vec![], vec![sum, recip_n], mean);
        push("Sub", vec![], vec![x, mean], centered);
        push(
            "atlas.MeanOfSquares",
            vec![("axes", axes), ("count", Attr::Int(n as i64))],
            vec![centered],
            mos,
        );
        push("Add", vec![], vec![mos, eps_c], var_eps);
        push("atlas.Rsqrt", vec![], vec![var_eps], inv);
        push("Mul", vec![], vec![centered, inv], normed);
        match (gamma, beta) {
            (Some(gm), Some(bt)) => {
                let scaled = g.fresh_edge(&format!("{name}/scaled"), f32_info(&x_dims));
                g.push_node(INode {
                    name: format!("{name}/gamma"),
                    op: "Mul".into(),
                    attrs: Default::default(),
                    inputs: vec![normed, gm],
                    outputs: vec![scaled],
                });
                g.push_node(INode {
                    name: format!("{name}/beta"),
                    op: "Add".into(),
                    attrs: Default::default(),
                    inputs: vec![scaled, bt],
                    outputs: vec![out],
                });
            }
            (Some(gm), None) => {
                g.push_node(INode {
                    name: format!("{name}/gamma"),
                    op: "Mul".into(),
                    attrs: Default::default(),
                    inputs: vec![normed, gm],
                    outputs: vec![out],
                });
            }
            _ => {
                // No affine part: the normed value is the output.
                g.redirect_uses(out, normed);
            }
        }
        g.remove_node(idx);
        rewrites += 1;
    }

    // (b)–(d) Reductions.
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        let op = node.op.clone();
        if op != "ReduceMean" && op != "ReduceSum" {
            continue;
        }
        let name = node.name.clone();
        let keepdims = node.attr_int("keepdims").unwrap_or(1) != 0;
        let x = node.inputs[0];
        let x_dims = dims_of(g, x);
        let axes: Vec<usize> = match node.attr_ints("axes") {
            Some(a) => a
                .iter()
                .map(|&v| crate::shape_fns::norm_axis(v, x_dims.len()))
                .collect::<Result<_, _>>()
                .map_err(|reason| ImportError::Pass {
                    pass: "norm_fusion",
                    node: name.clone(),
                    reason,
                })?,
            None => (0..x_dims.len()).collect(),
        };
        let out = node.outputs[0];
        let n: usize = axes.iter().map(|&a| x_dims[a]).product();
        if n == 0 {
            // 1/N for the mean would be inf (and quantize to a silent
            // saturation); a zero-size reduction has no lowerable form.
            return Err(ImportError::Pass {
                pass: "norm_fusion",
                node: name,
                reason: format!("{op} over zero elements"),
            });
        }
        let axes_attr = Attr::Ints(axes.iter().map(|&a| a as i64).collect());

        // Output edge of the kept-dims reduction (atlas.Sum keeps dims=1).
        let mut kept_dims = x_dims.clone();
        for &a in &axes {
            kept_dims[a] = 1;
        }

        // Where the kept-dims result goes: directly to `out`, or through a
        // Reshape when keepdims=0.
        let make_target = |g: &mut IGraph, hint: &str| -> (EdgeId, Option<EdgeId>) {
            if keepdims {
                (out, None)
            } else {
                let kept = g.fresh_edge(hint, f32_info(&kept_dims));
                (kept, Some(kept))
            }
        };

        // (b) mean of squares?
        let mos_input = if op == "ReduceMean" {
            g.producer(x).and_then(|(_, p)| match p.op.as_str() {
                "atlas.Square" => Some(p.inputs[0]),
                "Mul" if p.inputs[0] == p.inputs[1] => Some(p.inputs[0]),
                _ => None,
            })
        } else {
            None
        };

        if let Some(sq_in) = mos_input {
            let (target, kept) = make_target(g, &format!("{name}/mos_kept"));
            g.push_node(INode {
                name: format!("{name}/mos"),
                op: "atlas.MeanOfSquares".into(),
                attrs: [
                    ("axes".to_string(), axes_attr.clone()),
                    ("count".to_string(), Attr::Int(n as i64)),
                ]
                .into(),
                inputs: vec![sq_in],
                outputs: vec![target],
            });
            if let Some(kept) = kept {
                add_reshape_to(g, &name, kept, out);
            }
        } else if op == "ReduceSum" {
            let (target, kept) = make_target(g, &format!("{name}/sum_kept"));
            g.push_node(INode {
                name: format!("{name}/sum"),
                op: "atlas.Sum".into(),
                attrs: [("axes".to_string(), axes_attr)].into(),
                inputs: vec![x],
                outputs: vec![target],
            });
            if let Some(kept) = kept {
                add_reshape_to(g, &name, kept, out);
            }
        } else {
            // (c) plain mean: Sum then multiply by 1/N.
            let sum = g.fresh_edge(&format!("{name}/sum"), f32_info(&kept_dims));
            let recip = f32_scalar_const(g, &format!("{name}/recip_n"), 1.0 / n as f32);
            g.push_node(INode {
                name: format!("{name}/sum"),
                op: "atlas.Sum".into(),
                attrs: [("axes".to_string(), axes_attr)].into(),
                inputs: vec![x],
                outputs: vec![sum],
            });
            let (target, kept) = make_target(g, &format!("{name}/mean_kept"));
            g.push_node(INode {
                name: format!("{name}/mean"),
                op: "Mul".into(),
                attrs: Default::default(),
                inputs: vec![sum, recip],
                outputs: vec![target],
            });
            if let Some(kept) = kept {
                add_reshape_to(g, &name, kept, out);
            }
        }
        g.remove_node(idx);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Reshape `from` into the dims of the existing edge `out`.
fn add_reshape_to(g: &mut IGraph, name: &str, from: EdgeId, out: EdgeId) {
    let shape = dims_of(g, out);
    add_reshape_to_edge(g, name, from, out, &shape);
}

/// Reshape `from` into `to`, declaring `shape`.
fn add_reshape_to_edge(g: &mut IGraph, name: &str, from: EdgeId, to: EdgeId, shape: &[usize]) {
    g.push_node(INode {
        name: format!("{name}/reshape"),
        op: "Reshape".into(),
        attrs: [(
            "shape".to_string(),
            Attr::Ints(shape.iter().map(|&d| d as i64).collect()),
        )]
        .into(),
        inputs: vec![from],
        outputs: vec![to],
    });
}

/// Division elimination:
/// - `Reciprocal(Sqrt(v))` → atlas.Rsqrt(v)
/// - `Div(x, Sqrt(v))` → Mul(x, atlas.Rsqrt(v))
/// - `Div(x, const c)` → Mul(x, const 1/c)
///
/// Anything that leaves a runtime f32 Div/Sqrt/Reciprocal is rejected at
/// lowering.
fn div_rewrites(g: &mut IGraph, scale: Scale) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        match node.op.as_str() {
            "Reciprocal" => {
                let input = node.inputs[0];
                if let Some((_, p)) = g.producer(input)
                    && p.op == "Sqrt"
                {
                    let v = p.inputs[0];
                    let node = g.nodes[idx].as_mut().expect("live");
                    node.op = "atlas.Rsqrt".into();
                    node.inputs = vec![v];
                    rewrites += 1;
                }
            }
            "Div" => {
                let (x, d) = (node.inputs[0], node.inputs[1]);
                let name = node.name.clone();
                let out = node.outputs[0];
                if let Some(k) = g.konst(d) {
                    // Constant divisor: multiply by the reciprocal.
                    let Some(vals) = k.as_f32s() else {
                        continue; // integer Div would have folded; leave it
                    };
                    // 1/0 would silently become a huge finite constant
                    // under quantization; ONNX produces inf/NaN there.
                    if vals.iter().any(|&v| !(1.0 / v).is_finite()) {
                        return Err(ImportError::Pass {
                            pass: "div_rewrites",
                            node: name,
                            reason: "Div by a zero or non-finite constant".into(),
                        });
                    }
                    // A reciprocal below the fixed-point resolution would
                    // quantize to 0 and silently zero the quotient (the old
                    // crate's flush-to-±1 instead inflated it to 2^-s — a
                    // division by 2^s rather than by `v`). Reject loudly.
                    let mult = scale.denom_i64() as f64;
                    if let Some(&v) = vals
                        .iter()
                        .find(|&&v| ((1.0 / v as f64) * mult).round() == 0.0)
                    {
                        return Err(ImportError::Pass {
                            pass: "div_rewrites",
                            node: name,
                            reason: format!(
                                "Div by {v:e} exceeds fixed-point resolution at scale {}                                  (reciprocal quantizes to 0)",
                                scale.log2()
                            ),
                        });
                    }
                    let recip = ConstTensor {
                        dtype: DType::F32,
                        dims: k.dims.clone(),
                        data: ConstData::F32(vals.iter().map(|&v| 1.0 / v).collect()),
                    };
                    let re = g.fresh_edge(
                        &format!("{name}/recip"),
                        TensorInfo {
                            dtype: DType::F32,
                            dims: recip.dims.clone(),
                        },
                    );
                    g.edge_mut(re).konst = Some(recip);
                    g.push_node(INode {
                        name: format!("{name}/mul_recip"),
                        op: "Mul".into(),
                        attrs: Default::default(),
                        inputs: vec![x, re],
                        outputs: vec![out],
                    });
                    g.remove_node(idx);
                    rewrites += 1;
                } else if let Some((_, p)) = g.producer(d)
                    && p.op == "Sqrt"
                {
                    let v = p.inputs[0];
                    let inv = g.fresh_edge(&format!("{name}/rsqrt"), f32_info(&dims_of(g, d)));
                    g.push_node(INode {
                        name: format!("{name}/rsqrt"),
                        op: "atlas.Rsqrt".into(),
                        attrs: Default::default(),
                        inputs: vec![v],
                        outputs: vec![inv],
                    });
                    g.push_node(INode {
                        name: format!("{name}/mul_rsqrt"),
                        op: "Mul".into(),
                        attrs: Default::default(),
                        inputs: vec![x, inv],
                        outputs: vec![out],
                    });
                    g.remove_node(idx);
                    rewrites += 1;
                }
            }
            _ => {}
        }
    }
    Ok(rewrites)
}

/// Floor the eps constant feeding every `atlas.Rsqrt` at one fixed-point ulp.
///
/// `MeanOfSquares` floor-divides, so a small-but-nonzero row can quantize to
/// variance 0; `rsqrt(var + ⌈eps⌉₁)` then still normalizes it approximately
/// correctly, where `rsqrt(0)` would zero it — and the prover's rsqrt
/// division witness requires a positive operand outright. The old crate got
/// this implicitly from the global flush-to-±1 quantizer wart; this pass is
/// the explicit, local replacement: a positive sub-resolution constant added
/// directly under an `atlas.Rsqrt` is bumped to exactly one ulp (`2^-s`).
/// Every `atlas.Rsqrt` is produced by `norm_fusion` or `div_rewrites`, so
/// running after both covers synthesized and model-provided eps alike.
fn rsqrt_eps_floor(g: &mut IGraph, scale: Scale) -> Result<usize, ImportError> {
    let mult = scale.denom_i64() as f64;
    let ulp = (1.0 / mult) as f32;
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "atlas.Rsqrt" {
            continue;
        }
        let v = node.inputs[0];
        let Some((add_idx, add)) = g.producer(v) else {
            continue;
        };
        if add.op != "Add" {
            continue;
        }
        let eps_slot = add.inputs.iter().position(|&e| {
            g.konst(e).and_then(|k| k.as_f32s()).is_some_and(|vals| {
                !vals.is_empty()
                    && vals
                        .iter()
                        .all(|&c| c > 0.0 && ((c as f64) * mult).round() == 0.0)
            })
        });
        let Some(slot) = eps_slot else {
            continue;
        };
        let old_edge = add.inputs[slot];
        let dims = g.konst(old_edge).expect("checked above").dims.clone();
        let len: usize = dims.iter().product::<usize>().max(1);
        let name = format!("{}/eps_floor", g.nodes[idx].as_ref().expect("live").name);
        let fe = g.fresh_edge(
            &name,
            TensorInfo {
                dtype: DType::F32,
                dims: dims.clone(),
            },
        );
        g.edge_mut(fe).konst = Some(ConstTensor {
            dtype: DType::F32,
            dims,
            data: ConstData::F32(vec![ulp; len]),
        });
        g.nodes[add_idx].as_mut().expect("live").inputs[slot] = fe;
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Whether `perm` swaps only the last two axes.
fn is_last_two_swap(perm: &[usize]) -> bool {
    let r = perm.len();
    r >= 2
        && perm[r - 1] == r - 2
        && perm[r - 2] == r - 1
        && perm[..r - 2].iter().enumerate().all(|(i, &p)| p == i)
}

/// MatMul / Gemm → canonical `atlas.Einsum`, absorbing a last-two-axes
/// Transpose on the RHS (the Q·Kᵀ attention pattern) into the equation.
///
/// Every emitted equation names every axis of every operand — no dropped
/// letters, no size-1 reinterpretation. Non-canonical layouts get explicit
/// Reshape/MoveAxis nodes instead of exotic equations.
fn matmul_gemm_to_einsum(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "MatMul" && node.op != "Gemm" {
            continue;
        }
        let name = node.name.clone();
        let out = node.outputs[0];
        let is_gemm = node.op == "Gemm";
        // atlas.Einsum has no raw (rebase-free) form: its eval always divides
        // the contraction by 2^scale, so an integer matmul's result would be
        // silently shrunk by that factor. Reject it (lowering gates this too).
        if g.info(out).is_some_and(|i| i.dtype != DType::F32) {
            return Err(ImportError::Pass {
                pass: "matmul_gemm_to_einsum",
                node: name,
                reason: format!(
                    "{} over non-float operands cannot be lowered: \
                     einsum always rebases by 2^scale",
                    if is_gemm { "Gemm" } else { "MatMul" }
                ),
            });
        }
        let (mut a, mut b) = (node.inputs[0], node.inputs[1]);
        let bias = if is_gemm {
            node.inputs.get(2).copied()
        } else {
            None
        };

        if is_gemm {
            let node = g.nodes[idx].as_ref().expect("live");
            if node.attr_float("alpha").unwrap_or(1.0) != 1.0
                || node.attr_float("beta").unwrap_or(1.0) != 1.0
            {
                return Err(ImportError::Pass {
                    pass: "matmul_gemm_to_einsum",
                    node: name,
                    reason: "Gemm with alpha/beta != 1 unsupported".into(),
                });
            }
            if node.attr_int("transA").unwrap_or(0) != 0 {
                return Err(ImportError::Pass {
                    pass: "matmul_gemm_to_einsum",
                    node: name,
                    reason: "Gemm transA unsupported".into(),
                });
            }
        }

        // Gemm transB needs no materialization: `A·Bᵀ` is the canonical
        // "mk,nk->mn" contraction with B used as stored.
        let b_transposed = is_gemm
            && g.nodes[idx]
                .as_ref()
                .expect("live")
                .attr_int("transB")
                .unwrap_or(0)
                != 0;

        // Absorb Transpose(last two axes) feeding the RHS: both RHS layouts
        // ("kn"/"nk") are canonical. The LHS has no "km" einsum, so a
        // transposed LHS keeps its Transpose (it becomes a MoveAxis) rather
        // than being absorbed into an equation lowering would reject.
        let mut b_swapped = b_transposed;
        if let Some((_, p)) = g.producer(b)
            && (p.op == "Transpose")
            && p.attrs.get("perm").is_some_and(|attr| {
                matches!(attr, Attr::Ints(perm)
                    if is_last_two_swap(&perm.iter().map(|&v| v as usize).collect::<Vec<_>>()))
            })
        {
            b = p.inputs[0];
            b_swapped = !b_swapped;
        }

        let mut a_dims = dims_of(g, a);
        let b_dims = dims_of(g, b);

        // Linear-projection pattern: [1,…,1,m,k] · [k,n] (or transposed
        // RHS). Squeeze the unit batch dims, contract in 2-D, and reshape
        // back — the same shape tract reached via RmAxis/AddAxis.
        let mut restore_shape: Option<Vec<usize>> = None;
        if b_dims.len() == 2
            && a_dims.len() > 2
            && a_dims[..a_dims.len() - 2].iter().all(|&d| d == 1)
        {
            let squeezed: Vec<usize> = a_dims[a_dims.len() - 2..].to_vec();
            let sq_edge = g.fresh_edge(&format!("{name}/squeeze"), f32_info(&squeezed));
            add_reshape_to_edge(g, &format!("{name}/squeeze"), a, sq_edge, &squeezed);
            a = sq_edge;
            a_dims = squeezed;
            restore_shape = Some(dims_of(g, out));
        }
        let (ra, rb) = (a_dims.len(), b_dims.len());

        // Canonical equations: batch letters a, b, …; then m/k on the LHS
        // and k/n on the RHS (swap on a transposed operand).
        let batch = &b"abcd"[..];
        let equation = match (ra, rb) {
            (2, 2) => {
                let rhs = if b_swapped { "nk" } else { "kn" };
                format!("mk,{rhs}->mn")
            }
            (r, s) if r == s && (3..=4).contains(&r) => {
                // Batched forms normalize to a contraction-last RHS
                // ("…nk"), the one layout family the proof system supports
                // (`rbmk,rbnk->bmn`). An RHS that arrives in `…kn` layout
                // (no transpose to absorb, e.g. attention·V) gets an
                // explicit MoveAxis — a cheap structural node — instead of
                // its own einsum layout.
                if !b_swapped {
                    let mut t_dims = b_dims.clone();
                    t_dims.swap(r - 2, r - 1);
                    let t = g.fresh_edge(&format!("{name}/rhsT"), f32_info(&t_dims));
                    g.push_node(INode {
                        name: format!("{name}/rhsT"),
                        op: "atlas.MoveAxis".into(),
                        attrs: [
                            ("source".to_string(), Attr::Int((r - 2) as i64)),
                            ("destination".to_string(), Attr::Int((r - 1) as i64)),
                        ]
                        .into(),
                        inputs: vec![b],
                        outputs: vec![t],
                    });
                    b = t;
                }
                let batch_letters: String = batch[..r - 2].iter().map(|&c| c as char).collect();
                format!("{batch_letters}mk,{batch_letters}nk->{batch_letters}mn")
            }
            _ => {
                return Err(ImportError::Pass {
                    pass: "matmul_gemm_to_einsum",
                    node: name,
                    reason: format!(
                        "unsupported MatMul ranks {ra}x{rb} (dims {a_dims:?} · {b_dims:?}); \
                         broadcasted-batch matmul needs an explicit Broadcast"
                    ),
                });
            }
        };

        // Where the contraction result lands: directly on `out`, or on a
        // fresh edge when a bias add or a batch-dim restore follows.
        // (restore and bias never combine: Gemm is always rank 2.)
        let einsum_out = match (&restore_shape, bias) {
            (Some(full), None) => {
                let m = full[full.len() - 2];
                let n = full[full.len() - 1];
                g.fresh_edge(&format!("{name}/mm2d"), f32_info(&[m, n]))
            }
            (None, Some(_)) => g.fresh_edge(&format!("{name}/mm"), f32_info(&dims_of(g, out))),
            (None, None) => out,
            (Some(_), Some(_)) => {
                return Err(ImportError::Pass {
                    pass: "matmul_gemm_to_einsum",
                    node: name,
                    reason: "batched Gemm with bias unsupported".into(),
                });
            }
        };
        g.push_node(INode {
            name: format!("{name}/einsum"),
            op: "atlas.Einsum".into(),
            attrs: [("equation".to_string(), Attr::Str(equation))].into(),
            inputs: vec![a, b],
            outputs: vec![einsum_out],
        });
        if let Some(bias) = bias {
            g.push_node(INode {
                name: format!("{name}/bias"),
                op: "Add".into(),
                attrs: Default::default(),
                inputs: vec![einsum_out, bias],
                outputs: vec![out],
            });
        } else if restore_shape.is_some() {
            add_reshape_to(g, &format!("{name}/unsqueeze"), einsum_out, out);
        }
        g.remove_node(idx);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// A last-axis slice of a linear projection becomes a projection against
/// the sliced weight (and sliced bias): `Slice(x·W + b)` → `x·W' + b'`.
///
/// This is the rewrite tract applied to the fused QKV projections in GPT-2
/// (the per-head weight columns are materialized as their own constants),
/// and it is what keeps slice extents like 768 out of the padded graph.
fn slice_through_projection(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "atlas.Slice" && node.op != "Slice" {
            continue;
        }
        // Only the canonical single-axis form produced by split_to_slices.
        let (Some(axis), Some(start), Some(end)) = (
            node.attr_int("axis"),
            node.attr_int("start"),
            node.attr_int("end"),
        ) else {
            continue;
        };
        let (axis, start, end) = (axis as usize, start as usize, end as usize);
        let out = node.outputs[0];
        let input = node.inputs[0];
        let name = node.name.clone();
        let rank = dims_of(g, input).len();
        if axis != rank - 1 || (end - start).is_power_of_two() {
            continue; // power-of-two extents survive padding as-is
        }

        // Match, walking through at most one bias Add and one Reshape in
        // any order: input ← {Add(·, bias const), Reshape}* ← atlas.Einsum.
        // Slicing the last axis commutes with both (the Reshape must leave
        // the last axis unchanged — checked), so the rebuild can use a
        // fixed einsum → bias → reshape order.
        let mut inner = input;
        let mut bias: Option<EdgeId> = None;
        let mut through_reshape = false;
        let einsum_node = loop {
            let Some((_, p)) = g.producer(inner) else {
                break None;
            };
            match p.op.as_str() {
                "Add" if bias.is_none() => {
                    let (lhs, rhs) = (p.inputs[0], p.inputs[1]);
                    if g.konst(rhs).is_some() {
                        bias = Some(rhs);
                        inner = lhs;
                    } else if g.konst(lhs).is_some() {
                        bias = Some(lhs);
                        inner = rhs;
                    } else {
                        break None;
                    }
                }
                "Reshape" if !through_reshape => {
                    let in_last = dims_of(g, p.inputs[0]).last().copied();
                    let out_last = dims_of(g, p.outputs[0]).last().copied();
                    if in_last != out_last {
                        break None; // reshape disturbs the sliced axis
                    }
                    through_reshape = true;
                    inner = p.inputs[0];
                }
                "atlas.Einsum" => break Some(p.clone()),
                _ => break None,
            }
        };
        let Some(einsum_node) = einsum_node else {
            continue;
        };
        let Some(Attr::Str(eq)) = einsum_node.attrs.get("equation") else {
            continue;
        };
        // The output letter `n` must be the last axis of the contraction
        // output and come from the RHS weight: "…k,kn->…n" (n = W dim 1) or
        // "…k,nk->…n" (n = W dim 0).
        let Some((_, rhs_and_out)) = eq.split_once(',') else {
            continue;
        };
        let Some((rhs_letters, _)) = rhs_and_out.split_once("->") else {
            continue;
        };
        let n_axis = match rhs_letters {
            "kn" => 1,
            "nk" => 0,
            _ => continue,
        };
        let w_edge = einsum_node.inputs[1];
        let Some(w) = g.konst(w_edge) else {
            continue;
        };
        let Some(w_vals) = w.as_f32s() else { continue };

        // Slice the weight along its n axis.
        let (rows, cols) = (w.dims[0], w.dims[1]);
        let sliced: Vec<f32> = if n_axis == 1 {
            (0..rows)
                .flat_map(|r| (start..end).map(move |c| (r, c)))
                .map(|(r, c)| w_vals[r * cols + c])
                .collect()
        } else {
            w_vals[start * cols..end * cols].to_vec()
        };
        let mut w_dims = w.dims.clone();
        w_dims[n_axis] = end - start;
        let w_new = g.fresh_edge(
            &format!("{name}/w"),
            TensorInfo {
                dtype: DType::F32,
                dims: w_dims.clone(),
            },
        );
        g.edge_mut(w_new).konst = Some(ConstTensor {
            dtype: DType::F32,
            dims: w_dims,
            data: ConstData::F32(sliced),
        });

        // Slice the bias (broadcast biases are 1-D or [..,n]).
        let bias_new = match bias {
            Some(b_edge) => {
                let bk = g.konst(b_edge).expect("checked");
                let Some(b_vals) = bk.as_f32s() else { continue };
                let b_len = bk.len();
                if !b_len.is_multiple_of(*bk.dims.last().unwrap_or(&1))
                    || *bk.dims.last().unwrap_or(&0) < end
                {
                    continue;
                }
                let last = *bk.dims.last().unwrap();
                let outer = b_len / last;
                let data: Vec<f32> = (0..outer)
                    .flat_map(|o| (start..end).map(move |c| (o, c)))
                    .map(|(o, c)| b_vals[o * last + c])
                    .collect();
                let mut b_dims = bk.dims.clone();
                *b_dims.last_mut().unwrap() = end - start;
                let e = g.fresh_edge(
                    &format!("{name}/bias"),
                    TensorInfo {
                        dtype: DType::F32,
                        dims: b_dims.clone(),
                    },
                );
                g.edge_mut(e).konst = Some(ConstTensor {
                    dtype: DType::F32,
                    dims: b_dims,
                    data: ConstData::F32(data),
                });
                Some(e)
            }
            None => None,
        };

        // Rebuild the sliced projection (einsum' → [reshape] → [bias add]),
        // writing into the slice's output edge. The original projection
        // stays for other users and falls to DCE once every slice rewrites.
        let extent = end - start;
        let mut mm_dims = dims_of(g, einsum_node.outputs[0]);
        *mm_dims.last_mut().expect("einsum output rank >= 1") = extent;
        let needs_tail = through_reshape || bias_new.is_some();
        let mm_out = if needs_tail {
            g.fresh_edge(&format!("{name}/mm"), f32_info(&mm_dims))
        } else {
            out
        };
        g.push_node(INode {
            name: format!("{name}/einsum"),
            op: "atlas.Einsum".into(),
            attrs: einsum_node.attrs.clone(),
            inputs: vec![einsum_node.inputs[0], w_new],
            outputs: vec![mm_out],
        });
        let mut current = mm_out;
        if let Some(b) = bias_new {
            let dst = if through_reshape {
                g.fresh_edge(&format!("{name}/biased"), f32_info(&mm_dims))
            } else {
                out
            };
            g.push_node(INode {
                name: format!("{name}/bias_add"),
                op: "Add".into(),
                attrs: Default::default(),
                inputs: vec![current, b],
                outputs: vec![dst],
            });
            current = dst;
        }
        if through_reshape {
            let target = dims_of(g, out);
            add_reshape_to_edge(g, &format!("{name}/unsqueeze"), current, out, &target);
        }
        g.remove_node(idx);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Transpose → a minimal chain of atlas.MoveAxis nodes.
fn transpose_to_moveaxis(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Transpose" {
            continue;
        }
        let name = node.name.clone();
        let input = node.inputs[0];
        let out = node.outputs[0];
        let in_dims = dims_of(g, input);
        let rank = in_dims.len();
        let perm: Vec<usize> = match node.attr_ints("perm") {
            Some(p) => p.iter().map(|&v| v as usize).collect(),
            None => (0..rank).rev().collect(),
        };

        // Greedy decomposition: repeatedly move the axis that must land at
        // the leftmost mismatched position (selection-sort over axes).
        let mut current: Vec<usize> = (0..rank).collect();
        let mut moves: Vec<(usize, usize)> = Vec::new();
        for (target_pos, &axis) in perm.iter().enumerate() {
            let cur_pos = current.iter().position(|&x| x == axis).expect("axis");
            if cur_pos != target_pos {
                let v = current.remove(cur_pos);
                current.insert(target_pos, v);
                moves.push((cur_pos, target_pos));
            }
        }

        let mut src_edge = input;
        let mut dims = in_dims;
        for (i, &(from, to)) in moves.iter().enumerate() {
            let d = dims.remove(from);
            dims.insert(to, d);
            let dst_edge = if i + 1 == moves.len() {
                out
            } else {
                g.fresh_edge(&format!("{name}/move{i}"), f32_info(&dims))
            };
            g.push_node(INode {
                name: format!("{name}/move{i}"),
                op: "atlas.MoveAxis".into(),
                attrs: [
                    ("source".to_string(), Attr::Int(from as i64)),
                    ("destination".to_string(), Attr::Int(to as i64)),
                ]
                .into(),
                inputs: vec![src_edge],
                outputs: vec![dst_edge],
            });
            src_edge = dst_edge;
        }
        if moves.is_empty() {
            g.redirect_uses(out, input);
        }
        g.remove_node(idx);
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Expand (and Tile that only repeats size-1 axes) → atlas.Broadcast.
fn expand_to_broadcast(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Expand" && node.op != "Tile" {
            continue;
        }
        let name = node.name.clone();
        let input = node.inputs[0];
        let out = node.outputs[0];
        let out_dims = dims_of(g, out);
        if node.op == "Tile" {
            // Only broadcast-like tiles (repeats on size-1 axes) survive.
            let in_dims = dims_of(g, input);
            let ok = in_dims
                .iter()
                .zip(&out_dims)
                .all(|(&i, &o)| i == o || i == 1);
            if !ok {
                return Err(ImportError::Pass {
                    pass: "expand_to_broadcast",
                    node: name,
                    reason: "Tile with repeats on non-1 axes unsupported".into(),
                });
            }
        }
        let node = g.nodes[idx].as_mut().expect("live");
        node.op = "atlas.Broadcast".into();
        node.inputs = vec![input];
        node.attrs.clear();
        node.attrs.insert(
            "shape".into(),
            Attr::Ints(out_dims.iter().map(|&d| d as i64).collect()),
        );
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Rewrite the grouped-KV repeat pattern (`repeat_kv` in GQA attention)
/// into a padding-coherent head gather.
///
/// Pattern: `Broadcast` expanding one axis `1 → r`, followed by a `Reshape`
/// merging that axis into its predecessor (`[.., G, r, ..] → [.., G·r, ..]`).
/// When `r` is not a power of two, padding `r` upward would insert zero
/// slots *inside* the merged dimension and scramble the layout (the
/// soundness check in atlas-trace rejects that reshape). The equivalent
/// gather — indices `[0×r, 1×r, …]` on the group axis — pads coherently:
/// padded index lanes read group 0 and are annihilated by the zero-padded
/// projection weights downstream.
fn repeat_to_gather(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Reshape" {
            continue;
        }
        let input = node.inputs[0];
        let out = node.outputs[0];
        let name = node.name.clone();
        let Some((b_idx, b)) = g.producer(input) else {
            continue;
        };
        if b.op != "atlas.Broadcast" && b.op != "Expand" {
            continue;
        }
        let src = b.inputs[0];
        let src_dims = dims_of(g, src);
        let bcast_dims = dims_of(g, input);
        let out_dims = dims_of(g, out);
        // Exactly one axis expanded 1 → r, and it has a predecessor axis.
        let expanded: Vec<usize> = (0..src_dims.len())
            .filter(|&a| src_dims[a] != bcast_dims[a])
            .collect();
        let [bx] = expanded[..] else { continue };
        if src_dims[bx] != 1 || bx == 0 {
            continue;
        }
        let r = bcast_dims[bx];
        if r.is_power_of_two() {
            continue; // plain reshape pads coherently
        }
        let grp = bx - 1;
        let group = bcast_dims[grp];
        // The reshape must merge exactly (grp, bx) and keep the rest.
        let mut merged = bcast_dims.clone();
        merged[grp] = group * r;
        merged.remove(bx);
        if merged != out_dims {
            continue;
        }

        // src without the broadcast 1-axis.
        let mut squeezed = src_dims.clone();
        squeezed.remove(bx);
        let info_of = |dims: &[usize]| TensorInfo {
            dtype: DType::F32,
            dims: dims.to_vec(),
        };
        let sq = g.fresh_edge(&format!("{name}/kv"), info_of(&squeezed));
        add_reshape_to_edge(g, &format!("{name}/squeeze"), src, sq, &squeezed);

        // Move the group axis to the front, gather, move it back.
        let mut fronted = squeezed.clone();
        let gdim = fronted.remove(grp);
        fronted.insert(0, gdim);
        let m1 = g.fresh_edge(&format!("{name}/front"), info_of(&fronted));
        g.push_node(INode {
            name: format!("{name}/front"),
            op: "atlas.MoveAxis".into(),
            attrs: [
                ("source".to_string(), Attr::Int(grp as i64)),
                ("destination".to_string(), Attr::Int(0)),
            ]
            .into(),
            inputs: vec![sq],
            outputs: vec![m1],
        });

        let indices: Vec<i64> = (0..group as i64)
            .flat_map(|gi| std::iter::repeat_n(gi, r))
            .collect();
        let idx_edge = g.fresh_edge(
            &format!("{name}/repeat_idx"),
            TensorInfo {
                dtype: DType::I64,
                dims: vec![indices.len()],
            },
        );
        g.edge_mut(idx_edge).konst = Some(ConstTensor {
            dtype: DType::I64,
            dims: vec![indices.len()],
            data: ConstData::I64(indices),
        });
        let mut gathered = fronted.clone();
        gathered[0] = group * r;
        let ge = g.fresh_edge(&format!("{name}/gathered"), info_of(&gathered));
        g.push_node(INode {
            name: format!("{name}/repeat"),
            op: "Gather".into(),
            attrs: [("axis".to_string(), Attr::Int(0))].into(),
            inputs: vec![m1, idx_edge],
            outputs: vec![ge],
        });
        g.push_node(INode {
            name: format!("{name}/back"),
            op: "atlas.MoveAxis".into(),
            attrs: [
                ("source".to_string(), Attr::Int(0)),
                ("destination".to_string(), Attr::Int(grp as i64)),
            ]
            .into(),
            inputs: vec![ge],
            outputs: vec![out],
        });
        g.remove_node(idx);
        let _ = b_idx; // broadcast falls to DCE when unused
        rewrites += 1;
    }
    Ok(rewrites)
}

/// Cast policy in the quantized domain:
/// - float → int/bool: `atlas.Descale` (divide by `2^scale`; maps a scaled
///   0/S mask to raw 0/1) — old-crate parity;
/// - int/bool → float: identity, but ONLY when the operand provably carries
///   already-scaled values (see [`scaled_lineage`]); anything else would
///   leak raw integers into the scaled domain, silently off by `2^scale`.
fn cast_policy(g: &mut IGraph) -> Result<usize, ImportError> {
    let mut rewrites = 0;
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if node.op != "Cast" {
            continue;
        }
        let name = node.name.clone();
        let input = node.inputs[0];
        let src = g.info(input).map(|i| i.dtype).expect("analyzed");
        let dst = g.info(node.outputs[0]).map(|i| i.dtype).expect("analyzed");
        if matches!(src, DType::I64 | DType::I32 | DType::Bool) && dst == DType::F32 {
            match (scaled_lineage(g, input), raw_lineage(g, input)) {
                // Pre-scaled by the input descriptor: the cast is a no-op.
                (Ok(()), _) => {
                    let node = g.nodes[idx].as_mut().expect("live");
                    node.attrs.clear();
                    node.op = "Identity".into();
                }
                // Provably raw (e.g. the floor idiom Cast(Cast(x→int)→f32)):
                // entering the scaled domain multiplies by 2^scale, done as
                // a raw Mul with constant 1.0 (which quantizes to 2^scale).
                (Err(_), Ok(())) => {
                    let one = g.fresh_edge(
                        &format!("{name}/rescale"),
                        TensorInfo {
                            dtype: DType::F32,
                            dims: vec![1],
                        },
                    );
                    g.edge_mut(one).konst = Some(ConstTensor {
                        dtype: DType::F32,
                        dims: vec![1],
                        data: ConstData::F32(vec![1.0]),
                    });
                    let node = g.nodes[idx].as_mut().expect("live");
                    node.attrs.clear();
                    node.op = "atlas.RawMul".into();
                    node.inputs = vec![input, one];
                }
                // Mixed or unprovable: refuse rather than guess the domain.
                (Err(scaled), Err(raw)) => {
                    return Err(ImportError::Pass {
                        pass: "cast_policy",
                        node: name,
                        reason: format!(
                            "int→float Cast is neither provably scaled ({scaled}) nor provably raw ({raw})"
                        ),
                    });
                }
            }
            rewrites += 1;
            continue;
        }
        let node = g.nodes[idx].as_mut().expect("live");
        node.attrs.clear();
        match (src, dst) {
            (DType::F32, DType::I64 | DType::I32 | DType::Bool) => {
                node.op = "atlas.Descale".into();
            }
            _ => {
                node.op = "Identity".into();
            }
        }
        rewrites += 1;
    }
    Ok(rewrites)
}

/// ONNX ops that carry their (first) data operand's values through unchanged
/// — pure data movement. The single source of truth shared by input
/// classification ([`crate::inputs`]) and the cast-policy lineage walks
/// below: the two walks must cover the same op set, or an input could be
/// classified `RawInt` while its Cast lowers to identity — silently off by
/// `2^scale`. `Concat` merges several operands, so lineage walks handle it
/// in its own arm (every operand is part of the lineage).
pub(crate) const VALUE_PRESERVING_OPS: &[&str] = &[
    "Unsqueeze",
    "Squeeze",
    "Reshape",
    "Flatten",
    "Identity",
    "Slice",
    "Expand",
    "Tile",
    "Transpose",
    "Split",
    "Gather",
    "Concat",
];

/// Pass-introduced structural aliases of [`VALUE_PRESERVING_OPS`] entries.
/// The rewrite passes run before `cast_policy`, so its lineage walks see
/// these names too; input classification runs before the passes and never
/// does.
const VALUE_PRESERVING_ATLAS_OPS: &[&str] = &["atlas.Slice", "atlas.MoveAxis", "atlas.Broadcast"];

/// Whether `op` passes its data operand(s) through unchanged (either name
/// surface).
fn value_preserving(op: &str) -> bool {
    VALUE_PRESERVING_OPS.contains(&op) || VALUE_PRESERVING_ATLAS_OPS.contains(&op)
}

/// Verify that an int→float Cast operand is scaled by construction: its
/// lineage must pass only through value-preserving ops and terminate at
/// graph inputs (which `classify_inputs` then marks Scaled through the very
/// same ops — [`VALUE_PRESERVING_OPS`] — so the caller supplies pre-scaled
/// values). Integer arithmetic, a descaled float, or an int constant in the
/// lineage would be raw, and lowering the Cast to identity would misread it
/// as scaled — so reject.
fn scaled_lineage(g: &IGraph, start: EdgeId) -> Result<(), String> {
    let mut frontier = vec![start];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = frontier.pop() {
        if !seen.insert(e) {
            continue;
        }
        if g.inputs.contains(&e) {
            continue;
        }
        if g.konst(e).is_some() {
            return Err(format!(
                "int→float Cast lineage reaches constant {}, whose raw value \
                 would be misread as scaled",
                g.edge(e).name
            ));
        }
        let Some((_, p)) = g.producer(e) else {
            return Err(format!("{} has no producer", g.edge(e).name));
        };
        match p.op.as_str() {
            "Concat" => frontier.extend(p.inputs.iter().copied()),
            // Gather/Slice carry their data operand through unchanged; the
            // index/shape operands are not part of the value lineage.
            op if value_preserving(op) => frontier.push(p.inputs[0]),
            other => {
                return Err(format!(
                    "int→float Cast fed by `{other}`; only value-preserving \
                     lineages from classified inputs lower to identity"
                ));
            }
        }
    }
    Ok(())
}

/// The dual of [`scaled_lineage`]: the operand provably carries raw integer
/// values — every leaf is an int constant or the output of a float→int Cast
/// (both raw by construction), reached through value-preserving ops or raw
/// integer arithmetic, and never a graph input (whose classification this
/// pass cannot see).
fn raw_lineage(g: &IGraph, start: EdgeId) -> Result<(), String> {
    let mut frontier = vec![start];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = frontier.pop() {
        if !seen.insert(e) {
            continue;
        }
        if g.inputs.contains(&e) {
            return Err(format!(
                "{} is a graph input; its quantization kind is not visible here",
                g.edge(e).name
            ));
        }
        if g.konst(e).is_some() {
            continue; // unfolded int constants are raw
        }
        let Some((_, p)) = g.producer(e) else {
            return Err(format!("{} has no producer", g.edge(e).name));
        };
        match p.op.as_str() {
            // A float→int cast produces raw values, whatever fed it.
            "atlas.Descale" => continue,
            "Cast" if g.info(e).is_some_and(|i| i.dtype != DType::F32) => continue,
            // Raw integer arithmetic stays raw (integer Mul lowers without
            // the fixed-point rebase).
            "Add" | "Sub" | "Mul" | "Neg" | "Concat" | "Where" | "And" | "Equal" | "Less"
            | "LessOrEqual" | "Greater" => frontier.extend(p.inputs.iter().copied()),
            op if value_preserving(op) => frontier.push(p.inputs[0]),
            other => return Err(format!("`{other}` is not provably raw")),
        }
    }
    Ok(())
}

/// Make `Edge` debug-printable for pass tests.
impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An integer MatMul must be rejected: atlas.Einsum always rebases by
    // 2^scale, which would silently divide raw integer products.
    #[test]
    fn integer_matmul_is_rejected() {
        let mut g = IGraph::default();
        let info = TensorInfo {
            dtype: DType::I64,
            dims: vec![2, 2],
        };
        let a = g.fresh_edge("a", info.clone());
        let b = g.fresh_edge("b", info.clone());
        let out = g.fresh_edge("out", info);
        g.push_node(INode {
            name: "mm".into(),
            op: "MatMul".into(),
            attrs: Default::default(),
            inputs: vec![a, b],
            outputs: vec![out],
        });
        g.outputs.push(out);
        let err = matmul_gemm_to_einsum(&mut g).unwrap_err();
        assert!(matches!(
            err,
            ImportError::Pass {
                pass: "matmul_gemm_to_einsum",
                ..
            }
        ));
    }

    // Div by a constant whose reciprocal is below the fixed-point resolution
    // must be rejected: quantizing it would silently zero the quotient (and
    // the old flush-to-±1 turned it into a division by 2^scale instead).
    #[test]
    fn div_by_sub_resolution_reciprocal_is_rejected() {
        let scale = Scale::new(14).unwrap();
        let build = |divisor: f32| {
            let mut g = IGraph::default();
            let info = TensorInfo {
                dtype: DType::F32,
                dims: vec![2],
            };
            let x = g.fresh_edge("x", info.clone());
            let d = g.fresh_edge("d", info.clone());
            g.edge_mut(d).konst = Some(ConstTensor {
                dtype: DType::F32,
                dims: vec![2],
                data: ConstData::F32(vec![divisor, divisor]),
            });
            let out = g.fresh_edge("out", info);
            g.push_node(INode {
                name: "div".into(),
                op: "Div".into(),
                attrs: Default::default(),
                inputs: vec![x, d],
                outputs: vec![out],
            });
            g.outputs.push(out);
            g
        };

        // 1/1e7 quantizes to 0 at scale 14: rejected.
        let err = div_rewrites(&mut build(1e7), scale).unwrap_err();
        assert!(matches!(
            err,
            ImportError::Pass {
                pass: "div_rewrites",
                ..
            }
        ));
        // 1/100 is representable: rewritten to a Mul by the reciprocal.
        assert_eq!(div_rewrites(&mut build(100.0), scale).unwrap(), 1);
    }

    // A positive sub-resolution eps under an atlas.Rsqrt is floored to one
    // ulp; a representable eps is left alone.
    #[test]
    fn rsqrt_eps_is_floored_to_one_ulp() {
        let scale = Scale::new(14).unwrap();
        let build = |eps: f32| {
            let mut g = IGraph::default();
            let info = TensorInfo {
                dtype: DType::F32,
                dims: vec![2, 1],
            };
            let var = g.fresh_edge("var", info.clone());
            let e = g.fresh_edge(
                "eps",
                TensorInfo {
                    dtype: DType::F32,
                    dims: vec![1],
                },
            );
            g.edge_mut(e).konst = Some(ConstTensor {
                dtype: DType::F32,
                dims: vec![1],
                data: ConstData::F32(vec![eps]),
            });
            let ve = g.fresh_edge("var_eps", info.clone());
            g.push_node(INode {
                name: "add".into(),
                op: "Add".into(),
                attrs: Default::default(),
                inputs: vec![var, e],
                outputs: vec![ve],
            });
            let inv = g.fresh_edge("inv", info);
            g.push_node(INode {
                name: "rsqrt".into(),
                op: "atlas.Rsqrt".into(),
                attrs: Default::default(),
                inputs: vec![ve],
                outputs: vec![inv],
            });
            g.outputs.push(inv);
            g
        };

        let mut g = build(1e-5); // sub-resolution at scale 14
        assert_eq!(rsqrt_eps_floor(&mut g, scale).unwrap(), 1);
        let (_, add) = g
            .producer(
                g.nodes
                    .iter()
                    .flatten()
                    .find(|n| n.op == "atlas.Rsqrt")
                    .unwrap()
                    .inputs[0],
            )
            .unwrap();
        let floored = g.konst(add.inputs[1]).unwrap().as_f32s().unwrap();
        assert_eq!(floored, vec![1.0 / 16384.0]);

        let mut g = build(1e-2); // representable: untouched
        assert_eq!(rsqrt_eps_floor(&mut g, scale).unwrap(), 0);
    }
}
