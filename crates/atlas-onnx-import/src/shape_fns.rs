//! Static, monomorphic shape inference: one function per supported ONNX op,
//! over fully bound dims. No symbolic algebra — a shape either resolves to
//! concrete `usize`s or the import fails with the node's name attached.

use crate::graph::{DType, IGraph, INode, TensorInfo, checked_element_count};

/// Infer the output infos of one (normalized) ONNX node.
pub fn infer(node: &INode, g: &IGraph) -> Result<Vec<TensorInfo>, String> {
    let info = |i: usize| -> Result<&TensorInfo, String> {
        let &edge = node
            .inputs
            .get(i)
            .ok_or_else(|| format!("missing input {i}"))?;
        g.info(edge)
            .ok_or_else(|| format!("input {} has no shape yet", g.edge(edge).name))
    };
    let one = |dtype: DType, dims: Vec<usize>| Ok(vec![TensorInfo { dtype, dims }]);

    match node.op.as_str() {
        // Elementwise, same shape, same type.
        "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Sin" | "Cos" | "Sqrt" | "Reciprocal" | "Neg"
        | "Identity" | "Softmax" => {
            let x = info(0)?;
            one(x.dtype, x.dims.clone())
        }
        "IsNaN" => {
            let x = info(0)?;
            one(DType::Bool, x.dims.clone())
        }
        "Cast" => {
            let x = info(0)?;
            let to = node.attr_int("to").ok_or("Cast missing `to`")?;
            use crate::proto::tensor_proto::DataType;
            let dtype = match DataType::try_from(to as i32) {
                Ok(DataType::Float) => DType::F32,
                Ok(DataType::Int64) => DType::I64,
                Ok(DataType::Int32) => DType::I32,
                Ok(DataType::Bool) => DType::Bool,
                other => return Err(format!("unsupported Cast target {other:?}")),
            };
            one(dtype, x.dims.clone())
        }

        // Binary broadcasting arithmetic.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            let (a, b) = (info(0)?, info(1)?);
            let dtype = if node.op == "Pow" {
                a.dtype
            } else {
                promote(a.dtype, b.dtype)?
            };
            one(dtype, broadcast(&a.dims, &b.dims)?)
        }
        "Equal" | "LessOrEqual" | "Less" | "Greater" => {
            let (a, b) = (info(0)?, info(1)?);
            one(DType::Bool, broadcast(&a.dims, &b.dims)?)
        }
        "And" => {
            let (a, b) = (info(0)?, info(1)?);
            one(DType::Bool, broadcast(&a.dims, &b.dims)?)
        }
        "Where" => {
            let (c, a, b) = (info(0)?, info(1)?, info(2)?);
            let dtype = promote(a.dtype, b.dtype)?;
            one(dtype, broadcast(&broadcast(&c.dims, &a.dims)?, &b.dims)?)
        }

        "MatMul" => {
            let (a, b) = (info(0)?, info(1)?);
            matmul_shape(&a.dims, &b.dims).map(|dims| {
                vec![TensorInfo {
                    dtype: a.dtype,
                    dims,
                }]
            })
        }
        "Gemm" => {
            let (a, b) = (info(0)?, info(1)?);
            if a.dims.len() != 2 || b.dims.len() != 2 {
                return Err("Gemm operands must be rank 2".into());
            }
            let ta = node.attr_int("transA").unwrap_or(0) != 0;
            let tb = node.attr_int("transB").unwrap_or(0) != 0;
            let (m, ka) = if ta {
                (a.dims[1], a.dims[0])
            } else {
                (a.dims[0], a.dims[1])
            };
            let (kb, n) = if tb {
                (b.dims[1], b.dims[0])
            } else {
                (b.dims[0], b.dims[1])
            };
            if ka != kb {
                return Err(format!("Gemm contraction mismatch {ka} vs {kb}"));
            }
            one(a.dtype, vec![m, n])
        }

        "ReduceMean" | "ReduceSum" => {
            let x = info(0)?;
            let keepdims = node.attr_int("keepdims").unwrap_or(1) != 0;
            let axes: Vec<usize> = match node.attr_ints("axes") {
                Some(axes) => axes
                    .iter()
                    .map(|&a| norm_axis(a, x.dims.len()))
                    .collect::<Result<_, _>>()?,
                None => (0..x.dims.len()).collect(),
            };
            // Repeated axes would silently skew the mean's element count
            // downstream (norm fusion multiplies the axis dims).
            let mut dedup = axes.clone();
            dedup.sort_unstable();
            dedup.dedup();
            if dedup.len() != axes.len() {
                return Err(format!("{} repeats a reduction axis", node.op));
            }
            let mut dims = Vec::new();
            for (i, &d) in x.dims.iter().enumerate() {
                if axes.contains(&i) {
                    if keepdims {
                        dims.push(1);
                    }
                } else {
                    dims.push(d);
                }
            }
            one(x.dtype, dims)
        }

        "Reshape" => {
            let x = info(0)?;
            let target = g
                .konst(*node.inputs.get(1).ok_or("Reshape missing shape input")?)
                .ok_or("Reshape target must be constant")?
                .as_i64s()
                .ok_or("Reshape target must be integer")?
                .to_vec();
            // With allowzero=1 (opset 14) a 0 in the target is a literal
            // zero dim, not "copy the input dim" — unsupported, so reject
            // rather than silently reinterpret.
            if node.attr_int("allowzero").unwrap_or(0) != 0 && target.contains(&0) {
                return Err("Reshape with allowzero=1 and a 0 target dim".into());
            }
            one(x.dtype, reshape_dims(&x.dims, &target)?)
        }
        "Squeeze" => {
            let x = info(0)?;
            let dims = match node.attr_ints("axes") {
                Some(axes) => {
                    let axes: Vec<usize> = axes
                        .iter()
                        .map(|&a| norm_axis(a, x.dims.len()))
                        .collect::<Result<_, _>>()?;
                    // ONNX requires squeezed dims to be 1; dropping a larger
                    // dim would desynchronize dims from the element count.
                    for &a in &axes {
                        if x.dims[a] != 1 {
                            return Err(format!(
                                "Squeeze axis {a} has size {}, expected 1",
                                x.dims[a]
                            ));
                        }
                    }
                    x.dims
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes.contains(i))
                        .map(|(_, &d)| d)
                        .collect()
                }
                None => x.dims.iter().copied().filter(|&d| d != 1).collect(),
            };
            one(x.dtype, dims)
        }
        "Unsqueeze" => {
            let x = info(0)?;
            let axes = node.attr_ints("axes").ok_or("Unsqueeze missing axes")?;
            let out_rank = x.dims.len() + axes.len();
            let axes: Vec<usize> = axes
                .iter()
                .map(|&a| norm_axis(a, out_rank))
                .collect::<Result<_, _>>()?;
            let mut dims = Vec::with_capacity(out_rank);
            let mut src = x.dims.iter();
            for i in 0..out_rank {
                if axes.contains(&i) {
                    dims.push(1);
                } else {
                    dims.push(*src.next().ok_or("Unsqueeze axes malformed")?);
                }
            }
            one(x.dtype, dims)
        }
        "Flatten" => {
            let x = info(0)?;
            // ONNX Flatten: axis in [-r, r]; axis == r is valid (everything
            // in the leading dim), but a negative axis counts from r, so it
            // normalizes against the rank, not rank + 1. Zero-size dims stay
            // zero: the empty product is already 1.
            let raw = node.attr_int("axis").unwrap_or(1);
            let axis = if raw < 0 {
                norm_axis(raw, x.dims.len())?
            } else {
                norm_axis(raw, x.dims.len() + 1)?
            };
            let (lead, tail) = x.dims.split_at(axis);
            one(
                x.dtype,
                vec![
                    lead.iter().product::<usize>(),
                    tail.iter().product::<usize>(),
                ],
            )
        }
        "Transpose" => {
            let x = info(0)?;
            let perm: Vec<usize> = match node.attr_ints("perm") {
                Some(p) => validate_perm(p, x.dims.len())?,
                None => (0..x.dims.len()).rev().collect(),
            };
            one(x.dtype, perm.iter().map(|&p| x.dims[p]).collect())
        }
        "Concat" => {
            let first = info(0)?.clone();
            let axis = norm_axis(
                node.attr_int("axis").ok_or("missing axis")?,
                first.dims.len(),
            )?;
            let mut dims = first.dims.clone();
            dims[axis] = 0;
            for i in 0..node.inputs.len() {
                let x = info(i)?;
                let compatible = x.dims.len() == first.dims.len()
                    && x.dims
                        .iter()
                        .zip(&first.dims)
                        .enumerate()
                        .all(|(d, (a, b))| d == axis || a == b);
                if !compatible {
                    return Err(format!(
                        "Concat shape mismatch: {:?} vs {:?} on axis {axis}",
                        x.dims, first.dims
                    ));
                }
                dims[axis] += x.dims[axis];
            }
            one(first.dtype, dims)
        }
        "Slice" => {
            let x = info(0)?;
            let (_, _, _, _, dims) = slice_params(node, &x.dims)?;
            one(x.dtype, dims)
        }
        "Split" => {
            let x = info(0)?;
            let axis = norm_axis(node.attr_int("axis").unwrap_or(0), x.dims.len())?;
            let n = node.outputs.len();
            if n == 0 {
                return Err("Split with no outputs".into());
            }
            let sizes: Vec<usize> = match node.attr_ints("split") {
                Some(s) => s
                    .iter()
                    .map(|&v| checked_dim(v, "Split size"))
                    .collect::<Result<_, _>>()?,
                None => {
                    if let Some(num) = node.attr_int("num_outputs") {
                        // Opset 18 form: uneven splits allowed, chunks of
                        // ceil(size/n) with a smaller final chunk.
                        if num != n as i64 {
                            return Err(format!(
                                "Split num_outputs {num} but node has {n} outputs"
                            ));
                        }
                        let chunk = x.dims[axis].div_ceil(n);
                        let mut remaining = x.dims[axis];
                        (0..n)
                            .map(|_| {
                                let s = chunk.min(remaining);
                                remaining -= s;
                                s
                            })
                            .collect()
                    } else {
                        if x.dims[axis] % n != 0 {
                            return Err(format!(
                                "Split axis {} of size {} not divisible by {n}",
                                axis, x.dims[axis]
                            ));
                        }
                        vec![x.dims[axis] / n; n]
                    }
                }
            };
            let covered = sizes.iter().try_fold(0usize, |acc, &s| acc.checked_add(s));
            if sizes.len() != n || covered != Some(x.dims[axis]) {
                return Err("Split sizes do not cover the axis".into());
            }
            Ok(sizes
                .into_iter()
                .map(|s| {
                    let mut dims = x.dims.clone();
                    dims[axis] = s;
                    TensorInfo {
                        dtype: x.dtype,
                        dims,
                    }
                })
                .collect())
        }
        "Gather" => {
            let (data, idx) = (info(0)?, info(1)?);
            let axis = norm_axis(node.attr_int("axis").unwrap_or(0), data.dims.len())?;
            let mut dims = data.dims[..axis].to_vec();
            dims.extend_from_slice(&idx.dims);
            dims.extend_from_slice(&data.dims[axis + 1..]);
            one(data.dtype, dims)
        }
        "Shape" => {
            let x = info(0)?;
            let (start, end) = shape_slice_bounds(node, x.dims.len());
            one(DType::I64, vec![end.saturating_sub(start)])
        }
        "ConstantOfShape" => {
            let shape = g
                .konst(*node.inputs.first().ok_or("ConstantOfShape missing input")?)
                .ok_or("ConstantOfShape input must be constant")?
                .as_i64s()
                .ok_or("ConstantOfShape input must be integer")?;
            let dtype = match node.attrs.get("value") {
                Some(crate::graph::Attr::Tensor(t)) => t.dtype,
                _ => DType::F32,
            };
            let dims = shape
                .iter()
                .map(|&d| checked_dim(d, "ConstantOfShape"))
                .collect::<Result<_, _>>()?;
            one(dtype, dims)
        }
        "Expand" => {
            let x = info(0)?;
            let shape: Vec<usize> = g
                .konst(*node.inputs.get(1).ok_or("Expand missing shape input")?)
                .ok_or("Expand shape must be constant")?
                .as_i64s()
                .ok_or("Expand shape must be integer")?
                .iter()
                .map(|&d| checked_dim(d, "Expand"))
                .collect::<Result<_, _>>()?;
            one(x.dtype, broadcast(&x.dims, &shape)?)
        }
        "Tile" => {
            let x = info(0)?;
            let repeats = g
                .konst(*node.inputs.get(1).ok_or("Tile missing repeats input")?)
                .ok_or("Tile repeats must be constant")?
                .as_i64s()
                .ok_or("Tile repeats must be integer")?;
            if repeats.len() != x.dims.len() {
                return Err("Tile repeats rank mismatch".into());
            }
            let dims = x
                .dims
                .iter()
                .zip(repeats)
                .map(|(&d, &r)| {
                    checked_dim(r, "Tile repeat")?
                        .checked_mul(d)
                        .ok_or_else(|| format!("Tile dim {d} x {r} overflows"))
                })
                .collect::<Result<_, _>>()?;
            one(x.dtype, dims)
        }
        "LayerNormalization" => {
            let x = info(0)?;
            if node.outputs.len() != 1 {
                return Err("LayerNormalization with mean/invstd outputs unsupported".into());
            }
            one(x.dtype, x.dims.clone())
        }
        // Data-dependent shapes: these must constant-fold; reaching shape
        // inference means a runtime operand.
        "Range" | "NonZero" => Err(format!("{} must be constant at import time", node.op)),
        other => Err(format!("unsupported operator {other}")),
    }
}

/// The resolved `[start, end)` window of ONNX `Shape` (opset 15 added the
/// `start`/`end` attributes: negative values count from the rank, then both
/// clamp to `[0, rank]`).
pub fn shape_slice_bounds(node: &INode, rank: usize) -> (usize, usize) {
    let r = rank as i64;
    let clamp = |v: i64| (if v < 0 { v + r } else { v }).clamp(0, r) as usize;
    (
        clamp(node.attr_int("start").unwrap_or(0)),
        clamp(node.attr_int("end").unwrap_or(r)),
    )
}

/// ONNX Reshape semantics: `0` copies the input dim, one `-1` is inferred.
pub fn reshape_dims(input: &[usize], target: &[i64]) -> Result<Vec<usize>, String> {
    // The target dims are untrusted i64s: every product is checked (and
    // capped) so a hostile shape can't overflow into a bogus element count.
    let numel = |dims: &[usize]| -> Result<usize, String> {
        checked_element_count(dims).ok_or_else(|| format!("Reshape shape {dims:?} overflows"))
    };
    let total = numel(input)?;
    let mut out: Vec<usize> = Vec::with_capacity(target.len());
    let mut infer_pos = None;
    for (i, &t) in target.iter().enumerate() {
        match t {
            0 => out.push(*input.get(i).ok_or("Reshape 0-dim out of range")?),
            -1 => {
                if infer_pos.replace(i).is_some() {
                    return Err("Reshape has two -1 dims".into());
                }
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            _ => return Err(format!("invalid Reshape target dim {t}")),
        }
    }
    if let Some(i) = infer_pos {
        let known = numel(&out)?;
        if known == 0 || !total.is_multiple_of(known) {
            return Err(format!("cannot infer -1 dim: {total} vs {out:?}"));
        }
        out[i] = total / known;
    }
    if numel(&out)? != total {
        return Err(format!(
            "Reshape {input:?} -> {out:?} changes element count"
        ));
    }
    Ok(out)
}

/// Per-sliced-axis `(axes, starts, ends, steps, output_dims)`.
pub type SliceParams = (Vec<usize>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<usize>);

/// Slice parameter resolution with ONNX clamping semantics.
pub fn slice_params(node: &INode, input_dims: &[usize]) -> Result<SliceParams, String> {
    let starts = node.attr_ints("starts").ok_or("Slice missing starts")?;
    let ends = node.attr_ints("ends").ok_or("Slice missing ends")?;
    let axes: Vec<usize> = match node.attr_ints("axes") {
        Some(a) => a
            .iter()
            .map(|&v| norm_axis(v, input_dims.len()))
            .collect::<Result<_, _>>()?,
        None => (0..starts.len()).collect(),
    };
    let steps: Vec<i64> = match node.attr_ints("steps") {
        Some(s) => s.to_vec(),
        None => vec![1; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err("Slice parameter lengths disagree".into());
    }
    // ONNX requires unique axes. A repeated axis would desynchronize the
    // consumers of these params: the output dims keep the LAST occurrence's
    // extent while the fold's coordinate mapping and the lowered chain of
    // single-axis slices apply the FIRST occurrence's bounds.
    let mut dedup = axes.clone();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != axes.len() {
        return Err("Slice repeats an axis".into());
    }
    let mut dims = input_dims.to_vec();
    let mut starts_out = Vec::new();
    let mut ends_out = Vec::new();
    for (((&start, &end), &axis), &step) in starts.iter().zip(ends).zip(&axes).zip(&steps) {
        if step != 1 {
            return Err(format!("Slice step {step} unsupported (only 1)"));
        }
        let size = input_dims[axis] as i64;
        let clamp = |v: i64| -> i64 {
            let v = if v < 0 { v + size } else { v };
            v.clamp(0, size)
        };
        let (s, e) = (clamp(start), clamp(end));
        // ONNX: start >= end (after clamping) yields an empty output, not an
        // error — shape-subgraph-computed bounds produce this legitimately.
        let e = e.max(s);
        dims[axis] = (e - s) as usize;
        starts_out.push(s);
        ends_out.push(e);
    }
    Ok((axes, starts_out, ends_out, steps, dims))
}

/// MatMul shape with NumPy batch broadcasting and 1-D promotion.
fn matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => Err("MatMul with rank-0 operand".into()),
        (1, 1) => {
            if a[0] != b[0] {
                return Err(format!("MatMul 1-D mismatch {a:?} vs {b:?}"));
            }
            Ok(vec![])
        }
        (1, _) => {
            // [k] · [..., k, n] -> [..., n]
            let k = a[0];
            if b[b.len() - 2] != k {
                return Err(format!("MatMul mismatch {a:?} vs {b:?}"));
            }
            let mut out = b[..b.len() - 2].to_vec();
            out.push(b[b.len() - 1]);
            Ok(out)
        }
        (_, 1) => {
            let k = b[0];
            if a[a.len() - 1] != k {
                return Err(format!("MatMul mismatch {a:?} vs {b:?}"));
            }
            Ok(a[..a.len() - 1].to_vec())
        }
        _ => {
            let (m, ka) = (a[a.len() - 2], a[a.len() - 1]);
            let (kb, n) = (b[b.len() - 2], b[b.len() - 1]);
            if ka != kb {
                return Err(format!("MatMul contraction mismatch {a:?} vs {b:?}"));
            }
            let batch = broadcast(&a[..a.len() - 2], &b[..b.len() - 2])?;
            let mut out = batch;
            out.push(m);
            out.push(n);
            Ok(out)
        }
    }
}

/// A non-negative `i64` dim (shape operands are untrusted model data).
fn checked_dim(d: i64, context: &str) -> Result<usize, String> {
    usize::try_from(d).map_err(|_| format!("{context}: negative dim {d}"))
}

/// Validate a transpose permutation: a bijection over `0..rank`.
pub fn validate_perm(perm: &[i64], rank: usize) -> Result<Vec<usize>, String> {
    if perm.len() != rank {
        return Err("Transpose perm rank mismatch".into());
    }
    let mut seen = vec![false; rank];
    let mut out = Vec::with_capacity(rank);
    for &v in perm {
        let p = usize::try_from(v)
            .ok()
            .filter(|&p| p < rank)
            .ok_or_else(|| format!("Transpose perm entry {v} out of range for rank {rank}"))?;
        if seen[p] {
            return Err(format!("Transpose perm repeats axis {p}"));
        }
        seen[p] = true;
        out.push(p);
    }
    Ok(out)
}

/// NumPy broadcast of two shapes — the same rule the lowered graph's
/// `Broadcast` op enforces, so it is delegated to atlas-trace rather than
/// reimplemented (two copies of the rule could drift apart).
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    atlas_trace::tensor::broadcast_shape(a, b).map_err(|e| e.to_string())
}

/// Normalize a possibly negative axis.
pub fn norm_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

/// Promote dtypes for arithmetic (ONNX requires equal types; we allow
/// I32/I64 mixing since we widen payloads).
pub(crate) fn promote(a: DType, b: DType) -> Result<DType, String> {
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (DType::I32, DType::I64) | (DType::I64, DType::I32) => Ok(DType::I64),
        _ => Err(format!("mismatched operand types {a:?} vs {b:?}")),
    }
}
