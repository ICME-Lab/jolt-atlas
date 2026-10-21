//! Whitelist constant folding: small, pure evaluators for exactly the ops
//! that appear in shape/mask-construction subgraphs. No generic interpreter.
//!
//! Folding computes in the declared dtype (f32 stays f32, matching what the
//! exporter's own constant folding would produce).

use crate::graph::{Attr, ConstData, ConstTensor, DType, IGraph, INode, checked_element_count};
use crate::shape_fns::{self, norm_axis};

/// Base materialization budget for a fold, in elements. Shape/mask subgraphs
/// are small; this admits an 8192x8192 mask.
const FOLD_BASE_ELEMENTS: usize = 1 << 26;

/// How many elements a fold of these inputs may materialize: the base
/// budget, or twice the input payload for folds over real weights
/// (transposed/concatenated initializers).
fn fold_budget<'a>(inputs: impl Iterator<Item = &'a ConstTensor>) -> usize {
    let input_elems: usize = inputs.map(|t| t.len()).sum();
    FOLD_BASE_ELEMENTS.max(input_elems.saturating_mul(2))
}

/// Checked element count of a fold result, against `budget`. Untrusted
/// shapes reach folding before the analysis-level cap, so every fold checks
/// its output size *before* allocating.
fn fold_numel(dims: &[usize], budget: usize) -> Result<usize, String> {
    checked_element_count(dims)
        .filter(|&n| n <= budget)
        .ok_or_else(|| format!("fold result shape {dims:?} exceeds the materialization budget"))
}

/// Row-major strides.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// Flat source index of broadcast element `flat` of `out_dims` in a tensor
/// of `src_dims` (right-aligned NumPy broadcast).
fn bcast_src_index(flat: usize, out_dims: &[usize], src_dims: &[usize]) -> usize {
    let out_strides = strides(out_dims);
    let offset = out_dims.len() - src_dims.len();
    let mut idx = 0;
    for (axis, &sd) in src_dims.iter().enumerate() {
        let coord = (flat / out_strides[axis + offset]) % out_dims[axis + offset];
        idx = idx * sd + if sd == 1 { 0 } else { coord };
    }
    idx
}

/// Broadcast-expand a constant to `dims`, within `budget` elements.
fn expand(t: &ConstTensor, dims: &[usize], budget: usize) -> Result<ConstTensor, String> {
    let total = fold_numel(dims, budget)?;
    let data = match &t.data {
        ConstData::F32(v) => ConstData::F32(
            (0..total)
                .map(|i| v[bcast_src_index(i, dims, &t.dims)])
                .collect(),
        ),
        ConstData::I64(v) => ConstData::I64(
            (0..total)
                .map(|i| v[bcast_src_index(i, dims, &t.dims)])
                .collect(),
        ),
        ConstData::Bool(v) => ConstData::Bool(
            (0..total)
                .map(|i| v[bcast_src_index(i, dims, &t.dims)])
                .collect(),
        ),
    };
    Ok(ConstTensor {
        dtype: t.dtype,
        dims: dims.to_vec(),
        data,
    })
}

/// Narrow an I32-labeled payload to int32 range (two's-complement wrap,
/// matching what onnxruntime's int32 arithmetic/casts compute). The importer
/// stores I32 widened to i64, so without this a fold could carry values no
/// int32 runtime ever produces.
fn wrap_i32(t: &mut ConstTensor) {
    if t.dtype == DType::I32
        && let ConstData::I64(v) = &mut t.data
    {
        for x in v {
            *x = *x as i32 as i64;
        }
    }
}

fn binary_numeric(
    a: &ConstTensor,
    b: &ConstTensor,
    ff: impl Fn(f32, f32) -> f32,
    fi: impl Fn(i64, i64) -> Result<i64, String>,
) -> Result<ConstTensor, String> {
    let dims = shape_fns::broadcast(&a.dims, &b.dims)?;
    let budget = fold_budget([a, b].into_iter());
    let (ea, eb) = (expand(a, &dims, budget)?, expand(b, &dims, budget)?);
    match (&ea.data, &eb.data) {
        (ConstData::F32(x), ConstData::F32(y)) => Ok(ConstTensor {
            dtype: DType::F32,
            dims,
            data: ConstData::F32(x.iter().zip(y).map(|(&p, &q)| ff(p, q)).collect()),
        }),
        (ConstData::I64(x), ConstData::I64(y)) => {
            let v: Result<Vec<i64>, String> = x.iter().zip(y).map(|(&p, &q)| fi(p, q)).collect();
            let mut t = ConstTensor {
                dtype: if a.dtype == DType::I32 && b.dtype == DType::I32 {
                    DType::I32
                } else {
                    DType::I64
                },
                dims,
                data: ConstData::I64(v?),
            };
            wrap_i32(&mut t);
            Ok(t)
        }
        _ => Err("mixed/unsupported operand types in constant fold".into()),
    }
}

fn compare(
    a: &ConstTensor,
    b: &ConstTensor,
    ff: impl Fn(f32, f32) -> bool,
    fi: impl Fn(i64, i64) -> bool,
) -> Result<ConstTensor, String> {
    let dims = shape_fns::broadcast(&a.dims, &b.dims)?;
    let budget = fold_budget([a, b].into_iter());
    let (ea, eb) = (expand(a, &dims, budget)?, expand(b, &dims, budget)?);
    let data = match (&ea.data, &eb.data) {
        (ConstData::F32(x), ConstData::F32(y)) => {
            x.iter().zip(y).map(|(&p, &q)| ff(p, q)).collect()
        }
        (ConstData::I64(x), ConstData::I64(y)) => {
            x.iter().zip(y).map(|(&p, &q)| fi(p, q)).collect()
        }
        (ConstData::Bool(x), ConstData::Bool(y)) => x
            .iter()
            .zip(y)
            .map(|(&p, &q)| fi(p as i64, q as i64))
            .collect(),
        _ => return Err("mixed operand types in comparison fold".into()),
    };
    Ok(ConstTensor {
        dtype: DType::Bool,
        dims,
        data: ConstData::Bool(data),
    })
}

fn unary_f32(t: &ConstTensor, f: impl Fn(f32) -> f32) -> Result<ConstTensor, String> {
    let v = t.as_f32s().ok_or("expected float operand")?;
    Ok(ConstTensor {
        dtype: DType::F32,
        dims: t.dims.clone(),
        data: ConstData::F32(v.iter().map(|&x| f(x)).collect()),
    })
}

/// Try to fold `node`. `Ok(None)` means "not a foldable op" (caller falls
/// back to shape inference); `Err` means the fold ran and failed.
///
/// Precondition (checked by the caller): every input the op needs folded is
/// constant. `Shape` only needs its input's dims.
pub fn fold(node: &INode, g: &IGraph) -> Result<Option<Vec<ConstTensor>>, String> {
    let k = |i: usize| -> Result<&ConstTensor, String> {
        node.inputs
            .get(i)
            .and_then(|&e| g.konst(e))
            .ok_or_else(|| format!("input {i} not constant"))
    };
    let out = |t: ConstTensor| Ok(Some(vec![t]));

    match node.op.as_str() {
        "Shape" => {
            let dims = g
                .dims(*node.inputs.first().ok_or("Shape missing input")?)
                .ok_or("Shape input has no inferred dims")?;
            // Opset 15 Shape takes start/end attributes and returns
            // dims[start..end]; ignoring them would fold the wrong shape.
            let (start, end) = shape_fns::shape_slice_bounds(node, dims.len());
            let window = &dims[start..end.max(start)];
            out(ConstTensor {
                dtype: DType::I64,
                dims: vec![window.len()],
                data: ConstData::I64(window.iter().map(|&d| d as i64).collect()),
            })
        }
        "Identity" => out(k(0)?.clone()),
        "Neg" => {
            let t = k(0)?;
            match &t.data {
                ConstData::F32(v) => out(ConstTensor {
                    dtype: t.dtype,
                    dims: t.dims.clone(),
                    data: ConstData::F32(v.iter().map(|&x| -x).collect()),
                }),
                ConstData::I64(v) => {
                    let mut r = ConstTensor {
                        dtype: t.dtype,
                        dims: t.dims.clone(),
                        data: ConstData::I64(
                            v.iter()
                                .map(|&x| x.checked_neg().ok_or("i64 overflow in Neg fold"))
                                .collect::<Result<_, _>>()?,
                        ),
                    };
                    wrap_i32(&mut r);
                    out(r)
                }
                ConstData::Bool(_) => Err("Neg of bool".into()),
            }
        }
        "Sqrt" => out(unary_f32(k(0)?, f32::sqrt)?),
        "Reciprocal" => out(unary_f32(k(0)?, |x| 1.0 / x)?),
        "Cast" => {
            let t = k(0)?;
            let to = node.attr_int("to").ok_or("Cast missing to")?;
            use crate::proto::tensor_proto::DataType;
            let target = DataType::try_from(to as i32).map_err(|_| "bad Cast target")?;
            let cast = |t: &ConstTensor| -> Result<ConstTensor, String> {
                let dims = t.dims.clone();
                Ok(match target {
                    DataType::Float => ConstTensor {
                        dtype: DType::F32,
                        dims,
                        data: ConstData::F32(match &t.data {
                            ConstData::F32(v) => v.clone(),
                            ConstData::I64(v) => v.iter().map(|&x| x as f32).collect(),
                            ConstData::Bool(v) => v.iter().map(|&x| x as u8 as f32).collect(),
                        }),
                    },
                    DataType::Int64 | DataType::Int32 => {
                        // Int32 results are narrowed with two's-complement
                        // wrap (C static_cast, what onnxruntime computes).
                        // Non-finite float→int is undefined in ONNX and
                        // runtime-dependent, so refuse rather than guess.
                        let narrow = target == DataType::Int32;
                        let vals: Vec<i64> = match &t.data {
                            ConstData::F32(v) => v
                                .iter()
                                .map(|&x| {
                                    if !x.is_finite() {
                                        return Err(format!(
                                            "Cast of non-finite {x} to int in fold"
                                        ));
                                    }
                                    Ok(if narrow { x as i32 as i64 } else { x as i64 })
                                })
                                .collect::<Result<_, _>>()?,
                            ConstData::I64(v) => {
                                if narrow {
                                    v.iter().map(|&x| x as i32 as i64).collect()
                                } else {
                                    v.clone()
                                }
                            }
                            ConstData::Bool(v) => v.iter().map(|&x| x as i64).collect(),
                        };
                        ConstTensor {
                            dtype: if narrow { DType::I32 } else { DType::I64 },
                            dims,
                            data: ConstData::I64(vals),
                        }
                    }
                    DataType::Bool => ConstTensor {
                        dtype: DType::Bool,
                        dims,
                        data: ConstData::Bool(match &t.data {
                            ConstData::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
                            ConstData::I64(v) => v.iter().map(|&x| x != 0).collect(),
                            ConstData::Bool(v) => v.clone(),
                        }),
                    },
                    other => return Err(format!("unsupported Cast target {other:?}")),
                })
            };
            out(cast(t)?)
        }
        // Integer folds use checked arithmetic: a wrapped constant would
        // silently change the model's semantics.
        "Add" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a + b,
            |a, b| {
                a.checked_add(b)
                    .ok_or_else(|| "i64 overflow in Add fold".into())
            },
        )?),
        "Sub" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a - b,
            |a, b| {
                a.checked_sub(b)
                    .ok_or_else(|| "i64 overflow in Sub fold".into())
            },
        )?),
        "Mul" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a * b,
            |a, b| {
                a.checked_mul(b)
                    .ok_or_else(|| "i64 overflow in Mul fold".into())
            },
        )?),
        "Div" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a / b,
            // ONNX integer division truncates toward zero (C semantics).
            |a, b| {
                if b == 0 {
                    Err("integer division by zero in constant fold".into())
                } else {
                    Ok(a / b)
                }
            },
        )?),
        "Pow" => {
            // ONNX Pow's result type is the base type, not the promotion.
            let mut r = binary_numeric(k(0)?, k(1)?, f32::powf, |a, b| {
                u32::try_from(b)
                    .ok()
                    .and_then(|e| a.checked_pow(e))
                    .ok_or_else(|| "integer Pow out of range".to_string())
            })?;
            r.dtype = k(0)?.dtype;
            wrap_i32(&mut r);
            out(r)
        }
        "Equal" => out(compare(k(0)?, k(1)?, |a, b| a == b, |a, b| a == b)?),
        "LessOrEqual" => out(compare(k(0)?, k(1)?, |a, b| a <= b, |a, b| a <= b)?),
        "Less" => out(compare(k(0)?, k(1)?, |a, b| a < b, |a, b| a < b)?),
        "Greater" => out(compare(k(0)?, k(1)?, |a, b| a > b, |a, b| a > b)?),
        "And" => {
            let dims = shape_fns::broadcast(&k(0)?.dims, &k(1)?.dims)?;
            let budget = fold_budget([k(0)?, k(1)?].into_iter());
            let (a, b) = (expand(k(0)?, &dims, budget)?, expand(k(1)?, &dims, budget)?);
            match (&a.data, &b.data) {
                (ConstData::Bool(x), ConstData::Bool(y)) => out(ConstTensor {
                    dtype: DType::Bool,
                    dims,
                    data: ConstData::Bool(x.iter().zip(y).map(|(&p, &q)| p && q).collect()),
                }),
                _ => Err("And expects booleans".into()),
            }
        }
        "Where" => {
            let (c, a, b) = (k(0)?, k(1)?, k(2)?);
            let dims = shape_fns::broadcast(&shape_fns::broadcast(&c.dims, &a.dims)?, &b.dims)?;
            let budget = fold_budget([c, a, b].into_iter());
            let (ec, ea, eb) = (
                expand(c, &dims, budget)?,
                expand(a, &dims, budget)?,
                expand(b, &dims, budget)?,
            );
            let ConstData::Bool(cond) = &ec.data else {
                return Err("Where condition must be bool".into());
            };
            let data = match (&ea.data, &eb.data) {
                (ConstData::F32(x), ConstData::F32(y)) => ConstData::F32(
                    cond.iter()
                        .zip(x.iter().zip(y))
                        .map(|(&c, (&p, &q))| if c { p } else { q })
                        .collect(),
                ),
                (ConstData::I64(x), ConstData::I64(y)) => ConstData::I64(
                    cond.iter()
                        .zip(x.iter().zip(y))
                        .map(|(&c, (&p, &q))| if c { p } else { q })
                        .collect(),
                ),
                (ConstData::Bool(x), ConstData::Bool(y)) => ConstData::Bool(
                    cond.iter()
                        .zip(x.iter().zip(y))
                        .map(|(&c, (&p, &q))| if c { p } else { q })
                        .collect(),
                ),
                _ => return Err("Where branch types disagree".into()),
            };
            out(ConstTensor {
                // Match shape inference's dtype promotion (I32 + I64 → I64),
                // not whichever branch happens to be X.
                dtype: shape_fns::promote(ea.dtype, eb.dtype)?,
                dims,
                data,
            })
        }
        // Pure reinterpretations: data unchanged, dims from shape inference.
        "Reshape" | "Squeeze" | "Unsqueeze" | "Flatten" => {
            let infos = shape_fns::infer(node, g)?;
            let mut t = k(0)?.clone();
            let n = checked_element_count(&infos[0].dims).ok_or("reinterpreted shape overflows")?;
            if n != t.len() {
                return Err(format!(
                    "{} fold would change the element count: {:?} holds {} elements",
                    node.op,
                    infos[0].dims,
                    t.len()
                ));
            }
            t.dims = infos[0].dims.clone();
            out(t)
        }
        "Transpose" => {
            let t = k(0)?;
            let perm: Vec<usize> = match node.attr_ints("perm") {
                Some(p) => shape_fns::validate_perm(p, t.dims.len())?,
                None => (0..t.dims.len()).rev().collect(),
            };
            let out_dims: Vec<usize> = perm.iter().map(|&p| t.dims[p]).collect();
            let src_strides = strides(&t.dims);
            let out_strides = strides(&out_dims);
            // A valid perm preserves the element count, so this is t.len().
            let total = t.len();
            let map_idx = |flat: usize| -> usize {
                let mut src = 0;
                for (axis, &od) in out_dims.iter().enumerate() {
                    let coord = (flat / out_strides[axis]) % od;
                    src += coord * src_strides[perm[axis]];
                }
                src
            };
            let data = match &t.data {
                ConstData::F32(v) => ConstData::F32((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::I64(v) => ConstData::I64((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::Bool(v) => ConstData::Bool((0..total).map(|i| v[map_idx(i)]).collect()),
            };
            out(ConstTensor {
                dtype: t.dtype,
                dims: out_dims,
                data,
            })
        }
        "Concat" => {
            let first = k(0)?;
            let axis = norm_axis(
                node.attr_int("axis").ok_or("missing axis")?,
                first.dims.len(),
            )?;
            let parts: Vec<&ConstTensor> =
                (0..node.inputs.len()).map(k).collect::<Result<_, _>>()?;
            for p in &parts {
                let compatible = p.dtype == first.dtype
                    && p.dims.len() == first.dims.len()
                    && p.dims
                        .iter()
                        .zip(&first.dims)
                        .enumerate()
                        .all(|(i, (a, b))| i == axis || a == b);
                if !compatible {
                    return Err(format!(
                        "Concat shape mismatch: {:?} vs {:?} on axis {axis}",
                        p.dims, first.dims
                    ));
                }
            }
            let mut dims = first.dims.clone();
            dims[axis] = parts.iter().map(|p| p.dims[axis]).sum();
            let outer: usize = first.dims[..axis].iter().product();
            macro_rules! cat {
                ($variant:ident) => {{
                    let mut data = Vec::new();
                    for o in 0..outer {
                        for p in &parts {
                            let inner: usize = p.dims[axis..].iter().product();
                            let ConstData::$variant(v) = &p.data else {
                                return Err("mixed concat types".into());
                            };
                            data.extend_from_slice(&v[o * inner..(o + 1) * inner]);
                        }
                    }
                    ConstData::$variant(data)
                }};
            }
            let data = match &first.data {
                ConstData::F32(_) => cat!(F32),
                ConstData::I64(_) => cat!(I64),
                ConstData::Bool(_) => cat!(Bool),
            };
            out(ConstTensor {
                dtype: first.dtype,
                dims,
                data,
            })
        }
        "Slice" => {
            let t = k(0)?;
            let (axes, starts, ends, _, out_dims) = shape_fns::slice_params(node, &t.dims)?;
            // Gather element-by-element through coordinate mapping.
            let out_strides = strides(&out_dims);
            let src_strides = strides(&t.dims);
            let total: usize = out_dims.iter().product();
            let map_idx = |flat: usize| -> usize {
                let mut src = 0;
                for (axis, &od) in out_dims.iter().enumerate() {
                    let mut coord = (flat / out_strides[axis]) % od;
                    if let Some(pos) = axes.iter().position(|&a| a == axis) {
                        coord += starts[pos] as usize;
                        let _ = ends;
                    }
                    src += coord * src_strides[axis];
                }
                src
            };
            let data = match &t.data {
                ConstData::F32(v) => ConstData::F32((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::I64(v) => ConstData::I64((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::Bool(v) => ConstData::Bool((0..total).map(|i| v[map_idx(i)]).collect()),
            };
            out(ConstTensor {
                dtype: t.dtype,
                dims: out_dims,
                data,
            })
        }
        "Gather" => {
            let (data_t, idx_t) = (k(0)?, k(1)?);
            let axis = norm_axis(node.attr_int("axis").unwrap_or(0), data_t.dims.len())?;
            let indices = idx_t.as_i64s().ok_or("Gather indices must be integer")?;
            let axis_size = data_t.dims[axis] as i64;
            let resolved: Vec<usize> = indices
                .iter()
                .map(|&i| {
                    let i = if i < 0 { i + axis_size } else { i };
                    if i < 0 || i >= axis_size {
                        Err(format!("Gather index {i} out of range {axis_size}"))
                    } else {
                        Ok(i as usize)
                    }
                })
                .collect::<Result<_, _>>()?;
            let outer: usize = data_t.dims[..axis].iter().product();
            let inner: usize = data_t.dims[axis + 1..].iter().product();
            let mut dims = data_t.dims[..axis].to_vec();
            dims.extend_from_slice(&idx_t.dims);
            dims.extend_from_slice(&data_t.dims[axis + 1..]);
            fold_numel(&dims, fold_budget([data_t, idx_t].into_iter()))?;
            macro_rules! gat {
                ($v:expr, $variant:ident) => {{
                    let mut data = Vec::with_capacity(outer * resolved.len() * inner);
                    for o in 0..outer {
                        for &ix in &resolved {
                            let base = (o * data_t.dims[axis] + ix) * inner;
                            data.extend_from_slice(&$v[base..base + inner]);
                        }
                    }
                    ConstData::$variant(data)
                }};
            }
            let data = match &data_t.data {
                ConstData::F32(v) => gat!(v, F32),
                ConstData::I64(v) => gat!(v, I64),
                ConstData::Bool(v) => gat!(v, Bool),
            };
            out(ConstTensor {
                dtype: data_t.dtype,
                dims,
                data,
            })
        }
        "Expand" => {
            let t = k(0)?;
            let shape: Vec<usize> = k(1)?
                .as_i64s()
                .ok_or("Expand shape must be integer")?
                .iter()
                .map(|&d| usize::try_from(d).map_err(|_| format!("Expand: negative dim {d}")))
                .collect::<Result<_, _>>()?;
            let dims = shape_fns::broadcast(&t.dims, &shape)?;
            let budget = fold_budget([t].into_iter());
            out(expand(t, &dims, budget)?)
        }
        "Tile" => {
            let t = k(0)?;
            let repeats: Vec<usize> = k(1)?
                .as_i64s()
                .ok_or("Tile repeats must be integer")?
                .iter()
                .map(|&r| usize::try_from(r).map_err(|_| format!("Tile: negative repeat {r}")))
                .collect::<Result<_, _>>()?;
            if repeats.len() != t.dims.len() {
                return Err("Tile repeats rank mismatch".into());
            }
            let out_dims: Vec<usize> = t
                .dims
                .iter()
                .zip(&repeats)
                .map(|(&d, &r)| {
                    d.checked_mul(r)
                        .ok_or_else(|| format!("Tile dim {d} x {r} overflows"))
                })
                .collect::<Result<_, _>>()?;
            let total = fold_numel(&out_dims, fold_budget([t].into_iter()))?;
            let out_strides = strides(&out_dims);
            let src_strides = strides(&t.dims);
            let map_idx = |flat: usize| -> usize {
                let mut src = 0;
                for (axis, &od) in out_dims.iter().enumerate() {
                    let coord = (flat / out_strides[axis]) % od % t.dims[axis];
                    src += coord * src_strides[axis];
                }
                src
            };
            let data = match &t.data {
                ConstData::F32(v) => ConstData::F32((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::I64(v) => ConstData::I64((0..total).map(|i| v[map_idx(i)]).collect()),
                ConstData::Bool(v) => ConstData::Bool((0..total).map(|i| v[map_idx(i)]).collect()),
            };
            out(ConstTensor {
                dtype: t.dtype,
                dims: out_dims,
                data,
            })
        }
        "ConstantOfShape" => {
            let shape: Vec<usize> = k(0)?
                .as_i64s()
                .ok_or("ConstantOfShape input must be integer")?
                .iter()
                .map(|&d| {
                    usize::try_from(d).map_err(|_| format!("ConstantOfShape: negative dim {d}"))
                })
                .collect::<Result<_, _>>()?;
            let total = fold_numel(&shape, FOLD_BASE_ELEMENTS)?;
            let fill = match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => t.clone(),
                _ => ConstTensor {
                    dtype: DType::F32,
                    dims: vec![1],
                    data: ConstData::F32(vec![0.0]),
                },
            };
            let data = match &fill.data {
                ConstData::F32(v) => ConstData::F32(vec![v[0]; total]),
                ConstData::I64(v) => ConstData::I64(vec![v[0]; total]),
                ConstData::Bool(v) => ConstData::Bool(vec![v[0]; total]),
            };
            out(ConstTensor {
                dtype: fill.dtype,
                dims: shape,
                data,
            })
        }
        "Range" => {
            let (s, l, d) = (k(0)?, k(1)?, k(2)?);
            match (&s.data, &l.data, &d.data) {
                (ConstData::I64(s), ConstData::I64(l), ConstData::I64(d)) => {
                    let (s, l, d) = (s[0], l[0], d[0]);
                    if d == 0 {
                        return Err("Range delta is zero".into());
                    }
                    // ONNX: max(ceil((limit - start) / delta), 0), which is
                    // non-empty for a negative delta with limit < start. The
                    // operands are untrusted i64s, so count in i128.
                    let (diff, delta) = (l as i128 - s as i128, d as i128);
                    let n = if diff != 0 && (diff > 0) == (delta > 0) {
                        ((diff.abs() + delta.abs() - 1) / delta.abs()) as u64 as usize
                    } else {
                        0
                    };
                    fold_numel(&[n], FOLD_BASE_ELEMENTS)?;
                    // Every element lies between start and limit, so the
                    // i128 intermediate always fits back into i64.
                    let data = (0..n)
                        .map(|i| (s as i128 + i as i128 * delta) as i64)
                        .collect();
                    out(ConstTensor {
                        dtype: DType::I64,
                        dims: vec![n],
                        data: ConstData::I64(data),
                    })
                }
                (ConstData::F32(s), ConstData::F32(l), ConstData::F32(d)) => {
                    let (s, l, d) = (s[0], l[0], d[0]);
                    let n = (((l - s) / d).ceil()).max(0.0) as usize;
                    fold_numel(&[n], FOLD_BASE_ELEMENTS)?;
                    out(ConstTensor {
                        dtype: DType::F32,
                        dims: vec![n],
                        data: ConstData::F32((0..n).map(|i| s + i as f32 * d).collect()),
                    })
                }
                _ => Err("Range operands must share a numeric type".into()),
            }
        }
        "NonZero" => {
            let t = k(0)?;
            let rank = t.dims.len().max(1);
            let nonzero: Vec<usize> = match &t.data {
                ConstData::F32(v) => (0..v.len()).filter(|&i| v[i] != 0.0).collect(),
                ConstData::I64(v) => (0..v.len()).filter(|&i| v[i] != 0).collect(),
                ConstData::Bool(v) => (0..v.len()).filter(|&i| v[i]).collect(),
            };
            let st = strides(&t.dims);
            // Output is [rank, n]: coordinate matrix.
            let mut data = vec![0i64; rank * nonzero.len()];
            for (col, &flat) in nonzero.iter().enumerate() {
                for axis in 0..t.dims.len() {
                    data[axis * nonzero.len() + col] = ((flat / st[axis]) % t.dims[axis]) as i64;
                }
            }
            out(ConstTensor {
                dtype: DType::I64,
                dims: vec![rank, nonzero.len()],
                data: ConstData::I64(data),
            })
        }
        _ => Ok(None),
    }
}

/// Whether `op` is in the fold whitelist (used by analysis to decide whether
/// to attempt folding when all needed operands are constant).
pub fn foldable(op: &str) -> bool {
    matches!(
        op,
        "Shape"
            | "Identity"
            | "Neg"
            | "Sqrt"
            | "Reciprocal"
            | "Cast"
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Pow"
            | "Equal"
            | "LessOrEqual"
            | "Less"
            | "Greater"
            | "And"
            | "Where"
            | "Reshape"
            | "Squeeze"
            | "Unsqueeze"
            | "Flatten"
            | "Transpose"
            | "Concat"
            | "Slice"
            | "Gather"
            | "Expand"
            | "Tile"
            | "ConstantOfShape"
            | "Range"
            | "NonZero"
    )
}
