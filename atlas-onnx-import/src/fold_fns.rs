//! Whitelist constant folding: small, pure evaluators for exactly the ops
//! that appear in shape/mask-construction subgraphs. No generic interpreter.
//!
//! Folding computes in the declared dtype (f32 stays f32, matching what the
//! exporter's own constant folding would produce).

use crate::graph::{Attr, ConstData, ConstTensor, DType, IGraph, INode};
use crate::shape_fns::{self, norm_axis};

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

/// Broadcast-expand a constant to `dims`.
fn expand(t: &ConstTensor, dims: &[usize]) -> ConstTensor {
    let total: usize = dims.iter().product();
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
    ConstTensor {
        dtype: t.dtype,
        dims: dims.to_vec(),
        data,
    }
}

fn binary_numeric(
    a: &ConstTensor,
    b: &ConstTensor,
    ff: impl Fn(f32, f32) -> f32,
    fi: impl Fn(i64, i64) -> Result<i64, String>,
) -> Result<ConstTensor, String> {
    let dims = shape_fns::broadcast(&a.dims, &b.dims)?;
    let (ea, eb) = (expand(a, &dims), expand(b, &dims));
    match (&ea.data, &eb.data) {
        (ConstData::F32(x), ConstData::F32(y)) => Ok(ConstTensor {
            dtype: DType::F32,
            dims,
            data: ConstData::F32(x.iter().zip(y).map(|(&p, &q)| ff(p, q)).collect()),
        }),
        (ConstData::I64(x), ConstData::I64(y)) => {
            let v: Result<Vec<i64>, String> = x.iter().zip(y).map(|(&p, &q)| fi(p, q)).collect();
            Ok(ConstTensor {
                dtype: if a.dtype == DType::I32 && b.dtype == DType::I32 {
                    DType::I32
                } else {
                    DType::I64
                },
                dims,
                data: ConstData::I64(v?),
            })
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
    let (ea, eb) = (expand(a, &dims), expand(b, &dims));
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
                .dims(node.inputs[0])
                .ok_or("Shape input has no inferred dims")?;
            out(ConstTensor {
                dtype: DType::I64,
                dims: vec![dims.len()],
                data: ConstData::I64(dims.iter().map(|&d| d as i64).collect()),
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
                ConstData::I64(v) => out(ConstTensor {
                    dtype: t.dtype,
                    dims: t.dims.clone(),
                    data: ConstData::I64(v.iter().map(|&x| -x).collect()),
                }),
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
                    DataType::Int64 | DataType::Int32 => ConstTensor {
                        dtype: if target == DataType::Int64 {
                            DType::I64
                        } else {
                            DType::I32
                        },
                        dims,
                        data: ConstData::I64(match &t.data {
                            ConstData::F32(v) => v.iter().map(|&x| x as i64).collect(),
                            ConstData::I64(v) => v.clone(),
                            ConstData::Bool(v) => v.iter().map(|&x| x as i64).collect(),
                        }),
                    },
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
        "Add" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a + b,
            |a, b| Ok(a + b),
        )?),
        "Sub" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a - b,
            |a, b| Ok(a - b),
        )?),
        "Mul" => out(binary_numeric(
            k(0)?,
            k(1)?,
            |a, b| a * b,
            |a, b| Ok(a * b),
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
        "Pow" => out(binary_numeric(k(0)?, k(1)?, f32::powf, |a, b| {
            u32::try_from(b)
                .map(|e| a.pow(e))
                .map_err(|_| "negative integer exponent".to_string())
        })?),
        "Equal" => out(compare(k(0)?, k(1)?, |a, b| a == b, |a, b| a == b)?),
        "LessOrEqual" => out(compare(k(0)?, k(1)?, |a, b| a <= b, |a, b| a <= b)?),
        "Less" => out(compare(k(0)?, k(1)?, |a, b| a < b, |a, b| a < b)?),
        "Greater" => out(compare(k(0)?, k(1)?, |a, b| a > b, |a, b| a > b)?),
        "And" => {
            let dims = shape_fns::broadcast(&k(0)?.dims, &k(1)?.dims)?;
            let (a, b) = (expand(k(0)?, &dims), expand(k(1)?, &dims));
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
            let (ec, ea, eb) = (expand(c, &dims), expand(a, &dims), expand(b, &dims));
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
                dtype: ea.dtype,
                dims,
                data,
            })
        }
        // Pure reinterpretations: data unchanged, dims from shape inference.
        "Reshape" | "Squeeze" | "Unsqueeze" | "Flatten" => {
            let infos = shape_fns::infer(node, g)?;
            let mut t = k(0)?.clone();
            t.dims = infos[0].dims.clone();
            out(t)
        }
        "Transpose" => {
            let t = k(0)?;
            let perm: Vec<usize> = match node.attr_ints("perm") {
                Some(p) => p.iter().map(|&v| v as usize).collect(),
                None => (0..t.dims.len()).rev().collect(),
            };
            let out_dims: Vec<usize> = perm.iter().map(|&p| t.dims[p]).collect();
            let src_strides = strides(&t.dims);
            let out_strides = strides(&out_dims);
            let total: usize = out_dims.len().pow(0).max(out_dims.iter().product());
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
                .map(|&d| d as usize)
                .collect();
            let dims = shape_fns::broadcast(&t.dims, &shape)?;
            out(expand(t, &dims))
        }
        "Tile" => {
            let t = k(0)?;
            let repeats: Vec<usize> = k(1)?
                .as_i64s()
                .ok_or("Tile repeats must be integer")?
                .iter()
                .map(|&r| r as usize)
                .collect();
            let out_dims: Vec<usize> = t.dims.iter().zip(&repeats).map(|(&d, &r)| d * r).collect();
            let out_strides = strides(&out_dims);
            let src_strides = strides(&t.dims);
            let total: usize = out_dims.iter().product();
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
                .map(|&d| d as usize)
                .collect();
            let total: usize = shape.iter().product();
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
                    let n = ((l - s).max(0) as f64 / d as f64).ceil() as usize;
                    out(ConstTensor {
                        dtype: DType::I64,
                        dims: vec![n],
                        data: ConstData::I64((0..n).map(|i| s + i as i64 * d).collect()),
                    })
                }
                (ConstData::F32(s), ConstData::F32(l), ConstData::F32(d)) => {
                    let (s, l, d) = (s[0], l[0], d[0]);
                    let n = (((l - s) / d).ceil()).max(0.0) as usize;
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
