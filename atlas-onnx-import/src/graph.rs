//! The importer's internal graph: SSA edges with interned names, nodes with
//! typed attributes, and constant tensors attached to edges.
//!
//! Nodes keep their ONNX `op_type` string through the pass pipeline (that is
//! the external format's own naming); rewrite passes introduce `atlas.*` ops,
//! and `verify_supported` is the single choke point guaranteeing only
//! lowerable ops remain.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use crate::{error::ImportError, proto};

/// Index of an SSA edge (a named value) in [`IGraph::edges`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeId(pub usize);

/// One SSA value.
#[derive(Clone, Debug)]
pub struct Edge {
    /// The ONNX value name (or a synthesized one for pass-created edges).
    pub name: String,
    /// Shape/type (filled by analysis; pass-created edges set it directly).
    pub info: Option<TensorInfo>,
    /// Constant payload, if the value is known at import time.
    pub konst: Option<ConstTensor>,
}

/// One operation.
#[derive(Clone, Debug)]
pub struct INode {
    /// The ONNX node name (synthesized when empty).
    pub name: String,
    /// The ONNX `op_type`, or an `atlas.*` target op introduced by a pass.
    pub op: String,
    /// Typed attributes.
    pub attrs: BTreeMap<String, Attr>,
    /// Operand edges.
    pub inputs: Vec<EdgeId>,
    /// Result edges (most ops have one; `Split` has several).
    pub outputs: Vec<EdgeId>,
}

impl INode {
    /// Integer attribute.
    pub fn attr_int(&self, name: &str) -> Option<i64> {
        match self.attrs.get(name)? {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Integer-list attribute.
    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match self.attrs.get(name)? {
            Attr::Ints(v) => Some(v),
            _ => None,
        }
    }

    /// Float attribute.
    pub fn attr_float(&self, name: &str) -> Option<f32> {
        match self.attrs.get(name)? {
            Attr::Float(v) => Some(*v),
            _ => None,
        }
    }
}

/// The importer graph. Nodes are tombstoned (`None`) when removed by a pass.
///
/// An edge→producer index is maintained by [`IGraph::push_node`] and
/// [`IGraph::remove_node`], making [`IGraph::producer`] O(1) — pass loops
/// call it per node, and a linear scan there made the pipeline quadratic on
/// transformer-scale graphs. The index requires that a live node's `outputs`
/// are never reassigned in place: replace the node (push + remove) instead.
#[derive(Debug, Default)]
pub struct IGraph {
    /// Nodes (tombstoned when removed).
    pub nodes: Vec<Option<INode>>,
    /// SSA values.
    pub edges: Vec<Edge>,
    /// Graph inputs.
    pub inputs: Vec<EdgeId>,
    /// Graph outputs.
    pub outputs: Vec<EdgeId>,
    names: HashMap<String, EdgeId>,
    producers: HashMap<EdgeId, usize>,
}

impl IGraph {
    /// The edge behind an id.
    pub fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id.0]
    }

    /// Mutable edge access.
    pub fn edge_mut(&mut self, id: EdgeId) -> &mut Edge {
        &mut self.edges[id.0]
    }

    /// The constant payload of an edge, if known.
    pub fn konst(&self, id: EdgeId) -> Option<&ConstTensor> {
        self.edges[id.0].konst.as_ref()
    }

    /// The shape/type of an edge (post-analysis).
    pub fn info(&self, id: EdgeId) -> Option<&TensorInfo> {
        self.edges[id.0].info.as_ref()
    }

    /// Static dims of an edge (post-analysis).
    pub fn dims(&self, id: EdgeId) -> Option<&[usize]> {
        self.edges[id.0].info.as_ref().map(|i| i.dims.as_slice())
    }

    /// The live node producing an edge (O(1) via the maintained index).
    pub fn producer(&self, id: EdgeId) -> Option<(usize, &INode)> {
        let &idx = self.producers.get(&id)?;
        self.nodes[idx].as_ref().map(|n| (idx, n))
    }

    /// How many live node inputs plus graph outputs reference the edge.
    /// One O(nodes) scan — for a per-edge query over many edges, build
    /// [`IGraph::use_counts`] once instead.
    pub fn use_count(&self, id: EdgeId) -> usize {
        self.nodes
            .iter()
            .flatten()
            .map(|n| n.inputs.iter().filter(|&&e| e == id).count())
            .sum::<usize>()
            + self.outputs.iter().filter(|&&e| e == id).count()
    }

    /// Use counts of every edge (indexed by `EdgeId`), computed in one scan.
    pub fn use_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.edges.len()];
        for node in self.nodes.iter().flatten() {
            for &e in &node.inputs {
                counts[e.0] += 1;
            }
        }
        for &e in &self.outputs {
            counts[e.0] += 1;
        }
        counts
    }

    /// Intern (or look up) an edge by name.
    pub fn edge_by_name(&mut self, name: &str) -> EdgeId {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = EdgeId(self.edges.len());
        self.edges.push(Edge {
            name: name.to_string(),
            info: None,
            konst: None,
        });
        self.names.insert(name.to_string(), id);
        id
    }

    /// Create a fresh edge with a unique synthesized name.
    pub fn fresh_edge(&mut self, hint: &str, info: TensorInfo) -> EdgeId {
        let mut name = format!("atlas::{hint}");
        let mut n = 0;
        while self.names.contains_key(&name) {
            n += 1;
            name = format!("atlas::{hint}#{n}");
        }
        let id = self.edge_by_name(&name);
        self.edges[id.0].info = Some(info);
        id
    }

    /// Append a node. The new node becomes the producer of its outputs
    /// (passes push a replacement before removing the node it supersedes).
    pub fn push_node(&mut self, node: INode) -> usize {
        let idx = self.nodes.len();
        for &o in &node.outputs {
            self.producers.insert(o, idx);
        }
        self.nodes.push(Some(node));
        idx
    }

    /// Remove a node, dropping its producer entries (unless a replacement
    /// node already took an output over).
    pub fn remove_node(&mut self, idx: usize) {
        if let Some(node) = self.nodes[idx].take() {
            for &o in &node.outputs {
                if self.producers.get(&o) == Some(&idx) {
                    self.producers.remove(&o);
                }
            }
        }
    }

    /// Live nodes with their indices.
    pub fn live_nodes(&self) -> impl Iterator<Item = (usize, &INode)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|n| (i, n)))
    }

    /// Redirect every use of `from` (node inputs and graph outputs) to `to`.
    pub fn redirect_uses(&mut self, from: EdgeId, to: EdgeId) {
        for node in self.nodes.iter_mut().flatten() {
            for input in &mut node.inputs {
                if *input == from {
                    *input = to;
                }
            }
        }
        for output in &mut self.outputs {
            if *output == from {
                *output = to;
            }
        }
    }

    /// A topological order over live nodes (Kahn's algorithm over edge
    /// dependencies). Errors on cycles or dangling operands.
    pub fn topo_order(&self) -> Result<Vec<usize>, ImportError> {
        let mut produced: HashMap<EdgeId, usize> = HashMap::new();
        for (i, n) in self.live_nodes() {
            for &o in &n.outputs {
                if produced.insert(o, i).is_some() {
                    return Err(ImportError::Malformed(format!(
                        "edge {} produced twice",
                        self.edge(o).name
                    )));
                }
            }
        }
        // An edge is "available" if it is a graph input, a constant, or
        // produced by an already-visited node.
        let mut pending: Vec<usize> = self.live_nodes().map(|(i, _)| i).collect();
        let mut done: Vec<bool> = vec![false; self.nodes.len()];
        let available = |g: &IGraph, done: &[bool], e: EdgeId| -> bool {
            g.konst(e).is_some()
                || g.inputs.contains(&e)
                || produced.get(&e).map(|&p| done[p]).unwrap_or(false)
        };
        let mut order = Vec::with_capacity(pending.len());
        while !pending.is_empty() {
            let before = pending.len();
            pending.retain(|&i| {
                let node = self.nodes[i].as_ref().expect("live");
                if node.inputs.iter().all(|&e| available(self, &done, e)) {
                    done[i] = true;
                    order.push(i);
                    false
                } else {
                    true
                }
            });
            if pending.len() == before {
                let stuck = self.nodes[pending[0]].as_ref().expect("live");
                return Err(ImportError::Malformed(format!(
                    "cycle or dangling operand at node {} ({})",
                    stuck.name, stuck.op
                )));
            }
        }
        Ok(order)
    }
}

/// Build the importer graph from the decoded model: intern all edges, attach
/// initializers as constants, resolve graph-input shapes against `bindings`.
pub fn from_proto(
    model: &proto::ModelProto,
    model_dir: &Path,
    bindings: &HashMap<String, usize>,
) -> Result<IGraph, ImportError> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| ImportError::Malformed("model has no graph".into()))?;

    let mut g = IGraph::default();

    // Initializers become constant edges.
    for init in &graph.initializer {
        let tensor = decode_tensor(init, model_dir)?;
        let id = g.edge_by_name(&init.name);
        g.edges[id.0].info = Some(TensorInfo {
            dtype: tensor.dtype,
            dims: tensor.dims.clone(),
        });
        g.edges[id.0].konst = Some(tensor);
    }

    // Graph inputs (skip initializer-shadowed ones, an ONNX legacy pattern).
    for input in &graph.input {
        let id = g.edge_by_name(&input.name);
        if g.edges[id.0].konst.is_some() {
            continue;
        }
        let (dtype, dims) = input_signature(input, bindings)?;
        g.edges[id.0].info = Some(TensorInfo { dtype, dims });
        g.inputs.push(id);
    }

    // Nodes. Output `value_info` shape annotations are deliberately ignored;
    // analysis recomputes everything.
    for (i, node) in graph.node.iter().enumerate() {
        let name = if node.name.is_empty() {
            format!("{}_{i}", node.op_type)
        } else {
            node.name.clone()
        };
        let mut attrs = BTreeMap::new();
        for a in &node.attribute {
            use proto::attribute_proto::AttributeType as AT;
            let ty = AT::try_from(a.r#type).unwrap_or(AT::Undefined);
            let value = match ty {
                AT::Int => Attr::Int(a.i),
                AT::Ints => Attr::Ints(a.ints.clone()),
                AT::Float => Attr::Float(a.f),
                AT::String => Attr::Str(String::from_utf8_lossy(&a.s).into_owned()),
                AT::Tensor => Attr::Tensor(decode_tensor(
                    a.t.as_ref().ok_or_else(|| {
                        ImportError::Malformed(format!("{name}: empty tensor attribute"))
                    })?,
                    model_dir,
                )?),
                AT::Graph | AT::Graphs => {
                    return Err(ImportError::UnsupportedFeature {
                        what: format!("subgraph attribute on {}", node.op_type),
                        context: name,
                    });
                }
                other => {
                    return Err(ImportError::UnsupportedFeature {
                        what: format!("attribute type {other:?}"),
                        context: name,
                    });
                }
            };
            attrs.insert(a.name.clone(), value);
        }
        // Optional trailing inputs/outputs are encoded as "" in ONNX; drop
        // them. An omitted *interior* operand (e.g. Slice without axes but
        // with steps) must not be dropped — later operands would silently
        // shift into its position — so it is rejected instead.
        let mut positional = |names: &[String], kind: &str| -> Result<Vec<EdgeId>, ImportError> {
            let mut names = names;
            while names.last().is_some_and(|n| n.is_empty()) {
                names = &names[..names.len() - 1];
            }
            if names.iter().any(|n| n.is_empty()) {
                return Err(ImportError::UnsupportedFeature {
                    what: format!("omitted optional {kind} before a supplied one"),
                    context: name.clone(),
                });
            }
            Ok(names.iter().map(|n| g.edge_by_name(n)).collect())
        };
        let inputs = positional(&node.input, "input")?;
        let outputs = positional(&node.output, "output")?;
        g.push_node(INode {
            name,
            op: node.op_type.clone(),
            attrs,
            inputs,
            outputs,
        });
    }

    for output in &graph.output {
        let id = g.edge_by_name(&output.name);
        g.outputs.push(id);
    }

    Ok(g)
}

/// Decode a `TensorProto` into a [`ConstTensor`].
pub fn decode_tensor(t: &proto::TensorProto, model_dir: &Path) -> Result<ConstTensor, ImportError> {
    use proto::tensor_proto::DataType;

    if t.segment.is_some() {
        return Err(ImportError::UnsupportedFeature {
            what: "TensorProto.segment".into(),
            context: t.name.clone(),
        });
    }
    // Model files are untrusted: dims are checked (no negative sizes, no
    // overflow in the element count, and a hard cap) before any allocation.
    let dims: Vec<usize> = t
        .dims
        .iter()
        .map(|&d| {
            usize::try_from(d)
                .map_err(|_| ImportError::Malformed(format!("tensor {}: negative dim {d}", t.name)))
        })
        .collect::<Result<_, _>>()?;
    let expected = checked_element_count(&dims)
        .ok_or_else(|| ImportError::Malformed(format!("tensor {}: dims overflow", t.name)))?;
    let mut raw = proto::tensor_payload(t, model_dir)?;
    let external = t.data_location == proto::tensor_proto::DataLocation::External as i32;

    let dtype = DataType::try_from(t.data_type).map_err(|_| ImportError::UnsupportedFeature {
        what: format!("tensor data_type {}", t.data_type),
        context: t.name.clone(),
    })?;
    let has_raw_data = !raw.is_empty() || external;
    if has_raw_data {
        validate_raw_payload(t, dtype, expected, &mut raw)?;
    }
    let (dtype, data) = match dtype {
        DataType::Float => {
            let v: Vec<f32> = if has_raw_data {
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk of 4")))
                    .collect()
            } else {
                t.float_data.clone()
            };
            (DType::F32, ConstData::F32(v))
        }
        DataType::Int64 => {
            let v: Vec<i64> = if has_raw_data {
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().expect("chunk of 8")))
                    .collect()
            } else {
                t.int64_data.clone()
            };
            (DType::I64, ConstData::I64(v))
        }
        DataType::Int32 => {
            let v: Vec<i64> = if has_raw_data {
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().expect("chunk of 4")) as i64)
                    .collect()
            } else {
                t.int32_data.iter().map(|&v| v as i64).collect()
            };
            (DType::I32, ConstData::I64(v))
        }
        DataType::Bool => {
            let v: Vec<bool> = if has_raw_data {
                raw.iter().map(|&b| b != 0).collect()
            } else {
                t.int32_data.iter().map(|&v| v != 0).collect()
            };
            (DType::Bool, ConstData::Bool(v))
        }
        other => {
            return Err(ImportError::UnsupportedFeature {
                what: format!("tensor data type {other:?} (re-export the model in fp32)"),
                context: t.name.clone(),
            });
        }
    };
    let tensor = ConstTensor { dtype, dims, data };
    if tensor.len() != expected {
        return Err(ImportError::Malformed(format!(
            "tensor {}: {} elements, dims {:?} expect {expected}",
            t.name,
            tensor.len(),
            tensor.dims
        )));
    }
    Ok(tensor)
}

/// Validate the size of a tensor's raw payload. External data without an
/// explicit length may share a file with later tensors, so keep only the
/// number of bytes described by this tensor's shape and type.
fn validate_raw_payload(
    tensor: &proto::TensorProto,
    dtype: proto::tensor_proto::DataType,
    expected_elements: usize,
    raw: &mut Vec<u8>,
) -> Result<(), ImportError> {
    use proto::tensor_proto::{DataLocation, DataType};

    let bytes_per_element = match dtype {
        DataType::Float | DataType::Int32 => 4,
        DataType::Int64 => 8,
        DataType::Bool => 1,
        _ => return Ok(()),
    };
    let expected_bytes = expected_elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| {
            ImportError::Malformed(format!("tensor {}: byte size overflow", tensor.name))
        })?;
    let external = tensor.data_location == DataLocation::External as i32;
    let external_without_length =
        external && !tensor.external_data.iter().any(|kv| kv.key == "length");

    if external_without_length {
        if raw.len() < expected_bytes {
            return Err(ImportError::Malformed(format!(
                "tensor {}: raw payload has {} bytes, dims {:?} and data type {:?} need {expected_bytes}",
                tensor.name,
                raw.len(),
                tensor.dims,
                dtype
            )));
        }
        raw.truncate(expected_bytes);
    } else if raw.len() != expected_bytes {
        return Err(ImportError::Malformed(format!(
            "tensor {}: raw payload has {} bytes, dims {:?} and data type {:?} expect {expected_bytes}",
            tensor.name,
            raw.len(),
            tensor.dims,
            dtype
        )));
    }
    Ok(())
}

/// Hard cap on the element count of any tensor the importer will represent:
/// far beyond anything provable, but small enough that no partial product of
/// capped dims (including power-of-two padding) can overflow `usize`.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 34;

/// The element type of a value.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DType {
    /// 32-bit float (quantized at lowering).
    F32,
    /// 64-bit int (indices, shapes, masks).
    I64,
    /// 32-bit int (kept raw at lowering).
    I32,
    /// Boolean.
    Bool,
}

/// A constant tensor's payload. Integer data is widened to `i64`; the
/// [`DType`] remembers the declared width.
#[derive(Clone, PartialEq, Debug)]
pub enum ConstData {
    /// Float payload.
    F32(Vec<f32>),
    /// Integer payload (i32 widened).
    I64(Vec<i64>),
    /// Boolean payload.
    Bool(Vec<bool>),
}

/// A constant tensor.
#[derive(Clone, PartialEq, Debug)]
pub struct ConstTensor {
    /// Declared element type.
    pub dtype: DType,
    /// Dimensions.
    pub dims: Vec<usize>,
    /// Payload.
    pub data: ConstData,
}

impl ConstTensor {
    /// Element count.
    pub fn len(&self) -> usize {
        match &self.data {
            ConstData::F32(v) => v.len(),
            ConstData::I64(v) => v.len(),
            ConstData::Bool(v) => v.len(),
        }
    }

    /// Whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The payload as i64 values, if integer or bool.
    pub fn as_i64s(&self) -> Option<&[i64]> {
        match &self.data {
            ConstData::I64(v) => Some(v),
            _ => None,
        }
    }

    /// The payload as f32 values, if float.
    pub fn as_f32s(&self) -> Option<&[f32]> {
        match &self.data {
            ConstData::F32(v) => Some(v),
            _ => None,
        }
    }

    /// A scalar (single-element) value as i64.
    pub fn scalar_i64(&self) -> Option<i64> {
        match (&self.data, self.len()) {
            (ConstData::I64(v), 1) => Some(v[0]),
            (ConstData::Bool(v), 1) => Some(v[0] as i64),
            _ => None,
        }
    }

    /// A scalar (single-element) value as f32.
    pub fn scalar_f32(&self) -> Option<f32> {
        match (&self.data, self.len()) {
            (ConstData::F32(v), 1) => Some(v[0]),
            _ => None,
        }
    }
}

/// Shape and type of a value, filled by shape inference.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorInfo {
    /// Element type.
    pub dtype: DType,
    /// Static dims.
    pub dims: Vec<usize>,
}

/// A typed attribute.
#[derive(Clone, PartialEq, Debug)]
pub enum Attr {
    /// Integer.
    Int(i64),
    /// Integer list.
    Ints(Vec<i64>),
    /// Float.
    Float(f32),
    /// String.
    Str(String),
    /// Tensor.
    Tensor(ConstTensor),
}

/// Checked product of dims with the [`MAX_TENSOR_ELEMENTS`] cap so untrusted
/// shape metadata can never drive an unbounded allocation. A zero dim makes
/// the count 0, but the remaining dims must still fit the cap (partial
/// products of a zero-element shape are computed all over the pipeline).
pub fn checked_element_count(dims: &[usize]) -> Option<usize> {
    let mut nonzero: usize = 1;
    let mut has_zero = false;
    for &d in dims {
        if d == 0 {
            has_zero = true;
            continue;
        }
        nonzero = nonzero.checked_mul(d)?;
        if nonzero > MAX_TENSOR_ELEMENTS {
            return None;
        }
    }
    Some(if has_zero { 0 } else { nonzero })
}

/// Resolve a symbolic dimension: a binding name, an integer literal, or a
/// `+`-separated sum of those (exporters put derived dims like
/// `"past_sequence_length + sequence_length"` on some inputs).
fn resolve_dim_expr(expr: &str, bindings: &HashMap<String, usize>) -> Option<usize> {
    expr.split('+')
        .map(str::trim)
        .map(|term| {
            term.parse::<usize>()
                .ok()
                .or_else(|| bindings.get(term).copied())
        })
        .sum()
}

/// Resolve a graph input's dtype and static dims, substituting `bindings`
/// for symbolic dims.
fn input_signature(
    input: &proto::ValueInfoProto,
    bindings: &HashMap<String, usize>,
) -> Result<(DType, Vec<usize>), ImportError> {
    use proto::type_proto::Value;

    let ty = input
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .ok_or_else(|| ImportError::Malformed(format!("input {} has no type", input.name)))?;
    let Value::TensorType(tensor) = ty else {
        return Err(ImportError::UnsupportedFeature {
            what: "non-tensor graph input".into(),
            context: input.name.clone(),
        });
    };
    let dtype = match proto::tensor_proto::DataType::try_from(tensor.elem_type) {
        Ok(proto::tensor_proto::DataType::Float) => DType::F32,
        Ok(proto::tensor_proto::DataType::Int64) => DType::I64,
        Ok(proto::tensor_proto::DataType::Int32) => DType::I32,
        Ok(proto::tensor_proto::DataType::Bool) => DType::Bool,
        other => {
            return Err(ImportError::UnsupportedFeature {
                what: format!("input element type {other:?}"),
                context: input.name.clone(),
            });
        }
    };
    let mut dims = Vec::new();
    for (i, d) in tensor
        .shape
        .as_ref()
        .map(|s| s.dim.as_slice())
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        use proto::tensor_shape_proto::dimension::Value as DimValue;
        let size = match &d.value {
            Some(DimValue::DimValue(v)) if *v > 0 => *v as usize,
            Some(DimValue::DimParam(p)) => {
                resolve_dim_expr(p, bindings).ok_or_else(|| ImportError::UnboundDim {
                    input: input.name.clone(),
                    dim: i,
                    dim_param: p.clone(),
                })?
            }
            _ => {
                return Err(ImportError::UnboundDim {
                    input: input.name.clone(),
                    dim: i,
                    dim_param: String::new(),
                });
            }
        };
        dims.push(size);
    }
    if checked_element_count(&dims).is_none() {
        return Err(ImportError::Malformed(format!(
            "input {}: shape {dims:?} exceeds the element cap",
            input.name
        )));
    }
    Ok((dtype, dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str, inputs: Vec<EdgeId>, outputs: Vec<EdgeId>) -> INode {
        INode {
            name: name.into(),
            op: "Identity".into(),
            attrs: BTreeMap::new(),
            inputs,
            outputs,
        }
    }

    #[test]
    fn producer_index_tracks_node_replacement() {
        let mut g = IGraph::default();
        let a = g.edge_by_name("a");
        let b = g.edge_by_name("b");
        let old = g.push_node(node("old", vec![a], vec![b]));
        assert_eq!(g.producer(b).map(|(i, _)| i), Some(old));
        // The pass replacement pattern: push the successor, then remove the
        // superseded node — the index must end up on the new producer.
        let new = g.push_node(node("new", vec![a], vec![b]));
        g.remove_node(old);
        assert_eq!(g.producer(b).map(|(i, _)| i), Some(new));
        g.remove_node(new);
        assert!(g.producer(b).is_none());
    }

    #[test]
    fn decode_tensor_rejects_raw_payload_with_trailing_bytes() {
        let tensor = proto::TensorProto {
            dims: vec![1],
            data_type: proto::tensor_proto::DataType::Float as i32,
            raw_data: vec![0; 5].into(),
            ..Default::default()
        };

        let error = decode_tensor(&tensor, Path::new("."))
            .expect_err("raw payload with a trailing byte must be rejected");
        assert!(error.to_string().contains("raw payload has 5 bytes"));
    }
}
