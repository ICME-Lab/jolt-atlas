//! Lowering: the analyzed, rewritten importer graph → `atlas_trace::Graph`.
//!
//! This is the single choke point: every surviving node must map 1:1 onto
//! the atlas op set or the import fails with a typed error. It also:
//! - quantizes constants (floats scale by `2^s`, integers stay raw);
//! - inserts explicit `Broadcast` nodes so every elementwise op receives
//!   operands already at its output shape (the prover's assumption);
//! - gates einsum equations to the canonical set (full-letter equations
//!   only — no dropped letters, no size-1 reinterpretation);
//! - splits multi-axis ONNX slices into chains of single-axis slices.

use std::collections::HashMap;

use atlas_trace::{Graph, Node, NodeId, Scale, Tensor, op, op::Op};

use crate::{
    error::ImportError,
    graph::{EdgeId, IGraph, INode},
    inputs::InputDescriptor,
    quantize::quantize_const,
};

/// Einsum equations the proof layer supports. The matmul pass only emits
/// these; the gate makes that a checked invariant rather than a convention.
const CANONICAL_EINSUMS: &[&str] = &[
    "mk,kn->mn",
    "mk,nk->mn",
    "amk,akn->amn",
    "amk,ank->amn",
    "abmk,abkn->abmn",
    "abmk,abnk->abmn",
];

/// Gather dictionaries up to this many (power-of-two-rounded) rows prove
/// with the small-table path; larger ones use the large path. Old-crate
/// parity (`node/handlers/index.rs`).
const GATHER_SMALL_LIMIT: usize = 65_536;

/// The lowering result.
pub struct Lowered {
    /// The atlas-trace graph (uncompiled).
    pub graph: Graph,
    /// How to quantize each input, in graph-input order.
    pub input_descriptors: Vec<InputDescriptor>,
}

/// Scalars are rank-1 `[1]` in the atlas IR.
fn lower_dims(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        vec![1]
    } else {
        dims.to_vec()
    }
}

struct Lowerer<'g> {
    g: &'g IGraph,
    scale: Scale,
    nodes: Vec<Node>,
    edge_to_node: HashMap<EdgeId, NodeId>,
    inputs: Vec<NodeId>,
}

impl<'g> Lowerer<'g> {
    fn push(&mut self, op: Op, inputs: Vec<NodeId>, output_dims: Vec<usize>) -> NodeId {
        self.nodes.push(Node {
            op,
            inputs,
            output_dims,
        });
        NodeId(self.nodes.len() as u32 - 1)
    }

    /// The atlas node holding an edge's value, materializing constants on
    /// first use.
    fn resolve(&mut self, edge: EdgeId) -> Result<NodeId, ImportError> {
        if let Some(&id) = self.edge_to_node.get(&edge) {
            return Ok(id);
        }
        let e = self.g.edge(edge);
        let Some(konst) = &e.konst else {
            return Err(ImportError::Malformed(format!(
                "edge {} is neither produced, constant, nor an input",
                e.name
            )));
        };
        let data = quantize_const(konst, self.scale, &e.name)?;
        let dims = lower_dims(&konst.dims);
        let tensor = Tensor::new(data, &dims)
            .map_err(|err| ImportError::Malformed(format!("constant {}: {err}", e.name)))?;
        let id = self.push(Op::Constant(op::Constant(tensor)), vec![], dims);
        self.edge_to_node.insert(edge, id);
        Ok(id)
    }

    /// Resolve an operand and broadcast it to `target` dims if needed.
    fn resolve_broadcast(&mut self, edge: EdgeId, target: &[usize]) -> Result<NodeId, ImportError> {
        let id = self.resolve(edge)?;
        let dims = lower_dims(self.g.dims(edge).unwrap_or(&[]));
        if dims == target {
            return Ok(id);
        }
        Ok(self.push(
            Op::Broadcast(op::Broadcast {
                shape: target.to_vec(),
            }),
            vec![id],
            target.to_vec(),
        ))
    }

    fn out_dims(&self, node: &INode) -> Vec<usize> {
        lower_dims(self.g.dims(node.outputs[0]).expect("analyzed"))
    }

    fn lower_node(&mut self, node: &INode) -> Result<(), ImportError> {
        let out = self.out_dims(node);
        let unlowered = || ImportError::Unlowered {
            node: node.name.clone(),
            op_type: node.op.clone(),
        };
        let attr_err = |reason: String| ImportError::Pass {
            pass: "lower",
            node: node.name.clone(),
            reason,
        };

        // Elementwise ops whose operands must match the output shape.
        let elementwise = |me: &mut Self, op: Op| -> Result<NodeId, ImportError> {
            let inputs = node
                .inputs
                .iter()
                .map(|&e| me.resolve_broadcast(e, &out))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(me.push(op, inputs, out.clone()))
        };
        // Ops that take operands as-is.
        let direct = |me: &mut Self, op: Op| -> Result<NodeId, ImportError> {
            let inputs = node
                .inputs
                .iter()
                .map(|&e| me.resolve(e))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(me.push(op, inputs, out.clone()))
        };

        let id = match node.op.as_str() {
            "Add" => elementwise(self, Op::Add(op::Add))?,
            "Sub" => elementwise(self, Op::Sub(op::Sub))?,
            "Mul" => elementwise(self, Op::Mul(op::Mul { rebase: true }))?,
            "Neg" => direct(self, Op::Neg(op::Neg))?,
            "Relu" => direct(self, Op::ReLU(op::ReLU))?,
            "Tanh" => direct(self, Op::Tanh(op::Tanh))?,
            "Sigmoid" => direct(self, Op::Sigmoid(op::Sigmoid))?,
            "Erf" => direct(self, Op::Erf(op::Erf))?,
            "Sin" => direct(self, Op::Sin(op::Sin))?,
            "Cos" => direct(self, Op::Cos(op::Cos))?,
            "IsNaN" => direct(self, Op::IsNan(op::IsNan))?,
            "And" => elementwise(self, Op::And(op::And))?,
            "Where" => elementwise(self, Op::Iff(op::Iff))?,
            "Identity" => direct(self, Op::Identity(op::Identity))?,
            "atlas.Square" => direct(self, Op::Square(op::Square { rebase: true }))?,
            "atlas.Cube" => direct(self, Op::Cube(op::Cube { rebase: true }))?,
            "atlas.Rsqrt" => direct(self, Op::Rsqrt(op::Rsqrt))?,
            "atlas.Descale" => direct(
                self,
                Op::ScalarConstDiv(op::ScalarConstDiv {
                    divisor: self.scale.denom_i32(),
                }),
            )?,
            "atlas.MoveAxis" => {
                let source = node.attr_int("source").ok_or_else(unlowered)? as usize;
                let destination = node.attr_int("destination").ok_or_else(unlowered)? as usize;
                direct(
                    self,
                    Op::MoveAxis(op::MoveAxis {
                        source,
                        destination,
                    }),
                )?
            }
            "atlas.Broadcast" => direct(self, Op::Broadcast(op::Broadcast { shape: out.clone() }))?,
            "Reshape" => direct(self, Op::Reshape(op::Reshape { shape: out.clone() }))?,
            "Concat" => {
                let rank = out.len();
                let axis =
                    crate::shape_fns::norm_axis(node.attr_int("axis").ok_or_else(unlowered)?, rank)
                        .map_err(attr_err)?;
                direct(self, Op::Concat(op::Concat { axis }))?
            }
            "atlas.Slice" => {
                let axis = node.attr_int("axis").ok_or_else(unlowered)? as usize;
                let start = node.attr_int("start").ok_or_else(unlowered)? as usize;
                let end = node.attr_int("end").ok_or_else(unlowered)? as usize;
                direct(self, Op::Slice(op::Slice { axis, start, end }))?
            }
            "Slice" => {
                // Multi-axis ONNX slice → chain of single-axis slices.
                let in_dims = lower_dims(self.g.dims(node.inputs[0]).expect("analyzed"));
                let (axes, starts, ends, _, _) =
                    crate::shape_fns::slice_params(node, &in_dims).map_err(attr_err)?;
                let mut current = self.resolve(node.inputs[0])?;
                let mut dims = in_dims;
                let mut last = current;
                for ((&axis, &start), &end) in axes.iter().zip(&starts).zip(&ends) {
                    dims[axis] = (end - start) as usize;
                    last = self.push(
                        Op::Slice(op::Slice {
                            axis,
                            start: start as usize,
                            end: end as usize,
                        }),
                        vec![current],
                        dims.clone(),
                    );
                    current = last;
                }
                last
            }
            "Gather" => {
                let axis = node.attr_int("axis").unwrap_or(0);
                if axis != 0 {
                    return Err(attr_err(format!("Gather axis {axis}; only axis 0 lowers")));
                }
                let dict_dims = lower_dims(self.g.dims(node.inputs[0]).expect("analyzed"));
                let dict_len = dict_dims[0];
                let gather = if dict_len.next_power_of_two() <= GATHER_SMALL_LIMIT {
                    Op::GatherSmall(op::GatherSmall { axis: 0, dict_len })
                } else {
                    Op::GatherLarge(op::GatherLarge { axis: 0, dict_len })
                };
                direct(self, gather)?
            }
            "Softmax" => {
                let rank = out.len();
                let axis =
                    crate::shape_fns::norm_axis(node.attr_int("axis").ok_or_else(unlowered)?, rank)
                        .map_err(attr_err)?;
                if axis != rank - 1 {
                    return Err(attr_err(format!(
                        "softmax axis {axis} of rank {rank}; transpose so the softmax axis is last"
                    )));
                }
                direct(
                    self,
                    Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
                )?
            }
            "atlas.Sum" => {
                let axes = node
                    .attr_ints("axes")
                    .ok_or_else(unlowered)?
                    .iter()
                    .map(|&a| a as usize)
                    .collect();
                direct(self, Op::Sum(op::Sum { axes }))?
            }
            "atlas.MeanOfSquares" => {
                let axes: Vec<usize> = node
                    .attr_ints("axes")
                    .ok_or_else(unlowered)?
                    .iter()
                    .map(|&a| a as usize)
                    .collect();
                let count = node.attr_int("count").ok_or_else(unlowered)? as usize;
                let in_dims = lower_dims(self.g.dims(node.inputs[0]).expect("analyzed"));
                let padded_count = axes
                    .iter()
                    .map(|&a| in_dims.get(a).copied().unwrap_or(1).next_power_of_two())
                    .product();
                direct(
                    self,
                    Op::MeanOfSquares(op::MeanOfSquares {
                        axes,
                        count,
                        padded_count,
                    }),
                )?
            }
            "atlas.Einsum" => {
                let Some(crate::graph::Attr::Str(eq)) = node.attrs.get("equation") else {
                    return Err(unlowered());
                };
                if !CANONICAL_EINSUMS.contains(&eq.as_str()) {
                    return Err(attr_err(format!(
                        "einsum {eq:?} is outside the canonical set {CANONICAL_EINSUMS:?}"
                    )));
                }
                let equation = eq.parse::<op::EinsumEq>().map_err(attr_err)?;
                direct(self, Op::Einsum(op::Einsum { equation }))?
            }
            _ => return Err(unlowered()),
        };
        self.edge_to_node.insert(node.outputs[0], id);
        Ok(())
    }
}

/// Lower the rewritten importer graph. `descriptors` come from
/// [`classify_inputs`] run *before* the rewrite passes (classification
/// inspects `Cast`/`Gather` uses that passes rewrite away).
pub fn lower(
    g: &IGraph,
    scale: Scale,
    descriptors: &[InputDescriptor],
) -> Result<Lowered, ImportError> {
    // Inputs that nothing consumes (e.g. empty KV-cache tensors whose
    // concats were elided) are pruned — same behavior as the old pipeline.
    let live_inputs: Vec<EdgeId> = g
        .inputs
        .iter()
        .copied()
        .filter(|&e| g.use_count(e) > 0)
        .collect();
    let input_descriptors: Vec<InputDescriptor> = live_inputs
        .iter()
        .map(|&e| {
            let name = &g.edge(e).name;
            descriptors
                .iter()
                .find(|d| &d.name == name)
                .cloned()
                .ok_or_else(|| ImportError::Malformed(format!("input {name} has no descriptor")))
        })
        .collect::<Result<_, _>>()?;

    let mut lw = Lowerer {
        g,
        scale,
        nodes: Vec::new(),
        edge_to_node: HashMap::new(),
        inputs: Vec::new(),
    };

    // Graph inputs first, in declared order.
    for &edge in &live_inputs {
        let dims = lower_dims(g.dims(edge).ok_or_else(|| {
            ImportError::Malformed(format!("input {} has no shape", g.edge(edge).name))
        })?);
        let id = lw.push(Op::Input(op::Input), vec![], dims);
        lw.inputs.push(id);
        lw.edge_to_node.insert(edge, id);
    }

    for idx in g.topo_order()? {
        let node = g.nodes[idx].as_ref().expect("live");
        if node.outputs.len() != 1 {
            return Err(ImportError::Malformed(format!(
                "node {} still has {} outputs at lowering",
                node.name,
                node.outputs.len()
            )));
        }
        lw.lower_node(node)?;
    }

    let outputs = g
        .outputs
        .iter()
        .map(|&e| {
            lw.edge_to_node.get(&e).copied().ok_or_else(|| {
                ImportError::Malformed(format!(
                    "graph output {} was never produced",
                    g.edge(e).name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Lowered {
        graph: Graph {
            nodes: lw.nodes,
            inputs: lw.inputs,
            outputs,
        },
        input_descriptors,
    })
}
