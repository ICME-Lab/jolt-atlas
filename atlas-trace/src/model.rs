//! Compilation: [`Graph`] → [`Model`] through validate → shape-check → pad →
//! clamp-width annotate. `Model::compile` is the only constructor, so a
//! `Model` is always fully analyzed.

use serde::{Deserialize, Serialize};

use crate::{
    error::CompileError,
    graph::{Graph, Node},
    id::{NodeId, Scale, SemVersion},
    op::{BoundCtx, Op, PadCtx, TableCache},
    tensor::Tensor,
};

/// Options for [`Model::compile`].
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// The global fixed-point scale.
    pub scale: Scale,
    /// Pad every dimension to the next power of two (what the prover needs).
    pub pad_pow2: bool,
    /// The semantics version to trace under.
    pub version: SemVersion,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            scale: Scale::new(14).expect("14 is in range"),
            pad_pow2: true,
            version: SemVersion::V1,
        }
    }
}

/// Supported saturating-clamp lookup widths (multiples of 8: the read-raf
/// sumcheck runs 8 address phases).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ClampWidth {
    /// 40-bit lookup.
    W40,
    /// 48-bit lookup.
    W48,
    /// 56-bit lookup.
    W56,
    /// Full 64-bit lookup (an i64 accumulation).
    W64,
}

impl ClampWidth {
    /// The width in bits.
    pub fn bits(self) -> usize {
        match self {
            ClampWidth::W40 => 40,
            ClampWidth::W48 => 48,
            ClampWidth::W56 => 56,
            ClampWidth::W64 => 64,
        }
    }

    /// Smallest supported width whose signed range contains `[-bound, bound]`.
    /// Ported verbatim from the old crate's `width_for_bound`.
    pub fn for_bound(bound: u128) -> ClampWidth {
        if bound >= (1u128 << 63) {
            return ClampWidth::W64;
        }
        // `bound < 2^b`; a w-bit two's-complement holds [-2^(w-1), 2^(w-1)).
        let needed = (u128::BITS - bound.leading_zeros()) as usize + 1;
        [ClampWidth::W40, ClampWidth::W48, ClampWidth::W56]
            .into_iter()
            .find(|w| w.bits() >= needed)
            .unwrap_or(ClampWidth::W64)
    }
}

/// Analysis results attached to a node by compilation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMeta {
    /// Width of the node's saturating-clamp lookup; `None` = no clamp.
    pub sat_clamp_bits: Option<ClampWidth>,
}

/// Original (pre-padding) IO dims, for padding inputs and cropping outputs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoDims {
    /// Original dims per graph input (same order as `graph.inputs`).
    pub inputs: Vec<Vec<usize>>,
    /// Original dims per graph output (same order as `graph.outputs`).
    pub outputs: Vec<Vec<usize>>,
}

/// A compiled, analyzed, immutable model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    graph: Graph,
    scale: Scale,
    version: SemVersion,
    meta: Vec<NodeMeta>,
    io: IoDims,
    /// Memoized activation tables (regenerated on demand; not serialized).
    #[serde(skip)]
    pub(crate) tables: TableCache,
}

impl Model {
    /// Compile a graph: validate, check shapes, pad, annotate clamp widths.
    pub fn compile(mut graph: Graph, opts: CompileOptions) -> Result<Model, CompileError> {
        graph.validate()?;
        check_shapes(&graph)?;

        let io = IoDims {
            inputs: graph
                .inputs
                .iter()
                .map(|&id| graph.node(id).output_dims.clone())
                .collect(),
            outputs: graph
                .outputs
                .iter()
                .map(|&id| graph.node(id).output_dims.clone())
                .collect(),
        };

        if opts.pad_pow2 {
            pad_graph(&mut graph)?;
            // The padded graph must still shape-check; this catches shape
            // rewrites that diverge from the padded operand dims.
            check_shapes(&graph)?;
        }

        let meta = annotate_clamp_widths(&graph, opts.scale);

        Ok(Model {
            graph,
            scale: opts.scale,
            version: opts.version,
            meta,
            io,
            tables: TableCache::default(),
        })
    }

    /// The compiled (padded) graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The global fixed-point scale.
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// The semantics version.
    pub fn version(&self) -> SemVersion {
        self.version
    }

    /// Original (pre-padding) IO dims.
    pub fn io_dims(&self) -> &IoDims {
        &self.io
    }

    /// A node together with its analysis results.
    pub fn node(&self, id: NodeId) -> (&Node, &NodeMeta) {
        (self.graph.node(id), &self.meta[id.index()])
    }

    /// The graph-input node ids.
    pub fn inputs(&self) -> &[NodeId] {
        &self.graph.inputs
    }

    /// The graph-output node ids.
    pub fn outputs(&self) -> &[NodeId] {
        &self.graph.outputs
    }

    /// Iterate nodes in topological order with ids and analysis results.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node, &NodeMeta)> {
        self.graph
            .nodes
            .iter()
            .zip(&self.meta)
            .enumerate()
            .map(|(i, (n, m))| (NodeId(i as u32), n, m))
    }
}

/// Infer every node's shape from its operands and compare with the declared
/// dims (Input nodes keep their declaration).
fn check_shapes(graph: &Graph) -> Result<(), CompileError> {
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Input(_)) {
            continue;
        }
        let input_dims: Vec<&[usize]> = node
            .inputs
            .iter()
            .map(|&id| graph.node(id).output_dims.as_slice())
            .collect();
        let inferred =
            node.op
                .spec()
                .infer_shape(&input_dims)
                .map_err(|source| CompileError::Shape {
                    node: NodeId(i as u32),
                    op: node.op.to_string(),
                    source,
                })?;
        if inferred != node.output_dims {
            return Err(CompileError::ShapeMismatch {
                node: NodeId(i as u32),
                op: node.op.to_string(),
                inferred,
                declared: node.output_dims.clone(),
            });
        }
    }
    Ok(())
}

/// Pad every dimension to the next power of two: rewrite each op's
/// shape-carrying fields, then pad the node's output dims.
fn pad_graph(graph: &mut Graph) -> Result<(), CompileError> {
    let originals: Vec<Vec<usize>> = graph.nodes.iter().map(|n| n.output_dims.clone()).collect();

    for i in 0..graph.nodes.len() {
        let input_dims: Vec<&[usize]> = graph.nodes[i]
            .inputs
            .iter()
            .map(|&id| originals[id.index()].as_slice())
            .collect();
        let ctx = PadCtx {
            input_dims,
            output_dims: &originals[i],
        };
        let node = &mut graph.nodes[i];
        node.op
            .spec_mut()
            .apply_padding(&ctx)
            .map_err(|reason| CompileError::Padding {
                node: NodeId(i as u32),
                op: node.op.to_string(),
                reason,
            })?;
        node.output_dims = node
            .output_dims
            .iter()
            .map(|d| d.next_power_of_two())
            .collect();
    }
    Ok(())
}

/// Fold [`crate::op::OpSpec::clamp_bound`] over the graph.
fn annotate_clamp_widths(graph: &Graph, scale: Scale) -> Vec<NodeMeta> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let input_dims: Vec<&[usize]> = node
                .inputs
                .iter()
                .map(|&id| graph.node(id).output_dims.as_slice())
                .collect();
            let input_consts: Vec<Option<&Tensor<i32>>> = node
                .inputs
                .iter()
                .map(|&id| match &graph.node(id).op {
                    Op::Constant(c) => Some(c.0.as_ref()),
                    _ => None,
                })
                .collect();
            let ctx = BoundCtx {
                scale,
                input_dims,
                input_consts,
            };
            NodeMeta {
                sat_clamp_bits: node.op.spec().clamp_bound(&ctx).map(ClampWidth::for_bound),
            }
        })
        .collect()
}

impl std::ops::Index<NodeId> for Model {
    type Output = crate::graph::Node;

    fn index(&self, id: NodeId) -> &crate::graph::Node {
        self.graph.node(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_rounding_matches_old_crate() {
        // Pinned in the old crate's clamp_width tests.
        assert_eq!(ClampWidth::for_bound(1 << 32), ClampWidth::W40);
        assert_eq!(ClampWidth::for_bound((1 << 39) - 1), ClampWidth::W40);
        assert_eq!(ClampWidth::for_bound(1 << 39), ClampWidth::W48);
        assert_eq!(ClampWidth::for_bound(1 << 48), ClampWidth::W56);
        assert_eq!(ClampWidth::for_bound(1 << 60), ClampWidth::W64);
        assert_eq!(ClampWidth::for_bound(u128::MAX), ClampWidth::W64);
    }
}
