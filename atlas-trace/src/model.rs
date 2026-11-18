//! Compilation: [`Graph`] → [`Model`] through validate → shape-check → pad.
//! `Model::compile` is the only constructor, so a `Model` is always fully
//! analyzed.

use serde::{Deserialize, Serialize};

use crate::{
    error::CompileError,
    graph::{Graph, Node},
    id::{NodeId, Scale, SemVersion},
    op::{Op, PadCtx, TableCache},
};

/// A compiled, analyzed, immutable model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    graph: Graph,
    scale: Scale,
    version: SemVersion,
    io: IoDims,
    /// Each node's level: its value `v` represents `v / 2^(level·scale)`. Level 1 is an
    /// `i32`; a higher level is a wide value ([`crate::Trace::wide`]).
    levels: Vec<u8>,
    /// A bound on the magnitude of each node's entries: the largest absolute entry of a
    /// constant, `2^31` for any other level-1 value, and for a wide value what its op gives
    /// from its operands' bounds.
    bounds: Vec<u128>,
    /// Memoized activation tables (regenerated on demand; not serialized).
    #[serde(skip)]
    pub(crate) tables: TableCache,
}

impl Model {
    /// Compile a graph: validate, check shapes, pad.
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

        let (levels, bounds) = levels_and_bounds(&graph)?;
        Ok(Model {
            graph,
            levels,
            bounds,
            scale: opts.scale,
            version: opts.version,
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

    /// Recompute every node's level and magnitude bound from the graph and require them to be
    /// the ones the model holds: a model that was deserialized, not compiled, is checked here.
    pub fn check_levels(&self) -> Result<(), CompileError> {
        let (levels, bounds) = levels_and_bounds(&self.graph)?;
        if levels == self.levels && bounds == self.bounds {
            return Ok(());
        }
        let node = (0..levels.len())
            .find(|&i| {
                self.levels.get(i) != Some(&levels[i]) || self.bounds.get(i) != Some(&bounds[i])
            })
            .unwrap_or(levels.len());
        Err(CompileError::Level {
            node: NodeId(node as u32),
            op: self
                .graph
                .nodes
                .get(node)
                .map_or_else(String::new, |n| n.op.to_string()),
            reason: "the model's levels or bounds are not its graph's".to_string(),
        })
    }

    /// The level of node `id`'s value.
    pub fn level(&self, id: NodeId) -> u8 {
        self.levels[id.index()]
    }

    /// The bound on the magnitude of node `id`'s entries.
    pub fn bound(&self, id: NodeId) -> u128 {
        self.bounds[id.index()]
    }

    /// The node `id`.
    pub fn node(&self, id: NodeId) -> &Node {
        self.graph.node(id)
    }

    /// The graph-input node ids.
    pub fn inputs(&self) -> &[NodeId] {
        &self.graph.inputs
    }

    /// The graph-output node ids.
    pub fn outputs(&self) -> &[NodeId] {
        &self.graph.outputs
    }

    /// Iterate nodes in topological order with their ids.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId(i as u32), n))
    }
}

/// Every node's level and magnitude bound, in node order. `Err` if an op has no form for its
/// operands' levels, a graph output is wide, or a wide value's bound reaches its op's limit.
fn levels_and_bounds(graph: &Graph) -> Result<(Vec<u8>, Vec<u128>), CompileError> {
    let mut levels: Vec<u8> = Vec::with_capacity(graph.nodes.len());
    let mut bounds: Vec<u128> = Vec::with_capacity(graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(i as u32);
        let operand_levels: Vec<u8> = node.inputs.iter().map(|x| levels[x.index()]).collect();
        let level =
            node.op
                .spec()
                .level(&operand_levels)
                .map_err(|reason| CompileError::Level {
                    node: id,
                    op: node.op.to_string(),
                    reason,
                })?;
        let bound = if level > 1 {
            let operand_bounds: Vec<u128> = node.inputs.iter().map(|x| bounds[x.index()]).collect();
            let dims: Vec<&[usize]> = node
                .inputs
                .iter()
                .map(|x| graph.node(*x).output_dims.as_slice())
                .collect();
            let bound = node.op.spec().wide_bound(&operand_bounds, &dims);
            let limit = node.op.spec().wide_limit(&operand_levels);
            if bound >= 1u128 << limit {
                return Err(CompileError::Bound {
                    node: id,
                    op: node.op.to_string(),
                    log2: u128::BITS - bound.leading_zeros(),
                    limit,
                });
            }
            bound
        } else if let Op::Constant(constant) = &node.op {
            constant
                .0
                .data()
                .iter()
                .map(|v| v.unsigned_abs() as u128)
                .max()
                .unwrap_or(0)
                .max(1)
        } else {
            1u128 << 31
        };
        levels.push(level);
        bounds.push(bound);
    }
    if let Some(&id) = graph.outputs.iter().find(|id| levels[id.index()] != 1) {
        return Err(CompileError::Level {
            node: id,
            op: graph.node(id).op.to_string(),
            reason: "a graph output is a level-1 value".to_string(),
        });
    }
    Ok((levels, bounds))
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
///
/// KNOWN GAP: this pass does not verify that padded lanes stay inert. An op
/// that writes live values into padded lanes (`Broadcast` expanding a
/// non-power-of-two axis, `Gather` on zero-padded index lanes, the
/// `ConstantFill` activations) is only sound when something downstream
/// absorbs them (zero-padded weights, a masked reduction, output cropping).
/// Today that absorption holds by construction of the importer's lowering
/// and is checked by nothing; a per-axis padded-lane analysis is the
/// designed fix (see the ignored `broadcast_fill_inflates_padded_reduction`
/// test below, which pins the failing case).
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

impl std::ops::Index<NodeId> for Model {
    type Output = crate::graph::Node;

    fn index(&self, id: NodeId) -> &crate::graph::Node {
        self.graph.node(id)
    }
}

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
            version: SemVersion::V3,
        }
    }
}

/// Original (pre-padding) IO dims, for padding inputs and cropping outputs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoDims {
    /// Original dims per graph input (same order as `graph.inputs`).
    pub inputs: Vec<Vec<usize>>,
    /// Original dims per graph output (same order as `graph.outputs`).
    pub outputs: Vec<Vec<usize>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // Pins the padding-soundness gap documented on `pad_graph`: Broadcast of
    // a size-1 axis to a non-power-of-two extent fills the padded lanes with
    // the live value, so a reduction over that axis counts the padded lanes.
    // The assertion states the CORRECT (semantic) result; un-ignore it when
    // the padded-lane analysis (or a masked Broadcast) lands.
    /// Pins `rsqrt`'s totality at the kernel level: a zero-variance row
    /// traces to zero output instead of a panic. Note this is the *tracer*
    /// contract only — the prover's rsqrt division witness needs a positive
    /// operand, which imports guarantee via the `rsqrt_eps_floor` pass; a
    /// quantized-zero variance with a nonzero row also under-normalizes, so
    /// graphs headed for proving should always carry a floored eps.
    #[test]
    fn zero_variance_row_normalizes_to_zero_without_eps() {
        use crate::{GraphBuilder, op};
        let mut b = GraphBuilder::new();
        let x = b.input(&[2, 4]);
        let var = b.push(
            Op::MeanOfSquares(op::MeanOfSquares {
                axes: vec![1],
                count: 4,
                padded_count: 4,
            }),
            &[x],
        );
        let inv = b.push(Op::Rsqrt(op::Rsqrt), &[var]);
        let inv_b = b.push(Op::Broadcast(op::Broadcast { shape: vec![2, 4] }), &[inv]);
        let y = b.push(
            Op::Mul(op::Mul {
                rebase: true,
                deferred: false,
            }),
            &[x, inv_b],
        );
        b.output(y);
        let model = b.build(CompileOptions::default()).unwrap();

        let s = 1i32 << 14;
        // Row 0: all zeros (zero variance). Row 1: all 1.0 (variance 1.0).
        let input = Tensor::new(vec![0, 0, 0, 0, s, s, s, s], &[2, 4]).unwrap();
        let trace = model.trace(&[input]).unwrap();
        let out = trace.output(model.outputs()[0]);
        assert_eq!(&out.data()[..4], &[0, 0, 0, 0], "zero row stays zero");
        assert_eq!(
            &out.data()[4..8],
            &[s, s, s, s],
            "unit row normalizes to 1.0"
        );
    }

    /// Concat whose non-last operand is non-pow2 on the axis would pad
    /// *interior* lanes ([3]+[4] → [a0,a1,a2,0,b0..b3]), shifting the second
    /// operand off its semantic positions — compile must reject it. The
    /// mirrored pow2-first order is coherent and must execute correctly.
    #[test]
    fn concat_interior_padding_rejected() {
        use crate::{GraphBuilder, op};
        let build = |first: Vec<i32>, second: Vec<i32>| {
            let mut b = GraphBuilder::new();
            let d1 = vec![first.len()];
            let d2 = vec![second.len()];
            let x = b.input(&d1);
            let c = b.constant(Tensor::new(second, &d2).unwrap());
            let cat = b.push(Op::Concat(op::Concat { axis: 0 }), &[x, c]);
            b.output(cat);
            (b.build(CompileOptions::default()), first)
        };

        let (rejected, _) = build(vec![1, 2, 3], vec![101, 102, 103, 104]);
        assert!(rejected.is_err(), "non-pow2 first operand must be rejected");

        let (model, first) = build(vec![1, 2, 3, 4], vec![101, 102, 103]);
        let model = model.expect("pow2-first concat is pad-coherent");
        let input = Tensor::new(first, &[4]).unwrap();
        let trace = model.trace(&[input]).unwrap();
        let out = trace.output(model.outputs()[0]);
        assert_eq!(&out.data()[..7], &[1, 2, 3, 4, 101, 102, 103]);
    }

    /// A slice must be one of the extent-sized blocks of its axis and a concat's operands must
    /// each start at a multiple of their own padded extent: only then are they subcubes of the
    /// padded tensors, which is what the prover's layout ties select. Compile rejects the rest.
    #[test]
    fn unaligned_slice_and_concat_rejected() {
        use crate::{GraphBuilder, op};
        let slice = |size: usize, start: usize, end: usize| {
            let mut b = GraphBuilder::new();
            let x = b.input(&[size]);
            let s = b.push(
                Op::Slice(op::Slice {
                    axis: 0,
                    start,
                    end,
                }),
                &[x],
            );
            b.output(s);
            b.build(CompileOptions::default())
        };
        let err = slice(64, 8, 40).expect_err("an unaligned slice must be rejected");
        assert!(
            err.to_string()
                .contains("slice start 8 is not a multiple of its extent 32"),
            "{err}"
        );
        slice(64, 32, 64).expect("a slice at a multiple of its extent compiles");
        slice(48, 16, 32).expect("the input extent need not be a power of two");

        let concat = |extents: &[usize]| {
            let mut b = GraphBuilder::new();
            let inputs: Vec<_> = extents.iter().map(|&extent| b.input(&[extent])).collect();
            let cat = b.push(Op::Concat(op::Concat { axis: 0 }), &inputs);
            b.output(cat);
            b.build(CompileOptions::default())
        };
        let err = concat(&[4, 8, 4]).expect_err("an unaligned concat must be rejected");
        assert!(
            err.to_string()
                .contains("operand 1 starts at 4, not a multiple of its extent 8"),
            "{err}"
        );
        concat(&[8, 4, 4]).expect("operands at multiples of their extents compile");
        concat(&[8, 4, 3]).expect("the last operand need not be a power of two");
    }

    /// Size-1 axes never move elements in the row-major layout, so
    /// squeeze/unsqueeze reshapes are pad-coherent even around non-pow2
    /// axes; genuinely merged non-pow2 axes stay rejected.
    #[test]
    fn squeeze_reshape_is_pad_coherent() {
        use crate::{GraphBuilder, op};
        let reshape = |in_dims: &[usize], shape: Vec<usize>| {
            let mut b = GraphBuilder::new();
            let x = b.input(in_dims);
            let r = b.push(Op::Reshape(op::Reshape { shape }), &[x]);
            b.output(r);
            b.build(CompileOptions::default())
        };

        let model = reshape(&[1, 7, 8], vec![7, 8]).expect("squeeze of a leading 1");
        let input = Tensor::new((0..56).collect(), &[1, 7, 8]).unwrap();
        let trace = model.trace(&[input]).unwrap();
        let out = trace.output(model.outputs()[0]);
        assert_eq!(out.data()[..56], (0..56).collect::<Vec<i32>>()[..]);

        reshape(&[7, 8], vec![1, 7, 8]).expect("unsqueeze to a leading 1");
        reshape(&[7, 1, 8], vec![7, 8]).expect("squeeze of an interior 1");
        assert!(
            reshape(&[2, 7], vec![14]).is_err(),
            "merging a non-pow2 axis stays rejected"
        );
    }

    #[test]
    #[ignore = "known padding-soundness gap: broadcast fills padded lanes with live values"]
    fn broadcast_fill_inflates_padded_reduction() {
        use crate::{GraphBuilder, op};
        let mut b = GraphBuilder::new();
        let x = b.input(&[2, 1]);
        let bc = b.push(Op::Broadcast(op::Broadcast { shape: vec![2, 3] }), &[x]);
        let sum = b.push(Op::Sum(op::Sum { axes: vec![1] }), &[bc]);
        b.output(sum);
        let model = b.build(CompileOptions::default()).unwrap();
        let input = Tensor::new(vec![5, 7], &[2, 1]).unwrap();
        let trace = model.trace(&[input]).unwrap();
        let out = trace.output(model.outputs()[0]);
        // Semantic result: 3·v per row. Today the padded 4th lane also holds
        // v, so the trace (and the proof, which agrees with it) yields 4·v.
        assert_eq!(&out.data()[..2], &[15, 21]);
    }
}
