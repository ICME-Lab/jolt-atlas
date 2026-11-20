//! Placement of rescales: which products keep their exact accumulation as a wide value, and
//! where a chain of them is taken back to level 1.
//!
//! A fused product rescales at once, and its remainder and quotient are both range-checked by
//! a proof. A product whose every consumer is itself a product that can read a wide operand
//! need not: it is lowered *deferred*, its consumers take the wide value, and the chain is
//! rescaled once where a level-1 value is needed, by a [`Rescale`](crate::op::Rescale) node
//! placed after its last product. A value is wide for all of its consumers or rescaled for
//! all of them.

use crate::{
    graph::{Graph, Node},
    id::{NodeId, Scale},
    op::{self, MAX_LEVEL, Op, WIDE_LIMIT, WideShape, lifted_bound},
};

/// What [`defer_rescales`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferReport {
    /// Products lowered deferred.
    pub deferred: usize,
    /// `Rescale` nodes placed.
    pub rescales: usize,
    /// Output cells of the nodes that rescaled before the pass.
    pub cells_before: usize,
    /// Output cells of the nodes that rescale after it.
    pub cells_after: usize,
}

/// The bound a level-1 node's entries are known by: its largest absolute entry for a
/// constant, `2^31` otherwise. The rule of `Model::compile`.
fn narrow_bound(node: &Node) -> u128 {
    match &node.op {
        Op::Constant(constant) => crate::model::largest_entry(&constant.0),
        _ => 1u128 << 31,
    }
}

/// The deferred form of a fused product, or `None` for any other op.
fn deferred_form(op: &Op) -> Option<Op> {
    match op {
        Op::Einsum(einsum) if !einsum.deferred => Some(Op::Einsum(op::Einsum {
            equation: einsum.equation.clone(),
            deferred: true,
        })),
        Op::Mul(mul) if mul.rebase => Some(Op::Mul(op::Mul {
            rebase: false,
            deferred: true,
        })),
        _ => None,
    }
}

/// Whether `node` is a two-operand `Add` or `Sub`, which reads wide operands as they are.
fn is_sum(node: &Node) -> bool {
    matches!(node.op, Op::Add(_) | Op::Sub(_)) && node.inputs.len() == 2
}

/// Whether `node`'s value is at level 1 whatever the pass decides: it is not a product, a
/// sum or a reshape.
fn always_narrow(node: &Node) -> bool {
    deferred_form(&node.op).is_none() && !is_sum(node) && !matches!(node.op, Op::Reshape(_))
}

/// The node a chain of reshapes ending at `index` reshapes.
fn source(graph: &Graph, mut index: usize) -> usize {
    while let (Op::Reshape(_), [operand]) =
        (&graph.nodes[index].op, graph.nodes[index].inputs.as_slice())
    {
        index = operand.index();
    }
    index
}

/// The consumers of `index`'s value that are not reshapes, each with the operand position it
/// reads at, looking through reshapes (which move no element and keep a value's level).
/// `None` if the value, or a reshape of it, is a graph output.
fn value_readers(
    graph: &Graph,
    consumers: &[Vec<(usize, usize)>],
    index: usize,
) -> Option<Vec<(usize, usize)>> {
    if graph.outputs.iter().any(|id| id.index() == index) {
        return None;
    }
    let mut readers = Vec::new();
    for &(reader, position) in &consumers[index] {
        if matches!(graph.nodes[reader].op, Op::Reshape(_)) {
            readers.extend(value_readers(graph, consumers, reader)?);
        } else {
            readers.push((reader, position));
        }
    }
    Some(readers)
}

/// Lower every product of `graph` whose consumers can all read its exact accumulation as a
/// deferred product, and place a `Rescale` after the last product of each chain.
///
/// A product stays wide when every consumer of its value, looking through reshapes, is a
/// fused `Mul`, a two-operand `Add` or `Sub` (which lifts its other operand), or the first
/// operand of a fused `mk,kn->mn` contraction against a constant, such that the consumer's level stays
/// within [`MAX_LEVEL`] and its magnitude bound within [`WIDE_LIMIT`], and when the bits a
/// rescale of the product would commit exceed the bits its absence adds to the remainders of
/// its consumers' rescales. Dims are taken padded to powers of two, as the compiled graph
/// has them.
pub fn defer_rescales(graph: Graph, scale: Scale) -> (Graph, DeferReport) {
    let count = graph.nodes.len();
    let scale_bits = scale.log2() as usize;
    let padded = |node: &Node| -> Vec<usize> {
        node.output_dims
            .iter()
            .map(|d| d.next_power_of_two())
            .collect()
    };
    let cells = |node: &Node| -> usize { padded(node).iter().product() };
    let mut consumers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); count];
    for (index, node) in graph.nodes.iter().enumerate() {
        for (position, input) in node.inputs.iter().enumerate() {
            consumers[input.index()].push((index, position));
        }
    }

    // What each node presents to its consumers: its level and magnitude bound, after its
    // rescale if one is placed after it.
    let mut level = vec![1u8; count];
    let mut bound = vec![0u128; count];
    let mut deferred = vec![false; count];
    let mut rescale_after = vec![false; count];
    let mut report = DeferReport::default();

    for index in 0..count {
        let node = &graph.nodes[index];
        bound[index] = narrow_bound(node);
        if node.op.spec().rebase_divisor(scale).is_some() {
            report.cells_before += cells(node);
        }
        let in_levels: Vec<u8> = node.inputs.iter().map(|x| level[x.index()]).collect();
        let wide_operand = in_levels.iter().any(|&l| l > 1);
        // A product in its deferred form, or a sum that reads a wide operand: the nodes whose
        // value may be wide.
        let form = match deferred_form(&node.op) {
            Some(form) => Some(form),
            None if is_sum(node) && wide_operand => Some(node.op.clone()),
            None => None,
        };
        let Some(form) = form else {
            if node.op.spec().rebase_divisor(scale).is_some() {
                report.cells_after += cells(node);
            }
            // A reshape presents what its operand does.
            if let (Op::Reshape(_), [operand]) = (&node.op, node.inputs.as_slice()) {
                level[index] = level[operand.index()];
                bound[index] = bound[operand.index()];
            }
            continue;
        };
        let product = deferred_form(&node.op).is_some();
        let in_bounds: Vec<u128> = node.inputs.iter().map(|x| bound[x.index()]).collect();
        let in_dims: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .map(|x| padded(&graph.nodes[x.index()]))
            .collect();
        let in_dims: Vec<&[usize]> = in_dims.iter().map(Vec::as_slice).collect();
        let out_level = form.spec().level(&in_levels).ok();
        let wide_bound = form.spec().wide_bound(
            &in_bounds,
            &WideShape {
                levels: &in_levels,
                dims: &in_dims,
                scale,
            },
        );
        let fits = out_level.is_some_and(|l| l <= MAX_LEVEL)
            && wide_bound < 1u128 << form.spec().wide_limit(&in_levels);
        let out_level = out_level.unwrap_or(1);

        // The consumers of this node's value, looking through reshapes; `None` if the value
        // or a reshape of it is a graph output.
        let readers = value_readers(&graph, &consumers, index);
        // Whether consumer `reader`, reading this node's value at `position`, has a form for
        // its exact accumulation.
        let accepts = |&(reader, position): &(usize, usize)| -> bool {
            let reader_node = &graph.nodes[reader];
            let within = |reader_level: u8, reader_bound: u128| {
                reader_level <= MAX_LEVEL && reader_bound < 1u128 << WIDE_LIMIT
            };
            match &reader_node.op {
                Op::Einsum(einsum)
                    if !einsum.deferred
                        && position == 0
                        && einsum.equation.to_string() == "mk,kn->mn" =>
                {
                    let weights = &graph.nodes[reader_node.inputs[1].index()];
                    let terms = padded(weights).first().copied().unwrap_or(1) as u128;
                    matches!(weights.op, Op::Constant(_))
                        && within(
                            out_level + 1,
                            wide_bound
                                .saturating_mul(narrow_bound(weights))
                                .saturating_mul(terms),
                        )
                }
                Op::Mul(mul) if mul.rebase => {
                    let other = reader_node.inputs[1 - position].index();
                    if source(&graph, other) == index {
                        within(2 * out_level, wide_bound.saturating_mul(wide_bound))
                    } else if other < index {
                        within(
                            out_level + level[other],
                            wide_bound.saturating_mul(bound[other]),
                        )
                    } else {
                        // Not yet placed: a level-1 value unless it may be wide.
                        let other = &graph.nodes[other];
                        always_narrow(other)
                            && within(
                                out_level + 1,
                                wide_bound.saturating_mul(narrow_bound(other)),
                            )
                    }
                }
                // A sum lifts its lower operand to the level of its higher one.
                _ if is_sum(reader_node) => {
                    let other = reader_node.inputs[1 - position].index();
                    let (other_level, other_bound) = if source(&graph, other) == index {
                        (out_level, wide_bound)
                    } else if other < index {
                        (level[other], bound[other])
                    } else if always_narrow(&graph.nodes[other]) {
                        (1, narrow_bound(&graph.nodes[other]))
                    } else {
                        return false;
                    };
                    let top = out_level.max(other_level);
                    within(
                        top,
                        lifted_bound(wide_bound, out_level, top, scale)
                            .saturating_add(lifted_bound(other_bound, other_level, top, scale)),
                    )
                }
                _ => false,
            }
        };
        // A rescale here would commit a remainder and a quotient per cell; without it each
        // reader's rescale carries a remainder wider by this node's levels above the first.
        let rescale_bits = cells(node) * ((out_level as usize - 1) * scale_bits + 32);
        let stays_wide = fits
            && readers.as_ref().is_some_and(|readers| {
                // A sum that reads it trades its own output's range claim for the rescale's.
                let added_bits: usize = readers
                    .iter()
                    .filter(|&&(reader, _)| !is_sum(&graph.nodes[reader]))
                    .map(|&(reader, _)| {
                        cells(&graph.nodes[reader]) * (out_level as usize - 1) * scale_bits
                    })
                    .sum();
                !readers.is_empty() && readers.iter().all(accepts) && rescale_bits > added_bits
            });

        if wide_operand {
            assert!(fits, "a node was given a wide operand it has no form for");
        }
        if product && (wide_operand || stays_wide) {
            deferred[index] = true;
            report.deferred += 1;
        }
        if stays_wide {
            level[index] = out_level;
            bound[index] = wide_bound;
        } else {
            rescale_after[index] = wide_operand;
            report.cells_after += cells(node);
        }
    }

    // Rebuild: a deferred product in its deferred form, a `Rescale` after the last product
    // of each chain, and every consumer reading what its operand presents.
    let mut nodes: Vec<Node> = Vec::with_capacity(count);
    let mut presented: Vec<NodeId> = Vec::with_capacity(count);
    for (index, node) in graph.nodes.into_iter().enumerate() {
        let op = if deferred[index] {
            deferred_form(&node.op).expect("only a product is deferred")
        } else {
            node.op
        };
        let id = NodeId(nodes.len() as u32);
        nodes.push(Node {
            op,
            inputs: node.inputs.iter().map(|x| presented[x.index()]).collect(),
            output_dims: node.output_dims.clone(),
        });
        if rescale_after[index] {
            report.rescales += 1;
            let rescale = NodeId(nodes.len() as u32);
            nodes.push(Node {
                op: Op::Rescale(op::Rescale),
                inputs: vec![id],
                output_dims: node.output_dims,
            });
            presented.push(rescale);
        } else {
            presented.push(id);
        }
    }
    let remap = |ids: Vec<NodeId>| ids.into_iter().map(|id| presented[id.index()]).collect();
    (
        Graph {
            nodes,
            inputs: remap(graph.inputs),
            outputs: remap(graph.outputs),
        },
        report,
    )
}
