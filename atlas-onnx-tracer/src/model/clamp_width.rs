//! Static width analysis for the saturating-clamp lookups.
//!
//! Every clamped node (`Add`/`Sub`/`Sum`/`Einsum`/`Mul`/`Square`/`Cube`/
//! `MeanOfSquares`) proves `output = SatClamp(v)` with a lookup indexed by the
//! two's-complement encoding of the pre-clamp value `v`. That lookup was always
//! 64 bits wide, i.e. 16 four-bit one-hot chunks per node — but `v` is bounded
//! by the operator and its operands' ranges, and for most nodes the bound needs
//! far fewer bits. This pass computes a *sound* per-node width from the model
//! alone (so prover and verifier agree without communication):
//!
//! - every node output is an `i32` (that is exactly what the producers' own
//!   clamps / lookups establish, so it may be assumed inductively);
//! - constant operands (weights) are known exactly, so an einsum against a
//!   weight tensor is bounded by `2^31 · max_row Σ_k |w_k|`;
//! - the fused rescale ops' pre-clamp value is the *quotient* `acc >> S`.
//!
//! Widths are rounded up to a multiple of 8 (the read-raf sumcheck runs 8
//! address phases) in `{40, 48, 56, 64}`; anything that could exceed 63 bits
//! keeps the full 64 (the i64 accumulation itself is assumed not to overflow,
//! as before).
use crate::{model::Model, node::ComputationNode, ops::Operator, tensor::Tensor};
use std::collections::{BTreeMap, HashSet};

/// Widest saturating-clamp lookup (an i64 accumulation).
pub const CLAMP_WIDTH_MAX: usize = 64;
/// The supported saturating-clamp lookup widths.
pub const CLAMP_WIDTHS: [usize; 4] = [40, 48, 56, 64];

/// Smallest supported width whose signed range contains `[-bound, bound]`.
fn width_for_bound(bound: u128) -> usize {
    // `bound < 2^b` with `b = bits(bound)`; a `w`-bit two's-complement value
    // holds `[-2^(w-1), 2^(w-1))`, so `w = b + 1` suffices.
    let magnitude_bits = (u128::BITS - bound.leading_zeros()) as usize;
    let needed = magnitude_bits + 1;
    CLAMP_WIDTHS
        .into_iter()
        .find(|&w| w >= needed)
        .unwrap_or(CLAMP_WIDTH_MAX)
}

const I32_MAG: u128 = 1 << 31;

/// `max` over the operand's free indices of `Σ` over its contracted indices of
/// `|w|`, for a constant einsum operand `w` with index string `idx`. Free
/// indices are the letters that survive into the output; everything else is
/// summed over (possibly against the other operand, possibly alone).
fn const_operand_row_sum_max(
    w: &Tensor<i32>,
    idx: &str,
    output_letters: &HashSet<char>,
) -> Option<u128> {
    let letters: Vec<char> = idx.chars().collect();
    let dims = w.dims();
    if letters.len() != dims.len() {
        return None;
    }
    let free: Vec<usize> = (0..letters.len())
        .filter(|&i| output_letters.contains(&letters[i]))
        .collect();
    let free_size: usize = free.iter().map(|&i| dims[i]).product::<usize>().max(1);
    let mut sums = vec![0u128; free_size];
    let n = dims.len();
    let mut coord = vec![0usize; n];
    for &v in w.data() {
        let mut key = 0usize;
        for &f in &free {
            key = key * dims[f] + coord[f];
        }
        sums[key] += (v as i64).unsigned_abs() as u128;
        // advance coord (row-major)
        for d in (0..n).rev() {
            coord[d] += 1;
            if coord[d] < dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    sums.into_iter().max()
}

/// Bound on `|Σ_k left·right|` for an einsum: `2^31 · max_row Σ|w|` when an
/// operand is a constant tensor, else `2^62 · (contracted size)`.
fn einsum_acc_bound(
    node: &ComputationNode,
    equation: &str,
    nodes: &BTreeMap<usize, ComputationNode>,
) -> u128 {
    let mut parts = equation.split("->");
    let (Some(inputs_eq), Some(output_eq)) = (parts.next(), parts.next()) else {
        return u128::MAX;
    };
    let inputs_eq: Vec<&str> = inputs_eq.split(',').collect();
    if inputs_eq.len() != node.inputs.len() {
        return u128::MAX;
    }
    let output_letters: HashSet<char> = output_eq.chars().collect();

    // Prefer a constant operand: its row sums bound the contraction exactly.
    let mut best: Option<u128> = None;
    for (i, &input_idx) in node.inputs.iter().enumerate() {
        if let Some(Operator::Constant(w)) = nodes.get(&input_idx).map(|n| &n.operator) {
            if let Some(m) = const_operand_row_sum_max(&w.0, inputs_eq[i], &output_letters) {
                let b = I32_MAG * m;
                best = Some(best.map_or(b, |x: u128| x.min(b)));
            }
        }
    }
    if let Some(b) = best {
        return b;
    }

    // Both operands dynamic: each product is at most 2^62, summed over the
    // contracted index space.
    let mut sizes = std::collections::HashMap::new();
    for (i, &input_idx) in node.inputs.iter().enumerate() {
        let Some(input) = nodes.get(&input_idx) else {
            return u128::MAX;
        };
        for (j, c) in inputs_eq[i].chars().enumerate() {
            if let Some(&d) = input.output_dims.get(j) {
                sizes.insert(c, d);
            }
        }
    }
    let contracted: u128 = sizes
        .iter()
        .filter(|(c, _)| !output_letters.contains(c))
        .map(|(_, &d)| d as u128)
        .product::<u128>()
        .max(1);
    (1u128 << 62) * contracted
}

/// Sound bound on the magnitude of a clamped node's pre-clamp value, or `None`
/// if the node has no saturating clamp.
pub fn clamp_value_bound(
    node: &ComputationNode,
    nodes: &BTreeMap<usize, ComputationNode>,
) -> Option<u128> {
    let shift = |acc: u128, s: i32| acc >> s.max(0) as u32;
    Some(match &node.operator {
        Operator::Add(_) | Operator::Sub(_) => 2 * I32_MAG,
        Operator::Sum(s) => {
            let input = nodes.get(node.inputs.first()?)?;
            let n: u128 = s
                .axes
                .iter()
                .map(|&a| *input.output_dims.get(a).unwrap_or(&1) as u128)
                .product::<u128>()
                .max(1);
            n * I32_MAG
        }
        Operator::Mul(op) if op.scale != 0 => shift(1 << 62, op.scale),
        Operator::Square(op) if op.scale != 0 => shift(1 << 62, op.scale),
        // x^3 needs 93 bits before the rebase; keep the full width.
        Operator::Cube(op) if op.scale != 0 => u128::MAX,
        Operator::MeanOfSquares(op) => shift(1 << 62, op.scale),
        Operator::Einsum(op) => shift(einsum_acc_bound(node, &op.equation, nodes), op.scale),
        _ => return None,
    })
}

impl Model {
    /// Stamp every clamped node with its saturating-clamp lookup width (see the
    /// module docs). Idempotent; called by the loader, and by anything that
    /// builds a [`Model`] by hand and wants narrower-than-64-bit clamps.
    pub fn annotate_clamp_widths(&mut self) {
        let widths: Vec<(usize, usize)> = self
            .graph
            .nodes
            .values()
            .filter_map(|node| {
                clamp_value_bound(node, &self.graph.nodes).map(|bound| {
                    let w = if bound >= (1u128 << 63) {
                        CLAMP_WIDTH_MAX
                    } else {
                        width_for_bound(bound)
                    };
                    (node.idx, w)
                })
            })
            .collect();
        for (idx, w) in widths {
            if let Some(node) = self.graph.nodes.get_mut(&idx) {
                node.sat_clamp_bits = w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_rounding() {
        assert_eq!(width_for_bound(1 << 32), 40);
        assert_eq!(width_for_bound((1 << 39) - 1), 40);
        assert_eq!(width_for_bound(1 << 39), 48);
        assert_eq!(width_for_bound(1 << 48), 56);
        assert_eq!(width_for_bound(1 << 60), 64);
    }
}
