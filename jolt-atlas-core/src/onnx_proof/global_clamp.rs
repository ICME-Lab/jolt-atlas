//! Packed ("global") saturating-clamp and rescale-remainder buckets.
//!
//! Every clamped node used to commit its own one-hot chunks and run its own
//! lookup sumchecks. Here nodes of the same width are packed into *buckets*:
//! one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has one set of committed chunks
//! and one set of checks whatever the node count.
//!
//! A bucket sumcheck over the packed cycle space uses the cycle weight
//! `Σ_n γ_n·eq(r_n, t − off_n)` ([`CycleWeight::Packed`]) so its input claim
//! is a `γ`-combination of the member nodes' existing openings. The clamp
//! buckets are proven by the interior / saturated split and the remainder
//! buckets by a value sumcheck, both in [`super::clamp_split`]. Gaps at the
//! end of a bucket hold zero-weight dummy cycles.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
use super::{
    clamp_lookups::is_scalar,
    clamp_split,
    fused_rebase::{rebase_bits, rebase_remainder, remainder_lookup_bits},
    witness::build_one_hot_rad_witness,
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::model::{trace::Trace, Model};
use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, opening_proof::OpeningAccumulator},
    subprotocols::ps_shout::{CycleWeight, PackedSegment},
    utils::{lookup_bits::LookupBits, math::Math},
};

/// A node's segment inside a bucket's cycle space.
#[derive(Clone, Debug)]
pub struct BucketNode {
    /// Node index.
    pub idx: usize,
    /// Start of the node's segment (a multiple of its own padded size).
    pub offset: usize,
    /// log₂ of the node's padded output size.
    pub log_t: usize,
}

/// Which lookup family a bucket packs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketKind {
    /// Saturating clamp (`output = SatClamp(acc)`), width = clamp address bits.
    Clamp,
    /// Fused rescale remainder range check (`R < 2^width`).
    Remainder,
}

/// One packed lookup instance.
#[derive(Clone, Debug)]
pub struct ClampBucket {
    /// Lookup family.
    pub kind: BucketKind,
    /// Bucket index (unique within the kind).
    pub idx: usize,
    /// Clamp lookup width shared by every node in the bucket.
    pub width: usize,
    /// log₂ of the bucket's cycle space.
    pub log_t: usize,
    /// Member nodes, in layout order.
    pub nodes: Vec<BucketNode>,
}

/// Whether the node contributes a saturating clamp. Bucket construction needs
/// only this classification and the width already registered in the model.
/// Recomputing the magnitude bound here would scan every constant einsum
/// operand each time the prover or verifier reconstructs the bucket layout.
fn has_saturating_clamp(node: &ComputationNode, model: &Model) -> bool {
    match &node.operator {
        Operator::Add(_) | Operator::Sub(_) | Operator::MeanOfSquares(_) | Operator::Einsum(_) => {
            true
        }
        Operator::Sum(_) => node
            .inputs
            .first()
            .is_some_and(|idx| model.graph.nodes.contains_key(idx)),
        Operator::Mul(op) => op.scale != 0,
        Operator::Square(op) => op.scale != 0,
        Operator::Cube(op) => op.scale != 0,
        _ => false,
    }
}

/// Partition the model's clamped (non-scalar) nodes into buckets: grouped by
/// width, packed first-fit-decreasing by padded size into a capacity equal to
/// the largest clamped node (so offsets are automatically aligned).
pub fn clamp_buckets(model: &Model) -> Vec<ClampBucket> {
    let nodes = &model.graph.nodes;
    let clamped: Vec<(usize, usize, usize)> = nodes
        .values()
        .filter(|n| !is_scalar(n) && has_saturating_clamp(n, model))
        .map(|n| (n.idx, n.sat_clamp_bits, n.pow2_padded_num_output_elements()))
        .collect();
    pack_buckets(BucketKind::Clamp, clamped)
}

/// Whether `node` proves a fused rescale remainder range check (Einsum / Mul /
/// Square / Cube with a nonzero rebase, non-scalar).
pub fn has_remainder_rc(node: &ComputationNode) -> bool {
    matches!(
        node.operator,
        Operator::Einsum(_) | Operator::Mul(_) | Operator::Square(_) | Operator::Cube(_)
    ) && rebase_bits(&node.operator).is_some_and(|b| b > 0)
        && !is_scalar(node)
}

/// Partition the fused-rescale remainder range checks into buckets keyed by
/// rebase width.
pub fn remainder_buckets(model: &Model) -> Vec<ClampBucket> {
    let items: Vec<(usize, usize, usize)> = model
        .graph
        .nodes
        .values()
        .filter(|n| has_remainder_rc(n))
        .map(|n| {
            (
                n.idx,
                rebase_bits(&n.operator).unwrap() as usize,
                n.pow2_padded_num_output_elements(),
            )
        })
        .collect();
    pack_buckets(BucketKind::Remainder, items)
}

/// First-fit-decreasing packing of `(node, width, padded size)` items, grouped
/// by width, into a capacity equal to the largest item.
fn pack_buckets(kind: BucketKind, clamped: Vec<(usize, usize, usize)>) -> Vec<ClampBucket> {
    if clamped.is_empty() {
        return Vec::new();
    }
    let capacity = clamped.iter().map(|c| c.2).max().unwrap();
    let mut widths: Vec<usize> = clamped.iter().map(|c| c.1).collect();
    widths.sort_unstable();
    widths.dedup();

    let mut buckets: Vec<ClampBucket> = Vec::new();
    for w in widths {
        let mut group: Vec<(usize, usize)> = clamped
            .iter()
            .filter(|c| c.1 == w)
            .map(|c| (c.0, c.2))
            .collect();
        // Decreasing size (then node index): every earlier size is a multiple
        // of every later one, so first-fit keeps segments aligned.
        group.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut open: Vec<(usize, Vec<BucketNode>)> = Vec::new();
        for (idx, t) in group {
            let node = |offset: usize| BucketNode {
                idx,
                offset,
                log_t: t.log_2(),
            };
            match open.iter_mut().find(|(fill, _)| *fill + t <= capacity) {
                Some((fill, members)) => {
                    members.push(node(*fill));
                    *fill += t;
                }
                None => open.push((t, vec![node(0)])),
            }
        }
        for (fill, members) in open {
            buckets.push(ClampBucket {
                kind,
                idx: buckets.len(),
                width: w,
                log_t: fill.next_power_of_two().log_2(),
                nodes: members,
            });
        }
    }
    buckets
}

impl ClampBucket {
    /// Number of one-hot chunks (`width / LOG_K_CHUNK`).
    pub fn num_chunks(&self) -> usize {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.width).instruction_d
    }

    /// The `d`-th committed one-hot chunk polynomial.
    pub fn committed_poly(&self, d: usize) -> CommittedPoly {
        match self.kind {
            BucketKind::Clamp => CommittedPoly::GlobalClampRaD(self.idx, d),
            BucketKind::Remainder => CommittedPoly::GlobalRemainderRaD(self.idx, d),
        }
    }

    /// The bucket's committed one-hot chunk polynomials.
    pub fn committed_polys(&self) -> Vec<CommittedPoly> {
        match self.kind {
            BucketKind::Clamp => clamp_split::chunk_polys(self),
            BucketKind::Remainder => (0..self.num_chunks())
                .map(|d| self.committed_poly(d))
                .collect(),
        }
    }

    /// Size of the bucket's cycle space.
    pub fn cycle_len(&self) -> usize {
        1 << self.log_t
    }

    /// Lay each node's lookup addresses into the bucket's cycle space; the
    /// unused tail holds index-0 lookups.
    pub fn assemble_bits(
        &self,
        mut per_node: impl FnMut(usize) -> Vec<LookupBits>,
    ) -> Vec<LookupBits> {
        let mut bits = vec![LookupBits::new(0, self.width); self.cycle_len()];
        for n in &self.nodes {
            let b = per_node(n.idx);
            assert_eq!(b.len(), 1 << n.log_t, "node {} lookup count", n.idx);
            bits[n.offset..n.offset + b.len()].copy_from_slice(&b);
        }
        bits
    }

    /// The packed remainder range check's input claim `Σ_n γ_n·R_n(r_n)` and
    /// cycle weight, from the member nodes' remainder advice openings.
    pub fn remainder_inputs<F: JoltField>(
        &self,
        gammas: &[F],
        accumulator: &dyn OpeningAccumulator<F>,
        model: &Model,
    ) -> (F, CycleWeight<F>) {
        assert_eq!(gammas.len(), self.nodes.len());
        let mut input_claim = F::zero();
        let mut segments = Vec::with_capacity(self.nodes.len());
        for (n, gamma) in self.nodes.iter().zip(gammas) {
            let node = &model.graph.nodes[&n.idx];
            let (r, claim) = AccOpeningAccessor::new(accumulator, node)
                .get_advice(VirtualPoly::RescaleRemainder);
            assert_eq!(r.len(), n.log_t, "node {} remainder point length", n.idx);
            input_claim += *gamma * claim;
            segments.push(PackedSegment {
                gamma: *gamma,
                prefix: n.offset >> n.log_t,
                prefix_len: self.log_t - n.log_t,
                r: r.r,
            });
        }
        (
            input_claim,
            CycleWeight::Packed {
                log_T: self.log_t,
                segments,
            },
        )
    }
}

/// All buckets' committed polynomials (the model-level replacement for the
/// per-node `ClampRaD` / `RescaleRemainderRaD` chunks).
pub fn bucket_committed_polys(model: &Model) -> Vec<CommittedPoly> {
    clamp_buckets(model)
        .iter()
        .chain(remainder_buckets(model).iter())
        .flat_map(ClampBucket::committed_polys)
        .collect()
}

/// Witnesses for every bucket's chunk polynomials.
#[tracing::instrument(skip_all, name = "global_clamp::bucket_witnesses")]
pub fn bucket_witnesses<F: JoltField>(
    model: &Model,
    trace: &Trace,
) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
    let nodes = &model.graph.nodes;
    clamp_buckets(model)
        .iter()
        .chain(remainder_buckets(model).iter())
        .flat_map(|bucket| match bucket.kind {
            BucketKind::Clamp => {
                match clamp_split::live_bucket(bucket, |idx| {
                    !clamp_split::exact_output_prover(model, trace, idx)
                }) {
                    Some(_) => clamp_split::BucketSplit::from_trace(bucket, trace, model)
                        .witness_polys(bucket),
                    None => clamp_split::BucketSplit::empty_witness_polys(bucket),
                }
            }
            BucketKind::Remainder => {
                let bits = bucket.assemble_bits(|idx| {
                    remainder_lookup_bits(
                        &rebase_remainder(&nodes[&idx], trace),
                        bucket.width as i32,
                    )
                });
                (0..bucket.num_chunks())
                    .map(|d| {
                        (
                            bucket.committed_poly(d),
                            build_one_hot_rad_witness(&bits, d, bucket.width),
                        )
                    })
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}

#[cfg(test)]
mod bucket_membership_tests {
    use super::*;
    use atlas_onnx_tracer::{model::test::ModelBuilder, ops::*, tensor::Tensor};
    use std::collections::BTreeSet;

    #[test]
    fn clamp_buckets_contain_exactly_the_required_vector_clamps() {
        let mut builder = ModelBuilder::with_scale(14);
        let input = builder.input(vec![4]);
        let weight = builder.constant(Tensor::new(Some(&[2, -3, 5, 7]), &[4]).unwrap());
        let add = builder.add(input, weight);
        let sub = builder.sub(input, weight);
        let sum = builder.sum(input, vec![], vec![4]);
        let einsum = builder.einsum("i,i->i", vec![input, weight], vec![4]);
        let mul = builder.mul(input, weight);
        let square = builder.square(input);
        let cube = builder.cube(input, 14);
        // ReLU and Clamp have their own lookups, not saturating-clamp buckets.
        builder.relu(input);
        builder.clamp(input, 7);
        let raw_cube = builder.cube(input, 0);
        let raw_square = builder.square(input);
        let raw_mul = builder.mul(input, weight);
        let scalar = builder.input(vec![1]);
        builder.add(scalar, scalar);
        let mut model = builder.build();
        model.graph.nodes.get_mut(&raw_square).unwrap().operator =
            Operator::Square(Square { scale: 0 });
        model.graph.nodes.get_mut(&raw_mul).unwrap().operator = Operator::Mul(Mul { scale: 0 });
        let mos = model.graph.nodes.len();
        model.graph.nodes.insert(
            mos,
            ComputationNode::new(
                mos,
                Operator::MeanOfSquares(MeanOfSquares {
                    axes: vec![],
                    scale: 14,
                    count: 1,
                    padded_count: 1,
                }),
                vec![input],
                vec![4],
            ),
        );
        model.annotate_clamp_widths();
        let buckets = clamp_buckets(&model);
        let actual: BTreeSet<_> = buckets
            .iter()
            .flat_map(|b| b.nodes.iter().map(|n| n.idx))
            .collect();
        assert_eq!(
            actual,
            BTreeSet::from([add, sub, sum, einsum, mul, square, cube, mos])
        );
        assert!(!actual.contains(&raw_cube));
        for bucket in buckets {
            for member in bucket.nodes {
                assert_eq!(bucket.width, model.graph.nodes[&member.idx].sat_clamp_bits);
                assert_eq!(member.offset % (1 << member.log_t), 0);
            }
        }
    }

    #[test]
    fn incomplete_sum_does_not_enter_a_bucket() {
        let mut model = Model::default();
        let node = ComputationNode::new(0, Operator::Sum(Sum { axes: vec![0] }), vec![99], vec![4]);
        model.graph.nodes.insert(0, node);
        assert!(clamp_buckets(&model).is_empty());
        model.graph.nodes.get_mut(&0).unwrap().inputs.clear();
        assert!(clamp_buckets(&model).is_empty());
    }
}
