//! Packed ("global") saturating-clamp lookup buckets.
//!
//! Every clamped node used to commit its own `w/4` one-hot chunks and run its
//! own read-raf + one-hot checks. Here nodes of the same clamp width are packed
//! into *buckets*: one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has `w/4` committed chunks in
//! total and one read-raf / one-hot-check triple, whatever the node count.
//!
//! The read-raf sumcheck over a bucket proves `Σ_n γ_n·(rv_n(r_n) + γ·acc_n(r_n))`
//! with the packed cycle weight `Σ_n γ_n·eq(r_n, t − off_n)`
//! ([`CycleWeight::Packed`]); the one-hot checks run over the bucket's cycle
//! space unchanged. Gaps at the end of a bucket hold dummy index-0 lookups so
//! the Hamming-weight check (exactly one set address per cycle) stays exact.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
use super::{
    clamp_lookups::is_scalar,
    clamp_split,
    deferred_lookups::DEFERRED_PROOF_IDX,
    fused_rebase::{rebase_bits, rebase_remainder, remainder_lookup_bits},
    witness::build_one_hot_rad_witness,
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::model::{clamp_width::clamp_value_bound, trace::Trace, Model};
use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    },
    subprotocols::{
        ps_shout::{CycleWeight, PackedSegment},
        shout::RaOneHotEncoding,
    },
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

/// Partition the model's clamped (non-scalar) nodes into buckets: grouped by
/// width, packed first-fit-decreasing by padded size into a capacity equal to
/// the largest clamped node (so offsets are automatically aligned).
pub fn clamp_buckets(model: &Model) -> Vec<ClampBucket> {
    let nodes = &model.graph.nodes;
    let clamped: Vec<(usize, usize, usize)> = nodes
        .values()
        .filter(|n| clamp_value_bound(n, nodes).is_some() && !is_scalar(n))
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

    /// The bucket's (virtual) read-address polynomial and its sumcheck id.
    pub fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        let vp = match self.kind {
            BucketKind::Clamp => VirtualPoly::GlobalClampRa(self.idx),
            BucketKind::Remainder => VirtualPoly::GlobalRemainderRa(self.idx),
        };
        (vp, SumcheckId::NodeExecution(DEFERRED_PROOF_IDX))
    }

    /// Opening id of the bucket's (virtual) read-address polynomial.
    pub fn ra_opening_id(&self) -> OpeningId {
        let (vp, sid) = self.ra_poly();
        OpeningId::new(vp, sid)
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
                clamp_split::BucketSplit::from_trace(bucket, trace, model).witness_polys(bucket)
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

/// [`RaOneHotEncoding`] for a bucket's one-hot checks.
pub struct GlobalClampEncoding<'a>(pub &'a ClampBucket);

impl RaOneHotEncoding for GlobalClampEncoding<'_> {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        self.0.committed_poly(d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        self.0.ra_opening_id()
    }

    fn ra_source(&self) -> OpeningId {
        self.0.ra_opening_id()
    }

    fn log_k(&self) -> usize {
        self.0.width
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.0.width)
    }

    /// The cycle half of the bucket read-raf's output point.
    fn r_cycle<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F> {
        let (r, _) = accumulator.get_virtual_polynomial_opening(self.0.ra_opening_id());
        r.r[self.0.width..].to_vec()
    }
}
