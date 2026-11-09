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
use super::{clamp_lookups::is_scalar, clamp_split, fused_rebase::rebase_bits};
use crate::node_view::clamp_value_bound;
use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{field::JoltField, utils::math::Math};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::{CycleWeight, PackedSegment},
};
use atlas_trace::{Model, Op};
use common::{CommittedPoly, VirtualPoly};

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
/// the largest clamped node (so offsets are automatically aligned). `Err` if a node's operand
/// is not a node of the model.
pub fn clamp_buckets(model: &Model) -> Result<Vec<ClampBucket>, ProofVerifyError> {
    let mut clamped: Vec<(usize, usize, usize)> = Vec::new();
    for n in &model.views() {
        if clamp_value_bound(model, n)?.is_some() && !is_scalar(n) {
            clamped.push((n.idx, n.sat_clamp_bits, n.pow2_padded_num_output_elements()));
        }
    }
    Ok(pack_buckets(BucketKind::Clamp, clamped))
}

/// Whether `node` proves a fused rescale remainder range check (Einsum / Mul /
/// Square / Cube with a nonzero rebase, non-scalar).
pub fn has_remainder_rc(node: &ComputationNode) -> bool {
    matches!(
        node.operator,
        Op::Einsum(_) | Op::Mul(_) | Op::Square(_) | Op::Cube(_)
    ) && rebase_bits(node).is_some_and(|b| b > 0)
        && !is_scalar(node)
}

/// Partition the fused-rescale remainder range checks into buckets keyed by
/// rebase width.
pub fn remainder_buckets(model: &Model) -> Vec<ClampBucket> {
    let items: Vec<(usize, usize, usize)> = model
        .views()
        .iter()
        .filter(|n| has_remainder_rc(n))
        .filter_map(|n| {
            Some((
                n.idx,
                rebase_bits(n)? as usize,
                n.pow2_padded_num_output_elements(),
            ))
        })
        .collect();
    pack_buckets(BucketKind::Remainder, items)
}

/// First-fit-decreasing packing of `(node, width, padded size)` items, grouped
/// by width, into a capacity equal to the largest item.
fn pack_buckets(kind: BucketKind, clamped: Vec<(usize, usize, usize)>) -> Vec<ClampBucket> {
    let Some(capacity) = clamped.iter().map(|c| c.2).max() else {
        return Vec::new();
    };
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

    /// The packed remainder range check's input claim `Σ_n γ_n·R_n(r_n)` and
    /// cycle weight, from the member nodes' remainder advice openings.
    /// `Err` if the accumulator lacks a claim it reads.
    pub fn remainder_inputs<F: JoltField>(
        &self,
        gammas: &[F],
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<(F, CycleWeight<F>), ProofVerifyError> {
        if gammas.len() != self.nodes.len() {
            return Err(ProofVerifyError::Malformed(format!(
                "{} batching coefficients for a bucket of {} nodes",
                gammas.len(),
                self.nodes.len()
            )));
        }
        let mut input_claim = F::zero();
        let mut segments = Vec::with_capacity(self.nodes.len());
        for (n, gamma) in self.nodes.iter().zip(gammas) {
            let (r, claim) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::RescaleRemainder(n.idx),
                SumcheckId::NodeExecution(n.idx),
            ))?;
            if r.len() != n.log_t {
                return Err(ProofVerifyError::Malformed(format!(
                    "node {} remainder point has {} coordinates, expected {}",
                    n.idx,
                    r.len(),
                    n.log_t
                )));
            }
            input_claim += *gamma * claim;
            segments.push(PackedSegment {
                gamma: *gamma,
                prefix: n.offset >> n.log_t,
                prefix_len: self.log_t - n.log_t,
                r: r.r,
            });
        }
        Ok((
            input_claim,
            CycleWeight::Packed {
                log_T: self.log_t,
                segments,
            },
        ))
    }
}

/// All buckets' committed polynomials (the model-level replacement for the
/// per-node `ClampRaD` / `RescaleRemainderRaD` chunks). `Err` if a node's operand is not a node
/// of the model.
pub fn bucket_committed_polys(model: &Model) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
    Ok(clamp_buckets(model)?
        .iter()
        .chain(remainder_buckets(model).iter())
        .flat_map(ClampBucket::committed_polys)
        .collect())
}
