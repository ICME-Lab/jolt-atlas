//! Packed ("global") saturating-clamp and rescale-remainder buckets.
//!
//! Every clamped node used to commit its own one-hot chunks and run its own
//! lookup sumchecks. Here nodes of the same width are packed into *buckets*:
//! one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has one set of committed columns
//! (a clamp bucket's dense output digits and one-hot slack chunks, a remainder
//! bucket's dense digits) and one set of checks whatever the node count.
//!
//! A bucket sumcheck over the packed cycle space uses the cycle weight
//! `Σ_n γ_n·eq(r_n, t − off_n)` ([`CycleWeight::Packed`]) so its input claim
//! is a `γ`-combination of the member nodes' existing openings. The clamp
//! buckets are proven by the interior / saturated split and the remainder
//! buckets by a value sumcheck and a digit range check, both in [`super::clamp_split`]. Gaps at the
//! end of a bucket hold zero-weight dummy cycles.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
use super::{clamp_split, fused_rebase::rebase_bits, ops::mean_of_squares::mos_divisor};
use crate::node_view::clamp_value_bound;
use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{field::JoltField, utils::math::Math};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::{CycleWeight, PackedSegment},
};
use atlas_trace::{trace::ModelExecutionIO, Model, Op};
use common::{consts::TRIG_PERIOD_MODULUS, CommittedPoly, VirtualPoly};

/// A node's segment inside a bucket's cycle space.
#[derive(Clone, Debug)]
pub struct BucketNode {
    /// Node index.
    pub idx: usize,
    /// What the segment range-checks in a remainder bucket; `None` in a clamp bucket, whose
    /// segment is the node's clamped output.
    pub claim: Option<RangeClaim>,
    /// Start of the node's segment (a multiple of its own padded size).
    pub offset: usize,
    /// log₂ of the node's padded output size.
    pub log_t: usize,
}

/// A remainder a node's relation leaves to be bounded. Each is a virtual advice opening under
/// the node's execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Remainder {
    /// A fused rescale's remainder (`Einsum`, `Mul`, `Square`, `Cube`, `MeanOfSquares`).
    Rescale,
    /// `Div`'s remainder.
    Div,
    /// `ScalarConstDiv`'s remainder.
    ScalarConstDiv,
    /// `Rsqrt`'s division remainder `S³ mod x`.
    RsqrtDiv,
    /// `Rsqrt`'s square-root remainder `q − out²`.
    RsqrtSqrt,
    /// A trig op's teleport remainder `x mod P`.
    Teleport,
    /// Softmax's reciprocal-multiplication remainder.
    SoftmaxRecipMult,
    /// Softmax's exponentiation remainder.
    SoftmaxExp,
}

impl Remainder {
    /// The remainder's opening for node `node_idx`.
    pub fn opening_id(self, node_idx: usize) -> OpeningId {
        let poly = match self {
            Self::Rescale => VirtualPoly::RescaleRemainder(node_idx),
            Self::Div | Self::RsqrtDiv => VirtualPoly::DivRemainder(node_idx),
            Self::ScalarConstDiv => VirtualPoly::ScalarConstDivRemainder(node_idx),
            Self::RsqrtSqrt => VirtualPoly::SqrtRemainder(node_idx),
            Self::Teleport => VirtualPoly::TeleportRemainder(node_idx),
            Self::SoftmaxRecipMult => VirtualPoly::SoftmaxRecipMultRemainder(node_idx),
            Self::SoftmaxExp => VirtualPoly::SoftmaxExpRemainder(node_idx),
        };
        OpeningId::new(poly, SumcheckId::NodeExecution(node_idx))
    }
}

/// The bound `B` a remainder lies strictly below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// A constant.
    Const(i64),
    /// The output of node `.0`, as the remainder's node opened it beside the remainder.
    Node(usize),
    /// `2·v + 1` for `v` the output of node `.0`, opened the same way.
    TwiceNodePlusOne(usize),
}

/// What one bucket member range-checks: a remainder `R` itself (`R ∈ [0, 2^w)`), or the slack
/// `B − 1 − R` below its bound (`R < B`). Both are affine in claims the node's turn bound at
/// one point.
///
/// A member is non-negative and below `2^w` as a field element, and `|B| < 2^33`, so with
/// `w ≤ 32` each statement holds over the integers: the magnitudes are far below the field
/// modulus. `R ≥ 0` and `B − 1 − R ≥ 0` together give `B ≥ 1`: a bound that is not positive
/// has no proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeClaim {
    /// The remainder.
    pub remainder: Remainder,
    /// `None` for `R` itself, the bound for the slack `B − 1 − R`.
    pub slack: Option<Bound>,
}

/// Width of a two-sided range claim: wide enough for every `i32`-sized bound.
pub const TWO_SIDED_WIDTH: usize = 32;

/// The range claims that bound a remainder below `bound`, each with its width: `R` alone at
/// width `w` when the bound is `2^w` with `w` a whole number of digits, otherwise `R` and the
/// slack `B − 1 − R` at [`TWO_SIDED_WIDTH`].
pub fn range_claims(remainder: Remainder, bound: Bound) -> Vec<(RangeClaim, usize)> {
    let digit_bits = clamp_split::DIGIT_BASE.trailing_zeros() as usize;
    if let Bound::Const(b) = bound {
        if b > 0 && b.count_ones() == 1 {
            let width = b.trailing_zeros() as usize;
            if width.is_multiple_of(digit_bits) {
                return vec![(
                    RangeClaim {
                        remainder,
                        slack: None,
                    },
                    width,
                )];
            }
        }
    }
    vec![
        (
            RangeClaim {
                remainder,
                slack: None,
            },
            TWO_SIDED_WIDTH,
        ),
        (
            RangeClaim {
                remainder,
                slack: Some(bound),
            },
            TWO_SIDED_WIDTH,
        ),
    ]
}

impl RangeClaim {
    /// The member's point and claim for node `node_idx`, from the openings its node bound.
    /// `Err` if a claim is missing, or the bound is not opened at the remainder's point.
    pub fn evaluate<F: JoltField>(
        &self,
        node_idx: usize,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<(Vec<F>, F), ProofVerifyError> {
        let (point, remainder) =
            accumulator.get_virtual_polynomial_opening(self.remainder.opening_id(node_idx))?;
        let bound_at = |bound_node: usize| -> Result<F, ProofVerifyError> {
            let (bound_point, bound) =
                accumulator.get_virtual_polynomial_opening(OpeningId::new(
                    VirtualPoly::NodeOutput(bound_node),
                    SumcheckId::NodeExecution(node_idx),
                ))?;
            if bound_point.r != point.r {
                return Err(ProofVerifyError::Malformed(format!(
                    "node {node_idx}: {:?} and its bound are opened at different points",
                    self.remainder
                )));
            }
            Ok(bound)
        };
        let claim = match self.slack {
            None => remainder,
            Some(Bound::Const(b)) => F::from_i64(b - 1) - remainder,
            Some(Bound::Node(n)) => bound_at(n)? - F::one() - remainder,
            Some(Bound::TwiceNodePlusOne(n)) => {
                let v = bound_at(n)?;
                v + v - remainder
            }
        };
        Ok((point.r, claim))
    }
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

/// Partition the model's clamped nodes into buckets: grouped by
/// width, packed first-fit-decreasing by padded size into a capacity equal to
/// the largest clamped node (so offsets are automatically aligned). `Err` if a node's operand
/// is not a node of the model.
pub fn clamp_buckets(model: &Model) -> Result<Vec<ClampBucket>, ProofVerifyError> {
    let mut clamped: Vec<BucketItem> = Vec::new();
    for n in &model.views() {
        if clamp_value_bound(model, n)?.is_some() {
            clamped.push((
                n.idx,
                None,
                n.sat_clamp_bits,
                n.pow2_padded_num_output_elements(),
            ));
        }
    }
    Ok(pack_buckets(BucketKind::Clamp, clamped))
}

/// The remainders `node`'s relations leave to be bounded, each with its bound. `Err` for a
/// node whose constant bound cannot be range-checked.
pub fn node_remainders(
    node: &ComputationNode,
) -> Result<Vec<(Remainder, Bound)>, ProofVerifyError> {
    let malformed =
        |what: String| ProofVerifyError::Malformed(format!("model: node {}: {what}", node.idx));
    let operand = |position: usize| {
        node.inputs
            .get(position)
            .copied()
            .ok_or_else(|| malformed(format!("no operand {position}")))
    };
    Ok(match &node.operator {
        Op::Einsum(_) | Op::Mul(_) | Op::Square(_) | Op::Cube(_) => match rebase_bits(node) {
            Some(bits) if bits > 0 => vec![(Remainder::Rescale, Bound::Const(1i64 << bits))],
            _ => vec![],
        },
        Op::MeanOfSquares(op) => {
            let divisor = mos_divisor(op, node.scale);
            if i32::try_from(divisor).is_err() {
                return Err(malformed(format!(
                    "MeanOfSquares divisor N·2^S = {divisor} must fit i32 for the range check"
                )));
            }
            vec![(Remainder::Rescale, Bound::Const(divisor))]
        }
        Op::Div(_) => vec![(Remainder::Div, Bound::Node(operand(1)?))],
        Op::ScalarConstDiv(op) => {
            if op.divisor < 1 {
                return Err(malformed(format!(
                    "ScalarConstDiv divisor {} is not positive",
                    op.divisor
                )));
            }
            vec![(
                Remainder::ScalarConstDiv,
                Bound::Const(i64::from(op.divisor)),
            )]
        }
        Op::Rsqrt(_) => vec![
            (Remainder::RsqrtDiv, Bound::Node(operand(0)?)),
            (Remainder::RsqrtSqrt, Bound::TwiceNodePlusOne(node.idx)),
        ],
        Op::Sin(_) | Op::Cos(_) => vec![(
            Remainder::Teleport,
            Bound::Const(i64::from(TRIG_PERIOD_MODULUS)),
        )],
        Op::SoftmaxLastAxis(_) => {
            let bound = Bound::Const(1i64 << node.scale);
            vec![
                (Remainder::SoftmaxRecipMult, bound),
                (Remainder::SoftmaxExp, bound),
            ]
        }
        _ => vec![],
    })
}

/// Partition the model's range claims into buckets keyed by width: every remainder a node's
/// relations leave to be bounded ([`node_remainders`]), as the claims [`range_claims`] gives
/// for its bound. `Err` for a node whose bound cannot be range-checked.
pub fn remainder_buckets(model: &Model) -> Result<Vec<ClampBucket>, ProofVerifyError> {
    let mut items: Vec<BucketItem> = Vec::new();
    for n in &model.views() {
        for (remainder, bound) in node_remainders(n)? {
            for (claim, width) in range_claims(remainder, bound) {
                items.push((
                    n.idx,
                    Some(claim),
                    width,
                    n.pow2_padded_num_output_elements(),
                ));
            }
        }
    }
    Ok(pack_buckets(BucketKind::Remainder, items))
}

/// A bucket member before packing: node, range claim (remainder buckets), width, padded size.
type BucketItem = (usize, Option<RangeClaim>, usize, usize);

/// First-fit-decreasing packing of `(node, width, padded size)` items, grouped
/// by width, into a capacity equal to the largest item.
fn pack_buckets(kind: BucketKind, clamped: Vec<BucketItem>) -> Vec<ClampBucket> {
    let Some(capacity) = clamped.iter().map(|c| c.3).max() else {
        return Vec::new();
    };
    let mut widths: Vec<usize> = clamped.iter().map(|c| c.2).collect();
    widths.sort_unstable();
    widths.dedup();

    let mut buckets: Vec<ClampBucket> = Vec::new();
    for w in widths {
        let mut group: Vec<(usize, Option<RangeClaim>, usize)> = clamped
            .iter()
            .filter(|c| c.2 == w)
            .map(|c| (c.0, c.1, c.3))
            .collect();
        // Decreasing size (then node index, a node's claims in the order given): every
        // earlier size is a multiple of every later one, so first-fit keeps segments aligned.
        group.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        let mut open: Vec<(usize, Vec<BucketNode>)> = Vec::new();
        for (idx, claim, t) in group {
            let node = |offset: usize| BucketNode {
                idx,
                claim,
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
    /// The bucket's committed polynomials: [`clamp_split::chunk_polys`] for a clamp bucket,
    /// [`clamp_split::remainder_digit_polys`] for a remainder bucket.
    pub fn committed_polys(&self) -> Vec<CommittedPoly> {
        match self.kind {
            BucketKind::Clamp => clamp_split::chunk_polys(self),
            BucketKind::Remainder => clamp_split::remainder_digit_polys(self),
        }
    }

    /// Size of the bucket's cycle space.
    pub fn cycle_len(&self) -> usize {
        1 << self.log_t
    }

    /// The packed range check's input claim `Σ_m γ_m·claim_m` and cycle weight, from the
    /// openings the members' nodes bound ([`RangeClaim::evaluate`]). `Err` if the accumulator
    /// lacks a claim a member reads, or a member's claims are not at one point.
    pub fn remainder_inputs<F: JoltField>(
        &self,
        gammas: &[F],
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<(F, CycleWeight<F>), ProofVerifyError> {
        if gammas.len() != self.nodes.len() {
            return Err(ProofVerifyError::Malformed(format!(
                "{} batching coefficients for a bucket of {} members",
                gammas.len(),
                self.nodes.len()
            )));
        }
        let mut input_claim = F::zero();
        let mut segments = Vec::with_capacity(self.nodes.len());
        for (n, gamma) in self.nodes.iter().zip(gammas) {
            let range_claim = n.claim.ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "node {} is in a remainder bucket without a range claim",
                    n.idx
                ))
            })?;
            let (r, claim) = range_claim.evaluate(n.idx, accumulator)?;
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
                r,
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

/// The buckets' committed polynomials for the statement `io`: every remainder bucket's, and the
/// clamp buckets' that have a live node. A clamp bucket whose nodes are all unsaturated model
/// outputs ([`clamp_split::exact_output_verifier`]) proves nothing through its columns, so it
/// commits none. `Err` if a node's operand is not a node of the model.
pub fn bucket_committed_polys(
    model: &Model,
    io: &ModelExecutionIO,
) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
    let live = clamp_buckets(model)?.into_iter().filter(|bucket| {
        clamp_split::live_bucket(bucket, |idx| {
            !clamp_split::exact_output_verifier(model, io, idx)
        })
        .is_some()
    });
    Ok(live
        .chain(remainder_buckets(model)?)
        .flat_map(|bucket| bucket.committed_polys())
        .collect())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::{pack_buckets, BucketKind};

    /// `(log_t, member node indices)` per bucket, in bucket order.
    fn layout(items: &[(usize, usize, usize)]) -> Vec<(usize, usize, Vec<usize>)> {
        let items = items.iter().map(|&(idx, w, t)| (idx, None, w, t)).collect();
        pack_buckets(BucketKind::Clamp, items)
            .into_iter()
            .map(|b| (b.width, b.log_t, b.nodes.iter().map(|n| n.idx).collect()))
            .collect()
    }

    /// A one-element node joining a bucket filled to an exact power of two below capacity
    /// raises it to the next power of two.
    #[test]
    fn one_element_node_can_double_a_bucket() {
        let without = layout(&[(0, 40, 8), (1, 40, 2)]);
        assert_eq!(without, vec![(40, 3, vec![0]), (40, 1, vec![1])]);
        let with = layout(&[(0, 40, 8), (1, 40, 2), (2, 40, 1)]);
        assert_eq!(with, vec![(40, 3, vec![0]), (40, 2, vec![1, 2])]);
    }

    /// A one-element node that fits in no bucket of its width opens one, and every later bucket
    /// moves up an index.
    #[test]
    fn one_element_node_can_open_a_bucket() {
        let without = layout(&[(0, 40, 4), (1, 48, 4)]);
        assert_eq!(without, vec![(40, 2, vec![0]), (48, 2, vec![1])]);
        let with = layout(&[(0, 40, 4), (1, 48, 4), (2, 40, 1)]);
        assert_eq!(
            with,
            vec![(40, 2, vec![0]), (40, 0, vec![2]), (48, 2, vec![1])]
        );
    }

    /// With no larger node the capacity is one cell, so each one-element node is its own
    /// bucket.
    #[test]
    fn one_element_nodes_alone_get_a_bucket_each() {
        let both = layout(&[(0, 40, 1), (1, 40, 1)]);
        assert_eq!(both, vec![(40, 0, vec![0]), (40, 0, vec![1])]);
    }
}
