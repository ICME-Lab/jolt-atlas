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
//! A remainder bucket's member is a *range claim* ([`RangeClaim`]): a remainder `R` some
//! node's relation leaves to be bounded (`R ∈ [0, 2^w)`), or the slack `B − 1 − R` below its
//! bound (`R < B`), each an affine expression over claims the node bound at one point.
//! [`node_remainders`] lists each node's remainders and bounds, and [`range_claims`] picks the
//! form: `R` alone at width `w` for a bound `2^w` of a whole number of digits, both claims at
//! [`TWO_SIDED_WIDTH`] otherwise. Together the two claims give `B ≥ 1` and `0 ≤ R < B` over
//! the integers; a bound that is not positive has no proof.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
use super::{clamp_split, fused_rebase::rebase_bits, ops::mean_of_squares::mos_divisor};
use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{field::JoltField, utils::math::Math};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::{CycleWeight, PackedSegment},
};
use atlas_trace::{Model, Op};
use common::{consts::TRIG_PERIOD_MODULUS, CommittedPoly, VirtualPoly};

/// A node's segment inside a bucket's cycle space.
#[derive(Clone, Debug)]
pub struct BucketNode {
    /// Node index.
    pub idx: usize,
    /// What the segment range-checks.
    pub claim: Option<RangeClaim>,
    /// Start of the member's segment (a multiple of its own size).
    pub offset: usize,
    /// log₂ of the member's size: the node's padded output size, or its row count for a
    /// per-row remainder ([`Remainder::domain_len`]).
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
    /// The node's output offset by `2^31`: in `[0, 2^32)` exactly when the output is an `i32`.
    /// Read from the node's reduced output opening, not from an advice opening.
    Output,
    /// Softmax's gap of a row's sum-table reads below the sum scale, one value per row.
    SoftmaxSumGap,
    /// Softmax's excess of a row's shifted sum-table reads over the sum scale, one value per
    /// row.
    SoftmaxPrevSumGap,
}

impl Remainder {
    /// The remainder's advice opening for node `node_idx`; `None` for [`Self::Output`], which
    /// is read from the node's reduced output opening.
    pub fn opening_id(self, node_idx: usize) -> Option<OpeningId> {
        let poly = match self {
            Self::Output => return None,
            Self::Rescale => VirtualPoly::RescaleRemainder(node_idx),
            Self::Div | Self::RsqrtDiv => VirtualPoly::DivRemainder(node_idx),
            Self::ScalarConstDiv => VirtualPoly::ScalarConstDivRemainder(node_idx),
            Self::RsqrtSqrt => VirtualPoly::SqrtRemainder(node_idx),
            Self::Teleport => VirtualPoly::TeleportRemainder(node_idx),
            Self::SoftmaxSumGap => VirtualPoly::SoftmaxSumGap(node_idx),
            Self::SoftmaxPrevSumGap => VirtualPoly::SoftmaxPrevSumGap(node_idx),
        };
        Some(OpeningId::new(poly, SumcheckId::NodeExecution(node_idx)))
    }

    /// The remainder's point and claim for node `node_idx`. `Err` if the accumulator lacks
    /// the opening.
    pub fn opening<F: JoltField>(
        self,
        node_idx: usize,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<(Vec<F>, F), ProofVerifyError> {
        match self.opening_id(node_idx) {
            Some(id) => {
                let (point, claim) = accumulator.get_virtual_polynomial_opening(id)?;
                Ok((point.r, claim))
            }
            None => {
                let (point, out) = accumulator.get_node_output_opening(node_idx)?;
                Ok((point.r, out + F::from_u64(OUTPUT_OFFSET)))
            }
        }
    }

    /// The number of values the remainder has for `node`, a power of two: one per padded
    /// output cell, or one per row of the last axis for softmax's row gaps.
    pub fn domain_len(self, node: &ComputationNode) -> usize {
        let cells = node.pow2_padded_num_output_elements();
        match self {
            Self::SoftmaxSumGap | Self::SoftmaxPrevSumGap => {
                let lanes = node
                    .output_dims
                    .last()
                    .map_or(1, |n| n.next_power_of_two().max(1));
                cells / lanes
            }
            _ => cells,
        }
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

/// `2^31`: the offset that makes an `i32` output non-negative ([`Remainder::Output`]).
pub const OUTPUT_OFFSET: u64 = 1 << 31;

/// Width of an output range claim: `out + 2^31 ∈ [0, 2^32)`.
pub const OUTPUT_WIDTH: usize = 32;

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
        let (point, remainder) = self.remainder.opening(node_idx, accumulator)?;
        let bound_at = |bound_node: usize| -> Result<F, ProofVerifyError> {
            let (bound_point, bound) =
                accumulator.get_virtual_polynomial_opening(OpeningId::new(
                    VirtualPoly::NodeOutput(bound_node),
                    SumcheckId::NodeExecution(node_idx),
                ))?;
            if bound_point.r != point {
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
        Ok((point, claim))
    }
}

/// One packed range check: members whose values lie in `[0, 2^width)`.
#[derive(Clone, Debug)]
pub struct ClampBucket {
    /// Bucket index.
    pub idx: usize,
    /// The bit width every member's value is bounded by.
    pub width: usize,
    /// log₂ of the bucket's cycle space.
    pub log_t: usize,
    /// Member nodes, in layout order.
    pub nodes: Vec<BucketNode>,
}

/// Whether `node`'s op is exact arithmetic whose result must be shown to lie in `i32`: a sum,
/// difference, axis sum or rescaled quotient.
pub fn checks_its_output(node: &ComputationNode) -> bool {
    node.operator.spec().checks_output()
}

/// The range claims `node` contributes, each with its width: its remainders' ([`node_remainders`],
/// [`range_claims`]), and its output's when its op checks it ([`checks_its_output`]) and it is
/// not a model output (a public `i32` tensor needs no check). `Err` for a node whose bound
/// cannot be range-checked.
pub fn node_range_claims(
    model: &Model,
    node: &ComputationNode,
) -> Result<Vec<(RangeClaim, usize)>, ProofVerifyError> {
    let mut claims = Vec::new();
    for (remainder, bound) in node_remainders(node)? {
        claims.extend(range_claims(remainder, bound));
    }
    if checks_its_output(node) && !model.output_indices().contains(&node.idx) {
        claims.push((
            RangeClaim {
                remainder: Remainder::Output,
                slack: None,
            },
            OUTPUT_WIDTH,
        ));
    }
    Ok(claims)
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
            // `g ∈ [0, S']` and `g⁻ ∈ [0, N·(S' + 1))`, each bounded by the power of two with a
            // whole number of digits above it.
            let digit_bits = clamp_split::DIGIT_BASE.trailing_zeros() as usize;
            let width = |bits: usize| bits.next_multiple_of(digit_bits);
            let sum_scale_bits =
                node.scale as usize + atlas_trace::semantics::softmax::SUM_EXTRA_BITS as usize;
            let lanes = node
                .output_dims
                .last()
                .ok_or_else(|| malformed("softmax of rank 0".to_string()))?
                .next_power_of_two();
            let (gap_width, prev_gap_width) = (
                width(sum_scale_bits + 1),
                width(sum_scale_bits + 1 + lanes.log_2()),
            );
            if prev_gap_width > 62 {
                return Err(malformed(format!(
                    "softmax over {lanes} lanes: the row sums do not fit the range check"
                )));
            }
            vec![
                (Remainder::SoftmaxSumGap, Bound::Const(1i64 << gap_width)),
                (
                    Remainder::SoftmaxPrevSumGap,
                    Bound::Const(1i64 << prev_gap_width),
                ),
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
        for (claim, width) in node_range_claims(model, n)? {
            items.push((n.idx, Some(claim), width, claim.remainder.domain_len(n)));
        }
    }
    Ok(pack_buckets(items))
}

/// A bucket member before packing: node, range claim (remainder buckets), width, padded size.
type BucketItem = (usize, Option<RangeClaim>, usize, usize);

/// First-fit-decreasing packing of `(node, width, padded size)` items, grouped
/// by width, into a capacity equal to the largest item.
fn pack_buckets(clamped: Vec<BucketItem>) -> Vec<ClampBucket> {
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
    /// The bucket's committed polynomials: its digit columns.
    pub fn committed_polys(&self) -> Vec<CommittedPoly> {
        clamp_split::remainder_digit_polys(self)
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

/// The buckets' committed polynomials: every range bucket's digit columns. A function of the
/// model alone. `Err` for a node whose bound cannot be range-checked.
pub fn bucket_committed_polys(model: &Model) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
    Ok(remainder_buckets(model)?
        .iter()
        .flat_map(|bucket| bucket.committed_polys())
        .collect())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::{pack_buckets, range_claims, Bound, Remainder, TWO_SIDED_WIDTH};

    /// `(log_t, member node indices)` per bucket, in bucket order.
    fn layout(items: &[(usize, usize, usize)]) -> Vec<(usize, usize, Vec<usize>)> {
        let items = items.iter().map(|&(idx, w, t)| (idx, None, w, t)).collect();
        pack_buckets(items)
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

    /// One claim at width `w` exactly when the bound is `2^w` with `w` a whole number of
    /// base-4 digits; otherwise the remainder and its slack at the two-sided width.
    #[test]
    fn range_claims_follow_the_width_rule() {
        let shape = |bound: Bound| -> Vec<(bool, usize)> {
            range_claims(Remainder::Rescale, bound)
                .into_iter()
                .map(|(claim, width)| (claim.slack.is_some(), width))
                .collect()
        };
        let two_sided = vec![(false, TWO_SIDED_WIDTH), (true, TWO_SIDED_WIDTH)];
        assert_eq!(shape(Bound::Const(1 << 14)), vec![(false, 14)]);
        assert_eq!(shape(Bound::Const(1 << 28)), vec![(false, 28)]);
        assert_eq!(shape(Bound::Const(1)), vec![(false, 0)]);
        assert_eq!(shape(Bound::Const(1 << 7)), two_sided);
        assert_eq!(shape(Bound::Const(7)), two_sided);
        assert_eq!(shape(Bound::Const(768 << 14)), two_sided);
        assert_eq!(shape(Bound::Const(0)), two_sided);
        assert_eq!(shape(Bound::Const(-3)), two_sided);
        assert_eq!(shape(Bound::Node(3)), two_sided);
        assert_eq!(shape(Bound::TwiceNodePlusOne(3)), two_sided);
    }
}
